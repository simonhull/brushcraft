//! Residency on lavapipe, against a model of the store: what the GPU holds after each frame.
//!
//! The model's tiles carry their place and a version (one more than the last epoch naming
//! them) in their vertices, so a readback of the tree and the slots shows which data every
//! resident tile draws.

use std::collections::{BTreeMap, BTreeSet};
use std::convert::Infallible;
use std::time::{Duration, Instant};

use geom::{BaseMesh, Level, TileKey, cube_sphere};
use store::{Drained, Epoch};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BufferUsages, CommandEncoderDescriptor,
    ComputePassDescriptor, ShaderSource,
};

use super::*;
use crate::gpu::GpuTest;
use crate::tree::{GpuNode, NODE_LIVE, Place};
use crate::upload::{GpuVertex, TileSource};

/// The test projects' finest level: level 7, where a face has 2 × 2 tiles.
const FINEST: u8 = 7;

/// A small deterministic generator for the tests' random choices.
struct Rng(u64);

impl Rng {
    /// The next 64 random bits (splitmix64).
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A random index below `n` (`n > 0`).
    fn below(&mut self, n: usize) -> usize {
        usize::try_from(self.next() % n as u64).expect("below n")
    }

    /// True one time in `n`.
    fn one_in(&mut self, n: usize) -> bool {
        self.below(n) == 0
    }
}

/// The store as residency sees it: a version per tile, bumped by each epoch.
struct Model {
    /// The finest level.
    finest: Level,
    /// Per tile, one more than the last epoch naming it (0 if none has).
    versions: BTreeMap<TileKey, u32>,
    /// The next epoch's number.
    next_seq: u64,
    /// Epochs closed and not drained yet.
    undrained: Vec<Epoch>,
    /// Whether the undrained epochs are lost: the next drain is a resync.
    lost: bool,
    /// Every tile of the project.
    all: Vec<TileKey>,
}

impl Model {
    /// A model of `mesh`'s tiles, every version 0.
    fn new(mesh: &BaseMesh, finest: Level) -> Self {
        let levels: Vec<Level> = (0..=finest.get()).map(level).collect();
        let all = mesh.faces().flat_map(|face| {
            levels.iter().flat_map(move |&level| {
                let per_side = u16::try_from(level.tiles_per_side()).expect("small");
                (0..per_side).flat_map(move |ty| {
                    (0..per_side).filter_map(move |tx| {
                        Place::of(&TileKey::display(face, level, tx, ty).ok()?).key(finest)
                    })
                })
            })
        });
        Self {
            finest,
            versions: BTreeMap::new(),
            next_seq: 0,
            undrained: Vec::new(),
            lost: false,
            all: all.collect(),
        }
    }

    /// A random tile.
    fn any(&self, rng: &mut Rng) -> TileKey {
        self.all[rng.below(self.all.len())]
    }

    /// Closes an epoch of `keys` and their edge neighbours (an edit moves shared border
    /// vertices, so the tiles across them change too), bumping their versions.
    fn close(&mut self, mesh: &BaseMesh, seeds: &[TileKey]) {
        let around = seeds.iter().flat_map(|key| {
            let place = Place::of(key);
            let neighbours = place.neighbours(mesh.topology()).into_iter().flatten();
            std::iter::once(place).chain(neighbours)
        });
        let keys: BTreeSet<TileKey> = around.filter_map(|p| p.key(self.finest)).collect();
        let seq = self.next_seq;
        self.next_seq += 1;
        let version = u32::try_from(seq + 1).expect("few epochs");
        for key in &keys {
            self.versions.insert(*key, version);
        }
        self.undrained.push(Epoch { seq, keys });
    }

    /// What a drain gives: the epochs since the last, or a resync if they were lost.
    fn drain(&mut self) -> (Drained, Vec<Epoch>) {
        let epochs = std::mem::take(&mut self.undrained);
        if std::mem::take(&mut self.lost) {
            (Drained::Resync, Vec::new())
        } else {
            (Drained::Epochs(epochs.clone()), epochs)
        }
    }

    /// `key`'s version.
    fn version(&self, key: &TileKey) -> u32 {
        self.versions.get(key).copied().unwrap_or(0)
    }
}

impl TileSource for Model {
    type Error = Infallible;

    fn read(&mut self, key: TileKey, out: &mut [GpuVertex]) -> Result<Option<f32>, Infallible> {
        let version = self.version(&key);
        let tag = tag(&key, version);
        out.fill(tag);
        Ok(Some(tag.position[0]))
    }
}

/// The vertex the model writes for `key` at `version`: the version and the place, exactly
/// representable (and the version again in the reserved word, to read back).
fn tag(key: &TileKey, version: u32) -> GpuVertex {
    let small = |v: u32| f32::from(u16::try_from(v).expect("small values"));
    GpuVertex {
        position: [
            small(version),
            small(key.face().index()),
            f32::from(key.level().get()),
        ],
        roughness: 1.0,
        normal: [f32::from(key.tx()), f32::from(key.ty()), 0.0],
        // Exact, to read the version back from.
        reserved: version,
    }
}

/// Level `l`.
fn level(l: u8) -> Level {
    Level::new(l).expect("a level")
}

/// The configuration of the tiny-budget tests, `extra` slot buffers beyond the floor's.
///
/// Slot buffers hold two side-65 slots, the staging ring four such tiles, and every epoch of
/// two or more side-65 tiles goes through shadow slots.
fn tiny(mesh: &BaseMesh, extra: u64) -> ResidencyConfig {
    let big = SizeClass::of(level(FINEST)).slot_bytes();
    let faces = u64::from(mesh.face_count());
    let floor: u64 = (0..6)
        .filter_map(SizeClass::new)
        .map(|class| class.slot_bytes() * faces)
        .sum();
    ResidencyConfig {
        budget_bytes: floor + extra * 2 * big,
        buffer_bytes: 2 * big,
        staging_bytes: 2 * big,
        staging_cap: 4 * big,
        small_epoch_bytes: big + big / 2,
        request_capacity: 64,
        floor_top: level(5),
    }
}

/// What the GPU holds: every live node's tile, with the version its slot's first vertex
/// carries. Checks that the slot holds that tile's data, and that every vertex of it does.
fn gpu_view(gpu: &Gpu, residency: &Residency) -> BTreeMap<TileKey, u32> {
    let nodes: Vec<GpuNode> = gpu
        .read_buffer(residency.tree().buffer())
        .expect("the tree reads back");
    let mut slots: BTreeMap<(u8, u16), Vec<GpuVertex>> = BTreeMap::new();
    for class in (0..7).filter_map(SizeClass::new) {
        for (b, buffer) in residency.slot_buffers(class).iter().enumerate() {
            let vertices: Vec<GpuVertex> = gpu.read_buffer(buffer).expect("slots read back");
            slots.insert((class.get(), u16::try_from(b).expect("few")), vertices);
        }
    }
    let live = nodes.iter().filter(|node| node.flags & NODE_LIVE != 0);
    let mut view = BTreeMap::new();
    for node in live {
        let place = Place::from_packed(node.face, node.place).expect("a place");
        let key = place.key(residency.finest()).expect("a tile");
        let class = u8::try_from(node.slot >> 29).expect("3 bits");
        let buffer = u16::try_from((node.slot >> 21) & 0xFF).expect("8 bits");
        let index = node.slot & ((1 << 21) - 1);
        let class = SizeClass::new(class).expect("a class");
        let vertices = usize::try_from(class.vertices()).expect("small");
        let first = usize::try_from(index).expect("small") * vertices;
        let data = &slots[&(class.get(), buffer)][first..first + vertices];
        let version = data[0].reserved;
        assert_eq!(data[0], tag(&key, version), "{key:?}'s slot holds its data");
        assert!(
            data.iter().all(|v| *v == data[0]),
            "{key:?}'s slot holds one version, whole"
        );
        assert!(view.insert(key, version).is_none(), "{key:?} has one node");
    }
    view
}

/// One frame: begin, the feed's drain, the frame's use and requests, staging, then the
/// uploads encoded, submitted and waited for.
fn frame(
    gpu: &Gpu,
    residency: &mut Residency,
    model: &mut Model,
    drained: Drained,
    used: FrameUse,
    requests: Option<&Requests>,
) {
    residency.begin_frame(gpu).expect("a frame begins");
    residency.apply(drained);
    residency.use_frame(used);
    if let Some(requests) = requests {
        residency.want(requests);
    }
    let deadline = Instant::now() + Duration::from_secs(60);
    let uploads = residency.stage(gpu, model, deadline).expect("staged");
    submit(gpu, uploads, |_| {});
}

/// Encodes `uploads` around `passes`, submits them and waits for the GPU.
fn submit(gpu: &Gpu, uploads: FrameUploads, passes: impl FnOnce(&mut wgpu::CommandEncoder)) {
    gpu.scoped("frame", || {
        let mut encoder = gpu
            .device()
            .create_command_encoder(&CommandEncoderDescriptor::default());
        uploads.encode(&mut encoder);
        passes(&mut encoder);
        uploads.encode_requests(&mut encoder);
        gpu.queue().submit([encoder.finish()]);
        uploads.submitted();
    })
    .expect("the frame submits");
    gpu.device()
        .poll(PollType::wait_indefinitely())
        .expect("the GPU finishes");
}

/// Requests for `count` random tiles not resident, with random screen errors.
fn random_requests(rng: &mut Rng, model: &Model, residency: &Residency, count: usize) -> Requests {
    let wanted = (0..count).filter_map(|_| {
        let key = model.any(rng);
        (residency.slot(&key).is_none()).then(|| Request {
            key,
            screen_error: f32::from(u16::try_from(rng.below(1000)).expect("small")),
        })
    });
    Requests::new(wanted, 0)
}

/// A random cut among the resident tiles: up to `count` of them (none before any arrive).
fn random_cut(rng: &mut Rng, residency: &Residency, count: usize) -> BTreeSet<TileKey> {
    let resident: Vec<TileKey> = residency.resident().copied().collect();
    if resident.is_empty() {
        return BTreeSet::new();
    }
    let picked = (0..count).map(|_| resident[rng.below(resident.len())]);
    picked.collect()
}

#[test]
fn the_floor_is_resident_after_the_first_frame_whole() {
    GpuTest::run(|gpu| {
        let mesh = cube_sphere(1).expect("a mesh");
        let mut model = Model::new(&mesh, level(FINEST));
        let config = ResidencyConfig::default();
        let mut residency = Residency::new(gpu, &mesh, level(FINEST), config).expect("residency");
        assert!(!residency.floor_ready(), "nothing before the first frame");
        frame(
            gpu,
            &mut residency,
            &mut model,
            Drained::Epochs(Vec::new()),
            FrameUse::default(),
            None,
        );
        assert!(residency.floor_ready(), "the floor after it");
        let view = gpu_view(gpu, &residency);
        let floor: BTreeSet<TileKey> = model
            .all
            .iter()
            .filter(|key| key.level() <= level(5))
            .copied()
            .collect();
        let on_gpu: BTreeSet<TileKey> = view.keys().copied().collect();
        assert_eq!(
            on_gpu, floor,
            "every display tile up to level 5, nothing else"
        );
        assert_eq!(residency.stats().floor, floor.len(), "counted as the floor");
    });
}

#[test]
fn random_arrivals_under_a_tiny_budget_never_evict_the_floor_the_cut_or_a_drawn_childs_parent() {
    GpuTest::run(|gpu| {
        let mesh = cube_sphere(1).expect("a mesh");
        let mut model = Model::new(&mesh, level(FINEST));
        let config = tiny(&mesh, 3);
        let budget = config.budget_bytes;
        let mut residency = Residency::new(gpu, &mesh, level(FINEST), config).expect("residency");
        let mut rng = Rng(11);
        let mut drawn: BTreeSet<TileKey> = BTreeSet::new();
        let mut evicted = 0;
        for step in 0..300 {
            let count = 1 + rng.below(3);
            let cut = random_cut(&mut rng, &residency, count);
            // What must survive this frame: the floor, the cut, and the resident parents of
            // the cut's and the drawn tiles.
            let parents = cut.iter().chain(&drawn).filter_map(|key| {
                let parent = Place::of(key).parent()?.key(level(FINEST))?;
                residency.slot(&parent).map(|_| parent)
            });
            let mut kept: BTreeSet<TileKey> = parents.collect();
            kept.extend(cut.iter().copied());
            let asked = 1 + rng.below(6);
            let requests = random_requests(&mut rng, &model, &residency, asked);
            if rng.one_in(5) {
                let seed = model.any(&mut rng);
                model.close(&mesh, &[seed]);
            }
            let (drained, _) = model.drain();
            let before: BTreeSet<TileKey> = residency.resident().copied().collect();
            frame(
                gpu,
                &mut residency,
                &mut model,
                drained,
                FrameUse::new(cut.clone(), drawn.clone()),
                Some(&requests),
            );
            let after = gpu_view(gpu, &residency);
            let cpu: BTreeSet<TileKey> = residency.resident().copied().collect();
            let on_gpu: BTreeSet<TileKey> = after.keys().copied().collect();
            assert_eq!(on_gpu, cpu, "step {step}: the GPU's tree is the CPU's");
            assert!(residency.floor_ready(), "step {step}: the floor stays");
            for key in &kept {
                assert!(after.contains_key(key), "step {step}: {key:?} was evicted");
            }
            evicted += before.difference(&cpu).count();
            assert!(
                residency.stats().allocated_bytes <= budget,
                "step {step}: within the budget"
            );
            drawn = cut;
        }
        assert!(
            evicted > 50,
            "the budget is tiny enough to evict: {evicted}"
        );
    });
}

#[test]
fn a_request_waits_while_every_slot_it_could_take_is_protected() {
    GpuTest::run(|gpu| {
        let mesh = cube_sphere(1).expect("a mesh");
        let mut model = Model::new(&mesh, level(FINEST));
        // Room for two side-65 tiles beyond the floor.
        let config = tiny(&mesh, 1);
        let mut residency = Residency::new(gpu, &mesh, level(FINEST), config).expect("residency");
        let finest: Vec<TileKey> = model
            .all
            .iter()
            .filter(|key| key.level() == level(FINEST))
            .copied()
            .collect();
        let ask = |keys: &[TileKey]| {
            let requests = keys.iter().map(|&key| Request {
                key,
                screen_error: 1.0,
            });
            Requests::new(requests, 0)
        };
        let two = ask(&finest[..2]);
        // The floor fills most of the first frame's staging; the two take a frame or two more.
        for _ in 0..3 {
            frame(
                gpu,
                &mut residency,
                &mut model,
                Drained::Epochs(Vec::new()),
                FrameUse::default(),
                Some(&two),
            );
        }
        assert!(
            residency.slot(&finest[0]).is_some() && residency.slot(&finest[1]).is_some(),
            "both arrive"
        );
        let cut: BTreeSet<TileKey> = finest[..2].iter().copied().collect();
        let third = ask(&finest[2..3]);
        for _ in 0..3 {
            frame(
                gpu,
                &mut residency,
                &mut model,
                Drained::Epochs(Vec::new()),
                FrameUse::new(cut.clone(), cut.clone()),
                Some(&third),
            );
        }
        assert!(residency.slot(&finest[2]).is_none(), "the third waits");
        assert!(
            residency.stats().starved >= 3,
            "counted each frame: {:?}",
            residency.stats()
        );
        let view = gpu_view(gpu, &residency);
        assert!(
            cut.iter().all(|key| view.contains_key(key)),
            "the cut stays"
        );
        // Once the cut moves on, it comes in.
        frame(
            gpu,
            &mut residency,
            &mut model,
            Drained::Epochs(Vec::new()),
            FrameUse::default(),
            Some(&third),
        );
        assert!(residency.slot(&finest[2]).is_some(), "the third arrives");
    });
}

/// Checks that every delivered epoch is applied to all or none of the tiles the GPU holds.
fn assert_whole(view: &BTreeMap<TileKey, u32>, delivered: &[Epoch], step: usize) {
    for epoch in delivered {
        let applied = u32::try_from(epoch.seq + 1).expect("few epochs");
        let shown: Vec<(TileKey, bool)> = epoch
            .keys
            .iter()
            .filter_map(|key| view.get(key).map(|&v| (*key, v >= applied)))
            .collect();
        let new = shown.iter().filter(|(_, new)| *new).count();
        assert!(
            new == 0 || new == shown.len(),
            "step {step}: epoch {} half applied: {shown:?}",
            epoch.seq
        );
    }
}

/// Checks that every tile the GPU holds shows the store's data, when no epoch is pending.
fn assert_current(view: &BTreeMap<TileKey, u32>, model: &Model, step: usize) {
    for (key, version) in view {
        assert_eq!(
            *version,
            model.version(key),
            "step {step}: {key:?} shows old data with nothing pending"
        );
    }
}

/// Checks that epochs apply whole, with `extra` slot buffers beyond the floor's.
///
/// Random epochs, small (a tile and its neighbours) and large (several of them), are delivered
/// as they close, with random requests and cuts: after every frame each delivered epoch is applied
/// to all of its resident tiles or none, and with nothing pending every tile shows the store's
/// data.
fn epochs_apply_whole(extra: u64, seed: u64) {
    GpuTest::run(|gpu| {
        let mesh = cube_sphere(1).expect("a mesh");
        let mut model = Model::new(&mesh, level(FINEST));
        let config = tiny(&mesh, extra);
        let mut residency = Residency::new(gpu, &mesh, level(FINEST), config).expect("residency");
        let mut rng = Rng(seed);
        let mut delivered: Vec<Epoch> = Vec::new();
        let mut drawn: BTreeSet<TileKey> = BTreeSet::new();
        let (mut pending, mut settled) = (0, 0);
        for step in 0..300 {
            for _ in 0..rng.below(3) {
                let count = 1 + rng.below(4);
                let seeds: Vec<TileKey> = (0..count).map(|_| model.any(&mut rng)).collect();
                model.close(&mesh, &seeds);
            }
            let (drained, epochs) = model.drain();
            delivered.extend(epochs);
            let cut = random_cut(&mut rng, &residency, 2);
            let asked = rng.below(5);
            let requests = random_requests(&mut rng, &model, &residency, asked);
            frame(
                gpu,
                &mut residency,
                &mut model,
                drained,
                FrameUse::new(cut.clone(), drawn.clone()),
                Some(&requests),
            );
            let view = gpu_view(gpu, &residency);
            assert_whole(&view, &delivered, step);
            if residency.stats().pending_epochs == 0 {
                assert_current(&view, &model, step);
                settled += 1;
            } else {
                pending += 1;
            }
            drawn = cut;
        }
        assert!(
            pending > 10,
            "epochs stayed pending over several frames: {pending}"
        );
        assert!(settled > 10, "and settled: {settled}");
        assert!(residency.stats().forced > 0, "and some were forced whole");
    });
}

#[test]
fn an_epoch_is_never_half_applied_under_a_tiny_budget() {
    epochs_apply_whole(3, 23);
}

#[test]
fn an_epoch_is_never_half_applied_with_room_for_shadow_slots() {
    epochs_apply_whole(40, 29);
}

/// Checks that resyncs recover, with `extra` slot buffers beyond the floor's.
///
/// Random epochs, a seventh of the drains lost to a resync, then quiet frames: every tile kept ends
/// up showing the store's data, and the floor never leaves (its tiles keep their nodes throughout).
fn resyncs_recover(extra: u64, seed: u64) {
    GpuTest::run(|gpu| {
        let mesh = cube_sphere(1).expect("a mesh");
        let mut model = Model::new(&mesh, level(FINEST));
        let config = tiny(&mesh, extra);
        let mut residency = Residency::new(gpu, &mesh, level(FINEST), config).expect("residency");
        let mut rng = Rng(seed);
        let mut drawn: BTreeSet<TileKey> = BTreeSet::new();
        let mut cut: BTreeSet<TileKey> = BTreeSet::new();
        let mut floor_nodes: BTreeMap<TileKey, u32> = BTreeMap::new();
        for step in 0..200 {
            let quiet = step >= 150;
            if !quiet {
                for _ in 0..rng.below(3) {
                    let seed_key = model.any(&mut rng);
                    model.close(&mesh, &[seed_key]);
                }
                // The renderer falls behind: the epochs since its last drain are lost.
                model.lost |= rng.one_in(7);
            }
            let (drained, _) = model.drain();
            if !quiet {
                cut = random_cut(&mut rng, &residency, 2);
            }
            let asked = rng.below(4);
            let requests = random_requests(&mut rng, &model, &residency, asked);
            frame(
                gpu,
                &mut residency,
                &mut model,
                drained,
                FrameUse::new(cut.clone(), drawn.clone()),
                Some(&requests),
            );
            let floor = model.all.iter().filter(|key| key.level() <= level(5));
            for key in floor {
                let node = residency
                    .tree()
                    .node(Place::of(key))
                    .unwrap_or_else(|| panic!("step {step}: floor tile {key:?} left"));
                assert_eq!(
                    *floor_nodes.entry(*key).or_insert(node),
                    node,
                    "step {step}: floor tile {key:?} left and came back"
                );
            }
            drawn = cut.clone();
        }
        assert!(residency.stats().resyncs > 5, "resynced often");
        assert_eq!(residency.stats().pending_epochs, 0, "settled");
        assert_current(&gpu_view(gpu, &residency), &model, 200);
    });
}

#[test]
fn a_resync_recovers_every_tile_kept_under_a_tiny_budget() {
    resyncs_recover(6, 5);
}

#[test]
fn a_resync_recovers_every_tile_kept_with_room_for_shadow_slots() {
    resyncs_recover(40, 31);
}

/// A compute shader standing in for the level choice: every live node with a child that isn't
/// resident requests it, with the node's error as its screen error.
const REQUEST_CHILDREN: &str = r"
// The tile tree, read.
@group(0) @binding(0) var<storage, read> nodes: array<TileNode>;
// The request buffer, appended to.
@group(0) @binding(1) var<storage, read_write> requests: TileRequests;
// The finest level: no tile below it.
@group(0) @binding(2) var<uniform> finest: vec4<u32>;

// One invocation per node: requests each missing child of a live node.
@compute @workgroup_size(64)
fn request_children(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= arrayLength(&nodes) {
        return;
    }
    let node = nodes[id.x];
    let level = node.place >> 28u;
    if (node.flags & NODE_LIVE) == 0u || level >= finest.x {
        return;
    }
    let tx = (node.place >> 14u) & 0x3fffu;
    let ty = node.place & 0x3fffu;
    // Whole-face levels (up to 6) have one child, the finer ones four.
    var count = 1u;
    if level >= 6u {
        count = 4u;
    }
    for (var c = 0u; c < count; c++) {
        if node.children[c] != NONE {
            continue;
        }
        var cx = tx;
        var cy = ty;
        if count == 4u {
            cx = tx * 2u + (c & 1u);
            cy = ty * 2u + (c >> 1u);
        }
        let place = ((level + 1u) << 28u) | (cx << 14u) | cy;
        let i = atomicAdd(&requests.count, 1u);
        if i < arrayLength(&requests.entries) {
            requests.entries[i] = TileRequest(node.face, place, node.error + 1.0, 0u);
        } else {
            atomicAdd(&requests.dropped, 1u);
        }
    }
}
";

#[test]
fn requests_the_gpu_writes_are_read_back_a_frame_later_and_brought_in() {
    GpuTest::run(|gpu| {
        let mesh = cube_sphere(1).expect("a mesh");
        let mut model = Model::new(&mesh, level(FINEST));
        let config = ResidencyConfig {
            request_capacity: 16,
            ..ResidencyConfig::default()
        };
        let mut residency = Residency::new(gpu, &mesh, level(FINEST), config).expect("residency");
        let source = format!(
            "{}{}{REQUEST_CHILDREN}",
            crate::tree::NODE_WGSL,
            REQUESTS_WGSL
        );
        let pipeline = gpu
            .compute_pipeline(
                "request children",
                ShaderSource::Wgsl(source.into()),
                "request_children",
            )
            .expect("the stand-in compiles");
        let finest = gpu.device().create_buffer_init(&BufferInitDescriptor {
            label: Some("finest"),
            contents: bytemuck::cast_slice(&[u32::from(FINEST), 0, 0, 0]),
            usage: BufferUsages::UNIFORM,
        });
        let all = model.all.len();
        let mut frames = 0;
        let mut dropped = 0;
        while residency.stats().resident < all {
            frames += 1;
            assert!(frames < 40, "everything arrives: {:?}", residency.stats());
            residency.begin_frame(gpu).expect("begins");
            if let Some(requests) = residency.take_requests() {
                dropped += requests.dropped();
                assert!(
                    requests.iter().all(|r| residency.slot(&r.key).is_none()),
                    "only missing tiles are requested"
                );
                residency.want(&requests);
            }
            let deadline = Instant::now() + Duration::from_secs(60);
            let uploads = residency.stage(gpu, &mut model, deadline).expect("staged");
            let bind = gpu.device().create_bind_group(&BindGroupDescriptor {
                label: Some("request children"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: residency.tree().buffer().as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: residency.request_buffer().as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: finest.as_entire_binding(),
                    },
                ],
            });
            let groups = residency.tree().capacity().div_ceil(64);
            submit(gpu, uploads, |encoder| {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(groups, 1, 1);
            });
        }
        assert!(
            frames > 2,
            "tiles came down the levels, a frame behind each: {frames}"
        );
        assert!(
            dropped > 0,
            "a small request buffer drops what doesn't fit, and counts it"
        );
        assert_eq!(
            gpu_view(gpu, &residency).len(),
            all,
            "and the GPU holds them all"
        );
    });
}
