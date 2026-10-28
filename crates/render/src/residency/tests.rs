//! Residency on lavapipe, against a model of the store: what the GPU holds after each frame.
//!
//! The model's tiles carry their place and a version (one more than the last epoch naming
//! them) in their vertices, so a readback of the tree and the slots shows which data every
//! resident tile draws.

use std::collections::{BTreeMap, BTreeSet};
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
    /// The tiles the tests' level choice asks for, with their screen errors.
    asking: BTreeMap<TileKey, f32>,
    /// Tiles the model has nothing for: their reads give `None`.
    missing: BTreeSet<TileKey>,
    /// One read in this many fails (0: none), for the tests of failed frames.
    fail_one_in: usize,
    /// Reads done, to choose which fail.
    reads: usize,
    /// Tiles readied by `prepare` and not read since: reads take their data from here.
    prepared: BTreeMap<TileKey, u32>,
    /// Reads of tiles `prepare` didn't ready.
    unprepared: usize,
    /// Every epoch closed, delivered or lost.
    closed: Vec<Epoch>,
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
            asking: BTreeMap::new(),
            missing: BTreeSet::new(),
            fail_one_in: 0,
            reads: 0,
            prepared: BTreeMap::new(),
            unprepared: 0,
            closed: Vec::new(),
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
        let keys: Vec<TileKey> = keys.into_iter().collect();
        self.close_exact(&keys);
    }

    /// Closes an epoch of exactly `keys`, bumping their versions.
    fn close_exact(&mut self, keys: &[TileKey]) {
        let keys: BTreeSet<TileKey> = keys.iter().copied().collect();
        let seq = self.next_seq;
        self.next_seq += 1;
        let version = u32::try_from(seq + 1).expect("few epochs");
        for key in &keys {
            self.versions.insert(*key, version);
            // Data readied before the change is stale now.
            self.prepared.remove(key);
        }
        self.closed.push(Epoch {
            seq,
            keys: keys.clone(),
        });
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

/// A read the model was told to fail.
#[derive(Debug)]
struct Failure;

impl From<Failure> for ResidencyError {
    fn from(_: Failure) -> Self {
        Self::Invariant("the model failed a read, as told")
    }
}

impl TileSource for Model {
    type Error = Failure;

    fn prepare(&mut self, keys: &[TileKey]) -> Result<(), Failure> {
        for key in keys {
            self.prepared.insert(*key, self.version(key));
        }
        Ok(())
    }

    fn read(&mut self, key: TileKey, out: &mut [GpuVertex]) -> Result<Option<f32>, Failure> {
        self.reads += 1;
        if self.fail_one_in > 0 && self.reads.is_multiple_of(self.fail_one_in) {
            return Err(Failure);
        }
        let readied = self.prepared.remove(&key);
        if readied.is_none() {
            self.unprepared += 1;
        }
        let version = readied.unwrap_or_else(|| self.version(&key));
        if self.missing.contains(&key) {
            return Ok(None);
        }
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
/// Slot buffers hold two side-65 slots, the staging ring four such tiles, every epoch of two
/// or more side-65 tiles goes through shadow slots, and four such tiles may be forced into a
/// frame. The floor's classes take twice the floor (their shadow headroom), reserved first.
fn tiny(mesh: &BaseMesh, extra: u64) -> ResidencyConfig {
    let big = SizeClass::of(level(FINEST)).slot_bytes();
    let faces = u64::from(mesh.face_count());
    let floor: u64 = (0..6)
        .filter_map(SizeClass::new)
        .map(|class| 2 * class.slot_bytes() * faces)
        .sum();
    ResidencyConfig::new(
        floor + extra * 2 * big,
        2 * big,
        4 * big,
        big + big / 2,
        64,
        level(5),
    )
    .and_then(|config| config.with_buffer_bytes(2 * big))
    .expect("a valid configuration")
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
        uploads.submit(gpu.queue(), encoder);
    })
    .expect("the frame submits");
    gpu.device()
        .poll(PollType::wait_indefinitely())
        .expect("the GPU finishes");
}

/// The model's requests this frame, and `count` more random tiles not resident.
///
/// Those of earlier frames still missing stay asked for (a level choice asks for a tile until it
/// arrives; one in ten is given up, as a camera moves on); screen errors are random.
fn random_requests(
    rng: &mut Rng,
    model: &mut Model,
    residency: &Residency,
    count: usize,
) -> Requests {
    model.asking.retain(|key, _| residency.slot(key).is_none());
    let kept: Vec<TileKey> = model.asking.keys().copied().collect();
    for key in kept {
        if rng.one_in(10) {
            model.asking.remove(&key);
        }
    }
    for _ in 0..count {
        let key = model.any(rng);
        if residency.slot(&key).is_none() {
            let error = f32::from(u16::try_from(rng.below(1000)).expect("small"));
            model.asking.insert(key, error);
        }
    }
    let asked = model
        .asking
        .iter()
        .map(|(&key, &screen_error)| Request { key, screen_error });
    Requests::new(asked, 0)
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
fn a_budget_that_cannot_hold_the_floor_is_refused() {
    GpuTest::run(|gpu| {
        let mesh = cube_sphere(1).expect("a mesh");
        // Enough for the coarsest floor tiles, not for the rest.
        let config = ResidencyConfig::default()
            .with_budget(4096)
            .expect("an aligned budget");
        let refused = Residency::new(gpu, &mesh, level(FINEST), config).expect_err("too small");
        assert!(
            matches!(
                refused,
                ResidencyError::Config(ConfigError::FloorTooBig { tiles, slots, .. })
                    if tiles > slots
            ),
            "{refused}"
        );
        // The tiny budget the other tests use holds it.
        Residency::new(gpu, &mesh, level(FINEST), tiny(&mesh, 0)).expect("the floor fits");
    });
}

#[test]
fn random_arrivals_under_a_tiny_budget_never_evict_the_floor_the_cut_or_a_drawn_childs_parent() {
    GpuTest::run(|gpu| {
        let mesh = cube_sphere(1).expect("a mesh");
        let mut model = Model::new(&mesh, level(FINEST));
        let config = tiny(&mesh, 3);
        let budget = config.budget_bytes();
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
            let requests = random_requests(&mut rng, &mut model, &residency, asked);
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
        // Requests are staged from the frame after they're made; the floor fills most of the
        // first frame's staging; the two take a frame or two more.
        for _ in 0..4 {
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
        // The first of these frames only makes the request; the other three are starved.
        for _ in 0..4 {
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

/// Checks that, with nothing pending, every slot made is free or a resident tile's, once: no
/// slot leaked, none given back twice.
fn assert_census(residency: &Residency) {
    assert_eq!(
        residency.stats().pending_epochs,
        0,
        "nothing pending holds a slot"
    );
    let (made, free) = residency.pools.census();
    assert_eq!(
        made,
        free + residency.stats().resident as u64,
        "every slot is free or a resident tile's, once"
    );
    assert_floor_counts(residency);
}

/// Checks that the counts of floor tiles resident and unavailable match the sets, and say what
/// a scan of the floor would.
fn assert_floor_counts(residency: &Residency) {
    let floor = residency
        .entries
        .values()
        .filter(|entry| entry.floor)
        .count();
    assert_eq!(
        residency.stats().floor,
        floor,
        "the count of resident floor tiles is kept as they come and go"
    );
    let unavailable = residency.floor.intersection(&residency.unavailable).count();
    assert_eq!(
        residency.floor_unavailable, unavailable,
        "and of floor tiles the source had nothing for"
    );
    assert_eq!(
        residency.floor_settled(),
        residency
            .floor
            .iter()
            .all(|key| residency.entries.contains_key(key) || residency.unavailable.contains(key)),
        "the counts say what a scan would"
    );
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
            let requests = random_requests(&mut rng, &mut model, &residency, asked);
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
        assert_eq!(model.unprepared, 0, "every read was readied first");
        if extra < 10 {
            assert!(
                residency.stats().forced > 0,
                "a tiny budget stalls epochs, and forces them"
            );
        } else {
            assert!(
                residency.stats().shadowed > 50,
                "room for shadow slots is used: {:?}",
                residency.stats()
            );
        }
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
            let requests = random_requests(&mut rng, &mut model, &residency, asked);
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
            assert_whole(&gpu_view(gpu, &residency), &model.closed, step);
            drawn = cut.clone();
        }
        assert!(residency.stats().resyncs > 5, "resynced often");
        // Lost epochs included: a tile arriving while the resync is pending joins it.
        assert_whole(&gpu_view(gpu, &residency), &model.closed, 200);
        assert_eq!(residency.stats().pending_epochs, 0, "settled");
        assert_current(&gpu_view(gpu, &residency), &model, 200);
        assert_census(&residency);
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
        let config = ResidencyConfig::new(4 << 30, 16 << 20, 64 << 20, 8 << 20, 16, level(5))
            .expect("a valid configuration");
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

/// Runs one frame of nothing new: no epochs, no use, the model's open requests.
fn quiet_frame(gpu: &Gpu, residency: &mut Residency, model: &mut Model, asking: &[TileKey]) {
    let requests = Requests::new(
        asking.iter().map(|&key| Request {
            key,
            screen_error: 1.0,
        }),
        0,
    );
    frame(
        gpu,
        residency,
        model,
        Drained::Epochs(Vec::new()),
        FrameUse::default(),
        Some(&requests),
    );
}

#[test]
fn tiles_the_source_has_nothing_for_wait_for_an_epoch_naming_them() {
    GpuTest::run(|gpu| {
        let mesh = cube_sphere(1).expect("a mesh");
        let mut model = Model::new(&mesh, level(FINEST));
        let mut residency = Residency::new(gpu, &mesh, level(FINEST), ResidencyConfig::default())
            .expect("residency");
        let face = geom::FaceId::new(1);
        let floor = TileKey::display(face, level(3), 0, 0).expect("a floor tile");
        let fine = TileKey::displacement(face, level(FINEST), 1, 1).expect("a finest tile");
        let doomed = TileKey::displacement(face, level(FINEST), 0, 1).expect("a finest tile");
        model.missing.extend([floor, fine]);
        for _ in 0..3 {
            quiet_frame(gpu, &mut residency, &mut model, &[fine, doomed]);
        }
        assert!(
            residency.floor_ready(),
            "the floor is all there that can be"
        );
        assert!(
            residency.slot(&floor).is_none(),
            "the missing floor tile isn't"
        );
        assert!(
            residency.slot(&fine).is_none(),
            "nor the missing requested tile"
        );
        assert!(
            residency.slot(&doomed).is_some(),
            "the other request arrived"
        );
        assert_eq!(residency.stats().unavailable, 2, "both counted");
        assert_floor_counts(&residency);

        // The source has them now, but nothing says so: they aren't asked for again.
        model.missing.clear();
        let reads = model.reads;
        for _ in 0..2 {
            quiet_frame(gpu, &mut residency, &mut model, &[fine]);
        }
        assert_eq!(model.reads, reads, "an unavailable tile isn't read again");

        // An epoch naming them: the floor tile is wanted again at once, the request once the
        // level choice asks again.
        model.close(&mesh, &[floor]);
        model.close(&mesh, &[fine]);
        let (named, _) = model.drain();
        frame(
            gpu,
            &mut residency,
            &mut model,
            named,
            FrameUse::default(),
            None,
        );
        assert!(
            residency.slot(&floor).is_some(),
            "the floor tile came back with its epoch"
        );
        for _ in 0..2 {
            quiet_frame(gpu, &mut residency, &mut model, &[fine]);
        }
        assert!(
            residency.slot(&fine).is_some(),
            "and the request once asked again"
        );

        // A resident tile whose epoch leaves the source with nothing: dropped when it switches.
        model.missing.insert(doomed);
        model.close(&mesh, &[doomed]);
        let (drained, _) = model.drain();
        frame(
            gpu,
            &mut residency,
            &mut model,
            drained,
            FrameUse::default(),
            None,
        );
        assert!(residency.slot(&doomed).is_none(), "dropped");
        assert_eq!(residency.stats().dropped, 1, "and counted");
        let view = gpu_view(gpu, &residency);
        assert!(!view.contains_key(&doomed), "the GPU's tree lost it too");
        assert_floor_counts(&residency);
        assert_eq!(model.unprepared, 0, "every read was readied first");
    });
}

#[test]
fn a_resync_retries_the_floor_tiles_the_source_had_nothing_for() {
    GpuTest::run(|gpu| {
        let mesh = cube_sphere(1).expect("a mesh");
        let mut model = Model::new(&mesh, level(FINEST));
        let mut residency = Residency::new(gpu, &mesh, level(FINEST), ResidencyConfig::default())
            .expect("residency");
        let floor = TileKey::display(geom::FaceId::new(4), level(5), 0, 0).expect("a floor tile");
        model.missing.insert(floor);
        quiet_frame(gpu, &mut residency, &mut model, &[]);
        assert!(residency.slot(&floor).is_none(), "missing");
        assert_floor_counts(&residency);
        model.missing.clear();
        model.lost = true;
        let (drained, _) = model.drain();
        frame(
            gpu,
            &mut residency,
            &mut model,
            drained,
            FrameUse::default(),
            None,
        );
        quiet_frame(gpu, &mut residency, &mut model, &[]);
        assert!(residency.slot(&floor).is_some(), "retried by the resync");
        assert_floor_counts(&residency);
        assert_current(&gpu_view(gpu, &residency), &model, 0);
    });
}

#[test]
fn frames_in_flight_unwaited_still_apply_every_epoch_whole() {
    GpuTest::run(|gpu| {
        let mesh = cube_sphere(1).expect("a mesh");
        let mut model = Model::new(&mesh, level(FINEST));
        let config = tiny(&mesh, 20);
        let cap = config.staging_cap;
        let mut residency = Residency::new(gpu, &mesh, level(FINEST), config).expect("residency");
        let mut rng = Rng(41);
        for step in 0..150 {
            if step < 120 {
                for _ in 0..rng.below(3) {
                    let count = 1 + rng.below(3);
                    let seeds: Vec<TileKey> = (0..count).map(|_| model.any(&mut rng)).collect();
                    model.close(&mesh, &seeds);
                }
            }
            let (drained, _) = model.drain();
            let asked = rng.below(4);
            let requests = random_requests(&mut rng, &mut model, &residency, asked);
            residency.begin_frame(gpu).expect("begins");
            residency.apply(drained);
            residency.use_frame(FrameUse::default());
            residency.want(&requests);
            let deadline = Instant::now() + Duration::from_secs(60);
            let uploads = residency.stage(gpu, &mut model, deadline).expect("staged");
            // Submitted, never waited for: several frames are in flight, and the one poll a
            // frame (in begin_frame) brings the staging buffers back when the GPU is done.
            gpu.scoped("frame", || {
                let mut encoder = gpu
                    .device()
                    .create_command_encoder(&CommandEncoderDescriptor::default());
                uploads.encode(&mut encoder);
                uploads.submit(gpu.queue(), encoder);
            })
            .expect("submits");
            assert!(
                residency.ring.alive() <= cap,
                "step {step}: the ring stays within its cap"
            );
        }
        gpu.device()
            .poll(PollType::wait_indefinitely())
            .expect("the GPU finishes");
        for _ in 0..20 {
            quiet_frame(gpu, &mut residency, &mut model, &[]);
        }
        let view = gpu_view(gpu, &residency);
        assert_whole(&view, &model.closed, 150);
        assert_eq!(residency.stats().pending_epochs, 0, "settled");
        assert_current(&view, &model, 150);
        assert_census(&residency);
    });
}

#[test]
fn failed_and_dropped_frames_lose_only_their_own_tiles_and_leak_nothing() {
    GpuTest::run(|gpu| {
        let mesh = cube_sphere(1).expect("a mesh");
        let mut model = Model::new(&mesh, level(FINEST));
        let config = tiny(&mesh, 8);
        let cap = config.staging_cap;
        let mut residency = Residency::new(gpu, &mesh, level(FINEST), config).expect("residency");
        let mut rng = Rng(43);
        let (mut failed, mut dropped) = (0, 0);
        for step in 0..250 {
            let quiet = step >= 200;
            if !quiet {
                for _ in 0..rng.below(3) {
                    let count = 1 + rng.below(3);
                    let seeds: Vec<TileKey> = (0..count).map(|_| model.any(&mut rng)).collect();
                    model.close(&mesh, &seeds);
                }
            }
            model.fail_one_in = if !quiet && rng.one_in(6) { 3 } else { 0 };
            let (drained, _) = model.drain();
            let asked = rng.below(4);
            let requests = random_requests(&mut rng, &mut model, &residency, asked);
            residency.begin_frame(gpu).expect("begins");
            residency.apply(drained);
            residency.use_frame(FrameUse::default());
            residency.want(&requests);
            let deadline = Instant::now() + Duration::from_secs(60);
            match residency.stage(gpu, &mut model, deadline) {
                Err(_) => failed += 1,
                Ok(uploads) if !quiet && rng.one_in(6) => {
                    // The render thread gives up on it.
                    drop(uploads);
                    dropped += 1;
                }
                Ok(uploads) => {
                    submit(gpu, uploads, |_| {});
                    // A frame after a lost one writes the whole tree: what the GPU holds is
                    // then the CPU's, and every epoch is whole.
                    let view = gpu_view(gpu, &residency);
                    let cpu: BTreeSet<TileKey> = residency.resident().copied().collect();
                    let on_gpu: BTreeSet<TileKey> = view.keys().copied().collect();
                    assert_eq!(on_gpu, cpu, "step {step}: the GPU's tree is the CPU's");
                    assert_whole(&view, &model.closed, step);
                }
            }
            gpu.device()
                .poll(PollType::wait_indefinitely())
                .expect("the GPU finishes");
            assert!(
                residency.ring.alive() <= cap,
                "step {step}: no staging leaks"
            );
            assert!(
                residency.requests.made() <= 3,
                "step {step}: no readback leaks"
            );
        }
        assert!(
            failed > 5 && dropped > 5,
            "frames failed ({failed}) and were dropped ({dropped})"
        );
        assert!(residency.floor_ready(), "the floor came back");
        assert_eq!(residency.stats().pending_epochs, 0, "settled");
        assert_current(&gpu_view(gpu, &residency), &model, 250);
        assert!(
            residency.stats().lost_frames >= failed + dropped,
            "each counted"
        );
        assert_census(&residency);
        // One more frame hears the last map callbacks: then every buffer is back.
        quiet_frame(gpu, &mut residency, &mut model, &[]);
        residency.begin_frame(gpu).expect("begins");
        assert!(residency.ring.idle(), "every staging buffer came back");
        assert!(residency.requests.idle(), "every readback buffer came back");
    });
}

#[test]
fn a_frames_uploads_can_be_submitted_on_another_thread() {
    GpuTest::run(|gpu| {
        let mesh = cube_sphere(1).expect("a mesh");
        let mut model = Model::new(&mesh, level(FINEST));
        let mut residency = Residency::new(gpu, &mesh, level(FINEST), ResidencyConfig::default())
            .expect("residency");
        residency.begin_frame(gpu).expect("begins");
        let deadline = Instant::now() + Duration::from_secs(60);
        let uploads = residency.stage(gpu, &mut model, deadline).expect("staged");
        std::thread::scope(|scope| {
            scope.spawn(|| {
                let mut encoder = gpu
                    .device()
                    .create_command_encoder(&CommandEncoderDescriptor::default());
                uploads.encode(&mut encoder);
                uploads.submit(gpu.queue(), encoder);
            });
        });
        gpu.device()
            .poll(PollType::wait_indefinitely())
            .expect("the GPU finishes");
        assert_eq!(
            gpu_view(gpu, &residency).len(),
            residency.stats().floor,
            "the floor, uploaded from the render thread"
        );
    });
}

/// Stages one frame that has no time at all (its deadline has passed), with `used`, and
/// submits it.
fn rushed_frame(gpu: &Gpu, residency: &mut Residency, model: &mut Model, used: FrameUse) {
    residency.begin_frame(gpu).expect("begins");
    let (drained, _) = model.drain();
    residency.apply(drained);
    residency.use_frame(used);
    let uploads = residency.stage(gpu, model, Instant::now()).expect("staged");
    submit(gpu, uploads, |_| {});
}

#[test]
fn a_small_epoch_behind_a_pending_large_one_shows_in_the_next_frame() {
    GpuTest::run(|gpu| {
        let mesh = cube_sphere(1).expect("a mesh");
        let mut model = Model::new(&mesh, level(FINEST));
        let config = tiny(&mesh, 10);
        let mut residency = Residency::new(gpu, &mesh, level(FINEST), config).expect("residency");
        let face = geom::FaceId::new(0);
        let tiles: Vec<TileKey> = [(0, 0), (1, 0), (0, 1)]
            .into_iter()
            .map(|(tx, ty)| TileKey::displacement(face, level(FINEST), tx, ty).expect("a tile"))
            .collect();
        for _ in 0..4 {
            quiet_frame(gpu, &mut residency, &mut model, &tiles);
        }
        assert!(
            tiles.iter().all(|key| residency.slot(key).is_some()),
            "resident"
        );
        // A large epoch (three side-65 tiles, past the small epoch size) that a frame without
        // time leaves pending: no slot was missing, so it hasn't stalled.
        model.close_exact(&tiles);
        rushed_frame(gpu, &mut residency, &mut model, FrameUse::default());
        assert_eq!(residency.stats().pending_epochs, 1, "the large epoch waits");
        // Frames out of time aren't stalls: nothing is forced however many pass.
        for _ in 0..4 {
            rushed_frame(gpu, &mut residency, &mut model, FrameUse::default());
        }
        assert_eq!(residency.stats().pending_epochs, 1, "still waiting");
        assert_eq!(residency.stats().forced, 0, "and not forced");
        // A brush step over one of its tiles: it can't show without the rest, so the pending
        // epoch goes whole, now, deadline or not.
        model.close_exact(&tiles[..1]);
        rushed_frame(gpu, &mut residency, &mut model, FrameUse::default());
        assert_eq!(residency.stats().pending_epochs, 0, "both applied");
        assert_current(&gpu_view(gpu, &residency), &model, 0);
        assert_eq!(residency.stats().forced, 1, "forced once");
    });
}

/// Three finest tiles of face 0, and its level-5 floor tile, for the joining tests.
fn joining_tiles() -> (Vec<TileKey>, TileKey) {
    let face = geom::FaceId::new(0);
    let tiles = [(0, 0), (1, 0), (0, 1)]
        .into_iter()
        .map(|(tx, ty)| TileKey::displacement(face, level(FINEST), tx, ty).expect("a tile"))
        .collect();
    let floor = TileKey::display(face, level(5), 0, 0).expect("a floor tile");
    (tiles, floor)
}

#[test]
fn a_tile_arriving_while_an_epoch_naming_it_is_pending_joins_it() {
    GpuTest::run(|gpu| {
        let mesh = cube_sphere(1).expect("a mesh");
        let mut model = Model::new(&mesh, level(FINEST));
        let config = tiny(&mesh, 10);
        let mut residency = Residency::new(gpu, &mesh, level(FINEST), config).expect("residency");
        let (tiles, floor) = joining_tiles();
        model.missing.insert(floor);
        for _ in 0..4 {
            quiet_frame(gpu, &mut residency, &mut model, &tiles);
        }
        assert!(
            residency.slot(&floor).is_none(),
            "the floor tile is unavailable"
        );
        // A large epoch over the tiles and the floor tile, which the source has again: the floor
        // tile arrives whatever the deadline, the epoch waits for time.
        model.missing.clear();
        let mut named = tiles.clone();
        named.push(floor);
        model.close_exact(&named);
        rushed_frame(gpu, &mut residency, &mut model, FrameUse::default());
        assert_eq!(residency.stats().pending_epochs, 1, "the epoch waits");
        assert!(
            residency.slot(&floor).is_none(),
            "and the floor tile with it"
        );
        assert_whole(&gpu_view(gpu, &residency), &model.closed, 0);
        quiet_frame(gpu, &mut residency, &mut model, &[]);
        assert!(
            residency.slot(&floor).is_some(),
            "it arrives with the epoch"
        );
        assert_current(&gpu_view(gpu, &residency), &model, 1);
    });
}

#[test]
fn a_tile_arriving_while_a_resync_is_pending_joins_it() {
    GpuTest::run(|gpu| {
        let mesh = cube_sphere(1).expect("a mesh");
        let mut model = Model::new(&mesh, level(FINEST));
        let config = tiny(&mesh, 10);
        let mut residency = Residency::new(gpu, &mesh, level(FINEST), config).expect("residency");
        let (tiles, floor) = joining_tiles();
        model.missing.insert(floor);
        for _ in 0..4 {
            quiet_frame(gpu, &mut residency, &mut model, &tiles);
        }
        // An epoch over a kept tile and the floor tile, lost: the resync re-reads what it keeps
        // (the cut's tiles and the floor) and retries the floor tile, which may be new beside
        // the kept tiles' old data, so it waits with them.
        model.missing.clear();
        model.close_exact(&[tiles[0], floor]);
        model.lost = true;
        let cut: BTreeSet<TileKey> = tiles.iter().copied().collect();
        rushed_frame(
            gpu,
            &mut residency,
            &mut model,
            FrameUse::new(cut.clone(), cut),
        );
        assert_eq!(residency.stats().pending_epochs, 1, "the resync waits");
        assert!(
            residency.slot(&floor).is_none(),
            "and the floor tile with it"
        );
        assert_whole(&gpu_view(gpu, &residency), &model.closed, 0);
        quiet_frame(gpu, &mut residency, &mut model, &[]);
        assert!(
            residency.slot(&floor).is_some(),
            "it arrives with the resync"
        );
        assert_current(&gpu_view(gpu, &residency), &model, 1);
    });
}

#[test]
fn a_resync_of_the_floor_goes_through_its_shadow_headroom_not_forced() {
    GpuTest::run(|gpu| {
        let mesh = cube_sphere(1).expect("a mesh");
        let mut model = Model::new(&mesh, level(FINEST));
        // The floor is larger than a small epoch here: it can't go in place unforced.
        let config = tiny(&mesh, 2);
        let mut residency = Residency::new(gpu, &mesh, level(FINEST), config).expect("residency");
        quiet_frame(gpu, &mut residency, &mut model, &[]);
        let floor: Vec<TileKey> = residency.resident().copied().collect();
        model.close_exact(&floor);
        model.lost = true;
        let (drained, _) = model.drain();
        frame(
            gpu,
            &mut residency,
            &mut model,
            drained,
            FrameUse::default(),
            None,
        );
        assert_eq!(residency.stats().pending_epochs, 0, "re-read in one frame");
        assert_eq!(residency.stats().forced, 0, "into shadow slots, not forced");
        assert!(
            residency.stats().shadowed >= floor.len() as u64,
            "every floor tile shadowed"
        );
        assert_current(&gpu_view(gpu, &residency), &model, 0);
    });
}

#[test]
fn a_request_is_staged_the_frame_after_it_is_made_so_its_page_in_can_land() {
    GpuTest::run(|gpu| {
        let mesh = cube_sphere(1).expect("a mesh");
        let mut model = Model::new(&mesh, level(FINEST));
        let mut residency = Residency::new(gpu, &mesh, level(FINEST), ResidencyConfig::default())
            .expect("residency");
        let (tiles, _) = joining_tiles();
        quiet_frame(gpu, &mut residency, &mut model, &tiles);
        assert!(
            tiles.iter().all(|key| residency.slot(key).is_none()),
            "not in the frame that asked"
        );
        quiet_frame(gpu, &mut residency, &mut model, &tiles);
        assert!(
            tiles.iter().all(|key| residency.slot(key).is_some()),
            "in the next"
        );
    });
}
