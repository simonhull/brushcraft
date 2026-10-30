//! The normal pass against the CPU's normals, and across every seam.
//!
//! The fixture is the subdivided pentagonal prism, whose base vertices have valence 3 (the
//! prism's corners), 4 (its edges and quads) and 5 (its pentagons' centres), sculpted with
//! random corrections at a coarse level and the finest, and made resident through residency
//! from a source that reads the store's rings: what the app uploads, through the same path.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use geom::fixtures::{ring_normals, subdivided_pentagonal_prism};
use geom::{BaseMesh, Channel, FaceId, Level, TileKey, Vec3};
use store::{CacheError, Drained, Project, ProjectConfig, StoreConfig, TileStore};
use wgpu::CommandEncoderDescriptor;

use super::*;
use crate::gpu::GpuTest;
use crate::residency::{Request, Requests, Residency, ResidencyConfig};
use crate::upload::{TileData, TileDetail, TileSource};

mod records;

/// The finest level: four tiles a face, so seams inside faces as well as across them.
const FINEST: u8 = 7;

/// The angle the GPU's normals may be off the CPU's, in radians.
///
/// The arithmetic is the same; what differs is rounding. The GPU's square root and division are
/// within a few ULP, not correctly rounded, and its compiler may fuse a product into the
/// subtraction of a cross product (one rounding fewer). Each is a relative error of a few times
/// `f32::EPSILON` (1.2e-7) in a component, so a few 1e-7 rad, scaled by how ill-conditioned the
/// cross product is (`|du| |dv| / |du × dv|`, near 1 for the sculpted prism's quads, a few at
/// worst). 1e-5 rad leaves room for that conditioning and nothing more: a wrong stencil, a ring
/// summed out of order or a missing term is off by far more (a degree is 1.7e-2 rad).
const MAX_ANGLE: f64 = 1e-5;

/// A fresh recovery directory under the system temp dir, removed with everything in it on drop.
struct TempDir(PathBuf);

impl TempDir {
    /// Names a directory for this process and `name` (the project creates it).
    fn new(name: &str) -> Self {
        let dir =
            std::env::temp_dir().join(format!("brushcraft-normals-{}-{name}", std::process::id()));
        drop(std::fs::remove_dir_all(&dir));
        Self(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        drop(std::fs::remove_dir_all(&self.0));
    }
}

/// Level `l`.
fn level(l: u8) -> Level {
    Level::new(l).expect("a level")
}

/// A small deterministic generator (xorshift64*), so the sculpt is the same every run.
struct Rng(u64);

impl Rng {
    /// The next value in `0..n`.
    fn below(&mut self, n: u32) -> u32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let x = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        u32::try_from((x >> 32) % u64::from(n)).expect("below n")
    }

    /// A value in `-scale..scale`.
    fn signed(&mut self, scale: f32) -> f32 {
        let unit = f32::from(u16::try_from(self.below(1 << 16)).expect("16 bits")) / 65_536.0;
        (unit * 2.0 - 1.0) * scale
    }
}

/// A project over the subdivided prism, finest level [`FINEST`], sculpted with random
/// corrections at level 3 and at the finest level.
fn sculpted_prism(dir: &TempDir) -> (Project, BaseMesh) {
    let mesh = subdivided_pentagonal_prism().expect("a mesh");
    let config = ProjectConfig {
        store: StoreConfig::new(256 << 20, 1),
        recovery_dir: dir.0.clone(),
    };
    let mut project = Project::new(mesh.clone(), config).expect("a project");
    let store = project.store();
    let mut rng = Rng(0x5EED_1234_ABCD_0001);
    store.begin_op().expect("an op");
    for face in mesh.faces() {
        for (l, count, scale) in [(3, 20, 0.02), (FINEST, 200, 0.002)] {
            let n = level(l).face_quads();
            for _ in 0..count {
                let (i, j) = (rng.below(n + 1), rng.below(n + 1));
                let c = Vec3::new(rng.signed(scale), rng.signed(scale), rng.signed(scale));
                store
                    .set_correction(face, level(l), i, j, c)
                    .expect("a correction");
            }
        }
    }
    store.close_epoch().expect("closed");
    (project, mesh)
}

/// Every finest tile of `mesh`.
fn finest_tiles(mesh: &BaseMesh) -> Vec<TileKey> {
    let finest = level(FINEST);
    let per_side = u16::try_from(finest.tiles_per_side()).expect("small");
    mesh.faces()
        .flat_map(|face| {
            (0..per_side).flat_map(move |ty| {
                (0..per_side).map(move |tx| {
                    TileKey::displacement(face, finest, tx, ty).expect("a finest tile")
                })
            })
        })
        .collect()
}

/// A source of finest tiles only, each as its ring from the store; display tiles it has none of.
/// One tile's record can be changed after it is written, for the bite checks.
struct RingSource<'a> {
    /// The store.
    store: &'a mut TileStore,
    /// A tile whose record has this word changed by adding this to its `f32`.
    nudge: Option<(TileKey, usize, f32)>,
}

impl TileSource for RingSource<'_> {
    type Error = CacheError;

    fn read(&mut self, key: TileKey, out: &mut TileData) -> Result<Option<TileDetail>, CacheError> {
        if key.channel() == Channel::DISPLAY {
            return Ok(None);
        }
        let block = self.store.positions_with_ring(key)?;
        out.ring(self.store.mesh(), &block);
        if let Some((nudged, word, by)) = self.nudge
            && nudged == key
        {
            let record = out.ring_record_mut().expect("a ring was written");
            let w = &mut record.words_mut()[word];
            *w = (f32::from_bits(*w) + by).to_bits();
        }
        Ok(Some(TileDetail::FINEST))
    }
}

/// The record word holding the x of block vertex `(i, j)` of a tile of `q` quads a side.
fn position_word(q: u32, i: i64, j: i64) -> usize {
    let padded = i64::from(q) + 3;
    let index = (j + 1) * padded + i + 1;
    HEADER_WORDS + 3 * usize::try_from(index).expect("on the block")
}

/// Stages and submits one frame of `residency` from `source`, wanting `keys`.
fn frame(gpu: &Gpu, residency: &mut Residency, source: &mut RingSource<'_>, keys: &[TileKey]) {
    residency.begin_frame(gpu).expect("begins");
    let requests = keys.iter().map(|&key| Request {
        key,
        screen_error: 2.0,
    });
    residency.want(&Requests::new(requests, 0));
    let deadline = Instant::now() + Duration::from_secs(120);
    let uploads = residency.stage(gpu, source, deadline).expect("staged");
    gpu.scoped("frame", || {
        let encoder = gpu
            .device()
            .create_command_encoder(&CommandEncoderDescriptor::default());
        uploads.submit(gpu.queue(), encoder, |_| {});
    })
    .expect("submits");
}

/// Residency over `mesh` with `keys` resident, read from `source`: the floor's frame (which
/// finds no display tile), then the one staging the requests.
fn resident(
    gpu: &Gpu,
    mesh: &BaseMesh,
    source: &mut RingSource<'_>,
    keys: &[TileKey],
) -> Residency {
    let mut residency =
        Residency::new(gpu, mesh, level(FINEST), ResidencyConfig::default()).expect("residency");
    for _ in 0..2 {
        frame(gpu, &mut residency, source, keys);
    }
    for key in keys {
        assert!(residency.slot(key).is_some(), "{key:?} resident");
    }
    residency
}

/// Every tile of `keys`' vertices as its slot holds them.
fn slots(gpu: &Gpu, residency: &Residency, keys: &[TileKey]) -> BTreeMap<TileKey, Vec<GpuVertex>> {
    let mut buffers: BTreeMap<(u8, u16), Vec<GpuVertex>> = BTreeMap::new();
    keys.iter()
        .map(|key| {
            let slot = residency.slot(key).expect("resident");
            let class = slot.class();
            let all = buffers
                .entry((class.get(), slot.buffer()))
                .or_insert_with(|| {
                    let buffer = &residency.slot_buffers(class)[usize::from(slot.buffer())];
                    gpu.read_buffer(buffer).expect("reads back")
                });
            let n = usize::try_from(class.vertices()).expect("small");
            let first = usize::try_from(slot.index()).expect("small") * n;
            (*key, all[first..first + n].to_vec())
        })
        .collect()
}

/// The angle between `a` and `b`, in radians, in `f64`.
fn angle(a: [f32; 3], b: Vec3) -> f64 {
    let a = a.map(f64::from);
    let b = [f64::from(b.x), f64::from(b.y), f64::from(b.z)];
    let cross = [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    let sine = cross.iter().map(|c| c * c).sum::<f64>().sqrt();
    let cosine: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
    sine.atan2(cosine)
}

/// What [`seams`] found: how many copies of a vertex it compared, how many across faces, how
/// many round a base vertex, and how many differed.
#[derive(Debug, Default)]
struct Seams {
    /// Pairs compared.
    pairs: usize,
    /// Of them, pairs in different faces.
    across_faces: usize,
    /// Of them, copies of a base vertex (every face round it holds one).
    base_vertices: usize,
    /// Pairs whose normal's (or position's) bits differ.
    differ: usize,
}

/// Compares every border vertex of every tile in `vertices` with each other copy of it, in every
/// tile holding one (in this face and the faces across): their normals' and positions' bits.
fn seams(mesh: &BaseMesh, vertices: &BTreeMap<TileKey, Vec<GpuVertex>>) -> Seams {
    let finest = level(FINEST);
    let q = finest.face_quads().min(64);
    let side = q + 1;
    let topology = mesh.topology();
    let mut found = Seams::default();
    for (key, ours) in vertices {
        let (x0, y0) = (u32::from(key.tx()) * q, u32::from(key.ty()) * q);
        let border = (0..side)
            .flat_map(|j| (0..side).map(move |i| (i, j)))
            .filter(|&(i, j)| i == 0 || j == 0 || i == q || j == q);
        for (i, j) in border {
            let here = ours[usize::try_from(j * side + i).expect("small")];
            let copies = topology.copies(key.face(), finest, x0 + i, y0 + j);
            let faces: BTreeSet<FaceId> = copies.iter().map(|copy| copy.face).collect();
            for copy in &*copies {
                let theirs = TileKey::displacement(copy.face, finest, copy.tx, copy.ty)
                    .expect("a finest tile");
                if theirs == *key && (copy.i, copy.j) == (i, j) {
                    continue;
                }
                let there =
                    vertices[&theirs][usize::try_from(copy.j * side + copy.i).expect("small")];
                found.pairs += 1;
                if copy.face != key.face() {
                    found.across_faces += 1;
                }
                if faces.len() > 2 {
                    found.base_vertices += 1;
                }
                let bits =
                    |v: GpuVertex| (v.position.map(f32::to_bits), v.normal.map(f32::to_bits));
                if bits(here) != bits(there) {
                    found.differ += 1;
                }
            }
        }
    }
    found
}

#[test]
fn the_gpu_normals_are_the_cpus_within_a_tight_angle() {
    GpuTest::run(|gpu| {
        let dir = TempDir::new("cpu");
        let (mut project, mesh) = sculpted_prism(&dir);
        let keys = finest_tiles(&mesh);
        let store = project.store();
        let mut source = RingSource { store, nudge: None };
        let residency = resident(gpu, &mesh, &mut source, &keys);
        let held = slots(gpu, &residency, &keys);
        let mut valences = BTreeSet::new();
        let mut worst: f64 = 0.0;
        for key in &keys {
            let block = source.store.positions_with_ring(*key).expect("a ring");
            let corners = (0..4).filter_map(|c| block.corner(c));
            valences.extend(corners.map(geom::GatheredRing::valence));
            let mut record = RingRecord::default();
            record.write(&mesh, &block);
            assert_eq!(
                residency.upload_bytes(key),
                record.bytes().len() as u64 + JOB_BYTES,
                "{key:?}: residency counts its upload exactly"
            );
            let cpu = ring_normals(&mesh, &block);
            let gpu_vertices = &held[key];
            let q = i64::from(block.quads());
            let inner = (0..=q).flat_map(|j| (0..=q).map(move |i| (i, j)));
            for ((vertex, normal), (i, j)) in gpu_vertices.iter().zip(&cpu).zip(inner) {
                assert_eq!(
                    vertex.position.map(f32::to_bits),
                    block.get(i, j).to_bits(),
                    "{key:?} ({i}, {j})'s position"
                );
                assert_eq!(vertex.roughness, 1.0, "a finest tile's roughness term");
                let off = angle(vertex.normal, *normal);
                assert!(
                    off <= MAX_ANGLE,
                    "{key:?} ({i}, {j}): GPU {:?} against CPU {normal:?}, {off:e} rad",
                    vertex.normal
                );
                worst = worst.max(off);
            }
        }
        assert_eq!(
            valences,
            BTreeSet::from([3, 4, 5]),
            "base vertices of valence 3, 4 and 5 at tile corners"
        );
        eprintln!("the GPU's normals are within {worst:e} rad of the CPU's");
        drop(project.close(false));
    });
}

#[test]
fn every_copy_of_a_vertex_has_the_same_normal_bits_and_a_nudged_ring_breaks_them() {
    GpuTest::run(|gpu| {
        let dir = TempDir::new("seams");
        let (mut project, mesh) = sculpted_prism(&dir);
        let keys = finest_tiles(&mesh);
        let store = project.store();
        let mut source = RingSource { store, nudge: None };
        let residency = resident(gpu, &mesh, &mut source, &keys);
        let found = seams(&mesh, &slots(gpu, &residency, &keys));
        assert!(found.pairs > 10_000, "seams compared: {found:?}");
        assert!(found.across_faces > 1_000, "face seams too: {found:?}");
        assert!(found.base_vertices > 100, "and base vertices: {found:?}");
        assert_eq!(found.differ, 0, "every copy agrees: {found:?}");

        // Bite: one tile's ring nudged one step off its j = 0 edge, under vertex (10, 0). That
        // tile's normal there changes; the tile across the edge reads its own copy, so the two
        // must now differ.
        let nudged = keys[5];
        let q = 64;
        source.nudge = Some((nudged, position_word(q, 10, -1), 1e-3));
        let nudged_residency = resident(gpu, &mesh, &mut source, &keys);
        let bitten = seams(&mesh, &slots(gpu, &nudged_residency, &keys));
        assert!(
            bitten.differ > 0,
            "a nudged ring shows at the seam: {bitten:?}"
        );
        drop(project.close(false));
    });
}

#[test]
fn a_stroke_brings_its_resident_neighbours_border_normals_in_the_same_epoch() {
    GpuTest::run(|gpu| {
        let dir = TempDir::new("neighbour");
        let (mut project, mesh) = sculpted_prism(&dir);
        let finest = level(FINEST);
        let face = FaceId::new(3);
        // Two tiles across the edge x = 64 of the face: the stroked one on the left, its
        // neighbour on the right.
        let stroked = TileKey::displacement(face, finest, 0, 0).expect("a tile");
        let neighbour = TileKey::displacement(face, finest, 1, 0).expect("a tile");
        let keys = [stroked, neighbour];
        let cursor = project.store().subscribe_stale();
        let mut source = RingSource {
            store: project.store(),
            nudge: None,
        };
        let mut residency = resident(gpu, &mesh, &mut source, &keys);
        let before = slots(gpu, &residency, &keys);

        // A stroke on the left tile alone: one vertex a step in from the shared edge, which the
        // neighbour holds no copy of but its ring reads.
        let (column, row) = (63, 30);
        source.store.begin_op().expect("an op");
        let correction = source
            .store
            .correction(stroked)
            .expect("reads")
            .get(column, row)
            .expect("a vertex");
        let lifted = correction + Vec3::new(0.0, 0.0, 0.05);
        source
            .store
            .set_correction(face, finest, column, row, lifted)
            .expect("written");
        source.store.close_epoch().expect("closed");
        let drained = source.store.drain_stale(&cursor);
        let Drained::Epochs(epochs) = &drained else {
            panic!("epochs, not a resync");
        };
        assert!(
            epochs.iter().any(|epoch| epoch.keys.contains(&neighbour)),
            "the epoch names the neighbour, whose ring moved: {epochs:?}"
        );
        residency.apply(drained);
        frame(gpu, &mut residency, &mut source, &keys);
        let after = slots(gpu, &residency, &keys);

        // The neighbour's vertices on the shared edge, and the stroked tile's copies of them.
        let edge: Vec<(usize, usize)> = (0..=64_usize).map(|r| (r * 65 + 64, r * 65)).collect();
        let bits = |v: &GpuVertex| v.normal.map(f32::to_bits);
        let changed = edge
            .iter()
            .filter(|&&(_, theirs)| {
                bits(&before[&neighbour][theirs]) != bits(&after[&neighbour][theirs])
            })
            .count();
        assert!(
            changed > 0,
            "the neighbour's normals on the shared edge moved"
        );
        for &(ours, theirs) in &edge {
            assert_eq!(
                bits(&after[&stroked][ours]),
                bits(&after[&neighbour][theirs]),
                "both tiles agree on the shared edge, in the same frame"
            );
        }
        // And the neighbour's normals are the store's new ones.
        let block = source.store.positions_with_ring(neighbour).expect("a ring");
        let cpu = ring_normals(&mesh, &block);
        for (vertex, normal) in after[&neighbour].iter().zip(&cpu) {
            assert!(
                angle(vertex.normal, *normal) <= MAX_ANGLE,
                "the neighbour's normals are current"
            );
        }
        source.store.unsubscribe_stale(cursor);
        drop(project.close(false));
    });
}

#[test]
fn a_record_is_its_header_its_positions_and_its_rings() {
    let dir = TempDir::new("record");
    let (mut project, mesh) = sculpted_prism(&dir);
    let store = project.store();
    // Face 0's first tile: block corner 0 is a base vertex, the others aren't.
    let key = TileKey::displacement(FaceId::new(0), level(FINEST), 0, 0).expect("a tile");
    let block = store.positions_with_ring(key).expect("a ring");
    let mut record = RingRecord::default();
    record.write(&mesh, &block);
    let words: &[u32] = bytemuck::cast_slice(record.bytes());
    let ring = block.corner(0).expect("a base vertex at corner 0");
    let points = 3 * 67 * 67;
    let length = HEADER_WORDS + points + 3 * ring.valence();
    assert_eq!(record.quads(), 64);
    assert_eq!(words.len(), length.next_multiple_of(2), "padded to 8 bytes");
    assert_eq!(
        words[HEADER_STENCILS], STENCIL_CORNER,
        "corner 0 reads its ring"
    );
    assert_eq!(
        words[HEADER_STENCILS + STENCILS - 1],
        encode(OWN_AXES),
        "the inside's stencil is in the record too"
    );
    let header = |at: usize| usize::try_from(words[at]).expect("small");
    assert_eq!(
        header(HEADER_RINGS),
        HEADER_WORDS + points,
        "after the points"
    );
    assert_eq!(header(HEADER_VALENCES), ring.valence());
    let others: Vec<u32> = (1..4).map(|c| words[HEADER_VALENCES + c]).collect();
    assert_eq!(others, [0, 0, 0], "no other base vertex");
    let padded: Vec<u32> = block.padded().iter().flat_map(|p| p.to_bits()).collect();
    assert_eq!(
        words[HEADER_WORDS..HEADER_WORDS + points],
        padded[..],
        "the points' bits"
    );
    let edges: Vec<u32> = ring.edges().iter().flat_map(|p| p.to_bits()).collect();
    assert_eq!(
        words[HEADER_WORDS + points..length],
        edges[..],
        "the ring's"
    );
    eprintln!(
        "a finest tile's record: {} bytes, against {} of its vertices",
        record.bytes().len(),
        65 * 65 * size_of::<GpuVertex>()
    );
    drop(project.close(false));
}
