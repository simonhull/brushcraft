//! What the tests share: a small base mesh and a meta made for it, store configurations, random
//! edits and the dense oracle they are checked against.

#![expect(clippy::missing_panics_doc, reason = "tests panic to fail")]

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Arc;

use geom::{BaseMesh, CorrectionMap, DenseEvaluation, FaceId, Level, Region, TileKey, Vec3};

use crate::file::Loc;
use crate::meta::{Meta, ProjectId, Role};
use crate::slot::EvaluatedSlot;
use crate::tiles::{StoreConfig, TileStore};
use crate::workers::Workers;

/// The project id of the tests' files.
pub(crate) const PROJECT: ProjectId =
    ProjectId::from_bits(0x7E57_0000_0000_0000_0000_0000_0000_0001);

/// The tests' default base mesh: the benchmark's sphere at its smallest, a cube of 6 faces.
///
/// # Panics
///
/// Never: the cube is a valid mesh.
pub(crate) fn mesh() -> Arc<BaseMesh> {
    Arc::new(geom::cube_sphere(1).unwrap())
}

/// A project file's meta for `mesh`, of project [`PROJECT`].
pub(crate) fn meta_for(mesh: &BaseMesh) -> Meta {
    Meta::new(PROJECT, Role::Project, mesh)
}

/// A project file's meta for [`mesh`].
pub(crate) fn meta() -> Meta {
    meta_for(&mesh())
}

/// Bytes of one evaluated slot: test chunks are sized in these.
pub(crate) const EVAL: usize = size_of::<EvaluatedSlot>();

/// The finest level the tests evaluate.
pub(crate) const TOP: u8 = 8;

/// The site the tests' ops carry.
pub(crate) const SITE: u16 = 3;

/// Deterministic random numbers (splitmix64).
pub(crate) struct Rng(pub(crate) u64);

impl Rng {
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n`; `n > 0`.
    pub(crate) fn below(&mut self, n: u32) -> u32 {
        u32::try_from(self.next_u64() % u64::from(n)).unwrap()
    }

    /// Uniform in `[-amplitude, amplitude)`, per component.
    pub(crate) fn vec3(&mut self, amplitude: f32) -> Vec3 {
        let mut signed =
            || f32::from(u16::try_from(self.next_u64() >> 48).unwrap()) / 32768.0 - 1.0;
        Vec3::new(
            signed() * amplitude,
            signed() * amplitude,
            signed() * amplitude,
        )
    }
}

/// One `set_correction` call.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Edit {
    /// The face.
    pub(crate) face: FaceId,
    /// The level.
    pub(crate) level: Level,
    /// The vertex column on the face's grid.
    pub(crate) i: u32,
    /// The vertex row on the face's grid.
    pub(crate) j: u32,
    /// The correction.
    pub(crate) c: Vec3,
}

impl Edit {
    /// The tiles the edit writes: every one holding a copy of its vertex.
    pub(crate) fn keys(&self, mesh: &BaseMesh) -> Vec<TileKey> {
        mesh.topology()
            .copies(self.face, self.level, self.i, self.j)
            .iter()
            .map(|copy| copy.key(self.level))
            .collect()
    }
}

/// Level `n`, which the tests know is valid.
pub(crate) fn level(n: u8) -> Level {
    Level::new(n).unwrap()
}

/// Evaluation workers of a store whose test isn't about how many there are: two, so that
/// evaluation runs in parallel without every store of a test run starting a worker per
/// hardware thread.
pub(crate) const TEST_THREADS: Workers = Workers::Fixed(NonZeroUsize::new(2).unwrap());

/// Exactly `n` evaluation workers.
///
/// # Panics
///
/// If `n` is 0.
pub(crate) fn fixed(n: usize) -> Workers {
    Workers::Fixed(NonZeroUsize::new(n).expect("a positive worker count"))
}

/// A configuration with pools of `chunks` chunks of `chunk_slots` evaluated slots.
///
/// The default kept predictions and stroke state come besides (their shares of the budget, see
/// [`StoreConfig::kept_prediction_bytes`] and [`StoreConfig::stroke_state_bytes`]), with
/// [`TEST_THREADS`] workers.
pub(crate) fn config(chunks: usize, chunk_slots: usize) -> StoreConfig {
    with_reserved_shares(
        StoreConfig::new(chunks * chunk_slots * EVAL, SITE)
            .with_chunk_bytes(chunk_slots * EVAL)
            .with_workers(TEST_THREADS),
    )
}

/// The bytes of `config`'s reserved shares: the kept predictions' and the stroke state's.
fn reserved(config: &StoreConfig) -> usize {
    config.kept_prediction_bytes().unwrap() + config.stroke_state_bytes().unwrap()
}

/// The bytes `config`'s pools may hold: its budget less the reserved shares.
pub(crate) fn pools(config: &StoreConfig) -> usize {
    config.budget_bytes - reserved(config)
}

/// `config` with its reserved shares (the kept predictions' and the stroke state's) added to
/// its budget: pools of the budget it had.
pub(crate) fn with_reserved_shares(config: StoreConfig) -> StoreConfig {
    config.with_budget_bytes(config.budget_bytes + reserved(&config))
}

/// Room for every tile of levels 1–8 of the cube, twice over.
pub(crate) fn roomy() -> StoreConfig {
    config(40, 16)
}

/// Six chunks of four evaluated slots (or fifteen correction slots).
///
/// About a sixth of the cube's 156 tiles fit, so evaluation evicts constantly and dirty tiles
/// are written back. It is the cube's minimum budget rounded up to whole chunks: an evaluation
/// pins at most two cone levels of nine tiles, which take five evaluated chunks, and one
/// correction chunk goes beside them.
pub(crate) fn tight() -> StoreConfig {
    config(6, 4)
}

/// A closed quad mesh with a valence-`n` vertex at the centre of each cap: an `n`-sided prism
/// with each polygon split into quads around its centroid.
pub(crate) fn split_prism(n: usize) -> BaseMesh {
    let angle = |k: usize| {
        let k = f32::from(u16::try_from(k).unwrap());
        let n = f32::from(u16::try_from(n).unwrap());
        2.0 * std::f32::consts::PI * k / n
    };
    let mut corners: Vec<Vec3> = Vec::new();
    for z in [1.0, -1.0] {
        corners.extend((0..n).map(|k| Vec3::new(angle(k).cos(), angle(k).sin(), z)));
    }
    let mut polygons: Vec<Vec<usize>> =
        vec![(0..n).collect(), (0..n).rev().map(|k| n + k).collect()];
    polygons.extend((0..n).map(|k| vec![k, n + k, n + (k + 1) % n, (k + 1) % n]));
    // Each polygon becomes one quad per corner: the corner, its two edge midpoints and the
    // centroid. Midpoints are shared between the two polygons of an edge.
    let mut positions = corners.clone();
    let mut midpoints: HashMap<(usize, usize), usize> = HashMap::new();
    let mut quads: Vec<[u32; 4]> = Vec::new();
    for polygon in &polygons {
        let sides = polygon.len();
        let sum = polygon.iter().fold(Vec3::ZERO, |sum, &v| sum + corners[v]);
        let centroid = positions.len();
        positions.push(sum * (1.0 / f32::from(u16::try_from(sides).unwrap())));
        let mids: Vec<usize> = (0..sides)
            .map(|a| {
                let (u, v) = (polygon[a], polygon[(a + 1) % sides]);
                *midpoints.entry((u.min(v), u.max(v))).or_insert_with(|| {
                    positions.push((corners[u] + corners[v]) * 0.5);
                    positions.len() - 1
                })
            })
            .collect();
        for a in 0..sides {
            let quad = [polygon[a], mids[a], centroid, mids[(a + sides - 1) % sides]];
            quads.push(quad.map(|v| u32::try_from(v).unwrap()));
        }
    }
    BaseMesh::from_polygons(positions, &quads).unwrap()
}

/// Every tile of levels `from..=to`, coarse levels first.
pub(crate) fn tiles(mesh: &BaseMesh, from: u8, to: u8) -> Vec<TileKey> {
    (from..=to)
        .flat_map(|l| {
            let level = level(l);
            let per_side = u16::try_from(level.tiles_per_side()).unwrap();
            mesh.faces().flat_map(move |face| {
                (0..per_side).flat_map(move |ty| {
                    (0..per_side).map(move |tx| TileKey::displacement(face, level, tx, ty).unwrap())
                })
            })
        })
        .collect()
}

/// `count` random edits at levels `1..=top`, half of them on face or tile borders, where twins
/// must stay equal.
pub(crate) fn random_edits(mesh: &BaseMesh, top: u8, count: u32, seed: u64) -> Vec<Edit> {
    let mut rng = Rng(seed);
    (0..count)
        .map(|_| {
            let face = FaceId::new(rng.below(mesh.face_count()));
            let level = level(1 + u8::try_from(rng.below(u32::from(top))).unwrap());
            let (n, tq) = (level.face_quads(), level.tile_quads());
            let mut pick = || {
                if rng.below(2) == 0 {
                    let border = rng.below(level.tiles_per_side() + 1) * tq;
                    (border + rng.below(3)).saturating_sub(1).min(n)
                } else {
                    rng.below(n + 1)
                }
            };
            let (i, j) = (pick(), pick());
            let amplitude = 0.5 / f32::from(u16::try_from(n).unwrap());
            Edit {
                face,
                level,
                i,
                j,
                c: rng.vec3(amplitude),
            }
        })
        .collect()
}

/// Flips a bit in the middle of the record at `loc` of the file at `path`, from outside, as a
/// bad sector would: the record fails its checksum.
pub(crate) fn damage(path: &std::path::Path, loc: Loc) {
    use std::os::unix::fs::FileExt;
    let mut bytes = std::fs::read(path).unwrap();
    let at = usize::try_from(loc.middle()).unwrap();
    bytes[at] ^= 0x40;
    std::fs::OpenOptions::new()
        .write(true)
        .open(path)
        .unwrap()
        .write_all_at(&bytes[at..=at], loc.middle())
        .unwrap();
}

/// Applies `edits` to the oracle's map alone: what a store over `mesh` that made them holds.
pub(crate) fn record(mesh: &BaseMesh, map: &mut CorrectionMap, edits: &[Edit]) {
    for e in edits {
        map.set(mesh.topology(), e.face, e.level, e.i, e.j, e.c)
            .unwrap();
    }
}

/// The oracle: every level up to [`TOP`], evaluated densely.
pub(crate) fn dense(mesh: &BaseMesh, map: &CorrectionMap) -> DenseEvaluation {
    DenseEvaluation::new(mesh, level(TOP), map).unwrap()
}

/// Asserts that `got`, tile `key`'s values, equal `want`'s at the same face-grid vertices, bit
/// for bit.
pub(crate) fn assert_tile(
    got: &[Vec3],
    want: impl Fn(u32, u32) -> Vec3,
    key: &TileKey,
    what: &str,
) {
    let region = Region::of_tile(key);
    let side = key.side();
    assert_eq!(got.len(), (side * side) as usize, "{what} {key:?}: length");
    for j in 0..side {
        for i in 0..side {
            let have = got[(j * side + i) as usize];
            assert_eq!(
                have.to_bits(),
                want(region.x0() + i, region.y0() + j).to_bits(),
                "{what} {key:?} ({i}, {j})"
            );
        }
    }
}

/// Asserts that the store evaluates every tile of `keys` as the dense evaluation does.
pub(crate) fn assert_positions(
    store: &mut TileStore,
    dense: &DenseEvaluation,
    keys: &[TileKey],
    what: &str,
) {
    for key in keys {
        let got = store.evaluate(*key).unwrap();
        let grid = dense.grid(key.face(), key.level()).unwrap();
        assert_tile(got, |i, j| grid.get(i, j).unwrap(), key, what);
    }
}

/// Every display tile of the store's mesh, built from the store's own finest tiles at [`TOP`]:
/// what the store's display layer should hold for its current state.
pub(crate) fn display_layer(store: &mut TileStore) -> Vec<geom::DisplayTile> {
    let mesh = store.mesh().clone();
    let finest = level(TOP);
    let side = finest.face_quads() + 1;
    let (mut positions, mut normals) = (geom::FaceGrids::new(finest), geom::FaceGrids::new(finest));
    for face in mesh.faces() {
        let count = usize::try_from(side * side).unwrap();
        let (mut p, mut n) = (vec![Vec3::ZERO; count], vec![Vec3::ZERO; count]);
        for key in tiles(&mesh, TOP, TOP)
            .into_iter()
            .filter(|k| k.face() == face)
        {
            let region = Region::of_tile(&key);
            let got_p = store.evaluate(key).unwrap().to_vec();
            let got_n = store.normals(key).unwrap().to_vec();
            for (index, (gp, gn)) in got_p.into_iter().zip(got_n).enumerate() {
                let index = u32::try_from(index).unwrap();
                let (x, y) = (
                    region.x0() + index % key.side(),
                    region.y0() + index / key.side(),
                );
                let at = usize::try_from(y * side + x).unwrap();
                p[at] = gp;
                n[at] = gn;
            }
        }
        positions
            .insert(face, geom::Grid::new(side, p).unwrap())
            .unwrap();
        normals
            .insert(face, geom::Grid::new(side, n).unwrap())
            .unwrap();
    }
    geom::build_mesh(&mesh, &positions, &normals).unwrap()
}
