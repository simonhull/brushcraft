//! The display records and readers, tested through the store in `tiles::tests::display` and
//! the file harnesses in `file::fuzz`; this holds the unit tests of the pieces alone.

use super::*;

/// A display record header round-trips its state, finest level, bound and cone.
#[test]
fn headers_hold_the_state_the_finest_level_the_bound_and_the_cone() {
    let cone = NormalCone::from_bits(0x1234_5678);
    let record = DisplayRecord {
        error: 0.25,
        cone,
        finest: Level::new(9).unwrap(),
        encoded: Vec::new(),
    };
    assert_eq!(display_header(None), [VERSION, UNKNOWN], "unknown");
    let known = display_header(Some(&record));
    assert_eq!(known.len(), 11, "version, state, finest level, bound, cone");
    assert_eq!(known[..3], [VERSION, KNOWN, 9], "known, from level 9");
    assert_eq!(
        f32::from_le_bytes(known[3..7].try_into().unwrap()),
        0.25,
        "the bound"
    );
    assert_eq!(
        NormalCone::from_bits(u32::from_le_bytes(known[7..].try_into().unwrap())),
        cone,
        "the cone"
    );
    assert!(
        display_body(None).is_empty(),
        "an unknown record has no body"
    );
}

/// The display tiles reached from `moved` as first written: every border vertex of each moved
/// tile's window at each level walked one at a time, nothing shared between tiles. What
/// [`reached_by`] must give.
fn reached_one_vertex_at_a_time(
    mesh: &BaseMesh,
    moved: &[TileKey],
    finest: Level,
    levels: &BTreeSet<Level>,
) -> BTreeSet<TileKey> {
    let mut answer = BTreeSet::new();
    for key in moved {
        let region = Region::of_tile(key);
        for &level in levels {
            let shift = u32::from(finest.get() - level.get());
            let n = level.face_quads();
            let lo = |v: u32| (v >> shift).saturating_sub(1);
            let hi = |v: u32| ((v + region.quads()).div_ceil(1 << shift) + 1).min(n);
            let xs = (lo(region.x0()), hi(region.x0()));
            let ys = (lo(region.y0()), hi(region.y0()));
            tiles_over(&mut answer, key.face(), level, xs, ys);
            for j in ys.0..=ys.1 {
                for i in xs.0..=xs.1 {
                    if i == 0 || j == 0 || i == n || j == n {
                        walk_run(mesh, &mut answer, key.face(), level, n, (i, j), (i, j));
                    }
                }
            }
        }
    }
    answer
}

/// Walking border runs, each window once, reaches what walking every border vertex does.
///
/// Checked for every finest tile alone (the cross-face part only shows where nothing else covers
/// it) and for random sets of them, on meshes with base vertices of valence 3, 4 and 5.
#[test]
fn reaching_by_runs_matches_reaching_one_vertex_at_a_time() {
    use crate::fixture::{TOP, level, split_prism};
    let meshes = [(*crate::fixture::mesh()).clone(), split_prism(5)];
    let finest = level(TOP);
    let levels: BTreeSet<Level> = (0..TOP).map(level).collect();
    for (m, mesh) in meshes.iter().enumerate() {
        let all: Vec<TileKey> = mesh
            .faces()
            .flat_map(|face| {
                let per_side = u16::try_from(finest.tiles_per_side()).unwrap();
                (0..per_side).flat_map(move |ty| {
                    (0..per_side).map(move |tx| {
                        TileKey::new(LayerId::BASE, Channel::DISPLACEMENT, face, finest, tx, ty)
                            .unwrap()
                    })
                })
            })
            .collect();
        for key in &all {
            let one = std::slice::from_ref(key);
            assert_eq!(
                reached_by(mesh, one.iter().copied(), finest, &levels),
                reached_one_vertex_at_a_time(mesh, one, finest, &levels),
                "mesh {m}, {key:?}"
            );
        }
        let mut seed = 0x9E37_79B9_7F4A_7C15_u64;
        for round in 0..16 {
            let some: Vec<TileKey> = (0..=round % 5)
                .map(|_| {
                    seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    all[usize::try_from(seed >> 33).unwrap() % all.len()]
                })
                .collect();
            assert_eq!(
                reached_by(mesh, some.iter().copied(), finest, &levels),
                reached_one_vertex_at_a_time(mesh, &some, finest, &levels),
                "mesh {m}, round {round}"
            );
        }
    }
}

/// A source that gives every vertex and notes the tile of each it is asked for.
struct Recording {
    /// The display tiles read.
    read: std::cell::RefCell<BTreeSet<TileKey>>,
}

impl geom::PositionSource for Recording {
    fn position(&self, face: FaceId, level: Level, i: u32, j: u32) -> Option<geom::Vec3> {
        let (key, _, _) = TileKey::containing(LayerId::BASE, Channel::DISPLAY, face, level, i, j)?;
        self.read.borrow_mut().insert(key);
        Some(geom::Vec3::new(0.0, 0.0, 1.0))
    }
}

/// [`tent_reads`] holds every tile one level finer that a display tile's mip reads, and the
/// tiles under it, for every display tile of meshes with base vertices of valence 3, 4 and 5.
#[test]
fn tent_reads_hold_everything_a_tile_reads() {
    use crate::fixture::{TOP, level, split_prism, tiles};
    let meshes = [(*crate::fixture::mesh()).clone(), split_prism(5)];
    let numbered = meshes.iter().enumerate();
    for (m, mesh) in numbered {
        for k in tiles(mesh, 0, TOP - 1) {
            let key = TileKey::display(k.face(), k.level(), k.tx(), k.ty()).unwrap();
            let reads = tent_reads(mesh, &key).unwrap();
            let source = Recording {
                read: std::cell::RefCell::new(BTreeSet::new()),
            };
            geom::mip_normals_tile(mesh, &source, &key).unwrap();
            for read in source.read.into_inner() {
                assert!(
                    reads.contains(&read),
                    "mesh {m}, {key:?}: the mip reads {read:?}"
                );
            }
            let finer = level(key.level().get() + 1);
            let region = Region::of_tile(&key);
            let per = finer.tile_quads();
            for y in (2 * region.y0()..=2 * (region.y0() + region.quads())).step_by(8) {
                for x in (2 * region.x0()..=2 * (region.x0() + region.quads())).step_by(8) {
                    let (under, _, _) = TileKey::containing(
                        LayerId::BASE,
                        Channel::DISPLAY,
                        key.face(),
                        finer,
                        x,
                        y,
                    )
                    .unwrap();
                    assert!(
                        reads.contains(&under),
                        "mesh {m}, {key:?}: under it, {under:?} ({per})"
                    );
                }
            }
        }
    }
}
