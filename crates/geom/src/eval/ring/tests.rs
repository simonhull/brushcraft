//! The gathered block against the source it was gathered from: its layout, its ring's bits from
//! every side of every seam and base vertex, and its normals against `geom`'s and the plain
//! reference's.

use std::collections::BTreeMap;

use super::*;
use crate::fixtures::{ring_normals, subdivided_pentagonal_prism};
use crate::test_rng::Rng;
use crate::{
    Channel, CorrectionMap, DenseEvaluation, FaceId, LayerId, Level, TileKey, cube_sphere, normals,
};

/// Level `l`.
fn level(l: u8) -> Level {
    Level::new(l).expect("a valid level")
}

/// A cube sphere evaluated to level 2 without corrections, and one of its faces.
fn sphere() -> (BaseMesh, DenseEvaluation) {
    let mesh = cube_sphere(1).expect("a valid mesh");
    let dense = DenseEvaluation::new(&mesh, level(2), &CorrectionMap::new()).expect("evaluates");
    (mesh, dense)
}

/// `mesh` evaluated to `top` with random corrections at every level, a third of them on face
/// borders, where the seams are.
fn evaluated(mesh: &BaseMesh, top: u8, seed: u64) -> DenseEvaluation {
    let mut rng = Rng::new(seed);
    let mut map = CorrectionMap::new();
    for _ in 0..200 {
        let face = FaceId::new(rng.below(mesh.face_count()));
        let level = level(1 + u8::try_from(rng.below(u32::from(top))).expect("a level"));
        let n = level.face_quads();
        let mut pick = || match rng.below(3) {
            0 => [0, n][usize::try_from(rng.below(2)).expect("0 or 1")],
            _ => rng.below(n + 1),
        };
        let (i, j) = (pick(), pick());
        let amplitude = 0.5 / f32::from(u16::try_from(n).expect("a small face"));
        map.set(mesh.topology(), face, level, i, j, rng.vec3(amplitude))
            .expect("a vertex of the mesh");
    }
    DenseEvaluation::new(mesh, level(top), &map).expect("evaluates")
}

/// A source whose copies of a shared vertex disagree: every copy but the owner's is moved a
/// little. A gather reading anything but owner copies gives different bits on two sides of a
/// seam.
struct Twins<'a> {
    /// The mesh, for the owner copies.
    mesh: &'a BaseMesh,
    /// The positions.
    dense: &'a DenseEvaluation,
}

impl PositionSource for Twins<'_> {
    fn position(&self, face: FaceId, level: Level, i: u32, j: u32) -> Option<Vec3> {
        let p = self.dense.position(face, level, i, j)?;
        let owner = self
            .mesh
            .topology()
            .owner_copy(face, level.face_quads(), i, j)?;
        Some(if owner == (face, i, j) {
            p
        } else {
            p + Vec3::new(1e-3, -2e-3, 3e-3)
        })
    }
}

/// Every tile of `mesh` at `level`, gathered from `src`.
fn blocks(
    mesh: &BaseMesh,
    src: &impl PositionSource,
    level: Level,
) -> BTreeMap<TileKey, GatheredBlock> {
    let per_side = u16::try_from(level.tiles_per_side()).expect("a few tiles a side");
    let keys = mesh.faces().flat_map(|face| {
        (0..per_side).flat_map(move |ty| {
            (0..per_side).map(move |tx| TileKey::displacement(face, level, tx, ty).expect("a tile"))
        })
    });
    keys.map(|key| {
        let block = gather_block(mesh, src, Region::of_tile(&key)).expect("gathers");
        (key, block)
    })
    .collect()
}

#[test]
fn a_gathered_block_holds_the_padded_grid_and_the_corner_rings() {
    let (mesh, dense) = sphere();
    let face = mesh.faces().next().expect("a face");
    let region = Region::whole_face(face, level(2));
    let block = gather_block(&mesh, &dense, region).expect("gathers");
    let side = region.quads() as usize + 3;
    assert_eq!(block.side() as usize, side, "quads + 3 a side");
    assert_eq!(block.region(), region, "the block gathered");
    assert_eq!(block.padded().len(), side * side, "(quads + 3)² slots");
    let grid = dense.grid(face, level(2)).expect("the face's grid");
    for j in 0..=region.quads() {
        for i in 0..=region.quads() {
            let slot = block.padded()[(j as usize + 1) * side + i as usize + 1];
            assert_eq!(
                slot.to_bits(),
                grid.get(i, j).expect("in the grid").to_bits(),
                "({i}, {j}) is the face's own vertex"
            );
            assert_eq!(
                block.get(i64::from(i), i64::from(j)).to_bits(),
                slot.to_bits(),
                "get reads the same slot"
            );
            assert_eq!(
                block.row(i64::from(j))[i as usize + 1].to_bits(),
                slot.to_bits(),
                "and so does a row"
            );
        }
    }
    assert!(
        block.padded()[0].x.is_nan(),
        "nothing diagonally off a face corner"
    );
    assert!(block.get(-2, 0).x.is_nan(), "nothing past the ring");
    assert!(block.row(-2).is_empty(), "no row past the ring");
    for c in 0..4 {
        let ring = block.corner(c).expect("every face corner is a base vertex");
        assert_eq!(ring.valence(), 3, "cube corners have valence 3");
        assert_eq!(ring.edges().len(), 3, "an edge neighbour per quad");
        assert_eq!(ring.diagonals().len(), 3, "a diagonal per quad");
    }
    let corner = block.corner(0).expect("corner 0");
    assert_eq!(
        corner.center().to_bits(),
        block.padded()[side + 1].to_bits(),
        "the centre is the block's own corner"
    );
    assert!(block.corner(4).is_none(), "four corners only");
}

#[test]
fn a_block_inside_a_face_has_no_corner_rings_and_a_full_ring() {
    let mesh = cube_sphere(1).expect("a valid mesh");
    let dense = evaluated(&mesh, 7, 5);
    let key = TileKey::displacement(FaceId::new(0), level(7), 1, 1).expect("a tile");
    let block = gather_block(&mesh, &dense, Region::of_tile(&key)).expect("gathers");
    assert_eq!(
        block.side(),
        67,
        "a tile of 65² vertices has a ring of side 67"
    );
    let corners: Vec<usize> = (0..4).filter(|&c| block.corner(c).is_some()).collect();
    assert_eq!(
        corners,
        [2],
        "only the tile's corner on the face's corner is a base vertex"
    );
    let q = i64::from(block.quads());
    for (i, j) in [(-1, -1), (q + 1, -1), (-1, q + 1)] {
        assert!(
            block.get(i, j).is_finite(),
            "({i}, {j}): inside the face, the ring's corners are vertices"
        );
    }
    assert!(
        block.get(q + 1, q + 1).x.is_nan(),
        "off the face's corner, none"
    );
}

#[test]
fn ring_vertices_are_the_same_bits_from_every_side_of_every_seam_and_base_vertex() {
    let meshes = [
        ("cube", cube_sphere(1).expect("a valid mesh")),
        (
            "pentagonal prism",
            subdivided_pentagonal_prism().expect("a valid mesh"),
        ),
    ];
    for (name, mesh) in &meshes {
        // Level 7: two by two tiles a face, so both tile seams inside faces and face seams.
        let top = level(7);
        let dense = evaluated(mesh, 7, 11);
        let src = Twins {
            mesh,
            dense: &dense,
        };
        let blocks = blocks(mesh, &src, top);
        let n = top.face_quads();
        let topology = mesh.topology();
        let mut rings: BTreeMap<(FaceId, u32, u32), Vec<&GatheredRing>> = BTreeMap::new();
        let mut seams = 0;
        for (key, block) in &blocks {
            let region = block.region();
            let q = i64::from(region.quads());
            let (x0, y0) = (i64::from(region.x0()), i64::from(region.y0()));
            for j in -1..=q + 1 {
                for i in -1..=q + 1 {
                    let here = block.get(i, j);
                    let Some((g, gi, gj)) = topology.locate(region.face(), n, x0 + i, y0 + j)
                    else {
                        assert!(here.x.is_nan(), "{name} {key:?} ({i}, {j}): no vertex, NaN");
                        continue;
                    };
                    // The vertex as the tile holding it on face `g` holds it.
                    let (theirs, _, _) =
                        TileKey::containing(LayerId::BASE, Channel::DISPLACEMENT, g, top, gi, gj)
                            .expect("a tile holds it");
                    let other = &blocks[&theirs];
                    let (ox, oy) = (other.region().x0(), other.region().y0());
                    let there = other.get(i64::from(gi - ox), i64::from(gj - oy));
                    assert_eq!(
                        here.to_bits(),
                        there.to_bits(),
                        "{name} {key:?} ({i}, {j}) is {theirs:?}'s ({gi}, {gj})"
                    );
                    if theirs != *key {
                        seams += 1;
                    }
                }
            }
            let corners = [(0, 0), (q, 0), (q, q), (0, q)].into_iter().enumerate();
            for (c, (ci, cj)) in corners {
                let Some(ring) = block.corner(c) else {
                    continue;
                };
                assert_eq!(
                    ring.center().to_bits(),
                    block.get(ci, cj).to_bits(),
                    "{name} {key:?} corner {c}: the centre is the block's corner"
                );
                let fi = u32::try_from(x0 + ci).expect("on the face");
                let fj = u32::try_from(y0 + cj).expect("on the face");
                let owner = topology
                    .owner_copy(region.face(), n, fi, fj)
                    .expect("a vertex of the face");
                rings.entry(owner).or_default().push(ring);
            }
        }
        assert!(seams > 0, "{name}: the rings cross seams");
        let valences: Vec<usize> = rings.values().map(|r| r[0].valence()).collect();
        assert!(
            valences.iter().any(|&v| v != 4),
            "{name}: base vertices of valence other than 4"
        );
        for (vertex, copies) in &rings {
            let bits = |ring: &GatheredRing| {
                let points = ring.edges().iter().chain(ring.diagonals());
                let points: Vec<[u32; 3]> = points.map(|p| p.to_bits()).collect();
                (ring.center().to_bits(), points, ring.owner_corner())
            };
            assert_eq!(
                copies.len(),
                copies[0].valence(),
                "{name} {vertex:?}: a block corner on every face around it"
            );
            for ring in copies {
                assert_eq!(
                    bits(ring),
                    bits(copies[0]),
                    "{name} {vertex:?}: every face holds the same ring"
                );
            }
        }
    }
}

#[test]
fn normals_from_a_block_are_its_sources_normals_and_the_references() {
    let meshes = [
        cube_sphere(1).expect("a valid mesh"),
        subdivided_pentagonal_prism().expect("a valid mesh"),
    ];
    for mesh in &meshes {
        let dense = evaluated(mesh, 7, 23);
        let src = Twins {
            mesh,
            dense: &dense,
        };
        for l in 1..=7 {
            for (key, block) in blocks(mesh, &src, level(l)) {
                let region = Region::of_tile(&key);
                let want = normals(mesh, &src, region).expect("normals");
                let got = block.normals(mesh);
                let bits = |v: &[Vec3]| {
                    let bits: Vec<[u32; 3]> = v.iter().map(|p| p.to_bits()).collect();
                    bits
                };
                assert_eq!(
                    bits(got.as_slice()),
                    bits(want.as_slice()),
                    "{key:?}: the block's normals are geom's"
                );
                assert_eq!(
                    bits(&ring_normals(mesh, &block)),
                    bits(want.as_slice()),
                    "{key:?}: the plain reference's are too"
                );
                let mut into = vec![Vec3::ZERO; want.as_slice().len()];
                block.normals_into(mesh, &mut into).expect("the right size");
                assert_eq!(bits(&into), bits(want.as_slice()), "{key:?}: into a buffer");
            }
        }
    }
}

#[test]
fn degenerate_and_non_finite_blocks_give_the_fallback_and_nan_alike() {
    /// Every position the same point, but one that isn't finite.
    struct Flat;
    impl PositionSource for Flat {
        fn position(&self, _: FaceId, _: Level, i: u32, j: u32) -> Option<Vec3> {
            Some(if (i, j) == (3, 3) {
                Vec3::new(f32::INFINITY, 0.0, 0.0)
            } else {
                Vec3::new(1.0, 2.0, 3.0)
            })
        }
    }
    let mesh = cube_sphere(1).expect("a valid mesh");
    let region = Region::whole_face(FaceId::new(0), level(3));
    let block = gather_block(&mesh, &Flat, region).expect("gathers");
    let got = block.normals(&mesh);
    let reference = ring_normals(&mesh, &block);
    let bits = |v: &[Vec3]| {
        let bits: Vec<[u32; 3]> = v.iter().map(|p| p.to_bits()).collect();
        bits
    };
    assert_eq!(bits(got.as_slice()), bits(&reference), "the same bits");
    assert!(
        got.as_slice().iter().any(|n| n.x.is_nan()),
        "next to the infinite position: NaN"
    );
    assert!(
        got.as_slice().contains(&Vec3::new(0.0, 0.0, 1.0)),
        "elsewhere, flat: the fallback"
    );
    let mut short = vec![Vec3::ZERO; 3];
    assert!(
        matches!(
            block.normals_into(&mesh, &mut short),
            Err(EvalError::BufferSize { .. })
        ),
        "a buffer of the wrong size"
    );
}

#[test]
fn stencils_name_corners_borders_and_the_inside() {
    let (mesh, _) = sphere();
    let face = mesh.faces().next().expect("a face");
    let region = Region::whole_face(face, level(2));
    let q = region.quads();
    assert_eq!(
        vertex_stencil(&mesh, region, 0, 0),
        VertexStencil::Corner(0),
        "block corner 0"
    );
    assert_eq!(
        vertex_stencil(&mesh, region, q, q),
        VertexStencil::Corner(2),
        "block corner 2"
    );
    assert_eq!(
        vertex_stencil(&mesh, region, 1, 1),
        VertexStencil::Axes([[1, 0], [0, 1]]),
        "inside the face, the block's own axes"
    );
    assert!(
        matches!(vertex_stencil(&mesh, region, 1, 0), VertexStencil::Axes(_)),
        "on the face's border, some owner's axes"
    );
    assert_eq!(
        vertex_stencil(&mesh, region, q + 1, 0),
        VertexStencil::Outside,
        "one column past the block"
    );
    assert_eq!(
        vertex_stencil(&mesh, region, 0, q + 1),
        VertexStencil::Outside,
        "one row past the block"
    );
    assert_eq!(
        vertex_stencil(&mesh, region, u32::MAX, u32::MAX),
        VertexStencil::Outside,
        "far off"
    );
    // A block inside the face: its last vertex is the face's inside, still its own.
    let inner = Region::new(region.face(), region.level(), 0, 0, 2).expect("a 2-quad block");
    assert_eq!(
        vertex_stencil(&mesh, inner, 2, 2),
        VertexStencil::Axes([[1, 0], [0, 1]]),
        "the block's last vertex"
    );
    assert_eq!(
        vertex_stencil(&mesh, inner, 3, 2),
        VertexStencil::Outside,
        "a face vertex off the block"
    );
}

#[test]
fn a_rings_owner_corner_is_the_owner_faces_corner() {
    let mesh = subdivided_pentagonal_prism().expect("a valid mesh");
    let dense = evaluated(&mesh, 3, 29);
    let top = level(3);
    let n = top.face_quads();
    let mut owner_corners = std::collections::BTreeSet::new();
    for (key, block) in &blocks(&mesh, &dense, top) {
        let region = block.region();
        let q = region.quads();
        let corners = [(0, 0), (q, 0), (q, q), (0, q)].into_iter().enumerate();
        for (c, (ci, cj)) in corners {
            let ring = block
                .corner(c)
                .expect("a whole face's corners are base vertices");
            let (fi, fj) = (region.x0() + ci, region.y0() + cj);
            let around = mesh
                .topology()
                .ring(region.face(), crate::topology::corner_at(fi, fj, n))
                .expect("a base vertex");
            let &(_, owner_corner) = around.entries.first().expect("an owner");
            assert_eq!(ring.owner_corner(), owner_corner, "{key:?} corner {c}");
            owner_corners.insert(owner_corner);
        }
    }
    assert!(owner_corners.len() > 1, "owners at several corners");
}

#[test]
fn rings_are_equal_only_when_every_part_is() {
    /// A valence-3 ring: centre `c`, points `edge` and `diagonal` apart, owner corner `owner`.
    fn ring(c: f32, edge: f32, diagonal: f32, owner: u8) -> GatheredRing {
        let mut ring = CornerRing::new(Vec3::new(c, 0.0, 0.0), owner);
        for k in 0..3_u8 {
            let k = f32::from(k);
            ring.push(
                Vec3::new(edge + k, 1.0, 0.0),
                Vec3::new(diagonal + k, 1.0, 1.0),
            );
        }
        GatheredRing { ring }
    }
    let base = ring(0.0, 1.0, 2.0, 1);
    assert_eq!(base, ring(0.0, 1.0, 2.0, 1), "the same parts");
    assert_ne!(base, ring(0.5, 1.0, 2.0, 1), "another centre");
    assert_ne!(base, ring(0.0, 1.5, 2.0, 1), "other edge points");
    assert_ne!(base, ring(0.0, 1.0, 2.5, 1), "other diagonals");
    assert_ne!(base, ring(0.0, 1.0, 2.0, 2), "another owner corner");
    assert_eq!(base.valence(), 3, "three faces");
}

#[test]
fn a_ring_point_that_isnt_finite_gives_nan_at_its_base_vertex() {
    /// The dense evaluation, but for one edge neighbour of face 0's first corner, infinite.
    struct Poisoned<'a>(&'a DenseEvaluation);
    impl PositionSource for Poisoned<'_> {
        fn position(&self, face: FaceId, level: Level, i: u32, j: u32) -> Option<Vec3> {
            let p = self.0.position(face, level, i, j)?;
            Some(if (face, i, j) == (FaceId::new(0), 1, 0) {
                Vec3::new(f32::INFINITY, p.y, p.z)
            } else {
                p
            })
        }
    }
    let (mesh, dense) = sphere();
    let region = Region::whole_face(FaceId::new(0), level(2));
    let block = gather_block(&mesh, &Poisoned(&dense), region).expect("gathers");
    let got = block.normals(&mesh);
    let reference = ring_normals(&mesh, &block);
    assert!(
        got.get(0, 0).expect("a vertex").x.is_nan(),
        "NaN at the base vertex"
    );
    let bits = |v: &[Vec3]| {
        let bits: Vec<[u32; 3]> = v.iter().map(|p| p.to_bits()).collect();
        bits
    };
    assert_eq!(
        bits(got.as_slice()),
        bits(&reference),
        "the reference agrees"
    );
}

#[test]
fn blocks_displaced_by_one_function_of_their_points_still_meet_bit_for_bit() {
    let mesh = cube_sphere(1).expect("a valid mesh");
    let dense = evaluated(&mesh, 7, 13);
    let top = level(7);
    let mut displaced = blocks(&mesh, &dense, top);
    // A bump about a face's corner vertex, so it reaches across tile and face seams.
    let centre = dense
        .position(FaceId::new(0), top, 0, 0)
        .expect("a corner vertex");
    // Weighed at each point's counterpart in another gather of the same surface, as a preview
    // weighs at pen-down.
    let bump = |p: Vec3, at: Vec3| {
        let d = at - centre;
        let w = (0.3 - (d.x * d.x + d.y * d.y + d.z * d.z).sqrt()).max(0.0);
        p + Vec3::new(w, -w * 0.5, w * 0.25)
    };
    for block in displaced.values_mut() {
        let reference = block.clone();
        assert!(block.displace_along(&reference, bump), "the same region");
    }
    let n = top.face_quads();
    let topology = mesh.topology();
    let mut seen: BTreeMap<(FaceId, u32, u32), [u32; 3]> = BTreeMap::new();
    let (mut shared, mut moved) = (0, 0);
    for (key, block) in &displaced {
        let region = block.region();
        let q = region.quads();
        for j in 0..=q {
            for i in 0..=q {
                let (fi, fj) = (region.x0() + i, region.y0() + j);
                let owner = topology
                    .owner_copy(key.face(), n, fi, fj)
                    .expect("a vertex of the face");
                let bits = block.get(i64::from(i), i64::from(j)).to_bits();
                if let Some(&first) = seen.get(&owner) {
                    shared += 1;
                    assert_eq!(first, bits, "{owner:?}: every copy moved alike");
                } else {
                    seen.insert(owner, bits);
                }
                let before = dense
                    .position(key.face(), top, fi, fj)
                    .expect("evaluated")
                    .to_bits();
                if before != bits {
                    moved += 1;
                }
            }
        }
    }
    assert!(shared > 0, "tiles share border vertices");
    assert!(moved > 0, "the bump moved vertices");
}

#[test]
fn a_block_displaced_along_another_moves_each_point_by_its_counterpart() {
    let mesh = cube_sphere(1).expect("a valid mesh");
    let (now, then) = (evaluated(&mesh, 7, 3), evaluated(&mesh, 7, 13));
    let key = TileKey::displacement(FaceId::new(2), level(7), 0, 1).expect("a tile");
    let region = Region::of_tile(&key);
    let current = gather_block(&mesh, &now, region).expect("gathers");
    let reference = gather_block(&mesh, &then, region).expect("gathers");
    let shift = |p: Vec3, at: Vec3| p + Vec3::new(at.x * 0.5, -at.y, at.z * 0.25);
    let mut moved = current.clone();
    assert!(moved.displace_along(&reference, shift), "the blocks match");
    let triples = current
        .padded()
        .iter()
        .zip(reference.padded())
        .zip(moved.padded());
    for ((before, at), after) in triples {
        if before.x.is_nan() {
            assert!(after.x.is_nan(), "a gap stays a gap");
        } else {
            assert_eq!(after.to_bits(), shift(*before, *at).to_bits());
        }
    }
    let corners =
        (0..4).filter_map(|c| Some((current.corner(c)?, reference.corner(c)?, moved.corner(c)?)));
    let mut rings = 0;
    for (before, at, after) in corners {
        rings += 1;
        assert_eq!(
            after.center().to_bits(),
            shift(before.center(), at.center()).to_bits()
        );
        let points = before.edges().iter().chain(before.diagonals());
        let reference_points = at.edges().iter().chain(at.diagonals());
        let moved_points = after.edges().iter().chain(after.diagonals());
        for ((b, r), a) in points.zip(reference_points).zip(moved_points) {
            assert_eq!(a.to_bits(), shift(*b, *r).to_bits(), "a ring point moved");
        }
    }
    assert!(
        rings > 0,
        "the tile on a face corner has a base-vertex ring"
    );
}

#[test]
fn a_block_isnt_displaced_along_one_of_another_region() {
    let mesh = cube_sphere(1).expect("a valid mesh");
    let dense = evaluated(&mesh, 7, 3);
    let one = TileKey::displacement(FaceId::new(2), level(7), 0, 1).expect("a tile");
    let other = TileKey::displacement(FaceId::new(2), level(7), 1, 1).expect("a tile");
    let mut block = gather_block(&mesh, &dense, Region::of_tile(&one)).expect("gathers");
    let elsewhere = gather_block(&mesh, &dense, Region::of_tile(&other)).expect("gathers");
    let before = block.clone();
    assert!(!block.displace_along(&elsewhere, |p, _| p + Vec3::new(1.0, 0.0, 0.0)));
    let bits = |of: &GatheredBlock| {
        let bits: Vec<[u32; 3]> = of.padded().iter().map(|p| p.to_bits()).collect();
        bits
    };
    assert_eq!(bits(&block), bits(&before), "nothing moved");
}
