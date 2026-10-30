//! Tests of the cluster triangles and their edge variants, on the CPU.

use std::collections::{BTreeMap, BTreeSet};

use super::*;
use crate::cull::reference::extra_triangles;

/// A grid vertex `(u, v)`.
fn at(u: u8, v: u8) -> GridVertex {
    GridVertex::new(u, v).expect("on the grid")
}

/// Twice the signed area of `triangle` in the grid: positive counter-clockwise.
fn doubled_area(triangle: [GridVertex; 3]) -> i32 {
    let [(ax, ay), (bx, by), (cx, cy)] = triangle.map(|g| (i32::from(g.u()), i32::from(g.v())));
    (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
}

/// Whether `point` lies strictly inside `triangle`, counter-clockwise.
fn inside(triangle: [GridVertex; 3], point: (f64, f64)) -> bool {
    let side = |from: GridVertex, to: GridVertex| {
        let (fx, fy) = (f64::from(from.u()), f64::from(from.v()));
        let (tx, ty) = (f64::from(to.u()), f64::from(to.v()));
        (tx - fx) * (point.1 - fy) - (ty - fy) * (point.0 - fx)
    };
    let [first, second, third] = triangle;
    side(first, second) > 0.0 && side(second, third) > 0.0 && side(third, first) > 0.0
}

/// The drawn triangles of `size` and `variant`, without the padding.
fn real(size: ClusterSize, variant: Variant) -> Vec<[GridVertex; 3]> {
    drawn(size, variant).flatten().collect()
}

/// Whether `vertex` is an odd border vertex of `edge` in a cluster of `size`: on it, not a
/// corner, at an odd place along it.
fn odd_on(edge: Edge, size: ClusterSize, vertex: GridVertex) -> bool {
    let last = size.quads();
    let along = match edge {
        Edge::Bottom | Edge::Top => vertex.u(),
        Edge::Right | Edge::Left => vertex.v(),
    };
    edge.holds(size, vertex) && along % 2 == 1 && along < last
}

#[test]
fn the_static_triangles_split_every_quad_along_its_i_j_to_i1_j1_diagonal() {
    let all: Vec<[GridVertex; 3]> = triangles().collect();
    assert_eq!(all.len(), CLUSTER_TRIANGLES, "128 triangles");
    let buffer = indices();
    assert_eq!(buffer.len(), CLUSTER_INDICES as usize, "384 indices");
    assert!(
        buffer.iter().all(|&i| usize::from(i) < CLUSTER_VERTICES),
        "every index on the 81-vertex grid"
    );
    for y in 0..CLUSTER_QUADS {
        for x in 0..CLUSTER_QUADS {
            let first = 2 * (usize::from(y) * usize::from(CLUSTER_QUADS) + usize::from(x));
            let pair = [all[first], all[first + 1]];
            let (p00, p11) = (at(x, y), at(x + 1, y + 1));
            for triangle in pair {
                assert!(
                    triangle.contains(&p00) && triangle.contains(&p11),
                    "quad ({x}, {y}): both triangles share the (i, j)-(i + 1, j + 1) diagonal"
                );
                assert_eq!(
                    doubled_area(triangle),
                    1,
                    "quad ({x}, {y}): counter-clockwise"
                );
            }
            assert!(
                pair[0].contains(&at(x + 1, y)),
                "the lower-right half first"
            );
            assert!(pair[1].contains(&at(x, y + 1)), "then the upper-left half");
            let flat: Vec<u16> = pair
                .iter()
                .flatten()
                .map(|g| u16::from(g.index()))
                .collect();
            assert_eq!(&buffer[3 * first..3 * first + 6], &*flat, "in index order");
        }
    }
}

#[test]
fn the_variant_tables_hold_remap_at_each_variant_s_offset() {
    let table = remap_table();
    assert_eq!(
        table.len() * 4,
        TABLES * CLUSTER_VERTICES,
        "every table, packed"
    );
    let bytes: Vec<u8> = table.iter().flat_map(|word| word.to_le_bytes()).collect();
    let mut offsets = BTreeSet::new();
    for size in ClusterSize::ALL {
        for variant in Variant::all() {
            let offset = table_offset(size, variant);
            assert!(
                offsets.insert(offset),
                "{size:?} {variant}: an offset of its own"
            );
            for vertex in GridVertex::all() {
                assert_eq!(
                    bytes[offset + usize::from(vertex.index())],
                    remap(size, variant, vertex).index(),
                    "{size:?} {variant} {vertex:?}"
                );
            }
        }
    }
    assert_eq!(Variant::all().count(), Variant::COUNT, "16 variants");
    assert_eq!(Variant::new(16), None, "four bits");
    assert_eq!(
        Variant::of_edges(&[Edge::Right, Edge::Left]).bits(),
        0b1010,
        "bit k for edge k, in neighbour order"
    );
    for size in ClusterSize::ALL {
        assert_eq!(ClusterSize::of_quads(u32::from(size.quads())), Some(size));
    }
}

#[test]
fn a_variant_moves_only_the_odd_vertices_of_its_edges_each_onto_an_even_neighbour_on_the_edge() {
    let size = ClusterSize::Eight;
    for variant in Variant::all() {
        for vertex in GridVertex::all() {
            let moved = remap(size, variant, vertex);
            let skipped = Edge::ALL
                .into_iter()
                .filter(|&edge| variant.has(edge) && odd_on(edge, size, vertex));
            let skipped: Vec<Edge> = skipped.collect();
            match skipped.as_slice() {
                [] => assert_eq!(moved, vertex, "{variant} {vertex:?} stays"),
                [edge] => {
                    let step = (
                        i32::from(moved.u()) - i32::from(vertex.u()),
                        i32::from(moved.v()) - i32::from(vertex.v()),
                    );
                    let expected = match edge {
                        Edge::Bottom => (1, 0),
                        Edge::Right => (0, -1),
                        Edge::Top => (-1, 0),
                        Edge::Left => (0, 1),
                    };
                    assert_eq!(step, expected, "{variant} {vertex:?} along {edge:?}");
                    assert!(edge.holds(size, moved), "stays on its edge");
                    assert!(!odd_on(*edge, size, moved), "onto an even vertex");
                }
                more => panic!("{vertex:?} is odd on {more:?}: only corners are on two edges"),
            }
        }
    }
    // A smaller cluster clamps every vertex past its quads onto them.
    for small in [ClusterSize::One, ClusterSize::Two, ClusterSize::Four] {
        for vertex in GridVertex::all() {
            let moved = remap(small, Variant::PLAIN, vertex);
            assert_eq!(
                (moved.u(), moved.v()),
                (vertex.u().min(small.quads()), vertex.v().min(small.quads())),
                "{small:?} {vertex:?}"
            );
        }
    }
}

#[test]
fn every_variant_of_every_size_covers_each_quad_once_with_its_triangles_whole() {
    for size in ClusterSize::ALL {
        let last = size.quads();
        let holds = |g: GridVertex| g.u() <= last && g.v() <= last;
        for variant in Variant::all() {
            let tris = real(size, variant);
            let skipped_odd = GridVertex::all().filter(|&g| {
                g.u() <= last
                    && g.v() <= last
                    && Edge::ALL
                        .into_iter()
                        .any(|edge| variant.has(edge) && odd_on(edge, size, g))
            });
            let skipped = skipped_odd.count();
            let quads = usize::from(last) * usize::from(last);
            assert_eq!(
                tris.len(),
                2 * quads - skipped,
                "{size:?} {variant}: one triangle fewer for each skipped vertex"
            );
            let mut area = 0;
            for &t in &tris {
                assert!(t.iter().all(|&g| holds(g)), "{size:?} {variant}: inside");
                assert!(
                    doubled_area(t) > 0,
                    "{size:?} {variant} {t:?}: not degenerate"
                );
                area += doubled_area(t);
                for edge in Edge::ALL {
                    assert!(
                        !(variant.has(edge) && t.iter().any(|&g| odd_on(edge, size, g))),
                        "{size:?} {variant} {t:?}: no skipped vertex drawn"
                    );
                }
            }
            assert_eq!(
                area,
                2 * i32::try_from(quads).expect("small"),
                "{size:?} {variant}: the cluster's area"
            );
            // Every quad covered exactly once, at points off every edge and diagonal.
            for y in 0..last {
                for x in 0..last {
                    for (dx, dy) in [(0.31, 0.62), (0.73, 0.18), (0.52, 0.47)] {
                        let (px, py) = (f64::from(x) + dx, f64::from(y) + dy);
                        let covering = tris.iter().filter(|&&t| inside(t, (px, py))).count();
                        assert_eq!(covering, 1, "{size:?} {variant}: ({px}, {py}) covered once");
                    }
                }
            }
            check_manifold(size, variant, &tris);
        }
    }
}

/// Every edge inside the cluster is drawn once each way, every border edge once, and a border
/// edge beside a coarser neighbour spans two quads, between even vertices.
fn check_manifold(size: ClusterSize, variant: Variant, tris: &[[GridVertex; 3]]) {
    let mut directed: BTreeMap<(GridVertex, GridVertex), usize> = BTreeMap::new();
    for [a, b, c] in tris {
        for (p, q) in [(*a, *b), (*b, *c), (*c, *a)] {
            *directed.entry((p, q)).or_default() += 1;
        }
    }
    for (&(p, q), &count) in &directed {
        assert_eq!(count, 1, "{size:?} {variant}: {p:?}-{q:?} drawn once");
        let border = Edge::ALL
            .into_iter()
            .find(|&edge| edge.holds(size, p) && edge.holds(size, q));
        match border {
            Some(edge) => {
                assert!(
                    !directed.contains_key(&(q, p)),
                    "{size:?} {variant}: a border edge has one side"
                );
                let length = p.u().abs_diff(q.u()) + p.v().abs_diff(q.v());
                let expected = if variant.has(edge) && size.quads() > 1 {
                    2
                } else {
                    1
                };
                assert_eq!(
                    length, expected,
                    "{size:?} {variant}: {p:?}-{q:?} on {edge:?}"
                );
            }
            None => assert!(
                directed.contains_key(&(q, p)),
                "{size:?} {variant}: {p:?}-{q:?} inside has a triangle on its other side"
            ),
        }
    }
}

#[test]
fn every_variant_draws_only_triangles_whose_normals_the_bounds_pass_holds() {
    for size in ClusterSize::ALL {
        let side = usize::from(size.quads()) + 1;
        let held_extra = extra_triangles(side).into_iter().map(|t| {
            let set: BTreeSet<(i32, i32)> = t.into_iter().collect();
            set
        });
        let own = triangles().filter_map(|t| {
            let set: BTreeSet<(i32, i32)> = t
                .iter()
                .map(|g| (i32::from(g.u()), i32::from(g.v())))
                .collect();
            set.iter()
                .all(|&(u, v)| u <= i32::from(size.quads()) && v <= i32::from(size.quads()))
                .then_some(set)
        });
        let held: BTreeSet<BTreeSet<(i32, i32)>> = held_extra.chain(own).collect();
        for variant in Variant::all() {
            for t in real(size, variant) {
                let set: BTreeSet<(i32, i32)> = t
                    .iter()
                    .map(|g| (i32::from(g.u()), i32::from(g.v())))
                    .collect();
                assert!(
                    held.contains(&set),
                    "{size:?} {variant}: {t:?} is a tile triangle or a bounds-pass fan"
                );
            }
        }
    }
}

/// A vertex's position as a tile stores it: a function of its place on the finer level's face
/// grid, in `f32` bits, so two tiles holding the same vertex store the same bits.
fn position(x: u32, y: u32) -> [u32; 3] {
    let (fx, fy) = (f64::from(x), f64::from(y));
    #[expect(
        clippy::cast_possible_truncation,
        reason = "positions are stored in f32, rounded from these"
    )]
    let p = [
        (fx * 0.37).sin() as f32,
        (fy * 0.21).cos() as f32,
        (fx * fy * 0.013).sin() as f32,
    ];
    p.map(f32::to_bits)
}

/// A cluster's stored positions: its 81 grid vertices, vertex `(u, v)` at `origin + scale · (u,
/// v)` on the finer face grid.
fn stored(origin: (u32, u32), scale: u32) -> Vec<[u32; 3]> {
    let all = GridVertex::all().map(|g| {
        position(
            origin.0 + scale * u32::from(g.u()),
            origin.1 + scale * u32::from(g.v()),
        )
    });
    all.collect()
}

/// The border segments `tris` draw along `edge`, each as the stored bits of its two ends, in
/// either order.
fn drawn_border(
    size: ClusterSize,
    tris: &[[GridVertex; 3]],
    edge: Edge,
    positions: &[[u32; 3]],
) -> BTreeSet<([u32; 3], [u32; 3])> {
    let segments = tris.iter().flat_map(|&[a, b, c]| [(a, b), (b, c), (c, a)]);
    let on_edge = segments.filter(|&(p, q)| edge.holds(size, p) && edge.holds(size, q));
    on_edge
        .map(|(p, q)| {
            let (a, b) = (
                positions[usize::from(p.index())],
                positions[usize::from(q.index())],
            );
            (a.min(b), a.max(b))
        })
        .collect()
}

#[test]
fn shared_border_vertices_are_bit_identical_on_both_sides_of_every_variant_pairing() {
    // A full cluster of a finer tile at (2n, 2n) on the finer level's face grid, beside a
    // cluster across each edge: of a coarser tile (its grid vertex (x, y) at (2x, 2y) of the
    // finer grid) where the variant names the edge, else of a tile at the same level. Each
    // stores its own copy of the positions, by place. Both sides must draw that edge from the
    // same vertices: the same segments, the same stored bits at their ends.
    let size = ClusterSize::Eight;
    let n = u32::from(CLUSTER_QUADS);
    let finer = stored((2 * n, 2 * n), 1);
    let finer_bits: BTreeSet<[u32; 3]> = finer.iter().copied().collect();
    for variant in Variant::all() {
        let tris = real(size, variant);
        for edge in Edge::ALL {
            let ours = drawn_border(size, &tris, edge, &finer);
            let coarser = variant.has(edge);
            // The cluster across the edge: for a coarser one, the cluster whose edge the finer
            // edge is half of.
            let (origin, scale) = match (edge, coarser) {
                (Edge::Bottom, true) => ((2 * n, 0), 2),
                (Edge::Bottom, false) => ((2 * n, n), 1),
                (Edge::Right, _) => ((3 * n, 2 * n), if coarser { 2 } else { 1 }),
                (Edge::Top, _) => ((2 * n, 3 * n), if coarser { 2 } else { 1 }),
                (Edge::Left, true) => ((0, 2 * n), 2),
                (Edge::Left, false) => ((n, 2 * n), 1),
            };
            let theirs_positions = stored(origin, scale);
            let facing = match edge {
                Edge::Bottom => Edge::Top,
                Edge::Right => Edge::Left,
                Edge::Top => Edge::Bottom,
                Edge::Left => Edge::Right,
            };
            // Their edge facing ours is plain on their side (their neighbour, ours, is not
            // coarser), whatever their other edges' variant.
            for their_variant in Variant::all().filter(|v| !v.has(facing)) {
                let theirs_tris = real(size, their_variant);
                let theirs = drawn_border(size, &theirs_tris, facing, &theirs_positions);
                let shared = theirs
                    .iter()
                    .filter(|(a, b)| finer_bits.contains(a) && finer_bits.contains(b));
                let shared: BTreeSet<([u32; 3], [u32; 3])> = shared.copied().collect();
                assert_eq!(
                    ours, shared,
                    "{variant} {edge:?} beside {their_variant}: the same segments, the same bits"
                );
            }
            let span = if coarser { n / 2 } else { n };
            assert_eq!(
                ours.len(),
                usize::try_from(span).expect("small"),
                "{variant} {edge:?}: the whole edge"
            );
        }
    }
}
