//! The blocks' raycast against the exhaustive test, bit for bit.
//!
//! On grids flat, bumpy, crumpled, degenerate and with holes, with rays aimed, grazing, along
//! grid lines and block borders, through vertices, and missing; the sweep's bounds against the
//! quad-by-quad ones; and what a tile's blocks take.

#![expect(clippy::missing_panics_doc, reason = "tests panic to fail")]

use geom::{FaceId, Level};

use super::*;
use crate::index::bvh::tests::Rng;
use crate::index::intersect_grid;

/// A hit as bits, `−0` and `0` told apart: what "bit for bit" compares.
type HitBits = (TileKey, u32, u32, u32, bool, u32, u32);

/// `hit` as [`HitBits`].
fn bits(hit: Option<TriangleHit>) -> Option<HitBits> {
    let hit = hit?;
    let floats = (hit.t.to_bits(), hit.u.to_bits(), hit.v.to_bits());
    Some((
        hit.key, floats.0, hit.i, hit.j, hit.second, floats.1, floats.2,
    ))
}

/// The key the grids' hits are named for.
fn key() -> TileKey {
    TileKey::displacement(FaceId::new(0), Level::new(6).unwrap(), 0, 0).unwrap()
}

/// Checks that the blocks of `positions` (`side` a side) answer `ray` as the exhaustive test
/// does, bit for bit, and returns whether it hit.
fn agrees(positions: &[Vec3], side: u32, blocks: &TileBlocks, ray: &Ray, what: &str) -> bool {
    let exhaustive = intersect_grid(ray, positions, side, key());
    let fast = blocks.intersect(ray, &WideRay::new(ray), positions, key());
    assert_eq!(
        bits(fast),
        bits(exhaustive),
        "{what}: the blocks' hit is the exhaustive one ({ray:?})"
    );
    exhaustive.is_some()
}

/// The shapes of grid the tests raycast.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shape {
    /// The plane `z = 0`: rays in it, or along its lines, are exactly parallel.
    Flat,
    /// A smooth height field, a few waves across.
    Waves,
    /// Each vertex moved at random by about a quad: folds and overlaps.
    Crumpled,
    /// Waves with runs of vertices merged: degenerate triangles, zero normals.
    Degenerate,
    /// A sheet rolled most of the way round a cylinder: a ray can cross it twice, and its
    /// silhouette is inside the tile.
    Rolled,
    /// Waves with a few vertices not finite (NaN or infinite): their triangles are never hit,
    /// the quads' other triangles are.
    Holes,
}

impl Shape {
    /// Every shape.
    const ALL: [Self; 6] = [
        Self::Flat,
        Self::Waves,
        Self::Crumpled,
        Self::Degenerate,
        Self::Rolled,
        Self::Holes,
    ];

    /// The grid's positions, `side` a side, a quad `1 / (side − 1)` across.
    fn grid(self, side: u32, rng: &mut Rng) -> Vec<Vec3> {
        let quads = side.saturating_sub(1).max(1);
        #[expect(clippy::cast_precision_loss, reason = "a grid of at most 65 a side")]
        let step = 1.0 / quads as f32;
        let (phase, amplitude) = (rng.range(0.0, 6.0), rng.range(0.02, 0.2));
        let mut positions = Vec::new();
        for j in 0..side {
            for i in 0..side {
                #[expect(clippy::cast_precision_loss, reason = "a grid of at most 65 a side")]
                let (x, y) = (i as f32 * step, j as f32 * step);
                let p = match self {
                    Self::Flat => Vec3::new(x, y, 0.0),
                    Self::Waves | Self::Degenerate | Self::Holes => Vec3::new(
                        x,
                        y,
                        amplitude * ((7.0 * x + phase).sin() + (5.0 * y - phase).cos()),
                    ),
                    Self::Crumpled => Vec3::new(x, y, 0.0) + rng.point(-step, step) * 1.5,
                    Self::Rolled => {
                        let angle = 5.5 * x;
                        Vec3::new(0.3 * angle.cos(), y, 0.3 * angle.sin())
                    }
                };
                positions.push(p);
            }
        }
        if self == Self::Holes {
            let holes = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
            for hole in holes.into_iter().cycle().take(side as usize / 4 + 1) {
                let at = rng.below(positions.len());
                if let Some(p) = positions.get_mut(at) {
                    match rng.below(3) {
                        0 => p.x = hole,
                        1 => p.y = hole,
                        _ => p.z = hole,
                    }
                }
            }
        }
        if self == Self::Degenerate {
            let width = side as usize;
            for _ in 0..side {
                let at = rng.below(positions.len());
                let along = if rng.below(2) == 0 { 1 } else { width };
                if let Some(&p) = positions.get(at) {
                    for k in 1..4 {
                        if let Some(q) = positions.get_mut(at + k * along) {
                            *q = p;
                        }
                    }
                }
            }
        }
        positions
    }
}

/// A ray of one of the kinds the tests cast at `positions` (`side` a side), chosen by `kind`.
fn ray_of(kind: usize, positions: &[Vec3], side: u32, rng: &mut Rng) -> Option<Ray> {
    let vertex = |draw: &mut Rng| positions.get(draw.below(positions.len())).copied();
    let width = side as usize;
    // A point on the grid: a vertex, or between two neighbours.
    let on_grid = |draw: &mut Rng| {
        let at = draw.below(positions.len());
        let p = positions.get(at)?;
        let q = positions
            .get(at + 1)
            .or_else(|| positions.get(at + width))?;
        Some(*p + (*q - *p) * draw.unit())
    };
    let away = |draw: &mut Rng| draw.point(-1.0, 1.0).normalized();
    match kind % 8 {
        // Aimed at a point of the grid from afar.
        0 => {
            let target = on_grid(rng)?;
            let origin = target + away(rng)? * rng.range(0.5, 3.0);
            Ray::new(origin, target - origin)
        }
        // Through a vertex exactly, from close by.
        1 => {
            let target = vertex(rng)?;
            let origin = target + away(rng)? * rng.range(0.01, 0.5);
            Ray::new(origin, target - origin)
        }
        // Grazing: from a vertex along an edge out of it, tipped off it by a hair.
        2 => {
            let at = rng.below(positions.len());
            let (p, q) = (positions.get(at)?, positions.get(at + 1)?);
            let tip = rng.point(-1.0, 1.0) * 10.0_f32.powf(rng.range(-7.0, -2.0));
            let origin = *p - (*q - *p) * rng.range(0.0, 20.0);
            Ray::new(origin + tip, *q - *p + tip)
        }
        // Along a grid row or a block border, in the grid's plane.
        3 => {
            let row = BLOCK_QUADS as usize * rng.below(side as usize / BLOCK_QUADS as usize + 1);
            let row = if rng.below(2) == 0 {
                row
            } else {
                rng.below(width)
            };
            let start = positions.get((row * width).min(positions.len() - 1))?;
            let end = positions.get((row * width + width - 1).min(positions.len() - 1))?;
            let origin = *start - (*end - *start) * rng.range(0.0, 1.0);
            Ray::new(origin, *end - *start)
        }
        // Along a column, a block border most often.
        4 => {
            let column = BLOCK_QUADS as usize * rng.below(side as usize / BLOCK_QUADS as usize + 1);
            let column = column.min(width - 1);
            let start = positions.get(column)?;
            let end = positions.get((side as usize - 1) * width + column)?;
            let origin = *start - (*end - *start) * rng.range(0.0, 1.0);
            Ray::new(origin, *end - *start)
        }
        // Nearly in the plane z = 0, from beside the grid.
        5 => {
            let origin = Vec3::new(rng.range(-1.0, 2.0), rng.range(-1.0, 2.0), 0.0)
                + Vec3::new(0.0, 0.0, rng.range(-1e-3, 1e-3));
            let target = Vec3::new(rng.unit(), rng.unit(), rng.range(-1e-4, 1e-4));
            Ray::new(origin, target - origin)
        }
        // Missing: aimed beside the grid.
        6 => {
            let origin = rng.point(-2.0, 3.0);
            let target = Vec3::new(rng.range(1.1, 2.0), rng.range(-1.0, -0.1), rng.unit());
            Ray::new(origin, target - origin)
        }
        // Anywhere, any way, origin among the triangles.
        _ => Ray::new(rng.point(-0.5, 1.5), rng.point(-1.0, 1.0)),
    }
}

/// Casts `rays` rays of every kind at a grid of every shape and of `side` a side, from `seed`,
/// checking each against the exhaustive test; returns how many hit.
fn cast_at_every_shape(seed: u64, side: u32, rays: usize) -> usize {
    let mut rng = Rng(seed);
    let mut hits = 0;
    for shape in Shape::ALL {
        let positions = shape.grid(side, &mut rng);
        let blocks = TileBlocks::new(&positions, side);
        for k in 0..rays {
            if let Some(ray) = ray_of(k, &positions, side, &mut rng) {
                let what = format!("{shape:?}, side {side}, ray {k}");
                hits += usize::from(agrees(&positions, side, &blocks, &ray, &what));
            }
        }
    }
    hits
}

#[test]
fn blocks_hit_as_every_triangle_does() {
    let mut hits = 0;
    for (seed, side) in [(1, 65), (2, 17), (3, 9), (4, 2), (5, 30), (6, 65)] {
        hits += cast_at_every_shape(seed, side, 400);
    }
    assert!(hits > 2_000, "most rays hit: {hits}");
}

#[test]
fn blocks_hit_as_every_triangle_does_for_any_seed() {
    bolero::check!()
        .with_type::<(u64, u8)>()
        .for_each(|&(seed, side)| {
            let side = u32::from(side % 66).max(2);
            cast_at_every_shape(seed, side, 16);
        });
}

#[test]
fn ties_go_to_the_first_triangle_in_row_major_order() {
    // Two sheets at the same height over each other, in different groups of blocks: every hit
    // on one is at the same `t` on the other, and the first rows win. The first sheet runs
    // over block rows 0 and 1; block row 2 folds back well below the sheets; the second sheet,
    // block row 3, lies over the first's first block row. One vertex of the second sheet, away
    // from where the rays hit, is lifted toward the rays' origins, so its block is entered
    // first, and its group (the fold makes the group unbounded) is visited first: only the
    // comparison of places then gives the first sheet's triangle.
    let side = 33;
    let mut positions = Vec::new();
    for j in 0..side {
        for i in 0..side {
            let x = f32::from(u8::try_from(i).unwrap());
            let p = match j {
                0..=16 => Vec3::new(x, f32::from(u8::try_from(j).unwrap()), 0.0),
                17..=23 => Vec3::new(x, f32::from(u8::try_from(24 - j).unwrap()), -5.0),
                _ => {
                    let lift = if (i, j) == (6, 31) { 0.5 } else { 0.0 };
                    Vec3::new(x, f32::from(u8::try_from(j - 24).unwrap()), lift)
                }
            };
            positions.push(p);
        }
    }
    let blocks = TileBlocks::new(&positions, side);
    let mut rng = Rng(7);
    for k in 0..200 {
        let target = Vec3::new(rng.range(1.0, 3.0), rng.range(1.0, 3.0), 0.0);
        let origin = target + Vec3::new(rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), 3.0);
        let ray = Ray::new(origin, target - origin).unwrap();
        let wide = WideRay::new(&ray);
        let first = blocks.blocks.first().unwrap().reach(&wide).unwrap();
        let second = blocks.blocks[12].reach(&wide).unwrap();
        assert!(
            f64::NEG_INFINITY < second && second < first,
            "ray {k} enters the second sheet's block first: {second} before {first}"
        );
        assert!(
            agrees(&positions, side, &blocks, &ray, &format!("tie {k}")),
            "ray {k} hits the sheets"
        );
        let hit = blocks.intersect(&ray, &wide, &positions, key()).unwrap();
        assert!(
            hit.j < 8,
            "ray {k}: the first sheet's triangle wins the tie"
        );
    }
}

#[test]
fn a_ray_through_a_far_block_skips_the_rest() {
    // A ray straight down on a flat grid: one or two blocks may hold its hit, and the rest are
    // skipped by their boxes.
    let side = 65;
    let positions = Shape::Flat.grid(side, &mut Rng(8));
    let blocks = TileBlocks::new(&positions, side);
    let ray = Ray::new(Vec3::new(0.3, 0.6, 2.0), Vec3::new(0.0, 0.0, -1.0)).unwrap();
    let wide = WideRay::new(&ray);
    let reached = blocks
        .blocks
        .iter()
        .filter(|block| block.reach(&wide).is_some())
        .count();
    assert_eq!(reached, 1, "only the block under the ray is reached");
}

#[test]
fn a_tile_takes_a_few_kilobytes() {
    let side = 65;
    let positions = Shape::Waves.grid(side, &mut Rng(9));
    let blocks = TileBlocks::new(&positions, side);
    assert_eq!(blocks.blocks.len(), 64, "8 × 8 blocks");
    assert_eq!(size_of::<Block>(), 52, "13 f32s a block");
    assert!(
        blocks.bytes() <= 4_300,
        "a finest tile's blocks: {} bytes",
        blocks.bytes()
    );
}

#[test]
fn grazing_hits_far_off_their_triangle_are_kept() {
    // A ray all but in a triangle's plane makes the test's determinant small, and its rounding
    // errors over it can land an accepted hit well off the triangle, edges away. These rays
    // cross a quad's first triangle at a random point, tilted out of its plane just enough
    // that the determinant is 20 to 200 times its rounding (where the blocks may skip a block,
    // given the margin): the blocks must keep every hit. One flat quad a grid, so its own box
    // decides.
    let mut rng = Rng(10);
    let (mut hits, mut far_off) = (0, 0);
    for k in 0..100_000 {
        let scale = 0.01;
        let low = rng.point(-1.0, 1.0);
        // A parallelogram: both triangles in one plane, so the block's normals bound its
        // determinant tightly.
        let (right, up) = (
            low + rng.point(-scale, scale),
            low + rng.point(-scale, scale),
        );
        let far = right + (up - low);
        let normal = (right - low).cross(far - low);
        let (Some(along), Some(unit)) = ((right - low).normalized(), normal.normalized()) else {
            continue;
        };
        let edge = f64::from(one_norm(right - low).max(one_norm(far - low)));
        let rounding = ROUNDOFF * f64::from(one_norm(along)) * edge * edge;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "a small tilt, rounded to f32"
        )]
        let tilt =
            (f64::from(rng.range(20.0, 200.0)) * rounding / f64::from(normal.length())) as f32;
        let direction = along + unit * if rng.below(2) == 0 { tilt } else { -tilt };
        let (a, b) = (rng.unit(), rng.unit());
        let (a, b) = if a + b > 1.0 {
            (1.0 - a, 1.0 - b)
        } else {
            (a, b)
        };
        let target = low + (right - low) * a + (far - low) * b;
        // From well before the triangle, or from just past it, where only a hit the rounding
        // pulls back to `t ≥ 0` is accepted.
        let back = if rng.below(2) == 0 {
            rng.range(0.5, 3.0)
        } else {
            -rng.range(0.0, 0.05)
        };
        let Some(ray) = Ray::new(target - direction * back, direction) else {
            continue;
        };
        let positions = [low, right, up, far];
        let blocks = TileBlocks::new(&positions, 2);
        hits += usize::from(agrees(
            &positions,
            2,
            &blocks,
            &ray,
            &format!("grazing ray {k}"),
        ));
        if let Some(hit) = intersect_grid(&ray, &positions, 2, key()) {
            let point = ray.at(hit.t);
            let block = blocks.blocks.first().unwrap();
            let outside = one_norm(block.low - least(block.low, point))
                + one_norm(greatest(block.high, point) - block.high);
            far_off += usize::from(f64::from(outside) > 1e-3 * edge);
        }
    }
    assert!(
        hits > 25_000,
        "the rays from before the triangle hit: {hits}"
    );
    assert!(
        far_off > 100,
        "many accepted hits lie well off their quad: {far_off}"
    );
}

#[test]
fn grazing_hits_off_their_place_are_kept() {
    // The rounding puts a grazing ray's accepted hits off their triangle mostly along the ray.
    // These rays graze a flat grid of 2 × 2 blocks, tilted out of its plane as above, and
    // cross it within a quad or two of the border between two blocks, heading across: the
    // triangles of the farther block then give hits whose `t` the rounding puts before the
    // block's box, ahead of the nearer block's own hits. The blocks must keep the least `t`
    // all the same.
    let mut rng = Rng(11);
    let side = 17;
    let (mut hits, mut far_off) = (0, 0);
    for k in 0..20_000 {
        let scale = 0.01;
        let (low, across, along) = (
            rng.point(-1.0, 1.0),
            rng.point(-scale, scale),
            rng.point(-scale, scale),
        );
        let places = (0..side).flat_map(|j| (0..side).map(move |i| (i, j)));
        let positions: Vec<Vec3> = places
            .map(|(i, j)| {
                let (i, j) = (u16::try_from(i).unwrap(), u16::try_from(j).unwrap());
                low + across * f32::from(i) + along * f32::from(j)
            })
            .collect();
        let normal = across.cross(along);
        let (Some(heading), Some(unit)) = (across.normalized(), normal.normalized()) else {
            continue;
        };
        let blocks = TileBlocks::new(&positions, side);
        let edge = f64::from(
            blocks
                .blocks
                .iter()
                .map(|block| block.edge)
                .fold(0.0, f32::max),
        );
        let rounding = ROUNDOFF * f64::from(one_norm(heading)) * edge * edge;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "a small tilt, rounded to f32"
        )]
        let tilt =
            (f64::from(rng.range(20.0, 200.0)) * rounding / f64::from(normal.length())) as f32;
        let direction = heading + unit * if rng.below(2) == 0 { tilt } else { -tilt };
        let target = low + across * rng.range(6.5, 8.0) + along * rng.range(1.0, 15.0);
        let Some(ray) = Ray::new(target - direction * rng.range(0.5, 3.0), direction) else {
            continue;
        };
        hits += usize::from(agrees(
            &positions,
            side,
            &blocks,
            &ray,
            &format!("grazing ray {k}"),
        ));
        if let Some(hit) = intersect_grid(&ray, &positions, side, key()) {
            far_off += usize::from(hit.i >= 8);
        }
    }
    assert!(hits > 10_000, "most rays hit: {hits}");
    assert!(
        far_off > 100,
        "many rays' first hit is in the farther block: {far_off}"
    );
}

#[test]
fn a_sweep_bounds_a_block_as_its_quads_do() {
    let mut rng = Rng(12);
    for side in [65, 2, 9, 13, 17, 30] {
        for shape in Shape::ALL {
            let positions = shape.grid(side, &mut rng);
            let swept = Sweep::blocks(&positions, side).unwrap();
            let per_side = (side - 1).div_ceil(BLOCK_QUADS);
            let places =
                (0..per_side).flat_map(|row| (0..per_side).map(move |column| (column, row)));
            let quad_by_quad: Vec<Block> = places
                .map(|place| Block::of_quads(&positions, side, place))
                .collect();
            assert_eq!(swept, quad_by_quad, "{shape:?}, side {side}");
        }
    }
    let mut short = Shape::Waves.grid(17, &mut rng);
    short.truncate(17 * 12);
    let short = short.as_slice();
    assert_eq!(
        Sweep::blocks(short, 17).unwrap(),
        [0, 1]
            .into_iter()
            .flat_map(|row| [0, 1].map(|column| Block::of_quads(short, 17, (column, row))))
            .collect::<Vec<_>>(),
        "a grid short of rows: the rows it has"
    );
}

#[test]
fn a_block_as_near_as_the_best_hit_is_still_tested() {
    let hit = |t: f32| TriangleHit {
        key: key(),
        t,
        i: 0,
        j: 0,
        second: false,
        u: 0.0,
        v: 0.0,
    };
    assert!(can_hold_better(5.0, None), "nothing found yet");
    assert!(can_hold_better(1.0, Some(&hit(2.0))), "a nearer block");
    assert!(
        can_hold_better(2.0, Some(&hit(2.0))),
        "a block at the hit's `t`: a tie it may win by place"
    );
    assert!(
        can_hold_better(0.0, Some(&hit(-0.0))),
        "`0` and `−0` are the same `t`"
    );
    assert!(
        !can_hold_better(2.000_000_1, Some(&hit(2.0))),
        "a block beyond the hit"
    );
    assert!(
        !can_hold_better(f64::from(f32::INFINITY), Some(&hit(f32::MAX))),
        "a block at infinity, beyond a finite hit"
    );
}
