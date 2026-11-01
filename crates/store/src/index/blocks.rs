//! A tile's blocks, for raycasts: the triangles a ray can't hit, skipped without testing them.
//!
//! [`TileBlocks`] cuts a tile's grid of quads into blocks of [`BLOCK_QUADS`] × [`BLOCK_QUADS`]
//! quads and keeps, for each, what bounds its triangles: the box of its vertices, the box of
//! its triangles' unnormalised normals, and the longest of their edges. A raycast tests the ray
//! against each block first ([`Block::reach`]), visits the blocks it may hit nearest first,
//! and tests the triangles of those only, quad by quad, with the very test the whole tile would
//! have had ([`intersect_triangle`]). It stops once no block left can hold a nearer hit, or
//! one as near and earlier in the tile.
//!
//! # The same answer, bit for bit
//!
//! The answer is the exhaustive test's: of the triangles [`intersect_triangle`] accepts, the
//! least `t`, ties to the first in row-major order, a quad's first triangle before its second
//! (see [`hit_before`]). Every triangle that could be accepted is tested, and the blocks are
//! visited in any order but compared by `(t, place)`, so the order of the visit changes
//! nothing. A block is skipped only when no triangle of it can be accepted, or none at a `t`
//! at most the best found, and that has to hold for the test's rounded arithmetic, not the
//! exact geometry: a ray nearly in a triangle's plane makes the test's determinant small, and
//! its rounding errors, divided by it, can put an accepted hit well off the triangle (several
//! quads off for a ray a few thousandths of a radian from the surface). So the block's box is
//! widened by a margin that bounds how far off an accepted hit can be, worked out from the
//! test's own operations (see [`Block::reach`]):
//!
//! - Write `F = origin − corner`, `E1`, `E2` the triangle's edges and `D = E1 · (d × E2)` the
//!   determinant, all as the test computes them. Cramer's rule is an identity, so in exact
//!   arithmetic the hit `origin + t d` and the triangle's point `corner + u E1 + v E2` are the
//!   same. With the test's rounded numerators and determinant (each off by at most `γ₅` times the
//!   sum of its products' magnitudes, `γₙ = n u / (1 − n u)`, `u = 2⁻²⁴`), they differ by at most
//!   `4 γ₅ ‖d‖₁ e² ‖F‖₁ / |D|`, `e` the longest edge's 1-norm.
//! - `|D|` is bounded below for the whole block from the box of its normals `E1 × E2`: `D = −d ·
//!   (E1 × E2)`, and over a box the dot product's magnitude is least at a corner, so the interval
//!   arithmetic gives it, less the rounding of the normals and of `D` itself (`9 u ‖d‖₁ e²`). A
//!   block whose normals the ray may run perpendicular to (it crosses the block's silhouette, or
//!   the block holds a degenerate triangle) has no such bound and is always tested.
//! - The accepted barycentric weights reach [`EDGE_SLACK`] past the triangle's edges, and the final
//!   multiplications by the determinant's reciprocal round them and `t` by `γ₂`.
//!
//! The margin is taken generously (constants rounded up, a floor for underflow), in `f64`,
//! where the slab test's own rounding is far below it. A block whose numbers are too large for
//! the bound (a coordinate of `10¹²` or more, where the test's `f32` products could overflow)
//! is always tested. In practice the margin is a few millionths of the distance from the ray's
//! origin, and the blocks a ray passes far from are skipped; only near the silhouette, where
//! the ray runs along the surface, are more blocks tested whole.
//!
//! # Cost
//!
//! Blocks are grouped [`GROUP`] × [`GROUP`], each group bounding its blocks' triangles alike,
//! so a ray tests the 16 groups of a finest tile and the blocks of the few it may hit, not all
//! 64 blocks.
//!
//! Building a tile's blocks costs a pass over its quads (two cross products each, a row at a
//! time in vectorised passes, see [`Sweep`]): about 0.4 of testing every triangle once,
//! measured. Testing a ray against built blocks costs about a thirtieth of that. [`BlockCache`]
//! keeps the blocks of the tiles raycast last, while the store's tiles evaluate to the bits
//! they were built from (see [`Written`]), so the raycasts between two edits share them. Each
//! block or group is 52 bytes, 80 to a finest tile: 4.2 KB a tile, under a tenth of its
//! evaluated positions.

use std::cmp::Ordering;
use std::ops::Range;
use std::sync::{Mutex, MutexGuard, PoisonError};

use geom::{TILE_QUADS, TileKey, Vec3};

use super::{EDGE_SLACK, Ray, TriangleHit, intersect_triangle};
use crate::tiles::{StoreId, TileStore};

/// Quads per side of a block: a finest tile of 64 × 64 quads is 8 × 8 blocks.
pub(super) const BLOCK_QUADS: u32 = 8;

/// The unit roundoff of `f32`, `2⁻²⁴`, rounded up: the most one rounding changes a value,
/// relative to it.
const ROUNDOFF: f64 = 5.960_464_477_539_063e-8;

/// A floor for the absolute rounding errors of numbers that underflow: an `f32` operation whose
/// result is subnormal is off by at most half of `2⁻¹⁴⁹`, and the test does a few dozen.
const UNDERFLOW: f64 = 1e-40;

/// The largest magnitude the margin holds for.
///
/// A block's coordinates, edges and distance from the ray's origin must be below it, so the
/// test's products of three of them stay far below `f32::MAX`. A block beyond it is always
/// tested.
const LARGEST: f64 = 1e12;

/// The least determinant bound a block is skipped with: below it, the margin it gives is too
/// wide to skip anything anyway, and the reciprocal is far from overflowing `f32`.
const LEAST_DETERMINANT: f64 = 1e-30;

/// What bounds the triangles of one block of a tile: see the module notes.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Block {
    /// The least corner of the box of the block's vertices.
    low: Vec3,
    /// The greatest corner of that box.
    high: Vec3,
    /// The least corner of the box of the block's triangles' normals `E1 × E2`, unnormalised,
    /// as the raycast's test computes the edges.
    normal_low: Vec3,
    /// The greatest corner of that box.
    normal_high: Vec3,
    /// The greatest 1-norm of the triangles' edges from their first corner, as the test
    /// computes them, rounded as `f32` computes the sum.
    edge: f32,
}

impl Block {
    /// A block of no quads yet: every box empty (its least corner above its greatest).
    const EMPTY: Self = Self {
        low: Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY),
        high: Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
        normal_low: Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY),
        normal_high: Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
        edge: 0.0,
    };

    /// The bounds of both `self`'s triangles and `other`'s: each box the union of the two,
    /// the longer edge.
    fn union(self, other: &Self) -> Self {
        Self {
            low: least(self.low, other.low),
            high: greatest(self.high, other.high),
            normal_low: least(self.normal_low, other.normal_low),
            normal_high: greatest(self.normal_high, other.normal_high),
            edge: self.edge.max(other.edge),
        }
    }

    /// Block `(column, row)` of the grid of `side` vertices a side with `positions`, quad by
    /// quad: the plain statement of what [`Sweep`] computes faster, which its tests check it
    /// against, and the blocks of a grid too large for it.
    fn of_quads(positions: &[Vec3], side: u32, (column, row): (u32, u32)) -> Self {
        let mut block = Self::EMPTY;
        let quads = side.saturating_sub(1);
        let (i0, j0) = (column * BLOCK_QUADS, row * BLOCK_QUADS);
        let (i1, j1) = ((i0 + BLOCK_QUADS).min(quads), (j0 + BLOCK_QUADS).min(quads));
        let span = |j: u32| row_span(positions, side, j, i0..=i1);
        for j in j0..j1 {
            let (Some(bottom), Some(top)) = (span(j), span(j + 1)) else {
                continue;
            };
            let pairs = bottom.windows(2).zip(top.windows(2));
            for pair in pairs {
                if let (&[low, right], &[up, far]) = pair {
                    block.add_quad(low, right, up, far);
                }
            }
        }
        block
    }

    /// Takes in the quad `low`, `right`, `up`, `far` (its corners `(i, j)`, `(i + 1, j)`,
    /// `(i, j + 1)`, `(i + 1, j + 1)`): its corners, its two triangles' normals and its edges
    /// from `low`, as [`intersect_grid`](super::intersect_grid)'s triangles have them.
    ///
    /// A non-finite value is left out of the boxes (`f32::min` and `max` pass over a NaN): a
    /// triangle with a non-finite corner is never accepted, its determinant being non-finite,
    /// and one whose finite corners give a non-finite normal has an edge of `10¹⁹` or more,
    /// which [`LARGEST`] keeps from being skipped.
    fn add_quad(&mut self, low: Vec3, right: Vec3, up: Vec3, far: Vec3) {
        let corners = [low, right, up, far];
        for corner in corners {
            self.low = least(self.low, corner);
            self.high = greatest(self.high, corner);
        }
        let (to_right, to_far, to_up) = (right - low, far - low, up - low);
        for normal in [to_right.cross(to_far), to_far.cross(to_up)] {
            self.normal_low = least(self.normal_low, normal);
            self.normal_high = greatest(self.normal_high, normal);
        }
        self.edge = self
            .edge
            .max(one_norm(to_right))
            .max(one_norm(to_far))
            .max(one_norm(to_up));
    }

    /// Where along `ray` a triangle of the block may be hit: a lower bound of the `t` of any
    /// hit [`intersect_triangle`] accepts on one, or `None` if it accepts none. Negative
    /// infinity for a block that can't be bounded, to be tested whatever the hits found.
    ///
    /// The box of the block's vertices is widened by the margin of the module notes, and the
    /// ray's line cut against it: an accepted hit's `t`, before the test's last rounding, lies
    /// in the interval it is in the box over, and that rounding moves it by `γ₂` of itself. The
    /// test accepts `t ≥ 0` only, `−0` included, so a block whose interval ends below zero
    /// holds no accepted hit.
    fn reach(&self, ray: &WideRay) -> Option<f64> {
        let unbounded = Some(f64::NEG_INFINITY);
        let (low, high) = (wide(self.low), wide(self.high));
        let edge = f64::from(self.edge) * (1.0 + 4.0 * ROUNDOFF);
        let reach: [f64; 3] = std::array::from_fn(|axis| {
            let origin = at(ray.origin, axis);
            (origin - at(low, axis))
                .abs()
                .max((origin - at(high, axis)).abs())
        });
        let from_origin = reach.iter().sum::<f64>() * (1.0 + ROUNDOFF);
        let bounded = edge < LARGEST && from_origin < LARGEST && ray.length < LARGEST;
        if !bounded {
            return unbounded;
        }
        let determinant = self.least_determinant(ray, edge);
        if determinant.is_nan() || determinant < LEAST_DETERMINANT {
            return unbounded;
        }
        let errors = 24.0 * ROUNDOFF * ray.length * edge * edge * from_origin
            + UNDERFLOW * (from_origin + ray.length + 2.0 * edge + 1.0);
        let slack = 3.0 * (f64::from(EDGE_SLACK) + 4.0 * ROUNDOFF) + 2.0 * ROUNDOFF;
        let margin = errors / determinant + 2.0 * ROUNDOFF * from_origin + slack * edge;
        let (near, far) = ray.through(low, high, margin)?;
        let rounding = 4.0 * ROUNDOFF;
        if far + rounding * far.abs() + UNDERFLOW < 0.0 {
            return None;
        }
        Some(near - rounding * near.abs() - UNDERFLOW)
    }

    /// A lower bound of `|D|`, the determinant the test computes, over the block's triangles,
    /// for `ray`; zero or less if there is none (the ray may run along one of them). `edge` is
    /// the block's longest edge, its rounding allowed for.
    fn least_determinant(&self, ray: &WideRay, edge: f64) -> f64 {
        let (low, high) = (wide(self.normal_low), wide(self.normal_high));
        let (mut least, mut most) = (0.0, 0.0);
        for axis in 0..3 {
            let d = at(ray.direction, axis);
            let (a, b) = (d * at(low, axis), d * at(high, axis));
            least += a.min(b);
            most += a.max(b);
        }
        let across = if least > 0.0 {
            least
        } else if most < 0.0 {
            -most
        } else {
            return 0.0;
        };
        across * (1.0 - 1e-12) - 16.0 * ROUNDOFF * ray.length * edge * edge - UNDERFLOW
    }
}

/// The most quads a row of a tile has: [`Sweep`]'s rows are this long.
const ROW: usize = TILE_QUADS as usize;

/// One component of a row of quads, a value a quad.
type Quads = [f32; ROW];

/// One component of a row of vertices.
type Vertices = [f32; ROW + 1];

/// A row of vertices, a component at a time.
#[derive(Debug, Clone, Copy)]
struct Row {
    /// The x components.
    x: Vertices,
    /// The y components.
    y: Vertices,
    /// The z components.
    z: Vertices,
}

impl Row {
    /// A row of `value`s.
    const fn filled(value: f32) -> Self {
        Self {
            x: [value; ROW + 1],
            y: [value; ROW + 1],
            z: [value; ROW + 1],
        }
    }

    /// Takes the components of `points`, as many as there are and the row holds.
    fn load(&mut self, points: &[Vec3]) {
        let lanes = self
            .x
            .iter_mut()
            .zip(&mut self.y)
            .zip(&mut self.z)
            .zip(points);
        for (((x, y), z), p) in lanes {
            (*x, *y, *z) = (p.x, p.y, p.z);
        }
    }
}

/// One component of the bounds of a row of quads, a value a quad.
#[derive(Debug, Clone, Copy)]
struct QuadBounds {
    /// The least of the normals' x components.
    low_x: Quads,
    /// Their y components'.
    low_y: Quads,
    /// Their z components'.
    low_z: Quads,
    /// The greatest of the normals' x components.
    high_x: Quads,
    /// Their y components'.
    high_y: Quads,
    /// Their z components'.
    high_z: Quads,
    /// The greatest of the edges' 1-norms.
    edge: Quads,
}

impl QuadBounds {
    /// Bounds of nothing yet.
    const EMPTY: Self = Self {
        low_x: [f32::INFINITY; ROW],
        low_y: [f32::INFINITY; ROW],
        low_z: [f32::INFINITY; ROW],
        high_x: [f32::NEG_INFINITY; ROW],
        high_y: [f32::NEG_INFINITY; ROW],
        high_z: [f32::NEG_INFINITY; ROW],
        edge: [0.0; ROW],
    };

    /// Takes in the first `quads` quads between the rows of vertices `bottom` and `top`: each
    /// one's triangles' normals and its edges from its first corner, as [`Block::add_quad`]
    /// computes them, all in one pass, which the compiler vectorises.
    fn take(&mut self, bottom: &Row, top: &Row, quads: usize) {
        let at = |c: &Vertices, i: usize| c.get(i).copied().unwrap_or(0.0);
        let put = |c: &mut Quads, i: usize, value: f32, f: fn(f32, f32) -> f32| {
            if let Some(c) = c.get_mut(i) {
                *c = f(*c, value);
            }
        };
        for i in 0..quads.min(ROW) {
            let (low_x, low_y, low_z) = (at(&bottom.x, i), at(&bottom.y, i), at(&bottom.z, i));
            let right = (
                at(&bottom.x, i + 1) - low_x,
                at(&bottom.y, i + 1) - low_y,
                at(&bottom.z, i + 1) - low_z,
            );
            let far = (
                at(&top.x, i + 1) - low_x,
                at(&top.y, i + 1) - low_y,
                at(&top.z, i + 1) - low_z,
            );
            let up = (
                at(&top.x, i) - low_x,
                at(&top.y, i) - low_y,
                at(&top.z, i) - low_z,
            );
            let cross = |a: (f32, f32, f32), b: (f32, f32, f32)| {
                (
                    a.1 * b.2 - a.2 * b.1,
                    a.2 * b.0 - a.0 * b.2,
                    a.0 * b.1 - a.1 * b.0,
                )
            };
            let norm = |a: (f32, f32, f32)| a.0.abs() + a.1.abs() + a.2.abs();
            // Each value taken into its bound on its own: a NaN is passed over by the bound,
            // which never is one, and so leaves the quad's other values in.
            for normal in [cross(right, far), cross(far, up)] {
                put(&mut self.low_x, i, normal.0, lesser);
                put(&mut self.low_y, i, normal.1, lesser);
                put(&mut self.low_z, i, normal.2, lesser);
                put(&mut self.high_x, i, normal.0, greater);
                put(&mut self.high_y, i, normal.1, greater);
                put(&mut self.high_z, i, normal.2, greater);
            }
            for edge in [right, far, up] {
                put(&mut self.edge, i, norm(edge), greater);
            }
        }
    }
}

/// The lesser of `a` and `b`, keeping `a` if `b` is a NaN.
///
/// One instruction for a vector of them. Folded over values from infinity it is their least,
/// NaNs passed over, as `f32::min` folds them; `a`, the bound, is never a NaN.
fn lesser(a: f32, b: f32) -> f32 {
    if b < a { b } else { a }
}

/// The greater of `a` and `b`, keeping `a` if `b` is a NaN (see [`lesser`]).
fn greater(a: f32, b: f32) -> f32 {
    if b > a { b } else { a }
}

/// `values` folded with `pick` ([`lesser`] or [`greater`]) from `from`, which is no NaN.
///
/// Folded in eight lanes, then the lanes pairwise, so the steps don't wait on each other one
/// by one: the same value as folding them in order, as the least or greatest of numbers is.
fn reduce(values: &[f32], from: f32, pick: fn(f32, f32) -> f32) -> f32 {
    let mut lanes = [from; 8];
    for chunk in values.chunks(lanes.len()) {
        let pairs = lanes.iter_mut().zip(chunk);
        for (lane, &value) in pairs {
            *lane = pick(*lane, value);
        }
    }
    let [l0, l1, l2, l3, l4, l5, l6, l7] = lanes;
    let (quarter0, quarter1) = (pick(l0, l1), pick(l2, l3));
    let (quarter2, quarter3) = (pick(l4, l5), pick(l6, l7));
    pick(pick(quarter0, quarter1), pick(quarter2, quarter3))
}

/// A tile's blocks built a row of quads at a time, a whole row in one vectorised pass (see
/// [`QuadBounds::take`]): the same bounds as [`Block::of_quads`] gives each block, several
/// times faster.
///
/// The bounds are kept a column of quads (or vertices) each down a band of [`BLOCK_QUADS`]
/// rows of quads, and folded into the band's blocks at its end. For a tile of at most
/// [`TILE_QUADS`] quads a side, as every tile is.
#[derive(Debug)]
struct Sweep {
    /// The row of vertices below the quads.
    bottom: Row,
    /// The row of vertices above them.
    top: Row,
    /// The band's quads' bounds so far.
    quads: QuadBounds,
    /// The least of the band's vertices so far, a column each.
    low: Row,
    /// The greatest of them.
    high: Row,
}

impl Sweep {
    /// The blocks, row-major, of the grid of `side` vertices a side with `positions`; the
    /// rows of quads the positions don't reach have empty blocks. `None` for a grid of more
    /// than [`TILE_QUADS`] quads a side.
    fn blocks(positions: &[Vec3], side: u32) -> Option<Vec<Block>> {
        let width = side as usize;
        let quads = width.saturating_sub(1);
        if quads > ROW {
            return None;
        }
        let per_row = quads.div_ceil(BLOCK_QUADS as usize);
        let mut blocks = Vec::with_capacity(per_row * per_row);
        let mut rows = positions.chunks_exact(width.max(1)).take(width);
        if let (Some(first), true) = (rows.next(), quads > 0) {
            let mut sweep = Box::new(Self {
                bottom: Row::filled(0.0),
                top: Row::filled(0.0),
                quads: QuadBounds::EMPTY,
                low: Row::filled(f32::INFINITY),
                high: Row::filled(f32::NEG_INFINITY),
            });
            sweep.bottom.load(first);
            sweep.take_vertices(width);
            let numbered = rows.zip(1_usize..);
            let mut open = false;
            for (row, j) in numbered {
                sweep.top.load(row);
                sweep.quads.take(&sweep.bottom, &sweep.top, quads);
                std::mem::swap(&mut sweep.bottom, &mut sweep.top);
                sweep.take_vertices(width);
                open = j % BLOCK_QUADS as usize != 0;
                if !open {
                    sweep.end_band(&mut blocks, width);
                }
            }
            // The last band, short of rows of quads, or of positions.
            if open {
                sweep.end_band(&mut blocks, width);
            }
        }
        blocks.resize(per_row * per_row, Block::EMPTY);
        Some(blocks)
    }

    /// Takes the first `width` vertices of the row in [`Sweep::bottom`] into the band's
    /// bounds.
    fn take_vertices(&mut self, width: usize) {
        let pairs = [
            (&mut self.low.x, &mut self.high.x, &self.bottom.x),
            (&mut self.low.y, &mut self.high.y, &self.bottom.y),
            (&mut self.low.z, &mut self.high.z, &self.bottom.z),
        ];
        for (lows, highs, values) in pairs {
            let lanes = lows
                .iter_mut()
                .zip(highs.iter_mut())
                .zip(values)
                .take(width);
            for ((low, high), &value) in lanes {
                *low = lesser(*low, value);
                *high = greater(*high, value);
            }
        }
    }

    /// Folds the band's bounds into its blocks, appended to `blocks`, and starts the next band
    /// at the row in [`Sweep::bottom`], the band's last. `width` is the rows' vertices.
    fn end_band(&mut self, blocks: &mut Vec<Block>, width: usize) {
        let quads = width.saturating_sub(1);
        let step = BLOCK_QUADS as usize;
        let starts = (0..quads).step_by(step);
        for start in starts {
            let end = (start + step).min(quads);
            // A block's vertices run one column past its quads.
            let vertices = start..end + 1;
            let fold = |values: &[f32], range: Range<usize>, from: f32, f: fn(f32, f32) -> f32| {
                reduce(values.get(range).unwrap_or_default(), from, f)
            };
            let (up, down) = (f32::INFINITY, f32::NEG_INFINITY);
            let q = &self.quads;
            blocks.push(Block {
                low: Vec3::new(
                    fold(&self.low.x, vertices.clone(), up, lesser),
                    fold(&self.low.y, vertices.clone(), up, lesser),
                    fold(&self.low.z, vertices.clone(), up, lesser),
                ),
                high: Vec3::new(
                    fold(&self.high.x, vertices.clone(), down, greater),
                    fold(&self.high.y, vertices.clone(), down, greater),
                    fold(&self.high.z, vertices, down, greater),
                ),
                normal_low: Vec3::new(
                    fold(&q.low_x, start..end, up, lesser),
                    fold(&q.low_y, start..end, up, lesser),
                    fold(&q.low_z, start..end, up, lesser),
                ),
                normal_high: Vec3::new(
                    fold(&q.high_x, start..end, down, greater),
                    fold(&q.high_y, start..end, down, greater),
                    fold(&q.high_z, start..end, down, greater),
                ),
                edge: fold(&q.edge, start..end, 0.0, greater),
            });
        }
        self.quads = QuadBounds::EMPTY;
        self.low = Row::filled(f32::INFINITY);
        self.high = Row::filled(f32::NEG_INFINITY);
        self.take_vertices(width);
    }
}

/// The vertices `columns` of row `j` of the grid of `side` vertices a side with `positions`;
/// `None` unless the row is whole there, as the exhaustive test, which takes whole rows only,
/// sees it.
fn row_span(
    positions: &[Vec3],
    side: u32,
    j: u32,
    columns: std::ops::RangeInclusive<u32>,
) -> Option<&[Vec3]> {
    let width = side as usize;
    let row = positions.chunks_exact(width.max(1)).nth(j as usize)?;
    row.get(*columns.start() as usize..=*columns.end() as usize)
}

/// The componentwise least of `a` and `b`, passing over a NaN.
fn least(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(a.x.min(b.x), a.y.min(b.y), a.z.min(b.z))
}

/// The componentwise greatest of `a` and `b`, passing over a NaN.
fn greatest(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z))
}

/// `v`'s 1-norm, the sum of its components' magnitudes, rounded as `f32` sums it.
fn one_norm(v: Vec3) -> f32 {
    v.x.abs() + v.y.abs() + v.z.abs()
}

/// `v` in `f64`, exactly.
fn wide(v: Vec3) -> [f64; 3] {
    [f64::from(v.x), f64::from(v.y), f64::from(v.z)]
}

/// Component `axis` (0, 1 or 2) of `v`.
fn at(v: [f64; 3], axis: usize) -> f64 {
    v.get(axis).copied().unwrap_or(0.0)
}

/// A ray in `f64`, with its direction's 1-norm: what every block's [`Block::reach`] reads.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct WideRay {
    /// Where the ray starts.
    origin: [f64; 3],
    /// Which way it goes.
    direction: [f64; 3],
    /// The direction's components' reciprocals (infinite where one is zero, and unused there).
    inverse: [f64; 3],
    /// The direction's 1-norm, rounded up.
    length: f64,
}

impl WideRay {
    /// `ray` in `f64`.
    pub(super) fn new(ray: &Ray) -> Self {
        let direction = wide(ray.direction());
        Self {
            origin: wide(ray.origin()),
            direction,
            inverse: direction.map(f64::recip),
            length: direction.iter().map(|d| d.abs()).sum::<f64>() * (1.0 + 1e-12),
        }
    }

    /// The interval of `t` over which the ray's line is in the box `low`–`high` widened by
    /// `margin` on every side, itself widened past the slab test's rounding; `None` if the
    /// line misses it.
    ///
    /// Each plane is moved out by a further `10⁻⁹` of the magnitudes involved, far more than
    /// the `f64` subtraction's rounding, so the planes computed lie outside the true ones; the
    /// rounding of the multiplications by the direction's reciprocals (a few parts in `10¹⁶`)
    /// is then allowed for by widening the interval by `10⁻¹²` of its ends. An axis the
    /// direction is zero along keeps the whole line or none of it.
    fn through(&self, low: [f64; 3], high: [f64; 3], margin: f64) -> Option<(f64, f64)> {
        let (mut near, mut far) = (f64::NEG_INFINITY, f64::INFINITY);
        for axis in 0..3 {
            let (origin, direction) = (at(self.origin, axis), at(self.direction, axis));
            let inverse = at(self.inverse, axis);
            let (low, high) = (at(low, axis), at(high, axis));
            let scale = origin.abs() + low.abs().max(high.abs()) + margin + 1.0;
            let widen = margin + 1e-9 * scale;
            let (low, high) = (low - widen, high + widen);
            if direction == 0.0 {
                if !(low <= origin && origin <= high) {
                    return None;
                }
                continue;
            }
            let (t0, t1) = ((low - origin) * inverse, (high - origin) * inverse);
            near = near.max(t0.min(t1));
            far = far.min(t0.max(t1));
        }
        let (near, far) = (near - 1e-12 * near.abs(), far + 1e-12 * far.abs());
        (near <= far).then_some((near, far))
    }
}

/// A tile's blocks: see the module notes.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct TileBlocks {
    /// Vertices per side of the tile's grid.
    side: u32,
    /// Blocks per side: the tile's quads per side over [`BLOCK_QUADS`], rounded up.
    per_side: u32,
    /// The blocks, row-major: block `(bx, by)` holds the quads `(i, j)` with
    /// `i / BLOCK_QUADS == bx` and `j / BLOCK_QUADS == by`.
    blocks: Vec<Block>,
    /// The blocks' groups of [`GROUP`] × [`GROUP`], row-major, each the union of its blocks'
    /// bounds: a raycast tests a group first, and its blocks only if the group may be hit.
    groups: Vec<Block>,
}

impl TileBlocks {
    /// The blocks of a tile's grid of `side` vertices a side, its `side²` positions row-major.
    ///
    /// A tile always has them all. Were it short, the blocks would bound the whole rows there
    /// are, and [`TileBlocks::intersect`] test only those, as the exhaustive test does; a debug
    /// build asserts it isn't.
    pub(super) fn new(positions: &[Vec3], side: u32) -> Self {
        debug_assert!(
            positions.len() >= (side as usize).pow(2),
            "a tile's grid has all its positions"
        );
        let per_side = side.saturating_sub(1).div_ceil(BLOCK_QUADS);
        let blocks = Sweep::blocks(positions, side).unwrap_or_else(|| {
            let places =
                (0..per_side).flat_map(|row| (0..per_side).map(move |column| (column, row)));
            places
                .map(|place| Block::of_quads(positions, side, place))
                .collect()
        });
        let per_group = per_side.div_ceil(GROUP);
        let places = (0..per_group).flat_map(|row| (0..per_group).map(move |column| (column, row)));
        let groups = places
            .map(|(column, row)| {
                let members = group_members(per_side, (column, row));
                members
                    .filter_map(|n| blocks.get(n as usize))
                    .fold(Block::EMPTY, Block::union)
            })
            .collect();
        Self {
            side,
            per_side,
            blocks,
            groups,
        }
    }

    /// The bytes the blocks and their groups take, the vectors' spare capacity included.
    #[cfg(test)]
    fn bytes(&self) -> usize {
        size_of::<Self>() + (self.blocks.capacity() + self.groups.capacity()) * size_of::<Block>()
    }

    /// The nearest triangle `ray` crosses of the grid with `positions` these blocks were built
    /// from, named for tile `key`: what
    /// [`intersect_grid`](super::intersect_grid) gives, bit for bit, from the triangles of the
    /// blocks the ray may hit (see the module notes). `wide` is `ray` in `f64`.
    pub(super) fn intersect(
        &self,
        ray: &Ray,
        wide: &WideRay,
        positions: &[Vec3],
        key: TileKey,
    ) -> Option<TriangleHit> {
        let per_group = self.per_side.div_ceil(GROUP);
        let groups = nearest_first(&self.groups, 0..per_group * per_group, wide);
        let mut best: Option<TriangleHit> = None;
        for (group_from, group) in groups {
            if !can_hold_better(group_from, best.as_ref()) {
                break;
            }
            let members = group_members(self.per_side, (group % per_group, group / per_group));
            let blocks = nearest_first(&self.blocks, members, wide);
            for (from, n) in blocks {
                if !can_hold_better(from, best.as_ref()) {
                    break;
                }
                let (column, row) = (n % self.per_side, n / self.per_side);
                self.intersect_block(ray, positions, key, (column, row), &mut best);
            }
        }
        best
    }

    /// Tests `ray` against every triangle of block `(column, row)`, keeping in `best` the
    /// first of them and it by [`hit_before`].
    fn intersect_block(
        &self,
        ray: &Ray,
        positions: &[Vec3],
        key: TileKey,
        (column, row): (u32, u32),
        best: &mut Option<TriangleHit>,
    ) {
        let quads = self.side.saturating_sub(1);
        let (i0, j0) = (column * BLOCK_QUADS, row * BLOCK_QUADS);
        let (i1, j1) = ((i0 + BLOCK_QUADS).min(quads), (j0 + BLOCK_QUADS).min(quads));
        let span = |j: u32| row_span(positions, self.side, j, i0..=i1);
        for j in j0..j1 {
            let (Some(bottom), Some(top)) = (span(j), span(j + 1)) else {
                continue;
            };
            let pairs = bottom.windows(2).zip(top.windows(2)).zip(i0..);
            for (pair, i) in pairs {
                let (&[low, right], &[up, far]) = pair else {
                    continue;
                };
                let triangles = [(false, low, right, far), (true, low, far, up)];
                for (second, corner, p, q) in triangles {
                    let Some((t, u, v)) = intersect_triangle(ray, corner, p, q) else {
                        continue;
                    };
                    let hit = TriangleHit {
                        key,
                        t,
                        i,
                        j,
                        second,
                        u,
                        v,
                    };
                    if best.is_none_or(|best| hit_before(&hit, &best)) {
                        *best = Some(hit);
                    }
                }
            }
        }
    }
}

/// Blocks a side of a group: see [`TileBlocks::groups`].
const GROUP: u32 = 2;

/// The blocks of group `(column, row)` of a tile of `per_side` blocks a side, by their places
/// in [`TileBlocks::blocks`].
fn group_members(per_side: u32, (column, row): (u32, u32)) -> impl Iterator<Item = u32> {
    let (x0, y0) = (column * GROUP, row * GROUP);
    let (x1, y1) = ((x0 + GROUP).min(per_side), (y0 + GROUP).min(per_side));
    (y0..y1).flat_map(move |y| (x0..x1).map(move |x| y * per_side + x))
}

/// The blocks `which` of `blocks` that `ray` may hit, each with the least `t` it may hit one
/// at ([`Block::reach`]), least first (ties in order of place).
fn nearest_first(
    blocks: &[Block],
    which: impl Iterator<Item = u32>,
    ray: &WideRay,
) -> Vec<(f64, u32)> {
    let reached = which.filter_map(|n| Some((blocks.get(n as usize)?.reach(ray)?, n)));
    let mut order: Vec<(f64, u32)> = reached.collect();
    order.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    order
}

/// Whether a block whose hits are at `t ≥ from` may hold one before `best` (see
/// [`hit_before`]): a hit at the same `t` may still come first, by its place.
fn can_hold_better(from: f64, best: Option<&TriangleHit>) -> bool {
    best.is_none_or(|best| from <= f64::from(best.t))
}

/// Whether hit `a` comes before hit `b` on one tile, as the exhaustive test chooses.
///
/// A smaller `t`, or the same `t` (`−0` and `0` alike) and an earlier triangle in row-major
/// order, a quad's first triangle before its second: the exhaustive test keeps the first
/// triangle of the least `t`.
fn hit_before(a: &TriangleHit, b: &TriangleHit) -> bool {
    match a.t.partial_cmp(&b.t) {
        Some(Ordering::Less) => true,
        Some(Ordering::Equal) => (a.j, a.i, a.second) < (b.j, b.i, b.second),
        Some(Ordering::Greater) | None => false,
    }
}

/// The state of a store's tiles that blocks were built from.
///
/// The store, and its count of writes ([`TileStore::edits`]). While both stay the same every
/// tile evaluates to the same bits, the very guarantee [`WaveBounds`](crate::tiles::WaveBounds)
/// relies on, so blocks built then are still the tile's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Written {
    /// The store.
    store: StoreId,
    /// Its count of writes.
    edits: u64,
}

impl Written {
    /// `store`'s state now.
    pub(super) const fn of(store: &TileStore) -> Self {
        Self {
            store: store.id(),
            edits: store.edits(),
        }
    }
}

/// The blocks of the tiles raycast last, while the store's tiles stay as they were built from
/// (see [`Written`]): a change of state drops them all.
///
/// It holds [`BlockCache::TILES`] tiles at most, the least recently used making room: about
/// 270 KB for finest tiles (4.2 KB each), kept beside the store's budget rather than in it, as
/// the index's own memory is.
#[derive(Debug, Default)]
pub(super) struct BlockCache {
    /// The state the blocks were built in; `None` before any.
    written: Option<Written>,
    /// The tiles' blocks, each with when it was last used.
    tiles: Vec<Cached>,
    /// A count of uses, for the least recently used.
    clock: u64,
}

/// One tile's blocks in a [`BlockCache`].
#[derive(Debug)]
struct Cached {
    /// The tile.
    key: TileKey,
    /// The [`BlockCache::clock`] when they were last used.
    used: u64,
    /// The blocks.
    blocks: TileBlocks,
}

impl BlockCache {
    /// The most tiles kept.
    pub(super) const TILES: usize = 64;

    /// The blocks of tile `key`, whose grid of `side` vertices a side has `positions` now, in
    /// state `written`: kept if they were built in that state, else built and kept, in place of
    /// the least recently used if the cache is full.
    ///
    /// Always `Some`: the tile's place is the one found or just filled.
    pub(super) fn blocks(
        &mut self,
        written: Written,
        key: TileKey,
        positions: &[Vec3],
        side: u32,
    ) -> Option<&TileBlocks> {
        if self.written != Some(written) {
            self.tiles.clear();
            self.written = Some(written);
        }
        self.clock += 1;
        let used = self.clock;
        let found = self
            .tiles
            .iter()
            .position(|cached| cached.key == key && cached.blocks.side == side);
        let at = found.unwrap_or_else(|| {
            let cached = Cached {
                key,
                used,
                blocks: TileBlocks::new(positions, side),
            };
            let oldest = self
                .tiles
                .iter()
                .enumerate()
                .min_by_key(|(_, kept)| kept.used)
                .map(|(at, _)| at)
                .filter(|_| self.tiles.len() >= Self::TILES);
            if let Some((at, slot)) = oldest.and_then(|at| Some((at, self.tiles.get_mut(at)?))) {
                *slot = cached;
                at
            } else {
                self.tiles.push(cached);
                self.tiles.len() - 1
            }
        });
        let cached = self.tiles.get_mut(at)?;
        cached.used = used;
        Some(&cached.blocks)
    }
}

/// The index's [`BlockCache`], behind a lock: raycasts take the index by shared reference.
///
/// A raycast holds it throughout, so raycasts through one index from several threads take
/// turns. A clone starts empty: a cloned index builds its own blocks as it raycasts.
#[derive(Debug, Default)]
pub(super) struct SharedBlocks(Mutex<BlockCache>);

impl SharedBlocks {
    /// The cache, for one raycast.
    ///
    /// A raycast that panicked while holding it left it whole (each change is one assignment,
    /// and every tile's blocks are checked against the store before use), so a poisoned lock
    /// is taken as it is.
    pub(super) fn lock(&self) -> MutexGuard<'_, BlockCache> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Clone for SharedBlocks {
    /// An empty cache.
    fn clone(&self) -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests;
