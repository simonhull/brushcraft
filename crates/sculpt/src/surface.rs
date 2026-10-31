//! The store as the stroke engine's surface: [`StoreSurface`] implements [`stroke::Surface`]
//! over the spatial index and the tile store, at the index's level.
//!
//! # Raycasts
//!
//! [`Surface::raycast`] is picking's raycast ([`SpatialIndex::raycast`]): the index's tree is
//! walked nearest box first ([`Bvh::raycast`](store::Bvh::raycast)), each candidate tile
//! evaluated and its quads intersected as two triangles, split along their `(i, j)`–`(i + 1,
//! j + 1)` diagonal, with the same test, so it hits the same tile at the same ray parameter, bit
//! for bit. It also keeps which triangle was hit and where on it, for the [`Hit`]: the point is
//! interpolated on the triangle, the normal from the evaluated normals at its corners
//! ([`TileStore::normals`]), and the base face's coordinates from the vertices' grid places.
//!
//! # The stroke's region
//!
//! [`Surface::continuous`] and [`Surface::closest_on_region`] both walk the surface near a point
//! `p`, within `reach` of it, tile by tile:
//!
//! - the walk starts at the tile holding `p` (from its face and coordinates);
//! - from each tile it reaches, it steps into every tile that shares one of its border vertices
//!   lying within `reach` of `p` (the copies of the vertex, [`geom::Topology::copies`]), across
//!   faces too.
//!
//! So two tiles are connected when the surface itself joins them within the ball, at a seam
//! vertex inside it: the two faces of a thin slab are close but join only at its rim, which a
//! ball smaller than the slab doesn't reach; two parts with a gap between them share no vertex
//! at all. A tile is one piece: two folds of the surface inside one tile count as connected,
//! an approximation at the scale of a tile (64 quads a side at the finest levels).
//!
//! [`Surface::continuous`] asks whether the walk from `a` reaches the tile holding `b` (`b`
//! within `reach` of `a` first, and true at once for two hits on one tile).
//! [`Surface::closest_on_region`] walks the whole region and takes, of the vertices within
//! `reach` of `near` on the tiles reached, the one closest to the ray (ties to the least tile
//! key, then the first vertex in row-major order): the closest point up to the mesh's spacing,
//! about one quad, which the stroke engine allows ("approximate, but deterministic"). If no vertex
//! lies within `reach` (a reach under a quad), `near` itself is the answer.
//!
//! # Errors
//!
//! The [`Surface`] methods can't fail. A store or index error is kept
//! ([`StoreSurface::error`]), the method answers `None` or `false`, and every later call
//! answers so too without reading the store: the owner checks after each call into the stroke
//! engine and ends the stroke there, taking the error with [`StoreSurface::take_error`].
//!
//! # Cost
//!
//! Every method reads tiles through the store, which evaluates a tile the first time and finds
//! it cached after: the tiles near a stroke are the ones it brushes. The walk looks up each
//! tile's neighbours with at most eight copy lookups (one per corner, one per edge), whatever
//! the tile's size. Measured in release on `cube_sphere(4)` at level 8 (`surface_costs` in the
//! tests, 200 calls each, 2026-10-31), brushes of about 12,000 and 50,000 vertices (radius 62
//! and 125 quads, the reach three radii, a region of 36 and 118 tiles):
//!
//! | method | warm, 12k | warm, 50k | cold, 12k | cold, 50k |
//! |---|---|---|---|---|
//! | `raycast` | 42 µs | 49 µs | 2.7 ms | 2.8 ms |
//! | `continuous` | 6 µs | 19 µs | 2.3 ms | 4.4 ms |
//! | `closest_on_region` | 108 µs | 224 µs | 7.6 ms | 21 ms |
//!
//! Means; the worst warm call was 292 µs. A raycast costs what picking's does (the triangle
//! tests of the tiles it enters); the walk about 1.5 µs a tile; `closest_on_region` scans only
//! the tiles whose boxes come nearer the ray than the best vertex so far. Cold, each evaluates
//! the tiles it reads: for the walk, every tile of the region, nine times a dab's footprint.

use std::collections::{BTreeSet, VecDeque};

use geom::{Channel, LayerId, Region, TileKey, Vec3};
use store::{CacheError, SpatialIndex, TileStore};
use stroke::{DVec3, Hit, Ray, Surface};

use crate::SculptError;

/// The tile store and its spatial index as the stroke engine's [`Surface`], at the index's
/// level: see the module notes for what each method computes, and how errors are kept.
///
/// Borrows the store and the index for its life: make one for each call into the stroke engine
/// ([`stroke::Stroke::begin`], `push`, `end`), and drop it before applying the dabs.
#[derive(Debug)]
pub struct StoreSurface<'a> {
    /// The store the tiles are read through.
    store: &'a mut TileStore,
    /// The index raycasts walk.
    index: &'a SpatialIndex,
    /// The first error a method hit, kept for the owner.
    error: Option<SculptError>,
}

impl<'a> StoreSurface<'a> {
    /// The surface of `store` at `index`'s level.
    #[must_use]
    pub fn new(store: &'a mut TileStore, index: &'a SpatialIndex) -> Self {
        Self {
            store,
            index,
            error: None,
        }
    }

    /// The error a method hit, if one did: since then every method has answered `None` or
    /// `false`.
    #[must_use]
    pub const fn error(&self) -> Option<&SculptError> {
        self.error.as_ref()
    }

    /// Takes the error a method hit, if one did, and answers again from the store.
    pub const fn take_error(&mut self) -> Option<SculptError> {
        self.error.take()
    }

    /// `result`'s value, or `None` with its error kept.
    fn keep<T>(&mut self, result: Result<T, SculptError>) -> Option<T> {
        match result {
            Ok(value) => Some(value),
            Err(e) => {
                self.error = Some(e);
                None
            }
        }
    }

    /// The tile of the index's level holding `hit`, from its face and coordinates.
    fn tile_of(&self, hit: &Hit) -> Option<TileKey> {
        let level = self.index.level();
        let n = level.face_quads();
        let (i, j) = (grid_index(hit.uv[0], n), grid_index(hit.uv[1], n));
        let (key, _, _) =
            TileKey::containing(LayerId::BASE, Channel::DISPLACEMENT, hit.face, level, i, j)?;
        Some(key)
    }

    /// The walk of the module notes from `start` over the tiles joined within `reach` of
    /// `center`: the tiles reached, in the order reached. Stops early, with `target` last, once
    /// it reaches `target`.
    ///
    /// A tile's border vertices other than its corners have their copies in the same tiles
    /// edge by edge (the tile across that edge, on this face or the next: tiles are aligned on
    /// both), so each edge's copies are looked up once, through one of its vertices, if any of
    /// them is within reach, and each corner's if it is.
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate`].
    fn walk(
        &mut self,
        start: TileKey,
        center: DVec3,
        reach: f64,
        target: Option<TileKey>,
    ) -> Result<Vec<TileKey>, CacheError> {
        let reach_squared = reach * reach;
        let mut seen = BTreeSet::from([start]);
        let mut queue = VecDeque::from([start]);
        let mut reached = Vec::new();
        while let Some(key) = queue.pop_front() {
            reached.push(key);
            if Some(key) == target {
                break;
            }
            let side = key.side();
            let positions = self.store.evaluate(key)?;
            let within = |(i, j): (u32, u32)| {
                at(positions, side, i, j)
                    .is_some_and(|p| distance_squared(p, center) <= reach_squared)
            };
            let probes = border_probes(side, within);
            let level = key.level();
            let region = Region::of_tile(&key);
            let topology = self.store.mesh().topology();
            for (i, j) in probes.into_iter().flatten() {
                let copies = topology.copies(key.face(), level, region.x0() + i, region.y0() + j);
                for copy in &copies {
                    let tile = copy.key(level);
                    if seen.insert(tile) {
                        queue.push_back(tile);
                    }
                }
            }
        }
        Ok(reached)
    }

    /// [`Surface::raycast`], its errors returned.
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate`] and [`TileStore::normals`].
    fn try_raycast(&mut self, ray: &Ray) -> Result<Option<Hit>, SculptError> {
        let Some(narrow) = store::Ray::new(ray.origin().to_f32(), ray.direction().to_f32()) else {
            return Ok(None);
        };
        let Some((key, triangle)) = self.raycast_triangle(&narrow)? else {
            return Ok(None);
        };
        self.hit_on_triangle(ray, key, triangle).map(Some)
    }

    /// The tile and the triangle `ray` hits first: [`SpatialIndex::raycast`]'s walk and test,
    /// keeping the triangle.
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate`].
    fn raycast_triangle(
        &mut self,
        ray: &store::Ray,
    ) -> Result<Option<(TileKey, TriangleHit)>, CacheError> {
        let index = self.index;
        let store = &mut *self.store;
        let mut found: Vec<(TileKey, TriangleHit)> = Vec::new();
        let hit = index.bvh().raycast(ray, |key| {
            let side = key.side();
            let hit = intersect_tile(ray, store.evaluate(key)?, side);
            if let Some(hit) = hit {
                found.push((key, hit));
            }
            Ok::<_, CacheError>(hit.map(|hit| hit.t))
        })?;
        Ok(hit.and_then(|hit| found.into_iter().find(|(key, _)| *key == hit.key)))
    }

    /// The [`Hit`] where `ray` crosses `triangle` of tile `key`: the point interpolated on the
    /// triangle, the evaluated normals interpolated and normalised (the triangle's own normal,
    /// facing the ray's origin, where they cancel), and the face's coordinates.
    ///
    /// # Errors
    ///
    /// As [`TileStore::normals`].
    fn hit_on_triangle(
        &mut self,
        ray: &Ray,
        key: TileKey,
        triangle: TriangleHit,
    ) -> Result<Hit, SculptError> {
        let side = key.side();
        let corners = triangle.corners(side);
        let weights = triangle.weights();
        let (positions, normals) = self.tile_values(key, corners)?;
        let position = interpolate(&positions, weights);
        let flat = (positions[1] - positions[0]).cross(positions[2] - positions[0]);
        let facing = if flat.dot(ray.direction()) > 0.0 {
            -flat
        } else {
            flat
        };
        let normal = interpolate(&normals, weights)
            .normalized()
            .or_else(|| facing.normalized())
            .unwrap_or(DVec3::Z);
        let (gx, gy) = triangle.grid(weights);
        Ok(Hit {
            position,
            normal,
            face: key.face(),
            uv: face_uv(key, gx, gy),
        })
    }

    /// The evaluated positions and normals of tile `key`'s vertices `at` (row-major indices),
    /// in `f64`, its normals made current first.
    ///
    /// # Errors
    ///
    /// As [`TileStore::normals`].
    fn tile_values<const N: usize>(
        &mut self,
        key: TileKey,
        at: [usize; N],
    ) -> Result<([DVec3; N], [DVec3; N]), SculptError> {
        self.store.normals(key)?;
        let (Some(positions), Some(normals)) = (
            self.store.cached_positions(&key),
            self.store.cached_normals(&key),
        ) else {
            // Made current just above: only a store that dropped them at once could miss.
            let normals = self.store.normals(key)?.to_vec();
            let positions = self.store.evaluate(key)?.to_vec();
            return Ok((gather(&positions, at), gather(&normals, at)));
        };
        Ok((gather(positions, at), gather(normals, at)))
    }

    /// [`Surface::closest_on_region`], its errors returned.
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate`] and [`TileStore::normals`].
    fn try_closest(&mut self, ray: &Ray, near: &Hit, reach: f64) -> Result<Hit, SculptError> {
        let Some(start) = self.tile_of(near) else {
            return Ok(*near);
        };
        let region = self.walk(start, near.position, reach, None)?;
        // Nearest box first, so the scan can stop at the first tile whose box is farther from
        // the ray than the best vertex so far: the answer doesn't depend on the order, as ties
        // go to the least key and vertex either way.
        let mut tiles: Vec<(f64, TileKey)> = region
            .into_iter()
            .map(|key| (self.box_distance_squared(ray, &key), key))
            .collect();
        tiles.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let reach_squared = reach * reach;
        let mut best: Option<(f64, TileKey, usize)> = None;
        for (bound, key) in tiles {
            if best.is_some_and(|(least, _, _)| bound > least) {
                break;
            }
            let positions = self.store.evaluate(key)?;
            for (k, &p) in positions.iter().enumerate() {
                if distance_squared(p, near.position) > reach_squared {
                    continue;
                }
                let p = DVec3::from_f32(p);
                let off = p - ray.closest_to(p);
                let candidate = (off.dot(off), key, k);
                if best.is_none_or(|least| lexically_less(candidate, least)) {
                    best = Some(candidate);
                }
            }
        }
        let Some((_, key, k)) = best else {
            return Ok(*near);
        };
        let ([position], [normal]) = self.tile_values(key, [k])?;
        let side = key.side();
        let (i, j) = (k % side as usize, k / side as usize);
        Ok(Hit {
            position,
            normal: normal.normalized().unwrap_or(near.normal),
            face: key.face(),
            uv: face_uv(key, to_f64(i), to_f64(j)),
        })
    }

    /// A lower bound of the squared distance from `ray` to any point of tile `key`'s box in the
    /// index: the distance from the box's centre to the ray less its half diagonal, or `0` for
    /// a tile without bounds.
    fn box_distance_squared(&self, ray: &Ray, key: &TileKey) -> f64 {
        let Some(bounds) = self.index.bvh().bounds(key) else {
            return 0.0;
        };
        let (low, high) = (
            DVec3::from_f32(bounds.aabb.min()),
            DVec3::from_f32(bounds.aabb.max()),
        );
        let centre = (low + high) * 0.5;
        let half = (high - low).length() * 0.5;
        let gap = (centre.distance(ray.closest_to(centre)) - half).max(0.0);
        gap * gap
    }

    /// [`Surface::continuous`], its errors returned.
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate`].
    fn try_continuous(&mut self, a: &Hit, b: &Hit, reach: f64) -> Result<bool, SculptError> {
        if a.position.distance(b.position) > reach {
            return Ok(false);
        }
        let (Some(from), Some(to)) = (self.tile_of(a), self.tile_of(b)) else {
            return Ok(false);
        };
        if from == to {
            return Ok(true);
        }
        let walked = self.walk(from, a.position, reach, Some(to))?;
        Ok(walked.last() == Some(&to))
    }
}

impl Surface for StoreSurface<'_> {
    fn raycast(&mut self, ray: &Ray) -> Option<Hit> {
        if self.error.is_some() {
            return None;
        }
        let hit = self.try_raycast(ray);
        self.keep(hit).flatten()
    }

    fn closest_on_region(&mut self, ray: &Ray, near: &Hit, reach: f64) -> Option<Hit> {
        if self.error.is_some() {
            return None;
        }
        let hit = self.try_closest(ray, near, reach);
        self.keep(hit)
    }

    fn continuous(&mut self, a: &Hit, b: &Hit, reach: f64) -> bool {
        if self.error.is_some() {
            return false;
        }
        let joined = self.try_continuous(a, b, reach);
        self.keep(joined).unwrap_or(false)
    }
}

/// Where a ray crosses one triangle of a tile: the ray parameter, the quad, which of its two
/// triangles, and the barycentric coordinates on it.
#[derive(Debug, Clone, Copy, PartialEq)]
struct TriangleHit {
    /// The ray parameter.
    t: f32,
    /// The quad's corner `(i, j)` in the tile, its least.
    i: u32,
    /// See `i`.
    j: u32,
    /// Whether it is the quad's second triangle, `(i, j)`, `(i + 1, j + 1)`, `(i, j + 1)`;
    /// else its first, `(i, j)`, `(i + 1, j)`, `(i + 1, j + 1)`.
    second: bool,
    /// The weight of the triangle's second corner, as the ray test found it (a little outside
    /// `0..=1` at an edge, within the test's slack).
    u: f32,
    /// The weight of its third corner, likewise.
    v: f32,
}

impl TriangleHit {
    /// The row-major indices of the triangle's corners in a tile of `side` vertices a side, in
    /// the test's order: the corner, the second, the third.
    fn corners(&self, side: u32) -> [usize; 3] {
        let at = |i: u32, j: u32| (j * side + i) as usize;
        let (i, j) = (self.i, self.j);
        if self.second {
            [at(i, j), at(i + 1, j + 1), at(i, j + 1)]
        } else {
            [at(i, j), at(i + 1, j), at(i + 1, j + 1)]
        }
    }

    /// The corners' weights, clamped onto the triangle: each at least 0, summing to 1.
    fn weights(&self) -> [f64; 3] {
        let (u, v) = (f64::from(self.u).max(0.0), f64::from(self.v).max(0.0));
        let sum = u + v;
        let (u, v) = if sum > 1.0 {
            (u / sum, v / sum)
        } else {
            (u, v)
        };
        [1.0 - u - v, u, v]
    }

    /// The point with the corners' `weights`, in the tile's grid coordinates.
    fn grid(&self, [_, u, v]: [f64; 3]) -> (f64, f64) {
        let (i, j) = (f64::from(self.i), f64::from(self.j));
        if self.second {
            (i + u, j + u + v)
        } else {
            (i + u + v, j + v)
        }
    }
}

/// The nearest triangle `ray` crosses of a tile with `positions`, row-major `side × side`.
///
/// It is the index's own test (see `store`'s `intersect_tile`), keeping the triangle. The least
/// parameter is the one the index finds, bit for bit; of equal ones, the first in row-major
/// order, the first triangle before the second.
fn intersect_tile(ray: &store::Ray, positions: &[Vec3], side: u32) -> Option<TriangleHit> {
    let width = side as usize;
    let rows = positions.chunks_exact(width);
    let quads = rows
        .clone()
        .zip(rows.skip(1))
        .zip(0_u32..)
        .flat_map(|((row, next), j)| {
            row.windows(2)
                .zip(next.windows(2))
                .zip(0_u32..)
                .map(move |(quad, i)| (quad, i, j))
        });
    let mut best: Option<TriangleHit> = None;
    for (quad, i, j) in quads {
        let (&[low, right], &[up, far]) = quad else {
            continue;
        };
        let triangles = [(false, low, right, far), (true, low, far, up)];
        for (second, corner, p, q) in triangles {
            let Some((t, u, v)) = intersect_triangle(ray, corner, p, q) else {
                continue;
            };
            if best.is_none_or(|best| t < best.t) {
                best = Some(TriangleHit {
                    t,
                    i,
                    j,
                    second,
                    u,
                    v,
                });
            }
        }
    }
    best
}

/// How far outside a triangle's edges, in barycentric terms, a hit still counts: the index's.
const EDGE_SLACK: f32 = 1e-6;

/// Where `ray` crosses the triangle `corner`, `second`, `third` (from either side): its
/// parameter and the weights of the second and third corners; `None` if it misses or runs
/// parallel to it.
///
/// The index's test (Möller and Trumbore's, by Cramer's rule, the barycentric bounds widened by
/// [`EDGE_SLACK`]), operation for operation, so the parameter is the index's bit for bit.
fn intersect_triangle(
    ray: &store::Ray,
    corner: Vec3,
    second: Vec3,
    third: Vec3,
) -> Option<(f32, f32, f32)> {
    let (edge1, edge2) = (second - corner, third - corner);
    let across = ray.direction().cross(edge2);
    let det = edge1.dot(across);
    if det == 0.0 || !det.is_finite() {
        return None;
    }
    let inv = det.recip();
    let from_corner = ray.origin() - corner;
    let u = from_corner.dot(across) * inv;
    let up = from_corner.cross(edge1);
    let v = ray.direction().dot(up) * inv;
    let inside = u >= -EDGE_SLACK && v >= -EDGE_SLACK && u + v <= 1.0 + EDGE_SLACK;
    let t = edge2.dot(up) * inv;
    (inside && t >= 0.0).then_some((t, u, v))
}

/// The border vertices to look up the neighbours of a tile with `side` vertices a side through.
///
/// Their copies name every tile sharing a border vertex `within` takes: each corner `within`
/// takes, and for each edge with a vertex between its corners that `within` takes, that edge's
/// first such vertex.
fn border_probes(side: u32, within: impl Fn((u32, u32)) -> bool) -> [Option<(u32, u32)>; 8] {
    let last = side.saturating_sub(1);
    let edge = |at: &dyn Fn(u32) -> (u32, u32)| (1..last).map(at).find(|&v| within(v));
    let corner = |v: (u32, u32)| within(v).then_some(v);
    [
        corner((0, 0)),
        corner((last, 0)),
        corner((0, last)),
        corner((last, last)),
        edge(&|t| (t, 0)),
        edge(&|t| (t, last)),
        edge(&|t| (0, t)),
        edge(&|t| (last, t)),
    ]
}

/// Vertex `(i, j)`'s position in a tile's row-major `positions`.
fn at(positions: &[Vec3], side: u32, i: u32, j: u32) -> Option<Vec3> {
    positions.get(usize::try_from(j * side + i).ok()?).copied()
}

/// The squared distance from `p` to `center`, in `f64`.
fn distance_squared(p: Vec3, center: DVec3) -> f64 {
    let off = DVec3::from_f32(p) - center;
    off.dot(off)
}

/// The values `at` of `values`, in `f64`; a missing one is zero (never, for a tile's own
/// vertices).
fn gather<const N: usize>(values: &[Vec3], at: [usize; N]) -> [DVec3; N] {
    at.map(|k| values.get(k).copied().map_or(DVec3::ZERO, DVec3::from_f32))
}

/// The sum of `values` with `weights`.
fn interpolate(values: &[DVec3; 3], weights: [f64; 3]) -> DVec3 {
    values
        .iter()
        .zip(weights)
        .fold(DVec3::ZERO, |sum, (&value, w)| sum + value * w)
}

/// The base face's coordinates, each in `0..=1`, of grid point `(gx, gy)` of tile `key`.
fn face_uv(key: TileKey, gx: f64, gy: f64) -> [f64; 2] {
    let region = Region::of_tile(&key);
    let n = f64::from(key.level().face_quads());
    let along = |origin: u32, g: f64| ((f64::from(origin) + g) / n).clamp(0.0, 1.0);
    [along(region.x0(), gx), along(region.y0(), gy)]
}

/// The grid column (or row) of a face of `n` quads a side holding face coordinate `u`: `⌊u n⌋`,
/// clamped to `0..=n`.
fn grid_index(u: f64, n: u32) -> u32 {
    let g = (u * f64::from(n)).floor().clamp(0.0, f64::from(n));
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped to 0..=n just above, which a u32 holds; a NaN casts to 0"
    )]
    let g = g as u32;
    g
}

/// Whether candidate `a` comes before `b`: the nearer to the ray, then the least tile key, then
/// the first vertex.
fn lexically_less(a: (f64, TileKey, usize), b: (f64, TileKey, usize)) -> bool {
    a.0.total_cmp(&b.0)
        .then(a.1.cmp(&b.1))
        .then(a.2.cmp(&b.2))
        .is_lt()
}

/// `n` as an `f64`: a tile's grid coordinate, far below `2^53`.
fn to_f64(n: usize) -> f64 {
    u32::try_from(n).map_or(f64::MAX, f64::from)
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;
