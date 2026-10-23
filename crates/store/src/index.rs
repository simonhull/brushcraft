//! The spatial index: where every finest-level tile is, for brushes and picking.
//!
//! [`SpatialIndex`] keeps, for every tile of one level (the finest the store evaluates), the
//! tile's [`TileBounds`]: the box of its evaluated positions and a cone bounding its normals.
//! The bounds sit in a [`Bvh`], built with the binned surface area heuristic and refitted as
//! edits move tiles. It is small (a few tens of bytes per tile) and always resident; the tiles
//! themselves are evaluated through the [`TileStore`] only when a query needs them.
//!
//! Two queries:
//!
//! - [`SpatialIndex::query_sphere`]: the tiles a brush's sphere touches, from the bounds alone;
//! - [`SpatialIndex::raycast`]: the first point a ray hits, for picking. The tree hands out
//!   candidate tiles nearest-first, and each is evaluated and intersected exactly, quad by quad
//!   (two triangles each), until no remaining candidate can be nearer than the hit found.

mod bvh;
#[cfg(test)]
mod tests;

pub use bvh::{Aabb, Bvh, Hit, Ray, SphereQuery};
use geom::{BaseMesh, Level, TileKey, Vec3};
use thiserror::Error;

use crate::tiles::{CacheError, TileStore};

/// Why building or updating the spatial index failed.
#[derive(Debug, Error)]
pub enum IndexError {
    /// Evaluating a tile failed.
    #[error("evaluating a tile failed")]
    Cache(#[from] CacheError),
    /// A tile evaluated to no finite position, so it has no bounds.
    #[error("{0:?} has no finite position to bound")]
    NoBounds(TileKey),
    /// A tile was listed twice for a build.
    #[error("{0:?} is listed twice")]
    DuplicateTile(TileKey),
    /// A refit or an update named a tile the index doesn't hold.
    #[error("{0:?} is not in the index")]
    UnknownTile(TileKey),
}

/// A cone bounding a set of unit normals: every normal is within its half-angle of its axis.
///
/// The cone of a tile tells, without evaluating the tile, whether all of it faces away from a
/// viewer or a brush. When the normals spread too far for a cone to say anything (or aren't
/// all valid), the cone is [`NormalCone::EVERYTHING`], which holds every direction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalCone {
    /// The cone's axis, a unit vector.
    axis: Vec3,
    /// The cosine of the cone's half-angle, in `[-1, 1]`: `-1` holds every direction.
    cos_half_angle: f32,
}

impl NormalCone {
    /// The cone that holds every direction: what a tile gets when its normals can't be bounded
    /// any tighter.
    pub const EVERYTHING: Self = Self {
        axis: Vec3::new(0.0, 0.0, 1.0),
        cos_half_angle: -1.0,
    };
    /// How far below the least `cos` of the normals the bound is set, so rounding in the dot
    /// products never leaves one of them just outside the cone.
    const SLACK: f32 = 1e-5;

    /// A cone holding every one of `normals`: its axis is their normalised mean, its half-angle
    /// that of the normal furthest from it.
    ///
    /// That isn't the narrowest bounding cone, but it is cheap and close to it for the gently
    /// curved normals of one tile. [`NormalCone::EVERYTHING`] if `normals` is empty, holds a
    /// normal that isn't finite, or averages to nothing (normals spread all around).
    #[must_use]
    pub fn from_normals(normals: &[Vec3]) -> Self {
        if !normals.iter().all(|n| n.is_finite()) {
            return Self::EVERYTHING;
        }
        let Some(axis) = normals
            .iter()
            .fold(Vec3::ZERO, |sum, &n| sum + n)
            .normalized()
        else {
            return Self::EVERYTHING;
        };
        let least = normals
            .iter()
            .filter_map(|n| n.normalized())
            .map(|n| axis.dot(n))
            .fold(1.0_f32, f32::min);
        Self {
            axis,
            cos_half_angle: (least - Self::SLACK).max(-1.0),
        }
    }

    /// The cone's axis, a unit vector.
    #[must_use]
    pub const fn axis(&self) -> Vec3 {
        self.axis
    }

    /// The cosine of the cone's half-angle: `-1` for a cone that holds every direction.
    #[must_use]
    pub const fn cos_half_angle(&self) -> f32 {
        self.cos_half_angle
    }

    /// Whether the direction of `n` lies in the cone. A zero or non-finite `n` has no
    /// direction, and lies in no cone.
    #[must_use]
    pub fn contains(&self, n: Vec3) -> bool {
        n.normalized()
            .is_some_and(|n| self.axis.dot(n) >= self.cos_half_angle)
    }
}

/// What the index knows of one tile without evaluating it: where it is and which way it faces.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TileBounds {
    /// The box of the tile's evaluated positions.
    pub aabb: Aabb,
    /// A cone bounding the tile's normals.
    pub cone: NormalCone,
}

impl TileBounds {
    /// The bounds of a tile with these evaluated `positions` and `normals`; `None` if no
    /// position is finite.
    #[must_use]
    pub fn of_tile(positions: &[Vec3], normals: &[Vec3]) -> Option<Self> {
        Some(Self {
            aabb: Aabb::from_points(positions.iter().copied())?,
            cone: NormalCone::from_normals(normals),
        })
    }
}

/// The spatial index over every tile of one level (see the module notes).
#[derive(Debug, Clone)]
pub struct SpatialIndex {
    /// The level whose tiles the index holds.
    level: Level,
    /// The tiles' bounds.
    bvh: Bvh,
}

impl SpatialIndex {
    /// The index over every tile of `level` (the finest level the store evaluates): each tile
    /// is evaluated once, positions and normals, and its bounds recorded. `progress(done,
    /// total)` is called after each tile.
    ///
    /// For the benchmark's sphere at level 9 that is 153,600 tiles; the evaluations go through
    /// the store's cache, so each parent tile is built once while its children are evaluated.
    ///
    /// # Errors
    ///
    /// [`IndexError::Cache`] if a tile can't be evaluated (level 0 isn't tiled, for one);
    /// [`IndexError::NoBounds`] if a tile has no finite position.
    pub fn build(
        store: &mut TileStore,
        level: Level,
        mut progress: impl FnMut(usize, usize),
    ) -> Result<Self, IndexError> {
        let keys: Vec<TileKey> = tiles_of(store.mesh(), level).collect();
        let total = keys.len();
        let mut tiles = Vec::with_capacity(total);
        for (done, key) in (1..).zip(keys) {
            tiles.push((key, Self::bounds_of(store, key)?));
            progress(done, total);
        }
        Ok(Self {
            level,
            bvh: Bvh::build(tiles)?,
        })
    }

    /// `key`'s bounds as the store evaluates it now.
    ///
    /// # Errors
    ///
    /// [`IndexError::Cache`] if the tile can't be evaluated; [`IndexError::NoBounds`] if it has
    /// no finite position.
    pub fn bounds_of(store: &mut TileStore, key: TileKey) -> Result<TileBounds, IndexError> {
        let aabb = Aabb::from_points(store.evaluate(key)?.iter().copied())
            .ok_or(IndexError::NoBounds(key))?;
        let cone = NormalCone::from_normals(store.normals(key)?);
        Ok(TileBounds { aabb, cone })
    }

    /// Re-evaluates the tiles `keys` (after edits) and refits the tree to their new bounds.
    ///
    /// # Errors
    ///
    /// As [`SpatialIndex::bounds_of`]; [`IndexError::UnknownTile`] for a tile not in the index.
    /// The index is unchanged on error.
    pub fn update(&mut self, store: &mut TileStore, keys: &[TileKey]) -> Result<(), IndexError> {
        if let Some(&key) = keys.iter().find(|k| self.bvh.bounds(k).is_none()) {
            return Err(IndexError::UnknownTile(key));
        }
        let changed = keys
            .iter()
            .map(|&key| Ok((key, Self::bounds_of(store, key)?)))
            .collect::<Result<Vec<_>, IndexError>>()?;
        self.bvh.refit(&changed)
    }

    /// The level whose tiles the index holds.
    #[must_use]
    pub const fn level(&self) -> Level {
        self.level
    }

    /// The tree of the tiles' bounds.
    #[must_use]
    pub const fn bvh(&self) -> &Bvh {
        &self.bvh
    }

    /// Every tile whose box touches the sphere of `radius` around `center`: the tiles a brush
    /// there may change. See [`Bvh::query_sphere`].
    #[must_use]
    pub fn query_sphere(&self, center: Vec3, radius: f32) -> SphereQuery<'_> {
        self.bvh.query_sphere(center, radius)
    }

    /// Where `ray` first hits the surface at the index's level, evaluating candidate tiles
    /// through `store` nearest-first and intersecting each one's quads as two triangles; `None`
    /// if it hits nothing.
    ///
    /// # Errors
    ///
    /// [`IndexError::Cache`] if a candidate tile can't be evaluated.
    pub fn raycast(&self, store: &mut TileStore, ray: &Ray) -> Result<Option<Hit>, IndexError> {
        Ok(self.bvh.raycast(ray, |key| {
            Ok::<_, CacheError>(intersect_tile(ray, store.evaluate(key)?, key.side()))
        })?)
    }
}

/// Every base-layer displacement tile of `level` on `mesh`, face by face.
fn tiles_of(mesh: &BaseMesh, level: Level) -> impl Iterator<Item = TileKey> {
    // At most 64 tiles per side, even at the deepest level.
    let per_side = u16::try_from(level.tiles_per_side()).unwrap_or(u16::MAX);
    mesh.faces().flat_map(move |face| {
        (0..per_side).flat_map(move |ty| {
            (0..per_side).filter_map(move |tx| TileKey::displacement(face, level, tx, ty).ok())
        })
    })
}

/// The ray parameter of `ray`'s nearest hit on a tile with `positions`, row-major `side × side`,
/// each quad split into two triangles along its `(i, j)`–`(i + 1, j + 1)` diagonal; `None` if
/// it misses every one.
fn intersect_tile(ray: &Ray, positions: &[Vec3], side: u32) -> Option<f32> {
    let side = side as usize;
    let rows = positions.chunks_exact(side);
    rows.clone()
        .zip(rows.skip(1))
        .flat_map(|(row, next)| row.windows(2).zip(next.windows(2)))
        .filter_map(|quad| match quad {
            (&[a, b], &[c, d]) => {
                let first = intersect_triangle(ray, a, b, d);
                let second = intersect_triangle(ray, a, d, c);
                [first, second].into_iter().flatten().reduce(f32::min)
            }
            _ => None,
        })
        .reduce(f32::min)
}

/// How far outside a triangle's edges, in barycentric terms, a hit still counts: enough that a
/// ray through an edge two triangles share hits one of them despite rounding.
const EDGE_SLACK: f32 = 1e-6;

/// The ray parameter where `ray` crosses the triangle `corner`, `second`, `third` (from either
/// side), or `None` if it misses or runs parallel to it.
///
/// Möller and Trumbore's test: solve `origin + t·direction = corner + u (second − corner) +
/// v (third − corner)` by Cramer's rule, and accept `u ≥ 0`, `v ≥ 0`, `u + v ≤ 1`, `t ≥ 0`
/// (the barycentric bounds widened by [`EDGE_SLACK`]).
fn intersect_triangle(ray: &Ray, corner: Vec3, second: Vec3, third: Vec3) -> Option<f32> {
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
    (inside && t >= 0.0).then_some(t)
}
