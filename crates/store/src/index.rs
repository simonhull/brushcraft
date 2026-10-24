//! The spatial index: where every finest-level tile is, for brushes and picking.
//!
//! [`SpatialIndex`] keeps, for every tile of one level (the finest the store evaluates), the
//! tile's [`TileBounds`]: the box of its evaluated positions and a cone bounding its normals.
//! The bounds sit in a [`Bvh`], built with the binned surface area heuristic and refitted as
//! edits move tiles. It is small (under 100 bytes a tile, see [`Bvh`]) and always resident; the
//! tiles themselves are evaluated through the [`TileStore`] only when a query needs them.
//!
//! The store keeps every tile's bounds in its files, written with the tiles they describe (see
//! [`crate::bounds`]): [`SpatialIndex::build`], [`SpatialIndex::refit`] and
//! [`SpatialIndex::update`] record what they compute, and [`SpatialIndex::load`] builds the
//! index from what is stored, evaluating only the tiles whose bounds aren't known.
//!
//! Two queries:
//!
//! - [`SpatialIndex::query_sphere`]: the tiles a brush's sphere touches, from the bounds alone;
//! - [`SpatialIndex::raycast`]: the first point a ray hits, for picking. The tree hands out
//!   candidate tiles nearest-first, and each is evaluated and intersected exactly, quad by quad
//!   (two triangles each), until no remaining candidate can be nearer than the hit found.

mod bvh;
#[cfg(test)]
pub(crate) mod tests;

pub use bvh::{Aabb, Bvh, Hit, MAX_TILES, Ray, SphereQuery};
use geom::{BaseMesh, Level, MAX_LEVEL, TILE_QUADS, TileKey, Vec3};
use thiserror::Error;

use crate::tiles::{CacheError, Evaluation, TileStore};

/// Tiles [`SpatialIndex::build`] evaluates together before it reads their bounds.
///
/// Enough to keep the store's workers busy through [`TileStore::evaluate_many`]; few enough
/// (about 100 MB of evaluated slots) that a budget of a few hundred megabytes keeps them cached
/// until they are read. Under a smaller one, tiles evicted before they are read are evaluated
/// again, one at a time.
const BUILD_BATCH: usize = 1024;

/// Why building or updating the spatial index failed.
#[derive(Debug, Error)]
#[non_exhaustive]
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
    /// A build was given tiles of more than one level, layer or channel: this one differs from
    /// the first.
    #[error("{0:?} is not of the same level, layer and channel as the other tiles")]
    MixedTiles(TileKey),
    /// A build was given more than [`MAX_TILES`] tiles, or tiles of a face so high that the
    /// tiles of every face up to it are more: this many.
    #[error("{0} tiles are more than an index holds")]
    TooManyTiles(usize),
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
    /// How far from 1 a stored axis's squared length may be: a normalised vector's is within a
    /// few units in the last place, and anything further isn't an axis this code made.
    const UNIT_SLACK: f32 = 1e-4;

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

    /// The cone with `axis` and `cos_half_angle` as a stored bound holds them, bit for bit;
    /// `None` unless both are finite, the cosine is in `[-1, 1]` and the axis is a unit vector
    /// (to within [`NormalCone::UNIT_SLACK`], as normalising leaves it).
    pub(crate) fn from_parts(axis: Vec3, cos_half_angle: f32) -> Option<Self> {
        let unit = (axis.dot(axis) - 1.0).abs() <= Self::UNIT_SLACK;
        let valid = axis.is_finite() && unit && (-1.0..=1.0).contains(&cos_half_angle);
        valid.then_some(Self {
            axis,
            cos_half_angle,
        })
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
    /// How many `u32` [`TileBounds::to_bits`] gives.
    pub(crate) const BITS: usize = 10;

    /// The bounds as bits, in the order the project file stores them: the box's low corner and
    /// high corner, the cone's axis (each `x`, `y`, `z`), then the cone's cosine.
    pub(crate) fn to_bits(self) -> [u32; Self::BITS] {
        let [min, max, axis] =
            [self.aabb.min(), self.aabb.max(), self.cone.axis].map(Vec3::to_bits);
        let cos = self.cone.cos_half_angle.to_bits();
        let mut bits = [0; Self::BITS];
        let all = min.into_iter().chain(max).chain(axis).chain([cos]);
        for (slot, value) in bits.iter_mut().zip(all) {
            *slot = value;
        }
        bits
    }

    /// The bounds [`TileBounds::to_bits`] gave these bits, bit for bit; `None` unless they make
    /// a valid box ([`Aabb::new`]) and cone ([`NormalCone::from_parts`]).
    pub(crate) fn from_bits(bits: [u32; Self::BITS]) -> Option<Self> {
        let [
            min_x,
            min_y,
            min_z,
            max_x,
            max_y,
            max_z,
            axis_x,
            axis_y,
            axis_z,
            cos,
        ] = bits.map(f32::from_bits);
        Some(Self {
            aabb: Aabb::new(
                Vec3::new(min_x, min_y, min_z),
                Vec3::new(max_x, max_y, max_z),
            )?,
            cone: NormalCone::from_parts(Vec3::new(axis_x, axis_y, axis_z), cos)?,
        })
    }

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

/// A tile's cached positions and normals, which its bounds are computed from.
type Cached<'a> = (&'a [Vec3], &'a [Vec3]);

/// What [`SpatialIndex::load`] found stored, and what it had to evaluate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LoadStats {
    /// Tiles whose bounds were read from the store.
    pub stored: usize,
    /// Tiles evaluated because their bounds weren't stored, or a write since made them
    /// unknown.
    pub evaluated: usize,
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
    /// For the benchmark's sphere at level 9 that is 153,600 tiles. They are evaluated a
    /// thousand or so at a time with [`TileStore::evaluate_many`], in parallel, then read back
    /// from the store's cache; each parent tile is built once while its children are
    /// evaluated.
    ///
    /// Every bound is recorded in the store, to be written with the next flush or save: a
    /// project whose files hold them opens with [`SpatialIndex::load`] instead, which is what
    /// to call for one that may.
    ///
    /// # Errors
    ///
    /// [`IndexError::Cache`] if a tile can't be evaluated (level 0 isn't tiled, for one);
    /// [`IndexError::NoBounds`] if a tile has no finite position. Nothing is recorded then.
    pub fn build(
        store: &mut TileStore,
        level: Level,
        mut progress: impl FnMut(usize, usize),
    ) -> Result<Self, IndexError> {
        let keys: Vec<TileKey> = tiles_of(store.mesh(), level).collect();
        let total = keys.len();
        let mut tiles = Vec::with_capacity(total);
        for batch in keys.chunks(BUILD_BATCH) {
            store.evaluate_many(batch, Evaluation::Normals)?;
            for bounds in Self::bounds_of_all(store, batch)? {
                tiles.push(bounds);
                progress(tiles.len(), total);
            }
        }
        store.record_bounds(&tiles)?;
        Ok(Self {
            level,
            bvh: Bvh::build(tiles)?,
        })
    }

    /// The index over every tile of `level`, from the bounds the store keeps in its files with
    /// the tiles: the same index, bit for bit, as [`SpatialIndex::build`] makes of
    /// the store's current state, without evaluating a tile whose bounds are stored. A tile
    /// whose bounds aren't (a project saved before they were kept, an index never built, a
    /// write that made them unknown) is evaluated as `build` does, and its bounds recorded.
    /// `progress(done, total)` is called as tiles are read or evaluated.
    ///
    /// # Errors
    ///
    /// [`IndexError::Cache`] if the stored bounds can't be read or a tile can't be evaluated;
    /// [`IndexError::NoBounds`] as for `build`.
    pub fn load(
        store: &mut TileStore,
        level: Level,
        mut progress: impl FnMut(usize, usize),
    ) -> Result<(Self, LoadStats), IndexError> {
        let keys: Vec<TileKey> = tiles_of(store.mesh(), level).collect();
        let total = keys.len();
        let mut found = store.stored_bounds(&keys)?;
        let missing: Vec<TileKey> = keys
            .iter()
            .zip(&found)
            .filter(|(_, bounds)| bounds.is_none())
            .map(|(key, _)| *key)
            .collect();
        let stats = LoadStats {
            stored: total - missing.len(),
            evaluated: missing.len(),
        };
        progress(stats.stored, total);
        let mut computed = Vec::with_capacity(missing.len());
        for batch in missing.chunks(BUILD_BATCH) {
            store.evaluate_many(batch, Evaluation::Normals)?;
            computed.extend(Self::bounds_of_all(store, batch)?);
            progress(stats.stored + computed.len(), total);
        }
        store.record_bounds(&computed)?;
        let mut computed = computed.into_iter();
        for slot in found.iter_mut().filter(|bounds| bounds.is_none()) {
            *slot = computed.next().map(|(_, bounds)| bounds);
        }
        let tiles = keys
            .into_iter()
            .zip(found)
            .map(|(key, bounds)| {
                bounds
                    .map(|bounds| (key, bounds))
                    .ok_or(IndexError::NoBounds(key))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((
            Self {
                level,
                bvh: Bvh::build(tiles)?,
            },
            stats,
        ))
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

    /// Re-evaluates the tiles `keys` (after edits), together with
    /// [`TileStore::evaluate_many`], and refits the tree to their new bounds (see
    /// [`SpatialIndex::refit`]).
    ///
    /// # Errors
    ///
    /// As [`SpatialIndex::refit`]. The index is unchanged on error.
    pub fn update(&mut self, store: &mut TileStore, keys: &[TileKey]) -> Result<(), IndexError> {
        self.check_known(keys)?;
        store.evaluate_many(keys, Evaluation::Normals)?;
        self.refit(store, keys)
    }

    /// Refits the tree to the bounds of the tiles `keys` as the store evaluates them now, and
    /// records them in the store, to be written with the next flush or save.
    ///
    /// For a caller that has made them current itself, with [`TileStore::evaluate_many`]: their
    /// positions and normals are then read from the store's cache. A tile that isn't current is
    /// evaluated on its own, one at a time; [`SpatialIndex::update`] makes them all current
    /// together first.
    ///
    /// # Errors
    ///
    /// As [`SpatialIndex::bounds_of`]; [`IndexError::UnknownTile`] for a tile not in the index.
    /// The index is unchanged on error.
    pub fn refit(&mut self, store: &mut TileStore, keys: &[TileKey]) -> Result<(), IndexError> {
        self.check_known(keys)?;
        let changed = Self::bounds_of_all(store, keys)?;
        // The store first: the tree's refit changes nothing when it fails, and can't fail here
        // (every key is known), so a failure leaves the index as it was.
        store.record_bounds(&changed)?;
        self.bvh.refit(&changed)
    }

    /// The bounds of each of `keys`, in order, as [`SpatialIndex::bounds_of`] gives them.
    ///
    /// Each tile is made current as `bounds_of` makes it, one after another (a lookup for a
    /// tile already current), and then the bounds, which are a box and a cone over each tile's
    /// cached positions and normals, are computed on the store's workers
    /// ([`TileStore::par_map`]). A tile that a later tile's evaluation pushed out of the cache
    /// meanwhile is done on its own, with `bounds_of`. Either way the bounds are those of the
    /// tile as the store evaluates it.
    ///
    /// # Errors
    ///
    /// As [`SpatialIndex::bounds_of`], for the first tile that fails.
    fn bounds_of_all(
        store: &mut TileStore,
        keys: &[TileKey],
    ) -> Result<Vec<(TileKey, TileBounds)>, IndexError> {
        for &key in keys {
            store.evaluate(key)?;
            store.normals(key)?;
        }
        let computed: Vec<Option<Result<TileBounds, IndexError>>> = {
            let cache = &*store;
            let cached: Vec<(TileKey, Option<Cached<'_>>)> = keys
                .iter()
                .map(|key| {
                    let views = cache.cached_positions(key).zip(cache.cached_normals(key));
                    (*key, views)
                })
                .collect();
            cache.par_map(&cached, |&(key, views)| {
                views.map(|(positions, normals)| {
                    TileBounds::of_tile(positions, normals).ok_or(IndexError::NoBounds(key))
                })
            })
        };
        keys.iter()
            .zip(computed)
            .map(|(&key, bounds)| {
                let bounds = match bounds {
                    Some(bounds) => bounds?,
                    None => Self::bounds_of(store, key)?,
                };
                Ok((key, bounds))
            })
            .collect()
    }

    /// Checks that every tile of `keys` is in the index.
    ///
    /// # Errors
    ///
    /// [`IndexError::UnknownTile`] for the first that isn't.
    fn check_known(&self, keys: &[TileKey]) -> Result<(), IndexError> {
        match keys.iter().find(|k| self.bvh.bounds(k).is_none()) {
            Some(&key) => Err(IndexError::UnknownTile(key)),
            None => Ok(()),
        }
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

/// The most tiles per side of a face, at the deepest level: 64, which a tile coordinate (a
/// `u16`) holds, as the assertion below checks at compile time.
const MAX_TILES_PER_SIDE: u32 = (1 << MAX_LEVEL) / TILE_QUADS;
const _: () = assert!(
    MAX_TILES_PER_SIDE <= 0xFFFF,
    "a face's tile coordinates fit a u16 at every level"
);

/// Every base-layer displacement tile of `level` on `mesh`, face by face.
fn tiles_of(mesh: &BaseMesh, level: Level) -> impl Iterator<Item = TileKey> {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "at most MAX_TILES_PER_SIDE tiles per side, which fits a u16 (asserted above)"
    )]
    let per_side = level.tiles_per_side() as u16;
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
