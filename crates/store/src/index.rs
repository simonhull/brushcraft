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
//! - [`SpatialIndex::raycast`]: the first point a ray hits, for picking
//!   ([`SpatialIndex::raycast_triangle`] with the triangle and where on it). The tree hands out
//!   candidate tiles nearest-first, and each is evaluated and intersected exactly, quad by quad
//!   (two triangles each), until no remaining candidate can be nearer than the hit found. Inside a
//!   tile, its blocks of 8 × 8 quads ([`blocks`]) skip the triangles the ray can't hit, with the
//!   same answer, bit for bit, as testing them all; the blocks of the tiles raycast last are kept
//!   until the store's next write.

mod blocks;
mod bvh;
#[cfg(test)]
pub(crate) mod tests;

pub use bvh::{Bvh, Hit, MAX_TILES, Ray, SphereQuery};
use geom::{BaseMesh, Level, MAX_LEVEL, TILE_QUADS, TileKey, Vec3};
use thiserror::Error;

pub(crate) use self::blocks::TileBlocks;
use self::blocks::{BlockCache, SharedBlocks, WideRay, Written};
pub use crate::tile_bounds::{Aabb, NormalCone, TileBounds};
pub use crate::tiles::WaveBounds;
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
    /// The blocks of the tiles raycast last, which let a raycast skip most of a tile's
    /// triangles (see [`blocks`]).
    blocks: SharedBlocks,
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
            for bounds in Self::bounds_of_all(store, batch, &WaveBounds::default())? {
                tiles.push(bounds);
                progress(tiles.len(), total);
            }
        }
        store.record_bounds(&tiles)?;
        Ok(Self {
            level,
            bvh: Bvh::build(tiles)?,
            blocks: SharedBlocks::default(),
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
            computed.extend(Self::bounds_of_all(store, batch, &WaveBounds::default())?);
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
                blocks: SharedBlocks::default(),
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
        let wave = store.evaluate_many_with_bounds(keys)?;
        self.refit_with(store, keys, &wave)
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
        self.refit_with(store, keys, &WaveBounds::default())
    }

    /// [`SpatialIndex::refit`], taking the bounds of `wave` (from
    /// [`TileStore::evaluate_many_with_bounds`]) for the tiles it has, if they are still
    /// current (see [`WaveBounds`]): those tiles aren't read back from the store. The same
    /// bits as `refit`.
    ///
    /// # Errors
    ///
    /// As [`SpatialIndex::refit`].
    pub fn refit_with(
        &mut self,
        store: &mut TileStore,
        keys: &[TileKey],
        wave: &WaveBounds,
    ) -> Result<(), IndexError> {
        self.check_known(keys)?;
        let changed = Self::bounds_of_all(store, keys, wave)?;
        // The store first: the tree's refit changes nothing when it fails, and can't fail here
        // (every key is known), so a failure leaves the index as it was.
        store.record_bounds(&changed)?;
        self.bvh.refit(&changed)
    }

    /// The bounds of each of `keys`, in order, as [`SpatialIndex::bounds_of`] gives them.
    ///
    /// A tile `wave` has current bounds for takes those (see [`WaveBounds`]). Each other tile
    /// is made current as `bounds_of` makes it, one after another (a lookup for a tile already
    /// current), and then their bounds, which are a box and a cone over each tile's cached
    /// positions and normals, are computed on the store's workers ([`TileStore::par_map`]). A
    /// tile that a later tile's evaluation pushed out of the cache meanwhile is done on its
    /// own, with `bounds_of`. Either way the bounds are those of the tile as the store
    /// evaluates it.
    ///
    /// # Errors
    ///
    /// As [`SpatialIndex::bounds_of`], for the first tile that fails.
    fn bounds_of_all(
        store: &mut TileStore,
        keys: &[TileKey],
        wave: &WaveBounds,
    ) -> Result<Vec<(TileKey, TileBounds)>, IndexError> {
        let known: Vec<Option<TileBounds>> =
            keys.iter().map(|key| wave.current(store, key)).collect();
        let rest: Vec<TileKey> = keys
            .iter()
            .zip(&known)
            .filter(|(_, known)| known.is_none())
            .map(|(&key, _)| key)
            .collect();
        for &key in &rest {
            store.evaluate(key)?;
            store.normals(key)?;
        }
        let computed: Vec<Option<Result<TileBounds, IndexError>>> = {
            let cache = &*store;
            let cached: Vec<(TileKey, Option<Cached<'_>>)> = rest
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
        let mut computed = computed.into_iter();
        keys.iter()
            .zip(known)
            .map(|(&key, known)| {
                let bounds = match known {
                    Some(bounds) => bounds,
                    None => match computed.next().flatten() {
                        Some(bounds) => bounds?,
                        None => Self::bounds_of(store, key)?,
                    },
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
    /// if it hits nothing. [`SpatialIndex::raycast_triangle`] without the triangle.
    ///
    /// # Errors
    ///
    /// [`IndexError::Cache`] if a candidate tile can't be evaluated.
    pub fn raycast(&self, store: &mut TileStore, ray: &Ray) -> Result<Option<Hit>, IndexError> {
        let hit = self.raycast_triangle(store, ray)?;
        Ok(hit.map(|hit| Hit {
            key: hit.key,
            t: hit.t,
        }))
    }

    /// [`SpatialIndex::raycast`], with the triangle hit and where on it: the one raycast, for
    /// picking and for whatever needs the surface's point, normal or coordinates there.
    ///
    /// Of a tile's triangles at the same least parameter, the first in row-major order is
    /// taken, a quad's first triangle before its second.
    ///
    /// A tile's triangles are tested through its blocks (see the module notes), built on its first
    /// raycast since the store's last write: about 0.4 of testing every triangle, then about a
    /// thirtieth while they are kept. The answer is the exhaustive test's, bit for bit.
    /// Raycasts through one index take turns on its kept blocks.
    ///
    /// # Errors
    ///
    /// [`IndexError::Cache`] if a candidate tile can't be evaluated.
    pub fn raycast_triangle(
        &self,
        store: &mut TileStore,
        ray: &Ray,
    ) -> Result<Option<TriangleHit>, IndexError> {
        let wide = WideRay::new(ray);
        let mut cache = self.blocks.lock();
        self.raycast_with(store, ray, |store, key| {
            tile_hit(store, &mut cache, ray, &wide, key)
        })
    }

    /// [`SpatialIndex::raycast_triangle`] over the stroke's surface: a previewed tile
    /// ([`TileStore::preview_positions`]) is hit as its preview shows it, every other as the
    /// store evaluates it. What a stroke places its next dabs on, its dabs not landed yet shown.
    ///
    /// A previewed tile's blocks are built from its preview's positions and kept with the preview
    /// (`TileStore::preview_blocks`), not in the index's cache: the cache's blocks are of the
    /// store's bits, and a preview changes without the store's writes, its blocks with it. So the
    /// skipping stays exact on previews too: the exhaustive test's answer, bit for bit. The tree
    /// finds a previewed tile only where its box holds the preview: a brush showing
    /// one grows the box to it ([`SpatialIndex::cover`]).
    ///
    /// # Errors
    ///
    /// As [`SpatialIndex::raycast_triangle`].
    pub fn raycast_surface(
        &self,
        store: &mut TileStore,
        ray: &Ray,
    ) -> Result<Option<TriangleHit>, IndexError> {
        let wide = WideRay::new(ray);
        let mut cache = self.blocks.lock();
        self.raycast_with(store, ray, |store, key| {
            if let Some((blocks, positions)) = store.preview_blocks(&key) {
                return Ok(blocks.intersect(ray, &wide, positions, key));
            }
            tile_hit(store, &mut cache, ray, &wide, key)
        })
    }

    /// [`SpatialIndex::raycast_surface`] testing every triangle of each candidate tile, previews
    /// in place of their tiles: its oracle.
    ///
    /// # Errors
    ///
    /// As [`SpatialIndex::raycast_triangle`].
    #[cfg(any(test, feature = "test-oracle"))]
    pub fn raycast_surface_exhaustive(
        &self,
        store: &mut TileStore,
        ray: &Ray,
    ) -> Result<Option<TriangleHit>, IndexError> {
        self.raycast_with(store, ray, |store, key| {
            Ok(intersect_grid(
                ray,
                store.surface_positions(key)?,
                key.side(),
                key,
            ))
        })
    }

    /// Grows tile `key`'s box in the tree to hold `aabb` too: a preview's box, so that queries
    /// and raycasts find the tile where its preview lies. Only the tree's box: the bounds the
    /// store records (and saves) stay the tile's own, and the next refit of the tile gives it
    /// back its own box.
    ///
    /// # Errors
    ///
    /// [`IndexError::UnknownTile`] for a tile not in the index; nothing is changed then.
    pub fn cover(&mut self, key: &TileKey, aabb: Aabb) -> Result<(), IndexError> {
        let bounds = *self.bvh.bounds(key).ok_or(IndexError::UnknownTile(*key))?;
        let grown = bounds.aabb.union(aabb);
        if grown == bounds.aabb {
            return Ok(());
        }
        self.bvh.refit(&[(
            *key,
            TileBounds {
                aabb: grown,
                ..bounds
            },
        )])
    }

    /// [`SpatialIndex::raycast_triangle`] testing every triangle of each candidate tile: the
    /// oracle the blocks' answers are checked against, here and (with the `test-oracle`
    /// feature) in other crates' tests.
    ///
    /// # Errors
    ///
    /// As [`SpatialIndex::raycast_triangle`].
    #[cfg(any(test, feature = "test-oracle"))]
    pub fn raycast_exhaustive(
        &self,
        store: &mut TileStore,
        ray: &Ray,
    ) -> Result<Option<TriangleHit>, IndexError> {
        self.raycast_with(store, ray, |store, key| {
            Ok(intersect_grid(ray, store.evaluate(key)?, key.side(), key))
        })
    }

    /// The tree's raycast ([`Bvh::raycast`]) with `hit(store, key)` as the test of a candidate
    /// tile: the hit it gives on the tile the tree finds first.
    ///
    /// # Errors
    ///
    /// As `hit`.
    fn raycast_with(
        &self,
        store: &mut TileStore,
        ray: &Ray,
        mut hit: impl FnMut(&mut TileStore, TileKey) -> Result<Option<TriangleHit>, CacheError>,
    ) -> Result<Option<TriangleHit>, IndexError> {
        let mut found: Vec<TriangleHit> = Vec::new();
        let first = self.bvh.raycast(ray, |key| {
            let hit = hit(store, key)?;
            if let Some(hit) = hit {
                found.push(hit);
            }
            Ok::<_, CacheError>(hit.map(|hit| hit.t))
        })?;
        Ok(first.and_then(|first| found.into_iter().find(|tri| tri.key == first.key)))
    }
}

/// The nearest triangle `ray` (`wide` in `f64`) crosses of tile `key`, as [`intersect_grid`]
/// gives it.
///
/// The tile is evaluated through `store`, and intersected through its blocks: from `cache` if
/// they were built from the tile as it evaluates now, else built and kept there.
///
/// # Errors
///
/// As [`TileStore::evaluate`].
fn tile_hit(
    store: &mut TileStore,
    cache: &mut BlockCache,
    ray: &Ray,
    wide: &WideRay,
    key: TileKey,
) -> Result<Option<TriangleHit>, CacheError> {
    let side = key.side();
    store.evaluate(key)?;
    // The state is read with the tile current, so the blocks kept for it are of these bits.
    let written = Written::of(store);
    if let Some(positions) = store.cached_positions(&key)
        && let Some(blocks) = cache.blocks(written, key, positions, side)
    {
        return Ok(blocks.intersect(ray, wide, positions, key));
    }
    // Current a moment ago, so only a store that drops a tile at once lands here.
    let positions = store.evaluate(key)?;
    Ok(TileBlocks::new(positions, side).intersect(ray, wide, positions, key))
}

/// Where a ray first crosses a tile's surface: [`SpatialIndex::raycast_triangle`]'s answer.
///
/// The tile's quad `(i, j)`–`(i + 1, j + 1)` is split along that diagonal into two triangles:
/// the first `(i, j)`, `(i + 1, j)`, `(i + 1, j + 1)`, the second `(i, j)`, `(i + 1, j + 1)`,
/// `(i, j + 1)`. The point is `corner + u (second − corner) + v (third − corner)` on the
/// triangle hit, which lies on the ray at `t` up to rounding; `u` and `v` may stray outside the
/// triangle by the test's slack (about `1e-6`) at an edge.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TriangleHit {
    /// The tile hit.
    pub key: TileKey,
    /// The ray parameter of the hit, as [`Hit::t`].
    pub t: f32,
    /// The quad's least corner's column in the tile.
    pub i: u32,
    /// The quad's least corner's row in the tile.
    pub j: u32,
    /// Whether the quad's second triangle was hit, else its first.
    pub second: bool,
    /// The weight of the triangle's second corner.
    pub u: f32,
    /// The weight of its third corner.
    pub v: f32,
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
#[cfg(test)]
fn intersect_tile(ray: &Ray, positions: &[Vec3], side: u32) -> Option<f32> {
    // Any tile names the hit: only its parameter is kept.
    let key = TileKey::displacement(geom::FaceId::new(0), Level::new(6).ok()?, 0, 0).ok()?;
    intersect_grid(ray, positions, side, key).map(|hit| hit.t)
}

/// The nearest triangle `ray` crosses of a grid of `side` vertices a side with `positions`,
/// row-major, the hit named for tile `key` (see [`TriangleHit`]); `None` if it misses every one.
///
/// Every triangle tested, in row-major order: the exhaustive test, which [`TileBlocks`] answers
/// as, bit for bit, and the oracle its tests check it against.
#[cfg(any(test, feature = "test-oracle"))]
fn intersect_grid(ray: &Ray, positions: &[Vec3], side: u32, key: TileKey) -> Option<TriangleHit> {
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
                    key,
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

/// How far outside a triangle's edges, in barycentric terms, a hit still counts: enough that a
/// ray through an edge two triangles share hits one of them despite rounding.
const EDGE_SLACK: f32 = 1e-6;

/// Where `ray` crosses the triangle `corner`, `second`, `third` (from either side): the ray
/// parameter, and the weights of the second and third corners; `None` if it misses or runs
/// parallel to it.
///
/// Möller and Trumbore's test: solve `origin + t·direction = corner + u (second − corner) +
/// v (third − corner)` by Cramer's rule, and accept `u ≥ 0`, `v ≥ 0`, `u + v ≤ 1`, `t ≥ 0`
/// (the barycentric bounds widened by [`EDGE_SLACK`]).
fn intersect_triangle(
    ray: &Ray,
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
