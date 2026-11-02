//! The store as the stroke engine's surface: [`StoreSurface`] implements [`stroke::Surface`]
//! over the spatial index and the tile store, at the index's level.
//!
//! # The stroke's own surface
//!
//! Every read goes to the **stroke's surface**: the store's tiles with the previews of the dabs
//! placed and not landed yet in place of theirs ([`TileStore::surface_positions`],
//! [`TileStore::surface_normals`], [`SpatialIndex::raycast_surface`], and the index's boxes, which
//! a preview grows to hold it; see [`crate::Preview`]). A preview moves the store's tiles as the
//! finest work will, so a stroke places each dab on the surface its dabs before it leave, up to
//! the stored encoding, however far the finest work lags (the brush-engine spec, "The surface
//! placement reads"). With no preview, it is the store's surface.
//!
//! # Raycasts
//!
//! [`Surface::raycast`] is picking's raycast over the stroke's surface,
//! [`SpatialIndex::raycast_surface`]: the index's tree walked nearest box first, each candidate
//! tile evaluated (or its preview read) and its quads intersected as two triangles. With no
//! preview it hits the same tile at the same ray parameter as picking, and keeps which triangle and
//! where on it, for the [`Hit`]: the point is interpolated on the triangle (so it agrees with
//! picking's `origin + direction · t` to rounding, about `1e-5` of the scene's size, not bit for
//! bit), the normal is interpolated from the surface's normals at its corners
//! ([`TileStore::surface_normals`]), and the base face's coordinates from the vertices' grid
//! places.
//!
//! # The stroke's region
//!
//! [`Surface::continuous`] and [`Surface::closest_on_region`] both walk the surface near a point
//! `p`, within `reach` of it, vertex by vertex:
//!
//! - the walk starts at the vertex nearest `p` of the quad around it;
//! - inside a tile, it spreads along the grid's edges (rows, columns and the diagonal each quad is
//!   split along) through vertices within `reach` of `p`: the tile's vertices within reach fall
//!   into pieces, and the walk enters only the pieces it reaches;
//! - it crosses into a neighbouring tile at a border vertex of an entered piece, through the
//!   vertex's copies ([`geom::Topology::copies`]), across faces too.
//!
//! So two points are connected when the surface itself joins them within the ball: the two
//! faces of a thin slab, or two sides of a fold, are close but join only at the rim, which a
//! ball smaller than the slab doesn't reach, whether or not they share a tile; two parts with a
//! gap between them share no vertex at all. The resolution is the mesh's: a fold thinner than a
//! quad isn't seen.
//!
//! [`Surface::continuous`] asks whether the walk from `a` enters the piece holding `b`'s nearest
//! quad corner (`b` within `reach` of `a` first). [`Surface::closest_on_region`] walks the whole
//! region and takes, of its vertices, the one closest to the ray (ties to the least tile key,
//! then the first vertex in row-major order): the closest point up to the mesh's spacing, about
//! one quad, which the stroke engine allows ("approximate, but deterministic"). If `near`'s
//! nearest corner lies out of reach (a reach under a quad), `near` itself is the answer.
//!
//! # Area normals
//!
//! [`Surface::area_normal`] sums the stroke's surface's normals ([`TileStore::surface_normals`],
//! a preview's own where one is shown) at every vertex inside the dab's sphere, each weighted by
//! the draw action's falloff at its distance, made as hard as the dab
//! ([`brush::draw_imprint_vertex`] at unit strength: the weight the action gives the vertex). The
//! tiles are those whose boxes in the index touch the sphere; each vertex counts once, in the tile
//! that owns it ([`geom::Topology::owns`]). The normals are made together first
//! ([`TileStore::prepare_surface_normals`]: a preview's change with every dab shown, so most are
//! made afresh, on the store's workers); then each tile's part is summed row by row, a tile each
//! on the workers, and the parts are added in key order, all in `f64`. So the same surface and dab
//! give the same bits at any worker count. Placement normalises the sum, and takes the hit's
//! normal where it is zero or not finite.
//!
//! # Errors
//!
//! The [`Surface`] methods can't fail. A store or index error is kept, the method answers `None`
//! or `false`, and every later call answers so too without reading the store. End each call into
//! the stroke engine with [`StoreSurface::finish`], which gives the error back: the dabs that
//! call gave are then to be discarded and the stroke ended.
//!
//! # Cost
//!
//! Every method reads tiles through the store, which evaluates a tile the first time and finds
//! it cached after: the tiles near a stroke are the ones it brushes. A walk first evaluates,
//! together on the store's workers, the tiles in its ball not cached yet. Inside the walk, a
//! tile whose box lies wholly within reach is one piece at no cost; another has its vertices'
//! distances taken and grouped into pieces row by row; a tile's neighbours are looked up once
//! per corner and per run of a piece along an edge.
//!
//! Measured in release on `cube_sphere(4)` at level 8 (`surface_costs` in the tests, 200 calls
//! each, 2026-10-31), brushes of about 12,000 and 50,000 vertices (radius 62 and 125 quads, the
//! reach three radii, a region of 36 and 118 tiles, of which about 20 and 36 cut by the ball's
//! edge and labelled):
//!
//! | method | warm, 12k | warm, 50k | cold, 12k | cold, 50k |
//! |---|---|---|---|---|
//! | `raycast` | 42 µs | 45 µs | 2.3 ms | 2.4 ms |
//! | `continuous` | 10 µs | 21 µs | 1.9 ms | 2.4 ms |
//! | `closest_on_region` | 287 µs | 590 µs | 3.5 ms | 6.8 ms |
//!
//! Means; the worst warm call was 897 µs (`closest_on_region`, 50k). Most of a warm walk is
//! labelling the tiles the ball's edge cuts, a few microseconds each; cold, it is evaluating the
//! region's tiles, nine times a dab's footprint, which the prefetch spreads over the workers.

use std::collections::BTreeSet;

use brush::{Imprint, Smoothstep, draw_imprint_vertex};
use geom::{Channel, LayerId, Region, TileKey, Topology, Vec3};
use store::{CacheError, Evaluation, SpatialIndex, TileStore, TriangleHit};
use stroke::{DVec3, Hit, Ray, Surface};

use self::region::{Ball, Source, Vertex, Walked, distance_squared, walk};
use crate::SculptError;
use crate::apply::vertices;

mod region;

/// The tile store and its spatial index as the stroke engine's [`Surface`], at the index's
/// level: see the module notes for what each method computes, and how errors are kept.
///
/// Borrows the store and the index for its life: make one for each call into the stroke engine
/// ([`stroke::Stroke::begin`], `push`, `end`), and end it with [`StoreSurface::finish`] before
/// applying the dabs.
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

    /// Ends the surface's use, giving back the error a method hit, if one did: the way to end
    /// each call into the stroke engine, so a kept error can't be dropped unread.
    ///
    /// # Errors
    ///
    /// The error a method kept. The dabs the stroke engine gave from a call that ended in an
    /// error were placed with the surface answering `None` or `false` after it: discard them,
    /// and end the stroke there.
    pub fn finish(self) -> Result<(), SculptError> {
        self.error.map_or(Ok(()), Err)
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
    ///
    /// The grid point is `⌊uv · n⌋`. A hit on a border between tiles (or rounded to just
    /// across it) lies in both, and either holds the corners of the quad around it that
    /// [`StoreSurface::anchor`] starts from, the border vertices being in both; so which one
    /// this picks doesn't change the walk.
    fn tile_of(&self, hit: &Hit) -> Option<TileKey> {
        let level = self.index.level();
        let n = level.face_quads();
        let (i, j) = (grid_index(hit.uv[0], n), grid_index(hit.uv[1], n));
        let (key, _, _) =
            TileKey::containing(LayerId::BASE, Channel::DISPLACEMENT, hit.face, level, i, j)?;
        Some(key)
    }

    /// The vertex the walk starts from (or looks for) for `hit`: of the corners of the quad
    /// around it in its tile, the nearest to it (the first of equals, row-major); `None` if
    /// that one isn't within `reach_squared` of `center`, or the hit isn't on a tile.
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate`].
    fn anchor(
        &mut self,
        hit: &Hit,
        center: DVec3,
        reach_squared: f64,
    ) -> Result<Option<Vertex>, CacheError> {
        let Some(key) = self.tile_of(hit) else {
            return Ok(None);
        };
        let side = key.side();
        let region = Region::of_tile(&key);
        let n = f64::from(key.level().face_quads());
        let last = f64::from(side - 1);
        let local = |u: f64, origin: u32| (u * n - f64::from(origin)).clamp(0.0, last);
        let (x, y) = (local(hit.uv[0], region.x0()), local(hit.uv[1], region.y0()));
        let (x0, y0) = (floor_to(x, side - 1), floor_to(y, side - 1));
        let (x1, y1) = ((x0 + 1).min(side - 1), (y0 + 1).min(side - 1));
        let corners = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)];
        let positions = self.store.surface_positions(key)?;
        let nearest = corners
            .into_iter()
            .filter_map(|(i, j)| {
                let at = (j * side + i) as usize;
                Some((distance_squared(*positions.get(at)?, hit.position), at))
            })
            .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let Some((_, at)) = nearest else {
            return Ok(None);
        };
        let inside = positions
            .get(at)
            .is_some_and(|&p| Ball::new(center, reach_squared).holds(p));
        Ok(inside.then_some((key, at)))
    }

    /// Evaluates, together on the store's workers, every tile whose box touches the ball of
    /// `reach` around `center` and whose positions aren't cached: what a walk there reads, made
    /// current in parallel instead of one tile at a time as the walk comes to it.
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate_many`].
    fn prefetch(&mut self, center: DVec3, reach: f64) -> Result<(), CacheError> {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the index's query is f32; a tile it misses by rounding is evaluated by the \
                      walk when it reaches it"
        )]
        let radius = reach as f32;
        let store = &*self.store;
        let missing: Vec<TileKey> = self
            .index
            .query_sphere(center.to_f32(), radius)
            .filter(|key| store.cached_surface_positions(key).is_none())
            .collect();
        if !missing.is_empty() {
            self.store.evaluate_many(&missing, Evaluation::Positions)?;
        }
        Ok(())
    }

    /// [`Surface::raycast`], its errors returned.
    ///
    /// # Errors
    ///
    /// As [`SpatialIndex::raycast_triangle`] and [`TileStore::normals`].
    fn try_raycast(&mut self, ray: &Ray) -> Result<Option<Hit>, SculptError> {
        let Some(narrow) = store::Ray::new(ray.origin().to_f32(), ray.direction().to_f32()) else {
            return Ok(None);
        };
        let Some(triangle) = self.index.raycast_surface(self.store, &narrow)? else {
            return Ok(None);
        };
        self.hit_on_triangle(ray, triangle).map(Some)
    }

    /// The [`Hit`] where `ray` crosses `triangle`: the point interpolated on the triangle, the
    /// evaluated normals interpolated and normalised (the triangle's own normal, facing the
    /// ray's origin, where they cancel), and the face's coordinates.
    ///
    /// # Errors
    ///
    /// As [`TileStore::normals`].
    fn hit_on_triangle(&mut self, ray: &Ray, triangle: TriangleHit) -> Result<Hit, SculptError> {
        let key = triangle.key;
        let corners = corners(&triangle);
        let weights = weights(&triangle);
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
        let (gx, gy) = grid_point(&triangle, weights);
        Ok(Hit {
            position,
            normal,
            face: key.face(),
            uv: face_uv(key, gx, gy),
        })
    }

    /// The positions and normals of tile `key`'s vertices `at` (row-major indices) on the
    /// stroke's surface, in `f64`.
    ///
    /// # Errors
    ///
    /// As [`TileStore::surface_normals`] and [`TileStore::surface_positions`].
    fn tile_values<const N: usize>(
        &mut self,
        key: TileKey,
        at: [usize; N],
    ) -> Result<([DVec3; N], [DVec3; N]), SculptError> {
        let normals = gather(self.store.surface_normals(key)?, at);
        let positions = gather(self.store.surface_positions(key)?, at);
        Ok((positions, normals))
    }

    /// The walk from `near` over the region within `reach` of it (see the module notes), its
    /// tiles prefetched first; `None` if `near` has no vertex of its quad within reach.
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate`] and [`TileStore::evaluate_many`].
    fn region(
        &mut self,
        near: &Hit,
        reach: f64,
        target: Option<&Hit>,
    ) -> Result<Option<Walked>, CacheError> {
        let reach_squared = reach * reach;
        let center = near.position;
        let prefetched = target.map_or(reach, |b| b.position.distance(center).min(reach));
        self.prefetch(center, prefetched)?;
        let Some(start) = self.anchor(near, center, reach_squared)? else {
            return Ok(None);
        };
        let goal = match target {
            Some(b) => match self.anchor(b, center, reach_squared)? {
                Some(goal) => Some(goal),
                None => return Ok(Some(Walked::default())),
            },
            None => None,
        };
        walk(
            &mut Reader {
                store: self.store,
                index: self.index,
            },
            start,
            center,
            reach_squared,
            goal,
        )
        .map(Some)
    }

    /// [`Surface::closest_on_region`], its errors returned.
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate`] and [`TileStore::normals`].
    fn try_closest(&mut self, ray: &Ray, near: &Hit, reach: f64) -> Result<Hit, SculptError> {
        let Some(walked) = self.region(near, reach, None)? else {
            return Ok(*near);
        };
        // Nearest box first, so the scan can stop at the first tile whose box is farther from
        // the ray than the best vertex so far: the answer doesn't depend on the order, as ties
        // go to the least key and vertex either way.
        let mut tiles: Vec<(f64, TileKey)> = walked
            .tiles
            .keys()
            .map(|&key| (self.box_distance_squared(ray, &key), key))
            .collect();
        tiles.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut best: Option<(f64, TileKey, usize)> = None;
        for (bound, key) in tiles {
            if best.is_some_and(|(least, _, _)| bound > least) {
                break;
            }
            let Some(reached) = walked.tiles.get(&key) else {
                continue;
            };
            let positions = self.store.surface_positions(key)?;
            let on_region = positions
                .iter()
                .enumerate()
                .filter(|&(k, _)| reached.holds(k));
            for (k, &p) in on_region {
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
        let side = key.side() as usize;
        let (i, j) = (k % side, k / side);
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

    /// [`Surface::area_normal`], its errors returned: the stroke's surface's normals inside the
    /// dab's sphere, each weighted by the draw action's falloff there, summed in `f64` (see
    /// "Area normals" in the module notes).
    ///
    /// # Errors
    ///
    /// As [`TileStore::surface_normals`] and [`TileStore::surface_positions`].
    fn try_area_normal(
        &mut self,
        center: DVec3,
        radius: f64,
        hardness: f32,
    ) -> Result<Option<DVec3>, SculptError> {
        let Some(imprint) = weighing(center, radius, hardness) else {
            return Ok(None);
        };
        let keys: BTreeSet<TileKey> = self
            .index
            .query_sphere(centre_of(&imprint), area_reach(&imprint))
            .collect();
        let keys: Vec<TileKey> = keys.into_iter().collect();
        self.store.prepare_surface_normals(&keys)?;
        // Each tile's positions first, on this thread: a preview's are made from its block on
        // first read.
        for &key in &keys {
            self.store.surface_positions(key)?;
        }
        let store = &*self.store;
        let at_hand: Vec<Option<SurfaceTile<'_>>> = keys
            .iter()
            .map(|key| {
                let positions = store.cached_surface_positions(key)?;
                let normals = store.cached_surface_normals(key)?;
                Some((*key, positions, normals))
            })
            .collect();
        let topology = store.mesh().topology();
        let sums: Vec<Option<DVec3>> = store.par_map(&at_hand, |tile| {
            tile.map(|(key, positions, normals)| {
                tile_area_sum(topology, key, positions, normals, &imprint)
            })
        });
        let mut sum = DVec3::ZERO;
        let tiles = keys.iter().zip(sums);
        for (&key, tile) in tiles {
            let part = if let Some(part) = tile {
                part
            } else {
                // Pushed out of the cache before it was read: on its own.
                let normals = self.store.surface_normals(key)?.to_vec();
                let positions = self.store.surface_positions(key)?.to_vec();
                let mesh = self.store.mesh();
                tile_area_sum(mesh.topology(), key, &positions, &normals, &imprint)
            };
            sum = sum + part;
        }
        Ok(Some(sum))
    }

    /// [`Surface::continuous`], its errors returned.
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate`] and [`TileStore::evaluate_many`].
    fn try_continuous(&mut self, a: &Hit, b: &Hit, reach: f64) -> Result<bool, SculptError> {
        if a.position.distance(b.position) > reach {
            return Ok(false);
        }
        let walked = self.region(a, reach, Some(b))?;
        Ok(walked.is_some_and(|walked| walked.found))
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

    fn area_normal(&mut self, center: DVec3, radius: f64, hardness: f32) -> Option<DVec3> {
        if self.error.is_some() {
            return None;
        }
        let sum = self.try_area_normal(center, radius, hardness);
        self.keep(sum).flatten()
    }
}

/// A tile of the stroke's surface as an area normal reads it: its key, positions and normals.
type SurfaceTile<'a> = (TileKey, &'a [Vec3], &'a [Vec3]);

/// Tile `key`'s part of an area normal's sum for `imprint` (see "Area normals" in the module
/// notes).
///
/// Its own vertices' `normals` (those it owns, [`Topology::owns`]), each weighted by `imprint`'s
/// displacement at its position in `positions`, summed row by row in `f64`.
fn tile_area_sum(
    topology: &Topology,
    key: TileKey,
    positions: &[Vec3],
    normals: &[Vec3],
    imprint: &Imprint,
) -> DVec3 {
    let owned = vertices(key.side())
        .zip(positions.iter().zip(normals))
        .filter(|&((i, j), _)| topology.owns(&key, i, j));
    owned.fold(DVec3::ZERO, |sum, (_, (p, n))| {
        match draw_imprint_vertex(&Smoothstep, imprint, [p.x, p.y, p.z]) {
            Some(w) => sum + DVec3::from_f32(*n) * f64::from(w),
            None => sum,
        }
    })
}

/// The imprint whose draw weights the normals of a dab at `center` of `radius` and `hardness`.
///
/// Of unit strength, so its displacement at a vertex is the falloff's weight there. `None` for a
/// dab that covers nothing (a radius that isn't positive and finite in `f32`, a centre that isn't
/// finite).
fn weighing(center: DVec3, radius: f64, hardness: f32) -> Option<Imprint> {
    let centre = center.to_f32();
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a dab's radius is f32: the same rounding as the dab's own"
    )]
    let radius = radius as f32;
    (centre.is_finite() && radius > 0.0 && radius.is_finite()).then(|| {
        Imprint::new(
            [centre.x, centre.y, centre.z],
            [0.0, 0.0, 1.0],
            radius,
            1.0,
            hardness,
            [0.0; 4],
        )
    })
}

/// How far the index is queried around `imprint`'s centre for its area normal.
///
/// Its radius, rounded well up, so a vertex the falloff finds inside (on its squared distance in
/// `f32`) is never in a tile whose box the query misses.
fn area_reach(imprint: &Imprint) -> f32 {
    let [x, y, z] = imprint.center;
    let size = 1.0 + x.abs().max(y.abs()).max(z.abs());
    imprint.radius * 1.001 + 1e-6 * size
}

/// The store and index as the region walk reads them.
#[derive(Debug)]
struct Reader<'s> {
    /// The store, for positions and the topology.
    store: &'s mut TileStore,
    /// The index, for tiles' boxes.
    index: &'s SpatialIndex,
}

impl Source for Reader<'_> {
    type Error = CacheError;

    fn topology(&self) -> &Topology {
        self.store.mesh().topology()
    }

    fn positions(&mut self, key: TileKey) -> Result<&[Vec3], CacheError> {
        self.store.surface_positions(key)
    }

    /// Whether the tile's box in the index lies inside the ball: every corner of it within
    /// reach. The index is refitted after every edit, and grown to every preview, so its boxes
    /// bound the surface's positions.
    fn wholly_within(&self, key: &TileKey, center: DVec3, reach_squared: f64) -> bool {
        let Some(bounds) = self.index.bvh().bounds(key) else {
            return false;
        };
        let (low, high) = (bounds.aabb.min(), bounds.aabb.max());
        let corners = (0..8_u8).map(|c| {
            Vec3::new(
                if c & 1 == 0 { low.x } else { high.x },
                if c & 2 == 0 { low.y } else { high.y },
                if c & 4 == 0 { low.z } else { high.z },
            )
        });
        let mut corners = corners;
        corners.all(|corner| distance_squared(corner, center) <= reach_squared)
    }
}

/// `imprint`'s centre as a vector.
const fn centre_of(imprint: &Imprint) -> Vec3 {
    let [x, y, z] = imprint.center;
    Vec3::new(x, y, z)
}

/// The row-major indices of `triangle`'s corners in its tile, in the test's order: the corner,
/// the second, the third (see [`TriangleHit`]).
fn corners(triangle: &TriangleHit) -> [usize; 3] {
    let side = triangle.key.side();
    let at = |i: u32, j: u32| (j * side + i) as usize;
    let (i, j) = (triangle.i, triangle.j);
    if triangle.second {
        [at(i, j), at(i + 1, j + 1), at(i, j + 1)]
    } else {
        [at(i, j), at(i + 1, j), at(i + 1, j + 1)]
    }
}

/// `triangle`'s corners' weights, clamped onto the triangle: each at least 0, summing to 1.
fn weights(triangle: &TriangleHit) -> [f64; 3] {
    let (u, v) = (
        f64::from(triangle.u).max(0.0),
        f64::from(triangle.v).max(0.0),
    );
    let sum = u + v;
    let (u, v) = if sum > 1.0 {
        (u / sum, v / sum)
    } else {
        (u, v)
    };
    [1.0 - u - v, u, v]
}

/// The point of `triangle` with its corners' `weights`, in its tile's grid coordinates.
fn grid_point(triangle: &TriangleHit, [_, u, v]: [f64; 3]) -> (f64, f64) {
    let (i, j) = (f64::from(triangle.i), f64::from(triangle.j));
    if triangle.second {
        (i + u, j + u + v)
    } else {
        (i + u + v, j + v)
    }
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
    floor_to(u * f64::from(n), n)
}

/// `⌊x⌋`, clamped to `0..=most`; a NaN gives 0.
fn floor_to(x: f64, most: u32) -> u32 {
    let g = x.floor().clamp(0.0, f64::from(most));
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped to 0..=most just above, which a u32 holds; a NaN casts to 0"
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
