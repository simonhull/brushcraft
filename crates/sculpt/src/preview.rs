//! Previews: a brush's dabs shown at once on the finest tiles they reach, ahead of the finest
//! work that makes them exact ([`Preview`]).
//!
//! # What a preview is
//!
//! [`apply_dabs`](crate::apply_dabs) edits the corrections: it predicts each tile it reaches,
//! runs the brush, writes and re-evaluates, which is the store's authority and costs about a
//! millisecond or two a dab of a few dozen tiles. When a frame's dabs don't all fit its time, the
//! ones left over are **shown** here instead, and **land** later, through `apply_dabs` as ever, in
//! the same order: the saved result is that of applying them all at once, bit for bit (the
//! preview is never written).
//!
//! A shown dab moves the finest tiles its sphere reaches as the store holds them, each gathered
//! with its ring ([`TileStore::cached_positions_with_ring`]), by the draw action's own amount
//! ([`brush::draw_imprint_vertex`], the code `apply_dabs` runs): every point within its radius
//! moves `w(p)` along the dab's normal. That is the one difference from the finest work, which
//! moves each vertex along its frame normal of the predicted surface: the frames need the
//! prediction, the costly part. The two agree to within the surface's turn over the dab.
//!
//! A tile's preview is its points as the store holds them, moved by every dab shown and not yet
//! landed, in order. Each move is a function of a point's bits alone, so every tile holding a
//! vertex (on every face) moves it to the same bits ([`geom::GatheredBlock::displace`]): previews
//! meet at their seams bit for bit, as the store's tiles do.
//!
//! The store keeps the previews ([`TileStore::set_preview`]) and hands them to the renderer in
//! epochs of their own ([`TileStore::close_preview_epoch`]), one a call here; it never reads or
//! saves them (see the store's `preview` notes).
//!
//! # Which tiles
//!
//! A dab moves the points within its radius. A tile holds such a point only if its box (of every
//! point of its block, ring included) touches the dab's sphere: a previewed tile by the box of
//! its preview, any other by its box in the spatial index (the store's surface). So a dab shown
//! moves the previewed tiles whose box touches it, and gathers those of the index's query it
//! reaches that aren't previewed.
//!
//! A tile the index reaches whose points the dab doesn't move is kept as **untouched**, with its
//! box, so the next dabs don't gather it again. The index's boxes are of a tile's own vertices,
//! without its ring: a tile beside a preview whose ring alone a dab moves keeps the store's ring
//! (its border normals on the GPU, which it computes from the ring, lag until it is previewed or
//! the dab lands). Its own vertices, and so the seams, are the store's.
//!
//! # Landing
//!
//! When the oldest dabs shown land ([`Preview::land`]), the store's finest surface moves under
//! the previews, and each tile whose points they moved is made again: the store's tile now, moved
//! by the dabs still shown. A landed dab moved a point by at most its strength, plus the stored
//! quantization of the tile the point is inside (its own corrections only, re-quantized by its own
//! writes, which a dab makes only where it moves a vertex of the tile). So it moved a point of a
//! previewed tile only if the tile's preview box, grown by how far the dabs shown moved its points
//! (each dab at most its strength) and by the landed dabs' strengths, touches its sphere; those
//! tiles are made again, and a tile no dab shown moves any more goes back to the store's. A tile
//! that wasn't previewed and whose surface the landed dabs moved may now lie within a dab still
//! shown: the index's boxes, refitted by the landing, find those. Untouched tiles the landing
//! moved are forgotten, to be looked at again.
//!
//! Once no dab is shown, no tile is previewed.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use brush::{Imprint, Smoothstep, draw_imprint_vertex};
use geom::{GatheredBlock, TileKey, Vec3};
use store::{Aabb, CacheError, Evaluation, SpatialIndex, TileStore};
use stroke::Dab;

use crate::apply::{SculptError, imprint_of};

/// What a [`Preview::show`] or [`Preview::land`] did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PreviewStats {
    /// Dabs shown (by `show`) or landed (by `land`).
    pub dabs: usize,
    /// Tiles whose previews were set: moved by a dab shown, or made again over what landed.
    pub tiles_set: usize,
    /// Tiles whose previews were cleared: no dab shown moves them any more.
    pub tiles_cleared: usize,
    /// Tiles gathered from the store to preview (or look at) afresh.
    pub gathered: usize,
}

/// What the preview knows of a tile a dab shown has reached.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Reached {
    /// The box of every point of its block as drawn: its preview's, or (untouched) the store's;
    /// `None` for a block of no finite point (never, for a tile of the store), which no dab
    /// reaches.
    bounds: Option<Aabb>,
    /// How far the dabs shown moved its points at most: the strengths of those that moved any,
    /// summed. Zero for a tile untouched.
    slack: f32,
}

/// A tile being moved in one call: its block and what is known of it.
#[derive(Debug)]
struct Moving {
    /// The tile.
    key: TileKey,
    /// Its block: its preview, or the store's tile gathered with its ring.
    block: GatheredBlock,
    /// How far the dabs shown moved its points at most.
    slack: f32,
    /// Whether it had a preview as the call began.
    previewed: bool,
}

/// A brush's previews: the dabs shown and not yet landed, and the tiles they reach (see the
/// module notes).
///
/// The caller shows the dabs a step can't apply ([`Preview::show`]) and, as it applies the
/// oldest of them through [`apply_dabs`](crate::apply_dabs) in order, lands them
/// ([`Preview::land`]) after each call. Dabs dropped unapplied (a stroke ended by an error) are
/// landed too: nothing then moves under the previews, and they are made again without them.
#[derive(Debug, Default)]
pub struct Preview {
    /// The dabs shown and not landed, oldest first.
    shown: VecDeque<Imprint>,
    /// The tiles they reach: previewed ones, and untouched ones the index found.
    reached: BTreeMap<TileKey, Reached>,
}

impl Preview {
    /// No dab shown.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// How many dabs are shown and not yet landed.
    #[must_use]
    pub fn shown(&self) -> usize {
        self.shown.len()
    }

    /// Shows `dabs`, after those shown already: moves the previews of the finest tiles each
    /// reaches (gathering those it reaches first from the store), sets them in the store and
    /// closes their preview epoch.
    ///
    /// # Errors
    ///
    /// [`SculptError::Dab`] for a dab the brush can't apply, before anything is shown;
    /// [`SculptError::Cache`] if a tile can't be gathered: the dabs before the failing one stay
    /// shown, and the call's epoch is closed all the same.
    pub fn show(
        &mut self,
        store: &mut TileStore,
        index: &SpatialIndex,
        dabs: &[Dab],
    ) -> Result<PreviewStats, SculptError> {
        let imprints: Result<Vec<Imprint>, SculptError> = dabs
            .iter()
            .enumerate()
            .map(|(n, dab)| imprint_of(dab).map_err(|error| SculptError::Dab { dab: n, error }))
            .collect();
        let imprints = imprints?;
        let mut stats = PreviewStats::default();
        let shown = imprints
            .iter()
            .try_for_each(|imprint| self.show_one(store, index, imprint, &mut stats));
        let closed = store.close_preview_epoch();
        shown?;
        closed?;
        Ok(stats)
    }

    /// Shows one dab (see [`Preview::show`]).
    ///
    /// # Errors
    ///
    /// [`SculptError::Cache`] if a tile can't be gathered.
    fn show_one(
        &mut self,
        store: &mut TileStore,
        index: &SpatialIndex,
        imprint: &Imprint,
        stats: &mut PreviewStats,
    ) -> Result<(), SculptError> {
        self.shown.push_back(*imprint);
        stats.dabs += 1;
        let [x, y, z] = imprint.center;
        let center = Vec3::new(x, y, z);
        let near: Vec<TileKey> = self
            .reached
            .iter()
            .filter(|(_, reached)| touches(reached.bounds.as_ref(), imprint, 0.0))
            .map(|(key, _)| *key)
            .collect();
        let queried = index
            .query_sphere(center, reach(imprint, 0.0))
            .filter(|key| !self.reached.contains_key(key));
        let fresh: Vec<TileKey> = queried.collect();
        // Tiles already reached take this dab alone; fresh ones, the store's tile moved by every
        // dab shown (this one last), as are untouched ones (the dabs before moved nothing there).
        let mut moving: Vec<Moving> = Vec::with_capacity(near.len() + fresh.len());
        let mut from_store: Vec<TileKey> = fresh;
        for key in near {
            match store.take_preview(&key) {
                Some(block) => {
                    let slack = self.reached.get(&key).map_or(0.0, |r| r.slack);
                    moving.push(Moving {
                        key,
                        block,
                        slack,
                        previewed: true,
                    });
                }
                None => from_store.push(key),
            }
        }
        let gathered = gather(store, &from_store)?;
        stats.gathered += gathered.len();
        let all: Vec<Imprint> = self.shown.iter().copied().collect();
        let incremental = moving.len();
        moving.extend(gathered);
        let (old, new) = moving.split_at_mut(incremental);
        store.par_map_mut(old, |tile| {
            if move_by(&mut tile.block, imprint) {
                tile.slack = add_up(tile.slack, imprint.strength.abs());
            }
        });
        store.par_map_mut(new, |tile| move_by_every(tile, &all));
        self.keep(store, moving, stats)
    }

    /// Lands the oldest `count` dabs shown (fewer if fewer are): they were applied through
    /// [`apply_dabs`](crate::apply_dabs), in order, since they were shown, or dropped. Every tile
    /// whose points they may have moved is made again over the store's tile as it is now, by
    /// the dabs still shown, or goes back to the store's; and so is every tile the landing
    /// brought within a dab still shown. Closes the call's preview epoch.
    ///
    /// `index` is the spatial index as the landing left it (refitted).
    ///
    /// # Errors
    ///
    /// [`SculptError::Cache`] if a tile can't be gathered: its preview is cleared (the store's
    /// tile shows), and the epoch is closed all the same.
    pub fn land(
        &mut self,
        store: &mut TileStore,
        index: &SpatialIndex,
        count: usize,
    ) -> Result<PreviewStats, SculptError> {
        let count = count.min(self.shown.len());
        let landed: Vec<Imprint> = self.shown.drain(..count).collect();
        let mut stats = PreviewStats {
            dabs: landed.len(),
            ..PreviewStats::default()
        };
        if landed.is_empty() {
            return Ok(stats);
        }
        let made = self.land_tiles(store, index, &landed, &mut stats);
        let closed = store.close_preview_epoch();
        made?;
        closed?;
        Ok(stats)
    }

    /// The tiles of a landing of `landed` made again (see [`Preview::land`]).
    ///
    /// # Errors
    ///
    /// [`SculptError::Cache`] if a tile can't be gathered.
    fn land_tiles(
        &mut self,
        store: &mut TileStore,
        index: &SpatialIndex,
        landed: &[Imprint],
        stats: &mut PreviewStats,
    ) -> Result<(), SculptError> {
        if self.shown.is_empty() {
            // Nothing shown any more: every preview goes back to the store's tile.
            for key in std::mem::take(&mut self.reached).into_keys() {
                if store.preview(&key).is_some() {
                    store.clear_preview(&key);
                    stats.tiles_cleared += 1;
                }
            }
            return Ok(());
        }
        let landed_strength = landed
            .iter()
            .fold(0.0_f32, |sum, dab| add_up(sum, dab.strength.abs()));
        // Previewed tiles whose points the landing may have moved, and untouched ones whose
        // store tiles it may have moved (forgotten, to be looked at again).
        let mut again: BTreeSet<TileKey> = BTreeSet::new();
        let mut forget: Vec<TileKey> = Vec::new();
        for (key, reached) in &self.reached {
            let grown = add_up(reached.slack, landed_strength);
            if landed
                .iter()
                .any(|dab| touches(reached.bounds.as_ref(), dab, grown))
            {
                if reached.slack > 0.0 {
                    again.insert(*key);
                } else {
                    forget.push(*key);
                }
            }
        }
        for key in forget {
            self.reached.remove(&key);
        }
        // Tiles the landing moved that no preview holds yet: within a dab still shown now?
        let moved: BTreeSet<TileKey> = landed
            .iter()
            .flat_map(|dab| {
                let [x, y, z] = dab.center;
                index.query_sphere(Vec3::new(x, y, z), reach(dab, landed_strength))
            })
            .filter(|key| !self.reached.contains_key(key))
            .collect();
        if !moved.is_empty() {
            for dab in &self.shown {
                let [x, y, z] = dab.center;
                let within = index
                    .query_sphere(Vec3::new(x, y, z), reach(dab, 0.0))
                    .filter(|key| moved.contains(key));
                again.extend(within);
            }
        }
        self.remake(store, &again, stats)
    }

    /// Makes every previewed tile again over the store's tiles as they are now, by the dabs
    /// still shown, and looks again at the tiles a dab shown reaches: for a store whose tiles
    /// changed under the previews otherwise than by dabs landing (a frame of dabs that failed
    /// part way, leaving the failing dab's writes). Closes the call's preview epoch.
    ///
    /// # Errors
    ///
    /// As [`Preview::land`].
    pub fn rebase(
        &mut self,
        store: &mut TileStore,
        index: &SpatialIndex,
    ) -> Result<PreviewStats, SculptError> {
        let mut stats = PreviewStats::default();
        let mut again: BTreeSet<TileKey> = BTreeSet::new();
        let reached = std::mem::take(&mut self.reached);
        let previewed = reached
            .into_iter()
            .filter(|(_, reached)| reached.slack > 0.0);
        again.extend(previewed.map(|(key, _)| key));
        for dab in &self.shown {
            let [x, y, z] = dab.center;
            again.extend(index.query_sphere(Vec3::new(x, y, z), reach(dab, 0.0)));
        }
        let made = self.remake(store, &again, &mut stats);
        let closed = store.close_preview_epoch();
        made?;
        closed?;
        Ok(stats)
    }

    /// Makes tiles `again` over the store's tiles as they are now, moved by the dabs still
    /// shown: their old previews dropped, each set again if a dab moves it, else cleared.
    ///
    /// # Errors
    ///
    /// [`SculptError::Cache`] if a tile can't be gathered: their previews are cleared.
    fn remake(
        &mut self,
        store: &mut TileStore,
        again: &BTreeSet<TileKey>,
        stats: &mut PreviewStats,
    ) -> Result<(), SculptError> {
        let keys: Vec<TileKey> = again.iter().copied().collect();
        // The store's tiles are gathered afresh; the old previews are dropped.
        let previewed: BTreeSet<TileKey> = keys
            .iter()
            .filter(|key| store.take_preview(key).is_some())
            .copied()
            .collect();
        let mut moving = match gather(store, &keys) {
            Ok(moving) => moving,
            Err(error) => {
                for key in &keys {
                    self.reached.remove(key);
                    store.clear_preview(key);
                }
                return Err(error);
            }
        };
        stats.gathered += moving.len();
        for tile in &mut moving {
            tile.previewed = previewed.contains(&tile.key);
        }
        let shown: Vec<Imprint> = self.shown.iter().copied().collect();
        store.par_map_mut(&mut moving, |tile| move_by_every(tile, &shown));
        self.keep(store, moving, stats)
    }

    /// Hands the tiles of a call back to the store: a moved tile's preview set, an unmoved
    /// one's cleared (if it had one) and kept as untouched.
    ///
    /// # Errors
    ///
    /// [`SculptError::Cache`] if the store refuses a preview (never, for the finest tiles the
    /// index holds).
    fn keep(
        &mut self,
        store: &mut TileStore,
        moving: Vec<Moving>,
        stats: &mut PreviewStats,
    ) -> Result<(), SculptError> {
        for tile in moving {
            let bounds = bounds_of(&tile.block);
            self.reached.insert(
                tile.key,
                Reached {
                    bounds,
                    slack: tile.slack,
                },
            );
            if tile.slack > 0.0 {
                store.set_preview(tile.key, tile.block)?;
                stats.tiles_set += 1;
            } else {
                // Moved by no dab shown: the store's tile shows (a preview taken is cleared).
                store.clear_preview(&tile.key);
                if tile.previewed {
                    stats.tiles_cleared += 1;
                }
            }
        }
        Ok(())
    }
}

/// The store's tiles `keys`, gathered with their rings, as tiles to move: evaluated together on
/// the store's workers first, then gathered from its cache, on the workers too.
///
/// # Errors
///
/// [`SculptError::Cache`] if a tile can't be evaluated or gathered.
fn gather(store: &mut TileStore, keys: &[TileKey]) -> Result<Vec<Moving>, SculptError> {
    if keys.is_empty() {
        return Ok(Vec::new());
    }
    let mut inputs: BTreeSet<TileKey> = BTreeSet::new();
    for key in keys {
        inputs.extend(store.normal_inputs(key)?.iter().copied());
    }
    let inputs: Vec<TileKey> = inputs.into_iter().collect();
    store.evaluate_many(&inputs, Evaluation::Positions)?;
    let cached = store.cached_positions_with_ring_many(keys)?;
    keys.iter()
        .zip(cached)
        .map(|(&key, cached)| {
            let block = match cached {
                Some(block) => block,
                None => store.positions_with_ring(key)?,
            };
            Ok(Moving {
                key,
                block,
                slack: 0.0,
                previewed: false,
            })
        })
        .collect::<Result<Vec<Moving>, CacheError>>()
        .map_err(SculptError::from)
}

/// Moves `tile`, as the store holds it, by every dab of `shown` in order: those whose sphere
/// its box touches as each comes. Its slack grows by the strength of each that moves a point.
fn move_by_every(tile: &mut Moving, shown: &[Imprint]) {
    for dab in shown {
        if touches(bounds_of(&tile.block).as_ref(), dab, 0.0) && move_by(&mut tile.block, dab) {
            tile.slack = add_up(tile.slack, dab.strength.abs());
        }
    }
}

/// Moves every point of `block` within `dab`'s radius by the draw action's amount along the
/// dab's normal; whether it moved any.
fn move_by(block: &mut GatheredBlock, dab: &Imprint) -> bool {
    let [nx, ny, nz] = dab.normal;
    let normal = Vec3::new(nx, ny, nz);
    let mut moved = false;
    block.displace(
        |p| match draw_imprint_vertex(&Smoothstep, dab, [p.x, p.y, p.z]) {
            Some(w) if w != 0.0 => {
                moved = true;
                p + normal * w
            }
            _ => p,
        },
    );
    moved
}

/// The box of every point of `block`: its positions, ring included, and its base-vertex rings'
/// points; `None` for a block of no finite point (never, for a tile of the store).
fn bounds_of(block: &GatheredBlock) -> Option<Aabb> {
    let rings = (0..4).filter_map(|c| block.corner(c));
    let ring_points: Vec<Vec3> = rings
        .flat_map(|ring| ring.edges().iter().chain(ring.diagonals()).copied())
        .collect();
    let points = block.padded().iter().chain(&ring_points).copied();
    let finite = points.filter(|p| p.x.is_finite() && p.y.is_finite() && p.z.is_finite());
    Aabb::from_points(finite)
}

/// How far a test of `dab`'s sphere, grown by `grow`, reaches: its radius and `grow`, rounded
/// well up.
///
/// So a point the draw action finds within the radius (its squared distance below the squared
/// radius, in `f32`) is never outside a box the test finds it misses.
fn reach(dab: &Imprint, grow: f32) -> f32 {
    let [x, y, z] = dab.center;
    let size = 1.0 + x.abs().max(y.abs()).max(z.abs());
    (dab.radius + grow) * 1.001 + 1e-6 * size
}

/// Whether `bounds` touches `dab`'s sphere grown by `grow` (see [`reach`]); no box touches
/// nothing.
fn touches(bounds: Option<&Aabb>, dab: &Imprint, grow: f32) -> bool {
    let Some(bounds) = bounds else {
        return false;
    };
    let (lo, hi) = (bounds.min(), bounds.max());
    let gap = |c: f32, l: f32, h: f32| {
        if c < l {
            l - c
        } else if c > h {
            c - h
        } else {
            0.0
        }
    };
    let [x, y, z] = dab.center;
    let (dx, dy, dz) = (gap(x, lo.x, hi.x), gap(y, lo.y, hi.y), gap(z, lo.z, hi.z));
    let r = reach(dab, grow);
    dx * dx + dy * dy + dz * dz <= r * r
}

/// `a + b`, rounded up: a bound on two movements one after the other.
fn add_up(a: f32, b: f32) -> f32 {
    if a == 0.0 {
        b
    } else if b == 0.0 {
        a
    } else {
        (a + b).next_up()
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;
