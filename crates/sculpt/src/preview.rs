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
//! with its ring ([`TileStore::cached_positions_with_ring_many`]), as the finest work moves them:
//! every point within its radius **at pen-down** moves by the draw action's own move
//! ([`brush::draw_imprint_vertex`] at the point's pen-down position, and [`brush::draw_move`], the
//! code `apply_dabs` runs): its displacement along the dab's normal, the dab's area normal, one
//! direction for every point. The finest work writes that move into the vertex's correction, in
//! its tangent frame, weighed at the same pen-down position. So a preview differs from what lands
//! only by the stored encoding (the correction rounded to its tile's quantization once a stroke,
//! and the frame's arithmetic). Placement doesn't read previews: a stroke places its dabs on its
//! pen-down surface ([`StoreSurface`](crate::StoreSurface)).
//!
//! A tile's preview is its points as the store holds them, moved by every dab shown and not yet
//! landed, in order, each point weighed at its pen-down position in the tile's **pen-down block**
//! ([`TileStore::pen_down_block`]: the tile and its ring at pen-down;
//! [`geom::GatheredBlock::displace_along`]). Each move is a function of a point's bits, its
//! pen-down bits and the dab alone, so every tile holding the vertex (on every face) moves it to
//! the same bits: previews meet at their seams bit for bit, as the store's tiles do.
//!
//! The store keeps the previews ([`TileStore::set_preview`]) and hands them to the renderer in
//! epochs of their own ([`TileStore::close_preview_epoch`]), one a call here; it never reads or
//! saves them (see the store's `preview` notes). A previewed tile's pen-down block is kept here,
//! beside it, until no dab shown moves it: a second block of the same size, in memory only.
//!
//! # Which tiles
//!
//! A dab moves the points within its radius at pen-down. A tile holds such a point only if the
//! box of its pen-down block (of every point of the block, ring included) touches the dab's
//! sphere; a tile not reached yet is found by its box in the spatial index, which holds its
//! pen-down positions (see [`SpatialIndex::refit`]). So a dab shown moves the reached tiles whose
//! pen-down box touches it, and gathers those of the index's query it reaches that aren't reached
//! yet. Neither box moves with the dabs.
//!
//! A tile the index reaches whose points the dab doesn't move is kept as **untouched**, with its
//! box, so the next dabs don't gather it again. The index's boxes are of a tile's own vertices,
//! without its ring: a tile beside a preview whose ring alone a dab moves keeps the store's ring
//! (its border normals on the GPU, which it computes from the ring, lag until it is previewed or
//! the dab lands). Its own vertices, and so the seams, are the store's.
//!
//! # Landing
//!
//! When the oldest dabs shown land ([`Preview::land`]), the store's finest surface moves under the
//! previews. A landed dab moved the vertices within its radius at pen-down, and a tile's interior
//! quantization only by that tile's own writes, so it changed the positions of the tiles its
//! sphere reaches in the index, and the rings of their neighbours ([`TileStore::stale_after`]):
//! the landing's tiles. Each of them that is previewed is made again: the store's tile now, moved
//! by the dabs still shown, or the store's tile alone if none moves it. One that isn't may lie
//! within a dab still shown, the store having moved it: its box in the index says, and it is
//! looked at again if so. Untouched tiles among them are forgotten, to be looked at again.
//!
//! Once no dab is shown, no tile is previewed.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use brush::{Imprint, Smoothstep, draw_imprint_vertex, draw_move};
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

/// A tile being moved in one call: its block, its pen-down block and box, and whether a dab
/// shown moved it.
#[derive(Debug)]
struct Moving {
    /// The tile.
    key: TileKey,
    /// Its block: its preview, or the store's tile gathered with its ring.
    block: GatheredBlock,
    /// Its pen-down block, which every point's move is weighed at.
    pen_down: GatheredBlock,
    /// The box of every point of `pen_down` ([`bounds_of`]).
    bounds: Option<Aabb>,
    /// Whether a dab shown moved a point of it: it is previewed.
    moved: bool,
    /// Whether it had a preview as the call began.
    previewed: bool,
    /// Whether `block` is its preview as the call began, taken from the store and not moved
    /// since: it goes back as it was ([`TileStore::put_back_preview`]), which a renderer has
    /// already.
    unmoved: bool,
}

impl Moving {
    /// The store's `block` of tile `key`, not moved yet, with its `pen_down` block.
    fn of_store(key: TileKey, block: GatheredBlock, pen_down: GatheredBlock) -> Self {
        let bounds = bounds_of(&pen_down);
        Self {
            key,
            block,
            pen_down,
            bounds,
            moved: false,
            previewed: false,
            unmoved: false,
        }
    }

    /// Moves it by every one of `dabs`, in order: by each whose sphere its pen-down box touches.
    fn move_by_every(&mut self, dabs: &[Imprint]) {
        for dab in dabs {
            if !touches(self.bounds.as_ref(), dab) {
                continue;
            }
            if move_by(&mut self.block, &self.pen_down, dab) {
                self.moved = true;
                self.unmoved = false;
            }
        }
    }
}

/// A tile the shown dabs reach, as a [`Preview`] keeps it between calls.
#[derive(Debug)]
struct Reached {
    /// The box of its pen-down block's points; `None` for a block of no finite point, which no
    /// dab reaches.
    bounds: Option<Aabb>,
    /// Its pen-down block, while it is previewed; `None` for an untouched tile.
    pen_down: Option<GatheredBlock>,
}

/// A brush's previews: the dabs shown and not yet landed, and the tiles they reach (see the
/// module notes).
///
/// The caller shows the dabs a step can't apply ([`Preview::show`]) and, as it applies the
/// oldest of them through [`apply_dabs`](crate::apply_dabs) in order, lands them
/// ([`Preview::land`]) after each call. Dabs dropped unapplied (a stroke ended by an error) are
/// landed too: nothing then moves under the previews, and they are made again without them; a
/// store changed otherwise under them (the failing dab's writes) has them made again
/// ([`Preview::rebase`]).
#[derive(Debug, Default)]
pub struct Preview {
    /// The dabs shown and not landed, oldest first.
    shown: VecDeque<Imprint>,
    /// The tiles they reach, previewed (with their pen-down blocks) or untouched.
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

    /// Shows `dabs`, after those shown already: moves the previews of the finest tiles they
    /// reach (gathering those reached first from the store), sets them in the store and closes
    /// their preview epoch.
    ///
    /// The dabs are applied tile by tile, each tile in parallel taking every dab whose sphere its
    /// pen-down box touches, in order: the tile's points move as one dab at a time would move
    /// them, as nothing a dab does to a tile depends on another tile. The tiles are those a dab
    /// can reach: a tile already reached whose pen-down box touches a dab's sphere, and those of
    /// the index's query of each dab not reached yet.
    ///
    /// # Errors
    ///
    /// [`SculptError::Dab`] for a dab the brush can't apply, before anything is shown;
    /// [`SculptError::Cache`] if a tile can't be gathered: nothing is shown then, the previews as
    /// they were, and the call's epoch closed all the same.
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
        let shown = self.show_all(store, index, &imprints, &mut stats);
        let closed = store.close_preview_epoch();
        shown?;
        closed?;
        Ok(stats)
    }

    /// Shows `new` (see [`Preview::show`]).
    ///
    /// # Errors
    ///
    /// [`SculptError::Cache`] if a tile can't be gathered.
    fn show_all(
        &mut self,
        store: &mut TileStore,
        index: &SpatialIndex,
        new: &[Imprint],
        stats: &mut PreviewStats,
    ) -> Result<(), SculptError> {
        if new.is_empty() {
            return Ok(());
        }
        let near: Vec<(TileKey, bool)> = self
            .reached
            .iter()
            .filter(|(_, reached)| new.iter().any(|dab| touches(reached.bounds.as_ref(), dab)))
            .map(|(key, reached)| (*key, reached.pen_down.is_some()))
            .collect();
        let queried: BTreeSet<TileKey> = new
            .iter()
            .flat_map(|dab| index.query_sphere(centre(dab), reach(dab)))
            .filter(|key| !self.reached.contains_key(key))
            .collect();
        // Previewed tiles take the new dabs alone; the others, the store's tile moved by every
        // dab shown (for an untouched one, the dabs before moved nothing there).
        let mut from_store: Vec<TileKey> = queried.into_iter().collect();
        let previewed: Vec<TileKey> = near
            .iter()
            .filter(|&&(_, previewed)| previewed)
            .map(|&(key, _)| key)
            .collect();
        from_store.extend(near.iter().filter(|(_, p)| !p).map(|&(key, _)| key));
        let gathered = gather(store, &from_store)?;
        stats.gathered += gathered.len();
        let mut moving: Vec<Moving> = Vec::with_capacity(previewed.len() + gathered.len());
        for key in previewed {
            let Some(Reached {
                bounds,
                pen_down: Some(pen_down),
            }) = self.reached.remove(&key)
            else {
                continue;
            };
            let Some(block) = store.take_preview(&key) else {
                continue;
            };
            moving.push(Moving {
                key,
                block,
                pen_down,
                bounds,
                moved: true,
                previewed: true,
                unmoved: true,
            });
        }
        let incremental = moving.len();
        moving.extend(gathered);
        self.shown.extend(new.iter().copied());
        stats.dabs += new.len();
        let all: Vec<Imprint> = self.shown.iter().copied().collect();
        let (old, fresh) = moving.split_at_mut(incremental);
        store.par_map_mut(old, |tile| tile.move_by_every(new));
        store.par_map_mut(fresh, |tile| tile.move_by_every(&all));
        self.keep(store, moving, stats)
    }

    /// Lands the oldest `count` dabs shown (fewer if fewer are): they were applied through
    /// [`apply_dabs`](crate::apply_dabs), in order, since they were shown, or dropped. Every tile
    /// whose positions or ring their landing can have changed is made again (see "Landing" in the
    /// module notes). Closes the call's preview epoch.
    ///
    /// `index` is the spatial index as the landing left it (refitted).
    ///
    /// # Errors
    ///
    /// [`SculptError::Cache`] if a tile can't be gathered or listed: the previews of those being
    /// made again are cleared (the store's tiles show), and the epoch is closed all the same.
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
    /// [`SculptError::Cache`] if a tile can't be gathered or listed.
    fn land_tiles(
        &mut self,
        store: &mut TileStore,
        index: &SpatialIndex,
        landed: &[Imprint],
        stats: &mut PreviewStats,
    ) -> Result<(), SculptError> {
        if self.shown.is_empty() {
            // Nothing shown any more: every preview goes back to the store's tile.
            self.clear(store, stats);
            return Ok(());
        }
        let queried: BTreeSet<TileKey> = landed
            .iter()
            .flat_map(|dab| index.query_sphere(centre(dab), reach(dab)))
            .collect();
        let queried: Vec<TileKey> = queried.into_iter().collect();
        let changed = match store.stale_after(&queried) {
            Ok(changed) => changed,
            Err(error) => {
                self.clear(store, stats);
                return Err(error.into());
            }
        };
        let mut again: BTreeSet<TileKey> = BTreeSet::new();
        let mut unpreviewed: Vec<TileKey> = Vec::new();
        for key in changed {
            match self.reached.get(&key) {
                Some(reached) if reached.pen_down.is_some() => {
                    again.insert(key);
                }
                Some(_) => {
                    // Untouched, and the store moved it: looked at again below.
                    self.reached.remove(&key);
                    unpreviewed.push(key);
                }
                None => unpreviewed.push(key),
            }
        }
        // The store moved these: within a dab still shown, by their boxes in the index?
        let within = unpreviewed.into_iter().filter(|key| {
            let bounds = index.bvh().bounds(key).map(|bounds| bounds.aabb);
            // A tile without bounds is looked at whole: gathered, and kept only if moved.
            bounds.is_none() || self.shown.iter().any(|dab| touches(bounds.as_ref(), dab))
        });
        again.extend(within);
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
        let reached = std::mem::take(&mut self.reached);
        let previewed = reached
            .into_iter()
            .filter(|(_, reached)| reached.pen_down.is_some());
        let mut again: BTreeSet<TileKey> = previewed.map(|(key, _)| key).collect();
        for dab in &self.shown {
            again.extend(index.query_sphere(centre(dab), reach(dab)));
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
        store.par_map_mut(&mut moving, |tile| tile.move_by_every(&shown));
        self.keep(store, moving, stats)
    }

    /// Clears every preview and forgets every tile reached, the dabs shown kept: for a landing
    /// that can't tell which tiles it changed, or one that leaves no dab shown. The next dab
    /// shown gathers afresh.
    fn clear(&mut self, store: &mut TileStore, stats: &mut PreviewStats) {
        for (key, reached) in std::mem::take(&mut self.reached) {
            if reached.pen_down.is_some() {
                store.clear_preview(&key);
                stats.tiles_cleared += 1;
            }
        }
    }

    /// Hands the tiles of a call back to the store: a previewed tile no dab of the call moved put
    /// back as it was, any other moved tile's preview set, an unmoved one's cleared (if it had
    /// one) and kept as untouched.
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
            let pen_down = tile.moved.then_some(tile.pen_down);
            self.reached.insert(
                tile.key,
                Reached {
                    bounds: tile.bounds,
                    pen_down,
                },
            );
            if tile.unmoved {
                // Its preview as it was.
                store.put_back_preview(tile.key, tile.block)?;
            } else if tile.moved {
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

/// The store's tiles `keys`, gathered with their rings, as tiles to move, with their pen-down
/// blocks.
///
/// Evaluated together on the store's workers first, then gathered from its cache, on the workers
/// too. A tile whose block reads no tile the stroke changed is its own pen-down block.
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
    // The tiles whose blocks read a tile the stroke changed have pen-down blocks of their own;
    // each other's is its block.
    let mut changed: Vec<TileKey> = Vec::new();
    for key in keys {
        let read = store.normal_inputs(key)?;
        if read
            .iter()
            .any(|input| store.pen_down_shape(input).is_some())
        {
            changed.push(*key);
        }
    }
    let mut pen_down: BTreeMap<TileKey, GatheredBlock> = changed
        .iter()
        .copied()
        .zip(store.pen_down_blocks_many(&changed)?)
        .collect();
    let cached = store.cached_positions_with_ring_many(keys)?;
    keys.iter()
        .zip(cached)
        .map(|(&key, cached)| {
            let block = match cached {
                Some(block) => block,
                None => store.positions_with_ring(key)?,
            };
            let at_pen_down = pen_down.remove(&key).unwrap_or_else(|| block.clone());
            Ok(Moving::of_store(key, block, at_pen_down))
        })
        .collect::<Result<Vec<Moving>, CacheError>>()
        .map_err(SculptError::from)
}

/// Moves every point of `block` within `dab`'s radius at pen-down (its counterpart in
/// `pen_down`) by the draw action's move, its displacement along the dab's normal
/// ([`brush::draw_move`]); whether it moved any.
///
/// A point with no displacement (on the rim, or of none at the falloff's edge) keeps its bits:
/// adding a zero move would turn a `-0.0` into `0.0`.
fn move_by(block: &mut GatheredBlock, pen_down: &GatheredBlock, dab: &Imprint) -> bool {
    let mut moved = false;
    block.displace_along(pen_down, |p, at| {
        match draw_imprint_vertex(&Smoothstep, dab, [at.x, at.y, at.z]) {
            Some(w) if w != 0.0 => {
                moved = true;
                let [x, y, z] = draw_move(dab, w);
                p + Vec3::new(x, y, z)
            }
            _ => p,
        }
    });
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

/// `dab`'s centre.
const fn centre(dab: &Imprint) -> Vec3 {
    let [x, y, z] = dab.center;
    Vec3::new(x, y, z)
}

/// How far a test of `dab`'s sphere reaches: its radius, rounded well up.
///
/// So a point the draw action finds within the radius (its squared distance below the squared
/// radius, in `f32`) is never outside a box the test finds it misses.
fn reach(dab: &Imprint) -> f32 {
    let [x, y, z] = dab.center;
    let size = 1.0 + x.abs().max(y.abs()).max(z.abs());
    dab.radius * 1.001 + 1e-6 * size
}

/// Whether `bounds` touches `dab`'s sphere (see [`reach`]); no box touches nothing.
fn touches(bounds: Option<&Aabb>, dab: &Imprint) -> bool {
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
    let r = reach(dab);
    dx * dx + dy * dy + dz * dz <= r * r
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;
