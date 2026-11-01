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
//! with its ring ([`TileStore::cached_positions_with_ring_many`]), by the draw action's own amount
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
//! When the oldest dabs shown land ([`Preview::land`]), the store's finest surface moves under the
//! previews. A landed dab moved the vertices within its radius (each by at most its strength), and
//! a tile's interior quantization only by that tile's own writes, so it changed the positions of
//! the tiles its sphere reached in the index as it was, and the rings of their neighbours
//! ([`TileStore::stale_after`]). After the landing the index's boxes have moved by at most the
//! landed strengths, so its query with the spheres grown by them finds every tile reached; the
//! landing's tiles are those and their neighbours. Each of them that is previewed is made again:
//! the store's tile now, moved by the dabs still shown, or the store's tile alone if none moves
//! it. One that isn't may now lie within a dab still shown, the store having moved it: its box (of
//! its own positions, as the store holds them now) says, and it is previewed if so. Untouched
//! tiles among them are forgotten, to be looked at again.
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

/// A tile being moved in one call: its block, its box, and whether a dab shown moved it.
#[derive(Debug)]
struct Moving {
    /// The tile.
    key: TileKey,
    /// Its block: its preview, or the store's tile gathered with its ring.
    block: GatheredBlock,
    /// The box of every point of `block`, as it is now ([`bounds_of`]).
    bounds: Option<Aabb>,
    /// Whether a dab shown moved a point of it: it is previewed.
    moved: bool,
    /// Whether it had a preview as the call began.
    previewed: bool,
}

impl Moving {
    /// The store's `block` of tile `key`, not moved yet.
    fn of_store(key: TileKey, block: GatheredBlock) -> Self {
        let bounds = bounds_of(&block);
        Self {
            key,
            block,
            bounds,
            moved: false,
            previewed: false,
        }
    }

    /// Moves it by `dab` if its box touches the dab's sphere, its box made again if a point
    /// moved.
    fn move_by(&mut self, dab: &Imprint) {
        if touches(self.bounds.as_ref(), dab, 0.0) && move_by(&mut self.block, dab) {
            self.moved = true;
            self.bounds = bounds_of(&self.block);
        }
    }

    /// Moves it by every one of `dabs`, in order ([`Moving::move_by`]).
    fn move_by_every(&mut self, dabs: &[Imprint]) {
        for dab in dabs {
            self.move_by(dab);
        }
    }
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
    /// The tiles they reach, previewed or untouched, by the box of their points as drawn (`None`
    /// for a tile of no finite point, which no dab reaches), and whether previewed.
    reached: BTreeMap<TileKey, (Option<Aabb>, bool)>,
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
        let near: Vec<(TileKey, bool)> = self
            .reached
            .iter()
            .filter(|(_, (bounds, _))| touches(bounds.as_ref(), imprint, 0.0))
            .map(|(key, &(_, previewed))| (*key, previewed))
            .collect();
        let queried = index
            .query_sphere(centre(imprint), reach(imprint, 0.0))
            .filter(|key| !self.reached.contains_key(key));
        let mut from_store: Vec<TileKey> = queried.collect();
        // Previewed tiles take this dab alone; the others, the store's tile moved by every dab
        // shown, this one last (for an untouched one, the dabs before moved nothing there).
        let mut moving: Vec<Moving> = Vec::with_capacity(near.len() + from_store.len());
        for (key, previewed) in near {
            match store.take_preview(&key).filter(|_| previewed) {
                Some(block) => {
                    let bounds = self.reached.get(&key).and_then(|&(bounds, _)| bounds);
                    moving.push(Moving {
                        key,
                        block,
                        bounds,
                        moved: true,
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
        store.par_map_mut(old, |tile| tile.move_by(imprint));
        store.par_map_mut(new, |tile| tile.move_by_every(&all));
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
            for (key, (_, previewed)) in std::mem::take(&mut self.reached) {
                if previewed {
                    store.clear_preview(&key);
                    stats.tiles_cleared += 1;
                }
            }
            return Ok(());
        }
        let strength = landed
            .iter()
            .fold(0.0_f32, |sum, dab| add_up(sum, dab.strength.abs()));
        let reached: BTreeSet<TileKey> = landed
            .iter()
            .flat_map(|dab| index.query_sphere(centre(dab), reach(dab, strength)))
            .collect();
        let reached: Vec<TileKey> = reached.into_iter().collect();
        let changed = match store.stale_after(&reached) {
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
                Some(&(_, true)) => {
                    again.insert(key);
                }
                Some(&(_, false)) => {
                    // Untouched, and the store moved it: looked at again below.
                    self.reached.remove(&key);
                    unpreviewed.push(key);
                }
                None => unpreviewed.push(key),
            }
        }
        // The store moved these: within a dab still shown now, by their own positions' box?
        let shown: Vec<Imprint> = self.shown.iter().copied().collect();
        let within = unpreviewed.into_iter().filter(|key| {
            let positions = store.cached_positions(key);
            let bounds = positions.and_then(|points| Aabb::from_points(points.iter().copied()));
            // A tile not cached is looked at whole: gathered, and kept only if moved.
            bounds.is_none() || shown.iter().any(|dab| touches(bounds.as_ref(), dab, 0.0))
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
        let previewed = reached.into_iter().filter(|&(_, (_, previewed))| previewed);
        let mut again: BTreeSet<TileKey> = previewed.map(|(key, _)| key).collect();
        for dab in &self.shown {
            again.extend(index.query_sphere(centre(dab), reach(dab, 0.0)));
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
    /// that can't tell which tiles it changed. The next dab shown gathers afresh.
    fn clear(&mut self, store: &mut TileStore, stats: &mut PreviewStats) {
        for (key, (_, previewed)) in std::mem::take(&mut self.reached) {
            if previewed {
                store.clear_preview(&key);
                stats.tiles_cleared += 1;
            }
        }
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
            self.reached.insert(tile.key, (tile.bounds, tile.moved));
            if tile.moved {
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
            Ok(Moving::of_store(key, block))
        })
        .collect::<Result<Vec<Moving>, CacheError>>()
        .map_err(SculptError::from)
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

/// `dab`'s centre.
const fn centre(dab: &Imprint) -> Vec3 {
    let [x, y, z] = dab.center;
    Vec3::new(x, y, z)
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
