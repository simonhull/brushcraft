//! Previews: a brush's dabs shown at once on the tiles a renderer draws, ahead of the finest work
//! that makes them exact ([`Preview`]).
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
//! saves them (see the store's `preview` notes). A previewed tile's pen-down block is kept by the
//! store beside it ([`TileStore::keep_preview_pen_down`]), counted with the stroke's pen-down
//! data, until no dab shown moves it.
//!
//! # Display tiles
//!
//! Most of what a big brush reaches is drawn at display levels, far coarser than the finest:
//! moving the finest tiles there costs as much as the finest work itself. So a dab shown also
//! moves the **display tiles** a renderer draws above the finest tiles it reaches
//! ([`Drawn`]; every level for [`DrawsAll`]), each a display preview of its own (the `display`
//! module: its pen-down tile, moves, normals of the moved surface, and a bound against the finest
//! result), handed to the store ([`TileStore::set_display_preview`]). The finest previews are made
//! only of the finest tiles a renderer draws, which a big brush seen at its size has few of.
//!
//! Every finest tile a dab shown reaches by the index is **held** in the store until the dab
//! lands ([`TileStore::hold_preview`]): a renderer draws a held tile only from a preview, and
//! asks for those it wants that have none ([`Preview::make_waiting`]); the refresh leaves the
//! display tiles above alone. A display preview's tile released, its vertices are the store's
//! surface; it is drawn until the refresh replaces it. The stroke's dabs are kept, applied at once
//! ([`Preview::note_applied`]) or shown, for the display previews' normals and bounds, until the
//! stroke ends ([`Preview::end_stroke`]).
//!
//! # Which tiles
//!
//! A dab moves the points within its radius at pen-down. A finest tile holds such a point only if
//! the box of its pen-down block (of every point of the block, ring included) touches the dab's
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
use geom::{GatheredBlock, Level, TileKey, Vec3};
use store::{Aabb, CacheError, DisplayPreview, Evaluation, SpatialIndex, TileStore};
use stroke::Dab;

use crate::apply::{SculptError, imprint_of};

mod dabs;
mod display;

use dabs::DabIndex;
use display::{DisplayState, StrokeDabs};

/// The most previews [`Preview::make_waiting`] makes a call: what a renderer asks for at once can
/// be a whole refined region, and a preview takes some tens of microseconds to a millisecond.
pub const WAITING_A_CALL: usize = 8;

/// The most tiles [`Preview::make_waiting`] tries a call, counting those that get no preview.
const WAITING_TRIES: usize = 4 * WAITING_A_CALL;

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
    /// Dabs taken again by display previews made (the dabs near it): what a fresh tile costs,
    /// which follows the dabs near it, not the stroke's length.
    pub replayed: usize,
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
    /// What the call's dabs have made of it so far.
    fate: Fate,
    /// Whether it had a preview as the call began.
    previewed: bool,
}

/// What a call's dabs make of a tile being moved ([`Moving`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fate {
    /// No dab shown moved a point of it: the store's tile shows.
    Untouched,
    /// `block` is its preview as the call began, taken from the store, and no dab of the call
    /// moved it: it goes back as it was ([`TileStore::put_back_preview`]), which a renderer has
    /// already.
    AsItWas,
    /// A dab of the call moved a point of it: its preview is set.
    Moved,
    /// Its pen-down block turned out not to be its block's (never: both are gathered over the
    /// tile's region): it is let go, its preview cleared, to be gathered afresh.
    Lost,
}

impl Fate {
    /// Whether the tile is previewed after the call.
    const fn previewed(self) -> bool {
        matches!(self, Self::AsItWas | Self::Moved)
    }
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
            fate: Fate::Untouched,
            previewed: false,
        }
    }

    /// Moves it by every one of `dabs`, in order: by each whose sphere its pen-down box touches.
    fn move_by_every(&mut self, dabs: &[Imprint]) {
        for dab in dabs {
            if !touches(self.bounds.as_ref(), dab) {
                continue;
            }
            match move_by(&mut self.block, &self.pen_down, dab) {
                Some(true) => self.fate = Fate::Moved,
                Some(false) => {}
                None => {
                    self.fate = Fate::Lost;
                    return;
                }
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
    /// Whether it is previewed: the store then keeps its pen-down block for it
    /// ([`TileStore::keep_preview_pen_down`]); an untouched tile has none.
    previewed: bool,
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
    /// The finest tiles each dab shown reaches by the spatial index (its pen-down boxes), in
    /// the order of `shown`: what it holds in the store.
    reach: VecDeque<Vec<TileKey>>,
    /// The finest tiles reached, previewed or untouched.
    reached: BTreeMap<TileKey, Reached>,
    /// Every dab of the stroke, applied at once or shown, in order: what a display preview's
    /// normals and bound take.
    stroke: Vec<Imprint>,
    /// Which of `stroke`'s dabs can reach a place.
    dab_index: DabIndex,
    /// Per finest tile held, how many dabs shown reach it.
    held: BTreeMap<TileKey, usize>,
    /// The display tiles previewed in this stroke.
    displays: BTreeMap<TileKey, DisplayState>,
    /// Per finest tile a dab of the stroke reaches, the dabs' strengths summed: how far the
    /// stroke can grow the tile's quantization scales, for the display previews' bounds.
    strengths: BTreeMap<TileKey, f64>,
    /// Per dab of `stroke`, in order, the strengths of the stroke's dabs overlapping it (itself
    /// included): no place moves farther than the largest.
    overlaps: Vec<f64>,
    /// The largest of `overlaps`.
    overlap_most: f64,
    /// The tiles [`Preview::make_waiting`] tried and got no preview of (nothing moves them, or
    /// they have no record), left out until a landing changes the store under them; a new dab
    /// that moves one makes its preview itself.
    fruitless: BTreeSet<TileKey>,
}

/// Which tiles a renderer holds or wants: those a [`Preview`] makes previews of.
///
/// A tile it doesn't draw needs no preview: if a renderer wants one later while dabs still to
/// land reach it, it asks for it ([`Preview::make_waiting`]); the store says which those are
/// ([`TileStore::preview_wanted`]).
pub trait Drawn {
    /// Whether a renderer holds or wants tile `key`.
    fn draws(&self, key: &TileKey) -> bool;
}

/// Every tile is drawn: previews of every finest and display tile the dabs reach, at every
/// level (headless use, and the tests).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DrawsAll;

impl Drawn for DrawsAll {
    fn draws(&self, _key: &TileKey) -> bool {
        true
    }
}

impl<F: Fn(&TileKey) -> bool> Drawn for F {
    fn draws(&self, key: &TileKey) -> bool {
        self(key)
    }
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
        drawn: &impl Drawn,
    ) -> Result<PreviewStats, SculptError> {
        let imprints: Result<Vec<Imprint>, SculptError> = dabs
            .iter()
            .enumerate()
            .map(|(n, dab)| imprint_of(dab).map_err(|error| SculptError::Dab { dab: n, error }))
            .collect();
        let imprints = imprints?;
        let mut stats = PreviewStats::default();
        let before = self.shown.len();
        let mut shown = self.show_all(store, index, &imprints, drawn, &mut stats);
        if self.shown.len() > before {
            self.note_shown(store, index, &imprints);
            let displays = self.show_displays(store, &imprints, drawn, &mut stats);
            shown = shown.and(displays);
        }
        self.note_bytes(store);
        let closed = store.close_preview_epoch();
        shown?;
        closed?;
        Ok(stats)
    }

    /// Notes `dabs`, just shown: they join the stroke's dabs, and the finest tiles each reaches
    /// by the index are held in the store.
    fn note_shown(&mut self, store: &mut TileStore, index: &SpatialIndex, dabs: &[Imprint]) {
        for dab in dabs {
            self.push_stroke(*dab);
            let reached: Vec<TileKey> = index.query_sphere(centre(dab), reach(dab)).collect();
            for key in &reached {
                *self.strengths.entry(*key).or_insert(0.0) += f64::from(dab.strength.abs());
                let count = self.held.entry(*key).or_insert(0);
                if *count == 0 {
                    store.hold_preview(*key);
                }
                *count += 1;
            }
            self.reach.push_back(reached);
        }
    }

    /// Notes `dabs`, applied at once by [`apply_dabs`](crate::apply_dabs) with nothing shown:
    /// they join the stroke's dabs, which every display preview made later takes for its normals
    /// and bound. Those the stroke has take them as a landing would have: over the finest tiles
    /// they wrote, with their turn and bound; made again instead, each would replay the whole
    /// stroke and gather what it wrote, milliseconds in a frame the dabs were found to fit.
    /// Closes the call's preview epoch.
    ///
    /// # Errors
    ///
    /// [`SculptError::Cache`] if a tile can't be read or gathered, or the store refuses a preview.
    pub fn note_applied(
        &mut self,
        store: &mut TileStore,
        index: &SpatialIndex,
        dabs: &[Dab],
    ) -> Result<(), SculptError> {
        let imprints: Vec<Imprint> = dabs.iter().filter_map(|dab| imprint_of(dab).ok()).collect();
        let mut reached: BTreeSet<TileKey> = BTreeSet::new();
        for dab in &imprints {
            for key in index.query_sphere(centre(dab), reach(dab)) {
                *self.strengths.entry(key).or_insert(0.0) += f64::from(dab.strength.abs());
                reached.insert(key);
            }
        }
        for dab in &imprints {
            self.push_stroke(*dab);
        }
        if self.displays.is_empty() {
            return Ok(());
        }
        let mut stats = PreviewStats::default();
        let landed = self.land_applied(store, &imprints, &reached, &mut stats);
        self.note_bytes(store);
        let closed = store.close_preview_epoch();
        landed?;
        closed?;
        Ok(())
    }

    /// The display previews the stroke has take `dabs`, already in the store, which changed
    /// around the finest tiles `reached`; those they touched are handed to the store.
    ///
    /// # Errors
    ///
    /// [`SculptError::Cache`] if a tile can't be read or gathered, or the store refuses a preview.
    fn land_applied(
        &mut self,
        store: &mut TileStore,
        dabs: &[Imprint],
        reached: &BTreeSet<TileKey>,
        stats: &mut PreviewStats,
    ) -> Result<(), SculptError> {
        let queried: Vec<TileKey> = reached.iter().copied().collect();
        let changed: BTreeSet<TileKey> = store.stale_after(&queried)?.into_iter().collect();
        let above = reached.iter().flat_map(display_ancestors);
        let above: BTreeSet<TileKey> = above.collect();
        let keys = self.moved_by(store, &above, dabs, false);
        self.land_displays(store, Vec::new(), &changed, &keys, stats)
    }

    /// Tells the store what the display previews' states hold, for its budget.
    fn note_bytes(&self, store: &mut TileStore) {
        let states = self.displays.values().map(DisplayState::bytes).sum();
        store.note_preview_state_bytes(states);
    }

    /// Adds `dab` to the stroke's dabs, and its strength to the overlaps of those it overlaps.
    fn push_stroke(&mut self, dab: Imprint) {
        let strength = f64::from(dab.strength.abs());
        let mut own = strength;
        for (other, overlap) in self.stroke.iter().zip(&mut self.overlaps) {
            let apart = centre(other) - centre(&dab);
            let reach = f64::from(reach(other)) + f64::from(reach(&dab));
            if f64::from(apart.dot(apart)) < reach * reach {
                *overlap += strength;
                self.overlap_most = self.overlap_most.max(*overlap);
                own += f64::from(other.strength.abs());
            }
        }
        self.dab_index.push(&dab);
        self.stroke.push(dab);
        self.overlaps.push(own);
        self.overlap_most = self.overlap_most.max(own);
    }

    /// Twice the farthest the stroke's dabs move any place: what the moves' part of a display
    /// preview's bound is capped by (the `display` module's "The bound").
    fn moves_cap(&self) -> f64 {
        2.0 * self.overlap_most
    }

    /// How far the finest work's rounding can move a finest vertex under display tile `key`,
    /// once every dab of the stroke has landed: the largest of its finest tiles' bounds
    /// ([`display::rounding_bound`]), over the tiles a dab of the stroke reaches (the others keep
    /// their stored bits).
    ///
    /// # Errors
    ///
    /// [`SculptError::Cache`] if a correction tile can't be read.
    fn rounding(&self, store: &mut TileStore, key: &TileKey) -> Result<f64, SculptError> {
        let Some(finest) = store.display_finest() else {
            return Ok(0.0);
        };
        let mut most = 0.0_f64;
        for tile in display::finest_under(key, finest) {
            if let Some(&strength) = self.strengths.get(&tile) {
                let scales = store.correction(tile)?.scales();
                most = most.max(display::rounding_bound(scales, strength));
            }
        }
        Ok(most)
    }

    /// Ends the stroke: every dab of it has landed (or was dropped). The display previews are
    /// handed to the store released, to be drawn until the refresh replaces them; every hold is
    /// released; the stroke's dabs are forgotten. Closes the call's preview epoch.
    ///
    /// # Errors
    ///
    /// [`SculptError::Cache`] if the store refuses a preview (never, for tiles it made) or the
    /// epoch can't close.
    pub fn end_stroke(&mut self, store: &mut TileStore) -> Result<(), SculptError> {
        let mut stats = PreviewStats::default();
        let landed = self.shown.len();
        if landed > 0 {
            // Dropped dabs: nothing moves under the previews any more.
            self.shown.clear();
            self.clear(store, &mut stats);
        }
        for (key, _) in std::mem::take(&mut self.held) {
            store.release_preview(&key);
        }
        self.reach.clear();
        self.fruitless.clear();
        let mut published = Ok(());
        let all: Vec<TileKey> = self.displays.keys().copied().collect();
        let forgotten = self.forget_unchanged(store, all);
        for mut state in std::mem::take(&mut self.displays).into_values() {
            if state.published_held == Some(true) {
                let key = *state.key();
                let cap = self.moves_cap();
                let set = self.rounding(store, &key).and_then(|rounding| {
                    Ok(store.set_display_preview(key, state.publish(false, rounding, cap))?)
                });
                published = published.and(set);
            }
        }
        let forgotten = forgotten.map(|_| ());
        published = published.and(forgotten);
        self.stroke.clear();
        self.dab_index.clear();
        self.overlaps.clear();
        self.overlap_most = 0.0;
        self.strengths.clear();
        self.note_bytes(store);
        let closed = store.close_preview_epoch();
        published?;
        closed?;
        Ok(())
    }

    /// Makes the previews of `keys`, tiles a renderer waits for: held with no preview made
    /// ([`TileStore::preview_wanted`]), at most [`WAITING_A_CALL`] a call (the rest wait for the
    /// next); the others are left alone. Closes the call's preview epoch.
    ///
    /// # Errors
    ///
    /// [`SculptError::Cache`] if a tile can't be read or gathered.
    pub fn make_waiting(
        &mut self,
        store: &mut TileStore,
        keys: &[TileKey],
    ) -> Result<PreviewStats, SculptError> {
        let mut stats = PreviewStats::default();
        let mut made = Ok(());
        let mut left = WAITING_A_CALL;
        let mut attempts = WAITING_TRIES;
        let mut wanted = keys
            .iter()
            .filter(|key| store.preview_wanted(key) && !self.fruitless.contains(key))
            .copied()
            .collect::<Vec<TileKey>>()
            .into_iter();
        // Batch after batch until the call's previews are made: a tile that gets none (nothing
        // moves it, it has no record) costs its one try and is left out for a while.
        while left > 0 && attempts > 0 {
            let batch: Vec<TileKey> = wanted.by_ref().take(left.min(attempts)).collect();
            if batch.is_empty() {
                break;
            }
            let tried = self.make_batch(store, &batch, &mut stats);
            made = made.and(tried);
            let (done, none): (Vec<TileKey>, Vec<TileKey>) = batch
                .into_iter()
                .partition(|key| !store.preview_wanted(key));
            attempts -= done.len() + none.len();
            left -= done.len();
            self.fruitless.extend(none);
        }
        self.note_bytes(store);
        let closed = store.close_preview_epoch();
        made?;
        closed?;
        Ok(stats)
    }

    /// Tries to make the previews of `keys`: a display tile with a state has it handed to the
    /// store again (the refresh dropped it while it wasn't held), one without is made, and a
    /// finest tile is gathered.
    ///
    /// # Errors
    ///
    /// [`SculptError::Cache`] if a tile can't be read or gathered.
    fn make_batch(
        &mut self,
        store: &mut TileStore,
        keys: &[TileKey],
        stats: &mut PreviewStats,
    ) -> Result<(), SculptError> {
        let (display, finest): (Vec<TileKey>, Vec<TileKey>) = keys
            .iter()
            .partition(|key| key.channel() == geom::Channel::DISPLAY);
        let (known, fresh): (Vec<TileKey>, Vec<TileKey>) = display
            .into_iter()
            .partition(|key| self.displays.contains_key(key));
        let handed = self.publish(store, &known, stats);
        let displays = self.make_displays(store, &fresh, stats);
        let again: BTreeSet<TileKey> = finest.into_iter().collect();
        let remade = self.remake(store, &again, stats);
        handed.and(displays).and(remade)
    }

    /// The display tiles above the finest tiles `dabs` reach that `drawn` draws: those a
    /// display preview moves for them.
    fn display_tiles(&self, dabs: usize, drawn: &impl Drawn) -> BTreeSet<TileKey> {
        let first = self.reach.len().saturating_sub(dabs);
        let reached = self.reach.iter().skip(first).flatten();
        let mut tiles = BTreeSet::new();
        for key in reached {
            tiles.extend(display_ancestors(key).filter(|above| drawn.draws(above)));
        }
        tiles
    }

    /// Moves the display previews by `new`, just shown and noted: those the stroke has take
    /// them, those `drawn` draws that it hasn't are made. Hands every one that changed to the
    /// store.
    ///
    /// # Errors
    ///
    /// [`SculptError::Cache`] if a tile can't be read or gathered.
    fn show_displays(
        &mut self,
        store: &mut TileStore,
        new: &[Imprint],
        drawn: &impl Drawn,
        stats: &mut PreviewStats,
    ) -> Result<(), SculptError> {
        // Every display preview the stroke has that the dabs reach takes them, drawn now or not
        // (one drawn later must hold them); of the others, those drawn are made.
        let reached = self.display_tiles(new.len(), &DrawsAll);
        let (known, fresh): (BTreeSet<TileKey>, Vec<TileKey>) = (
            reached
                .iter()
                .filter(|key| self.displays.contains_key(key))
                .copied()
                .collect(),
            reached
                .iter()
                .filter(|key| !self.displays.contains_key(key) && drawn.draws(key))
                .copied()
                .collect(),
        );
        let changed = self.moved_by(store, &known, new, true);
        self.publish(store, &changed, stats)?;
        self.make_displays(store, &fresh, stats)
    }

    /// The display previews of `keys` the stroke has take `dabs`, as still to land if `drawn`;
    /// returns those they touched. Taken out to work on: the stroke may have thousands, a dab
    /// reaches a few.
    fn moved_by(
        &mut self,
        store: &TileStore,
        keys: &BTreeSet<TileKey>,
        dabs: &[Imprint],
        drawn: bool,
    ) -> Vec<TileKey> {
        let mut moving = self.take_states(keys);
        let moved: Vec<bool> = store.par_map_mut(&mut moving, |state| {
            dabs.iter()
                .fold(false, |any, dab| state.take(dab, drawn) | any)
        });
        let touched = moving.iter().zip(moved).filter(|&(_, moved)| moved);
        let touched: Vec<TileKey> = touched.map(|(state, _)| *state.key()).collect();
        self.put_states(moving);
        touched
    }

    /// The states of `keys` the stroke has, out of `displays` ([`Preview::put_states`] them back).
    fn take_states<'a>(
        &mut self,
        keys: impl IntoIterator<Item = &'a TileKey>,
    ) -> Vec<DisplayState> {
        let taken = keys.into_iter().filter_map(|key| self.displays.remove(key));
        taken.collect()
    }

    /// Puts `states` back.
    fn put_states(&mut self, states: Vec<DisplayState>) {
        for state in states {
            self.displays.insert(*state.key(), state);
        }
    }

    /// Makes display previews of `keys` that the stroke has none of, over the store as it is,
    /// and hands them to the store.
    ///
    /// # Errors
    ///
    /// [`SculptError::Cache`] if a tile can't be read or gathered.
    fn make_displays(
        &mut self,
        store: &mut TileStore,
        keys: &[TileKey],
        stats: &mut PreviewStats,
    ) -> Result<(), SculptError> {
        let fresh: Vec<TileKey> = keys
            .iter()
            .filter(|key| !self.displays.contains_key(key))
            .copied()
            .collect();
        let dabs = StrokeDabs::of(&self.stroke, &self.dab_index, self.shown.len());
        let built = DisplayState::make_many(store, &fresh, dabs)?;
        let mut made = Vec::with_capacity(built.len());
        for state in built {
            made.push(*state.key());
            stats.replayed += state.replayed();
            self.displays.insert(*state.key(), state);
            stats.gathered += 1;
        }
        self.publish(store, &made, stats)
    }

    /// Hands the display previews of `keys` to the store, each held as the store says now.
    ///
    /// # Errors
    ///
    /// [`SculptError::Cache`] if the store refuses one (never, for tiles it made).
    fn publish(
        &mut self,
        store: &mut TileStore,
        keys: &[TileKey],
        stats: &mut PreviewStats,
    ) -> Result<(), SculptError> {
        let mut rounding: BTreeMap<TileKey, f64> = BTreeMap::new();
        for key in keys {
            if self.displays.contains_key(key) {
                rounding.insert(*key, self.rounding(store, key)?);
            }
        }
        let cap = self.moves_cap();
        let mut taken = self.take_states(rounding.keys());
        let mut publishing: Vec<(bool, f64, &mut DisplayState)> = taken
            .iter_mut()
            .filter_map(|state| {
                let rounds = rounding.get(state.key())?;
                Some((store.preview_held(state.key()), *rounds, state))
            })
            .collect();
        // Made on the workers, handed over in order.
        let made: Vec<(TileKey, bool, DisplayPreview)> =
            store.par_map_mut(&mut publishing, |(held, rounds, state)| {
                state.published_held = Some(*held);
                (*state.key(), *held, state.publish(*held, *rounds, cap))
            });
        self.put_states(taken);
        for (key, _, preview) in made {
            store.set_display_preview(key, preview)?;
            stats.tiles_set += 1;
        }
        Ok(())
    }

    /// After a landing of the dabs whose reach was `landed`: releases the finest tiles no dab
    /// shown reaches any more, rebases the display previews over the finest tiles `changed` (the
    /// store moved them), and hands the store those that changed or were released, with
    /// `touched`.
    ///
    /// # Errors
    ///
    /// [`SculptError::Cache`] if a finest tile can't be gathered.
    fn land_displays(
        &mut self,
        store: &mut TileStore,
        landed: Vec<Vec<TileKey>>,
        changed: &BTreeSet<TileKey>,
        touched: &[TileKey],
        stats: &mut PreviewStats,
    ) -> Result<(), SculptError> {
        // The display tiles above a tile released: what may no longer be held.
        let mut released: BTreeSet<TileKey> = BTreeSet::new();
        for key in landed.into_iter().flatten() {
            if let Some(count) = self.held.get_mut(&key) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    self.held.remove(&key);
                    store.release_preview(&key);
                    released.extend(display_ancestors(&key));
                }
            }
        }
        let Some(finest) = store.display_finest() else {
            return Ok(());
        };
        // The tiles the stroke wrote among those the landing can have changed (a tile it reached
        // and moved nothing of keeps its bits, which the records above it hold already), and the
        // previews over them: a landing costs what it changed, however many the stroke has.
        let written = changed
            .iter()
            .filter(|key| store.pen_down_shape(key).is_some());
        let mut under: BTreeSet<TileKey> = BTreeSet::new();
        let mut above: BTreeSet<TileKey> = BTreeSet::new();
        for key in written {
            let mut over = display_ancestors(key).filter(|up| self.displays.contains_key(up));
            if let Some(first) = over.next() {
                under.insert(*key);
                above.insert(first);
                above.extend(over);
            }
        }
        let mut blocks = display::Blocks::default();
        blocks.add_now(store, &under)?;
        let mut taken = self.take_states(&above);
        let dabs = StrokeDabs::of(&self.stroke, &self.dab_index, self.shown.len());
        let rebased: Vec<bool> = store.par_map_mut(&mut taken, |state| {
            state.rebase(finest, &blocks, &under, dabs)
        });
        let moved = taken.iter().zip(rebased).filter(|&(_, rebased)| rebased);
        let moved: Vec<TileKey> = moved.map(|(state, _)| *state.key()).collect();
        self.put_states(taken);
        let unheld = released.iter().filter(|key| {
            self.displays
                .get(key)
                .is_some_and(|state| state.published_held != Some(store.preview_held(key)))
        });
        let again: Vec<TileKey> = moved.iter().chain(unheld).copied().collect();
        let again = self.forget_unchanged(store, again)?;
        let wanted: BTreeSet<TileKey> = again.into_iter().chain(touched.iter().copied()).collect();
        let wanted: Vec<TileKey> = wanted.into_iter().collect();
        self.publish(store, &wanted, stats)
    }

    /// Of `keys`, the display previews to hand the store again: a released one over a tile whose
    /// record stands ([`TileStore::display_record_stands`]: rebuilt since the stroke's last write
    /// there, or never moved) is dropped, its state with it, and the record is drawn. No refresh
    /// would replace it; one whose record doesn't stand is what is drawn until the refresh's.
    ///
    /// # Errors
    ///
    /// [`SculptError::Cache`] if a record can't be read.
    fn forget_unchanged(
        &mut self,
        store: &mut TileStore,
        keys: Vec<TileKey>,
    ) -> Result<Vec<TileKey>, SculptError> {
        let mut kept = Vec::with_capacity(keys.len());
        for key in keys {
            let released = !store.preview_held(&key) && store.display_changes(&key).is_none();
            if released && store.display_record_stands(&key)? {
                self.displays.remove(&key);
                store.clear_display_preview(&key);
            } else {
                kept.push(key);
            }
        }
        Ok(kept)
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
        drawn: &impl Drawn,
        stats: &mut PreviewStats,
    ) -> Result<(), SculptError> {
        if new.is_empty() {
            return Ok(());
        }
        let near: Vec<(TileKey, bool)> = self
            .reached
            .iter()
            .filter(|(_, reached)| new.iter().any(|dab| touches(reached.bounds.as_ref(), dab)))
            .map(|(key, reached)| (*key, reached.previewed))
            .collect();
        let queried: BTreeSet<TileKey> = new
            .iter()
            .flat_map(|dab| index.query_sphere(centre(dab), reach(dab)))
            .filter(|key| !self.reached.contains_key(key) && drawn.draws(key))
            .collect();
        // Previewed tiles take the new dabs alone; the others, the store's tile moved by every
        // dab shown (for an untouched one, the dabs before moved nothing there). A previewed tile
        // whose pen-down block the store no longer keeps is gathered afresh too.
        let mut from_store: Vec<TileKey> = queried.into_iter().collect();
        from_store.extend(near.iter().filter(|(_, p)| !p).map(|&(key, _)| key));
        let mut moving: Vec<Moving> = Vec::new();
        let previewed = near.iter().filter(|&&(_, previewed)| previewed);
        for &(key, _) in previewed {
            let bounds = self.reached.get(&key).and_then(|reached| reached.bounds);
            let pen_down = store.take_preview_pen_down(&key);
            match (pen_down, store.take_preview(&key)) {
                (Some(pen_down), Some(block)) => moving.push(Moving {
                    key,
                    block,
                    pen_down,
                    bounds,
                    fate: Fate::AsItWas,
                    previewed: true,
                }),
                _ => from_store.push(key),
            }
        }
        from_store.sort_unstable();
        let gathered = gather(store, &from_store)?;
        stats.gathered += gathered.len();
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
        drawn: &impl Drawn,
    ) -> Result<PreviewStats, SculptError> {
        let count = count.min(self.shown.len());
        self.fruitless.clear();
        let landed: Vec<Imprint> = self.shown.drain(..count).collect();
        let reach: Vec<Vec<TileKey>> = self.reach.drain(..count.min(self.reach.len())).collect();
        let mut stats = PreviewStats {
            dabs: landed.len(),
            ..PreviewStats::default()
        };
        if landed.is_empty() {
            return Ok(stats);
        }
        // The landing changed the positions, or rings, of the tiles the dabs reach in the index
        // (their pen-down boxes, which the landing didn't move) and of their neighbours.
        let queried: BTreeSet<TileKey> = reach.iter().flatten().copied().collect();
        let queried: Vec<TileKey> = queried.into_iter().collect();
        let made = match store.stale_after(&queried) {
            Ok(changed) => {
                let set: BTreeSet<TileKey> = changed.iter().copied().collect();
                let finest = self.land_tiles(store, index, changed, drawn, &mut stats);
                let displays = self.land_displays(store, reach, &set, &[], &mut stats);
                finest.and(displays)
            }
            Err(error) => {
                self.clear(store, &mut stats);
                let _released = self.land_displays(store, reach, &BTreeSet::new(), &[], &mut stats);
                Err(error.into())
            }
        };
        self.note_bytes(store);
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
        changed: Vec<TileKey>,
        drawn: &impl Drawn,
        stats: &mut PreviewStats,
    ) -> Result<(), SculptError> {
        if self.shown.is_empty() {
            // Nothing shown any more: every preview goes back to the store's tile.
            self.clear(store, stats);
            return Ok(());
        }
        let mut again: BTreeSet<TileKey> = BTreeSet::new();
        let mut unpreviewed: Vec<TileKey> = Vec::new();
        for key in changed {
            match self.reached.get(&key) {
                Some(reached) if reached.previewed => {
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
        let within = unpreviewed
            .into_iter()
            .filter(|key| drawn.draws(key))
            .filter(|key| {
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
        let previewed = reached.into_iter().filter(|(_, reached)| reached.previewed);
        let mut again: BTreeSet<TileKey> = previewed.map(|(key, _)| key).collect();
        for dab in &self.shown {
            again.extend(index.query_sphere(centre(dab), reach(dab)));
        }
        let mut made = self.remake(store, &again, &mut stats);
        // The display previews, made again over the store as the failure left it.
        let displays: Vec<TileKey> = std::mem::take(&mut self.displays).into_keys().collect();
        let displays_made = self.make_displays(store, &displays, &mut stats);
        made = made.and(displays_made);
        self.note_bytes(store);
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
        // The store's tiles are gathered afresh; the old previews are dropped, their pen-down
        // blocks with them.
        let previewed: BTreeSet<TileKey> = keys
            .iter()
            .filter(|key| store.take_preview(key).is_some())
            .copied()
            .collect();
        for key in &keys {
            store.drop_preview_pen_down(key);
        }
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
            if reached.previewed {
                store.clear_preview(&key);
                store.drop_preview_pen_down(&key);
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
            if tile.fate == Fate::Lost {
                self.reached.remove(&tile.key);
                store.clear_preview(&tile.key);
                store.drop_preview_pen_down(&tile.key);
                continue;
            }
            self.reached.insert(
                tile.key,
                Reached {
                    bounds: tile.bounds,
                    previewed: tile.fate.previewed(),
                },
            );
            match tile.fate {
                Fate::AsItWas => {
                    store.keep_preview_pen_down(tile.key, tile.pen_down);
                    store.put_back_preview(tile.key, tile.block)?;
                }
                Fate::Moved => {
                    store.keep_preview_pen_down(tile.key, tile.pen_down);
                    store.set_preview(tile.key, tile.block)?;
                    stats.tiles_set += 1;
                }
                Fate::Untouched | Fate::Lost => {
                    // Moved by no dab shown: the store's tile shows (a preview taken is cleared).
                    store.drop_preview_pen_down(&tile.key);
                    store.clear_preview(&tile.key);
                    if tile.previewed {
                        stats.tiles_cleared += 1;
                    }
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

/// Moves every point of `block` within `dab`'s radius at pen-down by the draw action's move.
///
/// A point is weighed at its counterpart in `pen_down`, and moves by its displacement along the
/// dab's normal ([`brush::draw_move`]). Returns whether it moved any, or `None` if `pen_down`
/// isn't gathered over `block`'s region and nothing moved.
///
/// A point with no displacement (on the rim, or of none at the falloff's edge) keeps its bits:
/// adding a zero move would turn a `-0.0` into `0.0`.
fn move_by(block: &mut GatheredBlock, pen_down: &GatheredBlock, dab: &Imprint) -> Option<bool> {
    let mut moved = false;
    let matched = block.displace_along(pen_down, |p, at| {
        match draw_imprint_vertex(&Smoothstep, dab, [at.x, at.y, at.z]) {
            Some(w) if w != 0.0 => {
                moved = true;
                let [x, y, z] = draw_move(dab, w);
                p + Vec3::new(x, y, z)
            }
            _ => p,
        }
    });
    matched.then_some(moved)
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

/// The display tiles above finest tile `key`: one at each level from 0 up to the one below its
/// own.
fn display_ancestors(key: &TileKey) -> impl Iterator<Item = TileKey> + use<> {
    let (face, finest, tx, ty) = (key.face(), key.level(), key.tx(), key.ty());
    let per_side = finest.tiles_per_side();
    (0..finest.get()).filter_map(move |l| {
        let level = Level::new(l).ok()?;
        let scale = (per_side / level.tiles_per_side().max(1)).max(1);
        let (x, y) = (u32::from(tx) / scale, u32::from(ty) / scale);
        let (x, y) = (u16::try_from(x).ok()?, u16::try_from(y).ok()?);
        TileKey::display(face, level, x, y).ok()
    })
}

/// The box of the finite points of `points`; `None` if there is none.
fn bounds_of_points(points: &[Vec3]) -> Option<Aabb> {
    let finite = points
        .iter()
        .filter(|p| p.x.is_finite() && p.y.is_finite() && p.z.is_finite());
    Aabb::from_points(finite.copied())
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
