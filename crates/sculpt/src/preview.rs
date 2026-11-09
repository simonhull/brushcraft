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
//! # Strokes over strokes still landing
//!
//! The dabs shown may be of several strokes ([`OpenStrokes`]), in the order they land: the oldest
//! stroke's op is begun, and a stroke begun while the ones before it still land has its op
//! reserved, its pen-down shapes the **futures** of the tiles they change, each the tile as they
//! will leave it, bit for bit (the store's "Ops reserved ahead"). A later stroke's dabs are shown
//! once every earlier stroke's are ([`Preview::show_over`]): the dabs of one stroke are never
//! shown after a later one's.
//!
//! Each previewed tile is **shown by** one stroke, the newest whose dabs shown move it, and is
//! that stroke's preview alone, made over its pen-down surface, its pen-down block kept in that
//! stroke's own set:
//!
//! - the oldest stroke's, as above: the store's tile moved by its dabs shown;
//! - a later stroke's, nothing of it landed: its pen-down block, which holds every earlier stroke's
//!   dabs exactly, moved by its own dabs shown, each point weighed at that same block. Its pen-down
//!   block is read in its own set ([`TileStore::read_pen_down_of`]), every future it reads
//!   installed first ([`OpenStrokes::ready`]): a tile the earlier strokes change is read at its
//!   future, any other as stored, which it stays. So it is the oldest stroke's preview as it will
//!   be once those strokes have landed, and the landing of theirs never moves it.
//!
//! A tile whose points a later stroke's dabs don't move stays the earlier stroke's preview (or
//! the store's tile); one they move is taken over. Which tiles a later stroke's dabs reach is
//! decided as they are shown, the spatial index refitted over its futures by then: the tiles each
//! stroke's dabs find there are noted, and a preview made again later is made of the newest
//! stroke that found it and moves it. When the oldest stroke ends, the next's op begins, its set
//! the current one: its pen-down block of every tile is then the store's tile, and its previews
//! carry on as the oldest stroke's.
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
//! A display preview is one stroke's too: the newest whose dabs reach a finest tile under it.
//! A later stroke's is made over its pen-down tile, every tile it replaces at its future (the
//! futures installed first: every tile the earlier strokes change under the display tile and
//! around those), and drawn from that pen-down tile moved by its own dabs. Those futures can be
//! much of an earlier stroke's backlog, so they come a call's worth at a time
//! ([`OpenStrokes::ready`]): until a later stroke's display preview can be made, the earlier
//! stroke's stays (never let go before), or one is made for the oldest stroke, over the store, and
//! the later stroke's dabs reaching it are **pending**: not drawn, their whole moves in its bound.
//! It is made the later stroke's once its futures are in ([`Preview::take_over_displays`], off
//! the frame's path).
//!
//! Every finest tile a dab shown reaches by the index is **held** in the store until the dab
//! lands ([`TileStore::hold_preview`]): a renderer draws a held tile only from a preview, and
//! asks for those it wants that have none ([`Preview::make_waiting`]); the refresh leaves the
//! display tiles above alone. A display preview's tile released, its vertices are the store's
//! surface; it is drawn until the refresh replaces it. The strokes' dabs are kept, applied at once
//! ([`Preview::note_applied`]) or shown, for the display previews' normals and bounds, until the
//! last stroke ends ([`Preview::end_stroke`]).
//!
//! # Seams
//!
//! Previews of one stroke meet bit for bit, each move a function of a vertex's bits. Previews of
//! two strokes would not: a later stroke's holds the earlier one's work rounded as it will land,
//! the earlier one's its moves unrounded, a difference of the stored encoding at the vertices
//! they share (and a later stroke's pending dabs, at a display preview still the earlier one's).
//! So every preview holding a vertex draws it as one does ([`Preview::stitch`]): the newest
//! stroke's, a finest preview's before a display one's of the same stroke, at every finest
//! preview's every point (its ring and its base-vertex rings included, which the normals read)
//! and every display preview's border vertices, at every level; display previews at one level
//! draw that one's normal there too. Each preview stays its own stroke's otherwise, and nothing
//! waits for it: no future is read but those a stroke's own previews read.
//!
//! A preview drawing another's bits there differs from what lands by what that one does: a later
//! stroke's finest preview by the stored encoding (that of the earlier strokes' work, which its
//! pen-down holds rounded, and its own), within the finest previews' bound; a display preview's
//! bound measures the vertices it draws (the `display` module's "The bound"). One whose vertex
//! another no longer wins (the later stroke's preview gone) is made again.
//!
//! # Which tiles
//!
//! A dab moves the points within its radius at pen-down. A finest tile holds such a point only if
//! the box of its pen-down block (of every point of the block, ring included) touches the dab's
//! sphere; a tile not reached yet is found by its box in the spatial index, which holds its
//! pen-down positions, and every open stroke's (see [`SpatialIndex::refit`]). So a dab shown moves
//! the tiles its stroke has reached whose pen-down box touches it, and gathers those of the
//! index's query it reaches that its stroke hasn't. Neither box moves with the dabs.
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
//! the landing's tiles. Each of them that the oldest stroke's preview shows is made again: the
//! store's tile now, moved by the dabs still shown, or the store's tile alone if none moves it. A
//! later stroke's preview stays: it was made over the futures, which the landing reaches. One that
//! isn't previewed may lie within a dab still shown, the store having moved it: its box in the
//! index says (or the later stroke that found it), and it is looked at again if so. Untouched
//! tiles among them are forgotten, to be looked at again.
//!
//! Once no dab is shown, no tile is previewed.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

use brush::{Imprint, Smoothstep, draw_imprint_vertex, draw_move};
use geom::{GatheredBlock, Level, TileKey, Vec3};
use store::{
    Aabb, CacheError, DisplayPreview, Evaluation, OpId, PenDownReading, SpatialIndex, TileStore,
};
use stroke::Dab;

use crate::apply::{SculptError, imprint_of};

mod dabs;
mod display;
mod seams;
mod strokes;

use dabs::DabIndex;
pub use display::rounding_bound;
use display::{Base, DisplayState, StrokeDabs};
use seams::Place;
use strokes::Lone;
pub use strokes::OpenStrokes;

/// The most previews [`Preview::make_waiting`] makes a call: what a renderer asks for at once can
/// be a whole refined region, and a preview takes some tens of microseconds to a millisecond.
pub const WAITING_A_CALL: usize = 8;

/// What a set's node keeps for each key in [`Readers::by_display`], beside the key: an estimate.
const READER_BYTES: usize = 16;

/// The most tiles [`Preview::make_waiting`] tries a call, counting those that get no preview.
const WAITING_TRIES: usize = 4 * WAITING_A_CALL;

/// An open stroke as the previews tell them apart: its op, or `None` for a stroke shown alone.
///
/// `None` is the store's current stroke, as [`Preview::show`] and the other calls taking a store
/// show it. The ops of the strokes under way grow with them, so a later stroke's is greater.
type Tag = Option<OpId>;

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
    /// Dabs taken again by the display previews made: what a fresh tile costs, which follows the
    /// dabs near it, not the stroke's length.
    pub replayed: usize,
}

/// A tile being moved in one call: its block, its pen-down block and box, and whether a dab
/// shown moved it.
#[derive(Debug)]
struct Moving {
    /// The tile.
    key: TileKey,
    /// Its block: its preview, or its base gathered with its ring (the store's tile, or for a
    /// reserved stroke its pen-down block).
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
    /// No dab shown moved a point of it: the store's tile shows (or an earlier stroke's preview).
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

impl Moving {
    /// The `block` of tile `key` (its base), not moved yet, with its `pen_down` block.
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

/// What a tile the shown dabs reach shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Showing {
    /// The store's tile: no dab shown moves it.
    Untouched,
    /// Stroke `0`'s preview: the newest stroke whose dabs shown move it, which keeps its pen-down
    /// block in its own set ([`TileStore::keep_preview_pen_down`]).
    Of(Tag),
}

/// A tile the shown dabs reach, as a [`Preview`] keeps it between calls.
#[derive(Debug, Clone, Copy)]
struct Reached {
    /// The newest stroke whose dabs it was looked at for (gathered in that stroke's set).
    looked: Tag,
    /// The box of its pen-down block's points in `looked`'s set; `None` for a block of no finite
    /// point, which no dab reaches.
    bounds: Option<Aabb>,
    /// What it shows.
    showing: Showing,
}

/// One open stroke's dabs among those shown: which stroke, and how many, in landing order.
#[derive(Debug, Clone, Copy)]
struct Segment {
    /// The stroke.
    op: Tag,
    /// How many of its dabs are shown and not landed.
    dabs: usize,
}

/// One open stroke's finest tiles under display tiles that read a tile it changed.
///
/// For a reserved stroke, a tile it has the future of. Those a display preview made there takes
/// at pen-down, kept up as the stroke changes tiles.
#[derive(Debug, Default)]
struct Readers {
    /// Per display tile, the finest tiles under it whose blocks read a tile the stroke changed.
    by_display: BTreeMap<TileKey, BTreeSet<TileKey>>,
    /// Tiles the stroke changed that `by_display` hasn't taken in yet.
    unread: Vec<TileKey>,
}

/// A brush's previews: the dabs shown and not yet landed, of one stroke or of several in a chain,
/// and the tiles they reach (see the module notes).
///
/// The caller shows the dabs a step can't apply ([`Preview::show`], or [`Preview::show_over`]
/// with strokes over strokes still landing) and, as it applies the oldest of them through
/// [`apply_dabs`](crate::apply_dabs) in order, lands them ([`Preview::land`]) after each call.
/// Dabs dropped unapplied (a stroke ended by an error) are landed too: nothing then moves under
/// the previews, and they are made again without them; a store changed otherwise under them (the
/// failing dab's writes) has them made again ([`Preview::rebase`]).
#[derive(Debug, Default)]
pub struct Preview {
    /// The dabs shown and not landed, oldest first.
    shown: VecDeque<Imprint>,
    /// Whose `shown` are, stroke by stroke, oldest first.
    segments: VecDeque<Segment>,
    /// The finest tiles each dab shown reaches by the spatial index (its pen-down boxes), in
    /// the order of `shown`: what it holds in the store.
    reach: VecDeque<Vec<TileKey>>,
    /// Per stroke with dabs shown, the finest tiles its dabs found in the index as they were
    /// shown (its futures there installed): those it may show.
    found: BTreeMap<Tag, BTreeSet<TileKey>>,
    /// The finest tiles reached, previewed or untouched.
    reached: BTreeMap<TileKey, Reached>,
    /// Every dab of the strokes, applied at once or shown, in order: what a display preview's
    /// normals and bound take.
    stroke: Vec<Imprint>,
    /// Each stroke's first dab in `stroke`, in order.
    stroke_ops: Vec<(Tag, usize)>,
    /// Which of `stroke`'s dabs can reach a place.
    dab_index: DabIndex,
    /// Per finest tile held, how many dabs shown reach it.
    held: BTreeMap<TileKey, usize>,
    /// The display tiles previewed.
    displays: BTreeMap<TileKey, DisplayState>,
    /// Per finest tile a dab reaches, the dabs' strengths summed, every stroke's: how far the
    /// strokes can grow the tile's quantization scales, for the display previews' bounds.
    strengths: BTreeMap<TileKey, f64>,
    /// Per dab of `stroke`, in order, the strengths of the dabs overlapping it (itself
    /// included): no place moves farther than the largest.
    overlaps: Vec<f64>,
    /// The largest of `overlaps`.
    overlap_most: f64,
    /// Per stroke, the finest tiles under each display tile whose blocks read a tile it changed.
    readers: BTreeMap<Tag, Readers>,
    /// The tiles [`Preview::make_waiting`] tried and got no preview of (nothing moves them, or
    /// they have no record), left out until a landing changes the store under them; a new dab
    /// that moves one makes its preview itself.
    fruitless: BTreeSet<TileKey>,
    /// Per display tile previewed, its border vertices' places: where it meets the tiles around
    /// it ([`seams::border_places`]).
    border_places: BTreeMap<TileKey, Vec<(usize, Place)>>,
    /// The finest tiles the call under way made, kept or cleared: where previews of different
    /// strokes may have come to meet or part ([`Preview::stitch`]).
    touched: Vec<TileKey>,
    /// The display tiles the call under way handed over, or let go of: the same for them.
    display_seeds: Vec<TileKey>,
    /// Per finest tile a call may stitch, its block's places ([`seams::Shared`]); `None` for
    /// a region that can't be gathered.
    finest_places: BTreeMap<TileKey, Option<seams::Shared>>,
    /// Per finest preview drawing a later stroke's bits at some vertices, their places.
    finest_subs: BTreeMap<TileKey, BTreeSet<Place>>,
    /// Per display preview, the vertices it draws as a later stroke's preview does.
    display_subs: BTreeMap<TileKey, Vec<Seam>>,
}

/// A display preview being handed to the store ([`Preview::publish`]): whether it is held, its
/// rounding's bound, the vertices it draws as a later stroke's (see "Seams" in the module notes),
/// and its state.
type Publishing<'a> = (bool, f64, &'a [Seam], &'a mut DisplayState);

/// A vertex a display preview draws as other previews do (see "Seams" in the module notes): its
/// index, and the position and the normal it draws there, where those are another's.
type Seam = (usize, Option<Vec3>, Option<Vec3>);

/// The preview whose bits a place shows ([`Preview::choose`]).
#[derive(Debug, Clone, Copy)]
struct Chosen {
    /// The stroke whose preview it is.
    owner: Tag,
    /// The preview: a finest tile's or a display tile's.
    holder: TileKey,
    /// What it draws there: its position, or for a display level's normals, its normal.
    at: Vec3,
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

/// Stroke `op`'s reading of `strokes`: the store, its pen-down reads at the stroke's shapes, and
/// the spatial index.
///
/// # Errors
///
/// As [`OpenStrokes::read`].
fn open<S: OpenStrokes>(
    strokes: &mut S,
    op: Tag,
) -> Result<(PenDownReading<'_>, &SpatialIndex), CacheError> {
    match op {
        None => Ok(strokes.current()),
        Some(op) => strokes.read(op),
    }
}

/// What stroke `op`'s previews are drawn from: its pen-down blocks while its op is reserved.
fn base_of<S: OpenStrokes>(strokes: &S, op: Tag) -> Base {
    if op.is_some_and(|op| strokes.reserved(op)) {
        Base::PenDown
    } else {
        Base::Store
    }
}

/// Whether every future stroke `op`'s previews read near `dabs` and under display tiles
/// `displays` is installed ([`OpenStrokes::ready`]); always for a stroke whose op is begun.
fn ready<S: OpenStrokes>(strokes: &mut S, op: Tag, dabs: &[Dab], displays: &[TileKey]) -> bool {
    match op {
        Some(op) if strokes.reserved(op) => strokes.ready(op, dabs, displays),
        _ => true,
    }
}

/// Drops the pen-down blocks of `blocks`, each previewed tile's in its stroke's set: the stroke no
/// longer shows it. A stroke whose set is gone has none to drop.
fn drop_blocks<S: OpenStrokes>(strokes: &mut S, blocks: BTreeMap<Tag, Vec<TileKey>>) {
    for (op, keys) in blocks {
        if let Ok((mut store, _)) = open(strokes, op) {
            for key in &keys {
                store.drop_preview_pen_down(key);
            }
        }
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

    /// Shows `dabs` of the store's current stroke, after those shown already: moves the previews
    /// of the finest tiles they reach (gathering those reached first from the store), sets them
    /// in the store and closes their preview epoch.
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
        self.show_tagged(&mut Lone { store, index }, None, dabs, drawn)
    }

    /// Shows `dabs` of stroke `op`, one of `strokes`, after those shown already: as
    /// [`Preview::show`], over the stroke's own pen-down surface (see "Strokes over strokes still
    /// landing" in the module notes).
    ///
    /// Dabs of a stroke older than one whose dabs are shown are never shown: nothing is then
    /// (the caller shows every dab of a stroke before the next's). Nor are a reserved stroke's
    /// dabs whose futures can't be installed ([`OpenStrokes::ready`]): nothing is shown, to be
    /// shown later. The stats say how many were.
    ///
    /// # Errors
    ///
    /// As [`Preview::show`]; [`SculptError::Cache`] too if the stroke's shapes can't be read.
    pub fn show_over(
        &mut self,
        strokes: &mut impl OpenStrokes,
        op: OpId,
        dabs: &[Dab],
        drawn: &impl Drawn,
    ) -> Result<PreviewStats, SculptError> {
        self.show_tagged(strokes, Some(op), dabs, drawn)
    }

    /// Shows `dabs` of stroke `op` (see [`Preview::show_over`]), and closes the call's epoch.
    ///
    /// # Errors
    ///
    /// As [`Preview::show_over`].
    fn show_tagged<S: OpenStrokes>(
        &mut self,
        strokes: &mut S,
        op: Tag,
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
        let mut shown = self.show_ready(strokes, op, dabs, &imprints, drawn, &mut stats);
        let touched = std::mem::take(&mut self.touched);
        let seamed = self.stitch(strokes, touched, &mut stats);
        shown = shown.and(seamed);
        let (mut store, _) = strokes.current();
        self.note_bytes(&mut store);
        let closed = store.close_preview_epoch();
        shown?;
        closed?;
        Ok(stats)
    }

    /// Shows `imprints`, the imprints of `dabs` of stroke `op`, if they may be shown now (see
    /// [`Preview::show_over`]).
    ///
    /// For a later stroke, the futures its finest previews read are installed first, and none of
    /// the dabs is shown until they are: its own previews' and nothing else.
    ///
    /// # Errors
    ///
    /// [`SculptError::Cache`] if the stroke's shapes can't be read or a tile can't be gathered.
    fn show_ready<S: OpenStrokes>(
        &mut self,
        strokes: &mut S,
        op: Tag,
        dabs: &[Dab],
        imprints: &[Imprint],
        drawn: &impl Drawn,
        stats: &mut PreviewStats,
    ) -> Result<(), SculptError> {
        let later_shown = self.segments.back().is_some_and(|last| last.op > op);
        if imprints.is_empty() || later_shown || !ready(strokes, op, dabs, &[]) {
            return Ok(());
        }
        let base = base_of(strokes, op);
        let before = self.shown.len();
        let mut taken = BTreeMap::new();
        let mut shown = {
            let (mut store, index) = open(strokes, op)?;
            let shown = self.show_all(
                &mut store,
                index,
                (op, base),
                imprints,
                drawn,
                stats,
                &mut taken,
            );
            if self.shown.len() > before {
                self.note_shown(&mut store, index, op, imprints);
            }
            shown
        };
        drop_blocks(strokes, taken);
        if self.shown.len() > before {
            let displays = self.show_displays(strokes, op, imprints, drawn, stats);
            shown = shown.and(displays);
        }
        shown
    }

    /// The stroke whose preview finest tile `key` shows, if it is previewed.
    fn shown_by(&self, key: &TileKey) -> Option<Tag> {
        match self.reached.get(key)?.showing {
            Showing::Of(op) => Some(op),
            Showing::Untouched => None,
        }
    }

    /// The stroke whose preview tile `key` shows: a finest tile's or a display tile's, if it has
    /// one.
    fn owner_of(&self, key: &TileKey) -> Option<Tag> {
        if key.channel() == geom::Channel::DISPLAY {
            self.displays.get(key).map(|state| state.op)
        } else {
            self.shown_by(key)
        }
    }

    /// Whether previews of more than one stroke are shown, finest or display: only then can two
    /// meet.
    fn mixed(&self) -> bool {
        let finest = self
            .reached
            .values()
            .filter_map(|reached| match reached.showing {
                Showing::Of(op) => Some(op),
                Showing::Untouched => None,
            });
        let display = self.displays.values().map(|state| state.op);
        let mut owners = finest.chain(display);
        let first = owners.next();
        owners.any(|owner| Some(owner) != first)
    }

    /// The previews that may hold a vertex preview `key` holds, `key`'s included: for a finest
    /// tile, the finest tiles of its ring and the display tiles above them; for a display tile,
    /// the same around the finest tiles along its border, where it meets the tiles around it.
    /// Only those previewed (a finest tile) or with a state (a display tile).
    ///
    /// # Errors
    ///
    /// As [`TileStore::normal_inputs`].
    fn around(&self, store: &TileStore, key: &TileKey) -> Result<BTreeSet<TileKey>, CacheError> {
        let finest: Vec<TileKey> = if key.channel() == geom::Channel::DISPLAY {
            let Some(level) = store.display_finest() else {
                return Ok(BTreeSet::new());
            };
            display::finest_on_border(key, level)
        } else {
            vec![*key]
        };
        let mut around = BTreeSet::new();
        around.insert(*key);
        for tile in &finest {
            for input in store.normal_inputs(tile)?.iter() {
                if self.shown_by(input).is_some() {
                    around.insert(*input);
                }
                let above = display_ancestors(input).filter(|up| self.displays.contains_key(up));
                around.extend(above);
            }
        }
        Ok(around)
    }

    /// The places of the vertices preview `key` may share with others, each with its slot: a
    /// finest tile's points along its border, on either side, and its base-vertex rings'
    /// ([`seams::Shared`]), at the finest level; a display tile's border vertices, its own index,
    /// scaled to the finest level.
    fn held_places(&mut self, store: &TileStore, key: &TileKey) -> Vec<(usize, Place)> {
        if key.channel() == geom::Channel::DISPLAY {
            let Some(finest) = store.display_finest() else {
                return Vec::new();
            };
            let shift = finest.get().saturating_sub(key.level().get());
            let border = self
                .border_places
                .entry(*key)
                .or_insert_with(|| seams::border_places(store, key));
            let scaled = border
                .iter()
                .map(|&(v, (face, i, j))| (v, (face, i << shift, j << shift)));
            return scaled.collect();
        }
        let shared = self
            .finest_places
            .entry(*key)
            .or_insert_with(|| seams::Shared::of(store, key));
        shared
            .as_ref()
            .map_or_else(Vec::new, |shared| shared.slots.clone())
    }

    /// Makes the previews of different strokes meet bit for bit where `seeds`, the finest tiles
    /// the call made, kept or cleared, and the display tiles it handed over lie (see "Seams" in
    /// the module notes): at a vertex previews of different strokes hold, every one draws the
    /// newest stroke's bits. A finest preview that drew a later stroke's bits at a vertex no later
    /// stroke's preview holds any more is made again first.
    ///
    /// # Errors
    ///
    /// [`SculptError::Cache`] if a tile's inputs can't be listed, a tile can't be gathered, or the
    /// store refuses a preview.
    fn stitch<S: OpenStrokes>(
        &mut self,
        strokes: &mut S,
        seeds: Vec<TileKey>,
        stats: &mut PreviewStats,
    ) -> Result<(), SculptError> {
        let mut seeds = seeds;
        seeds.append(&mut self.display_seeds);
        if seeds.is_empty() || (!self.mixed() && self.finest_subs.is_empty()) {
            self.display_subs.clear();
            return Ok(());
        }
        // The previews holding a vertex a seed holds that meet another stroke's (or drew one's
        // bits), and those holding a vertex they hold: where previews of one stroke alone meet,
        // they meet bit for bit already.
        let (affected, holders) = {
            let (store, _) = strokes.current();
            let mut near = BTreeSet::new();
            for seed in &seeds {
                near.extend(self.around(&store, seed)?);
            }
            let mut affected = BTreeSet::new();
            let mut holders = BTreeSet::new();
            for key in near {
                let around = self.around(&store, &key)?;
                let own = self.owner_of(&key);
                let meets = around.iter().any(|other| self.owner_of(other) != own);
                let drew =
                    self.finest_subs.contains_key(&key) || self.display_subs.contains_key(&key);
                if meets || drew {
                    affected.insert(key);
                    holders.extend(around);
                }
            }
            (affected, holders)
        };
        // A finest preview drawing another's bits where that one doesn't win any more: made again.
        let mut chosen = {
            let (store, _) = strokes.current();
            self.choose(&store, &holders)
        };
        let lost: BTreeSet<TileKey> = {
            let (winners, _) = &chosen;
            let lost = affected.iter().filter(|key| {
                let previewed = self.owner_of(key).is_some();
                previewed
                    && self.finest_subs.get(key).is_some_and(|subs| {
                        subs.iter().any(|place| {
                            winners
                                .get(place)
                                .is_none_or(|winner| winner.holder == **key)
                        })
                    })
            });
            lost.copied().collect()
        };
        if !lost.is_empty() {
            for key in &lost {
                self.finest_subs.remove(key);
            }
            self.remake(strokes, &lost, stats)?;
            let (store, _) = strokes.current();
            chosen = self.choose(&store, &holders);
        }
        let (mut store, _) = strokes.current();
        let (chosen, normals) = chosen;
        let mut republish: Vec<TileKey> = Vec::new();
        for key in &affected {
            if self.owner_of(key).is_none() {
                if key.channel() == geom::Channel::DISPLAY {
                    self.display_subs.remove(key);
                } else {
                    self.finest_subs.remove(key);
                }
                continue;
            }
            let held = self.held_places(&store, key);
            let other = |place: &Place| chosen.get(place).filter(|winner| winner.holder != *key);
            if key.channel() == geom::Channel::DISPLAY {
                let level = key.level();
                let seams = held.into_iter().filter_map(|(v, place)| {
                    let at = other(&place).map(|winner| winner.at);
                    let normal = normals.get(&(level, place));
                    let normal = normal.filter(|winner| winner.holder != *key);
                    let normal = normal.map(|winner| winner.at);
                    (at.is_some() || normal.is_some()).then_some((v, at, normal))
                });
                let seams: Vec<Seam> = seams.collect();
                self.display_subs.insert(*key, seams);
                republish.push(*key);
            } else {
                let taken = held
                    .into_iter()
                    .filter_map(|(slot, place)| Some((slot, place, other(&place)?.at)));
                let taken: Vec<(usize, Place, Vec3)> = taken.collect();
                self.substitute(&mut store, key, &taken, stats)?;
            }
        }
        self.publish(&mut store, &republish, stats)?;
        self.display_seeds.clear();
        Ok(())
    }

    /// What each place a preview of `holders` holds shows (see "Seams" in the module notes): the
    /// newest stroke's preview holding it, a finest one before a display one of the same stroke,
    /// and the position it draws there, its own; and per display level and place, the newest
    /// stroke's display preview holding it and its normal there.
    fn choose(
        &mut self,
        store: &TileStore,
        holders: &BTreeSet<TileKey>,
    ) -> (HashMap<Place, Chosen>, HashMap<(Level, Place), Chosen>) {
        let mut chosen: HashMap<Place, Chosen> = HashMap::new();
        let mut normals: HashMap<(Level, Place), Chosen> = HashMap::new();
        // The finest previews first: of one stroke, theirs win.
        let (finest, display): (Vec<TileKey>, Vec<TileKey>) = holders
            .iter()
            .partition(|key| key.channel() != geom::Channel::DISPLAY);
        let ordered = finest.into_iter().chain(display);
        for key in ordered {
            let Some(owner) = self.owner_of(&key) else {
                continue;
            };
            let held = self.held_places(store, &key);
            if key.channel() == geom::Channel::DISPLAY {
                let Some(state) = self.displays.get_mut(&key) else {
                    continue;
                };
                state.refresh_normals();
                for (v, place) in held {
                    let Some((at, normal)) = state.vertex(v) else {
                        continue;
                    };
                    let holder = key;
                    let wins = |was: &Chosen| was.owner < owner;
                    if chosen.get(&place).is_none_or(wins) {
                        chosen.insert(place, Chosen { owner, holder, at });
                    }
                    let level = key.level();
                    if normals.get(&(level, place)).is_none_or(wins) {
                        let drawn = Chosen {
                            owner,
                            holder,
                            at: normal,
                        };
                        normals.insert((level, place), drawn);
                    }
                }
            } else {
                let Some(block) = store.preview(&key) else {
                    continue;
                };
                for (slot, place) in held {
                    let Some(at) = seams::point_at(block, slot) else {
                        continue;
                    };
                    if chosen.get(&place).is_none_or(|was| was.owner < owner) {
                        let holder = key;
                        chosen.insert(place, Chosen { owner, holder, at });
                    }
                }
            }
        }
        (chosen, normals)
    }

    /// Finest preview `key` draws, at each of `newer` (a slot, its place, and the position the
    /// preview chosen there draws: see "Seams" in the module notes), that preview's bits; handed
    /// to the store if that changed it.
    ///
    /// # Errors
    ///
    /// [`SculptError::Cache`] if the store refuses the preview.
    fn substitute(
        &mut self,
        store: &mut TileStore,
        key: &TileKey,
        newer: &[(usize, Place, Vec3)],
        stats: &mut PreviewStats,
    ) -> Result<(), SculptError> {
        if newer.is_empty() && !self.finest_subs.contains_key(key) {
            return Ok(());
        }
        let places = self.finest_places.get(key).and_then(Option::as_ref);
        let places = places.map(|shared| &shared.block);
        let (Some(places), Some(block)) = (places, store.preview(key)) else {
            return Ok(());
        };
        // Drawn so already, as a preview mostly is a call later: left as it is.
        let differs = newer.iter().any(|&(slot, _, at)| {
            seams::point_at(block, slot).map(Vec3::to_bits) != Some(at.to_bits())
        });
        let drawn_as: BTreeSet<Place> = newer.iter().map(|&(_, place, _)| place).collect();
        if !differs {
            if drawn_as.is_empty() {
                self.finest_subs.remove(key);
            } else {
                self.finest_subs.insert(*key, drawn_as);
            }
            return Ok(());
        }
        let subs: HashMap<Place, Vec3> = newer.iter().map(|&(_, place, at)| (place, at)).collect();
        let mut stitched = block.clone();
        let matched = stitched.displace_along(places, |p, at| {
            seams::place_of(at)
                .and_then(|place| subs.get(&place))
                .copied()
                .unwrap_or(p)
        });
        let before = seams::points(block);
        let after = seams::points(&stitched);
        let same = before
            .iter()
            .zip(&after)
            .all(|(a, b)| a.to_bits() == b.to_bits());
        if matched && !same {
            store.set_preview(*key, stitched)?;
            stats.tiles_set += 1;
        }
        if subs.is_empty() {
            self.finest_subs.remove(key);
        } else {
            self.finest_subs.insert(*key, drawn_as);
        }
        Ok(())
    }

    /// Notes `dabs` of stroke `op`, just shown: they join the strokes' dabs, and the finest tiles
    /// each reaches by the index are held in the store and noted as the stroke's found.
    fn note_shown(
        &mut self,
        store: &mut TileStore,
        index: &SpatialIndex,
        op: Tag,
        dabs: &[Imprint],
    ) {
        for dab in dabs {
            self.push_stroke(op, *dab);
            let reached: Vec<TileKey> = index.query_sphere(centre(dab), reach(dab)).collect();
            let found = self.found.entry(op).or_default();
            for key in &reached {
                *self.strengths.entry(*key).or_insert(0.0) += f64::from(dab.strength.abs());
                found.insert(*key);
                let count = self.held.entry(*key).or_insert(0);
                if *count == 0 {
                    store.hold_preview(*key);
                }
                *count += 1;
            }
            self.reach.push_back(reached);
        }
    }

    /// Notes `dabs` of the store's current stroke, applied at once by
    /// [`apply_dabs`](crate::apply_dabs) with nothing shown: they join the stroke's dabs, which
    /// every display preview made later takes for its normals and bound. Those the stroke has
    /// take them as a landing would have: over the finest tiles they wrote, with their turn and
    /// bound; made again instead, each would replay the whole stroke and gather what it wrote,
    /// milliseconds in a frame the dabs were found to fit. Closes the call's preview epoch.
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
        self.note_applied_tagged(&mut Lone { store, index }, None, dabs)
    }

    /// [`Preview::note_applied`] of `dabs` of stroke `op`, one of `strokes`: the oldest, whose op
    /// is begun, and the newest (with nothing shown, it is the only one).
    ///
    /// # Errors
    ///
    /// As [`Preview::note_applied`].
    pub fn note_applied_over(
        &mut self,
        strokes: &mut impl OpenStrokes,
        op: OpId,
        dabs: &[Dab],
    ) -> Result<(), SculptError> {
        self.note_applied_tagged(strokes, Some(op), dabs)
    }

    /// [`Preview::note_applied`] of `dabs` of stroke `op`.
    ///
    /// # Errors
    ///
    /// As [`Preview::note_applied`].
    fn note_applied_tagged<S: OpenStrokes>(
        &mut self,
        strokes: &mut S,
        op: Tag,
        dabs: &[Dab],
    ) -> Result<(), SculptError> {
        let reserved = self.reserved_ops(strokes);
        let (mut store, index) = strokes.current();
        let imprints: Vec<Imprint> = dabs.iter().filter_map(|dab| imprint_of(dab).ok()).collect();
        let mut reached: BTreeSet<TileKey> = BTreeSet::new();
        for dab in &imprints {
            for key in index.query_sphere(centre(dab), reach(dab)) {
                *self.strengths.entry(key).or_insert(0.0) += f64::from(dab.strength.abs());
                reached.insert(key);
            }
        }
        for dab in &imprints {
            self.push_stroke(op, *dab);
        }
        if self.displays.is_empty() {
            return Ok(());
        }
        let mut stats = PreviewStats::default();
        let landed = self.land_applied(&mut store, op, &imprints, &reached, &reserved, &mut stats);
        self.note_bytes(&mut store);
        let closed = store.close_preview_epoch();
        landed?;
        closed?;
        Ok(())
    }

    /// The display previews of stroke `op` take `dabs`, its own, already in the store, which
    /// changed around the finest tiles `reached`; those they touched are handed to the store.
    /// `reserved` are the strokes whose ops are reserved.
    ///
    /// # Errors
    ///
    /// [`SculptError::Cache`] if a tile can't be read or gathered, or the store refuses a preview.
    fn land_applied(
        &mut self,
        store: &mut TileStore,
        op: Tag,
        dabs: &[Imprint],
        reached: &BTreeSet<TileKey>,
        reserved: &BTreeSet<Tag>,
        stats: &mut PreviewStats,
    ) -> Result<(), SculptError> {
        let queried: Vec<TileKey> = reached.iter().copied().collect();
        let changed: BTreeSet<TileKey> = store.stale_after(&queried)?.into_iter().collect();
        let above = reached.iter().flat_map(display_ancestors);
        let above = above.filter(|key| self.displays.get(key).is_some_and(|state| state.op == op));
        let above: BTreeSet<TileKey> = above.collect();
        let keys = self.moved_by(store, &above, dabs, false);
        self.land_displays(store, reserved, Vec::new(), &changed, &keys, stats)
    }

    /// The strokes with dabs shown whose ops are reserved: their previews are made over their
    /// futures, which no landing moves.
    fn reserved_ops<S: OpenStrokes>(&self, strokes: &S) -> BTreeSet<Tag> {
        let ops = self.segments.iter().map(|segment| segment.op);
        let reserved = ops.filter(|op| base_of(strokes, *op) == Base::PenDown);
        reserved.collect()
    }

    /// Tells the store what the display previews' states hold, for its budget.
    fn note_bytes(&self, store: &mut TileStore) {
        let states: usize = self.displays.values().map(DisplayState::bytes).sum();
        let sets = self
            .readers
            .values()
            .flat_map(|readers| readers.by_display.values());
        let lists = sets.map(BTreeSet::len);
        let readers = lists.sum::<usize>() * (size_of::<TileKey>() + READER_BYTES);
        let stroke = self.stroke.len() * size_of::<Imprint>();
        store.note_preview_state_bytes(states + readers + stroke + self.dab_index.bytes());
    }

    /// Adds `dab`, of stroke `op`, to the strokes' dabs, and its strength to the overlaps of
    /// those it overlaps.
    fn push_stroke(&mut self, op: Tag, dab: Imprint) {
        if self.stroke_ops.last().is_none_or(|&(last, _)| last != op) {
            self.stroke_ops.push((op, self.stroke.len()));
        }
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

    /// Stroke `op`'s dabs among the strokes', as a display preview of it takes them: none for a
    /// stroke with none.
    fn dabs_of(&self, op: Tag) -> StrokeDabs<'_> {
        let at = self.stroke_ops.iter().position(|&(of, _)| of == op);
        let start = at
            .and_then(|at| self.stroke_ops.get(at))
            .map_or(self.stroke.len(), |&(_, start)| start);
        let next = at.and_then(|at| self.stroke_ops.get(at + 1));
        let end = next.map_or(self.stroke.len(), |&(_, next_start)| next_start);
        StrokeDabs::of(&self.stroke, &self.dab_index, self.shown.len(), start..end)
    }

    /// Stroke `op`'s dabs shown and not landed, oldest first.
    fn segment_dabs(&self, op: Tag) -> Vec<Imprint> {
        let mut first = 0;
        for segment in &self.segments {
            if segment.op == op {
                let dabs = self.shown.range(first..first + segment.dabs);
                return dabs.copied().collect();
            }
            first += segment.dabs;
        }
        Vec::new()
    }

    /// Notes `count` more dabs of stroke `op` shown, after every dab shown.
    fn push_segment(&mut self, op: Tag, count: usize) {
        match self.segments.back_mut() {
            Some(last) if last.op == op => last.dabs += count,
            _ => self.segments.push_back(Segment { op, dabs: count }),
        }
    }

    /// Notes the oldest `count` dabs shown landed; returns the strokes none of whose dabs are shown
    /// any more, oldest first.
    fn pop_segments(&mut self, count: usize) -> Vec<Tag> {
        let mut left = count;
        let mut gone = Vec::new();
        while left > 0 {
            let Some(front) = self.segments.front_mut() else {
                break;
            };
            let taken = left.min(front.dabs);
            front.dabs -= taken;
            left -= taken;
            if front.dabs == 0 {
                gone.push(front.op);
                self.segments.pop_front();
            }
        }
        gone
    }

    /// Twice the farthest the strokes' dabs move any place: what the moves' part of a display
    /// preview's bound is capped by (the `display` module's "The bound").
    fn moves_cap(&self) -> f64 {
        2.0 * self.overlap_most
    }

    /// How far the finest work's rounding can move a finest vertex under display tile `key`,
    /// once every dab of the strokes has landed: the largest of its finest tiles' bounds
    /// ([`display::rounding_bound`]), over the tiles a dab of the strokes reaches (the others keep
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
                most = most.max(rounding_bound(scales, strength));
            }
        }
        Ok(most)
    }

    /// Ends the stroke, the last of the strokes under way: every dab has landed (or was dropped).
    /// The display previews are handed to the store released, to be drawn until the refresh
    /// replaces them; every hold is released; the strokes' dabs are forgotten. Closes the call's
    /// preview epoch.
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
            self.clear_current(store, &mut stats);
        }
        self.segments.clear();
        self.found.clear();
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
                    let preview = state.publish(false, rounding, cap, &[]);
                    Ok(store.set_display_preview(key, preview)?)
                });
                published = published.and(set);
            }
        }
        let forgotten = forgotten.map(|_| ());
        published = published.and(forgotten);
        self.stroke.clear();
        self.stroke_ops.clear();
        self.dab_index.clear();
        self.readers.clear();
        self.border_places.clear();
        self.touched.clear();
        self.display_seeds.clear();
        self.finest_places.clear();
        self.finest_subs.clear();
        self.display_subs.clear();
        store.take_pen_down_added();
        self.overlaps.clear();
        self.overlap_most = 0.0;
        self.strengths.clear();
        self.note_bytes(store);
        let closed = store.close_preview_epoch();
        published?;
        closed?;
        Ok(())
    }

    /// Lets go of the display previews of strokes `ops`, dropped with none of their dabs landed
    /// and their reserved ops given back: they showed those dabs. Their tiles show the store's
    /// surface again (the refresh makes them anew); the finest previews go as the dabs are landed
    /// ([`Preview::land`]), which the caller does next, in the same epoch: the previews that drew
    /// their bits at shared vertices draw their own again then (see "Seams" in the module notes),
    /// so this leaves the epoch open.
    ///
    /// # Errors
    ///
    /// [`SculptError::Cache`] if the tiles around one can't be listed.
    pub fn let_go(&mut self, store: &mut TileStore, ops: &[OpId]) -> Result<(), SculptError> {
        let dropped = self
            .displays
            .iter()
            .filter(|(_, state)| state.op.is_some_and(|op| ops.contains(&op)));
        let dropped: Vec<TileKey> = dropped.map(|(key, _)| *key).collect();
        let mut listed = Ok(());
        for key in &dropped {
            // The previews it met: their seams with it are made again.
            match self.around(store, key) {
                Ok(around) => {
                    let (display, finest): (Vec<TileKey>, Vec<TileKey>) = around
                        .into_iter()
                        .filter(|other| other != key)
                        .partition(|other| other.channel() == geom::Channel::DISPLAY);
                    self.display_seeds.extend(display);
                    self.touched.extend(finest);
                }
                Err(error) => listed = Err(error),
            }
        }
        for key in &dropped {
            self.displays.remove(key);
            self.display_subs.remove(key);
            store.clear_display_preview(key);
        }
        self.note_bytes(store);
        Ok(listed?)
    }

    /// Whether a display preview holds a later stroke's dabs pending, to be made that stroke's
    /// ([`Preview::take_over_displays`]).
    #[must_use]
    pub fn displays_pending(&self) -> bool {
        self.displays.values().any(|state| state.later.is_some())
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
        index: &SpatialIndex,
        keys: &[TileKey],
    ) -> Result<PreviewStats, SculptError> {
        self.make_waiting_over(&mut Lone { store, index }, keys)
    }

    /// [`Preview::make_waiting`] with the dabs shown of `strokes`: each preview made over the
    /// pen-down surface of the stroke that shows it (see "Strokes over strokes still landing" in
    /// the module notes).
    ///
    /// # Errors
    ///
    /// As [`Preview::make_waiting`].
    pub fn make_waiting_over(
        &mut self,
        strokes: &mut impl OpenStrokes,
        keys: &[TileKey],
    ) -> Result<PreviewStats, SculptError> {
        let mut stats = PreviewStats::default();
        let mut made = Ok(());
        let mut left = WAITING_A_CALL;
        let mut attempts = WAITING_TRIES;
        let wanted: Vec<TileKey> = {
            let (store, _) = strokes.current();
            let wanted = keys
                .iter()
                .filter(|key| store.preview_wanted(key) && !self.fruitless.contains(key));
            wanted.copied().collect()
        };
        let mut wanted = wanted.into_iter();
        // Batch after batch until the call's previews are made: a tile that gets none (nothing
        // moves it, it has no record) costs its one try and is left out for a while.
        while left > 0 && attempts > 0 {
            let batch: Vec<TileKey> = wanted.by_ref().take(left.min(attempts)).collect();
            if batch.is_empty() {
                break;
            }
            let tried = self.make_batch(strokes, &batch, &mut stats);
            made = made.and(tried);
            let (store, _) = strokes.current();
            let (done, none): (Vec<TileKey>, Vec<TileKey>) = batch
                .into_iter()
                .partition(|key| !store.preview_wanted(key));
            attempts -= done.len() + none.len();
            left -= done.len();
            self.fruitless.extend(none);
        }
        let touched = std::mem::take(&mut self.touched);
        let seamed = self.stitch(strokes, touched, &mut stats);
        let made = made.and(seamed);
        let (mut store, _) = strokes.current();
        self.note_bytes(&mut store);
        let closed = store.close_preview_epoch();
        made?;
        closed?;
        Ok(stats)
    }

    /// Tries to make the previews of `keys`: a display tile with a state has it handed to the
    /// store again (the refresh dropped it while it wasn't held), one without is made, as the
    /// newest stroke's that reaches it, and a finest tile is gathered.
    ///
    /// # Errors
    ///
    /// [`SculptError::Cache`] if a tile can't be read or gathered.
    fn make_batch<S: OpenStrokes>(
        &mut self,
        strokes: &mut S,
        keys: &[TileKey],
        stats: &mut PreviewStats,
    ) -> Result<(), SculptError> {
        let (display, finest): (Vec<TileKey>, Vec<TileKey>) = keys
            .iter()
            .partition(|key| key.channel() == geom::Channel::DISPLAY);
        let (known, fresh): (Vec<TileKey>, Vec<TileKey>) = display
            .into_iter()
            .partition(|key| self.displays.contains_key(key));
        let handed = {
            let (mut store, _) = strokes.current();
            self.publish(&mut store, &known, stats)
        };
        let mut by_owner: BTreeMap<Tag, Vec<TileKey>> = BTreeMap::new();
        for key in fresh {
            by_owner
                .entry(self.display_owner(&key))
                .or_default()
                .push(key);
        }
        let mut displays = Ok(());
        for (op, owned) in by_owner {
            let made = self.make_displays_of(strokes, op, &owned, stats);
            displays = displays.and(made);
        }
        let again: BTreeSet<TileKey> = finest.into_iter().collect();
        let remade = self.remake(strokes, &again, stats);
        handed.and(displays).and(remade)
    }

    /// The stroke a display preview of `key` is made for: the newest with dabs shown that found a
    /// finest tile under it, else the oldest with dabs shown (or the last with dabs at all).
    fn display_owner(&self, key: &TileKey) -> Tag {
        let reaching = self.segments.iter().rev().find(|segment| {
            self.found.get(&segment.op).is_some_and(|tiles| {
                tiles
                    .iter()
                    .any(|tile| display_ancestors(tile).any(|above| above == *key))
            })
        });
        let oldest = self.segments.front().map(|segment| segment.op);
        let last = self.stroke_ops.last().map(|&(op, _)| op);
        reaching
            .map(|segment| segment.op)
            .or(oldest)
            .or(last)
            .flatten()
    }

    /// Moves the display previews by `new`, stroke `op`'s, just shown and noted: those of the
    /// stroke take them. An earlier stroke's they reach, if drawn, and those drawn that none has,
    /// are made as this stroke's where its futures for them are in; the others stay the earlier
    /// stroke's, never let go before this one's is made, the new dabs pending in their bounds
    /// ([`DisplayState::pend`]). Hands every one that changed to the store.
    ///
    /// # Errors
    ///
    /// [`SculptError::Cache`] if a tile can't be read or gathered.
    fn show_displays<S: OpenStrokes>(
        &mut self,
        strokes: &mut S,
        op: Tag,
        new: &[Imprint],
        drawn: &impl Drawn,
        stats: &mut PreviewStats,
    ) -> Result<(), SculptError> {
        // Per display tile above the finest tiles the dabs reach, their strengths summed.
        let first = self.reach.len().saturating_sub(new.len());
        let mut reached: BTreeMap<TileKey, f64> = BTreeMap::new();
        let per_dab = self.reach.iter().skip(first).zip(new);
        for (tiles, dab) in per_dab {
            let above: BTreeSet<TileKey> = tiles.iter().flat_map(display_ancestors).collect();
            for key in above {
                *reached.entry(key).or_insert(0.0) += f64::from(dab.strength.abs());
            }
        }
        // Every display preview of the stroke that the dabs reach takes them, drawn now or not
        // (one drawn later must hold them); of the others, those drawn are made.
        let mut known: BTreeSet<TileKey> = BTreeSet::new();
        let mut taking: Vec<TileKey> = Vec::new();
        let mut earlier: Vec<TileKey> = Vec::new();
        for key in reached.keys() {
            match self.displays.get(key).map(|state| state.op) {
                Some(owner) if owner == op => {
                    known.insert(*key);
                }
                Some(_) => {
                    earlier.push(*key);
                    if drawn.draws(key) {
                        taking.push(*key);
                    }
                }
                None if drawn.draws(key) => taking.push(*key),
                None => {}
            }
        }
        {
            let (mut store, _) = strokes.current();
            let changed = self.moved_by(&store, &known, new, true);
            // One a dab reached but didn't touch is held again all the same: said so.
            let held_again = known.iter().filter(|key| {
                self.displays
                    .get(key)
                    .is_some_and(|state| state.published_held != Some(store.preview_held(key)))
            });
            let changed: BTreeSet<TileKey> = changed.iter().chain(held_again).copied().collect();
            let changed: Vec<TileKey> = changed.into_iter().collect();
            self.publish(&mut store, &changed, stats)?;
        }
        let made = self.make_displays_of(strokes, op, &taking, stats);
        // The earlier strokes' left as they were: the new dabs pending in their bounds.
        let mut pending: Vec<TileKey> = Vec::new();
        for key in earlier {
            if let (Some(state), Some(&strength)) = (self.displays.get_mut(&key), reached.get(&key))
                && state.op < op
            {
                state.pend(op, strength);
                pending.push(key);
            }
        }
        let (mut store, _) = strokes.current();
        let published = self.publish(&mut store, &pending, stats);
        made.and(published)
    }

    /// The later strokes' dabs shown that reach a finest tile under display tile `key`, those of
    /// strokes newer than `op`: the newest of those strokes and their strengths summed, as a
    /// display preview of `op`'s pends them ([`DisplayState::pend`]).
    fn later_on(&self, key: &TileKey, op: Tag) -> Option<(Tag, f64)> {
        let mut first = 0;
        let mut later: Option<(Tag, f64)> = None;
        for segment in &self.segments {
            let range = first..first + segment.dabs;
            first += segment.dabs;
            if segment.op <= op {
                continue;
            }
            let dabs = self.reach.range(range.clone()).zip(self.shown.range(range));
            for (tiles, dab) in dabs {
                let under = tiles
                    .iter()
                    .any(|tile| display_ancestors(tile).any(|above| above == *key));
                if under {
                    let sum = later.map_or(0.0, |(_, sum)| sum);
                    later = Some((segment.op, sum + f64::from(dab.strength.abs())));
                }
            }
        }
        later
    }

    /// The display previews of `keys` of the strokes' take `dabs`, as still to land if `drawn`;
    /// returns those they touched. Taken out to work on: the strokes may have thousands, a dab
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

    /// The states of `keys` the strokes have, out of `displays` ([`Preview::put_states`] them
    /// back).
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

    /// Makes display previews of `keys` for stroke `op`, over its pen-down surface, each once
    /// every future it reads is installed ([`OpenStrokes::ready`]), in place of any the strokes
    /// have of it. One that can't be made yet keeps the preview it has; one that has none is made
    /// for the oldest stroke, over the store, the later strokes' dabs pending in its bound, until
    /// it can be ([`Preview::take_over_displays`]).
    ///
    /// # Errors
    ///
    /// [`SculptError::Cache`] if the stroke's shapes or a tile can't be read.
    fn make_displays_of<S: OpenStrokes>(
        &mut self,
        strokes: &mut S,
        op: Tag,
        keys: &[TileKey],
        stats: &mut PreviewStats,
    ) -> Result<(), SculptError> {
        if keys.is_empty() {
            return Ok(());
        }
        let (now, waiting): (Vec<TileKey>, Vec<TileKey>) = keys
            .iter()
            .partition(|key| ready(strokes, op, &[], &[**key]));
        let made = if now.is_empty() {
            Ok(())
        } else {
            let base = base_of(strokes, op);
            let (mut store, _) = open(strokes, op)?;
            self.make_displays(&mut store, op, &now, base, stats)
        };
        let fresh: Vec<TileKey> = waiting
            .into_iter()
            .filter(|key| !self.displays.contains_key(key))
            .collect();
        let oldest = self.segments.front().map(|segment| segment.op);
        let stand_in = match oldest {
            Some(oldest)
                if oldest != op && !fresh.is_empty() && base_of(strokes, oldest) == Base::Store =>
            {
                let (mut store, _) = open(strokes, oldest)?;
                self.make_displays(&mut store, oldest, &fresh, Base::Store, stats)
            }
            _ => Ok(()),
        };
        made.and(stand_in)
    }

    /// Makes display previews of `keys` for stroke `op`, over the store as it is, its pen-down
    /// reads at the stroke's shapes, drawn from `base`, in place of any the strokes have of
    /// them, and hands them to the store. A later stroke's dabs shown that reach one are pending
    /// in its bound ([`Preview::later_on`]).
    ///
    /// # Errors
    ///
    /// [`SculptError::Cache`] if a tile can't be read or gathered.
    fn make_displays(
        &mut self,
        store: &mut TileStore,
        op: Tag,
        keys: &[TileKey],
        base: Base,
        stats: &mut PreviewStats,
    ) -> Result<(), SculptError> {
        let readers = self.readers.entry(op).or_default();
        readers.unread.extend(store.take_pen_down_added());
        while let Some(&changed) = readers.unread.last() {
            // Popped once read: a failure leaves it for the next call.
            for reader in store.normal_inputs(&changed)?.iter() {
                for above in display_ancestors(reader) {
                    readers.by_display.entry(above).or_default().insert(*reader);
                }
            }
            readers.unread.pop();
        }
        let by_display = self.readers.get(&op).map(|of_op| &of_op.by_display);
        let none = BTreeMap::new();
        let dabs = self.dabs_of(op);
        let built = DisplayState::make_many(store, keys, dabs, by_display.unwrap_or(&none), base)?;
        let mut made = Vec::with_capacity(built.len());
        for mut state in built {
            state.op = op;
            if let Some((later, strength)) = self.later_on(state.key(), op) {
                state.pend(later, strength);
            }
            made.push(*state.key());
            stats.replayed += state.replayed();
            self.displays.insert(*state.key(), state);
            stats.gathered += 1;
        }
        self.publish(store, &made, stats)
    }

    /// Hands the display previews of `keys` to the store, each held as the store says now, and
    /// drawing a later stroke's bits at the vertices it shares with that stroke's previews, as
    /// the call's last seams left them (see "Seams" in the module notes; [`Preview::stitch`]
    /// makes them again at the call's end).
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
        let keys = keys.iter().filter(|key| self.displays.contains_key(key));
        let keys: BTreeSet<TileKey> = keys.copied().collect();
        self.display_seeds.extend(keys.iter().copied());
        let mut rounding: BTreeMap<TileKey, f64> = BTreeMap::new();
        for key in &keys {
            rounding.insert(*key, self.rounding(store, key)?);
        }
        let cap = self.moves_cap();
        let none: Vec<Seam> = Vec::new();
        let mut taken = self.take_states(rounding.keys());
        let seams = &self.display_subs;
        let mut publishing: Vec<Publishing<'_>> = taken
            .iter_mut()
            .filter_map(|state| {
                let rounds = rounding.get(state.key())?;
                let shared = seams.get(state.key()).unwrap_or(&none);
                Some((
                    store.preview_held(state.key()),
                    *rounds,
                    shared.as_slice(),
                    state,
                ))
            })
            .collect();
        // Made on the workers, handed over in order.
        let made: Vec<(TileKey, bool, DisplayPreview)> =
            store.par_map_mut(&mut publishing, |(held, rounds, shared, state)| {
                state.published_held = Some(*held);
                (
                    *state.key(),
                    *held,
                    state.publish(*held, *rounds, cap, shared),
                )
            });
        drop(publishing);
        self.put_states(taken);
        for (key, _, preview) in made {
            store.set_display_preview(key, preview)?;
            stats.tiles_set += 1;
        }
        Ok(())
    }

    /// Makes as their newest strokes' the display previews that hold a later stroke's dabs
    /// pending (see "Display tiles" in the module notes) and `drawn` draws, those whose futures are
    /// in now (at most a call's worth, [`OpenStrokes::ready`]): off a frame's path, a few at a
    /// time, until every one has been. Closes the call's preview epoch.
    ///
    /// # Errors
    ///
    /// [`SculptError::Cache`] if a stroke's shapes or a tile can't be read.
    pub fn take_over_displays(
        &mut self,
        strokes: &mut impl OpenStrokes,
        drawn: &impl Drawn,
    ) -> Result<PreviewStats, SculptError> {
        let mut stats = PreviewStats::default();
        let mut by: BTreeMap<Tag, Vec<TileKey>> = BTreeMap::new();
        for (key, state) in &self.displays {
            let pending = state.later.map(|(newest, _)| newest);
            let shown = pending.filter(|newest| self.segments.iter().any(|s| s.op == *newest));
            if let Some(later) = shown
                && drawn.draws(key)
            {
                by.entry(later).or_default().push(*key);
            }
        }
        if by.is_empty() {
            return Ok(stats);
        }
        let mut made = Ok(());
        let newest_first = by.into_iter().rev();
        for (op, mut keys) in newest_first {
            // Those over the stroke's finest previews first, where the two meet, the finest
            // level first: what a call's futures can't all make waits the longest coarsest.
            let shown = self
                .reached
                .iter()
                .filter(|(_, reached)| reached.showing == Showing::Of(op));
            let over: BTreeSet<TileKey> =
                shown.flat_map(|(key, _)| display_ancestors(key)).collect();
            keys.sort_by_key(|key| (!over.contains(key), std::cmp::Reverse(key.level())));
            let taken = self.make_displays_of(strokes, op, &keys, &mut stats);
            made = made.and(taken);
        }
        let touched = std::mem::take(&mut self.touched);
        let seamed = self.stitch(strokes, touched, &mut stats);
        let made = made.and(seamed);
        let (mut store, _) = strokes.current();
        self.note_bytes(&mut store);
        let closed = store.close_preview_epoch();
        made?;
        closed?;
        Ok(stats)
    }

    /// After a landing of the dabs whose reach was `landed`: releases the finest tiles no dab
    /// shown reaches any more, rebases the display previews over the finest tiles `changed` (the
    /// store moved them; those of the strokes `reserved` are over their futures, which it
    /// didn't), and hands the store those that changed or were released, with `touched`.
    ///
    /// # Errors
    ///
    /// [`SculptError::Cache`] if a finest tile can't be gathered.
    fn land_displays(
        &mut self,
        store: &mut TileStore,
        reserved: &BTreeSet<Tag>,
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
        // previews over them: a landing costs what it changed, however many the strokes have.
        let written = changed
            .iter()
            .filter(|key| store.pen_down_shape(key).is_some());
        let mut under: BTreeSet<TileKey> = BTreeSet::new();
        let mut above: BTreeSet<TileKey> = BTreeSet::new();
        for key in written {
            let over = display_ancestors(key).filter(|up| {
                self.displays
                    .get(up)
                    .is_some_and(|state| !reserved.contains(&state.op))
            });
            let mut over = over.peekable();
            if over.peek().is_some() {
                under.insert(*key);
                above.extend(over);
            }
        }
        let mut blocks = display::Blocks::default();
        blocks.add_now(store, &under)?;
        let mut taken = self.take_states(&above);
        let ops: BTreeSet<Tag> = taken.iter().map(|state| state.op).collect();
        let dabs: BTreeMap<Tag, StrokeDabs<'_>> =
            ops.into_iter().map(|op| (op, self.dabs_of(op))).collect();
        let rebased: Vec<bool> = store.par_map_mut(&mut taken, |state| {
            dabs.get(&state.op)
                .is_some_and(|dabs| state.rebase(finest, &blocks, &under, *dabs))
        });
        drop(dabs);
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

    /// Shows `new`, stroke `op`'s, its previews drawn from `base` (see [`Preview::show`]), in a
    /// reading of its shapes: the pen-down blocks of the tiles it takes over from an earlier
    /// stroke, kept in that stroke's set, are added to `taken` for the caller to drop.
    ///
    /// # Errors
    ///
    /// [`SculptError::Cache`] if a tile can't be gathered.
    #[expect(
        clippy::too_many_arguments,
        reason = "the reading, the stroke and what the call adds to: split, each would be a struct \
                  of these"
    )]
    fn show_all(
        &mut self,
        store: &mut TileStore,
        index: &SpatialIndex,
        (op, base): (Tag, Base),
        new: &[Imprint],
        drawn: &impl Drawn,
        stats: &mut PreviewStats,
        taken: &mut BTreeMap<Tag, Vec<TileKey>>,
    ) -> Result<(), SculptError> {
        if new.is_empty() {
            return Ok(());
        }
        // The tiles the stroke looked at whose pen-down box in its set touches a new dab; each is
        // its own preview already or not.
        let near: Vec<(TileKey, bool)> = self
            .reached
            .iter()
            .filter(|(_, reached)| {
                reached.looked == op && new.iter().any(|dab| touches(reached.bounds.as_ref(), dab))
            })
            .map(|(key, reached)| (*key, reached.showing == Showing::Of(op)))
            .collect();
        let queried: BTreeSet<TileKey> = new
            .iter()
            .flat_map(|dab| index.query_sphere(centre(dab), reach(dab)))
            .filter(|key| {
                let looked = self
                    .reached
                    .get(key)
                    .is_some_and(|reached| reached.looked == op);
                !looked && drawn.draws(key)
            })
            .collect();
        // The stroke's previews take the new dabs alone; the others, its base moved by every dab
        // of it shown (for one it found untouched, the dabs before moved nothing there). A preview
        // whose pen-down block the stroke's set no longer keeps is gathered afresh too.
        let mut from_store: Vec<TileKey> = queried.into_iter().collect();
        from_store.extend(near.iter().filter(|(_, own)| !own).map(|&(key, _)| key));
        let mut moving: Vec<Moving> = Vec::new();
        let own = near.iter().filter(|&&(_, own)| own);
        for &(key, _) in own {
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
        from_store.dedup();
        let gathered = gather(store, &from_store, base)?;
        stats.gathered += gathered.len();
        let incremental = moving.len();
        moving.extend(gathered);
        self.shown.extend(new.iter().copied());
        self.push_segment(op, new.len());
        stats.dabs += new.len();
        let all: Vec<Imprint> = self.segment_dabs(op);
        let (old, fresh) = moving.split_at_mut(incremental);
        store.par_map_mut(old, |tile| tile.move_by_every(new));
        store.par_map_mut(fresh, |tile| tile.move_by_every(&all));
        self.keep(store, op, moving, stats, taken)
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
        self.land_over(&mut Lone { store, index }, count, drawn)
    }

    /// [`Preview::land`] with the dabs shown of `strokes`: the oldest's dabs land first (and with
    /// them, dropped, every later stroke's).
    ///
    /// # Errors
    ///
    /// As [`Preview::land`].
    pub fn land_over(
        &mut self,
        strokes: &mut impl OpenStrokes,
        count: usize,
        drawn: &impl Drawn,
    ) -> Result<PreviewStats, SculptError> {
        let count = count.min(self.shown.len());
        self.fruitless.clear();
        let landed: Vec<Imprint> = self.shown.drain(..count).collect();
        let reach: Vec<Vec<TileKey>> = self.reach.drain(..count.min(self.reach.len())).collect();
        for gone in self.pop_segments(count) {
            self.found.remove(&gone);
        }
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
        let stale = {
            let (store, _) = strokes.current();
            store.stale_after(&queried)
        };
        let reserved = self.reserved_ops(strokes);
        let made = match stale {
            Ok(changed) => {
                let set: BTreeSet<TileKey> = changed.iter().copied().collect();
                let finest = self.land_tiles(strokes, changed, drawn, &mut stats);
                let (mut store, _) = strokes.current();
                let displays =
                    self.land_displays(&mut store, &reserved, reach, &set, &[], &mut stats);
                finest.and(displays)
            }
            Err(error) => {
                self.clear(strokes, &mut stats);
                let (mut store, _) = strokes.current();
                let empty = BTreeSet::new();
                let _released =
                    self.land_displays(&mut store, &reserved, reach, &empty, &[], &mut stats);
                Err(error.into())
            }
        };
        let touched = std::mem::take(&mut self.touched);
        let seamed = self.stitch(strokes, touched, &mut stats);
        let made = made.and(seamed);
        let (mut store, _) = strokes.current();
        self.note_bytes(&mut store);
        let closed = store.close_preview_epoch();
        made?;
        closed?;
        Ok(stats)
    }

    /// The tiles of a landing that changed `changed` made again (see [`Preview::land`]).
    ///
    /// # Errors
    ///
    /// [`SculptError::Cache`] if a tile can't be gathered or listed.
    fn land_tiles<S: OpenStrokes>(
        &mut self,
        strokes: &mut S,
        changed: Vec<TileKey>,
        drawn: &impl Drawn,
        stats: &mut PreviewStats,
    ) -> Result<(), SculptError> {
        if self.shown.is_empty() {
            // Nothing shown any more: every preview goes back to the store's tile.
            self.clear(strokes, stats);
            return Ok(());
        }
        let ops: BTreeSet<Tag> = self.segments.iter().map(|segment| segment.op).collect();
        let reserved = self.reserved_ops(strokes);
        let mut again: BTreeSet<TileKey> = BTreeSet::new();
        let mut unpreviewed: Vec<TileKey> = Vec::new();
        for key in changed {
            match self.reached.get(&key).map(|reached| reached.showing) {
                // Over the futures, which the landing didn't move.
                Some(Showing::Of(op)) if reserved.contains(&op) => {}
                Some(Showing::Of(_)) => {
                    again.insert(key);
                }
                Some(Showing::Untouched) => {
                    // Untouched, and the store moved it: looked at again below.
                    self.reached.remove(&key);
                    unpreviewed.push(key);
                }
                None => unpreviewed.push(key),
            }
        }
        // Shown by a stroke none of whose dabs are shown any more: made again too.
        let orphans = self.reached.iter().filter(
            |(_, reached)| matches!(reached.showing, Showing::Of(op) if !ops.contains(&op)),
        );
        let orphans: Vec<TileKey> = orphans.map(|(key, _)| *key).collect();
        again.extend(orphans);
        // The store moved these: within a dab of the oldest stroke still shown, by their boxes in
        // the index, or found by a later stroke's?
        let oldest = self.segments.front().map(|segment| segment.op);
        let oldest_dabs = match oldest {
            Some(op) if !reserved.contains(&op) => self.segment_dabs(op),
            _ => Vec::new(),
        };
        let within: Vec<TileKey> = {
            let (_, index) = strokes.current();
            let within = unpreviewed
                .into_iter()
                .filter(|key| drawn.draws(key))
                .filter(|key| {
                    let found = reserved
                        .iter()
                        .any(|op| self.found.get(op).is_some_and(|tiles| tiles.contains(key)));
                    let bounds = index.bvh().bounds(key).map(|bounds| bounds.aabb);
                    // A tile without bounds is looked at whole: gathered, and kept only if moved.
                    found
                        || bounds.is_none()
                        || oldest_dabs.iter().any(|dab| touches(bounds.as_ref(), dab))
                });
            within.collect()
        };
        again.extend(within);
        self.remake(strokes, &again, stats)
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
        self.rebase_over(&mut Lone { store, index })
    }

    /// [`Preview::rebase`] with the dabs shown of `strokes`.
    ///
    /// # Errors
    ///
    /// As [`Preview::land`].
    pub fn rebase_over(
        &mut self,
        strokes: &mut impl OpenStrokes,
    ) -> Result<PreviewStats, SculptError> {
        let mut stats = PreviewStats::default();
        let previewed = self
            .reached
            .iter()
            .filter(|(_, reached)| matches!(reached.showing, Showing::Of(_)));
        let mut again: BTreeSet<TileKey> = previewed.map(|(key, _)| *key).collect();
        // The untouched ones are looked at afresh.
        self.reached
            .retain(|_, reached| matches!(reached.showing, Showing::Of(_)));
        {
            let (_, index) = strokes.current();
            for dab in &self.shown {
                again.extend(index.query_sphere(centre(dab), reach(dab)));
            }
        }
        let mut made = self.remake(strokes, &again, &mut stats);
        // The display previews, made again over the store as the failure left it, each for the
        // stroke it showed.
        let displays = std::mem::take(&mut self.displays).into_iter();
        let mut by_op: BTreeMap<Tag, Vec<TileKey>> = BTreeMap::new();
        for (key, state) in displays {
            by_op.entry(state.op).or_default().push(key);
        }
        for (op, keys) in by_op {
            let remade = self.make_displays_of(strokes, op, &keys, &mut stats);
            made = made.and(remade);
        }
        let touched = std::mem::take(&mut self.touched);
        let seamed = self.stitch(strokes, touched, &mut stats);
        let made = made.and(seamed);
        let (mut store, _) = strokes.current();
        self.note_bytes(&mut store);
        let closed = store.close_preview_epoch();
        made?;
        closed?;
        Ok(stats)
    }

    /// Makes tiles `again` over the store's tiles as they are now, moved by the dabs still
    /// shown: their old previews dropped, each set again as the newest stroke's that moves it
    /// (one with dabs shown, its op begun or the tile found by its dabs), else cleared.
    ///
    /// # Errors
    ///
    /// [`SculptError::Cache`] if a tile can't be gathered: their previews are cleared.
    fn remake<S: OpenStrokes>(
        &mut self,
        strokes: &mut S,
        again: &BTreeSet<TileKey>,
        stats: &mut PreviewStats,
    ) -> Result<(), SculptError> {
        // The store's tiles are gathered afresh; the old previews are dropped, their pen-down
        // blocks with them.
        let previewed: BTreeSet<TileKey> = {
            let (mut store, _) = strokes.current();
            let previewed = again.iter().filter(|key| store.take_preview(key).is_some());
            previewed.copied().collect()
        };
        let mut dropping: BTreeMap<Tag, Vec<TileKey>> = BTreeMap::new();
        for key in again {
            if let Some(reached) = self.reached.get_mut(key)
                && let Showing::Of(op) = reached.showing
            {
                reached.showing = Showing::Untouched;
                dropping.entry(op).or_default().push(*key);
            }
        }
        drop_blocks(strokes, dropping);
        let ops: Vec<Tag> = self
            .segments
            .iter()
            .rev()
            .map(|segment| segment.op)
            .collect();
        let mut left: Vec<TileKey> = again.iter().copied().collect();
        let numbered = ops.iter().enumerate();
        for (n, &op) in numbered {
            let base = base_of(strokes, op);
            let found = self.found.get(&op);
            // A reserved stroke shows only the tiles its dabs found, its futures there installed.
            let (mut trying, rest): (Vec<TileKey>, Vec<TileKey>) =
                left.into_iter().partition(|key| {
                    base == Base::Store || found.is_some_and(|tiles| tiles.contains(key))
                });
            left = rest;
            if trying.is_empty() {
                continue;
            }
            trying.sort_unstable();
            let dabs = self.segment_dabs(op);
            let Ok((mut store, _)) = open(strokes, op) else {
                left.extend(trying);
                continue;
            };
            let mut moving = match gather(&mut store, &trying, base) {
                Ok(moving) => moving,
                Err(error) => {
                    let cleared = trying.iter().chain(&left);
                    for key in cleared {
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
            store.par_map_mut(&mut moving, |tile| tile.move_by_every(&dabs));
            // The oldest stroke's are kept whatever their fate; a later one's only if it moves
            // them, the others going on to the strokes before it.
            let (kept, unmoved): (Vec<Moving>, Vec<Moving>) = if n + 1 == ops.len() {
                (moving, Vec::new())
            } else {
                moving
                    .into_iter()
                    .partition(|tile| tile.fate != Fate::Untouched)
            };
            left.extend(unmoved.iter().map(|tile| tile.key));
            let mut taken = BTreeMap::new();
            self.keep(&mut store, op, kept, stats, &mut taken)?;
        }
        // Moved by no stroke's dabs shown: the store's tile shows.
        let (mut store, _) = strokes.current();
        for key in left {
            self.touched.push(key);
            store.clear_preview(&key);
            if previewed.contains(&key) {
                stats.tiles_cleared += 1;
            }
        }
        Ok(())
    }

    /// Clears every preview and forgets every tile reached, the dabs shown kept: for a landing
    /// that can't tell which tiles it changed, or one that leaves no dab shown. The next dab
    /// shown gathers afresh.
    fn clear<S: OpenStrokes>(&mut self, strokes: &mut S, stats: &mut PreviewStats) {
        let mut dropping: BTreeMap<Tag, Vec<TileKey>> = BTreeMap::new();
        {
            let (mut store, _) = strokes.current();
            for (key, reached) in std::mem::take(&mut self.reached) {
                if let Showing::Of(op) = reached.showing {
                    store.clear_preview(&key);
                    stats.tiles_cleared += 1;
                    dropping.entry(op).or_default().push(key);
                }
            }
        }
        drop_blocks(strokes, dropping);
    }

    /// [`Preview::clear`] as the last stroke ends: the current stroke's set is the only one left.
    fn clear_current(&mut self, store: &mut TileStore, stats: &mut PreviewStats) {
        for (key, reached) in std::mem::take(&mut self.reached) {
            if let Showing::Of(_) = reached.showing {
                store.clear_preview(&key);
                store.drop_preview_pen_down(&key);
                stats.tiles_cleared += 1;
            }
        }
    }

    /// Hands the tiles of a call of stroke `op` back to the store, in a reading of its shapes: a
    /// previewed tile no dab of the call moved put back as it was, any other moved tile's preview
    /// set (the stroke taking it over from an earlier one, whose pen-down block is added to
    /// `taken`), an unmoved one's cleared (if it had one, and no earlier stroke's preview shows
    /// it) and kept as untouched.
    ///
    /// # Errors
    ///
    /// [`SculptError::Cache`] if the store refuses a preview (never, for the finest tiles the
    /// index holds).
    fn keep(
        &mut self,
        store: &mut TileStore,
        op: Tag,
        moving: Vec<Moving>,
        stats: &mut PreviewStats,
        taken: &mut BTreeMap<Tag, Vec<TileKey>>,
    ) -> Result<(), SculptError> {
        for tile in moving {
            self.touched.push(tile.key);
            if !tile.previewed {
                // Gathered afresh: it draws its own bits everywhere until stitched again.
                self.finest_subs.remove(&tile.key);
            }
            let prior = self.reached.get(&tile.key).copied();
            // An earlier stroke's preview showing it, which this stroke's dabs may take over.
            let earlier = prior.and_then(|prior| match prior.showing {
                Showing::Of(shown_by) if shown_by != op => Some(shown_by),
                Showing::Of(_) | Showing::Untouched => None,
            });
            // A later stroke that looked at it keeps its look: its dabs' tests go by its box.
            let (looked, bounds) = match prior {
                Some(prior) if prior.looked > op => (prior.looked, prior.bounds),
                _ => (op, tile.bounds),
            };
            if tile.fate == Fate::Lost {
                self.reached.remove(&tile.key);
                store.clear_preview(&tile.key);
                store.drop_preview_pen_down(&tile.key);
                if let Some(earlier) = earlier {
                    taken.entry(earlier).or_default().push(tile.key);
                }
                continue;
            }
            let showing = match tile.fate {
                Fate::AsItWas => {
                    store.keep_preview_pen_down(tile.key, tile.pen_down);
                    store.put_back_preview(tile.key, tile.block)?;
                    Showing::Of(op)
                }
                Fate::Moved => {
                    store.keep_preview_pen_down(tile.key, tile.pen_down);
                    store.set_preview(tile.key, tile.block)?;
                    stats.tiles_set += 1;
                    if let Some(earlier) = earlier {
                        taken.entry(earlier).or_default().push(tile.key);
                    }
                    Showing::Of(op)
                }
                Fate::Untouched | Fate::Lost => {
                    if let Some(earlier) = earlier {
                        // Its dabs don't move it: the earlier stroke's preview stays.
                        Showing::Of(earlier)
                    } else {
                        // Moved by no dab shown: the store's tile shows (a preview taken is
                        // cleared).
                        store.drop_preview_pen_down(&tile.key);
                        store.clear_preview(&tile.key);
                        if tile.previewed {
                            stats.tiles_cleared += 1;
                        }
                        Showing::Untouched
                    }
                }
            };
            let reached = Reached {
                looked,
                bounds,
                showing,
            };
            self.reached.insert(tile.key, reached);
        }
        Ok(())
    }
}

/// The tiles `keys`, gathered with their rings, as tiles to move, with their pen-down blocks: each
/// drawn from `base`, the store's tile as it is or its pen-down block.
///
/// Evaluated together on the store's workers first, then gathered from its cache, on the workers
/// too. A tile whose block reads no tile the stroke changed is its own pen-down block.
///
/// # Errors
///
/// [`SculptError::Cache`] if a tile can't be evaluated or gathered.
fn gather(store: &mut TileStore, keys: &[TileKey], base: Base) -> Result<Vec<Moving>, SculptError> {
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
            let at_pen_down = pen_down.remove(&key);
            let block = match (base, &at_pen_down) {
                (Base::PenDown, Some(at)) => at.clone(),
                _ => match cached {
                    Some(block) => block,
                    None => store.positions_with_ring(key)?,
                },
            };
            let at_pen_down = at_pen_down.unwrap_or_else(|| block.clone());
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
pub fn display_ancestors(key: &TileKey) -> impl Iterator<Item = TileKey> + use<> {
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
