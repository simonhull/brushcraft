//! A tile's future: the tile as it will be once the open strokes still landing have landed,
//! computed now, without writing.
//!
//! A stroke started while earlier strokes still land reads exactly the surface they leave (the
//! stroke-while-landing spec, "What a new stroke reads"): each tile in its reach is the tile as
//! stored with every earlier open stroke's remaining dabs applied, in op order and placement
//! order. Two parts make it:
//!
//! - [`StrokeEdits`], owned, one per open stroke, kept beside its queue for the stroke's life: its
//!   dabs not landed yet, each with the tiles it touches and the tiles it edits at the stroke's
//!   pen-down, and for each tile the dabs touching it. Dabs are added as they are placed
//!   ([`StrokeEdits::place`], the tiles brushed on the store's workers, off the pen's path) and
//!   dropped as they land ([`StrokeEdits::land`]). It depends on the stroke's pen-down shapes alone
//!   ([`PenDownTiles`]), which don't change while it is open (for a later stroke, they are the
//!   earlier strokes' futures, whose final surface their landing doesn't change). A later stroke
//!   names the stroke it follows ([`StrokeEdits::after`]) and reads its pen-down once the strokes
//!   before it are sealed, their pens up ([`StrokeEdits::seal`]); see "A chain" on [`StrokeEdits`].
//! - [`Futures`], a short borrow of the session's [`Stroke`] and the chain's [`StrokeEdits`]: the
//!   stroke state's decisions for each stroke's remaining dabs, made from its bookkeeping as it is
//!   now, and each tile's future from them ([`Futures::future_of`]).
//!
//! # Bit for bit
//!
//! The future is the landing's own arithmetic, run on a copy. Each stroke's dabs are applied to
//! the tile by [`crate::apply::replay_tile`], the very function [`crate::apply_dabs`] lands a run
//! of dabs with, tile by tile: a run's tiles are its outcomes written back, a future is one
//! tile's outcome kept apart. And [`crate::apply_dabs`] lands any dabs, however it splits them
//! into frames and runs, with the bits of one dab at a time (see "The same as one dab at a time"
//! in its module notes), so a future is what the stroke's landing leaves, whichever way the
//! catch-up lands it. The replay reads, as the landing does:
//!
//! - **The dabs that touch the tile**: those whose sphere touches the box of one of its rows at
//!   pen-down, each with its number among the stroke's remaining dabs, so the stroke state's
//!   decisions line up ([`crate::apply::Run::numbers`]). A dab touching none of its rows moves none
//!   of its vertices, its own or its copies of a neighbour's: the replay would skip it.
//! - **The stroke's pen-down shape of the tile**, its dabs' distances measured there. For the
//!   oldest stroke of the chain, the store's ([`TileStore::pen_down_positions`]) of the shapes it
//!   reads now, which must be that stroke's own ([`FutureError::ForeignPenDown`]): the current
//!   stroke's set, or a reserved op's in a reading of it ([`TileStore::read_pen_down_of`]); for a
//!   later one, the future the strokes before it leave, whose positions are the tile's prediction
//!   with those corrections applied, the bits the store's evaluation of them gives, and so the
//!   pen-down shape that stroke reads (it is its surface) and lands on.
//! - **The tile's frames**, from its prediction: the level below, which no brush edits.
//! - **The stroke's state** ([`Stroke`]): each touched vertex's correction built unrounded across
//!   the stroke's dabs and rounded on write. For the stroke whose op is the store's current op and
//!   the one the session's [`Stroke`] last wrote in, its state as held; for any other, none, as its
//!   first dab will find it (the state starts afresh when the op changes), so each vertex's start
//!   is captured from the tile as the strokes before left it, the future it lands on.
//! - **The grants of that state**: which tiles hold state, are released, or are written per dab
//!   ("Releasing tiles" on [`Stroke`]). They are decided from the tiles each dab edits and nothing
//!   else. [`StrokeEdits::place`] finds those with the brush's own pass over the rows a dab touches
//!   (a dab edits a tile when it moves a vertex the tile owns, the test that makes a dab's landing
//!   count the tile as edited), and [`Futures::new`] makes the stroke's decisions from a copy of
//!   its bookkeeping, dab by dab ([`crate::apply::Plan::decide`], what a run's second replay makes
//!   of its first), every tile replayed under them ([`crate::apply::Plan::Granted`]). When the
//!   tiles the remaining dabs edit that hold no state number at most the buffers free, no tile is
//!   released or denied, and every tile goes on with its state or takes fresh state at its first
//!   edit: the decisions are skipped, and the tiles replayed under that
//!   ([`crate::apply::Plan::Continuous`]), the same bits. So a future is exact however many tiles
//!   the stroke's state has room for.
//!
//! The tile's border vertices are held by its neighbours too, each written by its owner's state:
//! the replay carries a copy of each owner's state with the same arithmetic and the owner's
//! grants, as a run's tiles do, so a future's border is the bits the owner's landing gives every
//! copy.
//!
//! # Cost
//!
//! - **Placing**: each dab brushed over the rows it touches of the tiles given, without writing, on
//!   the store's workers, a tile each: about the brush pass of landing it, less the writes and the
//!   predictions.
//! - **Deciding** ([`Futures::new`]): nothing past a count, when the stroke's state has room for
//!   the tiles its remaining dabs edit; else the stroke state's decisions for every remaining dab,
//!   a few map operations a dab and tile.
//! - **A future**: one tile replayed over the dabs of each stroke that touch it: about what landing
//!   that tile costs.
//!
//! # What it writes
//!
//! Nothing a reader of the surface sees: no correction, no op, no change or stale-tile feed, no
//! pen-down shape, no stroke state. It reads tiles through the store's caches, which it may fill
//! as any read does: corrections and positions loaded, the pen-down positions of changed tiles
//! made, and predictions kept, which may push out the least recently used predictions the
//! landing would have reused (never changing a bit: a prediction is made again when wanted; how
//! much it costs, the stroke-while-landing plan's measurements say).

use std::borrow::Cow;
use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

use brush::{Imprint, Smoothstep, draw_imprint};
use geom::{CorrectionError, CorrectionTile, Prediction, TileKey, Topology, Vec3};
use store::{Aabb, CacheError, OpId, TileStore};
use stroke::Dab;
use thiserror::Error;

use crate::apply::{
    PenDownTile, Plan, Run, SculptError, Stopped, TileInput, imprint_of, replay_tile,
};
use crate::state::Stroke;

/// Why a tile's future couldn't be made. The new stroke then waits for the strokes before it to
/// land, as it would without futures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum FutureError {
    /// The store reads another op's pen-down shapes than the oldest stroke's: a reading of
    /// another reserved op is under way ([`TileStore::reading_op`]), or none is, the oldest
    /// stroke's op isn't current, and the current stroke's set holds a shape of the tile. Read
    /// the oldest stroke's own: outside any reading when its op is current, in a reading of it
    /// ([`TileStore::read_pen_down_of`]) while it is reserved.
    #[error("the store's pen-down shapes aren't the oldest stroke's")]
    ForeignPenDown,
    /// A dab of the stroke writes a value the tile can't hold: its landing will fail at that
    /// dab.
    #[error("stroke {stroke} of the chain writes a value the tile can't hold")]
    Refused {
        /// The stroke's place in the chain, the oldest 0.
        stroke: usize,
        /// What the tile refused.
        #[source]
        error: CorrectionError,
    },
    /// The tile's prediction couldn't be made, or a vertex the replay reads isn't where it looks:
    /// a correction, position, frame or border owner missing (never, for a tile of the store's
    /// mesh).
    #[error("the tile's prediction, corrections or border can't be read")]
    Unreadable,
    /// The strokes given aren't a chain: the oldest follows a stroke not landed
    /// ([`StrokeEdits::after`]), a later one doesn't follow the stroke before it, or a stroke's
    /// dabs are placed on the pen-down of another place in the chain ([`PenDownTiles::after`]).
    #[error("the strokes given aren't a chain of open strokes")]
    Chain,
    /// Dabs are placed in a stroke sealed already ([`StrokeEdits::seal`]): its pen is up.
    #[error("the stroke is sealed: no more dabs are placed in it")]
    Sealed,
    /// A later stroke's pen-down is read on the futures of a stroke still placing dabs: its
    /// futures would change as it places more ([`StrokeEdits::seal`]).
    #[error("an earlier stroke isn't sealed: its futures aren't final")]
    Unsealed,
}

/// Tiles at a stroke's pen-down: each one's positions, row-major, as the stroke reads them, with
/// the boxes of the tile and of each of its rows.
///
/// What [`StrokeEdits::place`] brushes a stroke's new dabs over: it must hold every tile one of
/// them touches by its box at the stroke's pen-down (more are harmless; fewer are unsound, a
/// dab's edits and touches of a tile left out being missed). A caller keeps one for a stroke and
/// extends it as the pen moves on ([`PenDownTiles::extend_oldest`],
/// [`PenDownTiles::extend_futures`]): a stroke's pen-down doesn't change while it is open.
///
/// For a later stroke, which tiles a dab touches is a question about the earlier strokes'
/// futures, not about the surface as stored: take them from the futures' boxes (or the stored
/// boxes grown by the earlier strokes' largest remaining displacement of each tile), never from
/// the spatial index's boxes, which hold the tiles as stored and at pen-down. An index query at
/// a later stroke's dab can miss a tile whose future moved toward it.
#[derive(Debug, Clone, Default)]
pub struct PenDownTiles {
    /// The op of the stroke whose futures the tiles are, the newest of the chain before the
    /// stroke they are the pen-down of; `None` for the oldest open stroke's own pen-down.
    after: Option<OpId>,
    /// Each tile at pen-down.
    tiles: BTreeMap<TileKey, KeptTile>,
}

/// One tile of [`PenDownTiles`].
#[derive(Debug, Clone)]
struct KeptTile {
    /// Its positions, row-major.
    positions: Vec<Vec3>,
    /// The box of them all; `None` for none.
    bounds: Option<Aabb>,
    /// The box of each row's.
    rows: Vec<Option<Aabb>>,
}

impl KeptTile {
    /// Tile `key` at `positions`, its boxes made.
    fn new(key: TileKey, positions: Vec<Vec3>) -> Self {
        let width = usize::try_from(key.side()).unwrap_or(0).max(1);
        let rows: Vec<Option<Aabb>> = positions
            .chunks(width)
            .map(|row| Aabb::from_points(row.iter().copied()))
            .collect();
        Self {
            bounds: Aabb::from_points(positions.iter().copied()),
            rows,
            positions,
        }
    }
}

impl PenDownTiles {
    /// The tiles of `keys` at the pen-down of the oldest open stroke, of op `op`: see
    /// [`PenDownTiles::extend_oldest`].
    ///
    /// # Errors
    ///
    /// As [`PenDownTiles::extend_oldest`].
    pub fn of_oldest(
        store: &mut TileStore,
        op: OpId,
        keys: &BTreeSet<TileKey>,
    ) -> Result<Self, SculptError> {
        let mut tiles = Self::default();
        tiles.extend_oldest(store, op, keys)?;
        Ok(tiles)
    }

    /// The tiles of `keys` at the pen-down of the stroke after those of `before`: see
    /// [`PenDownTiles::extend_futures`].
    ///
    /// # Errors
    ///
    /// As [`PenDownTiles::extend_futures`].
    pub fn of_futures(
        store: &mut TileStore,
        before: &Futures<'_>,
        keys: &BTreeSet<TileKey>,
    ) -> Result<Self, SculptError> {
        let mut tiles = Self {
            after: before.newest(),
            tiles: BTreeMap::new(),
        };
        tiles.extend_futures(store, before, keys)?;
        Ok(tiles)
    }

    /// Adds the tiles of `keys` it doesn't hold, at the pen-down of the oldest open stroke, of
    /// op `op`: the store's ([`TileStore::pen_down_positions`]), a changed tile's shape, any other
    /// as stored.
    ///
    /// # Errors
    ///
    /// [`SculptError::Future`] with [`FutureError::Chain`] if these are a later stroke's tiles
    /// ([`PenDownTiles::after`] isn't `None`), or [`FutureError::ForeignPenDown`] if the shapes the
    /// store reads now aren't `op`'s (see [`FutureError::ForeignPenDown`]); as
    /// [`TileStore::pen_down_positions`]. The tiles added before the failing one stay.
    pub fn extend_oldest(
        &mut self,
        store: &mut TileStore,
        op: OpId,
        keys: &BTreeSet<TileKey>,
    ) -> Result<(), SculptError> {
        if self.after.is_some() {
            return Err(FutureError::Chain.into());
        }
        for &key in keys {
            if let std::collections::btree_map::Entry::Vacant(entry) = self.tiles.entry(key) {
                let positions = own_pen_down(store, op, key)?;
                entry.insert(KeptTile::new(key, positions));
            }
        }
        Ok(())
    }

    /// Adds the tiles of `keys` it doesn't hold, at the pen-down of the stroke after those of
    /// `before`: each the future they leave, at its positions.
    ///
    /// # Errors
    ///
    /// [`SculptError::Future`] with [`FutureError::Chain`] if `before` ends with another stroke
    /// than the one these tiles follow, or [`FutureError::Unsealed`] if a stroke of `before` may
    /// still place dabs; as [`Futures::future_of`], and as [`TileStore::keep_predictions_of`] and
    /// [`TileStore::frames_many`] for the tile's prediction. The tiles added before the failing
    /// one stay.
    pub fn extend_futures(
        &mut self,
        store: &mut TileStore,
        before: &Futures<'_>,
        keys: &BTreeSet<TileKey>,
    ) -> Result<(), SculptError> {
        if before.newest() != self.after {
            return Err(FutureError::Chain.into());
        }
        if !before.sealed() {
            return Err(FutureError::Unsealed.into());
        }
        for &key in keys {
            if self.tiles.contains_key(&key) {
                continue;
            }
            let tile = before.future_of(store, key)?;
            let _pins = store.pin(&[key])?;
            let made = ensure_prediction(store, key)?;
            let prediction = prediction_of(store, key, made)?;
            let positions = positions_of(&prediction, &tile)?;
            self.tiles.insert(key, KeptTile::new(key, positions));
        }
        Ok(())
    }

    /// The op of the stroke whose futures the tiles are: the stroke before the one they are the
    /// pen-down of; `None` for the oldest open stroke's own pen-down.
    #[must_use]
    pub const fn after(&self) -> Option<OpId> {
        self.after
    }

    /// The tiles held.
    pub fn keys(&self) -> impl Iterator<Item = &TileKey> {
        self.tiles.keys()
    }
}

/// One open stroke's dabs not landed yet, with the tiles each touches and edits at its pen-down.
///
/// What its futures need beside the stroke's state (see the module notes), kept for the stroke's
/// life beside its queue.
///
/// # A chain
///
/// A stroke started while others land follows the newest of them ([`StrokeEdits::after`]); its
/// pen-down is their futures, read once each earlier stroke is sealed ([`StrokeEdits::seal`], at
/// its pen-up), so they don't change. An earlier stroke's error while landing (dabs dropped)
/// makes its futures wrong, and with them every later stroke's [`StrokeEdits`] and the dabs placed
/// on them: those strokes end too (the stroke-while-landing spec, "Errors").
#[derive(Debug, Clone)]
pub struct StrokeEdits {
    /// The stroke's op: the op its dabs land in.
    op: OpId,
    /// The op of the stroke it follows, still open; `None` once there is none.
    after: Option<OpId>,
    /// Whether its pen is up: no more dabs are placed.
    sealed: bool,
    /// The number of the first dab not landed: the dabs are numbered from the stroke's first,
    /// so a dab's number stays as earlier ones land.
    first: usize,
    /// The dabs not landed, in placement order, numbered from `first`.
    dabs: VecDeque<PlacedDab>,
    /// For each tile, the numbers of the dabs not landed that touch it, in order.
    touching: BTreeMap<TileKey, VecDeque<usize>>,
    /// For each tile, how many dabs not landed edit it: the tiles whose state the stroke's
    /// remaining dabs need.
    edited: BTreeMap<TileKey, usize>,
}

/// One dab of a [`StrokeEdits`].
#[derive(Debug, Clone)]
struct PlacedDab {
    /// The draw action's imprint of it.
    imprint: Imprint,
    /// The tiles it touches, in key order: those holding a row whose box at pen-down its sphere
    /// touches.
    touches: Vec<TileKey>,
    /// The tiles it edits, in key order: those owning a vertex it moves.
    edits: Vec<TileKey>,
}

impl StrokeEdits {
    /// The edits of the stroke of op `op`, before any dab is placed, following the open stroke of
    /// op `after` (`None`: no stroke before it is open).
    #[must_use]
    pub const fn new(op: OpId, after: Option<OpId>) -> Self {
        Self {
            op,
            after,
            sealed: false,
            first: 0,
            dabs: VecDeque::new(),
            touching: BTreeMap::new(),
            edited: BTreeMap::new(),
        }
    }

    /// The stroke's op.
    #[must_use]
    pub const fn op(&self) -> OpId {
        self.op
    }

    /// The op of the open stroke it follows; `None` if none is open before it.
    #[must_use]
    pub const fn after(&self) -> Option<OpId> {
        self.after
    }

    /// The stroke it followed has landed and ended: it is the oldest open stroke now. Its dabs
    /// placed stay right (they were placed on the surface that stroke left), and the next are
    /// placed on its own pen-down ([`PenDownTiles::of_oldest`]).
    pub const fn earlier_landed(&mut self) {
        self.after = None;
    }

    /// The stroke's pen is up: no more dabs are placed ([`StrokeEdits::place`] refuses them), so
    /// its futures are final, and a later stroke may read its pen-down on them.
    pub const fn seal(&mut self) {
        self.sealed = true;
    }

    /// Whether the stroke is sealed ([`StrokeEdits::seal`]).
    #[must_use]
    pub const fn is_sealed(&self) -> bool {
        self.sealed
    }

    /// How many of its dabs haven't landed.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.dabs.len()
    }

    /// Adds `dabs`, the stroke's next placed, in order, each brushed over the tiles of
    /// `pen_down`, the stroke's pen-down, without writing, on `store`'s workers, a tile each.
    ///
    /// `pen_down` must hold every tile one of the dabs touches by its box at the stroke's pen-down
    /// (see [`PenDownTiles`]); the store is read for its mesh and workers only.
    ///
    /// # Errors
    ///
    /// [`SculptError::Future`] with [`FutureError::Sealed`] for a sealed stroke, or
    /// [`FutureError::Chain`] if `pen_down` is the pen-down of another place in the chain (its
    /// [`PenDownTiles::after`] isn't this stroke's [`StrokeEdits::after`]); [`SculptError::Dab`]
    /// for a dab the brush can't apply (its place in `dabs`). Nothing is added then.
    pub fn place(
        &mut self,
        store: &TileStore,
        dabs: &[Dab],
        pen_down: &PenDownTiles,
    ) -> Result<(), SculptError> {
        if self.sealed {
            return Err(FutureError::Sealed.into());
        }
        if pen_down.after != self.after {
            return Err(FutureError::Chain.into());
        }
        let imprints = imprints_of(dabs)?;
        let topology = store.mesh().topology();
        let spheres: Vec<(Vec3, f32)> = imprints.iter().map(sphere).collect();
        // Only the tiles one of the dabs can reach are sent to the workers.
        let tiles: Vec<(&TileKey, &KeptTile)> = pen_down
            .tiles
            .iter()
            .filter(|(_, tile)| spheres.iter().any(|&each| touches(tile.bounds, each)))
            .collect();
        let found = store.par_map(&tiles, |&(key, tile)| {
            tile_edits(topology, *key, tile, &imprints, &spheres)
        });
        let mut placed: Vec<PlacedDab> = imprints
            .into_iter()
            .map(|imprint| PlacedDab {
                imprint,
                touches: Vec::new(),
                edits: Vec::new(),
            })
            .collect();
        // Tiles in key order, so each dab's lists are too.
        let per_tile = tiles.iter().zip(found);
        for (&(&key, _), touched) in per_tile {
            for (k, edits) in touched {
                let Some(dab) = placed.get_mut(k) else {
                    continue;
                };
                dab.touches.push(key);
                if edits {
                    dab.edits.push(key);
                }
            }
        }
        let numbered = placed.into_iter().zip(self.first + self.dabs.len()..);
        for (dab, number) in numbered {
            for key in &dab.touches {
                self.touching.entry(*key).or_default().push_back(number);
            }
            for key in &dab.edits {
                *self.edited.entry(*key).or_default() += 1;
            }
            self.dabs.push_back(dab);
        }
        Ok(())
    }

    /// Drops the oldest `count` dabs not landed (all, if fewer remain): they have landed.
    pub fn land(&mut self, count: usize) {
        for _ in 0..count {
            let Some(dab) = self.dabs.pop_front() else {
                break;
            };
            for key in &dab.touches {
                if let Some(numbers) = self.touching.get_mut(key) {
                    numbers.pop_front();
                    if numbers.is_empty() {
                        self.touching.remove(key);
                    }
                }
            }
            for key in &dab.edits {
                if let Some(editing) = self.edited.get_mut(key) {
                    *editing -= 1;
                    if *editing == 0 {
                        self.edited.remove(key);
                    }
                }
            }
            self.first += 1;
        }
    }

    /// The stroke state's decisions for the remaining dabs, from `book_for` the stroke's op (see
    /// "The grants of that state" in the module notes).
    fn plan(&self, store: &TileStore, stroke: &Stroke) -> Plan {
        let book = stroke.book_for(store, self.op);
        if book.has_room_for(self.edited.keys()) {
            return Plan::Continuous;
        }
        let edits = self.dabs.iter().map(|dab| dab.edits.as_slice());
        let (grants, _, _) = Plan::decide(&book, edits);
        Plan::Granted(grants)
    }

    /// The dabs not landed that touch `key`, in order: their imprints, and their numbers counted
    /// from the first not landed (the count the stroke state's decisions are made in).
    fn touching(&self, key: &TileKey) -> (Vec<Imprint>, Vec<usize>) {
        let numbers = self.touching.get(key).into_iter().flatten();
        numbers
            .filter_map(|&number| {
                let place = number.checked_sub(self.first)?;
                Some((self.dabs.get(place)?.imprint, place))
            })
            .unzip()
    }
}

/// `imprint`'s sphere: its centre and squared radius.
fn sphere(imprint: &Imprint) -> (Vec3, f32) {
    let [x, y, z] = imprint.center;
    (Vec3::new(x, y, z), imprint.radius * imprint.radius)
}

/// Whether the sphere of centre and squared radius `(centre, radius_squared)` touches `bounds`
/// (`None`: a box of no points, which nothing touches).
fn touches(bounds: Option<Aabb>, (centre, radius_squared): (Vec3, f32)) -> bool {
    bounds.is_some_and(|aabb| aabb.distance_squared(centre) <= radius_squared)
}

/// For each of `imprints` (with their `spheres`) touching tile `key` at pen-down: its place, and
/// whether it edits the tile (moves a vertex the tile owns).
///
/// A dab touches the tile when its sphere touches the box of one of its rows (a dab missing the
/// whole tile's box is passed over at once); it is brushed over those rows alone, as a replay
/// brushes it ([`draw_imprint`], each vertex on its own, so the bits of the whole tile's pass),
/// into one buffer reused for every dab.
fn tile_edits(
    topology: &Topology,
    key: TileKey,
    tile: &KeptTile,
    imprints: &[Imprint],
    spheres: &[(Vec3, f32)],
) -> Vec<(usize, bool)> {
    let side = key.side();
    let last = side.saturating_sub(1);
    let width = usize::try_from(side).unwrap_or(0);
    let mut deltas = vec![0.0_f32; width];
    let mut out = Vec::new();
    let dabs = imprints.iter().zip(spheres).enumerate();
    for (k, (imprint, &each)) in dabs {
        if !touches(tile.bounds, each) {
            continue;
        }
        let touched = (0..side)
            .zip(&tile.rows)
            .filter(|&(_, &aabb)| touches(aabb, each));
        let mut reached = false;
        let mut edits = false;
        for (j, _) in touched {
            reached = true;
            if edits {
                break;
            }
            let first = usize::try_from(j).unwrap_or(0) * width;
            let Some(row) = tile.positions.get(first..first + width) else {
                continue;
            };
            deltas.fill(0.0);
            if draw_imprint(&Smoothstep, imprint, bytemuck::cast_slice(row), &mut deltas) == 0 {
                continue;
            }
            let columns = (0..side).zip(&deltas);
            edits = columns.into_iter().any(|(i, &delta)| {
                let inside = i != 0 && j != 0 && i != last && j != last;
                delta != 0.0 && (inside || topology.owns(&key, i, j))
            });
        }
        if reached {
            out.push((k, edits));
        }
    }
    out
}

/// The futures of a chain of open strokes: each stroke's remaining dabs ([`StrokeEdits`]) and
/// the stroke state's decisions for them, made from the session's [`Stroke`] as it is now.
///
/// It borrows the [`Stroke`], so the strokes can't land while it is held: make it for a batch of
/// futures, and drop it before landing more.
#[derive(Debug)]
pub struct Futures<'a> {
    /// The session's stroke state, the one [`crate::apply_dabs`] lands with.
    stroke: &'a Stroke,
    /// The chain's strokes, oldest first, each with its decisions.
    strokes: Vec<(&'a StrokeEdits, Plan)>,
}

impl<'a> Futures<'a> {
    /// The futures of `chain`, the open strokes oldest first, which land one after another in
    /// this order; `stroke` is the session's one stroke state. The oldest's pen-down shapes are
    /// the store's: its op is the store's current op, or the store keeps no shape of a tile it
    /// reaches.
    ///
    /// # Errors
    ///
    /// [`SculptError::Future`] with [`FutureError::Chain`] unless the oldest follows no open
    /// stroke and each later one follows the stroke before it ([`StrokeEdits::after`]).
    pub fn new(
        store: &TileStore,
        stroke: &'a Stroke,
        chain: &[&'a StrokeEdits],
    ) -> Result<Self, SculptError> {
        let befores = std::iter::once(None).chain(chain.iter().map(|edits| Some(edits.op)));
        let mut links = chain.iter().zip(befores);
        if !links.all(|(edits, before)| edits.after == before) {
            return Err(FutureError::Chain.into());
        }
        let strokes = chain
            .iter()
            .map(|&edits| (edits, edits.plan(store, stroke)))
            .collect();
        Ok(Self { stroke, strokes })
    }

    /// The op of the newest stroke of the chain; `None` for none.
    fn newest(&self) -> Option<OpId> {
        self.strokes.last().map(|(edits, _)| edits.op)
    }

    /// Whether every stroke of the chain is sealed: its futures are final.
    fn sealed(&self) -> bool {
        self.strokes.iter().all(|(edits, _)| edits.sealed)
    }

    /// For each stroke of the chain, whether its decisions took the fast path
    /// ([`crate::apply::Plan::Continuous`]): for the tests.
    #[cfg(test)]
    pub(crate) fn continuous(&self) -> Vec<bool> {
        self.strokes
            .iter()
            .map(|(_, plan)| *plan == Plan::Continuous)
            .collect()
    }

    /// Tile `key` once every stroke of the chain has landed its remaining dabs, made without
    /// writing. A chain with no dabs touching it gives the tile as stored.
    ///
    /// # Errors
    ///
    /// [`SculptError::Cache`] if the store can't read or pin the tile, or keep its prediction;
    /// [`SculptError::Future`] with [`FutureError::ForeignPenDown`] if the oldest stroke touches
    /// the tile and the store's shape of it isn't that stroke's, [`FutureError::Refused`] for a
    /// value the tile can't hold, or [`FutureError::Unreadable`].
    pub fn future_of(
        &self,
        store: &mut TileStore,
        key: TileKey,
    ) -> Result<CorrectionTile, SculptError> {
        let touching: Vec<(Vec<Imprint>, Vec<usize>)> = self
            .strokes
            .iter()
            .map(|(edits, _)| edits.touching(&key))
            .collect();
        let mut tile: CorrectionTile = store.correction(key)?.to_tile();
        if touching.iter().all(|(imprints, _)| imprints.is_empty()) {
            return Ok(tile);
        }
        // The oldest stroke's pen-down positions, as its landing reads them.
        let first_pen_down: Option<Vec<Vec3>> = match (self.strokes.first(), touching.first()) {
            (Some((edits, _)), Some((imprints, _))) if !imprints.is_empty() => {
                Some(own_pen_down(store, edits.op, key)?)
            }
            _ => None,
        };
        let _pins = store.pin(&[key])?;
        let made = ensure_prediction(store, key)?;

        let store: &TileStore = store;
        let prediction = prediction_of(store, key, made)?;
        let landing = self.strokes.iter().zip(&touching).enumerate();
        for (n, ((edits, plan), (imprints, numbers))) in landing {
            if imprints.is_empty() {
                continue;
            }
            // A later stroke's pen-down shape of the tile is the future the strokes before it
            // leave.
            let pen_down: Cow<'_, [Vec3]> = if n == 0
                && let Some(positions) = &first_pen_down
            {
                Cow::Borrowed(positions)
            } else {
                Cow::Owned(positions_of(&prediction, &tile)?)
            };
            let run = Run {
                imprints,
                numbers: Some(numbers),
                plan,
                stroke: self.stroke.continued(store, edits.op),
                topology: store.mesh().topology(),
                #[cfg(test)]
                break_twins: false,
            };
            let input = TileInput {
                key,
                pen_down: PenDownTile::Positions(Some(&pen_down)),
                corrections: tile.view(),
                prediction: Some(&prediction),
                frames: None,
            };
            let outcome = replay_tile(run, input);
            match outcome.stopped {
                None => {}
                Some(Stopped::Refused(error)) => {
                    return Err(FutureError::Refused { stroke: n, error }.into());
                }
                Some(Stopped::NoBorder | Stopped::Inconsistent) => {
                    return Err(FutureError::Unreadable.into());
                }
            }
            if let Some(written) = outcome.written {
                tile = written;
            }
        }
        Ok(tile)
    }
}

/// The draw action's imprints of `dabs`, checked as [`crate::apply_dabs`] checks them.
///
/// # Errors
///
/// [`SculptError::Dab`] for the first the brush can't apply.
fn imprints_of(dabs: &[Dab]) -> Result<Vec<Imprint>, SculptError> {
    dabs.iter()
        .enumerate()
        .map(|(n, dab)| imprint_of(dab).map_err(|error| SculptError::Dab { dab: n, error }))
        .collect()
}

/// Tile `key`'s pen-down positions as the oldest open stroke, of op `op`, reads them.
///
/// The store's ([`TileStore::pen_down_positions`]) from the pen-down set it reads now: its shape
/// if the set keeps one, else its positions as stored. The set read must be `op`'s: a reserved op's
/// in a reading of it ([`TileStore::read_pen_down_of`], [`TileStore::reading_op`] naming `op`), or,
/// outside any reading of a reserved op, the current stroke's, `op` being current. With another op
/// current (or none), the current set may still be read if it keeps no shape of the tile: the tile
/// is then as stored, which is what an op not yet begun reads of it.
///
/// # Errors
///
/// [`FutureError::ForeignPenDown`] if the set read is another op's (see there); as
/// [`TileStore::pen_down_positions`].
fn own_pen_down(store: &mut TileStore, op: OpId, key: TileKey) -> Result<Vec<Vec3>, SculptError> {
    let own = match store.reading_op() {
        Some(reading) => reading == op,
        None => store.current_op() == Some(op) || store.pen_down_shape(&key).is_none(),
    };
    if !own {
        return Err(FutureError::ForeignPenDown.into());
    }
    Ok(store.pen_down_positions(key)?.to_vec())
}

/// Makes tile `key`'s prediction readable: kept by the store if it can, giving `None`, else made
/// afresh and given.
///
/// It holds the bits the tile's evaluation, and so a landing, uses. Kept
/// ([`TileStore::keep_predictions_of`]) when the store has room: the tile, pinned by the caller,
/// is about to be landed, and its landing reuses it.
///
/// Made afresh, it is predicted from the tile's parents, made current by
/// [`TileStore::frames_many`], whose frames are dropped: the replay needs the prediction's
/// positions too, to make a later stroke's pen-down positions, and the store has no public way to
/// make a tile's parents current without predicting it. One prediction more, only for a tile the
/// store can't keep one for.
///
/// # Errors
///
/// As [`TileStore::keep_predictions_of`] and [`TileStore::frames_many`];
/// [`FutureError::Unreadable`] if the parents aren't current to predict from (never, just after
/// they were made so).
fn ensure_prediction(
    store: &mut TileStore,
    key: TileKey,
) -> Result<Option<Prediction>, SculptError> {
    let kept = match store.keep_predictions_of(&[key]) {
        Ok(kept) => kept,
        Err(CacheError::Full | CacheError::OverBudget { .. }) => false,
        Err(e) => return Err(e.into()),
    };
    if kept && store.kept_prediction(&key).is_some() {
        return Ok(None);
    }
    store.frames_many(&[key])?;
    let parents = store.parent_slots(&key).ok_or(FutureError::Unreadable)?;
    let prediction = store
        .predictor()
        .predict(&key, &parents)
        .ok_or(FutureError::Unreadable)?;
    Ok(Some(prediction))
}

/// Tile `key`'s prediction: `made`, if [`ensure_prediction`] made one, else the store's.
///
/// # Errors
///
/// [`FutureError::Unreadable`] if neither is there (never, just after [`ensure_prediction`]).
fn prediction_of(
    store: &TileStore,
    key: TileKey,
    made: Option<Prediction>,
) -> Result<Cow<'_, Prediction>, FutureError> {
    match made {
        Some(prediction) => Ok(Cow::Owned(prediction)),
        None => store
            .kept_prediction(&key)
            .map(Cow::Borrowed)
            .ok_or(FutureError::Unreadable),
    }
}

/// The positions of a tile whose prediction is `prediction` and whose corrections are `tile`,
/// row-major: the bits the store's evaluation gives
/// ([`PenDownShape::positions`](store::PenDownShape::positions) makes them alike).
///
/// # Errors
///
/// [`FutureError::Unreadable`] if `tile` lacks a vertex of the prediction (never, for a tile and
/// its own prediction).
fn positions_of(prediction: &Prediction, tile: &CorrectionTile) -> Result<Vec<Vec3>, FutureError> {
    if prediction.side() != tile.side() {
        return Err(FutureError::Unreadable);
    }
    let view = tile.view();
    let missing = Cell::new(false);
    let positions = prediction
        .apply(|i, j| {
            view.get(i, j).unwrap_or_else(|| {
                missing.set(true);
                Vec3::ZERO
            })
        })
        .into_vec();
    if missing.get() {
        return Err(FutureError::Unreadable);
    }
    Ok(positions)
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;
