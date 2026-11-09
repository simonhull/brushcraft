//! A tile's future: the tile as it will be once the open strokes still landing have landed,
//! computed now, without writing.
//!
//! A stroke started while earlier strokes still land reads exactly the surface they leave (the
//! stroke-while-landing spec, "What a new stroke reads"): each tile in its reach is the tile as
//! stored with every earlier open stroke's remaining dabs applied, in op order and placement
//! order. A [`FuturePlan`] is made once for a chain of open strokes, and gives each tile's future
//! ([`FuturePlan::future_of`]).
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
//! - **The stroke's pen-down shape of the tile**, its dabs' distances measured there. For the
//!   oldest stroke of the chain, the store's ([`TileStore::pen_down_positions`]), which must be its
//!   own ([`FutureError::ForeignPenDown`]); for a later one, the future the strokes before it
//!   leave, whose positions are the tile's prediction with those corrections applied, the bits the
//!   store's evaluation of them gives, and so the pen-down shape that stroke reads (it is its
//!   surface) and lands on.
//! - **The tile's frames**, from its prediction: the level below, which no brush edits.
//! - **The stroke's state** ([`Stroke`]): each touched vertex's correction built unrounded across
//!   the stroke's dabs and rounded on write. For the stroke whose op is the store's current op and
//!   the one the session's [`Stroke`] last wrote in ([`Remaining::op`]), its state as held; for any
//!   other, none, as its first dab will find it (the state starts afresh when the op changes), so
//!   each vertex's start is captured from the tile as the strokes before left it, the future it
//!   lands on.
//! - **The grants of that state**: which tiles hold state, are released, or are written per dab
//!   ("Releasing tiles" on [`Stroke`]). Those are decided from the tiles each dab edits and nothing
//!   else, so the plan decides them ahead, once a stroke, for every tile: for each tile of the
//!   stroke's reach ([`Remaining::reach`]), at its pen-down positions, the brush's own pass
//!   ([`crate::apply::brush_tile`]) tells which remaining dabs move a vertex it owns, the very test
//!   that makes a dab's landing count the tile as edited; then a copy of the stroke's bookkeeping
//!   makes the stroke's decisions dab by dab ([`crate::apply::Plan::decide`], what a run's second
//!   replay makes of its first), and every tile is replayed under them
//!   ([`crate::apply::Plan::Granted`]). So a future is exact however many tiles the stroke's state
//!   has room for.
//!
//! The tile's border vertices are held by its neighbours too, each written by its owner's state:
//! the replay carries a copy of each owner's state with the same arithmetic and the owner's
//! grants, as a run's tiles do, so a future's border is the bits the owner's landing gives every
//! copy.
//!
//! # Cost
//!
//! Making the plan brushes each stroke's remaining dabs over its reach once, without writing: the
//! brush pass of landing them, less the writes. A later stroke's reach is read at its pen-down
//! shape, the earlier strokes' futures of those tiles. Each future after that replays one tile
//! over each stroke's remaining dabs, rows out of a dab's reach skipped by their boxes.
//!
//! # What it writes
//!
//! Nothing a reader of the surface sees: no correction, no op, no change or stale-tile feed, no
//! pen-down shape, no stroke state. It reads tiles through the store's caches, which it may fill
//! as any read does: corrections and positions loaded, predictions kept, the pen-down positions
//! of changed tiles made.

use std::borrow::Cow;
use std::cell::Cell;
use std::collections::BTreeSet;

use brush::Imprint;
use geom::{CorrectionError, CorrectionTile, Prediction, TileKey, Vec3};
use store::{Aabb, CacheError, OpId, TileStore};
use stroke::Dab;
use thiserror::Error;

use crate::apply::{
    PenDownTile, Plan, Run, SculptError, Stopped, TileInput, brush_tile, imprint_of, replay_tile,
};
use crate::state::Stroke;

/// One open stroke of a chain: its op, its dabs not landed yet, and the tiles they can reach.
///
/// # Contract
///
/// `dabs` are exactly the stroke's dabs not landed yet, in placement order: none landed already
/// (it would count twice), none left out (its edits, and its say in the stroke state's
/// decisions, would be missed). `reach` holds every tile one of them touches by its box at the
/// stroke's pen-down; more tiles are harmless (a tile no dab moves is no dab's edit), fewer are
/// unsound (an edit missed changes the decisions).
#[derive(Debug, Clone, Copy)]
pub struct Remaining<'a> {
    /// The stroke's op: the op its dabs land in. The session's [`Stroke`] is this stroke's state
    /// if this op is the store's current op and the one the state last wrote in.
    pub op: OpId,
    /// Its dabs not landed yet, in placement order.
    pub dabs: &'a [Dab],
    /// Every tile any of those dabs touches by its box at the stroke's pen-down, or more.
    pub reach: &'a BTreeSet<TileKey>,
}

/// Why a tile's future couldn't be made. The new stroke then waits for the strokes before it to
/// land, as it would without futures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum FutureError {
    /// The store keeps a pen-down shape of a tile the oldest stroke reaches, and its op isn't the
    /// oldest stroke's: the shape is another stroke's, not the surface the oldest one reads. (The
    /// store keeps one stroke's shapes, the current op's, until shapes are kept per op.)
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
}

/// The futures of a chain of open strokes: each stroke's remaining dabs and the stroke state's
/// decisions for them, made once, then each tile's future from them (see the module notes).
///
/// It borrows the session's [`Stroke`], so the strokes can't land while it is held: its futures
/// are those of the store and state it was made over. Drop it before landing more, and make
/// another for the dabs then remaining.
#[derive(Debug)]
pub struct FuturePlan<'a> {
    /// The session's stroke state, the one [`crate::apply_dabs`] lands with.
    stroke: &'a Stroke,
    /// The chain's strokes, oldest first.
    strokes: Vec<StrokePlan>,
}

/// One stroke of a [`FuturePlan`].
#[derive(Debug)]
struct StrokePlan {
    /// Its op.
    op: OpId,
    /// The draw action's imprints of its remaining dabs, in order.
    imprints: Vec<Imprint>,
    /// The stroke state's grants for them, by dab, counted from the first remaining.
    plan: Plan,
}

impl<'a> FuturePlan<'a> {
    /// The plan of `chain`, the open strokes oldest first, with `stroke` the session's one stroke
    /// state; they land one after another, in this order.
    ///
    /// Each stroke's remaining dabs are brushed over its reach at its pen-down, without writing,
    /// to find which tiles each edits, and the stroke state's decisions are made from those (see
    /// the module notes). The oldest stroke's pen-down shapes are the store's: its op is the
    /// store's current op, or the store keeps no shape of a tile it reaches.
    ///
    /// # Errors
    ///
    /// [`SculptError::Dab`] for a dab the brush can't apply (its place in its stroke's slice),
    /// before anything is read; [`SculptError::Cache`] if the store can't read or pin a tile of a
    /// reach, or keep its prediction; [`SculptError::Future`] with
    /// [`FutureError::ForeignPenDown`], or, for a later stroke's reach read on the earlier ones'
    /// futures, as [`FuturePlan::future_of`].
    pub fn new(
        store: &mut TileStore,
        stroke: &'a Stroke,
        chain: &[Remaining<'_>],
    ) -> Result<Self, SculptError> {
        let checked: Result<Vec<Vec<Imprint>>, SculptError> = chain
            .iter()
            .map(|remaining| imprints_of(remaining.dabs))
            .collect();
        let mut futures = Self {
            stroke,
            strokes: Vec::with_capacity(chain.len()),
        };
        let strokes = chain.iter().zip(checked?);
        for (remaining, imprints) in strokes {
            let plan = if imprints.is_empty() {
                Plan::Continuous
            } else {
                futures.decide(store, remaining, &imprints)?
            };
            futures.strokes.push(StrokePlan {
                op: remaining.op,
                imprints,
                plan,
            });
        }
        Ok(futures)
    }

    /// Tile `key` once every stroke of the chain has landed its remaining dabs, made without
    /// writing. A chain with no dabs gives the tile as stored.
    ///
    /// # Errors
    ///
    /// [`SculptError::Cache`] if the store can't read or pin the tile, or keep its prediction;
    /// [`SculptError::Future`] with [`FutureError::ForeignPenDown`] if the oldest stroke reaches
    /// the tile and the store's shape of it isn't that stroke's, [`FutureError::Refused`] for a
    /// value the tile can't hold, or [`FutureError::Unreadable`].
    pub fn future_of(
        &self,
        store: &mut TileStore,
        key: TileKey,
    ) -> Result<CorrectionTile, SculptError> {
        self.tile_after(store, self.strokes.len(), key)
    }

    /// The decisions of the stroke after those planned, `remaining`, whose dabs' imprints are
    /// `imprints`: the grants its state makes, dab by dab, for the tiles of its reach each edits
    /// at its pen-down.
    ///
    /// # Errors
    ///
    /// As [`FuturePlan::pen_down`].
    fn decide(
        &self,
        store: &mut TileStore,
        remaining: &Remaining<'_>,
        imprints: &[Imprint],
    ) -> Result<Plan, SculptError> {
        let mut edited: Vec<Vec<TileKey>> = vec![Vec::new(); imprints.len()];
        for &key in remaining.reach {
            let positions = self.pen_down(store, remaining.op, key)?;
            // A dab whose sphere misses the box of every position moves none of them.
            let Some(bounds) = Aabb::from_points(positions.iter().copied()) else {
                continue;
            };
            let topology = store.mesh().topology();
            let dabs = imprints.iter().zip(&mut edited);
            for (imprint, tiles) in dabs {
                let [x, y, z] = imprint.center;
                let near =
                    bounds.distance_squared(Vec3::new(x, y, z)) <= imprint.radius * imprint.radius;
                if near && brush_tile(topology, key, &positions, imprint).moved > 0 {
                    tiles.push(key);
                }
            }
        }
        let book = self.stroke.book_for(store, remaining.op);
        let (grants, _, _) = Plan::decide(&book, edited);
        Ok(Plan::Granted(grants))
    }

    /// Tile `key`'s pen-down positions as the stroke after those planned, of op `op`, reads
    /// them: the store's for the oldest, else the tile's prediction with the planned strokes'
    /// future applied.
    ///
    /// # Errors
    ///
    /// As [`own_pen_down`] for the oldest stroke; as [`FuturePlan::tile_after`] and
    /// [`ensure_prediction`] for a later one.
    fn pen_down(
        &self,
        store: &mut TileStore,
        op: OpId,
        key: TileKey,
    ) -> Result<Vec<Vec3>, SculptError> {
        if self.strokes.is_empty() {
            return own_pen_down(store, op, key);
        }
        let tile = self.tile_after(store, self.strokes.len(), key)?;
        let _pins = store.pin(&[key])?;
        let made = ensure_prediction(store, key)?;
        let prediction = prediction_of(store, key, made)?;
        Ok(positions_of(&prediction, &tile)?)
    }

    /// Tile `key` once the first `upto` strokes of the plan have landed their remaining dabs.
    ///
    /// # Errors
    ///
    /// As [`FuturePlan::future_of`].
    fn tile_after(
        &self,
        store: &mut TileStore,
        upto: usize,
        key: TileKey,
    ) -> Result<CorrectionTile, SculptError> {
        let strokes = self.strokes.get(..upto).unwrap_or_default();
        let mut tile: CorrectionTile = store.correction(key)?.to_tile();
        if strokes.iter().all(|planned| planned.imprints.is_empty()) {
            return Ok(tile);
        }
        // The oldest stroke's pen-down positions, as its landing reads them.
        let first_pen_down: Option<Vec<Vec3>> = match strokes.first() {
            Some(first) if !first.imprints.is_empty() => Some(own_pen_down(store, first.op, key)?),
            _ => None,
        };
        let _pins = store.pin(&[key])?;
        let made = ensure_prediction(store, key)?;

        let store: &TileStore = store;
        let prediction = prediction_of(store, key, made)?;
        let landing = strokes.iter().enumerate();
        for (n, planned) in landing {
            if planned.imprints.is_empty() {
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
                imprints: &planned.imprints,
                plan: &planned.plan,
                stroke: self.stroke.continued(store, planned.op),
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

/// Tile `key`'s pen-down positions as the oldest open stroke, of op `op`, reads them: the
/// store's ([`TileStore::pen_down_positions`]), its shape if it keeps one, else its positions as
/// stored.
///
/// # Errors
///
/// [`FutureError::ForeignPenDown`] if the store keeps a shape of the tile and `op` isn't its
/// current op, whose shapes it keeps; as [`TileStore::pen_down_positions`].
fn own_pen_down(store: &mut TileStore, op: OpId, key: TileKey) -> Result<Vec<Vec3>, SculptError> {
    if store.current_op() != Some(op) && store.pen_down_shape(&key).is_some() {
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
