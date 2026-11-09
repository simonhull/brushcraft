//! A tile's future: the tile as it will be once the open strokes still landing have landed,
//! computed now, without writing.
//!
//! A stroke started while earlier strokes still land reads exactly the surface they leave (the
//! stroke-while-landing spec, "What a new stroke reads"): each tile in its reach is the tile as
//! stored with every earlier open stroke's remaining dabs applied, in op order and placement
//! order. [`future_of`] makes that tile.
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
//!   oldest stroke of the chain, the store's ([`TileStore::pen_down_positions`]); for a later one,
//!   the future the strokes before it leave, whose positions are the tile's prediction with those
//!   corrections applied, the bits the store's evaluation of them gives, and so the pen-down shape
//!   that stroke reads (it is its surface) and lands on.
//! - **The tile's frames**, from its prediction: the level below, which no brush edits.
//! - **The stroke's state** ([`Stroke`]): each touched vertex's correction built unrounded across
//!   the stroke's dabs and rounded on write. For the stroke the session's [`Stroke`] last wrote
//!   ([`Remaining::op`]), its state as held; for any other, none, as its first dab will find it
//!   (the state starts afresh when the op changes), so each vertex's start is captured from the
//!   tile as the strokes before left it, the future it lands on.
//! - **The grants of that state**: which tiles hold state, are released or are written per dab
//!   ("Releasing tiles" on [`Stroke`]) is decided from every tile each dab edits, not from this
//!   tile alone. The future is made only where those decisions are known without landing: when the
//!   tiles the stroke's remaining dabs reach ([`Remaining::reach`]) that hold no state number at
//!   most the free buffers, no tile is released or denied, and every tile goes on with its state or
//!   takes fresh state at its first edit, which is what the replay assumes
//!   ([`crate::apply::Plan::Continuous`]). Past that, [`FutureError::StrokeState`]: the default
//!   reservation, 512 tiles, holds an ordinary stroke's 30 to 50 many times over.
//!
//! The tile's border vertices are held by its neighbours too, each written by its owner's state:
//! the replay carries a copy of each owner's state with the same arithmetic, as a run's tiles do,
//! so a future's border is the bits the owner's landing gives every copy.
//!
//! # What it writes
//!
//! Nothing a reader of the surface sees: no correction, no op, no change or stale-tile feed, no
//! pen-down shape, no stroke state. It reads the tile through the store's caches, which it may
//! fill as any read does: the tile's corrections and positions loaded, its prediction kept, the
//! pen-down positions of a changed tile made.

use std::borrow::Cow;
use std::collections::BTreeSet;

use brush::Imprint;
use geom::{CorrectionError, CorrectionTile, Prediction, TileKey, Vec3};
use store::{CacheError, OpId, TileStore};
use stroke::Dab;
use thiserror::Error;

use crate::apply::{
    PenDownTile, Plan, Run, SculptError, Stopped, TileInput, imprint_of, replay_tile,
};
use crate::state::Stroke;

/// One open stroke of a chain: its op, and its dabs not landed yet that bear on a tile.
#[derive(Debug, Clone, Copy)]
pub struct Remaining<'a> {
    /// The stroke's op: the op its dabs land in. The session's [`Stroke`] is this stroke's state
    /// if it last wrote in this op (see "Bit for bit" in the module notes).
    pub op: OpId,
    /// Its dabs not landed yet whose spheres touch the tile by its box at the stroke's pen-down,
    /// in placement order. More do no harm: a dab that reaches no vertex of the tile changes
    /// nothing.
    pub dabs: &'a [Dab],
    /// Every tile any of its dabs not landed yet touches by its box at the stroke's pen-down: the
    /// tiles they can edit, whose count decides whether the stroke's state has room for them
    /// (see [`FutureError::StrokeState`]).
    pub reach: &'a BTreeSet<TileKey>,
}

/// Why a tile's future couldn't be made. The new stroke then waits for the strokes before it to
/// land, as it would without futures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum FutureError {
    /// The stroke's remaining dabs reach more tiles holding no state than its state has buffers
    /// free: its landing may release a tile's state, or write one per dab, as decided from every
    /// tile its dabs edit, which a future of one tile doesn't know.
    #[error("stroke {stroke} of the chain reaches more tiles than its state has room for")]
    StrokeState {
        /// The stroke's place in the chain, the oldest 0.
        stroke: usize,
    },
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
    /// The tile's prediction couldn't be made, or a vertex the replay reads isn't where it
    /// looks (never, for a tile of the store's mesh).
    #[error("the tile's prediction or border can't be read")]
    Unreadable,
}

/// Tile `key` once every stroke of `chain` has landed its remaining dabs, made without writing
/// (see the module notes).
///
/// `chain` holds the open strokes, oldest first, each with its dabs not landed yet that bear on
/// `key` ([`Remaining`]); they land one after another, in this order. The oldest's pen-down shape
/// is the store's ([`TileStore::pen_down_positions`]): its op is the store's current op, or no
/// stroke has changed the store since the last stroke ended. `stroke` is the session's one stroke
/// state, the one [`crate::apply_dabs`] lands with. A chain with no dabs gives the tile as stored.
///
/// # Errors
///
/// [`SculptError::Dab`] for a dab the brush can't apply (its place in its stroke's slice), and
/// [`SculptError::Future`] with [`FutureError::StrokeState`] for a stroke whose state may not hold
/// its tiles, before anything is read; [`SculptError::Cache`] if the store can't read or pin the
/// tile, or keep its prediction; [`SculptError::Future`] with [`FutureError::Refused`] for a value
/// the tile can't hold, or [`FutureError::Unreadable`].
pub fn future_of(
    store: &mut TileStore,
    stroke: &Stroke,
    chain: &[Remaining<'_>],
    key: TileKey,
) -> Result<CorrectionTile, SculptError> {
    // Everything that can be refused without reading, first.
    let imprints: Result<Vec<Vec<Imprint>>, SculptError> = chain
        .iter()
        .map(|remaining| imprints_of(remaining.dabs))
        .collect();
    let imprints = imprints?;
    let checked = chain.iter().zip(&imprints).enumerate();
    for (n, (remaining, its)) in checked {
        let state = stroke.continued(store, remaining.op);
        if !its.is_empty() && !Stroke::has_room_for(state, store, remaining.reach) {
            return Err(FutureError::StrokeState { stroke: n }.into());
        }
    }

    let mut tile: CorrectionTile = store.correction(key)?.to_tile();
    if imprints.iter().all(Vec::is_empty) {
        return Ok(tile);
    }
    // The oldest stroke's pen-down positions, as its landing reads them.
    let first_pen_down: Option<Vec<Vec3>> = match imprints.first() {
        Some(its) if !its.is_empty() => Some(store.pen_down_positions(key)?.to_vec()),
        _ => None,
    };
    let _pins = store.pin(&[key])?;
    let prediction = prediction_of(store, key)?;

    let store: &TileStore = store;
    let landing = chain.iter().zip(&imprints).enumerate();
    for (n, (remaining, its)) in landing {
        if its.is_empty() {
            continue;
        }
        // A later stroke's pen-down shape of the tile is the future the strokes before it leave.
        let pen_down: Cow<'_, [Vec3]> = if n == 0
            && let Some(positions) = &first_pen_down
        {
            Cow::Borrowed(positions)
        } else {
            let view = tile.view();
            Cow::Owned(
                prediction
                    .apply(|i, j| view.get(i, j).unwrap_or(Vec3::ZERO))
                    .into_vec(),
            )
        };
        let run = Run {
            imprints: its,
            plan: &Plan::Continuous,
            stroke: stroke.continued(store, remaining.op),
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

/// Tile `key`'s prediction, its frames and predicted positions: the bits its evaluation, and so
/// a landing, uses.
///
/// The one the store keeps, kept now if it keeps none ([`TileStore::keep_predictions_of`]: the
/// tile, pinned by the caller, is about to be landed, and its landing reuses it); else made
/// afresh from its parents, made current for it ([`TileStore::frames_many`]), not kept.
///
/// # Errors
///
/// As [`TileStore::keep_predictions_of`] and [`TileStore::frames_many`];
/// [`FutureError::Unreadable`] if the parents aren't current to predict from (never, just after
/// they were made so).
fn prediction_of(store: &mut TileStore, key: TileKey) -> Result<Prediction, SculptError> {
    let kept = match store.keep_predictions_of(&[key]) {
        Ok(kept) => kept,
        Err(CacheError::Full | CacheError::OverBudget { .. }) => false,
        Err(e) => return Err(e.into()),
    };
    if kept && let Some(prediction) = store.kept_prediction(&key) {
        return Ok(prediction.clone());
    }
    store.frames_many(&[key])?;
    let parents = store.parent_slots(&key).ok_or(FutureError::Unreadable)?;
    store
        .predictor()
        .predict(&key, &parents)
        .ok_or_else(|| FutureError::Unreadable.into())
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;
