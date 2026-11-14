//! The action a dab carries out on the vertices it reaches: how it moves one, and how far it can.
//!
//! One action so far, [`Draw`]. A dab carries no action of its own yet ([`stroke::Dab`] has no
//! tag for one), so every dab is draw's, and the code that lands dabs, previews them and bounds
//! them names [`Draw`] where it would otherwise ask the dab.
//!
//! **The move and its span are one trait's.** Every path that moves a vertex by a dab (the finest
//! work, the previews shown ahead of it) takes the move from [`Action::moved`]; the bound on
//! where a stroke's dabs still to land can take a tile, which a later stroke relies on to skip the
//! tiles its reads can't reach ([`dab_span`]; the stroke-while-landing spec, "What a new stroke
//! reads"), takes [`Action::span`]. A read that trusts a span too small reads a stale tile and
//! nothing notices, so the span isn't a separate function to remember: an action can't be
//! written without one, beside the move it bounds. When dabs come to say their action,
//! [`dab_span`] dispatches on it as the landing does.

use brush::{Imprint, draw_move};
use stroke::Dab;

use crate::apply::imprint_of;

/// How far a vertex can be moved along each axis.
///
/// The least of the move's components (each at most 0) and the most (each at least 0). Spans of
/// moves made one after another add up, axis by axis ([`add_span`]).
pub type Span = ([f64; 3], [f64; 3]);

/// Adds `part` to `sum`, axis by axis: the span of a move of span `sum` followed by one of span
/// `part`.
pub fn add_span(sum: &mut Span, part: Span) {
    let (low, high) = sum;
    let lows = low.iter_mut().zip(part.0);
    for (total, least) in lows {
        *total += least;
    }
    let highs = high.iter_mut().zip(part.1);
    for (total, most) in highs {
        *total += most;
    }
}

/// What an action does to a vertex a dab of it reaches, and the bound on it.
pub(crate) trait Action {
    /// The move of a vertex whose displacement by the dab `imprint` is `w` (the action's
    /// weighing of the vertex, [`brush::draw_imprint_vertex`] for draw): what its position
    /// changes by.
    fn moved(imprint: &Imprint, w: f32) -> [f32; 3];

    /// The span of every move the dab `imprint` can give a vertex: whatever the vertex and its
    /// weight, each component of [`Action::moved`] lies within it, the rounding of the `f32`
    /// arithmetic included. Never too small: a later stroke skips the tiles it says a dab can't
    /// take into a read's way.
    fn span(imprint: &Imprint) -> Span;
}

/// The draw action: every vertex in the dab's sphere moves along the dab's normal, by the dab's
/// strength weighed by its falloff.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Draw;

impl Draw {
    /// How far past the exact bound the `f32` arithmetic of a move may land, as a share of the
    /// strength: the falloff's weight may round a few `1e-7` out of `0..=1`, and so may each
    /// product.
    const ROUNDING: f64 = 1e-6;
}

impl Action for Draw {
    #[inline]
    fn moved(imprint: &Imprint, w: f32) -> [f32; 3] {
        draw_move(imprint, w)
    }

    /// Draw's displacement of a vertex is the dab's strength times a weight between 0 and 1,
    /// and its move that displacement along the dab's normal: never against the strength's
    /// sign, never off the normal. So a move lies between nothing and the dab's whole move
    /// ([`Action::moved`] at the full strength, the very function, so the direction can't
    /// differ), component by component; the span is that, a little apart for the rounding.
    fn span(imprint: &Imprint) -> Span {
        let whole = Self::moved(imprint, imprint.strength).map(f64::from);
        let slack = Self::ROUNDING * f64::from(imprint.strength.abs());
        (
            whole.map(|part| part.min(0.0) - slack),
            whole.map(|part| part.max(0.0) + slack),
        )
    }
}

/// How far `dab` can move a vertex once it lands, along each axis: its action's span
/// (draw's, the one action a dab can be of; an action says its span beside its move).
///
/// A stroke's dabs still to land move a vertex by the sum of such moves ([`add_span`]): what a
/// later stroke bounds a tile's future by, before it is made.
///
/// A dab the brush can't apply (its landing fails at it, so it moves nothing) spans its strength
/// every way, and one whose strength isn't finite spans everything: a bound that can't be said
/// is no bound, and the tiles it reaches are all asked for.
#[must_use]
pub fn dab_span(dab: &Dab) -> Span {
    if let Ok(imprint) = imprint_of(dab) {
        return Draw::span(&imprint);
    }
    let strength = f64::from(dab.strength.abs());
    let every_way = if strength.is_finite() {
        strength
    } else {
        f64::INFINITY
    };
    ([-every_way; 3], [every_way; 3])
}
