//! `Stroke`: one stroke from pen-down to pen-up, samples in and dabs out.

use std::vec::{Drain, IntoIter};

use crate::place::{Placer, Segment};
use crate::surface::{Projector, Surface};
use crate::symmetry::{Copies, Placed};
use crate::{Dab, Path, Sample, StrokeRandom, StrokeSettings};

/// Which stroke a dab belongs to ([`Dab::stroke`]), and the seed of its random values.
///
/// The caller numbers its strokes; a stroke replayed with the same id draws the same random
/// values ([`StrokeRandom`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct StrokeId(u32);

impl StrokeId {
    /// The stroke numbered `id`.
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// The number.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// One stroke, from pen-down ([`Stroke::begin`]) through each sample ([`Stroke::push`]) to
/// pen-up ([`Stroke::end`]): the pen's samples in, [`Dab`]s out.
///
/// Each sample goes through the [`Path`]'s smoothing, then placement (spacing along the surface,
/// the stroke's region, the frame and dynamics; see the `place` module's notes in the source),
/// then the [`Symmetry`](crate::Symmetry) copies. The camera and the surface are passed to each
/// call rather than held, so the caller can edit the surface between calls (the dabs it applies)
/// and move the camera mid-stroke: the brush's size is frozen at the first hit, so a zoom
/// doesn't change it.
///
/// **Deterministic:** the same settings, id, samples, camera and surface give bit-identical
/// dabs. That's what makes a stroke's recorded input meaningful.
#[derive(Debug, Clone)]
pub struct Stroke {
    /// The stroke's number, for its dabs.
    id: StrokeId,
    /// The screen path's smoothing.
    path: Path,
    /// The placement state.
    placer: Placer,
    /// The symmetry maps.
    copies: Copies,
    /// The latest brush point: where the next segment starts.
    last: Sample,
    /// Dabs placed by the current call, before their copies: reused from call to call.
    placed: Vec<Placed>,
    /// The current call's dabs, copies included: reused from call to call.
    dabs: Vec<Dab>,
}

impl Stroke {
    /// Starts a stroke at the pen's first sample, or gives `None` if nothing is under it: no
    /// stroke starts, and that isn't an error.
    ///
    /// Freezes the brush's world radius here (see [`SizeMode`](crate::SizeMode)): a screen
    /// radius is converted at the first hit's depth. Also `None` if the camera gives no ray
    /// through the sample, or a pixel size there that makes no positive, finite radius.
    ///
    /// The first dab lies at this hit; it comes out of the first [`Stroke::push`] (or
    /// [`Stroke::end`]), once the pen's movement gives it a direction.
    pub fn begin<P: Projector, S: Surface>(
        settings: &StrokeSettings,
        id: StrokeId,
        projector: &P,
        surface: &mut S,
        first: Sample,
    ) -> Option<Self> {
        let random = StrokeRandom::new(u64::from(id.get()));
        let placer = Placer::begin(settings, random, projector, surface, &first)?;
        let mut path = Path::new(*settings.path());
        let last = path.push(first);
        Some(Self {
            id,
            path,
            placer,
            copies: Copies::new(settings.symmetry()),
            last,
            placed: Vec::new(),
            dabs: Vec::new(),
        })
    }

    /// The stroke's number.
    #[must_use]
    pub const fn id(&self) -> StrokeId {
        self.id
    }

    /// The brush's world radius, frozen at the stroke's first hit; each dab's radius is this
    /// times its dynamics' size factor.
    #[must_use]
    pub const fn radius(&self) -> f64 {
        self.placer.radius()
    }

    /// Takes the pen's next sample and gives the dabs it places, copies included, in order: by
    /// sequence number, each dab's copies together, copy `0` first.
    ///
    /// The first call also gives the stroke's first dab, at its first hit.
    pub fn push<P: Projector, S: Surface>(
        &mut self,
        projector: &P,
        surface: &mut S,
        sample: Sample,
    ) -> Drain<'_, Dab> {
        let brush = self.path.push(sample);
        let segment = Segment {
            from: self.last,
            to: brush,
        };
        self.last = brush;
        self.placed.clear();
        self.placer
            .walk(projector, surface, &segment, &mut self.placed);
        self.placer.flush(&mut self.placed);
        self.expand();
        self.dabs.drain(..)
    }

    /// Ends the stroke at pen-up and gives its last dabs: the path is completed to the pen's last
    /// raw sample, whatever the smoothing held back, and a final dab placed there, so the tail
    /// is never lost.
    pub fn end<P: Projector, S: Surface>(
        mut self,
        projector: &P,
        surface: &mut S,
    ) -> IntoIter<Dab> {
        self.placed.clear();
        let tail = self.path.end().unwrap_or(self.last);
        let segment = Segment {
            from: self.last,
            to: tail,
        };
        self.placer
            .walk(projector, surface, &segment, &mut self.placed);
        self.placer.tail(&segment, &mut self.placed);
        self.expand();
        self.dabs.into_iter()
    }

    /// Copies this call's placed dabs through the symmetry into [`Stroke::dabs`].
    fn expand(&mut self) {
        self.dabs.clear();
        for placed in &self.placed {
            self.copies.expand(placed, self.id.get(), &mut self.dabs);
        }
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;
