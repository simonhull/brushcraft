//! Picking: the point of the surface under the cursor or the pen.
//!
//! A CPU raycast against the store's spatial index (the M1 spec's "Picking": no GPU readback,
//! so no frame of lag, and it works headless).
//!
//! **The ray** ([`PickRay::through`]) goes from the view's near plane through a point of the
//! viewport, given in physical pixels from its top-left corner, as winit reports the pointer
//! (`WindowEvent::PointerMoved`'s `PhysicalPosition`) and as the frame's [`Size`] counts the
//! surface. Both are physical, so a high-DPI display's scale factor needs no conversion: it only
//! sizes the overlay's text ([`crate::frame::Owner::set_scale`]), never the scene. The position
//! is continuous: `(0, 0)` is the viewport's top-left corner and pixel `(i, j)`'s centre is
//! `(i + 0.5, j + 0.5)`, the same convention [`View::pixel`] projects to. The ray starts on the
//! near plane, not at the eye, because nothing nearer is drawn.
//!
//! **The hit** ([`Picker::pick`], [`Hit`]) is where [`SpatialIndex::raycast`] says the ray first
//! meets the surface: the index's tiles nearest-first, each evaluated through the store and
//! intersected quad by quad. The index is over the **finest level** (the store's display layer is
//! built from it), while the screen draws the level choice's cut, coarser where it is far. The
//! two differ by at most what the level choice allows on screen, about one pixel (its error
//! rule), and the brush works on the finest surface too, so the hit is the one it wants.
//! Positions are `f32`, as the mesh is: the ray is rounded to `f32` once, from the near plane
//! (close to the surface in a close view, which keeps the rounding small against the pixel).
//!
//! **When it runs:** on the owner thread, in the frame loop's input phase, after the input has
//! moved the camera ([`Picker::update`]), and only when the pointer or the view (camera or frame
//! size) changed since the last pick. Pointer motion that changes nothing else wants a frame of
//! its own ([`crate::frame::schedule::Reason::Pick`]), which only shades the last scene again
//! under the new overlay line. The pen picks too: its stream ([`crate::pen`]) turns hover into
//! pointer motion, which reaches the frame loop like the mouse's.
//!
//! **Cost:** a pick takes candidate tiles nearest-first until none left can be nearer than its
//! hit, usually one or two (more at a silhouette), and intersects every quad of each: a tile is
//! 64 × 64 quads at any level, so a warm pick costs the same at level 9 as at level 7. Measured
//! by `pick::tests::picks_are_timed_on_a_reopened_project` (a 24-face sphere at level 7, saved
//! and reopened so the index loads from stored bounds and the first picks find no tile cached;
//! the test build, `opt-level = 1`, on the development desktop): a warm pick **about 0.35 ms**, the
//! first cold one **about 4 ms**, later cold ones about 1 ms. Cold picks evaluate their tiles;
//! a level-9 tile of the benchmark project evaluates in about 0.3 ms in a release build (the M0
//! benchmark), plus its pages if they aren't in memory. So on the benchmark project a pick is
//! **a fraction of a millisecond warm, about a millisecond cold**, release build: well inside a
//! frame's input phase. Hovering stays warm, since the tiles under the pointer stay cached.
//! Not yet measured in a release build on the benchmark project (Task 7's scripted runs can).
//!
//! What M1a4's brush calls is [`Picker::pick`]: a pick for any view and point, uncached. Nothing
//! edits the surface yet, so the index is never refitted here and a still pointer over a still
//! view isn't picked again; once the brush edits, it refits the index
//! ([`SpatialIndex::update`]) and the overlay's pick should be redone after an edit.

use std::fmt;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result};
use geom::TileKey;
use render::camera::{Vec3, View};
use render::surface::Size;
use store::{Project, Ray, SpatialIndex, TileStore};

use crate::camera::Pixel;

/// A ray from the view through a point of the viewport: unit length, starting on the near plane.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PickRay {
    /// Where it starts: on the near plane.
    origin: Vec3,
    /// Which way it goes, unit length.
    direction: Vec3,
    /// How far `origin` is from the eye.
    start: f64,
}

impl PickRay {
    /// The ray from `view`'s near plane through `at`, a point of its viewport in physical pixels
    /// from the top-left corner (see the module notes); `None` if `at` isn't finite.
    ///
    /// A point `p` pixels right of the centre is `p / pixels_per_unit` units right per unit of
    /// depth, which is how [`View::pixel`] projects it, so a point on the ray projects back to
    /// `at`.
    pub(crate) fn through(view: &View, at: Pixel) -> Option<Self> {
        if !(at.x.is_finite() && at.y.is_finite()) {
            return None;
        }
        let (width, height) = dimensions(view.size());
        let per_unit = view.pixels_per_unit();
        let across = (at.x - width / 2.0) / per_unit;
        let down = (height / 2.0 - at.y) / per_unit;
        let forward = view.forward();
        let direction = (forward + view.right() * across + view.up() * down).normalized()?;
        // The near plane is `near` along `forward`; the ray reaches it after `near / cos θ`.
        let start = view.near() / direction.dot(forward);
        Some(Self {
            origin: view.eye() + direction * start,
            direction,
            start,
        })
    }

    /// Where it starts: on the near plane.
    #[cfg(test)]
    pub(crate) const fn origin(&self) -> Vec3 {
        self.origin
    }

    /// Which way it goes, unit length.
    #[cfg(test)]
    pub(crate) const fn direction(&self) -> Vec3 {
        self.direction
    }

    /// The ray in the store's `f32` terms; `None` if it doesn't fit them (a view so far out that
    /// its near plane overflows an `f32`).
    fn store_ray(&self) -> Option<Ray> {
        Ray::new(to_f32(self.origin), to_f32(self.direction))
    }
}

/// `size`'s width and height, as `f64`.
fn dimensions(size: Size) -> (f64, f64) {
    (f64::from(size.width()), f64::from(size.height()))
}

/// `v` rounded to `f32`, as the store's rays and the mesh are.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the store's ray is f32, as the mesh is: rounded once, here"
)]
fn to_f32(v: Vec3) -> geom::Vec3 {
    geom::Vec3::new(v.x as f32, v.y as f32, v.z as f32)
}

/// Where a pick's ray first meets the surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Hit {
    /// The point hit, in world space.
    pub(crate) position: Vec3,
    /// The finest-level tile it lies on: its face, level and place on the face.
    pub(crate) key: TileKey,
    /// How far the point is from the eye, in world units.
    pub(crate) distance: f64,
}

/// What the last pick found.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Outcome {
    /// No pointer over the window (or no view to pick in): nothing was picked.
    NoPointer,
    /// The ray under the pointer misses the surface.
    Miss,
    /// The ray under the pointer hits it.
    Hit(Hit),
}

/// The last pick: what it found, and how long it took.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Report {
    /// Its number: picks made so far, from 1 (0 before the first). The overlay shows it, so it
    /// shows that a still pointer over a still view doesn't pick again.
    pub(crate) number: u64,
    /// What it found.
    pub(crate) outcome: Outcome,
    /// How long it took.
    pub(crate) took: Duration,
}

impl Default for Report {
    /// Nothing picked yet: no pointer, no time.
    fn default() -> Self {
        Self {
            number: 0,
            outcome: Outcome::NoPointer,
            took: Duration::ZERO,
        }
    }
}

impl fmt::Display for Report {
    /// The overlay's pick line, after its label: the pick's number, then the tile hit (face,
    /// level, column and row), the point, its distance from the eye and the pick's time; or
    /// that it missed, or that there is no pointer.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let took = self.took.as_secs_f64() * 1e3;
        write!(f, "#{}  ", self.number)?;
        match self.outcome {
            Outcome::NoPointer => f.write_str("no pointer over the window"),
            Outcome::Miss => write!(f, "no hit  ({took:.2} ms)"),
            Outcome::Hit(Hit {
                position,
                key,
                distance,
            }) => write!(
                f,
                "{} L{} tile ({}, {})  at ({:.3}, {:.3}, {:.3})  distance {distance:.3}  \
                 ({took:.2} ms)",
                key.face(),
                key.level(),
                key.tx(),
                key.ty(),
                position.x,
                position.y,
                position.z
            ),
        }
    }
}

/// The overlay's pick line, after its label, for `report`; `None` when there is no picker.
pub(crate) fn summary(report: Option<&Report>) -> String {
    report.map_or_else(|| "off (no spatial index)".to_owned(), Report::to_string)
}

/// What a pick is asked for: a view, and the pointer in it.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Asked {
    /// The view; `None` if the camera gave none.
    view: Option<View>,
    /// The pointer, in physical pixels; `None` while it is outside the window.
    at: Option<Pixel>,
}

/// What the last pick was asked for, so a frame re-picks only when that changed.
#[derive(Debug, Clone, Copy, Default)]
struct Since {
    /// What the last pick was asked for; `None` before the first.
    last: Option<Asked>,
}

impl Since {
    /// Whether `asked` differs from what the last pick was asked for, and remembers it.
    fn changed(&mut self, asked: &Asked) -> bool {
        if self.last.as_ref() == Some(asked) {
            return false;
        }
        self.last = Some(*asked);
        true
    }
}

/// Picks the surface under the pointer: the spatial index over the finest level, the pointer,
/// and the last pick (see the module notes).
#[derive(Debug)]
pub(crate) struct Picker {
    /// The spatial index, over the finest level.
    index: SpatialIndex,
    /// The pointer, in physical pixels; `None` while it is outside the window.
    pointer: Option<Pixel>,
    /// What the last pick was asked for.
    since: Since,
    /// What the last pick found.
    report: Report,
}

impl Picker {
    /// A picker over `index`, the pointer outside the window.
    pub(crate) fn new(index: SpatialIndex) -> Self {
        Self {
            index,
            pointer: None,
            since: Since::default(),
            report: Report::default(),
        }
    }

    /// A picker over `project`'s finest level, the level its display layer is built from,
    /// with the index loaded from the bounds the project keeps (on the benchmark project, about
    /// 0.1 s and none evaluated; a project with no stored bounds evaluates every tile); `None`
    /// if the project keeps no display layer.
    ///
    /// # Errors
    ///
    /// If the index can't be loaded: a tile whose bounds aren't stored can't be evaluated.
    pub(crate) fn load(project: &mut Project) -> Result<Option<Self>> {
        let Some(level) = project.store().display_finest() else {
            return Ok(None);
        };
        let (index, _) = project
            .load_index(level, |_, _| {})
            .with_context(|| format!("loading the spatial index at level {level}"))?;
        Ok(Some(Self::new(index)))
    }

    /// The pointer moved to `at` (`None`: it left the window); returns whether that changed it,
    /// so the overlay's line needs a frame.
    pub(crate) fn point(&mut self, at: Option<Pixel>) -> bool {
        std::mem::replace(&mut self.pointer, at) != at
    }

    /// Where the ray through `at` in `view` first meets the surface, at the index's (finest)
    /// level, evaluating candidate tiles through `store`; `None` if it misses, or if the ray
    /// can't be built (a point that isn't finite). Always picks: what the brush calls.
    ///
    /// # Errors
    ///
    /// If a candidate tile can't be evaluated.
    pub(crate) fn pick(
        &self,
        store: &mut TileStore,
        view: &View,
        at: Pixel,
    ) -> Result<Option<Hit>> {
        let Some(ray) = PickRay::through(view, at) else {
            return Ok(None);
        };
        let Some(store_ray) = ray.store_ray() else {
            return Ok(None);
        };
        let Some(hit) = self
            .index
            .raycast(store, &store_ray)
            .context("picking by the spatial index")?
        else {
            return Ok(None);
        };
        // The point along the ray as the store measured it: from its f32 origin, along its f32
        // direction (unit length to rounding, so `t` is in world units).
        let origin = Vec3::from_f32(store_ray.origin());
        let direction = Vec3::from_f32(store_ray.direction());
        let t = f64::from(hit.t);
        Ok(Some(Hit {
            position: origin + direction * t,
            key: hit.key,
            distance: ray.start + t * direction.length(),
        }))
    }

    /// Picks under the pointer in `view` (`None`: the camera gave no view), unless neither has
    /// changed since the last pick; returns whether it picked. The frame loop calls it in its
    /// input phase, once the input has moved the camera.
    ///
    /// # Errors
    ///
    /// As [`Picker::pick`].
    pub(crate) fn update(&mut self, store: &mut TileStore, view: Option<&View>) -> Result<bool> {
        let asked = Asked {
            view: view.copied(),
            at: self.pointer,
        };
        if !self.since.changed(&asked) {
            return Ok(false);
        }
        let started = Instant::now();
        let outcome = match (view, self.pointer) {
            (Some(view), Some(at)) => match self.pick(store, view, at)? {
                Some(hit) => Outcome::Hit(hit),
                None => Outcome::Miss,
            },
            (None, _) | (_, None) => Outcome::NoPointer,
        };
        self.report = Report {
            number: self.report.number + 1,
            outcome,
            took: started.elapsed(),
        };
        Ok(true)
    }

    /// What the last pick found, for the overlay.
    pub(crate) const fn report(&self) -> &Report {
        &self.report
    }

    /// Picks [`Picker::update`] has made.
    #[cfg(test)]
    pub(crate) const fn picks(&self) -> u64 {
        self.report.number
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;
