//! The view: a camera that turns and zooms about a pivot outside the model's boundary, and the
//! arcball that turns a drag into a rotation.
//!
//! The camera looks along its orientation from its eye, through [`Camera::target`]
//! [`Camera::distance`] ahead. Three things shape what navigation does with it (the navigation
//! spec of 2026-11-03):
//!
//! - **The pivot** ([`Camera::pivot`]) is the point orbit and zoom turn and scale about: where the
//!   last stroke began, the surface under the pointer as a navigation drag starts over the model,
//!   or the model's centre once framed ([`crate::frame`] sets it, the latest event winning).
//!   Setting it moves no pixel, unless the boundary must grow to hold it (below).
//! - **Orbit** turns the eye and the view's orientation together about the pivot, so the pivot
//!   stays on its pixel. **Pan** slides the view in its plane so the pivot's depth keeps pace with
//!   the pointer. **Zoom** scales the pivot's apparent distance (the eye's distance to it over the
//!   magnification) by the step's factor.
//! - **The boundary** ([`Boundary`]) is a sphere around the model's bounds, [`Boundary::MARGIN`]
//!   larger, that the eye never enters. Zooming in moves the eye along its line to the pivot until
//!   it meets the boundary, then **magnifies** instead: the eye stops, and the image is enlarged
//!   about the pivot's pixel, which stays put. Zooming out widens it back before the eye moves
//!   again, and every step is the exact inverse of its opposite (to rounding). After an orbit or a
//!   pan the eye is put back on its line to the pivot at the distance that keeps the pivot's
//!   apparent size: pushed out of the boundary if the move carried it in, the image magnified to
//!   make up; drawn in to the boundary, the magnification lowered, if the image was magnified and
//!   the move carried the eye off it. So **the image is magnified only while the eye is on the
//!   boundary** (with no boundary known, at [`Camera::MIN_DISTANCE`] from the pivot), and no orbit,
//!   pan or zoom leaves the eye inside it. The perspective freezes there: while magnified it is the
//!   boundary's, flatter than a true close-up.
//! - **The boundary holds every pivot**, as it holds the model's points, a margin inside: one set
//!   outside it (a surface raised past the margin, picked before the boundary is fitted again)
//!   grows it ([`Boundary::holding`]), so the eye's line to the pivot leaves the boundary at one
//!   point. It grows, never shrinks, until the model is framed again.
//!
//! **Magnification is the lens's, not the eye's:** a magnification factor `m` and a shift `s`
//! carry the unmagnified view's image plane (right and up over depth) `t` to `m · t + s`, which
//! [`Camera::view`] gives the renderer as a narrower field with its principal point moved
//! ([`View::magnified`]). The eye, every depth and the near plane stay where the eye's true place
//! puts them; [`View::pixels_per_unit`] grows by `m`, so everything that measures the view in
//! pixels (the level choice, the brush's screen radius, picking's rays through
//! [`View::through`]) measures the magnified image's. A zoom keeps the pivot's pixel by changing
//! `s` with `m`. **An unmagnified image is never shifted:** if the pivot moved on the screen
//! while magnified (a pan, a new pivot), the shift left as the magnification falls back to 1 is
//! folded into the orientation, the view turned about the eye so the pivot keeps its pixel and
//! the line of sight meets the screen's centre again. The pivot's pixel is kept exactly; the rest
//! of the image steps by the difference between a shift and a turn, small for a small shift.
//!
//! **The limits scale with the model and the `f32` the GPU draws in:** the eye goes out to
//! [`Camera::MAX_DISTANCE`] model radii (for a model larger than 1), so any model frames
//! unmagnified; and the magnification stops where a pixel at the pivot, in a 4K-tall view,
//! still spans [`Camera::PIXEL_ULPS`] `f32` steps of the eye's and the pivot's coordinates
//! (about 70 times for a unit model with the eye on its boundary), beyond which the image would
//! move in steps and the pick ray's own rounding would disagree with it.
//!
//! **Known limit: the surface can pass the eye during a stroke.** The view holds still while a
//! stroke is drawn (its record keeps one camera), and the boundary is fitted again only once the
//! brush is idle. A strong stroke on a magnified close-up can raise the surface by more than the
//! margin, up to and past the eye: the part past it isn't drawn and the brush may work on a
//! surface the eye is inside, until the stroke has landed and the grown boundary pushes the eye
//! back out along its line to the pivot.
//!
//! A scripted camera's pose ([`Camera::orbiting`], [`Camera::slide`]) is set directly: no pivot
//! of its own (the target), no boundary, never magnified, its view exactly the plain one.
//!
//! The arcball is Shoemake's: a pointer position maps to a point on a unit sphere centred on
//! the viewport (outside the sphere's silhouette, to its rim), and a drag from one point to
//! another is the rotation carrying the first to the second, so dragging round the rim rolls
//! the view and a drag back to where it began undoes itself.
//!
//! Math is `f64`: the camera is a handful of numbers, and accumulated drags then stay unit
//! length for far longer.
//!
//! The renderer takes the view as `render::camera::View` (`f64`, handed to the GPU
//! eye-relative), which [`Camera::view`] makes from this camera's eye, target, up and lens.
//! `render` has a camera of its own with the older model (orbiting its target, no pivot or
//! boundary); moving this one there, leaving this module the arcball and what navigation does
//! with it, is still to do. `geom::Vec3` is `f32` and stays the mesh's.

use std::ops::{Add, Mul, Neg, Sub};

use render::camera::{NEAR_FRACTION, Vec3 as RenderVec3, View};
use render::surface::Size;

/// A vector in 3D, `f64`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct Vec3 {
    /// The x component.
    pub(crate) x: f64,
    /// The y component.
    pub(crate) y: f64,
    /// The z component.
    pub(crate) z: f64,
}

impl Vec3 {
    /// The vector `(x, y, z)`.
    pub(crate) const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// The dot product.
    pub(crate) fn dot(self, o: Self) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    /// The cross product, `self × o`.
    pub(crate) fn cross(self, o: Self) -> Self {
        Self::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }

    /// The length.
    pub(crate) fn length(self) -> f64 {
        self.dot(self).sqrt()
    }
}

impl Add for Vec3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Sub for Vec3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Mul<f64> for Vec3 {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl Neg for Vec3 {
    type Output = Self;

    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}

/// A rotation, as a unit quaternion `w + xi + yj + zk`.
///
/// Made only as a unit quaternion ([`Rotation::IDENTITY`], [`Rotation::between`]) and kept one
/// by [`Rotation::then`], which renormalizes, so drift from a long drag never skews the view.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Rotation {
    /// The scalar part.
    w: f64,
    /// The vector part.
    v: Vec3,
}

impl Rotation {
    /// No rotation.
    pub(crate) const IDENTITY: Self = Self {
        w: 1.0,
        v: Vec3::new(0.0, 0.0, 0.0),
    };

    /// The shortest rotation carrying the unit vector `from` to the unit vector `to`: about
    /// their common normal, by the angle between them. Opposite vectors have no shortest
    /// rotation; they give a half turn about an axis perpendicular to `from`.
    pub(crate) fn between(from: Vec3, to: Vec3) -> Self {
        // The half-angle form: (1 + cos θ, sin θ · axis), normalized, is the rotation by θ.
        let w = 1.0 + from.dot(to);
        if w <= f64::EPSILON {
            let side = if from.x.abs() < 0.9 {
                Vec3::new(1.0, 0.0, 0.0)
            } else {
                Vec3::new(0.0, 1.0, 0.0)
            };
            return Self {
                w: 0.0,
                v: from.cross(side),
            }
            .normalized();
        }
        Self {
            w,
            v: from.cross(to),
        }
        .normalized()
    }

    /// The rotation by `angle` radians about the unit vector `axis`, counter-clockwise looking
    /// down the axis.
    pub(crate) fn about(axis: Vec3, angle: f64) -> Self {
        let (sin, cos) = (angle / 2.0).sin_cos();
        Self {
            w: cos,
            v: axis * sin,
        }
        .normalized()
    }

    /// The rotation undoing this one.
    pub(crate) fn inverse(self) -> Self {
        Self {
            w: self.w,
            v: -self.v,
        }
    }

    /// This rotation, then `next`: the product `next · self`, renormalized.
    pub(crate) fn then(self, next: Self) -> Self {
        Self {
            w: next.w * self.w - next.v.dot(self.v),
            v: self.v * next.w + next.v * self.w + next.v.cross(self.v),
        }
        .normalized()
    }

    /// `p` rotated.
    pub(crate) fn apply(self, p: Vec3) -> Vec3 {
        // p + 2w(v × p) + 2v × (v × p), for a unit quaternion.
        let t = self.v.cross(p) * 2.0;
        p + t * self.w + self.v.cross(t)
    }

    /// The same rotation scaled back to unit length (a zero quaternion, which only rounding
    /// from a unit one can't reach, becomes the identity).
    fn normalized(self) -> Self {
        let length = (self.w * self.w + self.v.dot(self.v)).sqrt();
        if length <= f64::EPSILON {
            return Self::IDENTITY;
        }
        Self {
            w: self.w / length,
            v: self.v * length.recip(),
        }
    }
}

/// A point in the viewport, in physical pixels from its top-left corner.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct Pixel {
    /// Pixels right of the left edge.
    pub(crate) x: f64,
    /// Pixels below the top edge.
    pub(crate) y: f64,
}

/// The model's boundary: a sphere around its bounds, a [`Boundary::MARGIN`] larger, that the
/// eye never enters (see the module notes).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Boundary {
    /// Its centre.
    centre: Vec3,
    /// Its radius, positive and finite.
    radius: f64,
}

impl Boundary {
    /// How far the boundary lies outside the model's bounds, as a fraction of their radius.
    ///
    /// Small, so the perspective freezes close to the surface (a unit sphere's eye stops 0.05
    /// above it), yet clear of the surface a stroke raises while the view holds still: the
    /// boundary is fitted again only between strokes.
    pub(crate) const MARGIN: f64 = 0.05;

    /// The boundary around a model held by the sphere of `radius` about `centre`; `None` unless
    /// both are finite and `radius` is positive.
    pub(crate) fn around(centre: Vec3, radius: f64) -> Option<Self> {
        let finite = centre.x.is_finite() && centre.y.is_finite() && centre.z.is_finite();
        let radius = radius * (1.0 + Self::MARGIN);
        (finite && radius > 0.0 && radius.is_finite()).then_some(Self { centre, radius })
    }

    /// Its centre.
    #[cfg(test)]
    pub(crate) const fn centre(&self) -> Vec3 {
        self.centre
    }

    /// Its radius.
    #[cfg(test)]
    pub(crate) const fn radius(&self) -> f64 {
        self.radius
    }

    /// The smallest sphere holding both this boundary and `other`.
    pub(crate) fn union(self, other: Self) -> Self {
        let apart = other.centre - self.centre;
        let gap = apart.length();
        if gap + other.radius <= self.radius {
            return self;
        }
        if gap + self.radius <= other.radius {
            return other;
        }
        // Neither holds the other: the sphere across both, from the far side of each.
        let radius = (gap + self.radius + other.radius) / 2.0;
        Self {
            centre: self.centre + apart * ((radius - self.radius) / gap),
            radius,
        }
    }

    /// This boundary grown, if it must be, to hold `point` as it holds the model's points: the
    /// sphere of the model's points it was made around ([`Boundary::around`], its radius over 1
    /// plus [`Boundary::MARGIN`]) grown to the smallest holding `point` too, then the margin
    /// added again. Itself if that sphere holds `point` already.
    pub(crate) fn holding(self, point: Vec3) -> Self {
        let model = self.radius / (1.0 + Self::MARGIN);
        let apart = point - self.centre;
        let gap = apart.length();
        if gap <= model || !gap.is_finite() {
            return self;
        }
        let grown = f64::midpoint(model, gap);
        Self {
            centre: self.centre + apart * ((grown - model) / gap),
            radius: grown * (1.0 + Self::MARGIN),
        }
    }

    /// How far from `origin` along the unit vector `direction` a ray last leaves the sphere;
    /// `None` if it doesn't meet it at a positive distance.
    fn exit(self, origin: Vec3, direction: Vec3) -> Option<f64> {
        let from_centre = origin - self.centre;
        let along = from_centre.dot(direction);
        let squared = along * along - from_centre.dot(from_centre) + self.radius * self.radius;
        let exit = -along + squared.max(0.0).sqrt();
        (squared >= 0.0 && exit > 0.0).then_some(exit)
    }
}

/// How the view's image is magnified (see the module notes): a point the unmagnified view shows
/// at `t` on its image plane (right and up, over its depth) shows at `factor · t + shift`.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Lens {
    /// The magnification, at least 1.
    factor: f64,
    /// The image's shift, right and up, in units of the image plane at unit depth.
    shift: [f64; 2],
}

impl Lens {
    /// No magnification.
    const NONE: Self = Self {
        factor: 1.0,
        shift: [0.0, 0.0],
    };
}

/// The camera: where it looks, from how far, which way is up, the pivot it turns and zooms
/// about, and the model's boundary it stays out of.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Camera {
    /// A point on the line of sight, [`Camera::distance`] ahead of the eye: the point the
    /// scripted paths look at, and what the near plane is measured to.
    target: Vec3,
    /// How far the eye is from the target; always within [`Camera::MIN_DISTANCE`] and
    /// [`Camera::MAX_DISTANCE`].
    distance: f64,
    /// The view's orientation: carries the view's axes (x right, y up, z towards the eye) into
    /// the world.
    orientation: Rotation,
    /// The vertical field of view, unmagnified, in radians.
    fov_y: f64,
    /// The point orbit and zoom turn and scale about (see the module notes).
    pivot: Vec3,
    /// How the image is magnified.
    lens: Lens,
    /// The model's boundary, once known: the eye stays out of it.
    boundary: Option<Boundary>,
}

impl Default for Camera {
    /// Looking at the origin down −z from 3 units away, y up: a unit sphere, as the benchmark
    /// project is, fills about two thirds of the view's height. The pivot is the origin, the
    /// image unmagnified, and no boundary is known.
    fn default() -> Self {
        Self {
            target: Vec3::default(),
            distance: 3.0,
            orientation: Rotation::IDENTITY,
            fov_y: 45_f64.to_radians(),
            pivot: Vec3::default(),
            lens: Lens::NONE,
            boundary: None,
        }
    }
}

impl Camera {
    /// The farthest the eye goes from the target, or from the pivot, for a model of radius 1 or
    /// less (or none known): a larger model's limit is this many of its boundary's radii
    /// ([`Camera::far`]), so framing any model leaves it unmagnified.
    pub(crate) const MAX_DISTANCE: f64 = 1e4;
    /// The closest the eye comes to the target, or to the pivot with no boundary known.
    pub(crate) const MIN_DISTANCE: f64 = 1e-4;
    /// How many `f32` rounding steps (ulps) of the drawn positions' magnitude a magnified pixel
    /// spans at least, at the pivot's depth ([`Camera::most_magnification`]).
    pub(crate) const PIXEL_ULPS: f64 = 4.0;
    /// The tallest viewport the magnification limit is made for, in pixels: a 4K display's
    /// ([`Camera::most_magnification`]). A shorter one gets the same limit, its pixels coarser.
    pub(crate) const TALLEST_VIEWPORT: f64 = 2160.0;

    /// The default camera ([`Camera::default`]: looking at the origin, y up) turned to look from
    /// `yaw` radians about the world's y axis (counter-clockwise from above, from +z towards +x)
    /// and `pitch` radians above the horizon, `distance` away (clamped to
    /// [`Camera::MIN_DISTANCE`]..[`Camera::MAX_DISTANCE`]): a scripted camera's pose
    /// ([`crate::script`]), set directly, with no boundary.
    pub(crate) fn orbiting(yaw: f64, pitch: f64, distance: f64) -> Self {
        // The view's own x axis is the world's x before the yaw: pitch about it first (the eye
        // rises towards +y, so the view turns by −pitch), then yaw about the world's up.
        let tilt = Rotation::about(Vec3::new(1.0, 0.0, 0.0), -pitch);
        let turn = Rotation::about(Vec3::new(0.0, 1.0, 0.0), yaw);
        Self {
            orientation: tilt.then(turn),
            distance: distance.clamp(Self::MIN_DISTANCE, Self::MAX_DISTANCE),
            ..Self::default()
        }
    }

    /// The point on the line of sight [`Camera::distance`] ahead of the eye.
    pub(crate) const fn target(&self) -> Vec3 {
        self.target
    }

    /// How far the eye is from the target.
    #[cfg(test)]
    pub(crate) const fn distance(&self) -> f64 {
        self.distance
    }

    /// The vertical field of view, unmagnified, in radians.
    pub(crate) const fn fov_y(&self) -> f64 {
        self.fov_y
    }

    /// The vertical field of view as magnified: the unmagnified one narrowed by
    /// [`Camera::magnification`].
    #[cfg(test)]
    pub(crate) fn magnified_fov_y(&self) -> f64 {
        2.0 * ((self.fov_y / 2.0).tan() / self.lens.factor).atan()
    }

    /// The point orbit and zoom turn and scale about.
    pub(crate) const fn pivot(&self) -> Vec3 {
        self.pivot
    }

    /// How many times the image is magnified: 1 unless the eye is at the boundary.
    pub(crate) const fn magnification(&self) -> f64 {
        self.lens.factor
    }

    /// The magnified image's shift, right and up, in units of the unmagnified image plane at
    /// unit depth (see the module notes): 0 whenever the image isn't magnified.
    pub(crate) const fn shift(&self) -> [f64; 2] {
        self.lens.shift
    }

    /// The model's boundary, once known.
    #[cfg(test)]
    pub(crate) const fn boundary(&self) -> Option<Boundary> {
        self.boundary
    }

    /// Where the eye is.
    pub(crate) fn eye(&self) -> Vec3 {
        self.target + self.axis(Vec3::new(0.0, 0.0, 1.0)) * self.distance
    }

    /// The view's `axis` (x right, y up, z towards the eye) in world space.
    pub(crate) fn axis(&self, axis: Vec3) -> Vec3 {
        self.orientation.apply(axis)
    }

    /// Makes `pivot`, a point of the model, the point orbit and zoom turn and scale about. Moves
    /// no pixel: only the next orbit or zoom turns or scales about it.
    ///
    /// The boundary is grown to hold it as it holds the model's points, a [`Boundary::MARGIN`]
    /// inside ([`Boundary::holding`]), so every pivot is inside the boundary; a surface raised
    /// past the boundary since it was fitted (a pick there before the frame loop fits it again)
    /// grows it. Only then can a pixel move: if the grown boundary takes in the eye, the eye is
    /// held out of it as [`Camera::orbit`] does.
    pub(crate) fn set_pivot(&mut self, pivot: Vec3) {
        self.pivot = pivot;
        if let Some(boundary) = self.boundary {
            let holding = boundary.holding(pivot);
            if holding != boundary {
                self.boundary = Some(holding);
                self.settle(self.apparent());
            }
        }
    }

    /// Takes `boundary` as the model's, grown to hold the one known before (if any) and the
    /// pivot, and keeps the eye out of it as [`Camera::orbit`] does.
    ///
    /// The boundary only grows until the next [`Camera::frame`], and always holds the pivot
    /// ([`Camera::set_pivot`]): the eye keeps to its line to a pivot inside it, which leaves the
    /// boundary at exactly one point. A model sculpted smaller leaves the eye a little farther
    /// out.
    pub(crate) fn grow_boundary(&mut self, boundary: Boundary) {
        let grown = self
            .boundary
            .map_or(boundary, |known| known.union(boundary));
        self.boundary = Some(grown.holding(self.pivot));
        self.settle(self.apparent());
    }

    /// Frames the model held by the sphere of `radius` about `centre` in a viewport of `size`:
    /// the view keeps its orientation, its pivot and target move to the centre, the image is
    /// unmagnified, and the eye backs off until the sphere fits the narrower of the view's two
    /// fields. The boundary is fitted afresh around the sphere. Nothing, if the sphere isn't
    /// one ([`Boundary::around`]).
    pub(crate) fn frame(&mut self, centre: Vec3, radius: f64, size: Size) {
        let Some(boundary) = Boundary::around(centre, radius) else {
            return;
        };
        let half = self.fov_y / 2.0;
        let aspect = f64::from(size.width()) / f64::from(size.height());
        let fit = half.min((half.tan() * aspect).atan());
        self.target = centre;
        self.pivot = centre;
        self.lens = Lens::NONE;
        self.boundary = Some(boundary);
        self.distance = (radius / fit.sin()).clamp(Self::MIN_DISTANCE, self.far());
        self.settle(self.apparent());
    }

    /// Turns the camera about the pivot by `drag`, a rotation of the scene as seen in the view
    /// (as [`Arcball::drag`] gives): the scene appears to turn by it about the pivot, so the eye
    /// and the view turn the other way, and the pivot stays on its pixel. An eye the turn would
    /// carry into the boundary is held out of it ([`Camera::settle`]).
    pub(crate) fn orbit(&mut self, drag: Rotation) {
        // The view turns by o · d⁻¹; in the world, that is the turn w = o · d⁻¹ · o⁻¹ applied
        // to the view's axes, and to the target's (and so the eye's) place about the pivot.
        let turned = drag.inverse().then(self.orientation);
        let world = self.orientation.inverse().then(turned);
        self.target = self.pivot + world.apply(self.target - self.pivot);
        self.orientation = turned;
        self.settle(self.apparent());
    }

    /// Slides the view in its plane so the pivot's depth keeps pace with the pointer, moved by
    /// `dx`, `dy` pixels (right and down) in a viewport `height` pixels tall: the pivot follows
    /// the pointer, magnified or not. An eye the slide would carry into the boundary is held out
    /// of it ([`Camera::settle`]), the pivot still where the pointer took it.
    pub(crate) fn pan(&mut self, dx: f64, dy: f64, height: f64) {
        let forward = -self.axis(Vec3::new(0.0, 0.0, 1.0));
        let depth = (self.pivot - self.eye()).dot(forward);
        // A pivot not ahead of the eye (which navigation never makes) pans at the target's depth.
        let depth = if depth > Self::MIN_DISTANCE {
            depth
        } else {
            self.distance
        };
        let per_pixel =
            2.0 * depth * (self.fov_y / 2.0).tan() / (height.max(1.0) * self.lens.factor);
        self.slide(-dx * per_pixel, dy * per_pixel);
        self.settle(self.apparent());
    }

    /// Moves the target, and the eye with it, `right` and `up` world units along the view's own
    /// axes: the view slides across the scene without turning. Set directly, as a scripted pose
    /// is ([`crate::script`]): it doesn't look at the boundary.
    pub(crate) fn slide(&mut self, right: f64, up: f64) {
        let right_axis = self.axis(Vec3::new(1.0, 0.0, 0.0));
        let up_axis = self.axis(Vec3::new(0.0, 1.0, 0.0));
        self.target = self.target + right_axis * right + up_axis * up;
    }

    /// Zooms in (`factor` below 1) or out (above 1) about the pivot: the pivot's apparent
    /// distance (the eye's distance to it, over the magnification) is scaled by `factor`, the
    /// eye moving along its line to the pivot until it meets the boundary, the image magnified
    /// about the pivot's pixel beyond that ([`Camera::settle`]). A step and its inverse undo
    /// each other, across the boundary either way. A factor that isn't a positive finite number
    /// does nothing.
    pub(crate) fn zoom(&mut self, factor: f64) {
        if factor.is_finite() && factor > 0.0 {
            self.settle(self.apparent() * factor);
        }
    }

    /// The pivot's apparent distance: the eye's distance to it over the magnification, which
    /// sets the pivot's size on the screen.
    fn apparent(&self) -> f64 {
        (self.eye() - self.pivot).length() / self.lens.factor
    }

    /// The farthest the eye goes from the target or the pivot: [`Camera::MAX_DISTANCE`], or that
    /// many of the boundary's radii for a model larger than 1. Distances scale with the model,
    /// so a model of any size frames unmagnified and zooms out as far, in its own radii, as the
    /// benchmark's unit sphere does.
    fn far(&self) -> f64 {
        self.boundary.map_or(Self::MAX_DISTANCE, |boundary| {
            Self::MAX_DISTANCE * boundary.radius.max(1.0)
        })
    }

    /// The most the image may be magnified with the eye `reach` from the pivot along `line`: as
    /// far as a pixel at the pivot's depth, in a [`Camera::TALLEST_VIEWPORT`] tall view, still
    /// spans [`Camera::PIXEL_ULPS`] of the `f32` rounding of what is drawn there; at least 1.
    ///
    /// The GPU takes the eye and the surface as `f32` (the view eye-relative,
    /// `render::camera::GpuView`), each rounded to an ulp of its magnitude: so a pixel smaller
    /// than an ulp of the larger of the two, the eye's or the pivot's largest coordinate, shows
    /// the rounding as the eye moves (the image steps rather than slides), and the pick ray's
    /// own rounding (`crate::pick`) disagrees with the image by that much. Four ulps keeps both
    /// to about a quarter of a pixel. For a unit model with the eye on its boundary, 0.05 above
    /// the surface, that is about 40 times.
    fn most_magnification(&self, line: Vec3, reach: f64) -> f64 {
        let eye = self.pivot + line * reach;
        let forward = -self.axis(Vec3::new(0.0, 0.0, 1.0));
        let depth = (self.pivot - eye).dot(forward);
        let largest = [eye, self.pivot]
            .iter()
            .flat_map(|v| [v.x.abs(), v.y.abs(), v.z.abs()])
            .fold(0.0, f64::max);
        let per_unit = Self::TALLEST_VIEWPORT / 2.0 / (self.fov_y / 2.0).tan();
        let pixel = Self::PIXEL_ULPS * f32_ulp(largest);
        (depth / (pixel * per_unit)).max(1.0)
    }

    /// Puts the eye on its line to the pivot, and magnifies the image, so the pivot's apparent
    /// distance is `apparent` (clamped to the limits): the eye as close as that asks, but no
    /// closer than the boundary (or [`Camera::MIN_DISTANCE`]) along the line, the image
    /// magnified by what is left, up to [`Camera::most_magnification`]. The pivot keeps its
    /// pixel: the eye moves along the line through it, and the image scales about it. Whatever
    /// moved the view last, it leaves the eye out of the boundary, and on it while the image is
    /// magnified.
    ///
    /// **Back to 1, the shift is folded into the orientation:** a magnification that falls to 1
    /// with the image still shifted (the pivot moved on the screen while magnified) turns the
    /// view about the eye instead, so the pivot shows where the shifted image showed it and the
    /// view looks along its axis again: an unmagnified image is never shifted. The pivot keeps
    /// its pixel exactly; the rest of the image differs from the shifted one by the turn's
    /// perspective, which grows with the shift (a few pixels at the edge for a shift of a tenth
    /// of the field).
    ///
    /// **A pivot not ahead of the eye** (which navigation never makes) has no pixel to keep: the
    /// image keeps its centre's instead, and isn't magnified (its depth leaves
    /// [`Camera::most_magnification`] at 1), a magnified one falling back to 1 about its centre.
    fn settle(&mut self, apparent: f64) {
        let to_eye = self.eye() - self.pivot;
        let reach = to_eye.length();
        if !(reach > 0.0 && reach.is_finite()) {
            return;
        }
        let line = to_eye * reach.recip();
        let far = self.far();
        let floor = self
            .boundary
            .and_then(|boundary| boundary.exit(self.pivot, line))
            .unwrap_or(0.0)
            .clamp(Self::MIN_DISTANCE, far);
        let apparent = apparent
            .min(far)
            .max(floor / self.most_magnification(line, floor));
        let settled = apparent.max(floor);
        let factor = settled / apparent;
        #[expect(
            clippy::float_cmp,
            reason = "a move that leaves the eye and the lens where they are must change nothing, \
                      bit for bit, not round the eye back onto its line"
        )]
        if settled == reach && factor == self.lens.factor {
            return;
        }
        // What keeps its pixel: the pivot, its place on the unmagnified image plane unchanged by
        // the eye moving along its line; or, with the pivot not ahead, the image's centre.
        let [sx, sy] = self.lens.shift;
        let [x, y] = self
            .pivot_on_image()
            .unwrap_or([-sx / self.lens.factor, -sy / self.lens.factor]);
        let shown = [self.lens.factor * x + sx, self.lens.factor * y + sy];
        #[expect(
            clippy::float_cmp,
            reason = "factor is exactly 1 when the eye is at the apparent distance, and an \
                      unmagnified image must have no shift at all"
        )]
        let unmagnified = factor == 1.0;
        if unmagnified {
            self.turn_to_show([x, y], shown);
            self.lens.shift = [0.0, 0.0];
        } else {
            self.lens.shift = [shown[0] - factor * x, shown[1] - factor * y];
        }
        self.lens.factor = factor;
        let eye = self.pivot + line * settled;
        let back = self.axis(Vec3::new(0.0, 0.0, 1.0));
        self.distance = (self.distance * settled / reach).clamp(Self::MIN_DISTANCE, far);
        self.target = eye - back * self.distance;
    }

    /// Turns the view about the eye so the unmagnified view shows what it shows at `from` on its
    /// image plane (right and up, over depth) at `shown` instead: the shortest turn carrying the
    /// one direction in the view to the other. Nothing if they are the same. The eye stays; the
    /// target turns with the view.
    fn turn_to_show(&mut self, [x, y]: [f64; 2], shown: [f64; 2]) {
        #[expect(
            clippy::float_cmp,
            reason = "only a turn that changes nothing is skipped, so an unshifted view stays bit \
                      for bit as it was"
        )]
        if [x, y] == shown {
            return;
        }
        let unit = |v: Vec3| v * v.length().recip();
        let (now, wanted) = (
            unit(Vec3::new(x, y, -1.0)),
            unit(Vec3::new(shown[0], shown[1], -1.0)),
        );
        // In the view, the direction is to turn from `now` to `wanted`: the view turns the other
        // way, as an orbit's drag turns it ([`Camera::orbit`]), about the eye.
        let eye = self.eye();
        self.orientation = Rotation::between(now, wanted)
            .inverse()
            .then(self.orientation);
        self.target = eye - self.axis(Vec3::new(0.0, 0.0, 1.0)) * self.distance;
    }

    /// Where the unmagnified view shows the pivot on its image plane: right and up, over its
    /// depth. `None` if it isn't ahead of the eye.
    fn pivot_on_image(&self) -> Option<[f64; 2]> {
        let seen = self.pivot - self.eye();
        let depth = -seen.dot(self.axis(Vec3::new(0.0, 0.0, 1.0)));
        (depth > 0.0).then(|| {
            [
                seen.dot(self.axis(Vec3::new(1.0, 0.0, 0.0))) / depth,
                seen.dot(self.axis(Vec3::new(0.0, 1.0, 0.0))) / depth,
            ]
        })
    }

    /// The renderer's view (`f64`; the GPU gets it eye-relative) into a viewport of `size`: from
    /// the eye towards the target, the view's up as this camera's, the near plane at
    /// [`render::camera::NEAR_FRACTION`] of the distance, the image magnified as the lens has it
    /// ([`View::magnified`]; an unmagnified camera's view is exactly the plain one). `None` only
    /// if rounding made the camera degenerate, which its own operations never do.
    pub(crate) fn view(&self, size: Size) -> Option<View> {
        let point = |v: Vec3| RenderVec3::new(v.x, v.y, v.z);
        let plain = View::look_at(
            point(self.eye()),
            point(self.target),
            point(self.axis(Vec3::new(0.0, 1.0, 0.0))),
            self.fov_y,
            size,
            self.distance * NEAR_FRACTION,
        )?;
        if self.lens == Lens::NONE {
            return Some(plain);
        }
        lensed(&plain, self.lens.factor, self.lens.shift)
    }
}

/// `plain` with its image magnified `factor` times about the viewport's centre and moved by
/// `shift`, a lens's view.
///
/// The shift is right and up, in units of `plain`'s image plane at unit depth: a point `plain`
/// shows at `t` on its image plane shows at `factor · t + shift`. What [`Camera::view`] gives for
/// its lens, and what a stroke record's camera rebuilds ([`crate::sculpting`]). `None` as
/// [`View::magnified`] is.
pub(crate) fn lensed(plain: &View, factor: f64, shift: [f64; 2]) -> Option<View> {
    // The shift is in image-plane units; the plain view's scale turns it into its pixels (y
    // down).
    let [sx, sy] = shift;
    let per_unit = plain.pixels_per_unit();
    plain.magnified(factor, [sx * per_unit, -sy * per_unit])
}

/// The step between `f32`s at the magnitude `x` (an `f32`'s rounding of a coordinate that
/// large is up to half of it); the smallest normal `f32` for 0.
fn f32_ulp(x: f64) -> f64 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the magnitude as the GPU rounds it, to find its f32 step"
    )]
    let rounded = (x.abs() as f32).max(f32::MIN_POSITIVE);
    let next = f32::from_bits(rounded.to_bits().saturating_add(1));
    f64::from(next - rounded)
}

/// Shoemake's arcball over a viewport: a unit sphere centred on it, filling its shorter side.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Arcball {
    /// The viewport's centre, in pixels.
    centre: Pixel,
    /// The sphere's radius, in pixels: half the shorter side.
    radius: f64,
}

impl Arcball {
    /// The arcball of a viewport of `size`.
    pub(crate) fn new(size: Size) -> Self {
        let (width, height) = (f64::from(size.width()), f64::from(size.height()));
        Self {
            centre: Pixel {
                x: width / 2.0,
                y: height / 2.0,
            },
            radius: width.min(height) / 2.0,
        }
    }

    /// The point on the sphere under `at`, in view space (x right, y up, z towards the eye):
    /// on the front of the sphere inside its silhouette, on its rim outside.
    pub(crate) fn point(&self, at: Pixel) -> Vec3 {
        let x = (at.x - self.centre.x) / self.radius;
        let y = (self.centre.y - at.y) / self.radius;
        let r2 = x * x + y * y;
        if r2 <= 1.0 {
            Vec3::new(x, y, (1.0 - r2).sqrt())
        } else {
            let r = r2.sqrt();
            Vec3::new(x / r, y / r, 0.0)
        }
    }

    /// The rotation of the scene a drag from `from` to `to` makes, in view space.
    pub(crate) fn drag(&self, from: Pixel, to: Pixel) -> Rotation {
        Rotation::between(self.point(from), self.point(to))
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    /// How close two computed values must be to count as equal.
    const CLOSE: f64 = 1e-9;

    /// Whether `a` and `b` agree to within [`CLOSE`] in every component.
    fn close(a: Vec3, b: Vec3) -> bool {
        (a - b).length() < CLOSE
    }

    /// A 200 × 100 viewport: a sphere of radius 50 centred at (100, 50).
    fn arcball() -> Arcball {
        Arcball::new(Size::new(200, 100).expect("non-zero"))
    }

    #[test]
    fn a_rotation_between_two_vectors_carries_one_to_the_other() {
        let from = Vec3::new(1.0, 0.0, 0.0);
        let to = Vec3::new(0.0, 0.6, 0.8);
        let rotation = Rotation::between(from, to);
        assert!(
            close(rotation.apply(from), to),
            "{:?}",
            rotation.apply(from)
        );
        assert!(
            close(rotation.inverse().apply(to), from),
            "the inverse carries it back"
        );
        let half = Rotation::between(from, -from);
        assert!(
            close(half.apply(from), -from),
            "opposite vectors: a half turn"
        );
    }

    #[test]
    fn rotations_compose_in_order() {
        let z = Vec3::new(0.0, 0.0, 1.0);
        let quarter_z = Rotation::about(z, std::f64::consts::FRAC_PI_2);
        let quarter_x = Rotation::about(Vec3::new(1.0, 0.0, 0.0), std::f64::consts::FRAC_PI_2);
        let x = Vec3::new(1.0, 0.0, 0.0);
        // x turned a quarter about z is y; y turned a quarter about x is z.
        assert!(close(quarter_z.then(quarter_x).apply(x), z));
        // x turned a quarter about x stays x; then about z, y.
        assert!(close(
            quarter_x.then(quarter_z).apply(x),
            Vec3::new(0.0, 1.0, 0.0)
        ));
    }

    #[test]
    fn the_arcball_maps_the_centre_to_the_front_and_outside_to_the_rim() {
        let ball = arcball();
        assert!(close(
            ball.point(Pixel { x: 100.0, y: 50.0 }),
            Vec3::new(0.0, 0.0, 1.0)
        ));
        assert!(
            close(
                ball.point(Pixel { x: 100.0, y: 0.0 }),
                Vec3::new(0.0, 1.0, 0.0)
            ),
            "the top of the sphere is up"
        );
        let outside = ball.point(Pixel { x: 190.0, y: 50.0 });
        assert!(
            close(outside, Vec3::new(1.0, 0.0, 0.0)),
            "beyond the silhouette, the rim: {outside:?}"
        );
    }

    #[test]
    fn dragging_right_turns_the_scene_about_the_view_up_axis() {
        let ball = arcball();
        let drag = ball.drag(Pixel { x: 100.0, y: 50.0 }, Pixel { x: 150.0, y: 50.0 });
        // The front of the sphere moves to its right edge: a quarter turn about +y.
        assert!(close(
            drag.apply(Vec3::new(0.0, 0.0, 1.0)),
            Vec3::new(1.0, 0.0, 0.0)
        ));
        assert!(
            close(
                drag.apply(Vec3::new(0.0, 1.0, 0.0)),
                Vec3::new(0.0, 1.0, 0.0)
            ),
            "up stays up"
        );
    }

    #[test]
    fn orbiting_turns_the_eye_against_the_drag_and_back() {
        let mut camera = Camera::default();
        let ball = arcball();
        let (centre, right) = (Pixel { x: 100.0, y: 50.0 }, Pixel { x: 150.0, y: 50.0 });
        camera.orbit(ball.drag(centre, right));
        // The scene turned a quarter to the right, so the eye, which was on +z, is now on −x,
        // still 3 from the target.
        assert!(
            close(camera.eye(), Vec3::new(-3.0, 0.0, 0.0)),
            "{:?}",
            camera.eye()
        );
        camera.orbit(ball.drag(right, centre));
        assert!(
            close(camera.eye(), Vec3::new(0.0, 0.0, 3.0)),
            "a drag back undoes it: {:?}",
            camera.eye()
        );
    }

    #[test]
    fn panning_moves_the_target_with_the_pointer_at_its_depth() {
        let mut camera = Camera::default();
        // At distance 3 with a 45° field of view, a 100-pixel-tall view spans 6 tan 22.5° units.
        let span = 6.0 * (22.5_f64).to_radians().tan();
        camera.pan(50.0, 0.0, 100.0);
        // Dragging right by half the height moves the scene right, so the target goes left.
        assert!(close(camera.target(), Vec3::new(-span / 2.0, 0.0, 0.0)));
        camera.pan(0.0, 100.0, 100.0);
        assert!(
            close(camera.target(), Vec3::new(-span / 2.0, span, 0.0)),
            "dragging down moves the target up: {:?}",
            camera.target()
        );
        assert!(
            close(camera.eye() - camera.target(), Vec3::new(0.0, 0.0, 3.0)),
            "the eye keeps its offset"
        );
    }

    #[test]
    fn zooming_with_no_boundary_scales_the_distance_then_magnifies_at_the_closest() {
        let mut camera = Camera::default();
        camera.zoom(0.5);
        assert!((camera.distance() - 1.5).abs() < CLOSE);
        assert!((camera.magnification() - 1.0).abs() < CLOSE);
        camera.zoom(1e-6);
        assert!(
            (camera.distance() - Camera::MIN_DISTANCE).abs() < CLOSE,
            "the eye stops at the closest: {}",
            camera.distance()
        );
        assert!(
            (camera.magnification() - Camera::MIN_DISTANCE / 1.5e-6).abs() < 1e-6,
            "and the image is magnified by the rest: {}",
            camera.magnification()
        );
        camera.zoom(1e-12);
        // The eye 1e-4 from the pivot at the origin: a pixel of four f32 steps at 1e-4, at the
        // tallest viewport's scale.
        let per_unit = Camera::TALLEST_VIEWPORT / 2.0 / (camera.fov_y() / 2.0).tan();
        let most = Camera::MIN_DISTANCE / (Camera::PIXEL_ULPS * f32_ulp(1e-4) * per_unit);
        assert!(
            (camera.magnification() / most - 1.0).abs() < 1e-9,
            "magnified at most to its limit: {} {most}",
            camera.magnification()
        );
        camera.zoom(1e30);
        assert!(
            (camera.distance() - Camera::MAX_DISTANCE).abs() < 1e-6,
            "clamped far: {}",
            camera.distance()
        );
        assert!((camera.magnification() - 1.0).abs() < CLOSE);
        for bad in [0.0, -2.0, f64::NAN, f64::INFINITY] {
            let before = camera;
            camera.zoom(bad);
            assert_eq!(camera, before, "{bad} ignored");
        }
    }

    /// The viewport the camera tests project into: 320 × 200.
    fn size() -> Size {
        Size::new(320, 200).expect("non-zero")
    }

    /// A unit sphere's boundary, about the origin.
    fn unit_boundary() -> Boundary {
        Boundary::around(Vec3::default(), 1.0).expect("a boundary")
    }

    /// A point on the unit sphere's near side, off the view's centre: where a stroke began.
    fn on_surface() -> Vec3 {
        let (x, y) = (0.3, -0.2);
        Vec3::new(x, y, (1.0 - x * x - y * y).sqrt())
    }

    /// The default camera with the unit sphere's boundary and its pivot [`on_surface`].
    fn sculpting() -> Camera {
        let mut camera = Camera::default();
        camera.grow_boundary(unit_boundary());
        camera.set_pivot(on_surface());
        camera
    }

    /// Where `camera`'s view of [`size`] shows `point`, and its depth.
    fn shown(camera: &Camera, point: Vec3) -> (f64, f64, f64) {
        camera
            .view(size())
            .expect("a view")
            .pixel(RenderVec3::new(point.x, point.y, point.z))
            .expect("in front")
    }

    /// How far `camera`'s eye is from its boundary's centre, over the boundary's radius: 1 on
    /// it, more outside.
    fn out(camera: &Camera) -> f64 {
        let boundary = camera.boundary().expect("a boundary");
        (camera.eye() - boundary.centre()).length() / boundary.radius()
    }

    /// Whether two pixels agree to `tolerance`.
    fn same_pixel(a: (f64, f64, f64), b: (f64, f64, f64), tolerance: f64) -> bool {
        (a.0 - b.0).abs() < tolerance && (a.1 - b.1).abs() < tolerance
    }

    /// `camera` zoomed in until its image is magnified at least `at_least` times.
    fn magnified(mut camera: Camera, at_least: f64) -> Camera {
        for _ in 0..1000 {
            if camera.magnification() >= at_least {
                return camera;
            }
            camera.zoom(1.0 / 1.1);
        }
        panic!("never magnified {at_least} times: {camera:?}");
    }

    #[test]
    fn setting_the_pivot_moves_no_pixel() {
        let mut camera = magnified(sculpting(), 3.0);
        let before = camera.view(size());
        camera.set_pivot(Vec3::new(-0.5, 0.5, 0.5));
        assert_eq!(camera.view(size()), before);
    }

    #[test]
    fn orbiting_keeps_the_pivot_on_its_pixel_magnified_or_not() {
        let ball = Arcball::new(size());
        for start in [sculpting(), magnified(sculpting(), 4.0)] {
            let mut camera = start;
            let pivot = shown(&camera, camera.pivot());
            for (from, to) in [
                ((160.0, 100.0), (230.0, 60.0)),
                ((40.0, 20.0), (90.0, 180.0)),
            ] {
                camera.orbit(ball.drag(
                    Pixel {
                        x: from.0,
                        y: from.1,
                    },
                    Pixel { x: to.0, y: to.1 },
                ));
                let now = shown(&camera, camera.pivot());
                assert!(same_pixel(pivot, now, 1e-7), "{pivot:?} → {now:?}");
                assert!(out(&camera) >= 1.0 - 1e-12, "outside: {}", out(&camera));
            }
            assert_ne!(camera.eye(), start.eye(), "it turned");
        }
    }

    #[test]
    fn panning_keeps_the_pivot_with_the_pointer_magnified_or_not() {
        for start in [sculpting(), magnified(sculpting(), 6.0)] {
            let mut camera = start;
            for (dx, dy) in [(37.0, -12.0), (-80.0, 45.0), (3.5, 0.25)] {
                let before = shown(&camera, camera.pivot());
                camera.pan(dx, dy, 200.0);
                let after = shown(&camera, camera.pivot());
                let moved = (after.0 - before.0, after.1 - before.1);
                assert!(
                    (moved.0 - dx).abs() < 1e-6 && (moved.1 - dy).abs() < 1e-6,
                    "the pivot moved {moved:?} for ({dx}, {dy})"
                );
                assert!(out(&camera) >= 1.0 - 1e-12, "outside: {}", out(&camera));
                if camera.magnification() > 1.0 {
                    assert!(
                        (out(&camera) - 1.0).abs() < 1e-9,
                        "magnified on the boundary"
                    );
                }
            }
        }
    }

    #[test]
    fn zooming_in_meets_the_boundary_then_magnifies_with_the_pivot_fixed() {
        let mut camera = sculpting();
        let pivot = shown(&camera, camera.pivot());
        let mut last = out(&camera);
        let mut magnified_steps = 0;
        for _ in 0..80 {
            camera.zoom(1.0 / 1.1);
            let now = shown(&camera, camera.pivot());
            assert!(same_pixel(pivot, now, 1e-6), "{pivot:?} → {now:?}");
            assert!(out(&camera) >= 1.0 - 1e-12, "outside: {}", out(&camera));
            if camera.magnification() > 1.0 {
                magnified_steps += 1;
                assert!((out(&camera) - 1.0).abs() < 1e-9, "the eye stopped on it");
            } else {
                assert!(out(&camera) < last, "the eye came closer");
            }
            last = out(&camera);
        }
        assert!(
            magnified_steps > 40,
            "it reached the boundary: {magnified_steps}"
        );
        assert!(camera.magnification() > 10.0, "{}", camera.magnification());
    }

    #[test]
    fn notches_in_then_out_return_to_the_start_across_the_boundary_both_ways() {
        // From outside, in across the boundary and back out.
        let start = sculpting();
        let mut camera = start;
        for _ in 0..50 {
            camera.zoom(1.0 / 1.1);
        }
        assert!(camera.magnification() > 2.0, "{}", camera.magnification());
        for _ in 0..50 {
            camera.zoom(1.1);
        }
        assert!(close(camera.eye(), start.eye()), "{:?}", camera.eye());
        assert!((camera.magnification() - 1.0).abs() < 1e-9);
        let [px, py] = camera.view(size()).expect("a view").principal();
        assert!(
            (px - 160.0).abs() < 1e-6 && (py - 100.0).abs() < 1e-6,
            "the line of sight at the centre again: ({px}, {py})"
        );
        // From magnified, out across the boundary and back in.
        let close_up = magnified(sculpting(), 5.0);
        let mut backing = close_up;
        for _ in 0..40 {
            backing.zoom(1.1);
        }
        assert!((backing.magnification() - 1.0).abs() < CLOSE, "unmagnified");
        assert!(out(&backing) > 1.2, "and backed off");
        for _ in 0..40 {
            backing.zoom(1.0 / 1.1);
        }
        assert!(close(backing.eye(), close_up.eye()), "{:?}", backing.eye());
        assert!((backing.magnification() / close_up.magnification() - 1.0).abs() < 1e-9);
        let (now, then) = (
            shown(&backing, on_surface()),
            shown(&close_up, on_surface()),
        );
        assert!(same_pixel(now, then, 1e-6), "{now:?} {then:?}");
    }

    #[test]
    fn a_boundary_grown_over_the_eye_pushes_it_out_keeping_the_pivots_pixel_and_size() {
        let mut camera = magnified(sculpting(), 2.0);
        let up = camera.axis(Vec3::new(0.0, 1.0, 0.0));
        let size_of = |seen: &Camera| {
            let (a, b) = (
                shown(seen, on_surface()),
                shown(seen, on_surface() + up * 1e-4),
            );
            (a.1 - b.1).abs()
        };
        let (pixel, apparent) = (shown(&camera, on_surface()), size_of(&camera));
        let grown = Boundary::around(Vec3::default(), 1.2).expect("a boundary");
        camera.grow_boundary(grown);
        assert!((out(&camera) - 1.0).abs() < 1e-9, "on the grown boundary");
        assert!((camera.boundary().expect("known").radius() - grown.radius()).abs() < CLOSE);
        assert!(same_pixel(pixel, shown(&camera, on_surface()), 1e-6));
        assert!(
            (size_of(&camera) / apparent - 1.0).abs() < 1e-3,
            "the pivot's size: {apparent} → {}",
            size_of(&camera)
        );
        let smaller = Boundary::around(Vec3::default(), 0.5).expect("a boundary");
        camera.grow_boundary(smaller);
        assert!(
            (camera.boundary().expect("known").radius() - grown.radius()).abs() < CLOSE,
            "a boundary only grows"
        );
    }

    #[test]
    fn the_union_of_two_boundaries_holds_both() {
        let a = Boundary::around(Vec3::new(1.0, 0.0, 0.0), 1.0).expect("a boundary");
        let b = Boundary::around(Vec3::new(-2.0, 1.0, 0.5), 0.5).expect("a boundary");
        let both = a.union(b);
        for one in [a, b] {
            let far = (one.centre() - both.centre()).length() + one.radius();
            assert!(far <= both.radius() + CLOSE, "{far} {}", both.radius());
        }
        let again = a.union(both);
        assert!(
            close(again.centre(), both.centre()) && (again.radius() - both.radius()).abs() < CLOSE,
            "holding it already: {again:?} {both:?}"
        );
        assert!(Boundary::around(Vec3::default(), 0.0).is_none());
        assert!(Boundary::around(Vec3::new(f64::NAN, 0.0, 0.0), 1.0).is_none());
    }

    #[test]
    fn framing_fits_the_model_and_resets_the_pivot_and_the_magnification() {
        let mut camera = magnified(sculpting(), 3.0);
        camera.pan(40.0, -30.0, 200.0);
        let centre = Vec3::new(0.5, -1.0, 2.0);
        camera.frame(centre, 2.0, size());
        assert_eq!(camera.pivot(), centre);
        assert!((camera.magnification() - 1.0).abs() < CLOSE);
        let view = camera.view(size()).expect("a view");
        assert_eq!(view.principal(), [160.0, 100.0], "on the axis again");
        // The sphere's silhouette touches the top and bottom (the narrower field here).
        let distance = (camera.eye() - centre).length();
        let half = (camera.fov_y() / 2.0).sin();
        assert!((2.0 / distance - half).abs() < 1e-9, "{distance}");
        let boundary = camera.boundary().expect("fitted");
        assert!((boundary.radius() - 2.0 * (1.0 + Boundary::MARGIN)).abs() < CLOSE);
        assert!(close(boundary.centre(), centre));
    }

    #[test]
    fn a_scripted_pose_is_exactly_the_plain_view() {
        let mut camera = Camera::orbiting(0.7, -0.3, 2.2);
        camera.slide(0.0, -0.4);
        let point = |v: Vec3| RenderVec3::new(v.x, v.y, v.z);
        let plain = View::look_at(
            point(camera.eye()),
            point(camera.target()),
            point(camera.axis(Vec3::new(0.0, 1.0, 0.0))),
            camera.fov_y(),
            size(),
            camera.distance() * NEAR_FRACTION,
        );
        assert_eq!(camera.view(size()), plain);
        assert_eq!(
            camera.pivot(),
            Vec3::default(),
            "the target it was posed about"
        );
        assert_eq!(camera.boundary(), None);
    }

    /// The pixel the view's line of sight meets: the centre, unless the image is shifted.
    fn principal(camera: &Camera) -> [f64; 2] {
        camera.view(size()).expect("a view").principal()
    }

    /// The unit sphere's near side where `camera` shows `pixel` of [`size`], if it shows it.
    fn sphere_under(camera: &Camera, pixel: (f64, f64)) -> Option<Vec3> {
        let view = camera.view(size())?;
        let ray = view.through(pixel.0, pixel.1);
        let (eye, ray) = (
            Vec3::new(view.eye().x, view.eye().y, view.eye().z),
            Vec3::new(ray.x, ray.y, ray.z),
        );
        // |eye + t ray| = 1, the nearer root.
        let (square, linear, constant) = (ray.dot(ray), 2.0 * eye.dot(ray), eye.dot(eye) - 1.0);
        let root = linear * linear - 4.0 * square * constant;
        (root >= 0.0).then(|| eye + ray * ((-linear - root.sqrt()) / (2.0 * square)))
    }

    /// `camera` zoomed out a notch at a time until unmagnified, each step keeping the pivot on
    /// its pixel; returns how many steps it took.
    fn zoom_out_to_one(camera: &mut Camera) -> usize {
        for step in 0..1000 {
            if camera.magnification() <= 1.0 {
                return step;
            }
            let before = shown(camera, camera.pivot());
            camera.zoom(1.1);
            let after = shown(camera, camera.pivot());
            assert!(
                same_pixel(before, after, 1e-6),
                "step {step}: {before:?} → {after:?}"
            );
        }
        panic!("never unmagnified: {camera:?}");
    }

    #[test]
    fn back_to_one_the_shift_is_folded_into_the_orientation_and_the_pivot_keeps_its_pixel() {
        let mut camera = magnified(sculpting(), 6.0);
        camera.pan(70.0, -40.0, 200.0);
        // A new pivot elsewhere on the surface, seen off where the magnification was centred.
        let other = {
            let (x, y) = (-0.2, 0.25);
            Vec3::new(x, y, (1.0 - x * x - y * y).sqrt())
        };
        camera.set_pivot(other);
        assert_ne!(camera.shift(), [0.0, 0.0], "shifted while magnified");
        zoom_out_to_one(&mut camera);
        assert_eq!(camera.magnification(), 1.0);
        assert_eq!(camera.shift(), [0.0, 0.0], "no shift unmagnified");
        assert_eq!(principal(&camera), [160.0, 100.0], "on the axis");
        // The view is still a true one: its axes orthonormal.
        let [right, up, back] = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ]
        .map(|axis| camera.axis(axis));
        for (a, b) in [(right, up), (up, back), (back, right)] {
            assert!(a.dot(b).abs() < 1e-12 && (a.length() - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn cycles_of_magnifying_panning_and_new_pivots_never_leave_an_unmagnified_view_off_axis() {
        let mut camera = sculpting();
        let ball = Arcball::new(size());
        let pixels = [(200.0, 80.0), (120.0, 130.0), (175.0, 60.0), (140.0, 110.0)];
        for cycle in 0_u32..24 {
            camera = magnified(camera, 3.0 + f64::from(cycle % 5));
            camera.pan(25.0, 15.0 - f64::from(cycle), 200.0);
            camera.orbit(ball.drag(Pixel { x: 160.0, y: 100.0 }, Pixel { x: 163.0, y: 98.0 }));
            let pixel = pixels[usize::try_from(cycle).expect("small") % pixels.len()];
            camera.set_pivot(sphere_under(&camera, pixel).expect("the sphere is there"));
            zoom_out_to_one(&mut camera);
            assert_eq!(camera.shift(), [0.0, 0.0], "cycle {cycle}");
            assert_eq!(principal(&camera), [160.0, 100.0], "cycle {cycle}");
            assert!(out(&camera) >= 1.0 - 1e-12, "cycle {cycle}: outside");
            // Back to the start's distance, to magnify again next cycle.
            for _ in 0..20 {
                camera.zoom(1.1);
            }
        }
    }

    #[test]
    fn a_pivot_behind_the_eye_unmagnifies_about_the_images_centre() {
        let mut camera = magnified(sculpting(), 5.0);
        let behind = camera.eye() + camera.axis(Vec3::new(0.0, 0.0, 1.0)) * 0.5;
        // Not a point of the model, but the camera must stay sound if handed one.
        camera.set_pivot(behind);
        let centre_ray = camera.view(size()).expect("a view").through(160.0, 100.0);
        camera.zoom(0.5);
        assert_eq!(camera.magnification(), 1.0, "nothing to magnify about");
        assert_eq!(camera.shift(), [0.0, 0.0]);
        let now = camera.view(size()).expect("a view").through(160.0, 100.0);
        let unit = |v: RenderVec3| v * v.length().recip();
        assert!(
            (unit(now) - unit(centre_ray)).length() < 1e-9,
            "the centre shows what it showed"
        );
        assert!(out(&camera) >= 1.0 - 1e-12, "outside: {}", out(&camera));
    }

    #[test]
    fn a_pivot_outside_the_boundary_grows_it_to_hold_the_pivot() {
        let mut camera = magnified(sculpting(), 2.0);
        // A surface raised past the margin since the boundary was fitted.
        let raised = on_surface() * 1.3;
        camera.set_pivot(raised);
        let boundary = camera.boundary().expect("a boundary");
        let model = boundary.radius() / (1.0 + Boundary::MARGIN);
        assert!(
            (raised - boundary.centre()).length() <= model + 1e-9,
            "the pivot is a point of the grown model sphere"
        );
        assert!(
            (Vec3::default() - boundary.centre()).length() + 1.0 <= model + 1e-9,
            "which still holds the old one"
        );
        assert!(
            out(&camera) >= 1.0 - 1e-12,
            "the eye held out: {}",
            out(&camera)
        );
        let before = boundary;
        camera.set_pivot(on_surface());
        assert_eq!(
            camera.boundary(),
            Some(before),
            "a pivot inside grows nothing"
        );
    }

    #[test]
    fn a_huge_model_frames_unmagnified() {
        let mut camera = Camera::default();
        for radius in [1.0, 4.5e3, 1e5, 3e6] {
            let centre = Vec3::new(radius * 0.1, -radius, 2.0 * radius);
            camera.frame(centre, radius, size());
            assert_eq!(camera.magnification(), 1.0, "radius {radius}");
            let distance = (camera.eye() - centre).length();
            let half = (camera.fov_y() / 2.0).sin();
            assert!(
                (radius / distance / half - 1.0).abs() < 1e-9,
                "radius {radius}: fitted at {distance}"
            );
            camera.zoom(1e9);
            assert!(
                (camera.eye() - centre).length() <= Camera::MAX_DISTANCE * radius * 1.06,
                "radius {radius}: out as far as a unit model's limit in its radii"
            );
        }
    }

    #[test]
    fn the_magnification_stops_where_a_pixel_spans_four_f32_steps() {
        // A model of radius 0.8: every coordinate of the eye and the pivot well inside 0.5..1,
        // one f32 step.
        let mut camera = Camera::default();
        camera.grow_boundary(Boundary::around(Vec3::default(), 0.8).expect("a boundary"));
        camera.set_pivot(on_surface() * 0.8);
        camera.zoom(1e-9);
        let factor = camera.magnification();
        let eye = camera.eye();
        let forward = -camera.axis(Vec3::new(0.0, 0.0, 1.0));
        let depth = (camera.pivot() - eye).dot(forward);
        let pivot = camera.pivot();
        let coordinates = [eye.x, eye.y, eye.z, pivot.x, pivot.y, pivot.z];
        let largest = coordinates.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let per_unit = Camera::TALLEST_VIEWPORT / 2.0 / (camera.fov_y() / 2.0).tan();
        let pixel = depth / (per_unit * factor);
        let steps = pixel / f32_ulp(largest);
        assert!(
            (steps / Camera::PIXEL_ULPS - 1.0).abs() < 1e-6,
            "{steps} steps a pixel at {factor}x"
        );
        assert!(
            (20.0..100.0).contains(&factor),
            "a unit model's limit: {factor}"
        );
    }

    #[test]
    fn no_orbit_pan_or_zoom_leaves_the_eye_inside_the_boundary() {
        let ball = Arcball::new(size());
        bolero::check!()
            .with_type::<Vec<(u8, f64, f64)>>()
            .for_each(|steps| {
                let mut camera = sculpting();
                for (n, &(kind, a, b)) in steps.iter().enumerate() {
                    if !(a.is_finite() && b.is_finite()) {
                        continue;
                    }
                    let (a, b) = (a.clamp(-1e3, 1e3), b.clamp(-1e3, 1e3));
                    match kind % 7 {
                        0 => camera.orbit(ball.drag(
                            Pixel { x: 160.0, y: 100.0 },
                            Pixel {
                                x: 160.0 + a,
                                y: 100.0 + b,
                            },
                        )),
                        1 => camera.pan(a, b, 200.0),
                        2 => camera.zoom((a / 100.0).exp()),
                        3 => {
                            // A point of the model: inside its bounds.
                            let p = Vec3::new(a.sin(), b.cos(), (a * b).sin()) * 0.5;
                            camera.set_pivot(p);
                        }
                        4 => {
                            // A point the view shows, anywhere along the ray out to three
                            // times the eye's distance to the boundary's centre: inside the
                            // boundary or out past it (a surface raised past its margin).
                            let view = camera.view(size()).expect("a view");
                            let ray = view.through(160.0 + a % 160.0, 100.0 + b % 100.0);
                            let reach = (camera.eye() - camera.boundary().expect("known").centre())
                                .length();
                            let along = (0.01 + (a * b).abs() % 3.0) * reach;
                            let p = view.eye() + ray * (along / ray.length());
                            camera.set_pivot(Vec3::new(p.x, p.y, p.z));
                        }
                        5 => {
                            // A boundary off the centre, grown into the known one.
                            let centre = Vec3::new(a.sin(), b.sin(), (a + b).cos()) * 0.8;
                            let radius = 0.2 + (a * b).abs() % 1.5;
                            camera.grow_boundary(
                                Boundary::around(centre, radius).expect("a boundary"),
                            );
                        }
                        // Zooms at the clamps: far past either limit.
                        _ => camera.zoom(if a > 0.0 { 1e300 } else { 1e-300 }),
                    }
                    let out = out(&camera);
                    assert!(out >= 1.0 - 1e-9, "step {n}: the eye is inside, at {out}");
                    let boundary = camera.boundary().expect("a boundary");
                    let model = boundary.radius() / (1.0 + Boundary::MARGIN);
                    assert!(
                        (camera.pivot() - boundary.centre()).length() <= model * (1.0 + 1e-12),
                        "step {n}: the pivot is outside the boundary"
                    );
                    if camera.magnification() > 1.0 {
                        assert!(
                            (out - 1.0).abs() < 1e-6,
                            "step {n}: magnified off it, at {out}"
                        );
                    } else {
                        assert_eq!(camera.shift(), [0.0, 0.0], "step {n}: shifted at 1x");
                    }
                    assert!(camera.magnification() >= 1.0, "step {n}");
                    assert!(camera.view(size()).is_some(), "step {n}: a view");
                }
            });
    }
}
