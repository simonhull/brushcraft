//! The view: a camera orbiting a target point, and the arcball that turns a drag into a
//! rotation.
//!
//! The camera looks at [`Camera::target`] from [`Camera::distance`] away, along the direction
//! its orientation gives. Orbiting turns the orientation about the target, panning moves the
//! target in the view plane, zooming scales the distance. There is no inertia: the camera moves
//! only while the pointer does.
//!
//! The arcball is Shoemake's: a pointer position maps to a point on a unit sphere centred on
//! the viewport (outside the sphere's silhouette, to its rim), and a drag from one point to
//! another is the rotation carrying the first to the second, so dragging round the rim rolls
//! the view and a drag back to where it began undoes itself.
//!
//! Math is `f64`: the camera is a handful of numbers, and accumulated drags then stay unit
//! length for far longer.

use std::ops::{Add, Mul, Neg, Sub};

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
    #[cfg(test)]
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
    #[cfg(test)]
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

/// The camera: where it looks, from how far, and which way is up.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Camera {
    /// The point it orbits and looks at.
    target: Vec3,
    /// How far the eye is from the target; always within [`Camera::MIN_DISTANCE`] and
    /// [`Camera::MAX_DISTANCE`].
    distance: f64,
    /// The view's orientation: carries the view's axes (x right, y up, z towards the eye) into
    /// the world.
    orientation: Rotation,
    /// The vertical field of view, in radians.
    fov_y: f64,
}

impl Default for Camera {
    /// Looking at the origin down −z from 3 units away, y up: a unit sphere, as the benchmark
    /// project is, fills about two thirds of the view's height.
    fn default() -> Self {
        Self {
            target: Vec3::default(),
            distance: 3.0,
            orientation: Rotation::IDENTITY,
            fov_y: 45_f64.to_radians(),
        }
    }
}

impl Camera {
    /// The farthest the eye goes from the target.
    pub(crate) const MAX_DISTANCE: f64 = 1e4;
    /// The closest the eye comes to the target.
    pub(crate) const MIN_DISTANCE: f64 = 1e-4;

    /// The point it looks at.
    pub(crate) const fn target(&self) -> Vec3 {
        self.target
    }

    /// How far the eye is from the target.
    pub(crate) const fn distance(&self) -> f64 {
        self.distance
    }

    /// Where the eye is.
    pub(crate) fn eye(&self) -> Vec3 {
        self.target + self.axis(Vec3::new(0.0, 0.0, 1.0)) * self.distance
    }

    /// The view's `axis` (x right, y up, z towards the eye) in world space.
    pub(crate) fn axis(&self, axis: Vec3) -> Vec3 {
        self.orientation.apply(axis)
    }

    /// Turns the camera about the target by `drag`, a rotation of the scene as seen in the view
    /// (as [`Arcball::drag`] gives): the scene appears to turn by it, so the camera turns the
    /// other way.
    pub(crate) fn orbit(&mut self, drag: Rotation) {
        self.orientation = drag.inverse().then(self.orientation);
    }

    /// Moves the target in the view plane so the point under the pointer follows it, moved by
    /// `dx`, `dy` pixels (right and down) in a viewport `height` pixels tall: at the target's
    /// depth, the scene keeps pace with the pointer.
    pub(crate) fn pan(&mut self, dx: f64, dy: f64, height: f64) {
        let per_pixel = 2.0 * self.distance * (self.fov_y / 2.0).tan() / height.max(1.0);
        let right = self.axis(Vec3::new(1.0, 0.0, 0.0));
        let up = self.axis(Vec3::new(0.0, 1.0, 0.0));
        self.target = self.target - right * (dx * per_pixel) + up * (dy * per_pixel);
    }

    /// Moves the eye towards the target (`factor` below 1) or away (above 1), scaling the
    /// distance, clamped to [`Camera::MIN_DISTANCE`]..[`Camera::MAX_DISTANCE`]. A factor that
    /// isn't a positive finite number does nothing.
    pub(crate) fn zoom(&mut self, factor: f64) {
        if factor.is_finite() && factor > 0.0 {
            self.distance = (self.distance * factor).clamp(Self::MIN_DISTANCE, Self::MAX_DISTANCE);
        }
    }
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
    fn zooming_scales_the_distance_within_its_limits() {
        let mut camera = Camera::default();
        camera.zoom(0.5);
        assert!((camera.distance() - 1.5).abs() < CLOSE);
        camera.zoom(1e-12);
        assert!(
            (camera.distance() - Camera::MIN_DISTANCE).abs() < CLOSE,
            "clamped near"
        );
        camera.zoom(1e30);
        assert!(
            (camera.distance() - Camera::MAX_DISTANCE).abs() < CLOSE,
            "clamped far"
        );
        for bad in [0.0, -2.0, f64::NAN, f64::INFINITY] {
            camera.zoom(bad);
            assert!(
                (camera.distance() - Camera::MAX_DISTANCE).abs() < CLOSE,
                "{bad} ignored"
            );
        }
    }
}
