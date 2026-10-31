//! Test fixtures for placement: a pinhole camera, and analytic surfaces.
//!
//! The surfaces (a plane, a sphere, an overhang and a thin slab) each implement
//! [`Surface`](crate::Surface) exactly where that is cheap, and approximately (but
//! deterministically) where the trait allows. The fixtures don't model base-face coordinates: each
//! part is its own face, and every hit's `uv` is `[0, 0]`.

pub(crate) mod overhang;
pub(crate) mod plane;
pub(crate) mod slab;
pub(crate) mod sphere;

use crate::{DVec3, Projector, Ray, Sample, ScreenPoint, Unit};

/// A pinhole camera: a perspective [`Projector`] with square pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Camera {
    /// The eye.
    pub(crate) eye: DVec3,
    /// The view axis, unit length.
    forward: DVec3,
    /// Screen right, unit length.
    right: DVec3,
    /// Screen down, unit length.
    down: DVec3,
    /// The focal length, in pixels: a pixel spans `depth / focal` world units at `depth`.
    pub(crate) focal: f64,
    /// The screen point on the view axis.
    centre: [f64; 2],
}

impl Camera {
    /// A camera at `eye` looking at `target`, `up` towards the screen's top, with a focal length
    /// of `focal` pixels and the view axis through screen point `(400, 300)`.
    pub(crate) fn looking_at(eye: DVec3, target: DVec3, up: DVec3, focal: f64) -> Self {
        let forward = (target - eye).normalized().unwrap();
        let right = forward.cross(up).normalized().unwrap();
        let down = forward.cross(right);
        Self {
            eye,
            forward,
            right,
            down,
            focal,
            centre: [400.0, 300.0],
        }
    }

    /// This camera zoomed in by `factor` (the focal length scaled).
    pub(crate) fn zoomed(self, factor: f64) -> Self {
        Self {
            focal: self.focal * factor,
            ..self
        }
    }

    /// Where `point` appears on the screen.
    pub(crate) fn project(&self, point: DVec3) -> ScreenPoint {
        let v = point - self.eye;
        let depth = v.dot(self.forward);
        let [cx, cy] = self.centre;
        ScreenPoint::new(
            cx + self.focal * v.dot(self.right) / depth,
            cy + self.focal * v.dot(self.down) / depth,
        )
        .unwrap()
    }
}

impl Projector for Camera {
    fn ray(&self, screen: ScreenPoint) -> Option<Ray> {
        let [cx, cy] = self.centre;
        let direction = self.forward
            + self.right * ((screen.x() - cx) / self.focal)
            + self.down * ((screen.y() - cy) / self.focal);
        Ray::new(self.eye, direction)
    }

    fn world_per_pixel(&self, depth: f64) -> f64 {
        depth / self.focal
    }
}

/// A sample at `at` and `time` with `pressure`.
pub(crate) fn sample(at: ScreenPoint, time: f64, pressure: f32) -> Sample {
    Sample::new(at, time, Unit::new(pressure).unwrap()).unwrap()
}

/// `p` moved toward `near` so it lies within `0.9 · reach` of it (unchanged if it already does):
/// how the fixtures keep `closest_on_region` within reach before projecting back onto a face.
pub(crate) fn within_reach(near: DVec3, p: DVec3, reach: f64) -> DVec3 {
    let d = near.distance(p);
    if d <= 0.9 * reach {
        p
    } else {
        near.lerp(p, 0.9 * reach / d)
    }
}

/// The roots of `square · t² + 2 · half · t + constant = 0`, smaller first, for a positive
/// `square`; `None` if there are none.
pub(crate) fn roots(square: f64, half: f64, constant: f64) -> Option<(f64, f64)> {
    let discriminant = half.powi(2) - square * constant;
    if square <= 0.0 || discriminant < 0.0 {
        return None;
    }
    let root = discriminant.sqrt();
    Some(((-half - root) / square, (-half + root) / square))
}

/// Where `ray`'s line meets the sphere at `centre` of `radius`, as ray parameters, nearer first;
/// `None` if it misses or the sphere is wholly behind the origin.
pub(crate) fn sphere_roots(ray: &Ray, centre: DVec3, radius: f64) -> Option<(f64, f64)> {
    let dir = ray.direction();
    let from = ray.origin() - centre;
    roots(dir.dot(dir), from.dot(dir), from.dot(from) - radius.powi(2))
        .filter(|&(_, far)| far > 0.0)
}
