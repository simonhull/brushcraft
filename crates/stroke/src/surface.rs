//! What the stroke engine knows of the scene: a [`Projector`] from the camera and a [`Surface`]
//! to place dabs on, both implemented by its callers, so the engine never sees the GPU, the
//! window or the store.
//!
//! The app implements [`Projector`] from its camera, and `sculpt` implements [`Surface`] over the
//! store's spatial index. The tests implement both analytically (a pinhole camera; a plane, a
//! sphere, an overhang and a thin slab).

use geom::FaceId;

use crate::{DVec3, ScreenPoint};

/// A ray: the points `origin + t · direction` for `t ≥ 0`.
///
/// Its origin and direction are finite and the direction isn't zero; [`Ray::new`] checks it.
/// A [`Projector`]'s rays are scaled so that `t` is the **view depth**: one unit of `t` moves
/// one world unit along the camera's view axis, so a point's [`Ray::depth_of`] is the depth
/// [`Projector::world_per_pixel`] takes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ray {
    /// Where the ray starts: the eye, or a point on the near plane.
    origin: DVec3,
    /// Which way it goes, and how far one unit of `t` is.
    direction: DVec3,
}

impl Ray {
    /// The ray from `origin` along `direction`, or `None` unless both are finite and the
    /// direction isn't zero.
    #[must_use]
    pub fn new(origin: DVec3, direction: DVec3) -> Option<Self> {
        (origin.is_finite() && direction.is_finite() && direction != DVec3::ZERO)
            .then_some(Self { origin, direction })
    }

    /// Where the ray starts.
    #[must_use]
    pub const fn origin(&self) -> DVec3 {
        self.origin
    }

    /// Which way the ray goes; not necessarily unit length.
    #[must_use]
    pub const fn direction(&self) -> DVec3 {
        self.direction
    }

    /// The point at parameter `t`: `origin + t · direction`.
    #[must_use]
    pub fn at(&self, t: f64) -> DVec3 {
        self.origin + self.direction * t
    }

    /// The parameter of the point on the ray's line closest to `point`: for a point on the ray,
    /// its `t`, and so its view depth for a [`Projector`]'s ray. Negative behind the origin.
    #[must_use]
    pub fn depth_of(&self, point: DVec3) -> f64 {
        (point - self.origin).dot(self.direction) / self.direction.dot(self.direction)
    }

    /// The point on the ray (`t ≥ 0`) closest to `point`.
    #[must_use]
    pub fn closest_to(&self, point: DVec3) -> DVec3 {
        self.at(self.depth_of(point).max(0.0))
    }
}

/// Where a ray meets the surface: the point, its normal, and where it is on the base mesh.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    /// The point, in world space.
    pub position: DVec3,
    /// The surface normal there: unit length, pointing out of the surface.
    pub normal: DVec3,
    /// The base face the point lies on.
    pub face: FaceId,
    /// The point's coordinates on its base face, each in `0..=1`.
    pub uv: [f64; 2],
}

/// The camera, as the stroke engine sees it: a screen point to a world ray, and how large a
/// pixel is at a depth.
///
/// Implemented by the app from its camera. Both methods must be deterministic: the same camera
/// gives the same rays, bit for bit, or a replayed stroke differs.
pub trait Projector {
    /// The world ray through `screen` (physical pixels from the view's top-left corner), scaled
    /// so its parameter is the view depth (see [`Ray`]). `None` if the camera can't make one (a
    /// degenerate view); no dab is placed through it.
    fn ray(&self, screen: ScreenPoint) -> Option<Ray>;

    /// How many world units one physical pixel spans at view depth `depth`, across the view: for
    /// a perspective camera, `depth` times the pixel's angular size; for an orthographic one, a
    /// constant. A brush's screen radius becomes a world radius through it.
    fn world_per_pixel(&self, depth: f64) -> f64;
}

/// The surface dabs are placed on, as the stroke engine sees it.
///
/// Implemented by `sculpt` over the store's spatial index. Every method takes `&mut self` so an
/// implementation can evaluate and cache what it reads; each must be deterministic, or a replayed
/// stroke differs.
///
/// **Errors.** The methods can't fail: an implementation that hits an error (a store read) keeps
/// it, answers `None` or `false`, and its owner checks after each [`Stroke`](crate::Stroke) call
/// and ends the stroke there. The engine places no dab where the surface gave no answer.
pub trait Surface {
    /// Where `ray` first meets the surface, or `None` if it misses.
    fn raycast(&mut self, ray: &Ray) -> Option<Hit>;

    /// The point closest to `ray` on the part of the surface connected to `near` within `reach`
    /// (world units), or `None` if there is none (only a failing implementation has none: `near`
    /// itself qualifies).
    ///
    /// The result must be [`continuous`](Surface::continuous) with `near` within `reach`. "Closest
    /// to the ray" may be approximate (the closest of the region's points that the implementation
    /// examines), but it must be deterministic.
    ///
    /// This is where a dab goes when the pen's ray misses the surface or jumps to a separate
    /// part: past a silhouette, dabs keep following the near side.
    fn closest_on_region(&mut self, ray: &Ray, near: &Hit, reach: f64) -> Option<Hit>;

    /// Whether `a` and `b` lie on one connected piece of the surface within `reach` of each other
    /// (world units): connected through the surface itself, not just close in space. Two faces of
    /// a thin slab are close, but not connected nearby.
    fn continuous(&mut self, a: &Hit, b: &Hit, reach: f64) -> bool;

    /// The sum of the surface's normals inside the sphere of `radius` around `center` (a dab's),
    /// each weighted by the dab's falloff at its distance, made as hard as `hardness`: the
    /// dab's area normal before it is normalised.
    ///
    /// Only the near side counts: a vertex whose normal has a non-positive dot product with
    /// `facing` (the surface normal at the dab's hit) is left out, so the far wall of a feature
    /// thinner than the dab (an ear, a fin) doesn't cancel the side being sculpted. `None` if the
    /// implementation has none to give (an error, kept as for the other methods); a sum of zero,
    /// or one that isn't finite, is degenerate too. Placement normalises it, and takes the hit's
    /// normal where it can't.
    ///
    /// The falloff is the action's (the implementation applies it); each vertex of the surface
    /// counts once, however many of its pieces hold it. Deterministic, as every method is.
    fn area_normal(
        &mut self,
        center: DVec3,
        facing: DVec3,
        radius: f64,
        hardness: f32,
    ) -> Option<DVec3>;
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    #[test]
    fn a_ray_must_be_finite_with_a_direction() {
        assert!(Ray::new(DVec3::ZERO, DVec3::ZERO).is_none(), "no direction");
        assert!(
            Ray::new(DVec3::new(f64::NAN, 0.0, 0.0), DVec3::Z).is_none(),
            "NaN origin"
        );
        assert!(
            Ray::new(DVec3::ZERO, DVec3::new(0.0, f64::INFINITY, 0.0)).is_none(),
            "infinite direction"
        );
    }

    #[test]
    fn depth_is_the_ray_parameter() {
        let ray = Ray::new(DVec3::new(1.0, 0.0, 0.0), DVec3::new(0.0, 0.0, 2.0)).unwrap();
        assert_eq!(ray.at(1.5), DVec3::new(1.0, 0.0, 3.0), "at");
        assert_eq!(ray.depth_of(ray.at(1.5)), 1.5, "a point on the ray");
        assert_eq!(ray.depth_of(DVec3::new(5.0, 7.0, 4.0)), 2.0, "off the ray");
        assert_eq!(
            ray.closest_to(DVec3::new(5.0, 7.0, 4.0)),
            DVec3::new(1.0, 0.0, 4.0),
            "closest point"
        );
        assert_eq!(
            ray.closest_to(DVec3::new(0.0, 0.0, -4.0)),
            ray.origin(),
            "behind the origin: the origin"
        );
    }
}
