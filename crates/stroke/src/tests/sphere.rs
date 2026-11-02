//! A sphere: one connected, curved face.

use geom::FaceId;

use super::{sphere_roots, within_reach};
use crate::{DVec3, Hit, Ray, Surface};

/// A sphere, hit from outside.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Sphere {
    /// The centre.
    pub(crate) centre: DVec3,
    /// The radius.
    pub(crate) radius: f64,
}

impl Sphere {
    /// The unit sphere at the origin.
    pub(crate) const fn unit() -> Self {
        Self {
            centre: DVec3::ZERO,
            radius: 1.0,
        }
    }

    /// The hit on the sphere in the direction of `p` from the centre.
    pub(crate) fn toward(&self, p: DVec3) -> Hit {
        let normal = (p - self.centre).normalized().unwrap_or(DVec3::Z);
        Hit {
            position: self.centre + normal * self.radius,
            normal,
            face: FaceId::new(0),
            uv: [0.0, 0.0],
        }
    }

    /// The great-circle distance between two points on the sphere.
    pub(crate) fn arc(&self, a: DVec3, b: DVec3) -> f64 {
        let (u, v) = (
            (a - self.centre).normalized().unwrap(),
            (b - self.centre).normalized().unwrap(),
        );
        // atan2 of |u × v| and u · v stays accurate for small angles, where acos doesn't.
        self.radius * u.cross(v).length().atan2(u.dot(v))
    }

    /// The point of the sphere closest to `ray`: its first hit, or toward the ray's point nearest
    /// the centre.
    pub(crate) fn closest_to_ray(&self, ray: &Ray) -> DVec3 {
        match sphere_roots(ray, self.centre, self.radius) {
            Some((near, far)) => ray.at(if near > 0.0 { near } else { far }),
            None => self.toward(ray.closest_to(self.centre)).position,
        }
    }
}

impl Surface for Sphere {
    fn raycast(&mut self, ray: &Ray) -> Option<Hit> {
        let (near, _) = sphere_roots(ray, self.centre, self.radius)?;
        (near > 0.0).then(|| self.toward(ray.at(near)))
    }

    fn closest_on_region(&mut self, ray: &Ray, near: &Hit, reach: f64) -> Option<Hit> {
        let candidate = self.closest_to_ray(ray);
        Some(self.toward(within_reach(near.position, candidate, reach)))
    }

    fn continuous(&mut self, a: &Hit, b: &Hit, reach: f64) -> bool {
        self.arc(a.position, b.position) <= reach
    }

    fn area_normal(&mut self, center: DVec3, _radius: f64, _hardness: f32) -> Option<DVec3> {
        // A cap of a sphere around a point on it: by symmetry, its normals sum to the radial
        // direction through the point.
        Some(center - self.centre)
    }
}
