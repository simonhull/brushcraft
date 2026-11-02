//! The plane `z = 0`, unbounded, its normal `+Z`: one connected face.

use geom::FaceId;

use super::within_reach;
use crate::{DVec3, Hit, Ray, Surface};

/// The plane `z = 0`, hit from either side.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Plane;

impl Plane {
    /// The hit at `position`, projected onto the plane.
    pub(crate) fn at(position: DVec3) -> Hit {
        Hit {
            position: DVec3::new(position.x, position.y, 0.0),
            normal: DVec3::Z,
            face: FaceId::new(0),
            uv: [0.0, 0.0],
        }
    }
}

impl Surface for Plane {
    fn raycast(&mut self, ray: &Ray) -> Option<Hit> {
        let t = -ray.origin().z / ray.direction().z;
        (t.is_finite() && t > 0.0).then(|| Self::at(ray.at(t)))
    }

    fn closest_on_region(&mut self, ray: &Ray, near: &Hit, reach: f64) -> Option<Hit> {
        // Where the ray meets the plane, or (parallel or pointing away) its point nearest `near`.
        let on_plane = self
            .raycast(ray)
            .map_or_else(|| ray.closest_to(near.position), |hit| hit.position);
        let candidate = Self::at(on_plane).position;
        Some(Self::at(within_reach(near.position, candidate, reach)))
    }

    fn continuous(&mut self, a: &Hit, b: &Hit, reach: f64) -> bool {
        a.position.distance(b.position) <= reach
    }

    fn area_normal(&mut self, _center: DVec3, radius: f64, _hardness: f32) -> Option<DVec3> {
        // Every normal is +Z; the sum's length (the falloff's integral) doesn't matter here.
        Some(DVec3::Z * radius)
    }
}
