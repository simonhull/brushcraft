//! A thin slab: two faces close together that aren't connected nearby.
//!
//! - **The top** (face 0): the half-plane `z = 0`, `x ≤ 1`, facing up.
//! - **The bottom** (face 1): the half-plane `z = −THICKNESS`, `x ≤ 3`, facing down.
//!
//! The faces join only far away (at `x = −100`, beyond any reach a test uses), so a stroke
//! across the top's edge, seen from above, has rays that pass the edge and hit the bottom's back
//! a hair below the last dab: close in space, but not continuous.

use geom::FaceId;

use super::within_reach;
use crate::{DVec3, Hit, Ray, Surface};

/// The top's face.
pub(crate) const TOP: FaceId = FaceId::new(0);
/// The bottom's face.
pub(crate) const BOTTOM: FaceId = FaceId::new(1);
/// The gap between the faces.
pub(crate) const THICKNESS: f64 = 0.02;
/// Where the top ends.
pub(crate) const TOP_EDGE: f64 = 1.0;
/// Where the bottom ends.
const BOTTOM_EDGE: f64 = 3.0;
/// Where the faces join, far from anything a test strokes.
const FOLD: f64 = -100.0;

/// The slab described in the module's notes.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Slab;

/// The face at height `z` facing along `normal`, ending at `edge`: its hit nearest `p`.
fn onto(face: FaceId, p: DVec3) -> Hit {
    let (z, edge, normal) = if face == TOP {
        (0.0, TOP_EDGE, DVec3::Z)
    } else {
        (-THICKNESS, BOTTOM_EDGE, -DVec3::Z)
    };
    Hit {
        position: DVec3::new(p.x.clamp(FOLD, edge), p.y, z),
        normal,
        face,
        uv: [0.0, 0.0],
    }
}

/// Where `ray` meets the plane `z = height`, if ahead of it.
fn meet(ray: &Ray, height: f64) -> Option<(f64, DVec3)> {
    let t = (height - ray.origin().z) / ray.direction().z;
    (t.is_finite() && t > 0.0).then(|| (t, ray.at(t)))
}

impl Surface for Slab {
    fn raycast(&mut self, ray: &Ray) -> Option<Hit> {
        let top = meet(ray, 0.0).filter(|(_, p)| (FOLD..=TOP_EDGE).contains(&p.x));
        let bottom = meet(ray, -THICKNESS).filter(|(_, p)| (FOLD..=BOTTOM_EDGE).contains(&p.x));
        let first = match (top, bottom) {
            (Some(a), Some(b)) => Some(if a.0 <= b.0 {
                (TOP, a.1)
            } else {
                (BOTTOM, b.1)
            }),
            (Some(a), None) => Some((TOP, a.1)),
            (None, Some(b)) => Some((BOTTOM, b.1)),
            (None, None) => None,
        };
        first.map(|(face, p)| onto(face, p))
    }

    fn closest_on_region(&mut self, ray: &Ray, near: &Hit, reach: f64) -> Option<Hit> {
        let height = near.position.z;
        let candidate = meet(ray, height).map_or_else(|| ray.closest_to(near.position), |m| m.1);
        let first = onto(near.face, candidate);
        Some(onto(
            near.face,
            within_reach(near.position, first.position, reach),
        ))
    }

    fn continuous(&mut self, a: &Hit, b: &Hit, reach: f64) -> bool {
        // Different faces join only at the fold.
        let through = if a.face == b.face {
            a.position.distance(b.position)
        } else {
            (a.position.x - FOLD) + THICKNESS + (b.position.x - FOLD)
        };
        through <= reach
    }

    fn area_normal(
        &mut self,
        _center: DVec3,
        _facing: DVec3,
        _radius: f64,
        _hardness: f32,
    ) -> Option<DVec3> {
        // Not modelled: placement takes the hit's normal, as for a degenerate sum.
        None
    }
}
