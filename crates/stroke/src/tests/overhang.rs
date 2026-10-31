//! An overhang: a mushroom's cap over its stem, two regions with a silhouette between them.
//!
//! - **The dome** (face 0): the upper half of the unit sphere at the origin, `z ≥ 0`.
//! - **The underside** (face 1): the disc `z = 0`, `r ≤ 1`, facing down; it meets the dome at the
//!   rim and the stem at the stem's top.
//! - **The stem** (face 2): the cylinder `r = 0.3` about Z, from `z = −2` up to the underside.
//!
//! Seen from a little above, the rim is a silhouette: a ray just below it hits the stem, which
//! is near in space but 0.7 away through the surface (across the underside).

use std::f64::consts::TAU;

use geom::FaceId;

use super::sphere::Sphere;
use super::{roots, sphere_roots, within_reach};
use crate::{DVec3, Hit, Ray, Surface};

/// The dome's face.
pub(crate) const DOME: FaceId = FaceId::new(0);
/// The underside's face.
pub(crate) const UNDERSIDE: FaceId = FaceId::new(1);
/// The stem's face.
pub(crate) const STEM: FaceId = FaceId::new(2);
/// The stem's radius.
const STEM_RADIUS: f64 = 0.3;
/// The stem's bottom.
const STEM_BOTTOM: f64 = -2.0;
/// Points sampled on each seam circle to measure paths across it.
const SEAM_SAMPLES: u32 = 128;

/// The overhang described in the module's notes.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Overhang;

/// A hit on `face` at `position` with `normal`.
const fn hit(face: FaceId, position: DVec3, normal: DVec3) -> Hit {
    Hit {
        position,
        normal,
        face,
        uv: [0.0, 0.0],
    }
}

/// The point on the circle of `radius` about Z at height `z`, at `angle` radians from X.
fn on_circle(radius: f64, z: f64, angle: f64) -> DVec3 {
    let (sin, cos) = angle.sin_cos();
    DVec3::new(radius * cos, radius * sin, z)
}

/// The shortest path from `a` to `b` through a point of the circle of `radius` about Z at `z`,
/// over [`SEAM_SAMPLES`] points of it.
fn across(a: DVec3, b: DVec3, radius: f64) -> f64 {
    (0..SEAM_SAMPLES)
        .map(|i| on_circle(radius, 0.0, TAU * f64::from(i) / f64::from(SEAM_SAMPLES)))
        .map(|s| a.distance(s) + s.distance(b))
        .fold(f64::INFINITY, f64::min)
}

impl Overhang {
    /// The dome's hit toward `p` from the origin, dropped onto the rim if `p` is below it.
    fn dome(p: DVec3) -> Hit {
        let up = DVec3::new(p.x, p.y, p.z.max(0.0));
        let normal = up.normalized().unwrap_or(DVec3::Z);
        hit(DOME, normal, normal)
    }

    /// The underside's hit at `p` projected onto it, held inside the rim and outside the stem.
    fn underside(p: DVec3) -> Hit {
        let r = p.x.hypot(p.y);
        let held = r.clamp(STEM_RADIUS, 1.0);
        let scale = if r > 0.0 { held / r } else { 0.0 };
        let position = if r > 0.0 {
            DVec3::new(p.x * scale, p.y * scale, 0.0)
        } else {
            DVec3::new(STEM_RADIUS, 0.0, 0.0)
        };
        hit(UNDERSIDE, position, -DVec3::Z)
    }

    /// The stem's hit at `p` projected onto it, held between its bottom and the underside.
    fn stem(p: DVec3) -> Hit {
        let normal = DVec3::new(p.x, p.y, 0.0).normalized().unwrap_or(DVec3::X);
        let z = p.z.clamp(STEM_BOTTOM, 0.0);
        hit(STEM, normal * STEM_RADIUS + DVec3::Z * z, normal)
    }

    /// The hit of `face` nearest `p`.
    fn onto(face: FaceId, p: DVec3) -> Hit {
        match face.index() {
            0 => Self::dome(p),
            1 => Self::underside(p),
            _ => Self::stem(p),
        }
    }

    /// The length of the shortest path through the surface from `a` to `b` (straight within a
    /// face, through sampled seam points between faces).
    fn path(a: &Hit, b: &Hit) -> f64 {
        let (from, to) = (a.position, b.position);
        match (a.face.index(), b.face.index()) {
            (left, right) if left == right => from.distance(to),
            (0, 1) | (1, 0) => across(from, to, 1.0),
            (1, 2) | (2, 1) => across(from, to, STEM_RADIUS),
            // Dome and stem: across the rim, the underside and the stem's top.
            _ => {
                let (dome, stem) = if a.face == DOME {
                    (from, to)
                } else {
                    (to, from)
                };
                // Coarser seams here: this pairs every rim point with every stem-top point.
                let circle = |radius: f64| {
                    (0..64_u32).map(move |i| on_circle(radius, 0.0, TAU * f64::from(i) / 64.0))
                };
                circle(1.0)
                    .flat_map(|rim| {
                        circle(STEM_RADIUS).map(move |top| {
                            dome.distance(rim) + rim.distance(top) + top.distance(stem)
                        })
                    })
                    .fold(f64::INFINITY, f64::min)
            }
        }
    }
}

impl Surface for Overhang {
    fn raycast(&mut self, ray: &Ray) -> Option<Hit> {
        let mut best: Option<(f64, Hit)> = None;
        let mut offer = |at: f64, candidate: Hit| {
            if at > 0.0 && best.is_none_or(|(nearest, _)| at < nearest) {
                best = Some((at, candidate));
            }
        };
        // The dome: the unit sphere's hits with z ≥ 0.
        let dome = sphere_roots(ray, DVec3::ZERO, 1.0);
        for at in dome.map(<[f64; 2]>::from).into_iter().flatten() {
            let point = ray.at(at);
            if point.z >= 0.0 {
                offer(at, Self::dome(point));
            }
        }
        // The underside: z = 0 between the stem and the rim.
        let flat = -ray.origin().z / ray.direction().z;
        if flat.is_finite() {
            let point = ray.at(flat);
            if (STEM_RADIUS..=1.0).contains(&point.x.hypot(point.y)) {
                offer(flat, Self::underside(point));
            }
        }
        // The stem: the infinite cylinder's hits between its ends.
        let (from, dir) = (ray.origin(), ray.direction());
        let stem = roots(
            dir.x.powi(2) + dir.y.powi(2),
            from.x * dir.x + from.y * dir.y,
            from.x.powi(2) + from.y.powi(2) - STEM_RADIUS.powi(2),
        );
        for at in stem.map(<[f64; 2]>::from).into_iter().flatten() {
            let point = ray.at(at);
            if (STEM_BOTTOM..=0.0).contains(&point.z) {
                offer(at, Self::stem(point));
            }
        }
        best.map(|(_, found)| found)
    }

    fn closest_on_region(&mut self, ray: &Ray, near: &Hit, reach: f64) -> Option<Hit> {
        // Near's own face: its point closest to the ray, held within reach of `near`.
        let candidate = match near.face.index() {
            0 => Sphere::unit().closest_to_ray(ray),
            1 => {
                let t = -ray.origin().z / ray.direction().z;
                if t.is_finite() && t > 0.0 {
                    ray.at(t)
                } else {
                    ray.closest_to(near.position)
                }
            }
            _ => ray.closest_to(near.position),
        };
        let first = Self::onto(near.face, candidate);
        let held = Self::onto(
            near.face,
            within_reach(near.position, first.position, reach),
        );
        Some(held)
    }

    fn continuous(&mut self, a: &Hit, b: &Hit, reach: f64) -> bool {
        Self::path(a, b) <= reach
    }
}
