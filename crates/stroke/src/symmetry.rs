//! Symmetry copies: each placed dab copied through the stroke's [`Symmetry`] before any action
//! runs, so every action and target gets symmetry the same way.
//!
//! **The maps.** A [`Symmetry`] turns on mirror planes (world X, Y, Z through the origin, in any
//! combination) and radial copies (`n` turns about a world axis through the origin). The copies
//! are the product: each mirror image (the identity first, then each non-empty subset of the
//! planes, in the order of their bits) turned by each radial step (`0` first). Copy `m · n + r`
//! is mirror image `m` turned `r` steps, so a copy index always names the same map, and copy `0`
//! is the dab under the pen. The order of mirroring and turning doesn't change the set of copies
//! for these maps: a plane either contains the axis or is perpendicular to it.
//!
//! **Mirrored frames.** A copy's position, normal and tangent all go through its map. A
//! reflection reverses rotation, so the alpha's angle is negated for a copy mirrored through an
//! odd number of planes: its rotation from the mirrored tangent then mirrors the original's.
//!
//! **Coincident copies.** A copy whose centre lies within 1 % of the dab's radius of an earlier
//! copy of the same dab is dropped: a dab on a mirror plane (or on the radial axis) is applied
//! once, not twice as hard. Copies that overlap without coinciding are all kept.

use std::f64::consts::TAU;

use crate::{ACTION_SLOTS, Axis, DVec3, Dab, MirrorPlanes, Symmetry};

/// How close two copies' centres must be, as a fraction of the dab's radius, to count as one.
pub(crate) const COINCIDENT: f64 = 0.01;

/// A dab as placement makes it, before it is copied and rounded to `f32`: world values in `f64`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Placed {
    /// The centre.
    pub(crate) position: DVec3,
    /// The unit area normal (see [`Dab::normal`](crate::Dab::normal)).
    pub(crate) normal: DVec3,
    /// The unit tangent along the stroke, perpendicular to the normal.
    pub(crate) tangent: DVec3,
    /// The radius, in world units.
    pub(crate) radius: f64,
    /// The action's strength.
    pub(crate) strength: f32,
    /// The alpha's rotation, in radians from the tangent.
    pub(crate) angle: f32,
    /// The falloff's hardness.
    pub(crate) hardness: f32,
    /// The action's parameters.
    pub(crate) params: [f32; ACTION_SLOTS],
    /// The dab's place in its stroke.
    pub(crate) seq: u32,
    /// Seconds since the stroke began.
    pub(crate) time: f32,
}

/// One symmetry map: a linear map of world space (a rotation, a reflection, or both), as the
/// columns of its matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Map {
    /// The images of world X, Y and Z.
    columns: [DVec3; 3],
    /// Whether the map reverses handedness (an odd number of mirror planes).
    mirrors: bool,
    /// The copy index this map gives.
    index: u32,
}

impl Map {
    /// `v` through the map.
    fn apply(&self, v: DVec3) -> DVec3 {
        let [x, y, z] = self.columns;
        x * v.x + y * v.y + z * v.z
    }
}

/// The world axis `axis` as a unit vector.
const fn unit(axis: Axis) -> DVec3 {
    match axis {
        Axis::X => DVec3::X,
        Axis::Y => DVec3::Y,
        Axis::Z => DVec3::Z,
    }
}

/// The rotation by `angle` radians about `axis`, applied to `v` (right-handed).
fn rotate(v: DVec3, axis: Axis, angle: f64) -> DVec3 {
    let (sin, cos) = angle.sin_cos();
    let k = unit(axis);
    // Rodrigues' formula, for a unit axis.
    v * cos + k.cross(v) * sin + k * (k.dot(v) * (1.0 - cos))
}

/// The stroke's symmetry maps, made once at the stroke's start.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Copies {
    /// The maps in copy order, the identity first.
    maps: Vec<Map>,
    /// The centres of the copies kept so far of the dab being expanded: scratch space, reused
    /// from dab to dab.
    kept: Vec<DVec3>,
}

impl Copies {
    /// The maps of `symmetry`.
    pub(crate) fn new(symmetry: Symmetry) -> Self {
        let planes = symmetry.mirror.map_or(0, MirrorPlanes::bits);
        // Each subset of the active planes, by its bits: 0 (no mirroring) first.
        let images: Vec<u8> = (0..8_u8).filter(|m| m & !planes == 0).collect();
        let (axis, turns) = symmetry
            .radial
            .map_or((Axis::Z, 1), |r| (r.axis(), r.copies()));
        let mut maps = Vec::with_capacity(images.len() * usize::from(turns));
        let mut index = 0;
        for &image in &images {
            let sign = |bit: u8| if image & bit == 0 { 1.0 } else { -1.0 };
            let mirrored = [
                DVec3::X * sign(MirrorPlanes::X.bits()),
                DVec3::Y * sign(MirrorPlanes::Y.bits()),
                DVec3::Z * sign(MirrorPlanes::Z.bits()),
            ];
            for turn in 0..turns {
                let angle = TAU * f64::from(turn) / f64::from(turns);
                let columns = if turn == 0 {
                    mirrored
                } else {
                    mirrored.map(|c| rotate(c, axis, angle))
                };
                maps.push(Map {
                    columns,
                    mirrors: image.count_ones() % 2 == 1,
                    index,
                });
                index += 1;
            }
        }
        Self {
            kept: Vec::with_capacity(maps.len()),
            maps,
        }
    }

    /// Appends `placed`'s copies to `out` as dabs of stroke `stroke`, dropping each copy that
    /// coincides with an earlier one (see the module's notes).
    pub(crate) fn expand(&mut self, placed: &Placed, stroke: u32, out: &mut Vec<Dab>) {
        let near = placed.radius * COINCIDENT;
        self.kept.clear();
        for map in &self.maps {
            let position = map.apply(placed.position);
            if self.kept.iter().any(|k| k.distance(position) <= near) {
                continue;
            }
            self.kept.push(position);
            out.push(Dab {
                position: position.to_f32_array(),
                normal: map.apply(placed.normal).to_f32_array(),
                tangent: map.apply(placed.tangent).to_f32_array(),
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "a dab is f32: rounded once, here"
                )]
                radius: placed.radius as f32,
                strength: placed.strength,
                angle: if map.mirrors {
                    -placed.angle
                } else {
                    placed.angle
                },
                hardness: placed.hardness,
                params: placed.params,
                stroke,
                seq: placed.seq,
                copy: map.index,
                time: placed.time,
            });
        }
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;
