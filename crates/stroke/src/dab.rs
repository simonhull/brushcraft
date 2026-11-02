//! `Dab`: one placed brush imprint, the only thing that passes from the stroke engine to an
//! action.

use crate::ACTION_SLOTS;

/// One dab: where a brush imprint lands and everything the action needs to apply it.
///
/// Plain data (`#[repr(C)]`, [`bytemuck::Pod`], 84 bytes of `f32`s and `u32`s with no padding),
/// shared as is by the CPU and the GPU. Every value is final: dynamics, size and symmetry have
/// been applied, so an action never evaluates a curve or knows about pens.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Dab {
    /// The centre, in world space.
    pub position: [f32; 3],
    /// The dab's area normal: the surface's normals inside its sphere, weighted by its falloff,
    /// summed and normalised as it was placed (the surface normal at the centre where that sum is
    /// degenerate). Unit length, pointing out of the surface; draw moves every vertex along it.
    pub normal: [f32; 3],
    /// The stroke's direction along the surface at the centre: unit length, perpendicular to the
    /// normal. With the normal and their cross product, the dab's local frame.
    pub tangent: [f32; 3],
    /// The radius, in world units.
    pub radius: f32,
    /// The action's strength, final: the brush's own ([`Strength`](crate::Strength)) times the
    /// dynamics' factor, in model units (for draw, how far a vertex at the centre moves).
    pub strength: f32,
    /// The alpha's rotation about the normal, in radians from the tangent.
    pub angle: f32,
    /// The falloff's hardness, final, in `0..=1`: the brush's own
    /// ([`Hardness`](crate::Hardness)) times the dynamics' factor, clamped.
    pub hardness: f32,
    /// The action's own parameters, as it defines them.
    pub params: [f32; ACTION_SLOTS],
    /// Which stroke placed the dab.
    pub stroke: u32,
    /// The dab's place in its stroke: `0` for the first dab, counting up. Symmetry copies of one
    /// dab share it.
    pub seq: u32,
    /// Which symmetry copy this is: `0` for the dab under the pen; the others are numbered as
    /// [`Symmetry`](crate::Symmetry) describes.
    pub copy: u32,
    /// When the dab was placed, in seconds since its stroke began on the pen's clock.
    pub time: f32,
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use core::mem::offset_of;

    use super::*;

    #[test]
    fn layout_is_pinned() {
        assert_eq!(size_of::<Dab>(), 84, "21 four-byte fields, no padding");
        assert_eq!(align_of::<Dab>(), 4, "four-byte alignment");
        let offsets = [
            ("position", offset_of!(Dab, position), 0),
            ("normal", offset_of!(Dab, normal), 12),
            ("tangent", offset_of!(Dab, tangent), 24),
            ("radius", offset_of!(Dab, radius), 36),
            ("strength", offset_of!(Dab, strength), 40),
            ("angle", offset_of!(Dab, angle), 44),
            ("hardness", offset_of!(Dab, hardness), 48),
            ("params", offset_of!(Dab, params), 52),
            ("stroke", offset_of!(Dab, stroke), 68),
            ("seq", offset_of!(Dab, seq), 72),
            ("copy", offset_of!(Dab, copy), 76),
            ("time", offset_of!(Dab, time), 80),
        ];
        for (field, got, expected) in offsets {
            assert_eq!(got, expected, "{field}'s offset");
        }
    }

    #[test]
    fn bytes_are_the_fields_in_order() {
        let dab = Dab {
            position: [1.0, 2.0, 3.0],
            normal: [0.0, 0.0, 1.0],
            tangent: [1.0, 0.0, 0.0],
            radius: 0.5,
            strength: 0.25,
            angle: 0.125,
            hardness: 0.75,
            params: [4.0, 5.0, 6.0, 7.0],
            stroke: 9,
            seq: 10,
            copy: 1,
            time: 0.0625,
        };
        let words: &[u32] = bytemuck::cast_slice(bytemuck::bytes_of(&dab));
        assert_eq!(words.first(), Some(&1.0_f32.to_bits()), "position first");
        assert_eq!(words.get(17), Some(&9), "stroke id after the params");
        assert_eq!(words.get(20), Some(&0.0625_f32.to_bits()), "time last");
        assert_eq!(
            *bytemuck::from_bytes::<Dab>(bytemuck::bytes_of(&dab)),
            dab,
            "round-trips through bytes"
        );
    }
}
