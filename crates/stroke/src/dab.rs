//! `Dab`: one placed brush imprint, the only thing that passes from the stroke engine to an
//! action.

use crate::ACTION_SLOTS;

/// The version of [`Dab`]'s layout and meaning, as the dab journal keeps a stroke's dabs: the
/// app sets it on every project it opens, and a restore refuses rows of another format.
///
/// **Bump it whenever `Dab`'s layout or the meaning of a field changes.** It is never 0, which
/// the journal reads as unset.
pub const DAB_FORMAT: u32 = 1;

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

    /// One dab's bytes as format 1 lays them out: 21 little-endian words (rows are the machine's
    /// bytes, and the app runs on little-endian machines), the fields in order.
    ///
    /// **Changing these bytes means bumping [`DAB_FORMAT`]**: a journal written in the old layout
    /// would otherwise restore as other dabs. The words are each field's bits, written out.
    const FORMAT_1: [u32; 21] = [
        0x3F80_0000, // position.x 1
        0x4000_0000, // position.y 2
        0x4040_0000, // position.z 3
        0x0000_0000, // normal.x 0
        0x0000_0000, // normal.y 0
        0x3F80_0000, // normal.z 1
        0x3F80_0000, // tangent.x 1
        0x0000_0000, // tangent.y 0
        0x0000_0000, // tangent.z 0
        0x3F00_0000, // radius 0.5
        0x3E80_0000, // strength 0.25
        0x3E00_0000, // angle 0.125
        0x3F40_0000, // hardness 0.75
        0x4080_0000, // params[0] 4
        0x40A0_0000, // params[1] 5
        0x40C0_0000, // params[2] 6
        0x40E0_0000, // params[3] 7
        0x0000_0009, // stroke 9
        0x0000_000A, // seq 10
        0x0000_0001, // copy 1
        0x3D80_0000, // time 0.0625
    ];

    /// A dab with every field set apart, as [`FORMAT_1`] holds it.
    const fn known_dab() -> Dab {
        Dab {
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
        }
    }

    /// The format number and the layout are pinned together: a change to [`Dab`]'s bytes without
    /// a bump of [`DAB_FORMAT`] fails here (and a bump without a new image fails too).
    #[test]
    fn dab_format_pins_the_byte_image() {
        assert_eq!(
            DAB_FORMAT, 1,
            "format 1's image below; a new format needs its own"
        );
        let image: Vec<u8> = FORMAT_1
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        let dab = known_dab();
        assert_eq!(
            bytemuck::bytes_of(&dab),
            image.as_slice(),
            "format 1's bytes, exactly"
        );
        let read: Dab = bytemuck::pod_read_unaligned(&image);
        assert_eq!(read, dab, "the image reads back as the dab");
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
