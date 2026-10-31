//! The symmetry maps on single dabs.

use super::*;
use crate::{Radial, Symmetry};

/// A dab at `position` with a normal and tangent that tell the maps apart.
fn placed(position: DVec3) -> Placed {
    Placed {
        position,
        normal: DVec3::new(0.6, 0.0, 0.8),
        tangent: DVec3::new(0.0, 1.0, 0.0),
        radius: 0.1,
        strength: 0.5,
        angle: 0.25,
        hardness: 1.0,
        params: [1.0, 2.0, 3.0, 4.0],
        seq: 9,
        time: 0.5,
    }
}

fn expand(symmetry: Symmetry, dab: &Placed) -> Vec<Dab> {
    let mut out = Vec::new();
    Copies::new(symmetry).expand(dab, 3, &mut out);
    out
}

fn close(a: [f32; 3], b: DVec3) -> bool {
    DVec3::from(geom::Vec3::new(a[0], a[1], a[2])).distance(b) < 1e-6
}

#[test]
fn mirror_copies_mirror_the_frame() {
    let dab = placed(DVec3::new(0.5, 0.2, 0.8));
    let out = expand(Symmetry::mirror(MirrorPlanes::X), &dab);
    assert_eq!(out.len(), 2, "the dab and its image");
    let (original, image) = (out[0], out[1]);
    assert_eq!((original.copy, image.copy), (0, 1), "copy indices");
    assert!(close(original.position, dab.position), "copy 0 is the dab");
    assert!(
        close(image.position, DVec3::new(-0.5, 0.2, 0.8)),
        "mirrored in X"
    );
    assert!(
        close(image.normal, DVec3::new(-0.6, 0.0, 0.8)),
        "normal mirrored"
    );
    assert!(
        close(image.tangent, DVec3::new(0.0, 1.0, 0.0)),
        "tangent mirrored"
    );
    assert_eq!(
        image.angle, -0.25,
        "a reflection reverses the alpha's rotation"
    );
    assert_eq!(
        (image.seq, image.stroke, image.radius, image.params),
        (9, 3, 0.1, [1.0, 2.0, 3.0, 4.0]),
        "the rest is shared"
    );
}

#[test]
fn two_planes_make_four_images_and_two_reflections_keep_the_angle() {
    let dab = placed(DVec3::new(0.5, 0.2, 0.8));
    let out = expand(Symmetry::mirror(MirrorPlanes::X | MirrorPlanes::Z), &dab);
    let expected = [
        (DVec3::new(0.5, 0.2, 0.8), 0.25),
        (DVec3::new(-0.5, 0.2, 0.8), -0.25),
        (DVec3::new(0.5, 0.2, -0.8), -0.25),
        (DVec3::new(-0.5, 0.2, -0.8), 0.25),
    ];
    assert_eq!(out.len(), 4, "four images");
    for (copy, (position, angle)) in out.iter().zip(expected) {
        assert!(
            close(copy.position, position),
            "copy {}: {:?}",
            copy.copy,
            copy.position
        );
        assert_eq!(copy.angle, angle, "copy {}'s angle", copy.copy);
    }
}

#[test]
fn radial_copies_turn_about_the_axis() {
    let dab = placed(DVec3::new(0.5, 0.0, 0.8));
    let out = expand(Symmetry::radial(Radial::new(Axis::Z, 4).unwrap()), &dab);
    let expected = [
        DVec3::new(0.5, 0.0, 0.8),
        DVec3::new(0.0, 0.5, 0.8),
        DVec3::new(-0.5, 0.0, 0.8),
        DVec3::new(0.0, -0.5, 0.8),
    ];
    assert_eq!(out.len(), 4, "four turns");
    for (copy, position) in out.iter().zip(expected) {
        assert!(
            close(copy.position, position),
            "copy {}: {:?}",
            copy.copy,
            copy.position
        );
        assert_eq!(copy.angle, 0.25, "a turn keeps the angle");
    }
    assert!(
        close(out[1].normal, DVec3::new(0.0, 0.6, 0.8)),
        "the normal turns"
    );
    assert!(
        close(out[1].tangent, DVec3::new(-1.0, 0.0, 0.0)),
        "the tangent turns"
    );
}

#[test]
fn mirror_and_radial_together_make_their_product() {
    let dab = placed(DVec3::new(0.5, 0.1, 0.8));
    let symmetry = Symmetry {
        mirror: Some(MirrorPlanes::X),
        radial: Some(Radial::new(Axis::Z, 3).unwrap()),
    };
    let out = expand(symmetry, &dab);
    assert_eq!(out.len(), 6, "three turns of each of two images");
    let copies: Vec<u32> = out.iter().map(|d| d.copy).collect();
    assert_eq!(copies, [0, 1, 2, 3, 4, 5], "image-major copy order");
    for (index, copy) in out.iter().enumerate() {
        let image = index / 3;
        let turn = f64::from(u32::try_from(index % 3).unwrap());
        let start = if image == 0 {
            dab.position
        } else {
            DVec3::new(-0.5, 0.1, 0.8)
        };
        let expected = rotate(start, Axis::Z, TAU * turn / 3.0);
        assert!(close(copy.position, expected), "copy {index}");
    }
}

#[test]
fn a_dab_on_a_mirror_plane_is_not_doubled() {
    let on_plane = placed(DVec3::new(0.0, 0.3, 0.5));
    assert_eq!(
        expand(Symmetry::mirror(MirrorPlanes::X), &on_plane).len(),
        1,
        "on the plane"
    );
    let on_axis = placed(DVec3::new(0.0, 0.0, 1.0));
    let six = Symmetry::radial(Radial::new(Axis::Z, 6).unwrap());
    assert_eq!(expand(six, &on_axis).len(), 1, "on the radial axis");
    // The copies are 2x apart: within 1 % of the radius (0.1) coincide, beyond don't.
    let inside = placed(DVec3::new(0.004 * 0.1, 0.3, 0.5));
    assert_eq!(
        expand(Symmetry::mirror(MirrorPlanes::X), &inside).len(),
        1,
        "0.8 % of the radius apart: one"
    );
    let outside = placed(DVec3::new(0.006 * 0.1, 0.3, 0.5));
    assert_eq!(
        expand(Symmetry::mirror(MirrorPlanes::X), &outside).len(),
        2,
        "1.2 % of the radius apart: two"
    );
}

#[test]
fn overlapping_copies_both_apply() {
    // A third of a radius from the plane: the copies overlap by far, but don't coincide.
    let near_plane = placed(DVec3::new(0.1 / 3.0, 0.3, 0.5));
    let out = expand(Symmetry::mirror(MirrorPlanes::X), &near_plane);
    assert_eq!(out.len(), 2, "both kept");
}

#[test]
fn no_symmetry_is_the_dab_alone() {
    let dab = placed(DVec3::new(0.5, 0.2, 0.8));
    let out = expand(Symmetry::NONE, &dab);
    assert_eq!(out.len(), 1, "one dab");
    assert_eq!(out[0].copy, 0, "copy 0");
    assert_eq!(out[0].angle, 0.25, "the angle as placed");
}
