//! Settings validation and the byte form.

use super::bytes::VERSION;
use super::*;
use crate::{ActionSlot, Curve, Entry, FullScale, Input, Param};

/// Settings with every part away from its default, every input and every kind of parameter.
fn everything() -> StrokeSettings {
    let curve = Curve::new(&[(0.0, 0.1), (0.3, 0.9), (1.0, 0.4)]).unwrap();
    let full = FullScale::new(12.5).unwrap();
    let inputs = [
        Input::Pressure,
        Input::TiltAmount,
        Input::TiltDirection,
        Input::Twist,
        Input::Speed(full),
        Input::Distance(full),
        Input::Random,
    ];
    let params = [
        Param::Size,
        Param::Strength,
        Param::Spacing,
        Param::Angle,
        Param::Hardness,
        Param::Action(ActionSlot::new(0).unwrap()),
        Param::Action(ActionSlot::new(3).unwrap()),
    ];
    let entries = inputs
        .into_iter()
        .zip(params)
        .map(|(input, param)| Entry::new(input, param, curve, 0.5, 2.0).unwrap())
        .collect();
    StrokeSettings::new(
        PathSettings::new(2.0, 0.1, 7.0, 12.0).unwrap(),
        Spacing::new(0.3).unwrap(),
        SizeMode::World,
        Radius::new(0.02).unwrap(),
        Dynamics::new(entries).unwrap(),
        Symmetry {
            mirror: Some(MirrorPlanes::X | MirrorPlanes::Y),
            radial: Some(Radial::new(Axis::Y, 6).unwrap()),
        },
        [1.0, -2.0, 0.5, 3.0],
    )
    .unwrap()
    .with_strength(Strength::new(-0.25).unwrap())
    .with_strength_mode(StrengthMode::Radius)
    .with_hardness(Hardness::new(0.75).unwrap())
}

#[test]
fn round_trips_through_bytes() {
    let mirrored = StrokeSettings {
        symmetry: Symmetry::mirror(MirrorPlanes::X | MirrorPlanes::Z),
        ..StrokeSettings::default()
    };
    for settings in [StrokeSettings::default(), everything(), mirrored] {
        let bytes = settings.to_bytes();
        assert_eq!(bytes.first(), Some(&VERSION), "starts with the version");
        assert_eq!(
            StrokeSettings::from_bytes(&bytes),
            Ok(settings.clone()),
            "decodes to the same settings"
        );
    }
}

#[test]
fn the_default_bytes_are_pinned() {
    // Version 4's layout must not drift: a stroke record stored today decodes the same later.
    let bytes = StrokeSettings::default().to_bytes();
    assert_eq!(
        bytes.len(),
        1 + 32 + 4 + 1 + 4 + 1 + 16 + 8 + 1 + 1 + 2 * 27,
        "length"
    );
    let head: Vec<u8> = std::iter::once(VERSION)
        .chain(1.0_f64.to_le_bytes())
        .chain(0.05_f64.to_le_bytes())
        .chain(5.0_f64.to_le_bytes())
        .chain(0.0_f64.to_le_bytes())
        .chain(0.15_f32.to_le_bytes())
        .chain([0])
        .chain(40.0_f32.to_le_bytes())
        .chain([0])
        .collect();
    assert_eq!(bytes.get(..head.len()), Some(head.as_slice()), "the head");
    let first_entry: Vec<u8> = [0, 0, 2]
        .into_iter()
        .chain(0.0_f32.to_le_bytes())
        .chain(0.0_f32.to_le_bytes())
        .chain(1.0_f32.to_le_bytes())
        .chain(1.0_f32.to_le_bytes())
        .chain(0.0_f32.to_le_bytes())
        .chain(1.0_f32.to_le_bytes())
        .collect();
    let brush: Vec<u8> = Strength::DEFAULT
        .get()
        .to_le_bytes()
        .into_iter()
        .chain(Hardness::DEFAULT.get().to_le_bytes())
        .collect();
    let brush_at = head.len() + 16;
    assert_eq!(
        bytes.get(brush_at..brush_at + 8),
        Some(brush.as_slice()),
        "the strength and hardness, after the action's parameters"
    );
    assert_eq!(
        bytes.get(brush_at + 8),
        Some(&0),
        "the strength in model units"
    );
    let at = head.len() + 16 + 8 + 1;
    assert_eq!(bytes.get(at), Some(&2), "two entries");
    assert_eq!(
        bytes.get(at + 1..at + 1 + first_entry.len()),
        Some(first_entry.as_slice()),
        "pressure → size, linear, 0..=1"
    );
}

#[test]
fn version_2_is_read_with_the_default_strength_and_hardness() {
    let settings = everything();
    let bytes = settings.to_bytes();
    assert_eq!(bytes.first(), Some(&4), "written as version 4");
    // Version 2 is version 4 without the strength, the hardness and the strength's mode.
    let brush_at = 1 + 32 + 4 + 1 + 4 + 1 + 1 + 1 + 1 + 16;
    let mut old: Vec<u8> = bytes;
    old.drain(brush_at..brush_at + 9);
    old[0] = 2;
    let read = StrokeSettings::from_bytes(&old).unwrap();
    assert_eq!(read.strength(), Strength::DEFAULT, "the default strength");
    assert_eq!(read.hardness(), Hardness::DEFAULT, "the default hardness");
    assert_eq!(read.strength_mode(), StrengthMode::Model, "in model units");
    assert_eq!(
        read.with_strength(settings.strength())
            .with_strength_mode(settings.strength_mode())
            .with_hardness(settings.hardness()),
        settings,
        "everything else as written"
    );
}

#[test]
fn version_3_is_read_in_model_units() {
    let settings = everything();
    // Version 3 is version 4 without the strength's mode, after the strength and hardness.
    let mode_at = 1 + 32 + 4 + 1 + 4 + 1 + 1 + 1 + 1 + 16 + 8;
    let mut old: Vec<u8> = settings.to_bytes();
    assert_eq!(old.get(mode_at), Some(&1), "written in radii");
    old.remove(mode_at);
    old[0] = 3;
    let read = StrokeSettings::from_bytes(&old).unwrap();
    assert_eq!(read.strength_mode(), StrengthMode::Model, "in model units");
    assert_eq!(
        read.with_strength_mode(StrengthMode::Radius),
        settings,
        "everything else as written"
    );
    let mut unknown = settings.to_bytes();
    unknown[mode_at] = 2;
    assert!(
        matches!(
            StrokeSettings::from_bytes(&unknown),
            Err(DecodeError::Tag {
                what: "strength mode",
                tag: 2
            })
        ),
        "an unknown mode is refused"
    );
}

#[test]
fn bad_bytes_are_refused() {
    let good = everything().to_bytes();
    assert_eq!(
        StrokeSettings::from_bytes(&[]),
        Err(DecodeError::Truncated),
        "empty"
    );
    let mut versioned = good.clone();
    versioned[0] = 1;
    assert_eq!(
        StrokeSettings::from_bytes(&versioned),
        Err(DecodeError::Version(1)),
        "version 1, never written, is not read"
    );
    for cut in 1..good.len() {
        assert!(
            StrokeSettings::from_bytes(&good[..cut]).is_err(),
            "cut at {cut}"
        );
    }
    let mut trailing = good;
    trailing.push(0);
    assert_eq!(
        StrokeSettings::from_bytes(&trailing),
        Err(DecodeError::Trailing(1)),
        "trailing byte"
    );
    let mut radius = StrokeSettings::default().to_bytes();
    let radius_at = 1 + 32 + 4 + 1;
    radius[radius_at..radius_at + 4].copy_from_slice(&(-1.0_f32).to_le_bytes());
    assert!(
        matches!(
            StrokeSettings::from_bytes(&radius),
            Err(DecodeError::Invalid(SettingsError::Radius(_)))
        ),
        "a negative radius"
    );
    let brush_at = 1 + 32 + 4 + 1 + 4 + 1 + 16;
    let mut strength = StrokeSettings::default().to_bytes();
    strength[brush_at..brush_at + 4].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(
        matches!(
            StrokeSettings::from_bytes(&strength),
            Err(DecodeError::Invalid(SettingsError::Strength(_)))
        ),
        "a NaN strength"
    );
    let mut hardness = StrokeSettings::default().to_bytes();
    hardness[brush_at + 4..brush_at + 8].copy_from_slice(&1.5_f32.to_le_bytes());
    assert!(
        matches!(
            StrokeSettings::from_bytes(&hardness),
            Err(DecodeError::Invalid(SettingsError::Hardness(_)))
        ),
        "a hardness past 1"
    );
    let mut mode = StrokeSettings::default().to_bytes();
    mode[1 + 32 + 4] = 9;
    assert_eq!(
        StrokeSettings::from_bytes(&mode),
        Err(DecodeError::Tag {
            what: "size mode",
            tag: 9
        }),
        "unknown size mode"
    );
}

#[test]
fn settings_are_validated() {
    assert!(Spacing::new(0.0).is_err(), "zero spacing");
    assert!(Spacing::new(f32::NAN).is_err(), "NaN spacing");
    assert!(Spacing::new(0.01).is_ok(), "the closest spacing");
    assert_eq!(Spacing::default().get(), Spacing::DEFAULT.get(), "default");
    assert!(MirrorPlanes::from_bits(0).is_err(), "no plane");
    assert!(MirrorPlanes::from_bits(0b1000).is_err(), "unknown plane");
    let xz = MirrorPlanes::X | MirrorPlanes::Z;
    assert!(
        xz.mirrors(Axis::X) && !xz.mirrors(Axis::Y) && xz.mirrors(Axis::Z),
        "X and Z"
    );
    assert!(Radial::new(Axis::Z, 1).is_err(), "one copy is no symmetry");
    assert!(
        Strength::new(f32::INFINITY).is_err(),
        "an infinite strength"
    );
    assert_eq!(
        Strength::new(-0.5).map(Strength::get),
        Ok(-0.5),
        "a strength inwards"
    );
    assert!(Hardness::new(-0.1).is_err(), "a negative hardness");
    assert!(Hardness::new(f32::NAN).is_err(), "a NaN hardness");
    assert_eq!(
        Hardness::new(1.0).map(Hardness::get),
        Ok(1.0),
        "the hardest"
    );
    assert!(Radial::new(Axis::Z, 65).is_err(), "too many copies");
    let nan_param = StrokeSettings::new(
        PathSettings::default(),
        Spacing::DEFAULT,
        SizeMode::Screen,
        Radius::new(1.0).unwrap(),
        Dynamics::default(),
        Symmetry::NONE,
        [0.0, f32::INFINITY, 0.0, 0.0],
    );
    assert!(
        matches!(nan_param, Err(SettingsError::ActionParam { index: 1, .. })),
        "an infinite action parameter"
    );
}

#[test]
fn symmetry_combines_mirror_and_radial() {
    assert_eq!(Symmetry::default(), Symmetry::NONE, "none by default");
    assert_eq!(Symmetry::NONE.copies(), 1, "the dab alone");
    let xz = Symmetry::mirror(MirrorPlanes::X | MirrorPlanes::Z);
    assert_eq!(xz.copies(), 4, "two planes, four images");
    let six = Symmetry::radial(Radial::new(Axis::Y, 6).unwrap());
    assert_eq!(six.copies(), 6, "six turns");
    let both = Symmetry {
        mirror: Some(MirrorPlanes::X),
        radial: Some(Radial::new(Axis::Y, 6).unwrap()),
    };
    assert_eq!(both.copies(), 12, "each turn of each image");
    let most = Symmetry {
        mirror: Some(MirrorPlanes::X | MirrorPlanes::Y | MirrorPlanes::Z),
        radial: Some(Radial::new(Axis::Z, Radial::MAX_COPIES).unwrap()),
    };
    assert_eq!(most.copies(), 8 * 64, "the most");
}

#[test]
fn unknown_symmetry_kinds_are_refused() {
    let settings = StrokeSettings {
        symmetry: Symmetry::mirror(MirrorPlanes::Y),
        ..StrokeSettings::default()
    };
    let at = 1 + 32 + 4 + 1 + 4;
    let good = settings.to_bytes();
    assert_eq!(good.get(at..at + 2), Some([1, 2].as_slice()), "mirror, Y");
    for kinds in [1 << 2, 1 << 3, 0x80] {
        let mut bad = good.clone();
        bad[at] = 1 | kinds;
        assert_eq!(
            StrokeSettings::from_bytes(&bad),
            Err(DecodeError::Tag {
                what: "symmetry",
                tag: 1 | kinds
            }),
            "kind bit {kinds:#04x} (topological is reserved)"
        );
    }
    let mut no_plane = good;
    no_plane[at + 1] = 0;
    assert!(
        matches!(
            StrokeSettings::from_bytes(&no_plane),
            Err(DecodeError::Invalid(SettingsError::Symmetry(_)))
        ),
        "mirroring through no plane"
    );
}

/// Any bytes decode without panicking, and whatever decodes encodes to bytes that decode to the
/// same settings.
#[test]
fn from_bytes_survives_any_input() {
    bolero::check!().for_each(|bytes: &[u8]| {
        if let Ok(settings) = StrokeSettings::from_bytes(bytes) {
            let again = StrokeSettings::from_bytes(&settings.to_bytes());
            assert_eq!(again, Ok(settings), "re-encoding round-trips");
        }
    });
}
