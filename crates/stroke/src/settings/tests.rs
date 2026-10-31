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
        Symmetry::Radial(Radial::new(Axis::Y, 6).unwrap()),
        [1.0, -2.0, 0.5, 3.0],
    )
    .unwrap()
}

#[test]
fn round_trips_through_bytes() {
    let mirrored = StrokeSettings {
        symmetry: Symmetry::Mirror(MirrorPlanes::X | MirrorPlanes::Z),
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
    // Version 1's layout must not drift: a stroke record stored today decodes the same later.
    let bytes = StrokeSettings::default().to_bytes();
    assert_eq!(
        bytes.len(),
        1 + 32 + 4 + 1 + 4 + 1 + 16 + 1 + 2 * 27,
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
    let at = head.len() + 16;
    assert_eq!(bytes.get(at), Some(&2), "two entries");
    assert_eq!(
        bytes.get(at + 1..at + 1 + first_entry.len()),
        Some(first_entry.as_slice()),
        "pressure → size, linear, 0..=1"
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
    versioned[0] = 2;
    assert_eq!(
        StrokeSettings::from_bytes(&versioned),
        Err(DecodeError::Version(2)),
        "unknown version"
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
    assert!(Radial::new(Axis::Z, 65).is_err(), "too many copies");
    let nan_param = StrokeSettings::new(
        PathSettings::default(),
        Spacing::DEFAULT,
        SizeMode::Screen,
        Radius::new(1.0).unwrap(),
        Dynamics::default(),
        Symmetry::None,
        [0.0, f32::INFINITY, 0.0, 0.0],
    );
    assert!(
        matches!(nan_param, Err(SettingsError::ActionParam { index: 1, .. })),
        "an infinite action parameter"
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
