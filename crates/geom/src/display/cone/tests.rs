//! Normal cones alone: their bits' round trip, the cone that covers everything, and what
//! covering and holding give, conservative after quantization.

use super::*;
use crate::test_rng::Rng;

/// `n` random unit vectors from `rng`, within `spread` (a vector length, not an angle) of
/// `centre`.
fn around(rng: &mut Rng, centre: Vec3, spread: f32, n: usize) -> Vec<Vec3> {
    (0..n)
        .map(|_| (centre + rng.vec3(spread)).normalized().unwrap_or(centre))
        .collect()
}

/// Every one of the 2^20 axis patterns (at a half-angle of 0) decodes to a finite unit axis.
#[test]
fn every_axis_pattern_decodes_to_a_unit_axis() {
    for bits in 0..1_u32 << (2 * AXIS_BITS) {
        let axis = FinestNormalCone::from_bits(bits).axis();
        assert!(axis.is_finite(), "{bits:#x}: {axis:?}");
        assert!((axis.length() - 1.0).abs() < 1e-6, "{bits:#x}: {axis:?}");
    }
}

/// Every `u32` decodes to a cone whose bits are the same `u32` again, with a unit axis and a
/// half-angle a whole number of steps up to 180°.
#[test]
fn bits_round_trip() {
    let mut rng = Rng::new(11);
    let samples = (0..4096).map(|_| rng.below(u32::MAX)).chain([
        0,
        1,
        u32::MAX,
        HALF_MAX << HALF_SHIFT,
        (HALF_MAX - 1) << HALF_SHIFT,
    ]);
    for bits in samples {
        let cone = FinestNormalCone::from_bits(bits);
        assert_eq!(cone.to_bits(), bits);
        assert!(
            (cone.axis().length() - 1.0).abs() < 1e-6,
            "{bits:#x}: {:?}",
            cone.axis()
        );
        let half = cone.half_angle();
        assert!(
            (0.0..=core::f32::consts::PI).contains(&half),
            "{bits:#x}: {half}"
        );
    }
}

/// The cone that covers everything holds every direction and every cone, and none but it
/// holds it; it is the default, and a half-angle reaching 180° saturates to it.
#[test]
fn everything_holds_everything() {
    let everything = FinestNormalCone::EVERYTHING;
    assert!(everything.covers_everything());
    assert_eq!(FinestNormalCone::default(), everything);
    assert_eq!(format!("{everything:?}"), "FinestNormalCone::EVERYTHING");
    let mut rng = Rng::new(5);
    for _ in 0..256 {
        let direction = rng.vec3(1.0);
        assert!(everything.contains(direction), "{direction:?}");
        let other = FinestNormalCone::from_bits(rng.below(u32::MAX));
        assert!(everything.holds(other), "{other:?}");
        assert_eq!(other.holds(everything), other.covers_everything());
    }
    // Normals all round the sphere need more than 180° around any axis.
    let all_round = [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 0.0, -1.0),
    ];
    assert_eq!(FinestNormalCone::covering(&all_round), Ok(everything));
    // A hemisphere's rim and its pole: 90° from the axis at best, not everything.
    let hemisphere = FinestNormalCone::covering(&all_round[..5]).unwrap();
    assert!(!hemisphere.covers_everything());
    assert!(
        hemisphere.half_angle().to_degrees() < 90.5,
        "{hemisphere:?}"
    );
    assert_eq!(
        FinestNormalCone::holding(&[hemisphere, everything]),
        Ok(everything),
        "a union with everything"
    );
    let opposite = FinestNormalCone::covering(&[Vec3::new(0.0, 0.0, -1.0)]).unwrap();
    let up = FinestNormalCone::covering(&[Vec3::new(0.0, 0.0, 1.0)]).unwrap();
    assert_eq!(
        FinestNormalCone::holding(&[up, opposite]),
        Ok(everything),
        "opposite directions span 180°"
    );
}

/// A cone covering normals holds each of them, however few or spread, and is no wider than the
/// exact cone around its own decoded axis by more than a step and the slack.
#[test]
fn covering_holds_every_normal() {
    let mut rng = Rng::new(17);
    for case in 0..512 {
        let centre = rng
            .vec3(1.0)
            .normalized()
            .unwrap_or(Vec3::new(0.0, 0.0, 1.0));
        let spread = [0.0, 1e-4, 0.01, 0.1, 0.5, 1.0, 3.0][case % 7];
        let n = 1 + usize::try_from(rng.below(64)).unwrap();
        let normals = around(&mut rng, centre, spread, n);
        let cone = FinestNormalCone::covering(&normals).unwrap();
        let exact = normals
            .iter()
            .map(|&normal| angle(wide(cone.axis()), wide(normal)))
            .fold(0.0, f64::max);
        for &normal in &normals {
            assert!(cone.contains(normal), "case {case}: {normal:?} in {cone:?}");
        }
        if !cone.covers_everything() {
            let over = f64::from(cone.half_angle()) - exact;
            assert!(
                (0.0..=f64::from(HALF_STEP) + 2.0 * SLACK).contains(&over),
                "case {case}: {over} over the exact {exact}"
            );
        }
    }
    assert_eq!(FinestNormalCone::covering(&[]), Err(DisplayError::Shape));
    assert_eq!(
        FinestNormalCone::covering(&[Vec3::new(f32::NAN, 0.0, 1.0)]),
        Err(DisplayError::NotFinite)
    );
    let zero = FinestNormalCone::covering(&[Vec3::ZERO]).unwrap();
    assert_eq!(zero.half_angle(), 0.0, "a zero normal widens nothing");
}

/// The union of cones holds each of them, and so every direction any of them holds, whatever
/// their order; it is a cone itself when they are close, and covers everything past 180°.
#[test]
fn holding_holds_each_cone_in_any_order() {
    let mut rng = Rng::new(23);
    for case in 0..512 {
        let centre = rng
            .vec3(1.0)
            .normalized()
            .unwrap_or(Vec3::new(0.0, 0.0, 1.0));
        let spread = [0.001, 0.05, 0.3, 1.0][case % 4];
        let count = 1 + usize::try_from(rng.below(4)).unwrap();
        let groups: Vec<Vec<Vec3>> = (0..count)
            .map(|_| {
                let at = (centre + rng.vec3(spread)).normalized().unwrap_or(centre);
                let n = 1 + usize::try_from(rng.below(16)).unwrap();
                around(&mut rng, at, spread * 0.5, n)
            })
            .collect();
        let cones: Vec<FinestNormalCone> = groups
            .iter()
            .map(|g| FinestNormalCone::covering(g).unwrap())
            .collect();
        let union = FinestNormalCone::holding(&cones).unwrap();
        let pairs = cones.iter().zip(&groups);
        for (cone, group) in pairs {
            assert!(union.holds(*cone), "case {case}: {cone:?} in {union:?}");
            for &normal in group {
                assert!(
                    union.contains(normal),
                    "case {case}: {normal:?} in {union:?}"
                );
            }
        }
        let mut reversed = cones.clone();
        reversed.reverse();
        assert_eq!(
            FinestNormalCone::holding(&reversed),
            Ok(union),
            "case {case}: the order doesn't matter"
        );
        if count == 1 {
            assert_eq!(
                union, cones[0],
                "case {case}: a cone alone is its own union"
            );
        }
    }
    assert_eq!(FinestNormalCone::holding(&[]), Err(DisplayError::Shape));
}

/// A cone's debug form names its axis and half-angle in degrees.
#[test]
fn debug_names_the_axis_and_the_half_angle() {
    let cone = FinestNormalCone::covering(&[Vec3::new(0.0, 0.0, 1.0)]).unwrap();
    let shown = format!("{cone:?}");
    assert!(shown.starts_with("FinestNormalCone { axis: "), "{shown}");
    assert!(shown.contains("half_degrees: "), "{shown}");
}
