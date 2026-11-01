//! Strokes on the analytic surfaces: spacing, the region rules, the size freeze, symmetry, the
//! tail and dynamics.
//!
//! The numbers each test checks are printed by `cargo nextest run -p stroke --no-capture --
//! stroke::tests`.

use super::*;
use crate::tests::overhang::{self, Overhang};
use crate::tests::plane::Plane;
use crate::tests::slab::{self, Slab};
use crate::tests::sphere::Sphere;
use crate::tests::{Camera, sample};
use crate::{
    Axis, Curve, DVec3, Dynamics, Entry, Hardness, Input, MirrorPlanes, Param, PathSettings,
    Radial, Radius, SPACING_TOLERANCE, ScreenPoint, SizeMode, Spacing, Strength, StrengthMode,
    Surface, Symmetry,
};

/// The spacing tests' tolerance: the plan's ±10 %.
const TOLERANCE: f64 = 0.10;

/// Settings with a `radius` (in `mode`'s unit), the default spacing, no smoothing lag to speak
/// of, `dynamics` and `symmetry`.
fn settings(mode: SizeMode, radius: f32, dynamics: Dynamics, symmetry: Symmetry) -> StrokeSettings {
    StrokeSettings::new(
        PathSettings::default(),
        Spacing::DEFAULT,
        mode,
        Radius::new(radius).unwrap(),
        dynamics,
        symmetry,
        [0.5, 1.0, 0.0, 0.0],
    )
    .unwrap()
}

/// A 20-pixel brush with the default dynamics and no symmetry.
fn plain() -> StrokeSettings {
    settings(SizeMode::Screen, 20.0, Dynamics::default(), Symmetry::NONE)
}

/// `count` samples moving evenly across the screen from `from` to `to` over `seconds`, at
/// `pressure`.
fn line(
    from: ScreenPoint,
    to: ScreenPoint,
    count: u32,
    seconds: f64,
    pressure: f32,
) -> Vec<Sample> {
    (0..count)
        .map(|i| {
            let u = f64::from(i) / f64::from(count - 1);
            sample(from.lerp(to, u), seconds * u, pressure)
        })
        .collect()
}

/// Runs a whole stroke of `samples` with `settings`, each sample through `cameras(index)`, and
/// returns every dab and the frozen radius.
fn run_with<S: Surface>(
    settings: &StrokeSettings,
    cameras: impl Fn(usize) -> Camera,
    surface: &mut S,
    samples: &[Sample],
) -> (Vec<Dab>, f64) {
    let (first, rest) = samples.split_first().unwrap();
    let mut stroke = Stroke::begin(settings, StrokeId::new(7), &cameras(0), surface, *first)
        .expect("the first sample is over the surface");
    let radius = stroke.radius();
    let mut dabs: Vec<Dab> = Vec::new();
    for (index, s) in rest.iter().enumerate() {
        dabs.extend(stroke.push(&cameras(index + 1), surface, *s));
    }
    dabs.extend(stroke.end(&cameras(samples.len()), surface));
    (dabs, radius)
}

/// [`run_with`] through one camera.
fn run<S: Surface>(
    settings: &StrokeSettings,
    camera: &Camera,
    surface: &mut S,
    samples: &[Sample],
) -> (Vec<Dab>, f64) {
    run_with(settings, |_| *camera, surface, samples)
}

fn position(dab: &Dab) -> DVec3 {
    let [x, y, z] = dab.position;
    DVec3::new(f64::from(x), f64::from(y), f64::from(z))
}

fn vector(v: [f32; 3]) -> DVec3 {
    DVec3::new(f64::from(v[0]), f64::from(v[1]), f64::from(v[2]))
}

/// The gaps between consecutive dabs (copy 0) as `distance` measures them, the tail's gap
/// (to the last dab, placed at the pen-up) left out.
fn gaps(dabs: &[Dab], distance: impl Fn(DVec3, DVec3) -> f64) -> Vec<f64> {
    let originals: Vec<DVec3> = dabs.iter().filter(|d| d.copy == 0).map(position).collect();
    let pairs = originals.windows(2).map(|w| distance(w[0], w[1]));
    let mut gaps: Vec<f64> = pairs.collect();
    gaps.pop();
    gaps
}

/// The worst relative error of `gaps` against `target`.
fn worst(gaps: &[f64], target: f64) -> f64 {
    gaps.iter()
        .map(|g| (g / target - 1.0).abs())
        .fold(0.0, f64::max)
}

/// A camera 5 units up Z looking at the unit sphere, 800 px focal length: the sphere spans about
/// ±200 px about the view's centre.
fn sphere_camera() -> Camera {
    Camera::looking_at(DVec3::new(0.0, 0.0, 5.0), DVec3::ZERO, DVec3::Y, 800.0)
}

fn screen(x: f64, y: f64) -> ScreenPoint {
    ScreenPoint::new(x, y).unwrap()
}

#[test]
fn the_same_inputs_give_bit_identical_dabs() {
    let random = Entry::new(
        Input::Random,
        Param::Angle,
        Curve::new(&[(0.0, 0.0), (1.0, 1.0)]).unwrap(),
        0.0,
        3.0,
    )
    .unwrap();
    let dynamics = Dynamics::new(vec![
        random,
        *Dynamics::default().entries().first().unwrap(),
    ])
    .unwrap();
    let symmetry = Symmetry {
        mirror: Some(MirrorPlanes::X),
        radial: Some(Radial::new(Axis::Z, 3).unwrap()),
    };
    let settings = settings(SizeMode::Screen, 15.0, dynamics, symmetry);
    let wiggle: Vec<Sample> = (0..120)
        .map(|i| {
            let t = f64::from(i) * 0.005;
            let at = screen(300.0 + 300.0 * t, 300.0 + 80.0 * (9.0 * t).sin());
            sample(
                at,
                t,
                0.3 + 0.6 * f32::from(u8::try_from(i % 7).unwrap()) / 7.0,
            )
        })
        .collect();
    let (first, radius) = run(&settings, &sphere_camera(), &mut Sphere::unit(), &wiggle);
    let (again, _) = run(&settings, &sphere_camera(), &mut Sphere::unit(), &wiggle);
    println!("replay: {} dabs, radius {radius:.5}", first.len());
    assert!(first.len() > 100, "a real stroke: {} dabs", first.len());
    assert_eq!(
        bytemuck::cast_slice::<Dab, u8>(&first),
        bytemuck::cast_slice::<Dab, u8>(&again),
        "bit-identical"
    );
}

#[test]
fn spacing_on_the_sphere_holds_slow_and_fast() {
    let camera = sphere_camera();
    let sphere = Sphere::unit();
    // Across the sphere's face, out to 0.85 of its 163 px radius on screen, where it slopes away.
    let (from, to) = (screen(262.0, 300.0), screen(538.0, 290.0));
    for (name, count, seconds) in [("slow", 400, 4.0), ("fast", 6, 0.03)] {
        let (dabs, radius) = run(
            &plain(),
            &camera,
            &mut Sphere::unit(),
            &line(from, to, count, seconds, 1.0),
        );
        let target = f64::from(Spacing::DEFAULT.get()) * radius;
        let gaps = gaps(&dabs, |a, b| sphere.arc(a, b));
        let error = worst(&gaps, target);
        println!(
            "sphere, {name}: {} gaps, worst {:.2} % off the spacing",
            gaps.len(),
            100.0 * error
        );
        assert!(gaps.len() > 50, "{name}: {} gaps", gaps.len());
        assert!(
            error <= TOLERANCE,
            "{name}: worst gap {:.1} % off",
            100.0 * error
        );
    }
}

#[test]
fn spacing_on_a_slanted_plane_holds() {
    // Ten degrees above the plane: a pixel near the horizon spans many times one near the eye.
    let camera = Camera::looking_at(
        DVec3::new(0.0, -5.0, 0.9),
        DVec3::new(0.0, 0.0, 0.0),
        DVec3::Z,
        800.0,
    );
    let (from, to) = (
        camera.project(DVec3::new(-0.5, -3.0, 0.0)),
        camera.project(DVec3::new(0.5, 4.0, 0.0)),
    );
    for (name, count) in [("slow", 300), ("fast", 5)] {
        let (dabs, radius) = run(
            &plain(),
            &camera,
            &mut Plane,
            &line(from, to, count, 1.0, 1.0),
        );
        let target = f64::from(Spacing::DEFAULT.get()) * radius;
        let gaps = gaps(&dabs, DVec3::distance);
        let error = worst(&gaps, target);
        println!(
            "slanted plane, {name}: {} gaps, worst {:.2} % off",
            gaps.len(),
            100.0 * error
        );
        assert!(gaps.len() > 50, "{name}: {} gaps", gaps.len());
        assert!(
            error <= TOLERANCE,
            "{name}: worst gap {:.1} % off",
            100.0 * error
        );
        // The walk lands each dab within its tolerance of the spacing, well inside the plan's.
        assert!(error <= 2.0 * SPACING_TOLERANCE, "{name}: {error}");
    }
}

#[test]
fn past_the_silhouette_the_stroke_keeps_to_the_caps_near_side() {
    // A little above the cap, so its rim is the silhouette and the stem shows below it.
    let eye = DVec3::new(5.0, 0.0, 1.0);
    let camera = Camera::looking_at(eye, DVec3::new(0.0, 0.0, -0.3), DVec3::Z, 600.0);
    let (from, to) = (
        camera.project(DVec3::new(0.6, 0.1, 0.8)),
        camera.project(DVec3::new(0.3, 0.1, -1.2)),
    );
    let samples = line(from, to, 60, 0.5, 1.0);
    let last_ray = camera.ray(to).unwrap();
    assert_eq!(
        Overhang.raycast(&last_ray).map(|h| h.face),
        Some(overhang::STEM),
        "the pen ends over the stem"
    );
    let (dabs, radius) = run(&plain(), &camera, &mut Overhang, &samples);
    assert!(
        radius * crate::REACH_RADII < 0.7,
        "the stem is out of reach"
    );
    let lowest = dabs
        .iter()
        .map(|d| d.position[2])
        .fold(f32::INFINITY, f32::min);
    println!(
        "overhang: {} dabs, lowest z {lowest:.4}, radius {radius:.4}",
        dabs.len()
    );
    for dab in &dabs {
        let p = position(dab);
        assert!(
            (p.length() - 1.0).abs() < 1e-5 && p.z >= -1e-6,
            "on the dome: {p:?}"
        );
        let facing = vector(dab.normal).dot(eye - p);
        assert!(facing >= -1e-5, "on the near side: {p:?} faces {facing}");
    }
    assert!(
        lowest < 0.02,
        "the stroke reached the rim: lowest z {lowest}"
    );
}

#[test]
fn on_a_thin_slab_the_stroke_never_jumps_to_the_far_face() {
    let camera = Camera::looking_at(
        DVec3::new(-1.0, -0.5, 2.0),
        DVec3::new(1.5, 0.0, 0.0),
        DVec3::Z,
        700.0,
    );
    let (from, to) = (
        camera.project(DVec3::new(0.0, 0.0, 0.0)),
        camera.project(DVec3::new(2.5, 0.2, -slab::THICKNESS)),
    );
    assert_eq!(
        Slab.raycast(&camera.ray(to).unwrap()).map(|h| h.face),
        Some(slab::BOTTOM),
        "the pen ends over the far face"
    );
    for (name, count) in [("slow", 200), ("fast", 4)] {
        let (dabs, radius) = run(
            &plain(),
            &camera,
            &mut Slab,
            &line(from, to, count, 0.5, 1.0),
        );
        let furthest = dabs
            .iter()
            .map(|d| d.position[0])
            .fold(f32::NEG_INFINITY, f32::max);
        println!(
            "slab, {name}: {} dabs, furthest x {furthest:.4}, radius {radius:.4}",
            dabs.len()
        );
        assert!(
            slab::THICKNESS < radius * crate::REACH_RADII,
            "the far face is within reach"
        );
        for dab in &dabs {
            let p = position(dab);
            assert!(
                p.z == 0.0 && p.x <= slab::TOP_EDGE + 1e-6,
                "{name}: on the top: {p:?}"
            );
            assert_eq!(dab.normal, [0.0, 0.0, 1.0], "{name}: facing up");
        }
        assert!(
            f64::from(furthest) > slab::TOP_EDGE - 0.01,
            "{name}: reached the edge"
        );
    }
}

#[test]
fn the_size_is_frozen_across_a_zoom_and_the_world_lock_holds() {
    let camera = Camera::looking_at(DVec3::new(0.0, 0.0, 4.0), DVec3::ZERO, DVec3::Y, 800.0);
    let samples = line(screen(200.0, 300.0), screen(600.0, 300.0), 80, 1.0, 1.0);
    let zooming = |index: usize| {
        if index < 40 {
            camera
        } else {
            camera.zoomed(2.0)
        }
    };
    let (dabs, radius) = run_with(&plain(), zooming, &mut Plane, &samples);
    // 20 px at depth 4 through an 800 px focal length.
    assert!(
        (radius - 20.0 * 4.0 / 800.0).abs() < 1e-12,
        "frozen at the first hit: {radius}"
    );
    #[expect(clippy::cast_possible_truncation, reason = "as the dab rounds it")]
    let expected = radius as f32;
    assert!(dabs.len() > 50, "{} dabs", dabs.len());
    assert!(
        dabs.iter()
            .all(|d| d.radius.to_bits() == expected.to_bits()),
        "every dab the same size"
    );
    // Between strokes, the zoom does change the world size.
    let (_, after) = run(&plain(), &camera.zoomed(2.0), &mut Plane, &samples);
    assert!(
        (after - radius / 2.0).abs() < 1e-12,
        "zoomed in 2x: half the world size"
    );
    // The world lock ignores the camera.
    let world = settings(SizeMode::World, 0.05, Dynamics::default(), Symmetry::NONE);
    for zoom in [1.0, 3.0] {
        let (locked_dabs, locked) = run(&world, &camera.zoomed(zoom), &mut Plane, &samples);
        assert_eq!(
            locked,
            f64::from(0.05_f32),
            "zoom {zoom}: the world radius as given"
        );
        assert!(
            locked_dabs
                .iter()
                .all(|d| d.radius.to_bits() == 0.05_f32.to_bits()),
            "zoom {zoom}: every dab"
        );
    }
}

/// A camera 4 units above the plane, looking down with screen X along world X.
fn top_camera() -> Camera {
    Camera::looking_at(DVec3::new(0.0, 0.0, 4.0), DVec3::ZERO, DVec3::Y, 800.0)
}

/// A stroke on the plane along `y`, at `x` (world), with `symmetry`.
fn symmetric_stroke(x: f64, symmetry: Symmetry) -> (Vec<Dab>, f64) {
    let camera = top_camera();
    let samples = line(
        camera.project(DVec3::new(x, -0.4, 0.0)),
        camera.project(DVec3::new(x, 0.4, 0.0)),
        40,
        0.5,
        1.0,
    );
    let settings = settings(SizeMode::Screen, 20.0, Dynamics::default(), symmetry);
    run(&settings, &camera, &mut Plane, &samples)
}

#[test]
fn mirror_and_radial_copies_land_where_they_should() {
    let (plain_dabs, _) = symmetric_stroke(0.3, Symmetry::NONE);
    let mirror = Symmetry::mirror(MirrorPlanes::X);
    let radial = Symmetry::radial(Radial::new(Axis::Z, 4).unwrap());
    let both = Symmetry {
        mirror: Some(MirrorPlanes::X),
        radial: Some(Radial::new(Axis::Z, 4).unwrap()),
    };
    for (name, symmetry) in [("mirror", mirror), ("radial", radial), ("both", both)] {
        let (dabs, _) = symmetric_stroke(0.3, symmetry);
        let copies = usize::try_from(symmetry.copies()).unwrap();
        println!("{name}: {} dabs, {copies} per placed dab", dabs.len());
        assert_eq!(
            dabs.len(),
            plain_dabs.len() * copies,
            "{name}: every copy kept"
        );
        for (original, group) in plain_dabs.iter().zip(dabs.chunks(copies)) {
            assert_eq!(group[0], *original, "{name}: copy 0 is the dab");
            let p = position(original);
            for copy in group {
                let q = position(copy);
                // On this plane every copy stays on it, at the original's distance from the
                // axis, and its tangent stays perpendicular to the normal.
                assert!(q.z.abs() < 1e-6, "{name}: on the plane");
                assert!(
                    (q.x.hypot(q.y) - p.x.hypot(p.y)).abs() < 1e-6,
                    "{name}: same radius"
                );
                assert!(
                    vector(copy.tangent).dot(vector(copy.normal)).abs() < 1e-6,
                    "{name}: frame"
                );
                assert_eq!(copy.seq, original.seq, "{name}: one sequence number");
            }
        }
        if name == "mirror" {
            for pair in dabs.chunks(2) {
                let (a, b) = (position(&pair[0]), position(&pair[1]));
                assert!(
                    (a.x + b.x).abs() < 1e-6 && (a.y - b.y).abs() < 1e-6,
                    "mirrored in X"
                );
            }
        }
    }
}

#[test]
fn a_stroke_on_the_mirror_plane_isnt_doubled() {
    let mirror = Symmetry::mirror(MirrorPlanes::X);
    let (on_plane, _) = symmetric_stroke(0.0, mirror);
    let (alone, _) = symmetric_stroke(0.0, Symmetry::NONE);
    assert_eq!(on_plane, alone, "on the plane: each dab once");
    // A third of a radius off the plane, the copies overlap but are distinct: both apply.
    let (_, radius) = symmetric_stroke(0.0, Symmetry::NONE);
    let (near_plane, _) = symmetric_stroke(radius / 3.0, mirror);
    let (single, _) = symmetric_stroke(radius / 3.0, Symmetry::NONE);
    assert_eq!(
        near_plane.len(),
        2 * single.len(),
        "overlapping copies both kept"
    );
    let apart = position(&near_plane[0]).distance(position(&near_plane[1]));
    assert!(
        apart < radius,
        "and they overlap: {apart} apart, radius {radius}"
    );
}

#[test]
fn a_fast_strokes_dabs_reach_its_last_sample() {
    let camera = top_camera();
    // Four samples 200 px apart in 15 ms: fast enough that the filter lags well behind.
    let samples = line(screen(150.0, 300.0), screen(750.0, 300.0), 4, 0.015, 1.0);
    let last = camera.ray(samples.last().unwrap().position()).unwrap();
    let pen_up = Plane.raycast(&last).unwrap().position;
    let (first, rest) = samples.split_first().unwrap();
    let mut stroke =
        Stroke::begin(&plain(), StrokeId::new(1), &camera, &mut Plane, *first).unwrap();
    let stroke_radius = stroke.radius();
    let mut pushed: Vec<Dab> = Vec::new();
    for s in rest {
        pushed.extend(stroke.push(&camera, &mut Plane, *s));
    }
    let behind = position(pushed.last().unwrap()).distance(pen_up);
    let tail: Vec<Dab> = stroke.end(&camera, &mut Plane).collect();
    let reached = position(tail.last().unwrap()).distance(pen_up);
    println!(
        "tail: {behind:.4} behind before end, {reached:.2e} after, {} tail dabs",
        tail.len()
    );
    let spacing = f64::from(Spacing::DEFAULT.get()) * stroke_radius;
    assert!(
        behind > 2.0 * spacing,
        "the filter held the brush back: {behind}"
    );
    assert!(reached < 1e-6, "the last dab is at the pen-up: {reached}");
}

#[test]
fn pressure_changes_size_and_strength_per_dab() {
    let camera = top_camera();
    let samples: Vec<Sample> = (0..60)
        .map(|i| {
            let u = f64::from(i) / 59.0;
            #[expect(clippy::cast_possible_truncation, reason = "a pressure")]
            let pressure = (0.2 + 0.8 * u) as f32;
            sample(screen(200.0 + 400.0 * u, 300.0), u, pressure)
        })
        .collect();
    // The brush's own strength 2 and hardness 0.5: each dab's are final, the factors times them.
    let brush = plain()
        .with_strength(Strength::new(2.0).unwrap())
        .with_hardness(Hardness::new(0.5).unwrap());
    let (dabs, radius) = run(&brush, &camera, &mut Plane, &samples);
    let first = dabs.first().unwrap();
    let last = dabs.last().unwrap();
    println!(
        "pressure: {} dabs, radius {:.4} to {:.4}, strength {:.3} to {:.3}",
        dabs.len(),
        first.radius,
        last.radius,
        first.strength,
        last.strength
    );
    assert!(
        (first.strength - 0.4).abs() < 1e-6,
        "starts at 0.2 of 2: {}",
        first.strength
    );
    assert!(
        (last.strength - 2.0).abs() < 1e-6,
        "ends at 2: {}",
        last.strength
    );
    assert!(
        dabs.iter()
            .all(|dab| dab.hardness.to_bits() == 0.5_f32.to_bits()),
        "the brush's hardness, its factor 1"
    );
    // Rising dab by dab until full pressure: the tail (from where the smoothing held the brush
    // to the pen-up) is all at the last sample's pressure.
    for pair in dabs.windows(2) {
        if pair[0].strength < 2.0 {
            assert!(pair[1].strength > pair[0].strength, "rising: {pair:?}");
            assert!(pair[1].radius > pair[0].radius, "growing: {pair:?}");
        } else {
            assert_eq!(pair[1].strength, 2.0, "held at full pressure");
        }
    }
    for dab in &dabs {
        let size = f64::from(dab.radius) / radius;
        assert!(
            (size - f64::from(dab.strength) / 2.0).abs() < 1e-5,
            "size and strength both follow the pressure"
        );
        assert_eq!(
            dab.params,
            [0.5, 1.0, 0.0, 0.0],
            "no dynamics on the action's parameters"
        );
    }
}

#[test]
fn a_strength_in_radii_follows_the_frozen_radius_at_any_zoom() {
    let samples = line(screen(300.0, 300.0), screen(500.0, 300.0), 30, 0.5, 1.0);
    let brush = plain()
        .with_strength(Strength::new(0.1).unwrap())
        .with_strength_mode(StrengthMode::Radius);
    // The same 20-pixel brush from twice as far: twice the world radius, twice the strength.
    let near = Camera::looking_at(DVec3::new(0.0, 0.0, 5.0), DVec3::ZERO, DVec3::Y, 800.0);
    let far = Camera::looking_at(DVec3::new(0.0, 0.0, 10.0), DVec3::ZERO, DVec3::Y, 800.0);
    for camera in [near, far] {
        let (dabs, radius) = run(&brush, &camera, &mut Plane, &samples);
        assert!(!dabs.is_empty(), "dabs placed");
        for dab in &dabs {
            let expected = 0.1 * radius;
            assert!(
                (f64::from(dab.strength) - expected).abs() < 1e-6 * expected,
                "a tenth of the radius {radius}, in model units: {}",
                dab.strength
            );
        }
    }
    let (model, _) = run(
        &plain().with_strength(Strength::new(0.1).unwrap()),
        &near,
        &mut Plane,
        &samples,
    );
    assert!(
        model
            .iter()
            .all(|dab| dab.strength.to_bits() == 0.1_f32.to_bits()),
        "in model units, as given"
    );
}

#[test]
fn nothing_under_the_first_sample_starts_no_stroke() {
    let camera = sphere_camera();
    let corner = sample(screen(5.0, 5.0), 0.0, 1.0);
    assert!(
        Stroke::begin(
            &plain(),
            StrokeId::new(1),
            &camera,
            &mut Sphere::unit(),
            corner
        )
        .is_none(),
        "off the sphere"
    );
}

#[test]
fn a_tap_places_one_dab_with_a_frame() {
    let camera = sphere_camera();
    let at = sample(screen(400.0, 300.0), 0.0, 1.0);
    let stroke =
        Stroke::begin(&plain(), StrokeId::new(4), &camera, &mut Sphere::unit(), at).unwrap();
    let dabs: Vec<Dab> = stroke.end(&camera, &mut Sphere::unit()).collect();
    assert_eq!(dabs.len(), 1, "one dab");
    let dab = dabs[0];
    assert_eq!((dab.seq, dab.stroke, dab.copy), (0, 4, 0), "numbered");
    assert!(
        (position(&dab) - DVec3::Z).length() < 1e-6,
        "at the sphere's top"
    );
    assert!(
        vector(dab.tangent).dot(vector(dab.normal)).abs() < 1e-6,
        "a tangent frame"
    );
}

#[test]
fn each_dab_is_framed_along_the_stroke() {
    let camera = top_camera();
    let samples = line(screen(200.0, 400.0), screen(600.0, 200.0), 50, 0.5, 1.0);
    let (dabs, _) = run(&plain(), &camera, &mut Plane, &samples);
    // 400 px right and 200 px up the screen: world (2, 1, 0).
    let along = DVec3::new(2.0, 1.0, 0.0).normalized().unwrap();
    for dab in &dabs {
        assert_eq!(dab.normal, [0.0, 0.0, 1.0], "the plane's normal");
        assert!(
            vector(dab.tangent).dot(along) > 0.999,
            "along the stroke: {:?}",
            dab.tangent
        );
    }
    let seqs: Vec<u32> = dabs.iter().map(|d| d.seq).collect();
    let expected: Vec<u32> = (0..u32::try_from(dabs.len()).unwrap()).collect();
    assert_eq!(seqs, expected, "numbered in order");
    assert!(
        dabs.windows(2).all(|w| w[1].time >= w[0].time),
        "in time order"
    );
}

/// A straight stroke for the properties: `length` pixels long, at a speed that changes.
///
/// It is centred on `centre` (offset by `offset`, each byte
/// `−32..32` px), heading `angle` (in 256ths of a turn), its samples spaced in proportion to
/// `steps`' first values and timed by their second (1–52 ms apart): the speed changes sample by
/// sample, from a crawl to hundreds of pixels per sample. `None` without steps.
fn property_stroke(
    centre: ScreenPoint,
    length: f64,
    offset: (u8, u8),
    angle: u8,
    steps: &[(u8, u8)],
) -> Option<Vec<Sample>> {
    if steps.is_empty() {
        return None;
    }
    let steps: Vec<(u8, u8)> = steps.iter().copied().cycle().take(40).collect();
    let total: f64 = steps.iter().map(|&(step, _)| f64::from(step) + 1.0).sum();
    let (sin, cos) = (f64::from(angle) / 256.0 * std::f64::consts::TAU).sin_cos();
    let middle = (
        centre.x() + f64::from(offset.0) / 4.0 - 32.0,
        centre.y() + f64::from(offset.1) / 4.0 - 32.0,
    );
    let at = |along: f64| {
        let d = along - length / 2.0;
        screen(middle.0 + d * cos, middle.1 + d * sin)
    };
    let mut travelled = 0.0;
    let mut time = 0.0;
    let mut samples = vec![sample(at(0.0), 0.0, 1.0)];
    for (step, dt) in steps {
        travelled += (f64::from(step) + 1.0) / total * length;
        time += 0.001 + f64::from(dt) * 0.0002;
        samples.push(sample(at(travelled), time, 1.0));
    }
    Some(samples)
}

/// Every dab of a stroke of `samples` on `surface` through `camera` passes `on_region`, if the
/// stroke starts on `start`'s face.
fn stays_on<S: Surface>(
    camera: &Camera,
    surface: &mut S,
    start: geom::FaceId,
    samples: &[Sample],
    on_region: impl Fn(DVec3) -> bool,
) {
    let first = samples.first().unwrap();
    let begun = surface.raycast(&camera.ray(first.position()).unwrap());
    if begun.map(|h| h.face) != Some(start) {
        return;
    }
    let (dabs, _) = run(&plain(), camera, surface, samples);
    for dab in &dabs {
        let p = position(dab);
        assert!(on_region(p), "off the stroke's region: {p:?}");
    }
}

#[test]
fn no_dab_is_off_the_strokes_region() {
    let overhang_camera = Camera::looking_at(
        DVec3::new(5.0, 0.0, 1.0),
        DVec3::new(0.0, 0.0, 0.2),
        DVec3::Z,
        600.0,
    );
    let slab_camera = Camera::looking_at(
        DVec3::new(-1.0, -0.5, 2.0),
        DVec3::new(0.6, 0.0, 0.0),
        DVec3::Z,
        700.0,
    );
    bolero::check!()
        .with_type::<((u8, u8), u8, Vec<(u8, u8)>, bool)>()
        .for_each(|(offset, angle, steps, on_slab)| {
            // Long strokes from the middle of each part, out past its silhouette or edge.
            if *on_slab {
                let centre = slab_camera.project(DVec3::new(0.6, 0.0, 0.0));
                let Some(samples) = property_stroke(centre, 900.0, *offset, *angle, steps) else {
                    return;
                };
                stays_on(&slab_camera, &mut Slab, slab::TOP, &samples, |p| {
                    p.z == 0.0 && p.x <= slab::TOP_EDGE + 1e-6
                });
            } else {
                let centre = overhang_camera.project(DVec3::new(0.8, 0.0, 0.5));
                let Some(samples) = property_stroke(centre, 700.0, *offset, *angle, steps) else {
                    return;
                };
                stays_on(
                    &overhang_camera,
                    &mut Overhang,
                    overhang::DOME,
                    &samples,
                    |p| (p.length() - 1.0).abs() < 1e-5 && p.z >= -1e-6,
                );
            }
        });
}

#[test]
fn spacing_holds_at_any_speed() {
    let camera = sphere_camera();
    let sphere = Sphere::unit();
    bolero::check!()
        .with_type::<((u8, u8), u8, Vec<(u8, u8)>)>()
        .for_each(|(offset, angle, steps)| {
            // 200 px through a point within 45 px of the centre: at most 145 px out, inside the
            // sphere's 163 px silhouette.
            let Some(samples) =
                property_stroke(screen(400.0, 300.0), 200.0, *offset, *angle, steps)
            else {
                return;
            };
            let (dabs, radius) = run(&plain(), &camera, &mut Sphere::unit(), &samples);
            let target = f64::from(Spacing::DEFAULT.get()) * radius;
            let gaps = gaps(&dabs, |a, b| sphere.arc(a, b));
            let error = worst(&gaps, target);
            assert!(
                error <= TOLERANCE,
                "worst gap {:.1} % off over {} gaps",
                100.0 * error,
                gaps.len()
            );
        });
}
