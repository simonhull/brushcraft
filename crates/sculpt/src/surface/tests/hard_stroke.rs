//! A hard stroke is one smooth ridge: placed on its pen-down surface, whatever its dabs raised.
//!
//! The stroke engine draws a stroke through a pinhole camera onto `cube_sphere(4)` at level 8,
//! its pressure rising from 0.15 to full along the way, at the app's settings (a strength of a
//! tenth of the radius, in radii), and each push's dabs are applied before the next push, as the
//! app applies them. Placement reading the stroke's own surface climbed the ridge its dabs had
//! just raised: under hard pressure each dab's centre went up the ridge's front, its area normal
//! tilted forward up to 83°, and the stroke broke into separate leaning spikes, in twice the dabs
//! (2026-11-02: 86 dabs, centres to 1.8 radii above the surface, the crest swinging between 1.6
//! and 2.2 radii). On its pen-down surface, the same stroke places the dabs it places on an
//! untouched twin, bit for bit, and leaves one ridge whose crest rises with the pressure.

use std::collections::BTreeSet;

use geom::{Level, TileKey, Vec3, cube_sphere};
use store::{Project, SpatialIndex};
use stroke::{
    DVec3, Dab, Sample, ScreenPoint, Strength, StrengthMode, Stroke as PenStroke, StrokeId,
    StrokeSettings, Unit,
};

use super::Pinhole;
use crate::StoreSurface;
use crate::apply::apply_dabs;
use crate::state::Stroke;
use crate::testing::{TestDir, fixed, project_over};

/// The pen's samples: down the screen from `(320, 280)` to `(320, 520)`, 1.5 s, the pressure
/// rising from 0.15 to full.
fn samples() -> Vec<Sample> {
    let count = 160_u32;
    (0..count)
        .map(|k| {
            let u = f64::from(k) / f64::from(count - 1);
            let at = ScreenPoint::new(320.0, 280.0 + 240.0 * u).unwrap();
            #[expect(
                clippy::cast_possible_truncation,
                reason = "a pressure in 0.15..=1: f32 holds it"
            )]
            let pressure = (0.15 + 0.85 * u) as f32;
            Sample::new(at, 1.5 * u, Unit::new(pressure).unwrap()).unwrap()
        })
        .collect()
}

/// The sphere at level 8 under a 1 GiB budget, with its index.
fn sphere(dir: &TestDir) -> (Project, SpatialIndex) {
    let level = Level::new(8).unwrap();
    project_over(dir, cube_sphere(4).unwrap(), level, 1024, fixed(4))
}

/// The stroke of [`samples`] at the app's settings over `project`, each push's dabs applied to
/// it before the next if `apply`; the dabs placed and the stroke's frozen radius.
fn draw(project: &mut Project, index: &mut SpatialIndex, apply: bool) -> (Vec<Dab>, f64) {
    let camera = Pinhole {
        eye: DVec3::new(0.0, 0.0, 4.0),
        focal: 800.0,
    };
    let settings = StrokeSettings::default()
        .with_strength_mode(StrengthMode::Radius)
        .with_strength(Strength::new(0.1).unwrap());
    let samples = samples();
    let (first, rest) = samples.split_first().unwrap();
    let mut state = Stroke::new(project.store());
    project.begin_stroke_and_wait().unwrap();
    let mut down = StoreSurface::new(project.store(), index);
    let mut pen =
        PenStroke::begin(&settings, StrokeId::new(1), &camera, &mut down, *first).unwrap();
    down.finish().unwrap();
    let radius = pen.radius();
    let mut all: Vec<Dab> = Vec::new();
    for sample in rest {
        let mut surface = StoreSurface::new(project.store(), index);
        let dabs: Vec<Dab> = pen.push(&camera, &mut surface, *sample).collect();
        surface.finish().unwrap();
        land(project, index, &mut state, &dabs, apply);
        all.extend(dabs);
    }
    let mut up = StoreSurface::new(project.store(), index);
    let tail: Vec<Dab> = pen.end(&camera, &mut up).collect();
    up.finish().unwrap();
    land(project, index, &mut state, &tail, apply);
    all.extend(tail);
    project.end_unrecorded_stroke().unwrap();
    (all, radius)
}

/// Applies `dabs` to `project` in the stroke `state` keeps, if `apply`.
fn land(
    project: &mut Project,
    index: &mut SpatialIndex,
    state: &mut Stroke,
    dabs: &[Dab],
    apply: bool,
) {
    if apply {
        apply_dabs(project.store(), index, state, dabs).unwrap();
    }
}

/// `dab`'s centre.
fn centre(dab: &Dab) -> Vec3 {
    let [x, y, z] = dab.position;
    Vec3::new(x, y, z)
}

/// The crest along the stroke: for each dab, how high its part of the ridge stands.
///
/// That is the highest any vertex nearer the dab's centre's direction than any other dab's (and
/// within two radii of it) stands above the `untouched` surface, along that surface's outward
/// direction.
fn crests(
    stroked: &mut Project,
    untouched: &mut Project,
    index: &SpatialIndex,
    dabs: &[Dab],
    radius: f32,
) -> Vec<f32> {
    let directions: Vec<Vec3> = dabs
        .iter()
        .map(|dab| centre(dab).normalized().unwrap())
        .collect();
    let near: BTreeSet<TileKey> = directions
        .iter()
        .flat_map(|&d| index.query_sphere(d, 4.0 * radius).collect::<Vec<_>>())
        .collect();
    let mut crest = vec![0.0_f32; dabs.len()];
    let reach = (2.0 * radius).cos();
    for key in near {
        let now = stroked.store().evaluate(key).unwrap().to_vec();
        let before = untouched.store().evaluate(key).unwrap().to_vec();
        for (p, q) in now.into_iter().zip(before) {
            let height = (p - q).dot(q.normalized().unwrap());
            if height <= 0.0 {
                continue;
            }
            let v = p.normalized().unwrap();
            let nearest = directions
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.dot(v).total_cmp(&b.1.dot(v)));
            if let Some((k, d)) = nearest
                && d.dot(v) > reach
                && let Some(top) = crest.get_mut(k)
            {
                *top = top.max(height);
            }
        }
    }
    crest
}

#[test]
fn a_hard_stroke_is_placed_at_pen_down_and_rises_as_one_smooth_ridge() {
    let (dir, twin_dir) = (TestDir::new(), TestDir::new());
    let (mut stroked, mut index) = sphere(&dir);
    let (mut untouched, mut twin_index) = sphere(&twin_dir);
    let (dabs, radius) = draw(&mut stroked, &mut index, true);
    let (placed, _) = draw(&mut untouched, &mut twin_index, false);
    #[expect(clippy::cast_possible_truncation, reason = "a dab's radius is f32")]
    let radius = radius as f32;

    // Placed on the pen-down surface: the dabs a stroke over the untouched twin places.
    assert_eq!(dabs.len(), placed.len(), "as many dabs");
    for (k, (dab, twin)) in dabs.iter().zip(&placed).enumerate() {
        assert_eq!(dab, twin, "dab {k}, whatever the dabs before it raised");
    }
    assert!(
        dabs.len() < 60,
        "{} dabs: the stroke doesn't pile up",
        dabs.len()
    );

    // Its normals are the pen-down surface's: within a few degrees of the sphere's radial.
    for (k, dab) in dabs.iter().enumerate() {
        let [nx, ny, nz] = dab.normal;
        let tilt = Vec3::new(nx, ny, nz)
            .dot(centre(dab).normalized().unwrap())
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees();
        assert!(tilt < 8.0, "dab {k} tilts {tilt}° off the radial");
    }

    // One ridge, its crest rising with the pressure up to the last radius (where no later dab
    // adds to it), never by a spike.
    let crest = crests(&mut stroked, &mut untouched, &index, &dabs, radius);
    let last = dabs.last().map(centre).unwrap();
    let tail = dabs
        .iter()
        .position(|dab| (centre(dab) - last).length() < 1.2 * radius)
        .unwrap();
    let (rising, _) = crest
        .split_at_checked(tail)
        .expect("the tail is the stroke's");
    for (k, pair) in rising.windows(2).enumerate() {
        let [was, now] = pair else { continue };
        assert!(
            *now >= *was - 0.005 * radius,
            "the crest falls at dab {}: {was} to {now} (radius {radius})",
            k + 1
        );
        assert!(
            *now - *was < 0.05 * radius,
            "a spike at dab {}: {was} to {now} (radius {radius})",
            k + 1
        );
    }
    let top = rising.iter().copied().fold(0.0_f32, f32::max);
    println!(
        "{} dabs, the crest rising to {:.3} radii by dab {tail}",
        dabs.len(),
        top / radius
    );
    assert!(
        top > 0.4 * radius,
        "the ridge stands {top}, over 0.4 radii at full pressure (radius {radius})"
    );
    drop(stroked.close(false));
    drop(untouched.close(false));
}
