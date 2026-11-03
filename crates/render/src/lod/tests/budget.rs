//! The triangle budget when it binds (spec, "Triangle budget", "When it binds").
//!
//! The worst errors first, tiles admitted up to the budget with no cut-off at a bucket, charged
//! by what they draw, and the charge the sweep takes back returned.

use super::*;

/// The most one refinement is charged: four finest-sized children drawn where their parent
/// wasn't, in triangles.
const ONE_TILE: u32 = 4 * 2 * 64 * 64;

/// The test sphere (`cube_sphere(4)` to level 8, every tile resident, `e` its quad edge), seen
/// from 2.7 away: about 440 tiles drawn, 3.6M triangles with no budget.
fn sphere(gpu: &Gpu) -> (Scene, View) {
    let mesh = cube_sphere(4).expect("a mesh");
    let scene = Scene::new(gpu, mesh, level(8), |_| true, |_| 1.0);
    let view = look(Vec3::new(0.4, 0.6, 2.6), Vec3::default());
    (scene, view)
}

/// One frame's choice on `scene` from `view` with a budget of `budget` triangles (`None`: the
/// default, which nothing here reaches).
fn choose(gpu: &Gpu, scene: &mut Scene, view: &View, budget: Option<u32>) -> LodOutput {
    let config = LodConfig::default()
        .with_relax_passes(MAX_RELAX_PASSES)
        .expect("valid");
    let config = budget.map_or(config, |b| config.with_triangle_budget(b));
    let mut lod = Lod::new(gpu, scene.tree.capacity(), config).expect("the level choice");
    let requests = request_buffer(gpu, 4096);
    run(gpu, &mut lod, scene, &requests, view, None, None)
}

/// Inflates the bound of every display tile `pick` picks a thousandfold and widens its cone to
/// cover everything, as a stand-in's formula bound over a long previewed stroke does.
fn inflate(scene: &mut Scene, mut pick: impl FnMut(&Scene, Place) -> bool) {
    let picked: Vec<Place> = scene
        .details
        .keys()
        .copied()
        .filter(|place| place.level() < scene.finest && pick(scene, *place))
        .collect();
    for place in picked {
        let before = scene.details[&place];
        scene.set_detail(
            place,
            TileDetail {
                stored: before.stored,
                error: before.error * 1000.0,
                cone: FinestNormalCone::EVERYTHING,
            },
        );
    }
}

/// Whether `place` lies on the side of the unit sphere away from `view`'s eye, well past the
/// horizon: it and every tile under it face away.
fn far_side(scene: &Scene, view: &View, place: Place) -> bool {
    let eye = view.gpu().eye;
    let length = (eye[0] * eye[0] + eye[1] * eye[1] + eye[2] * eye[2]).sqrt();
    let c = scene.bounds[&place].center;
    let along = (c[0] * eye[0] + c[1] * eye[1] + c[2] * eye[2]) / length;
    along < -0.3
}

/// Whether `place` lies within `radius` of the sphere's point facing `view`'s eye.
fn near_the_facing_point(scene: &Scene, view: &View, place: Place, radius: f32) -> bool {
    let eye = view.gpu().eye;
    let length = (eye[0] * eye[0] + eye[1] * eye[1] + eye[2] * eye[2]).sqrt();
    let facing = eye.map(|x| x / length);
    let c = scene.bounds[&place].center;
    let d: f32 = (0..3).map(|a| (c[a] - facing[a]).powi(2)).sum();
    d.sqrt() < radius
}

/// Asserts the choice `output` charged at most `budget` and filled it to within one tile's
/// charge, and drew no more than it charged.
fn assert_filled(output: &LodOutput, budget: u32) {
    let stats = &output.stats;
    assert!(
        stats.refined_triangles <= budget,
        "over the budget: {stats:?}"
    );
    assert!(
        stats.refined_triangles + ONE_TILE > budget,
        "{} of {budget} triangles charged: {stats:?}",
        stats.refined_triangles,
    );
    assert!(
        stats.drawn_triangles <= stats.refined_triangles,
        "drew more than it charged: {stats:?}"
    );
}

#[test]
fn a_binding_budget_is_filled_to_within_one_tile_s_charge() {
    GpuTest::run(|gpu| {
        let (mut scene, view) = sphere(gpu);
        let unlimited = choose(gpu, &mut scene, &view, None);
        assert_eq!(unlimited.stats.over_pixel, 0, "{:?}", unlimited.stats);
        for divisor in [2, 3, 5, 9] {
            let budget = unlimited.stats.drawn_triangles / divisor;
            let capped = choose(gpu, &mut scene, &view, Some(budget));
            check_partition(&scene, &capped);
            check_neighbours(&scene, &capped, Some(&view));
            assert!(
                capped.stats.budget_limited > 0,
                "the cap bites: {:?}",
                capped.stats
            );
            assert_filled(&capped, budget);
        }
    });
}

#[test]
fn tiles_out_of_view_or_facing_away_spend_nothing() {
    GpuTest::run(|gpu| {
        let (mut plain, view) = sphere(gpu);
        let unlimited = choose(gpu, &mut plain, &view, None);
        let budget = unlimited.stats.drawn_triangles / 3;
        let expected = choose(gpu, &mut plain, &view, Some(budget));
        let (mut scene, _) = sphere(gpu);
        inflate(&mut scene, |s, place| far_side(s, &view, place));
        let inflated = choose(gpu, &mut scene, &view, Some(budget));
        check_partition(&scene, &inflated);
        assert_eq!(
            inflated.drawn, expected.drawn,
            "the tiles drawn: {:?} against {:?}",
            inflated.stats, expected.stats
        );
        assert_eq!(
            inflated.stats.refined_triangles, expected.stats.refined_triangles,
            "the charge"
        );
    });
}

#[test]
fn an_inflated_region_in_view_takes_its_share_and_the_rest_of_the_view_the_remainder() {
    GpuTest::run(|gpu| {
        let (mut scene, view) = sphere(gpu);
        let unlimited = choose(gpu, &mut scene, &view, None);
        let budget = unlimited.stats.drawn_triangles / 2;
        inflate(&mut scene, |s, place| {
            near_the_facing_point(s, &view, place, 0.15)
        });
        let capped = choose(gpu, &mut scene, &view, Some(budget));
        check_partition(&scene, &capped);
        check_neighbours(&scene, &capped, Some(&view));
        assert_filled(&capped, budget);
        // Its errors are the worst: the region is drawn at the finest level.
        let region: Vec<&TileKey> = capped
            .cut
            .iter()
            .filter(|key| near_the_facing_point(&scene, &view, Place::of(key), 0.1))
            .collect();
        assert!(!region.is_empty(), "the region is in the cut");
        assert!(
            region.iter().all(|key| key.level() == scene.finest),
            "the region at the finest level: {region:?}"
        );
        // The rest of the view is refined with what is left, not left as base faces.
        let rest_refined = capped
            .drawn
            .iter()
            .filter(|key| !near_the_facing_point(&scene, &view, Place::of(key), 0.3))
            .any(|key| key.level() > level(2));
        assert!(
            rest_refined,
            "the rest of the view refined: {:?}",
            capped.stats
        );
    });
}

#[test]
fn an_inflated_region_over_the_whole_budget_never_drops_the_view_to_its_base_faces() {
    GpuTest::run(|gpu| {
        let (mut scene, view) = sphere(gpu);
        let unlimited = choose(gpu, &mut scene, &view, None);
        let budget = unlimited.stats.drawn_triangles / 6;
        // A region whose finest tiles alone need more than the budget: on the live run, 3,207
        // stand-ins that drew the view as base faces.
        inflate(&mut scene, |s, place| {
            near_the_facing_point(s, &view, place, 0.6)
        });
        let capped = choose(gpu, &mut scene, &view, Some(budget));
        check_partition(&scene, &capped);
        check_neighbours(&scene, &capped, Some(&view));
        assert!(
            capped.cut.iter().any(|key| key.level() > level(2)),
            "the cut refined past the base faces: {:?}",
            capped.stats
        );
        assert_filled(&capped, budget);
    });
}

#[test]
fn a_binding_budget_chooses_the_same_cut_whatever_order_the_threads_ran_in() {
    GpuTest::run(|gpu| {
        let (mut scene, view) = sphere(gpu);
        let unlimited = choose(gpu, &mut scene, &view, None);
        let budget = unlimited.stats.drawn_triangles / 6;
        inflate(&mut scene, |s, place| {
            near_the_facing_point(s, &view, place, 0.6)
        });
        let first = choose(gpu, &mut scene, &view, Some(budget));
        assert!(
            first.stats.filled + first.stats.given_back > 0,
            "{:?}",
            first.stats
        );
        for _ in 0..4 {
            let again = choose(gpu, &mut scene, &view, Some(budget));
            assert_eq!(
                again.cut, first.cut,
                "{:?} against {:?}",
                again.stats, first.stats
            );
        }
    });
}

#[test]
fn a_budget_far_from_binding_skips_its_passes_and_takes_them_up_a_frame_after_it_binds() {
    GpuTest::run(|gpu| {
        let (mut scene, near) = sphere(gpu);
        let unlimited = choose(gpu, &mut scene, &near, None);
        let budget = unlimited.stats.drawn_triangles / 3;
        let far = look(Vec3::new(0.4, 0.6, 2.6) * 4.0, Vec3::default());
        let config = LodConfig::default()
            .with_triangle_budget(budget)
            .with_relax_passes(MAX_RELAX_PASSES)
            .expect("valid");
        let mut lod = Lod::new(gpu, scene.tree.capacity(), config).expect("the level choice");
        let requests = request_buffer(gpu, 4096);
        let mut frame = |view: &View| run(gpu, &mut lod, &mut scene, &requests, view, None, None);
        let distant = frame(&far);
        assert!(
            distant.stats.refined_triangles < budget / 2,
            "far from its budget: {:?}",
            distant.stats
        );
        // The frame after one far from its budget encodes no budget pass: binding now, it
        // stops at a bucket, with no fill and nothing given back.
        let first = frame(&near);
        assert!(
            first.stats.partial_bucket.is_some(),
            "binds: {:?}",
            first.stats
        );
        assert_eq!(
            (first.stats.filled, first.stats.given_back),
            (0, 0),
            "{:?}",
            first.stats
        );
        // The next has heard it bind, and fills the budget.
        let second = frame(&near);
        assert_filled(&second, budget);
    });
}
