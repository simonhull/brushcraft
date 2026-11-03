//! The triangle budget when it binds (spec, "Triangle budget", "When it binds").
//!
//! The worst needs first over the whole view, tiles admitted up to the budget to within a fine
//! bin with no cut-off at a bucket, charged by what they draw, the same cut every frame.

use super::*;

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

/// Asserts the choice `output` bound and filled `budget` to within the slack it reports.
///
/// It charged at most `budget`, left unspent no more than its slack (the first fine bin it
/// couldn't take and its held tiles, plus the rounding of the budget to units), and drew no more
/// than it charged.
fn assert_filled(output: &LodOutput, budget: u32) {
    let stats = &output.stats;
    assert!(stats.budget_binds, "binds: {stats:?}");
    assert!(
        stats.refined_triangles <= budget,
        "over the budget: {stats:?}"
    );
    let unspent = budget - stats.refined_triangles;
    assert!(
        unspent <= stats.budget_slack + BUDGET_UNIT,
        "{unspent} of {budget} triangles unspent: {stats:?}"
    );
    assert!(
        stats.drawn_triangles <= stats.refined_triangles,
        "drew more than it charged: {stats:?}"
    );
}

/// `place`'s rank as the GPU orders it: its `e` projected over its own sphere, capped by its
/// ancestors', 0 out of view; `None` where in view is too close to call.
pub(super) fn cpu_rank(scene: &Scene, view: &View, place: Place) -> Option<f32> {
    let gpu = view.gpu();
    let eye = [gpu.eye[0], gpu.eye[1], gpu.eye[2]];
    let mut rank = f32::MAX;
    let mut at = Some(place);
    while let Some(p) = at {
        let b = scene.bounds[&p];
        let e = scene.details[&p].error;
        let wide = in_frustum(&gpu, b.center, (b.radius + e) * 1.001)
            && !scene.refine_facing_away(p, eye, e * 1.001 + b.radius * 0.001, b.radius * 0.001);
        let narrow = in_frustum(&gpu, b.center, (b.radius + e) * 0.999)
            && !scene.refine_facing_away(p, eye, e * 0.999 - b.radius * 0.001, -b.radius * 0.001);
        if wide != narrow {
            return None;
        }
        rank = rank.min(if wide {
            projected(&gpu, &b, 0.0, e)
        } else {
            0.0
        });
        at = p.parent();
    }
    Some(rank)
}

/// Asserts the budget's choice `output` went worst first.
///
/// No tile it left in the cut ranks above the bucket it stopped in ([`LodStats::budget_error`],
/// an eighth of an octave, with 5 % for the CPU's arithmetic against the GPU's), but those it
/// held for their seams.
pub(super) fn assert_worst_first(scene: &Scene, view: &View, output: &LodOutput) {
    let Some(error) = output.stats.budget_error() else {
        return;
    };
    let above = error * 2_f32.powf(1.0 / 8.0) * 1.05;
    let left_over: Vec<(Place, f32)> = places(output)
        .into_iter()
        .filter(|p| p.level() < scene.finest)
        .filter_map(|p| cpu_rank(scene, view, p).map(|rank| (p, rank)))
        .filter(|(_, rank)| *rank > above)
        .collect();
    let held = usize::try_from(output.stats.held).expect("small");
    assert!(
        left_over.len() <= held,
        "{} tiles of the cut ranked over {above} px: {left_over:?}, {:?}",
        left_over.len(),
        output.stats
    );
}

#[test]
fn a_binding_budget_is_filled_to_within_its_slack() {
    GpuTest::run(|gpu| {
        let (mut scene, view) = sphere(gpu);
        let unlimited = choose(gpu, &mut scene, &view, None);
        assert_eq!(unlimited.stats.over_pixel, 0, "{:?}", unlimited.stats);
        assert!(!unlimited.stats.budget_binds, "{:?}", unlimited.stats);
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
            // A fine bin is a 128th of an eighth of an octave, at one level: a tile or two.
            assert!(
                capped.stats.budget_slack <= 4 * 2 * 64 * 64 * 2,
                "{:?}",
                capped.stats
            );
        }
    });
}

#[test]
fn tiles_certainly_culled_out_of_view_or_facing_away_spend_nothing() {
    GpuTest::run(|gpu| {
        let (mut plain, view) = sphere(gpu);
        let unlimited = choose(gpu, &mut plain, &view, None);
        let budget = unlimited.stats.drawn_triangles / 3;
        let expected = choose(gpu, &mut plain, &view, Some(budget));
        let (mut scene, _) = sphere(gpu);
        inflate(&mut scene, |s, place| far_side(s, &view, place));
        // No morph: every tile facing away is settled, so certainly culled.
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
        // The budget's choice keeps the neighbour rule.
        assert!(capped.stats.admitted > 0, "{:?}", capped.stats);
        check_neighbours(&scene, &capped, Some(&view));
        assert!(
            capped.cut.iter().any(|key| key.level() > level(2)),
            "the cut refined past the base faces: {:?}",
            capped.stats
        );
        assert_filled(&capped, budget);
        // The sweep's takebacks of the first choice don't count: only what the budget's choice
        // held for its seams.
        assert_eq!(
            capped.stats.shortfall, capped.stats.held,
            "{:?}",
            capped.stats
        );
    });
}

#[test]
fn the_charge_goes_to_the_worst_needs_not_to_lower_errors_elsewhere() {
    GpuTest::run(|gpu| {
        // Errors a tenth of the quad edge: the tiles want refining by the quad edge's 4 px, at
        // low ranks; the inflated region's ranks are far above, and the neighbour rule needs
        // tiles of lower rank around it.
        let mesh = cube_sphere(4).expect("a mesh");
        let mut scene = Scene::new(gpu, mesh, level(8), |_| true, |_| 0.1);
        let view = look(Vec3::new(0.4, 0.6, 2.6), Vec3::default());
        let unlimited = choose(gpu, &mut scene, &view, None);
        let budget = unlimited.stats.drawn_triangles / 3;
        inflate(&mut scene, |s, place| {
            near_the_facing_point(s, &view, place, 0.15)
        });
        let capped = choose(gpu, &mut scene, &view, Some(budget));
        check_partition(&scene, &capped);
        check_neighbours(&scene, &capped, Some(&view));
        assert_filled(&capped, budget);
        let region_finest = capped
            .cut
            .iter()
            .filter(|key| near_the_facing_point(&scene, &view, Place::of(key), 0.1))
            .all(|key| key.level() == scene.finest);
        assert!(region_finest, "the worst first: {:?}", capped.stats);
        assert_worst_first(&scene, &view, &capped);
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
        assert!(first.stats.admitted > 0, "{:?}", first.stats);
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
fn the_frame_a_budget_starts_binding_in_is_chosen_by_it_with_no_cut_off() {
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
            !distant.stats.budget_binds,
            "far from its budget: {:?}",
            distant.stats
        );
        // The first frame that binds, with nothing read back saying so yet.
        let onset = frame(&near);
        assert_filled(&onset, budget);
    });
}

#[test]
fn a_budget_under_the_level_0_tiles_binds_and_stops_nowhere() {
    GpuTest::run(|gpu| {
        let (mut scene, view) = sphere(gpu);
        let tiny = choose(gpu, &mut scene, &view, Some(16));
        check_partition(&scene, &tiny);
        assert!(tiny.stats.budget_binds, "{:?}", tiny.stats);
        assert_eq!(tiny.stats.mark_bucket, None, "{:?}", tiny.stats);
        assert_eq!(tiny.stats.budget_bucket, None, "{:?}", tiny.stats);
        assert!(
            tiny.cut.iter().all(|key| key.level() == level(0) || {
                // Only refinements charged nothing: tiles out of view.
                !tiny.drawn.contains(key)
            }),
            "{:?}",
            tiny.stats
        );
    });
}

/// A scene and a budget where mark takes every bucket but the relaxation can't fit its raises.
///
/// An inflated region is refined to the finest level, and the relaxation must raise its
/// neighbours, which want less. The budget is the least mark takes whole, found by bisection
/// (mark takes every bucket exactly when the budget holds them all).
fn relax_refusing(gpu: &Gpu) -> (Scene, View, u32) {
    // Errors a twentieth of the quad edge: outside the region the quad edge's 4 px decides,
    // levels short of the region's, so the relaxation raises tiles in view around it.
    let mesh = cube_sphere(4).expect("a mesh");
    let mut scene = Scene::new(gpu, mesh, level(8), |_| true, |_| 0.05);
    let view = look(Vec3::new(0.4, 0.6, 2.6), Vec3::default());
    inflate(&mut scene, |s, place| {
        near_the_facing_point(s, &view, place, 0.15)
    });
    let unlimited = choose(gpu, &mut scene, &view, None);
    let (mut low, mut high) = (0, unlimited.stats.refined_triangles);
    while high - low > BUDGET_UNIT {
        let mid = low + (high - low) / 2;
        let output = choose(gpu, &mut scene, &view, Some(mid));
        if output.stats.mark_bucket.is_some() {
            low = mid;
        } else {
            high = mid;
        }
    }
    let output = choose(gpu, &mut scene, &view, Some(high));
    assert!(
        output.stats.mark_bucket.is_none() && output.stats.relax_refused > 0,
        "mark takes every bucket and the relaxation can't fit: {:?}",
        output.stats
    );
    (scene, view, high)
}

#[test]
fn a_relaxation_the_budget_refuses_makes_the_budget_bind_though_mark_took_every_bucket() {
    GpuTest::run(|gpu| {
        let (mut scene, view, budget) = relax_refusing(gpu);
        let output = choose(gpu, &mut scene, &view, Some(budget));
        check_partition(&scene, &output);
        check_neighbours(&scene, &output, Some(&view));
        assert!(output.stats.budget_binds, "{:?}", output.stats);
        assert!(
            output.stats.refined_triangles <= budget,
            "{:?}",
            output.stats
        );
        assert!(
            output.stats.drawn_triangles <= output.stats.refined_triangles,
            "{:?}",
            output.stats
        );
    });
}

#[test]
fn a_still_camera_over_a_refusing_relaxation_keeps_the_same_cut_every_frame() {
    GpuTest::run(|gpu| {
        let (mut scene, view, budget) = relax_refusing(gpu);
        let config = LodConfig::default()
            .with_triangle_budget(budget)
            .with_relax_passes(MAX_RELAX_PASSES)
            .expect("valid");
        let mut lod = Lod::new(gpu, scene.tree.capacity(), config).expect("the level choice");
        let requests = request_buffer(gpu, 4096);
        let outputs: Vec<LodOutput> = (0..6)
            .map(|_| run(gpu, &mut lod, &mut scene, &requests, &view, None, None))
            .collect();
        let settled = outputs.get(1).expect("frames");
        for (frame, output) in outputs.iter().enumerate().skip(2) {
            assert_eq!(
                output.cut, settled.cut,
                "frame {frame}: {:?} against {:?}",
                output.stats, settled.stats
            );
        }
    });
}

/// A frame's choice with the raster's morph on, the distance ramp at `ramp` pixels.
///
/// With a duration and every stamp zero (every tile new to the cut, the camera still), no tile
/// with a parent is settled at its end, and emit draws every tile facing away that isn't settled
/// at its start.
fn run_morphing(gpu: &Gpu, lod: &mut Lod, scene: &mut Scene, view: &View, ramp: f32) -> LodOutput {
    let requests = request_buffer(gpu, 4096);
    let stamps = gpu.device().create_buffer(&BufferDescriptor {
        label: Some("test stamps"),
        size: u64::from(scene.tree.capacity().max(1)) * 16,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params = gpu.device().create_buffer(&BufferDescriptor {
        label: Some("test raster params"),
        size: 32,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // GpuRasterParams: frame, now, duration, flags, ramp, count word, reserved.
    let words: [u32; 8] = [5, 1000, 150, 0, ramp.to_bits(), 0, 0, 0];
    gpu.queue()
        .write_buffer(&params, 0, bytemuck::cast_slice(&words));
    scene.upload(gpu, lod);
    gpu.queue().write_buffer(&requests, 0, &[0; 16]);
    let mut encoder = gpu
        .device()
        .create_command_encoder(&CommandEncoderDescriptor::default());
    let inputs = MainInputs {
        tree: scene.tree.buffer(),
        requests: &requests,
        view,
        previous: None,
        finest: scene.finest,
        morph: Some(MorphInputs {
            stamps: &stamps,
            params: &params,
        }),
    };
    lod.encode_main(gpu, &mut encoder, &inputs)
        .expect("encoded");
    let readback = lod
        .encode_readback(gpu, &mut encoder, 0)
        .expect("encoded")
        .expect("a readback buffer is free");
    gpu.queue().submit([encoder.finish()]);
    readback.submitted();
    gpu.device()
        .poll(PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .expect("the GPU finishes");
    gpu.check().expect("no GPU error");
    lod.collect(scene.finest)
        .expect("read back")
        .expect("the frame's results arrived")
}

#[test]
fn with_the_morph_on_the_tiles_drawn_facing_away_are_charged_and_the_budget_bounds_the_drawn() {
    GpuTest::run(|gpu| {
        let (mut scene, view) = sphere(gpu);
        let unlimited = choose(gpu, &mut scene, &view, None);
        for budget in [None, Some(unlimited.stats.drawn_triangles / 3)] {
            let config = LodConfig::default()
                .with_relax_passes(MAX_RELAX_PASSES)
                .expect("valid");
            let config = budget.map_or(config, |b| config.with_triangle_budget(b));
            let mut lod = Lod::new(gpu, scene.tree.capacity(), config).expect("the level choice");
            let output = run_morphing(gpu, &mut lod, &mut scene, &view, 1.0);
            let stats = &output.stats;
            assert!(stats.cone_exempt > 0, "tiles facing away drawn: {stats:?}");
            assert!(
                stats.drawn_triangles <= stats.refined_triangles,
                "drew more than it charged: {stats:?}"
            );
            if let Some(budget) = budget {
                assert_filled(&output, budget);
            }
            assert_eq!(stats.overflow, (0, 0, 0), "{stats:?}");
        }
    });
}

/// Whether `place`'s tile lies wholly outside `view`'s frustum, by its own sphere with a margin.
fn out_of_view(scene: &Scene, view: &View, place: Place) -> bool {
    let b = scene.bounds[&place];
    !in_frustum(&view.gpu(), b.center, b.radius * 1.01)
}

#[test]
fn with_the_morph_on_tiles_out_of_view_spend_nothing() {
    GpuTest::run(|gpu| {
        // Close enough that part of the sphere lies out of the frustum.
        let close = look(Vec3::new(0.2, 0.3, 1.3), Vec3::default());
        let (mut plain, _) = sphere(gpu);
        let config = LodConfig::default()
            .with_relax_passes(MAX_RELAX_PASSES)
            .expect("valid");
        let unlimited = {
            let mut lod = Lod::new(gpu, plain.tree.capacity(), config).expect("the level choice");
            run_morphing(gpu, &mut lod, &mut plain, &close, 1.0)
        };
        let config = config.with_triangle_budget(unlimited.stats.drawn_triangles / 3);
        let mut capped = Lod::new(gpu, plain.tree.capacity(), config).expect("the level choice");
        let expected = run_morphing(gpu, &mut capped, &mut plain, &close, 1.0);
        let (mut scene, _) = sphere(gpu);
        let mut picked = 0;
        inflate(&mut scene, |s, place| {
            let out = out_of_view(s, &close, place);
            picked += usize::from(out);
            out
        });
        assert!(picked > 0, "tiles out of view");
        let mut again = Lod::new(gpu, scene.tree.capacity(), config).expect("the level choice");
        let inflated = run_morphing(gpu, &mut again, &mut scene, &close, 1.0);
        check_partition(&scene, &inflated);
        assert!(inflated.stats.budget_binds, "{:?}", inflated.stats);
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
fn a_cascade_past_the_silhouette_to_the_back_faces_settles_and_keeps_the_cut() {
    GpuTest::run(|gpu| {
        // Deep detail just in front of the silhouette (levels 8 and 9 inflated, cube_sphere(1)
        // to level 10): its neighbours across the silhouette face away and want nothing, so the
        // neighbour rule needs them and their ancestors, back faces down to level 0, raised level
        // by level.
        let mesh = cube_sphere(1).expect("a mesh");
        let mut scene = Scene::new(gpu, mesh, level(10), |_| true, |_| 1.0);
        let view = look(Vec3::new(0.4, 0.6, 2.6), Vec3::default());
        let eye = view.gpu().eye;
        let length = (eye[0] * eye[0] + eye[1] * eye[1] + eye[2] * eye[2]).sqrt();
        inflate(&mut scene, |s, place| {
            let c = s.bounds[&place].center;
            let along = (c[0] * eye[0] + c[1] * eye[1] + c[2] * eye[2]) / length;
            place.level() >= level(8) && (0.0..0.15).contains(&along)
        });
        let unlimited = choose(gpu, &mut scene, &view, None);
        let config = LodConfig::default()
            .with_triangle_budget(unlimited.stats.drawn_triangles / 2)
            .with_relax_passes(MAX_RELAX_PASSES)
            .expect("valid");
        let mut lod = Lod::new(gpu, scene.tree.capacity(), config).expect("the level choice");
        let requests = request_buffer(gpu, 4096);
        let outputs: Vec<LodOutput> = (0..4)
            .map(|_| run(gpu, &mut lod, &mut scene, &requests, &view, None, None))
            .collect();
        let first = outputs.first().expect("frames");
        assert!(first.stats.budget_binds, "{:?}", first.stats);
        println!("spreading passes needed: {}", first.stats.need_passes);
        for (frame, output) in outputs.iter().enumerate() {
            assert!(
                !output.stats.need_unsettled,
                "frame {frame}: {:?}",
                output.stats
            );
            check_partition(&scene, output);
            check_neighbours(&scene, output, Some(&view));
        }
        let settled = outputs.get(1).expect("frames");
        for (frame, output) in outputs.iter().enumerate().skip(2) {
            assert_eq!(output.cut, settled.cut, "frame {frame}: {:?}", output.stats);
        }
    });
}

#[test]
fn needs_under_an_absent_tile_are_not_charged_and_the_budget_is_still_filled() {
    GpuTest::run(|gpu| {
        // A few level-7 tiles in view absent while all their descendants stay: their parents
        // are incomplete, so nothing under them can be reached, whatever it needs.
        let (probe, view) = sphere(gpu);
        let absent: BTreeSet<Place> = probe
            .bounds
            .keys()
            .copied()
            .filter(|p| p.level() == level(7))
            .filter(|p| near_the_facing_point(&probe, &view, *p, 0.4))
            .step_by(3)
            .collect();
        assert!(!absent.is_empty(), "tiles to leave out");
        drop(probe);
        let mesh = cube_sphere(4).expect("a mesh");
        let mut scene = Scene::new(gpu, mesh, level(8), |p| !absent.contains(&p), |_| 1.0);
        inflate(&mut scene, |s, place| {
            near_the_facing_point(s, &view, place, 0.5)
        });
        let unlimited = choose(gpu, &mut scene, &view, None);
        let budget = unlimited.stats.drawn_triangles / 3;
        let capped = choose(gpu, &mut scene, &view, Some(budget));
        check_partition(&scene, &capped);
        check_neighbours(&scene, &capped, Some(&view));
        assert_filled(&capped, budget);
        // Nothing counted that can't be refined: what is left is the stopping bin, a tile or two.
        let unspent = budget - capped.stats.refined_triangles;
        assert!(
            unspent <= 2 * 4 * 2 * 64 * 64,
            "{unspent} unspent: {:?}",
            capped.stats
        );
    });
}

#[test]
fn a_camera_moving_slowly_at_the_budget_s_edge_never_swaps_back_and_forth() {
    GpuTest::run(|gpu| {
        let (mut scene, view) = sphere(gpu);
        inflate(&mut scene, |s, place| {
            near_the_facing_point(s, &view, place, 0.15)
        });
        let unlimited = choose(gpu, &mut scene, &view, None);
        let config = LodConfig::default()
            .with_triangle_budget(unlimited.stats.drawn_triangles * 9 / 10)
            .with_relax_passes(MAX_RELAX_PASSES)
            .expect("valid");
        let mut lod = Lod::new(gpu, scene.tree.capacity(), config).expect("the level choice");
        let requests = request_buffer(gpu, 4096);
        let mut cuts: Vec<BTreeSet<TileKey>> = Vec::new();
        let mut binding = 0;
        for frame in 0..24_u32 {
            // A slow orbit: a tenth of a degree a frame, each view seen twice.
            let angle = f64::from(frame / 2) * 0.1_f64.to_radians();
            let eye = Vec3::new(2.7 * angle.sin(), 0.6, 2.7 * angle.cos());
            let output = run(
                gpu,
                &mut lod,
                &mut scene,
                &requests,
                &look(eye, Vec3::default()),
                None,
                None,
            );
            binding += usize::from(output.stats.budget_binds);
            cuts.push(output.cut);
        }
        assert!(binding > 0, "the budget binds");
        for at in 2..cuts.len() {
            let (a, b, c) = (&cuts[at - 2], &cuts[at - 1], &cuts[at]);
            assert!(
                !(a != b && a == c),
                "frames {} to {at} swap back and forth",
                at - 2
            );
        }
        // Each view seen twice: the second frame of a still view keeps the first's cut, after
        // the first frame's hysteresis has settled.
        for pair in cuts.chunks(2).skip(1) {
            assert_eq!(pair[0], pair[1], "a still view's cut changed");
        }
    });
}
