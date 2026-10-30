//! No tile the level choice culls by its normal cone has a triangle facing the eye mid-morph.

use super::*;

/// Whether `p` projects into `view`'s image, in front of the eye.
fn in_image(view: &View, p: [f32; 3]) -> bool {
    let point = Vec3::new(f64::from(p[0]), f64::from(p[1]), f64::from(p[2]));
    view.pixel(point).is_some_and(|(x, y, z)| {
        let (width, height) = (f64::from(size().width()), f64::from(size().height()));
        (0.0..width).contains(&x) && (0.0..height).contains(&y) && z > 0.0
    })
}

/// How many of the cluster's triangles, its 81 vertices drawn at `placed`, face `view`'s eye
/// with a corner in its image.
fn facing_in_view(view: &View, placed: &[[f32; 4]]) -> usize {
    let eye = view.eye();
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the eye near the unit sphere, in f32 as the GPU has it"
    )]
    let eye = [eye.x as f32, eye.y as f32, eye.z as f32];
    let at = |g: GridVertex| {
        let p = placed[usize::from(g.index())];
        [p[0], p[1], p[2]]
    };
    let facing = indices::triangles().filter(|triangle| {
        let [first, second, third] = triangle.map(at);
        let normal = crate::cull::reference::face_normal(first, second, third);
        let towards = [0, 1, 2].map(|a| eye[a] - first[a]);
        let faces = crate::cull::reference::dot(normal, towards) > 0.0;
        faces && [first, second, third].iter().any(|&p| in_image(view, p))
    });
    facing.count()
}

#[test]
fn a_tile_culled_by_its_cone_has_no_triangle_facing_the_eye_mid_morph() {
    GpuTest::run(|gpu| {
        let mut world = World::new(gpu);
        // Bounds thirty times the sphere's (its data unchanged), so the level choice refines to
        // the finest level's four tiles a face, small enough for the cone to cull some.
        let display = world.tiles.keys().filter(|p| p.level() < world.finest);
        let display: Vec<Place> = display.copied().collect();
        for place in display {
            let error = world.tiles[&place].detail.error * 30.0;
            set_error(gpu, &mut world, place, error);
        }
        let mut rng = Rng(17);
        let (mut lod, requests) = level_choice(gpu, &world);
        let mut raster = Raster::new(
            gpu,
            world.tree.capacity(),
            world.layout(),
            RasterConfig::default(),
        )
        .expect("the raster");
        let (mut culled_tiles, mut checked) = (0, 0);
        for seed in 0..4 {
            // Moving to a view, then a little closer with the camera taken as still, so the
            // tiles refining there start morphing by time; 50 ms on, they are a third of the
            // way (and by distance wherever their bounds put them).
            let (far, distance) = random_view(&mut rng, (1.5, 3.0));
            let far_eye = far.eye();
            let scale = (distance - 0.3) / distance;
            let near_eye = Vec3::new(far_eye.x * scale, far_eye.y * scale, far_eye.z * scale);
            let view = [Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, 1.0)]
                .into_iter()
                .find_map(|up| {
                    View::look_at(
                        near_eye,
                        Vec3::default(),
                        up,
                        45_f64.to_radians(),
                        size(),
                        0.01,
                    )
                })
                .expect("a view");
            let frames = [
                (&far, 900, true),
                (&view, 1000, false),
                (&view, 1050, false),
            ];
            let mut last = None;
            for (k, (at, ms, moving)) in frames.into_iter().enumerate() {
                let frame = RasterFrame {
                    view: at,
                    time: Duration::from_millis(ms),
                    moving,
                };
                let bounds = seed == 0 && k == 0;
                last = Some(lod_frame(
                    gpu,
                    &world,
                    &mut lod,
                    &requests,
                    &mut raster,
                    &frame,
                    bounds,
                ));
            }
            let (_, listed, cut) = last.expect("three frames");
            let drawn: BTreeSet<u32> = listed.iter().map(|c| c.node).collect();
            // Probe every vertex of every cluster of the tiles the frame didn't draw.
            let all = cut_clusters(&world, &cut, true);
            let (on_gpu, _) = cluster_buffers(gpu, &all);
            let culled = all
                .iter()
                .enumerate()
                .filter(|(_, c)| !drawn.contains(&c.node));
            let culled: Vec<(usize, &GpuCluster)> = culled.collect();
            let culled_nodes: BTreeSet<u32> = culled.iter().map(|(_, c)| c.node).collect();
            culled_tiles += culled_nodes.len();
            let mut probes = Vec::new();
            for &(instance, _) in &culled {
                let instance = u32::try_from(instance).expect("small");
                let corners = GridVertex::all().map(|g| [instance, u32::from(g.index())]);
                probes.extend(corners);
            }
            if probes.is_empty() {
                continue;
            }
            let placed = probe(gpu, &raster, &world, &on_gpu, &probes);
            for cluster in placed.chunks(indices::CLUSTER_VERTICES) {
                let faced = facing_in_view(&view, cluster);
                checked += indices::CLUSTER_TRIANGLES;
                assert_eq!(
                    faced, 0,
                    "seed {seed}: a culled tile's triangle faces the eye"
                );
            }
        }
        println!("{culled_tiles} culled tiles, {checked} triangles checked");
        assert!(culled_tiles > 0 && checked > 10_000, "culling was tested");
    });
}
