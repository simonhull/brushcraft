//! The level choice cone-culls a tile only when its morph is settled, so no tile it culls by its
//! cone has a triangle facing the eye mid-morph.

use super::*;

/// The patch that takes emit's settled test away: every tile its cone faces away is culled,
/// morphing or not (the bite).
pub(crate) const CULL_WHILE_MORPHING: (&str, &str) =
    ("if settled(n, b, edges >> 4u) {", "if true {");

/// The probe's `w`: the vertex's morph factor `t` where its tile vertex is odd (it may move), -1
/// where it is even (it never does).
const ODD_T: &str = "select(-1.0, placed.t, ((placed.at.x | placed.at.y) & 1u) == 1u)";

/// What the tiles the level choice culled by their cone showed, over the scene's frames.
#[derive(Debug, Default)]
struct Culled {
    /// Tiles culled by their cone.
    tiles: usize,
    /// Their triangles checked.
    triangles: usize,
    /// ... facing the eye, a corner in the image.
    facing: usize,
    /// Culled tiles with an odd vertex on its way (`0 < t < 1`), or some at each end.
    unsettled: usize,
    /// Tiles the cone would have culled but whose morph isn't settled, drawn.
    exempt: u32,
}

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

/// Whether a tile's odd vertices, `ts` their morph factors (-1 for even ones, skipped), are all
/// at one end of the morph: all 1 or all 0.
fn settled(ts: impl Iterator<Item = f32>) -> bool {
    let odd: Vec<f32> = ts.filter(|&t| t >= 0.0).collect();
    odd.iter().all(|&t| t >= 1.0) || odd.iter().all(|&t| t == 0.0)
}

/// The culled tiles of the scene's mid-morph frames, the raster morphing by `config`.
///
/// The scene: the sphere with its display tiles' bounds thirty times their size (data
/// unchanged), so the level choice refines to the finest level's four tiles a face, small enough
/// for the cone to cull some. From four random views, a frame moving, then a little closer with
/// the camera still, then 50 ms on, so tiles refining there morph by time a third of the way (and
/// by distance wherever their bounds put them, under `config`'s ramp). The level choice's shader
/// gets `patches`. Each third frame's culled tiles are probed.
fn culled_mid_morph(gpu: &Gpu, config: RasterConfig, patches: &[(&str, &str)]) -> Culled {
    let mut world = World::new(gpu);
    let display = world.tiles.keys().filter(|p| p.level() < world.finest);
    let display: Vec<Place> = display.copied().collect();
    for place in display {
        let error = world.tiles[&place].detail.error * 30.0;
        set_error(gpu, &mut world, place, error);
    }
    let mut rng = Rng(17);
    let (mut lod, requests) = level_choice_patched(gpu, &world, patches);
    let mut raster =
        Raster::new(gpu, world.tree.capacity(), world.layout(), config).expect("the raster");
    let mut culled = Culled::default();
    for seed in 0..4 {
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
        let steps = frames.into_iter().enumerate();
        for (k, (at, ms, moving)) in steps {
            let frame = RasterFrame {
                view: at,
                time: Duration::from_millis(ms),
                moving,
            };
            let bounds = seed == 0 && k == 0;
            last = Some(lod_frame_output(
                gpu,
                &world,
                &mut lod,
                &requests,
                &mut raster,
                &frame,
                bounds,
            ));
        }
        let (_, listed, output) = last.expect("three frames");
        tally(gpu, &world, &raster, &view, &listed, &output, &mut culled);
    }
    culled
}

/// Adds to `culled` what the tiles of `output`'s cut that `listed` doesn't draw and that lie in
/// `view`'s image show, probed where `raster` placed them.
fn tally(
    gpu: &Gpu,
    world: &World,
    raster: &Raster,
    view: &View,
    listed: &[GpuCluster],
    output: &LodOutput,
    culled: &mut Culled,
) {
    culled.exempt += output.stats.cone_exempt;
    let cut: BTreeSet<Place> = output.nodes.iter().map(|&(_, p)| p).collect();
    let drawn: BTreeSet<u32> = listed.iter().map(|c| c.node).collect();
    // Probe every vertex of every cluster of the tiles the frame didn't draw.
    let all = cut_clusters(world, &cut, true);
    let (on_gpu, _) = cluster_buffers(gpu, &all);
    let undrawn = all
        .iter()
        .enumerate()
        .filter(|(_, c)| !drawn.contains(&c.node));
    let undrawn: Vec<(usize, &GpuCluster)> = undrawn.collect();
    let mut probes = Vec::new();
    for &(instance, _) in &undrawn {
        let instance = u32::try_from(instance).expect("small");
        let corners = GridVertex::all().map(|g| [instance, u32::from(g.index())]);
        probes.extend(corners);
    }
    if probes.is_empty() {
        return;
    }
    let placed = probe_with(gpu, raster, world, &on_gpu, &probes, ODD_T);
    let mut by_node: BTreeMap<u32, Vec<&[[f32; 4]]>> = BTreeMap::new();
    let tiles = undrawn.iter().zip(placed.chunks(indices::CLUSTER_VERTICES));
    for ((_, c), cluster) in tiles {
        by_node.entry(c.node).or_default().push(cluster);
    }
    for tile in by_node.values() {
        // A tile with no vertex in the image was culled by the frustum (no occlusion here): the
        // rest by their cone.
        let points = tile.iter().flat_map(|c| c.iter());
        let in_view = points.clone().any(|p| in_image(view, [p[0], p[1], p[2]]));
        if !in_view {
            continue;
        }
        culled.tiles += 1;
        culled.triangles += tile.len() * indices::CLUSTER_TRIANGLES;
        culled.facing += tile.iter().map(|c| facing_in_view(view, c)).sum::<usize>();
        culled.unsettled += usize::from(!settled(points.map(|p| p[3])));
    }
}

/// The raster with its distance factor (the default), and by time alone.
fn configs() -> [RasterConfig; 2] {
    let by_time = RasterConfig::default().with_ramp(MorphThreshold::OFF);
    [RasterConfig::default(), by_time]
}

#[test]
fn a_tile_culled_by_its_cone_has_no_triangle_facing_the_eye_mid_morph() {
    GpuTest::run(|gpu| {
        let (mut tiles, mut triangles) = (0, 0);
        for config in configs() {
            let culled = culled_mid_morph(gpu, config, &[]);
            println!("{culled:?}");
            assert_eq!(culled.facing, 0, "a culled tile's triangle faces the eye");
            tiles += culled.tiles;
            triangles += culled.triangles;
        }
        assert!(tiles > 0 && triangles > 10_000, "culling was tested");
        // No bite here: this sphere's tiles are so curved that their cones are wide, and no
        // morph turns a triangle past them. `cull/tests/morph.rs` bites on bumpy tiles: there the
        // drawn cone, culling mid-morph, loses slivers that face an eye it says they face away
        // from; and the test below shows emit culling tiles mid-morph without its settled test.
    });
}

#[test]
fn the_level_choice_cone_culls_only_tiles_whose_morph_is_settled() {
    GpuTest::run(|gpu| {
        let (mut tiles, mut exempt, mut bitten) = (0, 0, 0);
        for config in configs() {
            let culled = culled_mid_morph(gpu, config, &[]);
            println!("{culled:?}");
            assert_eq!(culled.unsettled, 0, "a culled tile was morphing");
            tiles += culled.tiles;
            exempt += culled.exempt;
            // The bite: without the settled test, tiles mid-morph are culled (by distance; by
            // time alone, this scene's tiles still morphing are those refined, facing the eye).
            let without = culled_mid_morph(gpu, config, &[CULL_WHILE_MORPHING]);
            println!("without the settled test: {without:?}");
            bitten += without.unsettled;
        }
        assert!(
            tiles > 0 && exempt > 0,
            "both culled and exempt tiles were seen"
        );
        assert!(bitten > 0, "the test bites");
    });
}
