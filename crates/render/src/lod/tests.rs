//! Tests of the level choice on lavapipe, over synthetic tile trees.
//!
//! The tiles lie on a sphere, their bounds computed on the CPU and written straight into the
//! bounds buffer, their `e` a chosen fraction of their quad edge.

use std::collections::{BTreeMap, BTreeSet};

use geom::{BaseMesh, Level, cube_sphere};
use wgpu::{
    BufferDescriptor, BufferUsages, CommandEncoderDescriptor, PollType, TextureViewDescriptor,
};

use super::*;
use crate::camera::{Vec3, View};
use crate::cull::reference::{
    CpuPyramid, backfacing, bounds_of, depth_texture, in_frustum, occluded, projected,
};
use crate::cull::tests::Rng;
use crate::cull::{GpuBounds, PyramidPass};
use crate::gpu::GpuTest;
use crate::residency::{REQUEST_HEADER, SizeClass, Slot};
use crate::surface::Size;
use crate::tree::{NODE_BYTES, NODE_FINEST, TileTree};
use crate::upload::GpuVertex;

/// Level `l`.
fn level(l: u8) -> Level {
    Level::new(l).expect("a level")
}

/// The test viewport: 640 × 480.
fn size() -> Size {
    Size::new(640, 480).expect("non-zero")
}

/// The view from `eye` towards `target`, y up (or z, looking along y), 45°, near 0.01.
fn look(eye: Vec3, target: Vec3) -> View {
    View::look_at(
        eye,
        target,
        Vec3::new(0.0, 1.0, 0.0),
        45_f64.to_radians(),
        size(),
        0.01,
    )
    .or_else(|| {
        View::look_at(
            eye,
            target,
            Vec3::new(0.0, 0.0, 1.0),
            45_f64.to_radians(),
            size(),
            0.01,
        )
    })
    .expect("a view")
}

/// Every place of `mesh` at levels `0..=finest`.
fn all_places(mesh: &BaseMesh, finest: Level) -> Vec<Place> {
    let places = (0..=finest.get()).flat_map(|l| {
        let level = level(l);
        let per_side = u16::try_from(level.tiles_per_side()).expect("small");
        mesh.faces().flat_map(move |face| {
            (0..per_side).flat_map(move |ty| {
                (0..per_side).map(move |tx| {
                    Place::of(&TileKey::display(face, level, tx, ty).expect("a key"))
                })
            })
        })
    });
    places.collect()
}

/// `place`'s vertices: its face's corners bilinearly, pushed onto the unit sphere.
fn vertices(mesh: &BaseMesh, place: Place, finest: Level) -> (Vec<GpuVertex>, usize) {
    let key = place.key(finest).expect("a tile");
    let quads = place.level().tile_quads();
    let small = |n: u32| f32::from(u16::try_from(n).expect("a small grid"));
    let face_quads = small(place.level().face_quads());
    let corner = |c: u8| {
        let p = mesh.corner_position(place.face(), c).expect("a corner");
        [p.x, p.y, p.z]
    };
    let [c0, c1, c2, c3] = [corner(0), corner(1), corner(2), corner(3)];
    let side = usize::try_from(quads + 1).expect("small");
    let mut out = Vec::with_capacity(side * side);
    for j in 0..=quads {
        for i in 0..=quads {
            let u = small(u32::from(key.tx()) * quads + i) / face_quads;
            let v = small(u32::from(key.ty()) * quads + j) / face_quads;
            let p: [f32; 3] = [0, 1, 2].map(|a| {
                (1.0 - u) * (1.0 - v) * c0[a]
                    + u * (1.0 - v) * c1[a]
                    + u * v * c2[a]
                    + (1.0 - u) * v * c3[a]
            });
            let length = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            out.push(GpuVertex {
                position: p.map(|x| x / length),
                roughness: 1.0,
                normal: p.map(|x| x / length),
                reserved: 0,
            });
        }
    }
    (out, side)
}

/// A synthetic tile tree: the resident tiles of a sphere, their bounds and their `e`.
struct Scene {
    /// The base mesh.
    mesh: BaseMesh,
    /// The finest level.
    finest: Level,
    /// The tree.
    tree: TileTree,
    /// Every resident place's bounds.
    bounds: BTreeMap<Place, GpuBounds>,
    /// Every resident place's `e`.
    errors: BTreeMap<Place, f32>,
}

impl Scene {
    /// The tiles of `mesh` up to `finest`, those `keep` keeps resident, each with `e` =
    /// `error(place)` times its longest quad edge (0 at the finest level).
    fn new(
        gpu: &Gpu,
        mesh: BaseMesh,
        finest: Level,
        mut keep: impl FnMut(Place) -> bool,
        mut error: impl FnMut(Place) -> f32,
    ) -> Self {
        let kept: Vec<Place> = all_places(&mesh, finest)
            .into_iter()
            .filter(|place| keep(*place))
            .collect();
        let capacity = u32::try_from(kept.len()).expect("small");
        let mut tree = TileTree::new(gpu, &mesh, capacity).expect("a tree");
        let mut bounds = BTreeMap::new();
        let mut errors = BTreeMap::new();
        for place in kept {
            let (data, side) = vertices(&mesh, place, finest);
            let b = bounds_of(&data, side);
            let finest_tile = place.level() == finest;
            let e = if finest_tile {
                0.0
            } else {
                error(place) * b.quad_edge
            };
            bounds.insert(place, b);
            errors.insert(place, e);
            let flags = if finest_tile { NODE_FINEST } else { 0 };
            tree.insert(
                place,
                Slot::for_tests(SizeClass::of(place.level()), 0),
                e,
                flags,
            )
            .expect("inserted");
        }
        Self {
            mesh,
            finest,
            tree,
            bounds,
            errors,
        }
    }

    /// Writes the tree's changed nodes and every resident tile's bounds to the GPU.
    fn upload(&mut self, gpu: &Gpu, lod: &Lod) {
        for (start, nodes) in self.tree.take_dirty() {
            gpu.queue().write_buffer(
                self.tree.buffer(),
                u64::from(start) * NODE_BYTES,
                bytemuck::cast_slice(&nodes),
            );
        }
        for (place, b) in &self.bounds {
            if let Some(node) = self.tree.node(*place) {
                lod.bounds().write(gpu, node, &[*b]);
            }
        }
    }

    /// Whether `place` is resident.
    fn resident(&self, place: Place) -> bool {
        self.tree.node(place).is_some()
    }
}

/// A request buffer like residency's, holding `capacity` requests.
fn request_buffer(gpu: &Gpu, capacity: u64) -> Buffer {
    gpu.device().create_buffer(&BufferDescriptor {
        label: Some("test requests"),
        size: REQUEST_HEADER + capacity * 16,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// One frame's level choice, read back: `previous` for the first phase's occlusion test,
/// `post` the second phase's pyramid if it runs.
fn run(
    gpu: &Gpu,
    lod: &mut Lod,
    scene: &mut Scene,
    requests: &Buffer,
    view: &View,
    previous: Option<(&View, &DepthPyramid)>,
    post: Option<&DepthPyramid>,
) -> LodOutput {
    scene.upload(gpu, lod);
    gpu.queue().write_buffer(requests, 0, &[0; 16]);
    let mut encoder = gpu
        .device()
        .create_command_encoder(&CommandEncoderDescriptor::default());
    let inputs = MainInputs {
        tree: scene.tree.buffer(),
        requests,
        view,
        previous,
        finest: scene.finest,
    };
    lod.encode_main(gpu, &mut encoder, &inputs)
        .expect("encoded");
    if let Some(pyramid) = post {
        lod.encode_post(gpu, &mut encoder, pyramid)
            .expect("encoded");
    }
    let readback = lod
        .encode_readback(gpu, &mut encoder)
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

/// The tiles requested: (place, screen error).
fn read_requests(gpu: &Gpu, requests: &Buffer) -> Vec<(Place, f32)> {
    let words: Vec<u32> = gpu.read_buffer(requests).expect("read back");
    let count = usize::try_from(words[0]).expect("small");
    let entries = words[4..].chunks(4).take(count);
    entries
        .map(|e| {
            let place = Place::from_packed(e[0], e[1]).expect("a place");
            (place, f32::from_bits(e[2]))
        })
        .collect()
}

/// The places of a cut.
fn places(output: &LodOutput) -> BTreeSet<Place> {
    output.nodes.iter().map(|(_, place)| *place).collect()
}

/// The children of `place` along its edge `edge`.
fn edge_children(place: Place, edge: usize) -> Vec<Place> {
    let children = place.children();
    let picked: &[usize] = match (children[1].is_some(), edge) {
        (false, _) => &[0],
        (true, 0) => &[0, 1],
        (true, 1) => &[1, 3],
        (true, 2) => &[2, 3],
        (true, _) => &[0, 2],
    };
    picked.iter().filter_map(|&k| children[k]).collect()
}

/// The cut covers every face exactly once, with resident tiles.
fn check_partition(scene: &Scene, output: &LodOutput) {
    let cut = places(output);
    assert_eq!(cut.len(), output.nodes.len(), "no tile twice");
    let finest_per_side = u64::from(scene.finest.tiles_per_side());
    let mut area: BTreeMap<u32, u64> = BTreeMap::new();
    for &place in &cut {
        assert!(scene.resident(place), "{place:?} is resident");
        let mut above = place.parent();
        while let Some(p) = above {
            assert!(!cut.contains(&p), "{place:?} overlaps its ancestor {p:?}");
            above = p.parent();
        }
        let scale = finest_per_side / u64::from(place.level().tiles_per_side());
        *area.entry(place.face().index()).or_default() += scale * scale;
    }
    for face in scene.mesh.faces() {
        assert_eq!(
            area.get(&face.index()).copied().unwrap_or(0),
            finest_per_side * finest_per_side,
            "face {face:?} is covered exactly once"
        );
    }
}

/// Neighbours in the cut are at most one level apart across every edge.
fn check_neighbours(scene: &Scene, output: &LodOutput) {
    let cut = places(output);
    let topology = scene.mesh.topology();
    for &c in &cut {
        for (edge, m) in c.neighbours(topology).into_iter().enumerate() {
            let Some(m) = m else { continue };
            let mut covering = Some(m);
            while let Some(p) = covering {
                if cut.contains(&p) {
                    break;
                }
                covering = p.parent();
            }
            if let Some(p) = covering {
                assert!(
                    c.level().get() - p.level().get() <= 1,
                    "{c:?} edge {edge}: {p:?} is two levels coarser"
                );
                continue;
            }
            let back = m.neighbours(topology);
            let edges = back.iter().enumerate().filter(|(_, n)| **n == Some(c));
            for (k, _) in edges {
                for child in edge_children(m, k) {
                    assert!(
                        cut.contains(&child),
                        "{c:?} edge {edge}: {child:?}'s area is two levels finer"
                    );
                }
            }
        }
    }
}

/// Every cluster of the first list belongs to a drawn tile, every drawn tile has all its
/// clusters, and each cluster's edge variant marks exactly its coarser neighbours.
fn check_clusters(gpu: &Gpu, lod: &Lod, scene: &Scene, output: &LodOutput) {
    let (args, offset) = lod.draw_args(false);
    let words: Vec<u32> = gpu.read_buffer(args).expect("read back");
    let at = usize::try_from(offset / 4).expect("small");
    assert_eq!(words[at], CLUSTER_INDICES, "indices per cluster");
    let count = usize::try_from(words[at + 1]).expect("small");
    let clusters: Vec<GpuCluster> = gpu.read_buffer(lod.clusters(false)).expect("read back");
    let cut = places(output);
    let place_of: BTreeMap<u32, Place> = output.nodes.iter().copied().collect();
    let topology = scene.mesh.topology();
    let mut per_tile: BTreeMap<Place, usize> = BTreeMap::new();
    for cluster in &clusters[..count] {
        let place = place_of[&cluster.node];
        *per_tile.entry(place).or_default() += 1;
        let quads = place.level().tile_quads();
        let per = (quads / 8).max(1);
        assert_eq!(cluster.quads(), quads.min(8), "{place:?}: quads a cluster");
        let (cx, cy) = cluster.position();
        let on_edge = [cy == 0, cx == per - 1, cy == per - 1, cx == 0];
        let neighbours = place.neighbours(topology);
        for edge in 0..4 {
            let coarser = neighbours[edge].is_some_and(|m| {
                let mut above = m.parent();
                while let Some(p) = above {
                    if cut.contains(&p) {
                        return true;
                    }
                    above = p.parent();
                }
                false
            });
            assert_eq!(
                cluster.variant() & (1 << edge) != 0,
                on_edge[edge] && coarser,
                "{place:?} cluster ({cx}, {cy}) edge {edge}"
            );
        }
    }
    for (place, listed) in &per_tile {
        let per = usize::try_from((place.level().tile_quads() / 8).max(1)).expect("small");
        assert_eq!(*listed, per * per, "{place:?}: every cluster");
    }
    let drawn_first = output.stats.drawn.0;
    assert_eq!(
        per_tile.len(),
        usize::try_from(drawn_first).expect("small"),
        "drawn tiles"
    );
}

/// The capped screen error of `place` on the CPU, if it is decisively in view: its own and every
/// ancestor's projected `e`, the smallest; `None` near a culling boundary.
fn cpu_error(scene: &Scene, view: &View, place: Place) -> Option<f32> {
    let gpu = view.gpu();
    let eye = [gpu.eye[0], gpu.eye[1], gpu.eye[2]];
    let mut error = f32::MAX;
    let mut at = Some(place);
    while let Some(p) = at {
        let b = scene.bounds[&p];
        let e = scene.errors[&p];
        let wide = in_frustum(&gpu, b.center, (b.radius + e) * 1.001)
            && !backfacing(&b, eye, e * 1.001 + b.radius * 0.001);
        let narrow = in_frustum(&gpu, b.center, (b.radius + e) * 0.999)
            && !backfacing(&b, eye, e * 0.999 - b.radius * 0.001);
        if wide != narrow {
            return None;
        }
        error = error.min(if wide { projected(&gpu, &b, e, e) } else { 0.0 });
        at = p.parent();
    }
    Some(error)
}

#[test]
fn with_everything_resident_no_tile_in_view_is_left_over_a_pixel() {
    GpuTest::run(|gpu| {
        let mesh = cube_sphere(1).expect("a mesh");
        let mut scene = Scene::new(gpu, mesh, level(8), |_| true, |_| 0.3);
        let config = LodConfig::default()
            .with_relax_passes(MAX_RELAX_PASSES)
            .expect("valid");
        let mut lod = Lod::new(gpu, scene.tree.capacity(), config).expect("the level choice");
        let requests = request_buffer(gpu, 1024);
        let mut rng = Rng(11);
        for _ in 0..12 {
            let direction = Vec3::new(
                f64::from(rng.range(-1.0, 1.0)),
                f64::from(rng.range(-1.0, 1.0)),
                f64::from(rng.range(-1.0, 1.0)),
            )
            .normalized()
            .expect("a direction");
            let eye = direction * f64::from(rng.range(1.05, 6.0));
            let view = look(eye, Vec3::default());
            let output = run(gpu, &mut lod, &mut scene, &requests, &view, None, None);
            check_partition(&scene, &output);
            check_neighbours(&scene, &output);
            check_clusters(gpu, &lod, &scene, &output);
            assert_eq!(output.stats.over_pixel, 0, "{:?}", output.stats);
            assert_eq!(output.stats.shortfall, 0, "settled: {:?}", output.stats);
            assert!(
                output.drawn.is_subset(&output.cut),
                "drawn tiles are in the cut"
            );
            assert!(!output.drawn.is_empty(), "something is in view");
            assert!(
                read_requests(gpu, &requests).is_empty(),
                "nothing is missing"
            );
        }
    });
}

#[test]
fn hysteresis_keeps_the_cut_still_for_a_camera_moving_back_and_forth_across_a_threshold() {
    GpuTest::run(|gpu| {
        let mesh = cube_sphere(1).expect("a mesh");
        let mut scene = Scene::new(gpu, mesh, level(8), |_| true, |_| 0.3);
        let requests = request_buffer(gpu, 1024);
        let direction = Vec3::new(0.3, 0.5, 0.8).normalized().expect("a direction");
        let cuts = |tiles: &mut Scene, thresholds: Thresholds| {
            let config = LodConfig::default().with_thresholds(thresholds);
            let mut lod = Lod::new(gpu, tiles.tree.capacity(), config).expect("the level choice");
            let frames = (0..10).map(|frame| {
                // Back and forth by 10 %: inside the factor of 2 between refining and coarsening.
                let distance = if frame % 2 == 0 { 6.0 } else { 6.6 };
                let view = look(direction * distance, Vec3::default());
                places(&run(gpu, &mut lod, tiles, &requests, &view, None, None))
            });
            frames.collect::<Vec<BTreeSet<Place>>>()
        };
        let with = cuts(&mut scene, Thresholds::SPEC);
        for (frame, cut) in with.iter().enumerate().skip(2) {
            assert_eq!(cut, &with[1], "frame {frame} is frame 1's cut");
        }
        // The control: without the band, the same motion flips tiles every frame.
        let without = Thresholds {
            coarsen_error: Thresholds::SPEC.refine_error,
            coarsen_quad: Thresholds::SPEC.refine_quad,
            ..Thresholds::SPEC
        };
        let flipping = cuts(&mut scene, without);
        assert_ne!(flipping[2], flipping[3], "the motion crosses a threshold");
        assert_eq!(flipping[2], flipping[4], "and comes back");
    });
}

#[test]
fn the_cut_is_complete_with_no_hole_or_overlap_for_random_residency() {
    GpuTest::run(|gpu| {
        let mut rng = Rng(23);
        for round in 0..6 {
            let mesh = cube_sphere(1).expect("a mesh");
            // The floor always, then each finer tile with a chance that falls with its level.
            let mut keep = |place: Place| place.level().get() <= 5 || rng.below(4) != 0;
            let mut scene = Scene::new(gpu, mesh, level(8), &mut keep, |_| 0.3);
            let mut lod = Lod::new(gpu, scene.tree.capacity(), LodConfig::default())
                .expect("the level choice");
            let requests = request_buffer(gpu, 4096);
            for frame in 0..4 {
                let direction = Vec3::new(
                    f64::from(rng.range(-1.0, 1.0)),
                    f64::from(rng.range(-1.0, 1.0)),
                    f64::from(rng.range(-1.0, 1.0)),
                )
                .normalized()
                .expect("a direction");
                let view = look(direction * f64::from(rng.range(1.05, 4.0)), Vec3::default());
                let output = run(gpu, &mut lod, &mut scene, &requests, &view, None, None);
                check_partition(&scene, &output);
                check_neighbours(&scene, &output);
                check_clusters(gpu, &lod, &scene, &output);
                let cut = places(&output);
                let requested = read_requests(gpu, &requests);
                let requested_places: BTreeSet<Place> =
                    requested.iter().map(|(place, _)| *place).collect();
                for (place, error) in &requested {
                    assert!(
                        !scene.resident(*place),
                        "round {round} frame {frame}: {place:?}"
                    );
                    let parent = place.parent().expect("a parent");
                    assert!(cut.contains(&parent), "requested by a tile of the cut");
                    assert!(error.is_finite() && *error >= 0.0, "a screen error");
                }
                // A tile of the cut over a pixel, its children missing, asks for them.
                for &place in &cut {
                    let missing: Vec<Place> = place
                        .children()
                        .into_iter()
                        .flatten()
                        .filter(|child| child.level() <= scene.finest && !scene.resident(*child))
                        .collect();
                    if missing.is_empty() {
                        continue;
                    }
                    if cpu_error(&scene, &view, place).is_some_and(|error| error > 1.1) {
                        for child in missing {
                            assert!(
                                requested_places.contains(&child),
                                "round {round} frame {frame}: {child:?} is requested"
                            );
                        }
                    }
                }
                assert!(
                    output.stats.waiting <= output.stats.over_pixel,
                    "counted among those over a pixel"
                );
            }
        }
    });
}

#[test]
fn a_spike_of_detail_is_relaxed_into_neighbours_a_level_apart_or_drawn_coarser() {
    GpuTest::run(|gpu| {
        let mesh = cube_sphere(1).expect("a mesh");
        // Face 0's tiles carry a hundred times the error of the rest: it alone wants the finest
        // level, two or more levels past its neighbours.
        let spike = |place: Place| if place.face().index() == 0 { 30.0 } else { 0.3 };
        let mut scene = Scene::new(gpu, mesh, level(8), |_| true, spike);
        let requests = request_buffer(gpu, 1024);
        let view = look(Vec3::new(4.0, 0.5, 0.3), Vec3::default());
        let relaxed = |tiles: &mut Scene, passes: u32| {
            let config = LodConfig::default()
                .with_relax_passes(passes)
                .expect("valid");
            let mut lod = Lod::new(gpu, tiles.tree.capacity(), config).expect("the level choice");
            run(gpu, &mut lod, tiles, &requests, &view, None, None)
        };
        let settled = relaxed(&mut scene, MAX_RELAX_PASSES);
        check_partition(&scene, &settled);
        check_neighbours(&scene, &settled);
        let used = settled.stats.passes_used();
        assert!(used >= 2, "the spike needs passes: {:?}", settled.stats);
        assert!(
            used < usize::try_from(MAX_RELAX_PASSES).expect("small"),
            "and settles"
        );
        assert_eq!(settled.stats.shortfall, 0, "{:?}", settled.stats);
        // Too few passes: the sweep still keeps neighbours a level apart, drawing the area
        // coarser, and counts the shortfall.
        for passes in [0, 1] {
            let short = relaxed(&mut scene, passes);
            check_partition(&scene, &short);
            check_neighbours(&scene, &short);
            assert!(
                short.stats.shortfall > 0,
                "{passes} passes: {:?}",
                short.stats
            );
            assert!(
                short.stats.cut_triangles < settled.stats.cut_triangles,
                "coarser"
            );
        }
    });
}

#[test]
fn the_triangle_budget_caps_the_cut_and_refines_the_largest_errors_first() {
    GpuTest::run(|gpu| {
        let mesh = cube_sphere(1).expect("a mesh");
        let mut scene = Scene::new(gpu, mesh, level(8), |_| true, |_| 0.3);
        let requests = request_buffer(gpu, 1024);
        let view = look(Vec3::new(0.2, 0.3, 1.3), Vec3::default());
        let unlimited = {
            let mut lod = Lod::new(gpu, scene.tree.capacity(), LodConfig::default())
                .expect("the level choice");
            run(gpu, &mut lod, &mut scene, &requests, &view, None, None)
        };
        let budget = unlimited.stats.cut_triangles / 3;
        let config = LodConfig::default()
            .with_triangle_budget(budget)
            .with_relax_passes(MAX_RELAX_PASSES)
            .expect("valid");
        let mut lod = Lod::new(gpu, scene.tree.capacity(), config).expect("the level choice");
        let capped = run(gpu, &mut lod, &mut scene, &requests, &view, None, None);
        check_partition(&scene, &capped);
        check_neighbours(&scene, &capped);
        assert!(capped.stats.cut_triangles <= budget, "{:?}", capped.stats);
        assert!(
            capped.stats.refined_triangles <= budget,
            "{:?}",
            capped.stats
        );
        assert!(capped.stats.over_pixel > 0, "the cap bites");
        assert!(capped.stats.budget_limited > 0, "and is counted");
        // Largest errors first: no tile of the cut, its children resident, is much further over
        // than the least of the refined tiles (one histogram bucket, an eighth of an octave),
        // but for those the sweep or the budget took back.
        let cut = places(&capped);
        let mut refined = Vec::new();
        for place in scene.bounds.keys() {
            let inside = cut.iter().any(|c| {
                c.level() > place.level() && {
                    let mut above = c.parent();
                    while let Some(p) = above {
                        if p == *place {
                            break;
                        }
                        above = p.parent();
                    }
                    above == Some(*place)
                }
            });
            if inside && let Some(error) = cpu_error(&scene, &view, *place) {
                refined.push(error);
            }
        }
        let least = refined
            .iter()
            .copied()
            .filter(|e| *e > 1.0)
            .fold(f32::MAX, f32::min);
        let exceptions = capped.stats.shortfall + capped.stats.refused;
        let worse: Vec<f32> = cut
            .iter()
            .filter(|place| {
                place
                    .children()
                    .into_iter()
                    .flatten()
                    .all(|c| c.level() <= scene.finest)
            })
            .filter_map(|place| cpu_error(&scene, &view, *place))
            .filter(|error| *error > least * 1.2)
            .collect();
        assert!(
            worse.len() <= usize::try_from(exceptions).expect("small"),
            "{} tiles of the cut over {least}: {worse:?}, {:?}",
            worse.len(),
            capped.stats
        );
    });
}

/// Random bounds: a sphere somewhere in a 6-unit cube, a cone of any axis and width.
fn random_bounds(rng: &mut Rng) -> GpuBounds {
    let axis = [
        rng.range(-1.0, 1.0),
        rng.range(-1.0, 1.0),
        rng.range(-1.0, 1.0),
    ];
    let length = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    GpuBounds {
        center: [
            rng.range(-3.0, 3.0),
            rng.range(-3.0, 3.0),
            rng.range(-3.0, 3.0),
        ],
        radius: rng.range(0.01, 0.4),
        axis: axis.map(|a| a / length),
        cone_cos: rng.range(-0.3, 0.99),
        quad_edge: 0.01,
        reserved: [0; 3],
    }
}

/// The depth pyramid of a depth image of the test viewport's size, row by row.
fn pyramid_of(gpu: &Gpu, depth: &[f32]) -> DepthPyramid {
    let (width, height) = (size().width(), size().height());
    let texture = depth_texture(gpu, width, height, depth);
    let view = texture.create_view(&TextureViewDescriptor::default());
    let pyramid = DepthPyramid::new(gpu, size()).expect("a pyramid");
    let pass = PyramidPass::new(gpu).expect("the pyramid's passes");
    let mut encoder = gpu
        .device()
        .create_command_encoder(&CommandEncoderDescriptor::default());
    pass.build(gpu, &mut encoder, &view, &pyramid, None)
        .expect("built");
    gpu.queue().submit([encoder.finish()]);
    pyramid
}

#[test]
fn culling_agrees_with_the_cpu_reference_in_both_phases() {
    GpuTest::run(|gpu| {
        // Level 0 only, of 2,400 faces: every face's tile is in the cut, with random bounds.
        let mesh = cube_sphere(20).expect("a mesh");
        let mut scene = Scene::new(gpu, mesh, level(9), |p| p.level().get() == 0, |_| 0.3);
        let mut rng = Rng(5);
        let mut lod =
            Lod::new(gpu, scene.tree.capacity(), LodConfig::default()).expect("the level choice");
        let requests = request_buffer(gpu, 1 << 14);
        let (width, height) = (size().width(), size().height());
        // A near wall over the left half of the image, nothing behind the right.
        let near_wall = |i: u32| {
            if i % width < width / 2 {
                0.01 / 2.0
            } else {
                0.0
            }
        };
        let wall: Vec<f32> = (0..width * height).map(near_wall).collect();
        let far: Vec<f32> = vec![0.0; usize::try_from(width * height).expect("small")];
        let wall_pyramid = pyramid_of(gpu, &wall);
        let far_pyramid = pyramid_of(gpu, &far);
        let cpu_wall = CpuPyramid::new(width, height, &wall);
        let (mut decisive, mut culled, mut hidden) = (0, 0, 0);
        for round in 0..8 {
            let roots: Vec<Place> = scene
                .bounds
                .keys()
                .copied()
                .filter(|p| scene.resident(*p))
                .collect();
            for &place in &roots {
                scene.bounds.insert(place, random_bounds(&mut rng));
            }
            let eye = Vec3::new(
                f64::from(rng.range(-1.0, 1.0)),
                f64::from(rng.range(-1.0, 1.0)),
                5.0,
            );
            let view = look(eye, Vec3::default());
            let gview = view.gpu();
            let e3 = [gview.eye[0], gview.eye[1], gview.eye[2]];
            // Frustum and cone, then the wall in the first phase; the second phase against the
            // wall again (nothing more shows) and against nothing (everything hidden shows).
            for (post, expect_hidden) in [(&wall_pyramid, true), (&far_pyramid, false)] {
                let output = run(
                    gpu,
                    &mut lod,
                    &mut scene,
                    &requests,
                    &view,
                    Some((&view, &wall_pyramid)),
                    Some(post),
                );
                let drawn: BTreeSet<Place> = output.drawn.iter().map(Place::of).collect();
                for &place in &roots {
                    let b = scene.bounds[&place];
                    let shown = |grow: f32| {
                        let r = b.radius * (1.0 + grow);
                        let wider = GpuBounds {
                            radius: r,
                            cone_cos: b.cone_cos - grow,
                            ..b
                        };
                        let visible =
                            in_frustum(&gview, b.center, r) && !backfacing(&wider, e3, 0.0);
                        let hides = expect_hidden && occluded(&gview, &cpu_wall, b.center, r);
                        (visible && !hides, visible && hides)
                    };
                    let (wide, narrow) = (shown(1e-3), shown(-1e-3));
                    if wide != narrow {
                        continue;
                    }
                    decisive += 1;
                    let (expected, behind) = wide;
                    culled += u32::from(!expected);
                    hidden += u32::from(behind);
                    assert_eq!(
                        drawn.contains(&place),
                        expected,
                        "round {round}, post hidden {expect_hidden}: {place:?} {b:?}"
                    );
                }
                assert_eq!(output.stats.overflow, (0, 0, 0), "nothing overflowed");
            }
        }
        // The test means something: most tiles decided, many of them culled, some by the wall.
        assert!(decisive > 30_000, "decisive {decisive}");
        assert!(
            culled > 5_000 && hidden > 500,
            "culled {culled}, hidden {hidden}"
        );
    });
}
