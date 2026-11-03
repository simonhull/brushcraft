//! The benchmark orbit, measured.
//!
//! How many relaxation passes the level choice needs (the spec's open question 1), and what
//! its passes cost on the GPU, at the benchmark project's tree shape (`cube_sphere(20)`,
//! finest level 9: 2,400 faces, 218,400 tiles).
//!
//! The real project can't be driven through residency here yet (the `app` wiring is another
//! change), so the tiles are synthetic: the sphere's own patches, their bounds from a 9 × 9
//! sample of each, their `e` a quarter of their quad edge (where the error and quad rules meet)
//! times a per-tile factor (1 for the smooth case, 0.25 to 2 at random for the rough one).
//!
//! With every tile resident it also runs with a budget that binds (4M triangles,
//! [`BINDING_BUDGET`]), for what the passes that run then cost.
//!
//! Ignored by default: run on the machine's GPU, in release, with
//! `cargo nextest run --release -p render --run-ignored only orbit --no-capture`
//! (`BRUSHCRAFT_ADAPTER=<name>` picks an adapter by name).

use std::time::Duration;

use geom::{Level, MAX_LEVEL, cube_sphere};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

use super::*;
use crate::cull::{BoundsUpdate, SlotBuffers};
use crate::gpu::AdapterRequest;
use crate::residency::CLASSES;

/// The binding budget measured beside the default one: 4M triangles, under what the benchmark
/// orbit draws close.
const BINDING_BUDGET: u32 = 4 << 20;

/// Samples a side of a tile's bounds are taken from.
const SAMPLES: u32 = 9;

/// The bounds of `place` on the unit sphere and the cone of the finest normals under it.
///
/// The bounds come from a `SAMPLES` × `SAMPLES` grid of its vertices, grown by the sphere's
/// bulge between samples; the cone covers the grid's normals, which holds them all, as
/// [`sphere_cone`]'s does.
fn sampled_bounds(mesh: &BaseMesh, place: Place, finest: Level) -> (GpuBounds, FinestNormalCone) {
    let key = place.key(finest).expect("a tile");
    let quads = place.level().tile_quads();
    let steps = SAMPLES.min(quads + 1) - 1;
    let face_quads = f64::from(place.level().face_quads());
    let corner = |c: u8| {
        let p = mesh.corner_position(place.face(), c).expect("a corner");
        [f64::from(p.x), f64::from(p.y), f64::from(p.z)]
    };
    let [c0, c1, c2, c3] = [corner(0), corner(1), corner(2), corner(3)];
    let at = |i: u32, j: u32| {
        let u = (f64::from(u32::from(key.tx()) * quads) + f64::from(i * quads) / f64::from(steps))
            / face_quads;
        let v = (f64::from(u32::from(key.ty()) * quads) + f64::from(j * quads) / f64::from(steps))
            / face_quads;
        let p: [f64; 3] = [0, 1, 2].map(|a| {
            (1.0 - u) * (1.0 - v) * c0[a]
                + u * (1.0 - v) * c1[a]
                + u * v * c2[a]
                + (1.0 - u) * v * c3[a]
        });
        let length = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        p.map(|x| x / length)
    };
    let grid: Vec<[f64; 3]> = (0..=steps)
        .flat_map(|j| (0..=steps).map(move |i| (i, j)))
        .map(|(i, j)| at(i, j))
        .collect();
    let distance = |a: [f64; 3], b: [f64; 3]| {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    };
    let lo = grid
        .iter()
        .fold([f64::MAX; 3], |m, p| [0, 1, 2].map(|a| m[a].min(p[a])));
    let hi = grid
        .iter()
        .fold([f64::MIN; 3], |m, p| [0, 1, 2].map(|a| m[a].max(p[a])));
    let center = [0, 1, 2].map(|a| f64::midpoint(lo[a], hi[a]));
    let spacing = distance(grid[0], grid[1]).max(distance(
        grid[0],
        grid[usize::try_from(steps + 1).expect("small")],
    ));
    let radius = grid
        .iter()
        .map(|p| distance(*p, center))
        .fold(0.0, f64::max)
        + spacing * spacing / 8.0;
    let sum = grid
        .iter()
        .fold([0.0; 3], |s, p| [s[0] + p[0], s[1] + p[1], s[2] + p[2]]);
    let length = (sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]).sqrt();
    let axis = sum.map(|x| x / length);
    // On a sphere the normal is the position: the widest sample, widened by one sample's angle.
    let widest = grid
        .iter()
        .map(|p| p[0] * axis[0] + p[1] * axis[1] + p[2] * axis[2])
        .fold(1.0, f64::min);
    let cone = (widest.clamp(-1.0, 1.0).acos() + spacing).cos();
    let quad_edge = spacing * f64::from(steps) / f64::from(quads);
    let narrow = |x: f64| {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "bounds go to the GPU as f32"
        )]
        let narrowed = x as f32;
        narrowed
    };
    let normals: Vec<geom::Vec3> = grid
        .iter()
        .map(|p| geom::Vec3::new(narrow(p[0]), narrow(p[1]), narrow(p[2])))
        .collect();
    let finest_cone = FinestNormalCone::covering(&normals).expect("a tile has normals");
    let bounds = GpuBounds {
        center: center.map(narrow),
        radius: narrow(radius * (1.0 + 1e-4)),
        axis: axis.map(narrow),
        cone_cos: narrow(cone - 1e-4),
        quad_edge: narrow(quad_edge),
        reserved: [0; 3],
    };
    (bounds, finest_cone)
}

/// The benchmark tree shape with every tile up to `top` resident, `rough` choosing the
/// per-tile error factor.
fn benchmark_scene(gpu: &Gpu, shape: Shape, rough: bool) -> Scene {
    let mesh = cube_sphere(shape.splits).expect("the mesh");
    let (finest, top) = (shape.finest, shape.top);
    let places: Vec<Place> = all_places(&mesh, finest)
        .into_iter()
        .filter(|place| place.level() <= top)
        .collect();
    let bounds: Vec<(GpuBounds, FinestNormalCone)> = places
        .iter()
        .map(|place| sampled_bounds(&mesh, *place, finest))
        .collect();
    let capacity = u32::try_from(places.len()).expect("small");
    let mut tree = TileTree::new(gpu, &mesh, capacity).expect("a tree");
    let mut rng = Rng(77);
    let mut scene_bounds = BTreeMap::new();
    let mut details = BTreeMap::new();
    for (place, (b, cone)) in places.into_iter().zip(bounds) {
        let finest_tile = place.level() == finest;
        let factor = if rough { rng.range(0.25, 2.0) } else { 1.0 };
        let detail = if finest_tile {
            TileDetail::FINEST
        } else {
            TileDetail {
                stored: None,
                error: 0.25 * factor * b.quad_edge,
                cone,
                held: false,
            }
        };
        let flags = if finest_tile { NODE_FINEST } else { 0 };
        tree.insert(
            place,
            Slot::for_tests(SizeClass::of(place.level()), 0),
            detail,
            flags,
        )
        .expect("inserted");
        scene_bounds.insert(place, b);
        details.insert(place, detail);
    }
    Scene {
        mesh,
        finest,
        tree,
        bounds: scene_bounds,
        details,
    }
}

/// A synthetic tree's shape: `cube_sphere(splits)` to `finest`, every tile up to `top`
/// resident.
#[derive(Debug, Clone, Copy)]
struct Shape {
    /// The sphere's splits.
    splits: u32,
    /// The finest level.
    finest: Level,
    /// The finest resident level.
    top: Level,
    /// What to call it.
    label: &'static str,
}

/// What a run of frames measured.
#[derive(Debug, Default)]
struct Measured {
    /// Per frame, the relaxation passes that raised anything.
    passes: Vec<usize>,
    /// The largest shortfall a frame had.
    shortfall: u32,
    /// The largest count of tiles over a pixel.
    over_pixel: u32,
    /// The largest cut, tiles and triangles.
    cut: (u32, u32),
    /// The most triangles the refinements were charged.
    charged: u32,
    /// The most spreading passes the budget's choice needed, and whether one didn't settle.
    need_passes: (u32, bool),
    /// Per frame, the stages' GPU times.
    timings: Vec<LodTimings>,
}

/// The `p`-th fraction's value of `values` (sorted here).
fn quantile(values: &mut [Duration], p: f64) -> Duration {
    values.sort_unstable();
    let last = values.len().saturating_sub(1);
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "an index into a short list"
    )]
    let at = ((last as f64) * p).round() as usize;
    values.get(at).copied().unwrap_or_default()
}

/// An orbit of `frames` frames at `distance` from the centre into a `width` × `height` view.
fn orbit(
    gpu: &Gpu,
    scene: &mut Scene,
    lod: &mut Lod,
    distance: f64,
    frames: u32,
    size: Size,
) -> Measured {
    let requests = request_buffer(gpu, 1 << 16);
    let mut measured = Measured::default();
    for frame in 0..frames {
        let angle = std::f64::consts::TAU * f64::from(frame) / f64::from(frames);
        let eye = Vec3::new(
            angle.sin() * distance,
            0.3 * distance,
            angle.cos() * distance,
        );
        let view = View::look_at(
            eye,
            Vec3::default(),
            Vec3::new(0.0, 1.0, 0.0),
            45_f64.to_radians(),
            size,
            distance * 1e-3,
        )
        .expect("a view");
        let output = run(gpu, lod, scene, &requests, &view, None, None);
        let stats = &output.stats;
        measured.passes.push(stats.passes_used());
        measured.shortfall = measured.shortfall.max(stats.shortfall);
        measured.over_pixel = measured.over_pixel.max(stats.over_pixel);
        measured.charged = measured.charged.max(stats.refined_triangles);
        measured.need_passes = (
            measured.need_passes.0.max(stats.need_passes),
            measured.need_passes.1 || stats.need_unsettled,
        );
        measured.cut = (
            measured.cut.0.max(stats.cut),
            measured.cut.1.max(stats.cut_triangles),
        );
        if let Some(timings) = lod.read_timings(gpu).expect("timings") {
            measured.timings.push(timings);
        }
    }
    measured
}

/// One line of the stages' p50 and max, milliseconds.
fn stages(timings: &[LodTimings]) -> String {
    let column = |pick: fn(&LodTimings) -> Duration| {
        let mut values: Vec<Duration> = timings.iter().map(pick).collect();
        let median = quantile(&mut values, 0.5);
        let max = quantile(&mut values, 1.0);
        format!(
            "{:.3}/{:.3}",
            median.as_secs_f64() * 1e3,
            max.as_secs_f64() * 1e3
        )
    };
    let total = |t: &LodTimings| t.choose + t.relax + t.sweep + t.emit;
    format!(
        "choose {} relax {} sweep {} emit {} total {} ms (p50/max)",
        column(|t| t.choose),
        column(|t| t.relax),
        column(|t| t.sweep),
        column(|t| t.emit),
        column(total),
    )
}

#[test]
#[ignore = "a measurement on the machine's GPU: run it by hand (see the module notes)"]
fn orbit_relaxation_passes_and_timings_at_the_benchmark_tree_shape() {
    let request =
        std::env::var("BRUSHCRAFT_ADAPTER").map_or(AdapterRequest::Default, AdapterRequest::Named);
    let gpu = Gpu::open(&request).expect("a GPU");
    let info = gpu.adapter_info();
    println!(
        "adapter: {} ({:?}, driver {} {})",
        info.name, info.backend, info.driver, info.driver_info
    );
    let sizes = [(3840, 2160), (1920, 1080)];
    let shapes = [
        Shape {
            splits: 20,
            finest: level(9),
            top: level(9),
            label: "benchmark, every tile (218,400 nodes)",
        },
        Shape {
            splits: 20,
            finest: level(9),
            top: level(8),
            label: "benchmark, display tiles (64,800 nodes)",
        },
        // As deep as levels go: 13 levels, so the relaxation's climb from the back faces is
        // longest, against the cap of MAX_RELAX_PASSES.
        Shape {
            splits: 1,
            finest: level(MAX_LEVEL),
            top: level(MAX_LEVEL),
            label: "deep, cube_sphere(1) to level 12 (32,802 nodes)",
        },
    ];
    for shape in shapes {
        for rough in [false, true] {
            let mut scene = benchmark_scene(&gpu, shape, rough);
            for (width, height) in sizes {
                let size = Size::new(width, height).expect("non-zero");
                for distance in [3.0, 1.5, 1.1, 1.02] {
                    // Every tile resident, also with a budget that binds: the cost of the
                    // passes that run then (spec, "When it binds").
                    let binding = shape.top == shape.finest && shape.splits == 20;
                    let cases = [(false, None), (true, None), (true, Some(BINDING_BUDGET))];
                    let cases = cases.into_iter().filter(|(_, b)| binding || b.is_none());
                    for (exempt, budget) in cases {
                        let config = LodConfig::default().with_offscreen_seams(exempt);
                        let config = budget.map_or(config, |b| config.with_triangle_budget(b));
                        let mut lod = Lod::new(&gpu, scene.tree.capacity(), config)
                            .expect("the level choice");
                        lod.enable_timing(&gpu).expect("timing");
                        let mut measured = orbit(&gpu, &mut scene, &mut lod, distance, 72, size);
                        let mut passes = measured.passes.clone();
                        passes.sort_unstable();
                        let max = passes.last().copied().unwrap_or(0);
                        let median = passes.get(passes.len() / 2).copied().unwrap_or(0);
                        // The first frame starts from nothing; the rest follow a moving camera.
                        let steady = measured.passes.iter().skip(1).copied().max().unwrap_or(0);
                        measured.timings.remove(0);
                        println!(
                            "{}, {}, {width}×{height}, distance {distance}, off-screen seams {}{}: passes p50 {median} max {max} (after frame 0: {steady}), shortfall max {}, over 1 px max {}, cut max {} tiles {} triangles, charged max {} triangles, spreading passes max {} (unsettled {}); {}",
                            shape.label,
                            if rough { "rough" } else { "smooth" },
                            if exempt { "exempt" } else { "held" },
                            if budget.is_some() { ", budget 4M" } else { "" },
                            measured.shortfall,
                            measured.over_pixel,
                            measured.cut.0,
                            measured.cut.1,
                            measured.charged,
                            measured.need_passes.0,
                            measured.need_passes.1,
                            stages(&measured.timings),
                        );
                    }
                }
            }
        }
    }
    bounds_timing(&gpu);
}

/// The bounds pass over 4,096 finest tiles in their own slots: its GPU time per tile.
fn bounds_timing(gpu: &Gpu) {
    let mesh = cube_sphere(20).expect("the benchmark mesh");
    let finest = level(9);
    let tiles = 4096;
    let places: Vec<Place> = all_places(&mesh, finest)
        .into_iter()
        .filter(|place| place.level() == finest)
        .take(tiles)
        .collect();
    let mut tree = TileTree::new(gpu, &mesh, u32::try_from(tiles).expect("small")).expect("a tree");
    let data: Vec<GpuVertex> = places
        .iter()
        .flat_map(|place| vertices(&mesh, *place, finest).0)
        .collect();
    for (index, place) in places.iter().enumerate() {
        tree.insert(
            *place,
            Slot::for_tests(SizeClass::of(finest), index),
            TileDetail::FINEST,
            NODE_FINEST,
        )
        .expect("inserted");
    }
    for (start, nodes) in tree.take_dirty() {
        gpu.queue().write_buffer(
            tree.buffer(),
            u64::from(start) * NODE_BYTES,
            bytemuck::cast_slice(&nodes),
        );
    }
    let slot_buffer = gpu.device().create_buffer_init(&BufferInitDescriptor {
        label: Some("bench slots"),
        contents: bytemuck::cast_slice(&data),
        usage: BufferUsages::STORAGE,
    });
    let mut classes: Vec<Vec<Buffer>> = vec![Vec::new(); CLASSES];
    classes[usize::from(SizeClass::of(finest).get())] = vec![slot_buffer];
    let slot_set = SlotBuffers::new(classes);
    let mut lod = Lod::new(gpu, tree.capacity(), LodConfig::default()).expect("the level choice");
    lod.enable_timing(gpu).expect("timing");
    let update = BoundsUpdate::since(&tree, 0);
    let mut elapsed = Vec::new();
    for _ in 0..20 {
        let mut encoder = gpu
            .device()
            .create_command_encoder(&CommandEncoderDescriptor::default());
        lod.encode_bounds(gpu, &mut encoder, tree.buffer(), &update, &slot_set)
            .expect("encoded");
        let readback = lod.encode_readback(gpu, &mut encoder, 0).expect("encoded");
        gpu.queue().submit([encoder.finish()]);
        drop(readback);
        if let Some(timings) = lod.read_timings(gpu).expect("timings") {
            elapsed.push(timings.bounds);
        }
    }
    let median = quantile(&mut elapsed, 0.5);
    println!(
        "bounds pass: {tiles} tiles of 65 × 65 in {:.3} ms (p50 of 20), {:.2} µs a tile",
        median.as_secs_f64() * 1e3,
        median.as_secs_f64() * 1e6 / f64::from(u32::try_from(tiles).expect("small")),
    );
}
