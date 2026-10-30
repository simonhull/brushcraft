//! Tests of the shading pass on lavapipe, over the raster's synthetic sphere
//! ([`crate::raster::tests::World`]).
//!
//! They check that the pass rebuilds the very surface the raster drew (the rebuilt point's depth
//! against the depth buffer, pixel by pixel, with edge variants, morphs by time and by distance,
//! and both of a quad's triangles in the scene), and that the check bites when the drawn edges,
//! the morph or the diagonal is ignored; that the fragment path shades as the compute path does;
//! and that a material or wireframe change is a shading pass alone. `golden` holds the image
//! tests.

use std::collections::BTreeSet;
use std::time::Duration;

use wgpu::{
    BufferDescriptor, BufferUsages, CommandEncoderDescriptor, ComputePipelineDescriptor,
    TexelCopyBufferInfo, TexelCopyBufferLayout,
};

use super::*;
use crate::cull::tests::Rng;
use crate::gpu::GpuTest;
use crate::lod::GpuCluster;
use crate::raster::tests::{
    World, draw_cut, finish, grown, level_choice, lod_frame, random_cut, random_view, refine,
};
use crate::raster::{RasterConfig, RasterFrame};

mod flip;
mod golden;
pub(crate) mod png;

/// The probe, appended to the pass's shader in tests only: for each pixel, the rebuilt point
/// relative to the eye and its depth (`w`, -1 where nothing is drawn), then its normal and
/// roughness term.
const PROBE_WGSL: &str = r"
// Three vec4s a pixel: (relative, depth), (normal, roughness), (behind, 0, 0, 0).
@group(3) @binding(1) var<storage, read_write> probed: array<vec4<f32>>;

// Rebuilds one pixel and writes what it found.
@compute @workgroup_size(8, 8)
fn probe(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(ids);
    if id.x >= size.x || id.y >= size.y {
        return;
    }
    let at = 3u * (id.y * size.x + id.x);
    let r = rebuild(id.xy);
    if !r.drawn {
        probed[at] = vec4<f32>(0.0, 0.0, 0.0, -1.0);
        probed[at + 1u] = vec4<f32>(0.0);
        probed[at + 2u] = vec4<f32>(0.0);
        return;
    }
    let clip = view.view_proj * vec4<f32>(r.relative, 1.0);
    probed[at] = vec4<f32>(r.relative, clip.z / clip.w);
    probed[at + 1u] = vec4<f32>(r.normal, r.roughness);
    probed[at + 2u] = vec4<f32>(select(0.0, 1.0, r.behind), 0.0, 0.0, 0.0);
}
";

/// One pixel as the probe rebuilt it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Probed {
    /// The point, relative to the eye.
    pub(crate) relative: [f32; 3],
    /// Its depth (reversed Z).
    pub(crate) depth: f32,
    /// The unit normal.
    pub(crate) normal: [f32; 3],
    /// The roughness term.
    pub(crate) roughness: f32,
    /// Whether a corner of its quad lies behind the eye.
    pub(crate) behind: bool,
}

/// The pass for `world`'s slots, compute, clay, no wireframe.
pub(crate) fn shade_for(gpu: &Gpu, world: &World) -> Shade {
    Shade::new(gpu, world.layout(), ShadePath::Compute, MatCapStyle::Clay)
        .expect("the shading pass")
}

/// The pass for `world`'s slots, compute, with `patches` in its shader.
fn patched(gpu: &Gpu, world: &World, patches: &[(&str, &str)]) -> Shade {
    let material = MatCap::new(gpu, MatCapStyle::Clay).expect("the matcap");
    Shade::patched(gpu, world.layout(), ShadePath::Compute, material, patches)
        .expect("the shading pass")
}

/// Shades the raster's last frame of `world` from `view` with `shade`, and waits for it.
pub(crate) fn shade_frame(
    gpu: &Gpu,
    shade: &mut Shade,
    world: &World,
    raster: &Raster,
    view: &View,
) {
    let mut encoder = gpu
        .device()
        .create_command_encoder(&CommandEncoderDescriptor::default());
    let scene = ShadeScene {
        tree: world.tree.buffer(),
        slots: &world.slots,
        raster,
        view,
    };
    shade.encode(gpu, &mut encoder, &scene).expect("encoded");
    gpu.queue().submit([encoder.finish()]);
    finish(gpu);
}

/// The value of half-float `bits`.
pub(crate) fn half(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = i32::from((bits >> 10) & 0x1F);
    let mantissa = f32::from(bits & 0x3FF);
    if exponent == 0 {
        return sign * mantissa * 2_f32.powi(-24);
    }
    if exponent == 0x1F {
        return if mantissa == 0.0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        };
    }
    sign * (1.0 + mantissa / 1024.0) * 2_f32.powi(exponent - 15)
}

/// The texels of `texture` (`bytes` a texel), row by row from the top, as `T`s.
fn read_texture<T: Pod>(gpu: &Gpu, texture: &Texture, bytes: u32) -> Vec<T> {
    let (width, height) = (texture.width(), texture.height());
    let row = (width * bytes).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let device = gpu.device();
    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some("texture readback"),
        size: u64::from(row) * u64::from(height),
        usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        TexelCopyBufferInfo {
            buffer: &buffer,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: Some(height),
            },
        },
        texture.size(),
    );
    gpu.queue().submit([encoder.finish()]);
    let padded: Vec<T> = gpu.read_buffer(&buffer).expect("read back");
    let stride = usize::try_from(row).expect("small") / size_of::<T>();
    let per_row = usize::try_from(width * bytes).expect("small") / size_of::<T>();
    padded
        .chunks(stride)
        .flat_map(|texels| texels.iter().take(per_row).copied())
        .collect()
}

/// The pass's target, linear RGBA, row by row from the top.
pub(crate) fn read_shaded(gpu: &Gpu, shade: &Shade) -> Vec<[f32; 4]> {
    let target = shade.target().expect("shaded");
    let halves: Vec<[u16; 4]> = read_texture(gpu, target, 8);
    halves.iter().map(|texel| texel.map(half)).collect()
}

/// The raster's depth buffer, row by row from the top.
fn read_depth(gpu: &Gpu, raster: &Raster) -> Vec<f32> {
    read_texture(gpu, raster.depth(), 4)
}

/// Every pixel of the frame `shade` last shaded, rebuilt by the probe (run with `shade`'s own
/// shader and bind groups); `None` where nothing is drawn.
pub(crate) fn probe(gpu: &Gpu, shade: &Shade) -> Vec<Option<Probed>> {
    let device = gpu.device();
    let bound = shade.bound.as_ref().expect("a frame shaded");
    let (width, height) = bound.size;
    let probe_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("probe"),
        entries: &[BindGroupLayoutEntry {
            binding: 1,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let source = format!("{}{PROBE_WGSL}", shade.source);
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("probe"),
        source: ShaderSource::Wgsl(source.into()),
    });
    let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("probe"),
        bind_group_layouts: &[
            Some(&shade.layout),
            Some(&shade.slot_layout),
            Some(shade.material.layout()),
            Some(&probe_layout),
        ],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("probe"),
        layout: Some(&layout),
        module: &module,
        entry_point: Some("probe"),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });
    let pixels = u64::from(width) * u64::from(height);
    let output = storage_buffer(gpu, "probed", pixels * 48).expect("a buffer");
    let group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("probe"),
        layout: &probe_layout,
        entries: &[entry(1, &output)],
    });
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bound.frame, &[]);
        pass.set_bind_group(1, &bound.slots, &[]);
        pass.set_bind_group(2, shade.material.bind_group(), &[]);
        pass.set_bind_group(3, &group, &[]);
        pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
    }
    gpu.queue().submit([encoder.finish()]);
    finish(gpu);
    let words: Vec<[f32; 4]> = gpu.read_buffer(&output).expect("read back");
    words
        .as_chunks::<3>()
        .0
        .iter()
        .map(|&[first, second, third]| {
            (first[3] >= 0.0).then_some(Probed {
                relative: [first[0], first[1], first[2]],
                depth: first[3],
                normal: [second[0], second[1], second[2]],
                roughness: second[3],
                behind: third[0] > 0.5,
            })
        })
        .collect()
}

/// `texel`'s bits, to compare texels exactly.
fn bits(texel: &[f32; 4]) -> [u32; 4] {
    texel.map(f32::to_bits)
}

/// How far the rebuilt depth may be from the raster's, relative.
///
/// That is float rounding of the same plane, interpolated two ways (up to 1.4e-4 seen on a fold
/// a few hundredths from the eye, where its two triangles lie within rounding of each other).
const DEPTH_TOLERANCE: f32 = 3e-4;

/// Pixels where `shade`'s rebuilt surface disagrees with what the raster drew: drawn on one side
/// only, or at a depth more than [`DEPTH_TOLERANCE`] off. And the drawn pixels.
fn disagreements(gpu: &Gpu, shade: &Shade, raster: &Raster) -> (usize, usize) {
    let depth = read_depth(gpu, raster);
    let ids = raster.read_visibility(gpu).expect("read back");
    let pixels = probe(gpu, shade);
    let mut wrong = 0;
    let mut drawn = 0;
    let compared = ids.iter().zip(&depth).zip(&pixels);
    for ((&id, &drawn_depth), rebuilt) in compared {
        match (id == crate::raster::BACKGROUND, rebuilt) {
            (true, None) => {}
            (false, Some(p)) => {
                drawn += 1;
                if (p.depth - drawn_depth).abs() > DEPTH_TOLERANCE * drawn_depth {
                    wrong += 1;
                }
            }
            _ => wrong += 1,
        }
    }
    (wrong, drawn)
}

/// A scene the raster drew, to rebuild: the raster after its last frame, and the frame's view.
pub(crate) struct Drawn {
    /// The raster.
    pub(crate) raster: Raster,
    /// The view.
    pub(crate) view: View,
    /// The clusters drawn.
    clusters: Vec<GpuCluster>,
}

/// `world`'s tiles at level `l`, drawn from 2.5 away with the camera still.
pub(crate) fn one_level(gpu: &Gpu, world: &World, l: u8) -> Drawn {
    let cut = crate::raster::tests::level_cut(world, l);
    let clusters = crate::raster::tests::cut_clusters(world, &cut, true);
    let view = View::look_at(
        Vec3::new(1.2, 0.9, 2.0),
        Vec3::default(),
        Vec3::new(0.0, 1.0, 0.0),
        45_f64.to_radians(),
        crate::raster::tests::size(),
        0.01,
    )
    .expect("a view");
    let frame = RasterFrame {
        view: &view,
        time: Duration::from_secs(1),
        moving: false,
    };
    let mut raster = crate::raster::tests::still_raster(gpu, world);
    draw_cut(gpu, &mut raster, world, &cut, &clusters, &frame);
    Drawn {
        raster,
        view,
        clusters,
    }
}

/// Scenes with every reason a vertex isn't where its tile holds it.
///
/// Random mixed-level cuts (edge variants), drawn moving (each tile's odd vertices morphed by
/// distance), the level choice's cuts (morphs by distance part way), and a refinement made with
/// the camera still, half way through its morph by time.
fn scenes(gpu: &Gpu, world: &World) -> Vec<Drawn> {
    let mut rng = Rng(23);
    let mut out = Vec::new();
    let raster = || {
        Raster::new(
            gpu,
            world.tree.capacity(),
            world.layout(),
            RasterConfig::default(),
        )
        .expect("the raster")
    };
    for _ in 0..2 {
        let (view, _) = random_view(&mut rng, (1.8, 3.0));
        let cut = random_cut(&mut rng, world, 3);
        let clusters = crate::raster::tests::cut_clusters(world, &cut, true);
        let frame = RasterFrame {
            view: &view,
            time: Duration::from_secs(1),
            moving: true,
        };
        let mut drawn = raster();
        draw_cut(gpu, &mut drawn, world, &cut, &clusters, &frame);
        out.push(Drawn {
            raster: drawn,
            view,
            clusters,
        });
    }
    let (mut lod, requests) = level_choice(gpu, world);
    for k in 0..2 {
        let (view, _) = random_view(&mut rng, (1.3, 2.5));
        let frame = RasterFrame {
            view: &view,
            time: Duration::from_secs(1),
            moving: true,
        };
        let mut drawn = raster();
        let (_, clusters, _) =
            lod_frame(gpu, world, &mut lod, &requests, &mut drawn, &frame, k == 0);
        out.push(Drawn {
            raster: drawn,
            view,
            clusters,
        });
    }
    // A refinement with the camera still: the new tiles morph in over 150 ms, drawn at 75.
    let (view, _) = random_view(&mut rng, (2.0, 2.6));
    let coarse = random_cut(&mut rng, world, 3);
    let mut fine = coarse.clone();
    let every_third = coarse.iter().copied().step_by(3);
    for place in every_third {
        if place.level() < world.finest {
            refine(&mut fine, place);
        }
    }
    let fine = grown(&mut rng, world, fine, 0);
    let config = RasterConfig::default().with_ramp(crate::raster::MorphThreshold::OFF);
    let mut drawn =
        Raster::new(gpu, world.tree.capacity(), world.layout(), config).expect("the raster");
    let mut clusters = Vec::new();
    for (cut, ms) in [(&coarse, 1000), (&fine, 1016), (&fine, 1091)] {
        clusters = crate::raster::tests::cut_clusters(world, cut, true);
        let frame = RasterFrame {
            view: &view,
            time: Duration::from_millis(ms),
            moving: false,
        };
        draw_cut(gpu, &mut drawn, world, cut, &clusters, &frame);
    }
    out.push(Drawn {
        raster: drawn,
        view,
        clusters,
    });
    out
}

#[test]
fn the_shading_pass_rebuilds_the_surface_the_raster_drew_at_every_pixel() {
    GpuTest::run(|gpu| {
        let world = World::new(gpu);
        let scenes = scenes(gpu, &world);
        let variants: BTreeSet<u32> = scenes
            .iter()
            .flat_map(|s| s.clusters.iter().map(|c| c.variant()))
            .collect();
        assert!(variants.len() > 4, "edge variants drawn: {variants:?}");
        let mut shade = shade_for(gpu, &world);
        let mut total = 0;
        for (k, scene) in scenes.iter().enumerate() {
            shade_frame(gpu, &mut shade, &world, &scene.raster, &scene.view);
            let (wrong, drawn) = disagreements(gpu, &shade, &scene.raster);
            assert!(drawn > 5000, "scene {k}: {drawn} pixels drawn");
            assert_eq!(
                wrong, 0,
                "scene {k}: {wrong} of {drawn} pixels rebuilt elsewhere"
            );
            total += drawn;
        }
        assert!(total > 50_000, "{total} pixels checked");
    });
}

#[test]
fn the_rebuild_check_bites_when_the_drawn_edges_the_morph_or_the_diagonal_is_ignored() {
    GpuTest::run(|gpu| {
        let world = World::new(gpu);
        let scenes = scenes(gpu, &world);
        let broken: [(&str, &[(&str, &str)]); 3] = [
            (
                "drawn edges",
                &[(
                    "cluster_cell(cluster % per, cluster / per, side - 1u, drawn_edges[n])",
                    "cluster_cell(cluster % per, cluster / per, side - 1u, 0u)",
                )],
            ),
            (
                "morph",
                &[(
                    "let t = min(time, morph_distance(e, own, p));",
                    "let t = 1.0;",
                )],
            ),
            (
                "diagonal",
                &[
                    ("second = quad_triangle(", "second = 0u * quad_triangle("),
                    ("if better(other, chosen) {", "if false {"),
                ],
            ),
        ];
        for (what, patches) in broken {
            let mut shade = patched(gpu, &world, patches);
            let wrong: usize = scenes
                .iter()
                .map(|scene| {
                    shade_frame(gpu, &mut shade, &world, &scene.raster, &scene.view);
                    disagreements(gpu, &shade, &scene.raster).0
                })
                .sum();
            assert!(wrong > 200, "ignoring the {what}: {wrong} pixels off");
        }
    });
}

#[test]
fn the_rebuilt_normals_are_unit_and_face_the_way_the_sphere_does() {
    GpuTest::run(|gpu| {
        let world = World::new(gpu);
        let scene = scenes(gpu, &world).remove(2);
        let mut shade = shade_for(gpu, &world);
        shade_frame(gpu, &mut shade, &world, &scene.raster, &scene.view);
        let eye = scene.view.eye();
        let mut checked = 0;
        let rebuilt = probe(gpu, &shade).into_iter().flatten();
        for p in rebuilt {
            let n = p.normal;
            let length = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((length - 1.0).abs() < 1e-4, "unit: {length}");
            // The world's normals are the sphere's: the point's direction.
            #[expect(
                clippy::cast_possible_truncation,
                reason = "the eye of a small test scene"
            )]
            let at = [
                p.relative[0] + eye.x as f32,
                p.relative[1] + eye.y as f32,
                p.relative[2] + eye.z as f32,
            ];
            let r = (at[0] * at[0] + at[1] * at[1] + at[2] * at[2]).sqrt();
            let cos = (at[0] * n[0] + at[1] * n[1] + at[2] * n[2]) / r;
            assert!(cos > 0.99, "the sphere's normal: {cos}");
            assert!(
                (p.roughness - 1.0).abs() < 1e-6,
                "the world's tiles are flat"
            );
            checked += 1;
        }
        assert!(checked > 5000, "{checked} pixels");
    });
}

#[test]
fn the_fragment_path_shades_as_the_compute_path_does() {
    GpuTest::run(|gpu| {
        let world = World::new(gpu);
        let scene = scenes(gpu, &world).remove(2);
        let images: Vec<Vec<[f32; 4]>> = [ShadePath::Compute, ShadePath::Fragment]
            .into_iter()
            .map(|path| {
                let mut shade = Shade::new(gpu, world.layout(), path, MatCapStyle::Glossy)
                    .expect("the shading pass");
                shade.set_settings(ShadeSettings { wireframe: true });
                assert_eq!(shade.path(), path);
                shade_frame(gpu, &mut shade, &world, &scene.raster, &scene.view);
                read_shaded(gpu, &shade)
            })
            .collect();
        let lit = images[0].iter().filter(|t| t[3] > 0.0).count();
        assert!(lit > 5000, "{lit} pixels shaded");
        let worst = images[0]
            .iter()
            .zip(&images[1])
            .flat_map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x - y).abs()))
            .fold(0.0_f32, f32::max);
        assert!(worst < 1e-3, "the paths differ by {worst}");
    });
}

#[test]
fn a_material_or_wireframe_change_is_a_shading_pass_alone() {
    GpuTest::run(|gpu| {
        let world = World::new(gpu);
        // Level 3, whose quads are some 25 pixels across, so the wireframe's lines are apart.
        let scene = one_level(gpu, &world, 3);
        let mut shade = shade_for(gpu, &world);
        shade_frame(gpu, &mut shade, &world, &scene.raster, &scene.view);
        let clay = read_shaded(gpu, &shade);
        let again = |pass: &mut Shade, (style, wireframe)| {
            pass.set_material(gpu, style);
            pass.set_settings(ShadeSettings { wireframe });
            let mut encoder = gpu
                .device()
                .create_command_encoder(&CommandEncoderDescriptor::default());
            assert!(
                pass.encode_again(gpu, &mut encoder).expect("encoded"),
                "a scene"
            );
            gpu.queue().submit([encoder.finish()]);
            finish(gpu);
            read_shaded(gpu, pass)
        };
        let glossy = again(&mut shade, (MatCapStyle::Glossy, false));
        // The same as a whole frame drawn and shaded glossy.
        let mut fresh = Shade::new(gpu, world.layout(), ShadePath::Compute, MatCapStyle::Glossy)
            .expect("the shading pass");
        shade_frame(gpu, &mut fresh, &world, &scene.raster, &scene.view);
        assert_eq!(glossy, read_shaded(gpu, &fresh), "as if drawn again");
        let changed = clay
            .iter()
            .zip(&glossy)
            .filter(|(a, b)| bits(a) != bits(b))
            .count();
        let lit = clay.iter().filter(|t| t[3] > 0.0).count();
        assert!(
            changed > lit / 2,
            "the material changed: {changed} of {lit}"
        );
        // The wireframe: lines over the triangles, not over everything.
        let wired = again(&mut shade, (MatCapStyle::Glossy, true));
        let lined = glossy
            .iter()
            .zip(&wired)
            .filter(|(a, b)| bits(a) != bits(b))
            .count();
        assert!(
            lined > lit / 20 && lined < lit / 2,
            "the wireframe covers {lined} of {lit}"
        );
        // At a display scale of 2 its lines are twice as wide: with the pixel's ramp each side,
        // a band of three pixels for two.
        shade.set_pixel_scale(2);
        let doubled = again(&mut shade, (MatCapStyle::Glossy, true));
        shade.set_pixel_scale(1);
        let wider = glossy
            .iter()
            .zip(&doubled)
            .filter(|(a, b)| bits(a) != bits(b))
            .count();
        assert!(
            wider > lined * 5 / 4,
            "{wider} pixels at scale 2, {lined} at 1"
        );
        // Background stays background.
        let background = clay.iter().zip(&wired).filter(|(a, _)| a[3] == 0.0);
        assert!(background.clone().count() > 1000, "background in view");
        assert!(
            background.into_iter().all(|(_, b)| bits(b) == [0; 4]),
            "untouched"
        );
    });
}

#[test]
fn shading_again_with_nothing_shaded_does_nothing() {
    GpuTest::run(|gpu| {
        let world = World::with_finest(gpu, 3);
        let mut shade = shade_for(gpu, &world);
        let mut encoder = gpu
            .device()
            .create_command_encoder(&CommandEncoderDescriptor::default());
        assert!(!shade.encode_again(gpu, &mut encoder).expect("encoded"));
        assert!(shade.target().is_none(), "no target yet");
    });
}

/// Scenes seen from just above the surface, along it, each with its clusters.
///
/// In random mixed-level cuts, big quads under the eye reach behind it, edge variants among
/// them, so the triangle is chosen without the screen; and quads seen grazing fold over
/// themselves.
fn grazing_scenes(gpu: &Gpu, world: &World) -> Vec<Drawn> {
    let mut out = grazing(gpu, world, 41, false);
    out.extend(grazing(gpu, world, 41, true));
    out
}

/// Six scenes for [`grazing_scenes`] from `seed`: of coarse cuts (levels 2 and 3), whose quads
/// under the eye reach behind it, or of random cuts from level 2, whose quads seen grazing
/// fold over themselves.
fn grazing(gpu: &Gpu, world: &World, seed: u64, coarse: bool) -> Vec<Drawn> {
    let mut rng = Rng(seed);
    let mut out = Vec::new();
    while out.len() < 6 {
        let d = [
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
        ]
        .map(f64::from);
        let Some(down) = Vec3::new(d[0], d[1], d[2]).normalized() else {
            continue;
        };
        let Some(along) = down.cross(Vec3::new(0.3, 1.0, 0.2)).normalized() else {
            continue;
        };
        let eye = down * 1.004;
        let target = eye + along - down * 0.25;
        let Some(view) = View::look_at(
            eye,
            target,
            down,
            70_f64.to_radians(),
            crate::raster::tests::size(),
            0.0005,
        ) else {
            continue;
        };
        let cut = if coarse {
            let level_2 = crate::raster::tests::level_cut(world, 2);
            grown(&mut rng, world, level_2, 1)
        } else {
            random_cut(&mut rng, world, 2)
        };
        let clusters = crate::raster::tests::cut_clusters(world, &cut, true);
        let frame = RasterFrame {
            view: &view,
            time: Duration::from_secs(1),
            moving: false,
        };
        let mut raster = crate::raster::tests::still_raster(gpu, world);
        draw_cut(gpu, &mut raster, world, &cut, &clusters, &frame);
        out.push(Drawn {
            raster,
            view,
            clusters,
        });
    }
    out
}

/// A bite check: what it breaks, the patches that break it, and the fewest pixels it must put
/// off.
type Bite<'a> = (&'a str, &'a [(&'a str, &'a str)], usize);

/// Pixels of `pixels` (the probe's) whose quad reaches behind the eye and lies on an edge whose
/// variant collapses one of its triangles, in `scene`.
fn behind_on_collapsed_edges(
    gpu: &Gpu,
    world: &World,
    scene: &Drawn,
    pixels: &[Option<Probed>],
) -> usize {
    let ids = scene.raster.read_visibility(gpu).expect("read back");
    let behind = ids
        .iter()
        .zip(pixels)
        .filter(|(_, p)| p.is_some_and(|p| p.behind))
        .filter_map(|(&id, _)| crate::raster::VisibilityId::decode(id));
    let on_variant = behind.filter(|id| {
        let quads = world.place_of(id.node).level().tile_quads();
        let per = (quads / 8).max(1);
        let (cx, cy) = (id.cluster % per, id.cluster / per);
        let Some(cluster) = scene
            .clusters
            .iter()
            .find(|c| c.node == id.node && c.position() == (cx, cy))
        else {
            return false;
        };
        let last = cluster.quads() - 1;
        let (qx, qy) = (id.quad % 8, id.quad / 8);
        let on = u32::from(qy == 0)
            | (u32::from(qx == last) << 1)
            | (u32::from(qy == last) << 2)
            | (u32::from(qx == 0) << 3);
        cluster.variant() & on != 0
    });
    on_variant.count()
}

#[test]
fn quads_reaching_behind_the_eye_are_rebuilt_as_drawn_collapsed_triangles_never_chosen() {
    GpuTest::run(|gpu| {
        let world = World::new(gpu);
        let scenes = grazing_scenes(gpu, &world);
        let mut shade = shade_for(gpu, &world);
        let mut exercised = 0;
        for (k, scene) in scenes.iter().enumerate() {
            shade_frame(gpu, &mut shade, &world, &scene.raster, &scene.view);
            let (wrong, drawn) = disagreements(gpu, &shade, &scene.raster);
            assert_eq!(
                wrong, 0,
                "scene {k}: {wrong} of {drawn} pixels rebuilt elsewhere"
            );
            let pixels = probe(gpu, &shade);
            exercised += behind_on_collapsed_edges(gpu, &world, scene, &pixels);
        }
        assert!(
            exercised > 100,
            "{exercised} pixels of collapsed border quads reach behind the eye"
        );
        // The bites. The choice as it was: the diagonal's side, the other triangle only where it
        // holds the centre better, nothing with a corner behind the eye, and a degenerate
        // triangle's coordinates as a point inside it. And a fold seen grazing, with neither the
        // facing test nor the nearer of two triangles holding the centre (either alone settles
        // it).
        let old_choice = [
            (
                "        return NO_TRIANGLE;",
                "        return vec3<f32>(1.0, 0.0, 0.0);",
            ),
            (
                "var second = select(0u, 1u, inside(second_tri.b) > inside(first.b));",
                "var second = 0u;",
            ),
            (
                "if better(other, chosen) {",
                "if in_front && inside(chosen.b) < 0.0 && inside(other.b) > inside(chosen.b) {",
            ),
        ];
        let fold_blind = [
            (
                "return inside(other.b) >= 0.0 && other.depth > chosen.depth;",
                "return false;",
            ),
            (
                "return t.faces && t.depth > 0.0 && inside(t.b) > -1e30;",
                "return t.depth > 0.0 && inside(t.b) > -1e30;",
            ),
        ];
        let bites: [Bite<'_>; 2] = [
            ("the old choice", &old_choice, 1000),
            ("a fold's hidden triangle chosen", &fold_blind, 5),
        ];
        for (what, patches, least) in bites {
            let mut broken = patched(gpu, &world, patches);
            let wrong: usize = scenes
                .iter()
                .map(|scene| {
                    shade_frame(gpu, &mut broken, &world, &scene.raster, &scene.view);
                    disagreements(gpu, &broken, &scene.raster).0
                })
                .sum();
            assert!(wrong > least, "{what}: {wrong} pixels off");
        }
    });
}
