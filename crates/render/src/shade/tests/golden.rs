//! Golden images (lavapipe), compared with the FLIP-style measure ([`super::flip`]):
//!
//! - **MatCap goldens:** a sphere shaded with each style (and the wireframe), drawn through the
//!   renderer's whole frame as the app draws it, against PNG files under the crate's
//!   `testdata/golden/`. Set `BRUSHCRAFT_WRITE_GOLDEN=1` to write them anew instead (then look at
//!   them, and commit them).
//! - **Against the finest level:** the level choice's frame, shaded, against a reference drawn
//!   entirely at the finest level, from a set of cameras.
//! - **The ≤ 1 px claim, on screen:** for the same cameras, how far the drawn surface lies from the
//!   finest, along the surface's normal, in pixels.
//!
//! Each has a bite: a check that the comparison fails for a frame it should refuse.

use std::path::{Path, PathBuf};

use super::flip::{Linear, flip};
use super::png::{self, Rgb};
use super::*;
use crate::Renderer;
use crate::cull::BoundsUpdate;
use crate::draw::{FrameInputs, Look, Scene, SceneView};
use crate::lod::{LodConfig, Thresholds};
use crate::raster::tests::{cut_clusters, finest_cut, still_raster};
use crate::raster::{FramePasses, MORPH, SceneInputs};
use crate::residency::REQUEST_HEADER;
use crate::surface::{Offscreen, Size, Target};

/// The environment variable that writes the golden files instead of checking them.
const WRITE: &str = "BRUSHCRAFT_WRITE_GOLDEN";

/// The golden images' side, pixels: small files (a PNG of stored blocks, ~28 KB).
const GOLDEN_SIDE: u32 = 96;

/// The frames' clear colour, linear.
const CLEAR: Color = Color {
    r: 0.018,
    g: 0.018,
    b: 0.022,
    a: 1.0,
};

/// Where the golden files live.
fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/golden")
}

/// The sRGB encoding of linear `c`, as a byte.
fn encode_srgb(c: f32) -> u8 {
    let c = c.clamp(0.0, 1.0);
    let v = if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a value in 0..=255"
    )]
    let byte = (v * 255.0).round() as u8;
    byte
}

/// The linear value of sRGB byte `b`.
fn decode_srgb(b: u8) -> f32 {
    let v = f32::from(b) / 255.0;
    if v <= 0.040_45 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// `image` (sRGB bytes) as linear RGB.
fn linear_of(image: &Rgb) -> Linear {
    Linear {
        width: usize::try_from(image.width).expect("small"),
        height: usize::try_from(image.height).expect("small"),
        pixels: image.pixels.iter().map(|p| p.map(decode_srgb)).collect(),
    }
}

/// `world` from `eye`, `size`, drawn through the renderer's whole frame as the app draws it (the
/// level choice, the raster, the shading, the tone map), shown as `look`: its pixels.
fn render(gpu: &Gpu, world: &World, size: Size, eye: Vec3, look: Look) -> Rgb {
    let mut target = Target::Offscreen(Offscreen::new(gpu, size).expect("target"));
    let mut renderer = Renderer::new(gpu, target.format()).expect("renderer");
    let passes = FramePasses::new(
        gpu,
        world.tree.capacity(),
        world.layout(),
        LodConfig::default(),
        MORPH,
        ShadePath::Compute,
    )
    .expect("the scene's passes");
    renderer.set_passes(passes);
    let requests = gpu.device().create_buffer(&BufferDescriptor {
        label: Some("test requests"),
        size: REQUEST_HEADER + 64 * 16,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let view = View::look_at(
        eye,
        Vec3::default(),
        Vec3::new(0.0, 1.0, 0.0),
        40_f64.to_radians(),
        size,
        0.01,
    )
    .expect("a view");
    let bounds = BoundsUpdate::since(&world.tree, 0);
    let scene = SceneInputs {
        tree: world.tree.buffer(),
        requests: &requests,
        slots: &world.slots,
        bounds: &bounds,
        view: &view,
        time: Duration::from_secs(1),
        moving: false,
        finest: world.finest,
        frame: 1,
    };
    let inputs = FrameInputs {
        size,
        clear: CLEAR,
        overlay: None,
        overlay_scale: 1,
        scene: Scene::Draw(&scene),
        look,
    };
    let drawn = renderer
        .draw(gpu, &mut target, &inputs, None, || {})
        .expect("drawn");
    assert!(drawn.scene && drawn.shown, "the scene was drawn and shown");
    finish(gpu);
    let Target::Offscreen(offscreen) = &target else {
        unreachable!("made offscreen");
    };
    let pixels = offscreen.read_pixels(gpu).expect("read back");
    Rgb {
        width: size.width(),
        height: size.height(),
        pixels: pixels.iter().map(|p| [p[0], p[1], p[2]]).collect(),
    }
}

/// The shaded look of `style`, with the wireframe or not.
const fn look(style: MatCapStyle, wireframe: bool) -> Look {
    Look {
        view: SceneView::Shaded,
        shading: ShadeSettings { wireframe },
        material: style,
    }
}

/// The goldens: a name and the look it shows.
fn goldens() -> [(&'static str, Look); 3] {
    [
        ("matcap-clay", look(MatCapStyle::Clay, false)),
        ("matcap-glossy", look(MatCapStyle::Glossy, false)),
        ("matcap-pale-wireframe", look(MatCapStyle::Pale, true)),
    ]
}

/// A golden frame matches its file if its FLIP-style error is at most this on average...
const GOLDEN_MEAN: f32 = 0.005;

/// ... and at most this at the 99th percentile.
const GOLDEN_P99: f32 = 0.05;

/// Where the goldens' camera is: in front, a little above and to the right, the whole sphere in
/// view.
const GOLDEN_EYE: Vec3 = Vec3::new(1.1, 0.9, 3.3);

/// The sphere for a golden, `world`, shown as `look` says.
fn golden_sphere(gpu: &Gpu, look: Look, world: &World) -> Rgb {
    let size = Size::new(GOLDEN_SIDE, GOLDEN_SIDE).expect("non-zero");
    render(gpu, world, size, GOLDEN_EYE, look)
}

#[test]
fn the_matcap_goldens_match_their_files() {
    GpuTest::run(|gpu| {
        // Finest level 5 for the shading; 3 for the wireframe, whose lines are then apart.
        let fine = World::with_finest(gpu, 5);
        let coarse = World::with_finest(gpu, 3);
        let write = std::env::var_os(WRITE).is_some();
        let dir = golden_dir();
        let mut frames = Vec::new();
        for (name, look) in goldens() {
            let world = if look.shading.wireframe {
                &coarse
            } else {
                &fine
            };
            let frame = golden_sphere(gpu, look, world);
            let path = dir.join(format!("{name}.png"));
            if write {
                std::fs::create_dir_all(&dir).expect("the golden directory");
                std::fs::write(&path, png::encode(&frame)).expect("written");
            }
            let bytes = std::fs::read(&path).unwrap_or_else(|error| {
                panic!("{}: {error}; write it with {WRITE}=1", path.display())
            });
            let golden = png::decode(&bytes).expect("a golden PNG");
            let errors = flip(&linear_of(&golden), &linear_of(&frame));
            let (mean, p99) = (errors.mean(), errors.percentile(99));
            assert!(
                mean <= GOLDEN_MEAN && p99 <= GOLDEN_P99,
                "{name}: FLIP mean {mean}, p99 {p99}"
            );
            frames.push((name, frame, golden));
        }
        // The bites: each frame against another style's golden is refused; and the wireframe's
        // golden refuses its sphere without the wireframe, the same style and world.
        let refused = |name: &str, frame: &Rgb, golden: &Rgb| {
            let errors = flip(&linear_of(golden), &linear_of(frame));
            assert!(
                errors.mean() > 4.0 * GOLDEN_MEAN && errors.percentile(99) > GOLDEN_P99,
                "{name}: mean {}, p99 {}",
                errors.mean(),
                errors.percentile(99)
            );
        };
        let numbered = frames.iter().enumerate();
        for (k, (name, frame, _)) in numbered {
            let (other, _, golden) = &frames[(k + 1) % frames.len()];
            refused(&format!("{name} against {other}'s golden"), frame, golden);
        }
        let unwired = golden_sphere(gpu, look(MatCapStyle::Pale, false), &coarse);
        let (_, _, wired) = &frames[2];
        refused("pale without the wireframe", &unwired, wired);
    });
}

/// Writes a larger shaded sphere for people to look at: `target/shaded-sphere-<style>.png`.
/// A tool, not a check: run it on its own with `--run-ignored only`.
#[test]
#[ignore = "a tool: writes PNGs of a shaded sphere into target/"]
fn a_large_shaded_sphere_for_people_to_look_at() {
    GpuTest::run(|gpu| {
        let world = World::new(gpu);
        let size = Size::new(640, 640).expect("non-zero");
        let out = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target");
        for style in MatCapStyle::ALL {
            let frame = render(gpu, &world, size, GOLDEN_EYE, look(style, false));
            let path = out.join(format!("shaded-sphere-{}.png", style.name()));
            std::fs::write(&path, png::encode(&frame)).expect("written");
        }
    });
}

/// The views the comparisons against the finest level are made from.
fn cameras() -> Vec<View> {
    let size = crate::raster::tests::size();
    let eyes = [
        Vec3::new(0.3, 0.4, 2.8),
        Vec3::new(-1.2, 0.5, 1.4),
        Vec3::new(1.1, -0.9, 1.0),
        Vec3::new(0.2, 1.45, 0.35),
        Vec3::new(-0.6, -0.3, 1.25),
    ];
    let views = eyes.into_iter().map(|eye| {
        View::look_at(
            eye,
            Vec3::new(0.05, -0.05, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            45_f64.to_radians(),
            size,
            0.01,
        )
        .expect("a view")
    });
    views.collect()
}

/// The shaded frame `shade` holds, tone mapped, the background the clear colour, linear.
fn shown(gpu: &Gpu, shade: &Shade, width: usize, height: usize) -> Linear {
    let clear = [CLEAR.r, CLEAR.g, CLEAR.b].map(|c| {
        #[expect(clippy::cast_possible_truncation, reason = "a colour")]
        let c = c as f32;
        c
    });
    let pixels = read_shaded(gpu, shade)
        .into_iter()
        .map(|t| {
            if t[3] == 0.0 {
                clear
            } else {
                [t[0], t[1], t[2]].map(crate::tone::tone)
            }
        })
        .collect();
    Linear {
        width,
        height,
        pixels,
    }
}

/// The level choice of `world` (refining by `e` and the quad cap at `refine_error` pixels),
/// drawn moving from `view` and shaded: what it shows, and what the probe rebuilds.
fn chosen(
    gpu: &Gpu,
    world: &World,
    view: &View,
    refine_error: f32,
) -> (Linear, Vec<Option<Probed>>) {
    let thresholds = Thresholds {
        refine_error,
        coarsen_error: refine_error * 0.5,
        refine_quad: 1e9,
        coarsen_quad: 1e9,
    };
    let config = LodConfig::default().with_thresholds(thresholds);
    let mut lod =
        crate::lod::Lod::new(gpu, world.tree.capacity(), config).expect("the level choice");
    let (_, requests) = level_choice(gpu, world);
    let raster_config = RasterConfig::for_level_choice(&config);
    let mut raster =
        Raster::new(gpu, world.tree.capacity(), world.layout(), raster_config).expect("the raster");
    let frame = RasterFrame {
        view,
        time: Duration::from_secs(1),
        moving: true,
    };
    lod_frame(gpu, world, &mut lod, &requests, &mut raster, &frame, true);
    let mut shade = shade_for(gpu, world);
    shade_frame(gpu, &mut shade, world, &raster, view);
    let size = view.size();
    let width = usize::try_from(size.width()).expect("small");
    let height = usize::try_from(size.height()).expect("small");
    (shown(gpu, &shade, width, height), probe(gpu, &shade))
}

/// `world` entirely at its finest level, unmorphed, from `view`, shaded: what it shows and what
/// the probe rebuilds.
fn finest(gpu: &Gpu, world: &World, view: &View) -> (Linear, Vec<Option<Probed>>) {
    let mut raster = still_raster(gpu, world);
    let cut = finest_cut(world);
    let clusters = cut_clusters(world, &cut, true);
    let frame = RasterFrame {
        view,
        time: Duration::from_secs(1),
        moving: false,
    };
    draw_cut(gpu, &mut raster, world, &cut, &clusters, &frame);
    let mut shade = shade_for(gpu, world);
    shade_frame(gpu, &mut shade, world, &raster, view);
    let size = view.size();
    let width = usize::try_from(size.width()).expect("small");
    let height = usize::try_from(size.height()).expect("small");
    (shown(gpu, &shade, width, height), probe(gpu, &shade))
}

/// A frame of the level choice matches the finest level if its FLIP-style error is at most this
/// on average...
const FINEST_MEAN: f32 = 0.01;

/// ... and at most this at the 99th percentile.
const FINEST_P99: f32 = 0.1;

/// How far, in pixels along the normal, the drawn surface lies from the finest one at a pixel
/// both draw: `None` where the ray grazes either (the normal within 75° of the line of sight).
fn screen_error(view: &View, drawn: &Probed, finest: &Probed) -> Option<f64> {
    let length = |v: [f32; 3]| {
        let [x, y, z] = v.map(f64::from);
        (x * x + y * y + z * z).sqrt()
    };
    let (t_drawn, t_finest) = (length(drawn.relative), length(finest.relative));
    let ray = finest.relative.map(|c| f64::from(c) / t_finest);
    let normal = finest.normal.map(f64::from);
    let facing = (ray[0] * normal[0] + ray[1] * normal[1] + ray[2] * normal[2]).abs();
    if facing < 0.26 {
        return None;
    }
    let forward = view.forward();
    let depth = t_finest * (ray[0] * forward.x + ray[1] * forward.y + ray[2] * forward.z);
    Some((t_drawn - t_finest).abs() * facing * view.pixels_per_unit() / depth)
}

/// The largest on-screen error of `drawn` against `finest`, over pixels both draw, and those
/// pixels.
fn largest_screen_error(
    view: &View,
    drawn: &[Option<Probed>],
    finest: &[Option<Probed>],
) -> (f64, usize) {
    let both = drawn
        .iter()
        .zip(finest)
        .filter_map(|(d, f)| Some((d.as_ref()?, f.as_ref()?)));
    let errors: Vec<f64> = both.filter_map(|(d, f)| screen_error(view, d, f)).collect();
    (errors.iter().copied().fold(0.0, f64::max), errors.len())
}

/// How far, in pixels, the silhouettes of two images lie apart.
///
/// That is the largest distance from a pixel only one of them draws to the nearest pixel the
/// other draws, searched up to `reach` pixels (`reach + 1` if none is that near). The ≤ 1 px
/// check skips grazing pixels, where the silhouette is; this measures it.
fn silhouette_gap(
    size: (usize, usize),
    first: &[Option<Probed>],
    second: &[Option<Probed>],
    reach: usize,
) -> f64 {
    let (width, height) = size;
    let drawn = |image: &[Option<Probed>], x: usize, y: usize| {
        image.get(y * width + x).is_some_and(Option::is_some)
    };
    let nearest = |image: &[Option<Probed>], x: usize, y: usize| {
        let window = (y.saturating_sub(reach)..=(y + reach).min(height - 1)).flat_map(|v| {
            (x.saturating_sub(reach)..=(x + reach).min(width - 1)).map(move |u| (u, v))
        });
        let found = window.filter(|&(u, v)| drawn(image, u, v)).map(|(u, v)| {
            let (du, dv) = (u.abs_diff(x), v.abs_diff(y));
            let squared = u32::try_from(du * du + dv * dv).expect("small");
            f64::from(squared).sqrt()
        });
        let limit = f64::from(u32::try_from(reach + 1).expect("small"));
        found.fold(limit, f64::min)
    };
    let pixels = (0..height).flat_map(|y| (0..width).map(move |x| (x, y)));
    let gaps = pixels.filter_map(|(x, y)| match (drawn(first, x, y), drawn(second, x, y)) {
        (true, false) => Some(nearest(second, x, y)),
        (false, true) => Some(nearest(first, x, y)),
        _ => None,
    });
    gaps.fold(0.0, f64::max)
}

/// The test viewport's width and height.
fn frame_size() -> (usize, usize) {
    let size = crate::raster::tests::size();
    (
        usize::try_from(size.width()).expect("small"),
        usize::try_from(size.height()).expect("small"),
    )
}

/// The farthest a pixel drawn by only one of two frames may lie from the other's silhouette.
///
/// The frames are the level choice's and the finest level's; the gap is the pixel of error the
/// level choice allows, a diagonal neighbour (√2) included. Measured: 1 px from the farther
/// cameras, 0 close up; a level choice at 4 px puts them 2 px apart.
const SILHOUETTE_GAP: f64 = 1.5;

#[test]
fn the_level_choice_s_shaded_frames_match_the_finest_level_and_stay_within_a_pixel_of_it() {
    GpuTest::run(|gpu| {
        let world = World::new(gpu);
        let views = cameras();
        let numbered = views.iter().enumerate();
        for (k, view) in numbered {
            let (reference, finest_surface) = finest(gpu, &world, view);
            let (image, surface) = chosen(gpu, &world, view, 1.0);
            let errors = flip(&reference, &image);
            let (mean, p99) = (errors.mean(), errors.percentile(99));
            assert!(
                mean <= FINEST_MEAN && p99 <= FINEST_P99,
                "camera {k}: FLIP mean {mean}, p99 {p99}"
            );
            let (largest, pixels) = largest_screen_error(view, &surface, &finest_surface);
            assert!(pixels > 10_000, "camera {k}: {pixels} pixels compared");
            assert!(
                largest <= 1.0,
                "camera {k}: {largest} px from the finest level"
            );
            let gap = silhouette_gap(frame_size(), &surface, &finest_surface, 6);
            assert!(
                gap <= SILHOUETTE_GAP,
                "camera {k}: silhouettes {gap} px apart"
            );
        }
    });
}

#[test]
fn the_comparisons_with_the_finest_level_bite_on_a_level_choice_four_times_too_coarse() {
    GpuTest::run(|gpu| {
        let world = World::new(gpu);
        let (mut worst_flip, mut worst_px, mut worst_gap) = (0.0_f32, 0.0_f64, 0.0_f64);
        for view in &cameras() {
            let (reference, finest_surface) = finest(gpu, &world, view);
            let (image, surface) = chosen(gpu, &world, view, 4.0);
            worst_flip = worst_flip.max(flip(&reference, &image).percentile(99));
            worst_px = worst_px.max(largest_screen_error(view, &surface, &finest_surface).0);
            let gap = silhouette_gap(frame_size(), &surface, &finest_surface, 6);
            worst_gap = worst_gap.max(gap);
        }
        assert!(worst_px > 1.5, "a 4 px threshold shows: {worst_px} px");
        assert!(
            worst_gap > SILHOUETTE_GAP,
            "and at the silhouette: {worst_gap} px"
        );
        assert!(
            worst_flip > FINEST_P99,
            "and the FLIP-style check sees it: {worst_flip}"
        );
    });
}

#[test]
fn the_srgb_round_trip_of_a_byte_is_exact() {
    for b in 0..=255_u8 {
        assert_eq!(encode_srgb(decode_srgb(b)), b, "{b}");
    }
}
