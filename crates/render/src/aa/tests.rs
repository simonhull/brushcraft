//! Accumulation and the edge filter, on lavapipe through the renderer's whole frame, over the
//! raster tests' sphere (every tile resident to level 4).
//!
//! The reference is a supersampled render of our own: the same scene drawn at an 8 × 8 grid of
//! offsets inside each pixel (64 frames, the level choice unjittered as the renderer draws it),
//! its shaded images averaged on the CPU and shown as the tone map shows a mean. It shares the
//! raster and shading with what it checks, but not the jitter sequence, the accumulation or the
//! means' half floats.

use std::collections::BTreeMap;
use std::time::Duration;

use wgpu::{Buffer, BufferDescriptor, BufferUsages, Color, CommandEncoderDescriptor, PollType};

use super::*;
use crate::camera::{Vec3, View};
use crate::draw::{Drawn, FrameInputs, Scene};
use crate::gpu::GpuTest;
use crate::lod::LodConfig;
use crate::material::MatCapStyle;
use crate::raster::tests::World;
use crate::raster::{FramePasses, MORPH, SceneInputs};
use crate::residency::REQUEST_HEADER;
use crate::shade::tests::png::{self, Rgb};
use crate::shade::tests::{half, read_shaded};
use crate::shade::{ShadePath, ShadeSettings};
use crate::surface::{Offscreen, Target};
use crate::{Renderer, tone};

/// The frames' clear colour, linear: unlike the sphere, so its silhouette is an edge.
const CLEAR: Color = Color {
    r: 0.2,
    g: 0.1,
    b: 0.05,
    a: 1.0,
};

/// Where the eye is: the whole sphere in view, a little off its axis.
const EYE: Vec3 = Vec3::new(0.4, 0.3, 2.6);

/// The frames' size.
fn size() -> Size {
    Size::new(96, 64).expect("non-zero")
}

/// The shaded look the tests show.
const LOOK: Look = Look {
    view: SceneView::Shaded,
    shading: ShadeSettings { wireframe: false },
    material: MatCapStyle::Clay,
};

/// The view from `eye` at `size`.
fn view_of(eye: Vec3, size: Size) -> View {
    View::look_at(
        eye,
        Vec3::default(),
        Vec3::new(0.0, 1.0, 0.0),
        45_f64.to_radians(),
        size,
        0.01,
    )
    .expect("a view")
}

/// A request buffer for the level choice.
fn requests(gpu: &Gpu) -> Buffer {
    gpu.device().create_buffer(&BufferDescriptor {
        label: Some("test requests"),
        size: REQUEST_HEADER + 64 * 16,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// The scene's passes for `world`.
fn passes_for(gpu: &Gpu, world: &World) -> FramePasses {
    FramePasses::new(
        gpu,
        world.tree.capacity(),
        world.layout(),
        LodConfig::default(),
        MORPH,
        ShadePath::Compute,
    )
    .expect("the scene's passes")
}

/// Waits for the GPU.
fn finish(gpu: &Gpu) {
    gpu.device()
        .poll(PollType::wait_indefinitely())
        .expect("the GPU finishes");
}

/// A renderer drawing frames of a world from one eye, as the app does: each frame a second after
/// the last (so a morph is over by the next), its bounds update the tiles stamped since the last.
struct Session {
    /// The renderer.
    renderer: Renderer,
    /// Its offscreen target.
    target: Target,
    /// The level choice's requests.
    requests: Buffer,
    /// The view.
    view: View,
    /// Frames drawn.
    frames: u64,
    /// The tree's generation the last frame's bounds brought the GPU to.
    generation: u64,
}

impl Session {
    /// A fresh renderer for `world`, seen from `eye` at `size`.
    fn new(gpu: &Gpu, world: &World, eye: Vec3, size: Size) -> Self {
        Self::with_view(gpu, world, &view_of(eye, size))
    }

    /// A fresh renderer for `world`, seen through `view`.
    fn with_view(gpu: &Gpu, world: &World, view: &View) -> Self {
        let size = view.size();
        let target = Target::Offscreen(Offscreen::new(gpu, size).expect("target"));
        let mut renderer = Renderer::new(gpu, target.format()).expect("renderer");
        renderer.set_passes(passes_for(gpu, world));
        Self {
            renderer,
            target,
            requests: requests(gpu),
            view: *view,
            frames: 0,
            generation: 0,
        }
    }

    /// Draws a frame, the camera `moving` or not, and waits for it.
    fn draw(&mut self, gpu: &Gpu, world: &World, moving: bool) -> Drawn {
        self.frames += 1;
        let bounds = BoundsUpdate::since(&world.tree, self.generation);
        self.generation = bounds.generation();
        let scene = SceneInputs {
            tree: world.tree.buffer(),
            requests: &self.requests,
            slots: &world.slots,
            bounds: &bounds,
            view: &self.view,
            time: Duration::from_secs(self.frames),
            moving,
            finest: world.finest,
            frame: self.frames,
        };
        let inputs = FrameInputs {
            size: self.view.size(),
            clear: CLEAR,
            overlay: None,
            overlay_scale: 1,
            scene: Scene::Draw(&scene),
            look: LOOK,
        };
        let drawn = self
            .renderer
            .draw(gpu, &mut self.target, &inputs, None, || {})
            .expect("drawn");
        finish(gpu);
        drawn
    }

    /// Every pixel's mean, premultiplied linear.
    fn means(&self, gpu: &Gpu) -> Vec<[f32; 4]> {
        let buffer = self.passes().aa().mean().expect("accumulated");
        let halves: Vec<[u16; 4]> = gpu.read_buffer(buffer).expect("read back");
        halves.iter().map(|texel| texel.map(half)).collect()
    }

    /// The means shown unfiltered, as sRGB bytes: after one frame, the one-sample image.
    fn unfiltered(&self, gpu: &Gpu) -> Vec<[u8; 4]> {
        let means = self.means(gpu);
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "rounded from 0..=255"
        )]
        let byte = |c: f32| (c * 255.0).round() as u8;
        means
            .into_iter()
            .map(|mean| {
                let [r, g, b] = shown(mean).map(byte);
                [r, g, b, 255]
            })
            .collect()
    }

    /// How many pixels of the frame differ by more than a code from the means shown unfiltered.
    fn filtered_pixels(&self, gpu: &Gpu) -> usize {
        let frame = self.pixels(gpu);
        let means = self.means(gpu);
        let unfiltered = means.into_iter().map(shown);
        let pairs = frame.iter().zip(unfiltered);
        pairs
            .filter(|(pixel, want)| {
                (0..3).any(|k| (f32::from(pixel[k]) - want[k] * 255.0).abs() > 1.5)
            })
            .count()
    }

    /// Shows the last scene again, as `look` (a look change, or a frame for the overlay alone).
    fn reshade(&mut self, gpu: &Gpu, look: Look) -> Drawn {
        let inputs = FrameInputs {
            size: self.view.size(),
            clear: CLEAR,
            overlay: None,
            overlay_scale: 1,
            scene: Scene::Reshade,
            look,
        };
        let drawn = self
            .renderer
            .draw(gpu, &mut self.target, &inputs, None, || {})
            .expect("drawn");
        finish(gpu);
        drawn
    }

    /// Draws still frames until every pixel holds all its samples; returns how many it took.
    fn converge(&mut self, gpu: &Gpu, world: &World) -> u32 {
        for drawn in 1..=4 * SAMPLES {
            self.draw(gpu, world, false);
            if self.counts(gpu).iter().all(|&count| count == SAMPLES) {
                return drawn;
            }
        }
        panic!("the accumulation never converged");
    }

    /// The scene's passes.
    fn passes(&self) -> &FramePasses {
        self.renderer.passes().expect("given")
    }

    /// The frame as the target holds it, sRGB bytes.
    fn pixels(&self, gpu: &Gpu) -> Vec<[u8; 4]> {
        let Target::Offscreen(offscreen) = &self.target else {
            unreachable!("made offscreen");
        };
        offscreen.read_pixels(gpu).expect("read back")
    }

    /// The last frame's visibility buffer.
    fn ids(&self, gpu: &Gpu) -> Vec<u32> {
        self.passes()
            .raster()
            .read_visibility(gpu)
            .expect("read back")
    }

    /// Every pixel's record: its start frame (overflow bit cleared), whether it overflowed, and
    /// the tiles it names.
    fn records(&self, gpu: &Gpu) -> Vec<Record> {
        let buffer = self.passes().aa().record().expect("accumulated");
        let words: Vec<[u32; 4]> = gpu.read_buffer(buffer).expect("read back");
        words.iter().map(|&w| Record::of(w)).collect()
    }

    /// Every pixel's sample count.
    fn counts(&self, gpu: &Gpu) -> Vec<u32> {
        let frame = self.passes().aa().frame();
        let records = self.records(gpu);
        records
            .iter()
            .map(|record| (frame - record.start + 1).min(SAMPLES))
            .collect()
    }

    /// Rebuilds the accumulation with `patches` in its pixel pass.
    fn patch(&mut self, gpu: &Gpu, world: &World, patches: &[(&str, &str)]) {
        let mut fresh = passes_for(gpu, world);
        fresh.patch_aa(gpu, world.tree.capacity(), patches);
        self.renderer.set_passes(fresh);
        self.frames = 0;
        self.generation = 0;
    }
}

/// A pixel's record, decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Record {
    /// The frame its history started.
    start: u32,
    /// Whether it saw more tiles than it names.
    overflow: bool,
    /// The tiles it names, [`EMPTY`] for none.
    seen: [u32; 4],
}

/// No tile, in a record.
const EMPTY: u32 = 0xF_FFFF;

impl Record {
    /// The record of `words`.
    const fn of(words: [u32; 4]) -> Self {
        let [w0, w1, w2, w3] = words;
        Self {
            start: w0 & 0x7FFF_FFFF,
            overflow: w0 >> 31 == 1,
            seen: [
                w1 & EMPTY,
                w2 & EMPTY,
                w3 & EMPTY,
                (w1 >> 20) | ((w2 >> 20) << 12),
            ],
        }
    }

    /// Whether it names `node`.
    fn names(&self, node: u32) -> bool {
        self.seen.contains(&node)
    }
}

/// The node an id names, `None` for the background.
const fn node_of(id: u32) -> Option<u32> {
    if id == crate::raster::BACKGROUND {
        None
    } else {
        Some(id >> 12)
    }
}

/// `world` from `eye` at `size`, supersampled on a `grid` × `grid` grid of offsets in each pixel:
/// the mean of the shaded images, premultiplied linear.
fn supersampled(gpu: &Gpu, world: &World, eye: Vec3, size: Size, grid: u32) -> Vec<[f32; 4]> {
    let mut passes = passes_for(gpu, world);
    let requests = requests(gpu);
    let view = view_of(eye, size);
    let pixels = usize::try_from(size.width() * size.height()).expect("small");
    let mut sum = vec![[0.0_f32; 4]; pixels];
    let offset = |i: u32| (f64::from(i) + 0.5) / f64::from(grid) - 0.5;
    // A few unjittered frames first, as a session starts: the level choice settles the same way.
    let warm = 3;
    let jitters = (0..grid * grid).map(|k| Jitter::new(offset(k % grid), offset(k / grid)));
    let frames: Vec<Option<Jitter>> = std::iter::repeat_n(None, warm).chain(jitters).collect();
    let mut generation = 0;
    for (frame, jitter) in (1_u64..).zip(&frames) {
        let bounds = BoundsUpdate::since(&world.tree, generation);
        generation = bounds.generation();
        let scene = SceneInputs {
            tree: world.tree.buffer(),
            requests: &requests,
            slots: &world.slots,
            bounds: &bounds,
            view: &view,
            time: Duration::from_secs(frame),
            moving: false,
            finest: world.finest,
            frame,
        };
        let mut encoder = gpu
            .device()
            .create_command_encoder(&CommandEncoderDescriptor::default());
        let at = jitter.unwrap_or(Jitter::CENTRE);
        let readback = passes
            .encode_jittered(gpu, &mut encoder, &scene, at)
            .expect("encoded");
        passes
            .encode_shading(gpu, &mut encoder, &scene)
            .expect("shaded");
        gpu.queue().submit([encoder.finish()]);
        if let Some(readback) = readback {
            readback.submitted();
        }
        finish(gpu);
        if jitter.is_some() {
            let shaded = read_shaded(gpu, passes.shade());
            for (total, texel) in sum.iter_mut().zip(shaded) {
                for (t, c) in total.iter_mut().zip(texel) {
                    *t += c;
                }
            }
        }
    }
    let count = f32::from(u16::try_from(grid * grid).expect("small"));
    sum.iter().map(|total| total.map(|t| t / count)).collect()
}

/// The sRGB encoding of linear `c` in `[0, 1]`.
fn encode(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// Premultiplied `c` as the tone map shows it over [`CLEAR`], sRGB-encoded, `[0, 1]`.
fn shown(c: [f32; 4]) -> [f32; 3] {
    #[expect(clippy::cast_possible_truncation, reason = "the GPU's f32")]
    let clear = [CLEAR.r as f32, CLEAR.g as f32, CLEAR.b as f32];
    let a = c[3].clamp(0.0, 1.0);
    let linear: [f32; 3] = std::array::from_fn(|k| {
        if a <= 0.0 {
            clear[k]
        } else {
            a * tone::tone(c[k] / a) + (1.0 - a) * clear[k]
        }
    });
    linear.map(encode)
}

/// The root mean square difference between `frame` (sRGB bytes) and `reference` (encoded, `[0,
/// 1]`), over every channel of every pixel.
fn rmse(frame: &[[u8; 4]], reference: &[[f32; 3]]) -> f32 {
    let squares: Vec<f32> = frame
        .iter()
        .zip(reference)
        .flat_map(|(pixel, want)| {
            (0..3).map(move |k| (f32::from(pixel[k]) / 255.0 - want[k]).powi(2))
        })
        .collect();
    let n = f32::from(u16::try_from(squares.len()).expect("small"));
    (squares.iter().sum::<f32>() / n).sqrt()
}

#[test]
fn the_jitters_are_the_halton_points_about_the_pixels_centre() {
    assert_eq!(jitter(0), Jitter::CENTRE, "the first sample is the centre");
    assert_eq!(jitter(SAMPLES + 3), jitter(3), "a cycle of SAMPLES");
    let points: Vec<Jitter> = (0..SAMPLES).map(jitter).collect();
    for (k, p) in points.iter().enumerate() {
        assert!((-0.5..0.5).contains(&p.x()) && (-0.5..0.5).contains(&p.y()));
        assert!(!points[..k].contains(p), "distinct: {p:?}");
    }
    // Well spread: every quarter of the pixel holds a quarter of them, give or take one.
    let mut quarters = [0_u32; 4];
    for p in &points {
        quarters[usize::from(p.x() >= 0.0) + 2 * usize::from(p.y() >= 0.0)] += 1;
    }
    assert!(
        quarters.iter().all(|&n| n.abs_diff(SAMPLES / 4) <= 1),
        "{quarters:?}"
    );
    assert!(
        (radical_inverse(2, 6) - 0.375).abs() < 1e-12,
        "0b110 mirrored"
    );
    assert!(
        (radical_inverse(3, 5) - 7.0 / 9.0).abs() < 1e-12,
        "12 in base 3 mirrored"
    );
}

#[test]
fn still_frames_accumulate_moving_ones_filter_and_a_look_change_restarts() {
    let mut state = AaState::default();
    let size = size();
    let first = state.draw(true, true, LOOK, size);
    assert_eq!(
        first,
        AaFrame::Accumulate {
            jitter: Jitter::CENTRE,
            restart: true
        }
    );
    let second = state.draw(true, true, LOOK, size);
    assert_eq!(
        second,
        AaFrame::Accumulate {
            jitter: jitter(1),
            restart: false
        }
    );
    assert_eq!(
        state.reshade(true, LOOK, size),
        AaFrame::Show,
        "nothing new to shade"
    );
    let glossy = Look {
        material: MatCapStyle::Glossy,
        ..LOOK
    };
    assert_eq!(
        state.reshade(true, glossy, size),
        AaFrame::Accumulate {
            jitter: jitter(1),
            restart: true
        },
        "the last raster's sample, shaded anew, starts the history"
    );
    assert_eq!(
        state.draw(true, true, glossy, size),
        AaFrame::Accumulate {
            jitter: jitter(2),
            restart: false
        }
    );
    assert_eq!(
        state.draw(false, true, glossy, size),
        AaFrame::Filter,
        "moving"
    );
    assert_eq!(
        state.draw(true, true, glossy, size),
        AaFrame::Accumulate {
            jitter: Jitter::CENTRE,
            restart: true
        },
        "stopped: from the centre again"
    );
    let other = Size::new(64, 64).expect("non-zero");
    assert!(matches!(
        state.draw(true, true, glossy, other),
        AaFrame::Accumulate { restart: true, .. }
    ));
    let clusters = Look {
        view: SceneView::Clusters,
        ..LOOK
    };
    assert_eq!(state.draw(true, true, clusters, other), AaFrame::Plain);
    state.lost();
    assert!(matches!(
        state.draw(true, true, glossy, other),
        AaFrame::Accumulate { restart: true, .. }
    ));
}

/// The convergence test's bounds: the accumulated frame's error against the reference...
const CONVERGED_RMSE: f32 = 0.006;

/// ... and at most this share of the one-sample frame's.
const CONVERGED_SHARE: f32 = 0.25;

#[test]
fn a_still_scene_converges_to_the_supersampled_reference() {
    GpuTest::run(|gpu| {
        let world = World::with_finest(gpu, 4);
        let reference = supersampled(gpu, &world, EYE, size(), 8);
        let reference: Vec<[f32; 3]> = reference.into_iter().map(shown).collect();
        let mut session = Session::new(gpu, &world, EYE, size());
        session.draw(gpu, &world, false);
        let aliased = session.unfiltered(gpu);
        let frames = session.converge(gpu, &world);
        let converged = session.pixels(gpu);
        let (before, after) = (rmse(&aliased, &reference), rmse(&converged, &reference));
        eprintln!("one sample {before:.5}, {frames} frames more {after:.5}");
        assert!(
            frames < 2 * SAMPLES,
            "converged in about SAMPLES frames: {frames}"
        );
        assert!(after <= CONVERGED_RMSE, "converged: {after}");
        assert!(
            after <= before * CONVERGED_SHARE,
            "far nearer than one sample: {after} against {before}"
        );
        if let Some(dir) = std::env::var_os("BRUSHCRAFT_WRITE_AA_PNG") {
            write_pair(gpu, &world, std::path::Path::new(&dir));
        }
    });
}

/// Writes a close-up of the sphere's edge, one sample and accumulated, magnified, to `dir`.
fn write_pair(gpu: &Gpu, world: &World, dir: &std::path::Path) {
    let size = Size::new(128, 96).expect("non-zero");
    // From 3 away, a point of the silhouette (on the circle at z = 1/3, radius √8/3), narrowly.
    let view = View::look_at(
        Vec3::new(0.0, 0.0, 3.0),
        Vec3::new(0.817, 0.471, 0.333),
        Vec3::new(0.0, 1.0, 0.0),
        6_f64.to_radians(),
        size,
        0.01,
    )
    .expect("a view");
    let mut session = Session::with_view(gpu, world, &view);
    session.draw(gpu, world, false);
    let aliased = session.unfiltered(gpu);
    session.converge(gpu, world);
    let converged = session.pixels(gpu);
    for (name, pixels) in [("aliased", aliased), ("accumulated", converged)] {
        let image = magnified(&pixels, size, 4);
        let path = dir.join(format!("aa-{name}.png"));
        std::fs::write(&path, png::encode(&image)).expect("written");
    }
}

/// `pixels` of `size`, each `by` × `by` pixels.
fn magnified(pixels: &[[u8; 4]], size: Size, by: u32) -> Rgb {
    let (width, height) = (size.width() * by, size.height() * by);
    let at = |x: u32, y: u32| {
        let p = pixels[usize::try_from((y / by) * size.width() + x / by).expect("small")];
        [p[0], p[1], p[2]]
    };
    let rows = (0..height).flat_map(|y| (0..width).map(move |x| at(x, y)));
    Rgb {
        width,
        height,
        pixels: rows.collect(),
    }
}

/// The pixel pass's check of the tiles a record names, and its stand-in that ignores them.
const NAMED_CHECK: (&str, &str) = (
    "reset = reset || (old[k] != EMPTY && changed_after(old[k], since));",
    "reset = reset || false;",
);

/// What a change made to a converged scene did.
struct Changed {
    /// The tile it was made to.
    tile: u32,
    /// Every pixel's record before it.
    before: Vec<Record>,
    /// Per pixel, whether its history started again with the frame after it.
    reset: Vec<bool>,
}

impl Changed {
    /// Per pixel, whether it reset and whether it should have: its record named the tile, or it
    /// was crowded.
    fn against_records(&self) -> Vec<(bool, bool)> {
        let wanted = self
            .before
            .iter()
            .map(|old| old.names(self.tile) || old.overflow);
        self.reset.iter().copied().zip(wanted).collect()
    }
}

/// Converges `session`, picks a tile (the smallest with at least 100 pixels: a patch with
/// neighbours all round), makes `change` to it, draws a still frame and reports.
fn after_change(
    gpu: &Gpu,
    world: &mut World,
    session: &mut Session,
    change: impl FnOnce(&Gpu, &mut World, u32),
) -> Changed {
    session.converge(gpu, world);
    let before = session.records(gpu);
    let ids = session.ids(gpu);
    let mut tally: BTreeMap<u32, usize> = BTreeMap::new();
    let nodes = ids.iter().filter_map(|&id| node_of(id));
    for node in nodes {
        *tally.entry(node).or_default() += 1;
    }
    let (&tile, _) = tally
        .iter()
        .filter(|&(_, &n)| n >= 100)
        .min_by_key(|&(_, &n)| n)
        .expect("a tile in view");
    change(gpu, world, tile);
    session.draw(gpu, world, false);
    let frame = session.passes().aa().frame();
    let after = session.records(gpu);
    let reset = before
        .iter()
        .zip(&after)
        .map(|(old, new)| {
            let reset = new.start == frame;
            assert!(
                reset || new.start == old.start,
                "kept or reset, nothing else"
            );
            reset
        })
        .collect();
    Changed {
        tile,
        before,
        reset,
    }
}

/// A new epoch for one tile, its data unchanged: only the bounds update names it.
fn restamped(gpu: &Gpu, world: &mut World, tile: u32) {
    let generation = world.tree.generation();
    world.restamp(gpu, tile);
    let update = BoundsUpdate::since(&world.tree, generation);
    let named: Vec<u32> = update.nodes().collect();
    assert_eq!(named, [tile], "the epoch names it");
}

#[test]
fn a_single_changed_tile_resets_only_the_pixels_that_saw_it() {
    GpuTest::run(|gpu| {
        let mut world = World::with_finest(gpu, 4);
        let mut session = Session::new(gpu, &world, EYE, size());
        let changed = after_change(gpu, &mut world, &mut session, restamped);
        let pixels = changed.against_records();
        let resets = pixels.iter().filter(|&&(reset, _)| reset).count();
        let saw = pixels.iter().filter(|&&(_, wanted)| wanted).count();
        let crowded = changed.before.iter().filter(|r| r.overflow).count();
        eprintln!(
            "{resets} reset, {saw} saw the tile, {crowded} crowded, of {}",
            pixels.len()
        );
        assert_eq!(
            crowded, 0,
            "four tiles a record: no pixel of the scene is crowded"
        );
        assert!(saw > 20, "the tile covers a patch: {saw}");
        assert!(saw < pixels.len() / 4, "and not most of the frame");
        assert!(
            pixels.iter().all(|&(reset, wanted)| reset == wanted),
            "exactly the pixels that saw it reset; the rest keep their samples"
        );
        // The bite: ignoring the tiles a record names, pixels at the tile's edge whose sample now
        // sees a neighbour keep stale samples of it.
        session.patch(gpu, &world, &[NAMED_CHECK]);
        let patched = after_change(gpu, &mut world, &mut session, restamped).against_records();
        let missed = patched
            .iter()
            .filter(|&&(reset, wanted)| wanted && !reset)
            .count();
        assert!(missed > 0, "the check bites");
    });
}

/// Records of one tile, as a stand-in to crowd pixels: the second to fourth names never fill.
const ONE_NAME: [(&str, &str); 3] = [
    ("} else if seen[1] == EMPTY {", "} else if false {"),
    ("} else if seen[2] == EMPTY {", "} else if false {"),
    ("} else if seen[3] == EMPTY {", "} else if false {"),
];

/// The crowded term of the reset rule, and its stand-in that drops it.
const CROWDED_CHECK: (&str, &str) = ("(crowded && latest[0] > since)", "false");

#[test]
fn a_crowded_pixel_resets_on_any_change() {
    GpuTest::run(|gpu| {
        let mut world = World::with_finest(gpu, 4);
        let mut session = Session::new(gpu, &world, EYE, size());
        // Records naming one tile crowd every pixel that sees two: seams and silhouettes.
        session.patch(gpu, &world, &ONE_NAME);
        let changed = after_change(gpu, &mut world, &mut session, restamped);
        let crowded_elsewhere = changed
            .before
            .iter()
            .filter(|r| r.overflow && !r.names(changed.tile))
            .count();
        eprintln!("{crowded_elsewhere} crowded pixels not naming the tile");
        assert!(crowded_elsewhere > 20, "crowded pixels away from the tile");
        let pixels = changed.against_records();
        assert!(
            pixels.iter().all(|&(reset, wanted)| reset == wanted),
            "the tile's pixels and every crowded one reset"
        );
        // The bite: without the crowded term, a crowded pixel keeps samples of tiles it no longer
        // names.
        let mut without_crowded = ONE_NAME.to_vec();
        without_crowded.push(CROWDED_CHECK);
        session.patch(gpu, &world, &without_crowded);
        let patched = after_change(gpu, &mut world, &mut session, restamped).against_records();
        let missed = patched
            .iter()
            .filter(|&&(reset, wanted)| wanted && !reset)
            .count();
        assert!(missed > 0, "the check bites");
    });
}

/// A reset rule comparing ids, as a stand-in for the bite: a pixel resets when its sample sees
/// another tile than the first it saw.
const ID_RULE: (&str, &str) = (
    "var reset = changed_after(node, since)",
    "var reset = (node != old[0] && node != EMPTY && old[0] != EMPTY)",
);

/// Pixels of `session` (converged) whose history started again over `frames` more still frames,
/// and pixels whose id changed between frames.
fn self_resets(gpu: &Gpu, world: &World, session: &mut Session, frames: u32) -> (usize, usize) {
    session.converge(gpu, world);
    let starts: Vec<u32> = session.records(gpu).iter().map(|r| r.start).collect();
    let first = session.ids(gpu);
    let mut moved = vec![false; first.len()];
    for _ in 0..frames {
        session.draw(gpu, world, false);
        let ids = session.ids(gpu);
        for ((flag, a), b) in moved.iter_mut().zip(&first).zip(&ids) {
            *flag |= a != b;
        }
    }
    let records = session.records(gpu);
    let reset = records
        .iter()
        .zip(&starts)
        .filter(|&(record, start)| record.start != *start)
        .count();
    (reset, moved.iter().filter(|&&m| m).count())
}

#[test]
fn the_jitter_alone_never_resets_a_pixel() {
    GpuTest::run(|gpu| {
        let world = World::with_finest(gpu, 4);
        let mut session = Session::new(gpu, &world, EYE, size());
        let (reset, moved) = self_resets(gpu, &world, &mut session, SAMPLES);
        eprintln!("{moved} pixels see other ids at other jitters; {reset} reset");
        assert!(
            moved > 50,
            "the jitter moves ids at edges and seams: {moved}"
        );
        assert_eq!(reset, 0, "no pixel started again");
        // The bite: a rule comparing ids resets pixels over and over.
        session.patch(gpu, &world, &[ID_RULE]);
        let repeated = self_resets_patched(gpu, &world, &mut session);
        assert!(repeated > 0, "the check bites: {repeated}");
    });
}

/// [`self_resets`] for a rule that may never converge: draws SAMPLES frames, then SAMPLES more,
/// and counts the pixels whose start moved over the second lot.
fn self_resets_patched(gpu: &Gpu, world: &World, session: &mut Session) -> usize {
    for _ in 0..SAMPLES {
        session.draw(gpu, world, false);
    }
    let starts: Vec<u32> = session.records(gpu).iter().map(|r| r.start).collect();
    for _ in 0..SAMPLES {
        session.draw(gpu, world, false);
    }
    let records = session.records(gpu);
    records
        .iter()
        .zip(&starts)
        .filter(|&(record, start)| record.start != *start)
        .count()
}

#[test]
fn the_filter_smooths_a_moving_frames_edges_and_leaves_flat_regions_alone() {
    GpuTest::run(|gpu| {
        let world = World::with_finest(gpu, 4);
        let reference: Vec<[f32; 3]> = supersampled(gpu, &world, EYE, size(), 8)
            .into_iter()
            .map(shown)
            .collect();
        let mut moving = Session::new(gpu, &world, EYE, size());
        moving.draw(gpu, &world, true);
        let filtered = moving.pixels(gpu);
        let mut still = Session::new(gpu, &world, EYE, size());
        still.draw(gpu, &world, false);
        let plain = still.unfiltered(gpu);
        let (filtered_error, plain_error) = (rmse(&filtered, &reference), rmse(&plain, &reference));
        eprintln!("filtered {filtered_error:.5}, unfiltered {plain_error:.5}");
        assert!(
            filtered_error < plain_error,
            "nearer the reference: {filtered_error} against {plain_error}"
        );
        let (width, height) = (size().width(), size().height());
        let at = |image: &[[u8; 4]], x: u32, y: u32| {
            image[usize::try_from(y * width + x).expect("small")]
        };
        let mut flat = 0;
        let mut changed = 0;
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let around: Vec<[u8; 4]> = (0..9)
                    .map(|k| at(&plain, x + k % 3 - 1, y + k / 3 - 1))
                    .collect();
                let spread = (0..3).all(|c| {
                    let lo = around.iter().map(|p| p[c]).min().unwrap_or(0);
                    let hi = around.iter().map(|p| p[c]).max().unwrap_or(0);
                    hi - lo <= 2
                });
                let (a, b) = (at(&filtered, x, y), at(&plain, x, y));
                let same = a.iter().zip(b).all(|(p, q)| p.abs_diff(q) <= 1);
                if spread {
                    flat += 1;
                    assert!(same, "a flat pixel ({x}, {y}) is unchanged");
                }
                changed += usize::from(!same);
            }
        }
        assert!(flat > 1000, "flat pixels checked: {flat}");
        assert!(changed > 20, "edges filtered: {changed}");
    });
}

#[test]
fn shading_again_in_the_same_look_keeps_the_image_and_a_new_look_restarts_it() {
    GpuTest::run(|gpu| {
        let world = World::with_finest(gpu, 4);
        let mut session = Session::new(gpu, &world, EYE, size());
        session.converge(gpu, &world);
        let (pixels, records) = (session.pixels(gpu), session.records(gpu));
        // A frame for the overlay alone (a pick line): nothing shaded, the image as it was.
        let same = session.reshade(gpu, LOOK);
        assert!(same.shown && !same.scene, "shown, not drawn: {same:?}");
        assert_eq!(session.pixels(gpu), pixels, "the accumulated image");
        assert_eq!(session.records(gpu), records, "no history touched");
        // A new look: the last sample shaded anew starts every history again.
        let glossy = Look {
            material: MatCapStyle::Glossy,
            ..LOOK
        };
        assert!(session.reshade(gpu, glossy).shown);
        let frame = session.passes().aa().frame();
        assert!(
            session.records(gpu).iter().all(|r| r.start == frame),
            "restarted"
        );
        assert_ne!(session.pixels(gpu), pixels, "in the new look");
    });
}

#[test]
fn an_image_too_large_to_accumulate_is_filtered_while_still_and_never_fails() {
    GpuTest::run(|gpu| {
        let world = World::with_finest(gpu, 4);
        let mut moving = Session::new(gpu, &world, EYE, size());
        moving.draw(gpu, &world, true);
        let filtered = moving.pixels(gpu);
        // A device binding less than this image's histories need (a 5K window on 128 MiB).
        let mut small = Session::new(gpu, &world, EYE, size());
        let mut passes = passes_for(gpu, &world);
        let pixels = u64::from(size().width() * size().height());
        passes.aa_mut().set_limit(pixels * 8 - 1);
        small.renderer.set_passes(passes);
        for _ in 0..3 {
            let drawn = small.draw(gpu, &world, false);
            assert!(drawn.shown, "still shown: {drawn:?}");
            assert!(
                drawn.aa.filtered_instead && !drawn.aa.restarted,
                "{drawn:?}"
            );
            assert_eq!(small.pixels(gpu), filtered, "filtered as a moving frame is");
        }
        let again = small.reshade(gpu, LOOK);
        assert!(
            again.shown && again.aa.filtered_instead,
            "shading again too: {again:?}"
        );
        assert!(small.passes().aa().record().is_none(), "no histories made");
    });
}

#[test]
fn the_first_still_frames_after_a_move_are_filtered_fading_as_samples_come() {
    GpuTest::run(|gpu| {
        let world = World::with_finest(gpu, 4);
        let mut session = Session::new(gpu, &world, EYE, size());
        session.draw(gpu, &world, true);
        let moving = session.pixels(gpu);
        let first = session.draw(gpu, &world, false);
        assert!(first.aa.restarted, "the camera stopped: {first:?}");
        let still = session.pixels(gpu);
        let differing = moving
            .iter()
            .zip(&still)
            .filter(|(a, b)| a.iter().zip(b.iter()).any(|(x, y)| x.abs_diff(*y) > 1))
            .count();
        assert_eq!(
            differing, 0,
            "no pop: the moving frame's filtered image again"
        );
        let mut filtered = vec![session.filtered_pixels(gpu)];
        for _ in 0..5 {
            session.draw(gpu, &world, false);
            filtered.push(session.filtered_pixels(gpu));
        }
        eprintln!("pixels filtered at 1 to 6 samples: {filtered:?}");
        assert!(filtered[0] > 20, "one sample is filtered at its edges");
        assert!(
            filtered.windows(2).take(4).all(|w| w[1] <= w[0]),
            "less and less: {filtered:?}"
        );
        assert_eq!(filtered[4], 0, "none from five samples: {filtered:?}");
    });
}
