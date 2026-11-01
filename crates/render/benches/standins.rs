//! A long stroke seen from a distance: the owner thread's cost of one brush step's frame (the
//! step's epoch applied, the frame staged with its stand-ins), early in the stroke and late, and
//! the GPU's (the frame submitted and waited for).
//!
//! `cargo bench -p render --bench standins`. The project is the 96-face cube sphere to level 8
//! (1,536 finest tiles), every display tile up to level 7 resident (the floor taken to level 7),
//! no finest tile: the stroke is seen from a distance. Each step names about a dozen finest tiles
//! along a path, three new ones a step, and the display tiles above them (the store's bump), so
//! [`STEPS`] steps edit about 900 finest tiles. Each sample is the owner thread's part of one
//! step's frame, after `done` steps of the same stroke: its epoch applied and the frame staged.
//! The frame is submitted and waited for between samples, untimed. It should stay flat as `done`
//! grows (it once grew with the stroke). `step_gpu` times the other half: each sample is the
//! frame's uploads submitted and waited for (copies, the stand-in pass's replace, normals and
//! settle, and the wait's own round trip), the frame staged untimed. The GPU is the default
//! adapter, or the first whose
//! name contains `BRUSHCRAFT_ADAPTER`; any failure panics, so a broken run never passes for a
//! fast one.

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::convert::Infallible;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use geom::{BaseMesh, Channel, FinestNormalCone, Level, TileKey, cube_sphere};
use render::gpu::{AdapterRequest, Gpu};
use render::residency::{FrameUse, Residency, ResidencyConfig};
use render::tree::Place;
use render::upload::{FrameUploads, TileData, TileDetail, TileSource};
use store::{Drained, Epoch, EpochKind};
use wgpu::{CommandEncoderDescriptor, PollType};

/// The project's finest level.
const FINEST: u8 = 8;

/// Steps in the stroke.
const STEPS: usize = 300;

/// Finest tiles each step names.
const PER_STEP: usize = 12;

/// New finest tiles each step reaches.
const ADVANCE: usize = 3;

/// Steps done before the samples: early in the stroke and late.
const DONE: [usize; 3] = [0, STEPS / 2, STEPS - 50];

/// Ends the run on a failure, saying what failed: a broken run never passes for a fast one.
trait OrDie<T> {
    /// The value, or a panic naming `what`.
    ///
    /// # Panics
    ///
    /// If there is no value.
    fn or_die(self, what: &str) -> T;
}

impl<T, E: std::fmt::Display> OrDie<T> for Result<T, E> {
    #[expect(
        clippy::panic,
        reason = "a bench ends on a failure: a broken run never passes for a fast one"
    )]
    fn or_die(self, what: &str) -> T {
        match self {
            Ok(value) => value,
            Err(error) => panic!("{what}: {error}"),
        }
    }
}

impl<T> OrDie<T> for Option<T> {
    #[expect(
        clippy::panic,
        reason = "a bench ends on a failure: a broken run never passes for a fast one"
    )]
    fn or_die(self, what: &str) -> T {
        match self {
            Some(value) => value,
            None => panic!("{what}"),
        }
    }
}

/// The GPU every bench runs on, opened once by `main`.
static GPU: OnceLock<Gpu> = OnceLock::new();

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let request =
        std::env::var("BRUSHCRAFT_ADAPTER").map_or(AdapterRequest::Default, AdapterRequest::Named);
    let opened = Gpu::open(&request)?;
    GPU.get_or_init(move || opened);
    divan::main();
    Ok(())
}

/// A source that has every tile, all zeros: a display tile with a small bound.
struct Zeros;

impl TileSource for Zeros {
    type Error = Infallible;

    fn read(
        &mut self,
        key: TileKey,
        _out: &mut TileData,
    ) -> Result<Option<TileDetail>, Infallible> {
        if key.channel() != Channel::DISPLAY {
            return Ok(Some(TileDetail::FINEST));
        }
        Ok(Some(TileDetail {
            error: 1e-3,
            cone: FinestNormalCone::EVERYTHING,
            stored: None,
        }))
    }
}

/// Level `l`.
fn level(l: u8) -> Level {
    Level::new(l).unwrap_or(Level::ZERO)
}

/// Every finest tile of `mesh`, face by face.
fn finest_tiles(mesh: &BaseMesh) -> Vec<TileKey> {
    let finest = level(FINEST);
    let per_side = u16::try_from(finest.tiles_per_side()).unwrap_or(0);
    let all = mesh.faces().flat_map(|face| {
        (0..per_side).flat_map(move |ty| {
            (0..per_side).filter_map(move |tx| TileKey::displacement(face, finest, tx, ty).ok())
        })
    });
    all.collect()
}

/// Step `n`'s epoch: [`PER_STEP`] finest tiles from `n × ADVANCE` along `path`, and the display
/// tiles above them.
fn step(path: &[TileKey], n: usize) -> Epoch {
    let start = (n * ADVANCE) % path.len().max(1);
    let finest = path.iter().cycle().skip(start).take(PER_STEP);
    let mut keys: BTreeSet<TileKey> = finest.copied().collect();
    let above: Vec<TileKey> = keys
        .iter()
        .flat_map(|key| {
            let mut above = Vec::new();
            let mut at = Place::of(key).parent();
            while let Some(place) = at {
                above.extend(place.key(level(FINEST)));
                at = place.parent();
            }
            above
        })
        .collect();
    keys.extend(above);
    Epoch {
        seq: u64::try_from(n).unwrap_or(u64::MAX),
        kind: EpochKind::Step { displacement: 1e-4 },
        keys,
    }
}

/// The owner thread's part of one step's frame: `epoch` applied, the frame staged.
///
/// # Panics
///
/// If residency fails.
fn stage(gpu: &Gpu, residency: &mut Residency, epoch: Epoch) -> FrameUploads {
    residency.begin_frame(gpu).or_die("a frame begins");
    residency.apply(Drained::Epochs(vec![epoch]));
    residency.use_frame(FrameUse::default());
    residency
        .stage(gpu, &mut Zeros, Instant::now() + Duration::from_secs(60))
        .or_die("staged")
}

/// Submits a frame's `uploads` and waits for the GPU.
///
/// # Panics
///
/// If the GPU fails.
fn submit(gpu: &Gpu, uploads: FrameUploads) {
    let encoder = gpu
        .device()
        .create_command_encoder(&CommandEncoderDescriptor::default());
    uploads.submit(gpu.queue(), encoder, |_| {});
    gpu.device()
        .poll(PollType::wait_indefinitely())
        .or_die("the GPU finishes");
}

/// Residency seen from a distance, its floor (every display tile to level 7) up, and the
/// stroke's path.
///
/// # Panics
///
/// If the mesh, the configuration, residency or the GPU fails.
fn far(gpu: &Gpu) -> (Residency, Vec<TileKey>) {
    let mesh = cube_sphere(4).or_die("a mesh");
    let config = ResidencyConfig::new(1 << 30, 16 << 20, 64 << 20, 8 << 20, 64, level(7))
        .or_die("a configuration");
    let mut residency = Residency::new(gpu, &mesh, level(FINEST), config).or_die("residency");
    residency.begin_frame(gpu).or_die("a frame begins");
    let uploads = residency
        .stage(gpu, &mut Zeros, Instant::now() + Duration::from_secs(60))
        .or_die("staged");
    submit(gpu, uploads);
    (residency, finest_tiles(&mesh))
}

/// The owner thread's part of one step's frame after `done` steps of the stroke.
///
/// # Panics
///
/// If the GPU or residency fails.
#[divan::bench(args = DONE, sample_count = 40, sample_size = 1)]
fn step_frame(bencher: divan::Bencher<'_, '_>, done: usize) {
    let gpu = GPU.get().or_die("the GPU, opened by main");
    let (mut residency, path) = far(gpu);
    for n in 0..done {
        let uploads = stage(gpu, &mut residency, step(&path, n));
        submit(gpu, uploads);
    }
    let state = RefCell::new((residency, done, None::<FrameUploads>));
    bencher
        .with_inputs(|| {
            // Untimed: the last sample's frame submitted, and the next step's epoch.
            let mut state = state.borrow_mut();
            if let Some(uploads) = state.2.take() {
                submit(gpu, uploads);
            }
            state.1 += 1;
            step(&path, state.1)
        })
        .bench_local_values(|epoch| {
            let mut state = state.borrow_mut();
            let uploads = stage(gpu, &mut state.0, epoch);
            state.2 = Some(uploads);
        });
    if let Some(uploads) = state.borrow_mut().2.take() {
        submit(gpu, uploads);
    }
}

/// The GPU's part of one step's frame after `done` steps of the stroke: the frame's uploads
/// submitted and waited for (their copies, the stand-in pass's replace, normals and settle),
/// staged untimed.
///
/// # Panics
///
/// If the GPU or residency fails.
#[divan::bench(args = DONE, sample_count = 40, sample_size = 1)]
fn step_gpu(bencher: divan::Bencher<'_, '_>, done: usize) {
    let gpu = GPU.get().or_die("the GPU, opened by main");
    let (mut residency, path) = far(gpu);
    for n in 0..done {
        let uploads = stage(gpu, &mut residency, step(&path, n));
        submit(gpu, uploads);
    }
    let state = RefCell::new((residency, done));
    bencher
        .with_inputs(|| {
            // Untimed: the next step's frame staged.
            let mut state = state.borrow_mut();
            state.1 += 1;
            let epoch = step(&path, state.1);
            stage(gpu, &mut state.0, epoch)
        })
        .bench_local_values(|uploads| submit(gpu, uploads));
}
