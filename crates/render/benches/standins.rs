//! A long stroke seen from a distance: the owner thread's cost of one brush step's frame (the
//! step's epoch applied, the frame staged with its stand-ins), early in the stroke and late.
//!
//! `cargo bench -p render --bench standins`. The project is the 96-face cube sphere to level 8
//! (1,536 finest tiles), every display tile up to level 7 resident (the floor taken to level 7),
//! no finest tile: the stroke is seen from a distance. Each step names about a dozen finest tiles
//! along a path, three new ones a step, and the display tiles above them (the store's bump), so
//! [`STEPS`] steps edit about 900 finest tiles. Each sample is one step's frame, after `done`
//! steps of the same stroke: its epoch applied, staged, submitted and waited for, so the GPU's
//! part is in it too, the same at every `done`. What should stay flat as `done` grows is the
//! owner thread's part, which once grew with the stroke. The GPU is the default adapter, or the
//! first whose name contains `BRUSHCRAFT_ADAPTER`.

use std::collections::BTreeSet;
use std::convert::Infallible;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use geom::{BaseMesh, Channel, FinestNormalCone, Level, TileKey, cube_sphere};
use render::gpu::{AdapterRequest, Gpu};
use render::residency::{FrameUse, Residency, ResidencyConfig};
use render::tree::Place;
use render::upload::{TileData, TileDetail, TileSource};
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

/// One step's frame: `epoch` applied, staged, submitted and waited for.
///
/// # Errors
///
/// If residency or the GPU fails.
fn frame(
    gpu: &Gpu,
    residency: &mut Residency,
    epoch: Epoch,
) -> Result<(), Box<dyn std::error::Error>> {
    residency.begin_frame(gpu)?;
    residency.apply(Drained::Epochs(vec![epoch]));
    residency.use_frame(FrameUse::default());
    let uploads = residency.stage(gpu, &mut Zeros, Instant::now() + Duration::from_secs(60))?;
    let encoder = gpu
        .device()
        .create_command_encoder(&CommandEncoderDescriptor::default());
    uploads.submit(gpu.queue(), encoder, |_| {});
    gpu.device().poll(PollType::wait_indefinitely())?;
    Ok(())
}

/// Residency seen from a distance, its floor (every display tile to level 7) up, and the
/// stroke's path.
///
/// # Errors
///
/// If the mesh, the configuration, residency or the GPU fails.
fn far() -> Result<(Residency, Vec<TileKey>), Box<dyn std::error::Error>> {
    let Some(gpu) = GPU.get() else {
        return Err("no GPU".into());
    };
    let mesh = cube_sphere(4)?;
    let config = ResidencyConfig::new(1 << 30, 16 << 20, 64 << 20, 8 << 20, 64, level(7))?;
    let mut residency = Residency::new(gpu, &mesh, level(FINEST), config)?;
    residency.begin_frame(gpu)?;
    let uploads = residency.stage(gpu, &mut Zeros, Instant::now() + Duration::from_secs(60))?;
    let encoder = gpu
        .device()
        .create_command_encoder(&CommandEncoderDescriptor::default());
    uploads.submit(gpu.queue(), encoder, |_| {});
    gpu.device().poll(PollType::wait_indefinitely())?;
    Ok((residency, finest_tiles(&mesh)))
}

/// One step's frame after `done` steps of the stroke.
#[divan::bench(args = DONE, sample_count = 40, sample_size = 1)]
fn step_frame(bencher: divan::Bencher<'_, '_>, done: usize) {
    let Some(gpu) = GPU.get() else { return };
    let Ok((mut residency, path)) = far() else {
        return;
    };
    for n in 0..done {
        if frame(gpu, &mut residency, step(&path, n)).is_err() {
            return;
        }
    }
    let mut n = done;
    bencher.bench_local(|| {
        n += 1;
        drop(frame(gpu, &mut residency, step(&path, n)));
    });
}
