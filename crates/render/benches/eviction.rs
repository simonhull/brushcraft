//! Eviction under a full budget: one frame that brings in [`ARRIVALS`] side-65 tiles, each
//! evicting the least recently drawn tile that may leave, with the [`PROTECTED`] least recently
//! drawn tiles in the cut, so every eviction's scan steps past them first.
//!
//! `cargo bench -p render --bench eviction`. The frame is staged and submitted, and waited for,
//! so each sample is a whole frame's CPU work: the difference between the `protected` sizes is
//! what the scan costs. The GPU is the default adapter, or the first whose name contains
//! `BRUSHCRAFT_ADAPTER`.

use std::collections::BTreeSet;
use std::convert::Infallible;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use geom::{BaseMesh, Level, TileKey, cube_sphere};
use render::gpu::{AdapterRequest, Gpu};
use render::residency::{FrameUse, Request, Requests, Residency, ResidencyConfig, SizeClass};
use render::upload::{GpuVertex, TileSource};
use store::Drained;
use wgpu::{CommandEncoderDescriptor, PollType};

/// The project's finest level: 2,016 side-65 tiles on the 96-face cube sphere.
const FINEST: u8 = 8;

/// Side-65 tiles the budget holds beyond the floor.
const RESIDENT: u64 = 1024;

/// Tiles each frame brings in, each evicting one.
const ARRIVALS: usize = 16;

/// Cut sizes: the least recently drawn tiles that every eviction's scan steps past.
const PROTECTED: [usize; 4] = [0, 256, 512, 960];

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

/// A source that has every tile, all zeros.
struct Zeros;

impl TileSource for Zeros {
    type Error = Infallible;

    fn read(&mut self, _key: TileKey, _out: &mut [GpuVertex]) -> Result<Option<f32>, Infallible> {
        Ok(Some(0.0))
    }
}

/// Level `l`.
fn level(l: u8) -> Level {
    Level::new(l).unwrap_or(Level::ZERO)
}

/// Every finest tile of `mesh`.
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

/// Runs one frame: begins it, takes `used`, wants `wanted`, stages, submits and waits.
///
/// # Errors
///
/// If residency or the GPU fails.
fn frame(
    gpu: &Gpu,
    residency: &mut Residency,
    used: FrameUse,
    wanted: &[TileKey],
) -> Result<(), Box<dyn std::error::Error>> {
    residency.begin_frame(gpu)?;
    residency.apply(Drained::Epochs(Vec::new()));
    residency.use_frame(used);
    let requests = wanted.iter().map(|&key| Request {
        key,
        screen_error: 1.0,
    });
    residency.want(&Requests::new(requests, 0));
    let uploads = residency.stage(gpu, &mut Zeros, Instant::now() + Duration::from_secs(60))?;
    let mut encoder = gpu
        .device()
        .create_command_encoder(&CommandEncoderDescriptor::default());
    uploads.encode(&mut encoder);
    uploads.submit(gpu.queue(), encoder);
    gpu.device().poll(PollType::wait_indefinitely())?;
    Ok(())
}

/// A residency holding [`RESIDENT`] finest tiles under a full budget, the oldest `protected`
/// of them in the cut, and the tiles not resident, to request next.
///
/// # Errors
///
/// If the mesh, the configuration, residency or the GPU fails.
fn full(
    gpu: &Gpu,
    protected: usize,
) -> Result<(Residency, Vec<TileKey>), Box<dyn std::error::Error>> {
    let mesh = cube_sphere(4)?;
    let big = SizeClass::of(level(FINEST)).slot_bytes();
    // The floor's classes, twice over (their shadow headroom), reserved first.
    let faces = u64::from(mesh.face_count());
    let floor_classes = (0..6).filter_map(SizeClass::new);
    let floor: u64 = floor_classes
        .map(|class| 2 * class.slot_bytes() * faces)
        .sum();
    let config = ResidencyConfig::default()
        .with_budget(floor + RESIDENT * big)?
        .with_buffer_bytes(64 * big)?;
    let mut residency = Residency::new(gpu, &mesh, level(FINEST), config)?;
    let tiles = finest_tiles(&mesh);
    let (resident, rest) = tiles.split_at(usize::try_from(RESIDENT).unwrap_or(0));
    // Brought in a batch at a time, in order, so the first are the least recently drawn.
    let mut drawn = BTreeSet::new();
    for batch in resident.chunks(64) {
        for _ in 0..3 {
            frame(
                gpu,
                &mut residency,
                FrameUse::new(BTreeSet::new(), drawn.clone()),
                batch,
            )?;
        }
        drawn = batch.iter().copied().collect();
    }
    let cut: BTreeSet<TileKey> = resident.iter().take(protected).copied().collect();
    frame(
        gpu,
        &mut residency,
        FrameUse::new(cut, BTreeSet::new()),
        &[],
    )?;
    Ok((residency, rest.to_vec()))
}

/// One frame bringing in [`ARRIVALS`] tiles under a full budget, the `protected` least recently
/// drawn tiles in the cut.
#[divan::bench(args = PROTECTED, sample_count = 50, sample_size = 1)]
fn arrivals_evicting(bencher: divan::Bencher<'_, '_>, protected: usize) {
    let Some(gpu) = GPU.get() else { return };
    let Ok((mut residency, mut waiting)) = full(gpu, protected) else {
        return;
    };
    let resident: Vec<TileKey> = residency.resident().copied().collect();
    let cut: BTreeSet<TileKey> = resident
        .iter()
        .filter(|key| key.level() == level(FINEST))
        .take(protected)
        .copied()
        .collect();
    bencher.bench_local(|| {
        let batch: Vec<TileKey> = waiting.drain(..ARRIVALS.min(waiting.len())).collect();
        // Wanted a frame ahead, as the readback gives them, then staged.
        let used = FrameUse::new(cut.clone(), BTreeSet::new());
        drop(frame(gpu, &mut residency, used.clone(), &batch));
        drop(frame(gpu, &mut residency, used, &batch));
        let gone = residency.stats().evictions;
        divan::black_box(gone);
        // The evicted tiles go to the back of the line, to come again.
        let now: BTreeSet<TileKey> = residency.resident().copied().collect();
        let evicted = resident.iter().filter(|key| !now.contains(key)).copied();
        let back: Vec<TileKey> = evicted
            .filter(|key| !waiting.contains(key))
            .take(ARRIVALS)
            .collect();
        waiting.extend(back);
    });
}
