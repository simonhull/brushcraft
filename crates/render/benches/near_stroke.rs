//! A strong stroke seen up close: the owner thread's cost of one brush step's frame when the
//! step's finest tiles are resident, so the frame reads each of them again (its positions with
//! the ring, which the normal pass turns into its vertices) and the display tiles above them
//! stand in, reading the same finest positions.
//!
//! `cargo bench -p render --bench near_stroke`. The project is the 96-face cube sphere to level 8
//! (1,536 finest tiles), every display tile up to level 7 resident (the floor taken to level 7),
//! and [`PATH`] finest tiles along the stroke's path resident too. Each step names [`PER_STEP`]
//! finest tiles along the path (a strong dab's reach on the benchmark project), [`ADVANCE`] new
//! ones a step, and the display tiles above them, as the store's epochs do. The source gives
//! every finest tile as a ring gathered once beforehand, so a sample times residency's own work:
//! the epoch applied, each finest tile's ring record written and staged, the positions kept for
//! the stand-ins, the stand-ins' scratch staged and their jobs built. The frame is submitted and
//! waited for between samples, untimed. The GPU is the default adapter, or the first whose name
//! contains `BRUSHCRAFT_ADAPTER`; any failure panics, so a broken run never passes for a fast
//! one.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::convert::Infallible;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use geom::{
    BaseMesh, Channel, FaceId, FinestNormalCone, GatheredBlock, Level, PositionSource, Region,
    TileKey, Vec3, cube_sphere, gather_block,
};
use render::gpu::{AdapterRequest, Gpu};
use render::residency::{FrameUse, Request, Requests, Residency, ResidencyConfig};
use render::tree::Place;
use render::upload::{FrameUploads, TileData, TileDetail, TileSource};
use store::{Drained, Epoch, EpochKind};
use wgpu::{CommandEncoderDescriptor, PollType};

/// The project's finest level.
const FINEST: u8 = 8;

/// Finest tiles along the stroke's path, all resident.
const PATH: usize = 160;

/// Finest tiles each step names: a strong dab's reach on the benchmark project.
const PER_STEP: usize = 25;

/// New finest tiles each step reaches.
const ADVANCE: usize = 2;

/// Steps done before the samples: early in the stroke and late.
const DONE: [usize; 2] = [0, 60];

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

/// Made-up positions for every face-grid vertex: distinct, finite, and the same whichever copy
/// of a vertex is read.
struct Made;

impl PositionSource for Made {
    #[expect(
        clippy::cast_precision_loss,
        reason = "made-up positions: any finite value does"
    )]
    fn position(&self, face: FaceId, level: Level, i: u32, j: u32) -> Option<Vec3> {
        let scale = 1.0 / level.face_quads().max(1) as f32;
        Some(Vec3::new(
            i as f32 * scale,
            j as f32 * scale,
            face.index() as f32,
        ))
    }
}

/// A source with every tile: display tiles all zeros, with a small bound; finest tiles as
/// their ring, gathered once beforehand.
struct Rings<'a> {
    /// The mesh the rings were gathered over.
    mesh: &'a BaseMesh,
    /// Each finest tile's ring.
    blocks: &'a BTreeMap<TileKey, GatheredBlock>,
}

impl TileSource for Rings<'_> {
    type Error = Infallible;

    fn read(&mut self, key: TileKey, out: &mut TileData) -> Result<Option<TileDetail>, Infallible> {
        if key.channel() != Channel::DISPLAY {
            if let Some(block) = self.blocks.get(&key) {
                out.ring(self.mesh, block);
            }
            return Ok(Some(TileDetail::FINEST));
        }
        Ok(Some(TileDetail {
            error: 1e-3,
            cone: FinestNormalCone::EVERYTHING,
            stored: None,
            held: false,
        }))
    }
}

/// Level `l`.
fn level(l: u8) -> Level {
    Level::new(l).unwrap_or(Level::ZERO)
}

/// The stroke's path: [`PATH`] finest tiles of `mesh`, face by face.
fn path(mesh: &BaseMesh) -> Vec<TileKey> {
    let finest = level(FINEST);
    let per_side = u16::try_from(finest.tiles_per_side()).unwrap_or(0);
    let all = mesh.faces().flat_map(|face| {
        (0..per_side).flat_map(move |ty| {
            (0..per_side).filter_map(move |tx| TileKey::displacement(face, finest, tx, ty).ok())
        })
    });
    all.take(PATH).collect()
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

/// The project: its mesh, its path, and each path tile's ring.
struct Project {
    /// The mesh.
    mesh: BaseMesh,
    /// The stroke's path.
    path: Vec<TileKey>,
    /// Each path tile's ring.
    blocks: BTreeMap<TileKey, GatheredBlock>,
}

/// The project, made once.
///
/// # Panics
///
/// If the mesh or a ring can't be made.
fn project() -> &'static Project {
    static PROJECT: OnceLock<Project> = OnceLock::new();
    PROJECT.get_or_init(|| {
        let mesh = cube_sphere(4).or_die("a mesh");
        let path = path(&mesh);
        let blocks = path
            .iter()
            .map(|&key| {
                let block = gather_block(&mesh, &Made, Region::of_tile(&key)).or_die("a ring");
                (key, block)
            })
            .collect();
        Project { mesh, path, blocks }
    })
}

/// One frame of residency with `source`: begun, `drained` applied, staged, submitted.
///
/// # Panics
///
/// If residency or the GPU fails.
fn stage(
    gpu: &Gpu,
    residency: &mut Residency,
    source: &mut Rings<'_>,
    drained: Drained,
) -> FrameUploads {
    residency.begin_frame(gpu).or_die("a frame begins");
    residency.apply(drained);
    residency.use_frame(FrameUse::default());
    residency
        .stage(gpu, source, Instant::now() + Duration::from_secs(60))
        .or_die("staged")
}

/// Residency seen up close: its floor (every display tile to level 7) and the path's finest
/// tiles resident.
///
/// # Panics
///
/// If the configuration, residency or the GPU fails, or the path's tiles don't arrive.
fn near(gpu: &Gpu, source: &mut Rings<'_>) -> Residency {
    let project = project();
    let config = ResidencyConfig::new(1 << 30, 16 << 20, 64 << 20, 8 << 20, 64, level(7))
        .or_die("a configuration");
    let mut residency =
        Residency::new(gpu, &project.mesh, level(FINEST), config).or_die("residency");
    let requests = project.path.iter().map(|&key| Request {
        key,
        screen_error: 1.0,
    });
    residency.want(&Requests::new(requests, 0));
    for _ in 0..20 {
        let uploads = stage(gpu, &mut residency, source, Drained::Epochs(Vec::new()));
        submit(gpu, uploads);
        if project.path.iter().all(|key| residency.slot(key).is_some()) {
            return residency;
        }
    }
    None.or_die("the path's finest tiles resident")
}

/// The owner thread's part of one step's frame after `done` steps of the stroke.
///
/// # Panics
///
/// If the GPU or residency fails.
#[divan::bench(args = DONE, sample_count = 100, sample_size = 1)]
fn step_frame(bencher: divan::Bencher<'_, '_>, done: usize) {
    let gpu = GPU.get().or_die("the GPU, opened by main");
    let project = project();
    let mut source = Rings {
        mesh: &project.mesh,
        blocks: &project.blocks,
    };
    let mut residency = near(gpu, &mut source);
    for n in 0..done {
        let epoch = Drained::Epochs(vec![step(&project.path, n)]);
        let uploads = stage(gpu, &mut residency, &mut source, epoch);
        submit(gpu, uploads);
    }
    let state = RefCell::new((residency, source, done, None::<FrameUploads>));
    bencher
        .with_inputs(|| {
            // Untimed: the last sample's frame submitted, and the next step's epoch.
            let mut state = state.borrow_mut();
            if let Some(uploads) = state.3.take() {
                submit(gpu, uploads);
            }
            state.2 += 1;
            step(&project.path, state.2)
        })
        .bench_local_values(|epoch| {
            let mut state = state.borrow_mut();
            let (staged, read, _, _) = &mut *state;
            let uploads = stage(gpu, staged, read, Drained::Epochs(vec![epoch]));
            state.3 = Some(uploads);
        });
    if let Some(uploads) = state.borrow_mut().3.take() {
        submit(gpu, uploads);
    }
}
