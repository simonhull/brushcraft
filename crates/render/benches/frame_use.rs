//! A frame's use on the owner thread: [`Residency::use_frame`] taking a cut of thousands of tiles
//! that changes every frame, or stays, as the app's frames do while the view moves or holds.
//!
//! `cargo bench -p render --bench frame_use`. The project is the 216-face cube sphere to level 9:
//! the floor (every display tile to level 6) and every display tile of levels 7 and 8 resident,
//! about 5,800 tiles. The cut is either every level-8 tile of the sphere (3,456) or, every other
//! frame, half the faces a level coarser (2,160); about three in five of its tiles are drawn.
//! Each sample is one call, its frame begun untimed before it. The GPU is the default adapter,
//! or the first whose name contains `BRUSHCRAFT_ADAPTER`; any failure panics, so a broken run
//! never passes for a fast one.

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::convert::Infallible;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use geom::{BaseMesh, FinestNormalCone, Level, TileKey, cube_sphere};
use render::gpu::{AdapterRequest, Gpu};
use render::residency::{FrameUse, Request, Requests, Residency, ResidencyConfig};
use render::upload::{FrameUploads, TileData, TileDetail, TileSource};
use store::Drained;
use wgpu::{CommandEncoderDescriptor, PollType};

/// The project's finest level: levels 7 and 8 are display tiles.
const FINEST: u8 = 9;

/// The floor's top level: below it, a tile is a whole face.
const FLOOR_TOP: u8 = 6;

/// The most frames the setup stages to bring every tile in.
const SETUP_FRAMES: usize = 400;

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

/// A source that has every display tile, all zeros, with a small bound.
struct Zeros;

impl TileSource for Zeros {
    type Error = Infallible;

    fn read(
        &mut self,
        _key: TileKey,
        _out: &mut TileData,
    ) -> Result<Option<TileDetail>, Infallible> {
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

/// Every display tile of `mesh` at level `l`, face by face.
fn display_tiles(mesh: &BaseMesh, l: u8) -> Vec<TileKey> {
    let at = level(l);
    let per_side = u16::try_from(at.tiles_per_side()).unwrap_or(0);
    let all = mesh.faces().flat_map(|face| {
        (0..per_side).flat_map(move |ty| {
            (0..per_side).filter_map(move |tx| TileKey::display(face, at, tx, ty).ok())
        })
    });
    all.collect()
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

/// Residency with the floor and every display tile of levels 7 and 8 resident, and the two
/// cuts: every level-8 tile, and half the faces at level 7 instead.
///
/// # Panics
///
/// If the mesh, the configuration, residency or the GPU fails, or the tiles don't all arrive.
fn resident(gpu: &Gpu) -> (Residency, [BTreeSet<TileKey>; 2]) {
    let mesh = cube_sphere(6).or_die("a mesh");
    let config = ResidencyConfig::new(2 << 30, 16 << 20, 64 << 20, 8 << 20, 64, level(FLOOR_TOP))
        .or_die("a configuration");
    let mut residency = Residency::new(gpu, &mesh, level(FINEST), config).or_die("residency");
    let coarse = display_tiles(&mesh, 7);
    let fine = display_tiles(&mesh, 8);
    let wanted = coarse.iter().chain(&fine).map(|&key| Request {
        key,
        screen_error: 1.0,
    });
    let requests = Requests::new(wanted, 0);
    let target = residency.floor().len() + coarse.len() + fine.len();
    for _ in 0..SETUP_FRAMES {
        if residency.stats().resident >= target {
            break;
        }
        residency.begin_frame(gpu).or_die("a frame begins");
        residency.apply(Drained::Epochs(Vec::new()));
        residency.want(&requests);
        let uploads = residency
            .stage(gpu, &mut Zeros, Instant::now() + Duration::from_secs(60))
            .or_die("staged");
        submit(gpu, uploads);
    }
    assert!(
        residency.stats().resident >= target,
        "every tile arrives: {} of {target}",
        residency.stats().resident
    );
    let all_fine: BTreeSet<TileKey> = fine.iter().copied().collect();
    // Even faces a level coarser: their level-7 tiles in place of their level-8 ones.
    let even = |key: &&TileKey| key.face().index().is_multiple_of(2);
    let odd_fine = fine.iter().filter(|key| !even(key));
    let even_coarse = coarse.iter().filter(even);
    let mixed: BTreeSet<TileKey> = odd_fine.chain(even_coarse).copied().collect();
    (residency, [all_fine, mixed])
}

/// A frame's use of `cut`: about three in five of its tiles drawn, the rest culled.
fn frame_use(cut: &BTreeSet<TileKey>) -> FrameUse {
    let drawn = cut
        .iter()
        .filter(|key| (u32::from(key.tx()) + u32::from(key.ty()) + key.face().index()) % 5 < 3);
    let drawn: BTreeSet<TileKey> = drawn.copied().collect();
    FrameUse::new(cut.clone(), drawn)
}

/// One [`Residency::use_frame`] of a cut of thousands of tiles: the cut changing every frame
/// (`changing`), or the same each frame.
///
/// # Panics
///
/// If the GPU or residency fails.
#[divan::bench(args = [true, false], sample_count = 200, sample_size = 1)]
fn use_frame(bencher: divan::Bencher<'_, '_>, changing: bool) {
    let gpu = GPU.get().or_die("the GPU, opened by main");
    let (residency, cuts) = resident(gpu);
    let uses = cuts.each_ref().map(frame_use);
    let state = RefCell::new((residency, 0_usize));
    bencher
        .with_inputs(|| {
            // Untimed: the frame begun, and its use made.
            let mut state = state.borrow_mut();
            state.0.begin_frame(gpu).or_die("a frame begins");
            state.1 += usize::from(changing);
            uses.get(state.1 % 2).cloned().unwrap_or_default()
        })
        .bench_local_values(|used| state.borrow_mut().0.use_frame(used));
}
