//! The drawing thread's share of a recovery flush of a stroke that dirtied 50 level-8 correction
//! tiles, in its two parts:
//!
//! - [`RecoveryWriter::end_stroke`]: the snapshot (one 26.9 KB slot copy per tile) and the hand-off
//!   to the writer thread. Encoding, compression and syncing happen there, outside the measurement.
//! - [`RecoveryWriter::poll`] once the writer thread has encoded the batch: appending the 50 framed
//!   records to the working file (page-cache writes) and asking for the sync.
//!
//! The working file is under [`std::env::temp_dir`]. Where that is `/tmp` on tmpfs (as on the
//! dev box), syncs are free, so a sync reply can come back within the measured `poll`, which
//! then also writes the save point; on a disk, the sync takes longer and the measurement is the
//! appends alone.

#![expect(
    clippy::expect_used,
    reason = "benchmark setup; a failure here should abort the run"
)]

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use geom::{FaceId, Level, TileKey, Vec3, cube_sphere};
use store::{Meta, ProjectFile, ProjectId, RecoveryWriter, Role, StoreConfig, TileStore};

/// Tiles the stroke dirties.
const TILES: u16 = 50;

fn main() {
    divan::main();
}

/// A store over a fresh working file with [`TILES`] dirty level-8 tiles, and its recovery writer.
///
/// The writer is subscribed before the edits. The file's directory is removed on drop, after the
/// writer (which abandons its flush) and the store.
struct Stroke {
    /// The writer, dropped first: it joins its thread.
    writer: RecoveryWriter,
    /// The store, holding the dirty tiles.
    store: TileStore,
    /// The directory holding the working file.
    dir: PathBuf,
}

impl Stroke {
    /// Makes the store, the writer and the stroke.
    ///
    /// # Panics
    ///
    /// If the fixed setup is rejected, or the temporary directory can't be made.
    fn new() -> Self {
        /// Distinguishes this process's directories.
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "brushcraft-bench-recovery-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).expect("temporary directory");
        let mesh = Arc::new(cube_sphere(1).expect("cube"));
        let meta = Meta::new(
            ProjectId::from_bits(1),
            Role::Recovery { base: None },
            &mesh,
        );
        let file =
            ProjectFile::create(&dir.join("recovery.bcrf"), meta, Arc::clone(&mesh)).expect("file");
        let mut config = StoreConfig {
            budget_bytes: 0,
            chunk_bytes: 4 << 20,
            site: 1,
            threads: None,
        };
        config.budget_bytes = config.minimum_budget(&mesh).max(64 << 20);
        let mut store = TileStore::open(file, config).expect("store");
        let mut writer = RecoveryWriter::new(&mut store).expect("writer");
        // Flush the empty start, so the measured call is a stroke end like any other.
        writer.finish(&mut store, Instant::now()).expect("flush");
        let level = Level::new(8).expect("level");
        let per_side = u16::try_from(level.tiles_per_side()).expect("tiles per side");
        let keys = mesh.faces().flat_map(|face: FaceId| {
            (0..per_side).flat_map(move |ty| {
                (0..per_side)
                    .map(move |tx| TileKey::displacement(face, level, tx, ty).expect("key"))
            })
        });
        // One stroke: every tile is a dab of one op.
        store.begin_op().expect("op");
        for key in keys.take(usize::from(TILES)) {
            store
                .edit_tile(key, |edit| edit.set(5, 7, Vec3::new(1e-4, 0.0, 0.0)))
                .expect("edit")
                .expect("set");
        }
        Self { writer, store, dir }
    }
}

impl Drop for Stroke {
    fn drop(&mut self) {
        drop(std::fs::remove_dir_all(&self.dir));
    }
}

/// `end_stroke` with 50 dirty level-8 tiles: the snapshot and the hand-off only.
///
/// # Panics
///
/// If the setup or the call fails, which would be a bug in the crate.
#[divan::bench(sample_count = 200, sample_size = 1)]
fn end_stroke_50_tiles(bencher: divan::Bencher<'_, '_>) {
    bencher.with_inputs(Stroke::new).bench_local_refs(|stroke| {
        stroke
            .writer
            .end_stroke(&mut stroke.store, Instant::now())
            .expect("end stroke");
    });
}

/// How long the `poll` bench gives the writer thread to encode the batch before measuring:
/// far longer than 50 tiles take (a few milliseconds).
const ENCODE_WAIT: std::time::Duration = std::time::Duration::from_millis(100);

/// `poll` after the writer thread has encoded the stroke's 50 tiles: appending their records
/// to the working file and asking for the sync.
///
/// # Panics
///
/// If the setup or the call fails, which would be a bug in the crate.
#[divan::bench(sample_count = 50, sample_size = 1)]
fn poll_appends_50_records(bencher: divan::Bencher<'_, '_>) {
    bencher
        .with_inputs(|| {
            let mut stroke = Stroke::new();
            stroke
                .writer
                .end_stroke(&mut stroke.store, Instant::now())
                .expect("end stroke");
            std::thread::sleep(ENCODE_WAIT);
            stroke
        })
        .bench_local_refs(|stroke| {
            stroke.writer.poll(&mut stroke.store).expect("poll");
        });
}
