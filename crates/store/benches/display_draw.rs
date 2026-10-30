//! [`TileStore::display_for_draw`] of `n` level-5 floor tiles (33² vertices each, in memory
//! since the open), either side of the size at which it decodes them on the workers instead of
//! the calling thread (64): whether that size is where the hand-off starts paying.
//!
//! The project is 216 faces (`cube_sphere(6)`) keeping its display layer under level 7, built,
//! saved, and opened again, so the floor is loaded from the file as on a real open. The file is
//! under [`std::env::temp_dir`], unlinked once open.

#![expect(
    clippy::expect_used,
    reason = "benchmark setup; a failure here should abort the run"
)]

use std::cell::RefCell;
use std::sync::Arc;
use std::time::{Duration, Instant};

use geom::{Level, TileKey, cube_sphere};
use store::{Meta, ProjectFile, ProjectId, Role, StoreConfig, TileStore};

fn main() {
    divan::main();
}

/// The opened store and its level-5 floor tiles.
struct Opened {
    /// The store.
    store: TileStore,
    /// Every level-5 display tile, one a face.
    keys: Vec<TileKey>,
}

thread_local! {
    /// The store the benches share, built once.
    static OPENED: RefCell<Option<Opened>> = const { RefCell::new(None) };
}

/// Builds the project over `mesh` at `path`, its display layer under level 7, and saves it;
/// returns the store's configuration.
///
/// # Panics
///
/// If the fixed setup is rejected.
fn build(path: &std::path::Path, mesh: &Arc<geom::BaseMesh>) -> StoreConfig {
    let finest = Level::new(7).expect("level");
    let config = StoreConfig::new(512 << 20, 1).with_display(finest);
    let meta = Meta::new(ProjectId::from_bits(1), Role::Project, mesh);
    let file = ProjectFile::create(path, meta, Arc::clone(mesh)).expect("file");
    let mut store = TileStore::open(file, config).expect("store");
    store.queue_display_rebuild(finest);
    while store.has_jobs() {
        store
            .run_jobs(Instant::now() + Duration::from_millis(100))
            .expect("the layer builds");
    }
    store.save().expect("saved");
    config
}

/// Builds the project, saves it and opens it again.
///
/// # Panics
///
/// If the fixed setup is rejected, or the temporary directory can't be made.
fn open() -> Opened {
    let dir = std::env::temp_dir().join(format!(
        "brushcraft-bench-display-draw-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("temporary directory");
    let path = dir.join("project.bcp");
    let mesh = Arc::new(cube_sphere(6).expect("mesh"));
    let config = build(&path, &mesh);
    let reopened = ProjectFile::open(&path).expect("reopened");
    let store = TileStore::open(reopened, config).expect("store");
    // Open files outlive their names.
    drop(std::fs::remove_dir_all(&dir));
    let level = Level::new(5).expect("level");
    let keys = mesh
        .faces()
        .map(|face| TileKey::display(face, level, 0, 0).expect("key"))
        .collect();
    Opened { store, keys }
}

/// `display_for_draw` of the first `n` level-5 floor tiles: below 64 decoded on this thread,
/// from 64 on the workers. Rebuild with a lower `PARALLEL_DECODE` to see the workers' cost at
/// the smaller sizes.
///
/// # Panics
///
/// If the setup or the call fails, which would be a bug in the crate.
#[divan::bench(args = [1, 4, 16, 32, 63, 64, 128, 216], sample_count = 200)]
fn floor_tiles_drawn(bencher: divan::Bencher<'_, '_>, n: usize) {
    OPENED.with_borrow_mut(|opened| {
        let opened = opened.get_or_insert_with(open);
        let keys: Vec<TileKey> = opened.keys.iter().take(n).copied().collect();
        bencher
            .counter(divan::counter::ItemsCount::new(n))
            .bench_local(|| {
                opened
                    .store
                    .display_for_draw(divan::black_box(&keys))
                    .expect("drawn")
            });
    });
}
