//! Opens any bytes as a project file and reads everything it names: an error or the file's
//! content, never a panic or an abort, and within libFuzzer's `-timeout`. Then opens a tile store
//! over it keeping a display layer (its book of display levels, the floor it loads, the rebuild
//! it queues) and runs its jobs for a few milliseconds.
//!
//! The input is a whole file. Every record is checksummed, so a change anywhere would fail the
//! checksum of the record it lands in and reach nothing behind it; the target therefore
//! recomputes each record's checksum first ([`store::fuzzing::fix_checksums`]), and the fuzzer's
//! changes reach the parsers as content. The seeds (`fuzz/corpus/project_open/seed-*`) are small
//! project and recovery files written by the store's tests (`write_fuzz_seeds`).

#![no_main]

use std::path::PathBuf;

use libfuzzer_sys::fuzz_target;
use store::{ProjectFile, StoreConfig, TileStore, Workers};

/// Where this process writes the file it opens: memory-backed, one per fuzzing process.
fn scratch() -> PathBuf {
    PathBuf::from(format!("/dev/shm/brushcraft-fuzz-{}", std::process::id()))
}

/// Opens the file at `path` and reads every tile it holds, raw and decoded; then opens it at
/// each of its save points and does the same.
///
/// # Errors
///
/// The first error of an open or a read: the answer to a hostile file, which ends the walk.
fn open_and_read(path: &std::path::Path) -> Result<(), store::FileError> {
    let file = ProjectFile::open(path)?;
    let points = file.save_points()?;
    read_all(&file)?;
    drop(file);
    for point in points {
        read_all(&ProjectFile::open_at(path, point)?)?;
    }
    Ok(())
}

/// Reads every tile of `file`, raw and decoded (correction tiles as displacement, display
/// records as display tiles).
///
/// # Errors
///
/// The first read's error.
fn read_all(file: &ProjectFile) -> Result<(), store::FileError> {
    let keys: Vec<_> = file.keys().copied().collect();
    for key in &keys {
        file.read_tile(key)?;
        if key.channel() == geom::Channel::DISPLACEMENT {
            file.read_displacement(key)?;
        }
    }
    store::fuzzing::read_displays(file)
}

/// Opens a tile store over the file at `path` keeping a display layer under level 2 (the seeds'
/// finest), with one worker, and runs its jobs for a few milliseconds.
///
/// # Errors
///
/// The open's or a job's error: the answer to a hostile file.
fn open_store(path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let finest = geom::Level::new(2)?;
    let config = StoreConfig::new(64 << 20, 1)
        .with_workers(Workers::Fixed(std::num::NonZeroUsize::MIN))
        .with_display(finest);
    let mut store = TileStore::open(ProjectFile::open(path)?, config)?;
    store.run_jobs(std::time::Instant::now() + std::time::Duration::from_millis(5))?;
    Ok(())
}

fuzz_target!(|bytes: &[u8]| {
    let mut bytes = bytes.to_vec();
    store::fuzzing::fix_checksums(&mut bytes);
    let path = scratch();
    if std::fs::write(&path, &bytes).is_ok() {
        // Any error is a valid answer to a hostile file; only panics and hangs are findings.
        drop(open_and_read(&path));
        drop(open_store(&path));
        drop(std::fs::remove_file(&path));
    }
});
