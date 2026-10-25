//! Paths that name no regular file, in a recovery file or a recovery directory: opening or
//! listing them never blocks, and only a regular file opens.

use std::sync::mpsc;
use std::time::Duration;

use rustix::fs::{AtFlags, CWD, FileType, Mode, Timespec, Timestamps, mknodat, utimensat};

use super::*;

/// Runs `f` on a thread of its own and returns what it returns, failing if that takes more than
/// ten seconds: a guard against a hang, not a timing.
fn without_hanging<T: Send + 'static>(what: &str, f: impl FnOnce() -> T + Send + 'static) -> T {
    let (sender, receiver) = mpsc::channel();
    std::thread::spawn(move || drop(sender.send(f())));
    receiver
        .recv_timeout(Duration::from_secs(10))
        .unwrap_or_else(|e| panic!("{what}: no answer ({e})"))
}

#[test]
fn a_recovery_file_naming_a_fifo_is_listed_without_blocking() {
    let dir = TestDir::new();
    let mesh = mesh();
    let path = dir.path("p.bcrf");
    let mut project = Project::new((*mesh).clone(), config(&dir)).unwrap();
    let mut map = CorrectionMap::new();
    stroke(&mut project, &mut map, 80, 191);
    assert_saved(project.save_as(&path));
    stroke(&mut project, &mut map, 80, 192);
    drop(project);
    // The project file the recovery file names is now a FIFO no one writes to.
    fs::remove_file(&path).unwrap();
    mknodat(CWD, &path, FileType::Fifo, Mode::RUSR | Mode::WUSR, 0).unwrap();

    let recovery_dir = dir.path("recovery");
    let candidates = without_hanging("listing", move || find_recoveries(&recovery_dir).unwrap());
    let [candidate] = candidates.as_slice() else {
        panic!("one recovery file: {candidates:?}");
    };
    assert_eq!(
        candidate.base_path.as_deref(),
        Some(path.as_path()),
        "{candidate:?}"
    );
    assert_eq!(candidate.status, CandidateStatus::Damaged, "{candidate:?}");

    let fifo = path.clone();
    let inspected = without_hanging("inspecting", move || ProjectFile::inspect(&fifo).map(drop));
    assert!(inspected.is_err(), "not a project file: {inspected:?}");
}

#[test]
fn only_a_regular_file_opens() {
    let dir = TestDir::new();
    let directory = dir.path("a directory");
    fs::create_dir_all(&directory).unwrap();
    for path in [Path::new("/dev/null"), directory.as_path()] {
        let inspected = ProjectFile::inspect(path).map(drop);
        assert!(
            matches!(&inspected, Err(StoreError::Io(e)) if e.kind() == io::ErrorKind::InvalidInput),
            "{}: {inspected:?}",
            path.display()
        );
    }
}

#[test]
fn a_fifo_at_a_temporary_recovery_name_is_listed_past_without_blocking() {
    let dir = TestDir::new();
    let recovery_dir = dir.path("recovery");
    fs::create_dir_all(&recovery_dir).unwrap();
    let name = |n: u8| format!("{n:032x}-{n:016x}.{EXTENSION}.tmp");
    let (fifo, stale) = (recovery_dir.join(name(1)), recovery_dir.join(name(2)));
    mknodat(CWD, &fifo, FileType::Fifo, Mode::RUSR | Mode::WUSR, 0).unwrap();
    fs::write(&stale, b"left by a crash").unwrap();
    // Both unwritten for long enough to be a crash's leftovers.
    let long_ago = Timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    let times = Timestamps {
        last_access: long_ago,
        last_modification: long_ago,
    };
    for path in [&fifo, &stale] {
        utimensat(CWD, path, &times, AtFlags::empty()).unwrap();
    }

    let candidates = without_hanging("listing", move || find_recoveries(&recovery_dir).unwrap());
    assert_eq!(candidates, [], "temporary files are not candidates");
    assert!(
        fs::symlink_metadata(&fifo).is_ok(),
        "the FIFO is left as it is"
    );
    assert!(!stale.exists(), "a stale temporary file is still swept");
}
