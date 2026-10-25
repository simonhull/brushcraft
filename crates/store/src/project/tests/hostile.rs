//! Recovery files naming paths no project file is at: listing them never blocks.

use std::sync::mpsc;
use std::time::Duration;

use rustix::fs::{CWD, FileType, Mode, mknodat};

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
