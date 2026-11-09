//! Helpers the app's tests share across modules.

use std::time::{Duration, Instant};

use store::{Project, SaveAhead};

/// Waits until `project`'s recovery writer has no flush under way: every cut taken so far is in
/// the recovery file, durable. A saved project's [`Project::save_ahead`] says so, and copies
/// nothing a reader sees.
///
/// # Panics
///
/// If a slice of the save ahead fails, or the flushes take more than a minute.
pub(crate) fn settle(project: &mut Project) {
    let began = Instant::now();
    while matches!(
        project.save_ahead(Instant::now()).expect("a slice"),
        SaveAhead::Flushing
    ) {
        std::thread::sleep(Duration::from_millis(1));
        assert!(began.elapsed() < Duration::from_secs(60), "the flush ends");
    }
}
