//! A temporary directory for tests, removed on drop. (Ours; no `tempfile` dependency.)

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Distinguishes the directories of one process.
static NEXT: AtomicU64 = AtomicU64::new(0);

/// A fresh directory under the system temp dir, deleted with everything in it on drop.
#[derive(Debug)]
pub(crate) struct TestDir(PathBuf);

impl TestDir {
    /// Creates the directory.
    ///
    /// # Panics
    ///
    /// If it can't be created.
    pub(crate) fn new() -> Self {
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("brushcraft-app-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    /// Takes on `path`, a directory made by the test process that started this one (a child a
    /// test kills), to work in: removed on drop, as one this made.
    pub(crate) const fn adopt(path: PathBuf) -> Self {
        Self(path)
    }

    /// The directory.
    pub(crate) fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        drop(std::fs::remove_dir_all(&self.0));
    }
}
