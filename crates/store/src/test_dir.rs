//! A temporary directory for tests, removed on drop. (Ours; no `tempfile` dependency.)

use std::path::PathBuf;
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
        let dir = std::env::temp_dir().join(format!("brushcraft-store-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    /// The path of `name` inside the directory.
    pub(crate) fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        drop(std::fs::remove_dir_all(&self.0));
    }
}
