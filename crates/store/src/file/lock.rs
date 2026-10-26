//! Opening project files and locking them: shared for reading, exclusive for writing, checked to
//! still be the file at their path, and never through a symlink.

use std::fs::{File, TryLockError};
use std::io;
use std::ops::Deref;
use std::os::unix::fs::MetadataExt;
use std::path::Path;

use rustix::fs::{FileType, Mode, OFlags};

use super::error::StoreError;

/// Whether a [`ProjectFile`](super::ProjectFile) accepts writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Access {
    /// Opened at the latest state: writes append. Holds the file's exclusive lock.
    ReadWrite,
    /// Opened at an earlier save point, without write permission, or as a compaction source.
    /// Holds a shared lock (an exclusive one as the source of an in-place compaction).
    ReadOnly,
}

/// A lock on the project file: exclusive to write (or replace) it, shared to read it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Lock {
    /// One holder only: the writer, or the source of an in-place compaction.
    Exclusive,
    /// Any number of readers, and no writer.
    Shared,
}

impl Lock {
    /// Takes this lock on `file` without waiting.
    ///
    /// # Errors
    ///
    /// [`StoreError::Locked`] if a conflicting lock is held, [`StoreError::Io`].
    fn take(self, file: &File) -> Result<(), StoreError> {
        let locked = match self {
            Self::Exclusive => file.try_lock(),
            Self::Shared => file.try_lock_shared(),
        };
        locked.map_err(|e| match e {
            TryLockError::WouldBlock => StoreError::Locked,
            TryLockError::Error(e) => e.into(),
        })
    }
}

/// An open file holding a [`Lock`], which it releases when dropped, before it closes.
///
/// A lock belongs to the open file description, which every duplicate of the descriptor shares,
/// and closing a descriptor releases it only once no duplicate is left. Spawning a process
/// duplicates all of this process's descriptors into the child until its `exec` (close-on-exec
/// closes them only then), so a file merely closed while another thread spawns stays locked for
/// that moment, against this very process: reopening it fails with [`StoreError::Locked`].
/// Unlocking releases the lock on the description itself, whichever descriptors still share it,
/// so a dropped `LockedFile` is unlocked at once.
#[derive(Debug)]
pub(super) struct LockedFile(File);

impl LockedFile {
    /// Takes `lock` on `file` without waiting.
    ///
    /// # Errors
    ///
    /// As [`Lock::take`].
    pub(super) fn new(file: File, lock: Lock) -> Result<Self, StoreError> {
        lock.take(&file)?;
        Ok(Self(file))
    }
}

impl Deref for LockedFile {
    type Target = File;

    /// The file, for reading, writing and syncing it.
    fn deref(&self) -> &File {
        &self.0
    }
}

impl Drop for LockedFile {
    /// Unlocks the file, then closes it.
    fn drop(&mut self) {
        // Best effort: if unlocking fails, closing still releases the lock once no duplicate of
        // the descriptor is left.
        drop(self.0.unlock());
    }
}

/// Opens `path` with `open` and takes `lock` on it (see [`lock_at`]).
///
/// # Errors
///
/// As [`lock_at`], and `open`'s error.
pub(super) fn open_locked(
    path: &Path,
    open: impl Fn(&Path) -> io::Result<File>,
    lock: Lock,
) -> Result<LockedFile, StoreError> {
    lock_at(path, open(path)?, open, lock)
}

/// Takes `lock` on `file`, opened from `path`, and checks that `path` still names it.
///
/// A lock belongs to the file, not the path, and compacting in place renames a new file over the
/// path: a handle opened before that rename and locked after it would hold the old, unlinked file.
/// So if `path` names another file once the lock is held, `file` is dropped and `path` opened
/// (with `open`) and locked once more.
///
/// # Errors
///
/// [`StoreError::Locked`] if a conflicting lock is held, or `path` is replaced again during the
/// retry; `open`'s error; [`StoreError::Io`].
pub(super) fn lock_at(
    path: &Path,
    file: File,
    open: impl Fn(&Path) -> io::Result<File>,
    lock: Lock,
) -> Result<LockedFile, StoreError> {
    let file = LockedFile::new(file, lock)?;
    if is_at(&file, path)? {
        return Ok(file);
    }
    drop(file);
    let reopened = LockedFile::new(open(path)?, lock)?;
    if is_at(&reopened, path)? {
        Ok(reopened)
    } else {
        Err(StoreError::Locked)
    }
}

/// Whether `path` names `file` (the same device and inode); `false` if it names nothing.
///
/// # Errors
///
/// The I/O error of either `stat`.
fn is_at(file: &File, path: &Path) -> io::Result<bool> {
    let held = file.metadata()?;
    match std::fs::metadata(path) {
        Ok(now) => Ok(held.dev() == now.dev() && held.ino() == now.ino()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e),
    }
}

/// Opens `path` for reading, if it is a regular file (see [`open_regular`]).
///
/// # Errors
///
/// As [`open_regular`].
pub(super) fn open_readable(path: &Path) -> io::Result<File> {
    open_regular(path, OFlags::RDONLY)
}

/// Opens `path` for reading and writing, if it is a regular file (see [`open_regular`]).
///
/// # Errors
///
/// As [`open_regular`].
pub(super) fn open_writable(path: &Path) -> io::Result<File> {
    open_regular(path, OFlags::RDWR)
}

/// Opens `path` with `access` (`RDONLY` or `RDWR`) and returns it as a blocking [`File`], if it
/// is a regular file.
///
/// A path may come from a file (a recovery file names its project) or a directory listing, so it
/// may name a FIFO or a device, and opening one must neither block nor do more than the open
/// itself: a plain open of a FIFO waits for a writer, and would hang listing recovery files at
/// every launch, and a terminal could become the process's controlling one. So the open doesn't
/// block (`O_NONBLOCK`) or take a terminal (`O_NOCTTY`), and closes on exec (`O_CLOEXEC`);
/// anything but a regular file is refused, and the descriptor is then made blocking again, so it
/// reads as any other.
///
/// # Errors
///
/// [`io::ErrorKind::InvalidInput`] if `path` names something other than a regular file; the I/O
/// error of the open, the `fstat` or the `fcntl`s.
pub(crate) fn open_regular(path: &Path, access: OFlags) -> io::Result<File> {
    let file = File::from(rustix::fs::open(
        path,
        access | OFlags::NONBLOCK | OFlags::NOCTTY | OFlags::CLOEXEC,
        Mode::empty(),
    )?);
    if FileType::from_raw_mode(rustix::fs::fstat(&file)?.st_mode) != FileType::RegularFile {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "not a regular file",
        ));
    }
    let flags = rustix::fs::fcntl_getfl(&file)?;
    rustix::fs::fcntl_setfl(&file, flags.difference(OFlags::NONBLOCK))?;
    Ok(file)
}
