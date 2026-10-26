//! Compacting a project file into a fresh one holding only its latest save point, and the
//! temporary sibling and directory sync that make the swap safe.

use std::fs::File;
use std::io;
use std::os::unix::fs::PermissionsExt as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use super::error::FileError;
use super::format::tile_record;
use super::lock::{Lock, open_locked, open_readable};
use super::{Loc, Name, ProjectFile, Repair};
use crate::display::is_display;
use crate::stamp::Stamp;

/// Writes a new project at `dst` holding `src`'s meta and mesh, its live tiles and bounds (raw
/// records copied, not recompressed) and one full Index.
///
/// `src` is recovered as by [`ProjectFile::open`], but never modified: whatever follows its last
/// save point is left in place (and not copied). It is held with a shared lock, so it can't be open
/// for writing. History before the latest state is dropped; which history to keep is decided with
/// stroke history (M2). On error, `dst` may be left partly written.
///
/// # Errors
///
/// Any [`FileError`] from reading `src` (a damaged live tile is [`FileError::Damaged`]; one
/// open for writing, [`FileError::Locked`]) or writing `dst`, which must not exist.
pub fn compact(src: &Path, dst: &Path) -> Result<(), FileError> {
    compact_into(&ProjectFile::inspect(src)?, dst)
}

/// Compacts the project at `path` in place, as [`compact`] does.
///
/// It compacts into a temporary file `<name>.compact-*.tmp` in the same directory, created with
/// no more permissions than the project's (so a private project is never exposed through it),
/// gives it the project's permissions, then renames it over `path` (resolved first, so a symlink
/// keeps pointing at the project rather than being replaced). Replacing the project is a write, so
/// the source is locked exclusively until after the rename: no writer or other compaction runs
/// meanwhile, and one that opened the old file locks it only to find the path replaced (see
/// [`ProjectFile::open`]). The directory is synced after the rename, so the swap is durable.
///
/// On error, `path` is unchanged (unless the rename succeeded and only the directory sync
/// failed) and the temporary file is removed. A crash before the rename can leave the temporary
/// file behind; it is never read, and can be deleted.
///
/// # Errors
///
/// As [`compact`], with [`FileError::Locked`] if the project is open at all.
pub fn compact_in_place(path: &Path) -> Result<(), FileError> {
    let path = path.canonicalize()?;
    let from = ProjectFile::open_latest(
        open_locked(&path, open_readable, Lock::Exclusive)?,
        Repair::Ignore,
        None,
    )?;
    let permissions = from.file.metadata()?.permissions();
    let tmp = temp_sibling(&path)?;
    let swapped = compact_into(&from, &tmp).and_then(|()| {
        // The umask may have masked bits of the project's mode at creation: give them back.
        std::fs::set_permissions(&tmp, permissions)?;
        std::fs::rename(&tmp, &path)?;
        sync_parent(&path)
    });
    if swapped.is_err() {
        // Best effort: gone already if the rename succeeded.
        drop(std::fs::remove_file(&tmp));
    }
    swapped
}

/// Writes `from`'s live tiles and bounds and one full Index to a new project at `dst`, created
/// with `from`'s permission bits (less the umask's), so it is never more open than `from`.
///
/// A bounds or display record that doesn't read is written as unknown, as
/// [`ProjectFile::copy_base`] does.
///
/// # Errors
///
/// As [`compact`].
pub(super) fn compact_into(from: &ProjectFile, dst: &Path) -> Result<(), FileError> {
    let mode = from.file.metadata()?.permissions().mode() & 0o777;
    let mut to = ProjectFile::create_like_mode(dst, from.meta.clone(), from, mode)?;
    for (name, &Loc { offset, len }) in &from.live {
        match name {
            Name::Tile(key) if is_display(key) => {
                // Derived: one that doesn't read goes across as unknown.
                match from.read_record(offset, len) {
                    Ok(raw) => to.append_display_copy(key, &raw)?,
                    Err(FileError::Damaged { .. }) => {
                        to.write_display_many([(*key, None)], Stamp::ZERO)?;
                    }
                    Err(e) => return Err(e),
                }
            }
            Name::Tile(key) => {
                let raw = from.read_record(offset, len)?;
                if tile_record(&raw, key).is_none() {
                    return Err(FileError::Damaged { offset });
                }
                let loc = to.append_record(&raw)?;
                to.note(*name, None, Stamp::ZERO, loc);
            }
            Name::Bounds(key) => match from.read_bounds_record(key)? {
                Some(raw) => {
                    let loc = to.append_record(&raw)?;
                    to.note(*name, None, Stamp::ZERO, loc);
                }
                None => to.write_bounds(*key, None, Stamp::ZERO)?,
            },
        }
    }
    to.head = from.head;
    to.save()?;
    Ok(())
}

/// A path for a temporary file next to `path`, unique within this process and among processes.
///
/// # Errors
///
/// [`FileError::Io`] if `path` has no file name.
fn temp_sibling(path: &Path) -> Result<PathBuf, FileError> {
    /// Distinguishes this process's temporary files.
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let name = path.file_name().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "the project path has no file name",
        )
    })?;
    let mut tmp = name.to_os_string();
    tmp.push(format!(
        ".compact-{}-{}.tmp",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    Ok(path.with_file_name(tmp))
}

/// Syncs the directory holding `path`, making a new or renamed entry in it durable.
///
/// # Errors
///
/// [`FileError::Io`].
pub(crate) fn sync_parent(path: &Path) -> Result<(), FileError> {
    let dir = match path.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir,
        _ => Path::new("."),
    };
    File::open(dir)?.sync_all()?;
    Ok(())
}
