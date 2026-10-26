//! Creating project, recovery and scratch files: each made under a temporary name, then moved
//! into place, and the files they replace retired.

use std::os::unix::fs::DirBuilderExt as _;
use std::path::{Path, PathBuf};
use std::{fs, io};

use super::{EXTENSION, KeptRecovery, ProjectError};
use crate::file::{FileError, PRIVATE_MODE, ProjectFile, sync_parent};
use crate::meta::{Meta, ProjectId};

/// Deletes the old recovery file at `old`, which its project has moved on from, and syncs its
/// directory.
///
/// Best effort: the project needs nothing in it, so a failure is reported, never fatal; a file
/// already gone is fine.
///
/// # Errors
///
/// The [`KeptRecovery`] if it can't be deleted, or the deletion made durable.
pub(super) fn retire(old: &Path) -> Result<(), KeptRecovery> {
    let kept = |source| KeptRecovery {
        path: old.to_path_buf(),
        source,
    };
    match fs::remove_file(old) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(kept(e)),
    }
    sync_parent(old).map_err(|e| {
        if let FileError::Io(io) = e {
            kept(io)
        } else {
            kept(io::Error::other(e))
        }
    })
}

/// A scratch working file with `meta` over `like`'s mesh, in the system's temporary directory.
///
/// It is what a save rebases the store onto when it can't make a fresh recovery file, so the
/// store has a file to write tiles back to (see [`scratch_in`]).
///
/// # Errors
///
/// As [`scratch_in`].
pub(super) fn scratch(meta: Meta, like: &ProjectFile) -> Result<ProjectFile, ProjectError> {
    scratch_in(&std::env::temp_dir(), meta, like)
}

/// A scratch working file with `meta` over `like`'s mesh, made in `dir`.
///
/// It is a file for the store to write tiles back to when it has no recovery file to write them
/// to, after a save ([`scratch`]) or a failed roll-over
/// ([`Project::strand`](super::Project::strand)).
///
/// It is created readable and writable by the user only ([`PRIVATE_MODE`]), as it holds the
/// user's unsaved work and may sit in a shared directory such as `/tmp` until it is unlinked.
/// It is unlinked at once, so it is only ever the project's open handle: no listing finds it,
/// and nothing is left of it after a crash, or once it is closed. It is no crash protection,
/// and doesn't pretend to be.
///
/// # Errors
///
/// [`ProjectError::Io`] if no random name can be drawn, or it can't be unlinked;
/// [`ProjectError::Store`] if it can't be made. Nothing is left behind, best effort.
pub(super) fn scratch_in(
    dir: &Path,
    meta: Meta,
    like: &ProjectFile,
) -> Result<ProjectFile, ProjectError> {
    let suffix = ProjectId::random()?.to_bits() >> 64;
    let at = dir.join(format!(
        "brushcraft-{}-{suffix:016x}.scratch",
        meta.project()
    ));
    let file = create_clean(&at, meta, like, PRIVATE_MODE)?;
    fs::remove_file(&at)?;
    Ok(file)
}

/// [`ProjectFile::create_like_mode`] at `at` with permission bits `mode`, removing the file
/// again, best effort, if it was made but couldn't be set up. A file already at `at` isn't this
/// call's, and is left alone.
///
/// # Errors
///
/// As [`ProjectFile::create_like_mode`].
pub(super) fn create_clean(
    at: &Path,
    meta: Meta,
    like: &ProjectFile,
    mode: u32,
) -> Result<ProjectFile, FileError> {
    ProjectFile::create_like_mode(at, meta, like, mode).inspect_err(|e| {
        if !matches!(e, FileError::Io(io) if io.kind() == io::ErrorKind::AlreadyExists) {
            drop(fs::remove_file(at));
        }
    })
}

/// A recovery file's two names: the temporary one it is made (and filled) under, and the real
/// one [`Unplaced::place`] moves it to (see the notes of the `project` module).
#[derive(Debug)]
pub(super) struct Unplaced {
    /// Where the file is made: `<name>.tmp`.
    tmp: PathBuf,
    /// Its name once placed: `<project id>-<suffix>.bcrf`.
    path: PathBuf,
    /// Test hook: [`Unplaced::place`] fails as a crash before the rename would, with an I/O
    /// error of this kind, and leaves the temporary file behind as a crash does.
    #[cfg(test)]
    crash: Option<io::ErrorKind>,
}

impl Unplaced {
    /// Moves the file to its real name, if nothing is there, and makes the move durable;
    /// returns the real name.
    ///
    /// # Errors
    ///
    /// [`ProjectError::Io`] if a file has the name already, or the rename or the directory's
    /// sync fails. The temporary file is removed then.
    pub(super) fn place(self) -> Result<PathBuf, ProjectError> {
        #[cfg(test)]
        if let Some(kind) = self.crash {
            return Err(io::Error::new(kind, "a crash before the rename, for the test").into());
        }
        let placed = (|| {
            place_new(&self.tmp, &self.path)?;
            sync_parent(&self.path)?;
            Ok::<_, ProjectError>(())
        })();
        match placed {
            Ok(()) => Ok(self.path),
            Err(e) => {
                self.discard();
                Err(e)
            }
        }
    }

    /// These names, with [`Unplaced::crash`] set to `crash`.
    #[cfg(test)]
    pub(super) fn crashing(self, crash: Option<io::ErrorKind>) -> Self {
        Self { crash, ..self }
    }

    /// Removes the temporary file: a creation given up. Best effort, as the error at hand is
    /// the one to report; a leftover is swept by [`find_recoveries`](super::find_recoveries).
    pub(super) fn discard(self) {
        drop(fs::remove_file(&self.tmp));
    }
}

/// Creates a recovery file of `project` in `dir` through `create`, under the temporary name of
/// a fresh one (see the notes of the `project` module), and returns it open, with its names for
/// [`Unplaced::place`].
///
/// `create` makes the file at the temporary path it is given, durable, and readable by the user
/// only ([`PRIVATE_MODE`]).
///
/// The directory is made if missing, with its parents, as the user's alone (`0700`): it holds
/// the user's unsaved work. A directory that already exists keeps its mode.
///
/// # Errors
///
/// [`ProjectError::Io`] if the directory can't be made or a random suffix drawn; `create`'s
/// error.
pub(super) fn create_recovery(
    dir: &Path,
    project: ProjectId,
    create: impl FnOnce(&Path) -> Result<ProjectFile, FileError>,
) -> Result<(ProjectFile, Unplaced), ProjectError> {
    fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(dir)?;
    let suffix = ProjectId::random()?.to_bits() >> 64;
    let name = format!("{project}-{suffix:016x}.{EXTENSION}");
    let names = Unplaced {
        tmp: dir.join(format!("{name}.tmp")),
        path: dir.join(&name),
        #[cfg(test)]
        crash: None,
    };
    let file = create(&names.tmp)?;
    Ok((file, names))
}

/// [`create_recovery`], then [`Unplaced::place`] at once: for a recovery file that starts
/// empty and needs no filling. Returns it open, with its path.
///
/// # Errors
///
/// As [`create_recovery`] and [`Unplaced::place`].
pub(super) fn create_placed(
    dir: &Path,
    project: ProjectId,
    create: impl FnOnce(&Path) -> Result<ProjectFile, FileError>,
) -> Result<(ProjectFile, PathBuf), ProjectError> {
    let (file, names) = create_recovery(dir, project, create)?;
    let path = names.place()?;
    Ok((file, path))
}

/// Gives the file at `tmp` the name `path`, if nothing has that name, and takes the name `tmp`
/// away: a rename that never replaces a file.
///
/// A hard link fails if `path` exists, however late it was made; a check before a rename
/// leaves a window in which a file made at `path` would be replaced. On a file system without
/// hard links, it falls back to that check and a rename. The name `tmp` is removed best effort:
/// a leftover is a second name of the placed file, which only takes a name.
///
/// # Errors
///
/// [`io::ErrorKind::AlreadyExists`] if `path` names a file; the I/O error of the link or the
/// rename.
pub(super) fn place_new(tmp: &Path, path: &Path) -> io::Result<()> {
    match fs::hard_link(tmp, path) {
        Ok(()) => {
            // Best effort: the file is in place either way.
            drop(fs::remove_file(tmp));
            Ok(())
        }
        Err(e)
            if matches!(
                e.kind(),
                io::ErrorKind::Unsupported | io::ErrorKind::PermissionDenied
            ) =>
        {
            if fs::symlink_metadata(path).is_ok() {
                return Err(io::ErrorKind::AlreadyExists.into());
            }
            fs::rename(tmp, path)
        }
        Err(e) => Err(e),
    }
}

/// A temporary path beside `path`, for a project file written before it is renamed there:
/// `<name>.<random>.tmp`.
///
/// # Errors
///
/// [`ProjectError::Io`] if `path` has no file name or no random suffix can be drawn.
pub(super) fn temporary(path: &Path) -> Result<PathBuf, ProjectError> {
    let name = path
        .file_name()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "the path has no file name"))?;
    let mut tmp = name.to_os_string();
    tmp.push(format!(
        ".{:016x}.tmp",
        ProjectId::random()?.to_bits() >> 64
    ));
    Ok(path.with_file_name(tmp))
}
