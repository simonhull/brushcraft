//! A project as the app holds it: the project file the user saves, a recovery file streamed in
//! the background, and the tile store layered over both.
//!
//! # Two files
//!
//! The **project file** is where the user saved the project. It changes only on
//! [`Project::save`] (or [`Project::save_as`], which starts a new one), each save appending the
//! tiles edited since the last plus a save point. The **recovery file** lives in app-private
//! storage ([`ProjectConfig::recovery_dir`]). Its meta names the project and the save point it
//! extends, and it holds only the edits since that save: a [`RecoveryWriter`] streams them in,
//! with a durable save point of its own after every stroke and at least once a second. A
//! project never saved has a recovery file only, which holds all of its work.
//!
//! # Saving
//!
//! A save finishes the flush in flight (and one more, so every edit is in the recovery file),
//! copies every tile edited since the last save into the project file
//! ([`TileStore::save_into`]), and saves it. Then a **fresh** recovery file is created on the new
//! save point, the store is rebased onto the pair, and only then, with the new recovery file
//! durable, is the old one deleted. A crash at any point leaves a recovery file that matches a
//! save point of the project file: the old one until the new one is in place.
//!
//! # Recovery file names
//!
//! A recovery file is named `<project id>-<suffix>.bcrf`, the id as 32 hex digits and the
//! suffix 16 random ones, so the old and the new file of one project coexist during a save. It
//! is created as `<name>.tmp`, synced, and renamed into place: a crash mid-create leaves at
//! most a `.tmp` file, never a half-made file under a recovery file's name.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use std::{fs, io};

use geom::BaseMesh;
use thiserror::Error;

use crate::file::{ProjectFile, StoreError, sync_parent};
use crate::meta::{Meta, ProjectId, Role};
use crate::recovery::{RecoveryError, RecoveryWriter};
use crate::tiles::{CacheError, Saved, StoreConfig, TileStore};

/// The extension of a recovery file (and of a project file).
const EXTENSION: &str = "bcrf";

/// How a [`Project`] is set up.
#[derive(Debug, Clone)]
pub struct ProjectConfig {
    /// The tile store's budget, chunks and site.
    pub store: StoreConfig,
    /// Where recovery files go: app-private storage, [`default_recovery_dir`] in the app. It is
    /// created if missing.
    pub recovery_dir: PathBuf,
}

/// The app's recovery directory on Linux: `$XDG_STATE_HOME/brushcraft/recovery`, else
/// `$HOME/.local/state/brushcraft/recovery`; `None` if neither variable names an absolute path.
#[must_use]
pub fn default_recovery_dir() -> Option<PathBuf> {
    /// The variable's value as a path, if it is an absolute one.
    fn absolute(name: &str) -> Option<PathBuf> {
        std::env::var_os(name)
            .map(PathBuf::from)
            .filter(|path| path.is_absolute())
    }
    let state = absolute("XDG_STATE_HOME")
        .or_else(|| absolute("HOME").map(|home| home.join(".local").join("state")))?;
    Some(state.join("brushcraft").join("recovery"))
}

/// Why a [`Project`] operation failed.
#[derive(Debug, Error)]
pub enum ProjectError {
    /// A file system call failed: creating the recovery directory, renaming a file into place,
    /// deleting a recovery file.
    #[error("project I/O failed")]
    Io(#[from] io::Error),
    /// The project file or a recovery file failed.
    #[error("a project or recovery file failed")]
    Store(#[from] StoreError),
    /// The tile store failed.
    #[error("the tile store failed")]
    Cache(#[from] CacheError),
    /// The recovery writer failed.
    #[error("the recovery writer failed")]
    Recovery(#[from] RecoveryError),
    /// [`Project::save`] on a project never saved: it has no project file yet. Use
    /// [`Project::save_as`].
    #[error("the project has never been saved: it needs a path")]
    NoPath,
    /// [`Project::save_as`] was given a path that already names a file. Nothing was written.
    #[error("{0:?} already exists")]
    Exists(PathBuf),
    /// [`Project::open`] was given a file that isn't a project file (a recovery file, say).
    #[error("not a project file")]
    NotAProject,
    /// [`Project::open`] was given a project file with no save point: there is no saved state
    /// to open.
    #[error("the project file has no save point")]
    NeverSaved,
    /// The latest edits of these tiles aren't in the recovery file: their slots were broken, or
    /// moving to a new recovery file couldn't read them. The rest of the edits are protected.
    #[error("the edits of {} tiles are not in the recovery file", .0.len())]
    Unprotected(Vec<geom::TileKey>),
}

/// An open project: the tile store over the project file's last save and a recovery file of
/// the edits since, the recovery writer streaming them, and the paths of both files (see the
/// module notes).
///
/// It belongs to one thread, the one that draws; [`Project::poll`] advances the recovery writer
/// and is meant to be called from the frame loop. Dropping a project closes its writer
/// (flushing what it can, as [`Project::close`] does) and keeps the recovery file, as a crash
/// would.
pub struct Project {
    /// Streams the store's edits into the recovery file. `None` only once closed: it is taken
    /// out by [`Project::close`] and the drop, which close it on the store.
    writer: Option<RecoveryWriter>,
    /// The tile store: the recovery file over the project file's last save point.
    store: TileStore,
    /// The project file, held open (and locked) for saves; the store's base reads it. `None`
    /// for a project never saved.
    file: Option<ProjectFile>,
    /// Where the project file is; `None` for a project never saved.
    path: Option<PathBuf>,
    /// Where the store's working file, the recovery file, is.
    recovery_path: PathBuf,
    /// The directory recovery files go in.
    recovery_dir: PathBuf,
    /// Makes the next recovery file's creation fail after its temporary file is written and
    /// synced, before the rename: a crash mid-create, for the tests.
    #[cfg(test)]
    fail_next_recovery: bool,
}

impl std::fmt::Debug for Project {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Project")
            .field("path", &self.path)
            .field("recovery_path", &self.recovery_path)
            .field("store", &self.store)
            .finish_non_exhaustive()
    }
}

impl Project {
    /// A new project over `mesh`, never saved: a fresh project id and a recovery file holding
    /// all of its work, from the first stroke.
    ///
    /// # Errors
    ///
    /// [`ProjectError::Io`] if the id can't be drawn or the recovery directory or file can't be
    /// made; [`ProjectError::Store`] if the recovery file can't be written;
    /// [`ProjectError::Cache`] if the store refuses `config.store`;
    /// [`ProjectError::Recovery`] if the writer's thread can't be started. No recovery file is
    /// left behind on error.
    pub fn new(mesh: BaseMesh, config: ProjectConfig) -> Result<Self, ProjectError> {
        let mesh = Arc::new(mesh);
        let meta = Meta::new(ProjectId::random()?, Role::Recovery { base: None }, &mesh);
        let (working, recovery_path) = create_recovery(
            &config.recovery_dir,
            meta,
            |path| ProjectFile::create(path, meta, Arc::clone(&mesh)),
            #[cfg(test)]
            false,
        )?;
        Self::start(
            TileStore::open(working, config.store),
            None,
            recovery_path,
            config,
        )
    }

    /// Opens the project file at `path` at its last save point, with a fresh recovery file
    /// layered on it.
    ///
    /// The project file is opened for writing and locked, so no other instance can open it
    /// meanwhile; whatever followed its last save point (a save cut short) is discarded.
    ///
    /// # Errors
    ///
    /// [`ProjectError::Store`] if the file can't be opened (including
    /// [`StoreError::Locked`]); [`ProjectError::NotAProject`]; [`ProjectError::NeverSaved`];
    /// as [`Project::new`] for the recovery file, the store and the writer.
    pub fn open(path: &Path, config: ProjectConfig) -> Result<Self, ProjectError> {
        let project = ProjectFile::open(path)?;
        if project.meta().role() != Role::Project {
            return Err(ProjectError::NotAProject);
        }
        let base = project.base()?.ok_or(ProjectError::NeverSaved)?;
        let meta = project.meta().with_role(Role::Recovery {
            base: Some(base.save_point()),
        });
        let (working, recovery_path) = create_recovery(
            &config.recovery_dir,
            meta,
            |at| ProjectFile::create_like(at, meta, &project),
            #[cfg(test)]
            false,
        )?;
        let store = TileStore::open_layered(working, Some(base), config.store);
        Self::start(
            store,
            Some((project, path.to_path_buf())),
            recovery_path,
            config,
        )
    }

    /// Starts the recovery writer on `store` and assembles the project; on error, deletes the
    /// recovery file at `recovery_path`, which nothing was written to.
    ///
    /// # Errors
    ///
    /// `store`'s error; [`ProjectError::Recovery`] if the writer's thread can't be started.
    fn start(
        opened: Result<TileStore, CacheError>,
        project: Option<(ProjectFile, PathBuf)>,
        recovery_path: PathBuf,
        config: ProjectConfig,
    ) -> Result<Self, ProjectError> {
        let started = opened.map_err(ProjectError::from).and_then(|mut store| {
            let writer = RecoveryWriter::new(&mut store)?;
            Ok((writer, store))
        });
        let (writer, store) = match started {
            Ok(started) => started,
            Err(e) => {
                // Best effort: the file is empty, and the error at hand is the one to report.
                drop(fs::remove_file(&recovery_path));
                return Err(e);
            }
        };
        let (file, path) = project.unzip();
        Ok(Self {
            writer: Some(writer),
            store,
            file,
            path,
            recovery_path,
            recovery_dir: config.recovery_dir,
            #[cfg(test)]
            fail_next_recovery: false,
        })
    }

    /// Saves the edits since the last save into the project file, as a new save point, and
    /// starts a fresh recovery file on it (see the module notes).
    ///
    /// # Errors
    ///
    /// [`ProjectError::NoPath`] for a project never saved; [`ProjectError::Recovery`] if the
    /// recovery writer can't finish its flushes (nothing is saved then);
    /// [`ProjectError::Cache`] if the save fails ([`CacheError::Save`]: the project file may
    /// hold the tiles without a save point; saving again writes them again); as
    /// [`Project::new`] for the fresh recovery file. A save that fails after the project file's
    /// save point is written leaves the project open on the old recovery file, which still
    /// holds every edit; saving again works.
    pub fn save(&mut self) -> Result<(), ProjectError> {
        self.poll()?;
        if self.file.is_none() {
            return Err(ProjectError::NoPath);
        }
        self.finish_flushes()?;
        let project = self.file.as_mut().ok_or(ProjectError::NoPath)?;
        let saved = self.store.save_into(project)?;
        self.switch_recovery(saved)
    }

    /// Saves the project as a new project file at `path`, with a new project id, and goes on
    /// with that one: every tile of the last save point copied (as records, not re-encoded),
    /// then the edits since, then a save point. The old project file isn't written to.
    ///
    /// The file is written as a temporary file beside `path` and renamed into place once its
    /// save point is durable, so `path` never names a half-written project.
    ///
    /// # Errors
    ///
    /// [`ProjectError::Exists`] if `path` names a file already; [`ProjectError::Recovery`] if
    /// the recovery writer can't finish its flushes; [`ProjectError::Store`] or
    /// [`ProjectError::Cache`] if writing the new file fails (the temporary file is removed);
    /// [`ProjectError::Io`] if the rename does; as [`Project::save`] for the fresh recovery
    /// file, where the new project file is in place but the project goes on with the old one.
    pub fn save_as(&mut self, path: &Path) -> Result<(), ProjectError> {
        self.poll()?;
        if fs::symlink_metadata(path).is_ok() {
            return Err(ProjectError::Exists(path.to_path_buf()));
        }
        self.finish_flushes()?;
        let meta = self
            .store
            .working()
            .meta()
            .with_project(ProjectId::random()?)
            .with_role(Role::Project);
        let tmp = temporary(path)?;
        let mut project = ProjectFile::create_like(&tmp, meta, self.store.working())?;
        let written = self.write_new_project(&mut project, &tmp, path);
        let saved = match written {
            Ok(saved) => saved,
            Err(e) => {
                // Best effort: the error at hand is the one to report.
                drop(fs::remove_file(&tmp));
                return Err(e);
            }
        };
        self.switch_recovery_to(saved, &project)?;
        // The old project file closes here, and its lock goes with it.
        self.file = Some(project);
        self.path = Some(path.to_path_buf());
        Ok(())
    }

    /// Fills `project`, the new file being written at `tmp`, with the last save point's tiles
    /// and the edits since, saves it, and renames it to `path`.
    ///
    /// # Errors
    ///
    /// As [`Project::save_as`].
    fn write_new_project(
        &mut self,
        project: &mut ProjectFile,
        tmp: &Path,
        path: &Path,
    ) -> Result<Saved, ProjectError> {
        if let Some(base) = self.store.base() {
            project.copy_base(base)?;
        }
        let saved = self.store.save_into(project)?;
        fs::rename(tmp, path)?;
        sync_parent(path)?;
        Ok(saved)
    }

    /// Closes the project: finishes the recovery writer's flushes and stops it, then deletes
    /// the recovery file unless `keep_recovery`.
    ///
    /// Whether to keep the recovery file of a project with unsaved edits is the user's choice,
    /// which the app asks for; this obeys the flag. Without it, edits since the last save are
    /// gone.
    ///
    /// # Errors
    ///
    /// [`ProjectError::Recovery`] if the last flushes fail: the recovery file is kept then,
    /// whatever `keep_recovery` says. [`ProjectError::Io`] or [`ProjectError::Store`] if it
    /// can't be deleted.
    pub fn close(mut self, keep_recovery: bool) -> Result<(), ProjectError> {
        // A failed writer rolls over first, so the file kept holds every edit.
        self.with_writer(|_, _| Ok(()))?;
        if let Some(writer) = self.writer.take() {
            writer.close(&mut self.store, Instant::now())?;
        }
        if !keep_recovery {
            fs::remove_file(&self.recovery_path)?;
            sync_parent(&self.recovery_path)?;
        }
        Ok(())
    }

    /// Whether the project has edits no save holds: any edit since the last save, or, for a
    /// project never saved, any edit at all.
    #[must_use]
    pub fn has_unsaved_edits(&self) -> bool {
        self.store.has_unsaved_edits()
    }

    /// The tile store, for edits and evaluation.
    pub const fn store(&mut self) -> &mut TileStore {
        &mut self.store
    }

    /// Where the project file is; `None` for a project never saved.
    #[must_use]
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// Where the recovery file is: it changes with every save.
    #[must_use]
    pub fn recovery_path(&self) -> &Path {
        &self.recovery_path
    }

    /// The project's id: the project file's, or for a project never saved, its recovery
    /// file's. A save as gives the project a new one.
    #[must_use]
    pub fn id(&self) -> ProjectId {
        self.store.working().meta().project()
    }

    /// A stroke ended: flushes its edits to the recovery file in the background (see
    /// [`RecoveryWriter::end_stroke`]).
    ///
    /// # Errors
    ///
    /// [`ProjectError::Recovery`] if the recovery writer fails, unless the failure was a sync,
    /// which the project deals with by moving to a new recovery file holding every edit;
    /// [`ProjectError::Unprotected`] for tiles whose edits aren't in the recovery file;
    /// [`ProjectError::Cache`], [`ProjectError::Io`] or [`ProjectError::Store`] if moving to a
    /// new recovery file fails.
    pub fn end_stroke(&mut self) -> Result<(), ProjectError> {
        self.with_writer(|writer, store| writer.end_stroke(store, Instant::now()))
    }

    /// Time passed, to `now`: flushes the edits of a long stroke at least once a second (see
    /// [`RecoveryWriter::tick`]).
    ///
    /// # Errors
    ///
    /// As [`Project::end_stroke`].
    pub fn tick(&mut self, now: Instant) -> Result<(), ProjectError> {
        self.with_writer(|writer, store| writer.tick(store, now))
    }

    /// Advances the recovery writer's flush in flight and installs finished prefetches,
    /// without blocking: for the app's frame loop.
    ///
    /// # Errors
    ///
    /// As [`Project::end_stroke`]; [`ProjectError::Cache`] from installing prefetched tiles.
    pub fn poll(&mut self) -> Result<(), ProjectError> {
        self.with_writer(RecoveryWriter::poll)?;
        self.store.poll()?;
        Ok(())
    }

    /// Flushes every edit into the recovery file and waits until it is durable: what a save
    /// starts with, so no flush is in flight while the store is rebased.
    ///
    /// # Errors
    ///
    /// As [`Project::with_writer`].
    fn finish_flushes(&mut self) -> Result<(), ProjectError> {
        self.with_writer(|writer, store| writer.finish(store, Instant::now()))
    }

    /// Runs `call` on the recovery writer, then deals with what it found.
    ///
    /// If the writer has failed (a sync of the recovery file failed, see [`crate::recovery`]),
    /// the project rolls over to a new recovery file holding every edit
    /// ([`Project::roll_over`]); the failure is then dealt with, and isn't returned. Tiles the
    /// writer left out of its flushes (broken slots), or the roll-over couldn't carry across,
    /// are reported.
    ///
    /// # Errors
    ///
    /// `call`'s error, as [`ProjectError::Recovery`], unless a roll-over dealt with it; as
    /// [`Project::roll_over`]; [`ProjectError::Unprotected`] for tiles whose edits aren't in the
    /// recovery file.
    fn with_writer(
        &mut self,
        call: impl FnOnce(&mut RecoveryWriter, &mut TileStore) -> Result<(), RecoveryError>,
    ) -> Result<(), ProjectError> {
        let writer = self.writer.as_mut().ok_or(RecoveryError::Stopped)?;
        let outcome = call(writer, &mut self.store);
        let mut unprotected = writer.take_skipped();
        if writer.is_failed() {
            unprotected.extend(self.roll_over()?);
        } else {
            outcome?;
        }
        if unprotected.is_empty() {
            Ok(())
        } else {
            Err(ProjectError::Unprotected(unprotected))
        }
    }

    /// Moves the edits to a new recovery file after the writer failed to sync the current one:
    /// the new file (made as at a save: temporary file and rename) names the same save point,
    /// takes every edit from the store ([`TileStore::roll_over`]: synced, a save point,
    /// synced), and becomes the working file; a new writer streams into it, and the old file is
    /// deleted. Returns the tiles that couldn't be carried across.
    ///
    /// # Errors
    ///
    /// As [`create_recovery`]; [`ProjectError::Cache`] if the store can't write the new file
    /// (it is deleted, and the project stays on the old one, failed, to try again);
    /// [`ProjectError::Recovery`] if the new writer's thread can't be started;
    /// [`ProjectError::Io`] or [`ProjectError::Store`] if the old file can't be deleted.
    fn roll_over(&mut self) -> Result<Vec<geom::TileKey>, ProjectError> {
        let meta = *self.store.working().meta();
        let (fresh, fresh_path) = create_recovery(
            &self.recovery_dir,
            meta,
            |at| ProjectFile::create_like(at, meta, self.store.working()),
            #[cfg(test)]
            false,
        )?;
        let (old, lost) = match self.store.roll_over(fresh) {
            Ok(rolled) => rolled,
            Err(e) => {
                // Best effort: the error at hand is the one to report.
                drop(fs::remove_file(&fresh_path));
                return Err(e.into());
            }
        };
        drop(old);
        if let Some(failed) = self.writer.take() {
            // It can only say that it needs a new file; closing releases its subscription.
            drop(failed.close(&mut self.store, Instant::now()));
        }
        self.writer = Some(RecoveryWriter::new(&mut self.store)?);
        let old_path = std::mem::replace(&mut self.recovery_path, fresh_path);
        fs::remove_file(&old_path)?;
        sync_parent(&old_path)?;
        Ok(lost)
    }

    /// After a save into the project file the project holds: [`Project::switch_recovery_to`]
    /// that file.
    ///
    /// # Errors
    ///
    /// As [`Project::switch_recovery_to`]; [`ProjectError::NoPath`] if there is no project file.
    fn switch_recovery(&mut self, saved: Saved) -> Result<(), ProjectError> {
        let project = self.file.take().ok_or(ProjectError::NoPath)?;
        let switched = self.switch_recovery_to(saved, &project);
        self.file = Some(project);
        switched
    }

    /// Creates a fresh recovery file on the save point `saved` proves in `project`, rebases the
    /// store onto it, and deletes the old recovery file, in that order: the old file goes only
    /// once the new one is durable and in use.
    ///
    /// # Errors
    ///
    /// As [`create_recovery`]; [`ProjectError::Cache`] if the rebase is refused (the new file is
    /// deleted again); [`ProjectError::Io`] or [`ProjectError::Store`] if the old file can't be
    /// deleted (the project goes on with the new one).
    fn switch_recovery_to(
        &mut self,
        saved: Saved,
        project: &ProjectFile,
    ) -> Result<(), ProjectError> {
        let meta = project.meta().with_role(Role::Recovery {
            base: Some(saved.save_point()),
        });
        let (fresh, fresh_path) = create_recovery(
            &self.recovery_dir,
            meta,
            |at| ProjectFile::create_like(at, meta, project),
            #[cfg(test)]
            std::mem::take(&mut self.fail_next_recovery),
        )?;
        let old = match self.store.rebase(saved, fresh) {
            Ok(old) => old,
            Err(e) => {
                // Best effort: the file is empty, and the error at hand is the one to report.
                drop(fs::remove_file(&fresh_path));
                return Err(e.into());
            }
        };
        drop(old);
        let old_path = std::mem::replace(&mut self.recovery_path, fresh_path);
        fs::remove_file(&old_path)?;
        sync_parent(&old_path)?;
        Ok(())
    }
}

/// Test-only controls.
#[cfg(test)]
impl Project {
    /// Makes the next recovery file's creation fail once its temporary file is written and
    /// synced, before the rename: what a crash mid-create leaves.
    pub(crate) const fn fail_next_recovery(&mut self) {
        self.fail_next_recovery = true;
    }
}

/// Closes the recovery writer on the store: its last flushes are finished as far as they can
/// be, and its subscription to the change feed is released. The recovery file is kept.
impl Drop for Project {
    fn drop(&mut self) {
        if let Some(writer) = self.writer.take() {
            // Nothing to report to; the recovery file holds whatever was durable.
            drop(writer.close(&mut self.store, Instant::now()));
        }
    }
}

/// Creates a recovery file with `meta` in `dir` through `create`, under a fresh name (see the
/// module notes), and returns it open with its path.
///
/// `create` makes the file at the temporary path it is given, durable; it is then renamed into
/// place and the directory synced. On error the temporary file is removed.
///
/// # Errors
///
/// [`ProjectError::Io`] if the directory can't be made, a random suffix drawn, or the rename
/// done; `create`'s error.
fn create_recovery(
    dir: &Path,
    meta: Meta,
    create: impl FnOnce(&Path) -> Result<ProjectFile, StoreError>,
    #[cfg(test)] fail: bool,
) -> Result<(ProjectFile, PathBuf), ProjectError> {
    fs::create_dir_all(dir)?;
    let suffix = ProjectId::random()?.to_bits() >> 64;
    let name = format!("{}-{suffix:016x}.{EXTENSION}", meta.project());
    let path = dir.join(&name);
    let tmp = dir.join(format!("{name}.tmp"));
    let file = create(&tmp)?;
    let placed = (|| {
        #[cfg(test)]
        if fail {
            return Err(io::Error::other("a crash before the rename, for the test").into());
        }
        if fs::symlink_metadata(&path).is_ok() {
            return Err(io::Error::from(io::ErrorKind::AlreadyExists).into());
        }
        fs::rename(&tmp, &path)?;
        sync_parent(&path)?;
        Ok(())
    })();
    match placed {
        Ok(()) => Ok((file, path)),
        Err(e) => {
            #[cfg(test)]
            if fail {
                // A crash leaves the synced temporary file behind: so does the test.
                return Err(e);
            }
            drop(file);
            // Best effort: the error at hand is the one to report.
            drop(fs::remove_file(&tmp));
            Err(e)
        }
    }
}

/// A temporary path beside `path`, for a project file written before it is renamed there:
/// `<name>.<random>.tmp`.
///
/// # Errors
///
/// [`ProjectError::Io`] if `path` has no file name or no random suffix can be drawn.
fn temporary(path: &Path) -> Result<PathBuf, ProjectError> {
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

#[cfg(test)]
mod tests;
