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
//! save point of the project file: the old one until the new one is in place. Once the store is
//! rebased, the save has happened, whatever follows: deleting the old file is best effort, and a
//! failure to is reported ([`ProjectError::OldRecoveryKept`]), never fatal.
//!
//! # Recovery file names
//!
//! A recovery file is named `<project id>-<suffix>.bcrf`, the id as 32 hex digits and the
//! suffix 16 random ones, so the old and the new file of one project coexist during a save. It
//! is created as `<name>.tmp`, synced, and moved into place: a crash mid-create leaves at
//! most a `.tmp` file, never a half-made file under a recovery file's name. A roll-over's new
//! file is filled with every edit and saved before the rename, so it never has its name while
//! it lacks the edits of the file it replaces; a save's new file is empty by rights (the save
//! holds every edit), and is renamed into place before the store moves to it.
//!
//! # After a crash
//!
//! A project's recovery file outlives a crash, or a close that keeps it. [`find_recoveries`]
//! lists the recovery files no running instance holds, each with the project file it extends
//! and whether it can be layered back on it ([`CandidateStatus`]): only on the very save point
//! it names, the project id, the save point's offset **and** its head op matching. Every save
//! is kept as a save point, so a project saved since still takes it, on that earlier save point
//! (the restored project then saves as a new file); a project file deleted, replaced or
//! rewritten can't, and the file is listed so the app can say so, never layered silently.
//! [`Project::restore`] opens a restorable one as the project, the recovered edits unsaved and
//! the recovery file going on as its own.
//!
//! What a restore gets is the recovery file's last durable save point. Whatever was appended
//! after it, a flush cut short or tiles of the stroke in progress written back to make room,
//! is discarded: a save point is the unit of durability, and replaying the tail could bring
//! back half a stroke. At worst, the stroke in progress is lost.

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use std::{fs, io};

use geom::BaseMesh;
use thiserror::Error;

use crate::OpId;
use crate::file::{BaseFile, ProjectFile, StoreError, sync_parent};
use crate::meta::{Meta, ProjectId, Role, SavePoint};
use crate::recovery::{RecoveryError, RecoveryWriter};
use crate::tiles::{CacheError, Saved, StoreConfig, TileStore};

/// The extension of a recovery file (and of a project file).
const EXTENSION: &str = "bcrf";

/// How long a project first waits to try again to move to a new recovery file.
///
/// The move follows a failed sync of the recovery file; if it fails too, it is tried again
/// after this wait, which doubles with each failure in a row. [`Project::poll`],
/// [`Project::tick`] and [`Project::end_stroke`] wait; [`Project::save`] and [`Project::close`]
/// try at once.
pub const ROLL_OVER_RETRY: Duration = Duration::from_secs(1);

/// The longest a project waits between two tries of a failing roll-over.
const ROLL_OVER_RETRY_MAX: Duration = Duration::from_secs(60);

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
    /// [`Project::save`] on a project with no path: never saved, so it has no project file yet,
    /// or restored onto an earlier save point of a project file that has moved on since. Use
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
    /// Reported by a save, they aren't in the project file either (the save couldn't read
    /// them): it holds their last saved state. The call that reports them otherwise did what it
    /// was asked; a save went through.
    #[error("the edits of {} tiles are not in the recovery file", .0.len())]
    Unprotected(Vec<geom::TileKey>),
    /// [`Project::restore`] was given a recovery file that can't be layered on its project:
    /// the status it was listed with, or the one found when the files were opened to restore
    /// it. Nothing was written, but opening them may have cut off their torn tails, which
    /// every open for writing does.
    #[error("the recovery file can't be restored: {0:?}")]
    NotRestorable(CandidateStatus),
    /// [`Project::restore`] found another file under the candidate's path than the one listed:
    /// another project's, or one on another save point. List again.
    #[error("the recovery file changed since it was listed")]
    CandidateChanged,
    /// The call did what it was asked (a save, say: the project goes on with its new recovery
    /// file), but the old recovery file at `path` couldn't be deleted. It holds nothing the
    /// project needs any more; until it is deleted, [`find_recoveries`] lists it, on the save
    /// point it was made on.
    #[error("the old recovery file {path:?} couldn't be deleted")]
    OldRecoveryKept {
        /// Where the old recovery file is.
        path: PathBuf,
        /// Why it couldn't be deleted.
        #[source]
        source: io::Error,
    },
}

/// A recovery file found by [`find_recoveries`]: what the app shows to offer a restore.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecoveryCandidate {
    /// Where the recovery file is.
    pub path: PathBuf,
    /// The project it belongs to; for a damaged file, the id its name gives.
    pub project: ProjectId,
    /// The save point of the project file it is layered on; `None` for a project never saved,
    /// or a damaged file.
    pub base: Option<SavePoint>,
    /// Where that project file was when the recovery file was made; `None` as for `base`.
    pub base_path: Option<PathBuf>,
    /// The newest edit a restore brings back: the recovery file's head op at its last durable
    /// save point. `None` if it holds no edits (or is damaged): restoring it gives the save
    /// point alone, or for a project never saved, an empty project.
    pub last_edit: Option<OpId>,
    /// When the recovery file was last written to.
    pub modified: SystemTime,
    /// Whether it can be restored, and if not, why.
    pub status: CandidateStatus,
}

/// Whether a [`RecoveryCandidate`] can be restored ([`Project::restore`]), and if not, why.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CandidateStatus {
    /// The recovery file of a project never saved: it holds all of the project's work, and
    /// restores on its own.
    NeverSaved,
    /// The project file is where the recovery file says, and its latest save point is the one
    /// the recovery file names: the recovered edits layer on it, and the restored project
    /// saves into the project file as usual.
    Restorable,
    /// The project file is where the recovery file says and still has the save point it names,
    /// but has been saved since (by another instance, or by a save the crash cut short after
    /// its save point was durable): the recovered edits layer on that earlier save point,
    /// which every save keeps. Saving into the project file would mix them with the later
    /// saves, so the restored project has no path and saves as a new file
    /// ([`Project::save_as`]).
    RestorableOnEarlierSave,
    /// No project file of this project is where the recovery file says: it was moved, deleted,
    /// or replaced by another project (or something else).
    ProjectMissing,
    /// The project file is there, but has no save point with the offset **and** head op the
    /// recovery file names: it was rewritten (compacted, or replaced by another copy of the
    /// project). The edits would be read through the wrong saved state.
    SavePointMissing,
    /// The project file is open in a running instance, which holds it: it can't be checked or
    /// layered on now.
    ProjectInUse,
    /// The recovery file, or the project file it extends, can't be read: its leading records
    /// are damaged, or it is damaged before a good save point.
    Damaged,
}

/// When [`Project::with_writer`] tries a failed writer's roll-over.
#[derive(Debug, Clone, Copy)]
enum Retry {
    /// At once: for a save or a close, which need every edit in a recovery file.
    Now,
    /// Only once the wait after the last failed try has passed, it being this instant now:
    /// for the frame loop's calls, which come every few milliseconds.
    Due(Instant),
}

/// Where a project's recovery file came from, which decides what happens to it if the project
/// fails to start.
#[derive(Debug)]
enum Recovery {
    /// Just made, empty: deleted if the project fails to start.
    Fresh(PathBuf),
    /// Restored after a crash, holding the user's work: kept whatever happens.
    Restored(PathBuf),
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
    /// Where saves go: the project file's path. `None` for a project never saved, or one
    /// restored onto an earlier save point of its project file, whose file is held (its base
    /// reads it) but not saved into.
    path: Option<PathBuf>,
    /// Where the store's working file, the recovery file, is.
    recovery_path: PathBuf,
    /// The directory recovery files go in.
    recovery_dir: PathBuf,
    /// Tiles whose latest edits were found unprotected (left out of a flush, or lost to a
    /// roll-over or a save), waiting to be reported: by the next call that has no other error
    /// to return ([`Project::report`]). So a report never stops a call from doing its work.
    unreported: BTreeSet<geom::TileKey>,
    /// When a failed writer's roll-over may be tried again; `None` to try at once. A frame loop
    /// polls every few milliseconds, and a roll-over that keeps failing (a full disk, say) is
    /// tried only this often; a save or a close tries at once.
    retry_at: Option<Instant>,
    /// How long the next failed roll-over makes the project wait before trying again:
    /// [`ROLL_OVER_RETRY`], doubling up to a minute while the failures go on.
    retry_after: Duration,
    /// Test hook: makes the next placement of a recovery file a save or a roll-over makes fail
    /// after its temporary file is written and synced, before the rename (see
    /// [`Unplaced::crash`]).
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
        let (working, recovery_path) = create_placed(&config.recovery_dir, meta.project(), |at| {
            ProjectFile::create(at, meta, Arc::clone(&mesh))
        })?;
        Self::start(
            TileStore::open(working, config.store),
            None,
            Recovery::Fresh(recovery_path),
            config,
        )
    }

    /// Opens the project file at `path` at its last save point, with a fresh recovery file
    /// layered on it.
    ///
    /// The project file is opened for writing and locked, so no other instance can open it
    /// meanwhile; whatever followed its last save point (a save cut short) is discarded.
    ///
    /// The project's ops start past the newest edit of every recovery file of the project in
    /// the recovery directory, whatever save point it is on: a session's
    /// edits are newer than every earlier session's, kept after a crash or not, so of two
    /// recovery files on one save point the one with the newest last edit is the latest work.
    ///
    /// # Errors
    ///
    /// [`ProjectError::Store`] if the file can't be opened (including
    /// [`StoreError::Locked`]); [`ProjectError::NotAProject`]; [`ProjectError::NeverSaved`];
    /// as [`Project::new`] for the recovery file, the store and the writer.
    pub fn open(path: &Path, config: ProjectConfig) -> Result<Self, ProjectError> {
        let path = std::path::absolute(path)?;
        let project = ProjectFile::open(&path)?;
        if project.meta().role() != Role::Project {
            return Err(ProjectError::NotAProject);
        }
        let base = project.base()?.ok_or(ProjectError::NeverSaved)?;
        let meta = project.meta().on_save_point(base.save_point(), &path);
        let earlier = recovery_heads(&config.recovery_dir, meta.project());
        let (working, recovery_path) = create_placed(&config.recovery_dir, meta.project(), |at| {
            ProjectFile::create_like(at, meta, &project)
        })?;
        let store = past(
            TileStore::open_layered(working, Some(base), config.store),
            earlier,
        );
        Self::start(
            store,
            Some((project, path)),
            Recovery::Fresh(recovery_path),
            config,
        )
    }

    /// Restores the project of a recovery file found by [`find_recoveries`]: a
    /// [`CandidateStatus::Restorable`] or [`CandidateStatus::RestorableOnEarlierSave`] one
    /// layered on its project file at the save point it names, a
    /// [`CandidateStatus::NeverSaved`] one on its own. The recovery file goes on as the
    /// project's, its edits unsaved (the next save copies them), and is kept until the project
    /// closes without keeping it.
    ///
    /// Everything the candidate was listed with is checked again under the locks a project
    /// holds, as the files may have changed since: the project file must hold the very save
    /// point the recovery file names (project id, offset and head op). If that is its latest,
    /// the project file is opened for writing (so no other instance writes it meanwhile), and
    /// the project saves into it as usual. If it has been saved since, the project file is held
    /// read-only, with a shared lock, as it is never written: the restored project reads that
    /// earlier save point, has no path ([`Project::path`] is `None`), and saves as a new file.
    /// Other readers may open the project file meanwhile, but no instance can write it.
    ///
    /// The recovery file opens at its last durable save point; whatever followed it (a flush
    /// cut short, tiles of the stroke in progress) is discarded. The project's ops start past
    /// the newest edit of every recovery file of the project, as [`Project::open`]'s do.
    ///
    /// # Errors
    ///
    /// [`ProjectError::NotRestorable`] with the candidate's status if it is neither of the two,
    /// or with the status the open files show if that has changed (the project file was saved,
    /// moved or deleted since, or is open elsewhere); [`ProjectError::CandidateChanged`] if
    /// another file is under the candidate's path; [`ProjectError::Store`] if either file
    /// can't be opened (the recovery file is held by a running instance:
    /// [`StoreError::Locked`]); [`ProjectError::Cache`] if the store refuses `config.store`;
    /// [`ProjectError::Recovery`] if the writer's thread can't be started. The recovery file is
    /// kept on error.
    pub fn restore(
        candidate: &RecoveryCandidate,
        config: ProjectConfig,
    ) -> Result<Self, ProjectError> {
        let recovery = Recovery::Restored(candidate.path.clone());
        match candidate.status {
            CandidateStatus::NeverSaved => {
                let working = ProjectFile::open(&candidate.path)?;
                let fits = working.meta().project() == candidate.project
                    && working.meta().role() == (Role::Recovery { base: None });
                if !fits {
                    return Err(ProjectError::CandidateChanged);
                }
                let earlier = recovery_heads(&config.recovery_dir, candidate.project);
                Self::start(
                    past(TileStore::open(working, config.store), earlier),
                    None,
                    recovery,
                    config,
                )
            }
            CandidateStatus::Restorable | CandidateStatus::RestorableOnEarlierSave => {
                let path = candidate
                    .base_path
                    .clone()
                    .ok_or(ProjectError::NotRestorable(CandidateStatus::ProjectMissing))?;
                // Looked at first, read-only: a restore onto an earlier save point never writes
                // the project file, and holds it shared, as any reader does.
                let looked = ProjectFile::inspect(&path).map_err(not_restorable)?;
                let working = ProjectFile::open_sharing_mesh(&candidate.path, &looked)?;
                let meta = working.meta();
                let fits = meta.project() == candidate.project
                    && meta.role()
                        == (Role::Recovery {
                            base: candidate.base,
                        })
                    && meta.base_path() == Some(path.as_path());
                if !fits {
                    return Err(ProjectError::CandidateChanged);
                }
                let (project, base, saves_into) = match standing(meta, &looked)? {
                    (CandidateStatus::RestorableOnEarlierSave, base) => (looked, base, None),
                    (CandidateStatus::Restorable, base) => {
                        // Saved into, so held for writing: opened again, its lock released
                        // first, and checked again, as it may have been saved meanwhile. Then
                        // it is held for writing all the same, but saved only as a new file.
                        drop(base);
                        drop(looked);
                        let project = ProjectFile::open(&path).map_err(not_restorable)?;
                        match standing(working.meta(), &project)? {
                            (CandidateStatus::Restorable, rechecked) => {
                                (project, rechecked, Some(path))
                            }
                            (CandidateStatus::RestorableOnEarlierSave, rechecked) => {
                                (project, rechecked, None)
                            }
                            (status, _) => return Err(ProjectError::NotRestorable(status)),
                        }
                    }
                    (status, _) => return Err(ProjectError::NotRestorable(status)),
                };
                let earlier = recovery_heads(&config.recovery_dir, candidate.project);
                let store = past(
                    TileStore::open_layered(working, base, config.store),
                    earlier,
                );
                Self::start_with(store, Some(project), saves_into, recovery, config)
            }
            status @ (CandidateStatus::ProjectMissing
            | CandidateStatus::SavePointMissing
            | CandidateStatus::ProjectInUse
            | CandidateStatus::Damaged) => Err(ProjectError::NotRestorable(status)),
        }
    }

    /// Starts the recovery writer on `store` and assembles the project, its recovery file at
    /// `recovery`; on error, deletes that file if it is a fresh one, which nothing was written
    /// to, and keeps it if it was restored.
    ///
    /// # Errors
    ///
    /// `store`'s error; [`ProjectError::Recovery`] if the writer's thread can't be started.
    fn start(
        opened: Result<TileStore, CacheError>,
        project: Option<(ProjectFile, PathBuf)>,
        recovery: Recovery,
        config: ProjectConfig,
    ) -> Result<Self, ProjectError> {
        let (file, path) = project.unzip();
        Self::start_with(opened, file, path, recovery, config)
    }

    /// [`Project::start`], with the project file and the path saves go to given apart: a
    /// project restored onto an earlier save point holds its project file (its base reads it)
    /// but saves only as a new file.
    ///
    /// # Errors
    ///
    /// As [`Project::start`].
    fn start_with(
        opened: Result<TileStore, CacheError>,
        file: Option<ProjectFile>,
        path: Option<PathBuf>,
        recovery: Recovery,
        config: ProjectConfig,
    ) -> Result<Self, ProjectError> {
        let started = opened.map_err(ProjectError::from).and_then(|mut store| {
            let writer = RecoveryWriter::new(&mut store)?;
            Ok((writer, store))
        });
        let (writer, store) = match started {
            Ok(started) => started,
            Err(e) => {
                if let Recovery::Fresh(fresh) = &recovery {
                    // Best effort: the file is empty, and the error at hand is the one to report.
                    drop(fs::remove_file(fresh));
                }
                return Err(e);
            }
        };
        let (Recovery::Fresh(recovery_path) | Recovery::Restored(recovery_path)) = recovery;
        Ok(Self {
            writer: Some(writer),
            store,
            file,
            path,
            recovery_path,
            recovery_dir: config.recovery_dir,
            unreported: BTreeSet::new(),
            retry_at: None,
            retry_after: ROLL_OVER_RETRY,
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
    /// holds every edit; saving again works. [`ProjectError::OldRecoveryKept`] if the save
    /// went through, but the old recovery file couldn't be deleted. [`ProjectError::Unprotected`]
    /// after the save went through, for tiles found unprotected before it or lost to it: a
    /// report never stops a save.
    pub fn save(&mut self) -> Result<(), ProjectError> {
        self.advance()?;
        if self.path.is_none() {
            return Err(ProjectError::NoPath);
        }
        self.finish_flushes()?;
        let path = self.path.clone().ok_or(ProjectError::NoPath)?;
        let mut project = self.file.take().ok_or(ProjectError::NoPath)?;
        let switched = self
            .store
            .save_into(&mut project)
            .map_err(ProjectError::from)
            .and_then(|saved| {
                self.unreported.extend(saved.lost());
                self.switch_recovery(saved, &project, &path)
            });
        self.file = Some(project);
        let old = switched?;
        retire(&old)?;
        self.report()
    }

    /// Saves the project as a new project file at `path`, with a new project id, and goes on
    /// with that one: every tile of the last save point copied (as records, not re-encoded),
    /// then the edits since, then a save point. The old project file isn't written to.
    ///
    /// The file is written as a temporary file beside `path` and moved into place once its
    /// save point is durable, so `path` never names a half-written project. The move never
    /// replaces a file (a hard link, not a rename), not even one made at `path` while the
    /// project was being written.
    ///
    /// # Errors
    ///
    /// [`ProjectError::Exists`] if `path` names a file already, or does by the time the new
    /// file is to be moved there (it is removed then); [`ProjectError::Recovery`] if
    /// the recovery writer can't finish its flushes; [`ProjectError::Store`] or
    /// [`ProjectError::Cache`] if creating or writing the new file fails (the temporary file is
    /// removed);
    /// [`ProjectError::Io`] if the rename does; as [`Project::save`] for the fresh recovery
    /// file, where the new project file is in place but the project goes on with the old one.
    /// Once the store is rebased onto the new file, the project goes on with it, its path and
    /// its id, whatever happens next: [`ProjectError::OldRecoveryKept`] if the old recovery file
    /// couldn't be deleted. [`ProjectError::Unprotected`] as for [`Project::save`].
    pub fn save_as(&mut self, path: &Path) -> Result<(), ProjectError> {
        self.advance()?;
        let path = std::path::absolute(path)?;
        if fs::symlink_metadata(&path).is_ok() {
            return Err(ProjectError::Exists(path));
        }
        self.finish_flushes()?;
        let meta = self
            .store
            .working()
            .meta()
            .with_project(ProjectId::random()?)
            .with_role(Role::Project);
        let tmp = temporary(&path)?;
        let mut project = match ProjectFile::create_like(&tmp, meta, self.store.working()) {
            Ok(project) => project,
            Err(e) => {
                // Made, then failed: removed, best effort. A file already there isn't this
                // save's, and is left alone.
                if !matches!(&e, StoreError::Io(io) if io.kind() == io::ErrorKind::AlreadyExists) {
                    drop(fs::remove_file(&tmp));
                }
                return Err(e.into());
            }
        };
        let written = self.write_new_project(&mut project, &tmp, &path);
        let saved = match written {
            Ok(saved) => saved,
            Err(e) => {
                // Best effort: the error at hand is the one to report.
                drop(fs::remove_file(&tmp));
                return Err(e);
            }
        };
        // The new project file lacks them whatever happens next, as `save` reports them.
        self.unreported.extend(saved.lost());
        let old = self.switch_recovery(saved, &project, &path)?;
        // The switch has happened: the store reads the new project file, so the project holds
        // it, and saves go to it. The old project file closes here, and its lock goes with it.
        self.file = Some(project);
        self.path = Some(path);
        retire(&old)?;
        self.report()
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
        place_new(tmp, path).map_err(|e| {
            if e.kind() == io::ErrorKind::AlreadyExists {
                ProjectError::Exists(path.to_path_buf())
            } else {
                e.into()
            }
        })?;
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
    /// The last flushes go through the roll-over as a save's do: a writer that has failed, or
    /// whose last sync fails, moves every edit to a new recovery file at once, and that is the
    /// file kept (or deleted).
    ///
    /// # Errors
    ///
    /// [`ProjectError::Recovery`] if the last flushes fail, or as [`Project::roll_over`] if
    /// moving to a new recovery file does: the recovery file is kept then, whatever
    /// `keep_recovery` says. [`ProjectError::Io`] or [`ProjectError::Store`] if it can't be
    /// deleted. [`ProjectError::Unprotected`] once closed, if a kept recovery file lacks the
    /// latest edits of some tiles.
    pub fn close(mut self, keep_recovery: bool) -> Result<(), ProjectError> {
        let finished = self.finish_flushes();
        if let Some(mut writer) = self.writer.take() {
            self.unreported.extend(writer.take_skipped());
            // Its flushes are done, or failed for good: closing releases its subscription.
            let closed = writer.close(&mut self.store, Instant::now());
            finished?;
            closed?;
        } else {
            finished?;
        }
        if !keep_recovery {
            fs::remove_file(&self.recovery_path)?;
            sync_parent(&self.recovery_path)?;
            // Nothing is kept to lack them.
            self.unreported.clear();
        }
        self.report()
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

    /// Where the project file saves go to is; `None` for a project never saved, or one
    /// restored onto an earlier save point ([`CandidateStatus::RestorableOnEarlierSave`]),
    /// which saves as a new file.
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

    /// A stroke begins: starts a new op in the store ([`TileStore::begin_op`]), which every
    /// edit of the stroke is part of, however many dabs it takes, and returns it.
    ///
    /// # Errors
    ///
    /// [`ProjectError::Cache`] with [`CacheError::OpsExhausted`] if the site's ops are used up.
    pub fn begin_stroke(&mut self) -> Result<OpId, ProjectError> {
        Ok(self.store.begin_op()?)
    }

    /// A stroke ended: flushes its edits to the recovery file in the background (see
    /// [`RecoveryWriter::end_stroke`]).
    ///
    /// # Errors
    ///
    /// [`ProjectError::Recovery`] if the recovery writer fails, unless the failure was a sync,
    /// which the project deals with by moving to a new recovery file holding every edit;
    /// [`ProjectError::Cache`], [`ProjectError::Io`] or [`ProjectError::Store`] if moving to a
    /// new recovery file fails (it is tried again after [`ROLL_OVER_RETRY`], then waits
    /// doubling up to a minute while it keeps failing); [`ProjectError::Unprotected`] for tiles
    /// whose edits aren't in the recovery file, once nothing else is to be reported.
    pub fn end_stroke(&mut self) -> Result<(), ProjectError> {
        let now = Instant::now();
        self.with_writer(
            |writer, store| writer.end_stroke(store, now),
            Retry::Due(now),
        )?;
        self.report()
    }

    /// Time passed, to `now`: flushes the edits of a long stroke at least once a second (see
    /// [`RecoveryWriter::tick`]).
    ///
    /// # Errors
    ///
    /// As [`Project::end_stroke`].
    pub fn tick(&mut self, now: Instant) -> Result<(), ProjectError> {
        self.with_writer(|writer, store| writer.tick(store, now), Retry::Due(now))?;
        self.report()
    }

    /// Advances the recovery writer's flush in flight and installs finished prefetches,
    /// without blocking: for the app's frame loop.
    ///
    /// # Errors
    ///
    /// As [`Project::end_stroke`]; [`ProjectError::Cache`] from installing prefetched tiles.
    pub fn poll(&mut self) -> Result<(), ProjectError> {
        self.advance()?;
        self.report()
    }

    /// [`Project::poll`] without the report: what a save starts with, before it saves, so a
    /// report waits until the save is made.
    ///
    /// # Errors
    ///
    /// As [`Project::with_writer`]; [`ProjectError::Cache`] from installing prefetched tiles.
    fn advance(&mut self) -> Result<(), ProjectError> {
        self.with_writer(RecoveryWriter::poll, Retry::Due(Instant::now()))?;
        self.store.poll()?;
        Ok(())
    }

    /// Flushes every edit into the recovery file and waits until it is durable: what a save
    /// starts with, so no flush is in flight while the store is rebased. A failed writer's
    /// roll-over is tried at once.
    ///
    /// # Errors
    ///
    /// As [`Project::with_writer`].
    fn finish_flushes(&mut self) -> Result<(), ProjectError> {
        self.with_writer(
            |writer, store| writer.finish(store, Instant::now()),
            Retry::Now,
        )
    }

    /// Reports the tiles found unprotected since the last report, if any, and forgets them.
    ///
    /// # Errors
    ///
    /// [`ProjectError::Unprotected`] with those tiles, in key order.
    fn report(&mut self) -> Result<(), ProjectError> {
        if self.unreported.is_empty() {
            return Ok(());
        }
        let tiles = std::mem::take(&mut self.unreported);
        Err(ProjectError::Unprotected(tiles.into_iter().collect()))
    }

    /// Runs `call` on the recovery writer, then deals with what it found.
    ///
    /// Tiles the writer left out of its flushes (broken slots) wait to be reported
    /// ([`Project::report`]). If the writer has failed (a sync of the recovery file failed, see
    /// [`crate::recovery`]), the project rolls over to a new recovery file holding every edit
    /// ([`Project::roll_over`]), and the failure, dealt with, isn't returned. A roll-over that
    /// fails is tried again by the next call, at once if `retry` says so, else only once a wait
    /// has passed: [`ROLL_OVER_RETRY`] after the first failure, doubling up to a minute while
    /// they go on. Until then the edits wait in memory.
    ///
    /// # Errors
    ///
    /// `call`'s error, as [`ProjectError::Recovery`], unless the writer has failed; as
    /// [`Project::roll_over`].
    fn with_writer(
        &mut self,
        call: impl FnOnce(&mut RecoveryWriter, &mut TileStore) -> Result<(), RecoveryError>,
        retry: Retry,
    ) -> Result<(), ProjectError> {
        let writer = self.writer.as_mut().ok_or(RecoveryError::Stopped)?;
        let outcome = call(writer, &mut self.store);
        self.unreported.extend(writer.take_skipped());
        if !writer.is_failed() {
            return Ok(outcome?);
        }
        let now = match retry {
            Retry::Due(now) if self.retry_at.is_some_and(|at| now < at) => return Ok(()),
            Retry::Due(now) => now,
            Retry::Now => Instant::now(),
        };
        let rolled = self.roll_over();
        if self.writer.as_ref().is_some_and(RecoveryWriter::is_failed) {
            self.retry_at = Some(now + self.retry_after);
            self.retry_after = (self.retry_after * 2).min(ROLL_OVER_RETRY_MAX);
        } else {
            self.retry_at = None;
            self.retry_after = ROLL_OVER_RETRY;
        }
        rolled
    }

    /// Moves the edits to a new recovery file after the writer failed to sync the current one.
    /// The tiles that couldn't be carried across wait to be reported ([`Project::report`]),
    /// whatever happens after the move.
    ///
    /// The new file names the same save point. It is made under its temporary name, filled with
    /// every edit from the store ([`TileStore::fill_roll_over`]: synced, a save point, synced),
    /// and only then renamed into place, so a crash meanwhile leaves no file under a recovery
    /// file's name that lacks the edits of the old one. Then the store swaps it in
    /// ([`TileStore::roll_over`]), a new writer streams into it, and the old file is deleted.
    ///
    /// # Errors
    ///
    /// As [`create_recovery`] and [`Unplaced::place`]; [`ProjectError::Cache`] if the store
    /// can't write the new file. The new file is deleted then, and the project stays on the old
    /// one, failed, to try again. [`ProjectError::Recovery`] if the new writer's thread can't
    /// be started; [`ProjectError::OldRecoveryKept`] if the old file can't be deleted.
    fn roll_over(&mut self) -> Result<(), ProjectError> {
        let meta = self.store.working().meta().clone();
        let (fresh, names) = create_recovery(&self.recovery_dir, meta.project(), |at| {
            ProjectFile::create_like(at, meta, self.store.working())
        })?;
        #[cfg(test)]
        let names = names.crashing(std::mem::take(&mut self.fail_next_recovery));
        let filled = match self.store.fill_roll_over(fresh) {
            Ok(filled) => filled,
            Err(e) => {
                names.discard();
                return Err(e.into());
            }
        };
        let fresh_path = names.place()?;
        let (old, lost) = match self.store.roll_over(filled) {
            Ok(rolled) => rolled,
            Err(e) => {
                // Best effort: the error at hand is the one to report.
                drop(fs::remove_file(&fresh_path));
                return Err(e.into());
            }
        };
        drop(old);
        self.unreported.extend(lost);
        if let Some(failed) = self.writer.take() {
            // It can only say that it needs a new file; closing releases its subscription.
            drop(failed.close(&mut self.store, Instant::now()));
        }
        let old_path = std::mem::replace(&mut self.recovery_path, fresh_path);
        self.writer = Some(RecoveryWriter::new(&mut self.store)?);
        retire(&old_path)
    }

    /// Creates a fresh recovery file on the save point `saved` proves in `project`, the project
    /// file at `path`, and rebases the store onto the pair; returns the old recovery file's
    /// path, for the caller to delete ([`retire`]) once it holds `project`: the old
    /// file goes only once the new one is durable and in use.
    ///
    /// Once this returns, the switch has happened: the store reads `project` and writes the
    /// new recovery file. On error nothing has changed.
    ///
    /// # Errors
    ///
    /// As [`create_recovery`] and [`Unplaced::place`]; [`ProjectError::Cache`] if the rebase is
    /// refused (the new file is deleted again).
    fn switch_recovery(
        &mut self,
        saved: Saved,
        project: &ProjectFile,
        path: &Path,
    ) -> Result<PathBuf, ProjectError> {
        let meta = project.meta().on_save_point(saved.save_point(), path);
        let (fresh, names) = create_recovery(&self.recovery_dir, meta.project(), |at| {
            ProjectFile::create_like(at, meta, project)
        })?;
        #[cfg(test)]
        let names = names.crashing(std::mem::take(&mut self.fail_next_recovery));
        // Empty, and on the new save point: a crash from here on leaves it as a valid recovery
        // file of that save point, holding nothing since. So it is placed before the rebase,
        // not filled first as a roll-over's is.
        let fresh_path = names.place()?;
        let old = match self.store.rebase(saved, fresh) {
            Ok(old) => old,
            Err(e) => {
                // Best effort: the file is empty, and the error at hand is the one to report.
                drop(fs::remove_file(&fresh_path));
                return Err(e.into());
            }
        };
        drop(old);
        Ok(std::mem::replace(&mut self.recovery_path, fresh_path))
    }
}

/// Deletes the old recovery file at `old`, which its project has moved on from, and syncs its
/// directory.
///
/// Best effort: the project needs nothing in it, so a failure is reported, never fatal; a file
/// already gone is fine.
///
/// # Errors
///
/// [`ProjectError::OldRecoveryKept`] if it can't be deleted, or the deletion made durable.
fn retire(old: &Path) -> Result<(), ProjectError> {
    let kept = |source| ProjectError::OldRecoveryKept {
        path: old.to_path_buf(),
        source,
    };
    match fs::remove_file(old) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(kept(e)),
    }
    sync_parent(old).map_err(|e| {
        if let StoreError::Io(io) = e {
            kept(io)
        } else {
            kept(io::Error::other(e))
        }
    })
}

/// Test-only controls.
#[cfg(test)]
impl Project {
    /// Makes the next placement of a recovery file fail once its temporary file is written and
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

/// `opened`, its op generator moved past `heads` ([`TileStore::skip_ops_past`]).
///
/// # Errors
///
/// `opened`'s error; [`CacheError::OpsExhausted`] if a head is the site's last op id.
fn past(
    opened: Result<TileStore, CacheError>,
    heads: Vec<Option<OpId>>,
) -> Result<TileStore, CacheError> {
    let mut store = opened?;
    store.skip_ops_past(heads)?;
    Ok(store)
}

/// The head ops of the recovery files of `project` in `dir`, on any save point: the newest
/// edit each would restore, which a new session's ops must be newer than.
///
/// Best effort, and read-only: a missing or unreadable `dir` has none, and a file a running
/// instance holds, or one that can't be read, is passed over. A file that can't be read is
/// listed as damaged, with no last edit, so it never competes with this session's; one held
/// is this session's own (a restored file), or another instance's, which only one restored
/// onto an earlier save point can be, as the project file is locked.
fn recovery_heads(dir: &Path, project: ProjectId) -> Vec<Option<OpId>> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    entries
        .filter_map(Result::ok)
        .filter(|entry| entry.file_name().to_str().and_then(recovery_name) == Some(project))
        .filter_map(|entry| ProjectFile::inspect(&entry.path()).ok())
        .filter(|file| file.meta().project() == project)
        .map(|file| file.head())
        .collect()
}

/// How long a recovery file's temporary file must have gone unwritten to be a crash's leftover.
///
/// [`find_recoveries`] removes `<name>.tmp` files older than this: an instance creating one
/// writes and syncs it in well under it.
const STALE_TEMPORARY: Duration = Duration::from_secs(60);

/// Lists the recovery files in `dir` that no running instance holds, each with whether it can
/// be restored ([`Project::restore`]) and, if not, why (see the module notes).
///
/// The list is ordered for choosing: projects with the most recently written recovery file
/// first, and a project's files on one save point together, **the one with the newest last
/// edit first** (then the most recently written). That one is the file to restore. One project
/// can leave several files on one save point: sessions that each opened it at that save point
/// and crashed, or a crash mid-roll-over, which leaves the old file and the new one holding the
/// same edits. Ops only grow across sessions, as every session's start past the newest edit of
/// every recovery file of the project ([`Project::open`], [`Project::restore`]), so the newest
/// last edit is the latest session's latest work. A write time can't say that: copying or
/// touching a file moves it, and a clock can be set back.
///
/// Only files named as recovery files are looked at (`<project id>-<suffix>.bcrf`). A file a
/// running instance holds is skipped: it is locked for writing, and a shared lock probe fails.
/// Nothing listed is written, not even a torn tail cut off: each file is opened to look at,
/// read-only, at its last durable save point, and so is the project file a recovery file
/// names. A missing `dir` has no recovery files.
///
/// Temporary files a crash left mid-create (`<recovery file name>.tmp`, never renamed into
/// place, so never holding edits) are removed, best effort: those unwritten for
/// a minute and not locked by an instance still making them.
///
/// # Errors
///
/// [`ProjectError::Io`] if `dir` can't be listed, or a file's metadata read.
pub fn find_recoveries(dir: &Path) -> Result<Vec<RecoveryCandidate>, ProjectError> {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e.into()),
    };
    let mut found = Vec::new();
    for entry in entries {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if let Some(made) = name.strip_suffix(".tmp")
            && recovery_name(made).is_some()
        {
            sweep(&entry.path());
            continue;
        }
        let Some(named) = recovery_name(name) else {
            continue;
        };
        let modified = match entry.metadata().and_then(|meta| meta.modified()) {
            Ok(modified) => modified,
            // Deleted since it was listed: a project closed or saved meanwhile.
            Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
            Err(e) => return Err(e.into()),
        };
        if let Some(candidate) = candidate(entry.path(), named, modified) {
            found.push(candidate);
        }
    }
    order(&mut found);
    Ok(found)
}

/// Orders candidates as [`find_recoveries`] lists them.
///
/// By project, the one whose newest file was written last first; within a project, by save
/// point (the latest first), and on one save point, the newest last edit first, then the newest
/// write; then by path, so the order is total.
fn order(found: &mut [RecoveryCandidate]) {
    let mut newest: HashMap<ProjectId, SystemTime> = HashMap::new();
    for candidate in &*found {
        let at = newest
            .entry(candidate.project)
            .or_insert(candidate.modified);
        *at = (*at).max(candidate.modified);
    }
    let offset = |c: &RecoveryCandidate| c.base.map(|point| point.offset);
    found.sort_by(|a, b| {
        let newest_of = |c: &RecoveryCandidate| newest.get(&c.project).copied();
        newest_of(b)
            .cmp(&newest_of(a))
            .then_with(|| a.project.cmp(&b.project))
            .then_with(|| offset(b).cmp(&offset(a)))
            .then_with(|| b.last_edit.cmp(&a.last_edit))
            .then_with(|| b.modified.cmp(&a.modified))
            .then_with(|| a.path.cmp(&b.path))
    });
}

/// Removes the temporary file at `path` if it is a crash's leftover.
///
/// It is one if unwritten for [`STALE_TEMPORARY`] and lockable, so no instance is still making
/// it. Best effort: a file that can't be checked or removed is left for the next listing.
fn sweep(path: &Path) {
    let stale = fs::metadata(path)
        .and_then(|meta| meta.modified())
        .ok()
        .and_then(|modified| modified.elapsed().ok())
        .is_some_and(|age| age >= STALE_TEMPORARY);
    if !stale {
        return;
    }
    let Ok(file) = fs::File::open(path) else {
        return;
    };
    // The lock is held across the removal, so an instance can't take the file meanwhile.
    if file.try_lock().is_ok() {
        drop(fs::remove_file(path));
    }
}

/// The project id a recovery file's name gives, if `name` is one: `<32 hex digits>-<16 hex
/// digits>.bcrf`, as [`create_recovery`] names them.
fn recovery_name(name: &str) -> Option<ProjectId> {
    let stem = name.strip_suffix(EXTENSION)?.strip_suffix('.')?;
    let (id, suffix) = stem.split_once('-')?;
    let hex = |s: &str, len| s.len() == len && s.bytes().all(|b| b.is_ascii_hexdigit());
    if !hex(id, 32) || !hex(suffix, 16) {
        return None;
    }
    u128::from_str_radix(id, 16).ok().map(ProjectId::from_bits)
}

/// The candidate the recovery file at `path` makes, named for project `named`; `None` if a
/// running instance holds it, or it is gone.
///
/// A file whose meta can't be read, or names another project than its name, or isn't a
/// recovery file at all, is listed as [`CandidateStatus::Damaged`].
fn candidate(path: PathBuf, named: ProjectId, modified: SystemTime) -> Option<RecoveryCandidate> {
    let damaged = |at| RecoveryCandidate {
        path: at,
        project: named,
        base: None,
        base_path: None,
        last_edit: None,
        modified,
        status: CandidateStatus::Damaged,
    };
    let file = match ProjectFile::inspect(&path) {
        Ok(file) => file,
        Err(StoreError::Locked) => return None,
        Err(StoreError::Io(e)) if e.kind() == io::ErrorKind::NotFound => return None,
        Err(_) => return Some(damaged(path)),
    };
    let meta = file.meta();
    let Role::Recovery { base } = meta.role() else {
        return Some(damaged(path));
    };
    if meta.project() != named {
        return Some(damaged(path));
    }
    let status = match meta.base_path() {
        None if base.is_none() => CandidateStatus::NeverSaved,
        None => CandidateStatus::ProjectMissing,
        Some(at) => match ProjectFile::inspect(at) {
            Ok(project) => standing(meta, &project).map_or(CandidateStatus::Damaged, |(s, _)| s),
            Err(e) => unavailable(&e).unwrap_or(CandidateStatus::Damaged),
        },
    };
    Some(RecoveryCandidate {
        project: named,
        base,
        base_path: meta.base_path().map(Path::to_path_buf),
        last_edit: file.head(),
        modified,
        status,
        path,
    })
}

/// How a recovery file with `meta`, layered on a save point, stands against `project`, the
/// file at its base path, with the base to layer it on if it can be.
///
/// `project` must be a project file of the same project and mesh, else the project is
/// missing. Then the save point `meta` names must be one of its save points, offset and head
/// op alike: [`CandidateStatus::Restorable`] if it is the latest,
/// [`CandidateStatus::RestorableOnEarlierSave`] if the project was saved since, and
/// [`CandidateStatus::SavePointMissing`] if there is none.
///
/// # Errors
///
/// [`StoreError::Damaged`] if a save point of the project file no longer loads;
/// [`StoreError::Io`].
fn standing(
    meta: &Meta,
    project: &ProjectFile,
) -> Result<(CandidateStatus, Option<BaseFile>), StoreError> {
    let Role::Recovery { base: Some(point) } = meta.role() else {
        return Ok((CandidateStatus::ProjectMissing, None));
    };
    let theirs = project.meta();
    if theirs.role() != Role::Project
        || theirs.project() != point.project
        || theirs.mesh_hash() != meta.mesh_hash()
    {
        return Ok((CandidateStatus::ProjectMissing, None));
    }
    let latest = project.base()?;
    if latest.as_ref().map(BaseFile::save_point) == Some(point) {
        return Ok((CandidateStatus::Restorable, latest));
    }
    Ok(match project.base_at(point)? {
        Some(base) => (CandidateStatus::RestorableOnEarlierSave, Some(base)),
        None => (CandidateStatus::SavePointMissing, None),
    })
}

/// The error a restore reports for a project file that failed to open with `e`:
/// [`ProjectError::NotRestorable`] if the project isn't available ([`unavailable`]), else the
/// failure itself.
fn not_restorable(e: StoreError) -> ProjectError {
    match unavailable(&e) {
        Some(status) => ProjectError::NotRestorable(status),
        None => e.into(),
    }
}

/// The status a project file that failed to open with `e` gives its recovery file, if the
/// failure says the project isn't available rather than damaged.
///
/// That is [`CandidateStatus::ProjectMissing`] for a file that isn't there, and
/// [`CandidateStatus::ProjectInUse`] for one a running instance holds.
fn unavailable(e: &StoreError) -> Option<CandidateStatus> {
    if let StoreError::Io(io) = e
        && io.kind() == io::ErrorKind::NotFound
    {
        Some(CandidateStatus::ProjectMissing)
    } else if matches!(e, StoreError::Locked) {
        Some(CandidateStatus::ProjectInUse)
    } else {
        None
    }
}

/// A recovery file's two names: the temporary one it is made (and filled) under, and the real
/// one [`Unplaced::place`] moves it to (see the module notes).
#[derive(Debug)]
struct Unplaced {
    /// Where the file is made: `<name>.tmp`.
    tmp: PathBuf,
    /// Its name once placed: `<project id>-<suffix>.bcrf`.
    path: PathBuf,
    /// Test hook: [`Unplaced::place`] fails as a crash before the rename would, and leaves the
    /// temporary file behind as a crash does.
    #[cfg(test)]
    crash: bool,
}

impl Unplaced {
    /// Moves the file to its real name, if nothing is there, and makes the move durable;
    /// returns the real name.
    ///
    /// # Errors
    ///
    /// [`ProjectError::Io`] if a file has the name already, or the rename or the directory's
    /// sync fails. The temporary file is removed then.
    fn place(self) -> Result<PathBuf, ProjectError> {
        #[cfg(test)]
        if self.crash {
            return Err(io::Error::other("a crash before the rename, for the test").into());
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
    fn crashing(self, crash: bool) -> Self {
        Self { crash, ..self }
    }

    /// Removes the temporary file: a creation given up. Best effort, as the error at hand is
    /// the one to report; a leftover is swept by [`find_recoveries`].
    fn discard(self) {
        drop(fs::remove_file(&self.tmp));
    }
}

/// Creates a recovery file of `project` in `dir` through `create`, under the temporary name of
/// a fresh one (see the module notes), and returns it open, with its names for
/// [`Unplaced::place`].
///
/// `create` makes the file at the temporary path it is given, durable.
///
/// # Errors
///
/// [`ProjectError::Io`] if the directory can't be made or a random suffix drawn; `create`'s
/// error.
fn create_recovery(
    dir: &Path,
    project: ProjectId,
    create: impl FnOnce(&Path) -> Result<ProjectFile, StoreError>,
) -> Result<(ProjectFile, Unplaced), ProjectError> {
    fs::create_dir_all(dir)?;
    let suffix = ProjectId::random()?.to_bits() >> 64;
    let name = format!("{project}-{suffix:016x}.{EXTENSION}");
    let names = Unplaced {
        tmp: dir.join(format!("{name}.tmp")),
        path: dir.join(&name),
        #[cfg(test)]
        crash: false,
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
fn create_placed(
    dir: &Path,
    project: ProjectId,
    create: impl FnOnce(&Path) -> Result<ProjectFile, StoreError>,
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
fn place_new(tmp: &Path, path: &Path) -> io::Result<()> {
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
