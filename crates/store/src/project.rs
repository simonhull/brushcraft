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
//! failure to is reported ([`SaveOutcome::kept`]), never fatal.
//!
//! A save blocks its thread for as long as all that takes: seconds after a long stroke, nearly
//! all of it waiting for the flush and copying records. An app that must go on answering calls
//! [`Project::save_ahead`] first, slice by slice, until it says [`SaveAhead::Ready`], and the
//! save has only its fixed part left. The records it copies are part of the save: no save point
//! names them until the save does, and if none ever does, the file discards them as it opens.
//!
//! # Saving comes first
//!
//! **A save always succeeds if the project file can be written.** The recovery directory never
//! stops one, full or read-only: the save copies every edit from the store, not from the recovery
//! file, so a flush that fails on the way, or a roll-over that can't be made, is passed over, and
//! the project file's save point is written first. If the fresh recovery file can't be made
//! then, the store is rebased onto a **scratch** working file instead, unlinked in the system's
//! temporary directory, and crash protection is **paused**: no recovery file holds the edits
//! since the save, and no writer runs. The save says so ([`SaveOutcome::protection`]), and so
//! does [`Project::protection`], for the app to show. The project keeps trying to put a recovery
//! file in place, as a roll-over that failed does (see [`ROLL_OVER_RETRY`]), and once one is,
//! protection runs again.
//!
//! A roll-over that fails pauses protection the same way, but leaves a recovery file: the one
//! whose writer failed, at its last durable save point. Nothing more may be appended to it (a
//! failed sync proves nothing about what follows), so the store is moved off it onto a scratch
//! file, **stranding** it: still held and locked, still the recovery file on disk
//! ([`Project::recovery_path`]), until a new recovery file replaces it or a save makes it
//! unneeded. The scratch file is made in the project file's directory or the recovery
//! directory, never the temporary directory, which is often memory-backed: it is filled with
//! every edit since the last save. If none can be made, the store stays on the failed file:
//! write-back still lands there, but no save point is ever written there again
//! ([`TileStore::save`] refuses).
//!
//! # The spatial index
//!
//! The bounds of the spatial index are kept with the tiles (see [`crate::bounds`]). What edits
//! change, and the index records over it, streams into the recovery file with the edits; a save
//! copies every bound recorded since into the project file, those the index computed with no edit
//! pending (a whole build, say) included, which the recovery file never holds. So
//! [`Project::load_index`] after [`Project::open`] reads the index instead of evaluating every
//! tile, and after [`Project::restore`] it evaluates at most the tiles whose bounds were derived
//! after the last save: a crash costs their evaluation, never a stale bound.
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

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use std::{fs, io};

use geom::{BaseMesh, Level};
use thiserror::Error;

use crate::OpId;
use crate::file::{FileError, PRIVATE_MODE, ProjectFile, SHARED_MODE, sync_parent};
use crate::index::{IndexError, LoadStats, SpatialIndex};
use crate::meta::{Meta, ProjectId, Role, SavePoint};
use crate::recovery::{RecoveryError, RecoveryWriter};
use crate::stroke_record::StrokeRecord;
use crate::tiles::{CacheError, Saved, StoreConfig, TileStore};

mod discover;
mod place;

pub use discover::find_recoveries;
use discover::{not_restorable, past, recovery_heads, standing};
use place::{
    create_clean, create_placed, create_recovery, place_new, retire, scratch, scratch_in, temporary,
};

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
#[non_exhaustive]
pub enum ProjectError {
    /// A file system call failed: creating the recovery directory, renaming a file into place,
    /// deleting a recovery file.
    #[error("project I/O failed")]
    Io(#[from] io::Error),
    /// The project file or a recovery file failed.
    #[error("a project or recovery file failed")]
    Store(#[from] FileError),
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
    /// The call that reports them otherwise did what it was asked. A save reports them in its
    /// outcome instead ([`SaveOutcome::unprotected`]), never as an error.
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
    /// Crash protection is paused: the try to put a new recovery file in place failed, for
    /// this reason, so no recovery file takes the edits (see [`Project::protection`]). It is
    /// tried again after [`ROLL_OVER_RETRY`], the wait doubling up to a minute while it keeps
    /// failing, and at once by a save or a close.
    #[error("crash protection is paused")]
    ProtectionPaused(#[source] Arc<Self>),
    /// The call did what it was asked (a roll-over, say: the project goes on with its new
    /// recovery file), but the old recovery file at `path` couldn't be deleted. It holds nothing
    /// the project needs any more. The project holds it, so no listing offers it meanwhile, and
    /// tries again to delete it whenever crash protection is next put in place. Once the
    /// project closes, until it is deleted, [`find_recoveries`] lists it, on the save point it
    /// was made on. A save reports it in its outcome instead ([`SaveOutcome::kept`]).
    #[error("the old recovery file {path:?} couldn't be deleted")]
    OldRecoveryKept {
        /// Where the old recovery file is.
        path: PathBuf,
        /// Why it couldn't be deleted.
        #[source]
        source: io::Error,
    },
    /// [`Project::end_stroke`] was given the record of another stroke than the one ending: its
    /// op isn't the one [`Project::begin_stroke`] returned for it. Nothing was recorded, and the
    /// stroke didn't end.
    #[error("a record of op {record:?} for the stroke of op {stroke:?}")]
    NotTheStroke {
        /// The record's op.
        record: OpId,
        /// The op of the stroke ending, if one is.
        stroke: Option<OpId>,
    },
    /// [`Project::end_stroke`] was given a record of an op it has kept a record of already: the
    /// stroke ended then, and its record is kept once. Nothing was recorded again.
    #[error("the stroke of op {0:?} has its record already")]
    AlreadyRecorded(OpId),
}

/// An old recovery file a save couldn't delete: it holds nothing the project needs any more
/// (see [`ProjectError::OldRecoveryKept`]).
#[derive(Debug, Error)]
#[error("the old recovery file {path:?} couldn't be deleted")]
pub struct KeptRecovery {
    /// Where the old recovery file is.
    pub path: PathBuf,
    /// Why it couldn't be deleted.
    #[source]
    pub source: io::Error,
}

/// The report of a call that deleted an old recovery file, and couldn't.
impl From<KeptRecovery> for ProjectError {
    fn from(kept: KeptRecovery) -> Self {
        Self::OldRecoveryKept {
            path: kept.path,
            source: kept.source,
        }
    }
}

/// Whether a project's edits are protected against a crash: streamed into a recovery file (see
/// the module notes).
#[derive(Debug, Clone)]
pub enum Protection {
    /// A recovery writer streams every edit into the recovery file, a durable save point after
    /// every stroke.
    Active,
    /// No recovery file takes the edits, for this reason: the last try to put one in place
    /// failed (the recovery directory is full or read-only, say). A crash loses the edits since
    /// the last durable save point of the last recovery file, or of the project file. The
    /// project keeps trying, and the app shows that protection is paused meanwhile.
    Paused(Arc<ProjectError>),
}

impl Protection {
    /// Whether crash protection runs.
    #[must_use]
    pub const fn is_active(&self) -> bool {
        matches!(self, Self::Active)
    }
}

/// What a save that went through ([`Project::save`], [`Project::save_as`]) has to report: the
/// project file holds the new save point, whatever it says.
#[derive(Debug)]
#[must_use = "a save can pause crash protection, which the app shows"]
pub struct SaveOutcome {
    /// Whether crash protection runs after the save: [`Protection::Paused`] if no new recovery
    /// file could be made on the save point, and none was put back in place before it.
    pub protection: Protection,
    /// Tiles whose latest edits are in no file, in key order: found unprotected before the save
    /// (left out of a flush, or lost to a roll-over), or lost to it, its save point holding
    /// their last saved state.
    pub unprotected: Vec<geom::TileKey>,
    /// Old recovery files that couldn't be deleted. The project holds them until it closes, so
    /// no listing offers them meanwhile.
    pub kept: Vec<KeptRecovery>,
}

/// How far [`Project::save_ahead`] has got.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaveAhead {
    /// A flush is under way: nothing to do until the recovery writer's thread is done with it.
    Flushing,
    /// Copying tile records into the project file, this many tiles left.
    Copying(usize),
    /// [`Project::save`] has only its fixed part left.
    Ready,
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

/// When [`Project::resume`] tries to put crash protection back in place.
#[derive(Debug, Clone, Copy)]
enum Retry {
    /// At once: for a save or a close, which want every edit in a recovery file.
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
    /// Streams the store's edits into the recovery file. `None` while crash protection is
    /// paused: a writer whose file failed to sync, or whose thread stopped, is closed at once, and
    /// none runs until a new recovery file is in place ([`Project::resume`]). Also `None` once
    /// closed: it is taken out by [`Project::close`] and the drop, which close it on the
    /// store.
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
    /// Where the recovery file on disk that protects the project is: the store's working file,
    /// or, after a failed roll-over, the failed file the store has moved off
    /// ([`Project::stranded`]), still at its last durable save point. `None` while crash
    /// protection has been paused since a save that couldn't make its new recovery file (the
    /// store works on a scratch file then, see [`scratch`]).
    recovery_path: Option<PathBuf>,
    /// The directory recovery files go in.
    recovery_dir: PathBuf,
    /// Tiles whose latest edits were found unprotected (left out of a flush, or lost to a
    /// roll-over or a save), waiting to be reported: by the next call that has no other error
    /// to return ([`Project::report`]). So a report never stops a call from doing its work.
    unreported: BTreeSet<geom::TileKey>,
    /// Why crash protection is paused, if it is: the error of the last try to put a recovery
    /// file in place. `None` while protection runs.
    paused: Option<Arc<ProjectError>>,
    /// When crash protection may be tried again; `None` to try at once. A frame loop polls
    /// every few milliseconds, and a try that keeps failing (a full disk, say) is made only this
    /// often; a save or a close tries at once.
    retry_at: Option<Instant>,
    /// How long the next failed try makes the project wait before trying again:
    /// [`ROLL_OVER_RETRY`], doubling up to a minute while the failures go on.
    retry_after: Duration,
    /// Old recovery files that couldn't be deleted, with their paths, held (and locked) until
    /// they are or the project closes, so no listing offers them while it runs (see
    /// [`Project::let_go`]). Deleting them is tried again whenever protection is put in place.
    held: Vec<(PathBuf, ProjectFile)>,
    /// The failed recovery file the store has moved off ([`Project::strand`]), still held (open
    /// and locked) while the store works on a scratch file instead. It stays the project's
    /// recovery file on disk ([`Project::recovery_path`]), at its last durable save point, until
    /// a new recovery file replaces it or a save makes it unneeded: it is released only through
    /// [`Project::let_go`].
    stranded: Option<ProjectFile>,
    /// The op of the stroke open now: the one [`Project::begin_stroke`] returned, until the
    /// stroke ends. Kept across a save, which ends the store's op (its later edits take a new
    /// one), so the stroke's record still lands.
    stroke: Option<OpId>,
    /// The op of the last stroke record kept: a second record of it is refused.
    recorded: Option<OpId>,
    /// Test hook: makes the next placement of a recovery file a save or a roll-over makes fail
    /// after its temporary file is written and synced, before the rename, with an I/O error of
    /// this kind (see [`Unplaced::crash`](place::Unplaced::crash)).
    #[cfg(test)]
    fail_next_recovery: Option<io::ErrorKind>,
    /// Test hook: makes the directory sync after the next save as moves its new file into
    /// place fail, with an I/O error (see [`Project::write_new_project`]).
    #[cfg(test)]
    fail_dir_sync: bool,
    /// Test record: each directory [`Project::strand`] tried a scratch file in, in order, and
    /// whether the store moved onto it there.
    #[cfg(test)]
    strand_tries: Vec<(PathBuf, bool)>,
}

impl std::fmt::Debug for Project {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Project")
            .field("path", &self.path)
            .field("recovery_path", &self.recovery_path)
            .field("paused", &self.paused)
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
            ProjectFile::create_private(at, meta, Arc::clone(&mesh))
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
    /// [`FileError::Locked`]); [`ProjectError::NotAProject`]; [`ProjectError::NeverSaved`];
    /// as [`Project::new`] for the recovery file, the store and the writer.
    pub fn open(path: &Path, config: ProjectConfig) -> Result<Self, ProjectError> {
        let path = std::path::absolute(path)?;
        let project = ProjectFile::open(&path)?;
        if project.meta().role() != Role::Project {
            return Err(ProjectError::NotAProject);
        }
        let base = project.base().ok_or(ProjectError::NeverSaved)?;
        let meta = project.meta().on_save_point(base.save_point(), &path);
        let earlier = recovery_heads(&config.recovery_dir, meta.project());
        let (working, recovery_path) = create_placed(&config.recovery_dir, meta.project(), |at| {
            ProjectFile::create_like_mode(at, meta, &project, PRIVATE_MODE)
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
    /// [`FileError::Locked`]); [`ProjectError::Cache`] if the store refuses `config.store`;
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
            recovery_path: Some(recovery_path),
            recovery_dir: config.recovery_dir,
            unreported: BTreeSet::new(),
            paused: None,
            retry_at: None,
            retry_after: ROLL_OVER_RETRY,
            held: Vec::new(),
            stranded: None,
            stroke: None,
            recorded: None,
            #[cfg(test)]
            fail_next_recovery: None,
            #[cfg(test)]
            fail_dir_sync: false,
            #[cfg(test)]
            strand_tries: Vec::new(),
        })
    }

    /// Saves the edits since the last save into the project file, as a new save point, and
    /// starts a fresh recovery file on it (see the module notes).
    ///
    /// Saving comes first: the recovery directory never stops a save. If the fresh recovery
    /// file can't be made, the save goes through all the same, and crash protection pauses
    /// until one can ([`SaveOutcome::protection`]).
    ///
    /// # Errors
    ///
    /// An error means the project isn't saved: it goes on as before, its edits unsaved, and
    /// saving again works. [`ProjectError::NoPath`] for a project never saved, or restored onto
    /// an earlier save point; [`ProjectError::Cache`] from installing prefetched tiles, or if
    /// the save fails ([`CacheError::Save`]: the project file may hold the tiles without a save
    /// point; saving again writes them again). Once the save point is written, only a store
    /// that can't be moved onto it fails the save: [`ProjectError::Cache`] if the rebase is
    /// refused, and the error of the fresh recovery file if no scratch file can be made in its
    /// place either. The project file holds the save point then, but the project goes on over
    /// the one before, and the next save saves its edits again.
    pub fn save(&mut self) -> Result<SaveOutcome, ProjectError> {
        let path = self.path.clone().ok_or(ProjectError::NoPath)?;
        let mut kept = self.prepare_save()?;
        let mut project = self.file.take().ok_or(ProjectError::NoPath)?;
        let switched = self
            .store
            .save_into(&mut project)
            .map_err(ProjectError::from)
            .and_then(|saved| {
                self.unreported.extend(saved.lost());
                self.switch(saved, &project, &path)
            });
        self.file = Some(project);
        if switched.is_err() {
            // What the file holds past its last save point is in doubt, so it is all written again.
            self.store.forget_copied_ahead();
        }
        kept.extend(switched?);
        Ok(self.outcome(kept))
    }

    /// A slice of the work of the next [`Project::save`], done ahead of it, for a caller that
    /// can't be blocked for all of it: never waits, and copies tile records into the project file
    /// until `deadline`, so a call takes about that long at most. Called until it says
    /// [`SaveAhead::Ready`], it leaves the save its fixed part only.
    ///
    /// While a flush is under way this only moves it on: the records to copy are the ones it
    /// writes. The project file changes in nothing a reader sees: its new records are named by no
    /// save point until the save. Edits may go on between slices, and the save takes them,
    /// however far this got. A save never needs this.
    ///
    /// Crash protection never stops a save: a flush that fails here is the writer's to deal with,
    /// and the save says what came of it.
    ///
    /// # Errors
    ///
    /// [`ProjectError::NoPath`] for a project never saved, or restored onto an earlier save
    /// point, as [`Project::save`] gives; [`ProjectError::Cache`] if a write to the project file
    /// fails. The project goes on as before, and saving again works.
    pub fn save_ahead(&mut self, deadline: Instant) -> Result<SaveAhead, ProjectError> {
        if self.path.is_none() {
            return Err(ProjectError::NoPath);
        }
        drop(self.advance());
        if self.writer.as_ref().is_some_and(|writer| !writer.is_idle()) {
            return Ok(SaveAhead::Flushing);
        }
        let project = self.file.as_mut().ok_or(ProjectError::NoPath)?;
        Ok(match self.store.copy_ahead(project, deadline)? {
            Some(left) => SaveAhead::Copying(left),
            None => SaveAhead::Ready,
        })
    }

    /// Saves the project as a new project file at `path`, with a new project id, and goes on
    /// with that one: every tile of the last save point copied (as records, not re-encoded),
    /// then the edits since, then a save point. The old project file isn't written to.
    ///
    /// The file is written as a temporary file beside `path` and moved into place once its
    /// save point is durable, so `path` never names a half-written project. The move never
    /// replaces a file (a hard link, not a rename), not even one made at `path` while the
    /// project was being written. Then the project moves onto it as [`Project::save`] does, and
    /// the recovery directory no more stops it.
    ///
    /// # Errors
    ///
    /// An error means the project isn't saved as `path`: it goes on as before, with its old
    /// project file, path and id. [`ProjectError::Exists`] if `path` names a file already, or
    /// does by the time the new file is to be moved there (it is removed then);
    /// [`ProjectError::Store`] or [`ProjectError::Cache`] if creating or writing the new file
    /// fails (the temporary file is removed); [`ProjectError::Io`] if the rename does, or the
    /// directory's sync after it (the new file is removed from `path` then, best effort, so a
    /// retry finds the path free); [`ProjectError::Cache`] from installing prefetched tiles. As
    /// [`Project::save`] once the new file is in place and durable: the project goes on with
    /// the old one then.
    pub fn save_as(&mut self, path: &Path) -> Result<SaveOutcome, ProjectError> {
        let path = std::path::absolute(path)?;
        if fs::symlink_metadata(&path).is_ok() {
            return Err(ProjectError::Exists(path));
        }
        let mut kept = self.prepare_save()?;
        // Copies ahead went to the old project file: the new one needs every tile.
        self.store.forget_copied_ahead();
        let meta = self
            .store
            .working()
            .meta()
            .with_project(ProjectId::random()?)
            .with_role(Role::Project);
        let tmp = temporary(&path)?;
        let mut project = create_clean(&tmp, meta, self.store.working(), SHARED_MODE)?;
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
        kept.extend(self.switch(saved, &project, &path)?);
        // The switch has happened: the store reads the new project file, so the project holds
        // it, and saves go to it. The old project file closes here, and its lock goes with it.
        self.file = Some(project);
        self.path = Some(path);
        Ok(self.outcome(kept))
    }

    /// Fills `project`, the new file being written at `tmp`, with the last save point's tiles
    /// and the edits since, saves it, and renames it to `path`.
    ///
    /// If the directory can't be synced after the rename, the file is removed from `path`
    /// again: the project doesn't adopt it, so it mustn't be left there looking saved.
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
        let synced = sync_parent(path);
        #[cfg(test)]
        let synced = if std::mem::take(&mut self.fail_dir_sync) {
            Err(io::Error::other("a directory sync failing, for the test").into())
        } else {
            synced
        };
        if let Err(e) = synced {
            // Best effort: the error at hand is the one to report.
            drop(fs::remove_file(path));
            return Err(e.into());
        }
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
    /// file kept (or deleted). So does a project whose crash protection is paused: it tries
    /// once more to put a recovery file in place.
    ///
    /// # Errors
    ///
    /// [`ProjectError::Recovery`] if the last flushes fail, or
    /// [`ProjectError::ProtectionPaused`] if no new recovery file can be put in place: the
    /// recovery file is kept then, whatever `keep_recovery` says, at its last durable save
    /// point (a project paused since a save has none, and keeps nothing).
    /// [`ProjectError::OldRecoveryKept`] if a roll-over on the way couldn't delete the file it
    /// replaced. [`ProjectError::Io`] or [`ProjectError::Store`] if the recovery file can't be
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
            if let Some(path) = &self.recovery_path {
                fs::remove_file(path)?;
                sync_parent(path)?;
            }
            // Nothing is kept to lack them.
            self.unreported.clear();
        }
        self.report()
    }

    /// Closes the project, dropping what no save holds: the user's choice, which a full disk
    /// mustn't overrule. So unlike [`Project::close`], which finishes the recovery writer's flushes
    /// first and keeps the recovery file if it can't, this stops the writer where it is, with no
    /// flush or roll-over, and removes the recovery file and any old one still held whatever state
    /// they are in. A file already gone is what was wanted.
    ///
    /// Otherwise the file would be found at the next open, and bring back work the user dropped.
    ///
    /// # Errors
    ///
    /// [`ProjectError::OldRecoveryKept`] if a recovery file can't be removed, or its removal made
    /// durable: the first that couldn't, the others tried all the same.
    pub fn discard(mut self) -> Result<(), ProjectError> {
        if let Some(writer) = self.writer.take() {
            writer.abandon(&mut self.store);
        }
        // Nothing is kept to lack them.
        self.unreported.clear();
        let held = std::mem::take(&mut self.held);
        let held = held.into_iter().map(|(path, _file)| path);
        let files = self.recovery_path.take().into_iter().chain(held);
        let mut first_failure = Ok(());
        for path in files {
            let removed = retire(&path);
            if first_failure.is_ok() {
                first_failure = removed;
            }
        }
        first_failure.map_err(ProjectError::from)
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

    /// The spatial index over every tile of `level`, loaded from the bounds the project keeps
    /// ([`SpatialIndex::load`]): read from the recovery file where it has them, else from the
    /// project file's save point, and evaluated only for tiles neither holds (a project saved
    /// before bounds were kept, or edited since without a refit). What an index built, refitted
    /// or loaded here computes is kept in turn: streamed into the recovery file where it replaces
    /// an edit's change, and saved into the project file by the next save.
    ///
    /// # Errors
    ///
    /// As [`SpatialIndex::load`].
    pub fn load_index(
        &mut self,
        level: Level,
        progress: impl FnMut(usize, usize),
    ) -> Result<(SpatialIndex, LoadStats), IndexError> {
        SpatialIndex::load(&mut self.store, level, progress)
    }

    /// Where the project file saves go to is; `None` for a project never saved, or one
    /// restored onto an earlier save point ([`CandidateStatus::RestorableOnEarlierSave`]),
    /// which saves as a new file.
    #[must_use]
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// Where the recovery file on disk that protects the project is: it changes with every save,
    /// and every roll-over. It is the store's working file or, after a roll-over failed, the
    /// failed file the store has moved off, still at its last durable save point. `None` while
    /// crash protection has been paused since a save that couldn't make its new recovery file.
    #[must_use]
    pub fn recovery_path(&self) -> Option<&Path> {
        self.recovery_path.as_deref()
    }

    /// Whether the project's edits are protected against a crash, or protection is paused, and
    /// why: for the app to show. Every call that advances the recovery writer can change it, a
    /// save too.
    #[must_use]
    pub fn protection(&self) -> Protection {
        self.paused.as_ref().map_or(Protection::Active, |cause| {
            Protection::Paused(Arc::clone(cause))
        })
    }

    /// The project's id: the project file's, or for a project never saved, its recovery
    /// file's. A save as gives the project a new one.
    #[must_use]
    pub fn id(&self) -> ProjectId {
        self.store.working().meta().project()
    }

    /// A stroke begins: starts a new op in the store ([`TileStore::begin_op`]), which every
    /// edit of the stroke is part of, however many dabs it takes, and returns it: the op the
    /// stroke's record is made with ([`Project::end_stroke`]).
    ///
    /// # Errors
    ///
    /// [`ProjectError::Cache`] with [`CacheError::OpsExhausted`] if the site's ops are used up.
    pub fn begin_stroke(&mut self) -> Result<OpId, ProjectError> {
        let op = self.store.begin_op()?;
        self.stroke = Some(op);
        Ok(op)
    }

    /// A stroke ended, leaving `record`: flushes its edits to the recovery file in the
    /// background (see [`RecoveryWriter::end_stroke`]), with the record kept beside them.
    ///
    /// The flush the stroke end starts appends the record before its save point, so the
    /// recovery file holds it from the save point that holds the stroke's last edits, and a save
    /// copies it into the project file. Opening a file never reads it (see [`StrokeRecord`]);
    /// [`ProjectFile::strokes`] reads them back, for tools and tests.
    ///
    /// `record`'s op must be the one [`Project::begin_stroke`] returned for this stroke. That
    /// holds across a save in the middle of the stroke, which moves the store's later edits to a
    /// new op: the record still lands, once.
    ///
    /// The record is compressed here, on the calling thread: a few kilobytes of samples.
    ///
    /// The stroke's pen-down shapes go once its record is kept
    /// ([`TileStore::release_pen_down`]): the next stroke reads the surface the stroke left.
    ///
    /// # Errors
    ///
    /// Refused, with nothing recorded and the stroke still open:
    /// [`ProjectError::NotTheStroke`] if `record`'s op isn't the open stroke's;
    /// [`ProjectError::Cache`] if compressing it fails.
    ///
    /// [`ProjectError::AlreadyRecorded`] if a record of its op was kept already: the stroke has
    /// ended, so a call made again after an error below never records it twice.
    ///
    /// Once the record is kept, the stroke has ended, whatever follows:
    /// [`ProjectError::Recovery`] if the recovery writer fails, unless the failure was a sync,
    /// or a stopped writer thread, which the project deals with by moving to a new recovery
    /// file holding every edit;
    /// [`ProjectError::ProtectionPaused`] if moving to a new recovery file fails (it is tried
    /// again after [`ROLL_OVER_RETRY`], then waits doubling up to a minute while it keeps
    /// failing); [`ProjectError::OldRecoveryKept`] if it moved, but couldn't delete the old
    /// file; [`ProjectError::Unprotected`] for tiles whose edits aren't in the recovery file,
    /// once nothing else is to be reported.
    pub fn end_stroke(&mut self, record: &StrokeRecord) -> Result<(), ProjectError> {
        let op = record.op();
        if self.recorded == Some(op) {
            return Err(ProjectError::AlreadyRecorded(op));
        }
        if self.stroke != Some(op) {
            return Err(ProjectError::NotTheStroke {
                record: op,
                stroke: self.stroke,
            });
        }
        self.store.record_stroke(record)?;
        self.recorded = Some(op);
        self.stroke = None;
        self.store.release_pen_down();
        self.end_stroke_at(Instant::now())
    }

    /// A stroke ended with no record of it: as [`Project::end_stroke`] otherwise. For tools,
    /// tests and benches only, whose edits no pen made; the app's strokes end with
    /// [`Project::end_stroke`].
    ///
    /// # Errors
    ///
    /// As [`Project::end_stroke`] once its record is kept.
    pub fn end_unrecorded_stroke(&mut self) -> Result<(), ProjectError> {
        self.stroke = None;
        self.store.release_pen_down();
        self.end_stroke_at(Instant::now())
    }

    /// [`Project::end_stroke`] at `now`.
    ///
    /// # Errors
    ///
    /// As [`Project::end_stroke`].
    fn end_stroke_at(&mut self, now: Instant) -> Result<(), ProjectError> {
        self.with_writer(
            |writer, store| writer.end_stroke(store, now),
            Retry::Due(now),
        )?;
        self.report()
    }

    /// Time passed, to `now`: flushes the edits of a long stroke at least once a second (see
    /// [`RecoveryWriter::tick`]), and tries to put paused crash protection back in place once
    /// the wait since the last try has passed.
    ///
    /// The app must call it regularly, every frame. A flush snapshots the tiles dirtied since the
    /// previous one outside the tile pool's budget, so ticking bounds that copy to about a second
    /// of edits; a caller that never ticks gets one large snapshot at [`Project::end_stroke`], of
    /// every tile the stroke dirtied.
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

    /// [`Project::poll`] without the report.
    ///
    /// # Errors
    ///
    /// As [`Project::with_writer`]; [`ProjectError::Cache`] from installing prefetched tiles.
    fn advance(&mut self) -> Result<(), ProjectError> {
        self.with_writer(RecoveryWriter::poll, Retry::Due(Instant::now()))?;
        self.store.poll()?;
        Ok(())
    }

    /// Flushes every edit into the recovery file and waits until it is durable, so no flush is
    /// in flight afterwards. A failed writer's roll-over, or paused protection, is tried at
    /// once.
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

    /// What a save starts with: prefetched tiles installed, and every edit flushed into the
    /// recovery file ([`Project::finish_flushes`]), so no flush is in flight while the store is
    /// rebased. Returns the old recovery file a roll-over on the way couldn't delete, if any.
    ///
    /// Crash protection never stops a save, so the flushes' failure is passed over: the save
    /// copies every edit from the store, not from the recovery file. A writer that fails here
    /// abandons its flush (none is in flight then), and one that can't be replaced leaves
    /// protection paused, which the save's outcome says if no fresh recovery file can be made
    /// on its save point either.
    ///
    /// # Errors
    ///
    /// [`ProjectError::Cache`] from installing prefetched tiles.
    fn prepare_save(&mut self) -> Result<Vec<KeptRecovery>, ProjectError> {
        self.store.poll()?;
        Ok(match self.finish_flushes() {
            Err(ProjectError::OldRecoveryKept { path, source }) => {
                vec![KeptRecovery { path, source }]
            }
            // Crash protection's, never the save's.
            Ok(()) | Err(_) => Vec::new(),
        })
    }

    /// The outcome of a save that went through, with `kept`, the old recovery files it couldn't
    /// delete: the protection it leaves, and the tiles found unprotected since the last report,
    /// which are forgotten.
    fn outcome(&mut self, kept: Vec<KeptRecovery>) -> SaveOutcome {
        SaveOutcome {
            protection: self.protection(),
            unprotected: std::mem::take(&mut self.unreported).into_iter().collect(),
            kept,
        }
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
    /// ([`Project::report`]). If the writer has failed (a sync of the recovery file failed, or
    /// its thread stopped, see [`crate::recovery`]), it is closed, the store's working file is
    /// marked failed ([`TileStore::fail_working`]), and the project moves to a new
    /// recovery file holding every edit ([`Project::resume`]); the failure, dealt with, isn't
    /// returned. With no writer, crash protection is paused: `call` isn't run, and protection
    /// is tried again.
    ///
    /// # Errors
    ///
    /// `call`'s error, as [`ProjectError::Recovery`], unless the writer has failed; as
    /// [`Project::resume`].
    fn with_writer(
        &mut self,
        call: impl FnOnce(&mut RecoveryWriter, &mut TileStore) -> Result<(), RecoveryError>,
        retry: Retry,
    ) -> Result<(), ProjectError> {
        if let Some(writer) = self.writer.as_mut() {
            let outcome = call(writer, &mut self.store);
            self.unreported.extend(writer.take_skipped());
            if !writer.is_failed() {
                return Ok(outcome?);
            }
        }
        if let Some(failed) = self.writer.take() {
            // It can only say that it needs a new file; closing releases its subscription.
            drop(failed.close(&mut self.store, Instant::now()));
            self.store.fail_working();
        }
        // With no writer, no flush drains save's cursor in the change feed, and nothing else
        // does between saves: drained here, the feed stays bounded while protection is paused.
        self.store.drain_saves();
        self.resume(retry)
    }

    /// Puts crash protection back in place, with no writer running: moves every edit to a new
    /// recovery file and starts a writer on it ([`Project::roll_over`]), if a try is due.
    ///
    /// A try is due at once if `retry` says so, else only once the wait after the last failed
    /// one has passed: [`ROLL_OVER_RETRY`] after the first failure, doubling up to a minute
    /// while they go on. Until then the edits wait in memory, and in the working file.
    ///
    /// # Errors
    ///
    /// [`ProjectError::ProtectionPaused`] if the try fails, with why; protection stays paused.
    /// [`ProjectError::OldRecoveryKept`] if it is back in place, but the file it replaced
    /// couldn't be deleted.
    fn resume(&mut self, retry: Retry) -> Result<(), ProjectError> {
        let now = match retry {
            Retry::Due(now) if self.retry_at.is_some_and(|at| now < at) => return Ok(()),
            Retry::Due(now) => now,
            Retry::Now => Instant::now(),
        };
        match self.roll_over() {
            Err(cause) if self.writer.is_none() => {
                self.strand(&cause);
                Err(self.pause(cause, now))
            }
            rolled => {
                self.protected();
                rolled
            }
        }
    }

    /// Crash protection is paused, for `cause`, at `now`: the next try waits, longer than the
    /// last did. Returns the error that says so.
    fn pause(&mut self, cause: ProjectError, now: Instant) -> ProjectError {
        let cause = Arc::new(cause);
        self.paused = Some(Arc::clone(&cause));
        self.retry_at = Some(now + self.retry_after);
        self.retry_after = (self.retry_after * 2).min(ROLL_OVER_RETRY_MAX);
        ProjectError::ProtectionPaused(cause)
    }

    /// Crash protection is in place: nothing is paused, a failure from now on is tried again at
    /// once, and the old recovery files that couldn't be deleted are tried again.
    fn protected(&mut self) {
        self.paused = None;
        self.retry_at = None;
        self.retry_after = ROLL_OVER_RETRY;
        // Best effort, as it was the first time: those still there are held on.
        self.held.retain(|(path, _)| retire(path).is_err());
    }

    /// Moves the edits to a new recovery file and starts a writer on it: after the writer
    /// failed to sync the current one, or while crash protection is paused. The tiles that
    /// couldn't be carried across wait to be reported ([`Project::report`]), whatever happens
    /// after the move.
    ///
    /// The new file names the same save point. It is made under its temporary name, filled with
    /// every edit from the store ([`TileStore::fill_roll_over`]: synced, a save point, synced),
    /// and only then renamed into place, so a crash meanwhile leaves no file under a recovery
    /// file's name that lacks the edits of the old one. Then the store swaps it in
    /// ([`TileStore::roll_over`]), the old file is deleted, and only then unlocked
    /// ([`Project::let_go`]), and a new writer streams into the new one. An old scratch file is
    /// only closed, which deletes it.
    ///
    /// # Errors
    ///
    /// As [`Project::fresh_recovery`] for the new file; [`ProjectError::Cache`] if the store
    /// can't write it. The new file is deleted then, and the project stays on the old one, with
    /// no writer. [`ProjectError::Recovery`] if the new writer's thread can't be started: the
    /// project goes on with the new file, with no writer. [`ProjectError::OldRecoveryKept`] if
    /// the old file can't be deleted: the project goes on with the new file and its writer.
    fn roll_over(&mut self) -> Result<(), ProjectError> {
        let meta = self.store.working().meta().clone();
        let (fresh, names) = create_recovery(&self.recovery_dir, meta.project(), |at| {
            ProjectFile::create_like_mode(at, meta, self.store.working(), PRIVATE_MODE)
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
        self.unreported.extend(lost);
        let retired = match self.recovery_path.replace(fresh_path) {
            Some(old_path) => self.let_go(old_path, old),
            // A scratch file, which no listing finds: closing it deletes it.
            None => None,
        };
        self.writer = Some(RecoveryWriter::new(&mut self.store)?);
        retired.map_or(Ok(()), |kept| Err(kept.into()))
    }

    /// Makes an empty recovery file with `meta` over `like`'s mesh, durable and in place: what
    /// a save starts on its save point. Returns it open, with its path.
    ///
    /// # Errors
    ///
    /// As [`create_recovery`] and [`Unplaced::place`](place::Unplaced::place).
    #[cfg_attr(
        not(test),
        expect(
            clippy::needless_pass_by_ref_mut,
            reason = "it takes the test hook that fails a placement, which tests only have"
        )
    )]
    fn fresh_recovery(
        &mut self,
        meta: Meta,
        like: &ProjectFile,
    ) -> Result<(ProjectFile, PathBuf), ProjectError> {
        let (fresh, names) = create_recovery(&self.recovery_dir, meta.project(), |at| {
            ProjectFile::create_like_mode(at, meta, like, PRIVATE_MODE)
        })?;
        #[cfg(test)]
        let names = names.crashing(std::mem::take(&mut self.fail_next_recovery));
        Ok((fresh, names.place()?))
    }

    /// Moves the store onto the save point `saved` proves in `project`, the project file at
    /// `path`, after a save: with a fresh recovery file on it, or if none can be made, a
    /// scratch working file ([`scratch`]), crash protection paused. Then deletes the old
    /// recovery file; returns it if it couldn't be.
    ///
    /// The fresh file is empty, and on the new save point: a crash from then on leaves it as a
    /// valid recovery file of that save point, holding nothing since. So it is placed before the
    /// rebase, not filled first as a roll-over's is; the old file goes only once the new one is
    /// durable and in use. With a fresh file, a running writer goes on into it, and paused
    /// protection runs again with a new writer. With a scratch file, the writer is closed.
    ///
    /// Once this returns, the switch has happened: the store reads `project` and writes the
    /// new working file. On error nothing has changed.
    ///
    /// # Errors
    ///
    /// The fresh recovery file's error if no scratch file can be made either;
    /// [`ProjectError::Cache`] if the rebase is refused (the new file is deleted again).
    fn switch(
        &mut self,
        saved: Saved,
        project: &ProjectFile,
        path: &Path,
    ) -> Result<Option<KeptRecovery>, ProjectError> {
        let meta = project.meta().on_save_point(saved.save_point(), path);
        let (working, fresh_path, failed) = match self.fresh_recovery(meta.clone(), project) {
            Ok((fresh, at)) => (fresh, Some(at), None),
            Err(cause) => match scratch(meta, project) {
                Ok(scratch) => (scratch, None, Some(cause)),
                Err(_) => return Err(cause),
            },
        };
        let old = match self.store.rebase(saved, working) {
            Ok(old) => old,
            Err(e) => {
                if let Some(at) = &fresh_path {
                    // Best effort: the file is empty, and the error at hand is the one to report.
                    drop(fs::remove_file(at));
                }
                return Err(e.into());
            }
        };
        let kept = match std::mem::replace(&mut self.recovery_path, fresh_path) {
            Some(old_path) => self.let_go(old_path, old),
            None => None,
        };
        match failed {
            Some(cause) => {
                if let Some(writer) = self.writer.take() {
                    // Nothing it could write is protection now.
                    drop(writer.close(&mut self.store, Instant::now()));
                }
                drop(self.pause(cause, Instant::now()));
            }
            None if self.writer.is_some() => self.protected(),
            None => match RecoveryWriter::new(&mut self.store) {
                Ok(writer) => {
                    self.writer = Some(writer);
                    self.protected();
                }
                Err(e) => drop(self.pause(e.into(), Instant::now())),
            },
        }
        Ok(kept)
    }

    /// Deletes the old recovery file at `path`, which the project has moved on from
    /// ([`retire`]), while still holding it, and only then closes it; returns it if it couldn't
    /// be deleted.
    ///
    /// The handle that holds `path` is `old`, the store's old working file, unless the store had
    /// moved off a failed recovery file onto a scratch file ([`Project::strand`]): then it is the
    /// stranded file, and `old` is the scratch file, deleted by dropping it.
    ///
    /// A recovery file is unlocked only once it is gone, so [`find_recoveries`] never finds a
    /// running project's old file unheld and offers to restore it. One that can't be deleted is
    /// held until it is (tried again when protection is next put in place), or the project
    /// closes (or is dropped): only then is it listed.
    fn let_go(&mut self, path: PathBuf, old: ProjectFile) -> Option<KeptRecovery> {
        let holder = self.stranded.take().unwrap_or(old);
        let kept = retire(&path).err();
        if kept.is_some() {
            self.held.push((path, holder));
        }
        kept
    }

    /// Moves the store off a failed recovery file onto an unlinked scratch file on disk-backed
    /// storage, so later write-backs never reach the failed one: after a roll-over failed, for
    /// `cause`, with crash protection pausing. The failed file is kept, held, as the recovery
    /// file on disk ([`Project::stranded`]), at its last durable save point.
    ///
    /// The scratch file is filled with every edit ([`TileStore::fill_roll_over`]) and swapped in
    /// ([`TileStore::roll_over`]), which clears the store's failed mark. It is made in the
    /// project file's directory, when the project has been saved, else (or if that fails) in the
    /// recovery directory, where the roll-over just failed, so it often fails too. If the
    /// roll-over failed for want of space ([`is_out_of_space`]), the recovery directory is
    /// skipped: filling a file there again would copy every edit since the last save on the
    /// drawing thread only to fail the same way. Never in the system's temporary directory: that
    /// is often memory-backed (tmpfs), and filling the file there would put every edit since the
    /// last save into memory outside the store's budget, under the very pressure that writes
    /// tiles back.
    ///
    /// A crash between making the scratch file and unlinking it leaves a stray
    /// `brushcraft-<project id>-<suffix>.scratch` in that directory, the project file's one
    /// included. No listing takes it for a recovery file (their names end in `.bcrf`); making it
    /// unnamed from the start (`O_TMPFILE`) would leave nothing, and is a later change.
    ///
    /// Best effort: does nothing unless the working file is a failed recovery file
    /// ([`TileStore::is_working_failed`]); if no scratch file can be made or filled, the store
    /// stays where it was, and [`TileStore::save`] refuses to write a save point there.
    fn strand(&mut self, cause: &ProjectError) {
        if !self.store.is_working_failed() {
            return;
        }
        debug_assert!(
            self.recovery_path.is_some(),
            "a failed recovery file is the project's recovery file on disk"
        );
        debug_assert!(
            self.stranded.is_none(),
            "a file stranded before was released when the store moved to a new one"
        );
        if self.stranded.is_some() {
            return;
        }
        let skip_recovery_dir = is_out_of_space(cause);
        let dirs: Vec<PathBuf> = self
            .path
            .as_deref()
            .and_then(Path::parent)
            .map(Path::to_path_buf)
            .into_iter()
            .chain((!skip_recovery_dir).then(|| self.recovery_dir.clone()))
            .collect();
        for dir in dirs {
            let moved = self.strand_in(&dir);
            #[cfg(test)]
            self.strand_tries.push((dir, moved));
            if moved {
                return;
            }
        }
    }

    /// [`Project::strand`] in `dir`: makes a scratch file there, fills it and moves the store
    /// onto it; whether it did. A failed fill changes nothing in the store, so the caller can try
    /// the next directory.
    fn strand_in(&mut self, dir: &Path) -> bool {
        let meta = self.store.working().meta().clone();
        let Ok(file) = scratch_in(dir, meta, self.store.working()) else {
            return false;
        };
        let Ok(filled) = self.store.fill_roll_over(file) else {
            return false;
        };
        // The store can't have moved on since the fill, within this call.
        let Ok((old, lost)) = self.store.roll_over(filled) else {
            return false;
        };
        self.unreported.extend(lost);
        self.stranded = Some(old);
        true
    }
}

/// Whether `e`, or an error it comes from, is an I/O error for want of space (`ENOSPC`).
fn is_out_of_space(e: &ProjectError) -> bool {
    let first: &(dyn std::error::Error + 'static) = e;
    std::iter::successors(Some(first), |err| err.source()).any(|err| {
        err.downcast_ref::<io::Error>()
            .is_some_and(|io| io.kind() == io::ErrorKind::StorageFull)
    })
}

/// Test-only controls.
#[cfg(test)]
impl Project {
    /// Makes the next placement of a recovery file fail once its temporary file is written and
    /// synced, before the rename: what a crash mid-create leaves.
    pub(crate) const fn fail_next_recovery(&mut self) {
        self.fail_next_recovery_with(io::ErrorKind::Other);
    }

    /// As [`Project::fail_next_recovery`], failing with an I/O error of `kind`: a full disk
    /// ([`io::ErrorKind::StorageFull`]), say.
    pub(crate) const fn fail_next_recovery_with(&mut self, kind: io::ErrorKind) {
        self.fail_next_recovery = Some(kind);
    }

    /// Makes the directory sync after the next save as moves its new file into place fail, as
    /// a sync that fails after the rename would.
    pub(crate) const fn fail_dir_sync(&mut self) {
        self.fail_dir_sync = true;
    }

    /// Each directory [`Project::strand`] has tried a scratch file in, in order, and whether
    /// the store moved onto it there.
    pub(crate) fn strand_tries(&self) -> &[(PathBuf, bool)] {
        &self.strand_tries
    }

    /// Whether a failed recovery file is stranded: held while the store works on a scratch file.
    pub(crate) const fn is_stranded(&self) -> bool {
        self.stranded.is_some()
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

#[cfg(test)]
mod tests;
