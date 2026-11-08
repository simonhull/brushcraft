//! Finding the recovery files a crash left: which can be restored, in the order to choose them,
//! and the head ops a new session's ops must start past.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};
use std::{fs, io};

use rustix::fs::OFlags;

use super::{CandidateStatus, EXTENSION, ProjectError, RecoveryCandidate};
use crate::OpId;
use crate::file::{BaseFile, FileError, ProjectFile, open_regular};
use crate::meta::{Meta, ProjectId, Role};
use crate::tiles::{CacheError, TileStore};

/// `opened`, its op generator moved past `heads` ([`TileStore::skip_ops_past`]).
///
/// # Errors
///
/// `opened`'s error; [`CacheError::OpsExhausted`] if a head is the site's last op id.
pub(super) fn past(
    opened: Result<TileStore, CacheError>,
    heads: Vec<Option<OpId>>,
) -> Result<TileStore, CacheError> {
    let mut store = opened?;
    store.skip_ops_past(heads)?;
    Ok(store)
}

/// The newest ops of the recovery files of `project` in `dir`, on any save point.
///
/// Each is [`ProjectFile::last_op`] (the head, or a stroke still landing that has landed
/// nothing): the newest work the file would restore, which a new session's ops must be newer
/// than.
///
/// Best effort, and read-only: a missing or unreadable `dir` has none, and a file a running
/// instance holds, or one that can't be read, is passed over. A file that can't be read is
/// listed as damaged, with no last edit, so it never competes with this session's; one held
/// is this session's own (a restored file), or another instance's, which only one restored
/// onto an earlier save point can be, as the project file is locked.
pub(super) fn recovery_heads(dir: &Path, project: ProjectId) -> Vec<Option<OpId>> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    entries
        .filter_map(Result::ok)
        .filter(|entry| entry.file_name().to_str().and_then(recovery_name) == Some(project))
        .filter_map(|entry| ProjectFile::inspect(&entry.path()).ok())
        .filter(|file| file.meta().project() == project)
        .map(|file| file.last_op())
        .collect()
}

/// How long a recovery file's temporary file must have gone unwritten to be a crash's leftover.
///
/// [`find_recoveries`] removes `<name>.tmp` files older than this: an instance creating one
/// writes and syncs it in well under it.
const STALE_TEMPORARY: Duration = Duration::from_secs(60);

/// Lists the recovery files in `dir` that no running instance holds, each with whether it can
/// be restored ([`Project::restore`](super::Project::restore)) and, if not, why (see the notes
/// of the `project` module).
///
/// The list is ordered for choosing: projects with the most recently written recovery file
/// first, and a project's files on one save point together, **the one with the newest last
/// edit first** (then the most recently written). That one is the file to restore. One project
/// can leave several files on one save point: sessions that each opened it at that save point
/// and crashed, or a crash mid-roll-over, which leaves the old file and the new one holding the
/// same edits. Ops only grow across sessions, as every session's start past the newest edit of
/// every recovery file of the project ([`Project::open`](super::Project::open),
/// [`Project::restore`](super::Project::restore)), so the newest last edit is the latest session's
/// latest work. A write time can't say that: copying or touching a file moves it, and a clock can
/// be set back.
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
/// it. Only a regular file is opened to check (see [`open_regular`]): anything else at a
/// temporary file's name, a FIFO say, is left as it is, never waited on. Best effort: a file
/// that can't be checked or removed is left for the next listing.
fn sweep(path: &Path) {
    let stale = fs::metadata(path)
        .and_then(|meta| meta.modified())
        .ok()
        .and_then(|modified| modified.elapsed().ok())
        .is_some_and(|age| age >= STALE_TEMPORARY);
    if !stale {
        return;
    }
    let Ok(file) = open_regular(path, OFlags::RDONLY) else {
        return;
    };
    // The lock is held across the removal, so an instance can't take the file meanwhile.
    if file.try_lock().is_ok() {
        drop(fs::remove_file(path));
    }
}

/// The project id a recovery file's name gives, if `name` is one: `<32 hex digits>-<16 hex
/// digits>.bcrf`, as [`create_recovery`](super::place::create_recovery) names them.
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
        Err(FileError::Locked) => return None,
        Err(FileError::Io(e)) if e.kind() == io::ErrorKind::NotFound => return None,
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
        last_edit: file.last_op(),
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
/// [`FileError::Damaged`] if a save point of the project file no longer loads;
/// [`FileError::Io`].
pub(super) fn standing(
    meta: &Meta,
    project: &ProjectFile,
) -> Result<(CandidateStatus, Option<BaseFile>), FileError> {
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
    let latest = project.base();
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
pub(super) fn not_restorable(e: FileError) -> ProjectError {
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
fn unavailable(e: &FileError) -> Option<CandidateStatus> {
    if let FileError::Io(io) = e
        && io.kind() == io::ErrorKind::NotFound
    {
        Some(CandidateStatus::ProjectMissing)
    } else if matches!(e, FileError::Locked) {
        Some(CandidateStatus::ProjectInUse)
    } else {
        None
    }
}
