//! Opening a store over its layers, saving it, and moving it onto a new save point or a fresh
//! working file.
//!
//! # Protocol
//!
//! [`crate::Project`] (`project.rs`) drives these:
//!
//! - It opens the store with [`TileStore::open_layered`]: the working file over the base, which
//!   must belong together.
//! - A save is two calls. [`TileStore::save_into`] copies every tile edited since the last rebase,
//!   with every bound changed since, into the project file and saves it; nothing in the store
//!   changes, and it returns the proof a rebase needs ([`Saved`]). [`TileStore::rebase`] then swaps
//!   in that save point as the base and a fresh working file over it, and refuses a proof the store
//!   was written after.
//! - After the working file fails to sync, a roll-over is two calls too.
//!   [`TileStore::fill_roll_over`] fills a fresh working file on the same base with every edit,
//!   changing nothing in the store; [`TileStore::roll_over`] swaps it in once the caller has named
//!   it, and refuses it if the store was written since.
//! - With no recovery writer running, it calls [`TileStore::drain_saves`] on every call that would
//!   have advanced one, so the change feed's backlog stays bounded between saves.

use std::collections::{BTreeSet, HashMap};
use std::rc::Rc;

use geom::TileKey;

#[cfg(test)]
use super::TestHooks;
use super::config::{Headroom, StoreConfig, reserved_shares};
use super::display::display_book_of;
use super::error::{CacheError, Invariant};
use super::layers::{
    Dirty, Filled, Saved, book_of, check_layers, first_op, saved_base, write_slot,
};
use super::{StoreId, TileStore};
use crate::activity::Activity;
use crate::deps::DepGraph;
use crate::display::is_display;
use crate::feed::{ChangeFeed, ConsumerId};
use crate::file::{BaseFile, ProjectFile, ReadScratch};
use crate::hash::TileMap;
use crate::kept::KeptLru;
use crate::meta::Role;
use crate::op::RESERVED_SITE;
use crate::pool::{Budget, Pool};
use crate::prefetch::Prefetcher;
use crate::stamp::Stamp;
use crate::workers::StartWorkersError;

impl TileStore {
    /// A store over `working`, the file it writes, with no base under it: a project file read
    /// and written directly, or the recovery file of a project never saved, which holds all of
    /// its work. The base mesh is the working file's.
    ///
    /// A project with a save is opened over its save point through [`crate::Project`], which
    /// layers a recovery file on it.
    ///
    /// The op generator starts after the working file's newest head op of `config.site`, else
    /// at sequence 0 of the site, so the store's edits are newer than everything it reads.
    ///
    /// # Errors
    ///
    /// [`CacheError::ReservedSite`]; [`CacheError::WrongBase`] if `working` is a recovery file
    /// that names a save point (it needs that save point under it);
    /// [`CacheError::BudgetTooSmall`] below [`StoreConfig::minimum_budget`] without the reserved
    /// shares, [`CacheError::TooManyKeptPredictions`] below it with the kept predictions',
    /// [`CacheError::TooManyStrokeTiles`] below it with the stroke state's besides;
    /// [`CacheError::OpsExhausted`] if the file's head is the site's last op id;
    /// [`CacheError::Store`] if the read buffers can't be made; [`CacheError::Workers`] if the
    /// evaluation workers can't be started.
    pub fn open(working: ProjectFile, config: StoreConfig) -> Result<Self, CacheError> {
        Self::open_layered(working, None, config)
    }

    /// A store over `working`, the file it writes, and `base`, the project file as of its last
    /// save, read where `working` has no record (`None` for a project never saved). The base
    /// mesh is the working file's.
    ///
    /// The files must belong together (see [`check_layers`]): over a base, `working` is a
    /// recovery file whose meta names the base's project and save point, offset and head op.
    ///
    /// The op generator starts after the newest head op of `config.site` in either file (all of
    /// the base's file, records after its save point included), else at sequence 0 of the site,
    /// so the store's edits are newer than everything it reads or might copy.
    ///
    /// The tiles a recovery file already holds (one restored after a crash) count as edits no
    /// save has made part of the project file: the next save copies them.
    ///
    /// # Errors
    ///
    /// [`CacheError::ReservedSite`]; [`CacheError::MeshMismatch`] if the base's mesh hash isn't
    /// the working file's; [`CacheError::WrongBase`] if the files don't belong together;
    /// [`CacheError::BudgetTooSmall`] below [`StoreConfig::minimum_budget`] without the reserved
    /// shares, [`CacheError::TooManyKeptPredictions`] below it with the kept predictions',
    /// [`CacheError::TooManyStrokeTiles`] below it with the stroke state's besides;
    /// [`CacheError::OpsExhausted`] if a file's head is the site's last op id;
    /// [`CacheError::Store`] if the read buffers can't be made; [`CacheError::Workers`] if the
    /// evaluation workers can't be started.
    pub(crate) fn open_layered(
        working: ProjectFile,
        base: Option<BaseFile>,
        config: StoreConfig,
    ) -> Result<Self, CacheError> {
        if config.site == RESERVED_SITE {
            return Err(CacheError::ReservedSite);
        }
        let mesh_hash = working.meta().mesh_hash();
        if base
            .as_ref()
            .is_some_and(|base| base.meta().mesh_hash() != mesh_hash)
        {
            return Err(CacheError::MeshMismatch);
        }
        check_layers(working.meta(), base.as_ref())?;
        let mesh = working.mesh();
        let headroom = Headroom::of(&mesh);
        let (reserved_bytes, cache_bytes) =
            reserved_shares(headroom, &config, config.floor_bytes(&mesh))?;
        let cache_caps = crate::jobs::CacheCaps::of(cache_bytes);
        let next_op = first_op(
            config.site,
            [working.head(), base.as_ref().and_then(BaseFile::head)],
        )?;
        // A recovery file holds only edits no save has: a restored one's are still to save.
        let since_rebase = match working.meta().role() {
            // Display records aren't edits: the display book carries them.
            Role::Recovery { .. } => working.keys().copied().filter(is_correction).collect(),
            Role::Project => BTreeSet::new(),
        };
        let bounds = book_of(&working, base.as_ref())?;
        let display = display_book_of(&working, base.as_ref(), config.display, mesh.face_count());
        let workers = rayon::ThreadPoolBuilder::new()
            .num_threads(config.worker_threads())
            .thread_name(|n| format!("tile-eval-{n}"))
            .build()
            .map_err(|e| CacheError::Workers(StartWorkersError::new(e)))?;
        let mut feed = ChangeFeed::new();
        let save_consumer = feed.subscribe();
        let mut store = Self {
            prefetch: Prefetcher::new(),
            mesh,
            mesh_hash,
            file: working,
            base,
            budget: Budget::new(config.budget_bytes - reserved_bytes),
            headroom,
            corrections: Pool::with_eviction_log(config.chunk_bytes),
            // Short chunks let the evaluated pool use the room the reserved shares
            // leave below a whole chunk (see "Kept predictions" on `TileStore`).
            evaluated: Pool::with_eviction_log(config.chunk_bytes).with_short_chunks(),
            deps: DepGraph::with_reach_memo(cache_caps.reach),
            unsaved: TileMap::default(),
            since_rebase,
            last_write: Stamp::ZERO,
            known_ops: TileMap::default(),
            generation: 0,
            working_failed: false,
            bounds,
            display,
            display_finest: config.display,
            jobs: std::collections::VecDeque::new(),
            bumped: std::collections::BTreeMap::new(),
            display_failed: std::collections::BTreeMap::new(),
            refresh_seen: u64::MAX,
            display_builds: 0,
            display_generations: std::collections::BTreeMap::new(),
            display_decoded: crate::jobs::DecodedTiles::new(cache_caps.decoded),
            read_ahead: super::read_ahead::ReadAhead::new(cache_caps.read_ahead),
            cache_caps,
            bump_readers: None,
            unbumped: BTreeSet::new(),
            stale: crate::stale::StaleFeed::new(config.stale_cap),
            epoch_writes: BTreeSet::new(),
            epoch_keys: BTreeSet::new(),
            floor: std::collections::BTreeMap::new(),
            floor_top: config.floor_top,
            feed,
            save_consumer,
            streams: HashMap::new(),
            next_op: Some(next_op),
            op: None,
            pins: Rc::default(),
            pin_sets: HashMap::new(),
            next_pin_set: 0,
            scratch: Vec::new(),
            workers,
            read_scratch: ReadScratch::new()?,
            worker_read_scratch: Vec::new(),
            kept: KeptLru::new(config.kept_predictions),
            stroke_tiles: config.stroke_tiles,
            id: StoreId::next(),
            edits: 0,
            own_edits: BTreeSet::new(),
            activity: Activity::default(),
            #[cfg(test)]
            hooks: TestHooks::default(),
        };
        if store.load_floor()? {
            // A layer built for another surface: none of it is trusted. Every tile is made
            // unknown, so no job takes a record the files hold for one it doesn't need to build.
            if let Some(finest) = store.display_finest {
                store.forget_foreign_display(finest);
                store.queue_display_rebuild(finest);
            }
        } else {
            store.queue_missing_display();
        }
        Ok(store)
    }

    /// Copies every tile edited since the last rebase (or since the store opened) into
    /// `project`, with every bound changed since, saves it, and returns the proof a rebase needs:
    /// the new save point as a base, and the store's latest write and bounds change.
    ///
    /// The bounds written describe the state the save point names: a tile whose latest edit is
    /// lost (below) has every bound its edit could change written as unknown.
    ///
    /// Each tile is copied from where its latest edit is: a dirty slot is encoded and written
    /// with the op of its latest write, else the working file's record is copied as it is,
    /// keeping its original op, else, if that record no longer reads, a clean resident slot
    /// (which holds what the record did) is written with the record's op as the store last knew
    /// it. A tile whose latest edit none of these gives is lost, and listed in the proof
    /// ([`Saved::lost`]), so one lost tile never blocks the saves after it: a broken dirty slot
    /// is saved as the working file's record, if that reads, and a tile neither way left out,
    /// at its last saved state. Save's
    /// consumer of the change feed is drained first; the tiles it lists stay listed until a
    /// rebase, so a save that fails, or one followed by more edits, is simply made again.
    /// Nothing in the store changes: dirty slots stay dirty until the rebase, which is what
    /// makes the save's copies the saved state.
    ///
    /// A project being saved under a new name has its old save point copied in first (see
    /// [`ProjectFile::copy_base`]); this adds the edits.
    ///
    /// # Errors
    ///
    /// [`CacheError::WrongBase`] if `project` isn't a project file; [`CacheError::MeshMismatch`]
    /// if it is over another mesh; [`CacheError::Save`] if a write to it or its save point
    /// fails; [`CacheError::Internal`] for an edited tile that has no record in the working file
    /// and isn't dirty ([`Invariant::NothingToSave`]: every edit is in one or the other), or a
    /// tile of the recovery file that wasn't copied ([`Invariant::NotInSave`], checked before
    /// the save point is written), or a save point whose derived maps disagree with it
    /// ([`Invariant::SavePointMismatch`], with nothing written: not retryable).
    pub(crate) fn save_into(&mut self, project: &mut ProjectFile) -> Result<Saved, CacheError> {
        self.sync_pins()?;
        if project.meta().role() != Role::Project {
            return Err(CacheError::WrongBase);
        }
        if project.meta().mesh_hash() != self.mesh_hash {
            return Err(CacheError::MeshMismatch);
        }
        self.drain_saves();
        let mut lost = BTreeSet::new();
        for key in &self.since_rebase {
            let resident = self.corrections.find(key);
            let tile = |id| {
                self.corrections
                    .get(id)
                    .and_then(|slot| slot.to_tile().ok())
            };
            let dirty = resident.filter(|&id| self.corrections.is_dirty(id) == Some(true));
            // The dirty slot: the latest edit.
            if let Some(id) = dirty {
                let op = self.unsaved.get(key).map(|write| write.op);
                if let (Some(op), Some(tile)) = (op, tile(id)) {
                    project
                        .write_displacement(*key, op, &tile)
                        .map_err(CacheError::Save)?;
                    continue;
                }
                // Broken: the latest edit is lost, and the record is the best left of it.
                lost.insert(*key);
            }
            // The working file's record.
            match self.file.read_tile_record(key) {
                Ok(Some(raw)) => {
                    project.append_copy(key, &raw).map_err(CacheError::Save)?;
                    continue;
                }
                Ok(None) if dirty.is_none() => {
                    return Err(Invariant::NothingToSave(*key).into());
                }
                Ok(None) => continue,
                Err(_) => {}
            }
            // The record is unreadable, but a clean resident slot holds what it did.
            let clean = resident.filter(|_| dirty.is_none());
            let op = clean.and_then(|_| self.known_ops.get(key).copied());
            match (op, clean.and_then(tile)) {
                (Some(op), Some(tile)) => project
                    .write_displacement(*key, op, &tile)
                    .map_err(CacheError::Save)?,
                _ => {
                    lost.insert(*key);
                }
            }
        }
        // Every tile of a recovery file is an edit since the last save point, so it must have
        // been copied: a save point without one would lose it once the store is rebased.
        let recovery = matches!(self.file.meta().role(), Role::Recovery { .. });
        if let Some(missing) = self
            .file
            .keys()
            .find(|key| recovery && is_correction(key) && !self.since_rebase.contains(key))
        {
            return Err(Invariant::NotInSave(*missing).into());
        }
        // The bounds of the state the save point names: the lost tiles' edits aren't in it.
        self.bounds.settle(&self.mesh, &self.deps)?;
        self.bounds
            .forget(&self.mesh, &self.deps, lost.iter().copied())?;
        project
            .write_bounds_many(
                self.bounds.all_changed().map(|(key, kept)| (*key, kept)),
                Stamp::ZERO,
            )
            .map_err(CacheError::Save)?;
        self.save_display(project, &lost)?;
        project
            .save()
            .map_err(|e| CacheError::from_save_point(e, CacheError::Save))?;
        let base = saved_base(project)?;
        Ok(Saved {
            base,
            stamp: self.last_write,
            lost,
            bounds: self.derived_changes(),
        })
    }

    /// Swaps in the save point `saved` proves, as the base, and `working`, a fresh working file
    /// over it; returns the working file it replaces.
    ///
    /// `saved` must be current: made by [`TileStore::save_into`] with no write since. Then every
    /// tile edited since the last rebase is in the new save point, and the old working file's
    /// records are no longer needed. That is checked, not assumed: every tile of the old
    /// working file and of the old base must be in the new save point (the old base's checked
    /// one by one only if the save point isn't known to extend it, see
    /// [`TileStore::missing_from_save`]), but for those the save listed as lost ([`Saved::lost`]),
    /// whose last saved state the store reads from now on (a lost tile's resident slot is
    /// dropped, or, if a pin holds it, filled again in place from the saved record, and every
    /// evaluation that followed from it is dropped too, as it may have come from the lost
    /// edit). Dirty slots are clean after the rebase (the save wrote them), and every other
    /// resident correction tile is kept, as it is what was just saved; so are the other cached
    /// evaluations, which stay valid: nothing they read has changed. Prefetches in flight are
    /// dropped (their workers stop and are joined), and later ones read the new files. The op
    /// generator continues, past the base's head if that is newer, and the current op ends: the
    /// next edit starts a new one, so the edits after a save point are newer than its head.
    ///
    /// # Errors
    ///
    /// [`CacheError::StaleSave`] if the store was written after `saved` was made;
    /// [`CacheError::MeshMismatch`] if either file isn't over the store's mesh;
    /// [`CacheError::WrongBase`] if `working` isn't a recovery file naming `saved`'s save point;
    /// [`CacheError::Internal`] with [`Invariant::NotInSave`] if a tile the store could read is
    /// missing from the save (a bug in the save); [`CacheError::OpsExhausted`] if the base's
    /// head is the site's last op id. The store is unchanged then, and the files are dropped.
    pub(crate) fn rebase(
        &mut self,
        saved: Saved,
        working: ProjectFile,
    ) -> Result<ProjectFile, CacheError> {
        self.sync_pins()?;
        let Saved {
            base,
            stamp,
            lost,
            bounds,
        } = saved;
        if stamp != self.last_write || bounds != self.derived_changes() {
            return Err(CacheError::StaleSave);
        }
        if base.meta().mesh_hash() != self.mesh_hash || working.meta().mesh_hash() != self.mesh_hash
        {
            return Err(CacheError::MeshMismatch);
        }
        check_layers(working.meta(), Some(&base))?;
        if let Some(missing) = self.missing_from_save(&base, &lost) {
            return Err(Invariant::NotInSave(missing).into());
        }
        let next_op = match self.next_op {
            Some(op) => {
                let after = first_op(op.site(), [working.head(), base.head()])?;
                Some(op.max(after))
            }
            None => None,
        };
        // The save wrote every dirty slot as it is now (no edit since), so none is unsaved.
        let dirty: Vec<(TileKey, Dirty)> = self.unsaved.drain().collect();
        for (key, write) in dirty {
            if let Some(id) = self.corrections.find(&key) {
                self.corrections.set_dirty(id, false)?;
                self.known_ops.insert(key, write.op);
            }
        }
        // A lost tile's slot doesn't hold the saved state, and has nothing to write: it is
        // dropped, or, if a pin holds it, reloaded in place once the store reads the new files.
        let mut pinned = Vec::new();
        for key in &lost {
            if self.drop_lost_slot(key)? {
                pinned.push(*key);
            }
        }
        // What was evaluated from a lost edit isn't the saved state the store reads from now on.
        self.invalidate_edited(lost.iter().copied());
        for key in &lost {
            self.note_epoch_write(*key);
        }
        self.close_epoch()?;
        self.forget_evicted_ops();
        #[cfg(test)]
        self.assert_known_ops_resident();
        self.since_rebase.clear();
        self.bounds.rebased();
        self.display.rebased();
        self.generation += 1;
        self.working_failed = false;
        // The old workers read the old files; dropping them joins them and drops their results.
        self.prefetch = Prefetcher::new();
        self.next_op = next_op;
        self.op = None;
        self.base = Some(base);
        let old = std::mem::replace(&mut self.file, working);
        for key in &pinned {
            self.reload_lost(key);
        }
        Ok(old)
    }

    /// The first tile the store could read that `saved`, a new save point, lacks, but for
    /// `lost`, the tiles the save listed as lost; `None` if it lacks none. What a rebase onto
    /// it checks, so that no tile the store reads is lost by moving on (a bug in a save).
    ///
    /// The tiles the store can read are the working file's, every tile edited since the last
    /// rebase, and the old base's. The first two are what the save wrote, and grow with the
    /// edits; the old base's are every tile of the project. So when the new save point is known
    /// to hold every tile of the old base ([`BaseFile::extends`]: it was made from the old
    /// base's save point, in the same open file, with records only put in), they aren't looked
    /// up one by one, and the check costs what the save wrote. Otherwise (a project saved under
    /// a new name, a save point loaded from the file, or one that doesn't descend from the
    /// base at all) every tile of the old base is looked up too.
    ///
    /// Tests run the full check as well whenever the short one passes.
    ///
    /// # Panics
    ///
    /// In tests only, if the full check finds a tile the short one missed: a save point taken
    /// to extend the base that doesn't.
    fn missing_from_save(&self, saved: &BaseFile, lost: &BTreeSet<TileKey>) -> Option<TileKey> {
        let old_base = self.base.as_ref().filter(|old| !saved.extends(old));
        let missing = self.missing_among(saved, lost, old_base);
        #[cfg(test)]
        if missing.is_none() {
            assert_eq!(
                self.missing_among(saved, lost, self.base.as_ref()),
                None,
                "a save point taken to extend the base lacks a tile of it"
            );
        }
        missing
    }

    /// The first tile of the working file, of those edited since the last rebase, or of
    /// `old_base` if given, that `saved` lacks and `lost` doesn't list (see
    /// [`TileStore::missing_from_save`]).
    fn missing_among(
        &self,
        saved: &BaseFile,
        lost: &BTreeSet<TileKey>,
        old_base: Option<&BaseFile>,
    ) -> Option<TileKey> {
        self.file
            .keys()
            .chain(&self.since_rebase)
            .chain(old_base.into_iter().flat_map(BaseFile::keys))
            .filter(|key| is_correction(key))
            .find(|key| !saved.contains(key) && !lost.contains(key))
            .copied()
    }

    /// Fills `fresh`, a new working file on the same base, with every edit, after the working
    /// file failed to sync: the first half of a roll-over, which changes nothing in the store.
    /// [`TileStore::roll_over`] swaps it in, once the caller has given it its name.
    ///
    /// Every tile of the old working file and every dirty slot is written to `fresh`, read
    /// through the store: a dirty slot as it is, with the op of its latest write; any other
    /// tile as the old file's record, copied after its checksum is checked. A clean resident
    /// tile whose record no longer reads is written from its slot, which holds what the record
    /// did, labelled with the record's op as the store last knew it (see
    /// [`TileStore::known_ops`]): never with an op that didn't edit it. A tile that can be read
    /// neither way, or a broken slot, is lost: listed, and left out. Then `fresh` is saved
    /// (synced, a save point, synced again). Every bound of the old working file and every bound
    /// changed since the last rebase go across too, as they are now; those a lost tile's edit could
    /// change, as unknown.
    ///
    /// # Errors
    ///
    /// [`CacheError::MeshMismatch`] or [`CacheError::WrongBase`] if `fresh` isn't over the
    /// store's mesh, or doesn't name the same base as the working file; [`CacheError::WriteBack`]
    /// if writing or saving `fresh` fails; [`CacheError::Internal`] if its save point's derived
    /// maps disagree with it ([`Invariant::SavePointMismatch`]).
    pub(crate) fn fill_roll_over(&mut self, mut fresh: ProjectFile) -> Result<Filled, CacheError> {
        self.sync_pins()?;
        if fresh.meta().mesh_hash() != self.mesh_hash {
            return Err(CacheError::MeshMismatch);
        }
        if fresh.meta().role() != self.file.meta().role()
            || fresh.meta().project() != self.file.meta().project()
        {
            return Err(CacheError::WrongBase);
        }
        check_layers(fresh.meta(), self.base.as_ref())?;
        let keys: BTreeSet<TileKey> = self
            .file
            .keys()
            .copied()
            .filter(is_correction)
            .chain(self.unsaved.keys().copied())
            .collect();
        let mut lost = Vec::new();
        let mut cleaned = Vec::new();
        for key in keys {
            let resident = self.corrections.find(&key);
            let dirty = resident.filter(|&id| self.corrections.is_dirty(id) == Some(true));
            let from_slot = match dirty {
                Some(_) => self.unsaved.get(&key).map(|write| write.op),
                None => match self.file.read_tile_record(&key) {
                    Ok(Some(raw)) => {
                        fresh
                            .append_copy(&key, &raw)
                            .map_err(CacheError::WriteBack)?;
                        continue;
                    }
                    // Unreadable (or, impossibly, gone): only the slot holds it now.
                    Ok(None) | Err(_) if resident.is_some() => self.known_ops.get(&key).copied(),
                    Ok(None) | Err(_) => None,
                },
            };
            let tile = resident
                .and_then(|id| self.corrections.get(id))
                .and_then(|slot| slot.to_tile().ok());
            let (Some(op), Some(tile)) = (from_slot, tile) else {
                lost.push(key);
                continue;
            };
            fresh
                .write_displacement(key, op, &tile)
                .map_err(CacheError::WriteBack)?;
            if let Some(id) = dirty {
                cleaned.push((key, id));
            }
        }
        self.bounds.settle(&self.mesh, &self.deps)?;
        self.bounds
            .forget(&self.mesh, &self.deps, lost.iter().copied())?;
        let carried: BTreeSet<TileKey> =
            self.file.bounds_locations().map(|(key, _)| *key).collect();
        for &key in &carried {
            if self.bounds.changed(&key).is_none() {
                let copied = match self.file.read_bounds_record(&key) {
                    Ok(Some(raw)) => fresh.append_bounds_copy(&key, &raw),
                    // A bound is derived: one that doesn't read goes across as unknown.
                    Ok(None) | Err(_) => fresh.write_bounds(key, None, Stamp::ZERO),
                };
                copied.map_err(CacheError::WriteBack)?;
            }
        }
        // What the old file held, as it is now, and what a flush would write; not the derived
        // bounds, which no recovery file holds.
        let bounds = &self.bounds;
        fresh
            .write_bounds_many(
                bounds
                    .all_changed()
                    .filter(|(key, _)| carried.contains(key) || bounds.is_unflushed(key))
                    .map(|(key, kept)| (*key, kept)),
                Stamp::ZERO,
            )
            .map_err(CacheError::WriteBack)?;
        self.carry_display(&mut fresh, &lost)?;
        fresh
            .save()
            .map_err(|e| CacheError::from_save_point(e, CacheError::WriteBack))?;
        Ok(Filled {
            file: fresh,
            lost,
            cleaned,
            stamp: self.last_write,
            bounds: self.derived_changes(),
        })
    }

    /// Swaps in the working file [`TileStore::fill_roll_over`] filled; returns the old working
    /// file and the tiles that couldn't be carried across (lost).
    ///
    /// Every slot written to the new file is clean, streaming consumers' marks are emptied (all
    /// they held is in it), and the generation moves on, so a flush started before is
    /// abandoned. The base is unchanged, and so are the tiles to save into the project, but for
    /// the lost ones: they are in no file the store writes, so the next save leaves them at
    /// their last saved state instead of failing to find them. A lost tile's slot, if it has
    /// one (a broken one), is dropped, or, if a pin holds it, filled again in place from the
    /// tile's last record, and every evaluation that followed from a lost tile is dropped, as it
    /// may have come from the lost edit.
    ///
    /// # Errors
    ///
    /// [`CacheError::StaleSave`] if the store was written since `filled` was: the new file
    /// doesn't hold those writes. The store is unchanged then, and the file is dropped.
    /// [`CacheError::Internal`] for a stale handle.
    pub(crate) fn roll_over(
        &mut self,
        filled: Filled,
    ) -> Result<(ProjectFile, Vec<TileKey>), CacheError> {
        let Filled {
            file,
            lost,
            cleaned,
            stamp,
            bounds,
        } = filled;
        if stamp != self.last_write || bounds != self.derived_changes() {
            return Err(CacheError::StaleSave);
        }
        // Drained first, so the feed can't bring a lost key back.
        self.drain_saves();
        // A lost tile's slot, if it has one, is broken: dropped as a rebase drops one, or, if a
        // pin holds it, reloaded in place once the store reads the new file.
        let mut pinned = Vec::new();
        for key in &lost {
            self.since_rebase.remove(key);
            if self.drop_lost_slot(key)? {
                pinned.push(*key);
            }
        }
        // What was evaluated from a lost edit isn't what the store reads from now on.
        self.invalidate_edited(lost.iter().copied());
        for key in &lost {
            self.note_epoch_write(*key);
        }
        self.close_epoch()?;
        for (key, id) in cleaned {
            self.corrections.set_dirty(id, false)?;
            if let Some(write) = self.unsaved.remove(&key) {
                self.known_ops.insert(key, write.op);
            }
        }
        self.forget_evicted_ops();
        #[cfg(test)]
        self.assert_known_ops_resident();
        let marks: Vec<ConsumerId> = self.streams.keys().copied().collect();
        for id in marks {
            if let Some(mark) = self.streams.get_mut(&id) {
                mark.clear();
            }
        }
        // The new file holds every bound as it is now.
        self.bounds.all_flushed();
        self.display.all_flushed();
        self.generation += 1;
        self.working_failed = false;
        // The workers' jobs name records of the old file.
        self.prefetch = Prefetcher::new();
        let old = std::mem::replace(&mut self.file, file);
        for key in &pinned {
            self.reload_lost(key);
        }
        Ok((old, lost))
    }

    /// The changes counted to the bounds and the display tiles the store keeps: a save or a
    /// roll-over made before a later one doesn't hold it.
    pub(super) const fn derived_changes(&self) -> u64 {
        self.bounds.changes().wrapping_add(self.display.changes())
    }

    /// Moves every key save's consumer hasn't drained from the change feed into
    /// [`TileStore::since_rebase`].
    ///
    /// Draining early changes nothing a save sees, as `since_rebase` is a set. A save drains it
    /// ([`TileStore::save_into`], [`TileStore::save`]), as do a roll-over and a lost tile, and so
    /// does every flush ([`TileStore::snapshot_dirty`]); with no recovery writer running, the
    /// project drains it on every call that would have advanced one
    /// ([`crate::Project::with_writer`]). So the feed's backlog stays bounded between saves.
    pub(crate) fn drain_saves(&mut self) {
        let drained = self.feed.drain(self.save_consumer);
        self.since_rebase
            .extend(drained.into_iter().map(|(key, _)| key));
    }

    /// The change feed's backlog: entries some consumer hasn't drained yet.
    #[cfg(test)]
    pub(crate) fn feed_backlog(&self) -> usize {
        self.feed.backlog()
    }

    /// Whether the store holds edits no save has made part of a save point of the project
    /// file: any edit since the last rebase, or since it opened.
    pub(crate) fn has_unsaved_edits(&self) -> bool {
        !self.since_rebase.is_empty() || self.feed.has_pending(self.save_consumer)
    }

    /// The working file the store reads first and writes.
    pub(crate) const fn working(&self) -> &ProjectFile {
        &self.file
    }

    /// The base under the working file, if the project has been saved.
    pub(crate) const fn base(&self) -> Option<&BaseFile> {
        self.base.as_ref()
    }

    /// Writes every dirty correction tile, and every spatial-index bound the working file lacks,
    /// to the working file and records a save point there. Returns its
    /// offset.
    ///
    /// This makes the working file hold every edit, as of a save point of its own; it doesn't
    /// save the project (see [`crate::Project::save`]). Save's consumer of the change feed is
    /// drained into the tiles the next project save copies. If a write fails, the tiles not
    /// written yet stay dirty for the next save.
    ///
    /// # Errors
    ///
    /// [`CacheError::Streaming`] while a recovery writer streams the working file: its save
    /// points are cuts at its snapshots, and one made here could name later edits than a save
    /// point the writer is about to write, so the durable state could go backwards.
    /// [`CacheError::FailedWorkingFile`] while the working file is a recovery file whose
    /// writer failed, as the [`crate::Project`] that owns the store marks it: a save point there
    /// would vouch for records a failed sync may have lost.
    /// [`CacheError::WriteBack`] if a tile's or a bound's write fails; [`CacheError::Eval`] if
    /// listing what the edits made stale fails (never, for tiles of the store's mesh);
    /// [`CacheError::Store`] if the save point's write does; [`CacheError::Internal`] if the
    /// save point's derived maps disagree with it ([`Invariant::SavePointMismatch`]).
    pub fn save(&mut self) -> Result<u64, CacheError> {
        if !self.streams.is_empty() {
            return Err(CacheError::Streaming);
        }
        if self.working_failed {
            return Err(CacheError::FailedWorkingFile);
        }
        self.sync_pins()?;
        self.drain_saves();
        let mut dirty: Vec<TileKey> = self.unsaved.keys().copied().collect();
        dirty.sort_unstable();
        for key in dirty {
            if let Some(id) = self.corrections.find(&key)
                && self.corrections.is_dirty(id) == Some(true)
            {
                let slot = self.corrections.get(id).ok_or(Invariant::StaleHandle)?;
                let op = write_slot(&mut self.file, &mut self.unsaved, key, slot)?;
                self.corrections.set_dirty(id, false)?;
                self.known_ops.insert(key, op);
            }
        }
        self.write_unsaved_bounds()?;
        self.write_unsaved_display()?;
        self.file
            .save()
            .map_err(|e| CacheError::from_save_point(e, CacheError::Store))
    }
}

/// Whether `key` names a tile record that is an edit, not a display record (see
/// [`crate::display`]): only those are saved, carried and checked as edits.
fn is_correction(key: &TileKey) -> bool {
    !is_display(key)
}
