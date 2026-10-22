//! The recovery writer: streams a tile store's edits into its working file, with a durable save
//! point after every stroke and at least once a second, without encoding or syncing on the
//! thread that draws.
//!
//! # A flush
//!
//! A flush moves through four phases, driven by the owner thread (the one that owns the
//! [`TileStore`]) and a writer thread of its own:
//!
//! 1. **Snapshot** (owner): [`TileStore::snapshot_dirty`] copies every correction slot edited since
//!    the writer's last flush and not yet in the working file, a 26.9 KB copy each, and the batch
//!    goes to the writer thread. This is all a stroke end costs the drawing thread.
//! 2. **Encoding** (writer): each copy is encoded, compressed and framed as a tile record, and the
//!    records come back.
//! 3. **Syncing tiles**: the owner appends the records to the working file
//!    ([`TileStore::append_streamed`], page-cache writes), then asks the writer to `fdatasync` the
//!    file. Appending a record clears its slot's dirty flag only if the slot is unchanged since the
//!    snapshot; a slot edited meanwhile stays dirty, and the next flush takes it.
//! 4. **Syncing the save point**: once that sync has completed, the owner appends a save point
//!    naming every tile in the file, and asks for a second sync. When it completes, the save point
//!    is durable: a crash from then on loses nothing up to the snapshot.
//!
//! The owner advances the phases in [`RecoveryWriter::poll`], which the app calls from its frame
//! loop (and every trigger calls first); it never blocks there.
//!
//! # Ordering
//!
//! A save point is written only after a sync that began after every record it names was
//! written. Records can be appended between the request for the tile sync and its completion:
//! a dirty tile written back to make room, say. So when the tile sync completes, the owner
//! checks that the file hasn't grown since it asked; if it has, it asks for another sync before
//! writing the save point.
//!
//! # Triggers
//!
//! A flush starts at [`RecoveryWriter::end_stroke`], and at [`RecoveryWriter::tick`] once a
//! second has passed since the last flush started and edits are waiting. At most one flush is in
//! flight: a trigger while one is queues the next, which starts as soon as the current one is
//! durable and takes every edit made in the meantime.
//!
//! # Failure
//!
//! An error in any phase is returned by the call that met it, and the flush is abandoned, along
//! with any flush queued behind it: the edits it held stay in the store's flush mark (or, if
//! already appended, in the file waiting for a save point), and the next trigger tries again.
//!
//! A failed sync is different, and final for the file. After one, the kernel may have dropped
//! the unsynced pages, so the records appended since the last durable save point may not be on
//! disk although the store counts them as written, and a later sync that succeeds proves
//! nothing about them. So the writer stops writing to the file: it is **failed**, it never
//! writes another save point there, and every trigger returns [`RecoveryError::NeedsNewFile`].
//! The file stays at its last durable save point. The owner ([`crate::Project`]) rolls over to
//! a new recovery file, copying every edit into it from the store, and starts a new writer.
//!
//! # Shutdown
//!
//! Dropping the writer stops its thread and joins it: the thread checks a stop flag between
//! tiles, so it returns after at most the tile or sync in hand. A flush in flight is abandoned;
//! the file reopens at its last save point, and what the flush had appended after it is
//! discarded.

use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use geom::TileKey;
use thiserror::Error;

use crate::feed::ConsumerId;
use crate::file::{FileSyncer, FramedTile, StoreError, frame_displacement};
use crate::tiles::{CacheError, Snapshot, TileStore};

/// The longest a [`RecoveryWriter::tick`] lets edits wait before it starts a flush.
pub const FLUSH_INTERVAL: Duration = Duration::from_secs(1);

/// Why a [`RecoveryWriter`] call failed. The flush it was part of is abandoned and its edits are
/// kept for the next one (see the module notes).
#[derive(Debug, Error)]
pub enum RecoveryError {
    /// The tile store failed: copying a slot, or appending a record or a save point to the
    /// working file.
    #[error("the tile store failed during a recovery flush")]
    Store(#[from] CacheError),
    /// Syncing the working file failed. What was appended since the last durable save point may
    /// not be on disk.
    #[error("syncing the recovery file failed")]
    Sync(#[source] StoreError),
    /// The writer thread couldn't be started.
    #[error("the recovery writer's thread couldn't be started")]
    Spawn(#[source] io::Error),
    /// The writer thread has stopped (it panicked): no flush can be finished.
    #[error("the recovery writer's thread has stopped")]
    Stopped,
    /// A sync of the working file failed earlier, so the writer no longer writes to it (see
    /// the module notes): the edits need a new recovery file and a new writer.
    #[error("the recovery file failed to sync: the edits need a new one")]
    NeedsNewFile,
    /// A reply arrived that the flush in flight wasn't waiting for: a bug in the writer.
    #[error("the recovery writer's thread replied out of step")]
    OutOfStep,
}

/// Where the flush in flight is (see the module notes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// No flush in flight.
    Idle,
    /// The writer thread is encoding the snapshot.
    Encoding,
    /// The writer thread is syncing the tiles appended so far, which end at byte `end`.
    SyncingTiles {
        /// The working file's length when the sync was asked for.
        end: u64,
    },
    /// The writer thread is syncing the save point at byte `offset`.
    SyncingSavePoint {
        /// Where the save point starts.
        offset: u64,
    },
    /// A sync failed: no flush will be made into this file again (see the module notes).
    Failed,
}

/// What the owner asks of the writer thread.
#[derive(Debug)]
enum Request {
    /// Encode, compress and frame these copies as tile records.
    Encode(Vec<Snapshot>),
    /// Sync this file. Each request carries the store's current working file, so a writer
    /// outlives a swap of the file.
    Sync(FileSyncer),
}

/// What the writer thread answers, one reply per request, in order.
#[derive(Debug)]
enum Reply {
    /// The framed records, in snapshot order, and the tiles whose copies didn't encode (broken
    /// slots), left out.
    Encoded(Vec<FramedTile>, Vec<TileKey>),
    /// The sync completed, or why it failed.
    Synced(Result<(), StoreError>),
}

/// Test hooks shared with the writer thread.
#[cfg(test)]
#[derive(Debug, Default)]
struct Hooks {
    /// Syncs still to run before the failing ones start, counting down.
    passing_syncs: std::sync::atomic::AtomicUsize,
    /// Syncs still to fail once the passing ones have run, counting down: each fails with an
    /// I/O error instead of syncing.
    failing_syncs: std::sync::atomic::AtomicUsize,
}

#[cfg(test)]
impl Hooks {
    /// Whether this sync is to fail, counting it down.
    fn sync_fails(&self) -> bool {
        /// Counts `n` down unless it is zero; whether it did.
        fn count_down(n: &std::sync::atomic::AtomicUsize) -> bool {
            n.try_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
                .is_ok()
        }
        !count_down(&self.passing_syncs) && count_down(&self.failing_syncs)
    }
}

/// Streams a [`TileStore`]'s edits into its working file in the background, a durable save point
/// per flush (see the module notes).
///
/// It belongs to the thread that owns the store, which passes the store to every call. Each
/// writer is a streaming consumer of the store's change feed, subscribed by
/// [`RecoveryWriter::new`] and removed by [`RecoveryWriter::close`].
#[derive(Debug)]
pub struct RecoveryWriter {
    /// The writer's place in the change feed, and its flush mark in the store.
    consumer: ConsumerId,
    /// Requests to the writer thread; `None` once the drop has closed it.
    requests: Option<Sender<Request>>,
    /// The writer thread's replies.
    replies: Receiver<Reply>,
    /// Set by the drop: the writer thread abandons the encoding in hand and returns.
    stop: Arc<AtomicBool>,
    /// The writer thread, joined by the drop.
    thread: Option<JoinHandle<()>>,
    /// Where the flush in flight is.
    phase: Phase,
    /// When a trigger that met a flush in flight came: the next flush is owed, and starts as
    /// soon as the current one is durable.
    queued: Option<Instant>,
    /// Whether the last flush failed: its edits may be in the file without a durable save point,
    /// so the next trigger flushes even if nothing new was edited.
    unconfirmed: bool,
    /// When the latest flush started (or failed to), for [`RecoveryWriter::tick`]'s interval. A
    /// trigger that finds nothing to flush leaves it.
    last_flush: Option<Instant>,
    /// The latest durable save point's offset in the working file.
    durable: Option<u64>,
    /// Flushes completed: save points made durable.
    flushes: u64,
    /// The store's working-file generation when the flush in flight started: a rebase since
    /// means the flush's records and save point would go to the wrong file.
    generation: u64,
    /// Tiles left out of a flush because their slots were broken, not yet reported (see
    /// [`RecoveryWriter::take_skipped`]).
    skipped: Vec<TileKey>,
    /// Test hooks shared with the writer thread.
    #[cfg(test)]
    hooks: Arc<Hooks>,
}

impl RecoveryWriter {
    /// A writer streaming `store`'s edits from now on, with its thread started.
    ///
    /// # Errors
    ///
    /// [`RecoveryError::Spawn`] if the thread can't be started; nothing is subscribed then.
    pub fn new(store: &mut TileStore) -> Result<Self, RecoveryError> {
        let (requests, inbox) = mpsc::channel();
        let (outbox, replies) = mpsc::channel();
        let stop = Arc::new(AtomicBool::new(false));
        #[cfg(test)]
        let hooks = Arc::new(Hooks::default());
        let worker = Worker {
            requests: inbox,
            replies: outbox,
            stop: Arc::clone(&stop),
            #[cfg(test)]
            hooks: Arc::clone(&hooks),
        };
        let thread = thread::Builder::new()
            .name("recovery-writer".into())
            .spawn(move || worker.run())
            .map_err(RecoveryError::Spawn)?;
        Ok(Self {
            consumer: store.subscribe_stream(),
            requests: Some(requests),
            replies,
            stop,
            thread: Some(thread),
            phase: Phase::Idle,
            queued: None,
            unconfirmed: false,
            last_flush: None,
            durable: None,
            flushes: 0,
            generation: store.generation(),
            skipped: Vec::new(),
            #[cfg(test)]
            hooks,
        })
    }

    /// A stroke ended at `now`: flushes every edit so far, or queues the flush if one is in
    /// flight. Advances the flush in flight first (see [`RecoveryWriter::poll`]).
    ///
    /// The drawing thread's cost is the snapshot: one slot copy per tile edited since the last
    /// flush. Encoding, compression and syncing happen on the writer thread.
    ///
    /// # Errors
    ///
    /// Any [`RecoveryError`] of advancing the flush in flight or of the snapshot. The flush is
    /// still started or queued when advancing fails.
    pub fn end_stroke(&mut self, store: &mut TileStore, now: Instant) -> Result<(), RecoveryError> {
        let polled = self.poll(store);
        let triggered = self.trigger(store, now);
        polled.and(triggered)
    }

    /// Time passed, to `now`, during a stroke or after it: starts a flush (or queues one, if one
    /// is in flight) when edits are waiting and [`FLUSH_INTERVAL`] has passed since the last
    /// flush started. Advances the flush in flight first.
    ///
    /// # Errors
    ///
    /// As [`RecoveryWriter::end_stroke`]; a tick whose advance fails starts nothing (the next
    /// one will).
    pub fn tick(&mut self, store: &mut TileStore, now: Instant) -> Result<(), RecoveryError> {
        self.poll(store)?;
        let due = self
            .last_flush
            .is_none_or(|last| now.saturating_duration_since(last) >= FLUSH_INTERVAL);
        if due && self.has_work(store) {
            self.trigger(store, now)?;
        }
        Ok(())
    }

    /// Advances the flush in flight as far as the writer thread's replies allow, without
    /// blocking: appends encoded records, writes the save point once the tiles are synced, and
    /// starts a queued flush once the save point is durable.
    ///
    /// # Errors
    ///
    /// Any [`RecoveryError`] of the step that failed; the flush is abandoned (see the module
    /// notes).
    pub fn poll(&mut self, store: &mut TileStore) -> Result<(), RecoveryError> {
        while !matches!(self.phase, Phase::Idle | Phase::Failed) {
            let reply = match self.replies.try_recv() {
                Ok(reply) => reply,
                Err(TryRecvError::Empty) => return Ok(()),
                Err(TryRecvError::Disconnected) => return self.settle(Err(RecoveryError::Stopped)),
            };
            self.advance(store, reply)?;
        }
        Ok(())
    }

    /// Flushes every edit made so far and waits until it is durable: the flush in flight and any
    /// queued after it, then one more for the edits since, all blocking the calling thread.
    /// `now` is when it started, for [`RecoveryWriter::tick`]'s interval.
    ///
    /// # Errors
    ///
    /// Any [`RecoveryError`] of a step on the way.
    pub fn finish(&mut self, store: &mut TileStore, now: Instant) -> Result<(), RecoveryError> {
        self.poll(store)?;
        self.trigger(store, now)?;
        while !matches!(self.phase, Phase::Idle | Phase::Failed) {
            self.wait_step(store)?;
        }
        Ok(())
    }

    /// Finishes every flush ([`RecoveryWriter::finish`]), unsubscribes from the store's change
    /// feed and stops the writer thread.
    ///
    /// # Errors
    ///
    /// As [`RecoveryWriter::finish`]; the writer is unsubscribed and stopped either way.
    pub fn close(mut self, store: &mut TileStore, now: Instant) -> Result<(), RecoveryError> {
        let finished = self.finish(store, now);
        store.unsubscribe_stream(self.consumer);
        finished
    }

    /// Whether no flush is in flight or queued: true once failed, too, as no flush will be.
    #[must_use]
    pub const fn is_idle(&self) -> bool {
        matches!(self.phase, Phase::Idle | Phase::Failed) && self.queued.is_none()
    }

    /// Whether a sync has failed, so the writer no longer writes to its file (see the module
    /// notes).
    #[must_use]
    pub const fn is_failed(&self) -> bool {
        matches!(self.phase, Phase::Failed)
    }

    /// The offset of the latest durable save point in the working file, for
    /// [`crate::ProjectFile::open_at`]; `None` before the first flush completes.
    #[must_use]
    pub const fn durable_save_point(&self) -> Option<u64> {
        self.durable
    }

    /// Flushes completed so far: save points made durable.
    #[must_use]
    pub const fn flushes(&self) -> u64 {
        self.flushes
    }

    /// The tiles left out of flushes since the last call because their correction slots were
    /// broken (a dirty slot with no op, or one that doesn't hold a valid tile): their latest
    /// edits are not in the recovery file. Each is dropped from the writer's flush mark when
    /// found, so one bad slot never stalls the flushes of the rest; an edit of it later is
    /// tried again.
    pub fn take_skipped(&mut self) -> Vec<TileKey> {
        std::mem::take(&mut self.skipped)
    }

    /// Whether a trigger now would have anything to flush: edits not yet in the working file,
    /// tiles in it since its last save point (written back to make room), or a failed flush to
    /// confirm.
    fn has_work(&self, store: &TileStore) -> bool {
        self.unconfirmed || store.has_unflushed(self.consumer) || store.has_unsaved_tiles()
    }

    /// A trigger at `now`: starts a flush, or queues one if a flush is in flight (keeping the
    /// earliest queued trigger's time).
    ///
    /// # Errors
    ///
    /// As [`RecoveryWriter::start`].
    fn trigger(&mut self, store: &mut TileStore, now: Instant) -> Result<(), RecoveryError> {
        if self.phase == Phase::Failed {
            return Err(RecoveryError::NeedsNewFile);
        }
        if self.phase == Phase::Idle {
            self.start(store, now)
        } else {
            self.queued.get_or_insert(now);
            Ok(())
        }
    }

    /// Starts a flush at `at`: snapshots the dirty slots and hands them to the writer thread.
    /// With nothing to encode it goes straight to syncing, if the file holds tiles no save point
    /// names yet or the last flush failed, and does nothing otherwise.
    ///
    /// # Errors
    ///
    /// [`RecoveryError::Store`] if the snapshot fails; [`RecoveryError::Stopped`] if the writer
    /// thread is gone.
    fn start(&mut self, store: &mut TileStore, at: Instant) -> Result<(), RecoveryError> {
        self.queued = None;
        let owed = std::mem::take(&mut self.unconfirmed);
        store.open_cut();
        self.generation = store.generation();
        let started = match store.snapshot_dirty(self.consumer, &mut self.skipped) {
            Ok(snapshots) if !snapshots.is_empty() => self
                .send(Request::Encode(snapshots))
                .map(|()| self.phase = Phase::Encoding),
            Ok(_) if owed || store.has_unsaved_tiles() => self.request_tile_sync(store),
            // Nothing to flush: the interval keeps running from the last real flush.
            Ok(_) => return Ok(()),
            Err(e) => Err(e.into()),
        };
        self.last_flush = Some(at);
        self.settle(started)
    }

    /// Blocks for the writer thread's next reply and advances the flush with it.
    ///
    /// # Errors
    ///
    /// As [`RecoveryWriter::poll`].
    fn wait_step(&mut self, store: &mut TileStore) -> Result<(), RecoveryError> {
        match self.replies.recv() {
            Ok(reply) => self.advance(store, reply),
            Err(_) => self.settle(Err(RecoveryError::Stopped)),
        }
    }

    /// Advances the flush in flight with `reply`, abandoning it on error.
    ///
    /// # Errors
    ///
    /// As [`RecoveryWriter::step`].
    fn advance(&mut self, store: &mut TileStore, reply: Reply) -> Result<(), RecoveryError> {
        let stepped = self.step(store, reply);
        self.settle(stepped)
    }

    /// One step of the flush in flight (see the module notes): `reply` answers the request of
    /// the current phase.
    ///
    /// # Errors
    ///
    /// [`RecoveryError::Sync`] as the writer thread reports it;
    /// [`RecoveryError::Store`] if an append or the save point's write fails;
    /// [`RecoveryError::Stopped`] if the next request can't be sent; [`RecoveryError::OutOfStep`]
    /// for a reply the phase didn't ask for, or if the store was rebased onto another working
    /// file since the flush started (the flush is abandoned: its records and save point belong
    /// to the old file).
    fn step(&mut self, store: &mut TileStore, reply: Reply) -> Result<(), RecoveryError> {
        if store.generation() != self.generation {
            return Err(RecoveryError::OutOfStep);
        }
        match (self.phase, reply) {
            (Phase::Encoding, Reply::Encoded(tiles, broken)) => {
                for key in broken {
                    store.drop_from_stream(self.consumer, &key);
                    self.skipped.push(key);
                }
                store.append_streamed(self.consumer, &tiles)?;
                self.request_tile_sync(store)
            }
            (Phase::SyncingTiles { end }, Reply::Synced(synced)) => {
                synced.map_err(RecoveryError::Sync)?;
                if store.working_len() != end {
                    // Appended to after the sync was asked for: those records aren't covered.
                    return self.request_tile_sync(store);
                }
                let offset = store.write_save_point()?;
                self.send(Request::Sync(store.working_syncer()))?;
                self.phase = Phase::SyncingSavePoint { offset };
                Ok(())
            }
            (Phase::SyncingSavePoint { offset }, Reply::Synced(synced)) => {
                synced.map_err(RecoveryError::Sync)?;
                self.durable = Some(offset);
                self.flushes += 1;
                self.phase = Phase::Idle;
                match self.queued {
                    Some(at) => self.start(store, at),
                    None => Ok(()),
                }
            }
            _ => Err(RecoveryError::OutOfStep),
        }
    }

    /// Asks the writer thread to sync the working file as it now ends.
    ///
    /// # Errors
    ///
    /// [`RecoveryError::Stopped`] if the writer thread is gone.
    fn request_tile_sync(&mut self, store: &TileStore) -> Result<(), RecoveryError> {
        self.send(Request::Sync(store.working_syncer()))?;
        self.phase = Phase::SyncingTiles {
            end: store.working_len(),
        };
        Ok(())
    }

    /// Sends `request` to the writer thread.
    ///
    /// # Errors
    ///
    /// [`RecoveryError::Stopped`] if the thread is gone.
    fn send(&self, request: Request) -> Result<(), RecoveryError> {
        self.requests
            .as_ref()
            .ok_or(RecoveryError::Stopped)?
            .send(request)
            .or(Err(RecoveryError::Stopped))
    }

    /// Passes `outcome` through. On error the flush in flight and any queued after it are
    /// abandoned, and the next trigger flushes even with no new edits, to confirm what it held;
    /// a failed sync fails the writer for good (see the module notes).
    ///
    /// # Errors
    ///
    /// `outcome`'s.
    fn settle(&mut self, outcome: Result<(), RecoveryError>) -> Result<(), RecoveryError> {
        if let Err(e) = &outcome {
            self.queued = None;
            self.unconfirmed = true;
            self.phase = match (self.phase, e) {
                (Phase::Failed, _) | (_, RecoveryError::Sync(_)) => Phase::Failed,
                _ => Phase::Idle,
            };
        }
        outcome
    }
}

/// Test-only controls.
#[cfg(test)]
impl RecoveryWriter {
    /// Lets the writer thread's next `pass` syncs run, then makes the `fail` after them fail
    /// with an I/O error.
    pub(crate) fn fail_syncs(&self, pass: usize, fail: usize) {
        self.hooks.passing_syncs.store(pass, Ordering::SeqCst);
        self.hooks.failing_syncs.store(fail, Ordering::SeqCst);
    }

    /// Blocks for one reply of the flush in flight and advances it, so a test can act between
    /// phases; does nothing when idle.
    ///
    /// # Errors
    ///
    /// As [`RecoveryWriter::poll`].
    pub(crate) fn step_blocking(&mut self, store: &mut TileStore) -> Result<(), RecoveryError> {
        if matches!(self.phase, Phase::Idle | Phase::Failed) {
            return Ok(());
        }
        self.wait_step(store)
    }

    /// The phase of the flush in flight, by name: `idle`, `encoding`, `syncing tiles`,
    /// `syncing save point` or `failed`.
    pub(crate) const fn phase(&self) -> &'static str {
        match self.phase {
            Phase::Idle => "idle",
            Phase::Encoding => "encoding",
            Phase::SyncingTiles { .. } => "syncing tiles",
            Phase::SyncingSavePoint { .. } => "syncing save point",
            Phase::Failed => "failed",
        }
    }

    /// A handle that upgrades only while the writer or its thread is alive: once the writer is
    /// dropped and its thread joined, it can't.
    pub(crate) fn liveness(&self) -> std::sync::Weak<AtomicBool> {
        Arc::downgrade(&self.stop)
    }
}

impl Drop for RecoveryWriter {
    /// Stops the writer thread and joins it: it abandons the encoding in hand at the next tile,
    /// or returns once the sync in hand completes.
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        drop(self.requests.take());
        if let Some(thread) = self.thread.take() {
            // A thread that panicked has nothing left to clean up; its panic was reported on
            // its own thread.
            drop(thread.join());
        }
    }
}

/// The writer thread's state: its end of the two channels and the stop flag.
struct Worker {
    /// Requests from the owner, in order.
    requests: Receiver<Request>,
    /// Replies to the owner, one per request.
    replies: Sender<Reply>,
    /// Set when the owner drops the writer.
    stop: Arc<AtomicBool>,
    /// Test hooks shared with the owner.
    #[cfg(test)]
    hooks: Arc<Hooks>,
}

impl Worker {
    /// Answers requests until the owner closes the channel, stops the writer, or stops
    /// listening.
    fn run(self) {
        while let Ok(request) = self.requests.recv() {
            let reply = match request {
                Request::Encode(snapshots) => match self.encode(&snapshots) {
                    Some((framed, broken)) => Reply::Encoded(framed, broken),
                    None => return,
                },
                Request::Sync(syncer) => Reply::Synced(sync(
                    &syncer,
                    #[cfg(test)]
                    &self.hooks,
                )),
            };
            if self.replies.send(reply).is_err() {
                return;
            }
        }
    }

    /// Frames every snapshot as a tile record, leaving out (and listing) any whose copy doesn't
    /// encode; `None` if the owner stopped the writer part-way.
    fn encode(&self, snapshots: &[Snapshot]) -> Option<(Vec<FramedTile>, Vec<TileKey>)> {
        let mut framed = Vec::with_capacity(snapshots.len());
        let mut broken = Vec::new();
        for snapshot in snapshots {
            if self.stop.load(Ordering::Relaxed) {
                return None;
            }
            match frame_displacement(snapshot.key, snapshot.op, &snapshot.slot) {
                Ok(tile) => framed.push(tile),
                Err(_) => broken.push(snapshot.key),
            }
        }
        Some((framed, broken))
    }
}

/// Syncs the file `syncer` names, or (in tests) fails if `hooks` says to.
///
/// # Errors
///
/// As [`FileSyncer::sync_data`].
fn sync(syncer: &FileSyncer, #[cfg(test)] hooks: &Hooks) -> Result<(), StoreError> {
    #[cfg(test)]
    if hooks.sync_fails() {
        return Err(io::Error::other("a failing sync, for the test").into());
    }
    syncer.sync_data()
}
