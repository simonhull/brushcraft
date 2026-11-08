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
//!    goes to the writer thread. At the same moment [`TileStore::flush_bounds`] appends, with one
//!    write, the spatial index's bounds that edits changed since the last flush (65 bytes a tile;
//!    the bounds an index computes with no edit pending stay out, see [`crate::bounds`]), so the
//!    save point names the bounds of the state the snapshot copies. The records of the strokes
//!    ended by then go in with the same write's neighbour ([`TileStore::flush_strokes`]), so the
//!    save point that holds a stroke's last edits holds its record. The open stroke's dab journal
//!    is cut at the same moment ([`TileStore::cut_journal`]): a [`crate::DabBatch`] of the dabs
//!    placed and not yet written or landed, with the count landed as of the snapshot, so the count
//!    describes exactly the tiles the snapshot copies (see `crate::tiles::journal`). This is all a
//!    stroke end costs the drawing thread, and it grows with the edits since the last flush: a slot
//!    copy per tile, and a copy of the dabs placed since (84 bytes each).
//! 2. **Encoding** (writer): each copy is encoded, compressed and framed as a tile record, and the
//!    records come back, with the journal's batches framed after them. A copy that doesn't encode
//!    is left out, and the bounds its edit could have changed are appended as unknown, at the cut,
//!    whatever its slot holds now. If its live slot is broken too, the tile is lost in the store,
//!    as a save loses one; a valid slot means the failure was transient, and it stays dirty, for
//!    the next flush to try again. A batch that doesn't encode abandons the flush instead (see
//!    "Failure").
//! 3. **Syncing tiles**: the owner appends the records to the working file
//!    ([`TileStore::append_streamed`], page-cache writes), then the batches
//!    ([`TileStore::append_journal`]), and asks the writer to `fdatasync` the file. Appending a
//!    record clears its slot's dirty flag only if the slot is unchanged since the snapshot; a slot
//!    edited meanwhile stays dirty, and the next flush takes it. The journal notes its batches
//!    written only once they are appended, so a flush abandoned before then leaves them for the
//!    next, and none is ever written twice. Both are appended before the sync is asked for, so the
//!    sync covers them and the save point holds them.
//! 4. **Syncing the save point**: once that sync has completed, the owner appends a save point
//!    naming every tile (and bound) in the file, and asks for a second sync. When it completes, the
//!    save point is durable: a crash from then on loses nothing up to the snapshot.
//!
//! The owner advances the phases in [`RecoveryWriter::poll`], which the app calls from its frame
//! loop (and every trigger calls first); it never blocks there.
//!
//! # Memory
//!
//! A flush's snapshot is the one copy it makes outside the tile pool's budget: a slot copy for
//! every tile dirtied since the previous flush, 26.9 KB each, held until the writer thread has
//! encoded them. The app bounds it by calling [`RecoveryWriter::tick`] regularly, every frame: a
//! tick starts a flush once [`FLUSH_INTERVAL`] has passed since the last, so each snapshot holds
//! about a second of edits however long the stroke runs. A caller that never ticks gets one large
//! snapshot at [`RecoveryWriter::end_stroke`], of every tile the stroke dirtied: for a stroke over
//! a whole level, gigabytes.
//!
//! The snapshot is taken whole, at the cut, because the save point names the state at that one
//! cut. A later option would bound the memory without relying on the tick and without changing
//! the cut rules: copy-on-write cuts, fixed at the trigger, where a slot dirty at the cut is
//! copied only if an edit is about to change it before the writer has taken it.
//!
//! # Ordering
//!
//! A save point is written only after a sync that began after every record it names was
//! written. Records can be appended between the request for the tile sync and its completion:
//! a dirty tile written back to make room, say. So when the tile sync completes, the owner
//! checks where the last record the save point will name ends
//! ([`TileStore::working_named_end`]); if that is past what the sync covered, it asks for
//! another sync before writing the save point. Records past the flush's cut don't count: the
//! save point doesn't name them, and they wait for the next one. So steady write-back of later
//! edits during a long stroke can't postpone a flush's save point, or the flushes queued behind
//! it.
//!
//! # Triggers
//!
//! A flush starts at [`RecoveryWriter::end_stroke`], and at [`RecoveryWriter::tick`] once a
//! second has passed since the last flush started and edits are waiting. At most one flush is in
//! flight: a trigger while one is queues the next, which starts as soon as the current one is
//! durable and takes every edit made in the meantime.
//!
//! # A stroke's first landing waits for its cut
//!
//! A restore of a journalled stroke needs its **pen-down surface**, every tile as the stroke
//! found them, in a save point (see [`crate::journal`]). A flush queued behind one in flight takes
//! its snapshot only when it starts, so strokes back to back could leave no save point at the
//! second's pen-down. So as a stroke's op begins, the owner asks
//! [`RecoveryWriter::cut_before_op`] which flush's cut will hold it:
//!
//! - **No flush in flight:** if anything waits to be flushed, a flush starts now and its cut does;
//!   otherwise the working file's latest save point does (or, before its first, the state the file
//!   starts from).
//! - **A flush in flight, no edit since its cut, and no valid tile left out of it:** that one's.
//! - **Otherwise:** the next flush's, queued behind the one in flight. The stroke begins all the
//!   same (its dabs are placed, journalled and shown ahead), but **lands nothing until that cut is
//!   taken** ([`crate::Project::may_land`]): at most the flush in flight's syncs, and only when
//!   strokes come back to back.
//!
//! One cut is in flight at a time: nothing is copied early, and the pen-down path costs nothing
//! beyond a flush that starts. A flush whose encode left a valid tile out (a copy that failed for a
//! reason that may pass) names that tile's older state, so it can't hold a pen-down: the next one
//! does. And a flush holding one that leaves a valid tile out is abandoned
//! ([`RecoveryError::Unpaired`]), as one carrying a batch is; a pen-down flush abandoned leaves the
//! stroke's pen-down unknown, and its journal unusable for a restore (the tiles restore alone).
//!
//! This holds while strokes don't overlap: a stroke begins only once the last one's op has ended
//! (every dab landed or dropped), so no edit of the last stroke follows the next one's cut.
//!
//! # Failure
//!
//! An error in any phase is returned by the call that met it, and the flush is abandoned, along
//! with any flush queued behind it: the edits it held stay in the store's flush mark (or, if
//! already appended, in the file waiting for a save point), and the next trigger tries again. A
//! stroke whose pen-down either was to hold learns it is unknown, and may land.
//!
//! A dab batch that doesn't encode ([`RecoveryError::Journal`]) is such an error, unlike a tile:
//! its dabs stay in the journal for the next flush, with nothing of the flush appended. Leaving
//! the batch out and writing the save point would name tiles with dabs landed past the count the
//! file's last batch states, and a restore would land those dabs a second time. For the same
//! reason a tile left out for a failure that may pass, in a flush that carries a batch, abandons
//! the flush too ([`RecoveryError::Unpaired`]): the save point would name the tile's older state
//! beside a count that counts its dabs landed. So does one in a flush holding a stroke's pen-down
//! surface, which would name a surface the stroke never found. A broken slot is lost either way,
//! and handled as in any flush first.
//!
//! So a valid tile that keeps failing to encode while a batch is due abandons every flush, with
//! no save point, until it encodes, and its key is reported late each time
//! ([`crate::ProjectError::Unprotected`]). That is accepted: a valid slot's copy fails only as
//! compression does, which passes, and a slot found broken is lost, as in any flush, and stops
//! abandoning them.
//!
//! A failed sync is different, and final for the file. After one, the kernel may have dropped
//! the unsynced pages, so the records appended since the last durable save point may not be on
//! disk although the store counts them as written, and a later sync that succeeds proves
//! nothing about them. So the writer stops writing to the file: it is **failed**, it never
//! writes another save point there, and every trigger returns [`RecoveryError::NeedsNewFile`].
//! The file stays at its last durable save point. The owner ([`crate::Project`]) rolls over to
//! a new recovery file, copying every edit into it from the store (and the open stroke's
//! pen-down surface and journal, so a stroke still landing survives it), and starts a new writer.
//!
//! A writer whose thread has stopped (it panicked) is failed too: no flush can be finished,
//! and the edits a flush in flight snapshotted would never reach the file. The call that finds
//! the thread gone returns [`RecoveryError::Stopped`], and every trigger after it
//! [`RecoveryError::NeedsNewFile`], so the owner rolls over the same way. Rolling over, rather
//! than starting a new writer on the same file, is what keeps every edit: a new writer
//! subscribes with an empty flush mark, so tiles already dirty would never be flushed, whereas
//! a roll-over copies every edit from the store.
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
use crate::file::{FileError, FileSyncer, FramedTile, frame_batch, frame_displacement};
use crate::journal::{DabBatch, PenDown};
use crate::stamp::Stamp;
use crate::tiles::{CacheError, CutId, PenDownPoint, Snapshot, TileStore};

/// The longest a [`RecoveryWriter::tick`] lets edits wait before it starts a flush.
pub const FLUSH_INTERVAL: Duration = Duration::from_secs(1);

/// Why a [`RecoveryWriter`] call failed. The flush it was part of is abandoned and its edits are
/// kept for the next one (see the module notes).
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum RecoveryError {
    /// The tile store failed: copying a slot, or appending a record or a save point to the
    /// working file.
    #[error("the tile store failed during a recovery flush")]
    Store(#[from] CacheError),
    /// Syncing the working file failed. What was appended since the last durable save point may
    /// not be on disk.
    #[error("syncing the recovery file failed")]
    Sync(#[source] FileError),
    /// The writer thread couldn't be started.
    #[error("the recovery writer's thread couldn't be started")]
    Spawn(#[source] io::Error),
    /// The writer thread has stopped (it panicked): no flush can be finished. The writer is
    /// failed from then on, as after a failed sync (see the module notes), and every later
    /// trigger returns [`RecoveryError::NeedsNewFile`].
    #[error("the recovery writer's thread has stopped")]
    Stopped,
    /// A sync of the working file failed earlier, so the writer no longer writes to it (see
    /// the module notes): the edits need a new recovery file and a new writer.
    #[error("the recovery file failed to sync: the edits need a new one")]
    NeedsNewFile,
    /// A dab batch of the open stroke's journal couldn't be encoded. The flush is abandoned
    /// with nothing of it appended, and its dabs wait in the journal for the next (see the module
    /// notes).
    #[error("a dab batch of the stroke's journal couldn't be encoded")]
    Journal(#[source] FileError),
    /// A tile's copy didn't encode, for a failure that may pass, in a flush that carries a dab
    /// batch or holds a stroke's pen-down surface. The flush is abandoned with none of its tiles
    /// or batches appended: its save point would name the tile's older state beside a landed
    /// count that counts its dabs, or as the surface the stroke found. The tile stays dirty and
    /// the dabs in the journal, for the next flush; the pen-down is unknown (see the module
    /// notes).
    #[error("a tile of a flush with dab batches or a pen-down surface didn't encode")]
    Unpaired,
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
        /// The working file's length when the sync was asked for: the sync covers every record
        /// before it.
        end: u64,
    },
    /// The writer thread is syncing the save point at byte `offset`.
    SyncingSavePoint {
        /// Where the save point starts.
        offset: u64,
    },
    /// A sync failed, or the writer thread stopped: no flush will be made into this file again
    /// (see the module notes).
    Failed,
}

/// The flush in flight's cut.
#[derive(Debug, Clone, Copy)]
struct Flight {
    /// The cut's name: its journals' cut, and the flush a stroke's pen-down may wait on.
    id: CutId,
    /// The store's write the cut is at.
    stamp: Stamp,
    /// Whether its encode left a valid tile out: its save point names that tile's older state, so
    /// it can't hold a pen-down surface.
    left_valid_out: bool,
}

/// What the owner asks of the writer thread.
#[derive(Debug)]
enum Request {
    /// Encode, compress and frame these copies as tile records, and these batches of the
    /// journal as dab-batch records.
    Encode(Vec<Snapshot>, Vec<DabBatch>),
    /// Sync this file. Each request carries the store's current working file, so a writer
    /// outlives a swap of the file.
    Sync(FileSyncer),
}

/// What the writer thread answers, one reply per request, in order.
#[derive(Debug)]
enum Reply {
    /// The framed records, in snapshot order, the tiles whose copies didn't encode, left out,
    /// and the framed batches, in order, or why one didn't encode.
    Encoded(
        Vec<FramedTile>,
        Vec<TileKey>,
        Result<Vec<Vec<u8>>, FileError>,
    ),
    /// The sync completed, or why it failed.
    Synced(Result<(), FileError>),
}

/// Counts `n` down unless it is zero; whether it did.
#[cfg(any(test, feature = "test-hooks"))]
fn count_down(n: &std::sync::atomic::AtomicUsize) -> bool {
    n.try_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
        .is_ok()
}

/// Test hooks shared with the writer thread.
#[cfg(any(test, feature = "test-hooks"))]
#[derive(Debug, Default)]
struct Hooks {
    /// Syncs still to run before the failing ones start, counting down.
    passing_syncs: std::sync::atomic::AtomicUsize,
    /// Syncs still to fail once the passing ones have run, counting down: each fails with an
    /// I/O error instead of syncing.
    failing_syncs: std::sync::atomic::AtomicUsize,
    /// Encodes still to fail, counting down: each fails as a transient compression error
    /// would, whatever the copy holds.
    failing_encodes: std::sync::atomic::AtomicUsize,
    /// Encodes of dab batches still to fail, counting down, as [`Hooks::failing_encodes`] for
    /// tiles.
    failing_batch_encodes: std::sync::atomic::AtomicUsize,
    /// Bytes of snapshot copies handed to the writer thread and not yet dropped by it.
    in_flight: std::sync::atomic::AtomicUsize,
    /// The most [`Hooks::in_flight`] has been.
    peak_in_flight: std::sync::atomic::AtomicUsize,
    /// Holds the writer thread back while set: it takes no request (or, see [`Pause`], no sync)
    /// until it is cleared, so a test can act while a flush is certainly in flight, its next
    /// reply not yet sent.
    paused: std::sync::Mutex<Pause>,
    /// Signalled when [`Hooks::paused`] is cleared.
    unpaused: std::sync::Condvar,
}

/// Which requests the writer thread is held back from, in tests.
#[cfg(any(test, feature = "test-hooks"))]
#[derive(Debug, Default, Clone, Copy)]
struct Pause {
    /// Every request.
    all: bool,
    /// Syncs only: a flush encodes, then waits in its tile sync.
    syncs: bool,
    /// Whether `syncs` lasts only until a stroke waits for its cut
    /// ([`RecoveryWriter::wait_for_cut`]), which clears it.
    until_waited: bool,
}

#[cfg(any(test, feature = "test-hooks"))]
impl Hooks {
    /// Whether this sync is to fail, counting it down.
    fn sync_fails(&self) -> bool {
        !count_down(&self.passing_syncs) && count_down(&self.failing_syncs)
    }

    /// Whether this encode is to fail, counting it down.
    fn encode_fails(&self) -> bool {
        count_down(&self.failing_encodes)
    }

    /// Whether this batch's encode is to fail, counting it down.
    fn batch_encode_fails(&self) -> bool {
        count_down(&self.failing_batch_encodes)
    }

    /// The bytes `snapshots` hold: their slots' copies.
    const fn bytes_of(snapshots: &[Snapshot]) -> usize {
        snapshots.len() * size_of::<crate::slot::CorrectionSlot>()
    }

    /// `snapshots` are handed to the writer thread: counts their bytes in flight, and the peak.
    fn sent(&self, snapshots: &[Snapshot]) {
        let bytes = Self::bytes_of(snapshots);
        let now = self.in_flight.fetch_add(bytes, Ordering::SeqCst) + bytes;
        self.peak_in_flight.fetch_max(now, Ordering::SeqCst);
    }

    /// The writer thread dropped `snapshots`: their bytes are no longer in flight.
    fn dropped(&self, snapshots: &[Snapshot]) {
        self.in_flight
            .fetch_sub(Self::bytes_of(snapshots), Ordering::SeqCst);
    }

    /// Holds the writer thread back from what `change` sets, or lets it go. A poisoned lock is
    /// taken over as it is: it guards two flags.
    fn pause(&self, change: impl FnOnce(&mut Pause)) {
        change(
            &mut self
                .paused
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
        self.unpaused.notify_all();
    }

    /// Blocks the writer thread for as long as it is held back from a request, a sync if `sync`.
    fn wait_while_paused(&self, sync: bool) {
        let paused = self
            .paused
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        drop(
            self.unpaused
                .wait_while(paused, |paused| paused.all || (sync && paused.syncs))
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
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
    /// The flush in flight's cut; `None` while idle.
    flight: Option<Flight>,
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
    /// Tiles left out of a flush, late or lost, not yet reported (see
    /// [`RecoveryWriter::take_skipped`]).
    skipped: Vec<TileKey>,
    /// Those of them gone from memory too ([`RecoveryWriter::take_lost`]).
    lost: Vec<TileKey>,
    /// Test hooks shared with the writer thread.
    #[cfg(any(test, feature = "test-hooks"))]
    hooks: Arc<Hooks>,
    /// Test record: the tiles each flush's snapshot copied, in the order taken.
    #[cfg(test)]
    copied: Vec<Vec<TileKey>>,
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
        #[cfg(any(test, feature = "test-hooks"))]
        let hooks = Arc::new(Hooks::default());
        let worker = Worker {
            requests: inbox,
            replies: outbox,
            stop: Arc::clone(&stop),
            #[cfg(any(test, feature = "test-hooks"))]
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
            flight: None,
            queued: None,
            unconfirmed: false,
            last_flush: None,
            durable: None,
            flushes: 0,
            generation: store.generation(),
            skipped: Vec::new(),
            lost: Vec::new(),
            #[cfg(any(test, feature = "test-hooks"))]
            hooks,
            #[cfg(test)]
            copied: Vec::new(),
        })
    }

    /// A stroke ended at `now`: flushes every edit so far, or queues the flush if one is in
    /// flight. Advances the flush in flight first (see [`RecoveryWriter::poll`]).
    ///
    /// The drawing thread's cost is the snapshot: one slot copy per tile edited since the last
    /// flush, held outside the tile pool until encoded. Encoding, compression and syncing happen
    /// on the writer thread. A stroke with no [`RecoveryWriter::tick`] during it snapshots every
    /// tile it dirtied here, at once (see the module notes on memory).
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
    /// The app must call it regularly, every frame. Besides keeping a save point at least once a
    /// second, it bounds a flush's snapshot, the copies made outside the tile pool, to the tiles
    /// dirtied since the previous flush: about a second of edits. A caller that never ticks gets
    /// one large snapshot at [`RecoveryWriter::end_stroke`], of every tile the stroke dirtied
    /// (see the module notes on memory).
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

    /// A stroke's op is about to begin, at `now`, before its first edit: says which flush's cut
    /// will hold every tile as the op finds them, its pen-down surface, starting it or queueing it
    /// if need be (see the module notes, "A stroke's first landing waits for its cut"). It
    /// doesn't advance the flush in flight, and copies nothing a flush wouldn't.
    ///
    /// [`PenDownPoint::Next`] says the stroke must land nothing until the flush queued behind
    /// the one in flight has taken its cut ([`TileStore::may_land`]).
    ///
    /// # Errors
    ///
    /// [`RecoveryError::NeedsNewFile`] if the writer has failed; any [`RecoveryError`] of
    /// starting a flush, which is abandoned (see the module notes). No save point holds the
    /// pen-down surface then.
    pub(crate) fn cut_before_op(
        &mut self,
        store: &mut TileStore,
        now: Instant,
    ) -> Result<PenDownPoint, RecoveryError> {
        if self.phase == Phase::Failed {
            return Err(RecoveryError::NeedsNewFile);
        }
        if self.phase == Phase::Idle {
            if self.has_work(store) {
                self.start(store, now)?;
            }
            return Ok(self.flight.map_or_else(
                || PenDownPoint::Known(latest_save_point(store)),
                |flight| PenDownPoint::Flush(flight.id),
            ));
        }
        if let Some(flight) = self.flight
            && !flight.left_valid_out
            && !store.written_since(flight.stamp)
        {
            return Ok(PenDownPoint::Flush(flight.id));
        }
        self.queued.get_or_insert(now);
        Ok(PenDownPoint::Next)
    }

    /// Holds the writer thread back before its syncs, as the tests' `pause_syncs` does,
    /// until a stroke waits for its cut ([`RecoveryWriter::wait_for_cut`]), which lets it go.
    #[cfg(any(test, feature = "test-hooks"))]
    pub(crate) fn pause_syncs_until_waited(&self) {
        self.hooks.pause(|pause| {
            pause.syncs = true;
            pause.until_waited = true;
        });
    }

    /// Blocks until the open stroke may land ([`TileStore::may_land`]): its pen-down flush,
    /// queued behind the one in flight, has taken its cut, or the flushes were abandoned (the
    /// pen-down is then unknown). For a landing that can't wait for frames: a close's or a save's.
    /// Returns at once if the stroke may land already.
    ///
    /// # Errors
    ///
    /// As [`RecoveryWriter::poll`]; the stroke may land then too, its pen-down unknown.
    pub(crate) fn wait_for_cut(&mut self, store: &mut TileStore) -> Result<(), RecoveryError> {
        // A test holding the syncs until a stroke waits: this one does.
        #[cfg(any(test, feature = "test-hooks"))]
        self.hooks.pause(|pause| {
            if std::mem::take(&mut pause.until_waited) {
                pause.syncs = false;
            }
        });
        while !store.may_land() {
            if matches!(self.phase, Phase::Idle | Phase::Failed) {
                // Nothing will take the cut: only a bug leaves a pen-down waiting with no flush
                // in flight, and the stroke must not wait on it for ever.
                store.pen_downs_abandoned();
                return Ok(());
            }
            self.wait_step(store)?;
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
                Err(TryRecvError::Disconnected) => {
                    return self.settle(store, Err(RecoveryError::Stopped));
                }
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

    /// Unsubscribes from the store's change feed and stops the writer thread at once, leaving
    /// whatever isn't flushed: for a recovery file about to be removed, which nothing need reach.
    pub(crate) fn abandon(self, store: &mut TileStore) {
        // No flush in flight will end: its cut, and the pen-down surfaces waiting on it, are given
        // up with the file.
        store.drop_journal_cut();
        store.pen_downs_abandoned();
        store.unsubscribe_stream(self.consumer);
    }

    /// Whether no flush is in flight or queued: true once failed, too, as no flush will be.
    #[must_use]
    pub const fn is_idle(&self) -> bool {
        matches!(self.phase, Phase::Idle | Phase::Failed) && self.queued.is_none()
    }

    /// Whether a sync has failed, or the writer thread has stopped, so the writer no longer
    /// writes to its file (see the module notes).
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

    /// The tiles left out of flushes since the last call: their latest edits are not in the save
    /// point the flush made.
    ///
    /// A broken slot (a dirty slot with no op, or one that doesn't hold a valid tile) is dropped
    /// from the writer's flush mark when found, so it never stalls the flushes of the rest, and
    /// lost in the store as a save loses a tile: its slot is dropped (or, if pinned, filled
    /// again from the tile's last record), what was evaluated from it too, and the store reads
    /// the tile's last record from then on; an edit of it later is tried again. A valid slot
    /// whose copy failed to encode (a transient failure) stays dirty, and the next flush tries
    /// it again.
    pub fn take_skipped(&mut self) -> Vec<TileKey> {
        std::mem::take(&mut self.skipped)
    }

    /// Takes the tiles left out so far whose latest edits are gone, from memory too: broken slots,
    /// as [`RecoveryWriter::take_skipped`] says. Those it lists besides are only late.
    pub fn take_lost(&mut self) -> Vec<TileKey> {
        std::mem::take(&mut self.lost)
    }

    /// Whether a trigger now would have anything to flush: edits not yet in the working file,
    /// tiles in it since its last save point (written back to make room), a dab batch due, or a
    /// failed flush to confirm.
    fn has_work(&self, store: &TileStore) -> bool {
        self.unconfirmed
            || store.has_unflushed(self.consumer)
            || store.has_unsaved_records()
            || store.has_journal_work()
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

    /// Starts a flush at `at`: snapshots the dirty slots and cuts the journal, and hands both to
    /// the writer thread. With nothing to encode it goes straight to syncing, if the file holds
    /// tiles no save point names yet or the last flush failed, and does nothing otherwise.
    ///
    /// # Errors
    ///
    /// [`RecoveryError::Store`] if the snapshot fails; [`RecoveryError::Stopped`] if the writer
    /// thread is gone.
    fn start(&mut self, store: &mut TileStore, at: Instant) -> Result<(), RecoveryError> {
        self.queued = None;
        let owed = std::mem::take(&mut self.unconfirmed);
        let stamp = store.open_cut();
        self.generation = store.generation();
        let before = self.skipped.len();
        let snapshot = store
            .snapshot_dirty(self.consumer, &mut self.skipped)
            .and_then(|snapshots| {
                // The journal as of the snapshot, so its landed count is the snapshot's; the
                // bounds of the snapshot's state, at the same cut; and the records of the strokes
                // it ends.
                let (id, batches) = store.cut_journal()?;
                store.flush_bounds(self.skipped.get(before..).unwrap_or_default())?;
                store.flush_strokes()?;
                Ok((snapshots, id, batches))
            });
        // A dirty slot with no op is a broken one: lost, not late.
        self.lost
            .extend(self.skipped.get(before..).unwrap_or_default());
        let started = match snapshot {
            Ok((snapshots, id, batches)) => {
                #[cfg(test)]
                self.copied.push(snapshots.iter().map(|s| s.key).collect());
                let flight = Flight {
                    id,
                    stamp,
                    left_valid_out: false,
                };
                if !snapshots.is_empty() || !batches.is_empty() {
                    #[cfg(any(test, feature = "test-hooks"))]
                    self.hooks.sent(&snapshots);
                    self.flight = Some(flight);
                    self.send(Request::Encode(snapshots, batches))
                        .map(|()| self.phase = Phase::Encoding)
                } else if owed || store.has_unsaved_records() {
                    self.flight = Some(flight);
                    self.request_tile_sync(store)
                } else {
                    // Nothing to flush: the interval keeps running from the last real flush, no
                    // save point follows the cut, and the latest holds what the cut does.
                    store.drop_cut();
                    store.pen_down_settled(id, latest_save_point(store));
                    return Ok(());
                }
            }
            Err(e) => Err(e.into()),
        };
        self.last_flush = Some(at);
        self.settle(store, started)
    }

    /// Blocks for the writer thread's next reply and advances the flush with it.
    ///
    /// # Errors
    ///
    /// As [`RecoveryWriter::poll`].
    fn wait_step(&mut self, store: &mut TileStore) -> Result<(), RecoveryError> {
        match self.replies.recv() {
            Ok(reply) => self.advance(store, reply),
            Err(_) => self.settle(store, Err(RecoveryError::Stopped)),
        }
    }

    /// Advances the flush in flight with `reply`, abandoning it on error.
    ///
    /// # Errors
    ///
    /// As [`RecoveryWriter::step`].
    fn advance(&mut self, store: &mut TileStore, reply: Reply) -> Result<(), RecoveryError> {
        let stepped = self.step(store, reply);
        self.settle(store, stepped)
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
            (Phase::Encoding, Reply::Encoded(tiles, broken, batches)) => {
                let flight = self.flight.ok_or(RecoveryError::OutOfStep)?;
                // A batch left out would leave the save point naming tiles past the landed
                // count of the file's last batch: the flush is abandoned before anything of it is
                // appended, and the journal keeps the dabs for the next.
                let batches = batches.map_err(RecoveryError::Journal)?;
                // Every tile left out is unprotected by this save point, which names unknown
                // bounds for whatever its at-cut edit could change. Only a broken live slot is
                // lost in memory; a valid one stays dirty, for the next flush to retry.
                let lost = store.left_out_at_reply(&broken);
                for key in &lost {
                    store.drop_from_stream(self.consumer, key);
                }
                self.skipped.extend(broken.iter().copied());
                self.lost.extend(lost.iter().copied());
                store.forget_bounds_at_cut(&broken, &lost)?;
                // A valid tile left out keeps its older state in the save point, which the
                // batch's landed count would count past, or a stroke would take for the surface
                // it found: nothing of the flush is appended, and the tile and the dabs wait for
                // the next. A lost one is gone either way.
                let left_valid_out = broken.len() > lost.len();
                // Whether a stroke's pen-down waits on it is the store's to say: a journal begun
                // on it, or one queued for the next flush, which its cut took up.
                if left_valid_out && (!batches.is_empty() || store.pen_down_on(flight.id)) {
                    return Err(RecoveryError::Unpaired);
                }
                // Then it can hold no pen-down a stroke begins on later.
                if let Some(in_flight) = &mut self.flight {
                    in_flight.left_valid_out = left_valid_out;
                }
                let id = flight.id;
                store.append_streamed(self.consumer, &tiles)?;
                // Before the sync is asked for, so it covers them and the save point holds them;
                // the journal moves on only once they are in the file.
                store.append_journal(id, &batches)?;
                self.request_tile_sync(store)
            }
            (Phase::SyncingTiles { end }, Reply::Synced(synced)) => {
                synced.map_err(RecoveryError::Sync)?;
                if store.working_named_end() > end {
                    // A record the save point names was appended after the sync was asked for.
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
                // Settled before the next flush cuts the journals: a stroke whose pen-down this
                // flush holds has batches only in later flushes.
                if let Some(flight) = self.flight.take() {
                    store.pen_down_settled(flight.id, PenDown::At(offset));
                }
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
    /// abandoned, with the cut it opened in `store`'s working file (no save point follows it,
    /// and the next flush opens its own), and the next trigger flushes even with no new edits,
    /// to confirm what it held; a failed sync, or a writer thread found stopped, fails the
    /// writer for good (see the module notes).
    ///
    /// # Errors
    ///
    /// `outcome`'s.
    fn settle(
        &mut self,
        store: &mut TileStore,
        outcome: Result<(), RecoveryError>,
    ) -> Result<(), RecoveryError> {
        if let Err(e) = &outcome {
            store.drop_cut();
            // A cut whose batches were appended is noted written already: only one still in
            // flight goes, for the next flush to write again.
            store.drop_journal_cut();
            // The flush in flight and the one queued behind it hold no pen-down surface now.
            store.pen_downs_abandoned();
            self.flight = None;
            self.queued = None;
            self.unconfirmed = true;
            self.phase = match (self.phase, e) {
                (Phase::Failed, _) | (_, RecoveryError::Sync(_) | RecoveryError::Stopped) => {
                    Phase::Failed
                }
                _ => Phase::Idle,
            };
        }
        outcome
    }
}

/// Test-only controls.
#[cfg(test)]
impl RecoveryWriter {
    /// Holds the writer thread back before it takes its next request, or lets it go.
    ///
    /// While it is held, a flush stays where the owner left it, with no reply to advance it,
    /// however the threads are scheduled: a trigger then meets the flush in flight and queues
    /// the next one. Dropping the writer lets the thread go.
    pub(crate) fn pause(&self, paused: bool) {
        self.hooks.pause(|pause| pause.all = paused);
    }

    /// Holds the writer thread back before it takes its next sync, or lets it go: a flush
    /// encodes, then stays in its tile sync once the owner has appended its records, whatever
    /// the owner polls.
    pub(crate) fn pause_syncs(&self, paused: bool) {
        self.hooks.pause(|pause| pause.syncs = paused);
    }

    /// Lets the writer thread's next `pass` syncs run, then makes the `fail` after them fail
    /// with an I/O error.
    pub(crate) fn fail_syncs(&self, pass: usize, fail: usize) {
        self.hooks.passing_syncs.store(pass, Ordering::SeqCst);
        self.hooks.failing_syncs.store(fail, Ordering::SeqCst);
    }

    /// Makes the writer thread's next `fail` encodes fail, as a transient compression error
    /// would, whatever the copies hold.
    pub(crate) fn fail_encodes(&self, fail: usize) {
        self.hooks.failing_encodes.store(fail, Ordering::SeqCst);
    }

    /// Makes the writer thread's next `fail` encodes of dab batches fail, as
    /// [`RecoveryWriter::fail_encodes`] does for tiles.
    pub(crate) fn fail_batch_encodes(&self, fail: usize) {
        self.hooks
            .failing_batch_encodes
            .store(fail, Ordering::SeqCst);
    }

    /// The tiles each flush's snapshot so far copied, in the order taken.
    pub(crate) fn copied(&self) -> &[Vec<TileKey>] {
        &self.copied
    }

    /// Stops the writer thread as a panic would: its channel closes and the thread is joined, so
    /// the next request finds it gone.
    pub(crate) fn stop_thread(&mut self) {
        drop(self.requests.take());
        self.hooks.pause(|pause| *pause = Pause::default());
        if let Some(thread) = self.thread.take() {
            drop(thread.join());
        }
    }

    /// Whether a trigger has queued a flush behind the one in flight.
    pub(crate) const fn is_queued(&self) -> bool {
        self.queued.is_some()
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

    /// The most bytes of snapshot copies ever in flight to the writer thread at once: handed
    /// over and not yet dropped by it.
    pub(crate) fn peak_snapshot_bytes(&self) -> usize {
        self.hooks.peak_in_flight.load(Ordering::SeqCst)
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
        #[cfg(any(test, feature = "test-hooks"))]
        self.hooks.pause(|pause| *pause = Pause::default());
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
    #[cfg(any(test, feature = "test-hooks"))]
    hooks: Arc<Hooks>,
}

impl Worker {
    /// Answers requests until the owner closes the channel, stops the writer, or stops
    /// listening.
    fn run(self) {
        while let Ok(request) = self.requests.recv() {
            #[cfg(any(test, feature = "test-hooks"))]
            self.hooks
                .wait_while_paused(matches!(request, Request::Sync(_)));
            let reply = match request {
                Request::Encode(snapshots, batches) => {
                    let encoded = self.encode(&snapshots);
                    #[cfg(any(test, feature = "test-hooks"))]
                    self.hooks.dropped(&snapshots);
                    drop(snapshots);
                    let Some((framed, broken)) = encoded else {
                        return;
                    };
                    let Some(framed_batches) = self.frame_batches(&batches) else {
                        return;
                    };
                    Reply::Encoded(framed, broken, framed_batches)
                }
                Request::Sync(syncer) => Reply::Synced(sync(
                    &syncer,
                    #[cfg(any(test, feature = "test-hooks"))]
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
            let encoded =
                frame_displacement(snapshot.key, snapshot.op, snapshot.stamp, &snapshot.slot);
            #[cfg(any(test, feature = "test-hooks"))]
            let encoded = if self.hooks.encode_fails() {
                Err(io::Error::other("a failing encode, for the test").into())
            } else {
                encoded
            };
            match encoded {
                Ok(tile) => framed.push(tile),
                Err(_) => broken.push(snapshot.key),
            }
        }
        Some((framed, broken))
    }

    /// Frames every batch as a dab-batch record, in order, or says why one didn't encode;
    /// `None` if the owner stopped the writer part-way.
    fn frame_batches(&self, batches: &[DabBatch]) -> Option<Result<Vec<Vec<u8>>, FileError>> {
        let mut framed = Vec::with_capacity(batches.len());
        for batch in batches {
            if self.stop.load(Ordering::Relaxed) {
                return None;
            }
            let encoded = frame_batch(batch);
            #[cfg(any(test, feature = "test-hooks"))]
            let encoded = if self.hooks.batch_encode_fails() {
                Err(io::Error::other("a failing batch encode, for the test").into())
            } else {
                encoded
            };
            match encoded {
                Ok(record) => framed.push(record),
                Err(e) => return Some(Err(e)),
            }
        }
        Some(Ok(framed))
    }
}

/// The working file's latest save point, as a pen-down: where a stroke's surface is when every
/// edit is in the file and named by it, the file's base before its first.
fn latest_save_point(store: &TileStore) -> PenDown {
    store
        .working_last_save_point()
        .map_or(PenDown::Base, PenDown::At)
}

/// Syncs the file `syncer` names, or (in tests) fails if `hooks` says to.
///
/// # Errors
///
/// As [`FileSyncer::sync_data`].
fn sync(
    syncer: &FileSyncer,
    #[cfg(any(test, feature = "test-hooks"))] hooks: &Hooks,
) -> Result<(), FileError> {
    #[cfg(any(test, feature = "test-hooks"))]
    if hooks.sync_fails() {
        return Err(io::Error::other("a failing sync, for the test").into());
    }
    syncer.sync_data()
}
