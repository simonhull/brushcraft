//! Display records read ahead on the workers: of a stroke, and of the tiles about to be drawn.
//!
//! A dab's bump ([`TileStore::bump_display_error`]) keeps, for every display tile it reaches
//! first, the tile's record as it was before the stroke. A record the store doesn't hold in
//! memory is in its files, and reading and decompressing it on the owner thread, in the middle
//! of a brush step, is what made the first dab on new ground slow. So when a dab first reaches a
//! display tile, the records of the tiles around it, which the next dabs are likely to reach,
//! are read on the workers; a bump finds them here.
//!
//! A renderer's requests are the other reader ([`TileStore::read_display_ahead`]): the display
//! tiles it is about to draw are read on the workers when it hears it wants them, so that
//! [`TileStore::display_for_draw`] finds them here rather than reading each on the owner thread
//! (about 140 µs a record on the benchmark project). Those reads are only for the tiles it
//! wants now: each call names them all, a tile it no longer names isn't read (a worker skips it)
//! and its record, read, goes; a tile staged goes too ([`TileStore::forget_drawn`]). They share
//! the cache's cap with the stroke's, three quarters of it at most
//! ([`TileStore::draw_ahead_cap`]), so a stroke's reads always have room. At the least budget
//! the cache holds 16 records (`CacheCaps::MIN_READ_AHEAD`), so 12 are read ahead for drawing
//! at once, about a frame's worth of staging; at the display caches' largest share, 256 MiB,
//! it holds 372, and 279 for drawing.
//!
//! Each record held remembers whom it was read for: a draw's requests adopt, and drop, only
//! records read for drawing, so a stroke's read-around records survive the tiles being
//! requested and drawn (a draw still uses them, as any record in memory).
//!
//! The cache drops its oldest record when full, whoever it was read for, so records still wanted
//! for drawing do take the room of a stroke's older ones, down to the quarter that stays the
//! stroke's: what is about to be drawn is sure to be used, what the next dabs reach only likely,
//! and a record gone is read again if a dab reaches it. A record no longer wanted takes no room:
//! a request's reads are kept only once the next request has said what it wants, however long
//! the owner thread took to come back to them.
//!
//! A record read ahead is used only while it is still the one the store would read: the same
//! open file, at the same place. Files are append-only while open, so a record superseded since
//! (by a flush, say) is at another place, and a file swapped since is another file; either way it
//! is read again, as it would have been.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, PoisonError, mpsc};

use geom::{Channel, TileKey};

use super::TileStore;
use crate::bounds::Kept;
use crate::display::{DisplayRecord, StoredDisplay};
use crate::file::{FileError, FileReader, Loc, display_record};
use crate::jobs::TileCache;

/// A record read ahead: where it was read from, and what it held (unknown, if damaged).
#[derive(Debug, Clone)]
pub(super) struct ReadRecord {
    /// The file it was read from.
    reader: FileReader,
    /// Where in it.
    loc: Loc,
    /// The record.
    record: StoredDisplay,
    /// Whom it was read for.
    purpose: Purpose,
}

/// Where a display record is, for a bump (see [`TileStore::bump_record`]).
#[derive(Debug)]
pub(super) enum Held {
    /// In memory: the record.
    Record(DisplayRecord),
    /// In a file, to read: the file, and where in it.
    In(FileReader, Loc),
    /// Nowhere: unknown, or no record at all.
    None,
}

/// Whom a record is read ahead for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Purpose {
    /// A stroke's next dabs ([`TileStore::read_around`]).
    Stroke,
    /// The tiles about to be drawn ([`TileStore::read_display_ahead`]): kept only while they
    /// are still wanted.
    Draw,
}

/// A record a worker read, sent back to the owner thread.
///
/// `None` if the read failed (the read on the owner thread reports it, if it comes to that), or,
/// read for drawing, if the tile was no longer wanted, and not read.
type Read = (TileKey, Option<ReadRecord>);

/// The tiles whose records are read ahead for drawing: shared with the workers, which skip a
/// tile no longer in it.
type Drawing = Arc<Mutex<BTreeSet<TileKey>>>;

/// The display records read ahead on the workers, and those still being read.
#[derive(Debug)]
pub(super) struct ReadAhead {
    /// Where the workers send what they read.
    sender: mpsc::Sender<Read>,
    /// What they read, taken on the owner thread.
    receiver: mpsc::Receiver<Read>,
    /// The tiles being read, and whom for.
    asked: BTreeMap<TileKey, Purpose>,
    /// The tiles a read-ahead looked at lately (read, or found needing none): looked at again
    /// only once it is emptied, at [`ReadAhead::LOOKED`] of them.
    looked: crate::hash::TileSet,
    /// The tiles about to be drawn whose records are read ahead (being read, or read and held),
    /// as the last [`TileStore::read_display_ahead`] named them, less those drawn since.
    drawing: Drawing,
    /// The records read, the oldest dropped first: a few strokes' worth of the tiles around
    /// them (a record of a level-8 tile is about 35 KB), and the tiles about to be drawn.
    pub(super) read: TileCache<ReadRecord>,
}

impl ReadAhead {
    /// The most records one worker task reads for drawing: a few, so the reads spread over the
    /// workers (one is about 140 µs on the benchmark project), at a task's spawn each.
    ///
    /// Chosen, not tuned: a request of 32 records spreads over 8 tasks, and the timing test
    /// `render::upload::tests::owner_time_staging_display_tiles` found them all read within the
    /// 100 ms it waits. The owner's cost is the spawns, a few microseconds each.
    const DRAW_TASK: usize = 4;
    /// The most tiles it remembers looking at.
    const LOOKED: usize = 4096;

    /// Nothing read ahead yet, keeping at most `cap` records (`jobs::CacheCaps::read_ahead`).
    pub(super) fn new(cap: usize) -> Self {
        let (sender, receiver) = mpsc::channel();
        Self {
            sender,
            receiver,
            asked: BTreeMap::new(),
            looked: crate::hash::TileSet::default(),
            drawing: Arc::default(),
            read: TileCache::new(cap),
        }
    }

    /// Runs `f` on the tiles read ahead for drawing. A worker that panicked holding the lock
    /// left the set whole (it only reads it), so a poisoned lock is taken as it is.
    fn drawing<R>(&self, f: impl FnOnce(&mut BTreeSet<TileKey>) -> R) -> R {
        f(&mut self.drawing.lock().unwrap_or_else(PoisonError::into_inner))
    }
}

/// Reads each of `reads` on a worker and sends it to `sender` (stopping once the store is gone
/// and nobody wants them). With `drawing`, a tile no longer in it is skipped, and sent unread.
fn read_records(
    reads: Vec<(TileKey, FileReader, Loc)>,
    sender: &mpsc::Sender<Read>,
    drawing: Option<&Drawing>,
) {
    let purpose = if drawing.is_some() {
        Purpose::Draw
    } else {
        Purpose::Stroke
    };
    for (key, reader, loc) in reads {
        let wanted = drawing.is_none_or(|drawing| {
            drawing
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .contains(&key)
        });
        let record = if wanted {
            match reader.read_record(loc) {
                Ok(raw) => Some(display_record(&raw, &key).unwrap_or(StoredDisplay::Unknown)),
                Err(FileError::Damaged { .. }) => Some(StoredDisplay::Unknown),
                Err(_) => None,
            }
        } else {
            None
        };
        let read = record.map(|record| ReadRecord {
            reader,
            loc,
            record,
            purpose,
        });
        if sender.send((key, read)).is_err() {
            return;
        }
    }
}

impl TileStore {
    /// Where the store's files hold `key`'s latest record: the working file's, else the
    /// base's, with a reader of that file.
    pub(super) fn record_place(&self, key: &TileKey) -> Option<(FileReader, Loc)> {
        if let Some(loc) = self.file.locate(key) {
            return Some((self.file.reader(), loc));
        }
        let base = self.base.as_ref()?;
        Some((base.reader(), base.locate(key)?))
    }

    /// `key`'s display record as read ahead, if it was and is still the record the store's
    /// files hold; `None` if it must be read.
    pub(super) fn read_ahead_record(&self, key: &TileKey) -> Option<&StoredDisplay> {
        self.read_ahead_entry(key).map(|found| &found.record)
    }

    /// [`TileStore::read_ahead_record`], with whom it was read for.
    fn read_ahead_entry(&self, key: &TileKey) -> Option<&ReadRecord> {
        let found = self.read_ahead.read.tiles.get(key)?;
        let (reader, loc) = self.record_place(key)?;
        (found.loc == loc && found.reader.same_file(&reader)).then_some(found)
    }

    /// Drops `key`'s record read ahead if it was read for drawing (freed with the cache's own);
    /// a stroke's stays.
    fn drop_drawn_record(&mut self, key: &TileKey) {
        let for_draw = self
            .read_ahead
            .read
            .tiles
            .get(key)
            .is_some_and(|found| found.purpose == Purpose::Draw);
        if for_draw {
            drop(self.read_ahead.read.remove(key));
        }
    }

    /// Where `key`'s display record is for a bump, which keeps it: in memory (the book's, or
    /// read ahead and still current, taken out so not copied), in a file, or nowhere (unknown,
    /// or built from another finest level).
    pub(super) fn bump_record(&mut self, key: &TileKey) -> Held {
        match self.display.kept(key) {
            Some(Kept::Known(record)) => return Held::Record(record.clone()),
            Some(Kept::Unknown) => return Held::None,
            Some(Kept::Written) | None => {}
        }
        if self.read_ahead_record(key).is_some()
            && let Some(found) = self.read_ahead.read.remove(key)
        {
            return found
                .record
                .known()
                .filter(|record| Some(record.finest) == self.display_finest)
                .map_or(Held::None, Held::Record);
        }
        self.record_place(key)
            .map_or(Held::None, |(reader, loc)| Held::In(reader, loc))
    }

    /// Takes in the records the workers have read ahead. One read for drawing a tile no longer
    /// wanted is dropped.
    pub(super) fn take_read_ahead(&mut self) {
        let reads = self.receive_reads();
        self.keep_reads(reads);
    }

    /// The records the workers have sent, not yet kept: their tiles stay among those asked until
    /// [`TileStore::keep_reads`], so a request made meanwhile takes them as being read. A tile
    /// answered without a record is no longer being read.
    fn receive_reads(&mut self) -> Vec<(TileKey, ReadRecord)> {
        let mut reads = Vec::new();
        while let Ok((key, read)) = self.read_ahead.receiver.try_recv() {
            match read {
                Some(read) => reads.push((key, read)),
                None => drop(self.read_ahead.asked.remove(&key)),
            }
        }
        reads
    }

    /// Keeps `reads`, from [`TileStore::receive_reads`], but for those read for drawing a tile
    /// no longer wanted, which take no room from the records held.
    fn keep_reads(&mut self, reads: Vec<(TileKey, ReadRecord)>) {
        for (key, read) in reads {
            self.read_ahead.asked.remove(&key);
            if read.purpose == Purpose::Draw
                && !self.read_ahead.drawing(|drawing| drawing.contains(&key))
            {
                continue;
            }
            // Replaced or dropped records are freed with the cache's own.
            drop(self.read_ahead.read.insert(key, read));
        }
    }

    /// The most tiles read ahead for drawing at once (being read, or held): three quarters of
    /// the read-ahead cache, the rest kept for a stroke's reads.
    pub(super) fn draw_ahead_cap(&self) -> usize {
        let cap = self.read_ahead.read.cap();
        cap - cap / 4
    }

    /// Reads ahead, on the workers, the display records of `keys`, the display tiles a renderer
    /// is about to draw, most wanted first, so that [`TileStore::display_for_draw`] finds them
    /// in memory rather than reading them on this thread. Returns how many reads it queued.
    ///
    /// Each call replaces the last: a tile it doesn't name is no longer read (a worker skips
    /// it), and its record, if read, goes. A record is used when the tile is drawn
    /// ([`TileStore::display_for_draw`]), and goes once the caller says it is staged
    /// ([`TileStore::forget_drawn`]). Tiles drawn without a file read are skipped (see
    /// [`TileStore::display_needs_read`]), a stroke's records read around included, as are keys
    /// that aren't the store's display tiles. The records read ahead for drawing share the
    /// read-ahead cache's cap, a part of the display caches' share of the budget
    /// ([`StoreConfig::display_cache_bytes`](super::StoreConfig::display_cache_bytes)): at most
    /// three quarters of it (12 records at the least budget), the first of `keys` taking it,
    /// the rest left to be read when drawn.
    pub fn read_display_ahead(&mut self, keys: &[TileKey]) -> usize {
        // Kept below, once this request has said what it wants: one it no longer wants would
        // push a record out of a full cache, and then be dropped itself.
        let received = self.receive_reads();
        let cap = self.draw_ahead_cap();
        let faces = self.mesh.face_count();
        let finest = self.display_finest;
        let mut wanted = BTreeSet::new();
        let mut reads = Vec::new();
        for key in keys {
            if wanted.len() >= cap {
                break;
            }
            let ours = key.channel() == Channel::DISPLAY
                && key.face().index() < faces
                && finest.is_some_and(|finest| key.level() < finest);
            if !ours || wanted.contains(key) {
                continue;
            }
            if self.in_memory(key) {
                continue;
            }
            // Held or being read: adopted if for drawing; a stroke's is used as it is.
            let held = self.read_ahead_entry(key).map(|found| found.purpose);
            match held.or_else(|| self.read_ahead.asked.get(key).copied()) {
                Some(Purpose::Draw) => {
                    wanted.insert(*key);
                    continue;
                }
                Some(Purpose::Stroke) => continue,
                None => {}
            }
            if let Some((reader, loc)) = self.record_place(key) {
                self.read_ahead.asked.insert(*key, Purpose::Draw);
                wanted.insert(*key);
                reads.push((*key, reader, loc));
            }
        }
        let gone: Vec<TileKey> = self.read_ahead.drawing(|drawing| {
            let old = std::mem::replace(drawing, wanted);
            old.into_iter()
                .filter(|key| !drawing.contains(key))
                .collect()
        });
        for key in &gone {
            self.drop_drawn_record(key);
        }
        self.keep_reads(received);
        let queued = reads.len();
        let mut reads = reads.into_iter().peekable();
        while reads.peek().is_some() {
            let task: Vec<(TileKey, FileReader, Loc)> =
                reads.by_ref().take(ReadAhead::DRAW_TASK).collect();
            let sender = self.read_ahead.sender.clone();
            let drawing = Arc::clone(&self.read_ahead.drawing);
            self.workers
                .spawn(move || read_records(task, &sender, Some(&drawing)));
        }
        queued
    }

    /// How many reads ahead the workers haven't answered yet.
    #[cfg(test)]
    pub(super) fn read_ahead_pending(&self) -> usize {
        self.read_ahead.asked.len()
    }

    /// Waits until the workers have answered every read asked of them, and leaves the answers
    /// waiting to be taken in, as an owner thread held up between two calls finds them.
    ///
    /// # Panics
    ///
    /// If the workers leave a read unanswered for ten seconds.
    #[cfg(test)]
    pub(super) fn await_read_ahead(&self) {
        let wait = std::time::Duration::from_secs(10);
        let answers: Vec<Read> = (0..self.read_ahead.asked.len())
            .map(|_| self.read_ahead.receiver.recv_timeout(wait).unwrap())
            .collect();
        for answer in answers {
            self.read_ahead.sender.send(answer).unwrap();
        }
    }

    /// Forgets the records read ahead for drawing `keys`, which the caller has staged: they
    /// are on their way to the GPU, and drawing them again reads them again. A tile drawn but
    /// not staged (no room, no slot this frame) keeps its record for the next frame. A stroke's
    /// records read around stay.
    pub fn forget_drawn(&mut self, keys: &[TileKey]) {
        let drawn: Vec<TileKey> = self.read_ahead.drawing(|drawing| {
            keys.iter()
                .filter(|key| drawing.remove(key))
                .copied()
                .collect()
        });
        for key in &drawn {
            self.drop_drawn_record(key);
        }
    }

    /// Reads ahead, on the workers, the display records of the tiles around each of `reached`
    /// (display tiles a dab reached for the first time this stroke): the tiles of the same
    /// level next to it, on its face and across its borders
    /// ([`crate::display::tiles_around`]), which the next
    /// dabs are likely to reach. Tiles whose record the store holds in memory (known or
    /// unknown), already read ahead or being read, or without a record in the files, are
    /// skipped.
    pub(super) fn read_around(&mut self, reached: &BTreeSet<TileKey>) {
        let mut around = BTreeSet::new();
        for key in reached {
            around.extend(
                self.deps
                    .reach_memo()
                    .around(&self.mesh, key)
                    .iter()
                    .copied(),
            );
        }
        if self.read_ahead.looked.len() > ReadAhead::LOOKED {
            self.read_ahead.looked.clear();
        }
        let mut reads = Vec::new();
        for key in around.difference(reached) {
            if !self.read_ahead.looked.insert(*key)
                || self.read_ahead.asked.contains_key(key)
                || matches!(self.display.kept(key), Some(Kept::Known(_) | Kept::Unknown))
                || self.read_ahead_record(key).is_some()
            {
                continue;
            }
            if let Some((reader, loc)) = self.record_place(key) {
                self.read_ahead.asked.insert(*key, Purpose::Stroke);
                reads.push((*key, reader, loc));
            }
        }
        if reads.is_empty() {
            return;
        }
        // One task for them all: a task per record would cost a worker's wake-up each, on the
        // owner thread, in the middle of a brush step.
        let sender = self.read_ahead.sender.clone();
        self.workers
            .spawn(move || read_records(reads, &sender, None));
    }
}
