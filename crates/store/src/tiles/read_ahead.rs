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
//! and its record, read, goes. They share the cache's cap with the stroke's, three quarters of
//! it at most ([`TileStore::draw_ahead_cap`]), so a stroke's reads always have room.
//!
//! A record read ahead is used only while it is still the one the store would read: the same
//! open file, at the same place. Files are append-only while open, so a record superseded since
//! (by a flush, say) is at another place, and a file swapped since is another file; either way it
//! is read again, as it would have been.

use std::collections::BTreeSet;
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

/// A record a worker read, sent back to the owner thread, and whom for.
///
/// `None` if the read failed (the read on the owner thread reports it, if it comes to that), or,
/// read for drawing, if the tile was no longer wanted, and not read.
type Read = (TileKey, Option<ReadRecord>, Purpose);

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
    /// The tiles being read.
    asked: BTreeSet<TileKey>,
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
    const DRAW_TASK: usize = 4;
    /// The most tiles it remembers looking at.
    const LOOKED: usize = 4096;

    /// Nothing read ahead yet, keeping at most `cap` records (`jobs::CacheCaps::read_ahead`).
    pub(super) fn new(cap: usize) -> Self {
        let (sender, receiver) = mpsc::channel();
        Self {
            sender,
            receiver,
            asked: BTreeSet::new(),
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
        });
        if sender.send((key, read, purpose)).is_err() {
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
        let found = self.read_ahead.read.tiles.get(key)?;
        let (reader, loc) = self.record_place(key)?;
        (found.loc == loc && found.reader.same_file(&reader)).then_some(&found.record)
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
        while let Ok((key, read, purpose)) = self.read_ahead.receiver.try_recv() {
            self.read_ahead.asked.remove(&key);
            let Some(read) = read else {
                continue;
            };
            if purpose == Purpose::Draw
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
    /// it), and its record, if read, goes. A record is used, and goes, when the tile is drawn.
    /// Tiles drawn without a file read are skipped (see [`TileStore::display_needs_read`]), as
    /// are keys that aren't the store's display tiles. The records read ahead for drawing
    /// share the read-ahead cache's cap, a part of the display caches' share of the budget
    /// ([`StoreConfig::display_cache_bytes`](super::StoreConfig::display_cache_bytes)): at most
    /// three quarters of it, the first of `keys` taking it, the rest left to be read when drawn.
    pub fn read_display_ahead(&mut self, keys: &[TileKey]) -> usize {
        self.take_read_ahead();
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
            if self.read_ahead.asked.contains(key) || self.read_ahead_record(key).is_some() {
                wanted.insert(*key);
                continue;
            }
            if !self.needs_read(key) {
                continue;
            }
            if let Some((reader, loc)) = self.record_place(key) {
                self.read_ahead.asked.insert(*key);
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
            // Freed with the cache's own.
            drop(self.read_ahead.read.remove(key));
        }
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

    /// Forgets the records read ahead for drawing `keys`, which were just drawn: they are on the
    /// GPU now, and drawing them again reads them again.
    pub(super) fn drawn_ahead(&mut self, keys: &[TileKey]) {
        let drawn: Vec<TileKey> = self.read_ahead.drawing(|drawing| {
            keys.iter()
                .filter(|key| drawing.remove(key))
                .copied()
                .collect()
        });
        for key in &drawn {
            if !self.read_ahead.asked.contains(key) {
                drop(self.read_ahead.read.remove(key));
            }
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
                || self.read_ahead.asked.contains(key)
                || matches!(self.display.kept(key), Some(Kept::Known(_) | Kept::Unknown))
                || self.read_ahead_record(key).is_some()
            {
                continue;
            }
            if let Some((reader, loc)) = self.record_place(key) {
                self.read_ahead.asked.insert(*key);
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
