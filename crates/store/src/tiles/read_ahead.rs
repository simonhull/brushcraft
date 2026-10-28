//! Display records read ahead of a stroke, on the workers.
//!
//! A dab's bump ([`TileStore::bump_display_error`]) keeps, for every display tile it reaches
//! first, the tile's record as it was before the stroke. A record the store doesn't hold in
//! memory is in its files, and reading and decompressing it on the owner thread, in the middle
//! of a brush step, is what made the first dab on new ground slow. So when a dab first reaches a
//! display tile, the records of the tiles around it, which the next dabs are likely to reach,
//! are read on the workers; a bump finds them here.
//!
//! A record read ahead is used only while it is still the one the store would read: the same
//! open file, at the same place. Files are append-only while open, so a record superseded since
//! (by a flush, say) is at another place, and a file swapped since is another file; either way it
//! is read again, as it would have been.

use std::collections::BTreeSet;
use std::sync::mpsc;

use geom::TileKey;

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

/// A record a worker read, sent back to the owner thread (`None`: the read failed, and the
/// read on the owner thread reports it, if it comes to that).
type Read = (TileKey, Option<ReadRecord>);

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
    /// The records read, the oldest dropped first: a few strokes' worth of the tiles around
    /// them (a record of a level-8 tile is about 35 KB).
    pub(super) read: TileCache<ReadRecord>,
}

impl ReadAhead {
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
            read: TileCache::new(cap),
        }
    }
}

impl TileStore {
    /// Where the store's files hold `key`'s latest record: the working file's, else the
    /// base's, with a reader of that file.
    fn record_place(&self, key: &TileKey) -> Option<(FileReader, Loc)> {
        if let Some(loc) = self.file.locate(key) {
            return Some((self.file.reader(), loc));
        }
        let base = self.base.as_ref()?;
        Some((base.reader(), base.locate(key)?))
    }

    /// `key`'s display record as read ahead, if it was and is still the record the store's
    /// files hold; `None` if it must be read.
    pub(super) fn read_ahead_record(&self, key: &TileKey) -> Option<StoredDisplay> {
        let found = self.read_ahead.read.tiles.get(key)?;
        let (reader, loc) = self.record_place(key)?;
        (found.loc == loc && found.reader.same_file(&reader)).then(|| found.record.clone())
    }

    /// Whether the store holds `key`'s display record read ahead, still current (see
    /// [`TileStore::read_ahead_record`]), without copying it.
    fn holds_read_ahead(&self, key: &TileKey) -> bool {
        self.read_ahead.read.tiles.get(key).is_some_and(|found| {
            self.record_place(key)
                .is_some_and(|(reader, loc)| found.loc == loc && found.reader.same_file(&reader))
        })
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
        if self.holds_read_ahead(key)
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

    /// Takes in the records the workers have read ahead.
    pub(super) fn take_read_ahead(&mut self) {
        while let Ok((key, read)) = self.read_ahead.receiver.try_recv() {
            self.read_ahead.asked.remove(&key);
            if let Some(read) = read {
                // Replaced or dropped records are freed with the cache's own.
                drop(self.read_ahead.read.insert(key, read));
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
                || self.holds_read_ahead(key)
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
        self.workers.spawn(move || {
            for (key, reader, loc) in reads {
                let record = match reader.read_record(loc) {
                    Ok(raw) => Some(display_record(&raw, &key).unwrap_or(StoredDisplay::Unknown)),
                    Err(FileError::Damaged { .. }) => Some(StoredDisplay::Unknown),
                    Err(_) => None,
                };
                let read = record.map(|record| ReadRecord {
                    reader,
                    loc,
                    record,
                });
                // The store may be gone: then nobody wants them.
                if sender.send((key, read)).is_err() {
                    return;
                }
            }
        });
    }
}
