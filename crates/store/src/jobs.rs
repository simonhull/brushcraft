//! Budgeted work on the owner thread: jobs that run in slices until a deadline, keeping their
//! place, so the frame loop can give the store what time a frame has left.
//!
//! The kinds are few and known, so a job is an enum ([`Job`]), not a trait object: bringing the
//! display layer up to date after edits ([`Job::Refresh`]) and building it whole for a file
//! without one ([`Job::Rebuild`]). Both are the same work, display tiles built finest level first
//! ([`Upkeep`]); they differ in what they start from.
//!
//! The display tiles are built on the store's workers. A slice on the owner thread only collects
//! the tiles the workers finished (and records them), hands the workers more (each with its
//! inputs, owned), or evaluates a small batch of the finest tiles they read; so a slice stays
//! short, and a deadline is honoured to within one slice.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::{Arc, mpsc};
use std::time::Duration;

use geom::{DisplayError, DisplayTile, Level, TileKey, Vec3};

use crate::CacheError;

/// How long before a deadline a wait for the workers ends: time for the waiting thread to be
/// woken, so the wait doesn't overrun the deadline.
pub(crate) const WAKE_MARGIN: Duration = Duration::from_micros(500);

/// The most spare finest copies a job keeps to copy into again (about 100 KB each at level 9).
const SPARE_COPIES: usize = 32;

/// The most tiles a round of evaluation for a job holds (see [`Upkeep::round_size`]).
pub(crate) const ROUND_MAX: usize = 8;

/// About how long a round of evaluation for a job should take: well under a slice's
/// millisecond, the rest of the slice being collecting and handing out.
const ROUND_TIME: Duration = Duration::from_micros(250);

/// What evaluating a tile is assumed to take before a job has measured it.
const TILE_COST: Duration = Duration::from_micros(150);

/// How long a job waits for the corrections it prefetched before it looks again: they are
/// installed by the owner thread, so nothing wakes it when they are read.
pub(crate) const READ_WAIT: Duration = Duration::from_micros(100);

/// What a slice of a job left.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Progress {
    /// More to do: run it again.
    More,
    /// Nothing to do on the owner thread until the workers finish a tile: wait for them.
    Waiting,
    /// Nothing to do on the owner thread until the corrections it prefetched land: wait a
    /// moment.
    Reading,
    /// Finished.
    Done,
}

/// The longest a slice of a job is made to take: a frame's slack.
pub(crate) const SLICE_TARGET: Duration = Duration::from_millis(1);

/// What [`TileStore::run_jobs`](crate::TileStore::run_jobs) did before its deadline.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct JobsReport {
    /// Slices run.
    pub slices: usize,
    /// Display tiles built and recorded.
    pub built: usize,
    /// Finest tiles evaluated for them.
    pub evaluated: usize,
    /// Jobs that finished.
    pub finished: usize,
    /// Display tiles still to build (or being built), over every job left.
    pub left: usize,
    /// The longest slice run: the most the deadline can be overrun by. Waiting for the workers
    /// isn't a slice: it ends at the deadline.
    pub longest_slice: Duration,
    /// Slices longer than the millisecond a slice is made to fit in
    /// ([`JobsReport::SLICE_TARGET`]).
    pub long_slices: usize,
    /// Display tiles that wouldn't build (extreme geometry, say), left unknown and skipped
    /// until a change lands (see [`TileStore::failed_display`](crate::TileStore::failed_display)).
    pub failed: usize,
}

impl JobsReport {
    /// The longest a slice is made to take, which [`JobsReport::long_slices`] counts past.
    pub const SLICE_TARGET: Duration = SLICE_TARGET;
}

/// A job: work the store does in slices on its owner thread.
#[derive(Debug)]
pub(crate) enum Job {
    /// Brings up to date the display tiles edits made unknown, finest level first.
    Refresh(Upkeep),
    /// Builds every display tile the files don't hold (a file without a display layer, or one
    /// whose records were all dropped), finest level first.
    Rebuild(Upkeep),
}

impl Job {
    /// The display work the job does.
    pub(crate) const fn upkeep(&mut self) -> &mut Upkeep {
        match self {
            Self::Refresh(upkeep) | Self::Rebuild(upkeep) => upkeep,
        }
    }

    /// Display tiles still to build, or being built.
    pub(crate) fn left(&self) -> usize {
        match self {
            Self::Refresh(upkeep) | Self::Rebuild(upkeep) => {
                upkeep.pending.len() + upkeep.in_flight.len()
            }
        }
    }
}

/// What a worker made of one display tile, sent back to the owner thread.
#[derive(Debug)]
pub(crate) struct Finished {
    /// The tile.
    pub(crate) key: TileKey,
    /// The store's count of correction changes its inputs were taken at
    /// ([`TileStore::edits`](crate::TileStore::edits)): if it moved since, the result is
    /// dropped and the tile built again.
    pub(crate) as_of: u64,
    /// For a decode, the generation of the record it decoded
    /// (`TileStore::display_generation`): if the record was replaced since, the result is
    /// dropped.
    pub(crate) record: u64,
    /// Whether the worker decoded the tile (for the tiles of the level above), not built it.
    pub(crate) decode: bool,
    /// What came of it.
    pub(crate) outcome: Outcome,
}

/// What came of building one display tile on a worker.
#[derive(Debug)]
pub(crate) enum Outcome {
    /// The tile, built.
    Built(Box<DisplayTile>),
    /// The tile, decoded from its record.
    Decoded(Box<DisplayTile>),
    /// The tile's record (to decode) doesn't read: build it.
    Unreadable,
    /// It won't build from its inputs.
    Failed(DisplayError),
    /// A record couldn't be read.
    Error(CacheError),
}

/// Display tiles to build, finest level first, for a display layer under `finest`.
#[derive(Debug)]
pub(crate) struct Upkeep {
    /// The finest level, which the display tiles are built from.
    pub(crate) finest: Level,
    /// The tiles still to build, finest level first, then in key order: a tile is built after
    /// every tile one level finer that it reads.
    pending: BTreeSet<(Reverse<Level>, TileKey)>,
    /// The tiles the workers are building, with the count of correction changes their inputs
    /// were taken at.
    pub(crate) in_flight: BTreeMap<TileKey, u64>,
    /// The tiles the workers are decoding for the tiles of the level above.
    pub(crate) decoding: BTreeSet<TileKey>,
    /// Where the workers send what they finished.
    pub(crate) sender: mpsc::Sender<Finished>,
    /// What they finished, read on the owner thread.
    pub(crate) receiver: mpsc::Receiver<Finished>,
    /// What a wait received, for the next slice.
    pub(crate) received: Vec<Finished>,
    /// The finest tiles' values the job copied for the workers.
    pub(crate) copies: FinestCopies,
    /// Copies no worker holds any more, kept to copy into again: a fresh allocation of a
    /// tile's size costs its page faults, several times the copy.
    spare: Vec<FinestTile>,
    /// The finest tiles being evaluated a step at a time, with their normals' cone (see
    /// `TileStore::evaluate_for`), listed once for every step.
    pub(crate) cone: Option<Cone>,
    /// The tiles the steps of that cone have evaluated so far: one asked for again was
    /// evicted since.
    pub(crate) cone_seen: BTreeSet<TileKey>,
    /// What evaluating one tile has taken lately, wall time over the tiles of a round (a
    /// moving average): how many a slice's round holds.
    tile_cost: Duration,
}

/// Finest tiles a job evaluates a step at a time, with the cone their normals read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Cone {
    /// The finest tiles.
    pub(crate) keys: Vec<TileKey>,
    /// The positions their normals read, then the tiles those are predicted from, level by
    /// level: coarsest level first.
    pub(crate) levels: Vec<Vec<TileKey>>,
    /// The store's count of correction changes it was listed at: a change since lists it
    /// again.
    pub(crate) as_of: u64,
}

/// A finest tile's positions and unit normals, copied for the workers.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FinestTile {
    /// Positions, row-major.
    pub(crate) positions: Arc<[Vec3]>,
    /// Unit normals, row-major.
    pub(crate) normals: Arc<[Vec3]>,
}

/// Tiles' values a job or the store holds for the workers, by key, at most its cap.
///
/// The oldest go first, and, for values a correction change outdates, they hold only as of one
/// count of them ([`TileStore::edits`](crate::TileStore::edits); see [`TileCache::since`]).
///
/// Dropping one tile at a time when full, not emptying it, keeps each slice's share of the
/// freeing small; what a change empties is handed back whole, for the caller to free off the
/// owner thread.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TileCache<V> {
    /// The store's count of correction changes the values are as of.
    as_of: u64,
    /// The values, by tile.
    pub(crate) tiles: BTreeMap<TileKey, V>,
    /// The tiles in the order they came in, the oldest first (a tile put in again moves last).
    order: VecDeque<TileKey>,
    /// The most tiles it holds (at least one).
    cap: usize,
}

impl<V> TileCache<V> {
    /// An empty cache of at most `cap` tiles (at least one).
    pub(crate) fn new(cap: usize) -> Self {
        Self {
            as_of: 0,
            tiles: BTreeMap::new(),
            order: VecDeque::new(),
            cap: cap.max(1),
        }
    }

    /// The most tiles it holds.
    pub(crate) const fn cap(&self) -> usize {
        self.cap
    }

    /// Drops the oldest tiles until it holds no more than its cap; returns the first dropped.
    fn shrink(&mut self) -> Option<V> {
        let mut out = None;
        while self.tiles.len() > self.cap {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            let dropped = self.tiles.remove(&oldest);
            out = out.or(dropped);
        }
        out
    }

    /// Empties it, handing back what it held (to free off the owner thread).
    pub(crate) fn take(&mut self) -> BTreeMap<TileKey, V> {
        self.order.clear();
        std::mem::take(&mut self.tiles)
    }

    /// Empties it unless its values are as of the store's count of correction changes `as_of`;
    /// returns what it held then, for the caller to free where it costs the owner thread
    /// nothing.
    #[must_use = "the values dropped: free them on the workers"]
    pub(crate) fn since(&mut self, as_of: u64) -> Option<BTreeMap<TileKey, V>> {
        if self.as_of == as_of {
            return None;
        }
        self.as_of = as_of;
        self.order.clear();
        Some(std::mem::take(&mut self.tiles))
    }

    /// Takes `key`'s value out, if it holds one.
    pub(crate) fn remove(&mut self, key: &TileKey) -> Option<V> {
        let value = self.tiles.remove(key)?;
        self.order.retain(|k| k != key);
        Some(value)
    }

    /// Puts `value` in for `key`, dropping the oldest tile if it is full; returns the value it
    /// replaced or dropped, if any (the first, were there several), for the caller to reuse.
    pub(crate) fn insert(&mut self, key: TileKey, value: V) -> Option<V> {
        let replaced = self.tiles.insert(key, value);
        if replaced.is_some() {
            self.order.retain(|k| *k != key);
        }
        self.order.push_back(key);
        let dropped = self.shrink();
        replaced.or(dropped)
    }
}

/// The finest positions and normals a job copied out of the store's cache for its workers.
///
/// Copied once each, and shared by every display tile reading them; a tile evaluated for a job
/// is copied right away, so a display tile reading more finest tiles than the budget holds at
/// once (under a small budget) is still built, never waiting for all of them to be resident
/// together. It holds more than one display tile reads (its block, apron and the tiles across
/// its face borders, a few dozen at most: [`CacheCaps::MIN_COPIES`]), so a tile can always
/// gather all it reads; [`CacheCaps::copies`] of them.
pub(crate) type FinestCopies = TileCache<FinestTile>;

/// Display tiles the store's jobs built, or decoded from their records, for the coarser tiles
/// that read them.
///
/// Each is held with the generation of the record it is (`TileStore::display_generation`): a
/// tile whose record was replaced since is decoded again.
///
/// Held by the store, not a job: a refresh's tiles of one level read those it just built a
/// level finer, never decoding them again, and the tiles no refresh rebuilt are decoded once
/// for many refreshes, not once each: a correction change doesn't touch a record, only a build
/// replaces one. [`CacheCaps::decoded`] of them, emptied when a call to
/// [`TileStore::run_jobs`](crate::TileStore::run_jobs) finds nothing to do.
pub(crate) type DecodedTiles = TileCache<(u64, Arc<DisplayTile>)>;

/// How many entries each of the display layer's caches holds, from the bytes the store's budget
/// sets aside for them
/// ([`StoreConfig::display_cache_bytes`](crate::StoreConfig::display_cache_bytes)).
///
/// The share is split by what each cache is for: half to the decoded display tiles (a coarser
/// tile reads 16 or so, and a stroke's refresh reads them again and again), three tenths to a
/// job's finest copies, a tenth each to the records read ahead and to the readers' memo. Each
/// is counted at its largest entry, and never below the least that lets the jobs make progress.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CacheCaps {
    /// Decoded display tiles the store keeps ([`DecodedTiles`]).
    pub(crate) decoded: usize,
    /// Finest tiles a job keeps copied ([`FinestCopies`]).
    pub(crate) copies: usize,
    /// Display records read ahead (`tiles::read_ahead`).
    pub(crate) read_ahead: usize,
    /// Tiles each of the display readers' memos keeps answers for
    /// (`display::ReachMemo`).
    pub(crate) reach: usize,
}

impl CacheCaps {
    /// A finest tile's copy: 65² positions and normals.
    const COPY_BYTES: usize = 65 * 65 * 24;
    /// A decoded display tile at its largest: 65² vertices of positions, normals and roughness,
    /// and its encoding.
    const DECODED_BYTES: usize = 65 * 65 * 25 + 65 * 65 * 17;
    /// The fewest finest copies: more than any display tile reads.
    pub(crate) const MIN_COPIES: usize = 64;
    /// The fewest decoded tiles kept: what the tiles of one coarser level read at once.
    const MIN_DECODED: usize = 64;
    /// The fewest memo answers: a dab's written tiles, many times over.
    const MIN_REACH: usize = 256;
    /// The fewest records read ahead: a dab's surroundings.
    const MIN_READ_AHEAD: usize = 16;
    /// One answer of the readers' memo: a few hundred keys at most.
    const REACH_BYTES: usize = 4096;
    /// A display record at its largest (`display::max_body` of a 65² tile).
    const RECORD_BYTES: usize = 65 * 65 * 17 + 256;

    /// The caps that `bytes` hold (each at least its minimum, which a tiny budget may take
    /// past `bytes`).
    pub(crate) fn of(bytes: usize) -> Self {
        let part =
            |tenths: usize, each: usize, least: usize| (bytes / 10 * tenths / each).max(least);
        Self {
            decoded: part(5, Self::DECODED_BYTES, Self::MIN_DECODED),
            copies: part(3, Self::COPY_BYTES, Self::MIN_COPIES),
            read_ahead: part(1, Self::RECORD_BYTES, Self::MIN_READ_AHEAD),
            reach: part(1, Self::REACH_BYTES, Self::MIN_REACH),
        }
    }
}

impl Upkeep {
    /// Work building `keys` (display tiles of levels below `finest`), keeping at most `copies`
    /// finest copies ([`CacheCaps::copies`]).
    pub(crate) fn new(
        finest: Level,
        keys: impl IntoIterator<Item = TileKey>,
        copies: usize,
    ) -> Self {
        let (sender, receiver) = mpsc::channel();
        Self {
            finest,
            pending: keys
                .into_iter()
                .filter(|key| key.level() < finest)
                .map(|key| (Reverse(key.level()), key))
                .collect(),
            in_flight: BTreeMap::new(),
            decoding: BTreeSet::new(),
            sender,
            receiver,
            received: Vec::new(),
            copies: FinestCopies::new(copies),
            spare: Vec::new(),
            cone: None,
            cone_seen: BTreeSet::new(),
            tile_cost: TILE_COST,
        }
    }

    /// The next tile to build: the finest level's first.
    pub(crate) fn next(&mut self) -> Option<TileKey> {
        self.pending.pop_first().map(|(_, key)| key)
    }

    /// Puts `key` (back) among the tiles to build.
    pub(crate) fn push(&mut self, key: TileKey) {
        if key.level() < self.finest {
            self.pending.insert((Reverse(key.level()), key));
        }
    }

    /// Whether every tile is built: none to build, none being built or decoded.
    pub(crate) fn is_empty(&self) -> bool {
        self.pending.is_empty() && self.in_flight.is_empty() && self.decoding.is_empty()
    }

    /// Tiles the workers are building or decoding.
    pub(crate) fn busy(&self) -> usize {
        self.in_flight.len() + self.decoding.len()
    }

    /// Copies finest tile `key`'s `positions` and `normals` into the job's copies, into a spare
    /// copy's buffers where one is free.
    pub(crate) fn copy_finest(&mut self, key: TileKey, positions: &[Vec3], normals: &[Vec3]) {
        let reused = self.spare.pop().and_then(|mut spare| {
            let p = Arc::get_mut(&mut spare.positions)?;
            p.get_mut(..)
                .filter(|p| p.len() == positions.len())?
                .copy_from_slice(positions);
            let n = Arc::get_mut(&mut spare.normals)?;
            n.get_mut(..)
                .filter(|n| n.len() == normals.len())?
                .copy_from_slice(normals);
            Some(spare)
        });
        let tile = reused.unwrap_or_else(|| FinestTile {
            positions: Arc::from(positions),
            normals: Arc::from(normals),
        });
        if let Some(old) = self.copies.insert(key, tile) {
            self.recycle(old);
        }
    }

    /// Whether the job keeps as many spare copies as it will.
    pub(crate) const fn spare_full(&self) -> bool {
        self.spare.len() >= SPARE_COPIES
    }

    /// Keeps `tile`'s buffers to copy into again, if no worker holds them and there is room.
    pub(crate) fn recycle(&mut self, mut tile: FinestTile) {
        if self.spare.len() < SPARE_COPIES
            && Arc::get_mut(&mut tile.positions).is_some()
            && Arc::get_mut(&mut tile.normals).is_some()
        {
            self.spare.push(tile);
        }
    }

    /// How many tiles the next round of evaluation holds, with `free` workers not building
    /// display tiles: as many as [`ROUND_TIME`] fits at the cost measured lately, at least one,
    /// at most one a free worker and [`ROUND_MAX`].
    pub(crate) fn round_size(&self, free: usize) -> usize {
        let fits = ROUND_TIME.as_nanos() / self.tile_cost.as_nanos().max(1);
        usize::try_from(fits)
            .unwrap_or(usize::MAX)
            .min(free)
            .clamp(1, ROUND_MAX)
    }

    /// Notes that a round of `tiles` tiles took `wall`, for [`Upkeep::round_size`].
    pub(crate) fn took(&mut self, wall: Duration, tiles: usize) {
        let Some(each) = u32::try_from(tiles.max(1)).ok().map(|n| wall / n) else {
            return;
        };
        self.tile_cost = (self.tile_cost * 3 + each) / 4;
    }

    /// Waits until a worker finishes a tile or `until`, keeping what arrives for the next
    /// slice.
    pub(crate) fn wait_until(&mut self, until: std::time::Instant) {
        let left = until.saturating_duration_since(std::time::Instant::now());
        if left.is_zero() {
            return;
        }
        if let Ok(finished) = self.receiver.recv_timeout(left) {
            self.received.push(finished);
        }
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use geom::FaceId;

    use super::*;

    /// A tile is handed out after every tile of a finer level, so a tile pushed back with the
    /// finer tiles it needs comes out after them.
    #[test]
    fn finer_levels_come_first() {
        let key = |l: u8, tx: u16| {
            TileKey::display(FaceId::new(0), Level::new(l).unwrap(), tx, 0).unwrap()
        };
        let mut upkeep = Upkeep::new(
            Level::new(8).unwrap(),
            [key(3, 0), key(7, 1), key(5, 0), key(8, 0)],
            CacheCaps::MIN_COPIES,
        );
        assert_eq!(
            upkeep.next(),
            Some(key(7, 1)),
            "the finest first; level 8 is the finest, none"
        );
        upkeep.push(key(7, 0));
        assert_eq!(
            upkeep.next(),
            Some(key(7, 0)),
            "a pushed finer tile comes before coarser ones"
        );
        assert_eq!(upkeep.next(), Some(key(5, 0)), "then level 5");
        assert_eq!(upkeep.next(), Some(key(3, 0)), "then level 3");
        assert!(upkeep.is_empty(), "nothing left");
    }

    /// The copies hold as of one count of correction changes: a change since empties them (and
    /// hands them back); when full, the oldest goes first.
    #[test]
    fn copies_are_dropped_after_a_change_or_oldest_first_when_full() {
        // Level 9 has 8 × 8 tiles a face, so 4 faces hold a copy for each up to the cap.
        let finest = |n: u16| {
            TileKey::new(
                geom::LayerId::BASE,
                geom::Channel::DISPLACEMENT,
                FaceId::new(u32::from(n / 64)),
                Level::new(9).unwrap(),
                n % 8,
                (n / 8) % 8,
            )
            .unwrap()
        };
        let one = FinestTile {
            positions: Arc::from(vec![Vec3::ZERO]),
            normals: Arc::from(vec![Vec3::ZERO]),
        };
        let mut copies = FinestCopies::new(CacheCaps::MIN_COPIES);
        assert!(copies.since(1).is_some(), "a first count empties it");
        copies.insert(finest(0), one.clone());
        assert!(copies.since(1).is_none(), "kept with no change since");
        assert_eq!(copies.tiles.len(), 1, "kept");
        let dropped = copies.since(2).unwrap();
        assert_eq!(dropped.len(), 1, "a change hands back what it held");
        assert!(copies.tiles.is_empty(), "a change empties them");
        let cap = u16::try_from(CacheCaps::MIN_COPIES).unwrap();
        for n in 0..cap {
            copies.insert(finest(n), one.clone());
        }
        assert_eq!(copies.tiles.len(), CacheCaps::MIN_COPIES, "full");
        copies.insert(finest(0), one.clone());
        assert_eq!(
            copies.tiles.len(),
            CacheCaps::MIN_COPIES,
            "put in again: no growth"
        );
        copies.insert(finest(cap), one);
        assert_eq!(copies.tiles.len(), CacheCaps::MIN_COPIES, "the cap holds");
        assert!(
            !copies.tiles.contains_key(&finest(1)),
            "the oldest went first"
        );
        assert!(
            copies.tiles.contains_key(&finest(0)),
            "a tile put in again counts as new"
        );
    }
}
