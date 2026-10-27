//! Budgeted work on the owner thread: jobs that run in slices until a deadline, keeping their
//! place, so the frame loop can give the store what time a frame has left.
//!
//! The kinds are few and known, so a job is an enum ([`Job`]), not a trait object: bringing the
//! display layer up to date after edits ([`Job::Refresh`]) and building it whole for a file
//! without one ([`Job::Rebuild`]). Both are the same work, display tiles built finest level first
//! ([`Upkeep`]); they differ in what they start from. A slice does one bounded piece: one display
//! tile built, or a bounded batch of the finest tiles it reads evaluated, so a deadline is
//! honoured to within one slice.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet};

use geom::{Level, TileKey, Vec3};

/// What a slice of a job left.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Progress {
    /// More to do: run it again.
    More,
    /// Finished.
    Done,
}

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
    /// Display tiles still to build, over every job left.
    pub left: usize,
    /// The longest slice run: the most the deadline can be overrun by.
    pub longest_slice: std::time::Duration,
}

/// A job: work the store does in slices on its owner thread.
#[derive(Debug, Clone, PartialEq)]
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

    /// Display tiles still to build.
    pub(crate) fn left(&self) -> usize {
        match self {
            Self::Refresh(upkeep) | Self::Rebuild(upkeep) => upkeep.pending.len(),
        }
    }
}

/// Display tiles to build, finest level first, for a display layer under `finest`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Upkeep {
    /// The finest level, which the display tiles are built from.
    pub(crate) finest: Level,
    /// The tiles still to build, finest level first, then in key order: a tile is built after
    /// every tile one level finer that it reads.
    pending: BTreeSet<(Reverse<Level>, TileKey)>,
    /// The finest tiles' values the tile being built reads, copied as they are evaluated.
    pub(crate) scratch: FinestScratch,
}

/// The finest positions and normals one display tile reads, copied as slices evaluate them.
///
/// A tile reading more finest tiles than the budget holds at once (a whole face's, under a
/// small budget) is still built, a batch at a time, never waiting for all of them to be
/// resident together. A tile still cached is read from the cache instead, so only the tiles a
/// job evaluated are copied. The copies hold as long as no correction changes: a write, or a
/// lost tile going back to its record, empties them, since it can change any finest tile they
/// hold.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct FinestScratch {
    /// The display tile they are for.
    pub(crate) tile: Option<TileKey>,
    /// The store's count of correction changes ([`TileStore::edits`](crate::TileStore::edits))
    /// when they were copied.
    as_of: u64,
    /// Finest tiles' positions, by key.
    pub(crate) positions: BTreeMap<TileKey, Vec<Vec3>>,
    /// Finest tiles' unit normals, by key.
    pub(crate) normals: BTreeMap<TileKey, Vec<Vec3>>,
}

impl FinestScratch {
    /// Empties it for display tile `tile` as of the store's count of correction changes
    /// `as_of`, unless it holds that tile's copies from then already.
    pub(crate) fn begin(&mut self, tile: TileKey, as_of: u64) {
        if self.tile != Some(tile) || self.as_of != as_of {
            self.tile = Some(tile);
            self.as_of = as_of;
            self.positions.clear();
            self.normals.clear();
        }
    }

    /// Empties it.
    pub(crate) fn clear(&mut self) {
        self.tile = None;
        self.positions.clear();
        self.normals.clear();
    }
}

impl Upkeep {
    /// Work building `keys` (display tiles of levels below `finest`).
    pub(crate) fn new(finest: Level, keys: impl IntoIterator<Item = TileKey>) -> Self {
        Self {
            finest,
            scratch: FinestScratch::default(),
            pending: keys
                .into_iter()
                .filter(|key| key.level() < finest)
                .map(|key| (Reverse(key.level()), key))
                .collect(),
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

    /// Whether every tile is built.
    pub(crate) fn is_empty(&self) -> bool {
        self.pending.is_empty()
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

    /// The copies hold for one tile as of one write: another tile, or a write since, empties
    /// them.
    #[test]
    fn copies_are_dropped_for_another_tile_or_after_a_write() {
        let tile = |tx| TileKey::display(FaceId::new(0), Level::new(7).unwrap(), tx, 0).unwrap();
        let finest = TileKey::new(
            geom::LayerId::BASE,
            geom::Channel::DISPLACEMENT,
            FaceId::new(0),
            Level::new(8).unwrap(),
            0,
            0,
        )
        .unwrap();
        let copy = |scratch: &mut FinestScratch| {
            scratch.positions.insert(finest, vec![Vec3::ZERO]);
            scratch.normals.insert(finest, vec![Vec3::ZERO]);
        };
        let held = |scratch: &FinestScratch| {
            scratch.positions.contains_key(&finest) && scratch.normals.contains_key(&finest)
        };
        let (before, after) = (1, 2);
        let mut scratch = FinestScratch::default();
        scratch.begin(tile(0), before);
        copy(&mut scratch);
        scratch.begin(tile(0), before);
        assert!(held(&scratch), "kept for the same tile with no write since");
        scratch.begin(tile(0), after);
        assert!(
            scratch.positions.is_empty() && scratch.normals.is_empty(),
            "a write empties them"
        );
        copy(&mut scratch);
        scratch.begin(tile(1), after);
        assert!(
            scratch.positions.is_empty() && scratch.normals.is_empty(),
            "another tile's"
        );
        copy(&mut scratch);
        scratch.clear();
        assert!(!held(&scratch) && scratch.tile.is_none(), "cleared");
    }
}
