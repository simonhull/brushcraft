//! Residency's side of the stand-ins ([`crate::standin`]): which resident display tiles are
//! behind a brush step, the finest tiles they read, and their rows.
//!
//! Each resident display tile's data has a version ([`Entry::version`](super::Entry)): the
//! steps it includes. A step's epoch names its finest tiles ([`Residency::apply_step`]); every
//! resident display ancestor of one whose data is older than the step stands in, in the next
//! frame staged, reading the finest tiles every step it is behind edited under it, its bound
//! raised by those steps' displacements summed. So a tile that arrives later, or whose refresh
//! switches in with data older than a step, stands in in its first frame too: no resident
//! display tile is drawn behind a step whose finest tiles it covers, which is what keeps its
//! seams closed (the stand-in module's notes, "Seams").
//!
//! A step is kept until every display tile above its finest tiles has caught up: resident
//! ones by a stand-in or a refresh, the rest by a refresh epoch naming them.

use std::collections::{BTreeMap, BTreeSet};

use geom::TileKey;

use super::{Behind, Residency, ResidencyError, SizeClass, Step, Stood};
use crate::gpu::{Gpu, GpuError};
use crate::standin::{RowBook, RowWrite, StandIn, StandInPass};
use crate::tree::Place;
use crate::upload::{Batch, TileData, TileSource};

/// Residency's stand-in state: the pass, the rows, the steps kept and the frame's plan.
#[derive(Debug)]
pub(super) struct StandIns {
    /// The stand-in pass.
    pub(super) pass: StandInPass,
    /// Which display tiles stand in, by row.
    pub(super) rows: RowBook,
    /// Rows freed since the last frame staged, to write free.
    pub(super) row_frees: Vec<u32>,
    /// Whether every row is to be written again (after a lost frame).
    pub(super) rows_rewrite: bool,
    /// The brush steps some display tile above them hasn't caught up with, oldest first.
    pub(super) history: Vec<Step>,
    /// Per display tile, the epoch number of the last refresh epoch naming it: its stored data
    /// includes every step before.
    pub(super) refreshed: BTreeMap<Place, u64>,
    /// Whether a display tile may be behind a step: a step came, a display tile's data was
    /// loaded, or a frame was lost since the stand-ins were last planned.
    pub(super) due: bool,
    /// The frame's stand-ins, planned while staging.
    pub(super) planned: Vec<StandIn>,
    /// The tiles the frame being staged stood in.
    pub(super) stood: Vec<Stood>,
    /// Per frame staged and not yet heard of as submitted, the tiles it stood in.
    pub(super) stood_in_flight: BTreeMap<u64, Vec<Stood>>,
    /// Tiles whose data replaced a stand-in in the last frame staged: they morph in
    /// ([`crate::tree::NODE_MORPH_IN`], for that frame).
    pub(super) morph_in: Vec<Place>,
    /// A finest tile as read for a stand-in.
    pub(super) data: TileData,
}

impl StandIns {
    /// No tile standing in, with a scratch of `tiles` finest tiles and `rows` rows.
    ///
    /// # Errors
    ///
    /// As [`StandInPass::new`].
    pub(super) fn new(gpu: &Gpu, tiles: u32, rows: u32) -> Result<Self, GpuError> {
        Ok(Self {
            pass: StandInPass::new(gpu, tiles, rows)?,
            rows: RowBook::new(rows),
            row_frees: Vec::new(),
            rows_rewrite: false,
            history: Vec::new(),
            refreshed: BTreeMap::new(),
            due: false,
            planned: Vec::new(),
            stood: Vec::new(),
            stood_in_flight: BTreeMap::new(),
            morph_in: Vec::new(),
            data: TileData::default(),
        })
    }
}

impl Residency {
    /// Plans the frame's stand-ins and stages what they read into `batch`: the rows freed
    /// since the last frame, every row again after a lost frame, the finest tiles read and the
    /// rows' steps; the tiles standing in are touched in the tree, so the bounds pass
    /// recomputes them.
    ///
    /// The tiles behind a step stand in together, as many as the scratch and the rows allow,
    /// the oldest data first; the rest are counted as skipped and try again the next frame.
    ///
    /// # Errors
    ///
    /// As [`Residency::stage`].
    pub(super) fn stage_stand_ins<S: TileSource>(
        &mut self,
        gpu: &Gpu,
        batch: &mut Batch,
        source: &mut S,
    ) -> Result<(), ResidencyError>
    where
        S::Error: Into<ResidencyError>,
    {
        for row in std::mem::take(&mut self.stand_ins.row_frees) {
            self.stand_ins
                .pass
                .write_row(gpu, batch, &mut self.ring, row, RowWrite::Free)?;
        }
        if std::mem::take(&mut self.stand_ins.rows_rewrite) {
            self.rewrite_rows(gpu, batch)?;
        }
        if !std::mem::take(&mut self.stand_ins.due) {
            return Ok(());
        }
        let behind = self.behind();
        if behind.is_empty() {
            return Ok(());
        }
        let capacity = usize::try_from(self.stand_ins.pass.scratch_tiles()).unwrap_or(usize::MAX);
        let mut reads: BTreeSet<TileKey> = BTreeSet::new();
        let mut chosen: Vec<(TileKey, Behind)> = Vec::new();
        for (key, tile) in behind {
            let more = tile.under.difference(&reads).count();
            if reads.len() + more <= capacity {
                reads.extend(tile.under.iter().copied());
                chosen.push((key, tile));
            } else {
                self.skip_stand_in();
            }
        }
        let read: Vec<TileKey> = reads.into_iter().collect();
        source.prepare(&read).map_err(Into::into)?;
        let vertices = usize::try_from(SizeClass::of(self.finest).vertices()).unwrap_or(0);
        let mut scratch: BTreeMap<TileKey, u32> = BTreeMap::new();
        let mut words = Vec::with_capacity(3 * vertices);
        for key in read {
            self.stand_ins.data.reset(vertices);
            if source
                .read(key, &mut self.stand_ins.data)
                .map_err(Into::into)?
                .is_none()
            {
                continue;
            }
            words.clear();
            self.stand_ins.data.positions(&mut words);
            if words.len() != 3 * vertices {
                continue;
            }
            let index = u32::try_from(scratch.len()).unwrap_or(u32::MAX);
            self.stand_ins
                .pass
                .stage_tile(gpu, batch, &mut self.ring, index, &words)?;
            scratch.insert(key, index);
        }
        for (key, tile) in chosen {
            let under: Option<Vec<(u32, Place)>> = tile
                .under
                .iter()
                .map(|f| scratch.get(f).map(|&index| (index, Place::of(f))))
                .collect();
            match under {
                Some(under) => self.stand_in(gpu, batch, key, &tile, under)?,
                None => self.skip_stand_in(),
            }
        }
        Ok(())
    }

    /// Stands `key` in, behind by `tile`, reading the finest tiles `under` (by their index in
    /// the scratch): its row's step staged, its node touched, its data's version moved past the
    /// steps. Skipped (counted) if every row is taken.
    ///
    /// # Errors
    ///
    /// As [`Residency::stage`].
    fn stand_in(
        &mut self,
        gpu: &Gpu,
        batch: &mut Batch,
        key: TileKey,
        tile: &Behind,
        under: Vec<(u32, Place)>,
    ) -> Result<(), ResidencyError> {
        let place = Place::of(&key);
        let (Some(entry), Some(node)) = (self.entries.get(&key).copied(), self.tree.node(place))
        else {
            return Ok(());
        };
        let (row, fresh) = if let Some(row) = self.stand_ins.rows.get(place) {
            (row, false)
        } else if let Some(row) = self.stand_ins.rows.take(place) {
            (row, true)
        } else {
            self.skip_stand_in();
            return Ok(());
        };
        let displacement = tile.displacement;
        let write = if fresh {
            RowWrite::Fresh {
                node,
                place,
                displacement,
            }
        } else {
            RowWrite::Step { displacement }
        };
        self.stand_ins
            .pass
            .write_row(gpu, batch, &mut self.ring, row, write)?;
        self.tree.touch(place);
        if let Some(live) = self.entries.get_mut(&key) {
            live.version = tile.newest + 1;
        }
        self.stand_ins.stood.push(Stood {
            key,
            slot: entry.slot,
            before: entry.version,
            row,
            fresh,
        });
        self.stand_ins.planned.push(StandIn {
            place,
            slot: entry.slot,
            row,
            under,
        });
        self.stats.stand_ins += 1;
        Ok(())
    }

    /// Whether the display tile at `place`, its data of `version`, is behind a kept step whose
    /// finest tiles it covers.
    pub(super) fn is_behind(&self, place: Place, version: u64) -> bool {
        let steps = self
            .stand_ins
            .history
            .iter()
            .filter(|step| step.seq >= version);
        let mut keys = steps.flat_map(|step| step.keys.iter());
        keys.any(|key| {
            let mut above = Place::of(key).parent();
            while let Some(at) = above {
                if at == place {
                    return true;
                }
                above = at.parent();
            }
            false
        })
    }

    /// Counts a stand-in that couldn't be made this frame, to try again the next.
    fn skip_stand_in(&mut self) {
        self.stats.stand_ins_skipped += 1;
        self.stand_ins.due = true;
    }

    /// Every resident display tile behind a kept step whose finest tiles it covers, with what
    /// it needs to catch up, oldest data first.
    fn behind(&self) -> Vec<(TileKey, Behind)> {
        let mut behind: BTreeMap<TileKey, Behind> = BTreeMap::new();
        for step in &self.stand_ins.history {
            for finest_key in &step.keys {
                let mut above = Place::of(finest_key).parent();
                while let Some(place) = above {
                    above = place.parent();
                    let Some(key) = place.key(self.finest) else {
                        continue;
                    };
                    let Some(entry) = self.entries.get(&key) else {
                        continue;
                    };
                    if entry.version > step.seq {
                        continue;
                    }
                    let tile = behind.entry(key).or_default();
                    // Steps come oldest first: each one's displacement is added once.
                    if tile.under.is_empty() || tile.newest != step.seq {
                        tile.displacement = tile.displacement.plus(step.displacement);
                        tile.newest = step.seq;
                    }
                    tile.under.insert(*finest_key);
                }
            }
        }
        let mut tiles: Vec<(TileKey, Behind)> = behind.into_iter().collect();
        let version = |key: &TileKey| self.entries.get(key).map_or(0, |entry| entry.version);
        tiles.sort_by_key(|(key, _)| version(key));
        tiles
    }

    /// Writes every row again: the taken rows' tiles (not their bounds, which the GPU keeps) and
    /// the free rows free. After a lost frame, whose row writes never reached the GPU.
    ///
    /// # Errors
    ///
    /// As [`Residency::stage`].
    fn rewrite_rows(&mut self, gpu: &Gpu, batch: &mut Batch) -> Result<(), ResidencyError> {
        let taken: BTreeMap<u32, Place> =
            self.stand_ins.rows.taken().map(|(p, r)| (r, p)).collect();
        for row in 0..self.stand_ins.rows.high() {
            let write = match taken.get(&row) {
                Some(&place) => match self.tree.node(place) {
                    Some(node) => RowWrite::Kept { node, place },
                    None => RowWrite::Free,
                },
                None => RowWrite::Free,
            };
            self.stand_ins
                .pass
                .write_row(gpu, batch, &mut self.ring, row, write)?;
        }
        Ok(())
    }

    /// Undoes the stand-ins of a frame that never reached the GPU: each tile still in its slot
    /// is behind its steps again, and a row made for it is freed. Every row is written again
    /// next frame.
    pub(super) fn unstand(&mut self, stood: Vec<Stood>) {
        for tile in stood {
            if let Some(entry) = self.entries.get_mut(&tile.key)
                && entry.slot == tile.slot
            {
                entry.version = entry.version.min(tile.before);
            }
            let place = Place::of(&tile.key);
            if tile.fresh && self.stand_ins.rows.get(place) == Some(tile.row) {
                self.stand_ins.rows.release(place);
            }
        }
        self.stand_ins.rows_rewrite = true;
        self.stand_ins.due = !self.stand_ins.history.is_empty();
    }

    /// Drops the steps every display tile above them has caught up with: each place above
    /// their finest tiles refreshed since, and its resident tile, if any, past them. The
    /// refreshes older than every step left are forgotten too: a tile they name is behind
    /// every step left either way.
    pub(super) fn prune_history(&mut self) {
        let (finest, entries, refreshed) = (self.finest, &self.entries, &self.stand_ins.refreshed);
        self.stand_ins.history.retain(|step| {
            step.keys.iter().any(|key| {
                let mut above = Place::of(key).parent();
                while let Some(place) = above {
                    let refreshed_since = refreshed.get(&place).is_some_and(|&r| r >= step.seq);
                    let resident = place.key(finest).and_then(|k| entries.get(&k));
                    let caught_up = resident.is_none_or(|entry| entry.version > step.seq);
                    if !(refreshed_since && caught_up) {
                        return true;
                    }
                    above = place.parent();
                }
                false
            })
        });
        match self.stand_ins.history.first() {
            None => self.stand_ins.refreshed.clear(),
            Some(oldest) => {
                let oldest = oldest.seq;
                self.stand_ins.refreshed.retain(|_, &mut seq| seq >= oldest);
            }
        }
    }
}
