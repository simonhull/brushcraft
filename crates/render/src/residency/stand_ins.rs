//! Residency's side of the stand-ins ([`crate::standin`]): which resident display tiles are
//! behind a brush step, the finest tiles they read, and their rows.
//!
//! # Versions and the index
//!
//! Each resident display tile's data has a version ([`Entry::version`](super::Entry)): it
//! includes every step of epoch number below it. Data the source gives includes every step
//! before the last refresh epoch naming the tile; a stand-in, every step it read.
//!
//! A step's epoch ([`store::EpochKind::Step`]) enters the **index** when its finest tiles show:
//! at once if none of them is resident, else when the pending epoch holding them switches, so a
//! stand-in never runs ahead of a drawn finest tile beside it. The index keeps, for every place
//! above a step's finest tiles, the step's displacement and the finest tiles under the place;
//! so what a display tile is behind is a range of its own place's entries, newer than its
//! version, whatever the stroke's length. Entering a step marks the places above it **due**,
//! as loading a display tile's data marks its own: only due places are looked at.
//!
//! Every due resident display tile behind a step stands in, in the frame, reading the finest
//! tiles the steps it is behind moved under it, its bound raised by their displacements summed
//! from its stored bound. So a tile that arrives later, or whose store data switches in older
//! than a step, stands in in its first frame too: no resident display tile is drawn behind a
//! step whose finest tiles show, which keeps its seams closed (the stand-in module's notes,
//! "Seams").
//!
//! A place's entries are kept until a refresh epoch names it: whatever holds its tile later
//! (an arrival, a reload after a lost frame) reads data as old as that refresh. A revert's
//! epoch ([`store::EpochKind::Revert`]) moved finest vertices by an unknown amount: the places
//! above them stop standing in, their tiles read again from the store, until a refresh names
//! them.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use geom::TileKey;

use super::{MAX_STEPS, Residency, ResidencyError, SizeClass};
use crate::gpu::{Gpu, GpuError};
use crate::residency::Slot;
use crate::standin::{Displacement, RowBook, RowWrite, StandIn, StandInPass};
use crate::tree::Place;
use crate::upload::{Batch, TileData, TileSource};

/// A brush step: its epoch's number, its displacement bound and its finest tiles.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Step {
    /// The step's epoch number.
    pub(super) seq: u64,
    /// How far it moved any finest vertex, at most.
    pub(super) displacement: Displacement,
    /// The finest tiles its epoch named.
    pub(super) keys: Vec<TileKey>,
}

/// One step's part under one place: what a display tile there needs of it.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Under {
    /// The step's displacement bound.
    displacement: Displacement,
    /// The finest tiles under the place it moved.
    finest: Vec<TileKey>,
}

/// What a resident display tile needs to stand in: the steps since its data's version.
#[derive(Debug, Clone, Default, PartialEq)]
pub(super) struct Behind {
    /// The steps' displacement bounds, summed: how far a finest vertex under it moved since.
    displacement: Displacement,
    /// The newest step's epoch number.
    newest: u64,
    /// The finest tiles the steps moved under it.
    under: BTreeSet<TileKey>,
}

/// A tile a frame stood in, undone if the frame is lost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Stood {
    /// The tile.
    pub(super) key: TileKey,
    /// Its slot.
    pub(super) slot: Slot,
}

/// Residency's stand-in state: the pass, the rows, the index of steps and the frame's plan.
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
    /// The index: per place, the steps that moved finest tiles under it, by epoch number,
    /// until a refresh epoch names it.
    pub(super) index: BTreeMap<Place, BTreeMap<u64, Under>>,
    /// Steps entered into the index, ever: what the tests count.
    pub(super) entered: u64,
    /// Per display tile, the epoch number of the last refresh epoch naming it: its stored data
    /// includes every step before.
    pub(super) refreshed: BTreeMap<Place, u64>,
    /// Places a revert's epoch reached, with its epoch number: they don't stand in until a
    /// refresh epoch names them after it.
    pub(super) blocked: BTreeMap<Place, u64>,
    /// Places whose tiles may be behind a step: a step entered under them, or their data was
    /// loaded, since they were last looked at.
    pub(super) due: BTreeSet<Place>,
    /// Places whose tiles need more finest tiles than a scratch binds: not looked at again
    /// until a step enters under them or their data is loaded.
    pub(super) oversized: BTreeSet<Place>,
    /// The frame's stand-ins, planned while staging.
    pub(super) planned: Vec<StandIn>,
    /// The tiles the frame being staged stood in.
    pub(super) stood: Vec<Stood>,
    /// Per frame staged and not yet heard of as submitted, the tiles it stood in.
    pub(super) stood_in_flight: BTreeMap<u64, Vec<Stood>>,
    /// Tiles whose data replaced a stand-in in the last frame staged: they morph in
    /// ([`crate::tree::NODE_MORPH_IN`], for that frame).
    pub(super) morph_in: Vec<Place>,
    /// Finest tiles' positions as an epoch's staging read them, for the stand-ins over them:
    /// so a tile is read once an epoch.
    pub(super) positions: BTreeMap<TileKey, Vec<u32>>,
    /// The finest tiles whose positions the stand-ins will read: those of steps not yet
    /// entered (their pending epochs') and those entered this frame.
    pub(super) keep_positions: BTreeSet<TileKey>,
    /// Whether the frame's tree upload wrote a node that has a row: the settle runs.
    pub(super) settle: bool,
    /// A finest tile as read for a stand-in.
    pub(super) data: TileData,
}

impl StandIns {
    /// No tile standing in, with a kept scratch of `tiles` finest tiles and `rows` rows.
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
            index: BTreeMap::new(),
            entered: 0,
            refreshed: BTreeMap::new(),
            blocked: BTreeMap::new(),
            due: BTreeSet::new(),
            oversized: BTreeSet::new(),
            planned: Vec::new(),
            stood: Vec::new(),
            stood_in_flight: BTreeMap::new(),
            morph_in: Vec::new(),
            positions: BTreeMap::new(),
            keep_positions: BTreeSet::new(),
            settle: false,
            data: TileData::default(),
        })
    }

    /// Enters `step` into the index: under every place above its finest tiles, those finest
    /// tiles, each place made due. A place over [`MAX_STEPS`] steps forgets its oldest, counted
    /// in `dropped`.
    pub(super) fn enter(&mut self, step: Step, dropped: &mut u64) {
        let mut by_place: BTreeMap<Place, Vec<TileKey>> = BTreeMap::new();
        for key in &step.keys {
            let mut above = Place::of(key).parent();
            while let Some(place) = above {
                by_place.entry(place).or_default().push(*key);
                above = place.parent();
            }
        }
        for (place, finest) in by_place {
            let steps = self.index.entry(place).or_default();
            let under = steps.entry(step.seq).or_insert(Under {
                displacement: step.displacement,
                finest: Vec::new(),
            });
            under.finest.extend(finest);
            while steps.len() > MAX_STEPS {
                steps.pop_first();
                *dropped += 1;
            }
            self.due.insert(place);
            self.oversized.remove(&place);
        }
        self.keep_positions.extend(step.keys);
        self.entered += 1;
    }

    /// Notes a refresh epoch numbered `seq` naming the display tile at `place`: its stored
    /// data includes every step before, so those steps leave its index, and a revert before it
    /// no longer blocks it.
    pub(super) fn refreshed(&mut self, place: Place, seq: u64) {
        self.refreshed.insert(place, seq);
        if let Some(steps) = self.index.get_mut(&place) {
            *steps = steps.split_off(&seq);
            if steps.is_empty() {
                self.index.remove(&place);
            }
        }
        if self.blocked.get(&place).is_some_and(|&revert| revert < seq) {
            self.blocked.remove(&place);
        }
    }

    /// Notes the positions of finest tile `key`, read for an epoch's staging, if the stand-ins
    /// will read them: `words` appends them.
    pub(super) fn note_positions(&mut self, key: TileKey, words: impl FnOnce(&mut Vec<u32>)) {
        if self.keep_positions.contains(&key) {
            let mut kept = Vec::new();
            words(&mut kept);
            self.positions.insert(key, kept);
        }
    }

    /// Rebuilds the index from what a resync hands over: every display tile a stroke changed
    /// since its last refresh, behind one step of their displacements summed. Everything else
    /// is as its refresh left it.
    pub(super) fn resync(&mut self, unrefreshed: Vec<store::Unrefreshed>) {
        self.index.clear();
        self.refreshed.clear();
        self.blocked.clear();
        self.oversized.clear();
        self.positions.clear();
        for tile in unrefreshed {
            let Ok(displacement) = Displacement::try_from(tile.displacement) else {
                continue;
            };
            let place = Place::of(&tile.tile);
            let under = Under {
                displacement,
                finest: tile.finest.into_iter().collect(),
            };
            self.index.entry(place).or_default().insert(0, under);
            self.due.insert(place);
        }
    }
}

impl Residency {
    /// Plans the frame's stand-ins and stages what they read into `batch`: the rows freed
    /// since the last frame, every row again after a lost frame, the finest tiles read and the
    /// rows' steps; the tiles standing in are touched in the tree, so the bounds pass
    /// recomputes them.
    ///
    /// Tiles in the cut stand in first, whatever the time; the rest while the frame is before
    /// `deadline`, the oldest data first. A frame reading more finest tiles than the kept
    /// scratch holds uses a one-off one. A tile needing more than a scratch binds is set
    /// aside until its inputs change; a tile with no row left waits; both are counted as
    /// skipped.
    ///
    /// # Errors
    ///
    /// As [`Residency::stage`].
    pub(super) fn stage_stand_ins<S: TileSource>(
        &mut self,
        gpu: &Gpu,
        batch: &mut Batch,
        source: &mut S,
        deadline: Instant,
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
        let behind = self.behind();
        if behind.is_empty() {
            return Ok(());
        }
        let most = usize::try_from(self.stand_ins.pass.most_tiles()).unwrap_or(usize::MAX);
        let mut reads: BTreeSet<TileKey> = BTreeSet::new();
        let mut chosen: Vec<(TileKey, Behind)> = Vec::new();
        for (key, tile) in behind {
            let place = Place::of(&key);
            if tile.under.len() > most {
                self.stand_ins.oversized.insert(place);
                self.stats.stand_ins_skipped += 1;
                continue;
            }
            let in_cut = self.cut.contains(&key);
            let more = tile.under.difference(&reads).count();
            if reads.len() + more > most || (!in_cut && Instant::now() >= deadline) {
                self.skip_stand_in(place);
                continue;
            }
            reads.extend(tile.under.iter().copied());
            chosen.push((key, tile));
        }
        let scratch = self.read_finest(gpu, batch, source, &reads)?;
        for (key, tile) in chosen {
            let under: Option<Vec<(u32, Place)>> = tile
                .under
                .iter()
                .map(|f| scratch.get(f).map(|&index| (index, Place::of(f))))
                .collect();
            match under {
                Some(under) => self.stand_in(gpu, batch, key, &tile, under)?,
                None => self.skip_stand_in(Place::of(&key)),
            }
        }
        Ok(())
    }

    /// Stages the finest tiles `reads` into the frame's scratch, from the positions the
    /// frame's epochs read where it has them, else from `source` (positions alone). Returns
    /// each staged tile's index in the scratch; a tile the source has nothing for is left out.
    ///
    /// # Errors
    ///
    /// As [`Residency::stage`].
    fn read_finest<S: TileSource>(
        &mut self,
        gpu: &Gpu,
        batch: &mut Batch,
        source: &mut S,
        reads: &BTreeSet<TileKey>,
    ) -> Result<BTreeMap<TileKey, u32>, ResidencyError>
    where
        S::Error: Into<ResidencyError>,
    {
        let tiles = u32::try_from(reads.len()).unwrap_or(u32::MAX);
        self.stand_ins.pass.begin(gpu, tiles)?;
        let unread: Vec<TileKey> = reads
            .iter()
            .filter(|key| !self.stand_ins.positions.contains_key(key))
            .copied()
            .collect();
        if !unread.is_empty() {
            source.prepare_positions(&unread).map_err(Into::into)?;
        }
        let vertices = usize::try_from(SizeClass::of(self.finest).vertices()).unwrap_or(0);
        let mut scratch: BTreeMap<TileKey, u32> = BTreeMap::new();
        let mut words = Vec::with_capacity(3 * vertices + 1);
        for &key in reads {
            words.clear();
            if let Some(kept) = self.stand_ins.positions.get(&key) {
                words.extend_from_slice(kept);
            } else {
                self.stats.stand_in_reads += 1;
                let data = &mut self.stand_ins.data;
                if !source
                    .read_positions(key, data, &mut words)
                    .map_err(Into::into)?
                {
                    continue;
                }
            }
            if words.len() != 3 * vertices {
                continue;
            }
            // A whole number of 8-byte words, as staging copies need.
            words.push(0);
            let index = u32::try_from(scratch.len()).unwrap_or(u32::MAX);
            self.stand_ins
                .pass
                .stage_tile(gpu, batch, &mut self.ring, index, &words)?;
            scratch.insert(key, index);
        }
        Ok(scratch)
    }

    /// Stands `key` in, behind by `tile`, reading the finest tiles `under` (by their index in
    /// the scratch): its row's step staged (a new row's bound from the tile's stored bound),
    /// its node touched, its data's version moved past the steps. Skipped (counted) if every
    /// row is taken.
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
        let displacement = tile.displacement;
        let row = if let Some(row) = self.stand_ins.rows.get(place) {
            self.stand_ins.pass.write_row(
                gpu,
                batch,
                &mut self.ring,
                row,
                RowWrite::Step { displacement },
            )?;
            row
        } else if let Some(row) = self.stand_ins.rows.take(place) {
            let write = RowWrite::Fresh {
                node,
                place,
                stored: entry.stored_error,
                displacement,
            };
            self.stand_ins
                .pass
                .write_row(gpu, batch, &mut self.ring, row, write)?;
            row
        } else {
            self.skip_stand_in(place);
            return Ok(());
        };
        self.tree.touch(place);
        if let Some(live) = self.entries.get_mut(&key) {
            live.version = tile.newest + 1;
        }
        self.stand_ins.stood.push(Stood {
            key,
            slot: entry.slot,
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

    /// Whether the display tile at `place`, its data of `version`, is behind a step in the
    /// index.
    pub(super) fn is_behind(&self, place: Place, version: u64) -> bool {
        self.stand_ins
            .index
            .get(&place)
            .is_some_and(|steps| steps.range(version..).next().is_some())
    }

    /// Counts a stand-in that couldn't be made this frame; its place is looked at again next.
    fn skip_stand_in(&mut self, place: Place) {
        self.stats.stand_ins_skipped += 1;
        self.stand_ins.due.insert(place);
    }

    /// Every due resident display tile behind a step in the index, with what it needs to catch
    /// up: the cut's tiles first, then the oldest data first. The due places are taken.
    fn behind(&mut self) -> Vec<(TileKey, Behind)> {
        let due = std::mem::take(&mut self.stand_ins.due);
        let mut behind: Vec<(TileKey, Behind)> = Vec::new();
        for place in due {
            let Some(key) = place.key(self.finest) else {
                continue;
            };
            let (Some(entry), Some(steps)) =
                (self.entries.get(&key), self.stand_ins.index.get(&place))
            else {
                continue;
            };
            if self.stand_ins.blocked.contains_key(&place)
                || self.stand_ins.oversized.contains(&place)
            {
                continue;
            }
            let mut tile = Behind::default();
            for (&seq, under) in steps.range(entry.version..) {
                tile.displacement = tile.displacement.plus(under.displacement);
                tile.newest = seq;
                tile.under.extend(under.finest.iter().copied());
            }
            if !tile.under.is_empty() {
                behind.push((key, tile));
            }
        }
        let order = |key: &TileKey| {
            let version = self.entries.get(key).map_or(0, |entry| entry.version);
            (!self.cut.contains(key), version)
        };
        behind.sort_by_key(|(key, _)| order(key));
        behind
    }

    /// Notes `nodes`, which a frame's tree upload writes: if one has a row, the frame settles the
    /// rows (the upload wrote its stored bound over the row's).
    pub(super) fn note_tree_write(&mut self, nodes: &[crate::tree::GpuNode]) {
        if self.stand_ins.rows.is_empty() {
            return;
        }
        let with_row = nodes.iter().any(|node| {
            Place::from_packed(node.face, node.place)
                .is_some_and(|place| self.stand_ins.rows.get(place).is_some())
        });
        self.stand_ins.settle |= with_row;
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

    /// Undoes the stand-ins of a frame that never reached the GPU: each tile still in the
    /// slot it stood in in leaves (its slot holds what the lost frame never changed, or a later
    /// frame's stand-in over it), to come back with the store's data, behind its steps again,
    /// as [`Residency::recover`] does with the tiles a lost frame wrote. Every row is written
    /// again next frame.
    pub(super) fn unstand(&mut self, stood: Vec<Stood>) {
        for tile in stood {
            let here = self
                .entries
                .get(&tile.key)
                .is_some_and(|entry| entry.slot == tile.slot);
            if here {
                self.remove_live(&tile.key);
            }
        }
        self.stand_ins.rows_rewrite = true;
        self.tree.mark_all_dirty();
    }

    /// Keeps the finest positions the stand-ins will read and forgets the rest, at the end of a
    /// frame: those of steps still pending, as their epochs' staging read them.
    pub(super) fn keep_positions(&mut self) {
        let pending: BTreeSet<TileKey> = self
            .groups
            .iter()
            .flat_map(|group| {
                group
                    .steps
                    .iter()
                    .flat_map(|step| step.keys.iter().copied())
            })
            .collect();
        self.stand_ins.keep_positions = pending;
        let keep = &self.stand_ins.keep_positions;
        self.stand_ins.positions.retain(|key, _| keep.contains(key));
    }
}
