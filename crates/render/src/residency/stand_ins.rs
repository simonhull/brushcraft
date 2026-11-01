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
//! A tile that can't stand in when it is behind (blocked by a revert, behind steps its place
//! forgot, past what a scratch binds, no row left, a finest tile missing) has its bound raised
//! instead, by twice the steps it is behind ([`Residency::raise_bound`]): never drawn with a
//! bound smaller than its error, though its seams may show the steps on one side only until it
//! stands in or is refreshed. Rows are as many as tiles can be resident, and the scratch grows,
//! so in practice only a revert or a forgotten step leaves a tile so.
//!
//! A place's entries are kept until a refresh epoch names it: whatever holds its tile later
//! (an arrival, a reload after a lost frame) reads data as old as that refresh. A revert's
//! epoch ([`store::EpochKind::Revert`]) moved finest vertices by an unknown amount: the places
//! above them stop standing in, their tiles read again from the store, until a refresh names
//! them.

use std::collections::{BTreeMap, BTreeSet};

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
    /// Whether its finest tiles show: its epoch's pending group switched (or it had none). A
    /// tile behind a step that doesn't show yet can't stand in (it would run ahead of a drawn
    /// finest tile), and has its bound raised meanwhile.
    shown: bool,
}

/// What a resident display tile needs to stand in: the steps since its data's version.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Behind {
    /// The steps' displacement bounds, summed: how far a finest vertex under it moved since.
    displacement: Displacement,
    /// The newest step's epoch number.
    newest: u64,
    /// The finest tiles the steps moved under it.
    under: BTreeSet<TileKey>,
    /// Whether it may stand in: not blocked by a revert, set aside as too large, or behind a
    /// step its place forgot.
    can_stand: bool,
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
    /// Per place, the steps its index forgot past [`MAX_STEPS`]: the newest one's epoch number
    /// and their displacements summed. A tile behind one doesn't stand in (its finest tiles are
    /// forgotten): its bound is raised instead, until a refresh names it.
    pub(super) dropped: BTreeMap<Place, (u64, Displacement)>,
    /// Places whose tiles may be behind a step: a step entered under them, or their data was
    /// loaded, since they were last looked at.
    pub(super) due: BTreeSet<Place>,
    /// Places whose tiles can't stand in until their inputs change: they need more finest tiles
    /// than a scratch binds, or one the source had nothing for. Not looked at again until a
    /// step enters under them or their data is loaded; their bound is raised meanwhile.
    pub(super) set_aside: BTreeSet<Place>,
    /// The frame's stand-ins, planned while staging.
    pub(super) planned: Vec<StandIn>,
    /// The tiles the frame being staged stood in.
    pub(super) stood: Vec<Stood>,
    /// Per frame staged and not yet heard of as submitted, the tiles it stood in.
    pub(super) stood_in_flight: BTreeMap<u64, Vec<Stood>>,
    /// Finest tiles' positions as an epoch's staging read them, for the stand-ins over them:
    /// so a tile is read once an epoch.
    pub(super) positions: BTreeMap<TileKey, Vec<u32>>,
    /// The finest tiles whose positions the stand-ins will read: those of steps not yet
    /// entered (their pending epochs') and those entered this frame.
    pub(super) keep_positions: BTreeSet<TileKey>,
    /// Whether the frame's tree upload wrote a node that has a row, or a row's bound was
    /// raised: the settle runs.
    pub(super) settle: bool,
    /// The newest epoch residency has applied, if any: what a tile read now has seen.
    pub(super) seen: Option<u64>,
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
            dropped: BTreeMap::new(),
            due: BTreeSet::new(),
            set_aside: BTreeSet::new(),
            planned: Vec::new(),
            stood: Vec::new(),
            stood_in_flight: BTreeMap::new(),
            positions: BTreeMap::new(),
            keep_positions: BTreeSet::new(),
            settle: false,
            seen: None,
            data: TileData::default(),
        })
    }

    /// Enters `step` into the index: under every place above its finest tiles, those finest
    /// tiles, each place made due. A place over [`MAX_STEPS`] steps forgets its oldest, counted
    /// in `dropped`.
    pub(super) fn enter(&mut self, step: Step, shown: bool, dropped: &mut u64) {
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
                shown,
            });
            under.finest.extend(finest);
            while steps.len() > MAX_STEPS {
                if let Some((seq, forgotten)) = steps.pop_first() {
                    let (through, sum) = self.dropped.entry(place).or_default();
                    *through = (*through).max(seq);
                    *sum = sum.plus(forgotten.displacement);
                }
                *dropped += 1;
            }
            self.due.insert(place);
            self.set_aside.remove(&place);
        }
        self.keep_positions.extend(step.keys);
        self.entered += 1;
    }

    /// Notes that `step`'s finest tiles show now (its pending group switched): the places above
    /// them may stand in for it.
    pub(super) fn show(&mut self, step: &Step) {
        let mut places = BTreeSet::new();
        for key in &step.keys {
            let mut above = Place::of(key).parent();
            while let Some(place) = above {
                places.insert(place);
                above = place.parent();
            }
        }
        for place in places {
            if let Some(under) = self
                .index
                .get_mut(&place)
                .and_then(|steps| steps.get_mut(&step.seq))
            {
                under.shown = true;
            }
            self.due.insert(place);
            self.set_aside.remove(&place);
        }
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
        if self
            .dropped
            .get(&place)
            .is_some_and(|&(through, _)| through < seq)
        {
            self.dropped.remove(&place);
        }
    }

    /// The steps the tile at `place` is behind from epoch number `from` on, its index's and
    /// those it forgot: their displacements summed, and the newest one's number; `None` if
    /// none.
    pub(super) fn since(&self, place: Place, from: u64) -> Option<(Displacement, u64)> {
        let forgotten = self.dropped.get(&place);
        let mut found = forgotten
            .filter(|&&(through, _)| through >= from)
            .map(|&(through, sum)| (sum, through));
        let kept = self
            .index
            .get(&place)
            .into_iter()
            .flat_map(|steps| steps.range(from..));
        for (&seq, under) in kept {
            let (sum, newest) = found.get_or_insert((Displacement::ZERO, seq));
            *sum = sum.plus(under.displacement);
            *newest = (*newest).max(seq);
        }
        found
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

    /// Forgets the index on a resync: the feed lost the epochs it was built from. What the
    /// resync hands over enters when the tiles' re-read switches ([`StandIns::enter_unrefreshed`]).
    pub(super) fn resync(&mut self) {
        self.index.clear();
        self.refreshed.clear();
        self.blocked.clear();
        self.dropped.clear();
        self.set_aside.clear();
        self.positions.clear();
    }

    /// Enters what a resync handed over: every display tile a stroke changed since its last
    /// refresh, behind one step of its displacements summed, under its own place. A tile whose
    /// displacement isn't one is blocked until a refresh, as after a revert.
    pub(super) fn enter_unrefreshed(&mut self, unrefreshed: Vec<store::Unrefreshed>) {
        for tile in unrefreshed {
            let Ok(displacement) = Displacement::try_from(tile.displacement) else {
                self.blocked.insert(Place::of(&tile.tile), 0);
                continue;
            };
            let place = Place::of(&tile.tile);
            let under = Under {
                displacement,
                finest: tile.finest.into_iter().collect(),
                shown: true,
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
    /// Every resident display tile behind a step stands in, whatever the time: the cut is a
    /// frame old, and the level choice may draw any of them. A frame reading more finest tiles
    /// than the kept scratch holds uses a one-off one, grown as it needs. A tile that can't
    /// stand in this frame (blocked by a revert, behind a step its place forgot, needing more
    /// finest tiles than a scratch binds, no row left, a finest tile the source has nothing
    /// for) keeps its data but has its bound raised by twice the steps it is behind
    /// ([`Residency::raise_bound`]): never drawn with a bound smaller than its error, though a
    /// seam beside it may show the steps on one side only until it stands in or is refreshed.
    /// Those are counted as skipped.
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
        let behind = self.behind();
        if behind.is_empty() {
            // A frame reading nothing: the one-off scratch counts it idle.
            self.stand_ins.pass.idle();
            return Ok(());
        }
        let most = usize::try_from(self.stand_ins.pass.most_tiles()).unwrap_or(usize::MAX);
        let mut reads: BTreeSet<TileKey> = BTreeSet::new();
        let mut chosen: Vec<(TileKey, Behind)> = Vec::new();
        let mut raised: Vec<TileKey> = Vec::new();
        let mut rows_left = self.stand_ins.rows.left();
        for (key, tile) in behind {
            let place = Place::of(&key);
            let has_row = self.stand_ins.rows.get(place).is_some();
            if !tile.can_stand {
                raised.push(key);
                continue;
            }
            if tile.under.len() > most {
                self.stand_ins.set_aside.insert(place);
                raised.push(key);
                continue;
            }
            let more = tile.under.difference(&reads).count();
            if reads.len() + more > most || (!has_row && rows_left == 0) {
                self.stand_ins.due.insert(place);
                raised.push(key);
                continue;
            }
            if !has_row {
                rows_left -= 1;
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
            if let Some(under) = under {
                self.stand_in(gpu, batch, key, &tile, under)?;
            } else {
                // A finest tile the source had nothing for: set aside until its inputs change.
                self.stand_ins.set_aside.insert(Place::of(&key));
                raised.push(key);
            }
        }
        for key in raised {
            self.stats.stand_ins_skipped += 1;
            self.raise_bound(gpu, batch, key)?;
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

    /// Raises the bound of `key`, a resident display tile behind steps it can't stand in for
    /// this frame, by twice those steps' displacements: its vertices stay, so a finest vertex
    /// that moved `D` from the surface it was within `e` of is within `e + D` (the second `D`
    /// the store's in-stroke margin).
    ///
    /// A stand-in's bound (in its row) is raised on the GPU by the steps since it last counted
    /// them, a row step replacing no vertex ([`Entry::bounded`](super::Entry)). Any other
    /// tile's node gets, on the CPU, the bound it was loaded with plus twice the steps since
    /// it was loaded (the store's bound counts every step it had made before), and the cone
    /// that covers everything; a node already holding that isn't written again.
    ///
    /// # Errors
    ///
    /// As [`Residency::stage`].
    pub(super) fn raise_bound(
        &mut self,
        gpu: &Gpu,
        batch: &mut Batch,
        key: TileKey,
    ) -> Result<(), ResidencyError> {
        let place = Place::of(&key);
        let Some(entry) = self.entries.get(&key).copied() else {
            return Ok(());
        };
        if let Some(row) = self.stand_ins.rows.get(place) {
            let Some((displacement, newest)) = self.stand_ins.since(place, entry.bounded) else {
                return Ok(());
            };
            self.stand_ins.pass.write_row(
                gpu,
                batch,
                &mut self.ring,
                row,
                RowWrite::Step { displacement },
            )?;
            self.stand_ins.settle = true;
            if let Some(live) = self.entries.get_mut(&key) {
                live.bounded = newest + 1;
            }
            // Lost with its frame, the raise would leave `bounded` counting steps the row's
            // bound doesn't: the tile then leaves, and comes back with the store's bound.
            self.stand_ins.stood.push(Stood {
                key,
                slot: entry.slot,
            });
            return Ok(());
        }
        let after = entry.loaded_after.map_or(0, |seen| seen + 1);
        let Some((displacement, _)) = self.stand_ins.since(place, after) else {
            return Ok(());
        };
        let raise = displacement.plus(displacement).get();
        let bound = (entry.loaded_error + raise).next_up().min(f32::MAX);
        self.tree.raise_bound(place, bound);
        Ok(())
    }

    /// Stands `key` in, behind by `tile`, reading the finest tiles `under` (by their index in
    /// the scratch): its row's step staged (a new row's bound from the tile's stored bound),
    /// its node touched, its data's version and its bound moved past the steps.
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
            // The row's bound counts the steps before `bounded` (a raise may have counted more
            // than the stand-in's vertices show): the steps since then are what moved.
            let moved = self
                .stand_ins
                .since(place, entry.bounded)
                .map_or(Displacement::ZERO, |(sum, _)| sum);
            self.stand_ins.pass.write_row(
                gpu,
                batch,
                &mut self.ring,
                row,
                RowWrite::Step {
                    displacement: moved,
                },
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
            // The rows were counted when the frame planned: never.
            return Err(ResidencyError::Invariant("a stand-in found no row"));
        };
        self.tree.touch(place);
        if let Some(live) = self.entries.get_mut(&key) {
            live.version = tile.newest + 1;
            live.bounded = tile.newest + 1;
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

    /// Whether the display tile at `place`, its data of `version`, is behind a step: one in
    /// the index, or one it forgot. For the tests.
    #[cfg(test)]
    pub(super) fn is_behind(&self, place: Place, version: u64) -> bool {
        self.stand_ins.since(place, version).is_some()
    }

    /// Every due resident display tile behind a step, with what it needs to catch up, the
    /// oldest data first. The due places are taken.
    fn behind(&mut self) -> Vec<(TileKey, Behind)> {
        let due = std::mem::take(&mut self.stand_ins.due);
        let mut behind: Vec<(TileKey, Behind)> = Vec::new();
        for place in due {
            let Some(key) = place.key(self.finest) else {
                continue;
            };
            let Some(entry) = self.entries.get(&key) else {
                continue;
            };
            let Some((displacement, newest)) = self.stand_ins.since(place, entry.version) else {
                continue;
            };
            let forgot = self
                .stand_ins
                .dropped
                .get(&place)
                .is_some_and(|&(through, _)| through >= entry.version);
            let kept = self.stand_ins.index.get(&place);
            let mut steps = kept
                .into_iter()
                .flat_map(|steps| steps.range(entry.version..));
            let can_stand = !forgot
                && !self.stand_ins.blocked.contains_key(&place)
                && !self.stand_ins.set_aside.contains(&place)
                && steps.clone().all(|(_, under)| under.shown);
            let mut tile = Behind {
                displacement,
                newest,
                under: BTreeSet::new(),
                can_stand,
            };
            // What it would read, for a tile that may stand in alone.
            if can_stand {
                for (_, under) in &mut steps {
                    tile.under.extend(under.finest.iter().copied());
                }
            }
            behind.push((key, tile));
        }
        let version = |key: &TileKey| self.entries.get(key).map_or(0, |entry| entry.version);
        behind.sort_by_key(|(key, _)| version(key));
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
