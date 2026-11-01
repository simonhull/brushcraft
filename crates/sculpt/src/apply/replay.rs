//! A run of a frame's dabs replayed tile by tile, tiles in parallel.
//!
//! Each tile takes every dab of the run that reaches it, in order, on a copy of its own
//! corrections, and the tiles are written back whole once the run is done (see "Tile by tile"
//! in the module notes of [`super`]).

use std::collections::BTreeMap;

use brush::{Imprint, Smoothstep, draw_imprint};
use geom::{
    CorrectionError, CorrectionTile, CorrectionView, Prediction, TileBorder, TileKey, Topology,
    Vec3,
};
use store::{Aabb, TileStore};

use super::vertices;
use crate::state::{Book, FinalState, Grant, Stroke, stroke_sum};

/// Which stroke state each tile's writes take, at each dab of a run.
///
/// What a replay assumes of the decisions [`Book::begin_dab`] makes, which depend on the tiles
/// every dab edits and so can only be checked once the run is replayed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Plan {
    /// No tile's state is released or denied during the run: each tile goes on with the state
    /// the stroke holds for it, or, if it holds none, with fresh state at its first edit. What
    /// every run whose dabs fit in the stroke's free buffers gets, and what nearly every other
    /// does (the tiles released are the stroke's oldest, which a run rarely edits again).
    Continuous,
    /// The grants a book gave each tile at each dab of the run that edited it, in dab order:
    /// what a second replay takes, after the first replay's dabs showed the first was wrong.
    Granted(Grants),
}

/// For each tile, the grant of its stroke state at each dab of a run that edited it, in dab
/// order (dabs counted from the run's first).
pub(super) type Grants = BTreeMap<TileKey, Vec<(usize, Grant)>>;

impl Plan {
    /// The grants a book gives the tiles that `outcomes` edited, dab by dab, from a copy of
    /// `book` (the stroke's as the run begins), with that copy after the run: the decisions the
    /// stroke makes for the dabs as replayed. And the tiles written per dab, summed over the
    /// dabs.
    pub(super) fn of(book: &Book, outcomes: &[TileOutcome], dabs: usize) -> (Grants, Book, usize) {
        let mut edited: Vec<Vec<TileKey>> = vec![Vec::new(); dabs];
        for outcome in outcomes {
            for edit in &outcome.edits {
                if let Some(tiles) = edited.get_mut(edit.dab) {
                    tiles.push(outcome.key);
                }
            }
        }
        let mut book = book.clone();
        let mut grants = Grants::new();
        let mut released = Vec::new();
        let mut without = 0;
        for (dab, mut tiles) in edited.into_iter().enumerate() {
            tiles.sort_unstable();
            let given = book.begin_dab(&tiles, &mut released);
            let granted = tiles.into_iter().zip(given);
            for (key, grant) in granted {
                without += usize::from(grant == Grant::Without);
                grants.entry(key).or_default().push((dab, grant));
            }
        }
        (grants, book, without)
    }

    /// `key`'s grant at `dab`, as the plan has it: under [`Plan::Continuous`], always
    /// [`Grant::Held`] (a fresh state at a first edit is the state a tile the stroke holds none
    /// for starts the run with, see [`StrokeState::found`]).
    fn grant(&self, key: &TileKey, dab: usize) -> Grant {
        match self {
            Self::Continuous => Grant::Held,
            Self::Granted(grants) => grants
                .get(key)
                .and_then(|grants| grants.iter().find(|&&(at, _)| at == dab))
                .map_or(Grant::Held, |&(_, grant)| grant),
        }
    }

    /// Whether `key`'s state is taken afresh at a dab of the run after `after` (all of them, for
    /// `None`) up to `dab`, included: its starts and sums restart from nothing then.
    fn taken_between(&self, key: &TileKey, after: Option<usize>, dab: usize) -> bool {
        let Self::Granted(grants) = self else {
            return false;
        };
        grants.get(key).is_some_and(|grants| {
            grants.iter().any(|&(at, grant)| {
                grant == Grant::Taken && after.is_none_or(|after| at > after) && at <= dab
            })
        })
    }
}

/// What every tile of a run reads: its dabs, the plan, the stroke and the mesh.
#[derive(Clone, Copy)]
pub(super) struct Run<'a> {
    /// The run's dabs, in order.
    pub(super) imprints: &'a [Imprint],
    /// What the replay assumes of the stroke state's grants.
    pub(super) plan: &'a Plan,
    /// The stroke, as the run begins: the state each tile and each vertex starts from.
    pub(super) stroke: &'a Stroke,
    /// The mesh's topology: the copies and owners of border vertices.
    pub(super) topology: &'a Topology,
}

/// One tile's part of a run, as the store holds it when the run begins.
#[derive(Clone, Copy)]
pub(super) struct TileInput<'a> {
    /// The tile.
    pub(super) key: TileKey,
    /// Its positions, current.
    pub(super) positions: &'a [Vec3],
    /// Its corrections, as stored.
    pub(super) corrections: CorrectionView<'a>,
    /// Its prediction, for its positions after each edit: needed when more than one dab can
    /// reach it, so for every tile of a run of two dabs or more.
    pub(super) prediction: Option<&'a Prediction>,
}

/// What one dab did to a tile it edited (one owning a vertex it moved).
#[derive(Debug, Clone, Copy)]
pub(super) struct DabEdit {
    /// The dab, counted from the run's first.
    pub(super) dab: usize,
    /// The vertices it moved that the tile owns.
    pub(super) moved: usize,
    /// The largest of their deltas, in magnitude.
    pub(super) delta: f32,
    /// The tile's quantization step after the dab's writes.
    pub(super) scale: f32,
}

/// Why a tile's replay stopped: the run is then applied one dab at a time instead, the way
/// that gives its own errors, as nothing has been written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Stopped {
    /// A value the tile can't hold ([`CorrectionError::TooLarge`] or
    /// [`CorrectionError::NotFinite`]).
    Refused(CorrectionError),
    /// A second dab reached the tile, which has no prediction to give its positions after the
    /// first's writes.
    NoPrediction,
    /// The mesh lists no border for the tile (a tile holding a vertex twice, which no mesh the
    /// store accepts has).
    NoBorder,
}

/// What a run did to one tile.
#[derive(Debug)]
pub(super) struct TileOutcome {
    /// The tile.
    pub(super) key: TileKey,
    /// Its corrections after the run, if a dab wrote any of its vertices (its own, or its copies
    /// of a neighbour's).
    pub(super) written: Option<CorrectionTile>,
    /// Whether a vertex of it was within a dab's radius: its positions moved.
    pub(super) reached: bool,
    /// Each dab that edited it, in order.
    pub(super) edits: Vec<DabEdit>,
    /// The grant its own writes took at each of those dabs, as the plan had it.
    pub(super) used: Vec<(usize, Grant)>,
    /// Its own stroke state after the run, if a dab wrote it with state.
    pub(super) state: Option<FinalState>,
    /// The border vertices written, each once.
    pub(super) border: Vec<(u32, u32)>,
    /// Why the replay stopped, if it did.
    pub(super) stopped: Option<Stopped>,
}

/// A vertex's stroke state as a replay carries it: the start and sum, and the dab of the run
/// it was last brought up to (`None`: as the run found it).
#[derive(Debug, Clone, Copy)]
struct Entry {
    /// The start, `NaN` if not captured.
    start: f32,
    /// The sum of the stroke's deltas.
    acc: f32,
    /// The last dab of the run that wrote it.
    as_of: Option<usize>,
}

impl Entry {
    /// `owner`'s entry `at` as the stroke holds it when the run begins: fresh if it holds no
    /// state for `owner`.
    fn found(stroke: &Stroke, owner: &TileKey, at: usize) -> Self {
        let (start, acc) = stroke
            .held(owner)
            .and_then(|state| state.entry(at))
            .unwrap_or((f32::NAN, 0.0));
        Self {
            start,
            acc,
            as_of: None,
        }
    }

    /// The `z` this entry's vertex, owned by `owner`, is written with at `dab`, whose stored
    /// `z` is `current` before the write, for a delta of `delta`: as the store's edit writes it
    /// (see `stroke_write` in [`super`]), the plan saying which state the owner has.
    fn write(&mut self, plan: &Plan, owner: &TileKey, dab: usize, current: f32, delta: f32) -> f32 {
        if plan.taken_between(owner, self.as_of, dab) {
            self.start = f32::NAN;
            self.acc = 0.0;
        }
        self.as_of = Some(dab);
        if plan.grant(owner, dab) == Grant::Without {
            return current + delta;
        }
        let (start, acc) = stroke_sum(self.start, self.acc, current, delta);
        self.start = start;
        self.acc = acc;
        start + acc
    }
}

/// A tile's own stroke state through a run: every entry of the tile, from the stroke's buffer
/// for it or fresh, and the dab it was last brought up to.
#[derive(Debug)]
struct StrokeState {
    /// Each vertex's start.
    start: Vec<f32>,
    /// Each vertex's sum.
    acc: Vec<f32>,
    /// The last dab of the run that wrote it.
    as_of: Option<usize>,
    /// Whether it still continues the buffer the stroke held for the tile.
    continuous: bool,
}

impl StrokeState {
    /// `key`'s state as the stroke holds it when the run begins (`entries` of them), or fresh.
    fn found(stroke: &Stroke, key: &TileKey, entries: usize) -> Self {
        let held = stroke.held(key);
        let mut start = vec![f32::NAN; entries];
        let mut acc = vec![0.0; entries];
        if let Some(state) = held {
            let slots = start.iter_mut().zip(acc.iter_mut()).enumerate();
            for (at, (to_start, to_acc)) in slots {
                (*to_start, *to_acc) = state.entry(at).unwrap_or((f32::NAN, 0.0));
            }
        }
        Self {
            start,
            acc,
            as_of: None,
            continuous: held.is_some(),
        }
    }

    /// Brings the state to `dab`, a dab that edits the tile: restarted if the plan takes it
    /// afresh since it was last brought up.
    fn reach(&mut self, plan: &Plan, key: &TileKey, dab: usize) {
        if plan.taken_between(key, self.as_of, dab) {
            self.start.fill(f32::NAN);
            self.acc.fill(0.0);
            self.continuous = false;
        }
        self.as_of = Some(dab);
    }

    /// The `z` entry `at` is written with, as [`Entry::write`] for the tile's own vertex, the
    /// state brought to the dab already, which `grant` the dab's.
    fn write(&mut self, grant: Grant, at: usize, current: f32, delta: f32) -> f32 {
        let (Some(start), Some(acc)) = (self.start.get_mut(at), self.acc.get_mut(at)) else {
            return current + delta;
        };
        if grant == Grant::Without {
            return current + delta;
        }
        let (new_start, new_acc) = stroke_sum(*start, *acc, current, delta);
        *start = new_start;
        *acc = new_acc;
        new_start + new_acc
    }
}

/// A border vertex of a tile, `(i, j)`, with its owner.
type Owned = ((u32, u32), Owner);

/// A border vertex of a tile: where its owner holds it.
#[derive(Debug, Clone, Copy)]
struct Owner {
    /// The owner's tile.
    key: TileKey,
    /// The vertex's entry in the owner's state, row-major in the owner's side.
    at: usize,
    /// Whether the owner is this tile.
    own: bool,
}

/// One tile's replay of the run: see "Tile by tile" in the module notes of [`super`].
pub(super) fn replay_tile(run: Run<'_>, input: TileInput<'_>) -> TileOutcome {
    let mut tile = Replay::new(run, input);
    let stopped = tile.dabs().err();
    tile.finish(stopped)
}

/// A tile's replay in progress.
struct Replay<'a> {
    /// What every tile reads.
    run: Run<'a>,
    /// What this tile reads.
    input: TileInput<'a>,
    /// Its corrections, as the dabs so far left them.
    tile: CorrectionTile,
    /// Its positions, as the dabs so far left them (current while `stale` is false).
    positions: Vec<Vec3>,
    /// For each row of the tile, a box holding every position the row has had since the run
    /// began (`None` while none is finite): it bounds the row's positions now.
    rows: Vec<Option<Aabb>>,
    /// The rows the dab being applied was brushed over: the only rows whose deltas may not be
    /// zero.
    brushed: Vec<u32>,
    /// Whether a write moved positions the tile has no prediction to give.
    stale: bool,
    /// Each vertex's delta of the dab being applied.
    deltas: Vec<f32>,
    /// The owner of each border vertex, found when first needed.
    owners: Option<BTreeMap<(u32, u32), Owner>>,
    /// The tile's own stroke state, from its first edit.
    own: Option<StrokeState>,
    /// The stroke state of the border vertices it holds copies of, from their first write.
    copies: BTreeMap<(u32, u32), Entry>,
    /// Whether a dab wrote the tile.
    written: bool,
    /// Whether a dab reached it.
    reached: bool,
    /// The dabs that edited it.
    edits: Vec<DabEdit>,
    /// The grants its own writes took.
    used: Vec<(usize, Grant)>,
    /// The border vertices written, each once.
    border: Vec<(u32, u32)>,
    /// Whether each vertex is among `border`, row-major.
    border_seen: Vec<bool>,
}

/// What a dab moves of a tile, from one scan of the rows it was brushed over.
#[derive(Debug, Default)]
struct Moves {
    /// The interior vertices it moves: all the tile's own.
    inside: usize,
    /// The first and last row holding one of them.
    rows: Option<(u32, u32)>,
    /// The largest of their deltas, in magnitude.
    largest: f32,
    /// The border vertices it moves, in row-major order.
    border: Vec<(u32, u32)>,
}

impl Moves {
    /// The moves of `deltas`, a tile's of `side` vertices a side, row-major, over `rows` (in
    /// order), the only rows with a delta that isn't zero.
    fn scan(side: u32, deltas: &[f32], rows: &[u32]) -> Self {
        let last = side.saturating_sub(1);
        let mut moves = Self::default();
        for &j in rows {
            let row = deltas.get(row_range(side, j)).unwrap_or_default();
            let columns = (0..side).zip(row);
            for (i, &delta) in columns {
                if delta == 0.0 {
                    continue;
                }
                if i == 0 || j == 0 || i == last || j == last {
                    moves.border.push((i, j));
                    continue;
                }
                moves.inside += 1;
                moves.largest = moves.largest.max(delta.abs());
                moves.rows = Some(moves.rows.map_or((j, j), |(first, _)| (first, j)));
            }
        }
        moves
    }
}

/// The places of row `j`'s vertices, row-major, in a tile of `side` vertices a side.
fn row_range(side: u32, j: u32) -> std::ops::Range<usize> {
    let width = usize::try_from(side).unwrap_or(0);
    let first = usize::try_from(j).unwrap_or(0) * width;
    first..first + width
}

impl<'a> Replay<'a> {
    /// The replay of `input` before any dab.
    fn new(run: Run<'a>, input: TileInput<'a>) -> Self {
        let positions = input.positions.to_vec();
        let side = input.key.side();
        let rows = (0..side)
            .map(|j| {
                let row = positions.get(row_range(side, j)).unwrap_or_default();
                Aabb::from_points(row.iter().copied())
            })
            .collect();
        Self {
            run,
            input,
            tile: input.corrections.to_tile(),
            deltas: vec![0.0; positions.len()],
            positions,
            rows,
            brushed: Vec::new(),
            stale: false,
            owners: None,
            own: None,
            copies: BTreeMap::new(),
            written: false,
            reached: false,
            edits: Vec::new(),
            used: Vec::new(),
            border: Vec::new(),
            border_seen: Vec::new(),
        }
    }

    /// Every dab of the run that can reach the tile, in order.
    ///
    /// # Errors
    ///
    /// Why it stopped (see [`Stopped`]).
    fn dabs(&mut self) -> Result<(), Stopped> {
        let imprints = self.run.imprints.iter().enumerate();
        for (dab, imprint) in imprints {
            // Without a prediction, the box no longer bounds positions a write moved: any dab
            // after one could reach them.
            if self.stale {
                return Err(Stopped::NoPrediction);
            }
            if self.brush(imprint) > 0 {
                self.reached = true;
                self.apply(dab)?;
            }
        }
        Ok(())
    }

    /// Brushes `imprint` over the rows its sphere touches by their boxes, the deltas of every
    /// other row zero, and returns how many vertices are within its radius.
    ///
    /// A vertex within the radius lies in its row's box, which so touches the sphere: the rows
    /// left out hold none. The test is the one the frame's tiles are chosen by (see "The frame's
    /// tiles" in the module notes of [`super`]).
    fn brush(&mut self, imprint: &Imprint) -> usize {
        let side = self.input.key.side();
        for &j in &self.brushed {
            if let Some(row) = self.deltas.get_mut(row_range(side, j)) {
                row.fill(0.0);
            }
        }
        let [x, y, z] = imprint.center;
        let centre = Vec3::new(x, y, z);
        let radius_squared = imprint.radius * imprint.radius;
        let touched = (0..side).zip(&self.rows).filter(|(_, aabb)| {
            aabb.is_some_and(|aabb| aabb.distance_squared(centre) <= radius_squared)
        });
        self.brushed = touched.map(|(j, _)| j).collect();
        let mut inside = 0;
        for &j in &self.brushed {
            let range = row_range(side, j);
            let (Some(positions), Some(deltas)) = (
                self.positions.get(range.clone()),
                self.deltas.get_mut(range),
            ) else {
                continue;
            };
            inside += draw_imprint(
                &Smoothstep,
                imprint,
                bytemuck::cast_slice(positions),
                deltas,
            );
        }
        inside
    }

    /// Writes dab `dab`'s deltas, as the store's edit of the tile would (interior row by row,
    /// then the border), and brings the positions up to date.
    ///
    /// # Errors
    ///
    /// [`Stopped::Refused`] for a value the tile can't hold; [`Stopped::NoBorder`].
    fn apply(&mut self, dab: usize) -> Result<(), Stopped> {
        let key = self.input.key;
        let side = key.side();
        let scan = Moves::scan(side, &self.deltas, &self.brushed);
        let border = self.border_owners(&scan.border)?;
        let owned_border = border.iter().filter(|(_, owner)| owner.own);
        let moved = scan.inside + owned_border.clone().count();
        let grant = (moved > 0).then(|| self.run.plan.grant(&key, dab));
        if let Some(grant) = grant {
            self.used.push((dab, grant));
            let entries = self.deltas.len();
            let own = self
                .own
                .get_or_insert_with(|| StrokeState::found(self.run.stroke, &key, entries));
            own.reach(self.run.plan, &key, dab);
        }
        let scale = self.tile.scale();
        let mut changed: Vec<usize> = Vec::new();
        if let Some(rows) = scan.rows {
            self.write_interior(grant.unwrap_or(Grant::Held), rows, &mut changed)?;
        }
        for &((i, j), owner) in &border {
            self.write_border(dab, (i, j), owner, grant.unwrap_or(Grant::Held))?;
            changed.push(at(side, i, j));
        }
        if moved > 0 {
            let border_deltas =
                owned_border.filter_map(|&((i, j), _)| self.deltas.get(at(side, i, j)).copied());
            let delta = border_deltas.fold(scan.largest, |m, d| m.max(d.abs()));
            self.edits.push(DabEdit {
                dab,
                moved,
                delta,
                scale: self.tile.scale(),
            });
        }
        if changed.is_empty() {
            return Ok(());
        }
        self.written = true;
        if self.tile.scale().to_bits() == scale.to_bits() {
            self.update_positions(&changed);
        } else {
            let all: Vec<usize> = (0..self.positions.len()).collect();
            self.update_positions(&all);
        }
        Ok(())
    }

    /// Each of `moved`, border vertices, with its owner.
    ///
    /// # Errors
    ///
    /// [`Stopped::NoBorder`] if the mesh lists no border for the tile.
    fn border_owners(&mut self, moved: &[(u32, u32)]) -> Result<Vec<Owned>, Stopped> {
        if moved.is_empty() {
            return Ok(Vec::new());
        }
        let owners = self.owners()?;
        let found = moved
            .iter()
            .map(|&vertex| owners.get(&vertex).map(|&owner| (vertex, owner)))
            .collect::<Option<Vec<_>>>();
        found.ok_or(Stopped::NoBorder)
    }

    /// The owner of each of the tile's border vertices, found once.
    ///
    /// # Errors
    ///
    /// [`Stopped::NoBorder`] if the mesh lists no border for the tile.
    fn owners(&mut self) -> Result<&BTreeMap<(u32, u32), Owner>, Stopped> {
        if self.owners.is_none() {
            self.border_seen = vec![false; self.deltas.len()];
            let key = self.input.key;
            let border = self
                .run
                .topology
                .tile_border(&key)
                .ok_or(Stopped::NoBorder)?;
            self.owners = Some(border_owners(&key, &border).ok_or(Stopped::NoBorder)?);
        }
        self.owners.as_ref().ok_or(Stopped::NoBorder)
    }

    /// Writes the interior's deltas row by row through the tile's own view
    /// ([`geom::CorrectionViewMut::update_interior_row`], the store's own edit of an interior),
    /// each vertex's `z` from the tile's stroke state under `grant`, noting each vertex written
    /// in `changed`. Only `rows`, the first and last row with a delta, are visited: the rows
    /// outside them would write nothing.
    ///
    /// # Errors
    ///
    /// [`Stopped::Refused`] for a value the tile can't hold.
    fn write_interior(
        &mut self,
        grant: Grant,
        (first, last): (u32, u32),
        changed: &mut Vec<usize>,
    ) -> Result<(), Stopped> {
        let side = self.input.key.side();
        let deltas = &self.deltas;
        let own = &mut self.own;
        let mut view = self.tile.view_mut();
        for j in first..=last {
            let row = view.update_interior_row(j, |i, j, c| {
                let at = at(side, i, j);
                let delta = deltas.get(at).copied().unwrap_or(0.0);
                if delta == 0.0 {
                    return None;
                }
                let z = own
                    .as_mut()
                    .map_or(c.z + delta, |own| own.write(grant, at, c.z, delta));
                changed.push(at);
                Some(Vec3::new(c.x, c.y, z))
            });
            if let Some(error) = row.error {
                return Err(Stopped::Refused(error));
            }
        }
        Ok(())
    }

    /// Writes border vertex `(i, j)`'s delta of dab `dab`, its `z` from its owner's stroke
    /// state: the tile's own under `grant` if it owns the vertex, else the copy of the owner's
    /// state this tile carries.
    ///
    /// # Errors
    ///
    /// [`Stopped::Refused`] for a value that isn't finite.
    fn write_border(
        &mut self,
        dab: usize,
        (i, j): (u32, u32),
        owner: Owner,
        grant: Grant,
    ) -> Result<(), Stopped> {
        let side = self.input.key.side();
        let at = at(side, i, j);
        let delta = self.deltas.get(at).copied().unwrap_or(0.0);
        let c = self.tile.get(i, j).unwrap_or(Vec3::ZERO);
        let z = if owner.own {
            self.own
                .as_mut()
                .map_or(c.z + delta, |own| own.write(grant, at, c.z, delta))
        } else {
            let stroke = self.run.stroke;
            let entry = self
                .copies
                .entry((i, j))
                .or_insert_with(|| Entry::found(stroke, &owner.key, owner.at));
            entry.write(self.run.plan, &owner.key, dab, c.z, delta)
        };
        self.tile
            .set(i, j, Vec3::new(c.x, c.y, z))
            .map_err(Stopped::Refused)?;
        if let Some(seen) = self.border_seen.get_mut(at)
            && !*seen
        {
            *seen = true;
            self.border.push((i, j));
        }
        Ok(())
    }

    /// The positions of the vertices at `changed` (row-major places, perhaps repeated) from the
    /// tile's prediction and corrections, as the store's evaluation gives them, and the box
    /// grown to hold them; with no prediction, the positions are marked stale instead.
    fn update_positions(&mut self, changed: &[usize]) {
        let Some(prediction) = self.input.prediction else {
            self.stale = true;
            return;
        };
        let side = self.input.key.side();
        let width = usize::try_from(side).unwrap_or(usize::MAX);
        for &at in changed {
            let (Ok(i), Ok(j)) = (u32::try_from(at % width), u32::try_from(at / width)) else {
                continue;
            };
            let (Some(p), Some(frame), Some(c)) = (
                prediction.predicted(i, j),
                prediction.frame(i, j),
                self.tile.get(i, j),
            ) else {
                continue;
            };
            let moved = frame.apply(p, c);
            if let Some(slot) = self.positions.get_mut(at) {
                *slot = moved;
            }
            if let (Some(point), Some(row)) =
                (Aabb::from_points([moved]), self.rows.get_mut(at / width))
            {
                *row = Some(row.map_or(point, |aabb| aabb.union(point)));
            }
        }
    }

    /// The tile's outcome, the replay stopped for `stopped` or done.
    fn finish(self, stopped: Option<Stopped>) -> TileOutcome {
        let key = self.input.key;
        let state = self.own.map(|own| FinalState {
            key,
            continuous: own.continuous,
            start: own.start,
            acc: own.acc,
        });
        TileOutcome {
            key,
            written: self.written.then_some(self.tile),
            reached: self.reached,
            edits: self.edits,
            used: self.used,
            state,
            border: self.border,
            stopped,
        }
    }
}

/// The owner of each border vertex of tile `key`, whose border is `border`: `None` if a vertex
/// has no copies (never, for a border the mesh lists).
fn border_owners(key: &TileKey, border: &TileBorder) -> Option<BTreeMap<(u32, u32), Owner>> {
    let side = key.side();
    let last = side.checked_sub(1)?;
    let ring = vertices(side).filter(|&(i, j)| i == 0 || j == 0 || i == last || j == last);
    let level = key.level();
    let mut owners = BTreeMap::new();
    for (i, j) in ring {
        let copies = border.copies(i, j)?;
        let owner = Topology::owner(&copies)?;
        let owner_key = owner.key(level);
        let own = owner_key == *key && (owner.i, owner.j) == (i, j);
        owners.insert(
            (i, j),
            Owner {
                key: owner_key,
                at: at(owner_key.side(), owner.i, owner.j),
                own,
            },
        );
    }
    Some(owners)
}

/// Vertex `(i, j)`'s place, row-major, in a tile of `side` vertices a side.
fn at(side: u32, i: u32, j: u32) -> usize {
    usize::try_from(j * side + i).unwrap_or(usize::MAX)
}

/// Whether the twins of every border vertex the run wrote agree, bit for bit.
///
/// Each tile holding a copy of one must be among `outcomes` (in key order) and hold the same
/// value there, as checked on the store's workers. What
/// [`TileStore::replace_corrections`] asks of its caller; a run that broke it (none should: see
/// "Tile by tile" in the module notes of [`super`]) is applied one dab at a time instead.
pub(super) fn twins_agree(store: &TileStore, outcomes: &[TileOutcome]) -> bool {
    let topology = store.mesh().topology();
    let value = |key: &TileKey, i: u32, j: u32| {
        let at = outcomes
            .binary_search_by_key(key, |outcome| outcome.key)
            .ok()?;
        let outcome = outcomes.get(at)?;
        outcome.written.as_ref()?.get(i, j)
    };
    let agree = store.par_map(outcomes, |outcome| {
        let key = outcome.key;
        if outcome.border.is_empty() {
            return true;
        }
        let Some(border) = topology.tile_border(&key) else {
            return false;
        };
        outcome.border.iter().all(|&(i, j)| {
            let Some(mine) = value(&key, i, j) else {
                return false;
            };
            border.copies(i, j).is_some_and(|copies| {
                copies.iter().all(|copy| {
                    value(&copy.key(key.level()), copy.i, copy.j)
                        .is_some_and(|theirs| theirs.to_bits() == mine.to_bits())
                })
            })
        })
    });
    agree.into_iter().all(|agrees| agrees)
}

/// The tiles `outcomes` wrote, with their corrections, for [`TileStore::replace_corrections`].
pub(super) fn written(outcomes: &[TileOutcome]) -> Vec<(TileKey, &CorrectionTile)> {
    outcomes
        .iter()
        .filter_map(|outcome| outcome.written.as_ref().map(|tile| (outcome.key, tile)))
        .collect()
}

/// Whether every tile of `outcomes` finished its replay.
pub(super) fn all_done(outcomes: &[TileOutcome]) -> bool {
    outcomes.iter().all(|outcome| outcome.stopped.is_none())
}

/// Each tile's grants as its replay took them, for the tiles that `outcomes` edited.
///
/// Under [`Plan::Continuous`], a tile the stroke holds no state for takes it fresh at its first
/// edit (which [`StrokeState::found`] gave it, fresh, from the start).
pub(super) fn used(plan: &Plan, book: &Book, outcomes: &[TileOutcome]) -> Grants {
    let mut grants = Grants::new();
    for outcome in outcomes.iter().filter(|outcome| !outcome.used.is_empty()) {
        let mut used = outcome.used.clone();
        if *plan == Plan::Continuous
            && !book.holds(&outcome.key)
            && let Some(first) = used.first_mut()
        {
            first.1 = Grant::Taken;
        }
        grants.insert(outcome.key, used);
    }
    grants
}

/// What each tile of `keys` is, for a run: positions, corrections and prediction.
///
/// Read from `store`; `None` for a tile whose positions or corrections aren't cached, or that
/// lacks the prediction `predicted` asks for.
pub(super) fn inputs<'a>(
    store: &'a TileStore,
    keys: &[TileKey],
    predicted: bool,
) -> Option<Vec<TileInput<'a>>> {
    keys.iter()
        .map(|key| {
            let prediction = store.kept_prediction(key);
            if predicted && prediction.is_none() {
                return None;
            }
            Some(TileInput {
                key: *key,
                positions: store.cached_positions(key)?,
                corrections: store.cached_correction(key)?,
                prediction,
            })
        })
        .collect()
}
