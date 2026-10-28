//! Tile bounds kept with the tiles, so opening a project loads its spatial index instead of
//! evaluating every tile.
//!
//! The spatial index ([`crate::SpatialIndex`]) holds a [`TileBounds`] for every tile of one level,
//! the finest the store evaluates. Computing them means evaluating every tile: 14.3 s for the
//! benchmark sphere on the dev box's workers, two minutes on one core. So the files keep them: a
//! `Bounds` record per tile, named by save points like a tile record (see [`crate::ProjectFile`]),
//! holding either the tile's bounds or the fact that they aren't known.
//!
//! # Derived data, never stale
//!
//! A tile's bounds follow from corrections: its own, those of every tile it is predicted from, at
//! every coarser level, and those of its neighbours, whose positions its normals read. A stored
//! bound is only worth anything if it is exactly what evaluating the tiles a save point names
//! gives. So the rule is that **a save point never names bounds its tiles don't give**: what the
//! store isn't sure of, it stores as unknown, and loading evaluates that tile as if nothing were
//! stored. An unknown bound costs one evaluation; a stale one would misplace a tile in the index.
//!
//! The store keeps that rule with a [`Book`]:
//!
//! - every correction write is noted, and before the store next hands out or records a bound it
//!   *settles* the writes: every tile of a tracked level whose bounds they can change (see
//!   [`readers`]) becomes unknown;
//! - the spatial index records the bounds it computes, which are those of the store's current
//!   state, and they replace whatever was there;
//! - the bounds edits changed since the working file last received them go to it together, at the
//!   moment a recovery flush cuts (or a save writes), so the save point names tiles and bounds of
//!   one state: the unknown markers a settle leaves, and the bounds recorded over them (a brush's
//!   refit);
//! - every bound changed since the last save of the project file stays in memory, to be copied into
//!   it by the next save.
//!
//! A bound recorded where no edit is waiting to be flushed (most of what an index build, or a load
//! that evaluates, records) is *derived*: the files already describe that tile's current state,
//! known or unknown, so the recovery file doesn't need it, and a crash costs only its evaluation
//! again. Derived bounds stay out of the recovery file, which so grows with edits, not with what
//! the index computes, and a stroke end's cost on the drawing thread with them; they reach the
//! project file at its next save.
//!
//! A tile whose latest edit is lost (a broken slot left out of a flush, a record that no longer
//! reads) makes the bounds that edit could change unknown too: the save point holds the tile's
//! older state, which the recorded bounds don't describe.
//!
//! # Layers
//!
//! The store reads a bound as it reads a correction tile: from memory if it changed since the
//! last save, else from the working file (the recovery file), else from the base (the project
//! file as of its last save). An unknown bound in a nearer layer hides a known one further down,
//! which is what keeps an edit in the recovery file from showing the project file's stale bound.

use std::collections::{BTreeMap, BTreeSet};

use geom::{BaseMesh, Channel, EvalError, LayerId, Level, Region, TileKey, parent_tiles};

use crate::bytes::{KEY_LEN, Reader, put_key};
use crate::deps::DepGraph;
use crate::record::{HEAD, TAIL};
use crate::tile_bounds::TileBounds;

/// The state byte of a stored bound whose tile's bounds are unknown: the tile is evaluated when
/// the index is loaded.
const UNKNOWN: u8 = 0;
/// The state byte of a stored bound that holds the tile's bounds.
const KNOWN: u8 = 1;
/// Bytes of a known bound's values: ten `f32`, bit for bit.
const VALUES_LEN: usize = 4 * TileBounds::BITS;

/// Bytes of a framed `Bounds` record holding an unknown bound: framing, key and state.
pub(crate) const UNKNOWN_RECORD_LEN: u64 = (HEAD + TAIL + KEY_LEN + 1) as u64;
/// Bytes of a framed `Bounds` record holding a known bound: framing, key, state and values.
pub(crate) const KNOWN_RECORD_LEN: u64 = UNKNOWN_RECORD_LEN + VALUES_LEN as u64;

/// Whether `len` is the length of a framed `Bounds` record: [`UNKNOWN_RECORD_LEN`] or
/// [`KNOWN_RECORD_LEN`].
pub(crate) const fn is_record_len(len: u64) -> bool {
    len == UNKNOWN_RECORD_LEN || len == KNOWN_RECORD_LEN
}

/// A `Bounds` record's payload: `key | state: u8`, then, if the state is known, the ten `f32`
/// of [`TileBounds::to_bits`] (little-endian). The state is 0 for unknown, 1 for known.
pub(crate) fn bounds_payload(key: &TileKey, bounds: Option<&TileBounds>) -> Vec<u8> {
    let mut out = Vec::with_capacity(KEY_LEN + 1 + VALUES_LEN);
    put_key(&mut out, key);
    match bounds {
        Some(bounds) => {
            out.push(KNOWN);
            for bits in bounds.to_bits() {
                out.extend_from_slice(&bits.to_le_bytes());
            }
        }
        None => out.push(UNKNOWN),
    }
    out
}

/// Parses [`bounds_payload`]'s output: the tile and its bound.
///
/// `None` on any shortfall or trailing byte, an invalid key, a state other than 0 or 1, or values
/// that aren't valid bounds (see [`TileBounds::from_bits`]).
pub(crate) fn parse_bounds_payload(payload: &[u8]) -> Option<(TileKey, StoredBound)> {
    let mut r = Reader::new(payload);
    let key = r.key()?;
    let kept = match r.u8()? {
        UNKNOWN => StoredBound::Unknown,
        KNOWN => {
            let mut bits = [0; TileBounds::BITS];
            for value in &mut bits {
                *value = r.u32()?;
            }
            StoredBound::Known(TileBounds::from_bits(bits)?)
        }
        _ => return None,
    };
    r.rest().is_empty().then_some((key, kept))
}

/// A tile's bound as the store keeps it: its bounds, or the fact that they aren't known (the
/// tile is evaluated when the index is loaded).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum StoredBound {
    /// The tile's bounds, as its tiles give them.
    Known(TileBounds),
    /// Not known: the tile is evaluated.
    Unknown,
}

impl StoredBound {
    /// The bounds, if known.
    pub(crate) const fn bounds(self) -> Option<TileBounds> {
        match self {
            Self::Known(bounds) => Some(bounds),
            Self::Unknown => None,
        }
    }
}

impl From<Option<TileBounds>> for StoredBound {
    /// Known bounds, or unknown for `None`.
    fn from(bounds: Option<TileBounds>) -> Self {
        bounds.map_or(Self::Unknown, Self::Known)
    }
}

/// Whether a write of `key`'s corrections can change any evaluated position: only base-layer
/// displacement tiles are evaluated.
fn is_evaluated(key: &TileKey) -> bool {
    key.layer() == LayerId::BASE && key.channel() == Channel::DISPLACEMENT
}

/// The tiles of `level` that subdivide `key`'s area: the tile itself where `level` keeps the same
/// tiles per side, else its four quarters. `level` must be the level after `key`'s.
fn children(key: &TileKey, level: Level) -> impl Iterator<Item = TileKey> {
    let split = level.tiles_per_side() > key.level().tiles_per_side();
    let (factor, steps) = if split { (2, 0..2) } else { (1, 0..1) };
    let (tx, ty, face) = (key.tx() * factor, key.ty() * factor, key.face());
    steps.clone().flat_map(move |dy| {
        steps
            .clone()
            .filter_map(move |dx| TileKey::displacement(face, level, tx + dx, ty + dy).ok())
    })
}

/// Every tile of a level in `levels` whose bounds a write of the corrections of the `edited` tiles
/// can change.
///
/// A tile's positions are predicted from the positions of its [`parent_tiles`] and changed by its
/// own corrections; its normals are computed from the positions of its
/// [`normal_tiles`](geom::normal_tiles). So the walk goes from the coarsest edited level down to
/// the finest tracked one: at each level, the tiles whose positions move are the edited ones and
/// every tile predicted from a tile that moved at the level above, and at a tracked level, those
/// tiles and every tile whose normals read one of them are the answer.
///
/// The readers of a tile are found among its neighbours, as the neighbourhood is symmetric: a
/// tile whose normals read `s` is among `normal_tiles(s)`, and a tile predicted from `s` subdivides
/// one of them (the tests check both, tile by tile, on meshes of valence 3 to 5). Candidates are
/// kept only if [`parent_tiles`] lists a moved tile, so a coarse edit doesn't spread a ring per
/// level. Every tile whose normals might read a moved one is taken, unfiltered: the set the draw
/// brush refits.
///
/// A tile's neighbours are its normal inputs as the store itself finds them
/// ([`DepGraph::normal_inputs_or_list`]): read from `deps` where it recorded them (a tile whose
/// normals were computed lists exactly [`normal_tiles`](geom::normal_tiles) as their inputs), and
/// listed from the mesh otherwise. Listing them is the costly part, and the tiles a brush writes
/// have all had their normals computed.
///
/// Writes of other layers or channels, and of levels finer than every tracked one, change nothing
/// the index holds.
///
/// # Errors
///
/// [`EvalError`] from listing a tile's neighbours or parents: never for tiles of `mesh`.
pub(crate) fn readers(
    mesh: &BaseMesh,
    deps: &DepGraph,
    edited: &BTreeSet<TileKey>,
    levels: &BTreeSet<Level>,
) -> Result<BTreeSet<TileKey>, EvalError> {
    let mut answer = BTreeSet::new();
    let Some(&finest) = levels.last() else {
        return Ok(answer);
    };
    let mut by_level: BTreeMap<Level, BTreeSet<TileKey>> = BTreeMap::new();
    for key in edited.iter().filter(|key| is_evaluated(key)) {
        if key.level() <= finest && key.level() > Level::ZERO {
            by_level.entry(key.level()).or_default().insert(*key);
        }
    }
    let Some(mut level) = by_level.keys().next().copied() else {
        return Ok(answer);
    };
    let mut moved = BTreeSet::new();
    loop {
        moved.extend(by_level.remove(&level).unwrap_or_default());
        let mut around = BTreeSet::new();
        for key in &moved {
            around.extend(deps.normal_inputs_or_list(mesh, key)?.iter().copied());
        }
        if levels.contains(&level) {
            answer.extend(moved.iter().copied());
            answer.extend(around.iter().copied());
        }
        let Some(next) = level.child().filter(|&next| next <= finest) else {
            return Ok(answer);
        };
        let mut predicted = BTreeSet::new();
        for candidate in around.iter().flat_map(|key| children(key, next)) {
            if parent_tiles(mesh, Region::of_tile(&candidate))?
                .iter()
                .any(|parent| moved.contains(parent))
            {
                predicted.insert(candidate);
            }
        }
        moved = predicted;
        level = next;
    }
}

/// A kind of data derived from the corrections that the store keeps in its files under the
/// bounds' rule (see the module notes).
///
/// The spatial index's bounds ([`Bounds`]), or the display layer
/// ([`crate::display::Display`]).
pub(crate) trait Derived {
    /// What the store keeps of one tile.
    type Value: Clone + std::fmt::Debug;

    /// Whether a recorded value always goes to the working file with the next flush, and leaves
    /// memory once there ([`Kept::Written`]): for values too large to keep in memory until a save,
    /// whose files grow with the edits anyway. Else a value recorded with no edit waiting is
    /// derived, kept in memory for the next save only.
    const SPILLS: bool;

    /// The bytes of memory `value` holds, itself and what it owns: what a [`Book`] of a kind
    /// that [spills](Derived::SPILLS) counts of the values waiting for the working file.
    fn bytes(value: &Self::Value) -> usize;

    /// Whether a write of `key`'s corrections can change a tile of a tracked level in `levels`.
    fn reaches(levels: &BTreeSet<Level>, key: &TileKey) -> bool;

    /// Every tile of a tracked level in `levels` whose value a write of the `edited` tiles can
    /// change.
    ///
    /// # Errors
    ///
    /// [`EvalError`] from listing a tile's neighbours or parents: never for tiles of `mesh`.
    fn readers(
        mesh: &BaseMesh,
        deps: &DepGraph,
        edited: &BTreeSet<TileKey>,
        levels: &BTreeSet<Level>,
    ) -> Result<BTreeSet<TileKey>, EvalError>;
}

/// The spatial index's bounds, as a [`Book`] keeps them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Bounds;

impl Derived for Bounds {
    type Value = TileBounds;

    const SPILLS: bool = false;

    fn bytes(_: &TileBounds) -> usize {
        size_of::<TileBounds>()
    }

    fn reaches(levels: &BTreeSet<Level>, key: &TileKey) -> bool {
        levels
            .last()
            .is_some_and(|&finest| key.level() <= finest && is_evaluated(key))
    }

    fn readers(
        mesh: &BaseMesh,
        deps: &DepGraph,
        edited: &BTreeSet<TileKey>,
        levels: &BTreeSet<Level>,
    ) -> Result<BTreeSet<TileKey>, EvalError> {
        readers(mesh, deps, edited, levels)
    }
}

/// One tile's value as a [`Book`] keeps it since the last rebase.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Kept<V> {
    /// The value, as the store's current state gives it.
    Known(V),
    /// Not known: whoever needs it computes it.
    Unknown,
    /// Known, and in the working file as its latest record, which reads find instead (for a
    /// [`Derived::SPILLS`] kind, once a flush or a save of the working file wrote it).
    Written,
}

impl<V> Kept<V> {
    /// The value, if it is held here and known.
    pub(crate) const fn known(&self) -> Option<&V> {
        match self {
            Self::Known(value) => Some(value),
            Self::Unknown | Self::Written => None,
        }
    }
}

/// What the store knows of the derived data of one kind (bounds by default) it keeps in its
/// files (see the module notes).
#[derive(Debug)]
pub(crate) struct Book<D: Derived = Bounds> {
    /// The levels whose values any file of the store, or this session, holds: writes are settled
    /// against these. Empty while none is kept anywhere, and then writes are not noted.
    levels: BTreeSet<Level>,
    /// Correction tiles written since the last settle that can reach a tracked level.
    edited: BTreeSet<TileKey>,
    /// Every value changed since the store opened or was last rebased. What a save copies into
    /// the project file, and what reads find first.
    changed: BTreeMap<TileKey, Kept<D::Value>>,
    /// The values of [`Book::changed`] edits changed that the working file hasn't received since
    /// (see the module notes), and for a [`Derived::SPILLS`] kind every value recorded since: what
    /// the next flush writes.
    unflushed: BTreeSet<TileKey>,
    /// The values of [`Book::changed`] whose latest change is derived (see the module notes): no
    /// flush writes them; a save of the project file, or of the working file itself, does.
    derived: BTreeSet<TileKey>,
    /// Counts the changes made to [`Book::changed`]: a save or a roll-over made before a later
    /// one doesn't hold it.
    changes: u64,
    /// For a kind that [spills](Derived::SPILLS), the values of [`Book::changed`] held in
    /// memory ([`Kept::Known`]): each waits for the working file, which every one of them goes to
    /// with the next flush. Empty for any other kind.
    waiting: BTreeSet<TileKey>,
    /// The bytes the values of [`Book::waiting`] hold ([`Derived::bytes`]).
    waiting_bytes: usize,
}

impl<D: Derived> Default for Book<D> {
    fn default() -> Self {
        Self {
            levels: BTreeSet::new(),
            edited: BTreeSet::new(),
            changed: BTreeMap::new(),
            unflushed: BTreeSet::new(),
            derived: BTreeSet::new(),
            changes: 0,
            waiting: BTreeSet::new(),
            waiting_bytes: 0,
        }
    }
}

impl<D: Derived> Book<D> {
    /// A book over files holding values of `levels`, with `changed` the values the working file
    /// holds that no save has copied into the project file (those of a restored recovery file).
    pub(crate) fn with(
        levels: BTreeSet<Level>,
        changed: BTreeMap<TileKey, Kept<D::Value>>,
    ) -> Self {
        let mut book = Self {
            levels,
            ..Self::default()
        };
        for (key, kept) in changed {
            book.put(key, kept);
        }
        book
    }

    /// A book tracking no level that holds every one of `keys` unknown, to be flushed: for files
    /// holding values the store can't keep up to date, so its next flush or save names none.
    pub(crate) fn all_unknown(keys: impl IntoIterator<Item = TileKey>) -> Self {
        let unflushed: BTreeSet<TileKey> = keys.into_iter().collect();
        let changes = unflushed.len();
        Self {
            changed: unflushed.iter().map(|&key| (key, Kept::Unknown)).collect(),
            unflushed,
            changes: u64::try_from(changes).unwrap_or(u64::MAX),
            ..Self::default()
        }
    }

    /// Notes a write of `key`'s corrections, if it can change a tracked level's values: the next
    /// [`Book::settle`] makes those unknown.
    pub(crate) fn note_write(&mut self, key: TileKey) {
        if D::reaches(&self.levels, &key) {
            self.edited.insert(key);
        }
    }

    /// [`Book::settle`], then [`Book::record`] of each of `values`, with the same result; a value
    /// the settle would make unknown and `values` records is set once, to its recorded value, to
    /// be flushed (as `record` over an unknown left by the settle does).
    ///
    /// # Errors
    ///
    /// As [`Book::settle`]; nothing is recorded then.
    pub(crate) fn settle_and_record(
        &mut self,
        mesh: &BaseMesh,
        deps: &DepGraph,
        values: &[(TileKey, D::Value)],
    ) -> Result<(), EvalError> {
        let mut stale = if self.edited.is_empty() {
            BTreeSet::new()
        } else {
            D::readers(mesh, deps, &self.edited, &self.levels)?
        };
        self.edited.clear();
        for (key, value) in values {
            if stale.remove(key) {
                self.set(*key, Some(value.clone()));
            } else {
                self.record(*key, value.clone());
            }
        }
        for key in stale {
            self.set(key, None);
        }
        Ok(())
    }

    /// [`Book::settle`] with the values the writes noted can change already listed: `stale`,
    /// [`Derived::readers`] of exactly `edited`. Does nothing and returns `false` unless
    /// `edited` is the writes noted since the last settle.
    pub(crate) fn settle_listed(
        &mut self,
        edited: &BTreeSet<TileKey>,
        stale: &BTreeSet<TileKey>,
    ) -> bool {
        if self.edited.is_empty() || self.edited != *edited {
            return false;
        }
        self.edited.clear();
        for &key in stale {
            self.set(key, None);
        }
        true
    }

    /// Makes unknown every value the writes noted since the last call can change (see
    /// [`readers`], which reads neighbours `deps` recorded).
    ///
    /// # Errors
    ///
    /// As [`readers`]; the writes stay noted then.
    pub(crate) fn settle(&mut self, mesh: &BaseMesh, deps: &DepGraph) -> Result<(), EvalError> {
        if self.edited.is_empty() {
            return Ok(());
        }
        let stale = D::readers(mesh, deps, &self.edited, &self.levels)?;
        self.edited.clear();
        for key in stale {
            self.set(key, None);
        }
        Ok(())
    }

    /// Makes unknown every value a write of the `lost` tiles could have changed: their latest
    /// edits are in no file, so values recorded since describe a state no save point holds. Done
    /// after [`Book::settle`]. Returns the values made unknown.
    ///
    /// # Errors
    ///
    /// As [`readers`].
    pub(crate) fn forget(
        &mut self,
        mesh: &BaseMesh,
        deps: &DepGraph,
        lost: impl IntoIterator<Item = TileKey>,
    ) -> Result<Vec<TileKey>, EvalError> {
        let lost: BTreeSet<TileKey> = lost.into_iter().collect();
        let stale: Vec<TileKey> = D::readers(mesh, deps, &lost, &self.levels)?
            .into_iter()
            .collect();
        for &key in &stale {
            self.set(key, None);
        }
        Ok(stale)
    }

    /// Sets `key`'s value to `value` (`None`: unknown), to be flushed, and tracks its level.
    pub(crate) fn set(&mut self, key: TileKey, value: Option<D::Value>) {
        self.levels.insert(key.level());
        self.put(key, value.map_or(Kept::Unknown, Kept::Known));
        self.unflushed.insert(key);
        self.derived.remove(&key);
        self.changes += 1;
    }

    /// Records `value`, computed from the store's current state, as `key`'s: flushed if an
    /// edit's change to it is waiting to be (it replaces that change, which the working file
    /// must not miss) or the kind [spills](Derived::SPILLS), else derived, kept for the next save
    /// only (see the module notes).
    pub(crate) fn record(&mut self, key: TileKey, value: D::Value) {
        if D::SPILLS || self.unflushed.contains(&key) {
            self.set(key, Some(value));
            return;
        }
        self.levels.insert(key.level());
        self.put(key, Kept::Known(value));
        self.derived.insert(key);
        self.changes += 1;
    }

    /// Makes `kept` `key`'s entry in [`Book::changed`], keeping [`Book::waiting`] and its bytes
    /// in step.
    fn put(&mut self, key: TileKey, kept: Kept<D::Value>) {
        let known = match &kept {
            Kept::Known(value) if D::SPILLS => Some(D::bytes(value)),
            Kept::Known(_) | Kept::Unknown | Kept::Written => None,
        };
        if let Some(Kept::Known(old)) = self.changed.insert(key, kept)
            && self.waiting.remove(&key)
        {
            self.waiting_bytes = self.waiting_bytes.saturating_sub(D::bytes(&old));
        }
        if let Some(bytes) = known {
            self.waiting.insert(key);
            self.waiting_bytes = self.waiting_bytes.saturating_add(bytes);
        }
    }

    /// `key`'s value if it changed since the last rebase; `None` if the files hold whatever is
    /// current.
    pub(crate) fn kept(&self, key: &TileKey) -> Option<&Kept<D::Value>> {
        self.changed.get(key)
    }

    /// Every value changed since the last rebase, in key order: what a save copies.
    pub(crate) fn all_kept(&self) -> impl Iterator<Item = (&TileKey, &Kept<D::Value>)> {
        self.changed.iter()
    }

    /// The values the working file hasn't received, in key order.
    pub(crate) fn unflushed_kept(&self) -> impl Iterator<Item = (&TileKey, &Kept<D::Value>)> {
        self.unflushed
            .iter()
            .filter_map(|key| Some((key, self.changed.get(key)?)))
    }

    /// The values a save of the working file itself writes, in key order: those a flush would,
    /// and the derived ones.
    pub(crate) fn unsaved_kept(&self) -> impl Iterator<Item = (&TileKey, &Kept<D::Value>)> {
        self.unflushed
            .union(&self.derived)
            .filter_map(|key| Some((key, self.changed.get(key)?)))
    }

    /// Whether `key`'s latest change is waiting for a flush.
    pub(crate) fn is_unflushed(&self, key: &TileKey) -> bool {
        self.unflushed.contains(key)
    }

    /// Whether the working file lacks a value's latest change made by an edit (or, for a kind
    /// that spills, any recorded value).
    pub(crate) fn has_unflushed(&self) -> bool {
        !self.unflushed.is_empty()
    }

    /// The working file has received `key`'s latest change.
    pub(crate) fn flushed(&mut self, key: &TileKey) {
        self.unflushed.remove(key);
        self.spill(key);
    }

    /// The working file has received `key`'s latest change in a save of its own: nothing of it
    /// is waiting, derived or not.
    pub(crate) fn saved(&mut self, key: &TileKey) {
        self.unflushed.remove(key);
        self.derived.remove(key);
        self.spill(key);
    }

    /// For a kind that [spills](Derived::SPILLS), drops `key`'s known value from memory once the
    /// working file holds it: reads find the file's record instead.
    fn spill(&mut self, key: &TileKey) {
        if D::SPILLS && matches!(self.changed.get(key), Some(Kept::Known(_))) {
            self.put(*key, Kept::Written);
        }
    }

    /// The values held in memory waiting for the working file, in key order: for a kind that
    /// [spills](Derived::SPILLS), every known value of [`Book::changed`]; none for any other.
    pub(crate) fn waiting(&self) -> impl Iterator<Item = (&TileKey, &D::Value)> {
        self.waiting
            .iter()
            .filter_map(|key| Some((key, self.changed.get(key)?.known()?)))
    }

    /// The bytes the values of [`Book::waiting`] hold ([`Derived::bytes`]).
    pub(crate) const fn waiting_bytes(&self) -> usize {
        self.waiting_bytes
    }

    /// A new working file holds every value a flush would write, as it is now (a roll-over
    /// filled it). Derived values stay derived: it doesn't hold them.
    pub(crate) fn all_flushed(&mut self) {
        let keys: Vec<TileKey> = self.unflushed.iter().copied().collect();
        self.unflushed.clear();
        for key in &keys {
            self.spill(key);
        }
    }

    /// The project file's new save point holds every changed value, and the working file is a
    /// new, empty one over it: nothing is changed since.
    pub(crate) fn rebased(&mut self) {
        self.changed.clear();
        self.unflushed.clear();
        self.derived.clear();
        self.waiting.clear();
        self.waiting_bytes = 0;
    }

    /// The changes counted so far, for telling whether a save is still current.
    pub(crate) const fn changes(&self) -> u64 {
        self.changes
    }

    /// The levels whose values the files, or this session, hold.
    pub(crate) const fn levels(&self) -> &BTreeSet<Level> {
        &self.levels
    }

    /// Every tile whose value is unknown since the last rebase, in key order: what a refresh
    /// brings up to date.
    pub(crate) fn unknown_keys(&self) -> Vec<TileKey> {
        self.changed
            .iter()
            .filter(|(_, kept)| matches!(kept, Kept::Unknown))
            .map(|(key, _)| *key)
            .collect()
    }

    /// Whether writes are noted and not yet settled.
    #[cfg(test)]
    pub(crate) fn has_unsettled(&self) -> bool {
        !self.edited.is_empty()
    }
}

impl Book<Bounds> {
    /// A book over files holding bounds of `levels`, with `changed` the bounds the working file
    /// holds that no save has copied into the project file (those of a restored recovery file).
    pub(crate) fn new(
        levels: BTreeSet<Level>,
        changed: BTreeMap<TileKey, Option<TileBounds>>,
    ) -> Self {
        let kept = changed
            .into_iter()
            .map(|(key, bounds)| (key, bounds.map_or(Kept::Unknown, Kept::Known)))
            .collect();
        Self::with(levels, kept)
    }

    /// `key`'s bound if it changed since the last rebase; `None` if the files hold whatever is
    /// current.
    pub(crate) fn changed(&self, key: &TileKey) -> Option<StoredBound> {
        self.kept(key)
            .map(|kept| StoredBound::from(kept.known().copied()))
    }

    /// Every bound changed since the last rebase, in key order, `None` if unknown: what a save
    /// copies.
    pub(crate) fn all_changed(&self) -> impl Iterator<Item = (&TileKey, Option<&TileBounds>)> {
        self.all_kept().map(|(key, kept)| (key, kept.known()))
    }

    /// The bounds the working file hasn't received, with their values, in key order.
    pub(crate) fn unflushed(&self) -> impl Iterator<Item = (&TileKey, Option<&TileBounds>)> {
        self.unflushed_kept().map(|(key, kept)| (key, kept.known()))
    }

    /// The bounds a save of the working file itself writes, with their values, in key order:
    /// those a flush would, and the derived ones.
    pub(crate) fn unsaved(&self) -> impl Iterator<Item = (&TileKey, Option<&TileBounds>)> {
        self.unsaved_kept().map(|(key, kept)| (key, kept.known()))
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;
