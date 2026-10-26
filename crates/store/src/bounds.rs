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
use crate::index::TileBounds;
use crate::record::{HEAD, TAIL};

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
pub(crate) fn parse_bounds_payload(payload: &[u8]) -> Option<(TileKey, Kept)> {
    let mut r = Reader::new(payload);
    let key = r.key()?;
    let kept = match r.u8()? {
        UNKNOWN => Kept::Unknown,
        KNOWN => {
            let mut bits = [0; TileBounds::BITS];
            for value in &mut bits {
                *value = r.u32()?;
            }
            Kept::Known(TileBounds::from_bits(bits)?)
        }
        _ => return None,
    };
    r.rest().is_empty().then_some((key, kept))
}

/// A tile's bound as the store keeps it: its bounds, or the fact that they aren't known (the
/// tile is evaluated when the index is loaded).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Kept {
    /// The tile's bounds, as its tiles give them.
    Known(TileBounds),
    /// Not known: the tile is evaluated.
    Unknown,
}

impl Kept {
    /// The bounds, if known.
    pub(crate) const fn bounds(self) -> Option<TileBounds> {
        match self {
            Self::Known(bounds) => Some(bounds),
            Self::Unknown => None,
        }
    }
}

impl From<Option<TileBounds>> for Kept {
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

/// What the store knows of the bounds it keeps in its files (see the module notes).
#[derive(Debug, Default)]
pub(crate) struct Book {
    /// The levels whose bounds any file of the store, or this session, holds: writes are settled
    /// against these. Empty while no bound is kept anywhere, and then writes are not noted.
    levels: BTreeSet<Level>,
    /// Correction tiles written since the last settle, of levels at most the finest tracked.
    edited: BTreeSet<TileKey>,
    /// Every bound changed since the store opened or was last rebased: its value now, `None` if
    /// unknown. What a save copies into the project file, and what reads find first.
    changed: BTreeMap<TileKey, Option<TileBounds>>,
    /// The bounds of [`Book::changed`] edits changed that the working file hasn't received since
    /// (see the module notes): what the next flush writes.
    unflushed: BTreeSet<TileKey>,
    /// The bounds of [`Book::changed`] whose latest change is derived (see the module notes): no
    /// flush writes them; a save of the project file, or of the working file itself, does.
    derived: BTreeSet<TileKey>,
    /// Counts the changes made to [`Book::changed`]: a save or a roll-over made before a later
    /// one doesn't hold it.
    changes: u64,
}

impl Book {
    /// A book over files holding bounds of `levels`, with `changed` the bounds the working file
    /// holds that no save has copied into the project file (those of a restored recovery file).
    pub(crate) fn new(
        levels: BTreeSet<Level>,
        changed: BTreeMap<TileKey, Option<TileBounds>>,
    ) -> Self {
        Self {
            levels,
            changed,
            ..Self::default()
        }
    }

    /// Notes a write of `key`'s corrections, if it can change a tracked level's bounds: the next
    /// [`Book::settle`] makes those unknown.
    pub(crate) fn note_write(&mut self, key: TileKey) {
        let reaches = self
            .levels
            .last()
            .is_some_and(|&finest| key.level() <= finest && is_evaluated(&key));
        if reaches {
            self.edited.insert(key);
        }
    }

    /// [`Book::settle`], then [`Book::record`] of each of `bounds`, with the same result; a bound
    /// the settle would make unknown and `bounds` records is set once, to its recorded value,
    /// to be flushed (as `record` over an unknown left by the settle does).
    ///
    /// # Errors
    ///
    /// As [`Book::settle`]; nothing is recorded then.
    pub(crate) fn settle_and_record(
        &mut self,
        mesh: &BaseMesh,
        deps: &DepGraph,
        bounds: &[(TileKey, TileBounds)],
    ) -> Result<(), EvalError> {
        let mut stale = if self.edited.is_empty() {
            BTreeSet::new()
        } else {
            readers(mesh, deps, &self.edited, &self.levels)?
        };
        self.edited.clear();
        for &(key, tile) in bounds {
            if stale.remove(&key) {
                self.set(key, Some(tile));
            } else {
                self.record(key, tile);
            }
        }
        for key in stale {
            self.set(key, None);
        }
        Ok(())
    }

    /// Makes unknown every bound the writes noted since the last call can change (see
    /// [`readers`], which reads neighbours `deps` recorded).
    ///
    /// # Errors
    ///
    /// As [`readers`]; the writes stay noted then.
    pub(crate) fn settle(&mut self, mesh: &BaseMesh, deps: &DepGraph) -> Result<(), EvalError> {
        if self.edited.is_empty() {
            return Ok(());
        }
        let stale = readers(mesh, deps, &self.edited, &self.levels)?;
        self.edited.clear();
        for key in stale {
            self.set(key, None);
        }
        Ok(())
    }

    /// Makes unknown every bound a write of the `lost` tiles could have changed: their latest
    /// edits are in no file, so bounds recorded since describe a state no save point holds. Done
    /// after [`Book::settle`]. Returns the bounds made unknown.
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
        let stale: Vec<TileKey> = readers(mesh, deps, &lost, &self.levels)?
            .into_iter()
            .collect();
        for &key in &stale {
            self.set(key, None);
        }
        Ok(stale)
    }

    /// Sets `key`'s bound to `bounds` (`None`: unknown), to be flushed, and tracks its level.
    pub(crate) fn set(&mut self, key: TileKey, bounds: Option<TileBounds>) {
        self.levels.insert(key.level());
        self.changed.insert(key, bounds);
        self.unflushed.insert(key);
        self.derived.remove(&key);
        self.changes += 1;
    }

    /// Records `bounds`, computed from the store's current state, as `key`'s bound: flushed if
    /// an edit's change to it is waiting to be (it replaces that change, which the working file
    /// must not miss), else derived, kept for the next save only (see the module notes).
    pub(crate) fn record(&mut self, key: TileKey, bounds: TileBounds) {
        if self.unflushed.contains(&key) {
            self.set(key, Some(bounds));
            return;
        }
        self.levels.insert(key.level());
        self.changed.insert(key, Some(bounds));
        self.derived.insert(key);
        self.changes += 1;
    }

    /// `key`'s bound if it changed since the last rebase; `None` if the files hold whatever is
    /// current.
    pub(crate) fn changed(&self, key: &TileKey) -> Option<Kept> {
        self.changed.get(key).map(|&bounds| Kept::from(bounds))
    }

    /// Every bound changed since the last rebase, in key order: what a save copies.
    pub(crate) fn all_changed(&self) -> impl Iterator<Item = (&TileKey, Option<&TileBounds>)> {
        self.changed
            .iter()
            .map(|(key, bounds)| (key, bounds.as_ref()))
    }

    /// The bounds the working file hasn't received, with their values, in key order.
    pub(crate) fn unflushed(&self) -> impl Iterator<Item = (&TileKey, Option<&TileBounds>)> {
        self.unflushed
            .iter()
            .filter_map(|key| Some((key, self.changed.get(key)?.as_ref())))
    }

    /// The bounds a save of the working file itself writes, with their values, in key order:
    /// those a flush would, and the derived ones.
    pub(crate) fn unsaved(&self) -> impl Iterator<Item = (&TileKey, Option<&TileBounds>)> {
        self.unflushed
            .union(&self.derived)
            .filter_map(|key| Some((key, self.changed.get(key)?.as_ref())))
    }

    /// Whether `key`'s latest change is waiting for a flush.
    pub(crate) fn is_unflushed(&self, key: &TileKey) -> bool {
        self.unflushed.contains(key)
    }

    /// Whether the working file lacks a bound's latest change made by an edit.
    pub(crate) fn has_unflushed(&self) -> bool {
        !self.unflushed.is_empty()
    }

    /// The working file has received `key`'s latest change.
    pub(crate) fn flushed(&mut self, key: &TileKey) {
        self.unflushed.remove(key);
    }

    /// The working file has received `key`'s latest change in a save of its own: nothing of it
    /// is waiting, derived or not.
    pub(crate) fn saved(&mut self, key: &TileKey) {
        self.unflushed.remove(key);
        self.derived.remove(key);
    }

    /// A new working file holds every bound a flush would write, as it is now (a roll-over
    /// filled it). Derived bounds stay derived: it doesn't hold them.
    pub(crate) fn all_flushed(&mut self) {
        self.unflushed.clear();
    }

    /// The project file's new save point holds every changed bound, and the working file is a
    /// new, empty one over it: nothing is changed since.
    pub(crate) fn rebased(&mut self) {
        self.changed.clear();
        self.unflushed.clear();
        self.derived.clear();
    }

    /// The changes counted so far, for telling whether a save is still current.
    pub(crate) const fn changes(&self) -> u64 {
        self.changes
    }

    /// Whether writes are noted and not yet settled.
    #[cfg(test)]
    pub(crate) fn has_unsettled(&self) -> bool {
        !self.edited.is_empty()
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;
