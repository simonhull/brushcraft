//! Display tiles kept with the tiles, so the renderer's coarse levels open with the project
//! instead of being rebuilt from every finest tile.
//!
//! A display tile (`geom::DisplayTile`) is derived from the finest level: its positions are the
//! finest ones subsampled, its normals a mip of the finest normals, its bound `e` and its normal
//! cone chained up from the finest level. Rebuilding the layer means evaluating every finest tile,
//! so the files keep it, under the same rule as the spatial index's bounds (see [`crate::bounds`]):
//! **a save point never names a display tile its tiles don't give**. A write makes every display
//! tile it can change unknown ([`readers`]); an unknown or damaged display tile is rebuilt, never
//! trusted.
//!
//! # In the file
//!
//! A display tile is a tile record of the base layer's [`Channel::DISPLAY`] channel, which the
//! format has always allowed (a channel a reader doesn't know round-trips): no new record kind,
//! and nothing a save point names changes. Its header is `version: u8, state: u8`, then, if the
//! state is known, `finest: u8` (the level it was built from), the bound `e: f32` and the normal
//! cone `cone: u32` (`geom::FinestNormalCone::to_bits`), little-endian; its body, compressed as
//! every tile body is, is the tile's encoding (`geom::DisplayTile::encoded`), or empty if unknown.
//! The record's op is [`DISPLAY_OP`]: a display tile is no edit.
//!
//! # Unlike bounds
//!
//! A display tile is tens of kilobytes, a bound forty bytes. So a recorded display tile always
//! goes to the working file with the next flush, never waiting in memory for the next save as a
//! derived bound does, and once there the store drops its copy and reads the file (see
//! [`crate::bounds::Kept::Written`]): a whole layer rebuilt in a session stays on disk. Between
//! flushes, the records waiting for one are held to a share of the budget, the rest written
//! through to the working file before the flush (`TileStore::write_through_display`), so a
//! session that doesn't flush holds no more of them either.

use std::collections::BTreeSet;

use geom::{
    BaseMesh, Channel, DisplayTile, EvalError, FaceId, FinestNormalCone, LayerId, Level, Region,
    TileKey,
};

use crate::bounds::{self, Derived};
use crate::deps::DepGraph;
use crate::jobs::TileCache;
use crate::op::OpId;

/// The version byte of the display record format this build writes and reads.
///
/// 2 adds the finest level a tile was built from, 3 its normal cone. A record of an older
/// version reads as unknown, and is rebuilt.
const VERSION: u8 = 3;
/// The state byte of a display record whose tile is unknown: it is rebuilt.
const UNKNOWN: u8 = 0;
/// The state byte of a display record that holds its tile.
const KNOWN: u8 = 1;

/// The op every display record carries: site 0's first op. A display tile is no edit, and the
/// store notes no op for it, so it never moves a file's head.
pub(crate) const DISPLAY_OP: OpId = match OpId::new(0, 0) {
    Some(op) => op,
    None => unreachable!(),
};

/// A display tile as the store keeps it: its bound, its normal cone, the finest level it was
/// built from, and its encoding, as the file holds them.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DisplayRecord {
    /// `e`.
    pub(crate) error: f32,
    /// The cone holding every finest normal under the tile.
    pub(crate) cone: FinestNormalCone,
    /// The finest level the tile was built from: a store keeping a display layer under another
    /// level reads the record as unknown (its bound is no bound on that store's surface).
    pub(crate) finest: Level,
    /// The tile's encoding (`geom::DisplayTile::encoded`).
    pub(crate) encoded: Vec<u8>,
}

impl DisplayRecord {
    /// The record of a tile built from the surface at level `finest`.
    pub(crate) fn of(tile: &DisplayTile, finest: Level) -> Self {
        Self {
            error: tile.error(),
            cone: tile.cone(),
            finest,
            encoded: tile.encoded().to_vec(),
        }
    }

    /// The tile `key` the record holds, decoded with its bound and cone; `None` if it doesn't
    /// decode as `key`'s.
    pub(crate) fn decode(&self, key: &TileKey) -> Option<DisplayTile> {
        DisplayTile::decode(*key, self.error, self.cone, &self.encoded).ok()
    }
}

/// A display tile as a record holds it: the tile, or the fact that it isn't known (it is
/// rebuilt).
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum StoredDisplay {
    /// The tile's bound and encoding.
    Known(DisplayRecord),
    /// Not known: the tile is rebuilt.
    Unknown,
}

impl StoredDisplay {
    /// The record, if known.
    pub(crate) fn known(self) -> Option<DisplayRecord> {
        match self {
            Self::Known(record) => Some(record),
            Self::Unknown => None,
        }
    }
}

/// A display record's header: the version, the state, and, if known, the finest level, the
/// bound and the cone.
pub(crate) fn display_header(record: Option<&DisplayRecord>) -> Vec<u8> {
    match record {
        Some(record) => {
            let mut out = vec![VERSION, KNOWN, record.finest.get()];
            out.extend_from_slice(&record.error.to_le_bytes());
            out.extend_from_slice(&record.cone.to_bits().to_le_bytes());
            out
        }
        None => vec![VERSION, UNKNOWN],
    }
}

/// The body of `key`'s display record: its encoding, or nothing if unknown.
pub(crate) fn display_body(record: Option<&DisplayRecord>) -> &[u8] {
    record.map_or(&[], |r| r.encoded.as_slice())
}

/// The most bytes a display tile of `key`'s level can encode to: every vertex at its widest
/// (12 bytes of lossless positions, 4 of normals, 1 of roughness), and room for the headers.
pub(crate) fn max_body(key: &TileKey) -> usize {
    let side = usize::try_from(key.side()).unwrap_or(usize::MAX);
    side.saturating_mul(side)
        .saturating_mul(17)
        .saturating_add(256)
}

/// What a display record of `key` with `header` and `body` holds.
///
/// `None` if it isn't one this build reads (another version, a state other than 0 or 1, a
/// finest level that isn't a level above `key`'s, a bound that isn't finite and non-negative, a
/// body longer than [`max_body`] or one that doesn't decode as `key`'s tile). Such a record
/// counts as unknown: the tile is rebuilt.
pub(crate) fn parse_display(key: &TileKey, header: &[u8], body: &[u8]) -> Option<StoredDisplay> {
    match *header {
        [VERSION, UNKNOWN] => body.is_empty().then_some(StoredDisplay::Unknown),
        [VERSION, KNOWN, finest, e0, e1, e2, e3, c0, c1, c2, c3] => {
            let finest = Level::new(finest).ok().filter(|f| *f > key.level())?;
            let error = f32::from_le_bytes([e0, e1, e2, e3]);
            // Every `u32` is a cone.
            let cone = FinestNormalCone::from_bits(u32::from_le_bytes([c0, c1, c2, c3]));
            if body.len() > max_body(key) {
                return None;
            }
            // Decoding checks every byte and the bound; only a tile that decodes is trusted.
            let tile = DisplayTile::decode(*key, error, cone, body).ok()?;
            Some(StoredDisplay::Known(DisplayRecord::of(&tile, finest)))
        }
        _ => None,
    }
}

/// Whether a display record's `header` is of the version this build writes: one that isn't
/// (an older build's) reads as unknown however intact it is.
pub(crate) fn is_current_version(header: &[u8]) -> bool {
    header.first() == Some(&VERSION)
}

/// Whether `key` names a display tile: the base layer's display channel.
pub(crate) fn is_display(key: &TileKey) -> bool {
    key.layer() == LayerId::BASE && key.channel() == Channel::DISPLAY
}

/// Whether a write of `key`'s corrections can change any evaluated position: only base-layer
/// displacement tiles are evaluated.
fn is_evaluated(key: &TileKey) -> bool {
    key.layer() == LayerId::BASE && key.channel() == Channel::DISPLACEMENT
}

/// The display tiles of `face` at `level` holding any vertex of columns `xs` and rows `ys`
/// (both inclusive, on the face's level grid).
fn tiles_over(
    out: &mut BTreeSet<TileKey>,
    face: FaceId,
    level: Level,
    (x0, x1): (u32, u32),
    (y0, y1): (u32, u32),
) {
    let tq = level.tile_quads();
    let last = level.tiles_per_side() - 1;
    // Tile `t` holds vertices `t·tq ..= (t + 1)·tq`.
    let range = |lo: u32, hi: u32| (lo.div_ceil(tq).saturating_sub(1))..=(hi / tq).min(last);
    for ty in range(y0, y1) {
        for tx in range(x0, x1) {
            let (Ok(tx), Ok(ty)) = (u16::try_from(tx), u16::try_from(ty)) else {
                continue;
            };
            if let Ok(key) = TileKey::display(face, level, tx, ty) {
                out.insert(key);
            }
        }
    }
}

/// Every display tile of a level in `levels` (the display levels; the finest evaluated level is
/// the one after the last) that a write of the corrections of the `edited` tiles can change.
///
/// The writes move the positions and normals of the finest tiles [`bounds::readers`] gives for
/// the finest level. A display tile at level `l` reads the finest vertices under it (its
/// positions and bound) and, through its normals' tent, every finest vertex less than one
/// level-`l` quad from its own vertices, a level at a time, across face borders through owner
/// copies. So the answer at each level is, for each finest tile that moved, the display tiles of
/// its face holding a vertex within one quad of it, and, where that window reaches the face's
/// border, the display tiles of every face sharing a border vertex in it, within one quad of
/// that vertex's copy there.
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
    let Some(finest) = levels.last().and_then(|l| l.child()) else {
        return Ok(BTreeSet::new());
    };
    if edited.is_empty() {
        return Ok(BTreeSet::new());
    }
    readers_memo(mesh, deps, edited, finest, levels)
}

/// The display tiles a write of each tile can change ([`readers`] of it alone), and those
/// around each display tile ([`tiles_around`]), kept between calls.
///
/// A stroke's dabs write the same tiles step after step, and the answers depend on the mesh
/// and the display levels alone (the neighbours [`bounds::readers`] reads from the dependency
/// graph are the ones it would list).
///
/// Behind a mutex, as the dependency graph that holds it is shared with the workers; only the
/// owner thread asks, so it is never contended. At most its cap of answers of each kind
/// (`jobs::CacheCaps::reach`), the oldest dropped one at a time, so a full memo never makes a
/// stroke list everything again at once.
#[derive(Debug)]
pub(crate) struct ReachMemo(std::sync::Mutex<Reached>);

/// What a [`ReachMemo`] holds.
#[derive(Debug)]
struct Reached {
    /// The display levels the answers are for: other levels empty it.
    levels: BTreeSet<Level>,
    /// The display tiles a write of each tile can change.
    by: TileCache<std::sync::Arc<[TileKey]>>,
    /// The display tiles around each display tile.
    around: TileCache<std::sync::Arc<[TileKey]>>,
}

impl Reached {
    /// No answers, for `levels`, at most `cap` of each kind.
    fn new(levels: BTreeSet<Level>, cap: usize) -> Self {
        Self {
            levels,
            by: TileCache::new(cap),
            around: TileCache::new(cap),
        }
    }
}

impl Default for ReachMemo {
    fn default() -> Self {
        Self::new(Self::DEFAULT_CAP)
    }
}

impl ReachMemo {
    /// The answers of each kind a memo made without a store's budget keeps.
    const DEFAULT_CAP: usize = 256;

    /// An empty memo keeping at most `cap` answers of each kind.
    pub(crate) fn new(cap: usize) -> Self {
        Self(std::sync::Mutex::new(Reached::new(BTreeSet::new(), cap)))
    }

    /// [`tiles_around`] `key`, kept for the next ask.
    pub(crate) fn around(&self, mesh: &BaseMesh, key: &TileKey) -> std::sync::Arc<[TileKey]> {
        let Ok(mut memo) = self.0.lock() else {
            return tiles_around(mesh, key).into_iter().collect();
        };
        if let Some(found) = memo.around.tiles.get(key) {
            return std::sync::Arc::clone(found);
        }
        let found: std::sync::Arc<[TileKey]> = tiles_around(mesh, key).into_iter().collect();
        drop(memo.around.insert(*key, std::sync::Arc::clone(&found)));
        found
    }
}

impl PartialEq for ReachMemo {
    /// Always equal: what it holds follows from the mesh, so two graphs that differ only in it
    /// are the same graph.
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

/// [`readers`] of `edited`, the union of each tile's own (a write's readers are the readers of
/// each tile written), taken from the dependency graph's memo where it holds them.
///
/// # Errors
///
/// As [`readers`].
fn readers_memo(
    mesh: &BaseMesh,
    deps: &DepGraph,
    edited: &BTreeSet<TileKey>,
    finest: Level,
    levels: &BTreeSet<Level>,
) -> Result<BTreeSet<TileKey>, EvalError> {
    let of = |key: TileKey| -> Result<BTreeSet<TileKey>, EvalError> {
        let moved = bounds::readers(
            mesh,
            deps,
            &BTreeSet::from([key]),
            &BTreeSet::from([finest]),
        )?;
        Ok(reached_by(mesh, moved, finest, levels))
    };
    let Ok(mut memo) = deps.reach_memo().0.lock() else {
        let moved = bounds::readers(mesh, deps, edited, &BTreeSet::from([finest]))?;
        return Ok(reached_by(mesh, moved, finest, levels));
    };
    if memo.levels != *levels {
        let cap = memo.by.cap();
        *memo = Reached::new(levels.clone(), cap);
    }
    // The tiles' answers overlap a lot: gathered in a hash set, then sorted once (a set built
    // from sorted keys is built in one pass).
    let mut answer = crate::hash::TileSet::with_capacity_and_hasher(1024, crate::hash::TileHash);
    for &key in edited {
        if let Some(found) = memo.by.tiles.get(&key) {
            answer.extend(found.iter().copied());
            continue;
        }
        let found: std::sync::Arc<[TileKey]> = of(key)?.into_iter().collect();
        answer.extend(found.iter().copied());
        drop(memo.by.insert(key, found));
    }
    let mut sorted: Vec<TileKey> = answer.into_iter().collect();
    sorted.sort_unstable();
    Ok(sorted.into_iter().collect())
}

/// The windows of `levels` that finest tile `key` (of level `finest`) reaches: at each level,
/// the vertices less than one quad of it (see [`readers`]).
fn windows_of(
    key: &TileKey,
    finest: Level,
    levels: &BTreeSet<Level>,
) -> impl Iterator<Item = Window> {
    let region = Region::of_tile(key);
    let face = key.face();
    levels.iter().map(move |&level| {
        let shift = u32::from(finest.get() - level.get());
        let n = level.face_quads();
        let lo = |v: u32| (v >> shift).saturating_sub(1);
        let hi = |v: u32| ((v + region.quads()).div_ceil(1 << shift) + 1).min(n);
        (
            face,
            level,
            (lo(region.x0()), hi(region.x0())),
            (lo(region.y0()), hi(region.y0())),
        )
    })
}

/// The display tiles of `levels` reading a finest vertex (of level `finest`) of the `moved`
/// tiles, or one less than one quad of their level from them (see [`readers`]).
fn reached_by(
    mesh: &BaseMesh,
    moved: impl IntoIterator<Item = TileKey>,
    finest: Level,
    levels: &BTreeSet<Level>,
) -> BTreeSet<TileKey> {
    let mut answer = BTreeSet::new();
    // The windows already walked: at coarse levels, many moved tiles share one.
    let mut walked: BTreeSet<Window> = BTreeSet::new();
    for key in moved {
        for (face, level, xs, ys) in windows_of(&key, finest, levels) {
            if walked.insert((face, level, xs, ys)) {
                window_and_around(mesh, &mut answer, face, level, xs, ys);
            }
        }
    }
    answer
}

/// The tiles one level finer that display tile `key` reads (display keys).
///
/// Those its mean normals read and its positions are subsampled from: every vertex less than one
/// quad of level `key.level() + 1` from the tile's own, on its face and, across its borders, on
/// every face sharing a border vertex (each base vertex's whole ring included). `None` at the
/// deepest level (no finer one).
///
/// A superset of what [`geom::mip_normals_tile`] reads: the tent of each vertex reads its eight
/// neighbours one level finer, the ring tent of a base vertex its neighbours around it.
pub(crate) fn tent_reads(mesh: &BaseMesh, key: &TileKey) -> Option<BTreeSet<TileKey>> {
    let finer = key.level().child()?;
    let region = Region::of_tile(key);
    let n = finer.face_quads();
    let xs = (
        (2 * region.x0()).saturating_sub(1),
        (2 * (region.x0() + region.quads()) + 1).min(n),
    );
    let ys = (
        (2 * region.y0()).saturating_sub(1),
        (2 * (region.y0() + region.quads()) + 1).min(n),
    );
    let mut answer = BTreeSet::new();
    window_and_around(mesh, &mut answer, key.face(), finer, xs, ys);
    Some(answer)
}

/// The display tiles around display tile `key`, `key` among them.
///
/// Those of its level holding a vertex less than one tile from its own, on its face and,
/// across its borders, the tiles of the faces sharing a border vertex next to the copies of it
/// (see [`window_and_around`]).
pub(crate) fn tiles_around(mesh: &BaseMesh, key: &TileKey) -> BTreeSet<TileKey> {
    let region = Region::of_tile(key);
    let n = key.level().face_quads();
    let q = region.quads();
    let xs = (region.x0().saturating_sub(q), (region.x0() + 2 * q).min(n));
    let ys = (region.y0().saturating_sub(q), (region.y0() + 2 * q).min(n));
    let mut answer = BTreeSet::new();
    window_and_around(mesh, &mut answer, key.face(), key.level(), xs, ys);
    answer
}

/// Adds to `out` the display tiles of `level` holding a vertex of the window `xs` × `ys` of
/// `face`, and those within one quad of each copy, on other faces, of the window's border
/// vertices.
fn window_and_around(
    mesh: &BaseMesh,
    out: &mut BTreeSet<TileKey>,
    face: FaceId,
    level: Level,
    xs: (u32, u32),
    ys: (u32, u32),
) {
    let n = level.face_quads();
    tiles_over(out, face, level, xs, ys);
    // A run of border vertices along one edge has its copies in a run along the same edge of
    // each face sharing it, so the windows around the copies of its two ends bound the windows
    // around all of them.
    for (first, last) in border_runs(n, xs, ys) {
        let at = |(i, j): (u32, u32)| -> Vec<(FaceId, u32, u32)> {
            mesh.topology()
                .copies(face, level, i, j)
                .iter()
                .map(|copy| {
                    let tile = copy.key_in(LayerId::BASE, Channel::DISPLAY, level);
                    let held = Region::of_tile(&tile);
                    (copy.face, held.x0() + copy.i, held.y0() + copy.j)
                })
                .collect()
        };
        let (starts, ends) = (at(first), at(last));
        let alone = starts
            .iter()
            .enumerate()
            .all(|(k, c)| starts.iter().skip(k + 1).all(|d| d.0 != c.0));
        if !alone {
            // A face meeting this edge twice (never on a mesh whose faces meet themselves
            // nowhere): its copies can't be paired by face, so walk them all.
            walk_run(mesh, out, face, level, n, first, last);
            continue;
        }
        for &(f, ci, cj) in &starts {
            let (ei, ej) = ends
                .iter()
                .find(|end| end.0 == f)
                .map_or((ci, cj), |&(_, ei, ej)| (ei, ej));
            let around = |a: u32, b: u32| (a.min(b).saturating_sub(1), (a.max(b) + 1).min(n));
            tiles_over(out, f, level, around(ci, ei), around(cj, ej));
        }
    }
}

/// Adds to `out` the display tiles of `level` within one quad of each copy of every border
/// vertex of `face` from `first` to `last` (a run along one edge, `n` quads a side), walked one
/// vertex at a time.
fn walk_run(
    mesh: &BaseMesh,
    out: &mut BTreeSet<TileKey>,
    face: FaceId,
    level: Level,
    n: u32,
    first: (u32, u32),
    last: (u32, u32),
) {
    for j in first.1..=last.1 {
        for i in first.0..=last.0 {
            for copy in &mesh.topology().copies(face, level, i, j) {
                let tile = copy.key_in(LayerId::BASE, Channel::DISPLAY, level);
                let at = Region::of_tile(&tile);
                let (ci, cj) = (at.x0() + copy.i, at.y0() + copy.j);
                let near = |c: u32| (c.saturating_sub(1), (c + 1).min(n));
                tiles_over(out, copy.face, level, near(ci), near(cj));
            }
        }
    }
}

/// A window of a face's vertex grid at a level: the face, the level, and its inclusive column
/// and row ranges.
type Window = (FaceId, Level, (u32, u32), (u32, u32));

/// The border vertices of the window `xs` × `ys` (inclusive) of a face grid `n` quads a side.
///
/// They come as runs `(first, last)`: each face corner in the window alone, and the rest of each
/// edge in the window as one run between its two ends (corners left out, so every vertex of a
/// run has its copies on the same faces).
fn border_runs(n: u32, xs: (u32, u32), ys: (u32, u32)) -> Vec<((u32, u32), (u32, u32))> {
    let within = |v: u32, (lo, hi): (u32, u32)| (lo..=hi).contains(&v);
    let mut runs = Vec::new();
    for j in [0, n].into_iter().filter(|&j| within(j, ys)) {
        for i in [0, n].into_iter().filter(|&i| within(i, xs)) {
            runs.push(((i, j), (i, j)));
        }
    }
    // The edge's own vertices in `range`, its corners left out.
    let inner = |(lo, hi): (u32, u32)| (lo.max(1), hi.min(n.saturating_sub(1)));
    for j in [0, n].into_iter().filter(|&j| within(j, ys)) {
        let (a, b) = inner(xs);
        if a <= b {
            runs.push(((a, j), (b, j)));
        }
    }
    for i in [0, n].into_iter().filter(|&i| within(i, xs)) {
        let (a, b) = inner(ys);
        if a <= b {
            runs.push(((i, a), (i, b)));
        }
    }
    runs
}

/// The display layer as the [`bounds::Book`] keeps it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Display;

impl Derived for Display {
    type Value = DisplayRecord;

    const SPILLS: bool = true;

    fn bytes(value: &DisplayRecord) -> usize {
        size_of::<DisplayRecord>().saturating_add(value.encoded.capacity())
    }

    fn reaches(levels: &BTreeSet<Level>, key: &TileKey) -> bool {
        levels
            .last()
            .and_then(|l| l.child())
            .is_some_and(|finest| key.level() <= finest && is_evaluated(key))
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

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;
