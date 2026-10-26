//! Display tiles kept with the tiles, so the renderer's coarse levels open with the project
//! instead of being rebuilt from every finest tile.
//!
//! A display tile (`geom::DisplayTile`) is derived from the finest level: its positions are the
//! finest ones subsampled, its normals a mip of the finest normals, its bound `e` chained up from
//! the finest level. Rebuilding the layer means evaluating every finest tile, so the files keep
//! it, under the same rule as the spatial index's bounds (see [`crate::bounds`]): **a save point
//! never names a display tile its tiles don't give**. A write makes every display tile it can
//! change unknown ([`readers`]); an unknown or damaged display tile is rebuilt, never trusted.
//!
//! # In the file
//!
//! A display tile is a tile record of the base layer's [`Channel::DISPLAY`] channel, which the
//! format has always allowed (a channel a reader doesn't know round-trips): no new record kind,
//! and nothing a save point names changes. Its header is `version: u8, state: u8`, then, if the
//! state is known, the bound `e: f32`; its body, compressed as every tile body is, is the tile's
//! encoding (`geom::DisplayTile::encoded`), or empty if unknown. The record's op is
//! [`DISPLAY_OP`]: a display tile is no edit.
//!
//! # Unlike bounds
//!
//! A display tile is tens of kilobytes, a bound forty bytes. So a recorded display tile always
//! goes to the working file with the next flush, never waiting in memory for the next save as a
//! derived bound does, and once there the store drops its copy and reads the file (see
//! [`crate::bounds::Kept::Written`]): a whole layer rebuilt in a session stays on disk.

use std::collections::BTreeSet;

use geom::{BaseMesh, Channel, DisplayTile, EvalError, FaceId, LayerId, Level, Region, TileKey};

use crate::bounds::{self, Derived};
use crate::deps::DepGraph;
use crate::op::OpId;

/// The version byte of the display record format this build writes and reads.
const VERSION: u8 = 1;
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

/// A display tile as the store keeps it: its bound and its encoding, as the file holds them.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DisplayRecord {
    /// `e`.
    pub(crate) error: f32,
    /// The tile's encoding (`geom::DisplayTile::encoded`).
    pub(crate) encoded: Vec<u8>,
}

impl DisplayRecord {
    /// The record of a built tile.
    pub(crate) fn of(tile: &DisplayTile) -> Self {
        Self {
            error: tile.error(),
            encoded: tile.encoded().to_vec(),
        }
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

/// A display record's header: the version, the state, and the bound if known.
pub(crate) fn display_header(record: Option<&DisplayRecord>) -> Vec<u8> {
    match record {
        Some(record) => {
            let mut out = vec![VERSION, KNOWN];
            out.extend_from_slice(&record.error.to_le_bytes());
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
/// `None` if it isn't one this build reads (another version, a state other than 0 or 1, a bound
/// that isn't finite and non-negative, a body longer than [`max_body`] or one that doesn't
/// decode as `key`'s tile). Such a record counts as unknown: the tile is rebuilt.
pub(crate) fn parse_display(key: &TileKey, header: &[u8], body: &[u8]) -> Option<StoredDisplay> {
    match *header {
        [VERSION, UNKNOWN] => body.is_empty().then_some(StoredDisplay::Unknown),
        [VERSION, KNOWN, a, b, c, d] => {
            let error = f32::from_le_bytes([a, b, c, d]);
            if body.len() > max_body(key) {
                return None;
            }
            // Decoding checks every byte and the bound; only a tile that decodes is trusted.
            let tile = DisplayTile::decode(*key, error, body).ok()?;
            Some(StoredDisplay::Known(DisplayRecord::of(&tile)))
        }
        _ => None,
    }
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
    let mut answer = BTreeSet::new();
    let Some(finest) = levels.last().and_then(|l| l.child()) else {
        return Ok(answer);
    };
    let moved = bounds::readers(mesh, deps, edited, &BTreeSet::from([finest]))?;
    for key in moved {
        let region = Region::of_tile(&key);
        let face = key.face();
        for &level in levels {
            let shift = u32::from(finest.get() - level.get());
            let n = level.face_quads();
            let lo = |v: u32| (v >> shift).saturating_sub(1);
            let hi = |v: u32| ((v + region.quads()).div_ceil(1 << shift) + 1).min(n);
            let xs = (lo(region.x0()), hi(region.x0()));
            let ys = (lo(region.y0()), hi(region.y0()));
            tiles_over(&mut answer, face, level, xs, ys);
            let on_border = |i: u32, j: u32| i == 0 || j == 0 || i == n || j == n;
            for j in ys.0..=ys.1 {
                for i in (xs.0..=xs.1).filter(|&i| on_border(i, j)) {
                    for copy in &mesh.topology().copies(face, level, i, j) {
                        let tile = copy.key_in(LayerId::BASE, Channel::DISPLAY, level);
                        let at = Region::of_tile(&tile);
                        let (ci, cj) = (at.x0() + copy.i, at.y0() + copy.j);
                        let near = |c: u32| (c.saturating_sub(1), (c + 1).min(n));
                        tiles_over(&mut answer, copy.face, level, near(ci), near(cj));
                    }
                }
            }
        }
    }
    Ok(answer)
}

/// The display layer as the [`bounds::Book`] keeps it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Display;

impl Derived for Display {
    type Value = DisplayRecord;

    const SPILLS: bool = true;

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
