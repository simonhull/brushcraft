//! Where a tile's vertices lie on the faces' grids: what tells which previews hold a vertex, so
//! previews of different strokes meet at their seams bit for bit (the parent module's "Seams").
//!
//! A vertex on a tile's border is held by every tile around it, and on a face's edge by tiles of
//! two faces, each in its own grid; the evaluation reads every vertex from its owner copy. So a
//! gather of a tile's region from a source that answers each vertex with its own place
//! ([`Places`]) tells, slot by slot, which vertex of which face the tile holds there: its
//! **place** ([`Place`]), the same in every tile holding it.

use geom::{FaceId, Level, PositionSource, Region, TileKey, Vec3, gather_block};
use store::TileStore;

/// A vertex of a face's grid at a level: its face's index, its column and its row.
pub(super) type Place = (u32, u32, u32);

/// A source answering each vertex with its own place, as exact small integers.
struct Places;

/// `n` as an exact `f32`, if it fits in 16 bits (every place of a level a tile can be at does).
fn exact(n: u32) -> Option<f32> {
    u16::try_from(n).ok().map(f32::from)
}

impl PositionSource for Places {
    fn position(&self, face: FaceId, _level: Level, i: u32, j: u32) -> Option<Vec3> {
        Some(Vec3::new(exact(face.index())?, exact(i)?, exact(j)?))
    }
}

/// The place written by [`Places`] as `point`; `None` for a slot with no vertex (NaN).
fn place_of(point: Vec3) -> Option<Place> {
    let whole = |c: f32| {
        if !(0.0..=65_535.0).contains(&c) || c.fract() != 0.0 {
            return None;
        }
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a whole number in 0..=65535, written exactly by `Places`"
        )]
        let n = c as u16;
        Some(u32::from(n))
    };
    Some((whole(point.x)?, whole(point.y)?, whole(point.z)?))
}

/// The places of display tile `key`'s border vertices, each with its index in the tile.
///
/// Row-major: those it shares with the tiles around it, at its level, its region gathered at its
/// own level. Empty if the region can't be gathered.
pub(super) fn border_places(store: &TileStore, key: &TileKey) -> Vec<(usize, Place)> {
    let Ok(block) = gather_block(store.mesh(), &Places, Region::of_tile(key)) else {
        return Vec::new();
    };
    let side = key.side();
    let last = side.saturating_sub(1);
    let grid = (0..side).flat_map(|j| (0..side).map(move |i| (i, j)));
    let border = grid.filter(|&(i, j)| i == 0 || j == 0 || i == last || j == last);
    let places = border.filter_map(|(i, j)| {
        let v = usize::try_from(j * side + i).ok()?;
        Some((v, place_of(block.get(i64::from(i), i64::from(j)))?))
    });
    places.collect()
}
