//! Where a tile's vertices lie on the faces' grids: what tells which previews hold a vertex, so
//! previews of different strokes meet at their seams bit for bit (the parent module's "Seams").
//!
//! A vertex on a tile's border is held by every tile around it, and on a face's edge by tiles of
//! two faces, each in its own grid; the evaluation reads every vertex from its owner copy. So a
//! gather of a tile's region from a source that answers each vertex with its own place
//! ([`Places`]) tells, slot by slot, which vertex of which face the tile holds there: its
//! **place** ([`Place`]), the same in every tile holding it.

use geom::{FaceId, GatheredBlock, Level, PositionSource, Region, TileKey, Vec3, gather_block};
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
pub(super) fn place_of(point: Vec3) -> Option<Place> {
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

/// Where finest tile `key`'s preview may meet others: its block of places, and the slots of the
/// points along its border it may share, with their places.
///
/// The block is gathered over the tile's region as its preview is, each point its place
/// ([`place_of`]); the slots are numbered as [`points`] numbers them.
#[derive(Debug)]
pub(super) struct Shared {
    /// The block of places: what a preview's points are matched to, slot by slot.
    pub(super) block: GatheredBlock,
    /// The points a tile around it may hold too: the block's three rows and columns along each
    /// side, its ring and the two lines inside it (its ring holds the next tile's line beside
    /// the border, whose ring holds its own), and its base-vertex rings' points. The others are
    /// its own alone.
    pub(super) slots: Vec<(usize, Place)>,
}

impl Shared {
    /// Finest tile `key`'s; `None` if its region can't be gathered.
    pub(super) fn of(store: &TileStore, key: &TileKey) -> Option<Self> {
        let block = gather_block(store.mesh(), &Places, Region::of_tile(key)).ok()?;
        let side = usize::try_from(block.side()).ok()?;
        let padded = side * side;
        // Padded lines 0 ..= side - 1 are the block's lines -1 ..= quads + 1.
        let near = |line: usize| line <= 2 || line + 3 >= side;
        let numbered = points(&block).into_iter().enumerate();
        let slots = numbered.filter_map(|(slot, point)| {
            let shared = slot >= padded || near(slot % side) || near(slot / side);
            if shared {
                Some((slot, place_of(point)?))
            } else {
                None
            }
        });
        let slots = slots.collect();
        Some(Self { block, slots })
    }
}

/// Every point `block` holds, in order: its positions with the ring, then each corner ring's
/// edges and diagonals. The same slots in every block gathered over the same region.
pub(super) fn points(block: &GatheredBlock) -> Vec<Vec3> {
    let rings = (0..4).filter_map(|c| block.corner(c));
    let ring_points = rings.flat_map(|ring| ring.edges().iter().chain(ring.diagonals()));
    let all = block.padded().iter().chain(ring_points);
    all.copied().collect()
}

/// Point `slot` of `block`, as [`points`] numbers them; `None` past them.
pub(super) fn point_at(block: &GatheredBlock, slot: usize) -> Option<Vec3> {
    let padded = block.padded();
    if let Some(&point) = padded.get(slot) {
        return Some(point);
    }
    let mut left = slot - padded.len();
    for corner in 0..4 {
        let Some(ring) = block.corner(corner) else {
            continue;
        };
        for part in [ring.edges(), ring.diagonals()] {
            if let Some(&point) = part.get(left) {
                return Some(point);
            }
            left -= part.len();
        }
    }
    None
}
