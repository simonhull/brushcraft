//! Identifiers and tile arithmetic: faces, levels, layers, channels, tile keys and regions.

use core::fmt;

use thiserror::Error;

/// Quads per side of a full tile.
pub const TILE_QUADS: u32 = 64;
/// The finest level a project can hold.
pub const MAX_LEVEL: u8 = 12;
/// Levels up to this one keep a face's whole grid in one tile.
const WHOLE_FACE_LEVEL: u8 = 6;

/// A base-mesh face, by its index in the face list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FaceId(u32);

impl FaceId {
    /// The face at `index` in the base mesh's face list.
    #[must_use]
    pub const fn new(index: u32) -> Self {
        Self(index)
    }

    /// Its index in the base mesh's face list.
    #[must_use]
    pub const fn index(self) -> u32 {
        self.0
    }
}

/// A subdivision level, `0..=MAX_LEVEL`. Level 0 is the base mesh; each level halves the quads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Level(u8);

/// A level above [`MAX_LEVEL`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("level {0} is above the maximum of {MAX_LEVEL}")]
pub struct LevelOutOfRange(pub u8);

impl Level {
    /// The base mesh.
    pub const ZERO: Self = Self(0);

    /// # Errors
    ///
    /// [`LevelOutOfRange`] above [`MAX_LEVEL`].
    pub const fn new(level: u8) -> Result<Self, LevelOutOfRange> {
        if level <= MAX_LEVEL {
            Ok(Self(level))
        } else {
            Err(LevelOutOfRange(level))
        }
    }

    /// The level number.
    #[must_use]
    pub const fn get(self) -> u8 {
        self.0
    }

    /// Quads per side of a face's grid: `2^level`.
    #[must_use]
    pub const fn face_quads(self) -> u32 {
        1 << self.0
    }

    /// Quads per side of one tile.
    #[must_use]
    pub const fn tile_quads(self) -> u32 {
        if self.0 <= WHOLE_FACE_LEVEL {
            self.face_quads()
        } else {
            TILE_QUADS
        }
    }

    /// Tiles per side of a face.
    #[must_use]
    pub const fn tiles_per_side(self) -> u32 {
        self.face_quads() / self.tile_quads()
    }

    /// The next coarser level, or `None` at level 0.
    #[must_use]
    pub const fn parent(self) -> Option<Self> {
        if self.0 == 0 {
            None
        } else {
            Some(Self(self.0 - 1))
        }
    }

    /// The next finer level, or `None` at [`MAX_LEVEL`].
    #[must_use]
    pub const fn child(self) -> Option<Self> {
        if self.0 < MAX_LEVEL {
            Some(Self(self.0 + 1))
        } else {
            None
        }
    }
}

impl TryFrom<u8> for Level {
    type Error = LevelOutOfRange;

    fn try_from(level: u8) -> Result<Self, LevelOutOfRange> {
        Self::new(level)
    }
}

impl fmt::Display for Level {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// A layer. M0 uses only [`LayerId::BASE`]; the key and file format carry it so the layer stack
/// needs no format change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LayerId(u16);

impl LayerId {
    /// The base layer.
    pub const BASE: Self = Self(0);

    /// A layer by number.
    #[must_use]
    pub const fn new(id: u16) -> Self {
        Self(id)
    }

    /// The layer number.
    #[must_use]
    pub const fn get(self) -> u16 {
        self.0
    }
}

/// What a tile holds.
///
/// Values are fixed in the file format: new channels take new values, never renumber. It is a
/// number rather than an enum so a file written by a newer version, with channels this one doesn't
/// know, still round-trips.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Channel(u8);

impl Channel {
    /// The correction pyramid (sculpted shape).
    pub const DISPLACEMENT: Self = Self(0);

    /// A channel by its file-format value.
    #[must_use]
    pub const fn from_raw(raw: u8) -> Self {
        Self(raw)
    }

    /// Its file-format value.
    #[must_use]
    pub const fn raw(self) -> u8 {
        self.0
    }
}

/// A tile position outside its face at that level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("tile ({tx}, {ty}) is outside a face with {per_side}×{per_side} tiles at level {level}")]
pub struct TileOutOfRange {
    /// Tile column.
    pub tx: u16,
    /// Tile row.
    pub ty: u16,
    /// The level.
    pub level: Level,
    /// Tiles per side at that level.
    pub per_side: u32,
}

/// One tile: a block of [`TILE_QUADS`]² quads (or a whole face at levels up to 6) of one channel of
/// one layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TileKey {
    layer: LayerId,
    channel: Channel,
    face: FaceId,
    level: Level,
    tx: u16,
    ty: u16,
}

impl TileKey {
    /// # Errors
    ///
    /// [`TileOutOfRange`] if `tx` or `ty` is not below `level.tiles_per_side()`.
    pub fn new(
        layer: LayerId,
        channel: Channel,
        face: FaceId,
        level: Level,
        tx: u16,
        ty: u16,
    ) -> Result<Self, TileOutOfRange> {
        let per_side = level.tiles_per_side();
        if u32::from(tx) < per_side && u32::from(ty) < per_side {
            Ok(Self {
                layer,
                channel,
                face,
                level,
                tx,
                ty,
            })
        } else {
            Err(TileOutOfRange {
                tx,
                ty,
                level,
                per_side,
            })
        }
    }

    /// A base-layer displacement tile.
    ///
    /// # Errors
    ///
    /// As [`TileKey::new`].
    pub fn displacement(
        face: FaceId,
        level: Level,
        tx: u16,
        ty: u16,
    ) -> Result<Self, TileOutOfRange> {
        Self::new(LayerId::BASE, Channel::DISPLACEMENT, face, level, tx, ty)
    }

    /// Without the range check, for callers that derived `tx`, `ty` from a valid grid position.
    pub(crate) const fn new_unchecked(
        layer: LayerId,
        channel: Channel,
        face: FaceId,
        level: Level,
        tx: u16,
        ty: u16,
    ) -> Self {
        Self {
            layer,
            channel,
            face,
            level,
            tx,
            ty,
        }
    }

    /// The tile holding face-grid vertex `(i, j)`, and the vertex's coordinates within it. A vertex
    /// on a border between tiles belongs to several; this picks the one with the higher index,
    /// except at the face's far edge. `None` if `(i, j)` is outside the face.
    #[must_use]
    pub fn containing(
        layer: LayerId,
        channel: Channel,
        face: FaceId,
        level: Level,
        i: u32,
        j: u32,
    ) -> Option<(Self, u32, u32)> {
        let n = level.face_quads();
        if i > n || j > n {
            return None;
        }
        let tq = level.tile_quads();
        let last = level.tiles_per_side() - 1;
        let (tx, ty) = ((i / tq).min(last), (j / tq).min(last));
        let key = Self::new_unchecked(
            layer,
            channel,
            face,
            level,
            u16::try_from(tx).ok()?,
            u16::try_from(ty).ok()?,
        );
        Some((key, i - tx * tq, j - ty * tq))
    }

    /// The layer.
    #[must_use]
    pub const fn layer(&self) -> LayerId {
        self.layer
    }

    /// The channel.
    #[must_use]
    pub const fn channel(&self) -> Channel {
        self.channel
    }

    /// The face.
    #[must_use]
    pub const fn face(&self) -> FaceId {
        self.face
    }

    /// The level.
    #[must_use]
    pub const fn level(&self) -> Level {
        self.level
    }

    /// Tile column within the face.
    #[must_use]
    pub const fn tx(&self) -> u16 {
        self.tx
    }

    /// Tile row within the face.
    #[must_use]
    pub const fn ty(&self) -> u16 {
        self.ty
    }

    /// Vertices per side of this tile.
    #[must_use]
    pub const fn side(&self) -> u32 {
        self.level.tile_quads() + 1
    }
}

/// A square block of one face's grid at one level: quads `x0 .. x0 + quads` by `y0 .. y0 + quads`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Region {
    face: FaceId,
    level: Level,
    x0: u32,
    y0: u32,
    quads: u32,
}

impl Region {
    /// A block, if it lies inside the face and (above level 0) starts and ends on even grid lines,
    /// so it has a parent.
    #[must_use]
    pub fn new(face: FaceId, level: Level, x0: u32, y0: u32, quads: u32) -> Option<Self> {
        let n = level.face_quads();
        let inside = quads >= 1 && x0.checked_add(quads)? <= n && y0.checked_add(quads)? <= n;
        let even = level == Level::ZERO
            || (x0.is_multiple_of(2) && y0.is_multiple_of(2) && quads.is_multiple_of(2));
        (inside && even).then_some(Self {
            face,
            level,
            x0,
            y0,
            quads,
        })
    }

    /// A face's whole grid.
    #[must_use]
    pub const fn whole_face(face: FaceId, level: Level) -> Self {
        Self {
            face,
            level,
            x0: 0,
            y0: 0,
            quads: level.face_quads(),
        }
    }

    /// The block a tile covers.
    #[must_use]
    pub fn of_tile(key: &TileKey) -> Self {
        let q = key.level.tile_quads();
        Self {
            face: key.face,
            level: key.level,
            x0: u32::from(key.tx) * q,
            y0: u32::from(key.ty) * q,
            quads: q,
        }
    }

    /// The same area one level coarser, or `None` at level 0.
    #[must_use]
    pub fn parent(self) -> Option<Self> {
        let level = self.level.parent()?;
        Some(Self {
            face: self.face,
            level,
            x0: self.x0 / 2,
            y0: self.y0 / 2,
            quads: (self.quads / 2).max(1),
        })
    }

    /// The face.
    #[must_use]
    pub const fn face(&self) -> FaceId {
        self.face
    }

    /// The level.
    #[must_use]
    pub const fn level(&self) -> Level {
        self.level
    }

    /// First quad column.
    #[must_use]
    pub const fn x0(&self) -> u32 {
        self.x0
    }

    /// First quad row.
    #[must_use]
    pub const fn y0(&self) -> u32 {
        self.y0
    }

    /// Quads per side.
    #[must_use]
    pub const fn quads(&self) -> u32 {
        self.quads
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    #[test]
    fn level_sizes() {
        let l = |n| Level::new(n).unwrap();
        assert_eq!(l(0).face_quads(), 1, "base");
        assert_eq!(l(6).tile_quads(), 64, "level 6 is one full tile");
        assert_eq!(l(6).tiles_per_side(), 1, "level 6: one tile");
        assert_eq!(l(9).tiles_per_side(), 8, "level 9: 8×8 tiles");
        assert_eq!(l(9).tile_quads(), 64, "full tiles");
        assert_eq!(l(3).tile_quads(), 8, "small levels: whole face");
        assert!(Level::new(MAX_LEVEL + 1).is_err(), "above max");
        assert_eq!(l(0).parent(), None, "level 0 has no parent");
    }

    #[test]
    fn tile_key_validates_its_position() {
        let face = FaceId::new(3);
        let level = Level::new(9).unwrap();
        assert!(
            TileKey::displacement(face, level, 7, 7).is_ok(),
            "last tile"
        );
        assert!(
            TileKey::displacement(face, level, 8, 0).is_err(),
            "past the face"
        );
        let l6 = Level::new(6).unwrap();
        assert!(
            TileKey::displacement(face, l6, 1, 0).is_err(),
            "level 6 has one tile"
        );
    }

    #[test]
    fn containing_prefers_the_lower_tile_on_a_border() {
        let level = Level::new(8).unwrap(); // 4×4 tiles of 64
        let (key, i, j) = TileKey::containing(
            LayerId::BASE,
            Channel::DISPLACEMENT,
            FaceId::new(0),
            level,
            64,
            256,
        )
        .unwrap();
        assert_eq!(
            (key.tx(), key.ty(), i, j),
            (1, 3, 0, 64),
            "i on a border: the tile to its right; j = N: last tile"
        );
        assert!(
            TileKey::containing(
                LayerId::BASE,
                Channel::DISPLACEMENT,
                FaceId::new(0),
                level,
                257,
                0
            )
            .is_none(),
            "outside the face"
        );
    }

    #[test]
    fn region_parents_halve() {
        let key = TileKey::displacement(FaceId::new(0), Level::new(9).unwrap(), 3, 5).unwrap();
        let r = Region::of_tile(&key);
        assert_eq!((r.x0(), r.y0(), r.quads()), (192, 320, 64), "tile region");
        let p = r.parent().unwrap();
        assert_eq!(
            (p.level().get(), p.x0(), p.y0(), p.quads()),
            (8, 96, 160, 32),
            "parent region"
        );
        let whole = Region::whole_face(FaceId::new(0), Level::new(1).unwrap());
        assert_eq!(
            whole.parent().unwrap().quads(),
            1,
            "level 1's parent is the base quad"
        );
        assert_eq!(
            Region::whole_face(FaceId::new(0), Level::ZERO).parent(),
            None,
            "no parent at level 0"
        );
    }
}
