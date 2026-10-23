//! The tree's key table: where each tile's item is, found from the tile's place on the mesh.

use geom::{Channel, LayerId, Level, TileKey};

use crate::index::IndexError;

/// Where each tile's item is in a [`super::Bvh`], by the tile's place on the mesh: a dense
/// table of `u32` item indices, one entry per tile of one level.
///
/// A tree holds the tiles of one level, layer and channel (the finest level a store evaluates),
/// and every tile of it once: the place of a tile in its face's grid is all that tells two of
/// them apart. So the table is an array indexed by `face · per_side² + ty · per_side + tx`, four
/// bytes a tile, with no hashing and no keys stored. It is sized by the highest face it holds,
/// so a table over a few tiles of a high face holds a slot for every tile of the faces before
/// it, absent; over a whole mesh's level, as the spatial index builds it, every slot is used.
#[derive(Debug, Clone, Default)]
pub(super) struct KeyTable {
    /// The layer, channel and level every tile of the table shares; `None` for an empty table.
    kind: Option<(LayerId, Channel, Level)>,
    /// Tiles per side of a face at the table's level.
    per_side: usize,
    /// The item index of each tile, at its place (see the type's notes);
    /// [`KeyTable::ABSENT`] where the tree holds no tile.
    slots: Vec<u32>,
}

impl KeyTable {
    /// A slot of no tile.
    const ABSENT: u32 = u32::MAX;

    /// The table of `keys`, each pointing at its position in `keys`.
    ///
    /// Every key is checked before anything else is built, so a tree's build refuses bad input
    /// before it does any work.
    ///
    /// # Errors
    ///
    /// [`IndexError::MixedTiles`] for the first key whose layer, channel or level isn't the
    /// first key's; [`IndexError::TooManyTiles`] if the table would need more than
    /// [`super::MAX_TILES`] slots (a key of a face so high that every tile of the faces before
    /// it is more than a tree holds), refused before anything is allocated;
    /// [`IndexError::DuplicateTile`] for the first key listed a second time. The caller keeps
    /// `keys` within [`super::MAX_TILES`] too, so every position fits a `u32` below
    /// [`KeyTable::ABSENT`].
    pub(super) fn new(keys: &[TileKey]) -> Result<Self, IndexError> {
        let Some(first) = keys.first() else {
            return Ok(Self::default());
        };
        let kind = (first.layer(), first.channel(), first.level());
        if let Some(&other) = keys
            .iter()
            .find(|k| (k.layer(), k.channel(), k.level()) != kind)
        {
            return Err(IndexError::MixedTiles(other));
        }
        let faces = keys.iter().map(|k| k.face().index()).max().map_or(0, |f| {
            // A face index is a `u32`, so one more fits a `usize` on every target we build for.
            usize::try_from(f).map_or(usize::MAX, |f| f.saturating_add(1))
        });
        let per_side = kind.2.tiles_per_side() as usize;
        let len = faces.saturating_mul(per_side * per_side);
        if len > super::MAX_TILES {
            return Err(IndexError::TooManyTiles(len));
        }
        let mut table = Self {
            kind: Some(kind),
            per_side,
            slots: vec![Self::ABSENT; len],
        };
        for (index, key) in (0..).zip(keys) {
            let slot = table.slot_mut(key).ok_or(IndexError::UnknownTile(*key))?;
            if *slot != Self::ABSENT {
                return Err(IndexError::DuplicateTile(*key));
            }
            *slot = index;
        }
        Ok(table)
    }

    /// Points each of `keys` at its position in `keys`: after the build reorders the items,
    /// with the keys in their new order. Keys the table doesn't hold are skipped (the build
    /// hands it exactly the keys it was made with).
    pub(super) fn reindex<'k>(&mut self, keys: impl IntoIterator<Item = &'k TileKey>) {
        for (index, key) in (0..).zip(keys) {
            if let Some(slot) = self.slot_mut(key) {
                *slot = index;
            }
        }
    }

    /// The item index of `key`; `None` if the table doesn't hold it.
    pub(super) fn get(&self, key: &TileKey) -> Option<usize> {
        let &index = self.slots.get(self.place(key)?)?;
        (index != Self::ABSENT).then_some(index as usize)
    }

    /// How many tiles the table holds.
    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.slots.iter().filter(|&&s| s != Self::ABSENT).count()
    }

    /// The bytes the table holds on the heap.
    pub(super) fn heap_bytes(&self) -> usize {
        self.slots.capacity() * size_of::<u32>()
    }

    /// `key`'s slot, mutably; `None` if the key isn't of the table's kind or is past its last
    /// face.
    fn slot_mut(&mut self, key: &TileKey) -> Option<&mut u32> {
        let place = self.place(key)?;
        self.slots.get_mut(place)
    }

    /// Where `key`'s slot is: `face · per_side² + ty · per_side + tx`; `None` if the key isn't
    /// of the table's layer, channel and level. The key's own range check keeps `tx` and `ty`
    /// below `per_side`.
    fn place(&self, key: &TileKey) -> Option<usize> {
        let kind = self.kind?;
        if (key.layer(), key.channel(), key.level()) != kind {
            return None;
        }
        let face = usize::try_from(key.face().index()).ok()?;
        let (tx, ty) = (usize::from(key.tx()), usize::from(key.ty()));
        face.checked_mul(self.per_side * self.per_side)?
            .checked_add(ty * self.per_side + tx)
    }
}
