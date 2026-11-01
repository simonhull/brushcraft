//! Previews: finest tiles shown ahead of their evaluation, held in memory beside the store's own
//! tiles and never saved.
//!
//! A brush whose finest work doesn't fit its frame shows its dabs at once on a **preview**: the
//! finest tiles they reach, gathered as the store holds them with their ring
//! ([`TileStore::positions_with_ring`]) and moved by the dabs, as a renderer would draw them once
//! the dabs land (the `sculpt` crate makes them). The dabs themselves land later, as finest work
//! through the store's edits; the brush then sets each tile's preview again over what landed, or
//! clears it once no dab it shows is left to land.
//!
//! The store keeps a tile's preview ([`TileStore::set_preview`]) and hands it to whoever draws
//! ([`TileStore::preview`]); it reads none of it itself. Evaluation, the spatial index, raycasts,
//! the display layer's refresh and every file the store writes see only the store's own tiles: a
//! preview is never saved, and a save mid-stroke holds what has landed.
//!
//! # Epochs
//!
//! The tiles whose previews were set or cleared since the last close are noted, and
//! [`TileStore::close_preview_epoch`] closes them as one epoch of the stale-tile feed, of kind
//! [`EpochKind::Preview`]: the positions a renderer reads for them changed, the store's did not.
//! A consumer that resyncs is handed every preview again, as an epoch closed right after its
//! resync.
//!
//! # The refresh waits
//!
//! A display tile with a previewed finest tile under it isn't refreshed while the preview lasts:
//! built from the store's finest level, it would show the surface without the dabs still to land,
//! replacing what a renderer draws from the preview. The refresh drops such tiles (and those
//! waiting on one), and looks for them again once a preview is cleared.

use std::collections::{BTreeMap, BTreeSet};

use geom::{Channel, GatheredBlock, Level, Region, TileKey};

use super::TileStore;
use super::error::CacheError;
use crate::stale::EpochKind;

/// The store's previews: each finest tile's, the display tiles above them, and what changed since
/// the last epoch.
#[derive(Debug, Default)]
pub(crate) struct Previews {
    /// Each previewed finest tile's block: its positions with the ring as a renderer draws them;
    /// `None` while its brush has taken it to move it ([`TileStore::take_preview`]).
    blocks: BTreeMap<TileKey, Option<GatheredBlock>>,
    /// Per display tile above a previewed finest tile, how many previewed finest tiles lie
    /// under it: the refresh leaves these alone.
    above: BTreeMap<TileKey, usize>,
    /// The finest tiles whose previews were set or cleared since the last preview epoch.
    changed: BTreeSet<TileKey>,
}

impl Previews {
    /// The display tiles above finest tile `key`: one at each level from 1 below its own.
    fn ancestors(key: &TileKey) -> impl Iterator<Item = TileKey> + use<> {
        let (face, finest, tx, ty) = (key.face(), key.level(), key.tx(), key.ty());
        let per_side = finest.tiles_per_side();
        (1..finest.get()).filter_map(move |l| {
            let level = Level::new(l).ok()?;
            let scale = per_side / level.tiles_per_side().max(1);
            let (x, y) = (u32::from(tx) / scale.max(1), u32::from(ty) / scale.max(1));
            let (x, y) = (u16::try_from(x).ok()?, u16::try_from(y).ok()?);
            TileKey::display(face, level, x, y).ok()
        })
    }
}

impl TileStore {
    /// Sets finest tile `key`'s preview to `block`: its positions with the ring, as a renderer
    /// is to draw them until the preview is set again or cleared. Noted for the next preview
    /// epoch ([`TileStore::close_preview_epoch`]).
    ///
    /// The store keeps it in memory, outside its budget, and never reads or saves it (see the
    /// module notes); the display tiles above it aren't refreshed while it lasts.
    ///
    /// # Errors
    ///
    /// [`CacheError::NotATile`] if `key` isn't a tile of the store's finest level (the display
    /// layer's, for a store keeping one), [`CacheError::NotItsBlock`] if `block` isn't gathered
    /// over `key`'s region. Nothing is kept then.
    pub fn set_preview(&mut self, key: TileKey, block: GatheredBlock) -> Result<(), CacheError> {
        self.check_key(&key)?;
        if self
            .display_finest
            .is_some_and(|finest| finest != key.level())
        {
            return Err(CacheError::NotATile(key));
        }
        if block.region() != Region::of_tile(&key) {
            return Err(CacheError::NotItsBlock(key));
        }
        let previews = &mut self.previews;
        if previews.blocks.insert(key, Some(block)).is_none() {
            for above in Previews::ancestors(&key) {
                *previews.above.entry(above).or_insert(0) += 1;
            }
        }
        previews.changed.insert(key);
        Ok(())
    }

    /// Clears finest tile `key`'s preview, if it has one: a renderer draws the store's own tile
    /// again. Noted for the next preview epoch; the refresh looks again for the display tiles it
    /// left alone.
    pub fn clear_preview(&mut self, key: &TileKey) {
        let previews = &mut self.previews;
        if previews.blocks.remove(key).is_none() {
            return;
        }
        for above in Previews::ancestors(key) {
            if let Some(count) = previews.above.get_mut(&above) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    previews.above.remove(&above);
                }
            }
        }
        previews.changed.insert(*key);
        // The refresh dropped the tiles above it: it looks for unknown tiles again.
        self.refresh_seen = u64::MAX;
    }

    /// Finest tile `key`'s preview, if it has one: its positions with the ring as a renderer is
    /// to draw them.
    #[must_use]
    pub fn preview(&self, key: &TileKey) -> Option<&GatheredBlock> {
        self.previews.blocks.get(key)?.as_ref()
    }

    /// Takes finest tile `key`'s preview out to move it, if it has one: the brush sets it again
    /// ([`TileStore::set_preview`]) or clears it ([`TileStore::clear_preview`]) before the next
    /// preview epoch closes. Meanwhile it counts as previewed (the refresh still waits for it),
    /// and [`TileStore::preview`] has nothing for it.
    pub fn take_preview(&mut self, key: &TileKey) -> Option<GatheredBlock> {
        self.previews.blocks.get_mut(key)?.take()
    }

    /// Every previewed finest tile, in key order.
    pub fn previewed(&self) -> impl Iterator<Item = &TileKey> {
        self.previews.blocks.keys()
    }

    /// Whether any finest tile has a preview.
    #[must_use]
    pub fn has_previews(&self) -> bool {
        !self.previews.blocks.is_empty()
    }

    /// Closes the finest tiles whose previews were set or cleared since the last close as one
    /// epoch of the stale-tile feed, of kind [`EpochKind::Preview`]; nothing for none. Any writes
    /// the open epoch holds are closed first, as their own epoch: a preview epoch holds no
    /// writes.
    ///
    /// # Errors
    ///
    /// As [`TileStore::close_epoch`], closing the writes first: the previews stay noted then.
    pub fn close_preview_epoch(&mut self) -> Result<(), CacheError> {
        if self.previews.changed.is_empty() {
            return Ok(());
        }
        self.close_writes()?;
        let changed = std::mem::take(&mut self.previews.changed);
        if self.stale.is_listened() {
            self.stale.close(changed, EpochKind::Preview);
        }
        Ok(())
    }

    /// Hands every previewed tile to the feed again, as one preview epoch: after a consumer's
    /// resync, which lost the epochs that named them.
    pub(super) fn reannounce_previews(&mut self) {
        let every: BTreeSet<TileKey> = self.previews.blocks.keys().copied().collect();
        if !every.is_empty() {
            self.stale.close(every, EpochKind::Preview);
        }
    }

    /// Whether display tile `key` waits for a preview under it to end before it is refreshed.
    pub(super) fn preview_holds(&self, key: &TileKey) -> bool {
        key.channel() == Channel::DISPLAY && self.previews.above.contains_key(key)
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use geom::FaceId;

    use super::*;

    #[test]
    fn a_finest_tile_has_one_display_ancestor_a_level_below_its_own() {
        let finest = Level::new(9).expect("a level");
        let key = TileKey::displacement(FaceId::new(3), finest, 5, 2).expect("a tile");
        let above: Vec<TileKey> = Previews::ancestors(&key).collect();
        assert_eq!(above.len(), 8, "levels 1 to 8");
        for (ancestor, l) in above.iter().zip(1..) {
            assert_eq!(ancestor.level().get(), l);
            assert_eq!(ancestor.channel(), Channel::DISPLAY);
            assert_eq!(ancestor.face(), FaceId::new(3));
        }
        let level_8 = above.last().expect("level 8");
        assert_eq!(
            (level_8.tx(), level_8.ty()),
            (2, 1),
            "half the tile's place"
        );
        let level_7 = above.get(6).expect("level 7");
        assert_eq!((level_7.tx(), level_7.ty()), (1, 0));
        assert!(
            above.iter().take(6).all(|a| (a.tx(), a.ty()) == (0, 0)),
            "a tile is a whole face up to level 6"
        );
    }
}
