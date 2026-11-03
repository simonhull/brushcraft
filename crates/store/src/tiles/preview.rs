//! Previews: tiles shown ahead of their finest work, held in memory beside the store's own
//! tiles and never saved.
//!
//! A brush whose finest work doesn't fit its frame shows its dabs at once on **previews** of the
//! tiles a renderer draws, and the dabs land later, as finest work through the store's edits (the
//! `sculpt` crate makes the previews). A preview is either:
//!
//! - a **finest tile's** ([`TileStore::set_preview`]): its positions with the ring, gathered as the
//!   store holds them ([`TileStore::positions_with_ring`]) and moved by the dabs still to land; or
//! - a **display tile's** ([`TileStore::set_display_preview`], [`DisplayPreview`]): the vertices a
//!   renderer draws at a display level (positions, normals, roughness) and the bound of their
//!   distance from the finest surface the dabs will leave.
//!
//! The store keeps them and hands them to whoever draws ([`TileStore::preview`],
//! [`TileStore::display_preview`]). Nothing else reads them: a stroke places its dabs on its
//! pen-down surface (see the `pen_down` module), and evaluation, the brush's edits, the display
//! layer's refresh and every file the store writes see only the store's own tiles: a preview is
//! never saved, and a save mid-stroke holds what has landed.
//!
//! # Held tiles
//!
//! The brush also tells the store which finest tiles its dabs still to land reach
//! ([`TileStore::hold_preview`], [`TileStore::release_preview`]): those and every display tile
//! above them are **held**. A held tile is drawn from its preview or not at all: a renderer asks
//! [`TileStore::preview_wanted`] before it loads one, and leaves a held tile with no preview yet
//! for the brush to make one ([`TileStore::preview_wanted`] names them). A held display tile isn't
//! refreshed: built from the store's finest level, it would show the surface without the dabs
//! still to land.
//!
//! A display preview stays after its tile is released, the dabs under it landed: its vertices are
//! then the store's surface itself, and it is drawn until the display layer's refresh rebuilds
//! the tile, which replaces it ([`TileStore::record_built`] drops it). A finest preview is cleared
//! by the brush once nothing still to land moves it.
//!
//! # Epochs
//!
//! The tiles whose previews were set or cleared since the last close are noted (a preview taken
//! to move and put back unmoved, [`TileStore::put_back_preview`], is not: a renderer has it), and
//! [`TileStore::close_preview_epoch`] closes them as one epoch of the stale-tile feed, of kind
//! [`EpochKind::Preview`]: the positions a renderer reads for them changed, the store's did not.
//! One epoch holds the finest and display tiles of one brush call, so a renderer switches them
//! together. A consumer that resyncs is handed every finest preview again with the resync, as
//! unrefreshed display tiles over the previewed finest tiles
//! ([`Drained::Resync`](crate::Drained::Resync)); the display previews it reads again with every
//! other tile it keeps.

use std::collections::{BTreeMap, BTreeSet};

use geom::{Channel, GatheredBlock, Level, Region, TileKey, Vec3};

use super::TileStore;
use super::error::CacheError;
use crate::stale::EpochKind;

/// The store's previews: each finest tile's and display tile's, the tiles held, and what changed
/// since the last epoch.
#[derive(Debug, Default)]
pub(crate) struct Previews {
    /// Each previewed finest tile: its positions with the ring as a renderer draws them, and
    /// what is made of them.
    blocks: BTreeMap<TileKey, Shown>,
    /// Each previewed display tile: what a renderer draws for it.
    display: BTreeMap<TileKey, DisplayPreview>,
    /// The finest tiles the brush's dabs still to land reach: held (see the module notes).
    held: BTreeSet<TileKey>,
    /// Per display tile above a held finest tile, how many held finest tiles lie under it: held
    /// too, and the refresh leaves them alone.
    above: BTreeMap<TileKey, usize>,
    /// The tiles whose previews were set or cleared since the last preview epoch.
    changed: BTreeSet<TileKey>,
}

/// A display tile's preview: the vertices a renderer draws for it while a brush's dabs are shown
/// ahead of their finest work, and their bound (see the module notes).
///
/// Its vertices are row-major, `side²` of each, as a [`geom::DisplayTile`]'s; the cone of its
/// finest normals is unknown (a renderer takes the one that covers everything).
#[derive(Debug, Clone, PartialEq)]
pub struct DisplayPreview {
    /// The positions drawn.
    pub positions: Vec<Vec3>,
    /// The unit normals drawn.
    pub normals: Vec<Vec3>,
    /// The roughness term per vertex, `0..=255` for `0..=1`.
    pub roughness: Vec<u8>,
    /// A bound, in world units, on the distance from the surface the positions draw to every
    /// finest vertex under the tile once every dab shown has landed.
    pub bound: f32,
    /// Whether dabs still to land reach it, when it was made: a renderer then leaves it to the
    /// brush (no stand-in of its own over it), which makes it again as they land. Once released,
    /// its positions are the store's surface, and a renderer treats it as any data it read.
    pub held: bool,
}

/// A previewed tile: its block, taken out while its brush moves it.
///
/// A brush moving the tile takes its block out ([`TileStore::take_preview`]) and either sets a
/// new one ([`TileStore::set_preview`]) or puts the same one back unmoved
/// ([`TileStore::put_back_preview`]), which a renderer has already.
#[derive(Debug)]
struct Shown {
    /// Its positions with the ring and the base-vertex rings; `None` while its brush has taken
    /// it to move it.
    block: Option<GatheredBlock>,
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
    /// module notes).
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
        previews.blocks.insert(key, Shown { block: Some(block) });
        previews.changed.insert(key);
        Ok(())
    }

    /// Clears finest tile `key`'s preview, if it has one: a renderer draws the store's own tile
    /// again. Noted for the next preview epoch.
    pub fn clear_preview(&mut self, key: &TileKey) {
        let previews = &mut self.previews;
        if previews.blocks.remove(key).is_some() {
            previews.changed.insert(*key);
        }
    }

    /// Holds finest tile `key` and every display tile above it: a dab still to land reaches it
    /// (see "Held tiles" in the module notes). Holding a held tile does nothing.
    pub fn hold_preview(&mut self, key: TileKey) {
        let previews = &mut self.previews;
        if previews.held.insert(key) {
            for above in Previews::ancestors(&key) {
                *previews.above.entry(above).or_insert(0) += 1;
            }
        }
    }

    /// Releases finest tile `key` from [`TileStore::hold_preview`]: no dab still to land reaches
    /// it. The display tiles above it no other held tile is under are released too; the refresh
    /// looks again for the tiles it left alone.
    pub fn release_preview(&mut self, key: &TileKey) {
        let previews = &mut self.previews;
        if !previews.held.remove(key) {
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
        // The refresh dropped the tiles above it: it looks for unknown tiles again.
        self.refresh_seen = u64::MAX;
    }

    /// Whether tile `key` is held: a finest tile a dab still to land reaches, or a display tile
    /// above one.
    #[must_use]
    pub fn preview_held(&self, key: &TileKey) -> bool {
        if key.channel() == Channel::DISPLAY {
            self.previews.above.contains_key(key)
        } else {
            self.previews.held.contains(key)
        }
    }

    /// Whether tile `key` is held with no preview made for it: a renderer must not draw it from
    /// the store's own data, and waits for the brush to make its preview.
    #[must_use]
    pub fn preview_wanted(&self, key: &TileKey) -> bool {
        self.preview_held(key)
            && if key.channel() == Channel::DISPLAY {
                !self.previews.display.contains_key(key)
            } else {
                !self.previews.blocks.contains_key(key)
            }
    }

    /// Every tile held, finest and display, in key order.
    pub fn held_previews(&self) -> impl Iterator<Item = &TileKey> {
        self.previews.held.iter().chain(self.previews.above.keys())
    }

    /// Sets display tile `key`'s preview to `preview`, noted for the next preview epoch.
    ///
    /// The store keeps it in memory, outside its budget, and never reads or saves it (see the
    /// module notes); the refresh replaces it once the tile is no longer held.
    ///
    /// # Errors
    ///
    /// [`CacheError::NotATile`] if `key` isn't a tile of the store's display layer,
    /// [`CacheError::NotItsBlock`] if `preview` doesn't hold `side²` of each vertex value.
    /// Nothing is kept then.
    pub fn set_display_preview(
        &mut self,
        key: TileKey,
        preview: DisplayPreview,
    ) -> Result<(), CacheError> {
        let below_finest = self
            .display_finest
            .is_some_and(|finest| key.level() < finest);
        let of_the_mesh = key.layer() == geom::LayerId::BASE
            && key.level() != Level::ZERO
            && self.mesh.topology().contains(key.face());
        if key.channel() != Channel::DISPLAY || !below_finest || !of_the_mesh {
            return Err(CacheError::NotATile(key));
        }
        let vertices = usize::try_from(key.side() * key.side()).unwrap_or(usize::MAX);
        if [
            preview.positions.len(),
            preview.normals.len(),
            preview.roughness.len(),
        ] != [vertices; 3]
        {
            return Err(CacheError::NotItsBlock(key));
        }
        self.previews.display.insert(key, preview);
        self.previews.changed.insert(key);
        Ok(())
    }

    /// Display tile `key`'s preview, if it has one: what a renderer is to draw for it.
    #[must_use]
    pub fn display_preview(&self, key: &TileKey) -> Option<&DisplayPreview> {
        self.previews.display.get(key)
    }

    /// Takes display tile `key`'s preview out, if it has one, for the brush to make it again;
    /// noted for the next preview epoch whether it is set again or not.
    pub fn take_display_preview(&mut self, key: &TileKey) -> Option<DisplayPreview> {
        let taken = self.previews.display.remove(key);
        if taken.is_some() {
            self.previews.changed.insert(*key);
        }
        taken
    }

    /// Every display tile with a preview, in key order.
    pub fn display_previewed(&self) -> impl Iterator<Item = &TileKey> {
        self.previews.display.keys()
    }

    /// Drops display tile `key`'s preview as the refresh records the tile: its own epoch names it.
    pub(super) fn refreshed_preview(&mut self, key: &TileKey) {
        self.previews.display.remove(key);
    }

    /// Finest tile `key`'s preview, if it has one: its positions with the ring as a renderer is
    /// to draw them.
    #[must_use]
    pub fn preview(&self, key: &TileKey) -> Option<&GatheredBlock> {
        self.previews.blocks.get(key)?.block.as_ref()
    }

    /// Takes finest tile `key`'s preview out to move it, if it has one: the brush sets it again
    /// ([`TileStore::set_preview`]), puts it back unmoved ([`TileStore::put_back_preview`]) or
    /// clears it ([`TileStore::clear_preview`]) before the next preview epoch closes. Meanwhile
    /// it counts as previewed (the refresh still waits for it), and [`TileStore::preview`] has
    /// nothing of it.
    pub fn take_preview(&mut self, key: &TileKey) -> Option<GatheredBlock> {
        self.previews.blocks.get_mut(key)?.block.take()
    }

    /// Puts back finest tile `key`'s preview `block`, taken with [`TileStore::take_preview`] and
    /// not moved since: the same bits, so nothing is noted for the next preview epoch (a renderer
    /// has it already).
    ///
    /// The caller vouches that `block` is the one taken, unchanged: a block that differs is to be
    /// set ([`TileStore::set_preview`]).
    ///
    /// # Errors
    ///
    /// [`CacheError::NotItsBlock`] if `block` isn't gathered over `key`'s region,
    /// [`CacheError::NoPreviewTaken`] if `key` has no preview taken; nothing is kept then.
    pub fn put_back_preview(
        &mut self,
        key: TileKey,
        block: GatheredBlock,
    ) -> Result<(), CacheError> {
        if block.region() != Region::of_tile(&key) {
            return Err(CacheError::NotItsBlock(key));
        }
        let Some(shown) = self
            .previews
            .blocks
            .get_mut(&key)
            .filter(|shown| shown.block.is_none())
        else {
            return Err(CacheError::NoPreviewTaken(key));
        };
        shown.block = Some(block);
        Ok(())
    }

    /// Every previewed finest tile, in key order.
    pub fn previewed(&self) -> impl Iterator<Item = &TileKey> {
        self.previews.blocks.keys()
    }

    /// Whether any tile has a preview, finest or display, or is held.
    #[must_use]
    pub fn has_previews(&self) -> bool {
        !self.previews.blocks.is_empty()
            || !self.previews.display.is_empty()
            || !self.previews.held.is_empty()
    }

    /// Closes the tiles whose previews were set or cleared since the last close as one
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

    /// Adds every previewed finest tile to `unrefreshed`, under each display tile above it (with
    /// no displacement where the display tile has none yet): what a consumer's resync, which lost
    /// the epochs naming the previews, needs to stand in for them again.
    pub(super) fn previews_unrefreshed(&self, unrefreshed: &mut Vec<crate::Unrefreshed>) {
        let mut at: BTreeMap<TileKey, usize> = unrefreshed
            .iter()
            .enumerate()
            .map(|(n, entry)| (entry.tile, n))
            .collect();
        for key in self.previews.blocks.keys() {
            for above in Previews::ancestors(key) {
                let n = *at.entry(above).or_insert_with(|| {
                    unrefreshed.push(crate::Unrefreshed {
                        tile: above,
                        displacement: 0.0,
                        finest: BTreeSet::new(),
                    });
                    unrefreshed.len() - 1
                });
                if let Some(entry) = unrefreshed.get_mut(n) {
                    entry.finest.insert(*key);
                }
            }
        }
    }

    /// Whether display tile `key` waits for the dabs under it to land before it is refreshed: it
    /// is held.
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
