//! The level jumps of a stroke's tiles: how far the level drawing a stroked part of the surface
//! changed from one cut to the next.
//!
//! The M1 spec's target for a stroke seen from a distance: no level jump bigger than one during
//! the stroke.
//!
//! A stroked place is a finest tile the stroke's epochs named (its steps and previews). The cut
//! is a partition of the surface into tiles, so each stroked place is drawn by exactly one cut
//! tile, the place itself or an ancestor; its jump between two cuts is the difference of those
//! tiles' levels. Two cut tiles drawing the same stroked place overlap, so one is an ancestor of
//! the other (or they are the same), and every such pair draws one: the largest jump over the
//! stroked places is the largest level difference over pairs of tiles, one from each cut, one an
//! ancestor-or-self of the other, the finer one drawing a stroked place.
//!
//! So [`LevelJumps`] keeps every ancestor of every stroked place (the places too: its cover),
//! and a cut tile draws stroked surface exactly when it is in the cover. Each cut heard walks only
//! the tiles of both cuts in the cover up through their ancestors (at most the finest level's
//! number of steps each), looking each up in the other cut: a few hundred lookups for a stroke's
//! tiles, whatever the cut's size.

use std::collections::{BTreeSet, HashSet};

use geom::{Level, TileKey};
use render::tree::Place;
use store::{Drained, EpochKind};

/// The places a stroke edited and their ancestors, and the jumps of the cuts drawing them.
#[derive(Debug, Default)]
pub(crate) struct LevelJumps {
    /// Every finest place the current stroke's epochs named, and every ancestor of each: the
    /// cut tiles drawing stroked surface are those in it.
    cover: HashSet<Place>,
    /// Whether the last frame's stroke was open, to tell when the next one begins.
    open: bool,
}

impl LevelJumps {
    /// A frame's brush stroke is open (`open`) and its drained epochs are `drained`, in a
    /// project whose finest level is `finest`: a stroke beginning forgets the last one's places,
    /// then the finest tiles its steps and previews named join the cover.
    ///
    /// A resync's display tiles carry the finest tiles under them that strokes moved
    /// ([`store::Unrefreshed::finest`]): they join it too.
    pub(crate) fn drained(&mut self, open: bool, drained: &Drained, finest: Level) {
        if open && !self.open {
            self.cover.clear();
        }
        self.open = open;
        let named: Vec<&TileKey> = match drained {
            Drained::Epochs(epochs) => epochs
                .iter()
                .filter(|epoch| matches!(epoch.kind, EpochKind::Step { .. } | EpochKind::Preview))
                .flat_map(|epoch| &epoch.keys)
                .collect(),
            Drained::Resync(unrefreshed) => {
                unrefreshed.iter().flat_map(|tile| &tile.finest).collect()
            }
        };
        let stroked = named.into_iter().filter(|key| key.level() == finest);
        for key in stroked {
            let up = std::iter::successors(Some(Place::of(key)), |place| place.parent());
            for place in up {
                if !self.cover.insert(place) {
                    // Its ancestors are in already: they went in with it.
                    break;
                }
            }
        }
    }

    /// The largest level change of a stroked place from cut `before` to cut `after` (the cut
    /// heard before it, and it), in a project whose finest level is `finest`: 0 if every stroked
    /// place in view kept its level; `None` if no stroked place is drawn by both (no stroke yet,
    /// or none in view).
    pub(crate) fn jump(
        &self,
        before: &BTreeSet<TileKey>,
        after: &BTreeSet<TileKey>,
        finest: Level,
    ) -> Option<u8> {
        if self.cover.is_empty() {
            return None;
        }
        let finer_after = self.coarser_in(after, before, finest);
        let finer_before = self.coarser_in(before, after, finest);
        finer_after.max(finer_before)
    }

    /// Over the tiles of `cut` drawing stroked surface, the largest number of levels each is
    /// finer than the tile of `other` holding it (itself or an ancestor), where one does; `None`
    /// where none does.
    fn coarser_in(
        &self,
        cut: &BTreeSet<TileKey>,
        other: &BTreeSet<TileKey>,
        finest: Level,
    ) -> Option<u8> {
        let drawing = cut
            .iter()
            .map(Place::of)
            .filter(|place| self.cover.contains(place));
        let jumps = drawing.filter_map(|place| {
            let up = std::iter::successors(Some(place), |p| p.parent());
            let mut held = up.filter(|p| p.key(finest).is_some_and(|key| other.contains(&key)));
            let holder = held.next()?;
            Some(place.level().get() - holder.level().get())
        });
        jumps.max()
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use std::collections::BTreeSet;

    use geom::{FaceId, Level, TileKey};
    use store::{Drained, Epoch, EpochKind};

    use super::*;

    /// The finest level of the tests' project.
    const FINEST: u8 = 9;

    /// The tile at `level`, `tx`, `ty` of face 0: a display tile below the finest level, a
    /// finest tile at it.
    fn tile(level: u8, tx: u16, ty: u16) -> TileKey {
        let finest = Level::new(FINEST).unwrap();
        let level = Level::new(level).unwrap();
        if level == finest {
            TileKey::displacement(FaceId::new(0), level, tx, ty).unwrap()
        } else {
            TileKey::display(FaceId::new(0), level, tx, ty).unwrap()
        }
    }

    /// One step epoch naming `keys`.
    fn step(keys: &[TileKey]) -> Drained {
        Drained::Epochs(vec![Epoch {
            seq: 0,
            kind: EpochKind::Step { displacement: 0.1 },
            keys: keys.iter().copied().collect(),
            written: keys.iter().copied().collect(),
        }])
    }

    /// The tile at `level` holding finest tile `(tx, ty)`: its ancestor there.
    fn above(level: u8, tx: u16, ty: u16) -> TileKey {
        let finest = Level::new(FINEST).unwrap();
        let finest_place = Place::of(&tile(FINEST, tx, ty));
        let up = std::iter::successors(Some(finest_place), |p| p.parent());
        let mut at = up.filter(|p| p.level().get() == level);
        at.next().unwrap().key(finest).unwrap()
    }

    #[test]
    fn a_stroked_place_drawn_coarser_or_finer_jumps_by_the_levels_between() {
        let finest = Level::new(FINEST).unwrap();
        let mut jumps = LevelJumps::default();
        jumps.drained(true, &step(&[tile(FINEST, 7, 7)]), finest);
        let at = |level| -> BTreeSet<TileKey> { [above(level, 7, 7)].into() };
        assert_eq!(jumps.jump(&at(5), &at(5), finest), Some(0), "kept");
        assert_eq!(jumps.jump(&at(5), &at(6), finest), Some(1), "one finer");
        assert_eq!(jumps.jump(&at(6), &at(5), finest), Some(1), "one coarser");
        assert_eq!(jumps.jump(&at(4), &at(9), finest), Some(5), "to the finest");
        assert_eq!(jumps.jump(&at(9), &at(3), finest), Some(6), "back up");
    }

    #[test]
    fn tiles_drawing_no_stroked_place_never_count() {
        let finest = Level::new(FINEST).unwrap();
        let mut jumps = LevelJumps::default();
        assert_eq!(
            jumps.jump(&BTreeSet::new(), &BTreeSet::new(), finest),
            None,
            "no stroke"
        );
        jumps.drained(true, &step(&[tile(FINEST, 7, 7)]), finest);
        // Elsewhere on the face, in another level-7 tile than the stroke's: a jump of two, which
        // doesn't count.
        let before: BTreeSet<TileKey> = [above(7, 0, 0), above(8, 7, 7)].into();
        let after: BTreeSet<TileKey> = [above(9, 0, 0), above(8, 7, 7)].into();
        assert_eq!(jumps.jump(&before, &after, finest), Some(0));
        // Display tiles an epoch names aren't stroked places: only the finest ones are.
        let mut display = LevelJumps::default();
        display.drained(true, &step(&[above(7, 0, 0)]), finest);
        assert_eq!(display.jump(&before, &after, finest), None);
    }

    #[test]
    fn a_new_stroke_forgets_the_last_ones_places() {
        let finest = Level::new(FINEST).unwrap();
        let mut jumps = LevelJumps::default();
        jumps.drained(true, &step(&[tile(FINEST, 7, 7)]), finest);
        jumps.drained(false, &Drained::Epochs(Vec::new()), finest);
        let before: BTreeSet<TileKey> = [above(7, 7, 7), above(7, 0, 0)].into();
        let after: BTreeSet<TileKey> = [above(9, 7, 7), above(8, 0, 0)].into();
        assert_eq!(jumps.jump(&before, &after, finest), Some(2), "after pen-up");
        jumps.drained(true, &step(&[tile(FINEST, 0, 0)]), finest);
        assert_eq!(
            jumps.jump(&before, &after, finest),
            Some(1),
            "the new stroke's alone"
        );
    }
}
