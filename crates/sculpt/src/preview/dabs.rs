//! Which of a stroke's dabs can reach a place ([`DabIndex`]).
//!
//! What a preview made or rebased late in a long stroke takes is the dabs near it, not every dab
//! so far: a stroke keeps entering fresh tiles, and a walk of all its dabs for each one makes its
//! cost grow with its length.

use std::collections::HashMap;

use brush::Imprint;
use store::Aabb;

use super::{centre, reach};

/// The most cells a dab is filed under: past this it counts as near everywhere.
const MOST_CELLS: i64 = 512;

/// The most cells a place is looked up in: past this each dab's box is looked at.
const MOST_LOOKED_AT: i64 = 4096;

/// A stroke's dabs by the cells of a grid their spheres (see [`reach`]) touch, in stroke order.
///
/// What it finds holds every dab touching a place, and maybe a few more: a caller's own test
/// sorts them out, so what it does with them is what it would do walking every dab, bit for bit.
#[derive(Debug, Default)]
pub(super) struct DabIndex {
    /// The cells' side: twice the first dab's reach.
    cell: f32,
    /// Per cell, the numbers of the dabs touching it, ascending.
    cells: HashMap<[i64; 3], Vec<usize>>,
    /// The dabs too big, or too strange, to file.
    everywhere: Vec<usize>,
    /// Every dab's box, in order.
    boxes: Vec<([f32; 3], [f32; 3])>,
}

impl DabIndex {
    /// Files `dab`, the next of the stroke.
    pub(super) fn push(&mut self, dab: &Imprint) {
        let number = self.boxes.len();
        let reached = reach(dab);
        if self.cell <= 0.0 && reached.is_finite() && reached > 0.0 {
            self.cell = 2.0 * reached;
        }
        let at = centre(dab);
        let low = [at.x - reached, at.y - reached, at.z - reached];
        let high = [at.x + reached, at.y + reached, at.z + reached];
        self.boxes.push((low, high));
        let Some(range) = self
            .range(low, high)
            .filter(|range| count(range) <= MOST_CELLS)
        else {
            self.everywhere.push(number);
            return;
        };
        for cell in cells(range) {
            self.cells.entry(cell).or_default().push(number);
        }
    }

    /// The numbers of the dabs from `from` on that may touch `bounds`, ascending, each once.
    pub(super) fn near(&self, bounds: &Aabb, from: usize) -> Vec<usize> {
        let (low, high) = (bounds.min(), bounds.max());
        let range = self.range([low.x, low.y, low.z], [high.x, high.y, high.z]);
        let Some(range) = range.filter(|range| count(range) <= MOST_LOOKED_AT) else {
            // A coarse tile's box: each dab's own is looked at instead.
            let (low, high) = ([low.x, low.y, low.z], [high.x, high.y, high.z]);
            let meets = |&(n, (first, last)): &(usize, &([f32; 3], [f32; 3]))| {
                let apart = (0..3).any(|axis| {
                    let axes = (
                        first.get(axis),
                        last.get(axis),
                        low.get(axis),
                        high.get(axis),
                    );
                    matches!(axes, (Some(a), Some(b), Some(c), Some(d)) if b < c || a > d)
                });
                (!apart).then_some(n)
            };
            let numbered = self.boxes.iter().enumerate().skip(from);
            return numbered.filter_map(|dab| meets(&dab)).collect();
        };
        let filed = cells(range)
            .filter_map(|cell| self.cells.get(&cell))
            .flatten();
        let mut found: Vec<usize> = filed
            .chain(&self.everywhere)
            .copied()
            .filter(|&number| number >= from)
            .collect();
        found.sort_unstable();
        found.dedup();
        found
    }

    /// Forgets every dab.
    pub(super) fn clear(&mut self) {
        *self = Self::default();
    }

    /// The cells from `low` to `high`, first and last along each axis; `None` with no grid yet
    /// or for a place off it.
    fn range(&self, low: [f32; 3], high: [f32; 3]) -> Option<[(i64, i64); 3]> {
        if self.cell <= 0.0 {
            return None;
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "a cell's number, checked finite and small first"
        )]
        let at = |v: f32| {
            let cell = (f64::from(v) / f64::from(self.cell)).floor();
            (cell.abs() < 1e15).then_some(cell as i64)
        };
        let axis = |n: usize| Some((at(*low.get(n)?)?, at(*high.get(n)?)?));
        Some([axis(0)?, axis(1)?, axis(2)?])
    }
}

/// How many cells `range` holds.
fn count(range: &[(i64, i64); 3]) -> i64 {
    range
        .iter()
        .map(|&(first, last)| last.saturating_sub(first).saturating_add(1))
        .fold(1, i64::saturating_mul)
}

/// The cells of `range`.
fn cells([x, y, z]: [(i64, i64); 3]) -> impl Iterator<Item = [i64; 3]> {
    (x.0..=x.1)
        .flat_map(move |i| (y.0..=y.1).flat_map(move |j| (z.0..=z.1).map(move |k| [i, j, k])))
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use brush::Imprint;
    use geom::Vec3;
    use store::Aabb;

    use super::DabIndex;
    use crate::preview::touches;

    /// A dab at `at`, of radius `radius`.
    fn dab(at: [f32; 3], radius: f32) -> Imprint {
        Imprint::new(at, [0.0, 0.0, 1.0], radius, 0.01, 0.0, [0.0; 4])
    }

    /// What it finds holds every dab touching a place, whatever their sizes, and few others.
    #[test]
    fn the_dabs_near_a_place_hold_every_dab_touching_it() {
        let mut index = DabIndex::default();
        let mut dabs = Vec::new();
        for n in 0..400_u16 {
            let t = f32::from(n) * 0.05;
            let radius = if n % 50 == 49 { 3.0 } else { 0.1 };
            let at = [t.cos() * t * 0.1, t.sin() * t * 0.1, -0.5];
            dabs.push(dab(at, radius));
            index.push(dabs.last().unwrap());
        }
        for n in 0..50_u16 {
            let t = f32::from(n) * 0.4;
            let centre = Vec3::new(t.cos() * t * 0.1, t.sin() * t * 0.1, -0.5);
            let spread = Vec3::new(0.05, 0.05, 0.05);
            let place = Aabb::new(centre - spread, centre + spread).unwrap();
            let want: Vec<usize> = (100..dabs.len())
                .filter(|&d| touches(Some(&place), &dabs[d]))
                .collect();
            let found = index.near(&place, 100);
            assert!(found.is_sorted(), "in stroke order");
            for d in &want {
                assert!(found.contains(d), "dab {d} touches place {n}");
            }
            assert!(found.len() < dabs.len() / 2, "{} found", found.len());
        }
    }

    /// A place too big to look up by cells still leaves out the dabs that miss it.
    #[test]
    fn a_place_too_big_to_look_up_looks_at_each_dab() {
        let mut index = DabIndex::default();
        index.push(&dab([0.0; 3], 0.001));
        index.push(&dab([1.0; 3], 0.001));
        index.push(&dab([1.5; 3], 0.001));
        index.push(&dab([20.0; 3], 0.001));
        let big = Aabb::new(Vec3::new(-9.0, -9.0, -9.0), Vec3::new(9.0, 9.0, 9.0)).unwrap();
        assert_eq!(index.near(&big, 1), vec![1, 2]);
    }
}
