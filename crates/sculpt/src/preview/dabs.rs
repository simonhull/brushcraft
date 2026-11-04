//! Which of a stroke's dabs can reach a place ([`DabIndex`]).
//!
//! What a preview made or rebased late in a long stroke takes is the dabs near it, not every dab
//! so far: a stroke keeps entering fresh tiles, and a walk of all its dabs for each one makes its
//! cost grow with its length.

use std::collections::{BTreeMap, HashMap};

use brush::Imprint;
use store::Aabb;

use super::{centre, reach};

/// The most cells a place is looked up in, per size of dab: past this each dab of that size has
/// its box looked at.
const MOST_LOOKED_AT: i64 = 4096;

/// A dab's box: its sphere's (see [`reach`]), low and high corners.
type DabBox = ([f32; 3], [f32; 3]);

/// The dabs of one size, by the cells of a grid of about twice their reach.
#[derive(Debug, Default)]
struct Grid {
    /// Per cell, the numbers of the dabs touching it, ascending.
    cells: HashMap<[i64; 3], Vec<usize>>,
    /// The dabs of this size, ascending: walked for a place too big to look up by cells.
    dabs: Vec<usize>,
}

/// A stroke's dabs by the cells they touch, in stroke order, one grid per size of dab.
///
/// Pressure sizes a stroke's dabs, so its first may be a tenth of the rest, and on any one grid
/// a dab would fill very many cells or a place take in very many dabs.
///
/// What it finds holds every dab touching a place, and maybe a few more: a caller's own test
/// sorts them out, so what it does with them is what it would do walking every dab, bit for bit.
#[derive(Debug, Default)]
pub(super) struct DabIndex {
    /// Per size, a power of two given by its exponent, the dabs up to that reach.
    grids: BTreeMap<i32, Grid>,
    /// The dabs with a place or size that isn't a number: near everywhere.
    strange: Vec<usize>,
    /// Every dab's box, in order.
    boxes: Vec<DabBox>,
}

impl DabIndex {
    /// Files `dab`, the next of the stroke.
    pub(super) fn push(&mut self, dab: &Imprint) {
        let number = self.boxes.len();
        let reached = reach(dab);
        let at = centre(dab);
        let low = [at.x - reached, at.y - reached, at.z - reached];
        let high = [at.x + reached, at.y + reached, at.z + reached];
        self.boxes.push((low, high));
        let size = size_of_reach(reached);
        let range = size.and_then(|size| range_of(size, low, high));
        let (Some(size), Some(range)) = (size, range) else {
            self.strange.push(number);
            return;
        };
        let grid = self.grids.entry(size).or_default();
        grid.dabs.push(number);
        for cell in cells(range) {
            grid.cells.entry(cell).or_default().push(number);
        }
    }

    /// The numbers of the dabs from `from` on that may touch `bounds`, ascending, each once.
    pub(super) fn near(&self, bounds: &Aabb, from: usize) -> Vec<usize> {
        self.looked_at(bounds, from).0
    }

    /// [`DabIndex::near`], with how many dabs it looked at to find them.
    fn looked_at(&self, bounds: &Aabb, from: usize) -> (Vec<usize>, usize) {
        let (low, high) = (bounds.min(), bounds.max());
        let (low, high) = ([low.x, low.y, low.z], [high.x, high.y, high.z]);
        let mut found: Vec<usize> = self
            .strange
            .iter()
            .copied()
            .filter(|&n| n >= from)
            .collect();
        let mut looked = found.len();
        for (&size, grid) in &self.grids {
            let range = range_of(size, low, high).filter(|range| count(range) <= MOST_LOOKED_AT);
            if let Some(range) = range {
                let filed = cells(range).filter_map(|cell| grid.cells.get(&cell));
                for numbers in filed {
                    // Ascending: those before `from` are skipped, not walked.
                    let later = numbers.partition_point(|&n| n < from);
                    let numbers = numbers.get(later..).unwrap_or_default();
                    looked += numbers.len();
                    found.extend_from_slice(numbers);
                }
            } else {
                // A coarse tile's box: each dab's own is looked at instead.
                let later = grid.dabs.partition_point(|&n| n < from);
                let numbers = grid.dabs.get(later..).unwrap_or_default();
                looked += numbers.len();
                let meeting = numbers.iter().filter(|&&n| self.meets(n, low, high));
                found.extend(meeting);
            }
        }
        found.sort_unstable();
        found.dedup();
        (found, looked)
    }

    /// Whether dab `number`'s box meets the box from `low` to `high`.
    fn meets(&self, number: usize, low: [f32; 3], high: [f32; 3]) -> bool {
        self.boxes.get(number).is_none_or(|(first, last)| {
            let axes = first.iter().zip(last).zip(low.iter().zip(&high));
            !axes.into_iter().any(|((a, b), (c, d))| b < c || a > d)
        })
    }

    /// Forgets every dab.
    pub(super) fn clear(&mut self) {
        *self = Self::default();
    }
}

/// The size of a dab of reach `reached`: the exponent of the power of two at or above it; `None`
/// if it isn't a positive number.
fn size_of_reach(reached: f32) -> Option<i32> {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the exponent of a finite f32, a few hundred at most"
    )]
    let size = f64::from(reached).log2().ceil() as i32;
    (reached.is_finite() && reached > 0.0).then_some(size)
}

/// The cells of side twice `2^size` from `low` to `high`, first and last along each axis; `None`
/// for a place off the grid.
fn range_of(size: i32, low: [f32; 3], high: [f32; 3]) -> Option<[(i64, i64); 3]> {
    let width = 2.0_f64.powi(size + 1);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a cell's number, checked finite and small first"
    )]
    let at = |v: f32| {
        let cell = (f64::from(v) / width).floor();
        (cell.abs() < 1e15).then_some(cell as i64)
    };
    let axis = |n: usize| Some((at(*low.get(n)?)?, at(*high.get(n)?)?));
    Some([axis(0)?, axis(1)?, axis(2)?])
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

    /// A place looks at the dabs near it, however a stroke's dabs vary.
    ///
    /// A stroke touching down lightly and pressed hard after, dabs of every size on a path: a few
    /// of each size, not the stroke. A walk of every dab, box test or no box test, fails this.
    #[test]
    fn a_light_touchdown_leaves_a_place_a_few_dabs_to_look_at() {
        let mut index = DabIndex::default();
        let mut dabs = Vec::new();
        for n in 0..5_000_u16 {
            let t = f32::from(n) * 0.01;
            let radius = 0.002 + 0.05 * (f32::from(n) / 200.0).min(1.0);
            dabs.push(dab([t.cos() * (1.0 + t), t.sin() * (1.0 + t), 0.0], radius));
            index.push(dabs.last().unwrap());
        }
        let last = dabs.len() - 1;
        let centre = Vec3::new(dabs[last].center[0], dabs[last].center[1], 0.0);
        let spread = Vec3::new(0.02, 0.02, 0.02);
        let place = Aabb::new(centre - spread, centre + spread).unwrap();
        let (found, looked) = index.looked_at(&place, 0);
        assert!(found.contains(&last), "the dab there");
        assert!(
            looked < dabs.len() / 20,
            "{looked} dabs looked at, {} found",
            found.len()
        );
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
