//! The stroke's region near a point: the pieces of surface within reach, joined vertex to vertex
//! inside each tile and across tile borders at their shared vertices.

use std::collections::{BTreeMap, VecDeque};

use geom::{Region, TileKey, Topology, Vec3};
use stroke::DVec3;

/// One tile's vertices within reach of the region's centre, grouped into the pieces of surface
/// they form inside the tile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Pieces {
    /// Every vertex is within reach (the tile's box lies inside the ball): one piece, number 1.
    Whole,
    /// Some vertices are: the runs of them along each row, with their pieces.
    Labelled(Spans),
}

/// The runs of vertices within reach along each row of a tile, each with its piece.
///
/// Pieces are numbered from 1 in order of first appearance, row-major. A few runs a row, where a
/// label per vertex would be thousands to allocate and fill.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Spans {
    /// The tile's vertices a side.
    side: usize,
    /// Where each row's runs start in `runs`, and one past the last row's end.
    rows: Vec<usize>,
    /// Each run: its first and last column, and its piece.
    runs: Vec<(usize, usize, u32)>,
}

impl Pieces {
    /// The pieces of a tile with `side` vertices a side and `positions`, row-major, within
    /// `reach_squared` of `center` (tested in `f32`, as [`Ball::holds`] does).
    ///
    /// Two vertices within reach are in one piece when a path of the tile's grid edges joins
    /// them through vertices within reach: a row's neighbours, a column's, and the diagonal
    /// `(i, j)`–`(i + 1, j + 1)` that splits each quad into the triangles the surface is drawn
    /// and raycast with. Found a row at a time, as runs of consecutive vertices within reach,
    /// each run joined to the runs of the row below that share a vertex or a diagonal with it.
    pub(super) fn label(positions: &[Vec3], side: u32, center: DVec3, reach_squared: f64) -> Self {
        let width = side as usize;
        let ball = Ball::new(center, reach_squared);
        let mut runs = Runs::default();
        let mut rows = Vec::with_capacity(width + 1);
        let mut every: Vec<Run> = Vec::new();
        let mut inside: Vec<bool> = vec![false; width];
        let mut below = 0..0;
        for row in positions.chunks_exact(width) {
            let tested = inside.iter_mut().zip(row);
            for (flag, &p) in tested {
                *flag = ball.holds(p);
            }
            let first_here = every.len();
            rows.push(first_here);
            let mut start = None;
            for (i, &flag) in inside.iter().enumerate() {
                match (flag, start) {
                    (true, None) => start = Some(i),
                    (false, Some(from)) => {
                        every.push(runs.make(from, i - 1));
                        start = None;
                    }
                    _ => {}
                }
            }
            if let Some(from) = start {
                every.push(runs.make(from, width - 1));
            }
            let (done, here) = every.split_at(first_here);
            let under_row = done.get(below.clone()).unwrap_or_default();
            for run in here {
                let touching = under_row
                    .iter()
                    .filter(|under| under.first <= run.last && run.first <= under.last + 1);
                for under in touching {
                    runs.join(run.id, under.id);
                }
            }
            below = first_here..every.len();
        }
        rows.push(every.len());
        // Pieces numbered by their first run in row-major order: a few runs a row, so a short
        // list of the roots seen is searched linearly.
        let mut roots: Vec<u32> = Vec::new();
        let numbered = every
            .iter()
            .map(|run| {
                let root = runs.root(run.id);
                let piece = if let Some(k) = roots.iter().position(|&r| r == root) {
                    k + 1
                } else {
                    roots.push(root);
                    roots.len()
                };
                (
                    run.first,
                    run.last,
                    u32::try_from(piece).unwrap_or(u32::MAX),
                )
            })
            .collect();
        Self::Labelled(Spans {
            side: width,
            rows,
            runs: numbered,
        })
    }

    /// The piece of vertex `at` (row-major), or 0 if it is out of reach.
    pub(super) fn of(&self, at: usize) -> u32 {
        match self {
            Self::Whole => 1,
            Self::Labelled(spans) => spans.of(at),
        }
    }
}

impl Spans {
    /// The piece of vertex `at` (row-major), or 0 if it is out of reach.
    fn of(&self, at: usize) -> u32 {
        let (row, column) = (at / self.side, at % self.side);
        let (Some(&from), Some(&to)) = (self.rows.get(row), self.rows.get(row + 1)) else {
            return 0;
        };
        self.runs
            .get(from..to)
            .unwrap_or_default()
            .iter()
            .find(|&&(first, last, _)| first <= column && column <= last)
            .map_or(0, |&(_, _, piece)| piece)
    }
}

/// A run of consecutive vertices within reach in one row: columns `first..=last`, run `id`.
#[derive(Debug, Clone, Copy)]
struct Run {
    /// Its first column.
    first: usize,
    /// Its last column.
    last: usize,
    /// Its number among the tile's runs.
    id: u32,
}

/// The runs of a tile, joined into pieces: a union-find over run numbers.
#[derive(Debug, Default)]
struct Runs {
    /// Each run's parent; a root is its own.
    parent: Vec<u32>,
}

impl Runs {
    /// A new run over columns `first..=last`, a piece of its own.
    fn make(&mut self, first: usize, last: usize) -> Run {
        let id = u32::try_from(self.parent.len()).unwrap_or(u32::MAX);
        self.parent.push(id);
        Run { first, last, id }
    }

    /// The root of `id`'s piece, the path to it shortened on the way.
    fn root(&mut self, id: u32) -> u32 {
        let mut root = id;
        while let Some(&up) = self.parent.get(root as usize)
            && up != root
        {
            root = up;
        }
        let mut at = id;
        while at != root {
            let Some(slot) = self.parent.get_mut(at as usize) else {
                break;
            };
            at = std::mem::replace(slot, root);
        }
        root
    }

    /// Joins `a`'s piece and `b`'s; the lesser root stays, so the result is the same whatever
    /// the order of joins.
    fn join(&mut self, a: u32, b: u32) {
        let (a, b) = (self.root(a), self.root(b));
        let (keep, drop) = if a <= b { (a, b) } else { (b, a) };
        if let Some(slot) = self.parent.get_mut(drop as usize) {
            *slot = keep;
        }
    }
}

/// A tile the walk reached: its pieces, and which of them it entered.
#[derive(Debug)]
pub(super) struct Reached {
    /// The tile's pieces.
    pub(super) pieces: Pieces,
    /// The pieces the walk entered, in the order it entered them.
    pub(super) entered: Vec<u32>,
}

impl Reached {
    /// Whether vertex `at` (row-major) is on a piece the walk entered.
    pub(super) fn holds(&self, at: usize) -> bool {
        let piece = self.pieces.of(at);
        piece != 0 && self.entered.contains(&piece)
    }
}

/// A vertex of a tile: the tile, and the vertex's place in it, row-major.
pub(super) type Vertex = (TileKey, usize);

/// What a walk found: every tile it reached, and whether it reached its target.
#[derive(Debug, Default)]
pub(super) struct Walked {
    /// The tiles reached, by key.
    pub(super) tiles: BTreeMap<TileKey, Reached>,
    /// Whether the target vertex is on a piece the walk entered.
    pub(super) found: bool,
}

/// How the walk reads the surface: the topology, each tile's positions, and whether a tile lies
/// wholly within reach.
pub(super) trait Source {
    /// The error reading a tile can give.
    type Error;

    /// The mesh's topology.
    fn topology(&self) -> &Topology;

    /// Tile `key`'s evaluated positions, row-major.
    ///
    /// # Errors
    ///
    /// If the tile can't be evaluated.
    fn positions(&mut self, key: TileKey) -> Result<&[Vec3], Self::Error>;

    /// Whether every vertex of tile `key` is surely within `reach_squared` of `center` (its
    /// box is); `false` when unsure.
    fn wholly_within(&self, key: &TileKey, center: DVec3, reach_squared: f64) -> bool;
}

/// Walks the surface from vertex `start` over everything joined to it within `reach_squared` of
/// `center` (see the module notes of `surface`). Stops as soon as `target`'s piece is entered,
/// if a target is given.
///
/// # Errors
///
/// As [`Source::positions`].
pub(super) fn walk<S: Source>(
    source: &mut S,
    start: Vertex,
    center: DVec3,
    reach_squared: f64,
    target: Option<Vertex>,
) -> Result<Walked, S::Error> {
    let mut walked = Walked::default();
    let mut queue = VecDeque::from([start]);
    while let Some((key, at)) = queue.pop_front() {
        let side = key.side();
        let reached = match walked.tiles.entry(key) {
            std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::btree_map::Entry::Vacant(entry) => {
                let pieces = if source.wholly_within(&key, center, reach_squared) {
                    Pieces::Whole
                } else {
                    Pieces::label(source.positions(key)?, side, center, reach_squared)
                };
                entry.insert(Reached {
                    pieces,
                    entered: Vec::new(),
                })
            }
        };
        let piece = reached.pieces.of(at);
        if piece == 0 || reached.entered.contains(&piece) {
            continue;
        }
        reached.entered.push(piece);
        if target.is_some_and(|(goal, spot)| goal == key && reached.holds(spot)) {
            walked.found = true;
            return Ok(walked);
        }
        let exits = exits(&reached.pieces, piece, side);
        let topology = source.topology();
        let region = Region::of_tile(&key);
        let level = key.level();
        for (i, j) in exits {
            let copies = topology.copies(key.face(), level, region.x0() + i, region.y0() + j);
            let elsewhere = copies.iter().filter(|copy| copy.key(level) != key);
            for copy in elsewhere {
                let next = copy.key(level);
                let spot = (copy.j * next.side() + copy.i) as usize;
                queue.push_back((next, spot));
            }
        }
    }
    Ok(walked)
}

/// The border vertices through which piece `piece` of `pieces` leaves a tile of `side` vertices
/// a side.
///
/// They are each corner on the piece, and the first vertex of each run of the piece along an edge
/// between its corners.
///
/// One vertex a run is enough: a run's vertices have their copies in the same tile across the
/// edge (on this face or the next: tiles are aligned on both), consecutive along its edge, so
/// they are one piece there too.
fn exits(pieces: &Pieces, piece: u32, side: u32) -> Vec<(u32, u32)> {
    let last = side.saturating_sub(1);
    let on = |(i, j): (u32, u32)| pieces.of((j * side + i) as usize) == piece;
    let corners = [(0, 0), (last, 0), (0, last), (last, last)];
    let mut found: Vec<(u32, u32)> = corners.into_iter().filter(|&v| on(v)).collect();
    let edges: [&dyn Fn(u32) -> (u32, u32); 4] =
        [&|t| (t, 0), &|t| (t, last), &|t| (0, t), &|t| (last, t)];
    for edge in edges {
        let mut in_run = false;
        for t in 1..last {
            let here = on(edge(t));
            if here && !in_run {
                found.push(edge(t));
            }
            in_run = here;
        }
    }
    found
}

/// A ball, in `f32` for the test of every vertex of a tile: whether a vertex is within reach is
/// decided here, the same way wherever it is asked.
#[derive(Debug, Clone, Copy)]
pub(super) struct Ball {
    /// The centre, rounded to `f32`.
    center: Vec3,
    /// The squared radius, rounded to `f32`.
    radius_squared: f32,
}

impl Ball {
    /// The ball of squared radius `radius_squared` around `center`, rounded to `f32`.
    pub(super) fn new(center: DVec3, radius_squared: f64) -> Self {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "positions are f32: the test is too, rounded once here"
        )]
        let radius_squared = radius_squared as f32;
        Self {
            center: center.to_f32(),
            radius_squared,
        }
    }

    /// Whether `p` is within the ball: its squared distance from the centre, in `f32`, at most
    /// the squared radius.
    pub(super) fn holds(&self, p: Vec3) -> bool {
        let off = p - self.center;
        off.dot(off) <= self.radius_squared
    }
}

/// The squared distance from `p` to `center`, in `f64`.
pub(super) fn distance_squared(p: Vec3, center: DVec3) -> f64 {
    let off = DVec3::from_f32(p) - center;
    off.dot(off)
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    /// A flat `side × side` grid of unit spacing at `z = 0`, but rows `lifted` raised to
    /// `z = 100`.
    fn grid(side: u32, lifted: std::ops::Range<u32>) -> Vec<Vec3> {
        (0..side)
            .flat_map(|j| {
                let z = if lifted.contains(&j) { 100.0 } else { 0.0 };
                let row = f32::from(u16::try_from(j).unwrap());
                (0..side).map(move |i| Vec3::new(f32::from(u16::try_from(i).unwrap()), row, z))
            })
            .collect()
    }

    /// One tile of face 0 at level 6 (the whole face, 65 vertices a side) with `positions`,
    /// over the cube's topology.
    struct Fake {
        /// The cube, for its topology.
        mesh: geom::BaseMesh,
        /// The tile's positions.
        positions: Vec<Vec3>,
    }

    impl Source for Fake {
        type Error = ();

        fn topology(&self) -> &Topology {
            self.mesh.topology()
        }

        fn positions(&mut self, _: TileKey) -> Result<&[Vec3], ()> {
            Ok(&self.positions)
        }

        fn wholly_within(&self, _: &TileKey, _: DVec3, _: f64) -> bool {
            false
        }
    }

    /// A hairpin fold inside one tile: rows `0..32` run along y from 0 to 1 at `z = 0`, row 32
    /// bends out to `y = 1.1`, and rows `33..65` run back from 1 to 0 at `z = 0.01`.
    fn hairpin() -> Vec<Vec3> {
        (0..65_u16)
            .flat_map(|j| {
                let (y, z) = match j {
                    0..32 => (f32::from(j) / 31.0, 0.0),
                    32 => (1.1, 0.005),
                    _ => (f32::from(64 - j) / 31.0, 0.01),
                };
                (0..65_u16).map(move |i| Vec3::new(f32::from(i) / 64.0, y, z))
            })
            .collect()
    }

    #[test]
    fn the_two_sides_of_a_fold_in_one_tile_join_only_round_its_bend() {
        let mut fake = Fake {
            mesh: geom::cube_sphere(1).unwrap(),
            positions: hairpin(),
        };
        let key = TileKey::displacement(geom::FaceId::new(0), geom::Level::new(6).unwrap(), 0, 0)
            .unwrap();
        // Column 32, row 9 (y = 0.29) on the first side and row 55 (y = 0.29) on the second.
        let (a, b) = (9 * 65 + 32, 55 * 65 + 32);
        let near = DVec3::new(0.5, 0.29, 0.0);
        let across = walk(&mut fake, (key, a), near, 0.05 * 0.05, Some((key, b))).unwrap();
        assert!(
            !across.found,
            "the sides are 0.01 apart but don't join near here"
        );
        let round = walk(&mut fake, (key, a), near, 1.0, Some((key, b))).unwrap();
        assert!(round.found, "a reach past the bend joins them");
    }

    #[test]
    fn a_band_out_of_reach_splits_a_tile_in_two() {
        let positions = grid(9, 4..5);
        let centre = DVec3::new(4.0, 4.0, 0.0);
        let pieces = Pieces::label(&positions, 9, centre, 100.0);
        assert_eq!(pieces.of(0), 1, "the rows below the band");
        assert_eq!(pieces.of(9 * 8), 2, "the rows above it, apart");
        assert_eq!(pieces.of(9 * 4 + 4), 0, "the band itself is out of reach");
        let below = exits(&pieces, 1, 9);
        assert!(
            below.iter().all(|&(_, j)| j < 4),
            "the lower piece leaves the tile below the band only: {below:?}"
        );
    }

    #[test]
    fn a_diagonal_joins_runs_a_column_apart() {
        // Only (1, 0) and (2, 1) within reach: joined by the quad's diagonal.
        let positions = grid(4, 0..0);
        let mut far = positions.clone();
        for (k, p) in far.iter_mut().enumerate() {
            if k != 1 && k != 4 + 2 {
                p.z = 100.0;
            }
        }
        let centre = DVec3::new(1.5, 0.5, 0.0);
        let pieces = Pieces::label(&far, 4, centre, 1.0);
        assert_eq!(pieces.of(1), 1, "the first vertex");
        assert_eq!(pieces.of(4 + 2), 1, "joined across the diagonal");
        // The other diagonal, (2, 0) to (1, 1), isn't an edge of the triangles.
        let mut crossed = positions;
        for (k, p) in crossed.iter_mut().enumerate() {
            if k != 2 && k != 4 + 1 {
                p.z = 100.0;
            }
        }
        let apart = Pieces::label(&crossed, 4, centre, 1.0);
        assert_ne!(apart.of(2), apart.of(4 + 1), "not joined");
    }
}
