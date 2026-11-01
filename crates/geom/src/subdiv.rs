//! Catmull–Clark prediction of one level from the level above, block by block.
//!
//! Every stencil sums commutative pairs in a fixed grouping, so a neighbouring face computing the
//! same vertex in its own orientation gets bit-identical results (spec, "Seams are exact").

use crate::{MAX_VALENCE, Vec3};

/// A square block of one level's grid plus a one-vertex ring: indices `-1 ..= quads + 1` on both
/// axes. Slots without a value (the ring's four corners) hold NaN.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Padded {
    /// Quads per side of the block, without the ring.
    quads: u32,
    /// The `(quads + 3)²` slots, row-major from `(-1, -1)`.
    data: Vec<Vec3>,
}

impl Padded {
    /// A block of `quads` quads per side with every slot, ring included, set to `value`.
    pub(crate) fn filled(quads: u32, value: Vec3) -> Self {
        let side = usize::try_from(quads).unwrap_or(0) + 3;
        Self {
            quads,
            data: vec![value; side * side],
        }
    }

    /// Quads per side, without the ring.
    pub(crate) const fn quads(&self) -> u32 {
        self.quads
    }

    /// How many slots the block has, ring included: `(quads + 3)²`.
    #[cfg(feature = "bench-stages")]
    pub(crate) const fn len(&self) -> usize {
        self.data.len()
    }

    /// Every slot, ring included, row-major from `(-1, -1)`.
    pub(crate) fn as_slice(&self) -> &[Vec3] {
        &self.data
    }

    /// Every slot, ring included, row-major from `(-1, -1)`, writable.
    pub(crate) fn as_mut_slice(&mut self) -> &mut [Vec3] {
        &mut self.data
    }

    /// The bytes the block holds on the heap: its slots' allocation.
    pub(crate) fn heap_bytes(&self) -> usize {
        self.data.capacity().saturating_mul(size_of::<Vec3>())
    }

    /// Row `j` (`-1 ..= quads + 1`) with its ring: `quads + 3` slots, column `i` at index `i + 1`.
    /// Empty outside the block and its ring.
    pub(crate) fn row(&self, j: i64) -> &[Vec3] {
        let side = usize::try_from(self.quads)
            .unwrap_or(usize::MAX)
            .saturating_add(3);
        usize::try_from(j + 1)
            .ok()
            .and_then(|r| self.data.get(r * side..(r + 1) * side))
            .unwrap_or_default()
    }

    /// Row `j` (`-1 ..= quads + 1`) with its ring, writable: as [`Padded::row`].
    pub(crate) fn row_mut(&mut self, j: i64) -> &mut [Vec3] {
        let side = usize::try_from(self.quads)
            .unwrap_or(usize::MAX)
            .saturating_add(3);
        usize::try_from(j + 1)
            .ok()
            .and_then(|r| self.data.get_mut(r * side..(r + 1) * side))
            .unwrap_or_default()
    }

    /// Where `(i, j)` is in `data`; `None` outside the block and its ring.
    fn index(&self, i: i64, j: i64) -> Option<usize> {
        let side = i64::from(self.quads) + 3;
        let (pi, pj) = (i + 1, j + 1);
        if (0..side).contains(&pi) && (0..side).contains(&pj) {
            usize::try_from(pj * side + pi).ok()
        } else {
            None
        }
    }

    /// The value at `(i, j)`; NaN outside the block and its ring.
    pub(crate) fn get(&self, i: i64, j: i64) -> Vec3 {
        self.index(i, j)
            .and_then(|k| self.data.get(k))
            .copied()
            .unwrap_or(Vec3::NAN)
    }

    /// Sets the value at `(i, j)`, which must be in the block or its ring.
    pub(crate) fn set(&mut self, i: i64, j: i64, v: Vec3) {
        debug_assert!(
            self.index(i, j).is_some(),
            "({i}, {j}) outside a padded block of {} quads",
            self.quads
        );
        if let Some(slot) = self.index(i, j).and_then(|k| self.data.get_mut(k)) {
            *slot = v;
        }
    }
}

/// The one-ring of a base vertex at one level, owner first (see `Topology`'s rings). Quad `k` is
/// `(center, edges[k], diagonals[k], edges[k + 1])`.
///
/// Held in fixed arrays of [`MAX_VALENCE`] (the mesh rejects higher valences), so gathering a
/// ring doesn't allocate.
#[derive(Debug, Clone)]
pub(crate) struct CornerRing {
    /// The base vertex itself.
    pub(crate) center: Vec3,
    /// The edge neighbours, one per face around the vertex, in ring order; NaN past `valence`.
    edges: [Vec3; MAX_VALENCE],
    /// The diagonal neighbours (each face's vertex opposite the centre), in ring order; NaN past
    /// `valence`.
    diagonals: [Vec3; MAX_VALENCE],
    /// How many of `edges` and `diagonals` are the ring's.
    valence: usize,
    /// The owner face's corner at this vertex: tells frames which ring points lie along the
    /// owner's axes.
    pub(crate) owner_corner: u8,
}

impl CornerRing {
    /// A ring with no points yet.
    pub(crate) const fn new(center: Vec3, owner_corner: u8) -> Self {
        Self {
            center,
            edges: [Vec3::NAN; MAX_VALENCE],
            diagonals: [Vec3::NAN; MAX_VALENCE],
            valence: 0,
            owner_corner,
        }
    }

    /// Appends quad `k`'s edge and diagonal points. A ring holds at most [`MAX_VALENCE`];
    /// `BaseMesh::from_polygons` guarantees no ring is longer.
    pub(crate) fn push(&mut self, edge: Vec3, diagonal: Vec3) {
        let k = self.valence;
        debug_assert!(k < MAX_VALENCE, "valence above MAX_VALENCE");
        if let (Some(e), Some(d)) = (self.edges.get_mut(k), self.diagonals.get_mut(k)) {
            *e = edge;
            *d = diagonal;
            self.valence = k + 1;
        }
    }

    /// The edge neighbours, in ring order.
    pub(crate) fn edges(&self) -> &[Vec3] {
        self.edges.get(..self.valence).unwrap_or_default()
    }

    /// The diagonal neighbours, in ring order.
    pub(crate) fn diagonals(&self) -> &[Vec3] {
        self.diagonals.get(..self.valence).unwrap_or_default()
    }

    /// Every point of the ring, the centre first, then the edge and the diagonal neighbours,
    /// writable.
    pub(crate) fn points_mut(&mut self) -> impl Iterator<Item = &mut Vec3> {
        let valence = self.valence;
        let edges = self.edges.iter_mut().take(valence);
        let diagonals = self.diagonals.iter_mut().take(valence);
        std::iter::once(&mut self.center)
            .chain(edges)
            .chain(diagonals)
    }
}

/// Face point; `a`/`c` and `b`/`d` are the quad's diagonals.
pub(crate) fn face_point(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> Vec3 {
    ((a + c) + (b + d)) * 0.25
}

/// Edge point of edge `p`–`q`; `(a1, a2)` are the far vertices of the quad on one side, `(b1, b2)`
/// of the other.
pub(crate) fn edge_point(p: Vec3, q: Vec3, a1: Vec3, a2: Vec3, b1: Vec3, b2: Vec3) -> Vec3 {
    ((p + q) * 6.0 + ((a1 + a2) + (b1 + b2))) * (1.0 / 16.0)
}

/// Regular (valence 4) vertex point. `e` and `d` are in rotation order, so `e[0]`/`e[2]`,
/// `e[1]`/`e[3]` (and the same for `d`) are opposite pairs.
pub(crate) fn vertex_point(v: Vec3, e: [Vec3; 4], d: [Vec3; 4]) -> Vec3 {
    let [e1, e2, e3, e4] = e;
    let [d1, d2, d3, d4] = d;
    (v * 36.0 + ((e1 + e3) + (e2 + e4)) * 6.0 + ((d1 + d3) + (d2 + d4))) * (1.0 / 64.0)
}

/// Vertex point at a base vertex of any valence `n`:
/// `V·(4n − 7)/(4n) + Σe·3/(2n²) + Σd·1/(4n²)`, summed in the ring's canonical order.
pub(crate) fn extraordinary(ring: &CornerRing) -> Vec3 {
    let n = f32::from(u16::try_from(ring.edges().len()).unwrap_or(u16::MAX));
    let sum = |vs: &[Vec3]| vs.iter().fold(Vec3::ZERO, |s, v| s + *v);
    ring.center * ((4.0 * n - 7.0) / (4.0 * n))
        + sum(ring.edges()) * (3.0 / (2.0 * n * n))
        + sum(ring.diagonals()) * (1.0 / (4.0 * n * n))
}

/// The next level's edge point on the ring's edge `k`, from the ring alone. Bit-identical to the
/// same point computed inside any face's grid (same pairs).
pub(crate) fn ring_edge_point(ring: &CornerRing, k: usize) -> Vec3 {
    let (edges, diagonals) = (ring.edges(), ring.diagonals());
    let n = edges.len().max(1);
    let e = |x: usize| edges.get(x % n).copied().unwrap_or(Vec3::NAN);
    let d = |x: usize| diagonals.get(x % n).copied().unwrap_or(Vec3::NAN);
    // Quad k: (v, e_k, d_k, e_{k+1}); quad k − 1: (v, e_{k−1}, d_{k−1}, e_k).
    edge_point(
        ring.center,
        e(k),
        d(k),
        e(k + 1),
        e(k + n - 1),
        d(k + n - 1),
    )
}

/// Predicts the next level for a block.
///
/// `parent` is the block and its ring at level `l − 1`, `corners[c]` the ring of the base vertex
/// at block corner `c` (0: `(0,0)`, 1: `(q,0)`, 2: `(q,q)`, 3: `(0,q)`) when that corner is one.
/// Returns the block at level `l` (twice the quads) with its ring; ring slots that would need a
/// missing diagonal are NaN.
///
/// Each child row is computed from two or three parent rows as slices: child row `2j`
/// ([`vertex_row`]) from parent rows `j − 1`, `j` and `j + 1`, child row `2b + 1`
/// ([`face_row`]) from parent rows `b` and `b + 1`. Every read lies in the padded parent block.
pub(crate) fn predict(parent: &Padded, corners: &[Option<CornerRing>; 4]) -> Padded {
    let q = i64::from(parent.quads());
    let big = 2 * q;
    let mut out = Padded::filled(parent.quads() * 2, Vec3::NAN);
    for j in 0..=q {
        let rows = [parent.row(j - 1), parent.row(j), parent.row(j + 1)];
        vertex_row(rows, out.row_mut(2 * j));
    }
    for b in -1..=q {
        face_row([parent.row(b), parent.row(b + 1)], out.row_mut(2 * b + 1));
    }
    // The ring rows' end slots, diagonally off the block's corners, have no point.
    for (i, j) in [(-1, -1), (big + 1, -1), (-1, big + 1), (big + 1, big + 1)] {
        out.set(i, j, Vec3::NAN);
    }
    for (c, (i, j)) in [(0, 0), (big, 0), (big, big), (0, big)]
        .into_iter()
        .enumerate()
    {
        if let Some(ring) = corners.get(c).and_then(Option::as_ref) {
            out.set(i, j, extraordinary(ring));
        }
    }
    out
}

/// Child row `2j` (padded, ring included) from parent rows `[j − 1, j, j + 1]` (padded): an edge
/// point at every odd child column and a regular vertex point at every even one.
///
/// Child columns `2i − 1` and `2i` read the parent's columns `i − 1 ..= i + 1`, a three-wide
/// window; the last child column reads the parent's last two. Base vertices get their own rule
/// afterwards, in [`predict`].
fn vertex_row([up, mid, down]: [&[Vec3]; 3], out: &mut [Vec3]) {
    let (pairs, rest) = out.as_chunks_mut::<2>();
    let windows = up
        .array_windows::<3>()
        .zip(mid.array_windows())
        .zip(down.array_windows());
    for ([edge, vertex], (([ul, uc, ur], [ml, mc, mr]), [dl, dc, dr])) in
        pairs.iter_mut().zip(windows)
    {
        *edge = edge_point(*ml, *mc, *ul, *uc, *dl, *dc);
        *vertex = vertex_point(*mc, [*mr, *dc, *ml, *uc], [*dr, *dl, *ul, *ur]);
    }
    let last = (up.last_chunk(), mid.last_chunk(), down.last_chunk());
    if let ([edge], (Some([ul, ur]), Some([ml, mr]), Some([dl, dr]))) = (rest, last) {
        *edge = edge_point(*ml, *mr, *ul, *ur, *dl, *dr);
    }
}

/// Child row `2b + 1` (padded, ring included) from parent rows `[b, b + 1]` (padded): a face
/// point at every odd child column and an edge point at every even one.
///
/// Child columns `2i − 1` and `2i` read the parent's columns `i − 1 ..= i + 1`, a three-wide
/// window; the last child column reads the parent's last two.
fn face_row([lo, hi]: [&[Vec3]; 2], out: &mut [Vec3]) {
    let (pairs, rest) = out.as_chunks_mut::<2>();
    let windows = lo.array_windows::<3>().zip(hi.array_windows());
    for ([face, edge], ([ll, lc, lr], [hl, hc, hr])) in pairs.iter_mut().zip(windows) {
        *face = face_point(*ll, *lc, *hc, *hl);
        *edge = edge_point(*lc, *hc, *ll, *hl, *lr, *hr);
    }
    if let ([face], Some([ll, lr]), Some([hl, hr])) = (rest, lo.last_chunk(), hi.last_chunk()) {
        *face = face_point(*ll, *lr, *hr, *hl);
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;
    use crate::test_rng::Rng;

    /// A ring with owner corner 0 from its points.
    fn ring_of(center: Vec3, edges: &[Vec3], diagonals: &[Vec3]) -> CornerRing {
        let mut ring = CornerRing::new(center, 0);
        for (&e, &d) in edges.iter().zip(diagonals) {
            ring.push(e, d);
        }
        ring
    }

    #[test]
    fn rings_hold_up_to_max_valence() {
        let mut ring = CornerRing::new(Vec3::ZERO, 0);
        for k in 0..MAX_VALENCE {
            let x = f32::from(u8::try_from(k).unwrap());
            ring.push(Vec3::new(x, 0.0, 0.0), Vec3::new(0.0, x, 0.0));
        }
        assert_eq!(ring.edges().len(), MAX_VALENCE, "edges");
        assert_eq!(ring.diagonals().len(), MAX_VALENCE, "diagonals");
        assert_eq!(ring.edges()[31].x, 31.0, "last edge");
        assert_eq!(ring.diagonals()[31].y, 31.0, "last diagonal");
    }

    /// A small grid coordinate as `f32`, exactly.
    fn coord(x: i64) -> f32 {
        f32::from(i16::try_from(x).unwrap())
    }

    /// A padded block sampled from an affine function: Catmull–Clark reproduces affine functions
    /// exactly in regular regions, so every predicted point must lie on the same function.
    fn affine(quads: u32) -> (Padded, impl Fn(f32, f32) -> Vec3) {
        let f = |x: f32, y: f32| Vec3::new(0.5 * x - 0.25 * y + 1.0, 0.75 * y, 0.1 * x + 0.2 * y);
        let mut p = Padded::filled(quads, Vec3::NAN);
        let q = i64::from(quads);
        for j in -1..=q + 1 {
            for i in -1..=q + 1 {
                p.set(i, j, f(coord(i), coord(j)));
            }
        }
        (p, f)
    }

    #[test]
    fn regular_blocks_reproduce_affine_functions() {
        let (parent, f) = affine(4);
        let child = predict(&parent, &[None, None, None, None]);
        let q = 8;
        for j in -1..=q + 1 {
            for i in -1..=q + 1 {
                let outside = |x: i64| x < 0 || x > q;
                if outside(i) && outside(j) {
                    continue;
                }
                let want = f(coord(i) * 0.5, coord(j) * 0.5);
                let got = child.get(i, j);
                assert!(
                    (got - want).length() < 1e-5,
                    "({i}, {j}): {got:?} vs {want:?}"
                );
            }
        }
    }

    #[test]
    #[expect(
        clippy::many_single_char_names,
        reason = "a quad's corners and an edge's stencil points keep their textbook names"
    )]
    fn stencils_are_symmetric_under_relabelling() {
        let mut rng = Rng::new(3);
        let mut v = || rng.vec3(1.0);
        let (a, b, c, d) = (v(), v(), v(), v());
        assert_eq!(
            face_point(a, b, c, d).to_bits(),
            face_point(b, c, d, a).to_bits(),
            "rotated quad"
        );
        assert_eq!(
            face_point(a, b, c, d).to_bits(),
            face_point(d, c, b, a).to_bits(),
            "reflected quad"
        );
        let (p, q, a1, a2, b1, b2) = (v(), v(), v(), v(), v(), v());
        let e = edge_point(p, q, a1, a2, b1, b2).to_bits();
        assert_eq!(
            e,
            edge_point(q, p, b2, b1, a2, a1).to_bits(),
            "edge seen from the other face"
        );
    }

    #[test]
    fn extraordinary_rule_matches_regular_at_valence_four() {
        let mut random = Rng::new(9);
        let center = random.vec3(1.0);
        let edges: Vec<Vec3> = (0..4).map(|_| random.vec3(1.0)).collect();
        let diagonals: Vec<Vec3> = (0..4).map(|_| random.vec3(1.0)).collect();
        let ring = ring_of(center, &edges, &diagonals);
        let regular = vertex_point(
            center,
            [edges[0], edges[1], edges[2], edges[3]],
            [diagonals[0], diagonals[1], diagonals[2], diagonals[3]],
        );
        assert!(
            (extraordinary(&ring) - regular).length() < 1e-6,
            "same surface (not necessarily same bits)"
        );
    }

    /// The ring's edge points are the grid's edge points, bit for bit: frames read them from the
    /// ring while the grid computes them with the in-face stencil.
    #[test]
    fn ring_edge_points_match_the_grid() {
        let mut random = Rng::new(17);
        let mut parent = Padded::filled(2, Vec3::NAN);
        for j in -1..=3 {
            for i in -1..=3 {
                parent.set(i, j, random.vec3(1.0));
            }
        }
        let g = |i, j| parent.get(i, j);
        // A regular vertex at block corner 0, as a ring starting at the quad towards +i, +j.
        let ring = ring_of(
            g(0, 0),
            &[g(1, 0), g(0, 1), g(-1, 0), g(0, -1)],
            &[g(1, 1), g(-1, 1), g(-1, -1), g(1, -1)],
        );
        let child = predict(&parent, &[Some(ring.clone()), None, None, None]);
        assert_eq!(
            child.get(0, 0).to_bits(),
            extraordinary(&ring).to_bits(),
            "vertex point"
        );
        assert_eq!(
            child.get(1, 0).to_bits(),
            ring_edge_point(&ring, 0).to_bits(),
            "edge 0"
        );
        assert_eq!(
            child.get(0, 1).to_bits(),
            ring_edge_point(&ring, 1).to_bits(),
            "edge 1"
        );
        assert_eq!(
            child.get(-1, 0).to_bits(),
            ring_edge_point(&ring, 2).to_bits(),
            "edge 2"
        );
        assert_eq!(
            child.get(0, -1).to_bits(),
            ring_edge_point(&ring, 3).to_bits(),
            "edge 3"
        );
    }
}
