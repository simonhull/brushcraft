//! Catmull–Clark prediction of one level from the level above, block by block.
//!
//! Every stencil sums commutative pairs in a fixed grouping, so a neighbouring face computing the
//! same vertex in its own orientation gets bit-identical results (spec, "Seams are exact").

use crate::Vec3;

/// A square block of one level's grid plus a one-vertex ring: indices `-1 ..= quads + 1` on both
/// axes. Slots without a value (the ring's four corners) hold NaN.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Padded {
    quads: u32,
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
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CornerRing {
    pub(crate) center: Vec3,
    pub(crate) edges: Vec<Vec3>,
    pub(crate) diagonals: Vec<Vec3>,
    /// The owner face's corner at this vertex: tells frames which ring points lie along the
    /// owner's axes.
    pub(crate) owner_corner: u8,
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
    let n = f32::from(u16::try_from(ring.edges.len()).unwrap_or(u16::MAX));
    let sum = |vs: &[Vec3]| vs.iter().fold(Vec3::ZERO, |s, v| s + *v);
    ring.center * ((4.0 * n - 7.0) / (4.0 * n))
        + sum(&ring.edges) * (3.0 / (2.0 * n * n))
        + sum(&ring.diagonals) * (1.0 / (4.0 * n * n))
}

/// The next level's edge point on the ring's edge `k`, from the ring alone. Bit-identical to the
/// same point computed inside any face's grid (same pairs).
pub(crate) fn ring_edge_point(ring: &CornerRing, k: usize) -> Vec3 {
    let n = ring.edges.len().max(1);
    let e = |x: usize| ring.edges.get(x % n).copied().unwrap_or(Vec3::NAN);
    let d = |x: usize| ring.diagonals.get(x % n).copied().unwrap_or(Vec3::NAN);
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
pub(crate) fn predict(parent: &Padded, corners: &[Option<CornerRing>; 4]) -> Padded {
    let q = i64::from(parent.quads());
    let big = 2 * q;
    let mut out = Padded::filled(parent.quads() * 2, Vec3::NAN);
    let g = |i: i64, j: i64| parent.get(i, j);
    let outside = |x: i64| x < 0 || x > big;
    for jj in -1..=big + 1 {
        for ii in -1..=big + 1 {
            if outside(ii) && outside(jj) {
                continue;
            }
            let value = match (ii.rem_euclid(2) == 1, jj.rem_euclid(2) == 1) {
                (false, false) => {
                    let (i, j) = (ii / 2, jj / 2);
                    corner_ring(corners, q, i, j).map_or_else(
                        || {
                            vertex_point(
                                g(i, j),
                                [g(i + 1, j), g(i, j + 1), g(i - 1, j), g(i, j - 1)],
                                [
                                    g(i + 1, j + 1),
                                    g(i - 1, j + 1),
                                    g(i - 1, j - 1),
                                    g(i + 1, j - 1),
                                ],
                            )
                        },
                        extraordinary,
                    )
                }
                (true, false) => {
                    let (a, b) = ((ii - 1).div_euclid(2), jj / 2);
                    edge_point(
                        g(a, b),
                        g(a + 1, b),
                        g(a, b - 1),
                        g(a + 1, b - 1),
                        g(a, b + 1),
                        g(a + 1, b + 1),
                    )
                }
                (false, true) => {
                    let (a, b) = (ii / 2, (jj - 1).div_euclid(2));
                    edge_point(
                        g(a, b),
                        g(a, b + 1),
                        g(a - 1, b),
                        g(a - 1, b + 1),
                        g(a + 1, b),
                        g(a + 1, b + 1),
                    )
                }
                (true, true) => {
                    let (a, b) = ((ii - 1).div_euclid(2), (jj - 1).div_euclid(2));
                    face_point(g(a, b), g(a + 1, b), g(a + 1, b + 1), g(a, b + 1))
                }
            };
            out.set(ii, jj, value);
        }
    }
    out
}

/// The base-vertex ring at block position `(i, j)`, if that is a block corner with one.
#[expect(
    clippy::many_single_char_names,
    reason = "grid math keeps the conventions' names: q, c and (i, j)"
)]
fn corner_ring(corners: &[Option<CornerRing>; 4], q: i64, i: i64, j: i64) -> Option<&CornerRing> {
    let c = match (i, j) {
        (0, 0) => 0,
        (x, 0) if x == q => 1,
        (x, y) if x == q && y == q => 2,
        (0, y) if y == q => 3,
        _ => return None,
    };
    corners.get(c)?.as_ref()
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;
    use crate::test_rng::Rng;

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
        let ring = CornerRing {
            center,
            edges: edges.clone(),
            diagonals: diagonals.clone(),
            owner_corner: 0,
        };
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
        let ring = CornerRing {
            center: g(0, 0),
            edges: vec![g(1, 0), g(0, 1), g(-1, 0), g(0, -1)],
            diagonals: vec![g(1, 1), g(-1, 1), g(-1, -1), g(1, -1)],
            owner_corner: 0,
        };
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
