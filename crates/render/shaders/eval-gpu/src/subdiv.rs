//! Catmull–Clark stencils: `geom/src/subdiv.rs`'s, with its summation grouping, so a vertex
//! computed by neighbouring tiles (in their own orientations) gets the same bits.

use crate::math::Vec3;
use crate::{Ring, load};

/// Face point; `a`/`c` and `b`/`d` are the quad's diagonals.
pub(crate) fn face_point(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> Vec3 {
    ((a + c) + (b + d)) * 0.25
}

/// Edge point of edge `p`–`q`; `(a1, a2)` are the far vertices of the quad on one side, `(b1,
/// b2)` of the other.
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
/// `V·(4n − 7)/(4n) + Σe·3/(2n²) + Σd·1/(4n²)`, summed in the ring's order from zero.
pub(crate) fn extraordinary(rings: &[[f32; 3]], ring: Ring) -> Vec3 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "a valence is at most 32, exact in f32"
    )]
    let n = ring.valence as f32;
    let mut edges = Vec3::ZERO;
    let mut diagonals = Vec3::ZERO;
    let mut k = 0;
    while k < ring.valence {
        edges = edges + ring.edge(rings, k);
        diagonals = diagonals + ring.diagonal(rings, k);
        k += 1;
    }
    load(rings, ring.offset) * ((4.0 * n - 7.0) / (4.0 * n))
        + edges * (3.0 / (2.0 * n * n))
        + diagonals * (1.0 / (4.0 * n * n))
}

/// The next level's edge point on the ring's edge `k`, from the ring alone: bit-identical to the
/// same point computed inside any face's grid (same pairs).
pub(crate) fn ring_edge_point(rings: &[[f32; 3]], ring: Ring, k: u32) -> Vec3 {
    let n = ring.valence;
    // Quad k: (v, e_k, d_k, e_{k+1}); quad k − 1: (v, e_{k−1}, d_{k−1}, e_k).
    edge_point(
        load(rings, ring.offset),
        ring.edge(rings, k % n),
        ring.diagonal(rings, k % n),
        ring.edge(rings, (k + 1) % n),
        ring.edge(rings, (k + n - 1) % n),
        ring.diagonal(rings, (k + n - 1) % n),
    )
}
