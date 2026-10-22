//! Tangent frames for corrections, taken in each shared vertex's owner orientation.

use crate::subdiv::Padded;
use crate::topology::{Topology, border_edge, owner_axes};
use crate::{FaceId, Vec3};

/// An orthonormal frame: `t` along the owner's grid u direction, `n` the surface normal,
/// `b = n × t`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frame {
    t: Vec3,
    b: Vec3,
    n: Vec3,
}

impl Frame {
    /// Used where the surface is degenerate. It is the same everywhere, so twins still agree.
    pub const FALLBACK: Self = Self {
        t: Vec3::new(1.0, 0.0, 0.0),
        b: Vec3::new(0.0, 1.0, 0.0),
        n: Vec3::new(0.0, 0.0, 1.0),
    };
    /// A frame whose every component is NaN: what non-finite derivatives give, so a bug upstream
    /// shows up in the output instead of being hidden behind [`Frame::FALLBACK`].
    const NAN: Self = Self {
        t: Vec3::NAN,
        b: Vec3::NAN,
        n: Vec3::NAN,
    };

    /// The frame from derivatives along u and v: [`Frame::FALLBACK`] where finite derivatives are
    /// degenerate, and all NaN where either derivative isn't finite.
    #[must_use]
    pub fn from_derivatives(du: Vec3, dv: Vec3) -> Self {
        if !(du.is_finite() && dv.is_finite()) {
            return Self::NAN;
        }
        let Some(n) = du.cross(dv).normalized() else {
            return Self::FALLBACK;
        };
        let Some(t) = (du - n * du.dot(n)).normalized() else {
            return Self::FALLBACK;
        };
        Self {
            t,
            b: n.cross(t),
            n,
        }
    }

    /// Tangent: along the owner's grid u direction.
    #[must_use]
    pub const fn t(&self) -> Vec3 {
        self.t
    }

    /// Bitangent: `n × t`.
    #[must_use]
    pub const fn b(&self) -> Vec3 {
        self.b
    }

    /// Normal.
    #[must_use]
    pub const fn n(&self) -> Vec3 {
        self.n
    }

    /// `position + t·c.x + b·c.y + n·c.z`, in that order (the order is part of the seam
    /// guarantee).
    #[must_use]
    pub fn apply(&self, position: Vec3, c: Vec3) -> Vec3 {
        position + self.t * c.x + self.b * c.y + self.n * c.z
    }
}

/// The next level's points around a base vertex, for one-sided derivatives there.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CornerAxes {
    /// The vertex point.
    pub(crate) center: Vec3,
    /// The edge point along the owner's outgoing edge.
    pub(crate) e0: Vec3,
    /// The edge point along the owner's incoming edge.
    pub(crate) e1: Vec3,
    /// The owner face's corner at this vertex.
    pub(crate) owner_corner: u8,
}

/// Where a block sits, so derivatives at shared vertices can be taken in the owner's orientation.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Block<'a> {
    pub(crate) topology: &'a Topology,
    pub(crate) face: FaceId,
    /// Quads per side of the face at this level.
    pub(crate) face_quads: u32,
    pub(crate) x0: u32,
    pub(crate) y0: u32,
    pub(crate) quads: u32,
}

/// `(du, dv)` at block vertex `(i, j)` of `grid`, in the owner's orientation. `corners[c]` holds
/// block corner `c`'s axes when it is a base vertex.
#[expect(
    clippy::many_single_char_names,
    reason = "grid math keeps the conventions' names: n, c, (i, j) and the axes (u, v)"
)]
fn derivatives(
    block: &Block<'_>,
    grid: &Padded,
    corners: &[Option<CornerAxes>; 4],
    i: i64,
    j: i64,
) -> (Vec3, Vec3) {
    let n = block.face_quads;
    let (Ok(fi), Ok(fj)) = (
        u32::try_from(i64::from(block.x0) + i),
        u32::try_from(i64::from(block.y0) + j),
    ) else {
        return (Vec3::NAN, Vec3::NAN);
    };
    let on = |x: u32| x == 0 || x == n;
    if on(fi) && on(fj) {
        let q = i64::from(block.quads);
        let c = match (i == q, j == q) {
            (false, false) => 0,
            (true, false) => 1,
            (true, true) => 2,
            (false, true) => 3,
        };
        return corners
            .get(c)
            .copied()
            .flatten()
            .map_or((Vec3::NAN, Vec3::NAN), corner_derivatives);
    }
    let own = [(1, 0), (0, 1)];
    let [u, v] = if on(fi) || on(fj) {
        let k = border_edge(fi, fj, n);
        match block.topology.across(block.face, k) {
            Some(across) if across.face < block.face => owner_axes(k, across.edge),
            _ => own,
        }
    } else {
        own
    };
    let du = grid.get(i + u.0, j + u.1) - grid.get(i - u.0, j - u.1);
    let dv = grid.get(i + v.0, j + v.1) - grid.get(i - v.0, j - v.1);
    (du, dv)
}

/// One-sided derivatives at a base vertex, along the owner's two edges.
fn corner_derivatives(c: CornerAxes) -> (Vec3, Vec3) {
    let CornerAxes {
        center: v,
        e0,
        e1,
        owner_corner,
    } = c;
    debug_assert!(owner_corner < 4, "a corner index is 0–3");
    match owner_corner & 3 {
        0 => (e0 - v, e1 - v),
        1 => (v - e1, e0 - v),
        2 => (v - e0, v - e1),
        _ => (e1 - v, v - e0),
    }
}

/// Frames for every vertex of a predicted block, row-major `(quads + 1)²`.
pub(crate) fn frames(
    block: &Block<'_>,
    predicted: &Padded,
    corners: &[Option<CornerAxes>; 4],
) -> Vec<Frame> {
    let q = i64::from(block.quads);
    (0..=q)
        .flat_map(|j| (0..=q).map(move |i| (i, j)))
        .map(|(i, j)| {
            let (du, dv) = derivatives(block, predicted, corners, i, j);
            debug_assert!(
                du.is_finite() && dv.is_finite(),
                "derivatives at ({i}, {j}) of {block:?} are not finite"
            );
            Frame::from_derivatives(du, dv)
        })
        .collect()
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    #[test]
    fn frames_are_orthonormal_and_right_handed() {
        let f = Frame::from_derivatives(Vec3::new(2.0, 0.1, 0.0), Vec3::new(0.0, 3.0, 0.2));
        assert!((f.t.length() - 1.0).abs() < 1e-6, "t unit");
        assert!((f.n.length() - 1.0).abs() < 1e-6, "n unit");
        assert!(f.t.dot(f.n).abs() < 1e-6, "t ⟂ n");
        assert!((f.b - f.n.cross(f.t)).length() < 1e-6, "b = n × t");
        assert!(f.n.z > 0.0, "normal follows du × dv");
    }

    #[test]
    fn degenerate_derivatives_fall_back_deterministically() {
        assert_eq!(
            Frame::from_derivatives(Vec3::ZERO, Vec3::ZERO),
            Frame::FALLBACK,
            "zero"
        );
        let nan = Frame::from_derivatives(Vec3::NAN, Vec3::new(0.0, 1.0, 0.0));
        assert!(
            !(nan.t.is_finite() || nan.b.is_finite() || nan.n.is_finite()),
            "NaN propagates instead of falling back: {nan:?}"
        );
        assert!(
            !Frame::from_derivatives(Vec3::new(1.0, 0.0, 0.0), Vec3::new(f32::INFINITY, 0.0, 0.0))
                .n
                .is_finite(),
            "so does infinity"
        );
    }

    #[test]
    fn apply_adds_in_frame_order() {
        let f = Frame::FALLBACK;
        assert_eq!(
            f.apply(Vec3::ZERO, Vec3::new(1.0, 2.0, 3.0)),
            Vec3::new(1.0, 2.0, 3.0),
            "identity frame"
        );
    }
}
