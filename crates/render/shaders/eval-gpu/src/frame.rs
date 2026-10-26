//! Tangent frames and normals: `geom/src/frame.rs`'s formula and policy, one vertex at a time.
//!
//! Which reads a vertex takes (its stencil) follows `geom`'s `stencil`: a base vertex reads its
//! ring, a vertex on the face's border central differences along its owner's axes, any other
//! along the block's own.
//!
//! The types are plain structs of scalars where `geom` has enums with array payloads: rust-gpu
//! compiles matching on those into pointer casts SPIR-V can't express.

use crate::math::{Vec3, normalizable};

/// No block corner: the vertex isn't a base vertex.
pub(crate) const NO_CORNER: u32 = 4;

/// Not on a face edge: the vertex takes the block's own axes.
pub(crate) const OWN_AXES: u32 = 4;

/// Which reads a vertex takes: `geom`'s `Stencil`, without `Outside` (every block vertex lies on
/// its face).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Stencil {
    /// The block corner (`0..4`) of a base vertex, which reads its ring; else [`NO_CORNER`].
    pub(crate) corner: u32,
    /// The face edge (`0..4`) whose owner axes a vertex on the face's border takes; else
    /// [`OWN_AXES`].
    pub(crate) edge: u32,
}

/// The stencil of block vertex `(i, j)`, for a block at `(x0, y0)` of `quads` quads on a face of
/// `n` quads.
pub(crate) const fn stencil([n, x0, y0, quads]: [u32; 4], i: u32, j: u32) -> Stencil {
    let (fi, fj) = (x0 + i, y0 + j);
    let on_i = fi == 0 || fi == n;
    let on_j = fj == 0 || fj == n;
    if on_i && on_j {
        let corner = match (i == quads, j == quads) {
            (false, false) => 0,
            (true, false) => 1,
            (true, true) => 2,
            (false, true) => 3,
        };
        return Stencil {
            corner,
            edge: OWN_AXES,
        };
    }
    // `geom`'s `border_edge`: which face edge a vertex on the border lies on.
    let edge = if !(on_i || on_j) {
        OWN_AXES
    } else if fj == 0 {
        0
    } else if fi == n {
        1
    } else if fj == n {
        2
    } else {
        3
    };
    Stencil {
        corner: NO_CORNER,
        edge,
    }
}

/// What a vertex's frame and normal are built from: `geom`'s `Local`, as one struct.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Local {
    /// The derivative along the owner's u axis (central differences), or the owner's first edge
    /// out of a base vertex.
    du: Vec3,
    /// The unnormalized normal: `du × dv`, or the ring normal.
    normal: Vec3,
    /// Whether the inputs were finite (always, for a ring).
    finite: bool,
    /// Whether there was anything to build from: `false` for `geom`'s `Local::NonFinite`.
    known: bool,
}

impl Local {
    /// Central differences from the reads `plus_u, minus_u, plus_v, minus_v`.
    pub(crate) fn central(plus_u: Vec3, minus_u: Vec3, plus_v: Vec3, minus_v: Vec3) -> Self {
        let du = plus_u - minus_u;
        let dv = plus_v - minus_v;
        Self {
            du,
            normal: du.cross(dv),
            finite: du.is_finite() && dv.is_finite(),
            known: true,
        }
    }

    /// A base vertex without a finite ring: its frame and normal are NaN.
    pub(crate) const fn unknown() -> Self {
        Self {
            du: Vec3::ZERO,
            normal: Vec3::ZERO,
            finite: false,
            known: false,
        }
    }

    /// A base vertex's local axes from its owner's first edge out of it and its ring normal.
    pub(crate) const fn ring(du: Vec3, normal: Vec3) -> Self {
        Self {
            du,
            normal,
            finite: true,
            known: true,
        }
    }

    /// The direction of the owner's first edge out of base vertex `center`, from its edge
    /// neighbours `e0` (along the owner's outgoing edge) and `e1` (its incoming edge).
    pub(crate) fn first_edge(center: Vec3, e0: Vec3, e1: Vec3, owner_corner: u32) -> Vec3 {
        match owner_corner & 3 {
            0 => e0 - center,
            1 => center - e1,
            2 => center - e0,
            _ => e1 - center,
        }
    }
}

/// A base vertex's [`Local`] from its centre and ring points: `geom`'s `CornerAxes::local`.
///
/// `corner_local!(center, valence, owner_corner, |k| point)` evaluates `point` for edge
/// neighbour `k` of `valence`, owner first, as often as it needs one. A macro rather than a
/// function taking a closure: rust-gpu can't compile a closure that captures a buffer.
macro_rules! corner_local {
    ($center:expr, $valence:expr, $owner_corner:expr, | $k:ident | $point:expr) => {{
        let center: $crate::math::Vec3 = $center;
        let valence: u32 = $valence;
        let mut finite = center.is_finite();
        let mut next = 0_u32;
        while next < valence {
            let $k = next;
            finite = finite && $point.is_finite();
            next += 1;
        }
        if !finite || valence < 2 {
            $crate::frame::Local::unknown()
        } else {
            let e0 = {
                let $k = 0_u32;
                $point
            };
            let e1 = {
                let $k = 1_u32;
                $point
            };
            let du = $crate::frame::Local::first_edge(center, e0, e1, $owner_corner);
            // Σ_k (e_k − v) × (e_{k+1} − v), from the owner, wrapping.
            let mut normal = $crate::math::Vec3::ZERO;
            let mut at = 0_u32;
            while at < valence {
                let a = {
                    let $k = at;
                    $point
                };
                let b = {
                    let $k = (at + 1) % valence;
                    $point
                };
                normal = normal + (a - center).cross(b - center);
                at += 1;
            }
            $crate::frame::Local::ring(du, normal)
        }
    }};
}
pub(crate) use corner_local;

/// A tangent frame: `geom::Frame`'s axes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Frame {
    /// The tangent, along the owner's u direction.
    t: Vec3,
    /// The bitangent, `n × t`.
    b: Vec3,
    /// The unit normal.
    n: Vec3,
}

impl Frame {
    /// The fallback frame, used where finite inputs don't normalize: the same everywhere, so
    /// every copy of a vertex agrees.
    const FALLBACK: Self = Self {
        t: Vec3::new(1.0, 0.0, 0.0),
        b: Vec3::new(0.0, 1.0, 0.0),
        n: Vec3::new(0.0, 0.0, 1.0),
    };

    /// Every axis `nan`.
    const fn nan(nan: Vec3) -> Self {
        Self {
            t: nan,
            b: nan,
            n: nan,
        }
    }

    /// The frame from `local`, by `geom`'s formula and policy: the formula where both `n` and
    /// `t` normalize, else the fallback where the inputs were finite, else `nan` everywhere.
    pub(crate) fn from_local(local: Local, nan: Vec3) -> Self {
        if !local.known {
            return Self::nan(nan);
        }
        let n_length = local.normal.length();
        let n = scaled(local.normal, n_length);
        let along = local.du.dot(n);
        let t = local.du - n * along;
        let t_length = t.length();
        let t = scaled(t, t_length);
        let b = n.cross(t);
        if normalizable(n_length) & normalizable(t_length) {
            Self { t, b, n }
        } else if local.finite {
            Self::FALLBACK
        } else {
            Self::nan(nan)
        }
    }

    /// `position + t·c.x + b·c.y + n·c.z`, in that order: `geom`'s `Frame::apply`.
    pub(crate) fn apply(self, position: Vec3, c: Vec3) -> Vec3 {
        position + self.t * c.x + self.b * c.y + self.n * c.z
    }
}

/// `v` scaled by the reciprocal of `length`: `geom`'s normalization.
fn scaled(v: Vec3, length: f32) -> Vec3 {
    v * (1.0 / length)
}

/// The unit normal from `local`, by `geom`'s policy: the formula where it normalizes, else the
/// fallback where the inputs were finite, else `nan`.
pub(crate) fn normal(local: Local, nan: Vec3) -> Vec3 {
    if !local.known {
        return nan;
    }
    let length = local.normal.length();
    if normalizable(length) {
        scaled(local.normal, length)
    } else if local.finite {
        Frame::FALLBACK.n
    } else {
        nan
    }
}
