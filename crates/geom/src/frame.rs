//! Tangent frames for corrections, taken in each shared vertex's owner orientation.

use core::ops::RangeInclusive;

use crate::math::normalizable;
use crate::soa::Vectors;
use crate::subdiv::Padded;
use crate::topology::{Topology, border_edge, owner_axes};
use crate::{FaceId, Vec3};

/// An orthonormal frame: `t` along the owner's grid u direction, `n` the surface normal,
/// `b = n × t`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frame {
    /// The tangent: along the owner's grid u direction, in the tangent plane.
    t: Vec3,
    /// The bitangent: `n × t`.
    b: Vec3,
    /// The unit surface normal.
    n: Vec3,
}

impl Frame {
    /// Used where the surface is degenerate. It is the same everywhere, so twins still agree.
    pub(crate) const FALLBACK: Self = Self {
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
    pub(crate) fn from_derivatives(du: Vec3, dv: Vec3) -> Self {
        if !(du.is_finite() && dv.is_finite()) {
            return Self::NAN;
        }
        Self::from_tangent_and_normal(du, du.cross(dv))
    }

    /// The frame from a tangent direction `du` and an unnormalized normal, both computed from
    /// finite inputs: `n` the normal normalized, `t` `du` projected onto its plane, or
    /// [`Frame::FALLBACK`] where either is degenerate.
    fn from_tangent_and_normal(du: Vec3, normal: Vec3) -> Self {
        let Some(n) = normal.normalized() else {
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

    /// The frame for a vertex's [`Local`] axes.
    fn from_local(local: Local) -> Self {
        match local {
            Local::Derivatives { du, dv } => Self::from_derivatives(du, dv),
            Local::Ring { du, normal } => Self::from_tangent_and_normal(du, normal),
            Local::NonFinite => Self::NAN,
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

/// A base vertex and its ring of edge neighbours at one level, for the frame and normal there.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CornerAxes<'a> {
    /// The base vertex.
    pub(crate) center: Vec3,
    /// Its edge neighbours in the ring's canonical order (owner first, see `CornerRing`):
    /// `points[0]` lies along the owner's outgoing edge, `points[1]` along its incoming edge.
    pub(crate) points: &'a [Vec3],
    /// The owner face's corner at this vertex.
    pub(crate) owner_corner: u8,
}

impl CornerAxes<'_> {
    /// The owner's first-edge direction and the ring normal, or [`Local::NonFinite`] if any input
    /// isn't finite.
    pub(crate) fn local(&self) -> Local {
        let finite = self.center.is_finite() && self.points.iter().all(|p| p.is_finite());
        let (Some(&e0), Some(&e1), true) = (self.points.first(), self.points.get(1), finite) else {
            return Local::NonFinite;
        };
        Local::Ring {
            du: self.first_edge(e0, e1),
            normal: self.ring_normal(),
        }
    }

    /// The one-sided derivative along the owner's u axis, from its edge neighbours `e0` (along
    /// its outgoing edge) and `e1` (along its incoming edge).
    fn first_edge(&self, e0: Vec3, e1: Vec3) -> Vec3 {
        let v = self.center;
        debug_assert!(self.owner_corner < 4, "a corner index is 0–3");
        match self.owner_corner & 3 {
            0 => e0 - v,
            1 => v - e1,
            2 => v - e0,
            _ => e1 - v,
        }
    }

    /// `Σ_k (e_k − v) × (e_{k+1} − v)` over the whole ring, `k` from the owner, wrapping. Every
    /// copy of the vertex sums the same terms in the same order, so all get the same bits.
    fn ring_normal(&self) -> Vec3 {
        let v = self.center;
        let next = self.points.iter().cycle().skip(1);
        self.points
            .iter()
            .zip(next)
            .fold(Vec3::ZERO, |sum, (&a, &b)| sum + (a - v).cross(b - v))
    }
}

/// What a vertex's frame and normal are built from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Local {
    /// Derivatives along the owner's u and v: the normal is `du × dv`.
    Derivatives {
        /// The derivative along the owner's u axis.
        du: Vec3,
        /// The derivative along the owner's v axis.
        dv: Vec3,
    },
    /// At a base vertex, from finite inputs: the owner's first-edge direction and the
    /// unnormalized ring normal.
    Ring {
        /// The direction of the owner's first edge out of the vertex: the frame's u axis.
        du: Vec3,
        /// The ring normal, summed over the whole ring, not normalized.
        normal: Vec3,
    },
    /// At a base vertex whose ring isn't finite (or isn't known).
    NonFinite,
}

impl Local {
    /// Whether the inputs were finite: frames assert it, normals pass NaN through.
    const fn is_finite(&self) -> bool {
        match self {
            Self::Derivatives { du, dv } => du.is_finite() && dv.is_finite(),
            Self::Ring { .. } => true,
            Self::NonFinite => false,
        }
    }
}

/// Where a block sits, so derivatives at shared vertices can be taken in the owner's orientation.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Block<'a> {
    /// The mesh's topology, for finding each shared vertex's owner.
    pub(crate) topology: &'a Topology,
    /// The face the block lies on.
    pub(crate) face: FaceId,
    /// Quads per side of the face at this level.
    pub(crate) face_quads: u32,
    /// The block's first column on the face grid.
    pub(crate) x0: u32,
    /// The block's first row on the face grid.
    pub(crate) y0: u32,
    /// Quads per side of the block.
    pub(crate) quads: u32,
}

/// What the frame and normal at one block vertex read: the vertex's stencil.
///
/// Away from base vertices that is four grid neighbours, one step either way along the owner's
/// two axes (never the vertex itself); at a base vertex, its ring. Which one is structural: it
/// depends on the vertex's place on the face and the topology, never on positions. Partial
/// normal updates ([`crate::eval::refresh_normals_into`]) use it to tell which vertices read a
/// position that changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Stencil {
    /// A base vertex (a face corner) at block corner `c`, `0..4` in the order `(0, 0)`,
    /// `(q, 0)`, `(q, q)`, `(0, q)`: it reads its ring.
    Corner(usize),
    /// Central differences along the grid steps `[u, v]`, the owner's `+i` and `+j` in this
    /// block's grid: the vertex `(i, j)` reads `(i, j) ± u` and `(i, j) ± v`.
    Axes([(i64, i64); 2]),
    /// Off the face: there is nothing to read.
    Outside,
}

/// The [`Stencil`] of block vertex `(i, j)`.
pub(crate) fn stencil(block: &Block<'_>, i: i64, j: i64) -> Stencil {
    let n = block.face_quads;
    let (Ok(fi), Ok(fj)) = (
        u32::try_from(i64::from(block.x0) + i),
        u32::try_from(i64::from(block.y0) + j),
    ) else {
        return Stencil::Outside;
    };
    let on = |x: u32| x == 0 || x == n;
    if on(fi) && on(fj) {
        let q = i64::from(block.quads);
        return Stencil::Corner(match (i == q, j == q) {
            (false, false) => 0,
            (true, false) => 1,
            (true, true) => 2,
            (false, true) => 3,
        });
    }
    let own = [(1, 0), (0, 1)];
    let axes = if on(fi) || on(fj) {
        let k = border_edge(fi, fj, n);
        match block.topology.across(block.face, k) {
            Some(across) if across.face < block.face => owner_axes(k, across.edge),
            _ => own,
        }
    } else {
        own
    };
    Stencil::Axes(axes)
}

/// The derivatives from a [`Stencil::Axes`] stencil's four reads.
///
/// They are `du = plus_u − minus_u` and `dv = plus_v − minus_v`. Every caller computes them
/// here, so a vertex gets the same bits however its reads were gathered.
pub(crate) fn central([plus_u, minus_u, plus_v, minus_v]: [Vec3; 4]) -> Local {
    Local::Derivatives {
        du: plus_u - minus_u,
        dv: plus_v - minus_v,
    }
}

/// The [`Local`] axes at block vertex `(i, j)` of `grid`, in the owner's orientation: central
/// differences, or the ring at a base vertex. `corners[c]` holds block corner `c`'s ring when it
/// is a base vertex.
fn derivatives(
    block: &Block<'_>,
    grid: &Padded,
    corners: &[Option<CornerAxes<'_>>; 4],
    i: i64,
    j: i64,
) -> Local {
    match stencil(block, i, j) {
        Stencil::Corner(c) => corners
            .get(c)
            .and_then(Option::as_ref)
            .map_or(Local::NonFinite, CornerAxes::local),
        Stencil::Axes([u, v]) => central([
            grid.get(i + u.0, j + u.1),
            grid.get(i - u.0, j - u.1),
            grid.get(i + v.0, j + v.1),
            grid.get(i - v.0, j - v.1),
        ]),
        Stencil::Outside => Local::NonFinite,
    }
}

/// `f(i, j, local)` for every vertex of a block, row-major `(quads + 1)²`.
fn per_vertex<T>(
    block: &Block<'_>,
    grid: &Padded,
    corners: &[Option<CornerAxes<'_>>; 4],
    f: impl Fn(i64, i64, Local) -> T,
) -> impl Iterator<Item = T> {
    let q = i64::from(block.quads);
    (0..=q)
        .flat_map(move |j| (0..=q).map(move |i| (i, j)))
        .map(move |(i, j)| f(i, j, derivatives(block, grid, corners, i, j)))
}

/// The block vertices strictly inside the face, as `(columns, rows)` in block coordinates (an
/// empty range where there are none).
///
/// Their stencils are all the block's own central differences ([`Stencil::Axes`] with `+i` and
/// `+j`), whose four reads lie in the padded block: rows of them can be walked as slices.
fn inside_face(block: &Block<'_>) -> (RangeInclusive<i64>, RangeInclusive<i64>) {
    let (n, q) = (i64::from(block.face_quads), i64::from(block.quads));
    let span = |origin: u32| {
        let origin = i64::from(origin);
        (1 - origin).max(0)..=(n - 1 - origin).min(q)
    };
    (span(block.x0), span(block.y0))
}

/// The slots of `row` (a [`Padded::row`]) holding columns `columns`, each shifted by `shift`.
///
/// They are the reads of a run of vertices one step along the row (`shift` ±1), or straight
/// across from it (`shift` 0, on the rows above and below).
fn run_of<'a>(row: &'a [Vec3], columns: &RangeInclusive<i64>, shift: i64) -> &'a [Vec3] {
    // Column `i` is at index `i + 1` of a padded row.
    let slot = |i: i64| usize::try_from(i + shift + 1).ok();
    slot(*columns.start())
        .zip(slot(*columns.end()))
        .and_then(|(first, last)| row.get(first..=last))
        .unwrap_or_default()
}

/// Working space for [`frames_of_run`], one array per quantity of the frame formula, reused
/// from row to row.
#[derive(Debug, Default)]
struct FrameRows {
    /// The derivatives along the owner's u axis.
    du: Vectors,
    /// The derivatives along the owner's v axis.
    dv: Vectors,
    /// `du × dv`, then normalized: the normals.
    n: Vectors,
    /// The lengths `n` was normalized by.
    n_lengths: Vec<f32>,
    /// `du · n`: how far `du` leans out of the tangent plane.
    along: Vec<f32>,
    /// `du − n (du · n)`, then normalized: the tangents.
    t: Vectors,
    /// The lengths `t` was normalized by.
    t_lengths: Vec<f32>,
    /// `n × t`: the bitangents.
    b: Vectors,
}

/// Frames from central differences for a run of consecutive vertices of one row, pushed onto
/// `out` in order.
///
/// `[plus_u, minus_u, plus_v, minus_v]` hold each vertex's four reads, as [`central`] takes
/// them. [`Frame::from_derivatives`]' formula runs a step at a time over the whole run, then
/// each vertex's policy (NaN where an input isn't finite, the fallback where the normal or the
/// tangent doesn't normalize) picks its frame: vertex `k` gets the bits
/// `Frame::from_local(central(...))` gives for its reads.
fn frames_of_run(reads: [&[Vec3]; 4], rows: &mut FrameRows, out: &mut Vec<Frame>) {
    let [plus_u, minus_u, plus_v, minus_v] = reads;
    rows.du.set_difference(plus_u, minus_u);
    rows.dv.set_difference(plus_v, minus_v);
    rows.n.set_cross(&rows.du, &rows.dv);
    rows.n.normalize(&mut rows.n_lengths);
    Vectors::dot_into(&rows.du, &rows.n, &mut rows.along);
    rows.t.set_minus_scaled(&rows.du, &rows.n, &rows.along);
    rows.t.normalize(&mut rows.t_lengths);
    rows.b.set_cross(&rows.n, &rows.t);
    let inputs = rows.du.points().zip(rows.dv.points());
    let axes = rows.t.points().zip(rows.b.points()).zip(rows.n.points());
    let lengths = rows.n_lengths.iter().zip(&rows.t_lengths);
    out.extend(inputs.zip(axes).zip(lengths).map(
        |(((du, dv), ((t, b), n)), (&n_length, &t_length))| {
            // The common case first. A normal that normalizes had finite inputs: an infinite or
            // NaN component of `du` or `dv` makes two components of `du × dv` infinite or NaN.
            if normalizable(n_length) & normalizable(t_length) {
                Frame { t, b, n }
            } else if du.is_finite() && dv.is_finite() {
                Frame::FALLBACK
            } else {
                Frame::NAN
            }
        },
    ));
}

/// Frames for every vertex of a predicted block, row-major `(quads + 1)²`. `corners` holds the
/// predicted rings at base-vertex block corners.
///
/// The vertices strictly inside the face go a row at a time through [`frames_of_run`]; the rest
/// (on the face's border, where the owner's axes or a base vertex's ring apply) one by one. Both
/// give a vertex the same bits.
pub(crate) fn frames(
    block: &Block<'_>,
    predicted: &Padded,
    corners: &[Option<CornerAxes<'_>>; 4],
) -> Vec<Frame> {
    let q = i64::from(block.quads);
    let side = usize::try_from(q + 1).unwrap_or(0);
    let mut out = Vec::with_capacity(side * side);
    let one = |frames: &mut Vec<Frame>, i: i64, j: i64| {
        let local = derivatives(block, predicted, corners, i, j);
        debug_assert!(
            local.is_finite(),
            "frame inputs at ({i}, {j}) of {block:?} are not finite"
        );
        frames.push(Frame::from_local(local));
    };
    let (columns, rows) = inside_face(block);
    let mut scratch = FrameRows::default();
    for j in 0..=q {
        if !rows.contains(&j) || columns.is_empty() {
            (0..=q).for_each(|i| one(&mut out, i, j));
            continue;
        }
        (0..*columns.start()).for_each(|i| one(&mut out, i, j));
        let row = predicted.row(j);
        let reads = [
            run_of(row, &columns, 1),
            run_of(row, &columns, -1),
            run_of(predicted.row(j + 1), &columns, 0),
            run_of(predicted.row(j - 1), &columns, 0),
        ];
        let run_start = out.len();
        frames_of_run(reads, &mut scratch, &mut out);
        debug_assert!(
            out.get(run_start..)
                .is_some_and(|run| run.iter().all(|f| f.n.is_finite())),
            "frame inputs in row {j} of {block:?} are not finite"
        );
        (*columns.end() + 1..=q).for_each(|i| one(&mut out, i, j));
    }
    out
}

/// The unit normal, with [`Frame::from_local`]'s policy: the fallback frame's normal where finite
/// inputs are degenerate, NaN where any isn't finite.
pub(crate) fn normal(local: Local) -> Vec3 {
    match local {
        Local::Derivatives { du, dv } if du.is_finite() && dv.is_finite() => du.cross(dv),
        Local::Ring { normal, .. } => normal,
        Local::Derivatives { .. } | Local::NonFinite => return Vec3::NAN,
    }
    .normalized()
    .unwrap_or(Frame::FALLBACK.n)
}

/// Unit normals for every vertex of a block of evaluated positions, into `out`.
///
/// `out` holds `(quads + 1)²` values, row-major. Positions come from a source, so non-finite
/// input is passed through as NaN, not asserted.
pub(crate) fn normals(
    block: &Block<'_>,
    positions: &Padded,
    corners: &[Option<CornerAxes<'_>>; 4],
    out: &mut [Vec3],
) {
    let q = usize::try_from(block.quads).unwrap_or(usize::MAX);
    debug_assert_eq!(
        Some(out.len()),
        (q + 1).checked_pow(2),
        "one normal per vertex"
    );
    for (slot, n) in out
        .iter_mut()
        .zip(per_vertex(block, positions, corners, |_, _, local| {
            normal(local)
        }))
    {
        *slot = n;
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;
    use crate::test_rng::Rng;

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
    fn normals_follow_the_frame_policy() {
        let (du, dv) = (Vec3::new(2.0, 0.1, 0.0), Vec3::new(0.0, 3.0, 0.2));
        let local = |a, b| Local::Derivatives { du: a, dv: b };
        assert_eq!(
            normal(local(du, dv)),
            Frame::from_derivatives(du, dv).n,
            "same normal as the frame"
        );
        assert_eq!(
            normal(local(Vec3::ZERO, dv)),
            Frame::FALLBACK.n,
            "degenerate: fallback"
        );
        assert!(
            !normal(local(Vec3::NAN, dv)).is_finite(),
            "non-finite: NaN, not the fallback"
        );
    }

    /// A valence-3 ring around the origin, in the z = 0 plane, counter-clockwise from +x.
    fn flat_ring() -> [Vec3; 3] {
        [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-0.5, 0.8, 0.0),
            Vec3::new(-0.5, -0.8, 0.0),
        ]
    }

    #[test]
    fn ring_frames_take_the_ring_normal_and_the_owners_first_edge() {
        let points = flat_ring();
        for (owner_corner, t) in [(0, 1.0), (2, -1.0)] {
            let axes = CornerAxes {
                center: Vec3::ZERO,
                points: &points,
                owner_corner,
            };
            let f = Frame::from_local(axes.local());
            assert_eq!(f.n, Vec3::new(0.0, 0.0, 1.0), "corner {owner_corner}: n");
            assert_eq!(f.t, Vec3::new(t, 0.0, 0.0), "corner {owner_corner}: t");
            assert_eq!(normal(axes.local()), f.n, "corner {owner_corner}: normal");
        }
    }

    #[test]
    fn ring_frames_follow_the_frame_policy() {
        let points = flat_ring();
        let collapsed = [Vec3::ZERO; 3];
        let degenerate = CornerAxes {
            center: Vec3::ZERO,
            points: &collapsed,
            owner_corner: 0,
        };
        assert_eq!(
            Frame::from_local(degenerate.local()),
            Frame::FALLBACK,
            "degenerate: fallback frame"
        );
        assert_eq!(
            normal(degenerate.local()),
            Frame::FALLBACK.n,
            "degenerate: fallback normal"
        );
        let mut far = points;
        far[2] = Vec3::NAN;
        let nan = CornerAxes {
            center: Vec3::ZERO,
            points: &far,
            owner_corner: 0,
        };
        assert_eq!(nan.local(), Local::NonFinite, "a NaN anywhere in the ring");
        assert!(
            !Frame::from_local(nan.local()).n.is_finite(),
            "NaN frame, not the fallback"
        );
        assert!(!normal(nan.local()).is_finite(), "NaN normal");
    }

    /// Derivatives that exercise every branch of the frame policy: random, parallel, zero,
    /// tiny, huge, infinite and NaN, and a normal that normalizes beside a tangent that doesn't.
    fn awkward_derivatives(rng: &mut Rng, k: usize) -> (Vec3, Vec3) {
        let du = rng.vec3(1.0);
        match k % 10 {
            7 => (Vec3::new(1e-39, 0.0, 0.0), Vec3::new(0.0, 1e20, 0.0)),
            0 => (du, du * 2.0),
            1 => (Vec3::ZERO, rng.vec3(1.0)),
            2 => (Vec3::new(1e-30, 2e-30, 0.0), Vec3::new(0.0, 1e-30, 3e-30)),
            3 => (Vec3::new(3e18, -2e19, 1e19), Vec3::new(-1e19, 2e19, 3e18)),
            4 => (Vec3::new(f32::INFINITY, 0.0, 1.0), rng.vec3(1.0)),
            5 => (du, Vec3::new(0.5, f32::NAN, 0.0)),
            6 => (Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, -1.0)),
            _ => (du, rng.vec3(1.0)),
        }
    }

    /// The row kernel gives every vertex the bits of the one-vertex frame, whatever branch of
    /// the policy it takes.
    #[test]
    fn runs_of_frames_match_one_vertex_frames() {
        let mut rng = Rng::new(77);
        for len in [0, 1, 7, 8, 9, 65, 200] {
            let (du, dv): (Vec<Vec3>, Vec<Vec3>) =
                (0..len).map(|k| awkward_derivatives(&mut rng, k)).unzip();
            let zero = vec![Vec3::ZERO; len];
            let mut out = Vec::new();
            frames_of_run(
                [&du, &zero, &dv, &zero],
                &mut FrameRows::default(),
                &mut out,
            );
            assert_eq!(out.len(), len, "one frame per vertex");
            for (k, ((&u, &v), got)) in du.iter().zip(&dv).zip(&out).enumerate() {
                let want = Frame::from_local(central([u, Vec3::ZERO, v, Vec3::ZERO]));
                let bits = |f: &Frame| [f.t, f.b, f.n].map(Vec3::to_bits);
                assert_eq!(bits(got), bits(&want), "vertex {k} of {len}: {u:?}, {v:?}");
            }
        }
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
