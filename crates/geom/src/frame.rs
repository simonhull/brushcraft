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
    /// A frame whose every component is NaN: what non-finite inputs give, so a bug upstream shows
    /// up in the output instead of being hidden behind the fallback frame used where the surface
    /// is degenerate; and what a caller listing frames can stand in for one it lacks, for the same
    /// reason.
    pub const NAN: Self = Self {
        t: Vec3::NAN,
        b: Vec3::NAN,
        n: Vec3::NAN,
    };

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
    /// Central differences: a [`Stencil::Axes`] stencil's four reads
    /// `[plus_u, minus_u, plus_v, minus_v]`. The derivatives are `du = plus_u − minus_u` and
    /// `dv = plus_v − minus_v`, the unnormalized normal `du × dv`.
    Central([Vec3; 4]),
    /// At a base vertex, from finite inputs: the owner's first-edge direction and the
    /// unnormalized ring normal.
    Ring {
        /// The direction of the owner's first edge out of the vertex: the frame's u axis.
        du: Vec3,
        /// The ring normal, summed over the whole ring, not normalized.
        normal: Vec3,
    },
    /// At a base vertex whose ring isn't finite (or isn't known), or off the face.
    NonFinite,
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
        Stencil::Axes([u, v]) => Local::Central([
            grid.get(i + u.0, j + u.1),
            grid.get(i - u.0, j - u.1),
            grid.get(i + v.0, j + v.1),
            grid.get(i - v.0, j - v.1),
        ]),
        Stencil::Outside => Local::NonFinite,
    }
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

/// A piece of a block's row, as [`row_parts`] hands them out.
#[derive(Debug, Clone, Copy)]
enum RowPart<'a> {
    /// Vertex `(i, j)`, on the face's border: its stencil ([`stencil`]) is read alone.
    Vertex(i64, i64),
    /// A run of consecutive vertices of a row, strictly inside the face: each vertex's four
    /// reads `[plus_u, minus_u, plus_v, minus_v]`, as [`Local::Central`] holds them.
    Run {
        /// The runs of reads, one slice per read, one entry per vertex.
        reads: [&'a [Vec3]; 4],
    },
}

/// A block's vertices in row-major order: in each row, a run of the vertices strictly inside the
/// face, read from `grid` as slices, and single vertices before and after it.
fn row_parts<'a>(block: &Block<'_>, grid: &'a Padded) -> impl Iterator<Item = RowPart<'a>> {
    let q = i64::from(block.quads);
    let (columns, rows) = inside_face(block);
    let run_len = columns.clone().count();
    (0..=q).flat_map(move |j| {
        let has_run = rows.contains(&j) && run_len > 0;
        // The vertices before the run end at `before_end`, those after it start at
        // `after_start`; a row without a run is all "before".
        let (before_end, after_start) = if has_run {
            (*columns.start(), *columns.end() + 1)
        } else {
            (q + 1, q + 1)
        };
        let run = has_run.then(|| {
            let row = grid.row(j);
            let reads = [
                run_of(row, &columns, 1),
                run_of(row, &columns, -1),
                run_of(grid.row(j + 1), &columns, 0),
                run_of(grid.row(j - 1), &columns, 0),
            ];
            debug_assert!(
                reads.iter().all(|r| r.len() == run_len),
                "row {j}'s reads lie in the padded block"
            );
            RowPart::Run { reads }
        });
        let vertex = move |i| RowPart::Vertex(i, j);
        (0..before_end)
            .map(vertex)
            .chain(run)
            .chain((after_start..=q).map(vertex))
    })
}

/// What [`policy`] gives a vertex.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Outcome {
    /// What the formula gave.
    Formula,
    /// The fallback: finite inputs that don't normalize.
    Fallback,
    /// NaN: inputs that aren't finite.
    NotFinite,
}

/// The frame and normal policy, the one place it is written: what the formula gave where it
/// `normalized`, else the fallback where the inputs were finite, else NaN.
///
/// `inputs_finite` is only asked when the formula didn't normalize. A normal that normalizes
/// had finite inputs: an infinite or NaN component of `du` or `dv` makes two components of
/// `du × dv` infinite or NaN, and a ring's inputs are finite by construction
/// ([`CornerAxes::local`]).
fn policy(normalized: bool, inputs_finite: impl FnOnce() -> bool) -> Outcome {
    if normalized {
        Outcome::Formula
    } else if inputs_finite() {
        Outcome::Fallback
    } else {
        Outcome::NotFinite
    }
}

/// Where a batch in [`FrameRows`] came from, for [`policy`]'s question about its inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Inputs {
    /// Central differences: `du` and `dv` hold them.
    Central,
    /// Base-vertex rings, whose inputs are finite.
    Ring,
}

/// The frame and normal formula over a batch of vertices, one array per quantity.
///
/// Every frame and normal goes through it: a row's run of vertices at a time, or a block's
/// scattered vertices together ([`Scattered`]). The arrays are reused from batch to batch.
///
/// The formula runs a step at a time over the whole batch, so each step is one loop over
/// independent vertices that the compiler vectorises.
#[derive(Debug, Default)]
pub(crate) struct FrameRows {
    /// The derivatives along the owner's u axis (central differences), or the owner's first
    /// edge (rings).
    du: Vectors,
    /// The derivatives along the owner's v axis (central differences only).
    dv: Vectors,
    /// The unnormalized normals (`du × dv`, or the ring normal), then normalized.
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

impl FrameRows {
    /// Loads a batch of central differences from their four reads, one slice per read, one
    /// entry per vertex.
    fn load_central(&mut self, [plus_u, minus_u, plus_v, minus_v]: [&[Vec3]; 4]) -> Inputs {
        self.du.set_difference(plus_u, minus_u);
        self.dv.set_difference(plus_v, minus_v);
        self.n.set_cross(&self.du, &self.dv);
        Inputs::Central
    }

    /// Loads a batch of rings: each one's first-edge direction and unnormalized ring normal.
    fn load_rings(&mut self, du: &[Vec3], normals: &[Vec3]) -> Inputs {
        self.du.set_points(du);
        self.n.set_points(normals);
        Inputs::Ring
    }

    /// The batch's unit normals, into `out` in order (one slot per vertex).
    fn normals<'o>(&mut self, inputs: Inputs, out: impl Iterator<Item = &'o mut Vec3>) {
        self.n.normalize(&mut self.n_lengths);
        // The batch's own iterator first: `zip` stops on it without taking a slot past it.
        let normals = self.n.points().zip(&self.n_lengths).zip(out);
        match inputs {
            Inputs::Central => {
                let derivatives = self.du.points().zip(self.dv.points());
                for (((n, &length), slot), (du, dv)) in normals.zip(derivatives) {
                    *slot = normal_policy(n, length, || du.is_finite() && dv.is_finite());
                }
            }
            Inputs::Ring => {
                for ((n, &length), slot) in normals {
                    *slot = normal_policy(n, length, || true);
                }
            }
        }
    }

    /// The batch's frames, appended to `out` in order: `n` normalized, `t` `du` projected onto
    /// its plane and normalized, `b = n × t`.
    fn frames(&mut self, inputs: Inputs, out: &mut Vec<Frame>) {
        self.n.normalize(&mut self.n_lengths);
        Vectors::dot_into(&self.du, &self.n, &mut self.along);
        self.t.set_minus_scaled(&self.du, &self.n, &self.along);
        self.t.normalize(&mut self.t_lengths);
        self.b.set_cross(&self.n, &self.t);
        let tbn = self.t.points().zip(self.b.points()).zip(self.n.points());
        let frames = tbn
            .zip(self.n_lengths.iter().zip(&self.t_lengths))
            .map(|(((t, b), n), (&nl, &tl))| ([t, b, n], [nl, tl]));
        match inputs {
            Inputs::Central => {
                let derivatives = self.du.points().zip(self.dv.points());
                out.extend(frames.zip(derivatives).map(|((axes, lengths), (du, dv))| {
                    frame_policy(axes, lengths, || du.is_finite() && dv.is_finite())
                }));
            }
            Inputs::Ring => {
                out.extend(frames.map(|(axes, lengths)| frame_policy(axes, lengths, || true)));
            }
        }
    }
}

/// One normal of a [`FrameRows`] batch, by [`policy`]: `n`, normalized by `length`.
fn normal_policy(n: Vec3, length: f32, inputs_finite: impl FnOnce() -> bool) -> Vec3 {
    match policy(normalizable(length), inputs_finite) {
        Outcome::Formula => n,
        Outcome::Fallback => Frame::FALLBACK.n,
        Outcome::NotFinite => Vec3::NAN,
    }
}

/// One frame of a [`FrameRows`] batch, by [`policy`]: its axes, and the lengths `n` and `t`
/// were normalized by.
fn frame_policy(
    [t, b, n]: [Vec3; 3],
    [n_length, t_length]: [f32; 2],
    inputs_finite: impl FnOnce() -> bool,
) -> Frame {
    match policy(
        normalizable(n_length) & normalizable(t_length),
        inputs_finite,
    ) {
        Outcome::Formula => Frame { t, b, n },
        Outcome::Fallback => Frame::FALLBACK,
        Outcome::NotFinite => Frame::NAN,
    }
}

/// Vertices whose frames or normals are computed apart from the rows' runs.
///
/// They are the vertices on the face's border, base vertices, and a refresh's stale vertices:
/// collected by kind, then computed together through [`FrameRows`], as the runs are.
#[derive(Debug, Default)]
pub(crate) struct Scattered {
    /// Where each central-difference vertex's result goes, in the order of `reads`.
    central: Vec<usize>,
    /// Their reads `[plus_u, minus_u, plus_v, minus_v]`, one array per read.
    reads: [Vec<Vec3>; 4],
    /// Where each ring vertex's result goes, in the order of `ring_du` and `ring_normals`.
    rings: Vec<usize>,
    /// Their first-edge directions.
    ring_du: Vec<Vec3>,
    /// Their unnormalized ring normals.
    ring_normals: Vec<Vec3>,
    /// Where the vertices without finite inputs go: they get NaN.
    non_finite: Vec<usize>,
}

impl Scattered {
    /// No vertices yet, with room for `capacity` central-difference vertices.
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            central: Vec::with_capacity(capacity),
            reads: std::array::from_fn(|_| Vec::with_capacity(capacity)),
            ..Self::default()
        }
    }

    /// Adds a vertex whose result goes to `place`.
    pub(crate) fn push(&mut self, place: usize, local: Local) {
        match local {
            Local::Central(reads) => {
                self.central.push(place);
                for (column, read) in self.reads.iter_mut().zip(reads) {
                    column.push(read);
                }
            }
            Local::Ring { du, normal } => {
                self.rings.push(place);
                self.ring_du.push(du);
                self.ring_normals.push(normal);
            }
            Local::NonFinite => self.non_finite.push(place),
        }
    }

    /// How many vertices were added.
    pub(crate) fn len(&self) -> usize {
        self.central.len() + self.rings.len() + self.non_finite.len()
    }

    /// The four read columns, as [`FrameRows::load_central`] takes them.
    fn reads(&self) -> [&[Vec3]; 4] {
        self.reads.each_ref().map(Vec::as_slice)
    }

    /// Each vertex's frame, written to `out` at its place.
    fn frames_into(&self, rows: &mut FrameRows, out: &mut [Frame]) {
        let mut central = Vec::with_capacity(self.central.len());
        let central_inputs = rows.load_central(self.reads());
        rows.frames(central_inputs, &mut central);
        place(&self.central, central.into_iter(), out);
        let mut rings = Vec::with_capacity(self.rings.len());
        let ring_inputs = rows.load_rings(&self.ring_du, &self.ring_normals);
        rows.frames(ring_inputs, &mut rings);
        place(&self.rings, rings.into_iter(), out);
        place(&self.non_finite, std::iter::repeat(Frame::NAN), out);
    }

    /// Each vertex's unit normal, written to `out` at its place.
    pub(crate) fn normals_into(&self, rows: &mut FrameRows, out: &mut [Vec3]) {
        let mut central = vec![Vec3::NAN; self.central.len()];
        let central_inputs = rows.load_central(self.reads());
        rows.normals(central_inputs, central.iter_mut());
        place(&self.central, central.into_iter(), out);
        let mut rings = vec![Vec3::NAN; self.rings.len()];
        let ring_inputs = rows.load_rings(&self.ring_du, &self.ring_normals);
        rows.normals(ring_inputs, rings.iter_mut());
        place(&self.rings, rings.into_iter(), out);
        place(&self.non_finite, std::iter::repeat(Vec3::NAN), out);
    }
}

/// Writes `values` to `out` at `places`, in order.
fn place<T>(places: &[usize], values: impl Iterator<Item = T>, out: &mut [T]) {
    for (&k, value) in places.iter().zip(values) {
        if let Some(slot) = out.get_mut(k) {
            *slot = value;
        }
    }
}

/// Frames for every vertex of a predicted block, row-major `(quads + 1)²`. `corners` holds the
/// predicted rings at base-vertex block corners.
///
/// The vertices strictly inside the face go a row's run at a time through [`FrameRows`]; the
/// rest (on the face's border, where the owner's axes or a base vertex's ring apply) are
/// collected and go through it together at the end.
///
/// Vertices whose inputs aren't finite get [`Frame::NAN`], as [`normals`] passes such input
/// through: positions come from a source, so this is not asserted.
pub(crate) fn frames(
    block: &Block<'_>,
    predicted: &Padded,
    corners: &[Option<CornerAxes<'_>>; 4],
) -> Vec<Frame> {
    let side = usize::try_from(block.quads).map_or(0, |q| q + 1);
    let mut out = Vec::with_capacity(side * side);
    let mut rows = FrameRows::default();
    // At most the block's border is on the face's border.
    let mut scattered = Scattered::with_capacity(4 * side);
    for part in row_parts(block, predicted) {
        match part {
            RowPart::Vertex(i, j) => {
                scattered.push(out.len(), derivatives(block, predicted, corners, i, j));
                out.push(Frame::NAN);
            }
            RowPart::Run { reads, .. } => {
                let inputs = rows.load_central(reads);
                rows.frames(inputs, &mut out);
            }
        }
    }
    scattered.frames_into(&mut rows, &mut out);
    out
}

/// Unit normals for every vertex of a block of evaluated positions, into `out`.
///
/// `out` holds `(quads + 1)²` values, row-major. Positions come from a source, so non-finite
/// input is passed through as NaN, not asserted. As in [`frames`], the vertices strictly inside
/// the face go a row's run at a time through [`FrameRows`], the rest together at the end.
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
    let mut rows = FrameRows::default();
    // At most the block's border is on the face's border.
    let mut scattered = Scattered::with_capacity(4 * (q + 1));
    let mut slots = out.iter_mut().enumerate();
    for part in row_parts(block, positions) {
        match part {
            RowPart::Vertex(i, j) => {
                if let Some((k, _)) = slots.next() {
                    scattered.push(k, derivatives(block, positions, corners, i, j));
                }
            }
            RowPart::Run { reads, .. } => {
                let inputs = rows.load_central(reads);
                rows.normals(inputs, slots.by_ref().map(|(_, slot)| slot));
            }
        }
    }
    scattered.normals_into(&mut rows, out);
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;
    use crate::test_rng::Rng;

    /// The frame and normal of one vertex, as rewritten from the specification with early
    /// returns: an oracle independent of [`FrameRows`] and [`policy`], which the row and batch code
    /// must match bit for bit.
    mod reference {
        use super::{Frame, Local, Vec3};

        /// The frame from a tangent direction and an unnormalized normal, from finite inputs.
        fn from_tangent_and_normal(du: Vec3, normal: Vec3) -> Frame {
            let Some(n) = normal.normalized() else {
                return Frame::FALLBACK;
            };
            let Some(t) = (du - n * du.dot(n)).normalized() else {
                return Frame::FALLBACK;
            };
            Frame {
                t,
                b: n.cross(t),
                n,
            }
        }

        /// A vertex's frame.
        pub(super) fn frame(local: Local) -> Frame {
            match local {
                Local::Central([plus_u, minus_u, plus_v, minus_v]) => {
                    let (du, dv) = (plus_u - minus_u, plus_v - minus_v);
                    if du.is_finite() && dv.is_finite() {
                        from_tangent_and_normal(du, du.cross(dv))
                    } else {
                        Frame::NAN
                    }
                }
                Local::Ring { du, normal } => from_tangent_and_normal(du, normal),
                Local::NonFinite => Frame::NAN,
            }
        }

        /// A vertex's unit normal.
        pub(super) fn normal(local: Local) -> Vec3 {
            let unnormalized = match local {
                Local::Central([plus_u, minus_u, plus_v, minus_v]) => {
                    let (du, dv) = (plus_u - minus_u, plus_v - minus_v);
                    if !(du.is_finite() && dv.is_finite()) {
                        return Vec3::NAN;
                    }
                    du.cross(dv)
                }
                Local::Ring { normal, .. } => normal,
                Local::NonFinite => return Vec3::NAN,
            };
            unnormalized.normalized().unwrap_or(Frame::FALLBACK.n)
        }
    }

    /// One vertex's frame, through the batch code.
    fn frame_of(local: Local) -> Frame {
        let mut scattered = Scattered::default();
        scattered.push(0, local);
        let mut out = [Frame::FALLBACK];
        scattered.frames_into(&mut FrameRows::default(), &mut out);
        out[0]
    }

    /// One vertex's normal, through the batch code.
    fn normal_of(local: Local) -> Vec3 {
        let mut scattered = Scattered::default();
        scattered.push(0, local);
        let mut out = [Vec3::ZERO];
        scattered.normals_into(&mut FrameRows::default(), &mut out);
        out[0]
    }

    /// Central differences `du` and `dv` as a stencil's reads.
    fn central(du: Vec3, dv: Vec3) -> Local {
        Local::Central([du, Vec3::ZERO, dv, Vec3::ZERO])
    }

    /// A frame's bits.
    fn bits(f: &Frame) -> [[u32; 3]; 3] {
        [f.t, f.b, f.n].map(Vec3::to_bits)
    }

    #[test]
    fn frames_are_orthonormal_and_right_handed() {
        let f = frame_of(central(Vec3::new(2.0, 0.1, 0.0), Vec3::new(0.0, 3.0, 0.2)));
        assert!((f.t.length() - 1.0).abs() < 1e-6, "t unit");
        assert!((f.n.length() - 1.0).abs() < 1e-6, "n unit");
        assert!(f.t.dot(f.n).abs() < 1e-6, "t ⟂ n");
        assert!((f.b - f.n.cross(f.t)).length() < 1e-6, "b = n × t");
        assert!(f.n.z > 0.0, "normal follows du × dv");
    }

    #[test]
    fn degenerate_derivatives_fall_back_deterministically() {
        assert_eq!(
            frame_of(central(Vec3::ZERO, Vec3::ZERO)),
            Frame::FALLBACK,
            "zero"
        );
        let nan = frame_of(central(Vec3::NAN, Vec3::new(0.0, 1.0, 0.0)));
        assert!(
            !(nan.t.is_finite() || nan.b.is_finite() || nan.n.is_finite()),
            "NaN propagates instead of falling back: {nan:?}"
        );
        assert!(
            !frame_of(central(
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(f32::INFINITY, 0.0, 0.0)
            ))
            .n
            .is_finite(),
            "so does infinity"
        );
    }

    #[test]
    fn normals_follow_the_frame_policy() {
        let (du, dv) = (Vec3::new(2.0, 0.1, 0.0), Vec3::new(0.0, 3.0, 0.2));
        assert_eq!(
            normal_of(central(du, dv)),
            frame_of(central(du, dv)).n,
            "same normal as the frame"
        );
        assert_eq!(
            normal_of(central(Vec3::ZERO, dv)),
            Frame::FALLBACK.n,
            "degenerate: fallback"
        );
        assert!(
            !normal_of(central(Vec3::NAN, dv)).is_finite(),
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
            let f = frame_of(axes.local());
            assert_eq!(f.n, Vec3::new(0.0, 0.0, 1.0), "corner {owner_corner}: n");
            assert_eq!(f.t, Vec3::new(t, 0.0, 0.0), "corner {owner_corner}: t");
            assert_eq!(
                normal_of(axes.local()),
                f.n,
                "corner {owner_corner}: normal"
            );
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
            frame_of(degenerate.local()),
            Frame::FALLBACK,
            "degenerate: fallback frame"
        );
        assert_eq!(
            normal_of(degenerate.local()),
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
            !frame_of(nan.local()).n.is_finite(),
            "NaN frame, not the fallback"
        );
        assert!(!normal_of(nan.local()).is_finite(), "NaN normal");
    }

    /// Derivatives that exercise every branch of the frame policy: random, parallel, zero,
    /// negative zero, tiny, huge, infinite and NaN, and a normal that normalizes beside a
    /// tangent that doesn't.
    fn awkward_derivatives(rng: &mut Rng, k: usize) -> (Vec3, Vec3) {
        let du = rng.vec3(1.0);
        match k % 12 {
            0 => (du, du * 2.0),
            1 => (Vec3::ZERO, rng.vec3(1.0)),
            2 => (Vec3::new(1e-30, 2e-30, 0.0), Vec3::new(0.0, 1e-30, 3e-30)),
            3 => (Vec3::new(3e18, -2e19, 1e19), Vec3::new(-1e19, 2e19, 3e18)),
            4 => (Vec3::new(f32::INFINITY, 0.0, 1.0), rng.vec3(1.0)),
            5 => (du, Vec3::new(0.5, f32::NAN, 0.0)),
            6 => (Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, -1.0)),
            7 => (Vec3::new(1e-39, 0.0, 0.0), Vec3::new(0.0, 1e20, 0.0)),
            8 => (Vec3::new(-0.0, 1.0, -0.0), Vec3::new(0.0, -0.0, 1.0)),
            9 => (Vec3::new(-0.0, -0.0, -0.0), rng.vec3(1.0)),
            _ => (du, rng.vec3(1.0)),
        }
    }

    /// Stencil reads giving [`awkward_derivatives`], with `+0.0`, `-0.0` or random points
    /// subtracted.
    fn awkward_reads(rng: &mut Rng, k: usize) -> [Vec3; 4] {
        let (du, dv) = awkward_derivatives(rng, k);
        let minus = match k % 3 {
            0 => Vec3::ZERO,
            1 => Vec3::new(-0.0, -0.0, -0.0),
            _ => rng.vec3(1.0),
        };
        [du + minus, minus, dv + minus, minus]
    }

    /// A batch of central differences gets the reference's bits, frames and normals, whatever
    /// branch of the policy each vertex takes; a normals batch takes only its own slots.
    #[test]
    fn central_batches_match_the_reference() {
        let mut rng = Rng::new(77);
        for len in [0, 1, 7, 8, 9, 65, 200] {
            let reads: Vec<[Vec3; 4]> = (0..len).map(|k| awkward_reads(&mut rng, k)).collect();
            let columns: [Vec<Vec3>; 4] =
                std::array::from_fn(|c| reads.iter().map(|r| r[c]).collect());
            let slices = columns.each_ref().map(Vec::as_slice);
            let mut rows = FrameRows::default();
            let mut frames = Vec::new();
            let inputs = rows.load_central(slices);
            rows.frames(inputs, &mut frames);
            assert_eq!(frames.len(), len, "one frame per vertex");
            let mut normals = vec![Vec3::ZERO; len + 1];
            let mut slots = normals.iter_mut();
            let again = rows.load_central(slices);
            rows.normals(again, slots.by_ref());
            *slots.next().unwrap() = Vec3::new(9.0, 9.0, 9.0);
            for (k, ((&r, got), n)) in reads.iter().zip(&frames).zip(&normals).enumerate() {
                let local = Local::Central(r);
                let what = format!("vertex {k} of {len}: {r:?}");
                assert_eq!(bits(got), bits(&reference::frame(local)), "frame, {what}");
                assert_eq!(
                    n.to_bits(),
                    reference::normal(local).to_bits(),
                    "normal, {what}"
                );
            }
            assert_eq!(
                normals[len],
                Vec3::new(9.0, 9.0, 9.0),
                "the batch took only its own slots"
            );
        }
    }

    /// Scattered vertices of every kind (central differences, rings including degenerate ones,
    /// no finite inputs) land at their places with the reference's bits.
    #[test]
    fn scattered_vertices_match_the_reference() {
        let mut rng = Rng::new(79);
        let locals: Vec<Local> = (0..60)
            .map(|k| match k % 4 {
                0 | 1 => Local::Central(awkward_reads(&mut rng, k)),
                2 => {
                    let (du, normal) = awkward_derivatives(&mut rng, k);
                    let finite = du.is_finite() && normal.is_finite();
                    if finite {
                        Local::Ring { du, normal }
                    } else {
                        Local::NonFinite
                    }
                }
                _ => Local::NonFinite,
            })
            .collect();
        // Places in a scrambled order.
        let places: Vec<usize> = (0..locals.len()).map(|k| (k * 37) % locals.len()).collect();
        let mut scattered = Scattered::default();
        for (&place, &local) in places.iter().zip(&locals) {
            scattered.push(place, local);
        }
        assert_eq!(scattered.len(), locals.len(), "every vertex counted");
        let mut frames = vec![Frame::FALLBACK; locals.len()];
        let mut normals = vec![Vec3::ZERO; locals.len()];
        let mut rows = FrameRows::default();
        scattered.frames_into(&mut rows, &mut frames);
        scattered.normals_into(&mut rows, &mut normals);
        for (&place, &local) in places.iter().zip(&locals) {
            assert_eq!(
                bits(&frames[place]),
                bits(&reference::frame(local)),
                "frame of {local:?}"
            );
            assert_eq!(
                normals[place].to_bits(),
                reference::normal(local).to_bits(),
                "normal of {local:?}"
            );
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
