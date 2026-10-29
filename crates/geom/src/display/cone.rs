//! Normal cones: a direction and a half-angle holding every finest normal under a display tile.
//!
//! Level choice refines a tile by its cone, not by its own coarse triangles: a tile whose
//! triangles face away can still hold finest detail facing the camera (just past a silhouette),
//! and the cone says so. So a cone must **hold every finest normal under its tile as stored**:
//! the angle from each to the decoded axis is at most the decoded half-angle.
//!
//! # Building one
//!
//! One level below the finest, a tile's cone is taken from the finest normals under it
//! ([`FinestNormalCone::covering`]); above, from its children's decoded cones
//! ([`FinestNormalCone::holding`]), so it chains up the levels as `e` does. Either way, an axis is
//! chosen (the normals' sum, or the children's cones merged two at a time), quantized, and
//! decoded; the half-angle is then measured against the *decoded* axis, from the inputs
//! themselves (`angle(axis, n)` for a normal, `angle(axis, child's axis) + child's half-angle`
//! for a child, which holds everything the child holds by the triangle inequality on the
//! sphere), each angle raised by [`SLACK`] unless it is exactly 0, and rounded **up** to its
//! stored step. So whatever the axis's quantization error, the stored cone holds its inputs, and
//! the chosen axis only decides how tight it is. A half-angle that reaches 180° (or past) is
//! stored as the cone that covers everything.
//!
//! The chaining costs width: a parent's cone holds its children's whole cones, not just the
//! normals in them, so it is wider than the least cone of those normals (on the display tests'
//! smooth sphere, about 6° wider than the exact cone around the same axis, where four children
//! of 30.6° meet).
//!
//! # Stored
//!
//! In one `u32` ([`FinestNormalCone::to_bits`]): the axis octahedral in the low [`AXIS_BITS`] × 2
//! (two signed fractions), the half-angle in the high [`HALF_BITS`] as a count of
//! `π / (2^HALF_BITS − 1)` steps, the largest count meaning 180°: everything. Every `u32` is a
//! cone. The split of the 32 bits is the measured one (see [`AXIS_BITS`]).

use core::f64::consts::PI;
use core::fmt;

use super::DisplayError;
use super::encode::{from_octahedral, levels, octahedral, quantize_unit};
use crate::Vec3;

/// Bits per octahedral component of the axis.
///
/// A cone grows, at each level, by up to the axis's quantization error (its half-angle is
/// measured from the decoded axis: under 1.3° at 8 bits, 0.33° at 10, 0.085° at 12) and by up
/// to one half-angle step (it is rounded up: 0.003° at 16 bits, 0.044° at 12, 0.71° at 8), and
/// the levels add these up. Of the splits of the 32 bits, 10 + 12 gives the tightest cones on
/// the display tests' fixtures (`cone_widths`): the smooth sphere's cones one level below the
/// finest are 30.59° (8 + 16 bits: 30.92°; 12 + 8: 31.06°), and above them 68.57° (68.88°,
/// 69.18°).
const AXIS_BITS: u8 = 10;

/// Bits of the half-angle's step count (see [`AXIS_BITS`]).
const HALF_BITS: u8 = 12;

// The cone fills one `u32` exactly, so every one of them is a cone.
const _: () = assert!(2 * AXIS_BITS + HALF_BITS == 32, "a cone is one u32");

/// The mask of one axis component's bits.
const AXIS_MASK: u32 = (1 << AXIS_BITS) - 1;

/// Where the half-angle's bits start.
const HALF_SHIFT: u32 = 2 * AXIS_BITS as u32;

/// The half-angle's largest step count: 180°, the cone that covers everything.
const HALF_MAX: u32 = (1 << HALF_BITS) - 1;

/// One half-angle step, in radians: `π / HALF_MAX`.
const HALF_STEP: f32 = core::f32::consts::PI / 4095.0;

// `HALF_STEP`'s literal is `HALF_MAX`.
const _: () = assert!(HALF_MAX == 4095, "HALF_STEP divides by HALF_MAX");

/// What each angle a half-angle is measured from is raised by, in radians.
///
/// Far more than the `f64` arithmetic measuring it can lose (an `atan2`, a few products and a
/// sum, each off by about 2⁻⁵² of π), and far less than a step. An angle of exactly 0 (a direction
/// the axis itself, the cross product exactly zero) is exact, and isn't raised, so a cone alone is
/// its own union and a level whose tiles each hold one child doesn't widen.
const SLACK: f64 = 1e-9;

/// How far over the least cosine a normal's cosine to the axis may lie and still be measured
/// exactly by [`FinestNormalCone::covering`]: every other normal is narrower.
///
/// The cosines are ranked in `f32`, against `a`, the decoded axis (unit to within 2⁻²³). For
/// normals of unit length (all of them within [`UNIT_LENGTH_STRAY`] of it), the plain `a · n`
/// stands for the cosine: `|a| |n|` is within 1.3·10⁻⁶ of 1, and the dot product's three
/// products and two sums round by at most 2⁻²⁴ of terms no larger than 1 each (3·10⁻⁷), so it
/// is within 1.6·10⁻⁶ of the true cosine. Otherwise the cosine is `(a · n) / √(n · n)`, the
/// root and the quotient adding 2⁻²⁴ relative each: within 10⁻⁶. Either way, a normal ranked
/// more than 10⁻⁵ over the least is truly narrower than the least-ranked normal (by over
/// 6·10⁻⁶ in cosine), and its exact `atan2` measure (off by about 10⁻¹⁵ rad) is under that
/// normal's plus [`SLACK`], which is what the half-angle is built from. The band is several
/// times the error, and narrow enough that only the normals beside the widest are measured
/// exactly. (Were the widest measured at exactly 0, every cosine would be within rounding of 1,
/// so none lies past the band.)
const COSINE_TOLERANCE: f32 = 1e-5;

/// How far `n · n` may stray from 1, for every normal, for [`FinestNormalCone::covering`] to
/// rank them by `a · n` alone, without a root and a quotient each: `2·10⁻⁶`, so `|n|` is within
/// about 10⁻⁶ of 1.
///
/// The ranking needs every normal near unit length: a long one looks narrower than it is and
/// could be left out of the band, and a short one (a zero one most of all) looks wider, so it
/// moves the band and can leave a truly wider one outside. The finest normals are unit to `f32`
/// precision: the evaluation scales each `du × dv` (or ring normal) by the reciprocal of its
/// length, or gives the unit fallback normal, so `|n|` is within a few 2⁻²⁴ of 1, and `n · n`
/// within about 6·10⁻⁷ (computing it and subtracting 1 add under 3·10⁻⁷), three times inside
/// this. A call holding any other normal ranks them all by the full cosine instead.
const UNIT_LENGTH_STRAY: f32 = 2e-6;

/// The normals [`lane_sum`] and [`Scan`] take side by side: four, a vector's width of
/// `f32` on every target.
const LANES: usize = 4;

/// The floats of a round of [`LANES`] normals.
const ROUND: usize = 3 * LANES;

/// A cone of directions: an axis and a half-angle, as a display tile stores the cone holding
/// every finest normal under it (see the module notes).
///
/// It bounds the **finest normals under the tile**, not the tile's own triangles: it is for level
/// choice (whether finer detail under the tile can face the camera), not for back-face culling
/// of what the tile draws, whose triangles can face outside it.
///
/// Every `u32` is one ([`FinestNormalCone::from_bits`]); a half-angle of 180° holds every
/// direction ([`FinestNormalCone::EVERYTHING`], also the [`Default`], what a tile whose normals
/// aren't known is given). Bits with the half-angle's top count and other axis bits cover
/// everything too but aren't equal to [`FinestNormalCone::EVERYTHING`] (the builds never make
/// them, but a file may hold them): ask [`FinestNormalCone::covers_everything`], never compare.
///
/// # Testing in `f32`
///
/// [`FinestNormalCone::contains`] measures in `f64`, and the guarantee is to that measure. A
/// consumer testing in `f32`, as a GPU's `dot(n, axis) >= cos(half)` does, must widen the cone by
/// a margin, such as one step (π/4095): `acos` is ill-conditioned near 1, so an `f32` dot product
/// of unit vectors puts the angle off by up to about 4·10⁻⁴ rad.
///
/// # Determinism
///
/// The same inputs give the same bits on one platform. The axis and the half-angle go through
/// `atan2`, `sin` and `cos`, so another platform's maths library may round differently and, near a
/// step's boundary, give a cone a step apart; either is conservative.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct FinestNormalCone(u32);

/// A direction in `f64`, not necessarily unit.
type Wide = [f64; 3];

impl FinestNormalCone {
    /// The cone that holds every direction: a half-angle of 180°, around `+z`.
    pub const EVERYTHING: Self = Self(HALF_MAX << HALF_SHIFT);

    /// The cone `bits` store, as [`FinestNormalCone::to_bits`] gave them: every `u32` is one.
    #[must_use]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    /// The cone as stored: the axis's two octahedral components in the low bits, the
    /// half-angle's step count in the high ones (see the module notes).
    #[must_use]
    pub const fn to_bits(self) -> u32 {
        self.0
    }

    /// Whether the cone holds every direction (its half-angle is 180°).
    #[must_use]
    pub const fn covers_everything(self) -> bool {
        self.steps() == HALF_MAX
    }

    /// The decoded axis, a unit vector (`+z` for [`FinestNormalCone::EVERYTHING`]).
    #[must_use]
    pub fn axis(self) -> Vec3 {
        let component = |shift: u32| {
            let raw = (self.0 >> shift) & AXIS_MASK;
            // Sign-extended from its `AXIS_BITS`: the top bit is the sign.
            let unused = 32 - u32::from(AXIS_BITS);
            let signed = (raw << unused).cast_signed() >> unused;
            // A sign-extended `AXIS_BITS`-bit value is within ±2^(AXIS_BITS − 1), which an
            // `i16` holds: the fallback is never taken.
            let small = i16::try_from(signed).unwrap_or(0);
            (f32::from(small) / levels(AXIS_BITS)).clamp(-1.0, 1.0)
        };
        from_octahedral(component(0), component(u32::from(AXIS_BITS)))
    }

    /// The decoded half-angle, in radians: a whole number of steps of `π / 4095`, up to π.
    #[must_use]
    pub fn half_angle(self) -> f32 {
        if self.covers_everything() {
            return core::f32::consts::PI;
        }
        half_of(self.steps())
    }

    /// Whether `direction` lies within the cone: its angle to the decoded axis is at most the
    /// decoded half-angle, measured in `f64` (a zero vector, which has no direction, is held by
    /// every cone).
    #[must_use]
    pub fn contains(self, direction: Vec3) -> bool {
        self.covers_everything()
            || angle(wide(self.axis()), wide(direction)) <= f64::from(self.half_angle())
    }

    /// Whether the cone holds every direction `other` holds: the angle between the axes plus
    /// `other`'s half-angle is at most this one's.
    #[must_use]
    pub fn holds(self, other: Self) -> bool {
        if self.covers_everything() {
            return true;
        }
        !other.covers_everything()
            && angle(wide(self.axis()), wide(other.axis())) + f64::from(other.half_angle())
                <= f64::from(self.half_angle())
    }

    /// The cone holding every one of `normals`, as stored: around their sum, its half-angle the
    /// largest angle from the decoded axis to any of them, rounded up (see the module notes).
    ///
    /// A zero normal has no direction, so it widens nothing. Normals that cancel out (a zero
    /// sum) get the axis `+z`; normals spread over more than a hemisphere around the axis may
    /// need the cone that covers everything.
    ///
    /// The sum is taken in `f32` in four lanes, each normal's lane set by its place in order, so
    /// the same normals give the same bits however they are cut into runs. It only picks the axis:
    /// the half-angle is measured from the decoded axis, which is what makes the cone hold
    /// them. The widest normal is found by its `f32` cosine to the decoded axis (`a · n` alone
    /// when every normal is of unit length, as the finest normals are; the full cosine
    /// otherwise), and only those within `COSINE_TOLERANCE` (10⁻⁵) of the least are measured
    /// with the `atan2` that [`FinestNormalCone::contains`] uses, so the half-angle bounds every
    /// normal as that measures it.
    ///
    /// # Errors
    ///
    /// [`DisplayError::Shape`] for no normals; [`DisplayError::NotFinite`] for a normal that
    /// isn't finite, or normals so large their sum is not.
    pub fn covering(normals: &[Vec3]) -> Result<Self, DisplayError> {
        Self::covering_runs(&[normals])
    }

    /// [`FinestNormalCone::covering`] of the normals in `runs`, one run after another: the same
    /// bits as the runs joined into one slice, read in place (the rows of a block, without
    /// copying them out).
    ///
    /// # Errors
    ///
    /// As [`FinestNormalCone::covering`].
    pub fn covering_runs(runs: &[&[Vec3]]) -> Result<Self, DisplayError> {
        if runs.iter().all(|run| run.is_empty()) {
            return Err(DisplayError::Shape);
        }
        // A normal that isn't finite makes the sum not finite: checked once, here.
        let sum = lane_sum(runs);
        if !sum.is_finite() {
            return Err(DisplayError::NotFinite);
        }
        Ok(Self::around(wide(sum), |axis| {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "the decoded axis, an f32 vector widened: narrowed back exactly"
            )]
            let a = Vec3::new(axis[0] as f32, axis[1] as f32, axis[2] as f32);
            let scan = Scan::of(runs, a);
            if !scan.unit() {
                return widest_by_cosine(runs, axis, a);
            }
            let band = scan.least.iter().fold(f32::INFINITY, |m, &l| m.min(l)) + COSINE_TOLERANCE;
            widest_unit(runs, &scan.least, axis, a, band)
        }))
    }

    /// The cone holding every one of `cones`, as stored: their union, as a parent tile's cone
    /// holds its children's.
    ///
    /// The axis is the cones merged two at a time (the least cone holding two cones, in `f64`),
    /// in the order of their bits, so the same cones give the same bits in any order; its
    /// half-angle is the largest `angle(axis, child's axis) + child's half-angle` from the
    /// decoded axis, rounded up (see the module notes). Any of them covering everything makes
    /// the union cover everything.
    ///
    /// # Errors
    ///
    /// [`DisplayError::Shape`] for no cones.
    pub fn holding(cones: &[Self]) -> Result<Self, DisplayError> {
        if cones.iter().any(|c| c.covers_everything()) {
            return Ok(Self::EVERYTHING);
        }
        // In the order of their bits, without sorting a copy: the next is the least past the
        // last taken (a parent has at most four children, so the walk is short). A cone equal to
        // one taken is skipped, which changes nothing: a cone merged with itself is itself.
        let after = |last: Option<u32>| {
            cones
                .iter()
                .map(|c| c.0)
                .filter(|&bits| last.is_none_or(|l| bits > l))
                .min()
        };
        let first = after(None).ok_or(DisplayError::Shape)?;
        let as_wide = |bits: u32| {
            let cone = Self(bits);
            (unit(wide(cone.axis())), f64::from(cone.half_angle()))
        };
        let mut merged = as_wide(first);
        let mut last = first;
        while let Some(next) = after(Some(last)) {
            merged = merge(merged, as_wide(next));
            last = next;
        }
        Ok(Self::around(merged.0, |decoded| {
            cones
                .iter()
                .map(|c| reach(decoded, wide(c.axis())) + f64::from(c.half_angle()))
                .fold(0.0, f64::max)
        }))
    }

    /// The cone around `axis` (quantized) whose half-angle is `needs` of the decoded axis
    /// (measured with [`reach`]), rounded up to a step; the cone that covers everything if that
    /// reaches 180°.
    fn around(axis: Wide, needs: impl Fn(Wide) -> f64) -> Self {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "only the direction matters, and it is quantized far coarser than an f32"
        )]
        let narrow = Vec3::new(axis[0] as f32, axis[1] as f32, axis[2] as f32);
        let (u, v) = octahedral(narrow);
        let pack = |c: f32| u32::from(quantize_unit(c, AXIS_BITS).cast_unsigned()) & AXIS_MASK;
        let axis_bits = pack(u) | pack(v) << AXIS_BITS;
        let need = needs(wide(Self(axis_bits).axis()));
        steps_up(need).map_or(Self::EVERYTHING, |steps| {
            Self(axis_bits | steps << HALF_SHIFT)
        })
    }

    /// The half-angle's step count.
    const fn steps(self) -> u32 {
        self.0 >> HALF_SHIFT
    }
}

impl Default for FinestNormalCone {
    /// [`FinestNormalCone::EVERYTHING`]: the cone of a tile whose normals aren't known.
    fn default() -> Self {
        Self::EVERYTHING
    }
}

impl fmt::Debug for FinestNormalCone {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.covers_everything() {
            return f.write_str("FinestNormalCone::EVERYTHING");
        }
        f.debug_struct("FinestNormalCone")
            .field("axis", &self.axis())
            .field("half_degrees", &self.half_angle().to_degrees())
            .finish()
    }
}

/// The sum of the normals in `runs`, in `f32`, a round of four normals at a time.
///
/// The normal at place `k` (counting through the runs in order) is summed in lane `k mod LANES`,
/// each lane adds in order, and the lanes are summed pairwise at the end. The lanes are
/// independent, so the additions run side by side as vector operations, not as one chain; and
/// each normal's lane depends on its place alone, not on how the normals are cut into runs, so a
/// block read row by row and one read tile run by tile run give the same bits.
fn lane_sum(runs: &[&[Vec3]]) -> Vec3 {
    // Lane `l`'s components at `3l..3l + 3`: a round's floats add to them one to one.
    let mut lanes = [0.0_f32; ROUND];
    // Where the next normal's floats go.
    let mut next = 0;
    for run in runs {
        let flat: &[f32] = bytemuck::cast_slice(run);
        // The floats that bring the lanes round to the first, then whole rounds of them.
        let head = ((ROUND - next) % ROUND).min(flat.len());
        let (first, rest) = flat.split_at_checked(head).unwrap_or((flat, &[]));
        let heads = lanes.iter_mut().skip(next).zip(first);
        for (lane, &c) in heads {
            *lane += c;
        }
        let (rounds, tail) = rest.as_chunks::<ROUND>();
        for round in rounds {
            let sums = lanes.iter_mut().zip(round);
            for (lane, &c) in sums {
                *lane += c;
            }
        }
        let tails = lanes.iter_mut().zip(tail);
        for (lane, &c) in tails {
            *lane += c;
        }
        next = (next + flat.len()) % ROUND;
    }
    let lane = |l: usize| {
        let at = |c: usize| lanes.get(3 * l + c).copied().unwrap_or(0.0);
        Vec3::new(at(0), at(1), at(2))
    };
    (lane(0) + lane(1)) + (lane(2) + lane(3))
}

/// What a pass over the normals tells against the decoded axis `a`: the least `a · n` of each
/// run, and the shortest and longest `n · n` of all.
#[derive(Debug, Clone, PartialEq)]
struct Scan {
    /// The least `a · n` of each run, in order.
    least: Vec<f32>,
    /// The least `n · n`.
    shortest: f32,
    /// The largest `n · n`.
    longest: f32,
}

impl Scan {
    /// The pass over `runs`, four normals side by side, each `a · n` summed in the order
    /// `Vec3::dot` takes it, so the band pass, which takes `a.dot(n)`, gets the same bits.
    fn of(runs: &[&[Vec3]], a: Vec3) -> Self {
        let mut run_least = Vec::with_capacity(runs.len());
        let mut lo = [f32::INFINITY; LANES];
        let mut hi = [0.0_f32; LANES];
        // A run's last normals, fewer than a round, one at a time.
        let (mut tail_lo, mut tail_hi) = (f32::INFINITY, 0.0_f32);
        for run in runs {
            let mut least = [f32::INFINITY; LANES];
            let flat: &[f32] = bytemuck::cast_slice(run);
            let (rounds, tail) = flat.as_chunks::<ROUND>();
            for round in rounds {
                let [x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3] = *round;
                let along = [
                    a.x * x0 + a.y * y0 + a.z * z0,
                    a.x * x1 + a.y * y1 + a.z * z1,
                    a.x * x2 + a.y * y2 + a.z * z2,
                    a.x * x3 + a.y * y3 + a.z * z3,
                ];
                let length = [
                    x0 * x0 + y0 * y0 + z0 * z0,
                    x1 * x1 + y1 * y1 + z1 * z1,
                    x2 * x2 + y2 * y2 + z2 * z2,
                    x3 * x3 + y3 * y3 + z3 * z3,
                ];
                // Selects, not `min`/`max`: four independent lanes, no NaN handling (a NaN
                // shows in the sum).
                let nearest = least.iter_mut().zip(along);
                for (l, d) in nearest {
                    *l = if d < *l { d } else { *l };
                }
                let lengths = lo.iter_mut().zip(&mut hi).zip(length);
                for ((s, g), q) in lengths {
                    *s = if q < *s { q } else { *s };
                    *g = if q > *g { q } else { *g };
                }
            }
            let [l0, l1, l2, l3] = least;
            let mut this = l0.min(l1).min(l2.min(l3));
            let (rest, _) = tail.as_chunks::<3>();
            for &[x, y, z] in rest {
                let n = Vec3::new(x, y, z);
                let (d, q) = (a.dot(n), n.dot(n));
                this = this.min(d);
                tail_lo = tail_lo.min(q);
                tail_hi = tail_hi.max(q);
            }
            run_least.push(this);
        }
        let [s0, s1, s2, s3] = lo;
        let [g0, g1, g2, g3] = hi;
        Self {
            least: run_least,
            shortest: s0.min(s1).min(s2.min(s3)).min(tail_lo),
            longest: g0.max(g1).max(g2.max(g3)).max(tail_hi),
        }
    }

    /// Whether every normal is of unit length to within [`UNIT_LENGTH_STRAY`].
    fn unit(&self) -> bool {
        1.0 - self.shortest <= UNIT_LENGTH_STRAY && self.longest - 1.0 <= UNIT_LENGTH_STRAY
    }
}

/// The widest angle, as [`reach`] measures it from `axis` (the decoded axis, `a` in `f32`), of
/// the normals in `runs` whose `a · n` is at most `band`, visiting only the runs whose `least`
/// (from [`Scan`], one a run) is.
///
/// For normals of unit length (see [`UNIT_LENGTH_STRAY`]), every normal past the band is
/// narrower than one inside it.
fn widest_unit(runs: &[&[Vec3]], least: &[f32], axis: Wide, a: Vec3, band: f32) -> f64 {
    let mut need = 0.0_f64;
    let reaching = runs
        .iter()
        .zip(least)
        .filter(|&(_, &l)| l <= band)
        .map(|(run, _)| run);
    for run in reaching {
        for &n in *run {
            if a.dot(n) <= band {
                need = need.max(reach(axis, wide(n)));
            }
        }
    }
    need
}

/// [`widest_unit`] for normals of any length.
///
/// They are ranked by their cosine to the axis, `(a · n) / √(n · n)`, zero normals left out
/// (they have no direction, and no angle), and the band is [`COSINE_TOLERANCE`] over the least.
fn widest_by_cosine(runs: &[&[Vec3]], axis: Wide, a: Vec3) -> f64 {
    let cosine = |n: Vec3| {
        let length = n.dot(n);
        (length > 0.0).then(|| a.dot(n) / length.sqrt())
    };
    let mut least = f32::INFINITY;
    for run in runs {
        for &n in *run {
            if let Some(c) = cosine(n) {
                least = least.min(c);
            }
        }
    }
    let band = least + COSINE_TOLERANCE;
    let mut need = 0.0_f64;
    for run in runs {
        for &n in *run {
            if cosine(n).is_some_and(|c| c <= band) {
                need = need.max(reach(axis, wide(n)));
            }
        }
    }
    need
}

/// The half-angle `steps` steps stand for, in radians, as decoded (an `f32` product, so it can
/// be computed the same way anywhere).
fn half_of(steps: u32) -> f32 {
    // A step count is a `HALF_BITS`-bit field (12 bits), which a `u16` holds: the fallback is
    // never taken.
    f32::from(u16::try_from(steps).unwrap_or(u16::MAX)) * HALF_STEP
}

/// The fewest steps whose decoded half-angle is at least `need` radians, or `None` if that
/// would be 180° or more (or `need` isn't a number): the cone that covers everything.
fn steps_up(need: f64) -> Option<u32> {
    if need.is_nan() || need >= PI {
        return None;
    }
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a non-negative count under HALF_MAX + 1, rounded down"
    )]
    let below = (need.max(0.0) / f64::from(HALF_STEP)).floor() as u32;
    // From a step under the quotient, up to the first decoded half-angle (an `f32` product,
    // which can round either way from the `f64` one) at or past `need`: one or two steps.
    let mut steps = below.saturating_sub(1);
    while steps < HALF_MAX && f64::from(half_of(steps)) < need {
        steps += 1;
    }
    (steps < HALF_MAX).then_some(steps)
}

/// The least cone (axis, half-angle in radians) holding the cones `a` and `b`, whose axes are
/// unit.
///
/// One of them if it holds the other, else the cone spanning both, its axis on the great circle
/// through theirs. Opposite axes give a cone of 180° around `a`'s.
fn merge(a: (Wide, f64), b: (Wide, f64)) -> (Wide, f64) {
    let ((a_axis, a_half), (b_axis, b_half)) = (a, b);
    let between = angle(a_axis, b_axis);
    if between + b_half <= a_half {
        return a;
    }
    if between + a_half <= b_half {
        return b;
    }
    let half = (between + a_half + b_half) / 2.0;
    // `b`'s axis less its part along `a`'s: the way from `a`'s axis towards `b`'s.
    let along = dot(a_axis, b_axis);
    let towards = [
        b_axis[0] - a_axis[0] * along,
        b_axis[1] - a_axis[1] * along,
        b_axis[2] - a_axis[2] * along,
    ];
    let length = dot(towards, towards).sqrt();
    if length <= 0.0 {
        return (a_axis, PI);
    }
    let turn = half - a_half;
    let (sin, cos) = turn.sin_cos();
    let axis = [
        a_axis[0] * cos + towards[0] / length * sin,
        a_axis[1] * cos + towards[1] / length * sin,
        a_axis[2] * cos + towards[2] / length * sin,
    ];
    (axis, half)
}

/// [`angle`] raised by [`SLACK`], or exactly 0 where the angle is: what a half-angle must reach
/// to hold `b` around `a`.
fn reach(a: Wide, b: Wide) -> f64 {
    let between = angle(a, b);
    if between > 0.0 { between + SLACK } else { 0.0 }
}

/// `v` in `f64`, exactly.
fn wide(v: Vec3) -> Wide {
    [f64::from(v.x), f64::from(v.y), f64::from(v.z)]
}

/// The dot product of `a` and `b`.
fn dot(a: Wide, b: Wide) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// `v` scaled to unit length (`v` itself if it is zero).
fn unit(v: Wide) -> Wide {
    let length = dot(v, v).sqrt();
    if length > 0.0 {
        [v[0] / length, v[1] / length, v[2] / length]
    } else {
        v
    }
}

/// The angle between `a` and `b` in radians, from `atan2(|a × b|, a · b)` (an `acos` of the dot
/// product loses small angles to rounding); 0 if either is zero. Neither needs to be unit.
fn angle(a: Wide, b: Wide) -> f64 {
    let cross = [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    let (sin, cos) = (dot(cross, cross).sqrt(), dot(a, b));
    // A zero vector gives `atan2(0, ±0)`, and `atan2(0, −0)` is π: a zero normal facing an axis
    // with negative components would read as opposite it. It has no direction: 0.
    if sin == 0.0 && cos == 0.0 {
        return 0.0;
    }
    sin.atan2(cos)
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;
