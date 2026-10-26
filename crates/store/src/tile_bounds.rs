//! What the index knows of a tile without evaluating it: its [`TileBounds`].
//!
//! The bounds are the box of the tile's positions ([`Aabb`]) and a cone bounding its normals
//! ([`NormalCone`]). A leaf module: it needs only `geom`, so the store, its files and the index
//! can all use it.

use geom::Vec3;

/// A cone bounding a set of unit normals: every normal is within its half-angle of its axis.
///
/// The cone of a tile tells, without evaluating the tile, whether all of it faces away from a
/// viewer or a brush. When the normals spread too far for a cone to say anything (or aren't
/// all valid), the cone is [`NormalCone::EVERYTHING`], which holds every direction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalCone {
    /// The cone's axis, a unit vector.
    axis: Vec3,
    /// The cosine of the cone's half-angle, in `[-1, 1]`: `-1` holds every direction.
    cos_half_angle: f32,
}

impl NormalCone {
    /// The cone that holds every direction: what a tile gets when its normals can't be bounded
    /// any tighter.
    pub const EVERYTHING: Self = Self {
        axis: Vec3::new(0.0, 0.0, 1.0),
        cos_half_angle: -1.0,
    };
    /// How far below the least `cos` of the normals the bound is set, so rounding in the dot
    /// products never leaves one of them just outside the cone.
    const SLACK: f32 = 1e-5;
    /// How far from 1 a stored axis's squared length may be: a normalised vector's is within a
    /// few units in the last place, and anything further isn't an axis this code made.
    const UNIT_SLACK: f32 = 1e-4;

    /// A cone holding every one of `normals`: its axis is their normalised mean, its half-angle
    /// that of the normal furthest from it.
    ///
    /// That isn't the narrowest bounding cone, but it is cheap and close to it for the gently
    /// curved normals of one tile. [`NormalCone::EVERYTHING`] if `normals` is empty, holds a
    /// normal that isn't finite, or averages to nothing (normals spread all around).
    #[must_use]
    pub fn from_normals(normals: &[Vec3]) -> Self {
        if !normals.iter().all(|n| n.is_finite()) {
            return Self::EVERYTHING;
        }
        let Some(axis) = normals
            .iter()
            .fold(Vec3::ZERO, |sum, &n| sum + n)
            .normalized()
        else {
            return Self::EVERYTHING;
        };
        let least = normals
            .iter()
            .filter_map(|n| n.normalized())
            .map(|n| axis.dot(n))
            .fold(1.0_f32, f32::min);
        Self {
            axis,
            cos_half_angle: (least - Self::SLACK).max(-1.0),
        }
    }

    /// The cone's axis, a unit vector.
    #[must_use]
    pub const fn axis(&self) -> Vec3 {
        self.axis
    }

    /// The cosine of the cone's half-angle: `-1` for a cone that holds every direction.
    #[must_use]
    pub const fn cos_half_angle(&self) -> f32 {
        self.cos_half_angle
    }

    /// The cone with `axis` and `cos_half_angle` as a stored bound holds them, bit for bit;
    /// `None` unless both are finite, the cosine is in `[-1, 1]` and the axis is a unit vector
    /// (to within [`NormalCone::UNIT_SLACK`], as normalising leaves it).
    pub(crate) fn from_parts(axis: Vec3, cos_half_angle: f32) -> Option<Self> {
        let unit = (axis.dot(axis) - 1.0).abs() <= Self::UNIT_SLACK;
        let valid = axis.is_finite() && unit && (-1.0..=1.0).contains(&cos_half_angle);
        valid.then_some(Self {
            axis,
            cos_half_angle,
        })
    }

    /// Whether the direction of `n` lies in the cone. A zero or non-finite `n` has no
    /// direction, and lies in no cone.
    #[must_use]
    pub fn contains(&self, n: Vec3) -> bool {
        n.normalized()
            .is_some_and(|n| self.axis.dot(n) >= self.cos_half_angle)
    }
}

/// What the index knows of one tile without evaluating it: where it is and which way it faces.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TileBounds {
    /// The box of the tile's evaluated positions.
    pub aabb: Aabb,
    /// A cone bounding the tile's normals.
    pub cone: NormalCone,
}

impl TileBounds {
    /// How many `u32` [`TileBounds::to_bits`] gives.
    pub(crate) const BITS: usize = 10;

    /// The bounds as bits, in the order the project file stores them: the box's low corner and
    /// high corner, the cone's axis (each `x`, `y`, `z`), then the cone's cosine.
    pub(crate) fn to_bits(self) -> [u32; Self::BITS] {
        let [min, max, axis] =
            [self.aabb.min(), self.aabb.max(), self.cone.axis].map(Vec3::to_bits);
        let cos = self.cone.cos_half_angle.to_bits();
        let mut bits = [0; Self::BITS];
        let all = min.into_iter().chain(max).chain(axis).chain([cos]);
        for (slot, value) in bits.iter_mut().zip(all) {
            *slot = value;
        }
        bits
    }

    /// The bounds [`TileBounds::to_bits`] gave these bits, bit for bit; `None` unless they make
    /// a valid box ([`Aabb::new`]) and cone ([`NormalCone::from_parts`]).
    pub(crate) fn from_bits(bits: [u32; Self::BITS]) -> Option<Self> {
        let [
            min_x,
            min_y,
            min_z,
            max_x,
            max_y,
            max_z,
            axis_x,
            axis_y,
            axis_z,
            cos,
        ] = bits.map(f32::from_bits);
        Some(Self {
            aabb: Aabb::new(
                Vec3::new(min_x, min_y, min_z),
                Vec3::new(max_x, max_y, max_z),
            )?,
            cone: NormalCone::from_parts(Vec3::new(axis_x, axis_y, axis_z), cos)?,
        })
    }

    /// The bounds of a tile with these evaluated `positions` and `normals`; `None` if no
    /// position is finite.
    #[must_use]
    pub fn of_tile(positions: &[Vec3], normals: &[Vec3]) -> Option<Self> {
        Some(Self {
            aabb: Aabb::from_points(positions.iter().copied())?,
            cone: NormalCone::from_normals(normals),
        })
    }
}

/// A coordinate axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Axis {
    /// The x axis.
    X,
    /// The y axis.
    Y,
    /// The z axis.
    Z,
}

impl Axis {
    /// All three axes, in order.
    pub(crate) const ALL: [Self; 3] = [Self::X, Self::Y, Self::Z];

    /// `v`'s component along this axis.
    pub(crate) const fn of(self, v: Vec3) -> f32 {
        match self {
            Self::X => v.x,
            Self::Y => v.y,
            Self::Z => v.z,
        }
    }
}

/// The component-wise minimum of `a` and `b`.
const fn component_min(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(a.x.min(b.x), a.y.min(b.y), a.z.min(b.z))
}

/// The component-wise maximum of `a` and `b`.
const fn component_max(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z))
}

/// An axis-aligned bounding box: every point between [`Aabb::min`] and [`Aabb::max`],
/// component by component.
///
/// Its corners are finite and `min ≤ max` in every component, so a box always holds at least
/// one point; the constructors check it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    /// The low corner. `pub(crate)`, as is `max`, for the tree in `index::bvh`, which reads
    /// both in its ray test and its splits; outside the crate they stay private.
    pub(crate) min: Vec3,
    /// The high corner (`pub(crate)`, see `min`).
    pub(crate) max: Vec3,
}

impl Aabb {
    /// The box from `min` to `max`, or `None` unless both are finite and `min ≤ max` in every
    /// component.
    #[must_use]
    pub fn new(min: Vec3, max: Vec3) -> Option<Self> {
        let ordered = Axis::ALL.iter().all(|&a| a.of(min) <= a.of(max));
        (min.is_finite() && max.is_finite() && ordered).then_some(Self { min, max })
    }

    /// The smallest box holding every finite point of `points`, or `None` if there is none.
    /// Points with a NaN or infinite component are skipped.
    #[must_use]
    pub fn from_points(points: impl IntoIterator<Item = Vec3>) -> Option<Self> {
        points
            .into_iter()
            .filter(|p| p.is_finite())
            .map(Self::point)
            .reduce(Self::union)
    }

    /// The box holding just `p`, which must be finite.
    const fn point(p: Vec3) -> Self {
        Self { min: p, max: p }
    }

    /// The low corner.
    #[must_use]
    pub const fn min(&self) -> Vec3 {
        self.min
    }

    /// The high corner.
    #[must_use]
    pub const fn max(&self) -> Vec3 {
        self.max
    }

    /// The smallest box holding both boxes.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self {
            min: component_min(self.min, other.min),
            max: component_max(self.max, other.max),
        }
    }

    /// The centre of the box.
    #[must_use]
    pub fn centroid(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// The box's surface area: `2 (dx dy + dy dz + dz dx)`.
    #[must_use]
    pub fn surface_area(&self) -> f32 {
        let d = self.max - self.min;
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }

    /// Whether `p` lies in the box (its faces included).
    #[must_use]
    pub fn contains(&self, p: Vec3) -> bool {
        Axis::ALL
            .iter()
            .all(|&a| (a.of(self.min)..=a.of(self.max)).contains(&a.of(p)))
    }

    /// The squared distance from `p` to the nearest point of the box: 0 inside it.
    #[must_use]
    pub fn distance_squared(&self, p: Vec3) -> f32 {
        let nearest = component_min(component_max(p, self.min), self.max);
        let d = p - nearest;
        d.dot(d)
    }
}
