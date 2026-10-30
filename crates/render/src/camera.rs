//! The camera, and the view it gives the GPU passes: view-projection, frustum planes, the eye
//! and the scale that turns a world length at a distance into pixels.
//!
//! The camera orbits a target point: it looks at [`Camera::target`] from [`Camera::distance`]
//! away, along the direction its orientation gives. Orbiting turns the orientation about the
//! target, panning moves the target in the view plane, zooming scales the distance.
//!
//! The projection is a perspective one with **reversed Z and an infinite far plane**: depth is
//! 1 on the near plane and falls towards 0 at infinity, which spreads a 32-bit float depth's
//! precision evenly over distance (the raster's `Depth32Float`). Nothing is clipped far away;
//! the near plane sits at [`NEAR_FRACTION`] of the distance to the target, so it follows the
//! camera in for close work.
//!
//! Math is `f64`, the camera and its matrices both: a handful of numbers, kept exact for long
//! drags and for views close to the surface. A [`View`] goes to the GPU as `f32`
//! ([`View::gpu`], the WGSL `View` of [`crate::cull::CULL_WGSL`]).
//!
//! `app` still has its own copy of [`Camera`] and [`Rotation`] (another change is editing it);
//! it moves to this one when that lands, keeping only the arcball and navigation.

use std::ops::{Add, Mul, Neg, Sub};

use bytemuck::{Pod, Zeroable};

use crate::surface::Size;

/// The near plane's distance, as a fraction of the camera's distance to its target.
pub const NEAR_FRACTION: f64 = 1e-3;

/// A vector in 3D, `f64`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Vec3 {
    /// The x component.
    pub x: f64,
    /// The y component.
    pub y: f64,
    /// The z component.
    pub z: f64,
}

impl Vec3 {
    /// The vector `(x, y, z)`.
    #[must_use]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// The dot product.
    #[must_use]
    pub fn dot(self, o: Self) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    /// The cross product, `self × o`.
    #[must_use]
    pub fn cross(self, o: Self) -> Self {
        Self::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }

    /// The length.
    #[must_use]
    pub fn length(self) -> f64 {
        self.dot(self).sqrt()
    }

    /// The vector scaled to unit length, or `None` for a vector too short to have a direction.
    #[must_use]
    pub fn normalized(self) -> Option<Self> {
        let length = self.length();
        (length > f64::EPSILON && length.is_finite()).then(|| self * length.recip())
    }

    /// The vector of `geom`'s `f32` vector.
    #[must_use]
    pub fn from_f32(v: geom::Vec3) -> Self {
        Self::new(f64::from(v.x), f64::from(v.y), f64::from(v.z))
    }
}

impl Add for Vec3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Sub for Vec3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Mul<f64> for Vec3 {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl Neg for Vec3 {
    type Output = Self;

    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}

/// A rotation, as a unit quaternion `w + xi + yj + zk`.
///
/// Made only as a unit quaternion ([`Rotation::IDENTITY`], [`Rotation::between`],
/// [`Rotation::about`]) and kept one by [`Rotation::then`], which renormalizes, so drift from a
/// long drag never skews the view.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rotation {
    /// The scalar part.
    w: f64,
    /// The vector part.
    v: Vec3,
}

impl Rotation {
    /// No rotation.
    pub const IDENTITY: Self = Self {
        w: 1.0,
        v: Vec3::new(0.0, 0.0, 0.0),
    };

    /// The shortest rotation carrying the unit vector `from` to the unit vector `to`: about
    /// their common normal, by the angle between them. Opposite vectors have no shortest
    /// rotation; they give a half turn about an axis perpendicular to `from`.
    #[must_use]
    pub fn between(from: Vec3, to: Vec3) -> Self {
        // The half-angle form: (1 + cos θ, sin θ · axis), normalized, is the rotation by θ.
        let w = 1.0 + from.dot(to);
        if w <= f64::EPSILON {
            let side = if from.x.abs() < 0.9 {
                Vec3::new(1.0, 0.0, 0.0)
            } else {
                Vec3::new(0.0, 1.0, 0.0)
            };
            return Self {
                w: 0.0,
                v: from.cross(side),
            }
            .normalized();
        }
        Self {
            w,
            v: from.cross(to),
        }
        .normalized()
    }

    /// The rotation by `angle` radians about the unit vector `axis`, counter-clockwise looking
    /// down the axis.
    #[must_use]
    pub fn about(axis: Vec3, angle: f64) -> Self {
        let (sin, cos) = (angle / 2.0).sin_cos();
        Self {
            w: cos,
            v: axis * sin,
        }
        .normalized()
    }

    /// The rotation undoing this one.
    #[must_use]
    pub fn inverse(self) -> Self {
        Self {
            w: self.w,
            v: -self.v,
        }
    }

    /// This rotation, then `next`: the product `next · self`, renormalized.
    #[must_use]
    pub fn then(self, next: Self) -> Self {
        Self {
            w: next.w * self.w - next.v.dot(self.v),
            v: self.v * next.w + next.v * self.w + next.v.cross(self.v),
        }
        .normalized()
    }

    /// `p` rotated.
    #[must_use]
    pub fn apply(self, p: Vec3) -> Vec3 {
        // p + 2w(v × p) + 2v × (v × p), for a unit quaternion.
        let t = self.v.cross(p) * 2.0;
        p + t * self.w + self.v.cross(t)
    }

    /// The same rotation scaled back to unit length (a zero quaternion, which only rounding
    /// from a unit one can't reach, becomes the identity).
    fn normalized(self) -> Self {
        let length = (self.w * self.w + self.v.dot(self.v)).sqrt();
        if length <= f64::EPSILON {
            return Self::IDENTITY;
        }
        Self {
            w: self.w / length,
            v: self.v * length.recip(),
        }
    }
}

/// The camera: where it looks, from how far, and which way is up.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    /// The point it orbits and looks at.
    target: Vec3,
    /// How far the eye is from the target; always within [`Camera::MIN_DISTANCE`] and
    /// [`Camera::MAX_DISTANCE`].
    distance: f64,
    /// The view's orientation: carries the view's axes (x right, y up, z towards the eye) into
    /// the world.
    orientation: Rotation,
    /// The vertical field of view, in radians.
    fov_y: f64,
}

impl Default for Camera {
    /// Looking at the origin down −z from 3 units away, y up: a unit sphere, as the benchmark
    /// project is, fills about two thirds of the view's height.
    fn default() -> Self {
        Self {
            target: Vec3::default(),
            distance: 3.0,
            orientation: Rotation::IDENTITY,
            fov_y: 45_f64.to_radians(),
        }
    }
}

impl Camera {
    /// The farthest the eye goes from the target.
    pub const MAX_DISTANCE: f64 = 1e4;
    /// The closest the eye comes to the target.
    pub const MIN_DISTANCE: f64 = 1e-4;

    /// A camera looking at `target` from `distance` away (clamped to its limits), oriented by
    /// `orientation`, with a vertical field of view of `fov_y` radians (clamped to 1° to 170°).
    #[must_use]
    pub fn new(target: Vec3, distance: f64, orientation: Rotation, fov_y: f64) -> Self {
        let mut camera = Self {
            target,
            distance: Self::default().distance,
            orientation,
            fov_y: fov_y.clamp(1_f64.to_radians(), 170_f64.to_radians()),
        };
        camera.zoom(distance / camera.distance);
        camera
    }

    /// The point it looks at.
    #[must_use]
    pub const fn target(&self) -> Vec3 {
        self.target
    }

    /// How far the eye is from the target.
    #[must_use]
    pub const fn distance(&self) -> f64 {
        self.distance
    }

    /// The vertical field of view, in radians.
    #[must_use]
    pub const fn fov_y(&self) -> f64 {
        self.fov_y
    }

    /// Where the eye is.
    #[must_use]
    pub fn eye(&self) -> Vec3 {
        self.target + self.axis(Vec3::new(0.0, 0.0, 1.0)) * self.distance
    }

    /// The view's `axis` (x right, y up, z towards the eye) in world space.
    #[must_use]
    pub fn axis(&self, axis: Vec3) -> Vec3 {
        self.orientation.apply(axis)
    }

    /// Turns the camera about the target by `drag`, a rotation of the scene as seen in the view
    /// (an arcball's): the scene appears to turn by it, so the camera turns the other way.
    pub fn orbit(&mut self, drag: Rotation) {
        self.orientation = drag.inverse().then(self.orientation);
    }

    /// Moves the target in the view plane so the point under the pointer follows it, moved by
    /// `dx`, `dy` pixels (right and down) in a viewport `height` pixels tall: at the target's
    /// depth, the scene keeps pace with the pointer.
    pub fn pan(&mut self, dx: f64, dy: f64, height: f64) {
        let per_pixel = 2.0 * self.distance * (self.fov_y / 2.0).tan() / height.max(1.0);
        let right = self.axis(Vec3::new(1.0, 0.0, 0.0));
        let up = self.axis(Vec3::new(0.0, 1.0, 0.0));
        self.target = self.target - right * (dx * per_pixel) + up * (dy * per_pixel);
    }

    /// Moves the eye towards the target (`factor` below 1) or away (above 1), scaling the
    /// distance, clamped to [`Camera::MIN_DISTANCE`]..[`Camera::MAX_DISTANCE`]. A factor that
    /// isn't a positive finite number does nothing.
    pub fn zoom(&mut self, factor: f64) {
        if factor.is_finite() && factor > 0.0 {
            self.distance = (self.distance * factor).clamp(Self::MIN_DISTANCE, Self::MAX_DISTANCE);
        }
    }

    /// The view into a viewport of `size`: the near plane at [`NEAR_FRACTION`] of the
    /// distance.
    #[must_use]
    pub fn view(&self, size: Size) -> View {
        let basis = [
            self.axis(Vec3::new(1.0, 0.0, 0.0)),
            self.axis(Vec3::new(0.0, 1.0, 0.0)),
            self.axis(Vec3::new(0.0, 0.0, 1.0)),
        ];
        View::from_basis(
            self.eye(),
            basis,
            self.fov_y,
            size,
            self.distance * NEAR_FRACTION,
        )
    }
}

/// A 4 × 4 matrix, `f64`, row-major: `rows[r][c]`. Points are columns: `M · (x, y, z, 1)`.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Mat4 {
    /// The rows.
    rows: [[f64; 4]; 4],
}

impl Mat4 {
    /// The product `self · rhs`.
    fn mul(&self, rhs: &Self) -> Self {
        let rows = self.rows.map(|row| {
            let column = |c: usize| {
                row.iter()
                    .zip(&rhs.rows)
                    .map(|(a, rhs_row)| a * rhs_row.get(c).copied().unwrap_or(0.0))
                    .sum()
            };
            [column(0), column(1), column(2), column(3)]
        });
        Self { rows }
    }

    /// `self · (p, 1)`: clip coordinates `(x, y, z, w)` for a view-projection.
    fn transform(&self, p: Vec3) -> [f64; 4] {
        self.rows
            .map(|[a, b, c, d]| a * p.x + b * p.y + c * p.z + d)
    }

    /// The matrix as WGSL's `mat4x4<f32>` holds it: columns.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the GPU works in f32: the matrix is built in f64 and rounded once, here"
    )]
    fn columns_f32(&self) -> [[f32; 4]; 4] {
        let [r0, r1, r2, r3] = self.rows;
        let column =
            |c: usize| [r0, r1, r2, r3].map(|row| row.get(c).copied().unwrap_or(0.0) as f32);
        [column(0), column(1), column(2), column(3)]
    }
}

/// The frustum's planes of the view-projection `clip_from_world`, as [`View::planes`] holds them.
fn frustum_planes(clip_from_world: &Mat4) -> [[f64; 4]; 6] {
    let [r0, r1, r2, r3] = clip_from_world.rows;
    let add = |a: [f64; 4], b: [f64; 4], sign: f64| {
        [
            a[0] + sign * b[0],
            a[1] + sign * b[1],
            a[2] + sign * b[2],
            a[3] + sign * b[3],
        ]
    };
    let normalize = |p: [f64; 4]| {
        let length = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        if length > 0.0 {
            p.map(|c| c / length)
        } else {
            [0.0, 0.0, 0.0, 1.0]
        }
    };
    [
        normalize(add(r3, r0, 1.0)),
        normalize(add(r3, r0, -1.0)),
        normalize(add(r3, r1, 1.0)),
        normalize(add(r3, r1, -1.0)),
        // Depth at most 1: z ≤ w.
        normalize(add(r3, r2, -1.0)),
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Where inside its pixel a frame samples the scene: an offset from the pixel's centre, in
/// pixels, each axis in `[-0.5, 0.5)` (x right, y down). [`View::jittered`] applies it.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Jitter {
    /// Right, pixels.
    x: f64,
    /// Down, pixels.
    y: f64,
}

impl Jitter {
    /// The pixel's centre: no offset.
    pub const CENTRE: Self = Self { x: 0.0, y: 0.0 };

    /// The offset `(x, y)` pixels from the centre; `None` unless both lie in `[-0.5, 0.5)`.
    #[must_use]
    pub fn new(x: f64, y: f64) -> Option<Self> {
        let inside = |v: f64| (-0.5..0.5).contains(&v);
        (inside(x) && inside(y)).then_some(Self { x, y })
    }

    /// The offset right, pixels.
    #[must_use]
    pub const fn x(self) -> f64 {
        self.x
    }

    /// The offset down, pixels.
    #[must_use]
    pub const fn y(self) -> f64 {
        self.y
    }
}

/// A view: where the eye is and how the world projects onto a viewport, `f64`.
///
/// Built from a [`Camera`] ([`Camera::view`]) or directly ([`View::look_at`]); handed to the
/// GPU as a [`GpuView`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct View {
    /// Where the eye is.
    eye: Vec3,
    /// The view's axes in the world: right, up, and towards the eye.
    basis: [Vec3; 3],
    /// The view-projection: world to clip coordinates, reversed Z, infinite far plane.
    clip_from_world: Mat4,
    /// The frustum's planes, `(n, d)` with `n · p + d ≥ 0` inside and `|n| = 1`: left, right,
    /// bottom, top, near; the sixth, the far plane at infinity, holds everything (`(0, 0, 0,
    /// 1)`).
    planes: [[f64; 4]; 6],
    /// The viewport.
    size: Size,
    /// Pixels a world length of 1 covers, facing the eye at a distance of 1.
    pixels_per_unit: f64,
    /// The near plane's distance from the eye.
    near: f64,
}

impl View {
    /// The view from `eye` looking at `target`, with `up` as the view's up as far as it can be
    /// (perpendicular to the line of sight), a vertical field of view of `fov_y` radians, into
    /// a viewport of `size`, the near plane `near` from the eye.
    ///
    /// `None` if `eye` and `target` coincide, `up` is along the line of sight, `fov_y` isn't
    /// strictly between 0 and π, or `near` isn't positive and finite.
    #[must_use]
    pub fn look_at(
        eye: Vec3,
        target: Vec3,
        up: Vec3,
        fov_y: f64,
        size: Size,
        near: f64,
    ) -> Option<Self> {
        let back = (eye - target).normalized()?;
        let right = up.cross(back).normalized()?;
        let true_up = back.cross(right);
        let valid = fov_y > 0.0 && fov_y < std::f64::consts::PI && near > 0.0 && near.is_finite();
        valid.then(|| Self::from_basis(eye, [right, true_up, back], fov_y, size, near))
    }

    /// The view from `eye` along the orthonormal `basis` (right, up, towards the eye).
    fn from_basis(eye: Vec3, basis: [Vec3; 3], fov_y: f64, size: Size, near: f64) -> Self {
        let [right, up, back] = basis;
        let view = Mat4 {
            rows: [
                [right.x, right.y, right.z, -right.dot(eye)],
                [up.x, up.y, up.z, -up.dot(eye)],
                [back.x, back.y, back.z, -back.dot(eye)],
                [0.0, 0.0, 0.0, 1.0],
            ],
        };
        let (width, height) = (f64::from(size.width()), f64::from(size.height()));
        let focal = (fov_y / 2.0).tan().recip();
        // Reversed Z, infinite far: clip z = near, clip w = −z_view, so depth = near / distance
        // along the line of sight: 1 on the near plane, towards 0 far away.
        let projection = Mat4 {
            rows: [
                [focal * height / width, 0.0, 0.0, 0.0],
                [0.0, focal, 0.0, 0.0],
                [0.0, 0.0, 0.0, near],
                [0.0, 0.0, -1.0, 0.0],
            ],
        };
        let clip_from_world = projection.mul(&view);
        Self {
            eye,
            basis,
            clip_from_world,
            planes: frustum_planes(&clip_from_world),
            size,
            pixels_per_unit: focal * height / 2.0,
            near,
        }
    }

    /// The same view with its image moved by `jitter`, a fraction of a pixel: pixel `(x, y)`'s
    /// centre then sees what this view shows at `(x + 0.5 + jitter.x, y + 0.5 + jitter.y)`
    /// (x right, y down), so frames drawn at different jitters sample the scene at different
    /// points inside each pixel ([`crate::aa`]).
    ///
    /// Only the projection's first two rows change (each gains a multiple of the last), so the
    /// eye, the depth of every point, the near plane, [`View::pixels_per_unit`] and every pixel
    /// length are bit for bit the same: what the level choice and the geomorph compute from them
    /// doesn't move with the jitter. The frustum's side planes move with the image.
    #[must_use]
    pub fn jittered(&self, jitter: Jitter) -> Self {
        let (width, height) = (f64::from(self.size.width()), f64::from(self.size.height()));
        // ndc x' = ndc x − 2 jx / width moves the image left by jx pixels; ndc y points up
        // while pixels count down, so y' = y + 2 jy / height moves it up by jy.
        let shift = [-2.0 * jitter.x / width, 2.0 * jitter.y / height];
        let mut clip_from_world = self.clip_from_world;
        let [r0, r1, _, r3] = &mut clip_from_world.rows;
        let shifted_rows = [r0, r1].into_iter().zip(shift);
        for (row, by) in shifted_rows {
            let entries = row.iter_mut().zip(*r3);
            for (value, last) in entries {
                *value += by * last;
            }
        }
        Self {
            clip_from_world,
            planes: frustum_planes(&clip_from_world),
            ..*self
        }
    }

    /// Where the eye is.
    #[must_use]
    pub const fn eye(&self) -> Vec3 {
        self.eye
    }

    /// The direction the view looks along, unit length.
    #[must_use]
    pub fn forward(&self) -> Vec3 {
        -self.basis[2]
    }

    /// The view's right, unit length.
    #[must_use]
    pub const fn right(&self) -> Vec3 {
        self.basis[0]
    }

    /// The view's up, unit length.
    #[must_use]
    pub const fn up(&self) -> Vec3 {
        self.basis[1]
    }

    /// The viewport.
    #[must_use]
    pub const fn size(&self) -> Size {
        self.size
    }

    /// Pixels a world length of 1 covers, facing the eye at a distance of 1: a length `l` at
    /// distance `d` covers about `l · pixels_per_unit / d` pixels.
    #[must_use]
    pub const fn pixels_per_unit(&self) -> f64 {
        self.pixels_per_unit
    }

    /// The near plane's distance from the eye.
    #[must_use]
    pub const fn near(&self) -> f64 {
        self.near
    }

    /// The frustum's planes, `[nx, ny, nz, d]` with `n · p + d ≥ 0` inside and `|n| = 1`: left,
    /// right, bottom, top, near, and the far plane at infinity (`[0, 0, 0, 1]`).
    #[must_use]
    pub const fn planes(&self) -> [[f64; 4]; 6] {
        self.planes
    }

    /// `p` in clip coordinates `[x, y, z, w]`: inside the view, `|x|, |y| ≤ w` and `0 < z ≤ w`,
    /// with depth `z / w` 1 on the near plane and towards 0 far away.
    #[must_use]
    pub fn clip(&self, p: Vec3) -> [f64; 4] {
        self.clip_from_world.transform(p)
    }

    /// `p`'s pixel in the viewport, `(x right, y down)` from its top-left corner, and its
    /// depth; `None` behind the eye.
    #[must_use]
    pub fn pixel(&self, point: Vec3) -> Option<(f64, f64, f64)> {
        let [x, y, z, w] = self.clip(point);
        if w <= 0.0 {
            return None;
        }
        let (width, height) = (f64::from(self.size.width()), f64::from(self.size.height()));
        Some((
            (x / w * 0.5 + 0.5) * width,
            (0.5 - y / w * 0.5) * height,
            z / w,
        ))
    }

    /// The view as the GPU takes it: `f32`, laid out as WGSL's `View`.
    #[must_use]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the GPU works in f32: the view is built in f64 and rounded once, here"
    )]
    pub fn gpu(&self) -> GpuView {
        let eye = self.eye;
        // Relative to the eye: p - eye is formed in f32 on the GPU from two nearby f32 points,
        // so a close view loses nothing to a large translation rounded into the matrix.
        let mut relative = self.clip_from_world;
        for row in &mut relative.rows {
            row[3] += row[0] * eye.x + row[1] * eye.y + row[2] * eye.z;
        }
        let planes = self
            .planes
            .map(|[a, b, c, d]| [a, b, c, d + a * eye.x + b * eye.y + c * eye.z].map(|x| x as f32));
        GpuView {
            view_proj: relative.columns_f32(),
            planes,
            eye: [eye.x as f32, eye.y as f32, eye.z as f32, 1.0],
            viewport: [
                f64::from(self.size.width()) as f32,
                f64::from(self.size.height()) as f32,
                self.pixels_per_unit as f32,
                self.near as f32,
            ],
        }
    }
}

/// A [`View`] as the GPU reads it: 192 bytes, WGSL's `View` ([`crate::cull::CULL_WGSL`]).
///
/// Everything is **relative to the eye**: a point `p` goes in as `p - eye`, formed in `f32`
/// from two nearby `f32` points, so a view close to the surface (down to
/// [`Camera::MIN_DISTANCE`]) keeps the precision of the tile data itself rather than losing it
/// to a world-space translation rounded into the matrix. The raster (Task 4) should transform
/// vertices the same way.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct GpuView {
    /// The view-projection from eye-relative positions (`p - eye`) to clip coordinates, by
    /// columns.
    pub view_proj: [[f32; 4]; 4],
    /// The frustum's planes in eye-relative form: `n · (p - eye) + d ≥ 0` inside, in
    /// [`View::planes`]' order. The near plane's normal is the view's forward direction, and
    /// its `d` is minus the near distance.
    pub planes: [[f32; 4]; 6],
    /// The eye, `w` 1.
    pub eye: [f32; 4],
    /// The viewport's width and height in pixels, [`View::pixels_per_unit`], and the near
    /// plane's distance.
    pub viewport: [f32; 4],
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    /// How close two computed values must be to count as equal.
    const CLOSE: f64 = 1e-9;

    /// A 200 × 100 viewport.
    fn size() -> Size {
        Size::new(200, 100).expect("non-zero")
    }

    #[test]
    fn the_target_projects_to_the_viewport_centre_and_the_near_plane_to_depth_one() {
        let camera = Camera::default();
        let view = camera.view(size());
        let (x, y, depth) = view.pixel(camera.target()).expect("in front");
        assert!(
            (x - 100.0).abs() < CLOSE && (y - 50.0).abs() < CLOSE,
            "({x}, {y})"
        );
        assert!(
            (depth - NEAR_FRACTION).abs() < CLOSE,
            "depth is near / distance: {depth}"
        );
        let on_near = camera.eye() + view.forward() * view.near();
        let (_, _, near_depth) = view.pixel(on_near).expect("in front");
        assert!((near_depth - 1.0).abs() < CLOSE, "{near_depth}");
        assert!(
            view.pixel(camera.eye() - view.forward()).is_none(),
            "behind"
        );
    }

    #[test]
    fn the_frustum_planes_hold_exactly_what_projects_inside_the_viewport() {
        let view = View::look_at(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(0.1, -0.2, 0.3),
            Vec3::new(0.0, 1.0, 0.0),
            60_f64.to_radians(),
            size(),
            0.01,
        )
        .expect("a view");
        let mut state = 5_u64;
        let mut random = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            #[expect(clippy::cast_precision_loss, reason = "53 random bits")]
            let unit = (state >> 11) as f64 / (1_u64 << 53) as f64;
            unit * 8.0 - 4.0
        };
        for _ in 0..2000 {
            let point = Vec3::new(random(), random(), random());
            let inside_planes = view
                .planes()
                .iter()
                .all(|[a, b, c, d]| a * point.x + b * point.y + c * point.z + d >= 0.0);
            let [x, y, z, w] = view.clip(point);
            let inside_clip = w > 0.0 && x.abs() <= w && y.abs() <= w && z <= w;
            assert_eq!(inside_planes, inside_clip, "{point:?}");
        }
    }

    #[test]
    fn a_jittered_view_moves_the_image_by_the_jitter_and_nothing_else() {
        let view = View::look_at(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(0.1, -0.2, 0.3),
            Vec3::new(0.0, 1.0, 0.0),
            60_f64.to_radians(),
            size(),
            0.01,
        )
        .expect("a view");
        let jitter = Jitter::new(0.25, -0.375).expect("inside the pixel");
        let moved = view.jittered(jitter);
        for p in [Vec3::new(0.1, -0.2, 0.3), Vec3::new(0.4, 0.1, -0.2)] {
            let (x, y, depth) = view.pixel(p).expect("in front");
            let (mx, my, moved_depth) = moved.pixel(p).expect("in front");
            assert!((mx - (x - 0.25)).abs() < 1e-9, "x: {x} {mx}");
            assert!((my - (y + 0.375)).abs() < 1e-9, "y: {y} {my}");
            assert!((moved_depth - depth).abs() < 1e-12, "the same depth");
        }
        let (gpu, moved_gpu) = (view.gpu(), moved.gpu());
        assert_eq!(
            gpu.planes[4], moved_gpu.planes[4],
            "the near plane, bit for bit"
        );
        assert_eq!(gpu.viewport, moved_gpu.viewport, "lengths in pixels");
        assert_eq!(gpu.eye, moved_gpu.eye);
        assert_ne!(gpu.planes[0], moved_gpu.planes[0], "the sides move");
        assert_eq!(
            view.jittered(Jitter::CENTRE),
            view,
            "the centre changes nothing"
        );
        assert!(Jitter::new(0.5, 0.0).is_none() && Jitter::new(0.0, -0.6).is_none());
    }

    #[test]
    fn pixels_per_unit_turns_a_length_facing_the_eye_into_pixels() {
        let camera = Camera::default();
        let view = camera.view(size());
        let up = camera.axis(Vec3::new(0.0, 1.0, 0.0));
        let a = view.pixel(camera.target()).expect("in front");
        let b = view.pixel(camera.target() + up * 0.1).expect("in front");
        let measured = (a.1 - b.1).abs();
        let predicted = 0.1 * view.pixels_per_unit() / camera.distance();
        assert!(
            (measured - predicted).abs() < 1e-6,
            "{measured} {predicted}"
        );
    }

    #[test]
    fn a_rotation_between_two_vectors_carries_one_to_the_other() {
        let from = Vec3::new(1.0, 0.0, 0.0);
        let to = Vec3::new(0.0, 0.6, 0.8);
        let rotation = Rotation::between(from, to);
        assert!((rotation.apply(from) - to).length() < CLOSE, "carried");
        let half = Rotation::between(from, -from);
        assert!((half.apply(from) + from).length() < CLOSE, "a half turn");
    }

    #[test]
    fn orbiting_moves_the_eye_round_the_target_at_its_distance() {
        let mut camera = Camera::default();
        let quarter = Rotation::about(Vec3::new(0.0, 1.0, 0.0), std::f64::consts::FRAC_PI_2);
        camera.orbit(quarter);
        assert!(
            (camera.eye() - Vec3::new(-3.0, 0.0, 0.0)).length() < CLOSE,
            "{:?}",
            camera.eye()
        );
        camera.zoom(0.5);
        camera.pan(0.0, 0.0, 100.0);
        assert!((camera.distance() - 1.5).abs() < CLOSE, "zoomed");
    }

    #[test]
    fn the_gpu_view_holds_the_eye_relative_matrix_by_columns() {
        let view = Camera::default().view(size());
        let gpu = view.gpu();
        let p = Vec3::new(0.3, -0.2, 0.1);
        let clip = view.clip(p);
        let [c0, c1, c2, c3] = gpu.view_proj;
        let q = p - view.eye();
        #[expect(clippy::cast_possible_truncation, reason = "comparing in f32")]
        let relative = [q.x as f32, q.y as f32, q.z as f32];
        for (row, expected) in clip.iter().enumerate() {
            let got =
                c0[row] * relative[0] + c1[row] * relative[1] + c2[row] * relative[2] + c3[row];
            assert!((f64::from(got) - expected).abs() < 1e-5, "row {row}");
        }
        let near = gpu.planes[4];
        let forward = view.forward();
        #[expect(clippy::cast_possible_truncation, reason = "comparing in f32")]
        let forward = [forward.x as f32, forward.y as f32, forward.z as f32];
        assert!(
            (0..3).all(|k| (near[k] - forward[k]).abs() < 1e-6),
            "the near plane faces forward"
        );
        #[expect(clippy::cast_possible_truncation, reason = "comparing in f32")]
        let near_distance = view.near() as f32;
        assert!(
            (near[3] + near_distance).abs() < 1e-6,
            "at the near distance"
        );
    }
}
