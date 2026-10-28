//! Ground truth for the culling tests.
//!
//! Each claim is checked by brute force against the geometry it is about (projected segments,
//! sampled normals and points, the depth image's pixels), not against a twin doing the same
//! arithmetic. The GPU agrees with the twins (the level choice's culling test), so these hold
//! for it too.

use super::Rng;
use crate::camera::{Vec3, View};
use crate::cull::GpuBounds;
use crate::cull::reference::{CpuPyramid, backfacing, occluded, projected};
use crate::surface::Size;

/// A view from the origin down −z, 45° vertically, into a viewport of `width` × `height`.
fn view(width: u32, height: u32) -> View {
    View::look_at(
        Vec3::default(),
        Vec3::new(0.0, 0.0, -1.0),
        Vec3::new(0.0, 1.0, 0.0),
        45_f64.to_radians(),
        Size::new(width, height).expect("non-zero"),
        0.01,
    )
    .expect("a view")
}

/// `v` in `f32`, for bounds.
#[expect(
    clippy::cast_possible_truncation,
    reason = "bounds go to the GPU as f32"
)]
const fn narrow(v: Vec3) -> [f32; 3] {
    [v.x as f32, v.y as f32, v.z as f32]
}

/// Bounds of a sphere at `center` of `radius`, no cone.
const fn sphere(center: [f32; 3], radius: f32) -> GpuBounds {
    GpuBounds {
        center,
        radius,
        axis: [0.0; 3],
        cone_cos: -1.0,
        quad_edge: 0.0,
        reserved: [0; 3],
    }
}

/// A random unit vector.
fn direction(rng: &mut Rng) -> Vec3 {
    loop {
        let v = Vec3::new(
            f64::from(rng.range(-1.0, 1.0)),
            f64::from(rng.range(-1.0, 1.0)),
            f64::from(rng.range(-1.0, 1.0)),
        );
        if v.length() <= 1.0
            && let Some(unit) = v.normalized()
        {
            return unit;
        }
    }
}

#[test]
fn projected_bounds_the_pixels_a_short_segment_covers_even_in_the_image_corners() {
    let view = view(1920, 1080);
    let gpu = view.gpu();
    let tan_y = 22.5_f64.to_radians().tan();
    let tan_x = tan_y * 1920.0 / 1080.0;
    let mut rng = Rng(41);
    let mut worst_naive = 0.0_f64;
    for sample in 0..400 {
        let z = f64::from(rng.range(0.5, 20.0));
        // Half the samples in a corner, the rest anywhere in the view.
        let (fx, fy) = if sample % 2 == 0 {
            (0.97, -0.97)
        } else {
            (
                f64::from(rng.range(-1.0, 1.0)),
                f64::from(rng.range(-1.0, 1.0)),
            )
        };
        let center =
            view.forward() * z + view.right() * (fx * tan_x * z) + view.up() * (fy * tan_y * z);
        let length = z * 1e-3;
        let measured = (0..64)
            .map(|_| {
                let half = direction(&mut rng) * (length / 2.0);
                let a = view.pixel(center - half).expect("in front");
                let b = view.pixel(center + half).expect("in front");
                (a.0 - b.0).hypot(a.1 - b.1)
            })
            .fold(0.0, f64::max);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "bounds go to the GPU as f32"
        )]
        let (radius, extent) = ((length / 2.0) as f32, length as f32);
        let bound = f64::from(projected(
            &gpu,
            &sphere(narrow(center), radius),
            0.0,
            extent,
        ));
        assert!(
            bound >= measured * (1.0 - 1e-4),
            "sample {sample}: {measured} px measured, {bound} px bounded"
        );
        let naive = length * view.pixels_per_unit() / center.length();
        worst_naive = worst_naive.max(measured / naive);
    }
    // The corners are where `length · f / distance` falls short: the test can tell.
    assert!(
        worst_naive > 1.4,
        "the naive bound is short by {worst_naive}× at worst"
    );
}

#[test]
fn a_tile_found_backfacing_faces_away_at_every_sampled_normal_and_point() {
    let mut rng = Rng(43);
    let mut culled = 0;
    for _ in 0..4000 {
        let axis = direction(&mut rng);
        let b = GpuBounds {
            center: [
                rng.range(-2.0, 2.0),
                rng.range(-2.0, 2.0),
                rng.range(-2.0, 2.0),
            ],
            radius: rng.range(0.01, 0.5),
            axis: narrow(axis),
            cone_cos: rng.range(0.05, 0.99),
            quad_edge: 0.0,
            reserved: [0; 3],
        };
        let eye = narrow(direction(&mut rng) * f64::from(rng.range(1.0, 6.0)));
        if !backfacing(&b, eye, 0.0) {
            continue;
        }
        culled += 1;
        let half_angle = f64::from(b.cone_cos).acos();
        let center = Vec3::new(
            f64::from(b.center[0]),
            f64::from(b.center[1]),
            f64::from(b.center[2]),
        );
        let eye = Vec3::new(f64::from(eye[0]), f64::from(eye[1]), f64::from(eye[2]));
        for _ in 0..200 {
            // A normal within the cone: the axis tilted towards a random side by up to α.
            let side = direction(&mut rng);
            let tangent = (side - axis * side.dot(axis)).normalized().unwrap_or(side);
            let tilt = half_angle * f64::from(rng.range(0.0, 1.0));
            let normal = axis * tilt.cos() + tangent * tilt.sin();
            let point = center
                + direction(&mut rng) * (f64::from(b.radius) * f64::from(rng.range(0.0, 1.0)));
            assert!(
                normal.dot(point - eye) > 0.0,
                "{b:?} from {eye:?}: a triangle faces the eye"
            );
        }
    }
    assert!(
        culled > 300,
        "only {culled} tiles faced away: the test means little"
    );
}

#[test]
fn a_sphere_found_occluded_has_nearer_depth_at_every_pixel_it_covers() {
    let (width, height) = (96_u32, 64_u32);
    let view = view(width, height);
    let gpu = view.gpu();
    let mut rng = Rng(47);
    let mut hidden = 0;
    for _ in 0..40 {
        // Blocks of random depth, between distances 1 and 8 (reversed depth near / distance).
        let block = 8;
        let blocks: Vec<f32> = (0..(width / block) * (height / block))
            .map(|_| {
                if rng.below(3) == 0 {
                    0.0
                } else {
                    0.01 / rng.range(1.0, 8.0)
                }
            })
            .collect();
        let depth: Vec<f32> = (0..width * height)
            .map(|i| {
                let (x, y) = (i % width / block, i / width / block);
                blocks[usize::try_from(y * (width / block) + x).expect("small")]
            })
            .collect();
        let pyramid = CpuPyramid::new(width, height, &depth);
        for _ in 0..200 {
            let center = view.forward() * f64::from(rng.range(1.0, 10.0))
                + view.right() * f64::from(rng.range(-3.0, 3.0))
                + view.up() * f64::from(rng.range(-2.0, 2.0));
            let radius = f64::from(rng.range(0.02, 0.6));
            #[expect(
                clippy::cast_possible_truncation,
                reason = "bounds go to the GPU as f32"
            )]
            let radius_f32 = radius as f32;
            if !occluded(&gpu, &pyramid, narrow(center), radius_f32) {
                continue;
            }
            hidden += 1;
            for _ in 0..400 {
                let point = center + direction(&mut rng) * radius;
                let Some((x, y, point_depth)) = view.pixel(point) else {
                    panic!("an occluded sphere lies in front of the eye");
                };
                if x < 0.0 || y < 0.0 || x >= f64::from(width) || y >= f64::from(height) {
                    continue;
                }
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "a pixel inside the image"
                )]
                let at = (y as u32) * width + (x as u32);
                let image = f64::from(depth[usize::try_from(at).expect("small")]);
                assert!(
                    image > point_depth,
                    "a point of an occluded sphere shows at ({x}, {y})"
                );
            }
        }
    }
    assert!(
        hidden > 300,
        "only {hidden} spheres hidden: the test means little"
    );
}
