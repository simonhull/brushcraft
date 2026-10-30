//! Tests of the culling passes: the bounds pass and the depth pyramid against their CPU twins.
//! The culling tests themselves run inside the level choice (`lod/tests.rs`).

use geom::{FaceId, Level, TileKey, cube_sphere};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{BufferUsages, CommandEncoderDescriptor, TextureViewDescriptor};

use super::reference::{
    CpuPyramid, bounds_of, depth_texture, distance, dot, extra_triangles, face_normal,
    morph_normal, morph_triangles, quad_triangles, read_mip,
};
use super::*;
use crate::gpu::GpuTest;
use crate::residency::Slot;
use crate::tree::{NODE_FINEST, Place, TileTree};
use crate::upload::{GpuVertex, TileDetail};

mod morph;
mod truth;

/// A small deterministic generator (splitmix64).
pub(crate) struct Rng(pub(crate) u64);

impl Rng {
    /// The next 64 random bits.
    pub(crate) fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform float in `[lo, hi)`.
    pub(crate) fn range(&mut self, lo: f32, hi: f32) -> f32 {
        let unit = f32::from(u16::try_from(self.next() >> 48).expect("16 bits")) / 65536.0;
        lo + (hi - lo) * unit
    }

    /// A random index below `n`.
    pub(crate) fn below(&mut self, n: usize) -> usize {
        usize::try_from(self.next() % n as u64).expect("below n")
    }
}

/// A bumpy patch of a sphere of `side` × `side` vertices around a random direction: a tile's
/// worth of data with an interesting cone.
fn patch(rng: &mut Rng, side: usize) -> Vec<GpuVertex> {
    let (u0, v0) = (rng.range(-1.0, 1.0), rng.range(-1.0, 1.0));
    let span = rng.range(0.05, 1.5);
    let bump = rng.range(0.0, 0.05);
    let small = |n: usize| f32::from(u16::try_from(n).expect("a small grid"));
    let steps = small((side - 1).max(1));
    let mut vertices = Vec::with_capacity(side * side);
    for j in 0..side {
        for i in 0..side {
            let u = u0 + span * (small(i) / steps - 0.5);
            let v = v0 + span * (small(j) / steps - 0.5);
            let p = [u, v, 1.0];
            let length = dot(p, p).sqrt();
            let r = 1.0 + bump * rng.range(-1.0, 1.0);
            vertices.push(GpuVertex {
                position: p.map(|x| x / length * r),
                roughness: 1.0,
                normal: [0.0, 0.0, 1.0],
                reserved: 0,
            });
        }
    }
    vertices
}

#[test]
fn the_bounds_pass_matches_its_cpu_twin_and_holds_every_vertex_and_triangle() {
    GpuTest::run(|gpu| {
        let mesh = cube_sphere(1).expect("a mesh");
        let mut tree = TileTree::new(gpu, &mesh, 64).expect("a tree");
        let mut rng = Rng(3);
        // Tiles of classes 0, 3 and 6, in two buffers of class 6 to check the grouping.
        let mut classes: Vec<Vec<Vec<GpuVertex>>> = vec![Vec::new(); CLASSES];
        let mut expected = Vec::new();
        for (n, level) in [0_u8, 3, 7, 7, 8, 8, 8].into_iter().enumerate() {
            let level = Level::new(level).expect("a level");
            let class = SizeClass::of(level);
            let side = usize::try_from(class.side()).expect("small");
            let vertices = patch(&mut rng, side);
            let per_class = &mut classes[usize::from(class.get())];
            let index = per_class.len();
            per_class.push(vertices.clone());
            let place = Place::of(
                &TileKey::display(FaceId::new(u32::try_from(n).expect("small")), level, 0, 0)
                    .expect("a key"),
            );
            let node = tree
                .insert(
                    place,
                    Slot::for_tests(class, index),
                    TileDetail::FINEST,
                    NODE_FINEST,
                )
                .expect("inserted");
            expected.push((node, bounds_of(&vertices, side), vertices, side));
        }
        let buffers: Vec<Vec<Buffer>> = classes
            .iter()
            .map(|tiles| {
                let data: Vec<GpuVertex> = tiles.iter().flatten().copied().collect();
                if data.is_empty() {
                    return Vec::new();
                }
                vec![gpu.device().create_buffer_init(&BufferInitDescriptor {
                    label: Some("slots"),
                    contents: bytemuck::cast_slice(&data),
                    usage: BufferUsages::STORAGE,
                })]
            })
            .collect();
        let slots = SlotBuffers::new(buffers);
        for (start, nodes) in tree.take_dirty() {
            gpu.queue().write_buffer(
                tree.buffer(),
                u64::from(start) * crate::tree::NODE_BYTES,
                bytemuck::cast_slice(&nodes),
            );
        }
        let mut pass = BoundsPass::new(gpu, 64).expect("the bounds pass");
        let update = BoundsUpdate::since(&tree, 0);
        assert_eq!(update.len(), expected.len(), "every tile changed");
        let mut encoder = gpu
            .device()
            .create_command_encoder(&CommandEncoderDescriptor::default());
        pass.encode(gpu, &mut encoder, tree.buffer(), &update, &slots, None)
            .expect("encoded");
        gpu.queue().submit([encoder.finish()]);
        let got: Vec<GpuBounds> = gpu.read_buffer(pass.buffer()).expect("read back");
        for (node, cpu, vertices, side) in &expected {
            let gpu_bounds = got[usize::try_from(*node).expect("small")];
            let close = |a: f32, b: f32, tolerance: f32| (a - b).abs() <= tolerance;
            assert!(
                (0..3).all(|k| close(gpu_bounds.center[k], cpu.center[k], 1e-6)),
                "centre: {gpu_bounds:?} {cpu:?}"
            );
            assert!(close(gpu_bounds.radius, cpu.radius, 1e-5), "radius");
            assert!(
                close(gpu_bounds.quad_edge, cpu.quad_edge, 1e-5),
                "quad edge"
            );
            assert!(close(gpu_bounds.cone_cos, cpu.cone_cos, 1e-4), "cone");
            assert_holds(&gpu_bounds, vertices, *side);
        }
        // Nothing changed since: an update after this one bounds nothing.
        assert!(BoundsUpdate::since(&tree, update.generation()).is_empty());
    });
}

/// The bounds are conservative for the tile of `vertices`, `side` a row.
///
/// Every vertex is in the sphere and every triangle's normal in the cone: the tile's own and
/// its extra ones (the parent triangulation's and every edge variant's), each extra one facing
/// out as the tile does.
fn assert_holds(gpu_bounds: &GpuBounds, vertices: &[GpuVertex], side: usize) {
    // Conservative: every vertex in the sphere, every triangle's normal in the cone.
    let points: Vec<[f32; 3]> = vertices.iter().map(|v| v.position).collect();
    assert!(
        points
            .iter()
            .all(|&p| distance(p, gpu_bounds.center) <= gpu_bounds.radius),
        "the sphere holds every vertex"
    );
    let quads = side - 1;
    let triangles =
        (0..quads * quads).flat_map(|q| quad_triangles(&points, side, q % quads, q / quads));
    for [a, b, c] in triangles {
        let n = face_normal(a, b, c);
        assert!(
            dot(n, n) == 0.0 || dot(n, gpu_bounds.axis) >= gpu_bounds.cone_cos,
            "the cone holds every triangle"
        );
    }
    // And the parent triangulation's and every edge variant's triangles, each facing
    // out of the sphere as the tile does.
    let extras = extra_triangles(side);
    let half = quads / 2;
    assert_eq!(
        extras.len(),
        2 * half * half + 12 * half,
        "every extra triangle"
    );
    let at = |(x, y): (i32, i32)| {
        points[usize::try_from(y).expect("on the grid") * side
            + usize::try_from(x).expect("on the grid")]
    };
    for [a, b, c] in extras {
        let n = face_normal(at(a), at(b), at(c));
        assert!(
            dot(n, n) == 0.0 || dot(n, gpu_bounds.axis) >= gpu_bounds.cone_cos,
            "the cone holds the extra triangle {a:?} {b:?} {c:?}"
        );
        assert!(dot(n, at(a)) > 0.0, "{a:?} {b:?} {c:?} faces outward");
    }
    // And each morph triangle: an own or fan triangle with one odd vertex at its start.
    let morphs = morph_triangles(side);
    let expected = if quads < 2 {
        0
    } else {
        12 * quads * quads + 48 * half
    };
    assert_eq!(morphs.len(), expected, "every morph triangle");
    for t in morphs {
        let n = morph_normal(&points, side, t);
        assert!(
            dot(n, n) == 0.0 || dot(n, gpu_bounds.axis) >= gpu_bounds.cone_cos,
            "the cone holds the morph triangle {t:?}"
        );
    }
}

#[test]
fn a_bounds_pass_that_fails_leaves_its_tiles_unknown_not_stale() {
    GpuTest::run(|gpu| {
        let mesh = cube_sphere(1).expect("a mesh");
        let mut tree = TileTree::new(gpu, &mesh, 8).expect("a tree");
        let place = Place::of(&TileKey::display(FaceId::new(0), Level::ZERO, 0, 0).expect("a key"));
        let class = SizeClass::of(Level::ZERO);
        // A display tile, its e and cone unused here (the bounds pass reads neither): e 0 and
        // the cone that covers everything.
        let unused = TileDetail {
            error: 0.0,
            cone: geom::FinestNormalCone::EVERYTHING,
        };
        let node = tree
            .insert(place, Slot::for_tests(class, 0), unused, 0)
            .expect("inserted");
        let mut pass = BoundsPass::new(gpu, 8).expect("the bounds pass");
        let stale = GpuBounds {
            radius: 1.0,
            cone_cos: 0.5,
            ..GpuBounds::UNKNOWN
        };
        pass.write(gpu, node, &[stale]);
        let mut encoder = gpu
            .device()
            .create_command_encoder(&CommandEncoderDescriptor::default());
        // No slot buffers at all: the tile's can't be found.
        let failed = pass.encode(
            gpu,
            &mut encoder,
            tree.buffer(),
            &BoundsUpdate::since(&tree, 0),
            &SlotBuffers::default(),
            None,
        );
        assert!(
            matches!(failed, Err(CullError::MissingSlotBuffer { .. })),
            "{failed:?}"
        );
        gpu.queue().submit([encoder.finish()]);
        let got: Vec<GpuBounds> = gpu.read_buffer(pass.buffer()).expect("read back");
        assert_eq!(
            got[usize::try_from(node).expect("small")],
            GpuBounds::UNKNOWN,
            "reset"
        );
    });
}

#[test]
fn the_depth_pyramid_holds_the_farthest_depth_of_what_it_covers_at_every_mip() {
    GpuTest::run(|gpu| {
        let pass = PyramidPass::new(gpu).expect("the pyramid's passes");
        // Odd and even sizes, so the clamped edges are covered.
        for (width, height) in [(37_u32, 23_u32), (64, 16), (1, 5)] {
            let mut rng = Rng(u64::from(width * height));
            let depth: Vec<f32> = (0..width * height).map(|_| rng.range(0.0, 1.0)).collect();
            let depth = &depth;
            let texture = depth_texture(gpu, width, height, depth);
            let view = texture.create_view(&TextureViewDescriptor::default());
            let size = Size::new(width, height).expect("non-zero");
            let pyramid = DepthPyramid::new(gpu, size).expect("a pyramid");
            let mut encoder = gpu
                .device()
                .create_command_encoder(&CommandEncoderDescriptor::default());
            pass.build(gpu, &mut encoder, &view, &pyramid, None)
                .expect("encoded");
            gpu.queue().submit([encoder.finish()]);
            let cpu = CpuPyramid::new(width, height, depth);
            assert_eq!(
                usize::try_from(pyramid.levels()).expect("small"),
                cpu.mips.len(),
                "{width} × {height}: mips"
            );
            for (level, (w, h, texels)) in cpu.mips.iter().enumerate() {
                let level = u32::try_from(level).expect("small");
                assert_eq!(
                    read_mip(gpu, pyramid.texture(), level, *w, *h),
                    *texels,
                    "{width} × {height}, mip {level}"
                );
            }
            // And each texel of mip 0 is the minimum of the depth texels it covers.
            let (w0, _, mip0) = &cpu.mips[0];
            for (i, value) in mip0.iter().enumerate() {
                let (x, y) = (
                    u32::try_from(i).expect("small") % w0,
                    u32::try_from(i).expect("small") / w0,
                );
                // Past the image's edge, its last row or column.
                let covered = (2 * y..2 * y + 2).flat_map(|py| {
                    (2 * x..2 * x + 2).map(move |px| {
                        let at = py.min(height - 1) * width + px.min(width - 1);
                        depth[usize::try_from(at).expect("small")]
                    })
                });
                let farthest = covered.fold(f32::MAX, f32::min);
                assert_eq!(*value, farthest, "texel ({x}, {y})");
            }
        }
    });
}
