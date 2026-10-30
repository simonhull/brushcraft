//! The normal cone holds every morph state of every triangle the raster draws: each of a
//! triangle's two odd vertices anywhere on its way, independently, in every edge variant.

use super::*;
use crate::cull::reference::bounds_with;
use crate::indices::{ClusterSize, Variant, drawn};

/// Bumpy tiles of 2, 4 and 8 quads a side (one cluster each, so every edge variant applies to
/// the tile's own edges), with their bounds from the bounds pass.
fn tiles_with_bounds(gpu: &Gpu, rng: &mut Rng) -> Vec<(Vec<GpuVertex>, usize, GpuBounds)> {
    let mesh = cube_sphere(1).expect("a mesh");
    let mut tree = TileTree::new(gpu, &mesh, 64).expect("a tree");
    let mut classes: Vec<Vec<GpuVertex>> = vec![Vec::new(); CLASSES];
    let mut tiles = Vec::new();
    for n in 0..18_u32 {
        let level = Level::new(u8::try_from(1 + n % 3).expect("small")).expect("a level");
        let class = SizeClass::of(level);
        let side = usize::try_from(class.side()).expect("small");
        let vertices = patch(rng, side);
        let per_class = &mut classes[usize::from(class.get())];
        let index = per_class.len() / (side * side);
        per_class.extend_from_slice(&vertices);
        let face = FaceId::new(n / 3);
        let place = Place::of(&TileKey::display(face, level, 0, 0).expect("a key"));
        let node = tree
            .insert(
                place,
                Slot::for_tests(class, index),
                TileDetail::FINEST,
                NODE_FINEST,
            )
            .expect("inserted");
        tiles.push((vertices, side, node));
    }
    let buffers = classes.iter().map(|data| {
        if data.is_empty() {
            return Vec::new();
        }
        vec![gpu.device().create_buffer_init(&BufferInitDescriptor {
            label: Some("slots"),
            contents: bytemuck::cast_slice(data),
            usage: BufferUsages::STORAGE,
        })]
    });
    let slots = SlotBuffers::new(buffers.collect());
    for (start, nodes) in tree.take_dirty() {
        gpu.queue().write_buffer(
            tree.buffer(),
            u64::from(start) * crate::tree::NODE_BYTES,
            bytemuck::cast_slice(&nodes),
        );
    }
    let mut pass = BoundsPass::new(gpu, 64).expect("the bounds pass");
    let update = BoundsUpdate::since(&tree, 0);
    let mut encoder = gpu
        .device()
        .create_command_encoder(&CommandEncoderDescriptor::default());
    pass.encode(gpu, &mut encoder, tree.buffer(), &update, &slots, None)
        .expect("encoded");
    gpu.queue().submit([encoder.finish()]);
    let all: Vec<GpuBounds> = gpu.read_buffer(pass.buffer()).expect("read back");
    tiles
        .into_iter()
        .map(|(vertices, side, node)| {
            let b = all[usize::try_from(node).expect("small")];
            (vertices, side, b)
        })
        .collect()
}

/// Triangles of `variant` on the tile of `points` (a `side` grid) facing outside `bounds`'
/// cone, over `draws` random morph states each (every odd vertex at its own random `t`).
fn outside_the_cone(
    rng: &mut Rng,
    points: &[[f32; 3]],
    side: usize,
    variant: Variant,
    bounds: &GpuBounds,
    draws: usize,
) -> usize {
    let cluster = ClusterSize::of_quads(u32::try_from(side - 1).expect("small")).expect("a size");
    let at = |u: u8, v: u8| points[usize::from(v) * side + usize::from(u)];
    let mut outside = 0;
    for _ in 0..draws {
        // One t per odd vertex, shared by every triangle holding it, as the raster draws.
        let morphs: Vec<f32> = (0..side * side).map(|_| rng.range(0.0, 1.0)).collect();
        let placed = |corner: crate::indices::GridVertex| -> [f32; 3] {
            let (column, line) = (corner.u(), corner.v());
            let (across, up) = (column % 2, line % 2);
            let position = at(column, line);
            if across == 0 && up == 0 {
                return position;
            }
            let low = at(column - across, line - up);
            let high = at(column + across, line + up);
            let factor = morphs[usize::from(line) * side + usize::from(column)];
            [0, 1, 2].map(|axis| {
                let mid = f32::midpoint(low[axis], high[axis]);
                mid + factor * (position[axis] - mid)
            })
        };
        let triangles = drawn(cluster, variant).flatten();
        for triangle in triangles {
            let [first, second, third] = triangle.map(placed);
            let normal = face_normal(first, second, third);
            if dot(normal, normal) > 0.0 && dot(normal, bounds.axis) < bounds.cone_cos {
                outside += 1;
            }
        }
    }
    outside
}

#[test]
fn every_morph_state_of_every_drawn_triangle_lies_in_the_tile_s_cone() {
    GpuTest::run(|gpu| {
        let mut rng = Rng(23);
        let tiles = tiles_with_bounds(gpu, &mut rng);
        let (mut outside, mut outside_without) = (0, 0);
        for (vertices, side, bounds) in &tiles {
            let points: Vec<[f32; 3]> = vertices.iter().map(|v| v.position).collect();
            // The first cut's cone, without the morph triangles: the bite.
            let without = bounds_with(vertices, *side, false);
            for variant in Variant::all() {
                outside += outside_the_cone(&mut rng, &points, *side, variant, bounds, 8);
                outside_without += outside_the_cone(&mut rng, &points, *side, variant, &without, 8);
            }
        }
        println!("{outside} triangles outside the cone, {outside_without} without the morph's");
        assert_eq!(outside, 0, "the cone holds every morph state");
        assert!(
            outside_without > 0,
            "without the morph triangles it doesn't"
        );
    });
}
