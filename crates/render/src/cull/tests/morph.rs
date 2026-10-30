//! The normal cone holds the triangles the raster draws at either end of the morph.
//!
//! In every edge variant, and not those on the way: the level choice culls by it only a tile
//! whose morph is settled (`lod.rs`, "Cone culling and the morph"; `raster/tests/cone.rs`).

use super::*;
use crate::cull::reference::backfacing;
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

/// Where the raster draws a tile's odd vertices: all where they are, all where their morph
/// starts, or each at a random point on its way.
#[derive(Debug, Clone, Copy)]
enum Morph {
    /// Every `t` 1.
    End,
    /// Every `t` 0.
    Start,
    /// Every `t` drawn at random, one per odd vertex.
    OnTheWay,
}

/// Whether a triangle is a sliver its tile's culling would lose.
///
/// That is, whether the triangle of unit normal `normal` with a corner at `corner`, its tile's
/// bounds `bounds`, faces an eye from which the cone test (`backfacing`, margin 0, as emit
/// culls) says the whole tile faces away. The eye is built: far away, so the test's allowance for
/// the sphere (`r / d`) is small, seen from the side across the axis opposite the normal's lean, at
/// the angle from the axis halfway between where the normal turns to the eye (`π/2 − β`, `β` the
/// normal's angle from the axis) and where the cone test stops saying the tile faces away (`acos(r
/// / d) − α`). A normal inside the cone (`β ≤ α`) has no such angle.
fn sliver(bounds: &GpuBounds, normal: [f32; 3], corner: [f32; 3]) -> bool {
    let axis = bounds.axis;
    let lean = dot(normal, axis);
    let across: [f32; 3] = [0, 1, 2].map(|k| lean * axis[k] - normal[k]);
    let length = dot(across, across).sqrt();
    if length == 0.0 {
        return false;
    }
    let side = across.map(|x| x / length);
    let distance = 1000.0 * bounds.radius;
    let beta = lean.clamp(-1.0, 1.0).acos();
    let alpha = bounds.cone_cos.acos();
    let theta =
        0.5 * (std::f32::consts::FRAC_PI_2 - beta + (bounds.radius / distance).acos() - alpha);
    // From the eye to the centre: θ from the axis, towards `side`.
    let towards: [f32; 3] = [0, 1, 2].map(|k| theta.cos() * axis[k] + theta.sin() * side[k]);
    let eye: [f32; 3] = [0, 1, 2].map(|k| bounds.center[k] - distance * towards[k]);
    let to_eye: [f32; 3] = [0, 1, 2].map(|k| eye[k] - corner[k]);
    backfacing(bounds, eye, 0.0) && dot(normal, to_eye) > 0.0
}

/// Triangles of `variant` on the tile of `points` (a `side` grid) facing outside `bounds`'
/// cone, and of those how many are slivers ([`sliver`]), over `draws` morph states each, every
/// odd vertex placed by `morph`.
fn outside_the_cone(
    rng: &mut Rng,
    points: &[[f32; 3]],
    side: usize,
    variant: Variant,
    bounds: &GpuBounds,
    (morph, draws): (Morph, usize),
) -> (usize, usize) {
    let cluster = ClusterSize::of_quads(u32::try_from(side - 1).expect("small")).expect("a size");
    let at = |u: u8, v: u8| points[usize::from(v) * side + usize::from(u)];
    let (mut outside, mut slivers) = (0, 0);
    for _ in 0..draws {
        // One t per odd vertex, shared by every triangle holding it, as the raster draws.
        let morphs: Vec<f32> = (0..side * side)
            .map(|_| match morph {
                Morph::End => 1.0,
                Morph::Start => 0.0,
                Morph::OnTheWay => rng.range(0.0, 1.0),
            })
            .collect();
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
            // place.wgsl's mix(mid, p, t), mid as (a + b) · 0.5.
            [0, 1, 2].map(|axis| {
                #[expect(
                    clippy::manual_midpoint,
                    reason = "place.wgsl's arithmetic, (a + b) · 0.5, bit for bit"
                )]
                let mid = (low[axis] + high[axis]) * 0.5;
                mid * (1.0 - factor) + position[axis] * factor
            })
        };
        let triangles = drawn(cluster, variant).flatten();
        for triangle in triangles {
            let [first, second, third] = triangle.map(placed);
            let normal = face_normal(first, second, third);
            if dot(normal, normal) > 0.0 && dot(normal, bounds.axis) < bounds.cone_cos {
                outside += 1;
                slivers += usize::from(sliver(bounds, normal, first));
            }
        }
    }
    (outside, slivers)
}

#[test]
fn the_drawn_cone_holds_every_drawn_triangle_at_either_end_of_the_morph_not_on_the_way() {
    GpuTest::run(|gpu| {
        let mut rng = Rng(23);
        let tiles = tiles_with_bounds(gpu, &mut rng);
        let (mut at_the_ends, mut on_the_way, mut slivers) = (0, 0, 0);
        for (vertices, side, bounds) in &tiles {
            let points: Vec<[f32; 3]> = vertices.iter().map(|v| v.position).collect();
            for variant in Variant::all() {
                for ends in [Morph::End, Morph::Start] {
                    let (outside, _) =
                        outside_the_cone(&mut rng, &points, *side, variant, bounds, (ends, 1));
                    at_the_ends += outside;
                }
                let way = (Morph::OnTheWay, 8);
                let (outside, lost) =
                    outside_the_cone(&mut rng, &points, *side, variant, bounds, way);
                on_the_way += outside;
                slivers += lost;
            }
        }
        println!(
            "{at_the_ends} triangles outside the cone at the ends, {on_the_way} on the way, \
             {slivers} of them slivers the cone test would cull"
        );
        assert_eq!(at_the_ends, 0, "the cone holds both ends of the morph");
        // Why emit culls only tiles whose morph is settled: on the way, triangles leave the
        // cone, and the cone test culls their tile from eyes they face (the bite of
        // raster/tests/cone.rs's silhouette test, which its sphere's wide cones don't reach).
        assert!(on_the_way > 0, "the cone doesn't hold the morph on its way");
        assert!(slivers > 0, "culling a tile mid-morph by it loses slivers");
    });
}
