//! Tests of the raster on lavapipe, over a synthetic tile tree of a sphere, every tile resident.
//!
//! They cover the pinhole test (random cameras and random mixed-level cuts against a render of
//! the finest level, and its bite), what the visibility buffer's ids decode to, their
//! stability, the geomorph's endpoints, and the level choice's clusters drawn.
//!
//! The sphere is `cube_sphere(1)` (six faces) to level 7: whole-face display tiles at levels 0
//! to 6, four finest tiles a face at level 7. Every vertex is computed from its place on the
//! face grid, a face border vertex from its copy on the face of lowest index, so every tile
//! holding a vertex stores the same bits for it, as the display layer and the store's owner
//! copies guarantee. A display tile's `e` is measured: the largest distance from a finest
//! vertex under it to the point of the tile's own triangles at the same place.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use geom::{BaseMesh, FaceId, FinestNormalCone, Level, TileKey, cube_sphere};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BufferDescriptor, CommandEncoderDescriptor, ComputePassDescriptor,
    ComputePipelineDescriptor, Origin3d, PollType, TexelCopyBufferInfo, TexelCopyBufferLayout,
    TexelCopyTextureInfo, TextureAspect,
};

use super::*;
use crate::camera::Vec3;
use crate::cull::BoundsUpdate;
use crate::cull::tests::Rng;
use crate::gpu::GpuTest;
use crate::indices::{ClusterSize, GridVertex, Variant, remap};
use crate::lod::{GpuCluster, GpuCutEntry, Lod, LodConfig, MainInputs, Thresholds};
use crate::residency::{CLASSES, REQUEST_HEADER, SizeClass, Slot, SlotLayout};
use crate::surface::Size;
use crate::tree::{NODE_BYTES, NODE_FINEST, Place, TileTree};
use crate::upload::{GpuVertex, TileDetail};

mod cone;
mod pop;
mod seams;

/// The finest level.
const FINEST: u8 = 7;

/// Level `l`.
fn level(l: u8) -> Level {
    Level::new(l).expect("a level")
}

/// The test viewport.
fn size() -> Size {
    Size::new(400, 300).expect("non-zero")
}

/// `n` as an `f32`, exactly (the grids here are small).
fn small(n: u32) -> f32 {
    f32::from(u16::try_from(n).expect("a small grid"))
}

/// A face's point at `(u, v)` of its bilinear patch, pushed onto the unit sphere.
fn sphere_point(mesh: &BaseMesh, face: FaceId, u: f32, v: f32) -> [f32; 3] {
    let corner = |c: u8| {
        let p = mesh.corner_position(face, c).expect("a corner");
        [p.x, p.y, p.z]
    };
    let [c0, c1, c2, c3] = [corner(0), corner(1), corner(2), corner(3)];
    let p: [f32; 3] = [0, 1, 2].map(|a| {
        (1.0 - u) * (1.0 - v) * c0[a]
            + u * (1.0 - v) * c1[a]
            + u * v * c2[a]
            + (1.0 - u) * v * c3[a]
    });
    let length = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
    p.map(|x| x / length)
}

/// The finest level's vertex `(x, y)` of `face`'s grid: from its copy on the face of lowest
/// index, so every face holding it agrees.
fn finest_point(mesh: &BaseMesh, finest: Level, face: FaceId, x: u32, y: u32) -> [f32; 3] {
    let n = finest.face_quads();
    let tile = finest.tile_quads();
    let (face, x, y) = if x == 0 || y == 0 || x == n || y == n {
        let copies = mesh.topology().copies(face, finest, x, y);
        let owners = copies.iter().map(|c| {
            let at = (u32::from(c.tx) * tile + c.i, u32::from(c.ty) * tile + c.j);
            (c.face.index(), at)
        });
        let (owner, at) = owners.min().expect("a vertex has a copy");
        (FaceId::new(owner), at.0, at.1)
    } else {
        (face, x, y)
    };
    sphere_point(mesh, face, small(x) / small(n), small(y) / small(n))
}

/// One resident tile.
#[derive(Debug)]
struct Tile {
    /// Its node.
    node: u32,
    /// Its slot.
    slot: Slot,
    /// Its vertices, row-major.
    vertices: Vec<GpuVertex>,
    /// Its `e` and cone.
    detail: TileDetail,
}

/// The synthetic tree: every tile of the sphere to its finest level, resident, with its data in
/// slot buffers (one per class).
struct World {
    /// The mesh.
    mesh: BaseMesh,
    /// The finest level.
    finest: Level,
    /// The tree.
    tree: TileTree,
    /// Every tile.
    tiles: BTreeMap<Place, Tile>,
    /// The slot buffers.
    slots: SlotBuffers,
}

impl World {
    /// Every tile resident to [`FINEST`], its data uploaded and its node written.
    fn new(gpu: &Gpu) -> Self {
        Self::with_finest(gpu, FINEST)
    }

    /// Every tile resident to level `top`, its data uploaded and its node written.
    fn with_finest(gpu: &Gpu, top: u8) -> Self {
        let mesh = cube_sphere(1).expect("a mesh");
        let finest = level(top);
        let n = finest.face_quads();
        let grids: BTreeMap<u32, Vec<[f32; 3]>> = mesh
            .faces()
            .map(|face| {
                let grid = (0..=n).flat_map(|y| (0..=n).map(move |x| (x, y)));
                let points = grid.map(|(x, y)| finest_point(&mesh, finest, face, x, y));
                (face.index(), points.collect())
            })
            .collect();
        let places: Vec<Place> = (0..=top)
            .flat_map(|l| {
                let at = level(l);
                let per_side = u16::try_from(at.tiles_per_side()).expect("small");
                mesh.faces().flat_map(move |face| {
                    (0..per_side).flat_map(move |ty| {
                        (0..per_side).map(move |tx| {
                            Place::of(&TileKey::display(face, at, tx, ty).expect("a key"))
                        })
                    })
                })
            })
            .collect();
        let capacity = u32::try_from(places.len()).expect("small");
        let mut tree = TileTree::new(gpu, &mesh, capacity).expect("a tree");
        let mut tiles = BTreeMap::new();
        let mut classes: Vec<Vec<GpuVertex>> = vec![Vec::new(); CLASSES];
        for place in places {
            let key = place.key(finest).expect("a tile");
            let quads = place.level().tile_quads();
            let scale = n / place.level().face_quads();
            let grid = &grids[&place.face().index()];
            let at = |i: u32, j: u32| {
                let x = (u32::from(key.tx()) * quads + i) * scale;
                let y = (u32::from(key.ty()) * quads + j) * scale;
                grid[usize::try_from(y * (n + 1) + x).expect("small")]
            };
            let vertices: Vec<GpuVertex> = (0..=quads)
                .flat_map(|j| (0..=quads).map(move |i| (i, j)))
                .map(|(i, j)| {
                    let p = at(i, j);
                    GpuVertex {
                        position: p,
                        roughness: 1.0,
                        normal: p,
                        reserved: 0,
                    }
                })
                .collect();
            let finest_tile = place.level() == finest;
            let detail = if finest_tile {
                TileDetail::FINEST
            } else {
                let normals: Vec<geom::Vec3> = vertices
                    .iter()
                    .map(|v| geom::Vec3::new(v.normal[0], v.normal[1], v.normal[2]))
                    .collect();
                TileDetail {
                    error: measured_error(place, finest, &vertices, grid, scale),
                    cone: FinestNormalCone::covering(&normals).expect("normals"),
                }
            };
            let class = SizeClass::of(place.level());
            let per_class = &mut classes[usize::from(class.get())];
            let side = usize::try_from(class.side()).expect("small");
            let index = per_class.len() / (side * side);
            per_class.extend_from_slice(&vertices);
            let flags = if finest_tile { NODE_FINEST } else { 0 };
            let slot = Slot::for_tests(class, index);
            let node = tree.insert(place, slot, detail, flags).expect("inserted");
            tiles.insert(
                place,
                Tile {
                    node,
                    slot,
                    vertices,
                    detail,
                },
            );
        }
        let buffers = classes.iter().map(|data| {
            if data.is_empty() {
                return Vec::new();
            }
            vec![gpu.device().create_buffer_init(&BufferInitDescriptor {
                label: Some("test slots"),
                contents: bytemuck::cast_slice(data),
                usage: BufferUsages::STORAGE,
            })]
        });
        let mut world = Self {
            mesh,
            finest,
            tree,
            tiles,
            slots: SlotBuffers::new(buffers.collect()),
        };
        world.write_tree(gpu);
        world
    }

    /// Writes the tree's changed nodes.
    fn write_tree(&mut self, gpu: &Gpu) {
        for (start, nodes) in self.tree.take_dirty() {
            gpu.queue().write_buffer(
                self.tree.buffer(),
                u64::from(start) * NODE_BYTES,
                bytemuck::cast_slice(&nodes),
            );
        }
    }

    /// The node of `place`.
    fn node(&self, place: Place) -> u32 {
        self.tiles[&place].node
    }

    /// The slot layout the raster binds: one buffer more of each class than it has, as
    /// residency's most is more than it has made, so every draw binds placeholders too.
    fn layout(&self) -> SlotLayout {
        let mut per_class = [0_u16; CLASSES];
        let classes = per_class.iter_mut().zip(self.slots.classes());
        for (most, buffers) in classes {
            *most = u16::try_from(buffers.len() + 1).expect("small");
        }
        SlotLayout::new(per_class)
    }

    /// The place of `node`.
    fn place_of(&self, node: u32) -> Place {
        let found = self.tiles.iter().find(|(_, tile)| tile.node == node);
        *found.expect("a resident node").0
    }

    /// The `e` a tile's level change trades, as the raster takes it: its parent's, 0 at level 0.
    fn parent_error(&self, place: Place) -> f32 {
        place
            .parent()
            .map_or(0.0, |parent| self.tiles[&parent].detail.error)
    }
}

/// The largest distance from a finest vertex under the tile at `place` (whose vertices are
/// `vertices`) to the point of the tile's own triangles at the same place, a little over.
///
/// The finest surface's triangles each lie in one of the tile's (decision 4's diagonals nest),
/// so the distance between the two surfaces is largest at a finest vertex.
fn measured_error(
    place: Place,
    finest: Level,
    vertices: &[GpuVertex],
    grid: &[[f32; 3]],
    scale: u32,
) -> f32 {
    let n = finest.face_quads();
    let quads = place.level().tile_quads();
    let side = quads + 1;
    let key = place.key(finest).expect("a tile");
    let (x0, y0) = (
        u32::from(key.tx()) * quads * scale,
        u32::from(key.ty()) * quads * scale,
    );
    let at = |i: u32, j: u32| vertices[usize::try_from(j * side + i).expect("small")].position;
    let mut worst = 0.0_f32;
    for y in 0..=quads * scale {
        for x in 0..=quads * scale {
            let fine = grid[usize::try_from((y0 + y) * (n + 1) + x0 + x).expect("small")];
            let (qx, qy) = ((x / scale).min(quads - 1), (y / scale).min(quads - 1));
            let fx = small(x - qx * scale) / small(scale);
            let fy = small(y - qy * scale) / small(scale);
            let (p00, p10, p11, p01) = (
                at(qx, qy),
                at(qx + 1, qy),
                at(qx + 1, qy + 1),
                at(qx, qy + 1),
            );
            let coarse: [f32; 3] = [0, 1, 2].map(|a| {
                if fx >= fy {
                    p00[a] + fx * (p10[a] - p00[a]) + fy * (p11[a] - p10[a])
                } else {
                    p00[a] + fx * (p11[a] - p01[a]) + fy * (p01[a] - p00[a])
                }
            });
            let d = [0, 1, 2].map(|a| fine[a] - coarse[a]);
            worst = worst.max((d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt());
        }
    }
    worst * 1.01 + 1e-6
}

/// What lies across an edge of a tile in a cut.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Across {
    /// The neighbour at the tile's level is in the cut.
    Same,
    /// An ancestor of it is.
    Coarser,
    /// Its descendants are.
    Finer,
}

/// The tile of `cut` covering `place`: it or an ancestor.
fn covering(cut: &BTreeSet<Place>, place: Place) -> Option<Place> {
    let mut at = Some(place);
    while let Some(p) = at {
        if cut.contains(&p) {
            return Some(p);
        }
        at = p.parent();
    }
    None
}

/// What lies across each edge of `place`, in `cut`.
fn across(world: &World, cut: &BTreeSet<Place>, place: Place) -> [Across; 4] {
    place.neighbours(world.mesh.topology()).map(|m| {
        let m = m.expect("a closed mesh");
        match covering(cut, m) {
            Some(p) if p == m => Across::Same,
            Some(_) => Across::Coarser,
            None => Across::Finer,
        }
    })
}

/// A random cut of tiles from level `low` to the finest, neighbours at most a level apart.
fn random_cut(rng: &mut Rng, world: &World, low: u8) -> BTreeSet<Place> {
    let start = world.tiles.keys().filter(|p| p.level().get() == low);
    let cut: BTreeSet<Place> = start.copied().collect();
    grown(rng, world, cut, 5)
}

/// `cut` with each tile refined at random (two in five, in each of `rounds`), then relaxed.
fn grown(rng: &mut Rng, world: &World, mut cut: BTreeSet<Place>, rounds: u32) -> BTreeSet<Place> {
    for _ in 0..rounds {
        let tiles: Vec<Place> = cut.iter().copied().collect();
        for place in tiles {
            if place.level() < world.finest && rng.below(100) < 40 {
                refine(&mut cut, place);
            }
        }
    }
    // Relax: refine any tile two levels coarser than a neighbour in the cut.
    loop {
        let raise = cut.iter().find_map(|&c| {
            let neighbours = c.neighbours(world.mesh.topology()).into_iter().flatten();
            neighbours
                .filter_map(|m| covering(&cut, m))
                .find(|a| c.level().get() - a.level().get() >= 2)
        });
        let Some(raise) = raise else { break };
        refine(&mut cut, raise);
    }
    cut
}

/// Replaces `place` in `cut` by its children.
fn refine(cut: &mut BTreeSet<Place>, place: Place) {
    cut.remove(&place);
    cut.extend(place.children().into_iter().flatten());
}

/// The clusters `emit` would list for `cut` with nothing culled: edge variants where the
/// neighbour is coarser, unless `variants` is false (the broken seam rule); finer edges where
/// it is finer.
fn cut_clusters(world: &World, cut: &BTreeSet<Place>, variants: bool) -> Vec<GpuCluster> {
    let mut clusters = Vec::new();
    for &place in cut {
        let sides = across(world, cut, place);
        let mask = |kind: Across| {
            sides
                .iter()
                .enumerate()
                .filter(|(_, s)| **s == kind)
                .fold(0_u32, |m, (k, _)| m | (1 << k))
        };
        let (coarser, finer) = (
            if variants { mask(Across::Coarser) } else { 0 },
            mask(Across::Finer),
        );
        let quads = place.level().tile_quads();
        let per = (quads / 8).max(1);
        for cy in 0..per {
            for cx in 0..per {
                let on = u32::from(cy == 0)
                    | (u32::from(cx == per - 1) << 1)
                    | (u32::from(cy == per - 1) << 2)
                    | (u32::from(cx == 0) << 3);
                let cell = cx
                    | (cy << 8)
                    | (quads.min(8) << 16)
                    | ((coarser & on) << 24)
                    | ((finer & on) << 28);
                clusters.push(GpuCluster {
                    node: world.node(place),
                    cell,
                });
            }
        }
    }
    clusters
}

/// A cluster list and its draw arguments on the GPU.
fn cluster_buffers(gpu: &Gpu, clusters: &[GpuCluster]) -> (Buffer, Buffer) {
    let device = gpu.device();
    let list = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("test clusters"),
        contents: bytemuck::cast_slice(if clusters.is_empty() {
            &[GpuCluster { node: 0, cell: 0 }]
        } else {
            clusters
        }),
        usage: BufferUsages::STORAGE,
    });
    let count = u32::try_from(clusters.len()).expect("small");
    let args = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("test draw args"),
        contents: bytemuck::cast_slice(&[crate::lod::CLUSTER_INDICES, count, 0, 0, 0]),
        usage: BufferUsages::INDIRECT,
    });
    (list, args)
}

/// A cut list on the GPU: its entries and its count, at word 0.
fn cut_buffers(gpu: &Gpu, world: &World, cut: &BTreeSet<Place>) -> (Buffer, Buffer) {
    let device = gpu.device();
    let entries: Vec<GpuCutEntry> = cut
        .iter()
        .map(|&p| GpuCutEntry {
            face: p.face().index(),
            place: p.packed(),
            node: world.node(p),
            flags: 0,
        })
        .collect();
    let list = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("test cut"),
        contents: bytemuck::cast_slice(&entries),
        usage: BufferUsages::STORAGE,
    });
    let count = u32::try_from(entries.len()).expect("small");
    let counts = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("test cut count"),
        contents: bytemuck::cast_slice(&[count, 0, 0, 0]),
        usage: BufferUsages::STORAGE,
    });
    (list, counts)
}

/// Waits for the GPU, checking it raised nothing.
fn finish(gpu: &Gpu) {
    gpu.device()
        .poll(PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .expect("the GPU finishes");
    gpu.check().expect("no GPU error");
}

/// The visibility buffer, row by row.
fn read_visibility(gpu: &Gpu, raster: &Raster) -> Vec<u32> {
    let texture = raster.visibility();

    let (width, height) = (texture.width(), texture.height());
    let row = (width * 4).next_multiple_of(256);
    let buffer = gpu.device().create_buffer(&BufferDescriptor {
        label: Some("visibility readback"),
        size: u64::from(row * height),
        usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device()
        .create_command_encoder(&CommandEncoderDescriptor::default());
    encoder.copy_texture_to_buffer(
        TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        TexelCopyBufferInfo {
            buffer: &buffer,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: Some(height),
            },
        },
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue().submit([encoder.finish()]);
    let padded: Vec<u32> = gpu.read_buffer(&buffer).expect("read back");
    let stride = usize::try_from(row / 4).expect("small");
    let width = usize::try_from(width).expect("small");
    padded
        .chunks(stride)
        .flat_map(|r| r[..width].to_vec())
        .collect()
}

/// Draws `cut` (its clusters `clusters`) as one frame at `time`, stamping it first, and returns
/// the visibility buffer.
fn draw_cut(
    gpu: &Gpu,
    raster: &mut Raster,
    world: &World,
    cut: &BTreeSet<Place>,
    clusters: &[GpuCluster],
    frame: &RasterFrame<'_>,
) -> Vec<u32> {
    raster.begin_frame(gpu, frame).expect("a frame");
    let (entries, counts) = cut_buffers(gpu, world, cut);
    let (list, args) = cluster_buffers(gpu, clusters);
    let mut encoder = gpu
        .device()
        .create_command_encoder(&CommandEncoderDescriptor::default());
    let cut_list = CutList {
        entries: &entries,
        counts: &counts,
        count_word: 0,
    };
    raster
        .encode_stamp(gpu, &mut encoder, world.tree.buffer(), cut_list)
        .expect("encoded");
    let draw = ClusterDraw {
        tree: world.tree.buffer(),
        slots: &world.slots,
        clusters: &list,
        args: (&args, 0),
    };
    raster
        .encode_draw(gpu, &mut encoder, &draw, Phase::First)
        .expect("encoded");
    gpu.queue().submit([encoder.finish()]);
    finish(gpu);
    read_visibility(gpu, raster)
}

/// A random view of the sphere: from `distance` (a random one in its range) along a random
/// direction, towards a point near the centre.
fn random_view(rng: &mut Rng, distances: (f32, f32)) -> (View, f64) {
    loop {
        let d = [
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
        ];
        let length = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if !(0.1..=1.0).contains(&length) {
            continue;
        }
        let distance = f64::from(rng.range(distances.0, distances.1));
        let direction = Vec3::new(f64::from(d[0]), f64::from(d[1]), f64::from(d[2]));
        let eye = direction.normalized().expect("not zero");
        let eye = Vec3::new(eye.x * distance, eye.y * distance, eye.z * distance);
        let target = Vec3::new(
            f64::from(rng.range(-0.2, 0.2)),
            f64::from(rng.range(-0.2, 0.2)),
            f64::from(rng.range(-0.2, 0.2)),
        );
        let up = [Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, 1.0)];
        let view = up
            .into_iter()
            .find_map(|up| View::look_at(eye, target, up, 45_f64.to_radians(), size(), 0.01));
        return (view.expect("a view"), distance);
    }
}

/// Every finest tile, the reference cut.
fn finest_cut(world: &World) -> BTreeSet<Place> {
    let finest = world.tiles.keys().filter(|p| p.level() == world.finest);
    finest.copied().collect()
}

/// A raster that never morphs: every vertex where it is.
fn still_raster(gpu: &Gpu, world: &World) -> Raster {
    let config = RasterConfig::default()
        .with_morph(Duration::ZERO)
        .with_ramp(MorphThreshold::OFF);
    Raster::new(gpu, world.tree.capacity(), world.layout(), config).expect("the raster")
}

/// The reference: the finest level, every vertex where it is, from `view`.
fn reference(gpu: &Gpu, world: &World, view: &View) -> Vec<u32> {
    let mut raster = still_raster(gpu, world);
    let cut = finest_cut(world);
    let clusters = cut_clusters(world, &cut, true);
    let frame = RasterFrame {
        view,
        time: Duration::ZERO,
        moving: false,
    };
    draw_cut(gpu, &mut raster, world, &cut, &clusters, &frame)
}

/// Pixels inside the reference's silhouette by at least `margin` pixels: every pixel within
/// `margin` of them (a square) is drawn in `reference`.
fn inside(reference: &[u32], margin: usize) -> Vec<bool> {
    let (width, height) = (
        usize::try_from(size().width()).expect("small"),
        usize::try_from(size().height()).expect("small"),
    );
    let drawn: Vec<bool> = reference.iter().map(|&id| id != BACKGROUND).collect();
    // Erode along rows, then columns.
    let erode = |source: &[bool], along_rows: bool| -> Vec<bool> {
        let mut out = vec![false; source.len()];
        for y in 0..height {
            for x in 0..width {
                let (at, limit) = if along_rows { (x, width) } else { (y, height) };
                let lo = at.checked_sub(margin);
                let hi = at + margin;
                out[y * width + x] = lo.is_some()
                    && hi < limit
                    && (at - margin..=hi).all(|k| {
                        let index = if along_rows {
                            y * width + k
                        } else {
                            k * width + x
                        };
                        source[index]
                    });
            }
        }
        out
    };
    let rows = erode(&drawn, true);
    erode(&rows, false)
}

/// Background pixels of `image` where `mask` says the surface must be.
fn pinholes(mask: &[bool], image: &[u32]) -> usize {
    mask.iter()
        .zip(image)
        .filter(|&(&must, &id)| must && id == BACKGROUND)
        .count()
}

/// The margin, in pixels, a cut's silhouette may lie inside the finest level's from a camera
/// `distance` from the centre, and two pixels for rasterization.
///
/// Every tile's surface lies within its `e` inside the unit sphere, and within its parent's for
/// the tiles of `morphing` (the surface a morph starts from). Shrinking the sphere by `e` moves the
/// silhouette in by `e / sqrt(d² - 1)` radians: the distance to the silhouette's points.
fn margin(
    world: &World,
    cut: &BTreeSet<Place>,
    view: &View,
    distance: f64,
    morphing: &BTreeSet<Place>,
) -> usize {
    let worst = cut
        .iter()
        .map(|&p| {
            let own = world.tiles[&p].detail.error;
            if morphing.contains(&p) {
                world.parent_error(p).max(own)
            } else {
                own
            }
        })
        .fold(0.0_f32, f32::max);
    let pixels = f64::from(worst) * view.pixels_per_unit() / (distance * distance - 1.0).sqrt();
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a small positive number of pixels"
    )]
    let margin = pixels.ceil() as usize;
    margin + 2
}

/// The edge variants of `clusters`, by size.
fn variants_of(clusters: &[GpuCluster]) -> BTreeSet<(u32, u32)> {
    clusters.iter().map(|c| (c.quads(), c.variant())).collect()
}

/// The view of `face` head on, from 2.5 away from the centre.
fn facing(world: &World, face: FaceId) -> View {
    let whole = Place::of(&TileKey::display(face, level(0), 0, 0).expect("a key"));
    let centre = world.tiles[&whole]
        .vertices
        .iter()
        .fold([0.0_f64; 3], |s, v| {
            [0, 1, 2].map(|a| s[a] + f64::from(v.position[a]))
        });
    let towards = Vec3::new(centre[0], centre[1], centre[2])
        .normalized()
        .expect("not zero");
    let eye = Vec3::new(towards.x * 2.5, towards.y * 2.5, towards.z * 2.5);
    [Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, 1.0)]
        .into_iter()
        .find_map(|up| View::look_at(eye, Vec3::default(), up, 45_f64.to_radians(), size(), 0.01))
        .expect("a view")
}

/// The lowest level of each random cut the pinhole tests draw.
const LOWS: [u8; 6] = [2, 3, 4, 3, 5, 2];

/// Whole-face tiles: `face` at level `at`, the faces across its edges named by `variant` a
/// level coarser, every other face at `at`.
fn variant_cut(world: &World, face: FaceId, at: u8, variant: Variant) -> BTreeSet<Place> {
    let key = TileKey::display(face, level(at), 0, 0).expect("a key");
    let centre = Place::of(&key);
    let neighbours = centre.neighbours(world.mesh.topology());
    let coarser: BTreeSet<u32> = neighbours
        .iter()
        .zip(indices::Edge::ALL)
        .filter(|(_, edge)| variant.has(*edge))
        .map(|(m, _)| m.expect("a closed mesh").face().index())
        .collect();
    world
        .mesh
        .faces()
        .map(|f| {
            let l = if coarser.contains(&f.index()) {
                at - 1
            } else {
                at
            };
            Place::of(&TileKey::display(f, level(l), 0, 0).expect("a key"))
        })
        .collect()
}

#[test]
fn the_pinhole_test_every_edge_variant_of_a_lone_cluster_shows_no_background_inside() {
    GpuTest::run(|gpu| {
        let world = World::new(gpu);
        // Every edge variant of a single-cluster tile, of 8 quads (level 3) and of 4 (level
        // 2), seen head on, every vertex in place.
        let face = FaceId::new(0);
        let head_on = facing(&world, face);
        let truth = reference(gpu, &world, &head_on);
        let mut still = still_raster(gpu, &world);
        let mut seen = BTreeSet::new();
        for at in [2, 3] {
            for variant in Variant::all() {
                let cut = variant_cut(&world, face, at, variant);
                let clusters = cut_clusters(&world, &cut, true);
                seen.extend(variants_of(&clusters));
                let mask = inside(
                    &truth,
                    margin(&world, &cut, &head_on, 2.5, &BTreeSet::new()),
                );
                let frame = RasterFrame {
                    view: &head_on,
                    time: Duration::ZERO,
                    moving: false,
                };
                let image = draw_cut(gpu, &mut still, &world, &cut, &clusters, &frame);
                let holes = pinholes(&mask, &image);
                assert_eq!(holes, 0, "level {at} variant {variant}: {holes} pinholes");
                assert!(
                    mask.iter().filter(|&&m| m).count() > 20_000,
                    "the face is checked"
                );
            }
        }
        let wanted = [8, 4]
            .into_iter()
            .flat_map(|q| Variant::all().map(move |v| (q, v)));
        for (quads, variant) in wanted {
            assert!(
                seen.contains(&(quads, u32::from(variant.bits()))),
                "variant {variant} of {quads} quads drawn"
            );
        }
    });
}

/// Face 0 and its neighbour across edge 0, as the stages of two refinements made with the
/// camera still: every face at level 2 (at 800 ms), face 0 at 3 (900 ms), and both at 3
/// (1000 ms).
fn two_refinements(world: &World) -> (FaceId, FaceId, [(BTreeSet<Place>, u64); 3]) {
    let face = FaceId::new(0);
    let neighbour = Place::of(&TileKey::display(face, level(3), 0, 0).expect("a key"))
        .neighbours(world.mesh.topology())[0]
        .expect("a closed mesh")
        .face();
    let faces_at = |levels: &dyn Fn(FaceId) -> u8| -> BTreeSet<Place> {
        let tiles = world
            .mesh
            .faces()
            .map(|f| Place::of(&TileKey::display(f, level(levels(f)), 0, 0).expect("a key")));
        tiles.collect()
    };
    let stages = [
        (faces_at(&|_| 2), 800),
        (faces_at(&|f| if f == face { 3 } else { 2 }), 900),
        (
            faces_at(&|f| if f == face || f == neighbour { 3 } else { 2 }),
            1000,
        ),
    ];
    (face, neighbour, stages)
}

#[test]
fn the_pinhole_test_a_seam_between_morphs_takes_both_sides_shared_factors() {
    GpuTest::run(|gpu| {
        let mut world = World::new(gpu);
        let (face, neighbour, stages) = two_refinements(&world);
        let (last, _) = &stages[2];
        let morphing = last - &stages[0].0;
        let views: Vec<(FaceId, View, Vec<bool>)> = [face, neighbour]
            .into_iter()
            .map(|seen_from| {
                let view = facing(&world, seen_from);
                let truth = reference(gpu, &world, &view);
                let mask = inside(&truth, margin(&world, last, &view, 2.5, &morphing));
                (seen_from, view, mask)
            })
            .collect();
        let last_clusters = cut_clusters(&world, last, true);
        // By time: at 1040 ms face 0 is 93 % through its morph and its neighbour 27 %, so
        // their seam is drawn by the less advanced stamp on both sides.
        let by_time = RasterConfig::default().with_ramp(MorphThreshold::OFF);
        let mut raster =
            Raster::new(gpu, world.tree.capacity(), world.layout(), by_time).expect("the raster");
        for (seen_from, view, mask) in &views {
            let at = |ms: u64| RasterFrame {
                view,
                time: Duration::from_millis(ms),
                moving: false,
            };
            for (stage, ms) in &stages {
                let stage_clusters = cut_clusters(&world, stage, true);
                draw_cut(gpu, &mut raster, &world, stage, &stage_clusters, &at(*ms));
            }
            let image = draw_cut(gpu, &mut raster, &world, last, &last_clusters, &at(1040));
            let holes = pinholes(mask, &image);
            assert_eq!(
                holes, 0,
                "stamps 100 ms apart, from {seen_from}: {holes} pinholes"
            );
        }
        // By distance: the two faces' parents' e set far apart (their data unchanged: only the
        // bound the morph reads; the margins are from the true bounds), so one side alone would
        // draw its odd vertices in place and the other at their midpoints: the seam takes the
        // larger.
        for (f, error) in [(face, 1.0), (neighbour, 1e-7)] {
            let parent = Place::of(&TileKey::display(f, level(2), 0, 0).expect("a key"));
            let tile = &world.tiles[&parent];
            let detail = TileDetail {
                error,
                ..tile.detail
            };
            let slot = tile.slot;
            assert!(world.tree.move_to(parent, slot, detail).is_some());
        }
        world.write_tree(gpu);
        let by_distance = RasterConfig::default().with_morph(Duration::ZERO);
        let mut distance_raster =
            Raster::new(gpu, world.tree.capacity(), world.layout(), by_distance)
                .expect("the raster");
        for (seen_from, view, mask) in &views {
            let frame = RasterFrame {
                view,
                time: Duration::ZERO,
                moving: true,
            };
            let image = draw_cut(
                gpu,
                &mut distance_raster,
                &world,
                last,
                &last_clusters,
                &frame,
            );
            let holes = pinholes(mask, &image);
            assert_eq!(
                holes, 0,
                "parent e far apart, from {seen_from}: {holes} pinholes"
            );
        }
    });
}

#[test]
fn the_pinhole_test_random_cameras_and_random_mixed_level_cuts_show_no_background_inside() {
    GpuTest::run(|gpu| {
        let world = World::new(gpu);
        let mut rng = Rng(41);
        // Morphing by time alone, so every refined tile's odd vertices move (the distance
        // factor is the level-choice pinhole test's).
        let config = RasterConfig::default().with_ramp(MorphThreshold::OFF);
        let mut raster =
            Raster::new(gpu, world.tree.capacity(), world.layout(), config).expect("the raster");
        let mut seen = BTreeSet::new();
        let mut checked = 0;
        // Random cuts from random cameras, refined twice with the camera still, so tiles of
        // different stamps meet at seams: frame 1 draws the tiles at `low`; frame 2 a random
        // cut refining them, at 900 ms; frame 3 that cut refined again, at 1000 ms. At 1040 ms
        // the first refinement's tiles are 93 % through their morph, the second's 27 %; at
        // 1200 ms all are done.
        let seeds = LOWS.iter().enumerate();
        for (seed, &low) in seeds {
            let (view, distance) = random_view(&mut rng, (2.0, 3.2));
            let base: BTreeSet<Place> = world
                .tiles
                .keys()
                .filter(|p| p.level().get() == low)
                .copied()
                .collect();
            let first = grown(&mut rng, &world, base.clone(), 3);
            let cut = grown(&mut rng, &world, first.clone(), 2);
            let clusters = cut_clusters(&world, &cut, true);
            seen.extend(variants_of(&clusters));
            let mask = inside(
                &reference(gpu, &world, &view),
                margin(&world, &cut, &view, distance, &(&cut - &base)),
            );
            let at = |ms: u64| RasterFrame {
                view: &view,
                time: Duration::from_millis(ms),
                moving: false,
            };
            for (earlier, ms) in [(&base, 800), (&first, 900)] {
                let earlier_clusters = cut_clusters(&world, earlier, true);
                draw_cut(
                    gpu,
                    &mut raster,
                    &world,
                    earlier,
                    &earlier_clusters,
                    &at(ms),
                );
            }
            draw_cut(gpu, &mut raster, &world, &cut, &clusters, &at(1000));
            for ms in [1040, 1200] {
                let image = draw_cut(gpu, &mut raster, &world, &cut, &clusters, &at(ms));
                let holes = pinholes(&mask, &image);
                assert_eq!(holes, 0, "seed {seed} at {ms} ms: {holes} pinholes");
                checked += mask.iter().filter(|&&m| m).count();
            }
        }
        println!("pinhole test: {checked} pixels checked, variants drawn {seen:?}");
        assert!(checked > 100_000, "{checked} pixels checked");
    });
}

#[test]
fn the_pinhole_test_bites_a_finer_tile_drawing_its_odd_vertices_beside_a_coarser_one_leaks() {
    GpuTest::run(|gpu| {
        let world = World::new(gpu);
        let mut rng = Rng(41);
        let mut raster = still_raster(gpu, &world);
        let (mut holes_kept, mut holes_broken) = (0, 0);
        for &low in &LOWS {
            let (view, distance) = random_view(&mut rng, (2.0, 3.2));
            let cut = random_cut(&mut rng, &world, low);
            let mask = inside(
                &reference(gpu, &world, &view),
                margin(&world, &cut, &view, distance, &BTreeSet::new()),
            );
            let frame = RasterFrame {
                view: &view,
                time: Duration::ZERO,
                moving: false,
            };
            let kept = cut_clusters(&world, &cut, true);
            holes_kept += pinholes(
                &mask,
                &draw_cut(gpu, &mut raster, &world, &cut, &kept, &frame),
            );
            let broken = cut_clusters(&world, &cut, false);
            let leaks = pinholes(
                &mask,
                &draw_cut(gpu, &mut raster, &world, &cut, &broken, &frame),
            );
            println!(
                "low {low}: margin {}, {} pixels inside, {leaks} leaks",
                margin(&world, &cut, &view, distance, &BTreeSet::new()),
                mask.iter().filter(|&&m| m).count()
            );
            holes_broken += leaks;
        }
        println!(
            "random cuts: {holes_kept} pinholes with the edge variants, {holes_broken} without"
        );
        assert_eq!(holes_kept, 0, "watertight with the variants");
        assert!(
            holes_broken > 0,
            "the plain variant beside a coarser tile leaks"
        );
        // Face 0 at level 3 beside one coarser face across each of its edges in turn, seen
        // from that coarser face: a finer edge's odd vertices then bulge into the finer tile on
        // the screen, so drawing them opens a gap there (seen from the finer side they bulge
        // over the coarser tile instead, and nothing shows).
        let face = FaceId::new(0);
        let centre = Place::of(&TileKey::display(face, level(3), 0, 0).expect("a key"));
        let neighbours = centre.neighbours(world.mesh.topology());
        let edges = indices::Edge::ALL.into_iter().zip(neighbours);
        for (edge, m) in edges {
            let coarser_face = m.expect("a closed mesh").face();
            let view = facing(&world, coarser_face);
            let cut = variant_cut(&world, face, 3, Variant::of_edges(&[edge]));
            let mask = inside(
                &reference(gpu, &world, &view),
                margin(&world, &cut, &view, 2.5, &BTreeSet::new()),
            );
            let frame = RasterFrame {
                view: &view,
                time: Duration::ZERO,
                moving: false,
            };
            let kept = cut_clusters(&world, &cut, true);
            let kept = pinholes(
                &mask,
                &draw_cut(gpu, &mut raster, &world, &cut, &kept, &frame),
            );
            let broken = cut_clusters(&world, &cut, false);
            let leaks = pinholes(
                &mask,
                &draw_cut(gpu, &mut raster, &world, &cut, &broken, &frame),
            );
            println!(
                "level 3 beside level 2 across {edge:?}: {kept} pinholes kept, {leaks} broken"
            );
            assert_eq!(kept, 0, "{edge:?}: watertight with the variant");
            assert!(leaks > 0, "{edge:?}: the plain variant leaks");
        }
    });
}

/// The level choice for `world`, refining by `e` alone (no quad cap), with a request buffer.
fn level_choice(gpu: &Gpu, world: &World) -> (Lod, Buffer) {
    let thresholds = Thresholds {
        refine_quad: 1e9,
        coarsen_quad: 1e9,
        ..Thresholds::SPEC
    };
    let config = LodConfig::default().with_thresholds(thresholds);
    let lod = Lod::new(gpu, world.tree.capacity(), config).expect("the level choice");
    let requests = gpu.device().create_buffer(&BufferDescriptor {
        label: Some("test requests"),
        size: REQUEST_HEADER + 64 * 16,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    (lod, requests)
}

/// One frame through the level choice's first phase and the raster: bounds (on the first
/// frame), the choice, the stamps, the draw. Returns the visibility buffer, the clusters listed
/// and the cut's places.
fn lod_frame(
    gpu: &Gpu,
    world: &World,
    lod: &mut Lod,
    requests: &Buffer,
    raster: &mut Raster,
    frame: &RasterFrame<'_>,
    bounds: bool,
) -> (Vec<u32>, Vec<GpuCluster>, BTreeSet<Place>) {
    raster.begin_frame(gpu, frame).expect("a frame");
    gpu.queue().write_buffer(requests, 0, &[0; 16]);
    let mut encoder = gpu
        .device()
        .create_command_encoder(&CommandEncoderDescriptor::default());
    if bounds {
        let update = BoundsUpdate::since(&world.tree, 0);
        lod.encode_bounds(
            gpu,
            &mut encoder,
            world.tree.buffer(),
            &update,
            &world.slots,
        )
        .expect("encoded");
    }
    let inputs = MainInputs {
        tree: world.tree.buffer(),
        requests,
        view: frame.view,
        previous: None,
        finest: world.finest,
    };
    lod.encode_main(gpu, &mut encoder, &inputs)
        .expect("encoded");
    raster
        .encode_stamp(gpu, &mut encoder, world.tree.buffer(), lod.cut_list())
        .expect("encoded");
    let (args, offset) = lod.draw_args(false);
    let args = args.clone();
    let draw = ClusterDraw {
        tree: world.tree.buffer(),
        slots: &world.slots,
        clusters: lod.clusters(false),
        args: (&args, offset),
    };
    raster
        .encode_draw(gpu, &mut encoder, &draw, Phase::First)
        .expect("encoded");
    let readback = lod
        .encode_readback(gpu, &mut encoder)
        .expect("encoded")
        .expect("a readback buffer");
    gpu.queue().submit([encoder.finish()]);
    readback.submitted();
    finish(gpu);
    let output = lod.collect(world.finest).expect("read").expect("arrived");
    let image = read_visibility(gpu, raster);
    let words: Vec<u32> = gpu.read_buffer(&args).expect("read back");
    let count =
        usize::try_from(words[usize::try_from(offset / 4).expect("small") + 1]).expect("small");
    let listed: Vec<GpuCluster> = gpu.read_buffer(lod.clusters(false)).expect("read back");
    let cut = output.nodes.iter().map(|(_, p)| *p).collect();
    (image, listed[..count].to_vec(), cut)
}

#[test]
fn the_pinhole_test_holds_for_the_level_choice_s_cuts_morphing_by_distance() {
    GpuTest::run(|gpu| {
        let world = World::new(gpu);
        let mut rng = Rng(7);
        let (mut lod, requests) = level_choice(gpu, &world);
        let mut raster = Raster::new(
            gpu,
            world.tree.capacity(),
            world.layout(),
            RasterConfig::default(),
        )
        .expect("the raster");
        let mut levels = BTreeSet::new();
        for seed in 0..5 {
            let (view, _) = random_view(&mut rng, (1.6, 6.0));
            let frame = RasterFrame {
                view: &view,
                time: Duration::from_secs(1),
                moving: true,
            };
            let (image, clusters, cut) = lod_frame(
                gpu,
                &world,
                &mut lod,
                &requests,
                &mut raster,
                &frame,
                seed == 0,
            );
            levels.extend(cut.iter().map(|p| p.level().get()));
            assert!(!clusters.is_empty(), "seed {seed}: something drawn");
            // The level choice holds what it draws within a pixel of the finest surface, and
            // the morph by distance within two of the pixel threshold: four pixels of margin.
            let mask = inside(&reference(gpu, &world, &view), 4);
            let holes = pinholes(&mask, &image);
            assert_eq!(holes, 0, "seed {seed}: {holes} pinholes");
        }
        // The sphere's true bounds are small: the level choice draws it at two levels from here.
        assert!(levels.len() >= 2, "mixed levels: {levels:?}");
    });
}

/// The quad a visibility id names, as its corners `[p00, p10, p11, p01]` in pixels, where the
/// cluster's edge variant draws them (morphing off).
fn named_quad(
    world: &World,
    clusters: &[GpuCluster],
    id: VisibilityId,
    view: &View,
) -> [(f64, f64); 4] {
    let place = world.place_of(id.node);
    let quads = place.level().tile_quads();
    let per = (quads / 8).max(1);
    let (cx, cy) = (id.cluster % per, id.cluster / per);
    let cluster = clusters
        .iter()
        .find(|c| c.node == id.node && c.position() == (cx, cy))
        .expect("the id's cluster was listed");
    let size = ClusterSize::of_quads(cluster.quads()).expect("a size");
    let variant =
        Variant::new(u8::try_from(cluster.variant()).expect("4 bits")).expect("a variant");
    let (qx, qy) = (
        u8::try_from(id.quad % 8).expect("small"),
        u8::try_from(id.quad / 8).expect("small"),
    );
    let tile = &world.tiles[&place];
    let row = quads + 1;
    [(0, 0), (1, 0), (1, 1), (0, 1)].map(|(du, dv)| {
        let corner = GridVertex::new(qx + du, qy + dv).expect("on the grid");
        let drawn = remap(size, variant, corner);
        let column = cx * 8 + u32::from(drawn.u());
        let line = cy * 8 + u32::from(drawn.v());
        let at = tile.vertices[usize::try_from(line * row + column).expect("small")].position;
        let point = Vec3::new(f64::from(at[0]), f64::from(at[1]), f64::from(at[2]));
        let (x, y, _) = view.pixel(point).expect("in front");
        (x, y)
    })
}

/// Quad `corners`' triangle `which`: 0 `(p00, p10, p11)`, 1 `(p00, p11, p01)`.
const fn quad_half(corners: [(f64, f64); 4], which: u32) -> [(f64, f64); 3] {
    let [p00, p10, p11, p01] = corners;
    if which == 0 {
        [p00, p10, p11]
    } else {
        [p00, p11, p01]
    }
}

/// Whether `pixel` (a centre) lies in `triangle`, or within `slack` pixels of it.
fn covers(triangle: [(f64, f64); 3], pixel: (f64, f64), slack: f64) -> bool {
    let [first, second, third] = triangle;
    let area =
        (second.0 - first.0) * (third.1 - first.1) - (second.1 - first.1) * (third.0 - first.0);
    let edge = |from: (f64, f64), to: (f64, f64)| {
        let length = (to.0 - from.0).hypot(to.1 - from.1);
        let cross = (to.0 - from.0) * (pixel.1 - from.1) - (to.1 - from.1) * (pixel.0 - from.0);
        cross * area.signum() / length
    };
    area != 0.0
        && edge(first, second) >= -slack
        && edge(second, third) >= -slack
        && edge(third, first) >= -slack
}

/// The centre of pixel `index` of the test viewport.
fn pixel_centre(index: usize) -> (f64, f64) {
    let index = u32::try_from(index).expect("small");
    let width = size().width();
    (
        f64::from(index % width) + 0.5,
        f64::from(index / width) + 0.5,
    )
}

/// `corners` in `f32`, for [`quad_triangle`].
#[expect(
    clippy::cast_possible_truncation,
    reason = "pixels on a small viewport, as a shading pass holds them"
)]
fn narrow_corners(corners: [(f64, f64); 4]) -> [[f32; 2]; 4] {
    corners.map(|(x, y)| [x as f32, y as f32])
}

#[test]
fn every_visibility_id_decodes_to_the_node_cluster_and_quad_drawn_there() {
    GpuTest::run(|gpu| {
        let world = World::new(gpu);
        let mut rng = Rng(5);
        let mut raster = still_raster(gpu, &world);
        let mut checked = 0;
        for _ in 0..2 {
            let (view, _) = random_view(&mut rng, (2.0, 3.0));
            let cut = random_cut(&mut rng, &world, 3);
            let clusters = cut_clusters(&world, &cut, true);
            let frame = RasterFrame {
                view: &view,
                time: Duration::ZERO,
                moving: false,
            };
            let image = draw_cut(gpu, &mut raster, &world, &cut, &clusters, &frame);
            let sampled = image.iter().enumerate().step_by(7);
            for (index, &pixel) in sampled {
                let Some(id) = VisibilityId::decode(pixel) else {
                    continue;
                };
                assert_eq!(id.encode(), pixel, "round trip");
                let place = world.place_of(id.node);
                assert!(cut.contains(&place), "{place:?} is in the cut");
                assert!(id.quad < 64, "a quad of the cluster");
                // The quad is the first vertex's: were it another's, the id would name a
                // neighbouring quad, which doesn't hold the pixel.
                let centre = pixel_centre(index);
                let quad = named_quad(&world, &clusters, id, &view);
                assert!(
                    (0..2).any(|which| covers(quad_half(quad, which), centre, 0.01)),
                    "pixel {centre:?}: {id:?} of {place:?} is {quad:?}"
                );
                checked += 1;
            }
        }
        assert!(checked > 2000, "{checked} pixels checked");
    });
}

#[test]
fn the_diagonal_test_picks_the_triangle_the_rasterizer_drew() {
    GpuTest::run(|gpu| {
        let world = World::new(gpu);
        let mut rng = Rng(9);
        let mut full = still_raster(gpu, &world);
        // The same raster with every quad's second triangle degenerate: a pixel keeps its id
        // only where the first triangle drew it (nothing faces the eye behind a convex sphere).
        let mut first_only = still_raster(gpu, &world);
        let mut halves = indices::indices();
        for quad in halves.chunks_mut(6) {
            let p00 = quad[0];
            quad[3..].fill(p00);
        }
        first_only.indices = gpu.device().create_buffer_init(&BufferInitDescriptor {
            label: Some("first triangles"),
            contents: bytemuck::cast_slice(&halves),
            usage: BufferUsages::INDEX,
        });
        let (mut checked, mut seconds) = (0, 0);
        for _ in 0..2 {
            let (view, _) = random_view(&mut rng, (2.0, 3.0));
            let cut = random_cut(&mut rng, &world, 3);
            let clusters = cut_clusters(&world, &cut, true);
            let frame = RasterFrame {
                view: &view,
                time: Duration::ZERO,
                moving: false,
            };
            let image = draw_cut(gpu, &mut full, &world, &cut, &clusters, &frame);
            let firsts = draw_cut(gpu, &mut first_only, &world, &cut, &clusters, &frame);
            let pixels = image.iter().zip(&firsts).enumerate();
            for (index, (&pixel, &first)) in pixels {
                let Some(id) = VisibilityId::decode(pixel) else {
                    continue;
                };
                let drew = u32::from(first != pixel);
                let centre = pixel_centre(index);
                let quad = named_quad(&world, &clusters, id, &view);
                // Pixels within a hundredth of a pixel of the diagonal go by the rasterizer's
                // tie rule, which the side test needn't follow.
                let [p00, _, p11, _] = quad;
                let (dx, dy) = (p11.0 - p00.0, p11.1 - p00.1);
                let off = (dx * (centre.1 - p00.1) - dy * (centre.0 - p00.0)) / dx.hypot(dy);
                if off.abs() < 0.01 {
                    continue;
                }
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "a pixel centre on a small viewport"
                )]
                let at = [centre.0 as f32, centre.1 as f32];
                let picked = quad_triangle(narrow_corners(quad), at);
                assert_eq!(picked, drew, "pixel {centre:?} of {id:?}");
                checked += 1;
                seconds += drew;
            }
        }
        assert!(checked > 20_000, "{checked} pixels checked");
        assert!(
            seconds > checked / 4,
            "both triangles drawn: {seconds} of {checked}"
        );
    });
}

/// The WGSL decode run over `cases` (corners and pixel): its answer for each.
fn quad_triangle_on_gpu(gpu: &Gpu, cases: &[[[f32; 2]; 5]]) -> Vec<u32> {
    let source = format!(
        "{QUAD_WGSL}
@group(0) @binding(0) var<storage, read> cases: array<array<vec2<f32>, 5>>;
@group(0) @binding(1) var<storage, read_write> picked: array<u32>;
@compute @workgroup_size(64)
fn decode(@builtin(global_invocation_id) id: vec3<u32>) {{
    if id.x < arrayLength(&cases) {{
        let c = cases[id.x];
        picked[id.x] = quad_triangle(c[0], c[1], c[2], c[3], c[4]);
    }}
}}
"
    );
    let pipeline = gpu
        .compute_pipeline("quad decode", ShaderSource::Wgsl(source.into()), "decode")
        .expect("the decode");
    let device = gpu.device();
    let input = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cases"),
        contents: bytemuck::cast_slice(cases),
        usage: BufferUsages::STORAGE,
    });
    let output = storage_buffer(gpu, "picked", cases.len() as u64 * 4).expect("a buffer");
    let group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("quad decode"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[entry(0, &input), entry(1, &output)],
    });
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        let groups = u32::try_from(cases.len()).expect("small").div_ceil(64);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    gpu.queue().submit([encoder.finish()]);
    finish(gpu);
    gpu.read_buffer(&output).expect("read back")
}

#[test]
fn the_shading_pass_s_diagonal_test_answers_as_the_cpu_s() {
    GpuTest::run(|gpu| {
        let mut rng = Rng(13);
        let mut cases = Vec::new();
        for k in 0..4096 {
            let mut point = || [rng.range(-50.0, 50.0), rng.range(-50.0, 50.0)];
            let (p00, mut p10, p11, mut p01) = (point(), point(), point(), point());
            // Collapsed triangles too, as the edge variants make them.
            if k % 7 == 0 {
                p10 = p11;
            }
            if k % 11 == 0 {
                p01 = p00;
            }
            cases.push([p00, p10, p11, p01, point()]);
        }
        let picked = quad_triangle_on_gpu(gpu, &cases);
        let answers = cases.iter().zip(&picked);
        for (case, gpu_pick) in answers {
            let [p00, p10, p11, p01, pixel] = *case;
            assert_eq!(
                *gpu_pick,
                quad_triangle([p00, p10, p11, p01], pixel),
                "{case:?}"
            );
        }
    });
}

#[test]
fn with_the_camera_still_the_ids_are_the_same_frame_after_frame() {
    GpuTest::run(|gpu| {
        let world = World::new(gpu);
        let mut rng = Rng(11);
        let (mut lod, requests) = level_choice(gpu, &world);
        let mut raster = Raster::new(
            gpu,
            world.tree.capacity(),
            world.layout(),
            RasterConfig::default(),
        )
        .expect("the raster");
        let (view, _) = random_view(&mut rng, (1.8, 3.0));
        let at = |ms: u64, moving: bool| RasterFrame {
            view: &view,
            time: Duration::from_millis(ms),
            moving,
        };
        let (first, _, _) = lod_frame(
            gpu,
            &world,
            &mut lod,
            &requests,
            &mut raster,
            &at(1000, true),
            true,
        );
        let (second, _, _) = lod_frame(
            gpu,
            &world,
            &mut lod,
            &requests,
            &mut raster,
            &at(1016, false),
            false,
        );
        let (third, _, _) = lod_frame(
            gpu,
            &world,
            &mut lod,
            &requests,
            &mut raster,
            &at(1033, false),
            false,
        );
        let drawn = first.iter().filter(|&&id| id != BACKGROUND).count();
        assert!(drawn > 10_000, "{drawn} pixels drawn");
        assert!(
            first == second && second == third,
            "the same ids, frame after frame"
        );
    });
}

/// The probe, appended to the raster's shader in tests only: `probes[k]` is (cluster instance,
/// grid vertex), and `probed[k]` gets where the vertex shader puts that vertex, `w` its node.
const PROBE_WGSL: &str = r"
// The probes: (cluster instance, grid vertex).
@group(2) @binding(0) var<storage, read> probes: array<vec2<u32>>;
// Where each probe's vertex is drawn, w its node.
@group(2) @binding(1) var<storage, read_write> probed: array<vec4<f32>>;

// Runs cluster_point for each probe.
@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) id: vec3<u32>) {
    let k = id.x;
    if k >= arrayLength(&probes) {
        return;
    }
    let placed = cluster_point(probes[k].x, probes[k].y);
    probed[k] = vec4<f32>(placed.position, f32(placed.id >> NODE_SHIFT));
}
";

/// Where the vertex shader puts each `(cluster, grid vertex)` of `probes`, from the list
/// `clusters`, with the raster's frame as last begun and stamped: its position and node.
fn probe(
    gpu: &Gpu,
    raster: &Raster,
    world: &World,
    clusters: &Buffer,
    probes: &[[u32; 2]],
) -> Vec<[f32; 4]> {
    let device = gpu.device();
    let slot_pipeline = &raster.pipeline;
    let probe_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("probe"),
        entries: &[read_entry(0), written_entry(1)],
    });
    let source = format!("{}{PROBE_WGSL}", slot_pipeline.source);
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("probe"),
        source: ShaderSource::Wgsl(source.into()),
    });
    let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("probe"),
        bind_group_layouts: &[
            Some(&raster.layout),
            Some(&slot_pipeline.slot_layout),
            Some(&probe_layout),
        ],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("probe"),
        layout: Some(&layout),
        module: &module,
        entry_point: Some("probe"),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });
    let input = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("probes"),
        contents: bytemuck::cast_slice(probes),
        usage: BufferUsages::STORAGE,
    });
    let output = storage_buffer(gpu, "probed", probes.len() as u64 * 16).expect("a buffer");
    let draw = ClusterDraw {
        tree: world.tree.buffer(),
        slots: &world.slots,
        clusters,
        args: (clusters, 0),
    };
    let (group, slot_group) = raster.groups(gpu, &draw).expect("bind groups");
    let probe_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("probe"),
        layout: &probe_layout,
        entries: &[entry(0, &input), entry(1, &output)],
    });
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.set_bind_group(1, &slot_group, &[]);
        pass.set_bind_group(2, &probe_group, &[]);
        pass.dispatch_workgroups(
            u32::try_from(probes.len()).expect("small").div_ceil(64),
            1,
            1,
        );
    }
    gpu.queue().submit([encoder.finish()]);
    finish(gpu);
    gpu.read_buffer(&output).expect("read back")
}

/// The geomorph tests' case: every face at level 3 but one at level 4, seen from 3 away, and
/// probes of grid vertices of that level-4 tile's first cluster.
struct MorphCase {
    /// The view.
    view: View,
    /// Every face at level 3.
    base: BTreeSet<Place>,
    /// The +z face at level 4 instead.
    cut: BTreeSet<Place>,
    /// The cut's clusters.
    clusters: Vec<GpuCluster>,
    /// The same on the GPU.
    list: Buffer,
    /// The level-3 tile refined.
    refined: Place,
    /// Its child, the level-4 tile probed.
    tile: Place,
    /// (instance, grid vertex) of each probe: an even vertex, one odd in i, one odd in both (on
    /// the diagonal), one odd in j, and an odd one on the tile's bottom edge, beside a coarser
    /// face, which the variant replaces by its even neighbour.
    probes: Vec<[u32; 2]>,
    /// Where each probe starts a morph: the even vertex, the three midpoints, the even
    /// neighbour.
    starts: [[f32; 3]; 5],
    /// Where each probe ends it: its own position, but the replaced one's neighbour.
    ends: [[f32; 3]; 5],
}

impl MorphCase {
    /// The case on `world`.
    fn new(gpu: &Gpu, world: &World) -> Self {
        let view = View::look_at(
            Vec3::new(0.0, 0.0, 3.0),
            Vec3::default(),
            Vec3::new(0.0, 1.0, 0.0),
            45_f64.to_radians(),
            size(),
            0.01,
        )
        .expect("a view");
        let base: BTreeSet<Place> = world
            .tiles
            .keys()
            .filter(|p| p.level().get() == 3)
            .copied()
            .collect();
        let facing_eye = base
            .iter()
            .find(|p| world.tiles[p].vertices[0].position[2] > 0.5);
        let refined = *facing_eye.expect("a +z face");
        let mut cut = base.clone();
        refine(&mut cut, refined);
        let tile = refined.children()[0].expect("a child");
        let clusters = cut_clusters(world, &cut, true);
        let (list, _) = cluster_buffers(gpu, &clusters);
        let instance = clusters.iter().position(|c| c.node == world.node(tile));
        let instance = u32::try_from(instance.expect("listed")).expect("small");
        let grid = [(2, 2), (3, 2), (3, 3), (2, 3), (3, 0)];
        let probes = grid.iter().map(|&(u, v)| {
            let vertex = GridVertex::new(u, v).expect("on the grid");
            [instance, u32::from(vertex.index())]
        });
        let data = &world.tiles[&tile].vertices;
        let row = tile.level().tile_quads() + 1;
        let at = |i: u32, j: u32| data[usize::try_from(j * row + i).expect("small")].position;
        #[expect(
            clippy::manual_midpoint,
            reason = "the raster's arithmetic, (a + b) · 0.5, bit for bit"
        )]
        let mid = |from: [f32; 3], to: [f32; 3]| [0, 1, 2].map(|k| (from[k] + to[k]) * 0.5);
        Self {
            view,
            base,
            cut,
            clusters,
            list,
            refined,
            tile,
            probes: probes.collect(),
            starts: [
                at(2, 2),
                mid(at(2, 2), at(4, 2)),
                mid(at(2, 2), at(4, 4)),
                mid(at(2, 2), at(2, 4)),
                at(4, 0),
            ],
            ends: [at(2, 2), at(3, 2), at(3, 3), at(2, 3), at(4, 0)],
        }
    }

    /// The frame at `ms` milliseconds, the camera still unless `moving`.
    const fn at(&self, ms: u64, moving: bool) -> RasterFrame<'_> {
        RasterFrame {
            view: &self.view,
            time: Duration::from_millis(ms),
            moving,
        }
    }

    /// Draws the cut in a frame and probes where the vertex shader put the probes.
    fn draw_and_probe(
        &self,
        gpu: &Gpu,
        raster: &mut Raster,
        world: &World,
        frame: &RasterFrame<'_>,
    ) -> Vec<[f32; 3]> {
        draw_cut(gpu, raster, world, &self.cut, &self.clusters, frame);
        let probed = probe(gpu, raster, world, &self.list, &self.probes);
        for p in &probed {
            assert!(
                (p[3] - small(world.node(self.tile))).abs() < 0.5,
                "the tile's node"
            );
        }
        probed.iter().map(|p| [p[0], p[1], p[2]]).collect()
    }
}

#[test]
fn the_geomorph_moves_odd_vertices_from_their_midpoint_to_their_place_over_150_ms() {
    GpuTest::run(|gpu| {
        let world = World::new(gpu);
        let case = MorphCase::new(gpu, &world);
        let config = RasterConfig::default().with_ramp(MorphThreshold::OFF);
        let mut raster =
            Raster::new(gpu, world.tree.capacity(), world.layout(), config).expect("the raster");
        // Frame 1 draws every face at level 3; frame 2 refines one with the camera still, so
        // its time factor starts at 0.
        let base_clusters = cut_clusters(&world, &case.base, true);
        draw_cut(
            gpu,
            &mut raster,
            &world,
            &case.base,
            &base_clusters,
            &case.at(900, false),
        );
        let start = case.draw_and_probe(gpu, &mut raster, &world, &case.at(1000, false));
        assert_eq!(start, case.starts.to_vec(), "at the start: the midpoints");
        let halfway = case.draw_and_probe(gpu, &mut raster, &world, &case.at(1075, false));
        let odd = halfway
            .iter()
            .zip(case.starts.iter().zip(&case.ends))
            .enumerate();
        let middle = odd.take(4).skip(1);
        for (k, (now, (from, to))) in middle {
            let along = |q: [f32; 3]| {
                let product: f32 = (0..3).map(|a| (q[a] - from[a]) * (to[a] - from[a])).sum();
                product
            };
            let fraction = along(*now) / along(*to);
            assert!(
                (fraction - 0.5).abs() < 0.01,
                "vertex {k} half way: {fraction}"
            );
        }
        assert_eq!(halfway[0], case.ends[0], "an even vertex never moves");
        let end = case.draw_and_probe(gpu, &mut raster, &world, &case.at(1150, false));
        assert_eq!(end, case.ends.to_vec(), "at 150 ms: exactly where they are");
        let later = case.draw_and_probe(gpu, &mut raster, &world, &case.at(5000, false));
        assert_eq!(later, case.ends.to_vec(), "and they stay there");
    });
}

/// Sets the bound `e` the tree holds for `place` (its data unchanged).
fn set_error(gpu: &Gpu, world: &mut World, place: Place, error: f32) {
    let tile = &world.tiles[&place];
    let detail = TileDetail {
        error,
        ..tile.detail
    };
    let slot = tile.slot;
    assert!(world.tree.move_to(place, slot, detail).is_some());
    if let Some(stored) = world.tiles.get_mut(&place) {
        stored.detail = detail;
    }
    world.write_tree(gpu);
}

#[test]
fn the_geomorph_by_distance_holds_the_drawn_error_at_the_threshold_till_the_tile_s_own_reaches_it()
{
    GpuTest::run(|gpu| {
        let mut world = World::new(gpu);
        let case = MorphCase::new(gpu, &world);
        let threshold = 1.0_f32;
        let config = RasterConfig::default()
            .with_morph(Duration::ZERO)
            .with_ramp(MorphThreshold::new(threshold).expect("a threshold"));
        let mut raster =
            Raster::new(gpu, world.tree.capacity(), world.layout(), config).expect("the raster");
        // Pixels a length of `e` covers at `p`, seen from the case's eye (0, 0, 3) along -z.
        let ppu = case.view.pixels_per_unit();
        let pixels = |e: f32, p: [f32; 3]| f64::from(e) * ppu / (3.0 - f64::from(p[2]));
        // (parent e, own e): the parent's under the threshold everywhere (no morph); far over
        // it with the tile's own at it or over (done); and in between, where each odd vertex
        // stands at t = (x - r) / (x - y) of its way, x and y the bounds in pixels at it. The
        // camera moves: no time factor.
        let cases = [(1e-7_f32, 0.0_f32), (1.0, 0.5), (0.012, 0.002), (0.02, 0.0)];
        for (parent, own) in cases {
            set_error(gpu, &mut world, case.refined, parent);
            set_error(gpu, &mut world, case.tile, own);
            let placed = case.draw_and_probe(gpu, &mut raster, &world, &case.at(2000, true));
            let vertices = placed.iter().enumerate();
            for (k, at) in vertices {
                let (from, to) = (case.starts[k], case.ends[k]);
                let (x, y) = (pixels(parent, to), pixels(own, to));
                let r = f64::from(threshold);
                let t = if x - y <= 0.0 {
                    if y <= r { 1.0 } else { 0.0 }
                } else {
                    ((x - r) / (x - y)).clamp(0.0, 1.0)
                };
                let expected: [f64; 3] = [0, 1, 2]
                    .map(|a| f64::from(from[a]) + t * (f64::from(to[a]) - f64::from(from[a])));
                let off = (0..3)
                    .map(|a| (f64::from(at[a]) - expected[a]).abs())
                    .fold(0.0, f64::max);
                assert!(
                    off < 1e-6,
                    "e ({parent}, {own}) vertex {k}: t {t}, {at:?} for {expected:?}"
                );
                if t <= 0.0 || t >= 1.0 {
                    let exact = if t <= 0.0 { from } else { to };
                    assert_eq!(
                        *at, exact,
                        "e ({parent}, {own}) vertex {k}: an endpoint, exactly"
                    );
                }
            }
        }
    });
}

#[test]
fn the_draw_draws_exactly_the_clusters_the_level_choice_listed() {
    GpuTest::run(|gpu| {
        let world = World::new(gpu);
        let mut rng = Rng(3);
        let (mut lod, requests) = level_choice(gpu, &world);
        let mut raster = still_raster(gpu, &world);
        for seed in 0..3 {
            let (view, _) = random_view(&mut rng, (1.8, 4.0));
            let frame = RasterFrame {
                view: &view,
                time: Duration::ZERO,
                moving: true,
            };
            let (image, clusters, _) = lod_frame(
                gpu,
                &world,
                &mut lod,
                &requests,
                &mut raster,
                &frame,
                seed == 0,
            );
            let listed: BTreeSet<(u32, u32)> = clusters
                .iter()
                .map(|c| {
                    let place = world.place_of(c.node);
                    let per = (place.level().tile_quads() / 8).max(1);
                    let (cx, cy) = c.position();
                    (c.node, cy * per + cx)
                })
                .collect();
            assert_eq!(
                listed.len(),
                clusters.len(),
                "seed {seed}: no cluster twice"
            );
            let seen: BTreeSet<(u32, u32)> = image
                .iter()
                .filter_map(|&pixel| VisibilityId::decode(pixel))
                .map(|id| (id.node, id.cluster))
                .collect();
            let extra: Vec<_> = seen.difference(&listed).collect();
            assert!(
                extra.is_empty(),
                "seed {seed}: drawn but not listed: {extra:?}"
            );
            // Every listed cluster big on the screen shows: on a convex sphere nothing facing
            // the eye is hidden.
            for c in &clusters {
                let place = world.place_of(c.node);
                let quads = place.level().tile_quads();
                let per = (quads / 8).max(1);
                let (cx, cy) = c.position();
                let tile = &world.tiles[&place];
                let side = quads + 1;
                let q = c.quads();
                let corners = [(0, 0), (q, 0), (q, q), (0, q)].map(|(u, v)| {
                    let (i, j) = (cx * 8 + u, cy * 8 + v);
                    let p = tile.vertices[usize::try_from(j * side + i).expect("small")].position;
                    view.pixel(Vec3::new(f64::from(p[0]), f64::from(p[1]), f64::from(p[2])))
                });
                let corners: Option<Vec<(f64, f64, f64)>> = corners.into_iter().collect();
                let Some(corners) = corners else {
                    continue;
                };
                let inside_view = corners
                    .iter()
                    .all(|&(x, y, _)| (2.0..398.0).contains(&x) && (2.0..298.0).contains(&y));
                let doubled: f64 = (0..4)
                    .map(|k| {
                        let (a, b) = (corners[k], corners[(k + 1) % 4]);
                        a.0 * b.1 - b.0 * a.1
                    })
                    .sum();
                let area = doubled / 2.0;
                // Counter-clockwise on screen with y down: negative signed area faces the eye.
                if inside_view && -area > 40.0 {
                    let key = (c.node, cy * per + cx);
                    assert!(
                        seen.contains(&key),
                        "seed {seed}: {place:?} {key:?} ({area}) shows"
                    );
                }
            }
        }
    });
}

#[test]
fn a_frame_draws_both_phases_lists_watertight_the_second_drawing_what_the_first_hid() {
    GpuTest::run(|gpu| {
        let world = World::new(gpu);
        let (_, requests) = level_choice(gpu, &world);
        let thresholds = Thresholds {
            refine_quad: 1e9,
            coarsen_quad: 1e9,
            ..Thresholds::SPEC
        };
        let config = LodConfig::default().with_thresholds(thresholds);
        let mut passes =
            FramePasses::new(gpu, world.tree.capacity(), world.layout(), config, MORPH)
                .expect("the passes");
        // A half turn between frames (a flick): what comes into view was behind the sphere in
        // the previous frame's depth, so the first phase hides it and the second draws it.
        let look = |eye: Vec3| {
            View::look_at(
                eye,
                Vec3::default(),
                Vec3::new(0.0, 1.0, 0.0),
                45_f64.to_radians(),
                size(),
                0.01,
            )
            .expect("a view")
        };
        let views = [
            look(Vec3::new(0.0, 0.0, 1.6)),
            look(Vec3::new(0.0, 0.0, -1.6)),
        ];
        let mut bounds = BoundsUpdate::since(&world.tree, 0);
        let mut last = None;
        for view in &views {
            gpu.queue().write_buffer(&requests, 0, &[0; 16]);
            let inputs = SceneInputs {
                tree: world.tree.buffer(),
                requests: &requests,
                slots: &world.slots,
                bounds: &bounds,
                view,
                time: Duration::from_secs(1),
                moving: true,
                finest: world.finest,
            };
            let mut encoder = gpu
                .device()
                .create_command_encoder(&CommandEncoderDescriptor::default());
            let readback = passes
                .encode(gpu, &mut encoder, &inputs)
                .expect("encoded")
                .expect("a readback buffer");
            gpu.queue().submit([encoder.finish()]);
            readback.submitted();
            finish(gpu);
            bounds = BoundsUpdate::since(&world.tree, bounds.generation());
            last = passes.lod_mut().collect(world.finest).expect("read");
        }
        let output = last.expect("the second frame's results");
        assert!(
            output.stats.drawn.1 > 0,
            "the second phase drew: {:?}",
            output.stats
        );
        let main = cluster_keys(&world, &listed(gpu, passes.lod(), false));
        let post = cluster_keys(&world, &listed(gpu, passes.lod(), true));
        let image = read_visibility(gpu, passes.raster());
        let seen: BTreeSet<(u32, u32)> = image
            .iter()
            .filter_map(|&pixel| VisibilityId::decode(pixel))
            .map(|id| (id.node, id.cluster))
            .collect();
        let both: BTreeSet<(u32, u32)> = main.union(&post).copied().collect();
        assert!(
            seen.is_subset(&both),
            "every id drawn was listed by a phase"
        );
        assert!(!seen.is_disjoint(&post), "the second phase's clusters show");
        let holes = pinholes(&inside(&reference(gpu, &world, &views[1]), 4), &image);
        assert_eq!(holes, 0, "{holes} pinholes");
    });
}

/// A phase's cluster list as the level choice wrote it: `post` the second phase's.
fn listed(gpu: &Gpu, lod: &Lod, post: bool) -> Vec<GpuCluster> {
    let (args, offset) = lod.draw_args(post);
    let words: Vec<u32> = gpu.read_buffer(args).expect("read back");
    let count = words[usize::try_from(offset / 4).expect("small") + 1];
    let all: Vec<GpuCluster> = gpu.read_buffer(lod.clusters(post)).expect("read back");
    all[..usize::try_from(count).expect("small")].to_vec()
}

/// Each cluster as the visibility buffer names it: its node and its index in its tile.
fn cluster_keys(world: &World, clusters: &[GpuCluster]) -> BTreeSet<(u32, u32)> {
    let keys = clusters.iter().map(|c| {
        let per = (world.place_of(c.node).level().tile_quads() / 8).max(1);
        let (cx, cy) = c.position();
        (c.node, cy * per + cx)
    });
    keys.collect()
}
