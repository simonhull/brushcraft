//! Stand-ins (`crate::standin`) in the synthetic world, every tile resident.
//!
//! A wavy bump edits
//! the finest level across a face border, the finest tiles it changed take their new data, and
//! the pass stands in for every display tile above them. Their seams, their bound, the pinhole
//! test beside stored tiles, the refresh swapped in over them with no morph, and the pass's cost.

use std::sync::mpsc;
use std::time::Instant;

use wgpu::{ComputePassTimestampWrites, QuerySetDescriptor, QueryType};

use super::*;
use crate::standin::{Displacement, RowWrite, StandIn, StandInPass, StandInWork};
use crate::upload::{Batch, StagingRing};

/// Where the bump is centred: on the border between the +x and +y faces (x = y), so it moves
/// face border vertices.
pub(crate) const CENTRE: [f32; 3] = [0.696_310_6, 0.696_310_6, 0.174_077_6];

/// How far the tests' bump reaches from its centre.
const RADIUS: f32 = 0.35;

/// The bumped point of `p`.
///
/// It is pushed out along itself by `amplitude` times a smooth bump of
/// `radius` around [`CENTRE`] times a wave, so its curvature is far above the sphere's and every
/// level's bound is outgrown. A function of `p`'s bits alone, so every copy of a vertex moves
/// alike.
pub(crate) fn bumped(p: [f32; 3], amplitude: f32, radius: f32) -> [f32; 3] {
    let d = [0, 1, 2].map(|a| p[a] - CENTRE[a]);
    let reach = 1.0 - (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) / (radius * radius);
    if reach <= 0.0 {
        return p;
    }
    let wave = (40.0 * (p[0] + 2.0 * p[1] + 3.0 * p[2])).sin();
    let scale = 1.0 + amplitude * reach * reach * wave;
    p.map(|c| c * scale)
}

/// A world's finest grids, by face, row-major: as built, or bumped.
fn grids(world: &World, amplitude: f32, radius: f32) -> BTreeMap<u32, Vec<[f32; 3]>> {
    let n = world.finest.face_quads();
    world
        .mesh
        .faces()
        .map(|face| {
            let grid = (0..=n).flat_map(|y| (0..=n).map(move |x| (x, y)));
            let points = grid.map(|(x, y)| {
                let p = finest_point(&world.mesh, world.finest, face, x, y);
                bumped(p, amplitude, radius)
            });
            (face.index(), points.collect())
        })
        .collect()
}

/// A finest grid's point `(x, y)`.
fn at(grid: &[[f32; 3]], n: u32, x: u32, y: u32) -> [f32; 3] {
    grid[usize::try_from(y * (n + 1) + x).expect("small")]
}

/// The tile at `place`'s block of the finest grids `grids`: every finest vertex under it,
/// `(x0, y0)` its first, `size` finest quads a side.
fn block(world: &World, place: Place) -> (u32, u32, u32) {
    let scale = world.finest.face_quads() / place.level().face_quads();
    let quads = place.level().tile_quads();
    let size = quads * scale;
    (
        u32::from(place.tx()) * size,
        u32::from(place.ty()) * size,
        size,
    )
}

/// A bumped world: what the finest tiles hold after the edit, the finest tiles it changed,
/// and how far it moved a vertex at most.
struct Edit {
    /// The finest grids after it.
    grids: BTreeMap<u32, Vec<[f32; 3]>>,
    /// The finest grids before it.
    before: BTreeMap<u32, Vec<[f32; 3]>>,
    /// The finest tiles it changed: its epoch.
    edited: BTreeSet<Place>,
    /// A bound on how far it moved a finest vertex.
    displacement: Displacement,
}

impl Edit {
    /// The bump of `amplitude` and `radius` on `world` as built.
    fn new(world: &World, amplitude: f32, radius: f32) -> Self {
        let before = grids(world, 0.0, radius);
        let after = grids(world, amplitude, radius);
        let n = world.finest.face_quads();
        let mut worst = 0.0_f64;
        for (face, grid) in &after {
            for (p, q) in grid.iter().zip(&before[face]) {
                let d = [0, 1, 2].map(|a| f64::from(p[a]) - f64::from(q[a]));
                worst = worst.max((d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt());
            }
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "rounded to an f32, then up a step"
        )]
        let displacement = Displacement::try_from((worst as f32).next_up()).expect("finite");
        let finest = world.tiles.keys().filter(|p| p.level() == world.finest);
        let edited = finest
            .filter(|&&place| {
                let (x0, y0, size) = block(world, place);
                let grid = (y0..=y0 + size).flat_map(|y| (x0..=x0 + size).map(move |x| (x, y)));
                let face = place.face().index();
                grid.into_iter().any(|(x, y)| {
                    at(&after[&face], n, x, y).map(f32::to_bits)
                        != at(&before[&face], n, x, y).map(f32::to_bits)
                })
            })
            .copied()
            .collect();
        Self {
            grids: after,
            before,
            edited,
            displacement,
        }
    }

    /// The vertices of the tile at `place` subsampled from the edited grids exactly, as the
    /// refresh's tile has them at every border vertex and a stand-in at every vertex it takes.
    fn exact(&self, world: &World, place: Place) -> Vec<GpuVertex> {
        let (x0, y0, size) = block(world, place);
        let quads = place.level().tile_quads();
        let scale = size / quads;
        let n = world.finest.face_quads();
        let grid = &self.grids[&place.face().index()];
        (0..=quads)
            .flat_map(|j| (0..=quads).map(move |i| (i, j)))
            .map(|(i, j)| {
                let p = at(grid, n, x0 + i * scale, y0 + j * scale);
                GpuVertex {
                    position: p,
                    roughness: 1.0,
                    normal: p,
                    reserved: 0,
                }
            })
            .collect()
    }

    /// The display tiles above the edited finest tiles: the tiles that stand in.
    fn above(&self) -> BTreeSet<Place> {
        let mut above = BTreeSet::new();
        for place in &self.edited {
            let mut at = place.parent();
            while let Some(p) = at {
                above.insert(p);
                at = p.parent();
            }
        }
        above
    }

    /// Whether finest vertex `(x, y)` of `face` lies in an edited finest tile.
    fn in_edited(&self, world: &World, face: FaceId, x: u32, y: u32) -> bool {
        self.edited.iter().any(|&place| {
            let (x0, y0, size) = block(world, place);
            place.face() == face && (x0..=x0 + size).contains(&x) && (y0..=y0 + size).contains(&y)
        })
    }
}

impl World {
    /// Writes `vertices` into the slot of the tile at `place` with `detail`, as an epoch
    /// switching it in place does.
    fn rewrite(&mut self, gpu: &Gpu, place: Place, vertices: Vec<GpuVertex>, detail: TileDetail) {
        let tile = self.tiles.get_mut(&place).expect("resident");
        let slot = tile.slot;
        let buffer =
            &self.slots.classes()[usize::from(slot.class().get())][usize::from(slot.buffer())];
        let offset =
            u64::from(slot.index() * slot.class().vertices()) * size_of::<GpuVertex>() as u64;
        gpu.queue()
            .write_buffer(buffer, offset, bytemuck::cast_slice(&vertices));
        tile.vertices = vertices;
        tile.detail = detail;
        self.tree.move_to(place, slot, detail);
        self.write_tree(gpu);
    }

    /// The vertices the slot of the tile at `place` holds on the GPU.
    fn read_slot(&self, gpu: &Gpu, place: Place) -> Vec<GpuVertex> {
        let slot = self.tiles[&place].slot;
        let buffer =
            &self.slots.classes()[usize::from(slot.class().get())][usize::from(slot.buffer())];
        let all: Vec<GpuVertex> = gpu.read_buffer(buffer).expect("read back");
        let count = usize::try_from(slot.class().vertices()).expect("small");
        let first = usize::try_from(slot.index()).expect("small") * count;
        all[first..first + count].to_vec()
    }
}

/// Gives the edited finest tiles their new data, as the step's epoch does.
fn apply_finest(gpu: &Gpu, world: &mut World, edit: &Edit) {
    for &place in &edit.edited {
        let vertices = edit.exact(world, place);
        world.rewrite(gpu, place, vertices, TileDetail::FINEST);
    }
}

/// Runs `pass` over every display tile above `edit`'s finest tiles, and reads back what it wrote.
///
/// Every tile is resident here; each stand-in has a fresh row. What it wrote, the tiles'
/// vertices and their nodes' bounds and cones, goes into the world's copy. Returns the tiles
/// that stood in and how long the pass took ([`submit_timed`]).
fn stand_in(
    gpu: &Gpu,
    world: &mut World,
    edit: &Edit,
    pass: &mut StandInPass,
) -> (BTreeSet<Place>, Duration) {
    let targets = edit.above();
    let edited: Vec<Place> = edit.edited.iter().copied().collect();
    let mut ring = StagingRing::new(64 << 20, 256 << 20);
    let mut batch = Batch::new(&ring);
    let n = world.finest.face_quads();
    for (index, &place) in edited.iter().enumerate() {
        let (x0, y0, size) = block(world, place);
        let grid = &edit.grids[&place.face().index()];
        let mut words: Vec<u32> = (y0..=y0 + size)
            .flat_map(|y| (x0..=x0 + size).map(move |x| (x, y)))
            .flat_map(|(x, y)| at(grid, n, x, y).map(f32::to_bits))
            .collect();
        // A whole number of 8-byte words, as staging copies need.
        words.push(0);
        let index = u32::try_from(index).expect("small");
        pass.stage_tile(gpu, &mut batch, &mut ring, index, &words)
            .expect("staged");
    }
    let mut stand_ins = Vec::new();
    for (row, &place) in targets.iter().enumerate() {
        let row = u32::try_from(row).expect("small");
        let node = world.node(place);
        let write = RowWrite::Fresh {
            node,
            place,
            stored: world.tiles[&place].detail.error,
            displacement: edit.displacement,
        };
        pass.write_row(gpu, &mut batch, &mut ring, row, write)
            .expect("staged");
        let under = edited.iter().enumerate().filter(|(_, f)| {
            let mut at = f.parent();
            while let Some(p) = at {
                if p == place {
                    return true;
                }
                at = p.parent();
            }
            false
        });
        stand_ins.push(StandIn {
            place,
            slot: world.tiles[&place].slot,
            row,
            under: under
                .map(|(i, &f)| (u32::try_from(i).expect("small"), f))
                .collect(),
        });
    }
    let rows = u32::try_from(targets.len()).expect("small");
    let slots = world.slots.clone();
    let locate = |slot: Slot| {
        let buffer =
            slots.classes()[usize::from(slot.class().get())][usize::from(slot.buffer())].clone();
        let offset = u64::from(slot.index()) * slot.class().slot_bytes();
        Some((buffer, offset))
    };
    let work = pass
        .finish(
            gpu,
            &mut batch,
            &mut ring,
            world.tree.buffer(),
            world.finest,
            &stand_ins,
            locate,
            rows,
        )
        .expect("finished");
    let took = submit_timed(gpu, batch, work.as_ref());
    let nodes: Vec<GpuNode> = gpu.read_buffer(world.tree.buffer()).expect("read back");
    for &place in &targets {
        let vertices = world.read_slot(gpu, place);
        let node = nodes[usize::try_from(world.node(place)).expect("small")];
        let detail = TileDetail {
            stored: None,
            error: node.error,
            cone: FinestNormalCone::from_bits(node.finest_cone),
        };
        let tile = world.tiles.get_mut(&place).expect("resident");
        tile.vertices = vertices;
        tile.detail = detail;
        let slot = tile.slot;
        world.tree.move_to(place, slot, detail);
    }
    world.write_tree(gpu);
    (targets, took)
}

/// Submits `batch`'s uploads with `work` after their copies, and times the pass.
///
/// `work` is encoded where a frame's uploads encode it. Returns how long it took on the GPU:
/// between timestamps where the device has them, else from the submission to the end (the
/// scratch's copies included).
fn submit_timed(gpu: &Gpu, batch: Batch, work: Option<&StandInWork>) -> Duration {
    let (outcomes, _heard) = mpsc::channel();
    // The pass encoded after the uploads' copies, as the frame's uploads encode it, between two
    // timestamps where the device has them.
    let uploads = batch.finish(0, None, None, None, outcomes);
    let encoder = gpu
        .device()
        .create_command_encoder(&CommandEncoderDescriptor::default());
    let device = gpu.device();
    let queries = gpu.timestamp_period().map(|period| {
        let set = device.create_query_set(&QuerySetDescriptor {
            label: Some("stand-in timing"),
            ty: QueryType::Timestamp,
            count: 2,
        });
        let resolved = device.create_buffer(&BufferDescriptor {
            label: Some("stand-in timing"),
            size: 16,
            usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        (set, resolved, period)
    });
    let mark = |marked: &mut CommandEncoder, index: u32| {
        if let Some((set, _, _)) = &queries {
            drop(marked.begin_compute_pass(&ComputePassDescriptor {
                label: Some("stand-in mark"),
                timestamp_writes: Some(ComputePassTimestampWrites {
                    query_set: set,
                    beginning_of_pass_write_index: Some(index),
                    end_of_pass_write_index: None,
                }),
            }));
        }
    };
    let start = Instant::now();
    uploads.submit(gpu.queue(), encoder, |frame_encoder| {
        mark(frame_encoder, 0);
        if let Some(work) = work {
            work.encode(frame_encoder);
        }
        mark(frame_encoder, 1);
        if let Some((set, resolved, _)) = &queries {
            frame_encoder.resolve_query_set(set, 0..2, resolved, 0);
        }
    });
    finish(gpu);
    match &queries {
        Some((_, resolved, period)) => {
            let ticks: Vec<u64> = gpu.read_buffer(resolved).expect("read back");
            #[expect(
                clippy::cast_precision_loss,
                reason = "a few milliseconds of nanosecond ticks, far inside f64's exact range"
            )]
            let ns = ticks[1].saturating_sub(ticks[0]) as f64 * f64::from(*period);
            Duration::from_secs_f64(ns * 1e-9)
        }
        None => start.elapsed(),
    }
}

/// The world bumped by `amplitude`: its finest tiles edited and every display tile above them
/// stood in by a pass made with `patches`. Returns the edit and the tiles that stood in.
fn bumped_world(
    gpu: &Gpu,
    top: u8,
    amplitude: f32,
    patches: &[(&str, &str)],
) -> (World, Edit, BTreeSet<Place>) {
    let mut world = World::with_finest(gpu, top);
    let edit = Edit::new(&world, amplitude, RADIUS);
    apply_finest(gpu, &mut world, &edit);
    let mut pass = StandInPass::with_patches(gpu, 256, 256, patches).expect("the pass");
    let (stood, _) = stand_in(gpu, &mut world, &edit, &mut pass);
    (world, edit, stood)
}

/// The amplitude of the tests' bump: the wave moves vertices up to 2 % of the radius.
const AMPLITUDE: f32 = 0.02;

/// The seams that don't close: what differs, as text.
///
/// Every border vertex of every display tile that doesn't hold the edited finest grid's bits
/// there; and every pair of copies of a face border vertex, in display tiles
/// of one level on two faces, holding different bits.
fn seam_failures(world: &World, edit: &Edit) -> Vec<String> {
    let n = world.finest.face_quads();
    let mut failures = Vec::new();
    for (&place, tile) in &world.tiles {
        if place.level() == world.finest {
            continue;
        }
        let (x0, y0, size) = block(world, place);
        let quads = place.level().tile_quads();
        let scale = size / quads;
        let grid = &edit.grids[&place.face().index()];
        let border = (0..=quads)
            .flat_map(|j| (0..=quads).map(move |i| (i, j)))
            .filter(|&(i, j)| i == 0 || j == 0 || i == quads || j == quads);
        for (i, j) in border {
            let v = tile.vertices[usize::try_from(j * (quads + 1) + i).expect("small")];
            let finest = at(grid, n, x0 + i * scale, y0 + j * scale);
            if v.position.map(f32::to_bits) != finest.map(f32::to_bits) {
                failures.push(format!(
                    "{place:?} ({i}, {j}) isn't the finest vertex's bits"
                ));
            }
            // Its copies on other faces, at the same level.
            let level = place.level();
            let (fx, fy) = (x0 / scale + i, y0 / scale + j);
            for copy in world.mesh.topology().copies(place.face(), level, fx, fy) {
                if copy.face == place.face() {
                    continue;
                }
                let other =
                    Place::of(&TileKey::display(copy.face, level, copy.tx, copy.ty).expect("key"));
                let row = quads + 1;
                let w = world.tiles[&other].vertices
                    [usize::try_from(copy.j * row + copy.i).expect("small")];
                if w.position.map(f32::to_bits) != v.position.map(f32::to_bits) {
                    failures.push(format!(
                        "{place:?} ({i}, {j}) differs from {other:?}'s copy"
                    ));
                }
            }
        }
    }
    failures
}

#[test]
fn a_stand_in_takes_the_finest_bits_inside_the_edit_and_keeps_its_own_outside() {
    GpuTest::run(|gpu| {
        let (world, edit, stood) = bumped_world(gpu, FINEST, AMPLITUDE, &[]);
        assert!(stood.len() >= 2 * 7, "{} tiles stood in", stood.len());
        let faces: BTreeSet<FaceId> = edit.edited.iter().map(|p| p.face()).collect();
        assert!(
            faces.len() >= 2,
            "the edit crosses a face border: {faces:?}"
        );
        let n = world.finest.face_quads();
        let (mut taken, mut kept) = (0, 0);
        for &place in &stood {
            let (x0, y0, size) = block(&world, place);
            let quads = place.level().tile_quads();
            let scale = size / quads;
            let face = place.face().index();
            let vertices = &world.tiles[&place].vertices;
            for j in 0..=quads {
                for i in 0..=quads {
                    let (x, y) = (x0 + i * scale, y0 + j * scale);
                    let v = vertices[usize::try_from(j * (quads + 1) + i).expect("small")];
                    let expected = if edit.in_edited(&world, place.face(), x, y) {
                        taken += 1;
                        at(&edit.grids[&face], n, x, y)
                    } else {
                        kept += 1;
                        at(&edit.before[&face], n, x, y)
                    };
                    assert_eq!(
                        v.position.map(f32::to_bits),
                        expected.map(f32::to_bits),
                        "{place:?} ({i}, {j})"
                    );
                }
            }
        }
        assert!(taken > 1000 && kept > 1000, "{taken} taken, {kept} kept");
    });
}

#[test]
fn every_seam_beside_a_stand_in_holds_the_same_bits_on_both_sides_across_faces_too() {
    GpuTest::run(|gpu| {
        let (world, edit, _) = bumped_world(gpu, FINEST, AMPLITUDE, &[]);
        let failures = seam_failures(&world, &edit);
        assert!(
            failures.is_empty(),
            "{} failures: {:?}",
            failures.len(),
            &failures[..5.min(failures.len())]
        );
    });
}

#[test]
fn the_seam_test_bites_a_stand_in_that_keeps_its_old_border() {
    GpuTest::run(|gpu| {
        let keep_border = [(
            "slots[at].position = fresh;",
            "let i = lx + a;\n        let j = ly + b;\n        \
             if i > 0u && j > 0u && i + 1u < job.side && j + 1u < job.side {\n            \
             slots[at].position = fresh;\n        }",
        )];
        let (world, edit, _) = bumped_world(gpu, FINEST, AMPLITUDE, &keep_border);
        let failures = seam_failures(&world, &edit);
        assert!(failures.len() > 10, "{} failures", failures.len());
    });
}

/// The largest distance, in `f64`, from a finest vertex under the tile at `place` (the edited
/// grids') to the point of the tile's own triangles at the same place, and the tile's bound.
fn worst_and_bound(world: &World, edit: &Edit, place: Place) -> (f64, f64) {
    let n = world.finest.face_quads();
    let (x0, y0, size) = block(world, place);
    let quads = place.level().tile_quads();
    let scale = size / quads;
    let grid = &edit.grids[&place.face().index()];
    let tile = &world.tiles[&place];
    let vertex = |i: u32, j: u32| {
        let v = tile.vertices[usize::try_from(j * (quads + 1) + i).expect("small")];
        v.position.map(f64::from)
    };
    let mut worst = 0.0_f64;
    for y in 0..=size {
        for x in 0..=size {
            let fine = at(grid, n, x0 + x, y0 + y).map(f64::from);
            let (qx, qy) = ((x / scale).min(quads - 1), (y / scale).min(quads - 1));
            let fx = f64::from(x - qx * scale) / f64::from(scale);
            let fy = f64::from(y - qy * scale) / f64::from(scale);
            let (p00, p10, p11, p01) = (
                vertex(qx, qy),
                vertex(qx + 1, qy),
                vertex(qx + 1, qy + 1),
                vertex(qx, qy + 1),
            );
            let drawn: [f64; 3] = [0, 1, 2].map(|a| {
                if fx >= fy {
                    p00[a] + fx * (p10[a] - p00[a]) + fy * (p11[a] - p10[a])
                } else {
                    p00[a] + fx * (p11[a] - p01[a]) + fy * (p01[a] - p00[a])
                }
            });
            let d = [0, 1, 2].map(|a| fine[a] - drawn[a]);
            worst = worst.max((d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt());
        }
    }
    (worst, f64::from(tile.detail.error))
}

#[test]
fn a_stand_in_s_bound_holds_every_finest_vertex_under_it_and_its_cone_covers_everything() {
    GpuTest::run(|gpu| {
        let (world, edit, stood) = bumped_world(gpu, FINEST, AMPLITUDE, &[]);
        for &place in &stood {
            let (worst, bound) = worst_and_bound(&world, &edit, place);
            assert!(worst <= bound, "{place:?}: {worst} past its bound {bound}");
            let cone = world.tiles[&place].detail.cone;
            assert_eq!(cone, FinestNormalCone::EVERYTHING, "{place:?}");
        }
        // The bump outgrows the stored bounds, so the bound's raise is what holds them.
        let before = World::with_finest(gpu, FINEST);
        let past_stored = stood.iter().filter(|&&place| {
            let (worst, _) = worst_and_bound(&world, &edit, place);
            worst > f64::from(before.tiles[&place].detail.error)
        });
        assert!(
            past_stored.count() >= 3,
            "the bump outgrows few stored bounds"
        );
    });
}

#[test]
fn the_bound_test_bites_a_stand_in_whose_bound_isnt_raised() {
    GpuTest::run(|gpu| {
        let no_raise = [(
            "error = min((error + d + max(d, moved)) * SLACK, F32_MAX);",
            "error = error;",
        )];
        let (world, edit, stood) = bumped_world(gpu, FINEST, AMPLITUDE, &no_raise);
        let broken = stood.iter().filter(|&&place| {
            let (worst, bound) = worst_and_bound(&world, &edit, place);
            worst > bound
        });
        assert!(
            broken.count() >= 3,
            "too few bounds broken without the raise"
        );
    });
}

#[test]
fn the_pinhole_test_holds_with_stand_ins_beside_stored_tiles() {
    GpuTest::run(|gpu| {
        let (world, _, stood) = bumped_world(gpu, FINEST, AMPLITUDE, &[]);
        let mut rng = Rng(73);
        let mut raster = still_raster(gpu, &world);
        let (mut checked, mut beside) = (0, 0);
        for (seed, &low) in LOWS.iter().enumerate() {
            let (view, distance) = random_view(&mut rng, (2.0, 3.2));
            let cut = random_cut(&mut rng, &world, low);
            let clusters = cut_clusters(&world, &cut, true);
            let mask = inside(
                &reference(gpu, &world, &view),
                margin(&world, &cut, &view, distance, &BTreeSet::new()),
            );
            let frame = RasterFrame {
                view: &view,
                time: Duration::ZERO,
                moving: false,
            };
            let image = draw_cut(gpu, &mut raster, &world, &cut, &clusters, &frame);
            let holes = pinholes(&mask, &image);
            assert_eq!(holes, 0, "seed {seed}: {holes} pinholes");
            checked += mask.iter().filter(|&&m| m).count();
            beside += cut.intersection(&stood).count();
        }
        println!("stand-in pinhole test: {checked} pixels, {beside} stand-ins drawn");
        assert!(
            checked > 50_000 && beside > 0,
            "{checked} pixels, {beside} stand-ins"
        );
    });
}

/// The refreshed tile at `place`: the edited grids subsampled exactly, its bound measured.
fn refreshed(world: &World, edit: &Edit, place: Place) -> (Vec<GpuVertex>, TileDetail) {
    let vertices = edit.exact(world, place);
    let n = world.finest.face_quads();
    let scale = n / place.level().face_quads();
    let grid = &edit.grids[&place.face().index()];
    let error = measured_error(place, world.finest, &vertices, grid, scale);
    (
        vertices,
        TileDetail {
            stored: None,
            error,
            cone: FinestNormalCone::EVERYTHING,
        },
    )
}

#[test]
fn the_refresh_swaps_in_over_the_stand_ins_at_once_with_no_crack() {
    GpuTest::run(|gpu| {
        let (mut world, edit, stood) = bumped_world(gpu, FINEST, AMPLITUDE, &[]);
        let face = edit.edited.first().expect("an edit").face();
        let view = facing(&world, face);
        let config = RasterConfig::default().with_ramp(MorphThreshold::OFF);
        let mut raster =
            Raster::new(gpu, world.tree.capacity(), world.layout(), config).expect("the raster");
        // A cut of whole-face tiles at level 3 and 4: stand-ins near the bump, stored ones away
        // from it.
        let cut: BTreeSet<Place> = world
            .tiles
            .keys()
            .filter(|p| p.level().get() == if p.face() == face { 4 } else { 3 })
            .copied()
            .collect();
        assert!(cut.iter().any(|p| stood.contains(p)), "no stand-in drawn");
        let clusters = cut_clusters(&world, &cut, true);
        let at = |ms: u64| RasterFrame {
            view: &view,
            time: Duration::from_millis(ms),
            moving: false,
        };
        // Frame 1: the stand-ins.
        let stand_in_mask = inside(
            &reference(gpu, &world, &view),
            margin(&world, &cut, &view, 2.5, &BTreeSet::new()),
        );
        let stand_in_image = draw_cut(gpu, &mut raster, &world, &cut, &clusters, &at(800));
        assert_eq!(
            pinholes(&stand_in_mask, &stand_in_image),
            0,
            "stand-ins: pinholes"
        );
        // The refresh replaces every stand-in in place: the tiles stay in the cut, at their
        // level, and draw their new data at once.
        for &place in &stood {
            let (vertices, detail) = refreshed(&world, &edit, place);
            world.rewrite(gpu, place, vertices, detail);
        }
        world.write_tree(gpu);
        let mask = inside(
            &reference(gpu, &world, &view),
            margin(&world, &cut, &view, 2.5, &BTreeSet::new()),
        );
        let mut first: Option<Vec<u32>> = None;
        for (frame, ms) in [1000, 1040, 1080, 1120, 1200].into_iter().enumerate() {
            let image = draw_cut(gpu, &mut raster, &world, &cut, &clusters, &at(ms));
            let holes = pinholes(&mask, &image);
            assert_eq!(
                holes, 0,
                "refresh frame {frame} at {ms} ms: {holes} pinholes"
            );
            // No morph: the swap's first frame is already its last.
            if let Some(first) = &first {
                let moved = first.iter().zip(&image).filter(|(a, b)| a != b).count();
                assert_eq!(
                    moved, 0,
                    "refresh frame {frame} at {ms} ms: {moved} pixels moved"
                );
            } else {
                first = Some(image);
            }
        }
    });
}

#[test]
fn the_stand_in_pass_s_cost_for_a_typical_step() {
    GpuTest::run(|gpu| {
        let mut world = World::with_finest(gpu, 9);
        let edit = Edit::new(&world, AMPLITUDE, 0.6);
        apply_finest(gpu, &mut world, &edit);
        let mut pass = StandInPass::new(gpu, 256, 256).expect("the pass");
        // The first run makes the pipelines' device state and the job list; the next are timed.
        let (stood, _) = stand_in(gpu, &mut world, &edit, &mut pass);
        let mut runs: Vec<Duration> = (0..5)
            .map(|_| stand_in(gpu, &mut world, &edit, &mut pass).1)
            .collect();
        runs.sort();
        println!(
            "stand-ins: {} finest tiles edited, {} display tiles stood in; the pass (timestamps: \
             {}): median {:?}, runs {runs:?}",
            edit.edited.len(),
            stood.len(),
            gpu.timestamp_period().is_some(),
            runs[runs.len() / 2],
        );
        assert!(
            edit.edited.len() >= 40,
            "{} finest tiles",
            edit.edited.len()
        );
    });
}
