//! Tests of the level choice on lavapipe, over synthetic tile trees.
//!
//! The tiles lie on a sphere, their bounds computed on the CPU and written straight into the
//! bounds buffer, their `e` a chosen fraction of their quad edge, their finest normals' cones
//! those of the sphere under them.

use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::FRAC_PI_2;

use geom::{BaseMesh, Channel, FinestNormalCone, LayerId, Level, TileKey, cube_sphere};
use wgpu::{
    BufferDescriptor, BufferUsages, CommandEncoderDescriptor, PollType, TextureViewDescriptor,
};

use super::*;
use crate::camera::{Vec3, View};
use crate::cull::reference::{
    CpuPyramid, ShaderErrors, backfacing, bounds_of, cone_cos_sin, depth_texture,
    finest_facing_away, finest_facing_away_with, in_frustum, occluded, projected,
};
use crate::cull::tests::Rng;
use crate::cull::{GpuBounds, PyramidPass};
use crate::gpu::GpuTest;
use crate::residency::{REQUEST_HEADER, SizeClass, Slot};
use crate::surface::Size;
use crate::tree::{GpuNode, NODE_BYTES, NODE_FINEST, TileTree};
use crate::upload::{GpuVertex, TileDetail};

mod orbit;

/// Level `l`.
fn level(l: u8) -> Level {
    Level::new(l).expect("a level")
}

/// The test viewport: 640 × 480.
fn size() -> Size {
    Size::new(640, 480).expect("non-zero")
}

/// The view from `eye` towards `target`, y up (or z, looking along y), 45°, near 0.01.
fn look(eye: Vec3, target: Vec3) -> View {
    View::look_at(
        eye,
        target,
        Vec3::new(0.0, 1.0, 0.0),
        45_f64.to_radians(),
        size(),
        0.01,
    )
    .or_else(|| {
        View::look_at(
            eye,
            target,
            Vec3::new(0.0, 0.0, 1.0),
            45_f64.to_radians(),
            size(),
            0.01,
        )
    })
    .expect("a view")
}

/// Every place of `mesh` at levels `0..=finest`.
fn all_places(mesh: &BaseMesh, finest: Level) -> Vec<Place> {
    let places = (0..=finest.get()).flat_map(|l| {
        let level = level(l);
        let per_side = u16::try_from(level.tiles_per_side()).expect("small");
        mesh.faces().flat_map(move |face| {
            (0..per_side).flat_map(move |ty| {
                (0..per_side).map(move |tx| {
                    Place::of(&TileKey::display(face, level, tx, ty).expect("a key"))
                })
            })
        })
    });
    places.collect()
}

/// `place`'s vertices: its face's corners bilinearly, pushed onto the unit sphere.
fn vertices(mesh: &BaseMesh, place: Place, finest: Level) -> (Vec<GpuVertex>, usize) {
    let key = place.key(finest).expect("a tile");
    let quads = place.level().tile_quads();
    let small = |n: u32| f32::from(u16::try_from(n).expect("a small grid"));
    let face_quads = small(place.level().face_quads());
    let corner = |c: u8| {
        let p = mesh.corner_position(place.face(), c).expect("a corner");
        [p.x, p.y, p.z]
    };
    let [c0, c1, c2, c3] = [corner(0), corner(1), corner(2), corner(3)];
    let side = usize::try_from(quads + 1).expect("small");
    let mut out = Vec::with_capacity(side * side);
    for j in 0..=quads {
        for i in 0..=quads {
            let u = small(u32::from(key.tx()) * quads + i) / face_quads;
            let v = small(u32::from(key.ty()) * quads + j) / face_quads;
            let p: [f32; 3] = [0, 1, 2].map(|a| {
                (1.0 - u) * (1.0 - v) * c0[a]
                    + u * (1.0 - v) * c1[a]
                    + u * v * c2[a]
                    + (1.0 - u) * v * c3[a]
            });
            let length = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            out.push(GpuVertex {
                position: p.map(|x| x / length),
                roughness: 1.0,
                normal: p.map(|x| x / length),
                reserved: 0,
            });
        }
    }
    (out, side)
}

/// The cone of the finest normals under a tile of the test sphere whose vertices are `data`:
/// [`FinestNormalCone::covering`] their normals.
///
/// Every finest normal under the tile lies in it: a tile's edges are straight in its face's
/// bilinear patch (lines of one parameter), so on the sphere they are great-circle arcs between
/// the boundary vertices, and a cone narrower than a hemisphere that holds the boundary holds
/// the region it encloses.
fn sphere_cone(data: &[GpuVertex]) -> FinestNormalCone {
    let normals: Vec<geom::Vec3> = data
        .iter()
        .map(|v| geom::Vec3::new(v.normal[0], v.normal[1], v.normal[2]))
        .collect();
    FinestNormalCone::covering(&normals).expect("a tile has normals")
}

/// A synthetic tile tree: the resident tiles of a sphere, their bounds, their `e` and their
/// finest normals' cones.
struct Scene {
    /// The base mesh.
    mesh: BaseMesh,
    /// The finest level.
    finest: Level,
    /// The tree.
    tree: TileTree,
    /// Every resident place's bounds.
    bounds: BTreeMap<Place, GpuBounds>,
    /// Every resident place's `e` and finest normals' cone.
    details: BTreeMap<Place, TileDetail>,
}

impl Scene {
    /// The tiles of `mesh` up to `finest`, those `keep` keeps resident, each with `e` =
    /// `error(place)` times its longest quad edge and the cone of the sphere's normals under it
    /// ([`sphere_cone`]); [`TileDetail::FINEST`] at the finest level.
    fn new(
        gpu: &Gpu,
        mesh: BaseMesh,
        finest: Level,
        mut keep: impl FnMut(Place) -> bool,
        mut error: impl FnMut(Place) -> f32,
    ) -> Self {
        let kept: Vec<Place> = all_places(&mesh, finest)
            .into_iter()
            .filter(|place| keep(*place))
            .collect();
        let capacity = u32::try_from(kept.len()).expect("small");
        let mut tree = TileTree::new(gpu, &mesh, capacity).expect("a tree");
        let mut bounds = BTreeMap::new();
        let mut details = BTreeMap::new();
        for place in kept {
            let (data, side) = vertices(&mesh, place, finest);
            let b = bounds_of(&data, side);
            let finest_tile = place.level() == finest;
            let detail = if finest_tile {
                TileDetail::FINEST
            } else {
                TileDetail {
                    stored: None,
                    error: error(place) * b.quad_edge,
                    cone: sphere_cone(&data),
                }
            };
            bounds.insert(place, b);
            details.insert(place, detail);
            let flags = if finest_tile { NODE_FINEST } else { 0 };
            tree.insert(
                place,
                Slot::for_tests(SizeClass::of(place.level()), 0),
                detail,
                flags,
            )
            .expect("inserted");
        }
        Self {
            mesh,
            finest,
            tree,
            bounds,
            details,
        }
    }

    /// Gives the resident tile at `place` a new `e` and cone, as an epoch switching it does.
    fn set_detail(&mut self, place: Place, detail: TileDetail) {
        let slot = Slot::for_tests(SizeClass::of(place.level()), 0);
        assert!(
            self.tree.move_to(place, slot, detail).is_some(),
            "{place:?} is resident"
        );
        self.details.insert(place, detail);
    }

    /// `lod.wgsl`'s `refine_facing_away` for the tile at `place`, the sphere grown by `margin`.
    ///
    /// A finest tile by its drawn cone; any other by its finest normals' cone, and only if its
    /// drawn cone faces away too with the sphere grown by `drawn` (0 on the GPU, as drawing
    /// culls it: a tile that is drawn is never skipped for refining).
    fn refine_facing_away(&self, place: Place, eye: [f32; 3], margin: f32, drawn: f32) -> bool {
        let b = &self.bounds[&place];
        if place.level() == self.finest {
            backfacing(b, eye, margin)
        } else {
            finest_facing_away(b, self.details[&place].cone, eye, margin)
                && backfacing(b, eye, drawn)
        }
    }

    /// Writes the tree's changed nodes and every resident tile's bounds (in one write) to the
    /// GPU.
    fn upload(&mut self, gpu: &Gpu, lod: &Lod) {
        for (start, nodes) in self.tree.take_dirty() {
            gpu.queue().write_buffer(
                self.tree.buffer(),
                u64::from(start) * NODE_BYTES,
                bytemuck::cast_slice(&nodes),
            );
        }
        let count = usize::try_from(self.tree.capacity()).expect("small");
        let mut all = vec![GpuBounds::UNKNOWN; count];
        for (place, b) in &self.bounds {
            if let Some(node) = self.tree.node(*place) {
                all[usize::try_from(node).expect("small")] = *b;
            }
        }
        lod.bounds().write(gpu, 0, &all);
    }

    /// Whether `place` is resident.
    fn resident(&self, place: Place) -> bool {
        self.tree.node(place).is_some()
    }
}

/// A request buffer like residency's, holding `capacity` requests.
fn request_buffer(gpu: &Gpu, capacity: u64) -> Buffer {
    gpu.device().create_buffer(&BufferDescriptor {
        label: Some("test requests"),
        size: REQUEST_HEADER + capacity * 16,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// One frame's level choice, read back: `previous` for the first phase's occlusion test,
/// `post` the second phase's pyramid if it runs.
fn run(
    gpu: &Gpu,
    lod: &mut Lod,
    scene: &mut Scene,
    requests: &Buffer,
    view: &View,
    previous: Option<(&View, &DepthPyramid)>,
    post: Option<&DepthPyramid>,
) -> LodOutput {
    scene.upload(gpu, lod);
    gpu.queue().write_buffer(requests, 0, &[0; 16]);
    let mut encoder = gpu
        .device()
        .create_command_encoder(&CommandEncoderDescriptor::default());
    let inputs = MainInputs {
        tree: scene.tree.buffer(),
        requests,
        view,
        previous,
        finest: scene.finest,
        morph: None,
    };
    lod.encode_main(gpu, &mut encoder, &inputs)
        .expect("encoded");
    if let Some(pyramid) = post {
        lod.encode_post(gpu, &mut encoder, pyramid)
            .expect("encoded");
    }
    let readback = lod
        .encode_readback(gpu, &mut encoder, 0)
        .expect("encoded")
        .expect("a readback buffer is free");
    gpu.queue().submit([encoder.finish()]);
    readback.submitted();
    gpu.device()
        .poll(PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .expect("the GPU finishes");
    gpu.check().expect("no GPU error");
    lod.collect(scene.finest)
        .expect("read back")
        .expect("the frame's results arrived")
}

/// The tiles requested: (place, screen error).
fn read_requests(gpu: &Gpu, requests: &Buffer) -> Vec<(Place, f32)> {
    let words: Vec<u32> = gpu.read_buffer(requests).expect("read back");
    let count = usize::try_from(words[0]).expect("small");
    let entries = words[4..].chunks(4).take(count);
    entries
        .map(|e| {
            let place = Place::from_packed(e[0], e[1]).expect("a place");
            (place, f32::from_bits(e[2]))
        })
        .collect()
}

/// The places of a cut.
fn places(output: &LodOutput) -> BTreeSet<Place> {
    output.nodes.iter().map(|(_, place)| *place).collect()
}

/// The children of `place` along its edge `edge`.
fn edge_children(place: Place, edge: usize) -> Vec<Place> {
    let children = place.children();
    let picked: &[usize] = match (children[1].is_some(), edge) {
        (false, _) => &[0],
        (true, 0) => &[0, 1],
        (true, 1) => &[1, 3],
        (true, 2) => &[2, 3],
        (true, _) => &[0, 2],
    };
    picked.iter().filter_map(|&k| children[k]).collect()
}

/// The cut covers every face exactly once, with resident tiles.
fn check_partition(scene: &Scene, output: &LodOutput) {
    let cut = places(output);
    assert_eq!(cut.len(), output.nodes.len(), "no tile twice");
    let finest_per_side = u64::from(scene.finest.tiles_per_side());
    let mut area: BTreeMap<u32, u64> = BTreeMap::new();
    for &place in &cut {
        assert!(scene.resident(place), "{place:?} is resident");
        let mut above = place.parent();
        while let Some(p) = above {
            assert!(!cut.contains(&p), "{place:?} overlaps its ancestor {p:?}");
            above = p.parent();
        }
        let scale = finest_per_side / u64::from(place.level().tiles_per_side());
        *area.entry(place.face().index()).or_default() += scale * scale;
    }
    for face in scene.mesh.faces() {
        assert_eq!(
            area.get(&face.index()).copied().unwrap_or(0),
            finest_per_side * finest_per_side,
            "face {face:?} is covered exactly once"
        );
    }
}

/// Whether `place` is resident and might lie outside `view`'s frustum.
///
/// That is, not decisively inside by the sphere of what it draws.
fn maybe_outside(scene: &Scene, view: &View, place: Place) -> bool {
    scene
        .bounds
        .get(&place)
        .is_some_and(|b| !in_frustum(&view.gpu(), b.center, b.radius * 0.999))
}

/// Neighbours in the cut are at most one level apart across every edge.
///
/// Given the `view` whose off-screen seams are exempt, a seam is skipped where a tile of it
/// (either side, their parents, the finer side's edge children, the cut's tile across and the
/// finer side's ancestors down to its level) might lie outside its frustum.
fn check_neighbours(scene: &Scene, output: &LodOutput, view: Option<&View>) {
    let cut = places(output);
    let topology = scene.mesh.topology();
    for &c in &cut {
        for (edge, m) in c.neighbours(topology).into_iter().enumerate() {
            let Some(m) = m else { continue };
            let mut covering = Some(m);
            while let Some(p) = covering {
                if cut.contains(&p) {
                    break;
                }
                covering = p.parent();
            }
            if let Some(view) = view {
                // The tile of the cut across, and c's ancestors down to its level: a seam is
                // judged at its coarser side's level.
                let coarser_side = covering.into_iter().chain(
                    std::iter::successors(c.parent(), |p| p.parent()).take_while(|p| {
                        covering.is_some_and(|q| p.level().get() >= q.level().get())
                    }),
                );
                let involved = [Some(c), c.parent(), Some(m), m.parent()]
                    .into_iter()
                    .flatten()
                    .chain(m.children().into_iter().flatten())
                    .chain(coarser_side);
                let mut involved = involved;
                if involved.any(|place| maybe_outside(scene, view, place)) {
                    continue;
                }
            }
            if let Some(p) = covering {
                assert!(
                    c.level().get() - p.level().get() <= 1,
                    "{c:?} edge {edge}: {p:?} is two levels coarser"
                );
                continue;
            }
            let back = m.neighbours(topology);
            let edges = back.iter().enumerate().filter(|(_, n)| **n == Some(c));
            for (k, _) in edges {
                for child in edge_children(m, k) {
                    assert!(
                        cut.contains(&child),
                        "{c:?} edge {edge}: {child:?}'s area is two levels finer"
                    );
                }
            }
        }
    }
}

/// The first cluster list is exactly the drawn tiles' clusters, with the right edges marked.
///
/// Every cluster belongs to a drawn tile, every drawn tile has all its clusters, and each
/// cluster's edge variant marks exactly its coarser neighbours, its finer edges exactly its
/// refined ones.
fn check_clusters(gpu: &Gpu, lod: &Lod, scene: &Scene, output: &LodOutput) {
    let (args, offset) = lod.draw_args(false);
    let words: Vec<u32> = gpu.read_buffer(args).expect("read back");
    let at = usize::try_from(offset / 4).expect("small");
    assert_eq!(words[at], CLUSTER_INDICES, "indices per cluster");
    let count = usize::try_from(words[at + 1]).expect("small");
    let clusters: Vec<GpuCluster> = gpu.read_buffer(lod.clusters(false)).expect("read back");
    let cut = places(output);
    let place_of: BTreeMap<u32, Place> = output.nodes.iter().copied().collect();
    let topology = scene.mesh.topology();
    let mut per_tile: BTreeMap<Place, usize> = BTreeMap::new();
    for cluster in &clusters[..count] {
        let place = place_of[&cluster.node];
        *per_tile.entry(place).or_default() += 1;
        let quads = place.level().tile_quads();
        let per = (quads / 8).max(1);
        assert_eq!(cluster.quads(), quads.min(8), "{place:?}: quads a cluster");
        let (cx, cy) = cluster.position();
        let on_edge = [cy == 0, cx == per - 1, cy == per - 1, cx == 0];
        let neighbours = place.neighbours(topology);
        for edge in 0..4 {
            let coarser = neighbours[edge].is_some_and(|m| {
                let mut above = m.parent();
                while let Some(p) = above {
                    if cut.contains(&p) {
                        return true;
                    }
                    above = p.parent();
                }
                false
            });
            assert_eq!(
                cluster.variant() & (1 << edge) != 0,
                on_edge[edge] && coarser,
                "{place:?} cluster ({cx}, {cy}) edge {edge}"
            );
            // Neither coarser nor in the cut itself: refined, its area drawn finer.
            let finer = neighbours[edge].is_some_and(|m| !coarser && !cut.contains(&m));
            assert_eq!(
                cluster.finer() & (1 << edge) != 0,
                on_edge[edge] && finer,
                "{place:?} cluster ({cx}, {cy}) edge {edge}: finer"
            );
        }
    }
    for (place, listed) in &per_tile {
        let per = usize::try_from((place.level().tile_quads() / 8).max(1)).expect("small");
        assert_eq!(*listed, per * per, "{place:?}: every cluster");
    }
    let drawn_first = output.stats.drawn.0;
    assert_eq!(
        per_tile.len(),
        usize::try_from(drawn_first).expect("small"),
        "drawn tiles"
    );
}

/// The capped screen error of `place` on the CPU, if it is decisively in view: its own and every
/// ancestor's projected `e`, the smallest; `None` near a culling boundary.
fn cpu_error(scene: &Scene, view: &View, place: Place) -> Option<f32> {
    let gpu = view.gpu();
    let eye = [gpu.eye[0], gpu.eye[1], gpu.eye[2]];
    let mut error = f32::MAX;
    let mut at = Some(place);
    while let Some(p) = at {
        let b = scene.bounds[&p];
        let e = scene.details[&p].error;
        let wide = in_frustum(&gpu, b.center, (b.radius + e) * 1.001)
            && !scene.refine_facing_away(p, eye, e * 1.001 + b.radius * 0.001, b.radius * 0.001);
        let narrow = in_frustum(&gpu, b.center, (b.radius + e) * 0.999)
            && !scene.refine_facing_away(p, eye, e * 0.999 - b.radius * 0.001, -b.radius * 0.001);
        if wide != narrow {
            return None;
        }
        error = error.min(if wide { projected(&gpu, &b, e, e) } else { 0.0 });
        at = p.parent();
    }
    Some(error)
}

#[test]
fn a_tree_too_small_or_a_pyramid_of_another_size_is_refused() {
    GpuTest::run(|gpu| {
        let mesh = cube_sphere(1).expect("a mesh");
        let small = TileTree::new(gpu, &mesh, 8).expect("a tree");
        let mut lod = Lod::new(gpu, 64, LodConfig::default()).expect("the level choice");
        let requests = request_buffer(gpu, 16);
        let view = look(Vec3::new(0.0, 0.0, 3.0), Vec3::default());
        let mut encoder = gpu
            .device()
            .create_command_encoder(&CommandEncoderDescriptor::default());
        let mut inputs = MainInputs {
            tree: small.buffer(),
            requests: &requests,
            view: &view,
            previous: None,
            finest: level(8),
            morph: None,
        };
        let refused = lod.encode_main(gpu, &mut encoder, &inputs);
        assert!(
            matches!(refused, Err(LodError::TreeTooSmall { nodes: 64, .. })),
            "{refused:?}"
        );
        let tree = TileTree::new(gpu, &mesh, 64).expect("a tree");
        let other =
            DepthPyramid::new(gpu, Size::new(320, 240).expect("non-zero")).expect("a pyramid");
        inputs.tree = tree.buffer();
        inputs.previous = Some((&view, &other));
        let wrong_size = lod.encode_main(gpu, &mut encoder, &inputs);
        assert!(
            matches!(
                wrong_size,
                Err(LodError::PyramidSize {
                    pyramid: (320, 240),
                    view: (640, 480)
                })
            ),
            "{wrong_size:?}"
        );
        inputs.previous = None;
        assert!(
            lod.encode_main(gpu, &mut encoder, &inputs).is_ok(),
            "the right inputs pass"
        );
    });
}

#[test]
fn seams_off_the_screen_are_exempt_from_the_neighbour_rule_and_save_triangles() {
    GpuTest::run(|gpu| {
        let mesh = cube_sphere(1).expect("a mesh");
        let mut scene = Scene::new(gpu, mesh, level(8), |_| true, |_| 0.3);
        let requests = request_buffer(gpu, 1024);
        let mut strict = Lod::new(
            gpu,
            scene.tree.capacity(),
            LodConfig::default().with_offscreen_seams(false),
        )
        .expect("the level choice");
        let mut exempt =
            Lod::new(gpu, scene.tree.capacity(), LodConfig::default()).expect("the level choice");
        let mut saved = 0;
        // Close to the surface, looking along it: much of the sphere is off the screen.
        for (k, height) in [1.02, 1.05, 1.1, 1.3].into_iter().enumerate() {
            let eye = Vec3::new(
                0.2,
                height,
                0.1 * f64::from(u8::try_from(k).expect("small")),
            );
            let view = look(eye, eye + Vec3::new(1.0, -0.2, 0.3));
            let all = run(gpu, &mut strict, &mut scene, &requests, &view, None, None);
            check_partition(&scene, &all);
            check_neighbours(&scene, &all, None);
            let some = run(gpu, &mut exempt, &mut scene, &requests, &view, None, None);
            check_partition(&scene, &some);
            check_neighbours(&scene, &some, Some(&view));
            assert!(
                some.stats.cut_triangles <= all.stats.cut_triangles,
                "{:?} against {:?}",
                some.stats,
                all.stats
            );
            saved += all.stats.cut_triangles - some.stats.cut_triangles;
        }
        assert!(saved > 0, "the exemption saves triangles somewhere");
    });
}

#[test]
fn a_tile_whose_neighbour_is_absent_under_an_off_screen_parent_is_still_refined() {
    GpuTest::run(|gpu| {
        let mesh = cube_sphere(1).expect("a mesh");
        let finest = level(8);
        // Close to the surface, looking along it: the frustum's sides cross the sphere.
        let eye = Vec3::new(0.2, 1.05, 0.1);
        let view = look(eye, eye + Vec3::new(1.0, -0.2, 0.3));
        let frustum = view.gpu();
        let error = 0.3;
        // Residency as requests leave it: a tile's children are resident only if it is in view,
        // so a tile on the screen can have its neighbour across an edge absent, that neighbour's
        // parent outside the frustum.
        let in_view = |place: Place| {
            let (data, side) = vertices(&mesh, place, finest);
            let b = bounds_of(&data, side);
            in_frustum(&frustum, b.center, b.radius + error * b.quad_edge)
        };
        let mut resident: BTreeSet<Place> = BTreeSet::new();
        for place in all_places(&mesh, finest) {
            let keep = place
                .parent()
                .is_none_or(|parent| resident.contains(&parent) && in_view(parent));
            if keep {
                resident.insert(place);
            }
        }
        let topology = mesh.topology();
        let held_apart = resident.iter().any(|&place| {
            let absent = place
                .neighbours(topology)
                .into_iter()
                .flatten()
                .filter(|m| !resident.contains(m));
            let mut absent = absent;
            in_view(place) && absent.any(|m| m.parent().is_some_and(|p| !in_view(p)))
        });
        assert!(
            held_apart,
            "some tile in view has an absent neighbour under an off-screen parent"
        );
        let mut scene = Scene::new(
            gpu,
            mesh,
            finest,
            |place| resident.contains(&place),
            |_| error,
        );
        let config = LodConfig::default()
            .with_relax_passes(MAX_RELAX_PASSES)
            .expect("valid");
        let mut lod = Lod::new(gpu, scene.tree.capacity(), config).expect("the level choice");
        let requests = request_buffer(gpu, 4096);
        // Twice: the second frame sees the first's choice as its memory, as a still camera does.
        for _ in 0..2 {
            let output = run(gpu, &mut lod, &mut scene, &requests, &view, None, None);
            check_partition(&scene, &output);
            check_neighbours(&scene, &output, Some(&view));
            assert_eq!(output.stats.over_pixel, 0, "{:?}", output.stats);
            assert_eq!(output.stats.shortfall, 0, "{:?}", output.stats);
            assert_eq!(output.stats.held_by_neighbours, 0, "{:?}", output.stats);
        }
    });
}

#[test]
fn with_everything_resident_no_tile_in_view_is_left_over_a_pixel() {
    GpuTest::run(|gpu| {
        let mesh = cube_sphere(1).expect("a mesh");
        let mut scene = Scene::new(gpu, mesh, level(8), |_| true, |_| 0.3);
        let config = LodConfig::default()
            .with_relax_passes(MAX_RELAX_PASSES)
            .expect("valid");
        let mut lod = Lod::new(gpu, scene.tree.capacity(), config).expect("the level choice");
        let requests = request_buffer(gpu, 1024);
        let mut rng = Rng(11);
        for _ in 0..12 {
            let direction = Vec3::new(
                f64::from(rng.range(-1.0, 1.0)),
                f64::from(rng.range(-1.0, 1.0)),
                f64::from(rng.range(-1.0, 1.0)),
            )
            .normalized()
            .expect("a direction");
            let eye = direction * f64::from(rng.range(1.05, 6.0));
            let view = look(eye, Vec3::default());
            let output = run(gpu, &mut lod, &mut scene, &requests, &view, None, None);
            check_partition(&scene, &output);
            check_neighbours(&scene, &output, Some(&view));
            check_clusters(gpu, &lod, &scene, &output);
            assert_eq!(output.stats.over_pixel, 0, "{:?}", output.stats);
            assert_eq!(output.stats.shortfall, 0, "settled: {:?}", output.stats);
            assert!(
                output.drawn.is_subset(&output.cut),
                "drawn tiles are in the cut"
            );
            assert!(!output.drawn.is_empty(), "something is in view");
            assert!(
                read_requests(gpu, &requests).is_empty(),
                "nothing is missing"
            );
        }
    });
}

/// The view from `eye` towards `target` as [`look`] makes it, its field narrowed `factor` times:
/// an unmagnified view of the image `look(eye, target).magnified(factor, [0.0, 0.0])` shows.
fn narrowed(eye: Vec3, target: Vec3, factor: f64) -> View {
    let wide = look(eye, target);
    let fov_y = 2.0 * ((45_f64.to_radians() / 2.0).tan() / factor).atan();
    View::look_at(eye, target, wide.up(), fov_y, size(), 0.01).expect("a view")
}

#[test]
fn a_magnified_view_chooses_the_cut_of_the_narrowed_view_of_the_same_image() {
    GpuTest::run(|gpu| {
        let mesh = cube_sphere(1).expect("a mesh");
        let mut scene = Scene::new(gpu, mesh, level(8), |_| true, |_| 0.3);
        let config = LodConfig::default()
            .with_relax_passes(MAX_RELAX_PASSES)
            .expect("valid");
        let mut lod = Lod::new(gpu, scene.tree.capacity(), config).expect("the level choice");
        let requests = request_buffer(gpu, 1024);
        // Eyes just off the sphere, as the app's boundary holds them, looking at its surface
        // (drawn finest, magnified or not); and two far out, whose unmagnified views draw the
        // sphere coarser than their magnified ones must.
        let poses = [
            (Vec3::new(0.0, 0.0, 1.1), Vec3::new(0.0, 0.0, 1.0), 6.0),
            (Vec3::new(0.7, 0.5, 0.6), Vec3::new(0.55, 0.45, 0.5), 20.0),
            (Vec3::new(-1.0, 0.3, 0.4), Vec3::new(-0.9, 0.2, 0.3), 3.0),
            (Vec3::new(0.0, 12.0, 1.5), Vec3::new(0.0, 1.0, 0.1), 4.0),
            (Vec3::new(9.0, 9.0, 9.0), Vec3::new(0.5, 0.6, 0.6), 12.0),
        ];
        let mut finer = 0;
        for (eye, target, factor) in poses {
            let magnified = look(eye, target)
                .magnified(factor, [0.0, 0.0])
                .expect("magnified");
            let narrowed = narrowed(eye, target, factor);
            let ours = run(gpu, &mut lod, &mut scene, &requests, &magnified, None, None);
            let theirs = run(gpu, &mut lod, &mut scene, &requests, &narrowed, None, None);
            assert_eq!(ours.cut, theirs.cut, "{factor}x from {eye:?}: the cut");
            assert_eq!(
                ours.drawn, theirs.drawn,
                "{factor}x from {eye:?}: what is drawn"
            );
            assert_eq!(ours.stats.over_pixel, 0, "{:?}", ours.stats);
            let wide = look(eye, target);
            let unmagnified = run(gpu, &mut lod, &mut scene, &requests, &wide, None, None);
            let deepest = |output: &LodOutput| {
                output
                    .drawn
                    .iter()
                    .map(|key| key.level().get())
                    .max()
                    .unwrap_or(0)
            };
            finer += usize::from(deepest(&ours) > deepest(&unmagnified));
        }
        assert_eq!(
            finer, 2,
            "magnified from far out, the surface in view is drawn finer"
        );
    });
}

#[test]
fn a_view_magnified_off_its_centre_leaves_no_tile_in_view_over_a_pixel() {
    GpuTest::run(|gpu| {
        let mesh = cube_sphere(1).expect("a mesh");
        let mut scene = Scene::new(gpu, mesh, level(8), |_| true, |_| 0.3);
        let config = LodConfig::default()
            .with_relax_passes(MAX_RELAX_PASSES)
            .expect("valid");
        let mut lod = Lod::new(gpu, scene.tree.capacity(), config).expect("the level choice");
        let requests = request_buffer(gpu, 1024);
        let mut rng = Rng(23);
        let mut seen = 0;
        for _ in 0..10 {
            let direction = Vec3::new(
                f64::from(rng.range(-1.0, 1.0)),
                f64::from(rng.range(-1.0, 1.0)),
                f64::from(rng.range(-1.0, 1.0)),
            )
            .normalized()
            .expect("a direction");
            let eye = direction * f64::from(rng.range(2.0, 12.0));
            let factor = f64::from(rng.range(1.5, 30.0));
            // Magnified about a pixel of the sphere's image, off the centre: the image's shift
            // is what keeps that pixel where it was.
            let about = [
                f64::from(rng.range(220.0, 420.0)),
                f64::from(rng.range(160.0, 320.0)),
            ];
            let shift = [
                (1.0 - factor) * (about[0] - 320.0),
                (1.0 - factor) * (about[1] - 240.0),
            ];
            let view = look(eye, Vec3::default())
                .magnified(factor, shift)
                .expect("magnified");
            let output = run(gpu, &mut lod, &mut scene, &requests, &view, None, None);
            check_partition(&scene, &output);
            check_neighbours(&scene, &output, Some(&view));
            assert_eq!(
                output.stats.over_pixel, 0,
                "{factor}x {shift:?}: {:?}",
                output.stats
            );
            assert_eq!(output.stats.shortfall, 0, "settled: {:?}", output.stats);
            assert!(
                output.drawn.is_subset(&output.cut),
                "drawn tiles are in the cut"
            );
            seen += usize::from(!output.drawn.is_empty());
            for place in output.nodes.iter().map(|(_, place)| *place) {
                if let Some(error) = cpu_error(&scene, &view, place) {
                    assert!(error <= 1.0 + 1e-3, "{place:?}: {error} pixels on the CPU");
                }
            }
        }
        assert!(seen > 5, "the sphere is in most views: {seen}");
    });
}

#[test]
fn hysteresis_keeps_the_cut_still_for_a_camera_moving_back_and_forth_across_a_threshold() {
    GpuTest::run(|gpu| {
        let mesh = cube_sphere(1).expect("a mesh");
        let mut scene = Scene::new(gpu, mesh, level(8), |_| true, |_| 0.3);
        let requests = request_buffer(gpu, 1024);
        let direction = Vec3::new(0.3, 0.5, 0.8).normalized().expect("a direction");
        let cuts = |tiles: &mut Scene, thresholds: Thresholds| {
            let config = LodConfig::default().with_thresholds(thresholds);
            let mut lod = Lod::new(gpu, tiles.tree.capacity(), config).expect("the level choice");
            let frames = (0..10).map(|frame| {
                // Back and forth by 10 %: inside the factor of 2 between refining and coarsening.
                let distance = if frame % 2 == 0 { 12.0 } else { 13.2 };
                let view = look(direction * distance, Vec3::default());
                places(&run(gpu, &mut lod, tiles, &requests, &view, None, None))
            });
            frames.collect::<Vec<BTreeSet<Place>>>()
        };
        let with = cuts(&mut scene, Thresholds::SPEC);
        for (frame, cut) in with.iter().enumerate().skip(2) {
            assert_eq!(cut, &with[1], "frame {frame} is frame 1's cut");
        }
        // The control: without the band, the same motion flips tiles every frame.
        let without = Thresholds {
            coarsen_error: Thresholds::SPEC.refine_error,
            coarsen_quad: Thresholds::SPEC.refine_quad,
            ..Thresholds::SPEC
        };
        let flipping = cuts(&mut scene, without);
        assert_ne!(flipping[2], flipping[3], "the motion crosses a threshold");
        assert_eq!(flipping[2], flipping[4], "and comes back");
    });
}

#[test]
fn the_cut_is_complete_with_no_hole_or_overlap_for_random_residency() {
    GpuTest::run(|gpu| {
        let mut rng = Rng(23);
        for round in 0..6 {
            let mesh = cube_sphere(1).expect("a mesh");
            // The floor always, then each finer tile with a chance that falls with its level.
            let mut keep = |place: Place| place.level().get() <= 5 || rng.below(4) != 0;
            let mut scene = Scene::new(gpu, mesh, level(8), &mut keep, |_| 0.3);
            let mut lod = Lod::new(gpu, scene.tree.capacity(), LodConfig::default())
                .expect("the level choice");
            let requests = request_buffer(gpu, 4096);
            for frame in 0..4 {
                let direction = Vec3::new(
                    f64::from(rng.range(-1.0, 1.0)),
                    f64::from(rng.range(-1.0, 1.0)),
                    f64::from(rng.range(-1.0, 1.0)),
                )
                .normalized()
                .expect("a direction");
                let view = look(direction * f64::from(rng.range(1.05, 4.0)), Vec3::default());
                let output = run(gpu, &mut lod, &mut scene, &requests, &view, None, None);
                check_partition(&scene, &output);
                check_neighbours(&scene, &output, Some(&view));
                check_clusters(gpu, &lod, &scene, &output);
                let cut = places(&output);
                let requested = read_requests(gpu, &requests);
                let requested_places: BTreeSet<Place> =
                    requested.iter().map(|(place, _)| *place).collect();
                for (place, error) in &requested {
                    assert!(
                        !scene.resident(*place),
                        "round {round} frame {frame}: {place:?}"
                    );
                    let parent = place.parent().expect("a parent");
                    assert!(cut.contains(&parent), "requested by a tile of the cut");
                    assert!(error.is_finite() && *error >= 0.0, "a screen error");
                }
                // A tile of the cut over a pixel, its children missing, asks for them.
                for &place in &cut {
                    let missing: Vec<Place> = place
                        .children()
                        .into_iter()
                        .flatten()
                        .filter(|child| child.level() <= scene.finest && !scene.resident(*child))
                        .collect();
                    if missing.is_empty() {
                        continue;
                    }
                    if cpu_error(&scene, &view, place).is_some_and(|error| error > 1.1) {
                        for child in missing {
                            assert!(
                                requested_places.contains(&child),
                                "round {round} frame {frame}: {child:?} is requested"
                            );
                        }
                    }
                }
                assert!(
                    output.stats.waiting <= output.stats.over_pixel,
                    "counted among those over a pixel"
                );
            }
        }
    });
}

#[test]
fn an_edge_across_an_open_border_holds_no_neighbour_to_the_rule() {
    GpuTest::run(|gpu| {
        let mesh = cube_sphere(1).expect("a mesh");
        // The spike: face 0 wants the finest level, its neighbours far coarser.
        let spike = |place: Place| if place.face().index() == 0 { 30.0 } else { 0.3 };
        let mut scene = Scene::new(gpu, mesh, level(8), |_| true, spike);
        let requests = request_buffer(gpu, 1024);
        let view = look(Vec3::new(4.0, 0.5, 0.3), Vec3::default());
        let config = LodConfig::default()
            .with_relax_passes(0)
            .expect("valid")
            .with_offscreen_seams(false);
        let mut lod = Lod::new(gpu, scene.tree.capacity(), config).expect("the level choice");
        let closed = run(gpu, &mut lod, &mut scene, &requests, &view, None, None);
        assert!(
            closed.stats.shortfall > 0,
            "a closed mesh holds the spike back"
        );
        // Every edge an open border, as the tree marks one with its `edges` bit clear.
        let open: Vec<GpuNode> = (0..scene.tree.capacity())
            .map(|node| GpuNode {
                edges: 0,
                ..scene.tree.gpu_node(node)
            })
            .collect();
        gpu.queue()
            .write_buffer(scene.tree.buffer(), 0, bytemuck::cast_slice(&open));
        let output = run(gpu, &mut lod, &mut scene, &requests, &view, None, None);
        check_partition(&scene, &output);
        assert_eq!(
            output.stats.shortfall, 0,
            "no neighbour to hold it: {:?}",
            output.stats
        );
        assert!(
            output.stats.cut_triangles > closed.stats.cut_triangles,
            "the spike refines freely"
        );
    });
}

#[test]
fn a_spike_of_detail_is_relaxed_into_neighbours_a_level_apart_or_drawn_coarser() {
    GpuTest::run(|gpu| {
        let mesh = cube_sphere(1).expect("a mesh");
        // Face 0's tiles carry a hundred times the error of the rest: it alone wants the finest
        // level, two or more levels past its neighbours.
        let spike = |place: Place| if place.face().index() == 0 { 30.0 } else { 0.3 };
        let mut scene = Scene::new(gpu, mesh, level(8), |_| true, spike);
        let requests = request_buffer(gpu, 1024);
        let view = look(Vec3::new(4.0, 0.5, 0.3), Vec3::default());
        let relaxed = |tiles: &mut Scene, passes: u32| {
            let config = LodConfig::default()
                .with_relax_passes(passes)
                .expect("valid");
            let mut lod = Lod::new(gpu, tiles.tree.capacity(), config).expect("the level choice");
            run(gpu, &mut lod, tiles, &requests, &view, None, None)
        };
        let settled = relaxed(&mut scene, MAX_RELAX_PASSES);
        check_partition(&scene, &settled);
        check_neighbours(&scene, &settled, Some(&view));
        let used = settled.stats.passes_used();
        assert!(used >= 2, "the spike needs passes: {:?}", settled.stats);
        assert!(
            used < usize::try_from(MAX_RELAX_PASSES).expect("small"),
            "and settles"
        );
        assert_eq!(settled.stats.shortfall, 0, "{:?}", settled.stats);
        // Too few passes: the sweep still keeps neighbours a level apart, drawing the area
        // coarser, and counts the shortfall.
        for passes in [0, 1] {
            let short = relaxed(&mut scene, passes);
            check_partition(&scene, &short);
            check_neighbours(&scene, &short, Some(&view));
            assert!(
                short.stats.shortfall > 0,
                "{passes} passes: {:?}",
                short.stats
            );
            assert!(
                short.stats.cut_triangles < settled.stats.cut_triangles,
                "coarser"
            );
        }
    });
}

#[test]
fn the_triangle_budget_caps_the_cut_and_refines_the_largest_errors_first() {
    GpuTest::run(|gpu| {
        let mesh = cube_sphere(1).expect("a mesh");
        let mut scene = Scene::new(gpu, mesh, level(8), |_| true, |_| 0.3);
        let requests = request_buffer(gpu, 1024);
        let view = look(Vec3::new(0.2, 0.3, 1.3), Vec3::default());
        let unlimited = {
            let mut lod = Lod::new(gpu, scene.tree.capacity(), LodConfig::default())
                .expect("the level choice");
            run(gpu, &mut lod, &mut scene, &requests, &view, None, None)
        };
        let budget = unlimited.stats.cut_triangles / 3;
        let config = LodConfig::default()
            .with_triangle_budget(budget)
            .with_relax_passes(MAX_RELAX_PASSES)
            .expect("valid");
        let mut lod = Lod::new(gpu, scene.tree.capacity(), config).expect("the level choice");
        let capped = run(gpu, &mut lod, &mut scene, &requests, &view, None, None);
        check_partition(&scene, &capped);
        check_neighbours(&scene, &capped, Some(&view));
        // Recounted from the cut on the CPU, not taken from the GPU's own count.
        let recount: u32 = places(&capped)
            .iter()
            .map(|place| 2 * place.level().tile_quads().pow(2))
            .sum();
        assert!(
            recount <= budget,
            "{recount} triangles in the cut: {:?}",
            capped.stats
        );
        assert_eq!(recount, capped.stats.cut_triangles, "the GPU's count");
        assert!(
            capped.stats.refined_triangles <= budget,
            "{:?}",
            capped.stats
        );
        assert!(capped.stats.over_pixel > 0, "the cap bites");
        assert!(capped.stats.budget_limited > 0, "and is counted");
        // Largest errors first: no tile of the cut, its children resident, is much further over
        // than the least of the refined tiles (one histogram bucket, an eighth of an octave),
        // but for those the sweep or the budget took back.
        let cut = places(&capped);
        let mut refined = Vec::new();
        for place in scene.bounds.keys() {
            let inside = cut.iter().any(|c| {
                c.level() > place.level() && {
                    let mut above = c.parent();
                    while let Some(p) = above {
                        if p == *place {
                            break;
                        }
                        above = p.parent();
                    }
                    above == Some(*place)
                }
            });
            if inside && let Some(error) = cpu_error(&scene, &view, *place) {
                refined.push(error);
            }
        }
        let least = refined
            .iter()
            .copied()
            .filter(|e| *e > 1.0)
            .fold(f32::MAX, f32::min);
        let exceptions = capped.stats.shortfall + capped.stats.refused;
        let worse: Vec<f32> = cut
            .iter()
            .filter(|place| {
                place
                    .children()
                    .into_iter()
                    .flatten()
                    .all(|c| c.level() <= scene.finest)
            })
            .filter_map(|place| cpu_error(&scene, &view, *place))
            .filter(|error| *error > least * 1.2)
            .collect();
        assert!(
            worse.len() <= usize::try_from(exceptions).expect("small"),
            "{} tiles of the cut over {least}: {worse:?}, {:?}",
            worse.len(),
            capped.stats
        );
    });
}

/// Random bounds: a sphere somewhere in a 6-unit cube, a cone of any axis and width.
fn random_bounds(rng: &mut Rng) -> GpuBounds {
    let axis = [
        rng.range(-1.0, 1.0),
        rng.range(-1.0, 1.0),
        rng.range(-1.0, 1.0),
    ];
    let length = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    GpuBounds {
        center: [
            rng.range(-3.0, 3.0),
            rng.range(-3.0, 3.0),
            rng.range(-3.0, 3.0),
        ],
        radius: rng.range(0.01, 0.4),
        axis: axis.map(|a| a / length),
        cone_cos: rng.range(-0.3, 0.99),
        quad_edge: 0.01,
        reserved: [0; 3],
    }
}

/// The depth pyramid of a depth image of the test viewport's size, row by row.
fn pyramid_of(gpu: &Gpu, depth: &[f32]) -> DepthPyramid {
    let (width, height) = (size().width(), size().height());
    let texture = depth_texture(gpu, width, height, depth);
    let view = texture.create_view(&TextureViewDescriptor::default());
    let pyramid = DepthPyramid::new(gpu, size()).expect("a pyramid");
    let pass = PyramidPass::new(gpu).expect("the pyramid's passes");
    let mut encoder = gpu
        .device()
        .create_command_encoder(&CommandEncoderDescriptor::default());
    pass.build(gpu, &mut encoder, &view, &pyramid, None)
        .expect("built");
    gpu.queue().submit([encoder.finish()]);
    pyramid
}

#[test]
fn culling_agrees_with_the_cpu_reference_in_both_phases() {
    GpuTest::run(|gpu| {
        // Level 0 only, of 2,400 faces: every face's tile is in the cut, with random bounds.
        let mesh = cube_sphere(20).expect("a mesh");
        let mut scene = Scene::new(gpu, mesh, level(9), |p| p.level().get() == 0, |_| 0.3);
        let mut rng = Rng(5);
        let mut lod =
            Lod::new(gpu, scene.tree.capacity(), LodConfig::default()).expect("the level choice");
        let requests = request_buffer(gpu, 1 << 14);
        let (width, height) = (size().width(), size().height());
        // A near wall over the left half of the image, nothing behind the right.
        let near_wall = |i: u32| {
            if i % width < width / 2 {
                0.01 / 2.0
            } else {
                0.0
            }
        };
        let wall: Vec<f32> = (0..width * height).map(near_wall).collect();
        let far: Vec<f32> = vec![0.0; usize::try_from(width * height).expect("small")];
        let wall_pyramid = pyramid_of(gpu, &wall);
        let far_pyramid = pyramid_of(gpu, &far);
        let cpu_wall = CpuPyramid::new(width, height, &wall);
        let (mut decisive, mut culled, mut hidden) = (0, 0, 0);
        for round in 0..8 {
            let roots: Vec<Place> = scene
                .bounds
                .keys()
                .copied()
                .filter(|p| scene.resident(*p))
                .collect();
            for &place in &roots {
                scene.bounds.insert(place, random_bounds(&mut rng));
            }
            let eye = Vec3::new(
                f64::from(rng.range(-1.0, 1.0)),
                f64::from(rng.range(-1.0, 1.0)),
                5.0,
            );
            let view = look(eye, Vec3::default());
            let gview = view.gpu();
            let e3 = [gview.eye[0], gview.eye[1], gview.eye[2]];
            // Frustum and cone, then the wall in the first phase; the second phase against the
            // wall again (nothing more shows) and against nothing (everything hidden shows).
            for (post, expect_hidden) in [(&wall_pyramid, true), (&far_pyramid, false)] {
                let output = run(
                    gpu,
                    &mut lod,
                    &mut scene,
                    &requests,
                    &view,
                    Some((&view, &wall_pyramid)),
                    Some(post),
                );
                let drawn: BTreeSet<Place> = output.drawn.iter().map(Place::of).collect();
                for &place in &roots {
                    let b = scene.bounds[&place];
                    let shown = |grow: f32| {
                        let r = b.radius * (1.0 + grow);
                        let wider = GpuBounds {
                            radius: r,
                            cone_cos: b.cone_cos - grow,
                            ..b
                        };
                        let visible =
                            in_frustum(&gview, b.center, r) && !backfacing(&wider, e3, 0.0);
                        let hides = expect_hidden && occluded(&gview, &cpu_wall, b.center, r);
                        (visible && !hides, visible && hides)
                    };
                    let (wide, narrow) = (shown(1e-3), shown(-1e-3));
                    if wide != narrow {
                        continue;
                    }
                    decisive += 1;
                    let (expected, behind) = wide;
                    culled += u32::from(!expected);
                    hidden += u32::from(behind);
                    assert_eq!(
                        drawn.contains(&place),
                        expected,
                        "round {round}, post hidden {expect_hidden}: {place:?} {b:?}"
                    );
                }
                assert_eq!(output.stats.overflow, (0, 0, 0), "nothing overflowed");
            }
        }
        // The test means something: most tiles decided, many of them culled, some by the wall.
        assert!(decisive > 30_000, "decisive {decisive}");
        assert!(
            culled > 5_000 && hidden > 500,
            "culled {culled}, hidden {hidden}"
        );
    });
}

/// `v` in `f32`, as the GPU holds it.
fn narrow(v: Vec3) -> geom::Vec3 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "directions go to the cone builder as f32"
    )]
    let narrowed = geom::Vec3::new(v.x as f32, v.y as f32, v.z as f32);
    narrowed
}

/// `v` as bounds hold it.
fn narrow_array(v: Vec3) -> [f32; 3] {
    let n = narrow(v);
    [n.x, n.y, n.z]
}

/// `p` as a camera vector.
fn wide(p: [f32; 3]) -> Vec3 {
    Vec3::new(f64::from(p[0]), f64::from(p[1]), f64::from(p[2]))
}

/// A unit vector perpendicular to the unit vector `axis`.
fn perpendicular(axis: Vec3) -> Vec3 {
    let helper = if axis.x.abs() < 0.9 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    axis.cross(helper).normalized().expect("not parallel")
}

/// A finest-normal cone around `axis`, holding the directions `half_degrees` from it.
///
/// Built by [`FinestNormalCone::covering`] from the axis and four directions on that circle, so
/// its stored half-angle is that or a little more (the axis's quantization, a step's rounding).
fn cone_around(axis: Vec3, half_degrees: f64) -> FinestNormalCone {
    let axis = axis.normalized().expect("a direction");
    let u = perpendicular(axis);
    let w = axis.cross(u);
    let (sin, cos) = half_degrees.to_radians().sin_cos();
    let rim = [u, w, -u, -w].map(|side| axis * cos + side * sin);
    let directions = std::iter::once(axis).chain(rim).map(narrow);
    let normals: Vec<geom::Vec3> = directions.collect();
    FinestNormalCone::covering(&normals).expect("finite normals")
}

/// A one-split cube sphere's level-0 and level-1 tiles, `e` 0.3 of their quad edge: a level-0
/// tile has its one child resident, so it is refined exactly when the level choice wants it.
fn two_levels(gpu: &Gpu) -> Scene {
    let mesh = cube_sphere(1).expect("a mesh");
    Scene::new(gpu, mesh, level(8), |p| p.level().get() <= 1, |_| 0.3)
}

/// The level-0 tile of face 0.
fn lone_root() -> Place {
    Place::of(&TileKey::display(geom::FaceId::new(0), level(0), 0, 0).expect("a key"))
}

/// What one frame's level choice from `eye`, looking at `place`'s centre, does with `place`:
/// whether it refines it, and whether it draws it.
///
/// `place` is a level-0 tile; the level choice is a fresh one, so no earlier frame's choice
/// holds it. A refined tile's child is in the cut in its stead.
fn choice_from(gpu: &Gpu, scene: &mut Scene, place: Place, eye: Vec3) -> (bool, bool) {
    let center = wide(scene.bounds[&place].center);
    let view = look(eye, center);
    let mut lod =
        Lod::new(gpu, scene.tree.capacity(), LodConfig::default()).expect("the level choice");
    let requests = request_buffer(gpu, 1024);
    let output = run(gpu, &mut lod, scene, &requests, &view, None, None);
    let cut = places(&output);
    let refined = !cut.contains(&place);
    let child = place.children()[0].expect("a child");
    assert_eq!(cut.contains(&child), refined, "the child stands in for it");
    let key = place.key(scene.finest).expect("a tile");
    (refined, output.drawn.contains(&key))
}

/// Whether one frame's level choice from `eye` refines `place` ([`choice_from`]).
fn refined_from(gpu: &Gpu, scene: &mut Scene, place: Place, eye: Vec3) -> bool {
    choice_from(gpu, scene, place, eye).0
}

/// The lone root's eye three units out along its centre's direction, its `e`, and `away`: the
/// direction from the eye towards the tile, along which a normal faces away.
fn lone_view(scene: &Scene, place: Place) -> (Vec3, f32, Vec3) {
    let center = wide(scene.bounds[&place].center);
    let out = center.normalized().expect("off the origin");
    (center + out * 3.0, scene.details[&place].error, -out)
}

/// [`lone_view`], with the lone root's own triangles made to face away (its drawn cone 8°
/// around `away`): the drawn cone alone would never refine it.
fn facing_away_drawn(scene: &mut Scene, place: Place) -> (Vec3, f32, Vec3) {
    let (eye, e, away) = lone_view(scene, place);
    let drawn = GpuBounds {
        axis: narrow_array(away),
        cone_cos: 0.99,
        ..scene.bounds[&place]
    };
    scene.bounds.insert(place, drawn);
    assert!(
        backfacing(&drawn, narrow_array(eye), e),
        "its own triangles face away, the sphere grown by e"
    );
    (eye, e, away)
}

#[test]
fn a_silhouette_tile_whose_finest_detail_faces_the_camera_is_refined_though_it_faces_away() {
    GpuTest::run(|gpu| {
        let mut scene = two_levels(gpu);
        let place = lone_root();
        let (eye, e, away) = facing_away_drawn(&mut scene, place);
        let b = scene.bounds[&place];
        let e3 = narrow_array(eye);
        // The finest detail turns past the silhouette: a cone 60° off `away` and 35° wide holds
        // directions 95° from it, which face the eye.
        let (sin, cos) = 60_f64.to_radians().sin_cos();
        let tilted = away * cos + perpendicular(away) * sin;
        let past = cone_around(tilted, 35.0);
        assert!(
            !finest_facing_away(&b, past, e3, e),
            "the CPU reference: some finest normal faces the eye"
        );
        scene.set_detail(
            place,
            TileDetail {
                stored: None,
                error: e,
                cone: past,
            },
        );
        assert!(
            refined_from(gpu, &mut scene, place, eye),
            "refined by the finest normals' cone"
        );
        // Refined by its own triangles' cone instead (8° around `away`), it would not be.
        let drawn = cone_around(away, 8.2);
        assert!(finest_facing_away(&b, drawn, e3, e), "the CPU reference");
        scene.set_detail(
            place,
            TileDetail {
                stored: None,
                error: e,
                cone: drawn,
            },
        );
        assert!(
            !refined_from(gpu, &mut scene, place, eye),
            "the drawn cone alone leaves it coarse"
        );
    });
}

#[test]
fn a_tile_whose_finest_normals_all_face_away_too_is_not_refined() {
    GpuTest::run(|gpu| {
        let mut scene = two_levels(gpu);
        let place = lone_root();
        let (eye, e, away) = facing_away_drawn(&mut scene, place);
        let b = scene.bounds[&place];
        // Finest normals 35° around `away`: every one faces away from every point of the grown
        // sphere.
        let all_away = cone_around(away, 35.0);
        assert!(
            finest_facing_away(&b, all_away, narrow_array(eye), e),
            "the CPU reference: every finest normal faces away"
        );
        scene.set_detail(
            place,
            TileDetail {
                stored: None,
                error: e,
                cone: all_away,
            },
        );
        assert_eq!(
            choice_from(gpu, &mut scene, place, eye),
            (false, false),
            "nothing under it can show, so it is neither refined nor drawn"
        );
    });
}

#[test]
fn a_tile_whose_finest_normals_face_away_but_whose_own_triangles_face_the_eye_is_refined() {
    GpuTest::run(|gpu| {
        let mut scene = two_levels(gpu);
        let place = lone_root();
        // Seen from in front: its own triangles (the sphere's face) face the eye.
        let (eye, e, away) = lone_view(&scene, place);
        let b = scene.bounds[&place];
        let e3 = narrow_array(eye);
        assert!(!backfacing(&b, e3, 0.0), "its own triangles face the eye");
        // Finest normals 35° around `away`: every one faces away, but the tile would be drawn,
        // coarse, at its full error, so it must be refined.
        let all_away = cone_around(away, 35.0);
        assert!(
            finest_facing_away(&b, all_away, e3, e),
            "the CPU reference: every finest normal faces away"
        );
        assert!(
            !scene.refine_facing_away(place, e3, e, 0.0),
            "the CPU reference: not facing away for refining"
        );
        scene.set_detail(
            place,
            TileDetail {
                stored: None,
                error: e,
                cone: all_away,
            },
        );
        assert!(
            refined_from(gpu, &mut scene, place, eye),
            "refined, not drawn coarse past its error"
        );
    });
}

#[test]
fn a_tile_whose_finest_cone_covers_everything_is_refined_whenever_its_error_asks() {
    GpuTest::run(|gpu| {
        let mut scene = two_levels(gpu);
        let place = lone_root();
        let (front, e, away) = lone_view(&scene, place);
        let b = scene.bounds[&place];
        let center = wide(b.center);
        // In front, behind (its own triangles and the sphere's normals under it facing away)
        // and beside it.
        let behind = center + away * 3.0;
        let eyes = [front, behind, center + perpendicular(away) * 3.0];
        let sphere = scene.details[&place];
        let unknown = TileDetail {
            stored: None,
            error: e,
            cone: FinestNormalCone::EVERYTHING,
        };
        for eye in eyes {
            let view = look(eye, center).gpu();
            assert!(
                projected(&view, &b, e, e) > 2.0 * Thresholds::SPEC.refine_error,
                "its error asks for refining from {eye:?}"
            );
            scene.set_detail(place, unknown);
            assert!(
                refined_from(gpu, &mut scene, place, eye),
                "refined from {eye:?}"
            );
        }
        // The control: from behind, the sphere's own cone faces away and leaves it coarse.
        let from_behind = narrow_array(behind);
        assert!(
            backfacing(&b, from_behind, e) && finest_facing_away(&b, sphere.cone, from_behind, e),
            "the CPU reference: from behind, both cones face away"
        );
        scene.set_detail(place, sphere);
        assert!(
            !refined_from(gpu, &mut scene, place, behind),
            "the sphere's cone, from behind"
        );
    });
}

/// A random finest-normal cone: any axis, 0° to 100° wide, or one in ten covering everything.
fn random_cone(rng: &mut Rng) -> FinestNormalCone {
    if rng.below(10) == 0 {
        return FinestNormalCone::EVERYTHING;
    }
    let axis = Vec3::new(
        f64::from(rng.range(-1.0, 1.0)),
        f64::from(rng.range(-1.0, 1.0)),
        f64::from(rng.range(-1.0, 1.0)),
    );
    let axis = axis.normalized().unwrap_or(Vec3::new(0.0, 0.0, 1.0));
    cone_around(axis, f64::from(rng.range(0.0, 100.0)))
}

#[test]
fn refining_by_the_finest_normal_cone_agrees_with_the_cpu_reference() {
    GpuTest::run(|gpu| {
        // 2,400 level-0 tiles, each with its one child resident: a root is refined exactly when
        // the level choice wants it, which with its error well over the threshold is when it is
        // in view and doesn't face away for refining (its finest normals' cone and, as it would
        // otherwise be drawn, its own triangles' cone).
        let mesh = cube_sphere(20).expect("a mesh");
        let mut scene = Scene::new(gpu, mesh, level(9), |p| p.level().get() <= 1, |_| 0.3);
        let roots: Vec<Place> = scene
            .bounds
            .keys()
            .copied()
            .filter(|p| p.level().get() == 0)
            .collect();
        let mut rng = Rng(41);
        let (mut decisive, mut refined, mut away) = (0, 0, 0);
        let (mut past_drawn, mut drawn_kept) = (0, 0);
        for round in 0..6 {
            for &place in &roots {
                scene.bounds.insert(place, random_bounds(&mut rng));
                let detail = TileDetail {
                    stored: None,
                    error: rng.range(0.02, 0.3),
                    cone: random_cone(&mut rng),
                };
                scene.set_detail(place, detail);
            }
            let eye = Vec3::new(
                f64::from(rng.range(-1.0, 1.0)),
                f64::from(rng.range(-1.0, 1.0)),
                5.0,
            );
            let view = look(eye, Vec3::default());
            let gview = view.gpu();
            let e3 = [gview.eye[0], gview.eye[1], gview.eye[2]];
            let mut lod = Lod::new(gpu, scene.tree.capacity(), LodConfig::default())
                .expect("the level choice");
            let requests = request_buffer(gpu, 1 << 14);
            let output = run(gpu, &mut lod, &mut scene, &requests, &view, None, None);
            let cut = places(&output);
            for &place in &roots {
                let b = scene.bounds[&place];
                let TileDetail { error: e, cone, .. } = scene.details[&place];
                if projected(&gview, &b, e, e) < 1.5 * Thresholds::SPEC.refine_error {
                    continue;
                }
                // Decisive only if the answer holds with the sphere a little larger and smaller.
                let slack = 1e-3 * (b.radius + e) + 1e-3;
                let wants = |grow: f32| {
                    let s = grow.signum() * slack;
                    in_frustum(&gview, b.center, (b.radius + e) * (1.0 + grow))
                        && !scene.refine_facing_away(place, e3, e + s, s)
                };
                let (wide, narrow) = (wants(1e-3), wants(-1e-3));
                if wide != narrow {
                    continue;
                }
                decisive += 1;
                refined += u32::from(wide);
                away += u32::from(!wide && in_frustum(&gview, b.center, b.radius + e));
                past_drawn += u32::from(wide && backfacing(&b, e3, e));
                drawn_kept += u32::from(wide && finest_facing_away(&b, cone, e3, e));
                assert_eq!(
                    !cut.contains(&place),
                    wide,
                    "round {round}: {place:?} {b:?} {cone:?} e {e}"
                );
            }
        }
        // The test means something: many decided, refined and left coarse by the cones, some
        // refined although their own triangles face away, and some although their finest
        // normals do (their own triangles face the eye, so they are drawn).
        assert!(decisive > 5_000, "decisive {decisive}");
        assert!(
            refined > 1_000 && away > 50 && past_drawn > 50 && drawn_kept > 50,
            "refined {refined}, facing away {away}, refined past the drawn cone {past_drawn}, \
             kept by the drawn cone {drawn_kept}"
        );
    });
}

/// A uniform `f64` in `[lo, hi)`.
fn uniform(rng: &mut Rng, lo: f64, hi: f64) -> f64 {
    let bits = u32::try_from(rng.next() >> 32).expect("32 bits");
    lo + (hi - lo) * f64::from(bits) / 4_294_967_296.0
}

/// `x` in `f32`, as the GPU holds it.
fn narrow_scalar(x: f64) -> f32 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "bounds go to the GPU as f32"
    )]
    let narrowed = x as f32;
    narrowed
}

/// A random unit vector.
fn random_direction(rng: &mut Rng) -> Vec3 {
    loop {
        let v = Vec3::new(
            uniform(rng, -1.0, 1.0),
            uniform(rng, -1.0, 1.0),
            uniform(rng, -1.0, 1.0),
        );
        if (0.1..=1.0).contains(&v.length()) {
            return v.normalized().expect("long enough");
        }
    }
}

/// A unit vector perpendicular to the unit vector `axis`, at a random turn about it.
fn random_perpendicular(rng: &mut Rng, axis: Vec3) -> Vec3 {
    let u = perpendicular(axis);
    let (sin, cos) = uniform(rng, 0.0, std::f64::consts::TAU).sin_cos();
    u * cos + axis.cross(u) * sin
}

/// One case of the finest-normal cone test, as the GPU holds its inputs.
struct ConeCase {
    /// The tile's bounds (only the sphere matters).
    bounds: GpuBounds,
    /// The tile's finest-normal cone.
    cone: FinestNormalCone,
    /// The eye.
    eye: [f32; 3],
    /// What the sphere is grown by: `e`.
    margin: f32,
}

/// Where a [`cone_case`] lies.
#[derive(Debug, Clone, Copy)]
enum ConeEdge {
    /// Anywhere.
    Random,
    /// θ + α just short of a quarter turn: the sphere's edge grazing the cone's rim.
    QuarterTurn,
    /// A narrow cone pointing almost straight away, the eye almost on the grown sphere:
    /// `d·cos(θ + α)` near `d`, where widening the angle buys almost nothing.
    StraightAway,
}

/// A random cone test, distances from 10⁻³ to 10⁴; a quarter of the spheres are tiny.
///
/// On an edge (`QuarterTurn`, `StraightAway`), the grown sphere's radius is near `d·cos(θ + α)`,
/// the most it can be and still face away, for a half-angle that is half the time the stored
/// one (where facing away truly ends), else one step wider (where the test, which widens by a
/// step, stops saying so).
fn cone_case(rng: &mut Rng, edge: ConeEdge) -> ConeCase {
    let most_steps = if matches!(edge, ConeEdge::StraightAway) {
        4
    } else {
        2100
    };
    let steps = u32::try_from(rng.below(most_steps)).expect("small");
    let axis_bits = u32::try_from(rng.next() & 0xF_FFFF).expect("20 bits");
    let cone = FinestNormalCone::from_bits((steps << 20) | axis_bits);
    let axis = Vec3::from_f32(cone.axis());
    let alpha = f64::from(cone.half_angle());
    let d = 10_f64.powf(uniform(rng, -3.0, 4.0));
    let widened = if rng.below(2) == 0 {
        alpha
    } else {
        alpha + std::f64::consts::PI / 4095.0
    };
    let side = random_perpendicular(rng, axis);
    let towards = |theta: f64| axis * theta.cos() + side * theta.sin();
    let (direction, reach) = match edge {
        ConeEdge::Random => {
            let reach = d * 10_f64.powf(uniform(rng, -6.0, 0.0));
            (random_direction(rng), reach)
        }
        ConeEdge::QuarterTurn => {
            let delta = 10_f64.powf(uniform(rng, -7.0, -0.5));
            let theta = (FRAC_PI_2 - widened - delta).max(0.0);
            let reach = d * (theta + widened).cos() * (1.0 + uniform(rng, -1e-3, 1e-3));
            (towards(theta), reach.max(0.0))
        }
        ConeEdge::StraightAway => {
            let theta = uniform(rng, 0.0, 1e-3);
            let reach = d * (theta + widened).cos() * (1.0 - uniform(rng, 0.0, 3e-5));
            (towards(theta), reach)
        }
    };
    let radius = reach
        * if rng.below(4) == 0 {
            1e-6
        } else {
            uniform(rng, 0.0, 1.0)
        };
    // The eye at a distance from the origin like d's, so f32 rounding of the centre and the
    // eye doesn't swamp where the sphere lies.
    let eye = random_direction(rng) * (d * 10_f64.powf(uniform(rng, -3.0, 0.5)));
    let center = eye + direction * d;
    ConeCase {
        bounds: GpuBounds {
            center: narrow_array(center),
            radius: narrow_scalar(radius),
            ..GpuBounds::UNKNOWN
        },
        cone,
        eye: narrow_array(eye),
        margin: narrow_scalar(reach - radius),
    }
}

/// Checks, in `f64`, that every direction the case's cone holds faces away from every point of
/// its grown sphere, as the case's inputs stand.
///
/// The bound `d·cos(θ + α) − r` is positive (θ the angle from the decoded axis to the centre,
/// α the stored half-angle), and so is
/// `n · (p − eye)` for directions [`FinestNormalCone::contains`] accepts on the cone's rim and
/// inside it, at the sphere's point nearest to facing each and at random points of it. Returns
/// the bound over `d` and how many directions were checked.
fn assert_faces_away(rng: &mut Rng, case: &ConeCase) -> (f64, u32) {
    let center = wide(case.bounds.center);
    let eye = wide(case.eye);
    let v = center - eye;
    let d = v.length();
    let reach = f64::from(case.bounds.radius) + f64::from(case.margin);
    let axis = Vec3::from_f32(case.cone.axis())
        .normalized()
        .expect("a unit axis");
    let alpha = f64::from(case.cone.half_angle());
    let theta = v.cross(axis).length().atan2(v.dot(axis));
    let bound = d * (theta + alpha).min(std::f64::consts::PI).cos() - reach;
    assert!(
        bound > 0.0,
        "the f64 bound {bound:e} (d {d:e}, θ + α {}, r {reach:e})",
        theta + alpha
    );
    let mut checked = 0;
    for _ in 0..12 {
        let side = random_perpendicular(rng, axis);
        let fractions = [1.0 - 1e-7, 1.0 - 1e-4, uniform(rng, 0.0, 1.0)];
        for fraction in fractions {
            let (sin, cos) = (alpha * fraction).sin_cos();
            let n32 = narrow(axis * cos + side * sin);
            if !case.cone.contains(n32) {
                continue;
            }
            let n = Vec3::from_f32(n32).normalized().expect("a direction");
            // The sphere's point where n · (p − eye) is least: the centre less `reach` along n.
            let least = n.dot(v) - reach;
            let anywhere = n.dot(center + random_direction(rng) * reach - eye);
            assert!(
                least > 0.0 && anywhere > 0.0,
                "{n:?} in the cone faces the eye: {least:e}, {anywhere:e}"
            );
            checked += 1;
        }
    }
    (bound / d, checked)
}

/// The cone test checked against the truth in `f64`, not against its twin.
///
/// Whenever it says a tile faces away, as the CPU computes it and with the shader's errors
/// pushed to the bounds `lod.wgsl` states, every direction the cone holds faces away.
///
/// Random cases, and cases on the two edges where the margins matter: θ + α just short of a
/// quarter turn, and a narrow cone pointing straight away with the eye almost on the sphere
/// (where the slack, not the widening, carries it). With the slack below the stated error the
/// second fails; the one-step widening alone isn't needed for these bounds (the slack covers
/// the axis's decode error too), so dropping it, while keeping the slack, passes.
#[test]
fn whenever_the_cone_test_says_facing_away_every_finest_normal_faces_away_in_f64() {
    let mut rng = Rng(97);
    let edges = [
        ConeEdge::Random,
        ConeEdge::QuarterTurn,
        ConeEdge::StraightAway,
    ];
    let (mut away, mut on_edge, mut straight, mut directions) = (0, 0, 0, 0);
    let mut closest = f64::MAX;
    for k in 0..300_000 {
        let edge = edges[k % edges.len()];
        let case = cone_case(&mut rng, edge);
        // The shader's own errors, at the bounds lod.wgsl states, the worst way: its decoded
        // axis turned 2·10⁻⁶ rad off the CPU's, its q over the true value by 6·10⁻⁶·d.
        let axis = Vec3::from_f32(case.cone.axis());
        let turned = axis + random_perpendicular(&mut rng, axis) * 2e-6;
        let shader = ShaderErrors {
            axis: narrow_array(turned.normalized().expect("a direction")),
            overestimate: 6e-6,
        };
        let said = finest_facing_away(&case.bounds, case.cone, case.eye, case.margin);
        let worst = finest_facing_away_with(&case.bounds, case.cone, case.eye, case.margin, shader);
        if !(said || worst) {
            continue;
        }
        let (bound, checked) = assert_faces_away(&mut rng, &case);
        away += 1;
        on_edge += u32::from(bound < 1e-3);
        straight += u32::from(matches!(edge, ConeEdge::StraightAway));
        closest = closest.min(bound);
        directions += checked;
    }
    // The test means something: many cases face away, many of them within 10⁻³·d of the
    // boundary (one step of widening is 7.7·10⁻⁴), some pointing straight away, and most rim
    // directions are ones the cone holds.
    assert!(
        away > 20_000 && on_edge > 2_000 && straight > 1_000 && directions > 20 * away,
        "facing away {away}, on the edge {on_edge}, straight away {straight}, directions \
         {directions}, closest {closest:e}"
    );
}

#[test]
fn the_cone_test_s_cosine_and_sine_are_within_a_millionth_up_to_a_quarter_turn() {
    let steps = 100_000_u32;
    let angles = (0..=steps).map(|k| f64::from(k) / f64::from(steps) * FRAC_PI_2);
    let errors = angles.map(|x| {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the shader's input is an f32"
        )]
        let narrowed = x as f32;
        let exact = f64::from(narrowed);
        let (c, s) = cone_cos_sin(narrowed);
        (f64::from(c) - exact.cos())
            .abs()
            .max((f64::from(s) - exact.sin()).abs())
    });
    let worst = errors.fold(0.0, f64::max);
    assert!(worst < 1e-6, "worst {worst:e}");
}

#[test]
fn a_stage_that_did_not_run_reads_zero_not_the_ticks_an_earlier_frame_left() {
    // Every stage's queries hold ticks, the bounds' an earlier frame's: this frame changed no
    // tile, so its bounds pass never ran.
    let ticks = [100, 160, 200, 210, 300, 340, 400, 401, 500, 520, 600, 630];
    let ran = [
        Stage::Choose,
        Stage::Relax,
        Stage::Sweep,
        Stage::Emit,
        Stage::Post,
    ]
    .into_iter()
    .fold(StagesRan::default(), StagesRan::with);
    let read = LodTimings::from_ticks(&ticks, StageClock { period: 2.0, ran });
    let ns = Duration::from_nanos;
    let wanted = LodTimings {
        bounds: Duration::ZERO,
        choose: ns(20),
        relax: ns(80),
        sweep: ns(2),
        emit: ns(40),
        post: ns(60),
    };
    assert_eq!(read, wanted);
    let only_bounds = StageClock {
        period: 1.0,
        ran: StagesRan::default().with(Stage::Bounds),
    };
    let bounds = LodTimings::from_ticks(&ticks, only_bounds);
    assert_eq!(bounds.bounds, ns(60), "once it runs, its ticks count");
    assert_eq!(bounds.choose, Duration::ZERO, "and only its");
}

/// A random key: any layer, channel and level of a few faces, some far apart, so every field's
/// bits are crossed.
fn random_key(rng: &mut Rng) -> TileKey {
    let level = level(u8::try_from(rng.below(10)).expect("small"));
    let per_side = usize::try_from(level.tiles_per_side()).expect("small");
    let tx = u16::try_from(rng.below(per_side)).expect("small");
    let ty = u16::try_from(rng.below(per_side)).expect("small");
    let layer = LayerId::new(u16::try_from(rng.below(3)).expect("small"));
    let channel = Channel::from_raw(u8::try_from(rng.below(3)).expect("small"));
    let face = geom::FaceId::new([0, 1, 7, 95, 1 << 20][rng.below(5)]);
    TileKey::new(layer, channel, face, level, tx, ty).expect("in range")
}

#[test]
fn keys_at_every_field_s_ends_order_by_their_packed_order_as_by_their_own() {
    let top = level(geom::MAX_LEVEL);
    let below_top = level(geom::MAX_LEVEL - 1);
    let last = |at: Level| u16::try_from(at.tiles_per_side() - 1).expect("small");
    let layers = [LayerId::new(0), LayerId::new(u16::MAX)];
    let channels = [Channel::from_raw(0), Channel::from_raw(u8::MAX)];
    let faces = [geom::FaceId::new(0), geom::FaceId::new(u32::MAX)];
    let mut keys: Vec<TileKey> = Vec::new();
    for layer in layers {
        for channel in channels {
            for face in faces {
                for at in [below_top, top] {
                    for tx in [0, last(at)] {
                        for ty in [0, last(at)] {
                            let key = TileKey::new(layer, channel, face, at, tx, ty);
                            keys.push(key.expect("in range"));
                        }
                    }
                }
            }
        }
    }
    // Every pair, both ways round.
    for a in &keys {
        for b in &keys {
            let packed = key_order(a).cmp(&key_order(b));
            assert_eq!(packed, a.cmp(b), "{a:?} against {b:?}");
        }
    }
}

#[test]
fn keys_sort_by_their_packed_order_as_by_their_own_and_a_cut_reads_back_as_the_same_sets() {
    let mut rng = Rng(41);
    let keys: Vec<TileKey> = (0..4000).map(|_| random_key(&mut rng)).collect();
    for pair in keys.windows(2) {
        let (a, b) = (&pair[0], &pair[1]);
        let packed = key_order(a).cmp(&key_order(b));
        assert_eq!(packed, a.cmp(b), "{a:?} against {b:?}");
    }
    // Duplicates too: every third key twice, drawn in either, both or neither.
    let twice = keys.iter().step_by(3);
    let read: Vec<(TileKey, bool)> = keys
        .iter()
        .chain(twice)
        .map(|&key| (key, rng.below(3) == 0))
        .collect();
    // One at a time, as the sets were once built.
    let mut cut: BTreeSet<TileKey> = BTreeSet::new();
    let mut drawn: BTreeSet<TileKey> = BTreeSet::new();
    for &(key, was_drawn) in &read {
        cut.insert(key);
        if was_drawn {
            drawn.insert(key);
        }
    }
    let tiles = read.iter().map(|&(key, was_drawn)| CutTile {
        order: key_order(&key),
        key,
        drawn: was_drawn,
    });
    assert_eq!(cut_sets(tiles.collect()), (cut, drawn));
}
