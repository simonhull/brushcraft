//! Stand-ins through residency ([`Residency::apply_step`]).
//!
//! A sphere whose stored display tiles lag its finest level, seen from a distance (no finest tile
//! resident). The edited finest tiles go up through the scratch, every resident display tile above
//! them stands in in the frame of the step (and one arriving after it, in its first frame), and the
//! refresh replaces them, morphing in.

use std::convert::Infallible;

use geom::{FaceId, FinestNormalCone};

use super::*;
use crate::raster::tests::standins::bumped;
use crate::raster::tests::{finest_point, measured_error};
use crate::standin::Displacement;
use crate::tree::NODE_MORPH_IN;

/// The sphere's finest grids by face, row-major.
type Grids = BTreeMap<u32, Vec<[f32; 3]>>;

/// The store as residency sees it: display tiles built from the grids as they were at their
/// last refresh, finest tiles from the grids as they are.
struct Sphere {
    /// The mesh.
    mesh: BaseMesh,
    /// The finest level.
    finest: Level,
    /// The grids the display tiles not refreshed are built from.
    stored: Grids,
    /// The grids now.
    current: Grids,
    /// Display tiles refreshed since the edits: built from the grids now.
    refreshed: BTreeSet<Place>,
    /// The next epoch's number.
    seq: u64,
}

impl Sphere {
    /// The sphere to level 7, unedited.
    fn new() -> Self {
        let mesh = cube_sphere(1).expect("a mesh");
        let finest = level(FINEST);
        let n = finest.face_quads();
        let grids: Grids = mesh
            .faces()
            .map(|face| {
                let grid = (0..=n).flat_map(|y| (0..=n).map(move |x| (x, y)));
                let points = grid.map(|(x, y)| finest_point(&mesh, finest, face, x, y));
                (face.index(), points.collect())
            })
            .collect();
        Self {
            mesh,
            finest,
            stored: grids.clone(),
            current: grids,
            refreshed: BTreeSet::new(),
            seq: 0,
        }
    }

    /// The grid point `(x, y)` of `face` in `grids`.
    fn point(&self, grids: &Grids, face: FaceId, x: u32, y: u32) -> [f32; 3] {
        let n = self.finest.face_quads();
        grids[&face.index()][usize::try_from(y * (n + 1) + x).expect("small")]
    }

    /// The tile at `place`'s first finest vertex and its finest quads a side.
    fn block(&self, place: Place) -> (u32, u32, u32) {
        let size = self.finest.face_quads() / place.level().tiles_per_side();
        (
            u32::from(place.tx()) * size,
            u32::from(place.ty()) * size,
            size,
        )
    }

    /// Bumps the finest grids by `amplitude`, and returns the step's epoch (every finest tile
    /// it changed) and its displacement bound.
    fn step(&mut self, amplitude: f32) -> (Epoch, Displacement) {
        let before = self.current.clone();
        for grid in self.current.values_mut() {
            for p in grid.iter_mut() {
                *p = bumped(*p, amplitude, 0.35);
            }
        }
        let mut worst = 0.0_f64;
        for (face, grid) in &self.current {
            for (p, q) in grid.iter().zip(&before[face]) {
                let d = [0, 1, 2].map(|a| f64::from(p[a]) - f64::from(q[a]));
                worst = worst.max((d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt());
            }
        }
        let per_side = u16::try_from(self.finest.tiles_per_side()).expect("small");
        let mut keys = BTreeSet::new();
        for face in self.mesh.faces() {
            for ty in 0..per_side {
                for tx in 0..per_side {
                    let key = TileKey::displacement(face, self.finest, tx, ty).expect("a key");
                    let (x0, y0, size) = self.block(Place::of(&key));
                    let changed = (y0..=y0 + size)
                        .flat_map(|y| (x0..=x0 + size).map(move |x| (x, y)))
                        .any(|(x, y)| {
                            self.point(&self.current, face, x, y).map(f32::to_bits)
                                != self.point(&before, face, x, y).map(f32::to_bits)
                        });
                    if changed {
                        keys.insert(key);
                    }
                }
            }
        }
        self.seq += 1;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "rounded to an f32, then up a step"
        )]
        let displacement = Displacement::new((worst as f32).next_up()).expect("finite");
        (
            Epoch {
                seq: self.seq,
                keys,
            },
            displacement,
        )
    }

    /// Refreshes the display tiles at `places`, and returns the refresh's epoch.
    fn refresh(&mut self, places: &BTreeSet<Place>) -> Epoch {
        self.refreshed.extend(places.iter().copied());
        self.seq += 1;
        let keys = places.iter().filter_map(|p| p.key(self.finest)).collect();
        Epoch {
            seq: self.seq,
            keys,
        }
    }

    /// The grids the tile at `place` is built from.
    fn grids_of(&self, place: Place) -> &Grids {
        if place.level() == self.finest || self.refreshed.contains(&place) {
            &self.current
        } else {
            &self.stored
        }
    }
}

impl TileSource for Sphere {
    type Error = Infallible;

    fn at_hand(&mut self, _key: TileKey) -> bool {
        true
    }

    fn read(&mut self, key: TileKey, out: &mut TileData) -> Result<Option<TileDetail>, Infallible> {
        let place = Place::of(&key);
        let grids = self.grids_of(place);
        let (x0, y0, size) = self.block(place);
        let quads = place.level().tile_quads();
        let scale = size / quads;
        let vertices: Vec<GpuVertex> = (0..=quads)
            .flat_map(|j| (0..=quads).map(move |i| (i, j)))
            .map(|(i, j)| {
                let p = self.point(grids, place.face(), x0 + i * scale, y0 + j * scale);
                GpuVertex {
                    position: p,
                    roughness: 1.0,
                    normal: p,
                    reserved: 0,
                }
            })
            .collect();
        out.vertices().copy_from_slice(&vertices);
        if place.level() == self.finest {
            return Ok(Some(TileDetail::FINEST));
        }
        let grid = &grids[&place.face().index()];
        Ok(Some(TileDetail {
            error: measured_error(place, self.finest, &vertices, grid, scale),
            cone: FinestNormalCone::EVERYTHING,
        }))
    }
}

/// Residency for the sphere with the floor up to `floor_top` and room to spare.
fn residency(gpu: &Gpu, sphere: &Sphere, floor_top: u8) -> Residency {
    residency_with(gpu, sphere, floor_top, 160)
}

/// [`residency`] with a kept scratch of `scratch` finest tiles.
fn residency_with(gpu: &Gpu, sphere: &Sphere, floor_top: u8, scratch: u32) -> Residency {
    let config = ResidencyConfig::new(256 << 20, 16 << 20, 64 << 20, 8 << 20, 64, level(floor_top))
        .expect("a valid configuration")
        .with_stand_ins(scratch, 64);
    Residency::new(gpu, &sphere.mesh, sphere.finest, config).expect("residency")
}

/// One frame: the drain applied (after a step of `displacement`, if one is given), `requests`
/// wanted, staged, submitted and waited for.
fn frame(
    gpu: &Gpu,
    residency: &mut Residency,
    sphere: &mut Sphere,
    drained: Drained,
    displacement: Option<Displacement>,
    requests: Option<&Requests>,
) {
    residency.begin_frame(gpu).expect("a frame begins");
    match displacement {
        Some(displacement) => residency.apply_step(drained, displacement),
        None => residency.apply(drained),
    }
    residency.use_frame(FrameUse::default());
    if let Some(requests) = requests {
        residency.want(requests);
    }
    let deadline = Instant::now() + Duration::from_secs(60);
    let uploads = residency.stage(gpu, sphere, deadline).expect("staged");
    submit(gpu, uploads, |_| {});
}

/// What the GPU holds for each resident display tile: its vertices and its node.
fn display_tiles(gpu: &Gpu, residency: &Residency) -> BTreeMap<Place, (Vec<GpuVertex>, GpuNode)> {
    let nodes: Vec<GpuNode> = gpu
        .read_buffer(residency.tree().buffer())
        .expect("the tree reads back");
    let mut buffers: BTreeMap<(u8, u16), Vec<GpuVertex>> = BTreeMap::new();
    let mut tiles = BTreeMap::new();
    for key in residency.resident() {
        if key.channel() != Channel::DISPLAY {
            continue;
        }
        let place = Place::of(key);
        let slot = residency.slot(key).expect("resident");
        let data = buffers
            .entry((slot.class().get(), slot.buffer()))
            .or_insert_with(|| {
                let buffer = &residency.slot_buffers(slot.class())[usize::from(slot.buffer())];
                gpu.read_buffer(buffer).expect("slots read back")
            });
        let count = usize::try_from(slot.class().vertices()).expect("small");
        let first = usize::try_from(slot.index()).expect("small") * count;
        let node = residency.tree().node(place).expect("a node");
        tiles.insert(
            place,
            (
                data[first..first + count].to_vec(),
                nodes[usize::try_from(node).expect("small")],
            ),
        );
    }
    tiles
}

/// Checks every resident display tile against the sphere now.
///
/// Its border vertices are the finest vertices' bits (so every seam between two of them closes),
/// and every finest vertex under it lies within its node's bound of its triangles. Returns how many
/// tiles' bounds were raised past what their stored data's would be.
fn check(gpu: &Gpu, residency: &Residency, sphere: &Sphere) -> usize {
    let mut raised = 0;
    for (place, (vertices, node)) in display_tiles(gpu, residency) {
        let (x0, y0, size) = sphere.block(place);
        let quads = place.level().tile_quads();
        let scale = size / quads;
        let border = (0..=quads)
            .flat_map(|j| (0..=quads).map(move |i| (i, j)))
            .filter(|&(i, j)| i == 0 || j == 0 || i == quads || j == quads);
        for (i, j) in border {
            let v = vertices[usize::try_from(j * (quads + 1) + i).expect("small")];
            let finest = sphere.point(
                &sphere.current,
                place.face(),
                x0 + i * scale,
                y0 + j * scale,
            );
            assert_eq!(
                v.position.map(f32::to_bits),
                finest.map(f32::to_bits),
                "{place:?} ({i}, {j}) isn't the finest vertex's bits"
            );
        }
        let grid = &sphere.current[&place.face().index()];
        let worst = measured_error(place, sphere.finest, &vertices, grid, scale);
        // `measured_error` adds a little; the bound must hold what it measures, less that.
        let measured = (worst - 1e-6) / 1.01;
        assert!(
            measured <= node.error,
            "{place:?}: a finest vertex {measured} from its triangles, past its bound {}",
            node.error
        );
        if node.finest_cone == FinestNormalCone::EVERYTHING.to_bits()
            && node.error > measured_error(place, sphere.finest, &vertices, grid, scale)
        {
            raised += 1;
        }
    }
    raised
}

/// The display tiles above the finest tiles `epoch` names.
fn above(epoch: &Epoch) -> BTreeSet<Place> {
    let mut above = BTreeSet::new();
    for key in &epoch.keys {
        let mut at = Place::of(key).parent();
        while let Some(place) = at {
            above.insert(place);
            at = place.parent();
        }
    }
    above
}

#[test]
fn two_steps_seen_from_a_distance_stand_in_for_every_display_tile_above_them() {
    GpuTest::run(|gpu| two_steps(gpu, 160));
}

#[test]
fn a_frame_reading_more_finest_tiles_than_the_kept_scratch_makes_a_one_off_one() {
    GpuTest::run(|gpu| two_steps(gpu, 1));
}

/// Two steps seen from a distance, with a kept scratch of `scratch` finest tiles.
///
/// Every display tile above them stands in, its seams closed and its bound a bound, and a frame
/// writing the whole tree again keeps the bounds.
fn two_steps(gpu: &Gpu, scratch: u32) {
    let mut sphere = Sphere::new();
    let mut residency = residency_with(gpu, &sphere, 6, scratch);
    frame(
        gpu,
        &mut residency,
        &mut sphere,
        Drained::Epochs(Vec::new()),
        None,
        None,
    );
    for (step, amplitude) in [0.02, -0.015].into_iter().enumerate() {
        let (epoch, displacement) = sphere.step(amplitude);
        let stood = above(&epoch);
        let drained = Drained::Epochs(vec![epoch]);
        frame(
            gpu,
            &mut residency,
            &mut sphere,
            drained,
            Some(displacement),
            None,
        );
        assert!(
            residency.resident().all(|key| key.level() < sphere.finest),
            "seen from a distance: no finest tile resident"
        );
        let stats = residency.stats();
        assert_eq!(stats.stand_ins_skipped, 0);
        assert_eq!(
            stats.stand_ins,
            (stood.len() * (step + 1)) as u64,
            "step {step}: every display tile above it stood in"
        );
        check(gpu, &residency, &sphere);
    }
    // A frame writing the whole tree again keeps the stand-ins' bounds: the settle writes
    // them over the stored ones the upload brings.
    let before = display_tiles(gpu, &residency);
    residency.tree.mark_all_dirty();
    frame(
        gpu,
        &mut residency,
        &mut sphere,
        Drained::Epochs(Vec::new()),
        None,
        None,
    );
    let after = display_tiles(gpu, &residency);
    for (place, (_, node)) in &after {
        assert_eq!(node.error, before[place].1.error, "{place:?}'s bound kept");
    }
    check(gpu, &residency, &sphere);
}

#[test]
fn the_refresh_replaces_the_stand_ins_and_they_morph_in() {
    GpuTest::run(|gpu| {
        let mut sphere = Sphere::new();
        let mut residency = residency(gpu, &sphere, 6);
        frame(
            gpu,
            &mut residency,
            &mut sphere,
            Drained::Epochs(Vec::new()),
            None,
            None,
        );
        let (epoch, displacement) = sphere.step(0.02);
        let stood = above(&epoch);
        let drained = Drained::Epochs(vec![epoch]);
        frame(
            gpu,
            &mut residency,
            &mut sphere,
            drained,
            Some(displacement),
            None,
        );
        let refresh = sphere.refresh(&stood);
        frame(
            gpu,
            &mut residency,
            &mut sphere,
            Drained::Epochs(vec![refresh]),
            None,
            None,
        );
        let tiles = display_tiles(gpu, &residency);
        for place in &stood {
            let (vertices, node) = &tiles[place];
            assert_ne!(node.flags & NODE_MORPH_IN, 0, "{place:?} morphs in");
            let grid = &sphere.current[&place.face().index()];
            let scale = sphere.block(*place).2 / place.level().tile_quads();
            let refreshed = measured_error(*place, sphere.finest, vertices, grid, scale);
            assert_eq!(
                node.error, refreshed,
                "{place:?} draws the refreshed tile's bound"
            );
        }
        assert!(residency.stand_ins.rows.is_empty(), "every row freed");
        assert!(
            residency.stand_ins.history.is_empty(),
            "the step is caught up with"
        );
        check(gpu, &residency, &sphere);
        frame(
            gpu,
            &mut residency,
            &mut sphere,
            Drained::Epochs(Vec::new()),
            None,
            None,
        );
        let after = display_tiles(gpu, &residency);
        assert!(
            after
                .values()
                .all(|(_, node)| node.flags & NODE_MORPH_IN == 0),
            "the morph-in flag is the refresh frame's alone"
        );
    });
}

#[test]
fn a_display_tile_arriving_after_a_step_stands_in_in_its_first_frame() {
    GpuTest::run(|gpu| {
        let mut sphere = Sphere::new();
        let mut residency = residency(gpu, &sphere, 5);
        frame(
            gpu,
            &mut residency,
            &mut sphere,
            Drained::Epochs(Vec::new()),
            None,
            None,
        );
        let (epoch, displacement) = sphere.step(0.02);
        let late: Vec<Place> = above(&epoch)
            .into_iter()
            .filter(|p| p.level().get() == 6)
            .collect();
        assert!(!late.is_empty(), "the edit is under level-6 tiles");
        let drained = Drained::Epochs(vec![epoch]);
        frame(
            gpu,
            &mut residency,
            &mut sphere,
            drained,
            Some(displacement),
            None,
        );
        // Level 6 isn't resident yet: the floor's tiles stood in. Then level 6 is asked for.
        let requests = Requests::new(
            late.iter().map(|place| Request {
                key: place.key(sphere.finest).expect("a key"),
                screen_error: 10.0,
            }),
            0,
        );
        frame(
            gpu,
            &mut residency,
            &mut sphere,
            Drained::Epochs(Vec::new()),
            None,
            Some(&requests),
        );
        for place in &late {
            let key = place.key(sphere.finest).expect("a key");
            assert!(residency.slot(&key).is_some(), "{place:?} arrived");
        }
        let raised = check(gpu, &residency, &sphere);
        assert!(
            raised >= late.len(),
            "the arrivals stood in: {raised} raised"
        );
    });
}
