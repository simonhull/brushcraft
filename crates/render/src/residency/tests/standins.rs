//! Stand-ins through residency ([`Residency::apply`] of the store's step epochs).
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
    /// Tiles read, display and finest.
    reads: usize,
}

/// `p` pushed out along itself by a smooth bump of `amplitude` and `radius` round `centre`:
/// a function of `p`'s bits alone, so every copy of a vertex moves alike.
fn dab(p: [f32; 3], centre: [f32; 3], amplitude: f32, radius: f32) -> [f32; 3] {
    let d = [0, 1, 2].map(|a| p[a] - centre[a]);
    let reach = 1.0 - (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) / (radius * radius);
    if reach <= 0.0 {
        return p;
    }
    let scale = 1.0 + amplitude * reach * reach;
    p.map(|c| c * scale)
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
            reads: 0,
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

    /// Bumps the finest grids by `amplitude` (the tests' wavy bump), and returns the step's
    /// epoch: every finest tile it changed, and its displacement bound.
    fn step(&mut self, amplitude: f32) -> Epoch {
        self.edit(|p| bumped(p, amplitude, 0.35))
    }

    /// A small dab of `amplitude` at `centre`: the step's epoch.
    fn dab(&mut self, centre: [f32; 3], amplitude: f32) -> Epoch {
        self.edit(|p| dab(p, centre, amplitude, 0.12))
    }

    /// Moves every finest vertex by `moved`, and returns the step's epoch: every finest tile it
    /// changed, and its displacement bound.
    fn edit(&mut self, moved: impl Fn([f32; 3]) -> [f32; 3]) -> Epoch {
        let before = self.current.clone();
        for grid in self.current.values_mut() {
            for p in grid.iter_mut() {
                *p = moved(*p);
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
        // The store's bump names the display tiles above, their bound raised.
        let raised: BTreeSet<TileKey> = keys
            .iter()
            .flat_map(|key| {
                let mut above = Vec::new();
                let mut at = Place::of(key).parent();
                while let Some(place) = at {
                    above.extend(place.key(self.finest));
                    at = place.parent();
                }
                above
            })
            .collect();
        keys.extend(raised);
        self.seq += 1;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "rounded to an f32, then up a step"
        )]
        let displacement = (worst as f32).next_up();
        Epoch {
            seq: self.seq,
            kind: EpochKind::Step { displacement },
            keys,
        }
    }

    /// Refreshes the display tiles at `places`, and returns the refresh's epoch.
    fn refresh(&mut self, places: &BTreeSet<Place>) -> Epoch {
        self.refreshed.extend(places.iter().copied());
        self.seq += 1;
        let keys = places.iter().filter_map(|p| p.key(self.finest)).collect();
        Epoch {
            seq: self.seq,
            kind: EpochKind::Refresh,
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
        self.reads += 1;
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
            stored: None,
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

/// One frame: the drain applied, `requests` wanted, staged, submitted and waited for.
fn frame(
    gpu: &Gpu,
    residency: &mut Residency,
    sphere: &mut Sphere,
    drained: Drained,
    requests: Option<&Requests>,
) {
    residency.begin_frame(gpu).expect("a frame begins");
    residency.apply(drained);
    residency.use_frame(FrameUse::default());
    if let Some(requests) = requests {
        residency.want(requests);
    }
    let deadline = Instant::now() + Duration::from_secs(60);
    let uploads = residency.stage(gpu, sphere, deadline).expect("staged");
    submit(gpu, uploads, |_| {});
}

/// A frame with nothing drained or wanted.
fn quiet(gpu: &Gpu, residency: &mut Residency, sphere: &mut Sphere) {
    frame(gpu, residency, sphere, Drained::Epochs(Vec::new()), None);
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
    );
    for (step, amplitude) in [0.02, -0.015].into_iter().enumerate() {
        let epoch = sphere.step(amplitude);
        let stood = above(&epoch);
        let drained = Drained::Epochs(vec![epoch]);
        frame(gpu, &mut residency, &mut sphere, drained, None);
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
        );
        let epoch = sphere.step(0.02);
        let stood = above(&epoch);
        let drained = Drained::Epochs(vec![epoch]);
        frame(gpu, &mut residency, &mut sphere, drained, None);
        let refresh = sphere.refresh(&stood);
        frame(
            gpu,
            &mut residency,
            &mut sphere,
            Drained::Epochs(vec![refresh]),
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
            residency.stand_ins.index.is_empty(),
            "the step is caught up with"
        );
        check(gpu, &residency, &sphere);
        frame(
            gpu,
            &mut residency,
            &mut sphere,
            Drained::Epochs(Vec::new()),
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
        );
        let epoch = sphere.step(0.02);
        let late: Vec<Place> = above(&epoch)
            .into_iter()
            .filter(|p| p.level().get() == 6)
            .collect();
        assert!(!late.is_empty(), "the edit is under level-6 tiles");
        let drained = Drained::Epochs(vec![epoch]);
        frame(gpu, &mut residency, &mut sphere, drained, None);
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

/// Every resident tile's vertices, display and finest.
fn resident_tiles(gpu: &Gpu, residency: &Residency) -> BTreeMap<Place, Vec<GpuVertex>> {
    let mut buffers: BTreeMap<(u8, u16), Vec<GpuVertex>> = BTreeMap::new();
    let mut tiles = BTreeMap::new();
    for key in residency.resident() {
        let slot = residency.slot(key).expect("resident");
        let data = buffers
            .entry((slot.class().get(), slot.buffer()))
            .or_insert_with(|| {
                let buffer = &residency.slot_buffers(slot.class())[usize::from(slot.buffer())];
                gpu.read_buffer(buffer).expect("slots read back")
            });
        let count = usize::try_from(slot.class().vertices()).expect("small");
        let first = usize::try_from(slot.index()).expect("small") * count;
        tiles.insert(Place::of(key), data[first..first + count].to_vec());
    }
    tiles
}

/// The face-grid vertices (finest level) that two resident tiles, display or finest, hold on
/// their borders with different bits: where a seam between them would crack.
fn cracks(gpu: &Gpu, residency: &Residency, sphere: &Sphere) -> usize {
    let mut seen: BTreeMap<(u32, u32, u32), BTreeSet<[u32; 3]>> = BTreeMap::new();
    for (place, vertices) in resident_tiles(gpu, residency) {
        let (x0, y0, size) = sphere.block(place);
        let quads = place.level().tile_quads();
        let scale = size / quads;
        let border = (0..=quads)
            .flat_map(|j| (0..=quads).map(move |i| (i, j)))
            .filter(|&(i, j)| i == 0 || j == 0 || i == quads || j == quads);
        for (i, j) in border {
            let v = vertices[usize::try_from(j * (quads + 1) + i).expect("small")];
            let at = (place.face().index(), x0 + i * scale, y0 + j * scale);
            seen.entry(at)
                .or_default()
                .insert(v.position.map(f32::to_bits));
        }
    }
    seen.values().filter(|bits| bits.len() > 1).count()
}

/// The finest tiles `epoch` names.
fn finest_of(epoch: &Epoch, finest: Level) -> BTreeSet<TileKey> {
    let keys = epoch.keys.iter().filter(|key| key.level() == finest);
    keys.copied().collect()
}

#[test]
fn the_reads_a_step_takes_stay_flat_however_long_the_stroke() {
    GpuTest::run(|gpu| {
        let mut sphere = Sphere::new();
        let mut residency = residency(gpu, &sphere, 6);
        quiet(gpu, &mut residency, &mut sphere);
        let centre = [0.6, 0.7, 0.387_298_33];
        let mut per_step = Vec::new();
        for step in 0..40 {
            let amplitude = if step % 2 == 0 { 0.004 } else { -0.003 };
            let epoch = sphere.dab(centre, amplitude);
            let finest = finest_of(&epoch, sphere.finest).len();
            let before = sphere.reads;
            frame(
                gpu,
                &mut residency,
                &mut sphere,
                Drained::Epochs(vec![epoch]),
                None,
            );
            // A step reads its finest tiles once, for the stand-ins, and no display tile: the
            // ones it names had only their bound raised.
            per_step.push((sphere.reads - before, finest));
        }
        for (step, &(reads, finest)) in per_step.iter().enumerate() {
            assert_eq!(
                reads, finest,
                "step {step}: {reads} reads for {finest} finest tiles"
            );
        }
        assert!(per_step.iter().all(|&(_, finest)| finest > 0));
        check(gpu, &residency, &sphere);
    });
}

#[test]
fn a_step_staged_over_several_frames_never_cracks_a_seam_beside_a_drawn_finest_tile() {
    GpuTest::run(|gpu| {
        let mut sphere = Sphere::new();
        let big = SizeClass::of(sphere.finest).slot_bytes();
        // Staging for two finest tiles a frame, and every epoch of more than one through
        // shadow slots, over several frames.
        let config = ResidencyConfig::new(256 << 20, big, big, big, 64, level(6))
            .expect("a valid configuration");
        let mut residency =
            Residency::new(gpu, &sphere.mesh, sphere.finest, config).expect("residency");
        // Every finest tile resident, as drawn up close.
        let all: Vec<TileKey> = sphere
            .mesh
            .faces()
            .flat_map(|face| {
                (0..2).flat_map(move |ty| {
                    (0..2).filter_map(move |tx| {
                        TileKey::displacement(face, level(FINEST), tx, ty).ok()
                    })
                })
            })
            .collect();
        let requests = Requests::new(
            all.iter().map(|&key| Request {
                key,
                screen_error: 1.0,
            }),
            0,
        );
        for _ in 0..40 {
            frame(
                gpu,
                &mut residency,
                &mut sphere,
                Drained::Epochs(Vec::new()),
                Some(&requests),
            );
            if all.iter().all(|key| residency.slot(key).is_some()) {
                break;
            }
        }
        assert!(
            all.iter().all(|key| residency.slot(key).is_some()),
            "all resident"
        );
        assert_eq!(cracks(gpu, &residency, &sphere), 0, "before the step");
        let epoch = sphere.step(0.02);
        assert!(
            finest_of(&epoch, sphere.finest).len() > 2,
            "a step of several tiles"
        );
        let mut drained = Drained::Epochs(vec![epoch]);
        let mut frames = 0;
        loop {
            frame(gpu, &mut residency, &mut sphere, drained, None);
            drained = Drained::Epochs(Vec::new());
            frames += 1;
            assert_eq!(
                cracks(gpu, &residency, &sphere),
                0,
                "frame {frames}: a crack"
            );
            if residency.stats().pending_epochs == 0 {
                break;
            }
            assert!(frames < 40, "the step never switched");
        }
        assert!(
            frames > 1,
            "the step went through shadow slots over several frames"
        );
        assert!(residency.stats().shadowed > 0);
        check(gpu, &residency, &sphere);
    });
}

#[test]
fn a_revert_in_the_drain_of_a_step_drops_its_stand_ins_and_reads_the_store_again() {
    GpuTest::run(|gpu| {
        let mut sphere = Sphere::new();
        let mut residency = residency(gpu, &sphere, 6);
        quiet(gpu, &mut residency, &mut sphere);
        let first = sphere.step(0.02);
        let stood = above(&first);
        frame(
            gpu,
            &mut residency,
            &mut sphere,
            Drained::Epochs(vec![first]),
            None,
        );
        assert!(!residency.stand_ins.rows.is_empty(), "stood in");
        // A step, then a revert of the same tiles, in one drain: the store's data for the
        // display tiles above them is what they draw.
        let step = sphere.step(-0.01);
        sphere.current = sphere.stored.clone();
        sphere.seq += 1;
        let revert = Epoch {
            seq: sphere.seq,
            kind: EpochKind::Revert,
            keys: finest_of(&step, sphere.finest),
        };
        frame(
            gpu,
            &mut residency,
            &mut sphere,
            Drained::Epochs(vec![step, revert]),
            None,
        );
        for place in &stood {
            assert!(
                residency.stand_ins.rows.get(*place).is_none(),
                "{place:?} stands in"
            );
        }
        // The reverted surface is the stored one: what the tiles drew from the store is right.
        assert_eq!(cracks(gpu, &residency, &sphere), 0);
        check(gpu, &residency, &sphere);
    });
}

#[test]
fn after_a_stroke_whose_refresh_never_comes_the_frames_go_idle() {
    GpuTest::run(|gpu| {
        let mut sphere = Sphere::new();
        let mut residency = residency(gpu, &sphere, 6);
        quiet(gpu, &mut residency, &mut sphere);
        for amplitude in [0.02, -0.01, 0.015] {
            let epoch = sphere.step(amplitude);
            frame(
                gpu,
                &mut residency,
                &mut sphere,
                Drained::Epochs(vec![epoch]),
                None,
            );
        }
        assert!(!residency.stand_ins.rows.is_empty(), "stand-ins live");
        for idle in 0..3 {
            residency.begin_frame(gpu).expect("a frame begins");
            residency.apply(Drained::Epochs(Vec::new()));
            residency.use_frame(FrameUse::default());
            let deadline = Instant::now() + Duration::from_secs(60);
            let uploads = residency.stage(gpu, &mut sphere, deadline).expect("staged");
            assert!(
                !uploads.changes_scene(),
                "idle frame {idle} changes the scene"
            );
            assert!(uploads.is_empty(), "idle frame {idle} has work");
            submit(gpu, uploads, |_| {});
        }
        check(gpu, &residency, &sphere);
    });
}

#[test]
fn a_lost_frame_s_stand_ins_come_back_from_the_store_and_stay_bounds() {
    GpuTest::run(|gpu| {
        let mut sphere = Sphere::new();
        let mut residency = residency(gpu, &sphere, 6);
        quiet(gpu, &mut residency, &mut sphere);
        let first = sphere.step(0.02);
        frame(
            gpu,
            &mut residency,
            &mut sphere,
            Drained::Epochs(vec![first]),
            None,
        );
        // Frame A stands in for a second step and is lost; frame B, staged before that is
        // heard, stands in for a third and is submitted.
        let second = sphere.step(-0.015);
        residency.begin_frame(gpu).expect("a frame begins");
        residency.apply(Drained::Epochs(vec![second]));
        residency.use_frame(FrameUse::default());
        let deadline = Instant::now() + Duration::from_secs(60);
        let lost = residency.stage(gpu, &mut sphere, deadline).expect("staged");
        let third = sphere.step(0.01);
        residency.begin_frame(gpu).expect("a frame begins");
        residency.apply(Drained::Epochs(vec![third]));
        residency.use_frame(FrameUse::default());
        let kept = residency.stage(gpu, &mut sphere, deadline).expect("staged");
        drop(lost);
        submit(gpu, kept, |_| {});
        for _ in 0..4 {
            quiet(gpu, &mut residency, &mut sphere);
        }
        assert!(residency.stats().lost_frames > 0);
        assert_eq!(cracks(gpu, &residency, &sphere), 0);
        check(gpu, &residency, &sphere);
    });
}

#[test]
fn a_stood_in_tile_evicted_and_back_stands_in_again_in_its_first_frame() {
    GpuTest::run(|gpu| {
        let mut sphere = Sphere::new();
        let mut residency = residency(gpu, &sphere, 5);
        quiet(gpu, &mut residency, &mut sphere);
        let epoch = sphere.step(0.02);
        let late: Vec<TileKey> = above(&epoch)
            .into_iter()
            .filter(|p| p.level().get() == 6)
            .filter_map(|p| p.key(sphere.finest))
            .collect();
        let requests = Requests::new(
            late.iter().map(|&key| Request {
                key,
                screen_error: 10.0,
            }),
            0,
        );
        frame(
            gpu,
            &mut residency,
            &mut sphere,
            Drained::Epochs(Vec::new()),
            Some(&requests),
        );
        frame(
            gpu,
            &mut residency,
            &mut sphere,
            Drained::Epochs(vec![epoch]),
            Some(&requests),
        );
        assert!(late.iter().all(|key| residency.slot(key).is_some()));
        assert!(
            late.iter()
                .all(|key| residency.stand_ins.rows.get(Place::of(key)).is_some())
        );
        // Evicted, then requested again: back with the store's data, behind the step.
        for key in &late {
            residency.remove_live(key);
        }
        quiet(gpu, &mut residency, &mut sphere);
        frame(
            gpu,
            &mut residency,
            &mut sphere,
            Drained::Epochs(Vec::new()),
            Some(&requests),
        );
        assert!(late.iter().all(|key| residency.slot(key).is_some()), "back");
        let raised = check(gpu, &residency, &sphere);
        assert!(
            raised >= late.len(),
            "the returns stood in: {raised} raised"
        );
    });
}

#[test]
fn a_place_keeps_at_most_max_steps_and_counts_the_ones_it_forgets() {
    GpuTest::run(|gpu| {
        let sphere = Sphere::new();
        let mut residency = residency(gpu, &sphere, 6);
        let key = TileKey::displacement(FaceId::new(0), sphere.finest, 0, 0).expect("a key");
        let steps = MAX_STEPS + 5;
        let epochs = (0..steps).map(|seq| Epoch {
            seq: seq as u64,
            kind: EpochKind::Step { displacement: 1e-4 },
            keys: BTreeSet::from([key]),
        });
        residency.apply(Drained::Epochs(epochs.collect()));
        let above = sphere.finest.get();
        assert_eq!(
            residency.stats().stand_in_steps_dropped,
            5 * u64::from(above),
            "five steps forgotten at each of the places above"
        );
        assert!(
            residency
                .stand_ins
                .index
                .values()
                .all(|kept| kept.len() == MAX_STEPS)
        );
    });
}

#[test]
fn with_every_row_taken_a_stand_in_waits_and_is_counted() {
    GpuTest::run(|gpu| {
        let mut sphere = Sphere::new();
        let config = ResidencyConfig::new(256 << 20, 16 << 20, 64 << 20, 8 << 20, 64, level(6))
            .expect("a valid configuration")
            .with_stand_ins(160, 2);
        let mut residency =
            Residency::new(gpu, &sphere.mesh, sphere.finest, config).expect("residency");
        quiet(gpu, &mut residency, &mut sphere);
        let epoch = sphere.step(0.02);
        let wanted = above(&epoch).len();
        frame(
            gpu,
            &mut residency,
            &mut sphere,
            Drained::Epochs(vec![epoch]),
            None,
        );
        let stats = residency.stats();
        assert_eq!(stats.stand_ins, 2, "two rows");
        assert_eq!(stats.stand_ins_skipped, (wanted - 2) as u64);
        // Each waits: looked at again next frame.
        quiet(gpu, &mut residency, &mut sphere);
        assert_eq!(
            residency.stats().stand_ins_skipped,
            2 * (wanted - 2) as u64,
            "tried again"
        );
    });
}

#[test]
fn a_tile_past_what_a_scratch_binds_is_set_aside_until_its_inputs_change() {
    GpuTest::run(|gpu| {
        let mut sphere = Sphere::new();
        let mut residency = residency(gpu, &sphere, 6);
        quiet(gpu, &mut residency, &mut sphere);
        residency.stand_ins.pass.set_most_tiles(1);
        let epoch = sphere.step(0.02);
        assert!(finest_of(&epoch, sphere.finest).len() > 1);
        frame(
            gpu,
            &mut residency,
            &mut sphere,
            Drained::Epochs(vec![epoch]),
            None,
        );
        let skipped = residency.stats().stand_ins_skipped;
        assert!(skipped > 0, "some tiles need more than one finest tile");
        assert!(!residency.stand_ins.oversized.is_empty());
        // Not walked again while nothing under them changes.
        quiet(gpu, &mut residency, &mut sphere);
        assert_eq!(residency.stats().stand_ins_skipped, skipped, "set aside");
        // A step under them: looked at again.
        let next = sphere.step(-0.01);
        frame(
            gpu,
            &mut residency,
            &mut sphere,
            Drained::Epochs(vec![next]),
            None,
        );
        assert!(
            residency.stats().stand_ins_skipped > skipped,
            "looked at again"
        );
    });
}
