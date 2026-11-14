//! Strokes over strokes still landing: a `Down` while an earlier stroke's dabs land begins at once,
//! on the surface that stroke will leave (the stroke-while-landing spec).
//!
//! Today's behaviour, a stroke that waits for the one before it to land, is the oracle: the same
//! samples, placed once every earlier stroke has landed, give the same dabs, bit for bit, and the
//! same tiles once everything has landed.

use geom::Vec3;
use store::{DabBatch, DisplayPreview, FLUSH_INTERVAL, PenDown};

use super::*;
use crate::testing::settle;

/// A brush stepped directly over a project saved in a directory of its own, as the journal tests
/// step one.
///
/// A dab at a time is too dear for any time ([`DabCost::one_at_a_time`]), so a frame's step shows
/// every dab and lands none, and the strokes land only when asked ([`Chain::land_all`]).
struct Chain {
    /// The project's directory, kept until the project closes.
    dir: TestDir,
    /// The project, saved: a saved project says when its flushes are done ([`settle`]).
    project: Project,
    /// Its spatial index.
    picker: Picker,
    /// The brush.
    brush: Sculpting,
    /// The camera, the default one.
    camera: Camera,
    /// When the last cut was ticked at ([`Chain::cut`]).
    ticked: Instant,
}

impl Chain {
    /// The default brush, its sample times counted from `epoch`, over a fresh project saved in a
    /// directory of its own, its store under `store` (the tests' own with `None`).
    fn new(epoch: Instant, store: Option<StoreConfig>) -> Self {
        let dir = TestDir::new();
        let mut config_of = config(&dir);
        if let Some(store) = store {
            config_of.store = store;
        }
        let mut project =
            Project::new(cube_sphere(2).expect("a cube sphere"), config_of).expect("a new project");
        drop(
            project
                .save_as(&dir.path().join("chain.bcrf"))
                .expect("saved"),
        );
        let picker = Picker::load(&mut project)
            .expect("the index loads")
            .expect("a display layer");
        let brush = Sculpting::new(&mut project)
            .with_cost(DabCost::one_at_a_time())
            .with_epoch(epoch);
        Self {
            dir,
            project,
            picker,
            brush,
            camera: Camera::default(),
            ticked: Instant::now(),
        }
    }

    /// A frame's step of the brush given `samples`, all routed to it: no dab fits, and none lands.
    fn frame(&mut self, samples: &[PenSample]) -> BrushReport {
        let mut pen = PenFrame::default();
        for sample in samples {
            pen.push(*sample, Some(Route::Brush));
        }
        let budget = BrushBudget::frame(Instant::now(), Duration::ZERO, setup().interval);
        let context = context(&mut self.project, self.picker.index_mut(), &self.camera);
        self.brush.step(context, &pen, &budget).expect("a step")
    }

    /// Routes `down` as the frame loop does before the step that takes it.
    fn route(&mut self, down: &PenSample) -> Route {
        let context = context(&mut self.project, self.picker.index_mut(), &self.camera);
        self.brush.route(context, down)
    }

    /// Whether routing left a stroke begun over those under way, its op reserved, for its `Down`
    /// to begin with.
    fn routed_over(&self) -> bool {
        matches!(self.brush.routed, Some(Routed::Over(_)))
    }

    /// Asserts that the store holds no op reserved: with one stroke under way, none begun over
    /// it, routing gave back whatever it reserved.
    fn assert_nothing_reserved(&mut self) {
        let reserved = self.project.store().drop_every_reserved_op();
        assert!(reserved.is_empty(), "left reserved: {reserved:?}");
    }

    /// A step that lands one dab of the oldest stroke, however dear: every pen quiet for long and
    /// the catch-up starved, so a step that must make progress takes one.
    fn land_one(&mut self) -> BrushReport {
        let long_ago = Instant::now()
            .checked_sub(RESTING_QUIET * 2)
            .expect("after the epoch");
        self.brush.landed_at = long_ago;
        self.brush.unframed = false;
        for stroke in &mut self.brush.strokes {
            stroke.placed_at = long_ago;
            stroke.last.received = long_ago;
        }
        let context = context(&mut self.project, self.picker.index_mut(), &self.camera);
        self.brush
            .step(context, &PenFrame::default(), &BrushBudget::unlimited())
            .expect("a step")
    }

    /// Lands every dab the brush holds, every stroke after another, as a flush does.
    fn land_all(&mut self) {
        let context = context(&mut self.project, self.picker.index_mut(), &self.camera);
        self.brush.flush(context).expect("flushed");
        assert_eq!(self.brush.landing(), 0, "everything landed");
        assert_eq!(self.brush.error(), None);
    }

    /// The rows of the dabs not yet landed of every stroke under way, oldest stroke first.
    fn rows(&self) -> Vec<Vec<DabRow>> {
        let strokes = self.brush.pending();
        let rows =
            strokes.map(|stroke| stroke.dabs.iter().map(|dab| bytemuck::cast(*dab)).collect());
        rows.collect()
    }

    /// The ops of the strokes under way, oldest first.
    fn ops(&self) -> Vec<OpId> {
        let strokes = self.brush.pending();
        strokes.map(Pending::op_id).collect()
    }

    /// Cuts the project's journal now, between brush calls as the frame loop does: a tick a flush
    /// interval past every tick before, with no flush under way first, then waits until it is
    /// durable.
    fn cut(&mut self) {
        settle(&mut self.project);
        self.ticked = self.ticked.max(Instant::now()) + FLUSH_INTERVAL * 2;
        self.project.tick(self.ticked).expect("ticked");
        settle(&mut self.project);
    }

    /// Builds the project's display layer, as a renderer's first frames have it built: what its
    /// display previews are made over.
    fn build_display(&mut self) {
        let store = self.project.store();
        let finest = store.display_finest().expect("a display layer");
        store.queue_display_rebuild(finest);
        for _ in 0..10_000 {
            store
                .run_jobs(Instant::now() + Duration::from_millis(50))
                .expect("the jobs run");
            if !store.has_jobs() {
                return;
            }
        }
        panic!("the display layer never finished");
    }

    /// Every tile's evaluated positions, then closes the project (keeping no recovery file).
    fn close(self) -> BTreeMap<TileKey, Vec<[u32; 3]>> {
        let Self {
            dir, mut project, ..
        } = self;
        let tiles = positions(&mut project);
        project.close(false).expect("closed");
        drop(dir);
        tiles
    }
}

/// A stroke along the sphere's middle lower down than [`ACROSS`], crossing it: `Down` to `Up`.
fn crossing() -> Vec<PenSample> {
    stroke(((150.0, 80.0), (170.0, 125.0)), 10)
}

/// A third stroke, over both: `Down` to `Up`.
fn over_both() -> Vec<PenSample> {
    stroke(((140.0, 115.0), (185.0, 95.0)), 10)
}

#[test]
fn a_down_while_a_stroke_lands_begins_at_once_and_its_first_frame_shows_its_dabs() {
    let mut run = Chain::new(Instant::now(), None);
    let first = run.frame(&stroke(ACROSS, 12));
    assert!(first.dabs_shown > 0 && first.dabs_applied == 0, "{first:?}");
    let second = run.frame(&crossing());
    assert!(
        run.brush.deferred.is_empty(),
        "the second stroke begins at once: nothing waits"
    );
    assert_eq!(run.ops().len(), 2, "both strokes under way");
    let placed = run.rows().last().map_or(0, Vec::len);
    assert!(placed > 0, "the second placed its dabs");
    assert_eq!(
        second.dabs_shown, placed,
        "its first frame shows every dab it placed"
    );
    assert_eq!(second.dabs_applied, 0, "none of it lands before the first");
    run.land_all();
    run.close();
}

/// What strokes drawn over each other leave: the dabs of each, and every tile once all have landed.
type Drawn = (Vec<Vec<DabRow>>, BTreeMap<TileKey, Vec<[u32; 3]>>);

/// What `strokes` leave drawn over each other, and drawn as before.
///
/// First under way at once, each `Down` while the strokes before still land; then one after
/// another, each once the one before has landed: today's behaviour.
fn chained_and_waited(strokes: &[Vec<PenSample>]) -> (Drawn, Drawn) {
    let epoch = Instant::now();
    let mut chained = Chain::new(epoch, None);
    for samples in strokes {
        chained.frame(samples);
    }
    let chained_rows = chained.rows();
    assert_eq!(chained_rows.len(), strokes.len(), "every stroke under way");
    chained.land_all();

    let mut waited = Chain::new(epoch, None);
    let mut waited_rows = Vec::new();
    for samples in strokes {
        waited.frame(samples);
        waited_rows.extend(waited.rows());
        waited.land_all();
    }
    (
        (chained_rows, chained.close()),
        (waited_rows, waited.close()),
    )
}

#[test]
fn a_stroke_over_one_landing_places_the_dabs_and_leaves_the_tiles_of_waiting() {
    let ((chained, chained_tiles), (waited, waited_tiles)) =
        chained_and_waited(&[stroke(ACROSS, 12), crossing()]);
    assert!(
        waited.iter().all(|rows| !rows.is_empty()),
        "each placed dabs"
    );
    assert_eq!(chained, waited, "the same dabs, bit for bit");
    assert!(chained_tiles == waited_tiles, "the same tiles, bit for bit");
}

#[test]
fn three_strokes_in_a_chain_place_and_land_as_one_after_another() {
    let ((chained, chained_tiles), (waited, waited_tiles)) =
        chained_and_waited(&[stroke(ACROSS, 12), crossing(), over_both()]);
    assert!(
        waited.iter().all(|rows| !rows.is_empty()),
        "each placed dabs"
    );
    assert_eq!(chained, waited, "the same dabs, bit for bit");
    assert!(chained_tiles == waited_tiles, "the same tiles, bit for bit");
}

/// The most strokes `past_the_open_strokes_share_the_next_down_waits_as_before` draws over
/// those still landing before one must wait.
const MOST_OPEN: usize = 256;

#[test]
fn past_the_open_strokes_share_the_next_down_waits_as_before() {
    // Small pools, in which one stroke's dabs land, and which the pen-down data of a few dozen
    // strokes fill: each stroke's futures are the tiles under it.
    let config = |budget: usize| {
        StoreConfig::new(budget, 1)
            .with_chunk_bytes(64 << 10)
            .with_kept_predictions(0)
            .with_stroke_tiles(8)
            .with_display(Level::new(LEVEL).expect("a level"))
    };
    let minimum = config(0).minimum_budget(&cube_sphere(2).expect("a cube sphere"));
    // 8 MiB over the least: a dab over the whole sphere pins its 24 tiles, about 8 MB.
    let mut run = Chain::new(Instant::now(), Some(config(minimum + (8 << 20))));
    run.brush.set_radius(40.0);
    let strokes = [stroke(ACROSS, 6), crossing(), over_both()];
    let mut began = 0;
    let drawn = strokes.iter().cycle().take(MOST_OPEN);
    for samples in drawn {
        run.frame(samples);
        if !run.brush.deferred.is_empty() {
            break;
        }
        began += 1;
    }
    assert!(
        !run.brush.deferred.is_empty(),
        "past the share, a stroke waits"
    );
    assert!(began > 1, "strokes began over those landing first: {began}");
    assert_eq!(run.ops().len(), began, "each of them under way");
    let store = run.project.store();
    let (held, share) = (store.pen_down_bytes(), store.open_strokes_share());
    assert!(
        held.saturating_mul(4) >= share.saturating_mul(3),
        "the open strokes hold about their share: {held} of {share}"
    );
    assert_eq!(run.brush.error(), None);
    run.land_all();
    run.close();
}

/// The batches of the stroke made as `op` in the recovery file at `path`, in the order written.
fn batches_at(path: &std::path::Path, op: OpId) -> Vec<DabBatch> {
    let file = ProjectFile::open(path).expect("the recovery file opens");
    let batches = file.batches().map(|batch| batch.expect("a batch"));
    let of_op = batches.filter(|batch| batch.op() == op);
    of_op.collect()
}

#[test]
fn a_later_strokes_batches_name_the_stroke_before_then_its_own_cut() {
    let mut run = Chain::new(Instant::now(), None);
    run.frame(&stroke(ACROSS, 12));
    run.frame(&crossing());
    let ops = run.ops();
    let [first, second] = ops.as_slice() else {
        panic!("two strokes under way: {ops:?}");
    };
    // Cut while the first still lands: the second's dabs are journalled, its pen-down the first's.
    run.cut();
    run.land_all();
    run.cut();
    let Chain {
        dir,
        project,
        brush,
        ..
    } = run;
    drop(brush);
    let path = project.recovery_path().expect("a recovery file").to_owned();
    project.close(true).expect("closed");
    let batches = batches_at(&path, *second);
    let pen_downs: Vec<PenDown> = batches.iter().map(DabBatch::pen_down).collect();
    assert!(
        matches!(pen_downs.first(), Some(PenDown::After(op)) if op == first),
        "first after the stroke before: {pen_downs:?}"
    );
    assert!(
        matches!(pen_downs.last(), Some(PenDown::At(_))),
        "then at its own cut: {pen_downs:?}"
    );
    drop(dir);
}

/// The dabs `strokes` place, and the tiles once all have landed, drawn as `draw` draws them on a
/// [`Chain`] of sample times counted from `epoch`, which ends with every stroke under way.
fn drawn(epoch: Instant, draw: impl FnOnce(&mut Chain)) -> Drawn {
    let mut run = Chain::new(epoch, None);
    draw(&mut run);
    let rows = run.rows();
    run.land_all();
    (rows, run.close())
}

/// [`stroke()`] along [`ACROSS`] and [`crossing`] drawn one after the other, each once the one
/// before has landed: the oracle.
fn waited(epoch: Instant) -> Drawn {
    let mut run = Chain::new(epoch, None);
    let mut rows = Vec::new();
    for samples in [stroke(ACROSS, 12), crossing()] {
        run.frame(&samples);
        rows.extend(run.rows());
        run.land_all();
    }
    (rows, run.close())
}

#[test]
fn a_stroke_over_one_landing_between_its_frames_places_the_dabs_of_waiting() {
    let epoch = Instant::now();
    let (first, second) = (stroke(ACROSS, 12), crossing());
    let (chained, chained_tiles) = drawn(epoch, |run| {
        run.frame(&first);
        run.cut();
        for _ in 0..3 {
            assert_eq!(run.land_one().dabs_applied, 1, "the first lands a dab");
        }
        let (early, late) = second.split_at(4);
        run.frame(early);
        for _ in 0..3 {
            assert_eq!(run.land_one().dabs_applied, 1, "more of the first lands");
        }
        run.frame(late);
        assert_eq!(run.ops().len(), 2, "both under way");
    });
    let (waited_rows, waited_tiles) = waited(epoch);
    assert_eq!(
        chained.last(),
        waited_rows.last(),
        "the second's dabs, bit for bit"
    );
    assert!(chained_tiles == waited_tiles, "the same tiles, bit for bit");
}

#[test]
fn a_stroke_whose_futures_cant_be_had_waits_its_samples_and_places_them_as_waiting() {
    let epoch = Instant::now();
    let (first, second) = (stroke(ACROSS, 12), crossing());
    let (chained, chained_tiles) = drawn(epoch, |run| {
        run.frame(&first);
        let (early, late) = second.split_at(3);
        run.frame(early);
        run.brush.knob = Knob::FuturesFail;
        run.frame(late);
        assert_eq!(
            run.brush.deferred.len(),
            late.len(),
            "the rest of its samples wait, whole"
        );
        assert!(run.brush.error().is_none(), "nothing dropped");
        // The first lands in frames; the second's op begins, and its samples are placed on its
        // own pen-down surface.
        run.cut();
        let mut steps = 0;
        while run.ops().len() > 1 {
            run.land_one();
            steps += 1;
            assert!(steps < 10_000, "the first lands");
        }
        assert!(
            run.brush.deferred.is_empty(),
            "its samples placed as its op began"
        );
        run.brush.knob = Knob::None;
    });
    let (waited_rows, waited_tiles) = waited(epoch);
    assert_eq!(
        chained.last(),
        waited_rows.last(),
        "the second's dabs, bit for bit"
    );
    assert!(chained_tiles == waited_tiles, "the same tiles, bit for bit");
}

/// What the previews show while strokes are under way.
struct Previews {
    /// Each previewed finest tile's own positions (its block without the ring), row-major.
    finest: BTreeMap<TileKey, Vec<Vec3>>,
    /// Each display preview.
    display: BTreeMap<TileKey, DisplayPreview>,
    /// Where two previews holding the same vertex draw it differently: each such vertex's place,
    /// once.
    cracks: Vec<String>,
}

impl Previews {
    /// What `store` previews now.
    fn of(store: &TileStore) -> Self {
        let finest = store.previewed().map(|&key| {
            let block = store.preview(&key).expect("previewed");
            let q = i64::from(block.quads());
            let rows = (0..=q).flat_map(|j| (0..=q).map(move |i| (i, j)));
            let own: Vec<Vec3> = rows.map(|(i, j)| block.get(i, j)).collect();
            (key, own)
        });
        let finest: BTreeMap<TileKey, Vec<Vec3>> = finest.collect();
        let display = store.display_previewed().filter_map(|&key| {
            let preview = store.display_preview(&key)?;
            Some((key, preview.clone()))
        });
        let display: BTreeMap<TileKey, DisplayPreview> = display.collect();
        let cracks = cracks(store);
        Self {
            finest,
            display,
            cracks,
        }
    }
}

/// A source answering each vertex with its own place (face, column, row) as exact small floats:
/// a gather from it says which vertex each slot of a block holds.
struct Places;

impl geom::PositionSource for Places {
    fn position(&self, face: geom::FaceId, _level: Level, i: u32, j: u32) -> Option<Vec3> {
        let exact = |n: u32| u16::try_from(n).ok().map(f32::from);
        Some(Vec3::new(exact(face.index())?, exact(i)?, exact(j)?))
    }
}

/// Every point a gathered block holds, in order: its positions with the ring, then each corner
/// ring's edges and diagonals.
fn points(block: &geom::GatheredBlock) -> Vec<Vec3> {
    let rings = (0..4).filter_map(|c| block.corner(c));
    let ring_points = rings.flat_map(|ring| ring.edges().iter().chain(ring.diagonals()));
    let all = block.padded().iter().chain(ring_points);
    all.copied().collect()
}

/// The vertices two previews in `store` hold and draw differently, bit for bit.
///
/// Positions: a finest preview's every point, ring included, and a display preview's border
/// vertices, every level's, against every other's at the same place (on the finest grid).
/// Normals: a display preview's border vertices against the other display previews' at its level.
fn cracks(store: &TileStore) -> Vec<String> {
    let mesh = store.mesh();
    let finest = store.display_finest().expect("a display layer");
    let mut positions: BTreeMap<[u32; 3], [u32; 3]> = BTreeMap::new();
    let mut normals: BTreeMap<(u8, [u32; 3]), [u32; 3]> = BTreeMap::new();
    let mut cracks = Vec::new();
    let bits = |v: Vec3| [v.x.to_bits(), v.y.to_bits(), v.z.to_bits()];
    // The place a gather of `Places` wrote as `place` at `level`, on the finest grid.
    let on_finest = |place: Vec3, level: Level| -> Option<[u32; 3]> {
        if place.x.is_nan() {
            return None;
        }
        let shift = finest.get() - level.get();
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "small whole numbers, written exactly by `Places`"
        )]
        let whole = |c: f32| c as u32;
        Some([
            whole(place.x),
            whole(place.y) << shift,
            whole(place.z) << shift,
        ])
    };
    for key in store.previewed() {
        let block = store.preview(key).expect("previewed");
        let places = geom::gather_block(mesh, &Places, geom::Region::of_tile(key)).expect("places");
        let pairs = points(&places).into_iter().zip(points(block));
        for (place, point) in pairs {
            let Some(at) = on_finest(place, key.level()) else {
                continue;
            };
            let was = *positions.entry(at).or_insert_with(|| bits(point));
            if was != bits(point) {
                cracks.push(format!("{key:?} at {at:?}"));
            }
        }
    }
    for key in store.display_previewed() {
        let preview = store.display_preview(key).expect("previewed");
        let places = geom::gather_block(mesh, &Places, geom::Region::of_tile(key)).expect("places");
        let side = key.side();
        let grid = (0..side).flat_map(|j| (0..side).map(move |i| (i, j)));
        let border = grid.filter(|&(i, j)| i == 0 || j == 0 || i + 1 == side || j + 1 == side);
        for (i, j) in border {
            let v = usize::try_from(j * side + i).expect("a vertex");
            let place = places.get(i64::from(i), i64::from(j));
            let Some(at) = on_finest(place, key.level()) else {
                continue;
            };
            let (position, normal) = (bits(preview.positions[v]), bits(preview.normals[v]));
            let was = *positions.entry(at).or_insert(position);
            let was_normal = *normals.entry((key.level().get(), at)).or_insert(normal);
            if was != position || was_normal != normal {
                cracks.push(format!("{key:?} at {at:?}"));
            }
        }
    }
    cracks
}

/// `bits` as positions.
fn from_bits(bits: &[[u32; 3]]) -> Vec<Vec3> {
    let points = bits
        .iter()
        .map(|&[x, y, z]| Vec3::new(f32::from_bits(x), f32::from_bits(y), f32::from_bits(z)));
    points.collect()
}

/// The point display tile `key`'s triangles over `positions` draw at finest grid place
/// `(gx, gy)` of level `finest`, in `f64`: its quad there, split along its `(i, j)`–`(i+1, j+1)`
/// diagonal, as every level is drawn.
fn drawn_at(key: &TileKey, positions: &[Vec3], finest: Level, gx: u32, gy: u32) -> [f64; 3] {
    let quads = key.side() - 1;
    let step = f64::from(1_u32 << (finest.get() - key.level().get()));
    let across = f64::from(gx) / step - f64::from(u32::from(key.tx()) * quads);
    let down = f64::from(gy) / step - f64::from(u32::from(key.ty()) * quads);
    let last = f64::from(quads - 1);
    let (col, row) = (
        across.floor().clamp(0.0, last),
        down.floor().clamp(0.0, last),
    );
    let (fx, fy) = (across - col, down - row);
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a quad index within the tile, clamped"
    )]
    let (i, j) = (col as usize, row as usize);
    let side = usize::try_from(key.side()).expect("a side");
    let p = |a: usize, b: usize| {
        let v = positions[b * side + a];
        [f64::from(v.x), f64::from(v.y), f64::from(v.z)]
    };
    let (p00, p10, p01, p11) = (p(i, j), p(i + 1, j), p(i, j + 1), p(i + 1, j + 1));
    let mut out = [0.0; 3];
    for c in 0..3 {
        out[c] = if fx >= fy {
            p00[c] + fx * (p10[c] - p00[c]) + fy * (p11[c] - p10[c])
        } else {
            p00[c] + fy * (p01[c] - p00[c]) + fx * (p11[c] - p01[c])
        };
    }
    out
}

/// The farthest a finest vertex of `tiles` (every tile of the finest level `finest`) under
/// display tile `key` lies from what `preview` draws at its place.
fn farthest(
    tiles: &BTreeMap<TileKey, Vec<Vec3>>,
    finest: Level,
    key: &TileKey,
    preview: &DisplayPreview,
) -> f64 {
    let quads = key.side() - 1;
    let step = 1_u32 << (finest.get() - key.level().get());
    let tile_quads = finest.tile_quads();
    let span = (quads * step / tile_quads).max(1);
    let side = tile_quads + 1;
    let mut worst = 0.0_f64;
    for fy in 0..span {
        for fx in 0..span {
            let (tx, ty) = (
                u32::from(key.tx()) * span + fx,
                u32::from(key.ty()) * span + fy,
            );
            let tile = TileKey::displacement(
                key.face(),
                finest,
                u16::try_from(tx).expect("a tile"),
                u16::try_from(ty).expect("a tile"),
            )
            .expect("a tile");
            let positions = &tiles[&tile];
            for b in 0..side {
                for a in 0..side {
                    let f = positions[usize::try_from(b * side + a).expect("a vertex")];
                    let (gx, gy) = (tx * tile_quads + a, ty * tile_quads + b);
                    let d = drawn_at(key, &preview.positions, finest, gx, gy);
                    let gap = (f64::from(f.x) - d[0])
                        .hypot(f64::from(f.y) - d[1])
                        .hypot(f64::from(f.z) - d[2]);
                    worst = worst.max(gap);
                }
            }
        }
    }
    worst
}

/// The farthest a vertex of `positions`, display tile `key`'s, lies from the finest vertex of
/// `tiles` (every tile of the finest level `finest`) it subsamples.
fn farthest_vertex(
    tiles: &BTreeMap<TileKey, Vec<Vec3>>,
    finest: Level,
    key: &TileKey,
    positions: &[Vec3],
) -> f32 {
    let side = key.side();
    let quads = side - 1;
    let step = 1_u32 << (finest.get() - key.level().get());
    let tile_quads = finest.tile_quads();
    let last = finest.tiles_per_side() - 1;
    let place = |t: u16, v: u32| {
        let g = (u32::from(t) * quads + v) * step;
        let tile = (g / tile_quads).min(last);
        (tile, g - tile * tile_quads)
    };
    let mut worst = 0.0_f32;
    for j in 0..side {
        for i in 0..side {
            let (tx, a) = place(key.tx(), i);
            let (ty, b) = place(key.ty(), j);
            let tile = TileKey::displacement(
                key.face(),
                finest,
                u16::try_from(tx).expect("a tile"),
                u16::try_from(ty).expect("a tile"),
            )
            .expect("a tile");
            let at = usize::try_from(b * (tile_quads + 1) + a).expect("a vertex");
            let drawn = positions[usize::try_from(j * side + i).expect("a vertex")];
            worst = worst.max((tiles[&tile][at] - drawn).length());
        }
    }
    worst
}

/// Every display tile's record, as a renderer draws it: how far its vertices lie from the finest
/// vertices of `tiles` (every tile, in bits) they subsample.
fn record_errors(
    store: &mut TileStore,
    tiles: &BTreeMap<TileKey, Vec<[u32; 3]>>,
) -> BTreeMap<TileKey, f32> {
    let finest = Level::new(LEVEL).expect("a level");
    let tiles: BTreeMap<TileKey, Vec<Vec3>> = tiles
        .iter()
        .map(|(key, bits)| (*key, from_bits(bits)))
        .collect();
    let faces: Vec<geom::FaceId> = store.mesh().faces().collect();
    let levels = (0..LEVEL).map(|level| Level::new(level).expect("a level"));
    let mut keys: Vec<TileKey> = Vec::new();
    for level in levels {
        let per_side = u16::try_from(level.tiles_per_side()).expect("a few");
        for &face in &faces {
            for ty in 0..per_side {
                for tx in 0..per_side {
                    keys.push(TileKey::display(face, level, tx, ty).expect("a display tile"));
                }
            }
        }
    }
    let records = store.display_for_draw(&keys).expect("the records");
    let errors = keys.iter().zip(records).filter_map(|(key, record)| {
        let record = record?;
        Some((
            *key,
            farthest_vertex(&tiles, finest, key, record.positions()),
        ))
    });
    errors.collect()
}

/// Asserts that `shown`, the previews while strokes were under way, show `landed`, every tile
/// once they had all landed, within the previews' bounds.
///
/// A finest preview differs from what lands only by the stored encoding (under a thousandth of
/// how far the strokes moved the tiles shown), and a display preview lies within its bound of
/// every finest vertex under it.
///
/// A display preview's bound, derived and asserted above, is loose on these coarse tiles, so its
/// vertices are also held closer: each within twice its record's error before the strokes
/// (`records`) of the finest vertex it subsamples, plus a thousandth of how far the strokes moved.
/// That tolerance is empirical, not derived: a vertex drawn from its record is weighed where the
/// record has it, and its moves differ from the finest vertex's by the dabs' slopes over that
/// error, which no term here bounds; measured, it stays under the record's error, where a
/// stroke's dabs weighed on another stroke's surface, or left out, put a vertex a dab's move away
/// (five to a hundred times further). It is a check that the previews show the right dabs, the
/// bound the one that is guaranteed. `before` is every tile before the strokes; every preview
/// also meets every other bit for bit.
fn assert_shown_as_landed(
    shown: &Previews,
    before: &BTreeMap<TileKey, Vec<[u32; 3]>>,
    records: &BTreeMap<TileKey, f32>,
    landed: &BTreeMap<TileKey, Vec<[u32; 3]>>,
    what: &str,
) {
    assert!(!shown.finest.is_empty(), "{what}: finest previews");
    assert!(!shown.display.is_empty(), "{what}: display previews");
    assert!(
        shown.cracks.is_empty(),
        "{what}: previews meet bit for bit, but not at {} vertices: {:?}",
        shown.cracks.len(),
        shown.cracks.first()
    );
    let landed: BTreeMap<TileKey, Vec<Vec3>> = landed
        .iter()
        .map(|(key, bits)| (*key, from_bits(bits)))
        .collect();
    let (mut moved, mut off, mut worst) = (0.0_f32, 0.0_f32, None);
    for (key, drawn) in &shown.finest {
        let was = from_bits(&before[key]);
        for ((p, l), b) in drawn.iter().zip(&landed[key]).zip(&was) {
            moved = moved.max((*l - *b).length());
            let gap = (*p - *l).length();
            if gap > off {
                off = gap;
                worst = Some(*key);
            }
        }
    }
    assert!(
        moved > 0.01,
        "{what}: the strokes moved the surface: {moved}"
    );
    assert!(
        off < moved * 1e-3,
        "{what}: a finest preview ({worst:?}) is {off} from what landed, of {moved} moved"
    );
    let finest = Level::new(LEVEL).expect("a level");
    for (key, preview) in &shown.display {
        let error = farthest(&landed, finest, key, preview);
        let bound = f64::from(preview.bound);
        assert!(
            error <= bound,
            "{what}: display {key:?} is {error} from what landed, past its bound {bound}"
        );
        let vertex = farthest_vertex(&landed, finest, key, &preview.positions);
        let recorded = 2.0 * records[key] + moved * 1e-3;
        assert!(
            vertex <= recorded,
            "{what}: a vertex of display {key:?} is {vertex} from what landed, past {recorded}"
        );
    }
}

#[test]
fn a_later_strokes_previews_show_both_strokes_as_they_will_land() {
    // The second crosses the first: tiles the first previewed that the second's dabs reach.
    let mut run = Chain::new(Instant::now(), None);
    run.build_display();
    let before = positions(&mut run.project);
    let records = record_errors(run.project.store(), &before);
    run.frame(&stroke(ACROSS, 12));
    run.frame(&crossing());
    assert_eq!(run.ops().len(), 2, "both under way");
    let shown = Previews::of(run.project.store());
    run.cut();
    for _ in 0..3 {
        assert_eq!(run.land_one().dabs_applied, 1, "the first lands a dab");
    }
    let part_landed = Previews::of(run.project.store());
    // The first lands whole: the second's op begins, its previews carried on over what landed.
    let mut steps = 0;
    while run.ops().len() > 1 {
        run.land_one();
        steps += 1;
        assert!(steps < 10_000, "the first lands");
    }
    let first_landed = Previews::of(run.project.store());
    run.cut();
    assert_eq!(run.land_one().dabs_applied, 1, "the second lands a dab");
    let second_landing = Previews::of(run.project.store());
    run.land_all();
    let landed = run.close();
    assert_shown_as_landed(&shown, &before, &records, &landed, "shown");
    assert_shown_as_landed(&part_landed, &before, &records, &landed, "part landed");
    assert_shown_as_landed(&first_landed, &before, &records, &landed, "first landed");
    assert_shown_as_landed(
        &second_landing,
        &before,
        &records,
        &landed,
        "second landing",
    );
}

/// A stroke along the crest of the ridge a stroke along [`ACROSS`] raises.
const ALONG_THE_RIDGE: ((f64, f64), (f64, f64)) = ((140.0, 101.5), (180.0, 108.5));

#[test]
fn a_small_stroke_on_a_ridge_taller_than_its_radius_is_shown_on_the_ridge() {
    // A broad stroke raises a ridge many times the small brush's radius; the small one draws
    // along its crest while the ridge lands, and is shown on it, then as part of it lands.
    let mut run = Chain::new(Instant::now(), None);
    run.build_display();
    let before = positions(&mut run.project);
    let records = record_errors(run.project.store(), &before);
    run.brush.set_radius(24.0);
    run.frame(&to_and_fro(3));
    run.frame(&[sample(Phase::Up, ACROSS.0, 0.0)]);
    let ridge = Previews::of(run.project.store());
    let raised = ridge.finest.iter().flat_map(|(key, shown)| {
        let was = from_bits(&before[key]);
        let gaps = shown.iter().zip(was).map(|(p, b)| (*p - b).length());
        let gaps: Vec<f32> = gaps.collect();
        gaps
    });
    let height = raised.fold(0.0_f32, f32::max);
    run.brush.set_radius(3.0);
    run.frame(&stroke(ALONG_THE_RIDGE, 10));
    assert_eq!(run.ops().len(), 2, "both under way");
    let small: Vec<DabRow> = run.rows().last().cloned().unwrap_or_default();
    let radius = small
        .iter()
        .map(|row| {
            let dab: Dab = bytemuck::cast(*row);
            dab.radius
        })
        .fold(0.0_f32, f32::max);
    assert!(radius > 0.0, "the small stroke placed dabs");
    assert!(
        height > 2.0 * radius,
        "a ridge {height} high, past the small brush's radius {radius}"
    );
    let shown = Previews::of(run.project.store());
    run.cut();
    for _ in 0..3 {
        assert_eq!(run.land_one().dabs_applied, 1, "the ridge lands a dab");
    }
    let part_landed = Previews::of(run.project.store());
    run.land_all();
    let landed = run.close();
    assert_shown_as_landed(&shown, &before, &records, &landed, "shown");
    assert_shown_as_landed(&part_landed, &before, &records, &landed, "part landed");
}

#[test]
fn a_stroke_over_one_that_drops_dabs_goes_with_it() {
    let mut run = Chain::new(Instant::now(), None);
    run.frame(&stroke(ACROSS, 12));
    run.frame(&crossing());
    let ops = run.ops();
    let [_, later] = ops.as_slice() else {
        panic!("two strokes under way: {ops:?}");
    };
    let later = *later;
    run.cut();
    run.brush.knob = Knob::LandingFails;
    let failed = run.land_one();
    assert_eq!(failed.dabs_applied, 0, "the first's landing failed");
    assert!(run.ops().is_empty(), "both strokes ended");
    assert!(run.brush.drops() > 0, "dabs dropped");
    let error = run.brush.error().map(str::to_owned);
    assert!(
        error
            .as_deref()
            .is_some_and(|error| error.contains("applying dabs")),
        "{error:?}"
    );
    run.brush.knob = Knob::None;
    run.cut();
    let Chain {
        dir,
        project,
        brush,
        ..
    } = run;
    drop(brush);
    let path = project.recovery_path().expect("a recovery file").to_owned();
    project.close(true).expect("closed");
    let batches = batches_at(&path, later);
    let last = batches.last().expect("the second was journalled");
    assert!(last.placed() > 0, "{last:?}");
    assert_eq!(
        last.landed(),
        last.placed(),
        "its journal closes counting every dab landed: a restore lands none"
    );
    drop(dir);
}

/// A `Down` whose futures run past the share partway waits, and lands as waiting.
///
/// The store takes one future, then refuses the next, in one of `on_futures`'s rounds: the future
/// installed is let go with its op; the stroke begins once the one before has landed, and leaves
/// the tiles of waiting.
///
/// The test sphere is too small for the same refusal mid-stroke: a `Down` there reaches every
/// tile the earlier stroke changes. The mid-stroke wait is driven by the knob
/// (`a_stroke_whose_futures_cant_be_had_waits_its_samples_and_places_them_as_waiting`).
#[test]
fn a_down_whose_futures_run_past_the_share_partway_waits_and_lands_as_waiting() {
    let epoch = Instant::now();
    let (first, second) = (stroke(ACROSS, 12), crossing());
    let mut run = Chain::new(epoch, None);
    run.frame(&first);
    installs_left(Some(1));
    run.frame(&second);
    let left = INSTALLS_LEFT.get();
    installs_left(None);
    assert_eq!(
        left,
        Some(0),
        "a round installed a future, and the next was refused"
    );
    assert!(!run.brush.deferred.is_empty(), "the second waits");
    assert_eq!(run.ops().len(), 1, "nothing of it reserved");
    assert_eq!(run.brush.error(), None, "nothing dropped");
    run.land_all();
    assert!(run.brush.deferred.is_empty(), "it began and landed");
    let chained_tiles = run.close();
    let (_, waited_tiles) = waited(epoch);
    assert!(chained_tiles == waited_tiles, "the same tiles, bit for bit");
}

#[test]
fn samples_waiting_for_futures_are_placed_by_a_flush_as_waiting() {
    let epoch = Instant::now();
    let (first, second) = (stroke(ACROSS, 12), crossing());
    let (chained, chained_tiles) = drawn(epoch, |run| {
        run.frame(&first);
        let (early, late) = second.split_at(3);
        run.frame(early);
        run.brush.knob = Knob::FuturesFail;
        run.frame(late);
        assert_eq!(run.brush.deferred.len(), late.len(), "its samples wait");
        // `drawn` flushes with them still waiting, the futures still unavailable: the flush
        // lands the first, then places them on the second's own surface.
    });
    assert_eq!(chained.len(), 2, "both under way as the flush began");
    let (_, waited_tiles) = waited(epoch);
    assert!(chained_tiles == waited_tiles, "the same tiles, bit for bit");
}

/// A finer level for
/// `a_stroke_over_a_long_backlog_shows_its_first_frame_installing_only_its_own`: a face of 4 × 4
/// tiles, so an earlier stroke changes many more tiles than a later one reads.
const BACKLOG_LEVEL: u8 = 8;

/// How many futures `a_stroke_over_a_long_backlog_shows_its_first_frame_installing_only_its_own`
/// lets its brush install: far more than it does, so what is left counts them.
const COUNTED: usize = 1 << 20;

#[test]
fn a_stroke_over_a_long_backlog_shows_its_first_frame_installing_only_its_own() {
    // A long stroke still to land, then a short one over its start: the short one's dabs show in
    // its first frame, and the futures it installs are a few of the tiles the long one changes,
    // not its whole backlog.
    let config =
        StoreConfig::new(256 << 20, 1).with_display(Level::new(BACKLOG_LEVEL).expect("a level"));
    let mut run = Chain::new(Instant::now(), Some(config));
    run.build_display();
    run.frame(&stroke(((110.0, 100.0), (210.0, 110.0)), 60));
    run.brush.set_radius(4.0);
    let short = stroke(((112.0, 100.0), (116.0, 100.5)), 3);
    installs_left(Some(COUNTED));
    let second = run.frame(&short);
    let left = INSTALLS_LEFT.get();
    installs_left(None);
    let installed = COUNTED - left.expect("counted");
    let cracks = Previews::of(run.project.store()).cracks;
    assert!(
        cracks.is_empty(),
        "the previews meet bit for bit, but not at {} vertices: {:?}",
        cracks.len(),
        cracks.first()
    );
    assert_eq!(run.ops().len(), 2, "both under way");
    let placed = run.rows().last().map_or(0, Vec::len);
    assert!(placed > 0, "the short stroke placed dabs");
    assert_eq!(
        second.dabs_shown, placed,
        "its first frame shows every dab it placed"
    );
    let backlog = run.brush.strokes.first().map_or(0, |long| {
        long.remaining
            .as_ref()
            .map_or(0, |remaining| remaining.reach.len())
    });
    assert!(
        installed < backlog,
        "it installed {installed} futures of the {backlog} tiles the long stroke changes"
    );
    // Frames on, a call's futures at a time, the display previews become the short stroke's:
    // meeting the finest previews and each other bit for bit all the while.
    for frame in 0..20 {
        run.frame(&[]);
        let previews = Previews::of(run.project.store());
        assert!(
            previews.cracks.is_empty(),
            "frame {frame}: the previews meet bit for bit, but not at {} vertices: {:?}",
            previews.cracks.len(),
            previews.cracks.first()
        );
    }
    run.land_all();
    run.close();
}

#[test]
fn a_down_routed_over_a_landing_stroke_begins_with_the_stroke_routing_began() {
    // Routed, then stepped, as a frame does: the step installs no future the routing didn't, and
    // the stroke places the dabs and leaves the tiles of one that waited.
    let epoch = Instant::now();
    let (first, second) = (stroke(ACROSS, 12), crossing());
    let (chained, chained_tiles) = drawn(epoch, |run| {
        run.frame(&first);
        let (down, rest) = second.split_first().expect("a Down");
        assert_eq!(run.route(down), Route::Brush);
        assert!(run.routed_over(), "the stroke is begun, its op reserved");
        installs_left(Some(COUNTED));
        run.frame(std::slice::from_ref(down));
        let left = INSTALLS_LEFT.get();
        installs_left(None);
        assert_eq!(left, Some(COUNTED), "its `Down` installs no future again");
        assert!(!run.routed_over(), "taken by its stroke");
        assert_eq!(run.ops().len(), 2, "begun over the first");
        run.frame(rest);
    });
    let (waited_rows, waited_tiles) = waited(epoch);
    assert_eq!(chained, waited_rows, "the dabs of waiting");
    assert!(chained_tiles == waited_tiles, "the same tiles, bit for bit");
}

/// A [`Chain`] with a stroke along [`ACROSS`] under way, none of it landed, and the `Down` of
/// [`crossing`] routed to the brush over it: the stroke routing began waits for its step.
fn routed_over_one() -> (Chain, PenSample) {
    let mut run = Chain::new(Instant::now(), None);
    run.frame(&stroke(ACROSS, 12));
    let down = *crossing().first().expect("a Down");
    assert_eq!(run.route(&down), Route::Brush);
    assert!(run.routed_over(), "the stroke is begun, its op reserved");
    (run, down)
}

/// Ends a [`routed_over_one`] run whose `Down` began no stroke: nothing is reserved, the first
/// stroke lands, and the recovery file holds batches of no op but its own.
fn assert_given_back(mut run: Chain) {
    assert!(!run.routed_over(), "given back");
    run.assert_nothing_reserved();
    let ops = run.ops();
    let [first] = ops.as_slice() else {
        panic!("the first stroke alone under way: {ops:?}");
    };
    let first = *first;
    run.cut();
    let path = run
        .project
        .recovery_path()
        .expect("a recovery file")
        .to_owned();
    run.land_all();
    let Chain { dir, project, .. } = run;
    project.close(true).expect("closed");
    let file = ProjectFile::open(&path).expect("the recovery file opens");
    let batches = file.batches().map(|batch| batch.expect("a batch"));
    let others: Vec<OpId> = batches
        .map(|batch| batch.op())
        .filter(|op| *op != first)
        .collect();
    assert!(others.is_empty(), "batches of ops never begun: {others:?}");
    drop(dir);
}

#[test]
fn a_down_routed_off_the_futures_navigates_and_reserves_nothing() {
    let mut run = Chain::new(Instant::now(), None);
    run.frame(&stroke(ACROSS, 12));
    let off = sample(Phase::Down, (3.0, 3.0), 0.8);
    assert_eq!(run.route(&off), Route::Navigate);
    assert!(run.brush.routed_off_surface(), "nothing under it");
    assert_given_back(run);
}

#[test]
fn a_down_routed_and_claimed_elsewhere_gives_its_op_back_as_the_step_ends() {
    let (mut run, _) = routed_over_one();
    run.frame(&[]);
    assert_given_back(run);
}

#[test]
fn a_down_routed_and_held_for_a_frame_with_a_view_gives_its_op_back() {
    let (mut run, down) = routed_over_one();
    let mut pen = PenFrame::default();
    pen.push(down, Some(Route::Brush));
    let blind = BrushContext {
        project: &mut run.project,
        index: None,
        camera: &run.camera,
        size: size(),
        drawn: Drawing::ALL,
        waiting: &[],
    };
    run.brush
        .step(blind, &pen, &BrushBudget::unlimited())
        .expect("a step");
    assert_eq!(run.brush.held.len(), 1, "the Down is held");
    assert!(!run.routed_over(), "given back");
    run.assert_nothing_reserved();
    // The frame that can place it begins the stroke afresh.
    run.frame(&crossing()[1..]);
    assert_eq!(run.ops().len(), 2, "begun over the first");
    run.land_all();
    run.close();
}

#[test]
fn a_down_routed_and_deferred_for_a_save_gives_its_op_back() {
    let mut run = Chain::new(Instant::now(), None);
    run.frame(&stroke(ACROSS, 12));
    run.brush.begin_saving(Instant::now());
    let down = sample(Phase::Down, (150.0, 80.0), 0.8);
    assert!(down.received > run.brush.save_asked.expect("asked"));
    assert_eq!(run.route(&down), Route::Brush);
    assert!(run.routed_over(), "the stroke is begun, its op reserved");
    run.frame(&[down]);
    assert_eq!(run.brush.deferred.len(), 1, "it waits for the save");
    assert!(!run.routed_over(), "given back");
    run.assert_nothing_reserved();
    run.brush.abandon();
    run.close();
}

#[test]
fn a_down_routed_as_every_stroke_before_it_ends_begins_alone() {
    let (mut run, _) = routed_over_one();
    // The catch-up, off the frame's step, lands the first whole before the `Down` is taken.
    let began = Instant::now();
    run.cut();
    while !run.ops().is_empty() {
        // Its pen quiet for long and the catch-up starved: each call lands a dab.
        let long_ago = Instant::now()
            .checked_sub(RESTING_QUIET * 2)
            .expect("after the epoch");
        run.brush.landed_at = long_ago;
        run.brush.unframed = false;
        for stroke in &mut run.brush.strokes {
            stroke.placed_at = long_ago;
            stroke.last.received = long_ago;
        }
        let prepared = context(&mut run.project, run.picker.index_mut(), &run.camera);
        run.brush
            .prepare(prepared, Instant::now())
            .expect("prepared");
        assert!(began.elapsed() < Duration::from_secs(60), "the first lands");
    }
    assert!(run.routed_over(), "still routed: no step ended");
    run.frame(&crossing());
    assert!(!run.routed_over(), "given back");
    let strokes = &run.brush.strokes;
    let [only] = strokes.as_slice() else {
        panic!("one stroke under way: {}", strokes.len());
    };
    assert!(
        matches!(only.op, StrokeOp::Begun(_)),
        "begun alone, on what the first left"
    );
    assert_eq!(run.brush.error(), None);
    run.land_all();
    run.close();
}

#[test]
fn a_second_down_routed_gives_the_firsts_op_back_and_a_close_or_an_abandon_the_last() {
    let (mut run, _) = routed_over_one();
    let again = sample(Phase::Down, (160.0, 110.0), 0.8);
    assert_eq!(run.route(&again), Route::Brush);
    assert!(run.routed_over(), "the second's stands");
    let closing = context(&mut run.project, run.picker.index_mut(), &run.camera);
    run.brush.begin_closing(closing).expect("closing");
    assert!(!run.routed_over(), "a close gives it back");
    run.assert_nothing_reserved();
    run.land_all();
    run.close();

    let (mut abandoned, _) = routed_over_one();
    abandoned.brush.abandon();
    assert!(!abandoned.routed_over(), "an abandon lets it go");
    abandoned.close();
}

/// How many times each timed case is drawn ([`time_downs`]): the median and the spread are said.
const TIMED_RUNS: usize = 7;

/// The most hover frames a warmed case is given before its `Down` ([`Timed::run`]).
const HOVER_FRAMES: usize = 2000;

/// The brush's radius in the ordinary timed cases, in pixels: about a tile of the 4 × 4 a face,
/// so a pass across the sphere places some eighty dabs.
const ORDINARY_RADIUS: f32 = 8.0;

/// The median of `times`, with the least and the most.
fn spread(times: &[Duration]) -> String {
    let mut sorted = times.to_vec();
    sorted.sort_unstable();
    let at = |place: usize| sorted.get(place).copied().unwrap_or_default();
    let last = sorted.len().saturating_sub(1);
    format!(
        "{:.2?} ({:.2?} to {:.2?})",
        at(sorted.len() / 2),
        at(0),
        at(last)
    )
}

/// The machine's load averages as it says them, for a measurement's record.
fn load() -> String {
    let said = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
    let averages: Vec<&str> = said.split_whitespace().take(3).collect();
    averages.join(" ")
}

/// How the surface under a timed `Down` stands as it comes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Before {
    /// The earlier stroke still lands, and the pen comes down with no hover before: every
    /// future the `Down` reads is made in it.
    Cold,
    /// The earlier stroke still lands, and the pen hovered where it comes down until the
    /// brush's warming had nothing left.
    Warmed,
    /// The earlier stroke has landed and ended: the `Down` of a stroke alone, what the same
    /// samples cost with nothing to read of another stroke.
    Landed,
}

/// A `Down` timed over an earlier stroke ([`Timed::run`]).
#[derive(Debug, Clone)]
struct Timed {
    /// The brush's radius, in pixels, for both strokes; `None` for the default.
    radius: Option<f32>,
    /// The earlier stroke's samples, `Down` to `Up`.
    earlier: Vec<PenSample>,
    /// Whether half the earlier stroke's dabs land before the `Down`.
    half_landed: bool,
    /// Where the pen comes down, and its first drag.
    pen: ((f64, f64), (f64, f64)),
}

/// What one run of a timed `Down` took ([`Timed::run`]).
#[derive(Debug, Clone, Copy)]
struct Took {
    /// Routing the `Down`.
    route: Duration,
    /// The step that takes the `Down` and its first drag, and shows its first dab.
    step: Duration,
    /// The futures installed for the stroke by the end of that step.
    installed: usize,
    /// The earlier stroke's dabs still to land as the pen came down.
    backlog: usize,
    /// The tiles those dabs reach.
    reach: usize,
    /// The hover frames before the `Down`, and what their warming took in all.
    hover: (usize, Duration),
}

impl Timed {
    /// The heavy case: a stroke to and fro over the sphere's middle, 64 passes, none of it
    /// landed, and a `Down` on its middle.
    fn heavy() -> Self {
        let mut earlier = to_and_fro(64);
        earlier.push(sample(Phase::Up, ACROSS.0, 0.0));
        Self {
            radius: None,
            earlier,
            half_landed: false,
            pen: ((160.0, 105.0), (163.0, 105.5)),
        }
    }

    /// An ordinary case: one pass across the sphere with a brush about a tile wide
    /// ([`ORDINARY_RADIUS`]), half of it landed, and a `Down` at `pen`.
    fn ordinary(pen: ((f64, f64), (f64, f64))) -> Self {
        Self {
            radius: Some(ORDINARY_RADIUS),
            earlier: stroke(((110.0, 100.0), (210.0, 110.0)), 60),
            half_landed: true,
            pen,
        }
    }

    /// Draws the case once, on a face of 4 × 4 tiles, and times its `Down`: routed, then
    /// stepped with its first drag, as a frame does.
    ///
    /// `prepare` has kept up what a later stroke reads of the earlier one, as the frames after
    /// it would. A warmed `Down` comes after hover frames at its pixel, each a step and a
    /// `prepare` with a 4 ms deadline, until `prepare` has no more to do; the catch-up is held
    /// meanwhile (as during a close), so a warmed `Down` is over the same dabs as a cold one.
    fn run(&self, before: Before) -> Took {
        let level = Level::new(BACKLOG_LEVEL).expect("a level");
        let config = StoreConfig::new(256 << 20, 1).with_display(level);
        let mut run = Chain::new(Instant::now(), Some(config));
        run.build_display();
        if let Some(radius) = self.radius {
            run.brush.set_radius(radius);
        }
        run.frame(&self.earlier);
        let placed = run.rows().first().map_or(0, Vec::len);
        if self.half_landed {
            run.cut();
            for _ in 0..placed / 2 {
                assert_eq!(run.land_one().dabs_applied, 1, "a dab lands");
            }
        }
        if before == Before::Landed {
            run.cut();
            run.land_all();
        }
        let kept_up = context(&mut run.project, run.picker.index_mut(), &run.camera);
        run.brush
            .prepare(kept_up, Instant::now())
            .expect("prepared");
        let backlog = run.rows().first().map_or(0, Vec::len);
        let reach = run.brush.strokes.first().map_or(0, |earlier| {
            let remaining = earlier.remaining.as_ref();
            remaining.map_or(0, |remaining| remaining.reach.len())
        });
        let (at, then) = self.pen;
        let mut hover = (0, Duration::ZERO);
        if before == Before::Warmed {
            run.brush.landing = Landing::InSteps;
            let mut pen = PenFrame::default();
            pen.push(sample(Phase::Hover, at, 0.0), None);
            let budget = BrushBudget::frame(Instant::now(), Duration::ZERO, setup().interval);
            for _ in 0..HOVER_FRAMES {
                let stepped = context(&mut run.project, run.picker.index_mut(), &run.camera);
                run.brush.step(stepped, &pen, &budget).expect("a step");
                let prepared = context(&mut run.project, run.picker.index_mut(), &run.camera);
                let began = Instant::now();
                let more = run
                    .brush
                    .prepare(prepared, began + Duration::from_millis(4))
                    .expect("prepared");
                hover = (hover.0 + 1, hover.1 + began.elapsed());
                if !more {
                    break;
                }
            }
            run.brush.landing = Landing::InFrames;
        }
        let down = sample(Phase::Down, at, 0.8);
        let drag = sample(Phase::Drag, then, 0.8);
        let began = Instant::now();
        let routed = run.route(&down);
        let route = began.elapsed();
        assert_eq!(routed, Route::Brush, "on the surface");
        let stepped = Instant::now();
        let first = run.frame(&[down, drag]);
        let step = stepped.elapsed();
        assert!(first.dabs_shown > 0, "its first dab is shown: {first:?}");
        let over = if before == Before::Landed { 1 } else { 2 };
        assert_eq!(run.ops().len(), over, "it began, over the earlier stroke");
        let installed = run
            .brush
            .strokes
            .last()
            .map_or(0, |newest| match &newest.op {
                StrokeOp::Reserved { installed, .. } => installed.len(),
                StrokeOp::Begun(_) => 0,
            });
        run.brush.abandon();
        run.close();
        Took {
            route,
            step,
            installed,
            backlog,
            reach,
            hover,
        }
    }
}

/// Times `case`'s `Down` with the surface under it as each of `befores` says, [`TIMED_RUNS`]
/// times each, and prints the medians with their spread and the machine's load: a measurement,
/// asserting nothing of time.
fn time_downs(name: &str, case: &Timed, befores: &[Before]) {
    println!("{name} (load {})", load());
    for &before in befores {
        let runs: Vec<Took> = (0..TIMED_RUNS).map(|_| case.run(before)).collect();
        let of = |part: fn(&Took) -> Duration| -> Vec<Duration> { runs.iter().map(part).collect() };
        let Some(last) = runs.last() else {
            return;
        };
        println!(
            "  {before:?}: over {} dabs still to land on {} tiles, {} futures installed; routed \
             in {}, stepped to its first dab in {}, {} in all; {} hover frames warming {:.2?}",
            last.backlog,
            last.reach,
            last.installed,
            spread(&of(|took| took.route)),
            spread(&of(|took| took.step)),
            spread(&of(|took| took.route + took.step)),
            last.hover.0,
            last.hover.1,
        );
    }
    println!(
        "  (load {} after; medians of {TIMED_RUNS}, least to most)",
        load()
    );
}

/// A measurement, printed and asserting nothing of time: a heavy stroke, then at once a stroke
/// over it, from its `Down`'s routing to the step that places and shows its first dab.
///
/// The heavy stroke goes to and fro over the sphere's middle on a face of 4 × 4 tiles, none of it
/// landed: its dabs dig far past the sphere's middle, so the `Down` on it, its radius frozen at
/// that depth, reaches every tile the stroke changes.
#[test]
#[ignore = "a measurement, printed: run it alone on a quiet machine, with --no-capture"]
fn a_down_over_a_heavy_stroke_is_timed() {
    let befores = [Before::Cold, Before::Warmed];
    time_downs("a Down over a heavy stroke", &Timed::heavy(), &befores);
}

/// The reference of `a_down_over_a_heavy_stroke_is_timed`: the same `Down` once the heavy stroke
/// has landed, a stroke alone. Apart, as landing the heavy stroke each time takes long.
#[test]
#[ignore = "a measurement, printed: run it alone on a quiet machine, with --no-capture"]
fn a_down_once_a_heavy_stroke_has_landed_is_timed() {
    let befores = [Before::Landed];
    time_downs(
        "a Down once a heavy stroke has landed",
        &Timed::heavy(),
        &befores,
    );
}

/// A measurement, printed and asserting nothing of time: an ordinary stroke across the sphere,
/// half of it landed, then a stroke on the half still to land, and another beside it.
#[test]
#[ignore = "a measurement, printed: run it alone on a quiet machine, with --no-capture"]
fn a_down_over_an_ordinary_stroke_is_timed() {
    let befores = [Before::Cold, Before::Warmed, Before::Landed];
    let over = Timed::ordinary(((185.0, 107.5), (188.0, 107.8)));
    let name = "a Down on an ordinary stroke's half still to land";
    time_downs(name, &over, &befores);
    let beside = Timed::ordinary(((185.0, 122.0), (188.0, 122.3)));
    time_downs("a Down beside an ordinary stroke", &beside, &befores);
}
