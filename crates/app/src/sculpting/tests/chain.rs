//! Strokes over strokes still landing: a `Down` while an earlier stroke's dabs land begins at once,
//! on the surface that stroke will leave (the stroke-while-landing spec).
//!
//! Today's behaviour, a stroke that waits for the one before it to land, is the oracle: the same
//! samples, placed once every earlier stroke has landed, give the same dabs, bit for bit, and the
//! same tiles once everything has landed.

use geom::Vec3;
use sculpt::{Future, Futures};
use store::{DabBatch, DisplayPreview, FLUSH_INTERVAL, PenDown};

use super::*;
use crate::frame::BETWEEN_SLICE;
use crate::sculpting::ahead::SEAMS_A_CALL;
use crate::sculpting::over::{
    Moves, Reach, Read, box_distance_squared, chain_of, needed, ray_meets_box, rounded_up,
};
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
    /// When the pen's last hover ([`Chain::hover`]) was reported.
    hovered: Instant,
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
            hovered: Instant::now(),
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
    ///
    /// Its report is the step's own: what a `prepare` before it caught up, which a step's report
    /// would carry too, is taken off first, so it says one dab applied whenever it lands one.
    fn land_one(&mut self) -> BrushReport {
        self.brush.caught_up = BrushReport::default();
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

    /// Lands `count` dabs of the oldest stroke, a step each ([`Chain::land_one`]).
    fn land_some(&mut self, count: usize) {
        for _ in 0..count {
            assert_eq!(self.land_one().dabs_applied, 1, "a dab lands");
        }
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

    /// Hovers the pen at `at` until the brush's warming has nothing left: a frame's step and a
    /// `prepare` with a 4 ms deadline at a time, as the frame loop does. Returns how many hover
    /// frames it took, and their `prepare`s' time in all.
    ///
    /// The catch-up is held meanwhile (as during a close), so nothing lands: the `Down` after it
    /// is over the same dabs as one with no hover before. The hover is reported a second after
    /// the one before it, wherever that was: a pen come there at its leisure, never fast.
    fn hover(&mut self, at: (f64, f64)) -> (usize, Duration) {
        let landing = std::mem::replace(&mut self.brush.landing, Landing::InSteps);
        self.hovered = self.hovered.max(Instant::now()) + Duration::from_secs(1);
        let mut hover = sample(Phase::Hover, at, 0.0);
        hover.timestamp = self.hovered;
        hover.received = self.hovered;
        let mut pen = PenFrame::default();
        pen.push(hover, None);
        let budget = BrushBudget::frame(Instant::now(), Duration::ZERO, setup().interval);
        let mut took = (0, Duration::ZERO);
        loop {
            let stepped = context(&mut self.project, self.picker.index_mut(), &self.camera);
            self.brush.step(stepped, &pen, &budget).expect("a step");
            let prepared = context(&mut self.project, self.picker.index_mut(), &self.camera);
            let began = Instant::now();
            let more = self
                .brush
                .prepare(prepared, began + Duration::from_millis(4))
                .expect("prepared");
            took = (took.0 + 1, took.1 + began.elapsed());
            if !more {
                break;
            }
            assert!(took.0 < HOVER_FRAMES, "the warming ends");
        }
        // The seams about those futures are noted in calls with time left, which the call
        // that made the last future may not have had: the frames after it, as a loop brings
        // them while strokes land.
        for _ in 0..HOVER_FRAMES {
            if self.brush.ahead.seamed >= self.brush.ahead.listed.len() {
                break;
            }
            let prepared = context(&mut self.project, self.picker.index_mut(), &self.camera);
            let by = Instant::now() + Duration::from_millis(4);
            self.brush.prepare(prepared, by).expect("prepared");
        }
        self.brush.landing = landing;
        took
    }

    /// A `prepare` of the brush with no time to spare, as after a frame's hand-off: what a later
    /// stroke reads of those under way is kept up, and each piece of work off the pen's path gets
    /// its one batch.
    ///
    /// The catch-up is held meanwhile (as during a close): whether it would land a dab depends
    /// on whether the stroke's pen-down flush has finished, which a test can't time. What lands
    /// is what the test lands ([`Chain::land_one`]).
    fn prepare(&mut self) {
        let landing = std::mem::replace(&mut self.brush.landing, Landing::InSteps);
        let prepared = context(&mut self.project, self.picker.index_mut(), &self.camera);
        self.brush
            .prepare(prepared, Instant::now())
            .expect("prepared");
        self.brush.landing = landing;
    }

    /// The tiles whose futures are installed for the newest stroke, its op reserved; none for one
    /// whose op has begun.
    fn installed(&self) -> BTreeSet<TileKey> {
        let newest = self.brush.strokes.last();
        newest.map_or_else(BTreeSet::new, |newest| match &newest.op {
            StrokeOp::Reserved { installed, .. } => installed.clone(),
            StrokeOp::Begun(_) => BTreeSet::new(),
        })
    }

    /// A frame's step with the pen hovering at `at`, as its tablet `reported` it then: the brush
    /// notes where it hovers, and how fast it got there.
    fn hover_step(&mut self, at: (f64, f64), reported: Instant) {
        let mut hover = sample(Phase::Hover, at, 0.0);
        hover.timestamp = reported;
        hover.received = reported;
        let mut pen = PenFrame::default();
        pen.push(hover, None);
        let budget = BrushBudget::frame(Instant::now(), Duration::ZERO, setup().interval);
        let stepped = context(&mut self.project, self.picker.index_mut(), &self.camera);
        self.brush.step(stepped, &pen, &budget).expect("a step");
    }

    /// `prepare`s of the brush, each with a 4 ms deadline, until it wants no more.
    fn prepare_out(&mut self) {
        for _ in 0..HOVER_FRAMES {
            let prepared = context(&mut self.project, self.picker.index_mut(), &self.camera);
            let by = Instant::now() + Duration::from_millis(4);
            if !self.brush.prepare(prepared, by).expect("prepared") {
                return;
            }
        }
        panic!("the brush's prepare never ends");
    }

    /// A `prepare` of the brush whose deadline has passed, the catch-up free to land: whether
    /// it wants more.
    fn prepare_late(&mut self) -> bool {
        let prepared = context(&mut self.project, self.picker.index_mut(), &self.camera);
        self.brush
            .prepare(prepared, Instant::now())
            .expect("prepared")
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

/// Where a sample's pen is, as the tests give it.
fn pixel_of(sample: &PenSample) -> (f64, f64) {
    (sample.at.x, sample.at.y)
}

#[test]
fn a_down_where_the_pen_hovered_over_a_landing_stroke_makes_no_future_and_draws_as_waiting() {
    let epoch = Instant::now();
    let (first, second) = (stroke(ACROSS, 12), crossing());
    let (chained, chained_tiles) = drawn(epoch, |run| {
        run.frame(&first);
        // Part of the first has landed, for sure: its cut is taken, and its dabs landed here.
        run.cut();
        run.land_some(3);
        let (down, rest) = second.split_first().expect("a Down");
        take_made();
        run.hover(pixel_of(down));
        let ahead = run.brush.ahead.tiles.len();
        assert!(ahead > 0, "futures are made as the pen hovers");
        assert_eq!(take_made(), ahead, "each made once");
        assert_eq!(run.brush.ahead.after, run.ops().last().copied());
        run.assert_nothing_reserved();
        // A still pen's are made once: more hover frames make none.
        run.hover(pixel_of(down));
        assert_eq!(take_made(), 0, "nothing more to make for a still pen");
        assert_eq!(run.route(down), Route::Brush);
        run.frame(std::slice::from_ref(down));
        assert_eq!(run.ops().len(), 2, "begun over the first");
        assert!(!run.installed().is_empty(), "its futures are installed");
        assert_eq!(
            take_made(),
            0,
            "the Down makes no future: those it reads were made as the pen hovered"
        );
        run.frame(rest);
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
fn futures_made_ahead_hold_as_their_stroke_lands_on_before_the_down() {
    // Warmed with nothing landed; then the first stroke lands on, some of its dabs on tiles
    // whose futures are made, the rest of those tiles' dabs still to land; then the `Down`: the
    // futures made before are what the landing since leaves, and are installed as they are.
    let epoch = Instant::now();
    let (first, second) = (stroke(ACROSS, 12), crossing());
    let (chained, chained_tiles) = drawn(epoch, |run| {
        run.frame(&first);
        let (down, rest) = second.split_first().expect("a Down");
        run.hover(pixel_of(down));
        let ahead: BTreeSet<TileKey> = run.brush.ahead.tiles.keys().copied().collect();
        assert!(!ahead.is_empty(), "futures are made as the pen hovers");
        // Each tile's dabs still to land, before and after some land.
        let to_land = |chain: &Chain| -> BTreeMap<TileKey, usize> {
            let earlier = chain.brush.strokes.first().expect("the first");
            let remaining = earlier.remaining.as_ref().expect("made as the pen hovered");
            let counts = remaining
                .reach
                .iter()
                .map(|(key, reach)| (*key, reach.dabs));
            counts.collect()
        };
        let before = to_land(run);
        run.cut();
        run.land_some(5);
        let after = to_land(run);
        // Tiles whose futures are made, some of their dabs landed since, the rest still to land.
        let part_landed = ahead.iter().filter(|key| {
            let (was, is) = (before.get(key).copied(), after.get(key).copied());
            was.zip(is).is_some_and(|(was, is)| is < was && is > 0)
        });
        let part_landed: BTreeSet<TileKey> = part_landed.copied().collect();
        assert!(
            !part_landed.is_empty(),
            "tiles with futures made, part of their dabs landed since: {before:?} then {after:?}"
        );
        take_made();
        assert_eq!(run.route(down), Route::Brush);
        run.frame(std::slice::from_ref(down));
        assert_eq!(run.ops().len(), 2, "begun over the first");
        let installed = run.installed();
        assert!(
            installed.intersection(&part_landed).next().is_some(),
            "a future made before the landing, of a tile part landed since, is installed"
        );
        assert_eq!(
            take_made(),
            0,
            "none made again: they hold as the stroke lands"
        );
        run.land_some(2);
        run.frame(rest);
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
fn futures_made_ahead_of_a_stroke_that_lands_whole_before_the_down_are_let_go() {
    let epoch = Instant::now();
    let (first, second) = (stroke(ACROSS, 12), crossing());
    let (chained, chained_tiles) = drawn(epoch, |run| {
        run.frame(&first);
        let down = second.first().expect("a Down");
        run.hover(pixel_of(down));
        assert!(!run.brush.ahead.tiles.is_empty(), "futures made ahead");
        run.cut();
        run.land_all();
        assert!(run.ops().is_empty(), "the first has landed and ended");
        take_made();
        run.frame(&second);
        assert_eq!(take_made(), 0, "a stroke alone reads the store");
        assert!(run.brush.ahead.tiles.is_empty(), "let go with the stroke");
        let strokes = &run.brush.strokes;
        let [only] = strokes.as_slice() else {
            panic!("one stroke under way: {}", strokes.len());
        };
        assert!(matches!(only.op, StrokeOp::Begun(_)), "begun alone");
        run.cut();
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
fn three_strokes_each_begun_where_the_pen_hovered_place_and_land_as_one_after_another() {
    // The futures made ahead of the second stroke are of the first alone: the third reads
    // those of both, made afresh as the pen hovers again.
    let strokes = [stroke(ACROSS, 12), crossing(), over_both()];
    let epoch = Instant::now();
    let mut chained = Chain::new(epoch, None);
    for samples in &strokes {
        if let Some(newest) = chained.ops().last().copied() {
            let down = samples.first().expect("a Down");
            chained.hover(pixel_of(down));
            assert_eq!(
                chained.brush.ahead.after,
                Some(newest),
                "made up to the newest"
            );
            assert!(!chained.brush.ahead.tiles.is_empty(), "futures made ahead");
        }
        chained.frame(samples);
    }
    let chained_rows = chained.rows();
    assert_eq!(chained_rows.len(), strokes.len(), "every stroke under way");
    chained.land_all();
    chained.frame(&[]);
    assert!(
        chained.brush.ahead.tiles.is_empty() && chained.brush.ahead.after.is_none(),
        "let go with the strokes"
    );
    let chained_tiles = chained.close();

    let mut waited = Chain::new(epoch, None);
    let mut waited_rows = Vec::new();
    for samples in &strokes {
        waited.frame(samples);
        waited_rows.extend(waited.rows());
        waited.land_all();
    }
    assert_eq!(chained_rows, waited_rows, "the same dabs, bit for bit");
    assert!(
        chained_tiles == waited.close(),
        "the same tiles, bit for bit"
    );
}

#[test]
fn the_futures_made_ahead_go_as_the_stroke_they_follow_ends_or_drops_dabs() {
    // Landed and ended: its futures are the tiles themselves.
    let mut run = Chain::new(Instant::now(), None);
    run.frame(&stroke(ACROSS, 12));
    run.hover((150.0, 80.0));
    assert!(!run.brush.ahead.tiles.is_empty(), "futures made ahead");
    run.cut();
    run.land_all();
    run.frame(&[]);
    assert!(run.brush.ahead.tiles.is_empty(), "gone with the stroke");
    assert_eq!(run.brush.ahead.after, None);
    assert!(run.brush.ahead.todo.is_empty(), "nothing left listed");
    // A stroke alone after it reads the store, and lands as ever.
    run.frame(&crossing());
    run.cut();
    run.land_all();
    run.close();

    // Dropped dabs: the futures were of dabs that never land.
    let mut failed = Chain::new(Instant::now(), None);
    failed.frame(&stroke(ACROSS, 12));
    failed.hover((150.0, 80.0));
    assert!(!failed.brush.ahead.tiles.is_empty(), "futures made ahead");
    failed.cut();
    failed.brush.knob = Knob::LandingFails;
    assert_eq!(failed.land_one().dabs_applied, 0, "the landing failed");
    assert!(failed.ops().is_empty(), "the stroke ended");
    failed.brush.knob = Knob::None;
    failed.frame(&[]);
    assert!(failed.brush.ahead.tiles.is_empty(), "gone with the stroke");
    failed.brush.abandon();
    failed.close();
}

#[test]
fn a_stroke_fed_a_sample_a_frame_with_its_futures_installed_ahead_places_the_dabs_of_waiting() {
    let epoch = Instant::now();
    let (first, second) = (stroke(ACROSS, 12), crossing());
    let (chained, chained_tiles) = drawn(epoch, |run| {
        run.frame(&first);
        // The first lands on between the second's frames, for sure: its cut is taken, and a dab
        // lands after every other sample, between the futures installed for the second.
        run.cut();
        let numbered = second.iter().enumerate();
        for (n, sample) in numbered {
            run.frame(std::slice::from_ref(sample));
            run.prepare();
            if n % 2 == 1 {
                run.land_some(1);
            }
        }
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

/// A [`Chain`] on a face of 4 × 4 tiles with an ordinary stroke across the sphere under way.
///
/// At [`BACKLOG_LEVEL`], none of the stroke landed, drawn with a brush of `radius` pixels (the
/// default with `None`): it changes many tiles, of which a later stroke reads a few.
fn across_many_tiles(radius: Option<f32>) -> Chain {
    let level = Level::new(BACKLOG_LEVEL).expect("a level");
    let config = StoreConfig::new(256 << 20, 1).with_display(level);
    let mut run = Chain::new(Instant::now(), Some(config));
    if let Some(radius) = radius {
        run.brush.set_radius(radius);
    }
    run.frame(&stroke(((110.0, 100.0), (210.0, 110.0)), 60));
    run.prepare();
    run
}

#[test]
fn between_frames_the_futures_ahead_of_a_drawing_pen_are_installed() {
    let mut run = across_many_tiles(None);
    run.brush.set_radius(4.0);
    // No previews: only the stroke's own reads ask for futures. Nothing lands meanwhile (held,
    // as during a close): a brush with no previews would land its dabs in its steps.
    run.brush.preview = None;
    run.brush.landing = Landing::InSteps;
    // Drawn along the first, a sample a frame, a `prepare` after each as the frame loop has.
    let along = stroke(((112.0, 100.0), (172.0, 106.0)), 40);
    let (up, drawn) = along.split_last().expect("an Up");
    let (mut in_steps, mut ahead_of_pen) = (0, 0);
    // Those made as the first stroke's pen lifted are let go: every future here is made, and
    // counted, as it is installed.
    run.brush.ahead.clear();
    take_made();
    for sample in drawn {
        run.frame(std::slice::from_ref(sample));
        in_steps += take_made();
        let before = run.installed();
        run.prepare();
        let after = run.installed();
        assert!(after.is_superset(&before), "none let go");
        let ahead = after.len() - before.len();
        assert_eq!(take_made(), ahead, "each made once, as it is installed");
        ahead_of_pen += ahead;
    }
    assert_eq!(run.ops().len(), 2, "begun over the first");
    assert!(
        ahead_of_pen > 0,
        "futures are installed ahead of the pen between its frames ({in_steps} in its steps)"
    );
    let changed = run.brush.strokes.first().map_or(0, |earlier| {
        let remaining = earlier.remaining.as_ref();
        remaining.map_or(0, |remaining| remaining.reach.len())
    });
    let installed = run.installed().len();
    assert_eq!(installed, in_steps + ahead_of_pen, "each future made once");
    assert!(
        installed < changed,
        "those the pen came near, not all {changed}"
    );
    run.frame(std::slice::from_ref(up));
    run.brush.abandon();
    run.close();
}

/// Every tile of [`BACKLOG_LEVEL`] of `store`'s mesh.
fn backlog_tiles(store: &TileStore) -> Vec<TileKey> {
    let level = Level::new(BACKLOG_LEVEL).expect("a level");
    let per_side = u16::try_from(level.tiles_per_side()).expect("a few");
    let tiles = store.mesh().faces().flat_map(|face| {
        (0..per_side).flat_map(move |ty| {
            (0..per_side).map(move |tx| TileKey::displacement(face, level, tx, ty).expect("a tile"))
        })
    });
    tiles.collect()
}

/// Reads across the view of `camera`, rays and balls.
///
/// A ray through each pixel of a grid, up to the sphere's near side, past it or without end, and
/// a ball about a point of each ray near the sphere's near side, of no radius up to a tile and
/// more.
fn reads_across(camera: &Camera) -> Vec<Read> {
    let projector = ViewProjector::new(&camera.view(size()).expect("a view"));
    let pixels = (0..12).flat_map(|i| (0..8).map(move |j| (i, j)));
    let rays = pixels.filter_map(|(i, j)| {
        let at = ScreenPoint::new(100.0 + 10.0 * f64::from(i), 70.0 + 9.0 * f64::from(j)).ok()?;
        projector.ray(at)
    });
    let rays: Vec<Ray> = rays.collect();
    assert!(rays.len() > 50, "rays through the view");
    let mut reads: Vec<Read> = Vec::new();
    let numbered = rays.iter().enumerate();
    for (n, ray) in numbered {
        let until = [1.9, 2.3, f64::INFINITY][n % 3];
        reads.push(Read::Ray { ray: *ray, until });
        let center = ray.origin() + ray.direction() * 2.0;
        let radius = [0.0, 0.02, 0.08, 0.3][n % 4];
        reads.push(Read::Ball {
            center,
            radius: rounded_up(center, radius),
        });
    }
    reads
}

/// Whether `read` (a ray up to its end, or a ball) meets `bounds` exactly, with no slack: the
/// tests' own reading of a [`Read`].
fn read_meets(read: &Read, bounds: store::Aabb) -> bool {
    let wide = |v: Vec3| DVec3::new(f64::from(v.x), f64::from(v.y), f64::from(v.z));
    let (low, high) = (wide(bounds.min()), wide(bounds.max()));
    match *read {
        Read::Ray { ray, until } => ray_meets_box(&ray, until, low, high),
        Read::Ball { center, radius } => box_distance_squared(center, low, high) <= radius * radius,
    }
}

/// Asserts that the box each tile is grown to holds its future, made now.
///
/// For every tile the strokes under way change ([`Moves::grown`]): what a read not reaching the
/// box relies on. Returns how many tiles' futures have a box (the tiles a dab touches).
fn assert_grown_holds_futures(run: &mut Chain, case: &str) -> usize {
    run.prepare();
    let moves = Moves::of(&run.brush.strokes);
    let changed: Vec<TileKey> = moves.tiles().copied().collect();
    let store = run.project.store();
    let chain = chain_of(&run.brush.strokes).expect("made by prepare");
    let futures = Futures::new(store, &run.brush.state, &chain).expect("a chain");
    let made = futures.futures_of(store, &changed);
    let index = run.picker.index_mut();
    let mut boxed = 0;
    let futures_made = changed.iter().zip(made);
    for (key, future) in futures_made {
        let Some(bounds) = future.expect("a future").bounds() else {
            continue;
        };
        boxed += 1;
        let (low, high) = moves.grown(index, key).expect("a tile of the index");
        let (least, most) = (bounds.min(), bounds.max());
        let inside = f64::from(least.x) >= low.x
            && f64::from(least.y) >= low.y
            && f64::from(least.z) >= low.z
            && f64::from(most.x) <= high.x
            && f64::from(most.y) <= high.y
            && f64::from(most.z) <= high.z;
        assert!(
            inside,
            "{case}, tile {key:?}: its future {bounds:?} isn't in its grown box {low:?} to {high:?}"
        );
    }
    boxed
}

#[test]
fn the_box_a_tile_is_grown_to_holds_its_future() {
    // An ordinary stroke, none landed, then half; a big brush's; one to and fro, its dabs piled
    // up; and a stroke over another, both still to land.
    let mut ordinary = across_many_tiles(Some(ORDINARY_RADIUS));
    assert!(assert_grown_holds_futures(&mut ordinary, "ordinary") > 4);
    ordinary.cut();
    ordinary.land_some(40);
    assert!(assert_grown_holds_futures(&mut ordinary, "half landed") > 2);
    ordinary.brush.abandon();
    ordinary.close();

    let mut big = across_many_tiles(None);
    assert!(assert_grown_holds_futures(&mut big, "a big brush") > 10);
    // A second stroke over it, placed on its futures: the two together.
    big.brush.set_radius(ORDINARY_RADIUS);
    big.frame(&stroke(((150.0, 80.0), (170.0, 125.0)), 30));
    assert_eq!(big.ops().len(), 2, "begun over the first");
    assert!(assert_grown_holds_futures(&mut big, "two strokes") > 10);
    big.brush.abandon();
    big.close();

    let level = Level::new(BACKLOG_LEVEL).expect("a level");
    let config = StoreConfig::new(256 << 20, 1).with_display(level);
    let mut piled = Chain::new(Instant::now(), Some(config));
    piled.brush.set_radius(ORDINARY_RADIUS);
    let mut fro = to_and_fro(6);
    fro.push(sample(Phase::Up, ACROSS.0, 0.0));
    piled.frame(&fro);
    assert!(assert_grown_holds_futures(&mut piled, "to and fro") > 4);
    piled.brush.abandon();
    piled.close();
}

#[test]
fn with_futures_at_hand_the_tiles_asked_for_lie_between_those_a_read_reaches_and_the_bound() {
    let mut run = across_many_tiles(Some(ORDINARY_RADIUS));
    // Every future of the stroke, and each tile's two boxes: as stored, and its future's.
    let changed: Vec<TileKey> = {
        let earlier = run.brush.strokes.first().expect("a stroke");
        let remaining = earlier.remaining.as_ref().expect("made by prepare");
        remaining.reach.keys().copied().collect()
    };
    let store = run.project.store();
    let chain = chain_of(&run.brush.strokes).expect("made by prepare");
    let futures = Futures::new(store, &run.brush.state, &chain).expect("a chain");
    let made = futures.futures_of(store, &changed);
    let all: BTreeMap<TileKey, Future> = changed
        .iter()
        .copied()
        .zip(made.into_iter().map(|future| future.expect("a future")))
        .collect();
    let half: BTreeMap<TileKey, Future> = all
        .iter()
        .step_by(2)
        .map(|(key, future)| (*key, future.clone()))
        .collect();
    let tiles = backlog_tiles(store);
    let stored: BTreeMap<TileKey, store::Aabb> = tiles
        .iter()
        .map(|key| {
            let positions = store.evaluate(*key).expect("evaluated");
            let bounds = store::Aabb::from_points(positions.iter().copied());
            (*key, bounds.expect("a tile of points"))
        })
        .collect();
    // The tiles the stroke changes that `read` can read anything of, before or after their
    // futures are installed: reached in either box, or in the ring of a tile a ball reaches.
    let store = &*store;
    let reached = |read: &Read| -> BTreeSet<TileKey> {
        let met = |key: &TileKey| {
            let future = all.get(key).and_then(Future::bounds);
            read_meets(read, stored[key]) || future.is_some_and(|bounds| read_meets(read, bounds))
        };
        let mut reached = BTreeSet::new();
        let met_tiles = tiles.iter().filter(|key| met(key));
        for key in met_tiles {
            if all.contains_key(key) {
                reached.insert(*key);
            }
            if matches!(read, Read::Ball { .. }) {
                let ring = store.normal_inputs(key).expect("a ring");
                reached.extend(ring.iter().filter(|input| all.contains_key(input)));
            }
        }
        reached
    };

    let reads = reads_across(&run.camera);
    let index = run.picker.index_mut();
    let strokes = &run.brush.strokes;
    let (mut fewer, mut some) = (0, 0);
    for read in &reads {
        let one = std::slice::from_ref(read);
        let asked = |ahead| needed(store, index, strokes, one, ahead).expect("listed");
        let (bound, with_half, with_all) = (asked(None), asked(Some(&half)), asked(Some(&all)));
        let reached = reached(read);
        assert!(
            reached.is_subset(&with_all),
            "{read:?}: a tile it reaches isn't asked for: {:?}",
            reached.difference(&with_all).collect::<Vec<_>>()
        );
        assert!(
            with_all.is_subset(&with_half),
            "{read:?}: more with every future"
        );
        assert!(
            with_half.is_subset(&bound),
            "{read:?}: more than the bound's"
        );
        assert_eq!(
            asked(Some(&BTreeMap::new())),
            bound,
            "{read:?}: with no future at hand, the bound's"
        );
        fewer += usize::from(with_all.len() < bound.len());
        some += usize::from(!reached.is_empty());
    }
    assert!(some > 10, "reads that reach the stroke's tiles: {some}");
    assert!(
        fewer > 10,
        "reads asked fewer futures with them at hand: {fewer}"
    );
    drop(futures);
    run.brush.abandon();
    run.close();
}

/// A display preview as the store holds it, bit for bit.
#[derive(Debug, PartialEq, Eq)]
struct DisplayBits {
    /// Its positions.
    positions: Vec<[u32; 3]>,
    /// Its normals.
    normals: Vec<[u32; 3]>,
    /// Its roughness.
    roughness: Vec<u8>,
    /// Its bound.
    bound: u32,
    /// Whether it is held.
    held: bool,
}

/// Everything the store's previews hold, bit for bit: what a renderer is handed.
#[derive(Debug, PartialEq, Eq)]
struct ShownBits {
    /// Each finest preview's every point, its ring and its base-vertex rings included.
    finest: BTreeMap<TileKey, Vec<[u32; 3]>>,
    /// Each display preview.
    display: BTreeMap<TileKey, DisplayBits>,
    /// Where two previews holding the same vertex draw it differently ([`cracks`]).
    cracks: Vec<String>,
}

impl ShownBits {
    /// What `store` previews now.
    fn of(store: &TileStore) -> Self {
        let bits = |v: &Vec3| [v.x.to_bits(), v.y.to_bits(), v.z.to_bits()];
        let finest = store.previewed().map(|&key| {
            let block = store.preview(&key).expect("previewed");
            (key, points(block).iter().map(bits).collect())
        });
        let display = store.display_previewed().filter_map(|&key| {
            let preview = store.display_preview(&key)?;
            let shown = DisplayBits {
                positions: preview.positions.iter().map(bits).collect(),
                normals: preview.normals.iter().map(bits).collect(),
                roughness: preview.roughness.clone(),
                bound: preview.bound.to_bits(),
                held: preview.held,
            };
            Some((key, shown))
        });
        Self {
            finest: finest.collect(),
            display: display.collect(),
            cracks: cracks(store),
        }
    }
}

/// How a run of [`seams_drawn`] ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SeamsEnd {
    /// Every stroke lands, one after another.
    Landed,
    /// The oldest stroke's landing fails: it drops its dabs, and the strokes after it go too.
    Dropped,
}

/// Draws three strokes of `strokes` (each `Down` to `Up`), each over those before it still
/// landing, and lands them, on a face of many tiles; returns what the previews hold after every
/// step, bit for bit.
///
/// The steps are those that change what previews of different strokes meet on: each sample of
/// the second stroke (its `Down` where the pen hovered first, so the seams there were noted
/// ahead), dabs of the first landing, each sample of the third stroke over both, the first
/// landing whole and ending, then, as `end` says, the second landing a few dabs and everything
/// landing, or the oldest dropping its dabs and the others going with it.
///
/// With `afresh`, the brush's previews forget before every step what their stitches keep
/// between calls ([`sculpt::Preview::forget_seams`]), so every stitch makes it again.
///
/// A step is settled before it is looked at: frames with no sample until every dab placed is
/// shown, and `prepare`s until no display preview waits to be made a later stroke's. How much of
/// that a step does itself goes by the clock ([`over::FUTURES_TIME`]); what is shown once it is
/// done doesn't.
fn seams_drawn(
    epoch: Instant,
    strokes: &[Vec<PenSample>; 3],
    afresh: bool,
    end: SeamsEnd,
) -> Vec<ShownBits> {
    let level = Level::new(BACKLOG_LEVEL).expect("a level");
    let config = StoreConfig::new(256 << 20, 1).with_display(level);
    let mut run = Chain::new(epoch, Some(config));
    run.build_display();
    run.brush.set_radius(ORDINARY_RADIUS);
    let [first, second, third] = strokes;
    let mut shown: Vec<ShownBits> = Vec::new();
    let forget = |chain: &mut Chain| {
        if let Some(preview) = chain.brush.preview.as_mut().filter(|_| afresh) {
            preview.forget_seams();
        }
    };
    let settle = |chain: &mut Chain, steps: &mut Vec<ShownBits>| {
        for _ in 0..HOVER_FRAMES {
            let unshown = chain.brush.strokes.iter().any(|stroke| {
                let mut sources = stroke.sources.iter();
                sources.any(|source| !source.shown)
            });
            let preview = chain.brush.preview.as_ref();
            let pending = preview.is_some_and(Preview::displays_pending);
            if !unshown && !pending {
                break;
            }
            forget(chain);
            if unshown {
                chain.frame(&[]);
            } else {
                chain.prepare();
            }
        }
        steps.push(ShownBits::of(chain.project.store()));
    };
    run.frame(first);
    run.prepare();
    run.cut();
    // The second stroke, where the pen hovered: its seams noted ahead.
    let down = second.first().expect("a Down");
    run.hover(pixel_of(down));
    if !afresh {
        let kept = run.brush.preview.as_ref().map(Preview::seams_kept);
        assert!(kept > Some(0), "the hover noted seams ahead: {kept:?}");
    }
    forget(&mut run);
    assert_eq!(run.route(down), Route::Brush, "on the surface");
    for sample in second {
        forget(&mut run);
        run.frame(std::slice::from_ref(sample));
        settle(&mut run, &mut shown);
    }
    assert_eq!(run.ops().len(), 2, "the second began over the first");
    let kept = run.brush.preview.as_ref().map(Preview::seams_kept);
    assert!(
        kept > Some(0),
        "the stitches kept what they looked up: {kept:?}"
    );
    // Dabs of the first land under both.
    for _ in 0..3 {
        forget(&mut run);
        assert_eq!(run.land_one().dabs_applied, 1, "the first lands a dab");
        settle(&mut run, &mut shown);
    }
    // A third stroke, over both.
    for sample in third {
        forget(&mut run);
        run.frame(std::slice::from_ref(sample));
        settle(&mut run, &mut shown);
    }
    assert_eq!(run.ops().len(), 3, "the third began over both");
    if end == SeamsEnd::Dropped {
        run.brush.knob = Knob::LandingFails;
        forget(&mut run);
        run.land_one();
        assert!(run.ops().is_empty(), "every stroke ended");
        run.brush.knob = Knob::None;
        // The strokes over, the display previews left are released, drawn until the refresh
        // replaces them: each draws its own bits again, and they meet as one stroke's do.
        shown.push(ShownBits::of(run.project.store()));
        run.brush.abandon();
        run.close();
        return shown;
    }
    // The first lands whole and ends: the second's op begins.
    let mut steps = 0;
    while run.ops().len() > 2 {
        forget(&mut run);
        run.land_one();
        settle(&mut run, &mut shown);
        steps += 1;
        assert!(steps < 10_000, "the first lands");
    }
    run.cut();
    for _ in 0..3 {
        forget(&mut run);
        assert_eq!(run.land_one().dabs_applied, 1, "the second lands a dab");
        settle(&mut run, &mut shown);
    }
    forget(&mut run);
    run.land_all();
    shown.push(ShownBits::of(run.project.store()));
    run.close();
    shown
}

#[test]
fn what_a_stitch_keeps_between_calls_changes_nothing_the_previews_show() {
    // The tiles around each preview and the previews' places, kept from stitch to stitch and
    // noted ahead under a hovering pen, against the same made afresh by every stitch: the same
    // previews after every step, bit for bit.
    for end in [SeamsEnd::Landed, SeamsEnd::Dropped] {
        let epoch = Instant::now();
        let strokes = [
            stroke(((110.0, 100.0), (210.0, 110.0)), 30),
            stroke(((150.0, 101.0), (176.0, 112.0)), 8),
            stroke(((140.0, 110.0), (185.0, 100.0)), 8),
        ];
        let kept = seams_drawn(epoch, &strokes, false, end);
        let afresh = seams_drawn(epoch, &strokes, true, end);
        assert_eq!(kept.len(), afresh.len(), "{end:?}: the same steps");
        let met = kept.iter().any(|shown| shown.finest.len() > 1);
        assert!(met, "{end:?}: finest previews were shown");
        let steps = kept.iter().zip(&afresh).enumerate();
        for (step, (with, without)) in steps {
            let same = with == without;
            assert!(same, "{end:?}: the previews differ after step {step}");
            assert!(
                with.cracks.is_empty(),
                "{end:?}: previews apart after step {step}: {} vertices, the first {:?}",
                with.cracks.len(),
                with.cracks.first()
            );
        }
    }
}

#[test]
fn a_down_refused_for_room_first_lets_go_of_the_seams_noted_ahead() {
    // Small pools, as `past_the_open_strokes_share_the_next_down_waits_as_before` has, and a
    // small brush: its strokes preview a few tiles of the sphere's 24.
    let config = |budget: usize| {
        StoreConfig::new(budget, 1)
            .with_chunk_bytes(64 << 10)
            .with_kept_predictions(0)
            .with_stroke_tiles(8)
            .with_display(Level::new(LEVEL).expect("a level"))
    };
    let minimum = config(0).minimum_budget(&cube_sphere(2).expect("a cube sphere"));
    let mut run = Chain::new(Instant::now(), Some(config(minimum + (8 << 20))));
    run.brush.set_radius(ORDINARY_RADIUS);
    let strokes = [stroke(ACROSS, 6), crossing(), over_both()];
    run.frame(&strokes[0]);
    // The seams about every tile noted ahead, while there is room: far more than is previewed.
    let every = tile_keys(&mut run.project);
    let store = run.project.store();
    let (held, share) = (store.pen_down_bytes(), store.open_strokes_share());
    let preview = run.brush.preview.as_mut().expect("the brush previews");
    let noted = preview.warm_seams(store, &every);
    assert!(noted, "room to note ahead: {held} of {share} held");
    let unheld = every.len() - store.previewed().count();
    assert!(unheld > 0, "tiles with seams noted and no preview");
    // Strokes over strokes until one is refused for room.
    let mut began = 1;
    let drawn = strokes.iter().cycle().skip(1).take(MOST_OPEN);
    for samples in drawn {
        run.frame(samples);
        if !run.brush.deferred.is_empty() {
            break;
        }
        began += 1;
    }
    assert!(
        !run.brush.deferred.is_empty(),
        "past the share, a stroke waits: {began} began"
    );
    // Before it waited, what was noted ahead went: none of it holds room a stroke was refused.
    let waiting = run.brush.preview.as_mut().expect("the brush previews");
    let left = waiting.let_go_unheld_seams(run.project.store(), &[]);
    assert_eq!(left, 0, "bytes noted ahead still held as a stroke waits");
    assert_eq!(run.brush.error(), None);
    run.land_all();
    run.close();
}

#[test]
fn a_step_that_shows_a_downs_dabs_lands_none_however_long_since_its_samples_came() {
    // A step that takes a `Down` over a stroke still landing may run long (its futures, its
    // previews): by its end the samples it took came long ago, as a resting pen's did. It lands
    // nothing of the earlier stroke all the same: what it showed reaches a frame first.
    let mut run = Chain::new(Instant::now(), None);
    run.frame(&stroke(ACROSS, 12));
    run.cut();
    let queued = |chain: &Chain| chain.rows().first().map_or(0, Vec::len);
    let before = queued(&run);
    assert!(before > 0, "the earlier stroke still lands");
    // A microsecond a dab: a catch-up given a sample's interval lands them all, one given no
    // time none.
    run.brush.cost = DabCost::new(0.0, 1000.0, 0.0, 0.0);
    let long_ago = Instant::now()
        .checked_sub(STROKE_QUIET * 2)
        .expect("after the epoch");
    let late = |mut taken: PenSample| {
        taken.timestamp = long_ago;
        taken.received = long_ago;
        taken
    };
    let mut pen = PenFrame::default();
    pen.push(
        late(sample(Phase::Down, (150.0, 80.0), 0.8)),
        Some(Route::Brush),
    );
    pen.push(
        late(sample(Phase::Drag, (153.0, 86.0), 0.8)),
        Some(Route::Brush),
    );
    // Between frames, as the frame loop steps the brush over a stroke's samples.
    let budget = BrushBudget::slice(Instant::now(), BETWEEN_SLICE);
    let stepped = context(&mut run.project, run.picker.index_mut(), &run.camera);
    let report = run.brush.step(stepped, &pen, &budget).expect("a step");
    assert_eq!(run.ops().len(), 2, "it began over the earlier stroke");
    assert!(report.dabs_shown > 0, "its dabs are shown: {report:?}");
    assert_eq!(
        report.dabs_applied, 0,
        "none landed in its step: {report:?}"
    );
    assert_eq!(queued(&run), before, "the earlier stroke's dabs wait");
    // The pen resting on from there, a step that shows nothing catches up as before.
    let rested = context(&mut run.project, run.picker.index_mut(), &run.camera);
    let slice = BrushBudget::slice(Instant::now(), BETWEEN_SLICE);
    run.brush.unframed = false;
    let resting = run.brush.step(rested, &PenFrame::default(), &slice);
    let resting = resting.expect("a step");
    assert!(resting.dabs_applied > 0, "a resting pen's: {resting:?}");
    run.brush.abandon();
    run.close();
}

/// How many times each timed case is drawn ([`time_downs`]): the median and the spread are said.
const TIMED_RUNS: usize = 7;

/// How many times the heavy stroke is landed whole for its reference
/// (`a_down_once_a_heavy_stroke_has_landed_is_timed`): each takes the better part of a minute.
const LANDED_RUNS: usize = 5;

/// The most hover frames a pen is given for the brush's warming to end ([`Chain::hover`]).
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
    /// The store's budget, in bytes.
    budget: usize,
}

/// The store's budget in the timed cases, unless one says another: the tests' own.
const TIMED_BUDGET: usize = 256 << 20;

/// What one run of a timed `Down` took ([`Timed::run`]).
#[derive(Debug, Clone)]
struct Took {
    /// Routing the `Down`.
    route: Duration,
    /// The step that takes the `Down` and its first drag, and shows its first dab.
    step: Duration,
    /// The futures installed for the stroke by its routing, and by the end of that step.
    installed: (usize, usize),
    /// The futures made (not taken from those made ahead) in the routing, and in that step.
    made: (usize, usize),
    /// The earlier stroke's dabs still to land as the pen came down.
    backlog: usize,
    /// The tiles those dabs reach.
    reach: usize,
    /// The hover frames before the `Down`, and what their warming took in all.
    hover: (usize, Duration),
    /// The steps of the frames after it, each given one more sample of the pen drawing on.
    later: Vec<Duration>,
    /// The most the open strokes held, over its frames.
    held: Held,
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
            budget: TIMED_BUDGET,
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
            budget: TIMED_BUDGET,
        }
    }

    /// Draws the case once, on a face of 4 × 4 tiles, and times its `Down`: routed, then
    /// stepped with its first drag, as a frame does.
    ///
    /// `prepare` has kept up what a later stroke reads of the earlier one, as the frames after
    /// it would; the futures it made ahead where the earlier stroke's pen lifted are let go, so
    /// a cold `Down` makes every future it reads. A warmed `Down` comes after hover frames at
    /// its pixel ([`Chain::hover`]), so it is over the same dabs as a cold one.
    fn run(&self, before: Before, frames: usize) -> Took {
        let level = Level::new(BACKLOG_LEVEL).expect("a level");
        let config = StoreConfig::new(self.budget, 1).with_display(level);
        let mut run = Chain::new(Instant::now(), Some(config));
        run.build_display();
        if let Some(radius) = self.radius {
            run.brush.set_radius(radius);
        }
        run.frame(&self.earlier);
        let placed = run.rows().first().map_or(0, Vec::len);
        if self.half_landed {
            run.cut();
            run.land_some(placed / 2);
        }
        if before == Before::Landed {
            run.cut();
            run.land_all();
        }
        run.prepare();
        // The earlier stroke's own lift is a hover, where that `prepare` made futures ahead: a
        // cold `Down` has none.
        run.brush.ahead.clear();
        let backlog = run.rows().first().map_or(0, Vec::len);
        let reach = run.brush.strokes.first().map_or(0, |earlier| {
            let remaining = earlier.remaining.as_ref();
            remaining.map_or(0, |remaining| remaining.reach.len())
        });
        let (at, then) = self.pen;
        let hover = if before == Before::Warmed {
            run.hover(at)
        } else {
            (0, Duration::ZERO)
        };
        // The pen's samples are a frame apart on its clock, the last of them taken about now.
        let interval = setup().interval;
        let ahead = u32::try_from(frames).expect("a few frames");
        let earliest = Instant::now().checked_sub(interval * ahead);
        let earliest = earliest.expect("after the epoch");
        let at_frame = |mut taken: PenSample, frame: u32| {
            taken.timestamp = earliest + interval * frame;
            taken.received = taken.timestamp;
            taken
        };
        let (mut down, mut drag) = (sample(Phase::Down, at, 0.8), sample(Phase::Drag, then, 0.8));
        if frames > 0 {
            (down, drag) = (at_frame(down, 0), at_frame(drag, 0));
        }
        take_made();
        let began = Instant::now();
        let routed = run.route(&down);
        let route = began.elapsed();
        assert_eq!(routed, Route::Brush, "on the surface");
        let by_routing = match &run.brush.routed {
            Some(Routed::Over(over)) => over.begun.installed.len(),
            _ => 0,
        };
        let made_routing = take_made();
        let stepped = Instant::now();
        let first = run.frame(&[down, drag]);
        let step = stepped.elapsed();
        assert!(first.dabs_shown > 0, "its first dab is shown: {first:?}");
        let over = if before == Before::Landed { 1 } else { 2 };
        assert_eq!(run.ops().len(), over, "it began, over the earlier stroke");
        let installed = (by_routing, run.installed().len());
        let made = (made_routing, take_made());
        let mut held = Held::of(&mut run);
        // The frames after it, the pen drawing on along its first drag, a sample a frame.
        let mut later = Vec::with_capacity(frames);
        for frame in 0..frames {
            let frame = u32::try_from(frame).expect("a few frames");
            let k = f64::from(frame) + 2.0;
            let on = (at.0 + (then.0 - at.0) * k, at.1 + (then.1 - at.1) * k);
            let next = at_frame(sample(Phase::Drag, on, 0.8), frame + 1);
            let drawn_at = Instant::now();
            let shown = run.frame(&[next]);
            later.push(drawn_at.elapsed());
            assert_eq!(run.brush.error(), None, "the frame is shown: {shown:?}");
            held = held.most(Held::of(&mut run));
        }
        run.brush.abandon();
        run.close();
        Took {
            route,
            step,
            installed,
            made,
            backlog,
            reach,
            hover,
            later,
            held,
        }
    }
}

/// Times `case`'s `Down` with the surface under it as each of `befores` says, `count` times
/// each, and prints the medians with their spread and the machine's load: a measurement,
/// asserting nothing of time.
fn time_downs(name: &str, case: &Timed, befores: &[Before], count: usize) {
    println!("{name} (load {})", load());
    for &before in befores {
        let runs: Vec<Took> = (0..count).map(|_| case.run(before, 0)).collect();
        let of = |part: fn(&Took) -> Duration| -> Vec<Duration> { runs.iter().map(part).collect() };
        let Some(last) = runs.last() else {
            return;
        };
        println!(
            "  {before:?}: over {} dabs still to land on {} tiles; {} futures installed by its \
             routing ({} made then) and {} by its first dab ({} more made); routed in {}, \
             stepped to its first dab in {}, {} in all; {} hover frames warming {:.2?}",
            last.backlog,
            last.reach,
            last.installed.0,
            last.made.0,
            last.installed.1,
            last.made.1,
            spread(&of(|took| took.route)),
            spread(&of(|took| took.step)),
            spread(&of(|took| took.route + took.step)),
            last.hover.0,
            last.hover.1,
        );
    }
    println!(
        "  (load {} after; medians of {count}, least to most)",
        load()
    );
}

/// Which of the frames after a timed `Down`'s is said as the steady one ([`time_frames`]): the
/// tenth of the pen drawing on.
const STEADY_FRAME: usize = 10;

/// Times the frames of a stroke drawn on from `case`'s `Down`: a measurement, asserting nothing
/// of time.
///
/// A sample a frame along its first drag, with the surface under it as each of `befores` says,
/// `count` times each; prints the medians of its first frame's step and its [`STEADY_FRAME`]th's
/// with their spread and the machine's load.
///
/// The earlier stroke lands nothing meanwhile: every frame shows one more sample's dabs beside
/// its previews, where the two strokes' previews meet.
fn time_frames(name: &str, case: &Timed, befores: &[Before], count: usize) {
    println!("{name} (load {})", load());
    for &before in befores {
        let runs: Vec<Took> = (0..count).map(|_| case.run(before, STEADY_FRAME)).collect();
        let first: Vec<Duration> = runs.iter().map(|took| took.step).collect();
        let steady = runs.iter().filter_map(|took| took.later.last().copied());
        let steady: Vec<Duration> = steady.collect();
        println!(
            "  {before:?}: its first frame stepped in {}, its frame {STEADY_FRAME} after in {}",
            spread(&first),
            spread(&steady),
        );
    }
    println!(
        "  (load {} after; medians of {count}, least to most)",
        load()
    );
}

/// A measurement, printed and asserting nothing of time: the frames of a stroke drawn along a
/// stroke still landing.
///
/// Its first and one well into it, where every frame's previews meet the earlier stroke's; and
/// those of the same stroke drawn once the earlier one has landed, alone.
#[test]
#[ignore = "a measurement, printed: run it alone on a quiet machine, with --no-capture"]
fn a_frame_drawn_over_a_landing_stroke_is_timed() {
    let befores = [Before::Warmed, Before::Landed];
    let ordinary = Timed::ordinary(((185.0, 107.5), (188.0, 107.8)));
    let on_ordinary = "frames drawn on an ordinary stroke's half still to land";
    time_frames(on_ordinary, &ordinary, &befores, TIMED_RUNS);
    let on_heavy = "frames drawn on a heavy stroke";
    time_frames(on_heavy, &Timed::heavy(), &[Before::Warmed], TIMED_RUNS);
}

/// The reference of `a_frame_drawn_over_a_landing_stroke_is_timed`'s heavy case: the same frames
/// once the heavy stroke has landed, a stroke alone. Apart, as landing the heavy stroke each time
/// takes long.
#[test]
#[ignore = "a measurement, printed: run it alone on a quiet machine, with --no-capture"]
fn a_frame_drawn_once_a_heavy_stroke_has_landed_is_timed() {
    let name = "frames drawn once a heavy stroke has landed";
    time_frames(name, &Timed::heavy(), &[Before::Landed], LANDED_RUNS);
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
    let name = "a Down over a heavy stroke";
    time_downs(name, &Timed::heavy(), &befores, TIMED_RUNS);
}

/// The reference of `a_down_over_a_heavy_stroke_is_timed`: the same `Down` once the heavy stroke
/// has landed, a stroke alone. Apart, as landing the heavy stroke each time takes long.
#[test]
#[ignore = "a measurement, printed: run it alone on a quiet machine, with --no-capture"]
fn a_down_once_a_heavy_stroke_has_landed_is_timed() {
    let befores = [Before::Landed];
    let name = "a Down once a heavy stroke has landed";
    time_downs(name, &Timed::heavy(), &befores, LANDED_RUNS);
}

/// A measurement, printed and asserting nothing of time: an ordinary stroke across the sphere,
/// half of it landed, then a stroke on the half still to land, and another beside it.
#[test]
#[ignore = "a measurement, printed: run it alone on a quiet machine, with --no-capture"]
fn a_down_over_an_ordinary_stroke_is_timed() {
    let befores = [Before::Cold, Before::Warmed, Before::Landed];
    let over = Timed::ordinary(((185.0, 107.5), (188.0, 107.8)));
    let on_it = "a Down on an ordinary stroke's half still to land";
    time_downs(on_it, &over, &befores, TIMED_RUNS);
    let beside = Timed::ordinary(((185.0, 122.0), (188.0, 122.3)));
    let beside_it = "a Down beside an ordinary stroke";
    time_downs(beside_it, &beside, &befores, TIMED_RUNS);
}

#[test]
fn a_dab_whose_strength_is_no_number_reaches_every_way() {
    // No bound can be said of it, so none is: its tile's box grows without end, and every read
    // asks for the tile.
    for strength in [f32::NAN, f32::INFINITY] {
        let mut reach = Reach::default();
        let wild = Dab {
            strength,
            normal: [0.0, 0.0, 1.0],
            radius: 0.1,
            ..Dab::default()
        };
        reach.count(&wild);
        assert_eq!(reach.dabs, 1);
        assert!(reach.strength.is_infinite() && reach.growth().is_infinite());
        assert_eq!(reach.span, ([f64::NEG_INFINITY; 3], [f64::INFINITY; 3]));
    }
}

/// How many workers the store of `a_refused_install_stops_the_making_of_futures` has: a batch
/// of futures is this many, far fewer than its `Down` reads.
const FEW_WORKERS: usize = 2;

#[test]
fn a_refused_install_stops_the_making_of_futures() {
    // The store refuses the second future (as past the open strokes' share): the batch it was
    // made in is the last made, however many the `Down` reads.
    let down = sample(Phase::Down, (160.0, 105.0), 0.8);
    let made_by_a_down = |refused_after: Option<usize>| {
        let level = Level::new(BACKLOG_LEVEL).expect("a level");
        let workers = std::num::NonZeroUsize::new(FEW_WORKERS).expect("some");
        let config = StoreConfig::new(256 << 20, 1)
            .with_display(level)
            .with_workers(store::Workers::Fixed(workers));
        let mut run = Chain::new(Instant::now(), Some(config));
        assert_eq!(run.project.store().workers(), FEW_WORKERS);
        run.frame(&stroke(((110.0, 100.0), (210.0, 110.0)), 60));
        run.prepare();
        run.brush.ahead.clear();
        take_made();
        installs_left(refused_after);
        assert_eq!(run.route(&down), Route::Brush);
        installs_left(None);
        let made = take_made();
        if refused_after.is_some() {
            assert!(
                !run.routed_over(),
                "nothing begun: its futures can't be had"
            );
            run.assert_nothing_reserved();
        }
        run.brush.abandon();
        run.close();
        made
    };
    let wanted = made_by_a_down(None);
    let refused = made_by_a_down(Some(1));
    assert!(
        wanted > 2 * FEW_WORKERS,
        "the Down reads more futures than a batch: {wanted}"
    );
    assert_eq!(
        refused, FEW_WORKERS,
        "the batch the refused future was made in is the last made, of the {wanted} it reads"
    );
}

/// How a pen hovers over a stroke still landing while its landing is timed
/// (`landing_under_a_hovering_pen_is_timed`).
#[derive(Debug, Clone, Copy, PartialEq)]
enum Hovering {
    /// Not at all: the pen is out of range.
    Away,
    /// Still, on the stroke's middle.
    Still,
    /// To and fro along the stroke, this many pixels a frame.
    Moving(f64),
}

/// How long `landing_under_a_hovering_pen_is_timed` lands each of its cases for.
const LANDING_TIMED: Duration = Duration::from_millis(1500);

/// A measurement, printed and asserting nothing of time: landing under a hovering pen.
///
/// How fast the heavy stroke lands, and how long a `prepare` holds the loop (what a `Down`
/// arriving in it waits), with the pen away, hovering still, and hovering in motion over it.
///
/// Frames of 8 ms: a step with the hover's sample, then `prepare`s until 6 ms into the frame or
/// until none is wanted, as the frame loop calls them; the dabs land by the calibrated cost model,
/// as the app's do.
#[test]
#[ignore = "a measurement, printed: run it alone on a quiet machine, with --no-capture"]
fn landing_under_a_hovering_pen_is_timed() {
    println!("landing under a hovering pen (load {})", load());
    let cases = [
        Hovering::Away,
        Hovering::Still,
        Hovering::Moving(1.0),
        Hovering::Moving(6.0),
        Hovering::Moving(40.0),
    ];
    for hovering in cases {
        let level = Level::new(BACKLOG_LEVEL).expect("a level");
        let config = StoreConfig::new(256 << 20, 1).with_display(level);
        let mut run = Chain::new(Instant::now(), Some(config));
        run.build_display();
        let timed = Timed::heavy();
        run.frame(&timed.earlier);
        run.cut();
        run.brush.cost = DabCost::CALIBRATED;
        let placed = run.rows().first().map_or(0, Vec::len);
        take_made();
        let began = Instant::now();
        let (mut frames, mut calls) = (0_u32, Vec::new());
        while began.elapsed() < LANDING_TIMED && !run.ops().is_empty() {
            let frame = Instant::now();
            let at = match hovering {
                Hovering::Away => None,
                Hovering::Still => Some((160.0, 105.0)),
                Hovering::Moving(by) => {
                    // To and fro along the stroke, 50 pixels each way.
                    let gone = (f64::from(frames) * by) % 100.0;
                    Some((135.0 + gone.min(100.0 - gone), 105.0))
                }
            };
            let mut pen = PenFrame::default();
            if let Some(at) = at {
                pen.push(sample(Phase::Hover, at, 0.0), None);
            }
            let budget = BrushBudget::frame(frame, Duration::ZERO, setup().interval);
            let stepped = context(&mut run.project, run.picker.index_mut(), &run.camera);
            run.brush.step(stepped, &pen, &budget).expect("a step");
            let by = frame + Duration::from_millis(6);
            loop {
                let prepared = context(&mut run.project, run.picker.index_mut(), &run.camera);
                let call = Instant::now();
                let more = run.brush.prepare(prepared, by).expect("prepared");
                calls.push(call.elapsed());
                if !more || Instant::now() >= by {
                    break;
                }
            }
            frames += 1;
            let rest = (frame + Duration::from_millis(8)).saturating_duration_since(Instant::now());
            std::thread::sleep(rest);
        }
        let took = began.elapsed();
        let left = run.rows().first().map_or(0, Vec::len);
        let landed = placed - left;
        calls.sort_unstable();
        let longest = calls.last().copied().unwrap_or_default();
        let median = calls.get(calls.len() / 2).copied().unwrap_or_default();
        #[expect(clippy::cast_precision_loss, reason = "a few hundred dabs, for a rate")]
        let rate = landed as f64 / took.as_secs_f64();
        println!(
            "  {hovering:?}: {landed} of {placed} dabs landed in {took:.2?} ({rate:.0} a second) \
             over {frames} frames; {} futures made ahead, {} kept; a prepare took {median:.2?} \
             at the median, {longest:.2?} at the longest",
            take_made(),
            run.brush.ahead.tiles.len(),
        );
        run.brush.abandon();
        run.close();
    }
    println!("  (load {} after)", load());
}

#[test]
fn the_catch_up_comes_before_the_futures_made_ahead_and_no_prepare_is_held_by_both() {
    let mut run = across_many_tiles(None);
    run.brush.ahead.clear();
    run.cut();
    let queued = |chain: &Chain| chain.rows().first().map_or(0, Vec::len);
    let before = queued(&run);
    // Reported a second after the first stroke's lift: a pen that came there at its leisure.
    run.hover_step((160.0, 105.0), Instant::now() + Duration::from_secs(1));
    take_made();
    // None made yet, so a batch is owed: it is made first, and with no time left the catch-up
    // waits for the next call.
    assert!(run.prepare_late(), "more is wanted");
    let first = take_made();
    assert!(
        (1..=run.project.store().workers()).contains(&first),
        "one batch made: {first}"
    );
    assert_eq!(
        queued(&run),
        before,
        "no dab landed in the call that made it"
    );
    // A batch was just made: the catch-up has this call, and with no time left no future is
    // made after it.
    run.prepare_late();
    assert_eq!(take_made(), 0, "no future made in the catch-up's call");
    assert_eq!(queued(&run), before - 1, "the catch-up landed its dab");
    // Left alone for long, the futures get a call again.
    if !run.brush.ahead.todo.is_empty() {
        run.brush.ahead.made_at = Instant::now().checked_sub(AHEAD_STARVED * 2);
        run.prepare_late();
        assert!(take_made() > 0, "a batch made once they waited long");
        assert_eq!(queued(&run), before - 1, "and no dab landed with it");
    }
    run.brush.abandon();
    run.close();
}

#[test]
fn a_hover_that_moves_keeps_the_futures_near_and_one_moving_fast_makes_none() {
    let mut run = across_many_tiles(Some(ORDINARY_RADIUS));
    run.brush.ahead.clear();
    // Nothing lands meanwhile (held, as during a close): the futures alone are looked at.
    run.brush.landing = Landing::InSteps;
    // Its tablet's reports, timed from well after the first stroke's lift: it comes to the
    // stroke at its leisure.
    let began = Instant::now() + Duration::from_secs(10);
    let at = |millis: u64| began + Duration::from_millis(millis);
    // Warmed where it holds still.
    run.hover_step((150.0, 104.0), at(0));
    take_made();
    run.prepare_out();
    let still: BTreeSet<TileKey> = run.brush.ahead.tiles.keys().copied().collect();
    assert_eq!(take_made(), still.len(), "each made once");
    assert!(!still.is_empty(), "futures made where it hovers");
    let listed_at = run.brush.ahead.target;
    // A pixel on, a tenth of a second later: the list stands, and nothing is made.
    run.hover_step((151.0, 104.0), at(100));
    run.prepare_out();
    assert_eq!(take_made(), 0, "within a quarter radius the list stands");
    assert_eq!(run.brush.ahead.target, listed_at, "not listed again");
    // Along the stroke at a walk: listed again, the futures still near kept, only new ones made.
    run.hover_step((166.0, 105.5), at(600));
    run.prepare_out();
    let walked: BTreeSet<TileKey> = run.brush.ahead.tiles.keys().copied().collect();
    assert_ne!(run.brush.ahead.target, listed_at, "listed where it is now");
    let kept = walked.intersection(&still).count();
    assert!(kept > 0, "the futures still near are kept");
    assert_eq!(
        take_made(),
        walked.len() - kept,
        "only the new ones are made"
    );
    // Far along it in a millisecond: fast, so nothing is listed or made, and the frame after is
    // asked for.
    let walked_at = run.brush.ahead.target;
    run.hover_step((200.0, 109.0), at(601));
    assert!(run.brush.warm.fast, "moving fast");
    assert!(
        !run.prepare_late(),
        "nothing more for this frame's prepare: it isn't called again for a pen moving fast"
    );
    assert_eq!(take_made(), 0, "none made for a place it is leaving");
    assert_eq!(run.brush.ahead.target, walked_at, "not listed");
    assert_eq!(
        run.brush.ahead.tiles.len(),
        walked.len(),
        "those made are kept"
    );
    // The same report carried over by the frame after (frames come faster than the tablet
    // reports): as fast as it was.
    let (_, reported) = run.brush.warm.hovered.expect("hovering");
    run.hover_step((200.0, 109.0), reported);
    assert!(
        run.brush.warm.fast,
        "still fast: the tablet has said nothing new yet"
    );
    assert!(!run.prepare_late(), "and still nothing to do");
    assert_eq!(take_made(), 0, "none made");
    // Nothing more from the tablet for long: the pen holds still, and its futures are made.
    let long_ago = Instant::now()
        .checked_sub(HOVER_SETTLED * 2)
        .expect("after the epoch");
    assert!(long_ago < reported, "an earlier time than the report's");
    if let Some((_, then)) = &mut run.brush.warm.hovered {
        *then = long_ago;
    }
    run.hover_step((200.0, 109.0), long_ago);
    assert!(!run.brush.warm.fast, "at rest");
    run.prepare_out();
    assert_ne!(run.brush.ahead.target, walked_at, "listed where it rests");
    run.brush.abandon();
    run.close();
}

#[test]
fn a_prepare_out_of_time_notes_no_seam_ahead_and_asks_for_no_more() {
    let mut run = across_many_tiles(Some(ORDINARY_RADIUS));
    run.brush.ahead.clear();
    // Nothing lands meanwhile (held, as during a close): the warming alone is looked at.
    run.brush.landing = Landing::InSteps;
    run.hover_step((150.0, 104.0), Instant::now() + Duration::from_secs(10));
    // The futures made, then the seams about them, in the calls with time left after.
    run.prepare_out();
    run.prepare_out();
    let listed = run.brush.ahead.listed.len();
    assert!(listed > 0, "futures listed where it hovers");
    assert_eq!(
        run.brush.ahead.seamed, listed,
        "with time, every seam noted"
    );
    // The same hover, its futures made, with no seam noted yet.
    let kept = |chain: &Chain| chain.brush.preview.as_ref().map(Preview::seams_kept);
    run.brush.ahead.seamed = 0;
    if let Some(preview) = &mut run.brush.preview {
        preview.forget_seams();
    }
    // Out of time, call after call (a loop whose frames overrun): nothing is noted, and no call
    // says more is wanted, as none did any of it.
    for _ in 0..3 {
        assert!(
            !run.prepare_late(),
            "no more wanted of a call that did none"
        );
        assert_eq!(run.brush.ahead.seamed, 0, "none noted out of time");
        assert_eq!(kept(&run), Some(0), "nothing kept");
    }
    // A call with time goes on from where they stand, and says more while it notes some.
    let by = Instant::now() + Duration::from_secs(1);
    let prepared = context(&mut run.project, run.picker.index_mut(), &run.camera);
    let more = run.brush.prepare(prepared, by).expect("prepared");
    let first = run.brush.ahead.seamed;
    assert_eq!(first, listed.min(SEAMS_A_CALL), "a call's worth noted");
    assert_eq!(more, first < listed, "more wanted while some are left");
    assert!(kept(&run) > Some(0), "and kept");
    run.prepare_out();
    assert_eq!(run.brush.ahead.seamed, listed, "all noted in the end");
    assert!(!run.prepare_late(), "and nothing more to do");
    run.brush.abandon();
    run.close();
}

#[test]
fn landing_refits_a_tile_without_losing_the_box_of_a_future_installed_for_it() {
    // A stroke over one still landing has futures installed; the first lands on, refitting the
    // tiles it writes: each tile's box in the index still holds the future the later stroke
    // reads there, so its raycasts find it.
    let mut run = Chain::new(Instant::now(), None);
    run.frame(&stroke(ACROSS, 12));
    run.frame(&crossing());
    assert_eq!(run.ops().len(), 2, "both under way");
    let installed = run.installed();
    assert!(!installed.is_empty(), "futures installed");
    run.cut();
    run.land_some(8);
    assert_eq!(run.ops().len(), 2, "the first still lands");
    let newest = run.brush.strokes.last().expect("the second");
    let StrokeOp::Reserved { token, .. } = &newest.op else {
        panic!("the second's op is reserved");
    };
    let shapes: Vec<(TileKey, store::Aabb)> = {
        let reading = run
            .project
            .store()
            .read_pen_down_of(token)
            .expect("reserved");
        let boxed = installed.iter().filter_map(|key| {
            let bounds = reading.pen_down_shape(key)?.bounds()?;
            Some((*key, bounds))
        });
        boxed.collect()
    };
    assert_eq!(shapes.len(), installed.len(), "a shape a future");
    let index = run.picker.index_mut();
    for (key, shape) in shapes {
        let kept = index.bvh().bounds(&key).expect("a tile of the index").aabb;
        assert_eq!(
            kept.union(shape),
            kept,
            "tile {key:?}: its box in the index holds the future installed"
        );
    }
    run.land_all();
    run.close();
}

#[test]
fn futures_made_ahead_are_read_only_over_their_own_store_with_the_oldest_strokes_op_current() {
    let mut run = Chain::new(Instant::now(), None);
    run.frame(&stroke(ACROSS, 12));
    run.hover((150.0, 80.0));
    assert!(!run.brush.ahead.tiles.is_empty(), "futures made ahead");
    let ours = run.project.store().id();
    assert_eq!(run.brush.ahead.store, Some(ours), "tagged with their store");
    let read =
        |ahead: &Ahead, store: &TileStore, strokes: &[Pending]| ahead.of(store, strokes).is_some();
    assert!(
        read(&run.brush.ahead, run.project.store(), &run.brush.strokes),
        "read by a stroke begun over the first, on its store"
    );
    // Another store: not theirs, whatever its ops.
    let mut other = Chain::new(Instant::now(), None);
    let theirs = other.project.store().id();
    assert_ne!(ours, theirs);
    assert!(
        !read(&run.brush.ahead, other.project.store(), &run.brush.strokes),
        "not over another store"
    );
    // The tag forged to that store's: its current op isn't the oldest stroke's.
    run.brush.ahead.store = Some(theirs);
    assert!(
        !read(&run.brush.ahead, other.project.store(), &run.brush.strokes),
        "not where the store's current op isn't the oldest stroke's"
    );
    run.brush.ahead.store = Some(ours);
    // No strokes, or others than those they were made up to.
    assert!(!read(&run.brush.ahead, run.project.store(), &[]));
    // A save, a flush and a close let them go outright.
    run.brush.begin_saving(Instant::now());
    assert!(run.brush.ahead.tiles.is_empty(), "let go by a save");
    run.brush.end_saving();
    run.hover((150.0, 80.0));
    assert!(!run.brush.ahead.tiles.is_empty(), "made again");
    run.cut();
    run.land_all();
    assert!(run.brush.ahead.tiles.is_empty(), "let go by a flush");
    other.close();
    run.close();
}

/// The level of the wide timed cases: 16 × 16 tiles a face, 6,144 on the sphere.
///
/// Some 800 of them lie on the face the camera sees: more than the stroke's state has buffers
/// for ([`StoreConfig::DEFAULT_STROKE_TILES`], 512). A level lower, a stroke over the whole face
/// reaches 204.
const WIDE_LEVEL: u8 = 10;

/// The brush's radius in the wide timed cases, in pixels: a dab reaches a dozen or so tiles.
const WIDE_RADIUS: f32 = 6.0;

/// The middle of the sphere's face as the default camera sees it and how far from it the wide
/// cases' raster runs, in pixels: most of the face, short of its limb.
const WIDE_FACE: ((f64, f64), f64) = ((160.0, 102.0), 56.0);

/// How far apart the passes of the wide cases' first raster are, in pixels: one coarse sweep of
/// the face, whose dabs land and fill the stroke's state.
const WIDE_SWEEP: f64 = 5.0;

/// How far apart the passes of the wide cases' second raster are, in pixels: the backlog, some
/// 26,000 dabs over every tile of the face (the plan asks for 16,000: the harder case).
const WIDE_BACKLOG: f64 = 0.42;

/// The store's budget in the wide timed cases: the app's default.
const WIDE_BUDGET: usize = 2048 << 20;

/// How many times each of the wide case's parts is timed
/// (`futures_over_a_full_stroke_state_are_timed`).
const WIDE_TIMES: usize = 21;

/// How many times the wide case's `Down` is drawn, each over a backlog placed afresh
/// (`a_down_over_a_full_stroke_state_is_timed`).
const WIDE_RUNS: usize = 5;

/// A raster over the sphere's face ([`WIDE_FACE`]): passes across it `apart` pixels apart, top
/// to bottom, each the other way from the last, a sample every 2.5 pixels; every sample a drag.
fn raster(apart: f64) -> Vec<PenSample> {
    let ((across, down), reach) = WIDE_FACE;
    // How many steps of `step` pixels fit `span` pixels, the first at its start.
    let steps = |span: f64, step: f64| {
        let counted = (0_u32..).take_while(move |&k| f64::from(k) * step <= span);
        counted.map(move |k| f64::from(k) * step)
    };
    let mut samples = Vec::new();
    let passes = steps(2.0 * reach, apart).zip([false, true].into_iter().cycle());
    for (lower, backwards) in passes {
        let off = lower - reach;
        let half = (reach.powi(2) - off.powi(2)).max(0.0).sqrt();
        let along = steps(2.0 * half, 2.5);
        let mut pass: Vec<PenSample> = along
            .map(|x| sample(Phase::Drag, (across - half + x, down + off), 0.8))
            .collect();
        if backwards {
            pass.reverse();
        }
        samples.extend(pass);
    }
    samples
}

/// The wide timed case, drawn: one stroke over the sphere's face on 16 × 16 tiles a face, under a
/// store of `budget` bytes.
///
/// Its first sweep of the face has landed, so the stroke's state holds as many tiles as it has
/// buffers for; its second, some 26,000 dabs over the same tiles, is placed and shown and none of
/// it landed; its pen is up, and what a later stroke reads of it is made. More tiles than the
/// state has buffers for are still to be edited, so the state's decisions for them are made dab
/// by dab ([`Futures::continuous`] says no).
fn wide(budget: usize) -> Chain {
    let level = Level::new(WIDE_LEVEL).expect("a level");
    let config = StoreConfig::new(budget, 1).with_display(level);
    // The backlog's samples are a minute's worth at the pen's rate, taken up to now.
    let epoch = Instant::now().checked_sub(Duration::from_secs(600));
    let mut run = Chain::new(epoch.expect("ten minutes up"), Some(config));
    run.build_display();
    run.brush.set_radius(WIDE_RADIUS);
    let (coarse, dense) = (raster(WIDE_SWEEP), raster(WIDE_BACKLOG));
    let swept = coarse.len();
    let mut samples: Vec<PenSample> = coarse.into_iter().chain(dense).collect();
    if let Some(down) = samples.first_mut() {
        down.phase = Phase::Down;
    }
    samples.push(sample(Phase::Up, WIDE_FACE.0, 0.0));
    let samples = taken_up_to_now(samples);
    let (touch, rest) = samples.split_at(2);
    let (sweep, backlog) = rest.split_at(swept - 2);
    // Its op begun and its cut taken, the first sweep lands as it is drawn.
    run.frame(touch);
    run.cut();
    run.brush.cost = DabCost::any_number();
    run.frame(sweep);
    assert_eq!(
        run.rows().first().map_or(0, Vec::len),
        0,
        "the sweep landed"
    );
    run.brush.cost = DabCost::one_at_a_time();
    run.frame(backlog);
    run.prepare();
    run.brush.ahead.clear();
    assert_eq!(run.brush.error(), None);
    let buffers = run.project.store().stroke_tiles();
    assert_eq!(
        run.brush.state.tiles_held(),
        buffers,
        "the stroke's state is full"
    );
    run
}

/// The tiles the oldest stroke's dabs still to land reach, those most dabs reach first, each
/// with how many.
fn busiest(run: &Chain) -> Vec<(usize, TileKey)> {
    let earlier = run.brush.strokes.first();
    let remaining = earlier.and_then(|earlier| earlier.remaining.as_ref());
    let mut tiles: Vec<(usize, TileKey)> = remaining.map_or_else(Vec::new, |remaining| {
        let reached = remaining.reach.iter();
        reached.map(|(key, reach)| (reach.dabs, *key)).collect()
    });
    tiles.sort_unstable();
    tiles.reverse();
    tiles
}

/// Makes the future of every tile of `keys` twice over, and prints what became of their
/// predictions.
///
/// A workers' batch at a time, in key order as a `Down` asks for them; each time, how many the
/// store had kept as their batch came, how many it kept as the batch ended and as the pass did,
/// and what making them took of the pass (each batch's made first, as its futures would make
/// them).
fn time_predictions(store: &mut TileStore, futures: &Futures<'_>, keys: &[TileKey]) {
    let workers = store.workers();
    let kept = |held: &TileStore, batch: &[TileKey]| {
        let kept = batch
            .iter()
            .filter(|key| held.kept_prediction(key).is_some());
        kept.count()
    };
    for pass in ["first", "second"] {
        let (mut had, mut keeps, mut batches) = (0, 0, 0);
        let mut predicting = Duration::ZERO;
        let began = Instant::now();
        for batch in keys.chunks(workers) {
            had += kept(store, batch);
            let pinned = store.pin(batch).expect("the batch pinned");
            let predicted = Instant::now();
            let together = store.keep_predictions_of(batch);
            predicting += predicted.elapsed();
            drop(pinned);
            assert!(together.is_ok(), "kept together: {together:?}");
            let made = futures.futures_of(store, batch);
            assert!(made.iter().all(Result::is_ok), "every future made");
            keeps += kept(store, batch);
            batches += 1;
        }
        let took = began.elapsed();
        let still = kept(store, keys);
        println!(
            "  the {pass} pass over all {} tiles, {batches} batches of {workers}, in {took:.2?} \
             ({predicting:.2?} of it making predictions): {had} predictions the store had kept, \
             {} made afresh; {keeps} kept as their batch ended, {still} still kept as the pass \
             did",
            keys.len(),
            keys.len() - had,
        );
    }
}

/// A measurement, printed and asserting nothing of time: the futures of a stroke whose state is
/// full, at a held-tile count as a long stroke on a real model has it (the stroke-while-landing
/// plan, Task 7).
///
/// On a sphere of 6,144 tiles, some 800 of them under the stroke: making the chain's [`Futures`]
/// (the stroke state's decisions for every dab still to land, as more tiles are to be edited
/// than it has buffers: afresh, and as the stroke's edits keep them), one future (the tile most
/// dabs reach, and one at the middle), a workers' batch of the busiest, and what becomes of the
/// predictions as every tile reached has its future made, a workers' batch at a time: how many
/// the store had kept, how many it keeps after, and what making them takes of the pass.
#[test]
#[ignore = "a measurement, printed: run it alone on a quiet machine, with --no-capture"]
fn futures_over_a_full_stroke_state_are_timed() {
    println!("futures over a full stroke state (load {})", load());
    let mut run = wide(WIDE_BUDGET);
    let tiles = busiest(&run);
    let backlog = run.rows().first().map_or(0, Vec::len);
    let chain = chain_of(&run.brush.strokes).expect("what a later stroke reads is made");
    let state = &run.brush.state;
    let store = run.project.store();
    let (held, buffers, workers) = (state.tiles_held(), store.stroke_tiles(), store.workers());
    // The plan and its decisions: made afresh for a copy of the stroke's edits, which keeps
    // none; kept by the stroke's own from the first time they were made.
    let copies: Vec<sculpt::StrokeEdits> = chain.iter().map(|&edits| edits.clone()).collect();
    let afresh: Vec<&sculpt::StrokeEdits> = copies.iter().collect();
    let decided = {
        let began = Instant::now();
        drop(Futures::new(store, state, &afresh).expect("a chain"));
        began.elapsed()
    };
    let futures_of = |over: &TileStore| Futures::new(over, state, &chain).expect("a chain");
    let mut built = Vec::with_capacity(WIDE_TIMES);
    for _ in 0..WIDE_TIMES {
        let began = Instant::now();
        let futures = futures_of(store);
        built.push(began.elapsed());
        drop(futures);
    }
    let futures = futures_of(store);
    let by_dab: Vec<bool> = futures.continuous().iter().map(|fast| !fast).collect();
    println!(
        "  {backlog} dabs still to land on {} tiles, the stroke's state holding {held} of \
         {buffers}; decided dab by dab: {by_dab:?}; the futures built in {decided:.2?} with \
         their decisions made afresh, and in {} with them kept",
        tiles.len(),
        spread(&built),
    );
    // One future: the tile most dabs reach, then the one at the middle of them.
    let picked = [tiles.first(), tiles.get(tiles.len() / 2)];
    for &(dabs, key) in picked.into_iter().flatten() {
        let mut made = Vec::with_capacity(WIDE_TIMES);
        for _ in 0..WIDE_TIMES {
            let began = Instant::now();
            futures.future_of(store, key).expect("a future");
            made.push(began.elapsed());
        }
        println!(
            "  one future, of a tile {dabs} dabs reach: {}",
            spread(&made)
        );
    }
    // A workers' batch: what a cold `Down` over the busiest tiles waits for, a batch at a time.
    let batch: Vec<TileKey> = tiles.iter().take(workers).map(|&(_, key)| key).collect();
    let mut together = Vec::with_capacity(WIDE_TIMES);
    for _ in 0..WIDE_TIMES {
        let began = Instant::now();
        let made = futures.futures_of(store, &batch);
        together.push(began.elapsed());
        assert!(made.iter().all(Result::is_ok), "every future made");
    }
    println!(
        "  the {} busiest together, a workers' batch: {}",
        batch.len(),
        spread(&together)
    );
    let mut keys: Vec<TileKey> = tiles.iter().map(|&(_, key)| key).collect();
    keys.sort_unstable();
    time_predictions(store, &futures, &keys);
    drop(futures);
    println!(
        "  (load {} after; medians of {WIDE_TIMES}, least to most)",
        load()
    );
    run.brush.abandon();
    run.close();
}

/// What the open strokes hold of a store, as a measurement says it.
#[derive(Debug, Clone, Copy)]
struct Held {
    /// The open strokes' pen-down data, in bytes ([`TileStore::pen_down_bytes`]).
    bytes: usize,
    /// What the stitches' tables hold of it between calls, in bytes.
    seams: usize,
    /// What the open strokes may hold together ([`TileStore::open_strokes_share`]).
    share: usize,
    /// Whether a `Down` waited for want of it.
    waited: bool,
}

impl Held {
    /// What `run`'s open strokes hold now.
    fn of(run: &mut Chain) -> Self {
        let seams = run.brush.preview.as_ref().map_or(0, Preview::seam_bytes);
        let waited = !run.brush.deferred.is_empty();
        let store = run.project.store();
        Self {
            bytes: store.pen_down_bytes(),
            seams,
            share: store.open_strokes_share(),
            waited,
        }
    }

    /// The more of this and `other`, by the bytes held.
    fn most(self, other: Self) -> Self {
        Self {
            waited: self.waited || other.waited,
            ..if other.bytes > self.bytes {
                other
            } else {
                self
            }
        }
    }
}

impl std::fmt::Display for Held {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let part = 100.0 * mib(self.bytes) / mib(self.share).max(f64::MIN_POSITIVE);
        write!(
            f,
            "{:.1} MiB held ({:.1} MiB of it the seam tables) of a share of {:.0} MiB: {part:.1} %{}",
            mib(self.bytes),
            mib(self.seams),
            mib(self.share),
            if self.waited { ", A DOWN WAITED" } else { "" }
        )
    }
}

/// A `Down` on the middle of the sphere's face and `frames` frames of the pen drawing on.
///
/// A sample a frame, as [`Timed::run`] draws them: the routing's time, the first step's, the
/// futures made in each, and the most the open strokes held ([`Held::waited`] if the `Down`
/// waited for the stroke before it, past their share: nothing of it is shown then).
fn down_and_draw(run: &mut Chain, frames: u32) -> (Duration, Duration, (usize, usize), Held) {
    let (at, by) = (WIDE_FACE.0, (0.6, 0.1));
    let down = sample(Phase::Down, at, 0.8);
    let drag = sample(Phase::Drag, (at.0 + by.0, at.1 + by.1), 0.8);
    take_made();
    let began = Instant::now();
    let routed = run.route(&down);
    let route = began.elapsed();
    assert_eq!(routed, Route::Brush, "on the surface");
    let made_routing = take_made();
    let stepped = Instant::now();
    let first = run.frame(&[down, drag]);
    let step = stepped.elapsed();
    let made = (made_routing, take_made());
    let mut held = Held::of(run);
    assert!(
        held.waited || first.dabs_shown > 0,
        "its first dab is shown, unless it waits: {first:?}"
    );
    for frame in 2..frames + 2 {
        let on = (
            at.0 + by.0 * f64::from(frame),
            at.1 + by.1 * f64::from(frame),
        );
        run.frame(&[sample(Phase::Drag, on, 0.8)]);
        run.prepare();
        held = held.most(Held::of(run));
    }
    assert_eq!(run.brush.error(), None);
    (route, step, made, held)
}

/// A measurement, printed and asserting nothing of time: a cold `Down` over a full stroke state.
///
/// Over the busiest tiles of a stroke whose state is full, on a sphere of 6,144 tiles (the
/// stroke-while-landing plan, Task 7), and what the open strokes hold as the pen draws on from
/// it.
#[test]
#[ignore = "a measurement, printed: run it alone on a quiet machine, with --no-capture"]
fn a_down_over_a_full_stroke_state_is_timed() {
    println!("a Down over a full stroke state (load {})", load());
    let mut routes = Vec::with_capacity(WIDE_RUNS);
    let mut steps = Vec::with_capacity(WIDE_RUNS);
    for _ in 0..WIDE_RUNS {
        let mut run = wide(WIDE_BUDGET);
        let backlog = run.rows().first().map_or(0, Vec::len);
        let reached = busiest(&run).len();
        let (route, step, made, held) = down_and_draw(&mut run, 10);
        assert!(!held.waited, "it began over the stroke: {held}");
        println!(
            "  over {backlog} dabs still to land on {reached} tiles: routed in {route:.2?} ({} \
             futures made), stepped to its first dab in {step:.2?} ({} more made), {} installed \
             ten frames on; {held}",
            made.0,
            made.1,
            run.installed().len(),
        );
        routes.push(route);
        steps.push(step);
        run.brush.abandon();
        run.close();
    }
    println!(
        "  routed in {}, stepped in {} (load {} after; medians of {WIDE_RUNS}, least to most)",
        spread(&routes),
        spread(&steps),
        load()
    );
}

/// A measurement, printed and asserting nothing: what the open strokes hold of their share.
///
/// At its most ([`TileStore::pen_down_bytes`] against [`TileStore::open_strokes_share`], the seam
/// tables counted in it), on the timed cases, under the app's default budget and a lean one.
///
/// The ordinary and the heavy stroke with a stroke drawn over each after a hover (the futures
/// made ahead are kept beside the store, so they aren't in the count until the `Down` installs
/// them); three strokes in a chain; and the stroke whose state is full. Nothing lands meanwhile,
/// as in the other timed cases: every stroke holds all it ever does.
#[test]
#[ignore = "a measurement, printed: run it alone, with --no-capture"]
fn what_the_open_strokes_hold_is_measured() {
    println!("what the open strokes hold");
    for budget in [WIDE_BUDGET, 256 << 20] {
        println!("  under a budget of {:.0} MiB:", mib(budget));
        let cases = [
            (
                "the ordinary stroke",
                Timed::ordinary(((185.0, 107.5), (188.0, 107.8))),
            ),
            ("the heavy stroke", Timed::heavy()),
        ];
        for (name, case) in cases {
            let case = Timed { budget, ..case };
            let took = case.run(Before::Warmed, STEADY_FRAME);
            println!(
                "    {name} and one over it, over {} dabs on {} tiles: {}",
                took.backlog, took.reach, took.held
            );
        }
        let (under_way, chained) = held_by_three(budget);
        println!("    three in a chain ({under_way} under way): {chained}");
        // The stroke whose state is full, and one over it.
        let mut run = wide(budget);
        let alone = Held::of(&mut run);
        let (_, _, _, over) = down_and_draw(&mut run, 10);
        println!("    the wide stroke alone: {alone}");
        println!("    the wide stroke and one over it: {over}");
        run.brush.abandon();
        run.close();
    }
}

/// Three strokes in a chain under a store of `budget` bytes: the heavy stroke, one crossing it
/// and one over both. How many are under way at the end, and the most they held.
fn held_by_three(budget: usize) -> (usize, Held) {
    let level = Level::new(BACKLOG_LEVEL).expect("a level");
    let config = StoreConfig::new(budget, 1).with_display(level);
    let mut run = Chain::new(Instant::now(), Some(config));
    run.build_display();
    let mut heavy = to_and_fro(64);
    heavy.push(sample(Phase::Up, ACROSS.0, 0.0));
    let mut held = Held::of(&mut run);
    for samples in [heavy, crossing(), over_both()] {
        run.frame(&samples);
        run.prepare();
        held = held.most(Held::of(&mut run));
    }
    let under_way = run.ops().len();
    run.brush.abandon();
    run.close();
    (under_way, held)
}
