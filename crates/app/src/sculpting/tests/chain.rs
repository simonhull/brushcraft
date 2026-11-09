//! Strokes over strokes still landing: a `Down` while an earlier stroke's dabs land begins at once,
//! on the surface that stroke will leave (the stroke-while-landing spec).
//!
//! Today's behaviour, a stroke that waits for the one before it to land, is the oracle: the same
//! samples, placed once every earlier stroke has landed, give the same dabs, bit for bit, and the
//! same tiles once everything has landed.

use store::{DabBatch, FLUSH_INTERVAL, PenDown};

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
    // A budget of small pools, which the pen-down data of a few dozen strokes fill: each
    // stroke's futures are the tiles under it.
    let tight = |budget: usize| {
        StoreConfig::new(budget, 1)
            .with_chunk_bytes(64 << 10)
            .with_kept_predictions(0)
            .with_stroke_tiles(8)
            .with_display(Level::new(LEVEL).expect("a level"))
    };
    let minimum = tight(0).minimum_budget(&cube_sphere(2).expect("a cube sphere"));
    let mut run = Chain::new(Instant::now(), Some(tight(minimum + (1 << 20))));
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
    // The pools at their least, these strokes' dabs pin more than they hold: they aren't landed
    // here, but let go with the project.
    run.brush.abandon();
    let Chain { dir, project, .. } = run;
    project.close(false).expect("closed");
    drop(dir);
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
