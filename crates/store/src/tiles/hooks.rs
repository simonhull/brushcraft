//! What the tests read and set on a store: the log of what it computed and read, and the faults
//! they inject.

use std::cell::Cell;

use geom::TileKey;

/// What the store computed and read, for tests that count re-evaluations and file reads.
#[derive(Debug, Default)]
pub(super) struct EvalLog {
    /// Tiles whose positions were computed, in order.
    pub(super) positions: Vec<TileKey>,
    /// Tiles whose positions were predicted straight into their slots on the workers, in
    /// order: also in `positions`.
    pub(super) in_place: Vec<TileKey>,
    /// Tiles whose normals were computed, in order, in full or brought up to date.
    pub(super) normals: Vec<TileKey>,
    /// Tiles whose stale normals were brought up to date (only the vertices reading a changed
    /// input recomputed), in order: also in `normals`.
    pub(super) refreshed: Vec<TileKey>,
    /// Correction tiles the owner thread looked up in its files, in order (prefetch
    /// workers' reads aren't counted).
    pub(super) file_reads: Vec<TileKey>,
    /// Prefetched correction tiles installed.
    pub(super) prefetched: usize,
    /// Invalidations run (dependency cascades walked) for edited tiles.
    pub(super) invalidations: usize,
    /// Parent lists read back from the dependency graph. (Counted through `&self`, as the
    /// lists are looked up, hence the cells.)
    pub(super) parents_reused: Cell<usize>,
    /// Parent lists listed with `parent_tiles`.
    pub(super) parents_listed: Cell<usize>,
    /// Normal input lists read back from the dependency graph.
    pub(super) normal_inputs_reused: Cell<usize>,
    /// Normal input lists listed with `normal_tiles`.
    pub(super) normal_inputs_listed: Cell<usize>,
    /// The most predictions held at once past a round's predicting: those kept, and those the
    /// round made to keep, before they are inserted (see
    /// [`TileStore::plan_predictions`](super::TileStore::plan_predictions)).
    pub(super) predictions_held: usize,
}

/// A store's test-only state: [`TileStore`](super::TileStore)'s one field for the tests.
#[derive(Debug, Default)]
pub(super) struct TestHooks {
    /// What was computed, for the tests.
    pub(super) log: EvalLog,
    /// A correction tile whose reads from the working file fail, for tests of error paths.
    pub(super) unreadable: Option<TileKey>,
    /// A tile whose normals computation fails on the workers, after every other task of its
    /// round has written its slot: for tests of that error path.
    pub(super) failing_normals: Option<TileKey>,
    /// A display tile that won't build, as one from extreme geometry wouldn't: for tests of a
    /// job skipping it.
    pub(super) unbuildable_display: Option<TileKey>,
    /// Whether what the workers finish for a display job is held back, as if they were still
    /// building it, until this is set back (then it arrives at the next slice, in order): for
    /// tests of results that arrive after the store has changed under them.
    pub(super) hold_finished: bool,
    /// The results held back while [`TestHooks::hold_finished`] is set.
    pub(super) held_finished: Vec<crate::jobs::Finished>,
    /// Whether the tiles the queued evaluation finishes are held back, as if still being
    /// evaluated, until this is set back (then they are kept for the caller at the next slice,
    /// as of the count of correction changes they were copied at): for tests of tiles finished
    /// before a write.
    pub(super) hold_evaluated: bool,
    /// The tiles held back while [`TestHooks::hold_evaluated`] is set.
    pub(super) held_evaluated: Vec<crate::queued::EvaluatedTile>,
    /// Every slice [`TileStore::run_jobs`](super::TileStore::run_jobs) ran, in order.
    pub(super) turns: Vec<Turned>,
    /// A clock for [`TileStore::run_jobs`](super::TileStore::run_jobs)'s deadlines in place of
    /// the system's, if a test sets one: it moves by [`TestClock::slice`] a slice, and to the
    /// end of each wait, so a call's slices are counted, never timed.
    pub(super) clock: Option<TestClock>,
    /// Whether [`TileStore::tile_border`](super::TileStore::tile_border) gives `None` for every
    /// tile, as it would for a tile holding a border vertex twice: for tests of the edits'
    /// fallback, which no valid mesh reaches.
    pub(super) without_tile_borders: bool,
    /// A delay added to each tile a step of the queued evaluation alone evaluates: for tests of
    /// a slow evaluation's deadlines.
    pub(super) slow_cone_tile: std::time::Duration,
}

/// One slice [`TileStore::run_jobs`](super::TileStore::run_jobs) ran, as the tests see it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Turned {
    /// Whether it was the queued evaluation's, else the display jobs'.
    pub(super) queue: bool,
    /// Whether both sides could have run it.
    pub(super) contested: bool,
    /// Tiles whose positions it computed.
    pub(super) positions: usize,
    /// Tiles whose normals it computed.
    pub(super) normals: usize,
}

/// A test's clock for [`TileStore::run_jobs`](super::TileStore::run_jobs): see
/// [`TestHooks::clock`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct TestClock {
    /// The time it reads now.
    pub(super) now: std::time::Instant,
    /// How far each slice moves it.
    pub(super) slice: std::time::Duration,
}

impl TestHooks {
    /// The test's clock's time, if it set one.
    pub(super) fn clock_now(&self) -> Option<std::time::Instant> {
        self.clock.map(|clock| clock.now)
    }

    /// Notes the start of a slice of `queue`'s side (`contested` if both could run it);
    /// returns what was computed so far, for [`TestHooks::end_turn`].
    pub(super) fn begin_turn(&mut self, queue: bool, contested: bool) -> (usize, usize) {
        self.turns.push(Turned {
            queue,
            contested,
            positions: 0,
            normals: 0,
        });
        (self.log.positions.len(), self.log.normals.len())
    }

    /// Notes the end of the slice [`TestHooks::begin_turn`] began, which found `before`
    /// computed, and moves the test's clock on by a slice.
    pub(super) fn end_turn(&mut self, (positions, normals): (usize, usize)) {
        let computed = (
            self.log.positions.len() - positions,
            self.log.normals.len() - normals,
        );
        if let Some(turn) = self.turns.last_mut() {
            (turn.positions, turn.normals) = computed;
        }
        if let Some(clock) = &mut self.clock {
            clock.now += clock.slice;
        }
    }

    /// Moves the test's clock to `until`, the end of a wait, if it is later.
    pub(super) fn wait_until(&mut self, until: std::time::Instant) {
        if let Some(clock) = &mut self.clock {
            clock.now = clock.now.max(until);
        }
    }
}
