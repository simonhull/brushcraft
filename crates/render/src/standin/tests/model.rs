//! A randomized state machine over the store and residency, hunting the stand-ins' timing holes.
//!
//! Each case is a short sequence of operations drawn from the bytes `bolero` generates, over a
//! small project (the cube sphere to level 7, its display layer built): brush steps of one to
//! four dabs (`sculpt::apply_dabs` frames) at random places, dabs shown ahead of their finest
//! work (`sculpt::Preview`, preview epochs) and their catch-up landing a few at a time, refresh
//! slices, reverts (writes no step covers), a frame of dabs that fails after its first (its epoch
//! closed all the same), lost and failed frames, camera moves
//! (near: a face's finest tiles wanted and in the cut; far: none), `mark_all_dirty`, under a
//! budget small enough to evict, an epoch size small enough to shadow, and sometimes a stale cap
//! small enough to resync. After every frame that reaches the GPU it checks the invariant
//! ([`invariant`]); once the sequence quiets down, that the frames go idle.
//!
//! Under `cargo nextest` it runs for bolero's default second; for longer, as fuzzing:
//! `cargo +$BRUSHCRAFT_NIGHTLY bolero test -p render
//! standin::tests::model::stand_ins_hold_their_bounds -T 60s`.

use std::time::{Duration, Instant};

use super::*;
use crate::residency::{Request, Requests, ResidencyError, SizeClass};
use crate::upload::{TileData, TileDetail, TileSource};

/// Operations a case holds at most.
const MOST_OPS: usize = 10;

/// A source over the store whose `fail_at`th read fails: a frame that fails to stage.
struct Flaky<'a> {
    /// The store's source.
    inner: StoreSource<'a>,
    /// Reads until the failing one, if one fails.
    fail_at: Option<usize>,
}

/// Why a [`Flaky`] read failed.
#[derive(Debug)]
enum Flake {
    /// The store failed.
    Store(store::CacheError),
    /// The read the case made fail.
    Injected,
}

impl From<Flake> for ResidencyError {
    fn from(flake: Flake) -> Self {
        match flake {
            Flake::Store(error) => Self::Store(error),
            Flake::Injected => Self::Invariant("a read the test made fail"),
        }
    }
}

impl Flaky<'_> {
    /// Counts a read, failing the one the case chose.
    ///
    /// # Errors
    ///
    /// [`Flake::Injected`] at that read.
    fn count(&mut self) -> Result<(), Flake> {
        match &mut self.fail_at {
            Some(0) => {
                self.fail_at = None;
                Err(Flake::Injected)
            }
            Some(left) => {
                *left -= 1;
                Ok(())
            }
            None => Ok(()),
        }
    }
}

impl TileSource for Flaky<'_> {
    type Error = Flake;

    fn prepare(&mut self, keys: &[TileKey]) -> Result<(), Flake> {
        self.inner.prepare(keys).map_err(Flake::Store)
    }

    fn at_hand(&mut self, key: TileKey) -> bool {
        self.inner.at_hand(key)
    }

    fn read(&mut self, key: TileKey, out: &mut TileData) -> Result<Option<TileDetail>, Flake> {
        self.count()?;
        self.inner.read(key, out).map_err(Flake::Store)
    }

    fn prepare_positions(&mut self, keys: &[TileKey]) -> Result<(), Flake> {
        self.inner.prepare_positions(keys).map_err(Flake::Store)
    }

    fn read_positions(
        &mut self,
        key: TileKey,
        scratch: &mut TileData,
        out: &mut Vec<u32>,
    ) -> Result<bool, Flake> {
        self.count()?;
        self.inner
            .read_positions(key, scratch, out)
            .map_err(Flake::Store)
    }
}

/// One case's world: the project, its feed's cursor, residency, and the camera's wants.
struct Case {
    /// The project (its store).
    project: Project,
    /// Its mesh.
    mesh: BaseMesh,
    /// The renderer's cursor on the stale-tile feed.
    cursor: StaleCursor,
    /// Residency.
    residency: Residency,
    /// The tiles the camera wants and has in its cut.
    near: Vec<TileKey>,
    /// Frames lost or failed, so far.
    lost: usize,
    /// The store's spatial index, for the brush.
    index: store::SpatialIndex,
    /// The brush's stroke state.
    stroke: sculpt::Stroke,
    /// The dabs shown ahead of their finest work.
    preview: sculpt::Preview,
    /// The dabs shown and not landed, oldest first: while any is, a brush step's dabs are shown
    /// after them (as the app's brush queues them), and a revert makes the previews again.
    shown: Vec<stroke::Dab>,
}

/// The vertex `(face, i, j)` of the finest grid a byte pair names.
fn vertex(first: u8, second: u8) -> (FaceId, u32, u32) {
    let quads = level(FINEST).face_quads();
    let face = FaceId::new(u32::from(first % 6));
    let column = u32::from(second) * quads / 255;
    let row = u32::from(first.wrapping_mul(37)) * quads / 255;
    (face, column, row)
}

impl Case {
    /// A case's world: the stale cap small (resyncs) or not.
    fn new(gpu: &Gpu, dir: &TempDir, small_cap: bool) -> Self {
        let cap = if small_cap { 24 } else { 1 << 16 };
        let (mut project, mesh) = project(dir, |config| config.with_stale_cap(cap));
        let cursor = project.store().subscribe_stale();
        let finest = level(FINEST);
        let big = SizeClass::of(finest).slot_bytes();
        let faces = u64::from(mesh.face_count());
        let floor: u64 = (0..=4)
            .filter_map(SizeClass::new)
            .map(|class| 2 * class.slot_bytes() * faces)
            .sum();
        // Ten side-65 slots past the floor (finest and level-6 tiles), so a camera move evicts;
        // epochs of more than one tile through shadow slots.
        let config = ResidencyConfig::new(floor + 10 * big, 2 * big, 4 * big, big, 64, level(4))
            .and_then(|config| config.with_buffer_bytes(2 * big))
            .expect("a configuration");
        let residency = Residency::new(gpu, &mesh, finest, config).expect("residency");
        let index =
            store::SpatialIndex::build(project.store(), finest, |_, _| {}).expect("an index");
        let stroke = sculpt::Stroke::new(project.store());
        let mut case = Self {
            project,
            mesh,
            cursor,
            residency,
            near: Vec::new(),
            lost: 0,
            index,
            stroke,
            preview: sculpt::Preview::new(),
            shown: Vec::new(),
        };
        case.frame(gpu);
        case
    }

    /// One frame: the feed drained and applied, the camera's wants, staged from a source whose
    /// `fail_at`th read fails (if any), submitted; or, with `lose`, dropped unsubmitted.
    /// Checks the invariant after a frame that reached the GPU.
    fn frame_with(&mut self, gpu: &Gpu, fail_at: Option<usize>, lose: bool) {
        let store = self.project.store();
        self.residency.begin_frame(gpu).expect("a frame begins");
        self.residency.apply(store.drain_stale(&self.cursor));
        let cut: BTreeSet<TileKey> = self.near.iter().copied().collect();
        self.residency.use_frame(FrameUse::new(cut.clone(), cut));
        let requests = self.near.iter().map(|&key| Request {
            key,
            screen_error: 4.0,
        });
        self.residency.want(&Requests::new(requests, 0));
        let mut source = Flaky {
            inner: StoreSource::new(store),
            fail_at,
        };
        let deadline = Instant::now() + Duration::from_secs(60);
        let staged = self.residency.stage(gpu, &mut source, deadline);
        let Ok(uploads) = staged else {
            self.lost += 1;
            return;
        };
        if lose {
            self.lost += 1;
            drop(uploads);
            return;
        }
        let encoder = gpu
            .device()
            .create_command_encoder(&CommandEncoderDescriptor::default());
        uploads.submit(gpu.queue(), encoder, |_| {});
        gpu.device()
            .poll(PollType::wait_indefinitely())
            .expect("the GPU finishes");
        gpu.check().expect("no GPU error");
        let truth = finest_grids(store, &self.mesh);
        let drawn = drawn_grids(store, &truth);
        invariant(gpu, &self.residency, &truth, &drawn);
    }

    /// A frame that reaches the GPU.
    fn frame(&mut self, gpu: &Gpu) {
        self.frame_with(gpu, None, false);
    }

    /// A write no step covers, at the vertex the bytes name: a revert's epoch, closed.
    fn revert(&mut self, (first, second): (u8, u8)) {
        let store = self.project.store();
        let finest = level(FINEST);
        store.begin_op().expect("an op");
        let (face, i, j) = vertex(first, second);
        let (key, li, lj) = TileKey::containing(
            geom::LayerId::BASE,
            Channel::DISPLACEMENT,
            face,
            finest,
            i,
            j,
        )
        .expect("a finest tile");
        let c = store
            .correction(key)
            .expect("reads")
            .get(li, lj)
            .expect("a vertex");
        let lift = (f32::from(second) - 127.5) / 127.5 * 0.02;
        store
            .set_correction(face, finest, i, j, c + Vec3::new(0.0, 0.0, lift))
            .expect("written");
        store.close_epoch().expect("closed");
        if !self.shown.is_empty() {
            // The store moved under the previews: they are made again.
            self.preview
                .rebase(store, &mut self.index)
                .expect("rebased");
        }
    }

    /// One brush dab at each place the byte pairs name: on a finest vertex as the store holds it,
    /// of a radius and strength the bytes choose, its normal the sphere's outward one there.
    fn placed(&mut self, places: &[(u8, u8)]) -> Vec<stroke::Dab> {
        let grids = finest_grids(self.project.store(), &self.mesh);
        let quads = level(FINEST).face_quads();
        let dabs = places.iter().zip(0_u32..).map(|(&(a, b), seq)| {
            let (face, i, j) = vertex(a, b);
            let at = usize::try_from(j * (quads + 1) + i).expect("small");
            let centre = grids[&face.index()][at];
            let length = centre.dot(centre).sqrt();
            stroke::Dab {
                position: [centre.x, centre.y, centre.z],
                normal: [centre.x / length, centre.y / length, centre.z / length],
                tangent: [1.0, 0.0, 0.0],
                radius: 0.02 + f32::from(a % 8) * 0.01,
                strength: (f32::from(b) - 127.5) / 127.5 * 0.01,
                seq,
                ..stroke::Dab::default()
            }
        });
        dabs.collect()
    }

    /// Shows a dab at each place the byte pairs name ahead of its finest work: a preview epoch.
    fn show(&mut self, places: &[(u8, u8)]) {
        let dabs = self.placed(places);
        self.preview
            .show(self.project.store(), &mut self.index, &dabs)
            .expect("shown");
        self.shown.extend(dabs);
    }

    /// Lands the oldest of the dabs shown, as many as `a` picks: their finest work in one call
    /// (a step's epoch), then the previews made again over it (a preview epoch).
    fn land(&mut self, a: u8) {
        if self.shown.is_empty() {
            return;
        }
        self.land_count(1 + usize::from(a) % self.shown.len());
    }

    /// Lands the oldest `count` dabs shown (see [`Case::land`]).
    fn land_count(&mut self, count: usize) {
        if count == 0 {
            return;
        }
        let landing: Vec<stroke::Dab> = self.shown.drain(..count).collect();
        self.project.begin_stroke().expect("a stroke");
        sculpt::apply_dabs(
            self.project.store(),
            &mut self.index,
            &mut self.stroke,
            &landing,
        )
        .expect("the dabs land");
        self.project
            .end_unrecorded_stroke()
            .expect("the stroke ends");
        self.preview
            .land(self.project.store(), &mut self.index, count)
            .expect("landed");
    }

    /// A frame of brush dabs, one at each place the byte pairs name ([`Case::placed`]), applied
    /// with `sculpt::apply_dabs` as one call: one epoch, its displacements summed. With `fail`, a
    /// last dab the store can't hold may make the call fail after the others (its epoch closed
    /// all the same). While dabs are shown, the frame's dabs are shown after them instead (and
    /// one to fail lands those first), as the app's brush queues them.
    fn dabs(&mut self, places: &[(u8, u8)], fail: bool) {
        if !self.shown.is_empty() {
            if fail {
                self.land_count(self.shown.len());
            } else {
                self.show(places);
                return;
            }
        }
        let mut frame = self.placed(places);
        if fail && let Some(last) = frame.last().copied() {
            frame.push(stroke::Dab {
                strength: 3e38,
                seq: last.seq + 1,
                ..last
            });
        }
        self.project.begin_stroke().expect("a stroke");
        let applied = sculpt::apply_dabs(
            self.project.store(),
            &mut self.index,
            &mut self.stroke,
            &frame,
        );
        // The dab the store can't hold fails the call where it lands on vertices of its own;
        // either way the call's epoch is closed.
        if !fail {
            applied.expect("the frame applies");
        }
        self.project
            .end_unrecorded_stroke()
            .expect("the stroke ends");
    }

    /// Runs one operation the bytes name, then a frame (one that may be lost or fail).
    fn op(&mut self, gpu: &Gpu, (kind, a, b): (u8, u8, u8)) {
        match kind % 11 {
            0 | 1 => {
                let count = usize::from(a % 4) + 1;
                let dabs: Vec<(u8, u8)> = (0..count)
                    .map(|n| {
                        let n = u8::try_from(n).expect("small");
                        (
                            a.wrapping_add(n.wrapping_mul(11)),
                            b.wrapping_add(n.wrapping_mul(29)),
                        )
                    })
                    .collect();
                self.dabs(&dabs, false);
            }
            2 => {
                // A refresh slice.
                self.project
                    .store()
                    .run_jobs(Instant::now() + Duration::from_millis(u64::from(a % 8)))
                    .expect("a slice");
            }
            // A write no step covers: a revert's epoch.
            3 => self.revert((a, b)),
            // A frame of dabs that fails after its first: its epoch closed all the same.
            4 => self.dabs(&[(a, b)], true),
            5 => {
                self.frame_with(gpu, None, true);
                return;
            }
            6 => {
                self.frame_with(gpu, Some(usize::from(a % 6)), false);
                return;
            }
            7 => {
                self.near = if a % 2 == 0 {
                    Vec::new()
                } else {
                    let face = FaceId::new(u32::from(b % 6));
                    let finest = level(FINEST);
                    (0..2)
                        .flat_map(|ty| (0..2).map(move |tx| (tx, ty)))
                        .filter_map(|(tx, ty)| TileKey::displacement(face, finest, tx, ty).ok())
                        .collect()
                };
            }
            8 => self.residency.mark_tree_dirty(),
            // Dabs shown ahead of their finest work.
            9 => {
                let count = usize::from(a % 3) + 1;
                let places: Vec<(u8, u8)> = (0..count)
                    .map(|n| {
                        let n = u8::try_from(n).expect("small");
                        (
                            a.wrapping_add(n.wrapping_mul(7)),
                            b.wrapping_add(n.wrapping_mul(13)),
                        )
                    })
                    .collect();
                self.show(&places);
            }
            _ => self.land(a),
        }
        self.frame(gpu);
    }

    /// Quiets the case down: every job run, every epoch applied, no camera wants. Then the
    /// frames must go idle.
    fn quiet(&mut self, gpu: &Gpu) {
        self.near.clear();
        self.land_count(self.shown.len());
        assert!(
            !self.project.store().has_previews(),
            "no preview once every dab shown has landed"
        );
        for _ in 0..40 {
            run_jobs(self.project.store());
            self.frame(gpu);
            if self.residency.stats().pending_epochs == 0
                && !self.project.store().has_stale(&self.cursor)
                && !self.project.store().has_jobs()
            {
                break;
            }
        }
        self.frame(gpu);
        let store = self.project.store();
        self.residency.begin_frame(gpu).expect("a frame begins");
        self.residency.apply(store.drain_stale(&self.cursor));
        self.residency.use_frame(FrameUse::default());
        let deadline = Instant::now() + Duration::from_secs(60);
        let uploads = self
            .residency
            .stage(gpu, &mut StoreSource::new(store), deadline)
            .expect("staged");
        assert!(!uploads.changes_scene(), "a quiet frame changes the scene");
        let encoder = gpu
            .device()
            .create_command_encoder(&CommandEncoderDescriptor::default());
        uploads.submit(gpu.queue(), encoder, |_| {});
    }
}

/// The finest level as a renderer draws it: `truth` (the store's, [`finest_grids`]) with every
/// previewed tile's own vertices from its preview.
pub(super) fn drawn_grids(
    store: &TileStore,
    truth: &BTreeMap<u32, Vec<Vec3>>,
) -> BTreeMap<u32, Vec<Vec3>> {
    let finest = level(FINEST);
    let (face_quads, quads) = (finest.face_quads(), finest.tile_quads());
    let mut drawn = truth.clone();
    for key in store.previewed() {
        let (Some(block), Some(grid)) = (store.preview(key), drawn.get_mut(&key.face().index()))
        else {
            continue;
        };
        for j in 0..=quads {
            for i in 0..=quads {
                let (x, y) = (
                    u32::from(key.tx()) * quads + i,
                    u32::from(key.ty()) * quads + j,
                );
                let at = usize::try_from(y * (face_quads + 1) + x).expect("small");
                grid[at] = block.get(i64::from(i), i64::from(j));
            }
        }
    }
    drawn
}

/// The invariant, against the store's finest level `truth` and the finest level as drawn,
/// `drawn` (previews in place of their tiles, [`drawn_grids`]).
///
/// Every resident display tile's node bounds every finest vertex under it, of the store's (in
/// `f64`), and each of its border vertices holds the drawn finest vertex's bits, unless the tile
/// is behind a step it couldn't stand in for (its bound raised) or its bound says nothing is
/// known.
fn invariant(
    gpu: &Gpu,
    residency: &Residency,
    truth: &BTreeMap<u32, Vec<Vec3>>,
    shown: &BTreeMap<u32, Vec<Vec3>>,
) {
    let finest = level(FINEST);
    let n = finest.face_quads();
    let at = |face: FaceId, x: u32, y: u32| {
        truth[&face.index()][usize::try_from(y * (n + 1) + x).expect("small")]
    };
    let drawn_at = |face: FaceId, x: u32, y: u32| {
        shown[&face.index()][usize::try_from(y * (n + 1) + x).expect("small")]
    };
    for (place, (vertices, node)) in display_tiles(gpu, residency) {
        let size = n / place.level().tiles_per_side();
        let (x0, y0) = (u32::from(place.tx()) * size, u32::from(place.ty()) * size);
        let quads = place.level().tile_quads();
        let scale = size / quads;
        let vertex =
            |i: u32, j: u32| vertices[usize::try_from(j * (quads + 1) + i).expect("small")];
        let unknown = node.error.to_bits() == f32::MAX.to_bits();
        let behind = residency.tile_is_behind(place);
        if !unknown && !behind {
            let border = (0..=quads)
                .flat_map(|j| (0..=quads).map(move |i| (i, j)))
                .filter(|&(i, j)| i == 0 || j == 0 || i == quads || j == quads);
            for (i, j) in border {
                let finest_bits = drawn_at(place.face(), x0 + i * scale, y0 + j * scale).to_bits();
                assert_eq!(
                    vertex(i, j).position.map(f32::to_bits),
                    finest_bits,
                    "{place:?} ({i}, {j}): a seam open on a tile neither behind nor unknown"
                );
            }
        }
        if unknown {
            continue;
        }
        let mut worst = 0.0_f64;
        for y in 0..=size {
            for x in 0..=size {
                let fine = at(place.face(), x0 + x, y0 + y);
                let fine = [fine.x, fine.y, fine.z].map(f64::from);
                let (qx, qy) = ((x / scale).min(quads - 1), (y / scale).min(quads - 1));
                let fx = f64::from(x - qx * scale) / f64::from(scale);
                let fy = f64::from(y - qy * scale) / f64::from(scale);
                let p = |i, j| vertex(i, j).position.map(f64::from);
                let (p00, p10, p11, p01) =
                    (p(qx, qy), p(qx + 1, qy), p(qx + 1, qy + 1), p(qx, qy + 1));
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
        assert!(
            worst <= f64::from(node.error),
            "{place:?}: a finest vertex {worst} from its triangles, past its bound {}",
            node.error
        );
    }
}

/// Runs the case the bytes `ops` name.
fn run(gpu: &Gpu, ops: &[(u8, u8, u8)]) {
    let dir = TempDir::new("model");
    let small_cap = ops.first().is_some_and(|&(kind, _, _)| kind >= 240);
    let mut case = Case::new(gpu, &dir, small_cap);
    for &op in ops.iter().take(MOST_OPS) {
        case.op(gpu, op);
    }
    case.quiet(gpu);
    let Case {
        project, cursor, ..
    } = case;
    let mut project = project;
    project.store().unsubscribe_stale(cursor);
    drop(project.close(false));
}

#[test]
fn stand_ins_hold_their_bounds() {
    GpuTest::run(|gpu| {
        // The GPU is shared by every case; a case that panics fails the check, nothing reuses
        // what it left.
        let gpu = std::panic::AssertUnwindSafe(gpu);
        // `BRUSHCRAFT_MODEL_SECS` runs the random engine longer than bolero's default second,
        // a soak without the fuzzing build's sanitizer (which makes a GPU case very slow).
        let secs = std::env::var("BRUSHCRAFT_MODEL_SECS")
            .ok()
            .and_then(|secs| secs.parse().ok())
            .unwrap_or(1);
        bolero::check!()
            .with_test_time(Duration::from_secs(secs))
            .with_type::<Vec<(u8, u8, u8)>>()
            .for_each(|ops| {
                let shared: &Gpu = *gpu;
                run(shared, ops);
            });
    });
}

/// The case the model found on 2026-10-31.
///
/// A frame of dabs that failed after its first wrote the failing dab's huge move into tiles the
/// first dab's bump had named, and the store closed the epoch as a step's with the first dab's
/// displacement. A write after the last bump now makes the epoch a revert's.
#[test]
fn regression_a_failed_dab_s_writes_after_an_earlier_bump_close_a_revert() {
    GpuTest::run(|gpu| {
        run(
            gpu,
            &[
                (208, 246, 196),
                (44, 76, 62),
                (246, 6, 52),
                (243, 241, 25),
                (254, 189, 212),
                (223, 122, 181),
                (19, 161, 242),
                (31, 2, 208),
                (27, 83, 37),
                (174, 106, 132),
            ],
        );
    });
}
