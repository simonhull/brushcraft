//! The futures made ahead of a `Down`, while the pen hovers over strokes still landing, and
//! installed ahead of a pen drawing over them.
//!
//! "Warming under a hovering pen" in the module notes of [`super`]: the futures kept made
//! ([`Ahead`]), when they are listed and made ([`Sculpting::list_to_warm`],
//! [`Sculpting::make_ahead`], [`Sculpting::list_ahead`]) and let go ([`Sculpting::keep_ahead`]),
//! and the futures installed ahead of a drawing pen ([`Sculpting::install_ahead`]).

use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

use geom::TileKey;
use render::camera::View;
use sculpt::{Future, Futures, StoreSurface};
use store::{OpId, TileStore};
use stroke::{DVec3, Dab, Projector, REACH_RADII, Radius, ScreenPoint, Surface};

use super::over::{FUTURES_TIME, Read, chain_of, install_in_time, needed, rounded_up};
use super::{AHEAD_DABS, CATCH_UP_STARVED, Pending, Scene, Sculpting, StrokeOp};
use crate::camera::Pixel;

/// How long the futures to make ahead wait for time the catch-up leaves before a batch of them
/// is made all the same, ahead of it ([`crate::frame::Brush::prepare`]).
///
/// The catch-up comes first: landing is what ends the strokes. But a `prepare` whose catch-up
/// takes all its time, call after call, still makes a batch of futures this often.
pub(super) const AHEAD_STARVED: Duration = CATCH_UP_STARVED;

/// How far a hovering pen moves, in brush radii, before the futures to make ahead are listed
/// again for where it is ([`Sculpting::list_to_warm`]).
///
/// Nearer than this, the list made for where it was stands: the futures kept and those still
/// to make are about the same tiles.
pub(super) const RELIST_RADII: f64 = 0.25;

/// How fast a hovering pen moves, in brush radii a second, past which no future is made ahead
/// for where it is: it is leaving, and the futures would be of a place it has left.
///
/// Ten times the stroke's reach a second: slower, the pen is still within reach of where the
/// list was made a tenth of a second on.
pub(super) const HOVER_FAST: f64 = 10.0 * REACH_RADII;

/// How long a tablet says nothing more of a hovering pen before the pen is taken to hold still,
/// however fast its last reports had it ([`HOVER_FAST`]).
///
/// A moving pen is reported every few milliseconds; a frame that finds the last report unchanged
/// may only have come sooner than the next one.
pub(super) const HOVER_SETTLED: Duration = Duration::from_millis(50);

/// The most futures kept made ahead of a `Down` ([`Ahead`]): those nearest the hovering pen.
///
/// A future is a tile's corrections, some tens of kilobytes, held beside the store's budget, so
/// they are few; a `Down` reading more makes the rest as it comes.
pub(super) const AHEAD_MOST: usize = 128;

/// The most tiles whose futures are made ahead that have the seams about them noted in one
/// `prepare` ([`Sculpting::note_seams_ahead`]).
///
/// The clock is looked at before each: a tile costs some tens of microseconds a preview about
/// it, the first time.
pub(super) const SEAMS_A_CALL: usize = 8;

/// What is to be done of the futures made ahead, in one `prepare`
/// ([`Sculpting::list_to_warm`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Warming {
    /// Nothing: no pen hovers over strokes still landing, or every future listed is made.
    Nothing,
    /// Nothing now, the pen moving fast: the frame after looks again, when it may have slowed.
    /// It is no work a `prepare` has left (nothing is to be called again for it): frames come
    /// while the pen moves, and while the strokes' dabs are queued.
    Later,
    /// Futures are listed to make.
    Now,
}

/// The futures made ahead of a `Down`, under a hovering pen, off the pen's path (see "Warming
/// under a hovering pen" in the module notes).
///
/// The futures of every stroke under way as they were made, up to the newest (`after`): what a
/// stroke begun after that one reads, so its `Down` installs them without making them
/// ([`super::over::install`]), and knows each tile's box exactly ([`needed`]). They are kept beside
/// the strokes, not in the store: no op is reserved for a pen that only hovers.
///
/// **They stay right** while the stroke `after` is under way and nothing but the strokes' own
/// landing writes the store. What that rests on, all of it:
///
/// - **A future doesn't change as its strokes land:** it is what their landing leaves, however far
///   the landing has got (a stroke before `after` that ends has landed just that), bit for bit
///   ([`sculpt::Futures`], "Bit for bit").
/// - **None of the strokes places a dab more:** they are made only with every pen up, the strokes
///   sealed ([`Sculpting::list_to_warm`]).
/// - **The strokes under way change in four ways only:** a stroke begins, the newest (it reads
///   them, if it begins right after `after`); the oldest ends, every dab landed (its futures are
///   the tiles then); the oldest drops dabs, and ends in that call with every stroke after it
///   ([`Sculpting::take_later`]), `after` among them; or all go (an abandon). An op is never handed
///   out twice, so `after` names one stroke for good.
/// - **Nothing but landing writes corrections while strokes are under way,** and the store's
///   current op is the oldest stroke's the whole time: every edit of it is that stroke's, and no
///   other op begins before the strokes' own. A restore's strokes resume before any `Down`, and no
///   future is made while one waits to resume.
///
/// **They fail closed** where one of those doesn't hold. They are read only by a stroke begun
/// right after `after`, over a store that is the one they were made over and whose current op is
/// still the oldest stroke's ([`Ahead::of`]): anything that began or ended an op behind the
/// brush's back (a save moving the op off, another writer) leaves them unread. They are let go
/// once `after` is no longer under way ([`Sculpting::keep_ahead`]), and outright by every path
/// of the brush that isn't a stroke's own placing and landing: a restore's strokes taken on, a
/// save, a close, a flush, an abandon. **A path added later that writes corrections outside a
/// stroke's landing (undo and redo, above all) must let them go too** ([`Ahead::clear`]): no
/// tag of the store's says such a write happened.
#[derive(Debug, Default)]
pub(super) struct Ahead {
    /// The op of the newest stroke under way as they were made; `None` with none made.
    pub(super) after: Option<OpId>,
    /// The store they were made over; `None` with none made.
    pub(super) store: Option<store::StoreId>,
    /// Each tile's future, with its box.
    pub(super) tiles: BTreeMap<TileKey, Future>,
    /// What they were listed for: where the pen hovered, the view, and the brush's radius in
    /// pixels then.
    pub(super) target: Option<(Pixel, View, Radius)>,
    /// The tiles still to make for `target`, the next to make last.
    pub(super) todo: Vec<TileKey>,
    /// When a batch of them was last made; `None` if none was since they were let go.
    pub(super) made_at: Option<Instant>,
    /// The tiles listed for `target`, in the order a `Down` there reads them, the nearest first
    /// in each group ([`Sculpting::list_ahead`]): those whose futures are made, or to make.
    pub(super) listed: Vec<TileKey>,
    /// How many of `listed`, from its first, have had the seams about them noted ahead
    /// ([`Sculpting::note_seams_ahead`]) since they were listed.
    pub(super) seamed: usize,
}

impl Ahead {
    /// The futures made ahead that a stroke begun over `earlier` reads from `store`: those of
    /// exactly these strokes, the newest of them the one they were made up to, over this very
    /// store, its current op still the oldest stroke's (see [`Ahead`], "They fail closed").
    /// `None` otherwise, or if none is made: the stroke makes its own.
    pub(super) fn of(
        &self,
        store: &TileStore,
        earlier: &[Pending],
    ) -> Option<&BTreeMap<TileKey, Future>> {
        let newest = earlier.last().map(|stroke| stroke.op.id());
        let oldest = earlier.first().and_then(|stroke| match &stroke.op {
            StrokeOp::Begun(op) => Some(*op),
            StrokeOp::Reserved { .. } => None,
        });
        let theirs = newest.is_some() && newest == self.after;
        let unchanged =
            self.store == Some(store.id()) && oldest.is_some() && store.current_op() == oldest;
        (theirs && unchanged).then_some(&self.tiles)
    }

    /// Lets every future go, and what was listed.
    pub(super) fn clear(&mut self) {
        *self = Self::default();
    }
}

impl Sculpting {
    /// Lets the futures made ahead go once the stroke they were made up to is no longer under
    /// way (see [`Ahead`]): looked at as each step and each `prepare` begins.
    pub(super) fn keep_ahead(&mut self) {
        let after = self.ahead.after;
        let mut under_way = self.strokes.iter().map(|stroke| stroke.op.id());
        if !under_way.any(|op| Some(op) == after) {
            self.ahead.clear();
        }
    }

    /// Whether futures are to be made ahead of a `Down` where the pen hovers over the strokes
    /// under way (see "Warming under a hovering pen" in the module notes), listing them first
    /// where the pen has moved on ([`Sculpting::list_ahead`]).
    ///
    /// Only where a `Down` would begin over the strokes under way
    /// ([`Sculpting::route_on_futures`]) and what a later stroke reads of each is made, every pen
    /// up: their futures are final then. Not for a pen moving fast ([`HOVER_FAST`]): the list
    /// stands as it is, and the frame after looks again, when the pen may have slowed. The
    /// list is made again only once the pen is a quarter of a radius from where it was made
    /// ([`RELIST_RADII`]), or the view or the radius changed; the futures made stay, those
    /// still near.
    pub(super) fn list_to_warm(&mut self, scene: &mut Scene<'_>) -> Warming {
        let Some((at, view)) = self.warm.wanted else {
            return Warming::Nothing;
        };
        let newest = self.strokes.last().map(|stroke| stroke.op.id());
        let over = newest.is_some() && self.restored.is_empty() && self.deferred.is_empty();
        let sealed = self.strokes.iter().all(|stroke| {
            let remaining = stroke.remaining.as_ref();
            remaining.is_some_and(|remaining| remaining.edits.is_sealed())
        });
        if !over || !sealed || self.pen_on_stroke() {
            return Warming::Nothing;
        }
        if self.warm.fast {
            return Warming::Later;
        }
        let theirs = Some(scene.project.store().id());
        if self.ahead.after != newest || self.ahead.store != theirs {
            self.ahead.clear();
            self.ahead.after = newest;
            self.ahead.store = theirs;
        }
        let radius = self.settings.radius();
        let stands = self.ahead.target.is_some_and(|(was, through, with)| {
            let gone = (at.x - was.x).hypot(at.y - was.y);
            through == view && with == radius && gone <= RELIST_RADII * f64::from(radius.get())
        });
        if !stands {
            self.ahead.target = Some((at, view, radius));
            self.list_ahead(scene, at);
        }
        if self.ahead.todo.is_empty() {
            Warming::Nothing
        } else {
            Warming::Now
        }
    }

    /// Makes futures listed ahead in the time a `prepare` has left before `deadline`, a call's
    /// futures time at most ([`FUTURES_TIME`]); none if this call made its batch first
    /// (`made_first`, the futures having waited long: [`AHEAD_STARVED`]). Returns whether some
    /// are still to make.
    pub(super) fn make_ahead_in_time(
        &mut self,
        scene: &mut Scene<'_>,
        made_first: bool,
        deadline: Instant,
    ) -> bool {
        if !made_first {
            let until = deadline.min(Instant::now() + FUTURES_TIME);
            self.make_ahead(scene, Some(until));
        }
        !self.ahead.todo.is_empty()
    }

    /// Makes the futures listed ahead ([`Sculpting::list_to_warm`]), a workers' batch together
    /// ([`TileStore::workers`]) on the store's workers: batches while the clock is before
    /// `until`, or with `None` exactly one, whatever the time.
    pub(super) fn make_ahead(&mut self, scene: &mut Scene<'_>, until: Option<Instant>) {
        if self.ahead.todo.is_empty() {
            return;
        }
        let store = scene.project.store();
        let futures = chain_of(&self.strokes).map(|chain| Futures::new(store, &self.state, &chain));
        let Some(Ok(futures)) = futures else {
            self.ahead.todo.clear();
            return;
        };
        let mut owed = until.is_none();
        while !self.ahead.todo.is_empty()
            && (owed || until.is_some_and(|until| Instant::now() < until))
        {
            owed = false;
            let most = store.workers().max(1);
            let split = self.ahead.todo.len().saturating_sub(most);
            let batch = self.ahead.todo.split_off(split);
            #[cfg(test)]
            super::tests::count_made(batch.len());
            let made = futures.futures_of(store, &batch);
            let made = batch.into_iter().zip(made);
            for (key, future) in made {
                // One that can't be made is left to the `Down` that reads it, which says why.
                if let Ok(future) = future {
                    self.ahead.tiles.insert(key, future);
                }
            }
            self.ahead.made_at = Some(Instant::now());
        }
    }

    /// Lists the futures to make ahead for a pen hovering at `at`: those a `Down` there may read
    /// ([`needed`]), in the order it reads them, the nearest first in each group.
    ///
    /// Along the pen's ray (the `Down` is routed by a raycast, however far the strokes' dabs dig
    /// or raise the surface there); then about its hit on the surface as stored: the hit's
    /// normals, the brush's radius (the first dab's area normal), and the stroke's reach
    /// ([`stroke::REACH_RADII`] radii, its first drag's walk). The hit on the surface the
    /// strokes leave may lie elsewhere along the ray, and so may what the `Down` reads about it:
    /// the list is a good guess, and the `Down` makes what it lacks. At most [`AHEAD_MOST`]; the
    /// futures made before that aren't listed now are let go, the others kept. So is what was
    /// noted ahead of the seams about the tiles listed before, where no preview holds it
    /// ([`sculpt::Preview::let_go_unheld_seams`]): a hover that wanders keeps its last place's.
    pub(super) fn list_ahead(&mut self, scene: &mut Scene<'_>, at: Pixel) {
        self.ahead.todo.clear();
        // What was noted of the seams about the place it hovered before, where no preview came
        // to hold it, goes: it is noted again for where the pen is now.
        if let Some(preview) = &mut self.preview {
            preview.let_go_unheld_seams(scene.project.store());
        }
        let projector = scene.projector;
        let screen = ScreenPoint::new(at.x, at.y).ok();
        let Some(ray) = screen.and_then(|screen| projector.ray(screen)) else {
            return;
        };
        let mut surface = StoreSurface::new(scene.project.store(), scene.index);
        let hit = surface.raycast(&ray);
        // Work ahead of a `Down` that may never come: a raycast that fails lists nothing and
        // says nothing. The `Down` itself reports what it meets.
        if surface.finish().is_err() {
            return;
        }
        let along = Read::Ray {
            ray,
            until: f64::INFINITY,
        };
        let mut reads = vec![along];
        if let Some(hit) = hit {
            let depth = ray.depth_of(hit.position);
            let radius = f64::from(self.settings.radius().get()) * projector.world_per_pixel(depth);
            let about = [0.0, radius, REACH_RADII * radius].map(|radius| Read::Ball {
                center: hit.position,
                radius: rounded_up(hit.position, radius),
            });
            reads.extend(about);
        }
        let (store, index) = (&*scene.project.store(), &*scene.index);
        let under = hit.map(|hit| hit.position.to_f32());
        // How far tile `key`'s box in the index lies from the hit: what each group is sorted by.
        let distance = |key: &TileKey| -> f32 {
            let from = under.zip(index.bvh().bounds(key));
            from.map_or(0.0, |(point, bounds)| bounds.aabb.distance_squared(point))
        };

        let made = self.ahead.of(store, &self.strokes);
        let mut listed: Vec<TileKey> = Vec::new();
        let mut seen: BTreeSet<TileKey> = BTreeSet::new();
        for read in &reads {
            let one = std::slice::from_ref(read);
            let Ok(wanted) = needed(store, index, &self.strokes, one, made) else {
                break;
            };
            let mut group: Vec<TileKey> = wanted.difference(&seen).copied().collect();
            group.sort_by(|a, b| distance(a).total_cmp(&distance(b)));
            seen.extend(group.iter().copied());
            listed.extend(group);
        }
        listed.truncate(AHEAD_MOST);
        let kept: BTreeSet<TileKey> = listed.iter().copied().collect();
        self.ahead.tiles.retain(|key, _| kept.contains(key));
        let todo = listed
            .iter()
            .rev()
            .filter(|key| !self.ahead.tiles.contains_key(key));
        self.ahead.todo = todo.copied().collect();
        self.ahead.listed = listed;
        self.ahead.seamed = 0;
    }

    /// Notes ahead, for the previews about the tiles whose futures are made ahead, what the
    /// first stitch of a stroke begun there looks up ([`sculpt::Preview::warm_seams`]): the
    /// tiles listed, nearest first as their futures were made, a tile at a time while the clock
    /// is before `deadline`, [`SEAMS_A_CALL`] a call at most. Returns whether it noted some and
    /// some are still to note.
    ///
    /// A call out of time notes nothing and asks for nothing: a `prepare` that says more is
    /// wanted has always done some of it, so a loop whose frames overrun isn't kept turning for
    /// work no call gets to. The tiles left wait for a `prepare` with time, which every frame
    /// brings while the strokes land; and a `Down` that finds none noted notes them itself.
    ///
    /// Only once every future listed for where the pen hovers is made, up to the newest stroke
    /// under way: the tiles those futures are of are the ones a `Down` there reads of the
    /// strokes under way, so the ones its first dabs' previews meet theirs on. Never for a pen
    /// moving fast (`warming`, [`Sculpting::list_to_warm`]'s answer), as no future is.
    ///
    /// What is noted follows from the mesh alone, so nothing here can go stale.
    pub(super) fn note_seams_ahead(
        &mut self,
        scene: &mut Scene<'_>,
        warming: Warming,
        deadline: Instant,
    ) -> bool {
        let newest = self.strokes.last().map(|stroke| stroke.op.id());
        let hovering = self.warm.wanted.is_some() && !self.warm.fast;
        let made = warming != Warming::Later
            && self.ahead.todo.is_empty()
            && self.ahead.target.is_some()
            && self.ahead.after == newest;
        let Some(preview) = self.preview.as_mut().filter(|_| hovering && made) else {
            return false;
        };
        let left = self.ahead.listed.iter().skip(self.ahead.seamed);
        let next: Vec<TileKey> = left.take(SEAMS_A_CALL).copied().collect();
        let mut noted = false;
        for tile in &next {
            // Looked at before every tile: one beside a coarse display tile lists many.
            if Instant::now() >= deadline {
                break;
            }
            preview.warm_seams(scene.project.store(), std::slice::from_ref(tile));
            self.ahead.seamed += 1;
            noted = true;
        }
        noted && self.ahead.seamed < self.ahead.listed.len()
    }

    /// Installs, ahead of a pen drawing over strokes still landing, the futures its next samples
    /// read: those about where its dabs head ([`AHEAD_DABS`] spacings past its last, within the
    /// stroke's reach, as [`Sculpting::warm_ahead`] warms the ground there), for as long as the
    /// call's futures time lasts ([`super::over::FUTURES_TIME`]). So a sample's own reads find them
    /// installed, and its step makes none.
    ///
    /// Installing a future early is always right: it is the surface the stroke reads there,
    /// whenever it comes to read it.
    pub(super) fn install_ahead(&mut self, scene: &mut Scene<'_>) {
        let Some((newest, earlier)) = self.strokes.split_last_mut() else {
            return;
        };
        let [Some(previous), Some(last)] = newest.heading else {
            return;
        };
        let StrokeOp::Reserved { token, installed } = &mut newest.op else {
            return;
        };
        let at = |dab: &Dab| {
            let [x, y, z] = dab.position;
            DVec3::new(f64::from(x), f64::from(y), f64::from(z))
        };
        #[expect(
            clippy::cast_precision_loss,
            reason = "a few spacings ahead, a small count"
        )]
        let center = at(&last) + (at(&last) - at(&previous)) * (AHEAD_DABS as f64);
        let reach = REACH_RADII * f64::from(last.radius);
        let read = Read::Ball {
            center,
            radius: rounded_up(center, reach),
        };
        let store = scene.project.store();
        let ahead = self.ahead.of(store, earlier);
        let Ok(wanted) = needed(store, scene.index, earlier, &[read], ahead) else {
            return;
        };
        let missing: Vec<TileKey> = wanted.difference(installed).copied().collect();
        install_in_time(
            scene.project,
            scene.index,
            (&self.state, ahead),
            earlier,
            (token, installed),
            &missing,
            &mut self.installing,
        );
    }
}
