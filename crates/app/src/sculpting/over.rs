//! What a stroke reads of the strokes still landing before it: which of their futures a read
//! needs, and getting them installed.
//!
//! The machinery of "Strokes over strokes still landing" in the module notes of [`super`]: how
//! far an earlier stroke's remaining dabs can move each tile ([`Reach`], [`Moves`]), what a
//! later stroke keeps of an earlier one ([`Remaining`]), the reads a call into the stroke engine
//! makes ([`Recorded`], [`Read`]) and the tiles whose futures they need ([`needed`]), installing
//! those ([`install`], [`install_in_time`]) until a call has read only installed futures
//! ([`on_futures`]), and the strokes as the previews read them ([`Open`]).

use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

use geom::TileKey;
use sculpt::{Future, Futures, OpenStrokes, PenDownTiles, SculptError, StoreSurface, StrokeEdits};
use store::{CacheError, OpId, PenDownReading, Project, ReservedOp, SpatialIndex, TileStore};
use stroke::{DVec3, Dab, Hit, Ray, Surface};

use super::ahead::Ahead;
use super::{Pending, Scene, Source, StrokeOp};

/// How far an earlier stroke's dabs still to land reach a tile: the tiles a later stroke must read
/// at their futures, and how far those futures can lie from the tile as the index holds it.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(super) struct Reach {
    /// The dabs not landed whose spheres touch the tile in the index, as each was queued.
    pub(super) dabs: usize,
    /// Their strengths' magnitudes summed, over every dab that ever counted here (landing
    /// lowers only the count): how far they can move one of its vertices together, as draw moves
    /// a vertex at most its strength a dab.
    pub(super) strength: f64,
    /// How far they can move one of its vertices together along each axis, the least (at most 0)
    /// and the most (at least 0): each dab's span, its action's ([`sculpt::dab_span`]), summed
    /// over every dab that ever counted here. Draw moves a vertex along its dab's normal, one
    /// way, so a stroke raising a ridge moves its tiles outward and hardly sideways: their boxes
    /// grow that way alone, and a read beside the ridge needs none of them.
    pub(super) span: sculpt::Span,
    /// The tile's correction scales as it was first reached: what its planes' rounding grows
    /// from ([`sculpt::rounding_bound`]).
    pub(super) scales: [f32; 3],
}

impl Reach {
    /// How far the dabs counted here, landing, can move a vertex of the tile from where it is
    /// now, whichever way: their strengths, and the finest work's rounding as its planes grow by
    /// them ([`sculpt::rounding_bound`]).
    pub(super) fn growth(&self) -> f64 {
        self.strength + sculpt::rounding_bound(self.scales, self.strength)
    }

    /// Counts `dab` here: one more not landed, its strength and its span added. A strength that
    /// isn't a number counts as endless, as its span does: the tile is then asked for by every
    /// read.
    pub(super) fn count(&mut self, dab: &Dab) {
        self.dabs += 1;
        let strength = f64::from(dab.strength.abs());
        self.strength += if strength.is_finite() {
            strength
        } else {
            f64::INFINITY
        };
        sculpt::add_span(&mut self.span, sculpt::dab_span(dab));
    }
}

/// How far the strokes still landing can move the tiles they reach, together: what a later
/// stroke's reads are tested against ([`needed`]).
#[derive(Debug)]
pub(super) struct Moves<'a> {
    /// What a later stroke reads of each of them, oldest first.
    pub(super) kept: Vec<&'a Remaining>,
}

impl<'a> Moves<'a> {
    /// Those of `earlier`, each with its [`Remaining`] (one without is passed over).
    pub(super) fn of(earlier: &'a [Pending]) -> Self {
        let remaining = earlier
            .iter()
            .filter_map(|stroke| stroke.remaining.as_ref());
        Self {
            kept: remaining.collect(),
        }
    }

    /// Whether a dab of theirs still to land reaches tile `key`.
    pub(super) fn changed(&self, key: &TileKey) -> bool {
        self.kept.iter().any(|kept| kept.reach.contains_key(key))
    }

    /// The tiles their dabs still to land reach.
    pub(super) fn tiles(&self) -> impl Iterator<Item = &TileKey> {
        self.kept.iter().flat_map(|kept| kept.reach.keys())
    }

    /// The farthest they can move a vertex of any tile, whichever way: a bound, never lowered.
    pub(super) fn most(&self) -> f64 {
        self.kept.iter().map(|kept| kept.most).sum()
    }

    /// How far they can move a vertex of tile `key` along each axis, the least and the most,
    /// their rounding included; `None` for a tile they don't reach.
    ///
    /// Each stroke's span of the tile ([`Reach`]) summed, the strokes landing one after another.
    /// And the finest work's rounding, either way along every axis, once for each stroke
    /// reaching the tile: each stroke builds its vertices unrounded and rounds them as it writes,
    /// so each adds an error of its own, on planes grown by then by at most all the strokes'
    /// strengths together ([`sculpt::rounding_bound`] of the largest scales any of them found
    /// and that sum, which bounds every stroke's). Sideways of a ridge this rounding is all the
    /// margin there is, so it is the sum, not one stroke's.
    pub(super) fn span(&self, key: &TileKey) -> Option<sculpt::Span> {
        let reaching = self.kept.iter().filter_map(|kept| kept.reach.get(key));
        let mut span = sculpt::Span::default();
        let (mut scales, mut strength, mut strokes) = ([0.0_f32; 3], 0.0_f64, 0.0_f64);
        for reach in reaching {
            sculpt::add_span(&mut span, reach.span);
            let found = scales.iter_mut().zip(reach.scales);
            for (most, scale) in found {
                *most = most.max(scale);
            }
            strength += reach.strength;
            strokes += 1.0;
        }
        if strokes == 0.0 {
            return None;
        }
        let rounding = strokes * sculpt::rounding_bound(scales, strength);
        sculpt::add_span(&mut span, ([-rounding; 3], [rounding; 3]));
        Some(span)
    }

    /// The box every point of tile `key` lies in once they have landed: its box in `index`,
    /// which holds it as stored, grown by their span of it ([`Moves::span`]; not at all for a
    /// tile they don't reach), and by the rounding of its `f32` bits. `None` for a tile `index`
    /// has no box of.
    pub(super) fn grown(&self, index: &SpatialIndex, key: &TileKey) -> Option<Corners> {
        let stored = index.bvh().bounds(key)?.aabb;
        let (least, most) = self.span(key).unwrap_or_default();
        let [lx, ly, lz] = least;
        let [hx, hy, hz] = most;
        let (low, high) = widened(stored);
        Some((low + DVec3::new(lx, ly, lz), high + DVec3::new(hx, hy, hz)))
    }
}

/// `aabb`'s corners, a little apart: by the rounding of its `f32` bits.
pub(super) fn widened(aabb: store::Aabb) -> Corners {
    let (low, high) = (aabb.min(), aabb.max());
    let (low, high) = (
        DVec3::new(f64::from(low.x), f64::from(low.y), f64::from(low.z)),
        DVec3::new(f64::from(high.x), f64::from(high.y), f64::from(high.z)),
    );
    let size = 1.0 + low.length().max(high.length());
    let slack = 1e-6 * size;
    let by = DVec3::new(slack, slack, slack);
    (low - by, high + by)
}

/// What a later stroke reads of an earlier one still landing (the stroke-while-landing spec,
/// "What a new stroke reads").
///
/// Kept up as the stroke places its dabs, off the pen's path ([`crate::frame::Brush::prepare`], and
/// at the next `Down` for those placed since): its dabs not landed, brushed over its pen-down tiles
/// without writing, from which the futures of the tiles they touch are made ([`Futures`]), and
/// which tiles those are, with how far they can move. Sealed at its pen's lift; its landing drops
/// what landed ([`Remaining::land`]).
#[derive(Debug)]
pub(super) struct Remaining {
    /// Its dabs placed here and not landed, with the tiles each touches and edits at its pen-down.
    pub(super) edits: StrokeEdits,
    /// Its pen-down tiles its dabs were brushed over: the store's for the oldest stroke, the
    /// futures of the strokes before it for any other. Extended as dabs come.
    pub(super) pen_down: PenDownTiles,
    /// The number, counted from the stroke's first dab, of the first not placed here yet.
    pub(super) next: u32,
    /// Every tile its dabs placed here and not landed reach in the index, as they were queued.
    pub(super) reach: BTreeMap<TileKey, Reach>,
    /// The largest growth ([`Reach::growth`]) of a tile of `reach`, ever: a bound, never lowered.
    pub(super) most: f64,
}

impl Remaining {
    /// Nothing placed yet of `stroke`, which follows `earlier`.
    pub(super) fn new(stroke: &Pending, earlier: &[Pending]) -> Self {
        let after = earlier.last().map(|before| before.op.id());
        Self {
            edits: StrokeEdits::new(stroke.op.id(), after),
            pen_down: PenDownTiles::default(),
            next: stroke.landed,
            reach: BTreeMap::new(),
            most: 0.0,
        }
    }

    /// Places here the dabs `stroke` has queued and not placed here yet, and seals it once its
    /// pen is up. Its pen-down tiles are the store's for the oldest stroke (`earlier` empty), the
    /// futures of `earlier` for any other, made from the session's stroke state `state`.
    ///
    /// # Errors
    ///
    /// [`SculptError`] as [`PenDownTiles::extend_oldest`], [`PenDownTiles::extend_futures`],
    /// [`Futures::new`] and [`StrokeEdits::place`]; [`sculpt::FutureError::Chain`] if one of
    /// `earlier` has none made. Nothing more is placed then, and the caller makes it again.
    pub(super) fn extend(
        &mut self,
        store: &mut TileStore,
        state: &sculpt::Stroke,
        earlier: &[Pending],
        stroke: &mut Pending,
    ) -> Result<(), SculptError> {
        let skip = usize::try_from(self.next.saturating_sub(stroke.landed)).unwrap_or(usize::MAX);
        let new = stroke.dabs.len().saturating_sub(skip);
        if new > 0 {
            let fresh_sources = stroke.sources.iter().skip(skip);
            let touched = fresh_sources.flat_map(|source| &source.tiles);
            let keys: BTreeSet<TileKey> = touched.copied().collect();
            if earlier.is_empty() {
                self.pen_down.extend_oldest(store, stroke.op.id(), &keys)?;
            } else {
                let chain = chain_of(earlier).ok_or(sculpt::FutureError::Chain)?;
                let futures = Futures::new(store, state, &chain)?;
                let before = earlier.last().map(|last| last.op.id());
                if self.pen_down.after() == before {
                    self.pen_down.extend_futures(store, &futures, &keys)?;
                } else {
                    self.pen_down = PenDownTiles::of_futures(store, &futures, &keys)?;
                }
            }
            let queued: &[Dab] = stroke.dabs.make_contiguous();
            let dabs = queued.get(skip..).unwrap_or_default();
            self.edits.place(store, dabs, &self.pen_down)?;
            let placed = stroke.dabs.iter().zip(&stroke.sources).skip(skip);
            for (dab, source) in placed {
                for key in &source.tiles {
                    let tile = match self.reach.entry(*key) {
                        std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
                        std::collections::btree_map::Entry::Vacant(entry) => {
                            let scales = store.correction(*key)?.scales();
                            entry.insert(Reach {
                                scales,
                                ..Reach::default()
                            })
                        }
                    };
                    tile.count(dab);
                    self.most = self.most.max(tile.growth());
                }
            }
            let count = u32::try_from(new).unwrap_or(u32::MAX);
            self.next = self.next.saturating_add(count);
        }
        if !stroke.open() && !self.edits.is_sealed() {
            self.edits.seal();
        }
        Ok(())
    }

    /// The oldest `count` dabs of `stroke` have landed, from its first not landed at `landed`
    /// with `sources` (theirs, oldest first): those placed here leave its edits and its reach.
    pub(super) fn land<'a>(
        &mut self,
        landed: u32,
        count: usize,
        sources: impl Iterator<Item = &'a Source>,
    ) {
        let here = usize::try_from(self.next.saturating_sub(landed)).unwrap_or(usize::MAX);
        let leaving = count.min(here);
        self.edits.land(leaving);
        for source in sources.take(leaving) {
            for key in &source.tiles {
                if let Some(tile) = self.reach.get_mut(key) {
                    tile.dabs = tile.dabs.saturating_sub(1);
                    if tile.dabs == 0 {
                        self.reach.remove(key);
                    }
                }
            }
        }
        let after = landed.saturating_add(u32::try_from(count).unwrap_or(u32::MAX));
        self.next = self.next.max(after);
    }

    /// The stroke it followed has ended, and its op has begun: it is the oldest stroke now, and
    /// its next dabs are brushed over its own pen-down tiles ([`StrokeEdits::earlier_landed`]).
    pub(super) fn earlier_landed(&mut self) {
        self.edits.earlier_landed();
        self.pen_down = PenDownTiles::default();
    }
}

/// The edits of `strokes`, oldest first, if each has its [`Remaining`] made: the chain of futures
/// a stroke after them reads.
pub(super) fn chain_of(strokes: &[Pending]) -> Option<Vec<&StrokeEdits>> {
    let edits = strokes
        .iter()
        .map(|stroke| stroke.remaining.as_ref().map(|remaining| &remaining.edits));
    edits.collect()
}

/// How many rounds of reading a later stroke's surface and installing the futures it found
/// missing ([`on_futures`]) may go before it gives up.
///
/// Each installs at least one tile, and the tiles that can be missing are those the earlier
/// strokes still change, so a few always do.
pub(super) const FUTURE_ROUNDS: usize = 32;

/// Why a call into the stroke engine on a reserved stroke's surface ([`on_futures`]) gave nothing.
#[derive(Debug)]
pub(super) enum OnFutures {
    /// The surface met a store error: the call's dabs are dropped, and its stroke ends, as any
    /// stroke's after a store error.
    Surface(SculptError),
    /// The futures it read couldn't be had (past the open strokes' share, a tile that can't be
    /// evaluated, a surface that kept reaching further): it must wait for the strokes before it
    /// to land.
    Futures,
}

/// What a call into the stroke engine read of a surface, as far as which tiles it can have read
/// goes ([`Recorded`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum Read {
    /// A raycast: the tiles along `ray` up to `until`, its hit's parameter (infinite for a miss).
    Ray {
        /// The ray.
        ray: Ray,
        /// The parameter its hit lies at; infinite if it hit nothing.
        until: f64,
    },
    /// The surface within `radius` of `center`, normals included: a walk, a region's closest
    /// point, an area normal, a hit's normal.
    Ball {
        /// The centre.
        center: DVec3,
        /// The radius, rounded up as the surface's own queries round it ([`rounded_up`]).
        radius: f64,
    },
}

/// `radius` around `center` rounded well up, as the surface's and the previews' index queries
/// round a dab's radius: so a vertex a read finds inside lies within it.
pub(super) fn rounded_up(center: DVec3, radius: f64) -> f64 {
    let size = 1.0 + center.x.abs().max(center.y.abs()).max(center.z.abs());
    radius.mul_add(1.001, 1e-6 * size)
}

/// A surface that notes what each call reads ([`Read`]): what a reserved stroke's call is checked
/// against ([`on_futures`]).
pub(super) struct Recorded<'s, S> {
    /// The surface.
    pub(super) surface: &'s mut S,
    /// What was read, in order.
    pub(super) reads: Vec<Read>,
}

impl<S> Recorded<'_, S> {
    /// Notes the surface within `radius` of `center` as read.
    pub(super) fn ball(&mut self, center: DVec3, radius: f64) {
        self.reads.push(Read::Ball {
            center,
            radius: rounded_up(center, radius),
        });
    }
}

impl<S: Surface> Surface for Recorded<'_, S> {
    fn raycast(&mut self, ray: &Ray) -> Option<Hit> {
        let hit = self.surface.raycast(ray);
        let until = hit.map_or(f64::INFINITY, |hit| ray.depth_of(hit.position));
        self.reads.push(Read::Ray { ray: *ray, until });
        if let Some(hit) = hit {
            // Its normal reads the tile it hit and that tile's ring.
            self.ball(hit.position, 0.0);
        }
        hit
    }

    fn closest_on_region(&mut self, ray: &Ray, near: &Hit, reach: f64) -> Option<Hit> {
        self.ball(near.position, reach);
        self.surface.closest_on_region(ray, near, reach)
    }

    fn continuous(&mut self, a: &Hit, b: &Hit, reach: f64) -> bool {
        self.ball(a.position, reach);
        self.surface.continuous(a, b, reach)
    }

    fn area_normal(
        &mut self,
        center: DVec3,
        facing: DVec3,
        radius: f64,
        hardness: f32,
    ) -> Option<DVec3> {
        // The area normal's own query rounds the radius up as `rounded_up` does; the preview of
        // a dab placed here reaches as far.
        self.ball(center, radius);
        self.surface.area_normal(center, facing, radius, hardness)
    }
}

/// The tiles `reads` may have read whose futures differ from the store's tiles.
///
/// Of those the strokes `earlier` still change: each whose box in `index`, grown by how far their
/// dabs can move it along each axis ([`Moves::grown`]), a read reaches, and those in the ring of
/// any tile a ball reaches by its normals.
///
/// What a later stroke's surface must have installed before that read for it to read the surface
/// those strokes leave (the store's "Ops reserved ahead"): a tile they don't change is read as it
/// is, which it stays. A ray is checked against each tile they change in turn (the index has no
/// query of boxes grown tile by tile), from the reach each [`Remaining`] keeps.
///
/// **Where the future is at hand** (`ahead`, the futures of `earlier` made ahead of the stroke,
/// [`Ahead::of`]), the tile's two boxes are known exactly, with no growth: the one it has in
/// `index`, which a read finds it in until its future is installed, and its future's
/// ([`Future::bounds`]), which a read finds it in from then on. A read reaching neither reads
/// nothing of the tile either way, so its future needn't be installed for it: the bound on how
/// far the tile can move, which a heavy stroke makes many tiles wide, is only for the tiles
/// whose futures aren't made yet. The tiles in the ring of a tile a ball reaches are asked for as
/// before, whatever their boxes: the ball reads that tile's normals, which read them.
///
/// # Errors
///
/// As [`TileStore::normal_inputs`].
pub(super) fn needed(
    store: &TileStore,
    index: &SpatialIndex,
    earlier: &[Pending],
    reads: &[Read],
    ahead: Option<&BTreeMap<TileKey, Future>>,
) -> Result<BTreeSet<TileKey>, CacheError> {
    let moves = Moves::of(earlier);
    let changed = |key: &TileKey| moves.changed(key);
    let most = moves.most();
    // The boxes a read must reach to read anything of tile `key`: its box in the index grown by
    // how far it can move; or, its future at hand, that box as it is and its future's.
    let boxes = |key: &TileKey| -> Option<(Corners, Option<Corners>)> {
        let exact = ahead
            .and_then(|ahead| ahead.get(key))
            .and_then(Future::bounds);
        match exact {
            Some(exact) => {
                let stored = index.bvh().bounds(key)?.aabb;
                Some((widened(stored), Some(widened(exact))))
            }
            None => Some((moves.grown(index, key)?, None)),
        }
    };
    // Whether a read reaches tile `key`, `meets` saying whether it meets a box; a tile with no
    // box in the index counts as reached.
    let reaches = |key: &TileKey, meets: &dyn Fn(DVec3, DVec3) -> bool| -> bool {
        boxes(key).is_none_or(|(stored, future)| {
            meets(stored.0, stored.1) || future.is_some_and(|future| meets(future.0, future.1))
        })
    };
    let mut needed: BTreeSet<TileKey> = BTreeSet::new();
    for read in reads {
        match *read {
            Read::Ray { ray, until } => {
                let crossed = moves.tiles().filter(|key| {
                    !needed.contains(*key)
                        && reaches(key, &|low, high| ray_meets_box(&ray, until, low, high))
                });
                let crossed: Vec<TileKey> = crossed.copied().collect();
                needed.extend(crossed);
            }
            Read::Ball { center, radius } => {
                let within = rounded_up(center, radius + most);
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "the index's query is f32; the radius is rounded up past its rounding"
                )]
                let (at, query) = (center.to_f32(), within as f32);
                let near = index.query_sphere(at, query);
                for key in near {
                    let touched = reaches(&key, &|low, high| {
                        box_distance_squared(center, low, high) <= radius * radius
                    });
                    if !touched {
                        continue;
                    }
                    if changed(&key) {
                        needed.insert(key);
                    }
                    let ring = store.normal_inputs(&key)?;
                    needed.extend(ring.iter().filter(|input| changed(input)));
                }
            }
        }
    }
    Ok(needed)
}

/// A box's corners, the least and the most: what a read is tested against ([`needed`]).
pub(super) type Corners = (DVec3, DVec3);

/// Whether `ray`, from its origin up to parameter `until`, meets the box from `low` to `high`
/// (the slab test, its boundary included).
pub(super) fn ray_meets_box(ray: &Ray, until: f64, low: DVec3, high: DVec3) -> bool {
    let (origin, direction) = (ray.origin(), ray.direction());
    let axes = [
        (origin.x, direction.x, low.x, high.x),
        (origin.y, direction.y, low.y, high.y),
        (origin.z, direction.z, low.z, high.z),
    ];
    let mut span = (0.0_f64, until);
    for (from, along, least, most) in axes {
        if along == 0.0 {
            if from < least || from > most {
                return false;
            }
            continue;
        }
        let (a, b) = ((least - from) / along, (most - from) / along);
        span = (span.0.max(a.min(b)), span.1.min(a.max(b)));
        if span.0 > span.1 {
            return false;
        }
    }
    true
}

/// The squared distance from `point` to the box from `low` to `high`: 0 inside it.
pub(super) fn box_distance_squared(point: DVec3, low: DVec3, high: DVec3) -> f64 {
    let outside = |at: f64, least: f64, most: f64| (least - at).max(at - most).max(0.0);
    let d = DVec3::new(
        outside(point.x, low.x, high.x),
        outside(point.y, low.y, high.y),
        outside(point.z, low.z, high.z),
    );
    d.dot(d)
}

/// Installs the futures of `keys` not `installed` yet for the stroke reserved as `token`, in
/// `project`'s store.
///
/// Each is the one made ahead (`ahead`, the futures of the strokes `earlier` made under a
/// hovering pen, [`Ahead::of`]) if there is one, else made now by `futures`, the futures of
/// `earlier` (each with its [`Remaining`]), those together on the store's workers, a batch of
/// them at a time ([`TileStore::workers`]), each batch installed before the next is made;
/// each is labelled with the newest of the strokes whose dabs reach the tile, and refitted in
/// `index` so the stroke's raycasts (and its previews' queries) find it there.
///
/// The label is the op of the record the tile is taken to hold once they land: provisional until
/// the store's restore and roll-over of a chain settle what a roll-over writes it as (the
/// stroke-while-landing plan, Task 4); a stroke that reaches a tile without changing it labels it
/// all the same.
///
/// # Errors
///
/// [`OnFutures::Futures`] if a future can't be made or installed (past the open strokes' share)
/// or a tile refitted: those installed before it stay.
pub(super) fn install(
    project: &mut Project,
    index: &mut SpatialIndex,
    (futures, ahead): (&Futures<'_>, Option<&BTreeMap<TileKey, Future>>),
    earlier: &[Pending],
    token: &ReservedOp,
    installed: &mut BTreeSet<TileKey>,
    keys: &BTreeSet<TileKey>,
) -> Result<(), OnFutures> {
    let fresh: Vec<TileKey> = keys.difference(installed).copied().collect();
    let store = project.store();
    let mut added: Vec<TileKey> = Vec::with_capacity(fresh.len());
    let mut outcome = Ok(());
    let at_hand = |key: &TileKey| ahead.and_then(|ahead| ahead.get(key));
    // A workers' batch at a time: those of it not made ahead are made now, together, and the
    // batch installed before the next is made. So no more futures are held made than are about
    // to go in, and a refusal (past the open strokes' share) stops the making with it.
    let batches = fresh.chunks(store.workers().max(1));
    'batches: for batch in batches {
        let cold = batch.iter().filter(|key| at_hand(key).is_none());
        let cold: Vec<TileKey> = cold.copied().collect();
        #[cfg(test)]
        super::tests::count_made(cold.len());
        let mut made_now = futures.futures_of(store, &cold).into_iter();
        for &key in batch {
            let tile = match at_hand(&key) {
                Some(future) => Some(future.tile().clone()),
                None => made_now.next().and_then(Result::ok).map(Future::into_tile),
            };
            let Some(tile) = tile else {
                outcome = Err(OnFutures::Futures);
                break 'batches;
            };
            let reaching = earlier.iter().rev().find(|stroke| {
                let remaining = stroke.remaining.as_ref();
                remaining.is_some_and(|remaining| remaining.reach.contains_key(&key))
            });
            let written_by = reaching.map(|stroke| stroke.op.id());
            #[cfg(test)]
            if !super::tests::may_install() {
                // As the store refuses one past the open strokes' share.
                outcome = Err(OnFutures::Futures);
                break 'batches;
            }
            if store
                .install_pen_down_of(token, key, tile, written_by)
                .is_err()
            {
                outcome = Err(OnFutures::Futures);
                break 'batches;
            }
            installed.insert(key);
            added.push(key);
        }
    }
    let refitted = index.refit(project.store(), &added);
    outcome?;
    refitted.map_err(|_why| OnFutures::Futures)
}

/// Runs `call`, a call into the stroke engine, on the surface of the stroke reserved as `token`,
/// after the strokes `earlier` still landing, until it has every future it reads.
///
/// It runs again after installing every future its reads found missing ([`needed`]), until one
/// run reads only tiles whose futures it had, and returns that run's result. The stroke engine is
/// deterministic, so that run is what the call gives on the surface those strokes leave; `call`
/// must start each run afresh (from a copy of the engine's stroke, say). The futures are made in
/// one [`Futures`] for the call, its decisions made once, from the session's stroke state
/// `state`.
///
/// # Errors
///
/// [`OnFutures::Surface`] for a store error the surface met; [`OnFutures::Futures`] if the
/// futures can't be had, or a run still finds some missing after [`FUTURE_ROUNDS`].
pub(super) fn on_futures<T>(
    scene: &mut Scene<'_>,
    (state, ahead): (&sculpt::Stroke, &Ahead),
    earlier: &[Pending],
    token: &ReservedOp,
    installed: &mut BTreeSet<TileKey>,
    mut call: impl FnMut(&mut Recorded<'_, StoreSurface<'_>>) -> T,
) -> Result<T, OnFutures> {
    let ahead = ahead.of(scene.project.store(), earlier);
    let mut futures: Option<Futures<'_>> = None;
    for _ in 0..FUTURE_ROUNDS {
        let (made, reads) = {
            let mut surface = StoreSurface::of_reserved(scene.project.store(), scene.index, token)
                .map_err(OnFutures::Surface)?;
            let mut recorded = Recorded {
                surface: &mut surface,
                reads: Vec::new(),
            };
            let made = call(&mut recorded);
            let reads = recorded.reads;
            surface.finish().map_err(OnFutures::Surface)?;
            (made, reads)
        };
        let wanted = needed(scene.project.store(), scene.index, earlier, &reads, ahead)
            .map_err(|_why| OnFutures::Futures)?;
        if wanted.is_subset(installed) {
            return Ok(made);
        }
        let made_futures = if let Some(kept) = futures.take() {
            kept
        } else {
            let chain = chain_of(earlier).ok_or(OnFutures::Futures)?;
            Futures::new(scene.project.store(), state, &chain).map_err(|_why| OnFutures::Futures)?
        };
        install(
            scene.project,
            scene.index,
            (&made_futures, ahead),
            earlier,
            token,
            installed,
            &wanted,
        )?;
        futures = Some(made_futures);
    }
    Err(OnFutures::Futures)
}

/// Per display tile, the tiles the strokes before a reserved stroke still change that its display
/// preview there reads ([`display_reads`]).
pub(super) type DisplayReads = BTreeMap<TileKey, BTreeSet<TileKey>>;

/// How long the previews of one call may spend installing futures for a reserved stroke.
///
/// [`Open`]'s [`OpenStrokes::ready`]: what they need past that is installed by the calls after,
/// so a display preview over much of an earlier stroke never holds a frame for its backlog. A
/// future costs about what landing the tile costs: a fraction of a millisecond for a tile few
/// dabs reach, several for one under many.
pub(super) const FUTURES_TIME: Duration = Duration::from_millis(2);

/// What one call's previews may still spend installing futures ([`FUTURES_TIME`]).
#[derive(Debug, Clone, Copy)]
pub(super) struct Installing {
    /// When its time is up.
    pub(super) until: Instant,
    /// Whether it has installed any: the first futures of a call are always installed, so the
    /// calls make progress however dear one is.
    pub(super) any: bool,
}

impl Installing {
    /// A call's, its time from now.
    pub(super) fn new() -> Self {
        Self {
            until: Instant::now() + FUTURES_TIME,
            any: false,
        }
    }

    /// Whether it may install more: always its first, then until its time is up.
    pub(super) fn more(&self) -> bool {
        !self.any || Instant::now() < self.until
    }
}

/// Per display tile, the tiles the strokes `earlier` still change that a later stroke's display
/// preview of it reads.
///
/// Every one whose ring (its normal inputs, itself included) holds a finest tile under it: the
/// tiles under it and around those, as neighbourhood goes both ways. A tile whose ring can't be
/// listed is left out, its future made when a finest preview reads it. Made once a stroke: what
/// the strokes before it change is set by then (their pens are up), and only shrinks as they land,
/// the future of a tile landed being the tile itself.
pub(super) fn display_reads(store: &TileStore, earlier: &[Pending]) -> DisplayReads {
    let kept = earlier
        .iter()
        .filter_map(|stroke| stroke.remaining.as_ref());
    let changed = kept.flat_map(|remaining| remaining.reach.keys());
    let mut reads = DisplayReads::new();
    for key in changed {
        let Ok(ring) = store.normal_inputs(key) else {
            continue;
        };
        let above = ring.iter().flat_map(sculpt::display_ancestors);
        for display in above {
            reads.entry(display).or_default().insert(*key);
        }
    }
    reads
}

/// The strokes under way as the previews read them ([`OpenStrokes`]).
///
/// Each one's pen-down shapes, and the futures a later one's previews need installed first (see
/// "Strokes over strokes still landing" in the module notes).
#[derive(Debug)]
pub(super) struct Open<'a> {
    /// The project.
    pub(super) project: &'a mut Project,
    /// Its spatial index, refitted as futures are installed.
    pub(super) index: &'a mut SpatialIndex,
    /// The strokes under way, oldest first.
    pub(super) strokes: &'a mut [Pending],
    /// The session's stroke state, which the futures are made from.
    pub(super) state: &'a sculpt::Stroke,
    /// What the brush's step (or its `prepare`) may still spend installing futures
    /// ([`FUTURES_TIME`]): one for all the step's calls.
    pub(super) installing: &'a mut Installing,
    /// The futures made ahead under a hovering pen: installed without being made again.
    pub(super) ahead: &'a Ahead,
}

impl OpenStrokes for Open<'_> {
    fn current(&mut self) -> (PenDownReading<'_>, &SpatialIndex) {
        (self.project.store().read_pen_down(), self.index)
    }

    fn read(&mut self, op: OpId) -> Result<(PenDownReading<'_>, &SpatialIndex), CacheError> {
        let stroke = self.strokes.iter().find(|stroke| stroke.op.id() == op);
        let store = self.project.store();
        match stroke.map(|stroke| &stroke.op) {
            Some(StrokeOp::Begun(_)) => Ok((store.read_pen_down(), self.index)),
            Some(StrokeOp::Reserved { token, .. }) => {
                Ok((store.read_pen_down_of(token)?, self.index))
            }
            None => Err(CacheError::NotReserved(op)),
        }
    }

    fn reserved(&self, op: OpId) -> bool {
        self.strokes
            .iter()
            .any(|stroke| stroke.op.id() == op && matches!(stroke.op, StrokeOp::Reserved { .. }))
    }

    fn ready(&mut self, op: OpId, dabs: &[Dab], displays: &[TileKey]) -> bool {
        let Some(at) = self.strokes.iter().position(|stroke| stroke.op.id() == op) else {
            return false;
        };
        let (earlier, rest) = self.strokes.split_at_mut(at);
        let Some(stroke) = rest.first_mut() else {
            return false;
        };
        let StrokeOp::Reserved { token, installed } = &mut stroke.op else {
            return true;
        };
        // Each dab's sphere, a little wider than the previews' query of it.
        let balls = dabs.iter().map(|dab| {
            let [x, y, z] = dab.position;
            let center = DVec3::new(f64::from(x), f64::from(y), f64::from(z));
            Read::Ball {
                center,
                radius: rounded_up(center, f64::from(dab.radius) * 1.001),
            }
        });
        let reads: Vec<Read> = balls.collect();
        let store = self.project.store();
        let ahead = self.ahead.of(store, earlier);
        let Ok(mut wanted) = needed(store, self.index, earlier, &reads, ahead) else {
            return false;
        };
        // The tiles the strokes before still change, as they land: a future of one landed is
        // the tile itself, so none is installed.
        let moves = Moves::of(earlier);
        let changed = |key: &TileKey| moves.changed(key);
        if !displays.is_empty() {
            let under = stroke
                .display_reads
                .get_or_insert_with(|| display_reads(store, earlier));
            let read = displays.iter().filter_map(|display| under.get(display));
            for tiles_read in read {
                wanted.extend(tiles_read.iter().filter(|key| changed(key)));
            }
        }
        let missing: Vec<TileKey> = wanted.difference(installed).copied().collect();
        install_in_time(
            self.project,
            self.index,
            (self.state, ahead),
            earlier,
            (token, installed),
            &missing,
            self.installing,
        )
    }
}

/// Installs the futures of `missing` for the stroke reserved as `token` after the strokes
/// `earlier`, as many as `installing` leaves time for. Returns whether every one is installed.
///
/// A workers' batch ([`TileStore::workers`]) between looks at the clock
/// ([`FUTURES_TIME`]); the rest are for
/// the calls after. They are those made ahead (`ahead`) where there are, else made now from the
/// session's stroke state `state` ([`install`]).
pub(super) fn install_in_time(
    project: &mut Project,
    index: &mut SpatialIndex,
    (state, ahead): (&sculpt::Stroke, Option<&BTreeMap<TileKey, Future>>),
    earlier: &[Pending],
    (token, installed): (&ReservedOp, &mut BTreeSet<TileKey>),
    missing: &[TileKey],
    installing: &mut Installing,
) -> bool {
    if missing.is_empty() {
        return true;
    }
    let Some(chain) = chain_of(earlier) else {
        return false;
    };
    let Ok(futures) = Futures::new(project.store(), state, &chain) else {
        return false;
    };
    let together = project.store().workers().max(1);
    let batches = missing.chunks(together);
    for some in batches {
        if !installing.more() {
            return false;
        }
        let now: BTreeSet<TileKey> = some.iter().copied().collect();
        let both = (&futures, ahead);
        if install(project, index, both, earlier, token, installed, &now).is_err() {
            return false;
        }
        installing.any = true;
    }
    true
}
