//! Reading the store: its mesh and identity, the tiles it has cached, evaluating tiles and their
//! normals on demand, and pinning them.

use std::borrow::Cow;
use std::collections::BTreeSet;
use std::rc::Rc;

use geom::{BaseMesh, TileKey, Vec3};

use super::config::Evaluation;
use super::error::{CacheError, Invariant};
use super::pins::PinGuard;
use super::round::{Targets, map_parallel, prefix, vertex_count};
use super::{StoreId, TileStore};
use crate::activity::Activity;
use crate::hash::TileMap;
use crate::pool::{Freshness, SlotId};
use crate::slot::EvaluatedSlot;
use crate::tile_bounds::TileBounds;

impl TileStore {
    /// The base mesh the store's tiles subdivide: level 0 and the topology.
    #[must_use]
    pub fn mesh(&self) -> &BaseMesh {
        &self.mesh
    }

    /// This store's identity: no other store the process opens has it, whatever its file or
    /// configuration, so state kept beside a store (a brush stroke's, say) can tell it is still
    /// over the same one. Op ids can't: every store of a site starts its ops at the same
    /// sequence number, and a project reopened after its edits were discarded hands out the
    /// discarded ops' ids again.
    #[must_use]
    pub const fn id(&self) -> StoreId {
        self.id
    }

    /// What the store has done since it opened: tiles loaded and evaluated, and evicted to make
    /// room (see [`Activity`]). Two counts around a call, through [`Activity::since`], give what
    /// that call did.
    #[must_use]
    pub const fn activity(&self) -> Activity {
        self.activity
    }

    /// `key`'s evaluated positions as cached, if they are current: no evaluation, and the read
    /// doesn't count as a use of the tile (which [`TileStore::evaluate`] does, for eviction).
    /// `None` if they aren't current; evaluate them first.
    ///
    /// With [`TileStore::par_map`], for reading many tiles' positions on the workers.
    #[must_use]
    pub fn cached_positions(&self, key: &TileKey) -> Option<&[Vec3]> {
        let id = self.fresh_slot(key)?;
        prefix(&self.evaluated.get(id)?.positions, vertex_count(key)).ok()
    }

    /// `key`'s unit normals as cached, if they are current, as [`TileStore::cached_positions`]
    /// gives positions (current normals go with current positions). `None` if they aren't
    /// current; compute them with [`TileStore::normals`] or [`TileStore::evaluate_many`] first.
    #[must_use]
    pub fn cached_normals(&self, key: &TileKey) -> Option<&[Vec3]> {
        let id = self.fresh_slot(key)?;
        if !self.deps.has_normals(key) {
            return None;
        }
        prefix(&self.evaluated.get(id)?.normals, vertex_count(key)).ok()
    }

    /// The tiles `key`'s normals are computed from, the tile itself included, in key order: the
    /// tile, its apron and the rings at its corners, as [`normal_tiles`](geom::normal_tiles) lists
    /// them for [`Region::of_tile`](geom::Region::of_tile). The list is structural, so it is read
    /// back from what the store recorded when it last computed the tile's normals, and listed
    /// from the mesh only for a tile it hasn't.
    ///
    /// Neighbourhood is symmetric, so these are also the tiles whose normals read `key`'s
    /// positions.
    ///
    /// # Errors
    ///
    /// [`CacheError::NotATile`]; [`CacheError::Eval`] (not for a valid key).
    pub fn normal_inputs(&self, key: &TileKey) -> Result<Cow<'_, [TileKey]>, CacheError> {
        self.check_key(key)?;
        Ok(self.deps.normal_inputs_or_list(&self.mesh, key)?)
    }

    /// `key`'s evaluated slot if its positions are current, found without counting as a use.
    fn fresh_slot(&self, key: &TileKey) -> Option<SlotId<EvaluatedSlot>> {
        (self.evaluated.freshness(key)? == Freshness::Fresh)
            .then(|| self.evaluated.find(key))
            .flatten()
    }

    /// `f` of every item of `items`, in order, computed on the store's evaluation workers
    /// ([`StoreConfig::workers`](super::config::StoreConfig::workers)); a single item is computed
    /// on the calling thread.
    ///
    /// For per-tile work over what the store has cached (with
    /// [`TileStore::cached_positions`] and [`TileStore::cached_normals`]) that is independent
    /// tile by tile: the results come back in item order however the work was spread.
    pub fn par_map<T: Sync, R: Send>(
        &self,
        items: &[T],
        f: impl Fn(&T) -> R + Sync + Send,
    ) -> Vec<R> {
        map_parallel(&self.workers, items, f)
    }

    /// `key`'s evaluated positions, row-major `side²`: cached, else computed coarse to fine
    /// from the base mesh, or from the finest level of its cone that is cached (see the module
    /// notes), with its corrections.
    ///
    /// # Errors
    ///
    /// [`CacheError::NotATile`]; [`CacheError::Full`] if the pins taken with
    /// [`TileStore::pin`] leave too little of the budget for the evaluation's own (never without
    /// such pins); [`CacheError::Store`] if a correction tile can't be read;
    /// [`CacheError::WriteBack`] if making room for one fails; [`CacheError::Eval`] (not for a
    /// valid mesh).
    pub fn evaluate(&mut self, key: TileKey) -> Result<&[Vec3], CacheError> {
        self.sync_pins()?;
        self.poll()?;
        self.check_key(&key)?;
        let id = self.ensure_positions(key)?;
        let slot = self.evaluated.get(id).ok_or(Invariant::StaleHandle)?;
        Ok(prefix(&slot.positions, vertex_count(&key))?)
    }

    /// `key`'s unit normals, row-major `side²`: cached, else computed from the evaluated
    /// positions of its own level's [`normal_tiles`](geom::normal_tiles) (the tile, its apron and
    /// the rings at its corners).
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate`].
    pub fn normals(&mut self, key: TileKey) -> Result<&[Vec3], CacheError> {
        self.sync_pins()?;
        self.poll()?;
        self.check_key(&key)?;
        let id = self.ensure_normals(key)?;
        let slot = self.evaluated.get(id).ok_or(Invariant::StaleHandle)?;
        Ok(prefix(&slot.normals, vertex_count(&key))?)
    }

    /// Makes every tile of `keys` current: its positions, as [`TileStore::evaluate`] gives them,
    /// and for [`Evaluation::Normals`] its normals too, as [`TileStore::normals`] does. The
    /// tiles are built on the store's worker threads, many at a time, with the same bits as one
    /// by one.
    ///
    /// The keys are taken level by level, coarse to fine, in batches: each batch's dependency
    /// cone is listed at once (the parents of the tiles not cached listed in parallel), then built
    /// coarse to fine, each level's tiles in parallel rounds (see the module notes); normals are
    /// computed in parallel once the positions of every tile they read are installed. A batch
    /// takes a share of the evaluated slots the budget has free of pins, at most a few
    /// thousand targets, and is halved and tried again if it runs out of room, down to one
    /// tile, which the budget always holds while nothing else is pinned.
    ///
    /// The tiles are left cached, not pinned: reading them right after with
    /// [`TileStore::evaluate`] or [`TileStore::normals`] finds them unless `keys` hold more than
    /// the budget does, when the first batches' tiles may be evicted again (and are recomputed
    /// when read). Duplicate keys are evaluated once.
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate`]: [`CacheError::NotATile`] before anything is evaluated. The
    /// batches made before an error stay cached.
    pub fn evaluate_many(&mut self, keys: &[TileKey], what: Evaluation) -> Result<(), CacheError> {
        self.evaluate_many_into(keys, what, None)
    }

    /// [`TileStore::evaluate_many`] with [`Evaluation::Normals`], which also hands back the
    /// bounds of every tile whose normals it computed: a box and a cone over the tile's
    /// positions and normals, as [`crate::SpatialIndex::bounds_of`] gives them, computed on the
    /// worker that computed the normals, while they are in its cache.
    ///
    /// For [`crate::SpatialIndex::refit_with`], which uses them in place of reading every tile
    /// back, as long as no correction is written in between (see [`WaveBounds`]). A tile whose
    /// normals were current already, or that has no finite position, gets none.
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate_many`]. The bounds computed before an error are lost.
    #[must_use = "the bounds are what this adds to evaluate_many: pass them to refit_with"]
    pub fn evaluate_many_with_bounds(
        &mut self,
        keys: &[TileKey],
    ) -> Result<WaveBounds, CacheError> {
        let mut bounds = TileMap::default();
        self.evaluate_many_into(keys, Evaluation::Normals, Some(&mut bounds))?;
        Ok(WaveBounds {
            store: Some(self.id),
            edits: self.edits,
            bounds,
        })
    }

    /// [`TileStore::evaluate_many`], putting the bounds of every tile whose normals are
    /// computed in `bounds`, if given (see [`TileStore::evaluate_many_with_bounds`]).
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate_many`].
    fn evaluate_many_into(
        &mut self,
        keys: &[TileKey],
        what: Evaluation,
        mut bounds: Option<&mut TileMap<TileBounds>>,
    ) -> Result<(), CacheError> {
        self.sync_pins()?;
        self.poll()?;
        for key in keys {
            self.check_key(key)?;
        }
        let keys: BTreeSet<TileKey> = keys.iter().copied().collect();
        self.in_batches(&keys, Targets::Released, |store, batch| match what {
            Evaluation::Positions => {
                let held = store.pin_current(batch.iter().copied().collect())?;
                store.unpin_held(&held)
            }
            Evaluation::Normals => store.normals_batch(batch, bounds.as_deref_mut()),
        })
    }

    /// Evaluates every tile of `keys` and pins it, with its correction tile, until the guard is
    /// dropped. The tiles are evaluated as [`TileStore::evaluate_many`] does, in parallel
    /// batches.
    ///
    /// The request is first counted against the budget: the chunks needed to hold its tiles
    /// besides every slot pinned already, plus the slots an evaluation pins while it works (two
    /// levels of a cone, see
    /// [`StoreConfig::minimum_budget`](super::config::StoreConfig::minimum_budget)). If that is
    /// more than the budget, nothing is evicted or pinned. The count packs slots into chunks,
    /// so pins spread over more chunks than that can still leave an evaluation short of room;
    /// then, as when a correction tile can't be read, the pins this call took are released and
    /// the pins held before are untouched.
    ///
    /// # Errors
    ///
    /// [`CacheError::NotATile`]; [`CacheError::OverBudget`]; as [`TileStore::evaluate`].
    pub fn pin(&mut self, keys: &[TileKey]) -> Result<PinGuard, CacheError> {
        self.sync_pins()?;
        self.poll()?;
        let keys: BTreeSet<TileKey> = keys.iter().copied().collect();
        for key in &keys {
            self.check_key(key)?;
        }
        self.check_pin_budget(&keys)?;
        let mut set = Vec::with_capacity(keys.len());
        let pinned = self.in_batches(&keys, Targets::Pinned, |store, batch| {
            store.pin_batch(batch, &mut set)
        });
        if let Err(e) = pinned {
            self.unpin_set(&set)?;
            return Err(e);
        }
        let id = self.next_pin_set;
        self.next_pin_set += 1;
        self.pin_sets.insert(id, set);
        Ok(PinGuard {
            id,
            keys: keys.into_iter().collect(),
            table: Rc::clone(&self.pins),
        })
    }
}

/// Bounds the normals wave of [`TileStore::evaluate_many_with_bounds`] computed, for
/// [`SpatialIndex::refit_with`](crate::index::SpatialIndex::refit_with).
///
/// They are the store's own: only the store makes them, and a refit uses them only over the
/// store that made them, and only while no correction has been written to it since. Every tile
/// then evaluates to the bits it had when they were computed, so each is the bound
/// [`SpatialIndex::bounds_of`](crate::index::SpatialIndex::bounds_of) would compute again.
/// Otherwise the refit computes every bound itself, as
/// [`SpatialIndex::refit`](crate::index::SpatialIndex::refit) does.
#[derive(Debug, Clone, Default)]
pub struct WaveBounds {
    /// The store that computed them.
    pub(crate) store: Option<StoreId>,
    /// That store's count of writes ([`TileStore::edits`]) when they were computed.
    pub(crate) edits: u64,
    /// The bounds, by tile.
    pub(crate) bounds: TileMap<TileBounds>,
}

impl WaveBounds {
    /// How many tiles have bounds.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.bounds.len()
    }

    /// `key`'s bounds, if they were computed and are still the tile's in `store`: the store
    /// that computed them, with no correction written since.
    pub(crate) fn current(&self, store: &TileStore, key: &TileKey) -> Option<TileBounds> {
        let fresh = self.store == Some(store.id()) && self.edits == store.edits();
        fresh.then(|| self.bounds.get(key).copied()).flatten()
    }
}
