//! A budgeted pool of fixed-size plain-data slots with clock eviction.
//!
//! Slots are carved from chunks, each one zeroed allocation (lazily backed by the OS), mapped
//! only while a shared [`Budget`] allows another whole chunk, so the pools' memory never exceeds
//! it. A slot holds one tile, named by its [`TileKey`]. When every mapped slot is in use, the
//! clock picks a victim among the unpinned ones, skipping those referenced since its last pass,
//! and writes it back first if it is dirty.

use std::collections::HashMap;

use bytemuck::Pod;
use geom::TileKey;
use thiserror::Error;

use crate::StoreError;

/// Bytes the pools may use, shared by all of them.
#[derive(Debug)]
pub(crate) struct Budget {
    limit: usize,
    used: usize,
}

impl Budget {
    /// A budget of `limit` bytes, none used.
    pub(crate) const fn new(limit: usize) -> Self {
        Self { limit, used: 0 }
    }

    /// The limit.
    pub(crate) const fn limit(&self) -> usize {
        self.limit
    }

    /// Bytes the pools' mapped chunks hold.
    pub(crate) const fn used(&self) -> usize {
        self.used
    }

    /// Takes `bytes` if they fit under the limit; whether they did.
    fn try_take(&mut self, bytes: usize) -> bool {
        let fits = self
            .used
            .checked_add(bytes)
            .is_some_and(|n| n <= self.limit);
        if fits {
            self.used += bytes;
        }
        fits
    }

    /// Gives back `bytes` taken earlier.
    fn give(&mut self, bytes: usize) {
        self.used = self.used.saturating_sub(bytes);
    }
}

/// A slot handle. Valid until the slot is evicted or removed: the pool checks the generation on
/// every use, so a stale handle finds nothing rather than another tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SlotId {
    chunk: u32,
    index: u32,
    generation: u32,
}

/// Why a pool operation failed.
#[derive(Debug, Error)]
pub(crate) enum PoolError {
    /// Every slot is pinned and the budget allows no more chunks.
    #[error("every slot in the pool is pinned and the budget is spent")]
    Full,
    /// The key already has a slot.
    #[error("the tile already has a slot")]
    Present,
    /// The handle's slot was evicted or removed.
    #[error("the slot handle is stale")]
    Stale,
    /// `unpin` of a slot that isn't pinned.
    #[error("the slot isn't pinned")]
    NotPinned,
    /// `remove` of a pinned slot.
    #[error("the slot is pinned")]
    Pinned,
    /// `remove` of a dirty slot, whose contents would be lost.
    #[error("the slot is dirty")]
    Dirty,
    /// Writing a dirty victim back failed; it keeps its slot.
    #[error("writing a dirty slot back failed")]
    WriteBack(#[from] StoreError),
}

/// One slot's bookkeeping.
#[derive(Debug, Clone, Copy, Default)]
struct Meta {
    /// The tile it holds; `None` while free.
    key: Option<TileKey>,
    /// Bumped whenever the slot is freed, so older handles go stale.
    generation: u32,
    pins: u32,
    /// Used since the clock last passed.
    referenced: bool,
    /// Changed since it was last written back.
    dirty: bool,
}

/// A slot position: chunk and index in it.
type Pos = (u32, u32);

/// A budgeted pool of `T` slots. Single-threaded: it belongs to the thread that owns the cache.
pub(crate) struct Pool<T: Pod> {
    /// Bytes per chunk: whole slots only, so a chunk's allocation is exactly this.
    chunk_bytes: usize,
    per_chunk: u32,
    /// Slot storage; `None` for a chunk given back. Its metadata stays, so generations survive.
    chunks: Vec<Option<Vec<T>>>,
    /// Per chunk, per slot.
    meta: Vec<Vec<Meta>>,
    /// Free slots of mapped chunks.
    free: Vec<Pos>,
    index: HashMap<TileKey, Pos>,
    /// Where the clock resumes, as a flat position (`chunk × per_chunk + index`).
    hand: u64,
}

impl<T: Pod> std::fmt::Debug for Pool<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pool")
            .field("chunk_bytes", &self.chunk_bytes)
            .field("per_chunk", &self.per_chunk)
            .field("bytes", &self.bytes())
            .field("tiles", &self.index.len())
            .field("free", &self.free.len())
            .finish_non_exhaustive()
    }
}

impl<T: Pod> Pool<T> {
    /// An empty pool whose chunks hold as many slots as fit in `chunk_bytes`, at least one.
    pub(crate) fn new(chunk_bytes: usize) -> Self {
        let size = size_of::<T>().max(1);
        let per_chunk = u32::try_from((chunk_bytes / size).max(1)).unwrap_or(u32::MAX);
        Self {
            chunk_bytes: size * per_chunk as usize,
            per_chunk,
            chunks: Vec::new(),
            meta: Vec::new(),
            free: Vec::new(),
            index: HashMap::new(),
            hand: 0,
        }
    }

    /// Bytes of the mapped chunks: what the budget counts for this pool.
    pub(crate) fn bytes(&self) -> usize {
        self.chunks.iter().flatten().count() * self.chunk_bytes
    }

    /// `key`'s slot, marking it referenced.
    pub(crate) fn lookup(&mut self, key: &TileKey) -> Option<SlotId> {
        let pos = *self.index.get(key)?;
        let meta = self.meta_mut(pos)?;
        meta.referenced = true;
        let generation = meta.generation;
        Some(SlotId {
            chunk: pos.0,
            index: pos.1,
            generation,
        })
    }

    /// The slot's contents, or `None` for a stale handle.
    pub(crate) fn get(&self, id: SlotId) -> Option<&T> {
        self.live(id)?;
        self.chunks
            .get(id.chunk as usize)?
            .as_ref()?
            .get(id.index as usize)
    }

    /// The slot's contents, or `None` for a stale handle.
    pub(crate) fn get_mut(&mut self, id: SlotId) -> Option<&mut T> {
        self.live(id)?;
        self.chunks
            .get_mut(id.chunk as usize)?
            .as_mut()?
            .get_mut(id.index as usize)
    }

    /// A zeroed slot for `key`, referenced and clean: a free slot, else one from a new chunk if
    /// the budget allows it, else the clock's victim, which is written back first if it is dirty.
    ///
    /// # Errors
    ///
    /// [`PoolError::Present`] if `key` already has a slot; [`PoolError::Full`] if every slot is
    /// pinned and the budget allows no chunk; [`PoolError::WriteBack`] if the victim's write-back
    /// failed (it keeps its slot).
    pub(crate) fn acquire(
        &mut self,
        key: TileKey,
        budget: &mut Budget,
        write_back: impl FnMut(TileKey, &T) -> Result<(), StoreError>,
    ) -> Result<SlotId, PoolError> {
        if self.index.contains_key(&key) {
            return Err(PoolError::Present);
        }
        let pos = match self.free.pop() {
            Some(pos) => pos,
            None if self.map_chunk(budget) => self.free.pop().ok_or(PoolError::Full)?,
            None => self.victim(write_back)?,
        };
        let generation = {
            let meta = self.meta_mut(pos).ok_or(PoolError::Full)?;
            *meta = Meta {
                key: Some(key),
                generation: meta.generation,
                pins: 0,
                referenced: true,
                dirty: false,
            };
            meta.generation
        };
        if let Some(slot) = self.slot_mut(pos) {
            bytemuck::bytes_of_mut(slot).fill(0);
        }
        self.index.insert(key, pos);
        Ok(SlotId {
            chunk: pos.0,
            index: pos.1,
            generation,
        })
    }

    /// Pins the slot: the clock and [`Pool::evict_all`] leave it alone until it is unpinned as
    /// many times.
    ///
    /// # Errors
    ///
    /// [`PoolError::Stale`].
    pub(crate) fn pin(&mut self, id: SlotId) -> Result<(), PoolError> {
        let meta = self.live_mut(id).ok_or(PoolError::Stale)?;
        meta.pins += 1;
        Ok(())
    }

    /// Undoes one [`Pool::pin`].
    ///
    /// # Errors
    ///
    /// [`PoolError::Stale`], [`PoolError::NotPinned`].
    pub(crate) fn unpin(&mut self, id: SlotId) -> Result<(), PoolError> {
        let meta = self.live_mut(id).ok_or(PoolError::Stale)?;
        meta.pins = meta.pins.checked_sub(1).ok_or(PoolError::NotPinned)?;
        Ok(())
    }

    /// Marks the slot as changed (to be written back before it is reused) or written back.
    ///
    /// # Errors
    ///
    /// [`PoolError::Stale`].
    pub(crate) fn set_dirty(&mut self, id: SlotId, dirty: bool) -> Result<(), PoolError> {
        self.live_mut(id).ok_or(PoolError::Stale)?.dirty = dirty;
        Ok(())
    }

    /// Whether the slot is dirty; `None` for a stale handle.
    pub(crate) fn is_dirty(&self, id: SlotId) -> Option<bool> {
        self.live(id).map(|meta| meta.dirty)
    }

    /// Frees `key`'s slot without writing it back (invalidation); whether it had one.
    ///
    /// # Errors
    ///
    /// [`PoolError::Pinned`] or [`PoolError::Dirty`], leaving the slot as it was.
    pub(crate) fn remove(&mut self, key: &TileKey) -> Result<bool, PoolError> {
        let Some(&pos) = self.index.get(key) else {
            return Ok(false);
        };
        let meta = self.meta_mut(pos).ok_or(PoolError::Stale)?;
        if meta.pins > 0 {
            return Err(PoolError::Pinned);
        }
        if meta.dirty {
            return Err(PoolError::Dirty);
        }
        self.release(pos);
        self.free.push(pos);
        Ok(true)
    }

    /// Frees every unpinned slot, writing dirty ones back first.
    ///
    /// # Errors
    ///
    /// [`PoolError::WriteBack`]: that slot and those not reached yet keep their tiles.
    pub(crate) fn evict_all(
        &mut self,
        mut write_back: impl FnMut(TileKey, &T) -> Result<(), StoreError>,
    ) -> Result<(), PoolError> {
        for chunk in 0..self.chunk_count() {
            self.evict_chunk(chunk, &mut write_back)?;
        }
        Ok(())
    }

    /// Gives chunks back until `budget.used() <= bytes` or nothing more can go: from the highest
    /// mapped chunk down, each chunk without pinned slots is emptied (dirty slots written back
    /// first) and unmapped. Chunks holding pinned slots stay.
    ///
    /// # Errors
    ///
    /// [`PoolError::WriteBack`]: that slot keeps its tile and its chunk stays mapped.
    pub(crate) fn shrink_to(
        &mut self,
        budget: &mut Budget,
        bytes: usize,
        mut write_back: impl FnMut(TileKey, &T) -> Result<(), StoreError>,
    ) -> Result<(), PoolError> {
        for chunk in (0..self.chunk_count()).rev() {
            if budget.used() <= bytes {
                break;
            }
            let pinned = self
                .meta
                .get(chunk as usize)
                .is_some_and(|slots| slots.iter().any(|m| m.pins > 0));
            if pinned || !self.is_mapped(chunk) {
                continue;
            }
            self.evict_chunk(chunk, &mut write_back)?;
            self.free.retain(|&(c, _)| c != chunk);
            if let Some(storage) = self.chunks.get_mut(chunk as usize) {
                *storage = None;
                budget.give(self.chunk_bytes);
            }
        }
        Ok(())
    }

    /// Maps a chunk if the budget allows one, reusing the lowest unmapped chunk's place, and puts
    /// its slots on the free list; whether it did.
    fn map_chunk(&mut self, budget: &mut Budget) -> bool {
        let unmapped = self.chunks.iter().position(Option::is_none);
        let Ok(chunk) = u32::try_from(unmapped.unwrap_or(self.chunks.len())) else {
            return false;
        };
        if !budget.try_take(self.chunk_bytes) {
            return false;
        }
        let storage = Some(bytemuck::zeroed_vec(self.per_chunk as usize));
        if let Some(place) = self.chunks.get_mut(chunk as usize) {
            *place = storage;
        } else {
            self.chunks.push(storage);
            self.meta
                .push(vec![Meta::default(); self.per_chunk as usize]);
        }
        // Reversed, so slots are handed out in order.
        self.free
            .extend((0..self.per_chunk).rev().map(|index| (chunk, index)));
        true
    }

    /// The clock's victim: sweeping from the hand over mapped slots, skipping pinned ones and
    /// clearing `referenced` on the first that have it, the first unpinned unreferenced slot,
    /// written back if dirty and then freed (but not put on the free list).
    ///
    /// # Errors
    ///
    /// [`PoolError::Full`] after two full sweeps without a victim; [`PoolError::WriteBack`].
    fn victim(
        &mut self,
        mut write_back: impl FnMut(TileKey, &T) -> Result<(), StoreError>,
    ) -> Result<Pos, PoolError> {
        let total = u64::from(self.chunk_count()) * u64::from(self.per_chunk);
        for _ in 0..2 * total {
            let flat = self.hand % total;
            self.hand = (flat + 1) % total;
            let pos = self.pos_of(flat);
            if !self.is_mapped(pos.0) {
                continue;
            }
            let Some(meta) = self.meta_mut(pos) else {
                continue;
            };
            if meta.pins > 0 || meta.key.is_none() {
                continue;
            }
            if meta.referenced {
                meta.referenced = false;
                continue;
            }
            self.write_back_slot(pos, &mut write_back)?;
            self.release(pos);
            return Ok(pos);
        }
        Err(PoolError::Full)
    }

    /// Frees every unpinned slot of `chunk`, writing dirty ones back first.
    ///
    /// # Errors
    ///
    /// [`PoolError::WriteBack`].
    fn evict_chunk(
        &mut self,
        chunk: u32,
        write_back: &mut impl FnMut(TileKey, &T) -> Result<(), StoreError>,
    ) -> Result<(), PoolError> {
        for index in 0..self.per_chunk {
            let pos = (chunk, index);
            let evictable = self
                .meta_of(pos)
                .is_some_and(|m| m.key.is_some() && m.pins == 0);
            if evictable {
                self.write_back_slot(pos, write_back)?;
                self.release(pos);
                self.free.push(pos);
            }
        }
        Ok(())
    }

    /// Writes the slot at `pos` back if it is dirty, then marks it clean.
    ///
    /// # Errors
    ///
    /// [`PoolError::WriteBack`], leaving it dirty.
    fn write_back_slot(
        &mut self,
        pos: Pos,
        write_back: &mut impl FnMut(TileKey, &T) -> Result<(), StoreError>,
    ) -> Result<(), PoolError> {
        let Some(Meta {
            key: Some(key),
            dirty: true,
            ..
        }) = self.meta_of(pos).copied()
        else {
            return Ok(());
        };
        if let Some(slot) = self.slot(pos) {
            write_back(key, slot)?;
        }
        if let Some(meta) = self.meta_mut(pos) {
            meta.dirty = false;
        }
        Ok(())
    }

    /// Frees the slot at `pos`: unkeyed, clean, a new generation. The caller puts it on the free
    /// list or reuses it.
    fn release(&mut self, pos: Pos) {
        let Some(meta) = self.meta_mut(pos) else {
            return;
        };
        let key = meta.key.take();
        *meta = Meta {
            generation: meta.generation.wrapping_add(1),
            ..Meta::default()
        };
        if let Some(key) = key {
            self.index.remove(&key);
        }
    }

    /// Chunks ever mapped (mapped now or not).
    fn chunk_count(&self) -> u32 {
        u32::try_from(self.chunks.len()).unwrap_or(u32::MAX)
    }

    /// Whether `chunk` is mapped.
    fn is_mapped(&self, chunk: u32) -> bool {
        self.chunks.get(chunk as usize).is_some_and(Option::is_some)
    }

    /// The position of flat index `flat`.
    fn pos_of(&self, flat: u64) -> Pos {
        let per = u64::from(self.per_chunk);
        (
            u32::try_from(flat / per).unwrap_or(u32::MAX),
            u32::try_from(flat % per).unwrap_or(u32::MAX),
        )
    }

    fn meta_of(&self, (chunk, index): Pos) -> Option<&Meta> {
        self.meta.get(chunk as usize)?.get(index as usize)
    }

    fn meta_mut(&mut self, (chunk, index): Pos) -> Option<&mut Meta> {
        self.meta.get_mut(chunk as usize)?.get_mut(index as usize)
    }

    fn slot(&self, (chunk, index): Pos) -> Option<&T> {
        self.chunks
            .get(chunk as usize)?
            .as_ref()?
            .get(index as usize)
    }

    fn slot_mut(&mut self, (chunk, index): Pos) -> Option<&mut T> {
        self.chunks
            .get_mut(chunk as usize)?
            .as_mut()?
            .get_mut(index as usize)
    }

    /// The live slot's bookkeeping, or `None` if `id` is stale.
    fn live(&self, id: SlotId) -> Option<&Meta> {
        self.meta_of((id.chunk, id.index))
            .filter(|m| m.generation == id.generation && m.key.is_some())
    }

    /// As [`Pool::live`], mutably.
    fn live_mut(&mut self, id: SlotId) -> Option<&mut Meta> {
        self.meta_mut((id.chunk, id.index))
            .filter(|m| m.generation == id.generation && m.key.is_some())
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use geom::{FaceId, Level};

    use super::*;
    use crate::slot::{CorrectionSlot, EvaluatedSlot};

    /// A small slot for the rule tests.
    type Small = [u64; 2];

    fn key(n: u32) -> TileKey {
        TileKey::displacement(FaceId::new(n), Level::new(9).unwrap(), 0, 0).unwrap()
    }

    /// A write-back for pools that must have nothing dirty to write.
    ///
    /// # Errors
    ///
    /// Never: it panics instead.
    fn no_write_back<T>(key: TileKey, _: &T) -> Result<(), StoreError> {
        panic!("unexpected write-back of {key:?}")
    }

    /// A pool of `slots` [`Small`] slots per chunk, and a budget of `chunks` of them.
    fn small(slots: usize, chunks: usize) -> (Pool<Small>, Budget) {
        let chunk = slots * size_of::<Small>();
        (Pool::new(chunk), Budget::new(chunks * chunk))
    }

    /// Acquires `key` and stores `value` in its slot.
    fn put(pool: &mut Pool<Small>, budget: &mut Budget, n: u32, value: u64) -> SlotId {
        let id = pool.acquire(key(n), budget, no_write_back).unwrap();
        *pool.get_mut(id).unwrap() = [value, u64::from(n)];
        id
    }

    #[test]
    fn never_exceeds_the_budget() {
        let chunk = 4 * size_of::<EvaluatedSlot>();
        let mut budget = Budget::new(3 * chunk);
        let mut corrections = Pool::<CorrectionSlot>::new(chunk);
        let mut evaluated = Pool::<EvaluatedSlot>::new(chunk);
        let mut rng = 0x2545_F491_4F6C_DD1D_u64;
        let mut next = |n: u64| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng % n
        };
        // Pinned slots: which pool, and the handle.
        let mut pinned: Vec<(bool, SlotId)> = Vec::new();
        let (mut fulls, mut grown) = (0, 0);
        for step in 0..20_000 {
            let in_corrections = next(2) == 0;
            let k = key(u32::try_from(next(48)).unwrap());
            macro_rules! with_pool {
                (| $pool:ident | $body:expr) => {
                    if in_corrections {
                        let $pool = &mut corrections;
                        $body
                    } else {
                        let $pool = &mut evaluated;
                        $body
                    }
                };
            }
            match next(16) {
                0..=5 => with_pool!(|pool| {
                    if pool.lookup(&k).is_none() {
                        let before = pool.bytes();
                        match pool.acquire(k, &mut budget, |_, _| Ok(())) {
                            Ok(id) => assert!(pool.get(id).is_some(), "fresh handle is live"),
                            Err(PoolError::Full) => fulls += 1,
                            Err(e) => panic!("acquire: {e}"),
                        }
                        grown += usize::from(pool.bytes() > before);
                    }
                }),
                6 | 7 => with_pool!(|pool| {
                    pool.lookup(&k);
                }),
                8 if pinned.len() < 6 => with_pool!(|pool| {
                    if let Some(id) = pool.lookup(&k) {
                        pool.pin(id).unwrap();
                        pinned.push((in_corrections, id));
                    }
                }),
                9 => {
                    if let Some((in_c, id)) = pinned.pop() {
                        if in_c {
                            corrections.unpin(id).unwrap();
                        } else {
                            evaluated.unpin(id).unwrap();
                        }
                    }
                }
                10 | 11 => with_pool!(|pool| {
                    if let Some(id) = pool.lookup(&k) {
                        pool.set_dirty(id, next(2) == 0).unwrap();
                    }
                }),
                12 | 13 => with_pool!(|pool| match pool.remove(&k) {
                    Ok(_) | Err(PoolError::Pinned | PoolError::Dirty) => {}
                    Err(e) => panic!("remove: {e}"),
                }),
                14 if step % 7 == 0 => with_pool!(|pool| pool.evict_all(|_, _| Ok(())).unwrap()),
                15 if step % 5 == 0 => with_pool!(|pool| {
                    let target = usize::try_from(next(3)).unwrap() * chunk;
                    pool.shrink_to(&mut budget, target, |_, _| Ok(())).unwrap();
                }),
                _ => {}
            }
            assert!(
                budget.used() <= budget.limit(),
                "step {step}: {} > {}",
                budget.used(),
                budget.limit()
            );
            assert_eq!(
                budget.used(),
                corrections.bytes() + evaluated.bytes(),
                "step {step}: the budget counts exactly the mapped chunks"
            );
            for &(in_c, id) in &pinned {
                let live = if in_c {
                    corrections.get(id).is_some()
                } else {
                    evaluated.get(id).is_some()
                };
                assert!(live, "step {step}: a pinned slot stays");
            }
        }
        assert!(fulls > 0 && grown > 10, "the workload hit the budget");
    }

    #[test]
    fn everything_pinned_is_full() {
        let (mut pool, mut budget) = small(4, 1);
        let ids: Vec<SlotId> = (0..4).map(|n| put(&mut pool, &mut budget, n, 7)).collect();
        for &id in &ids {
            pool.pin(id).unwrap();
        }
        assert!(
            matches!(
                pool.acquire(key(99), &mut budget, no_write_back),
                Err(PoolError::Full)
            ),
            "nothing to evict"
        );
        for (n, &id) in (0..).zip(&ids) {
            assert_eq!(pool.lookup(&key(n)), Some(id), "tile {n} kept its slot");
            assert_eq!(pool.get(id), Some(&[7, u64::from(n)]), "tile {n} unchanged");
        }
        assert!(
            pool.evict_all(no_write_back).is_ok(),
            "evict_all skips them"
        );
        assert_eq!(pool.lookup(&key(0)), Some(ids[0]), "still there");
        pool.unpin(ids[0]).unwrap();
        assert!(
            matches!(pool.unpin(ids[0]), Err(PoolError::NotPinned)),
            "pins balance"
        );
        let id = pool.acquire(key(99), &mut budget, no_write_back).unwrap();
        assert_eq!(
            id.index, ids[0].index,
            "the one unpinned slot is the victim"
        );
        assert_eq!(pool.get(id), Some(&[0, 0]), "and comes back zeroed");
    }

    #[test]
    fn dirty_victims_are_written_back_once() {
        let (mut pool, mut budget) = small(4, 1);
        for n in 0..4 {
            let id = put(&mut pool, &mut budget, n, 100 + u64::from(n));
            pool.set_dirty(id, n % 2 == 0).unwrap();
        }
        // A failing write-back leaves the victim dirty and in place.
        let failed = pool.acquire(key(10), &mut budget, |_, _: &Small| {
            Err(StoreError::ReadOnly)
        });
        assert!(
            matches!(failed, Err(PoolError::WriteBack(_))),
            "error returned"
        );
        let victim = pool.lookup(&key(0)).unwrap();
        assert_eq!(pool.is_dirty(victim), Some(true), "still dirty");
        assert_eq!(pool.get(victim), Some(&[100, 0]), "still there");

        let mut written = Vec::new();
        for n in 10..20 {
            pool.acquire(key(n), &mut budget, |k, v: &Small| {
                written.push((k, *v));
                Ok(())
            })
            .unwrap();
        }
        written.sort_unstable_by_key(|(k, _)| *k);
        assert_eq!(
            written,
            vec![(key(0), [100, 0]), (key(2), [102, 2])],
            "each dirty victim once, clean ones never"
        );
        written.clear();
        let id = pool.lookup(&key(19)).unwrap();
        pool.set_dirty(id, true).unwrap();
        pool.evict_all(|k, v: &Small| {
            written.push((k, *v));
            Ok(())
        })
        .unwrap();
        assert_eq!(
            written,
            vec![(key(19), [0, 0])],
            "evict_all writes back too"
        );
        assert_eq!(pool.lookup(&key(19)), None, "and frees");
    }

    #[test]
    fn stale_handles_are_rejected() {
        let (mut pool, mut budget) = small(2, 1);
        let a = put(&mut pool, &mut budget, 0, 1);
        put(&mut pool, &mut budget, 1, 2);
        // Both are referenced: the clock clears them, then takes slot 0 for tile 2.
        let c = put(&mut pool, &mut budget, 2, 3);
        assert_eq!((c.chunk, c.index), (a.chunk, a.index), "slot reused");
        assert_eq!(pool.get(a), None, "evicted handle");
        assert!(matches!(pool.pin(a), Err(PoolError::Stale)), "pin");
        assert_eq!(pool.is_dirty(a), None, "is_dirty");
        assert_eq!(pool.get(c), Some(&[3, 2]), "the new tile");

        assert!(pool.remove(&key(2)).unwrap(), "removed");
        assert_eq!(pool.get(c), None, "removed handle");
        assert!(!pool.remove(&key(2)).unwrap(), "already gone");
        let d = put(&mut pool, &mut budget, 3, 4);
        assert_eq!((d.chunk, d.index), (c.chunk, c.index), "slot reused");
        assert_eq!(pool.get(c), None, "still stale after reuse");
        assert!(
            matches!(
                pool.acquire(key(3), &mut budget, no_write_back),
                Err(PoolError::Present)
            ),
            "one slot per tile"
        );
    }

    #[test]
    fn shrink_returns_chunks() {
        let (mut pool, mut budget) = small(2, 3);
        let ids: Vec<SlotId> = (0..6).map(|n| put(&mut pool, &mut budget, n, 9)).collect();
        assert_eq!(pool.bytes(), budget.limit(), "filled");
        pool.set_dirty(ids[5], true).unwrap();
        pool.pin(ids[3]).unwrap();
        let mut written = Vec::new();
        pool.shrink_to(&mut budget, 0, |k, _: &Small| {
            written.push(k);
            Ok(())
        })
        .unwrap();
        assert_eq!(
            written,
            vec![key(5)],
            "dirty slots written back before going"
        );
        assert_eq!(
            pool.bytes(),
            2 * size_of::<Small>(),
            "the pinned slot's chunk stays"
        );
        assert_eq!(budget.used(), pool.bytes(), "budget given back");
        assert_eq!(pool.get(ids[3]), Some(&[9, 3]), "pinned tile kept");
        pool.unpin(ids[3]).unwrap();
        pool.evict_all(no_write_back).unwrap();
        pool.shrink_to(&mut budget, 0, no_write_back).unwrap();
        assert_eq!((pool.bytes(), budget.used()), (0, 0), "all gone");

        // Chunks come back, and handles from before stay stale.
        let id = put(&mut pool, &mut budget, 7, 1);
        assert_eq!(
            (id.chunk, id.index),
            (ids[0].chunk, ids[0].index),
            "same place"
        );
        assert_eq!(pool.get(ids[0]), None, "old handle stale");
        assert_eq!(pool.get(id), Some(&[1, 7]), "fresh slot");
    }
}
