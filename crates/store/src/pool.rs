//! A budgeted pool of fixed-size plain-data slots with clock eviction.
//!
//! Slots are carved from chunks, each one zeroed allocation (lazily backed by the OS), mapped
//! only while a shared [`Budget`] allows another whole chunk, so the pools' memory never exceeds
//! it. A slot holds one tile, named by its [`TileKey`]. When every mapped slot is in use, the
//! clock picks a victim among the unpinned ones (or the clean ones, for
//! [`Pool::acquire_clean`]), skipping those referenced since its last pass, and writes it back
//! first if it is dirty. Every chunk is charged the configured chunk size, whatever its slot type,
//! and a pool asked to shrink gives back its cheapest chunks first: the fewest dirty slots, then
//! the fewest in use.
//!
//! A pool made [`Pool::with_short_chunks`] may also map a *short* chunk when the budget has room
//! for less than a whole one: as many whole slots as the room holds, charged exactly their bytes,
//! so a budget that isn't a whole number of chunks (the tile store's, once its kept predictions
//! are set aside) loses at most part of a slot. Only a short chunk of at least half a chunk is
//! mapped, for the reason below.
//!
//! Chunks should be large: glibc serves an allocation from `mmap` (and so returns it to the OS
//! when it is dropped) only above its mmap threshold, which adapts upwards to as much as 32 MiB.
//! A smaller chunk may come from the heap and stay with the process after the pool gives it
//! back to the budget. The default 64 MB chunk is above it; tests use small chunks knowingly.

use std::cmp::Reverse;
use std::convert::Infallible;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

use bytemuck::Pod;
use geom::TileKey;
use thiserror::Error;

use crate::hash::TileMap;

/// Bytes the pools may use, shared by all of them.
///
/// A pool must always be used with the same `Budget`: it takes bytes from the budget it is given
/// when it maps a chunk and gives them back to the budget it is given when it unmaps one, and
/// can't tell two budgets apart.
#[derive(Debug)]
pub(crate) struct Budget {
    /// Bytes the pools may hold together.
    limit: usize,
    /// Bytes the pools' mapped chunks hold now: never more than `limit`.
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

    /// Bytes still to be had under the limit.
    pub(crate) const fn room(&self) -> usize {
        self.limit.saturating_sub(self.used)
    }

    /// Whether another `bytes` fit under the limit.
    pub(crate) fn fits(&self, bytes: usize) -> bool {
        self.used
            .checked_add(bytes)
            .is_some_and(|n| n <= self.limit)
    }

    /// Takes `bytes` if they fit under the limit; whether they did.
    fn try_take(&mut self, bytes: usize) -> bool {
        let fits = self.fits(bytes);
        if fits {
            self.used += bytes;
        }
        fits
    }

    /// Gives back `bytes` taken earlier.
    ///
    /// # Panics
    ///
    /// In debug builds, if more is given back than is in use: a pool was used with another
    /// budget.
    fn give(&mut self, bytes: usize) {
        debug_assert!(
            bytes <= self.used,
            "giving back {bytes} bytes of {} used: a pool changed budgets",
            self.used
        );
        self.used = self.used.saturating_sub(bytes);
    }
}

/// A handle to a slot of a [`Pool<T>`]. Valid until the slot is evicted or removed: the pool
/// checks the generation on every use, so a stale handle finds nothing rather than another tile.
///
/// The slot type is part of the handle's type, so a handle from one pool can't index another
/// pool of a different slot type. (`fn() -> T` keeps the handle `Send`, `Sync` and `Copy`
/// whatever `T` is.)
pub(crate) struct SlotId<T> {
    /// The chunk the slot is in.
    chunk: u32,
    /// The slot's index in its chunk.
    index: u32,
    /// The slot's generation when the handle was made.
    generation: u32,
    /// Ties the handle to its pool's slot type.
    pool: PhantomData<fn() -> T>,
}

impl<T> SlotId<T> {
    /// The handle of the slot at `pos` in its generation `generation`.
    const fn new((chunk, index): Pos, generation: u32) -> Self {
        Self {
            chunk,
            index,
            generation,
            pool: PhantomData,
        }
    }

    /// The fields that identify the slot, for the trait impls.
    const fn parts(&self) -> (u32, u32, u32) {
        (self.chunk, self.index, self.generation)
    }
}

/// Implemented by hand: deriving would require `T: Clone`.
impl<T> Clone for SlotId<T> {
    fn clone(&self) -> Self {
        *self
    }
}

/// Implemented by hand: deriving would require `T: Copy`.
impl<T> Copy for SlotId<T> {}

/// Implemented by hand: deriving would require `T: PartialEq`.
impl<T> PartialEq for SlotId<T> {
    fn eq(&self, other: &Self) -> bool {
        self.parts() == other.parts()
    }
}

/// Implemented by hand: deriving would require `T: Eq`.
impl<T> Eq for SlotId<T> {}

/// Implemented by hand: deriving would require `T: Hash`.
impl<T> Hash for SlotId<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.parts().hash(state);
    }
}

/// Implemented by hand: deriving would require `T: Debug`.
impl<T> std::fmt::Debug for SlotId<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SlotId")
            .field("chunk", &self.chunk)
            .field("index", &self.index)
            .field("generation", &self.generation)
            .finish()
    }
}

/// Whether a slot's contents are current, as [`Pool::lookup`] reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Freshness {
    /// The contents are current.
    Fresh,
    /// The tile was invalidated while the slot was pinned or dirty: the slot keeps its key and
    /// generation (so handles stay valid) but its contents must be recomputed in place, then
    /// marked with [`Pool::mark_fresh`].
    Stale,
}

/// What [`Pool::invalidate`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Invalidated {
    /// The key had no slot.
    Absent,
    /// The slot was unpinned and clean, and is free now.
    Freed,
    /// The slot is pinned or dirty, so it stays, marked [`Freshness::Stale`].
    Marked,
}

/// Why a pool operation failed. `W` is the error of the write-back the caller supplied, for
/// the operations that may write a dirty slot back; the others can't fail that way.
#[derive(Debug, Error)]
pub(crate) enum PoolError<W = Infallible> {
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
    WriteBack(#[source] W),
}

/// One slot's bookkeeping.
#[derive(Debug, Clone, Copy, Default)]
struct SlotMeta {
    /// The tile it holds; `None` while free.
    key: Option<TileKey>,
    /// Bumped whenever the slot is freed, so older handles go stale.
    generation: u32,
    /// Pins not yet undone.
    pins: u32,
    /// Used since the clock last passed.
    referenced: bool,
    /// Changed since it was last written back.
    dirty: bool,
    /// Invalidated while it couldn't be freed: the contents are to be recomputed in place.
    stale: bool,
}

/// Which slots the clock may take as its victim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Victims {
    /// Any unpinned slot; a dirty one is written back first.
    Any,
    /// Unpinned clean slots only, so nothing is written back.
    Clean,
}

/// A slot position: chunk and index in it.
type Pos = (u32, u32);

/// Plain data a [`Pool`] keeps in its slots, and how a slot is cleared when it is handed out
/// for another tile.
///
/// A recycled slot still holds its last tile's bytes. Zeroing all of them on every acquire is
/// the simple rule, but it is most of the cost of paging tiles in: an evaluated slot is about
/// 100 KB whatever its tile's level, a coarse tile uses a few hundred bytes of it, and on a
/// freshly mapped chunk the zeroing faults in every page of every slot. So each slot type says
/// what clearing means for it: enough that nothing of the previous tile can be read as the new
/// one's.
pub(crate) trait SlotData: Pod {
    /// Clears a slot about to be handed out for a new tile, so that it reads as holding no
    /// tile (or as zero, for a type that can't say so).
    fn clear(&mut self);
}

/// Pools of plain arrays (the pool's own tests and proofs) are zeroed.
#[cfg(any(test, kani))]
impl<T: Pod, const N: usize> SlotData for [T; N] {
    fn clear(&mut self) {
        bytemuck::bytes_of_mut(self).fill(0);
    }
}

/// A budgeted pool of `T` slots. Single-threaded: it belongs to the thread that owns the cache.
pub(crate) struct Pool<T: SlotData> {
    /// Bytes the budget is charged per chunk: the configured chunk size (see [`Pool::layout`]).
    /// A chunk's allocation is its whole slots, which may be a little less.
    chunk_bytes: usize,
    /// Slots per chunk: as many as fit in `chunk_bytes`, at least one.
    per_chunk: u32,
    /// Slot storage; `None` for a chunk given back. Its metadata stays, so generations survive.
    chunks: Vec<Option<Vec<T>>>,
    /// What the budget was charged for each chunk when it was mapped, given back exactly when
    /// it goes; zero for a chunk not mapped. Recorded rather than inferred from the storage, as
    /// a short chunk may hold as many slots as a whole one (whose charge includes the slack
    /// after its last slot).
    charges: Vec<usize>,
    /// Per chunk, per slot.
    meta: Vec<Vec<SlotMeta>>,
    /// Free slots of mapped chunks.
    free: Vec<Pos>,
    /// Where each tile's slot is.
    index: TileMap<Pos>,
    /// Slots with at least one pin.
    pinned: usize,
    /// Where the clock resumes, as a flat position (`chunk × per_chunk + index`).
    hand: u64,
    /// Whether the pool may map a short chunk, of fewer slots, with room for less than a whole
    /// one (see the module notes and [`Pool::with_short_chunks`]).
    short_chunks: bool,
    /// Keys the pool freed on its own (the clock's victims, [`Pool::evict_all`],
    /// [`Pool::shrink_to`]) since [`Pool::drain_evicted`] last ran; `None` unless the pool was
    /// made with [`Pool::with_eviction_log`].
    evicted: Option<Vec<TileKey>>,
}

impl<T: SlotData> std::fmt::Debug for Pool<T> {
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

impl<T: SlotData> Pool<T> {
    /// How a pool asked for chunks of `chunk_bytes` lays them out: slots per chunk (as many as
    /// fit, at least one) and the bytes the budget is charged per chunk.
    ///
    /// The charge is `chunk_bytes` itself (or one slot, if that is larger), whatever the slots
    /// leave unused at the chunk's end: so pools of different slot types are charged alike, and
    /// a chunk one pool gives back is exactly the room another needs for a chunk.
    pub(crate) fn layout(chunk_bytes: usize) -> (u32, usize) {
        let size = size_of::<T>().max(1);
        let charged = chunk_bytes.max(size);
        let per_chunk = u32::try_from(charged / size).unwrap_or(u32::MAX);
        (per_chunk, charged)
    }

    /// An empty pool whose chunks hold as many slots as fit in `chunk_bytes`, at least one, and
    /// are charged as [`Pool::layout`] says.
    pub(crate) fn new(chunk_bytes: usize) -> Self {
        let (per_chunk, chunk_bytes) = Self::layout(chunk_bytes);
        Self {
            chunk_bytes,
            per_chunk,
            chunks: Vec::new(),
            charges: Vec::new(),
            meta: Vec::new(),
            free: Vec::new(),
            index: TileMap::default(),
            pinned: 0,
            hand: 0,
            short_chunks: false,
            evicted: None,
        }
    }

    /// This pool, which may map a short chunk when the budget has room for at least half a
    /// chunk but not a whole one: as many slots as the room holds, charged exactly their bytes
    /// (see the module notes). Its other chunks are whole, so the room it gives back is still a
    /// whole chunk for another pool, except for the short one.
    pub(crate) fn with_short_chunks(self) -> Self {
        Self {
            short_chunks: true,
            ..self
        }
    }

    /// As [`Pool::new`], and the pool logs the keys it evicts for [`Pool::drain_evicted`], so an
    /// owner keeping bookkeeping per resident tile (the tile cache's dependency graph) can follow.
    /// The owner must drain the log, or it grows with every eviction.
    pub(crate) fn with_eviction_log(chunk_bytes: usize) -> Self {
        Self {
            evicted: Some(Vec::new()),
            ..Self::new(chunk_bytes)
        }
    }

    /// The keys evicted since the last call, oldest first: tiles the pool freed on its own, not
    /// those the caller freed with [`Pool::invalidate`] or [`Pool::remove`]. Always empty for a
    /// pool made without [`Pool::with_eviction_log`].
    pub(crate) fn drain_evicted(&mut self) -> Vec<TileKey> {
        self.evicted
            .as_mut()
            .map(std::mem::take)
            .unwrap_or_default()
    }

    /// Bytes the budget is charged per chunk: what mapping one more chunk takes from it.
    pub(crate) const fn chunk_bytes(&self) -> usize {
        self.chunk_bytes
    }

    /// Slots per chunk.
    pub(crate) const fn per_chunk(&self) -> u32 {
        self.per_chunk
    }

    /// Slots with at least one pin.
    pub(crate) const fn pinned_slots(&self) -> usize {
        self.pinned
    }

    /// Whether a mapped chunk has a free slot: [`Pool::acquire`] then needs no new chunk and
    /// evicts nothing.
    #[cfg(test)]
    pub(crate) fn has_free_slot(&self) -> bool {
        !self.free.is_empty()
    }

    /// Whether a mapped chunk holds no pinned slot, so that [`Pool::shrink_to`] can give it back.
    pub(crate) fn has_unpinned_chunk(&self) -> bool {
        (0..self.chunk_count()).any(|chunk| {
            self.is_mapped(chunk)
                && self
                    .meta
                    .get(chunk as usize)
                    .is_some_and(|slots| slots.iter().all(|m| m.pins == 0))
        })
    }

    /// Whether `key` has a pinned slot.
    pub(crate) fn is_pinned(&self, key: &TileKey) -> bool {
        self.index
            .get(key)
            .and_then(|&pos| self.meta_of(pos))
            .is_some_and(|m| m.pins > 0)
    }

    /// Bytes of the mapped chunks: what the budget counts for this pool.
    pub(crate) fn bytes(&self) -> usize {
        (0..self.chunk_count())
            .map(|chunk| self.charge(chunk))
            .sum()
    }

    /// What the budget was charged for `chunk` when it was mapped: [`Pool::chunk_bytes`] for a
    /// whole chunk, its slots' bytes for a short one, nothing if it isn't mapped.
    fn charge(&self, chunk: u32) -> usize {
        self.charges.get(chunk as usize).copied().unwrap_or(0)
    }

    /// `key`'s slot and whether its contents are current, marking it referenced.
    pub(crate) fn lookup(&mut self, key: &TileKey) -> Option<(SlotId<T>, Freshness)> {
        let pos = *self.index.get(key)?;
        let meta = self.meta_mut(pos)?;
        meta.referenced = true;
        let freshness = if meta.stale {
            Freshness::Stale
        } else {
            Freshness::Fresh
        };
        Some((SlotId::new(pos, meta.generation), freshness))
    }

    /// Whether `key`'s slot is current, without marking it referenced; `None` if it has none.
    pub(crate) fn freshness(&self, key: &TileKey) -> Option<Freshness> {
        let meta = self.meta_of(*self.index.get(key)?)?;
        Some(if meta.stale {
            Freshness::Stale
        } else {
            Freshness::Fresh
        })
    }

    /// `key`'s slot, without marking it referenced: for bookkeeping passes (saving) that
    /// shouldn't count as use.
    pub(crate) fn find(&self, key: &TileKey) -> Option<SlotId<T>> {
        let pos = *self.index.get(key)?;
        Some(SlotId::new(pos, self.meta_of(pos)?.generation))
    }

    /// The slot's contents, or `None` for a stale handle.
    pub(crate) fn get(&self, id: SlotId<T>) -> Option<&T> {
        self.live(id)?;
        self.chunks
            .get(id.chunk as usize)?
            .as_ref()?
            .get(id.index as usize)
    }

    /// The contents of every slot of `ids`, in the same order, all borrowed mutably at once: for
    /// filling several slots on several threads. `None` if a handle is stale or two name the
    /// same slot.
    ///
    /// The slots are found in one walk over the chunks, in slot order, splitting each chunk's
    /// slice at the slots wanted, so the borrows are disjoint by construction; a slot named twice
    /// is behind the split by its second turn, and refused.
    pub(crate) fn get_many_mut(&mut self, ids: &[SlotId<T>]) -> Option<Vec<&mut T>> {
        if ids.iter().any(|&id| self.live(id).is_none()) {
            return None;
        }
        let mut order: Vec<(Pos, usize)> = (0..)
            .zip(ids)
            .map(|(k, id)| ((id.chunk, id.index), k))
            .collect();
        order.sort_unstable();
        let mut found: Vec<Option<&mut T>> = Vec::with_capacity(ids.len());
        found.resize_with(ids.len(), || None);
        let mut wanted = order.into_iter().peekable();
        for (chunk, storage) in (0_u32..).zip(&mut self.chunks) {
            let mut rest: &mut [T] = storage.as_deref_mut().unwrap_or_default();
            // The index in the chunk of `rest`'s first slot.
            let mut start = 0;
            while let Some(((_, index), k)) = wanted.next_if(|&((c, _), _)| c == chunk) {
                let at = usize::try_from(index.checked_sub(start)?).ok()?;
                let (slot, tail) = std::mem::take(&mut rest).get_mut(at..)?.split_first_mut()?;
                *found.get_mut(k)? = Some(slot);
                rest = tail;
                start = index + 1;
            }
        }
        found.into_iter().collect()
    }

    /// The slot's contents, or `None` for a stale handle.
    pub(crate) fn get_mut(&mut self, id: SlotId<T>) -> Option<&mut T> {
        self.live(id)?;
        self.chunks
            .get_mut(id.chunk as usize)?
            .as_mut()?
            .get_mut(id.index as usize)
    }

    /// A cleared slot for `key` (see [`SlotData::clear`]), referenced and clean: a free slot, else
    /// one from a new chunk if the budget allows it, else the clock's victim, which is written
    /// back first if it is dirty.
    ///
    /// # Errors
    ///
    /// [`PoolError::Present`] if `key` already has a slot; [`PoolError::Full`] if every slot is
    /// pinned and the budget allows no chunk; [`PoolError::WriteBack`] if the victim's write-back
    /// failed (it keeps its slot).
    pub(crate) fn acquire<W>(
        &mut self,
        key: TileKey,
        budget: &mut Budget,
        write_back: impl FnMut(TileKey, &T) -> Result<(), W>,
    ) -> Result<SlotId<T>, PoolError<W>> {
        self.acquire_with(key, budget, Victims::Any, write_back)
    }

    /// As [`Pool::acquire`], but the clock takes only a clean victim: nothing is written back.
    ///
    /// # Errors
    ///
    /// [`PoolError::Present`] if `key` already has a slot; [`PoolError::Full`] if every unpinned
    /// slot is dirty and the budget allows no chunk.
    pub(crate) fn acquire_clean(
        &mut self,
        key: TileKey,
        budget: &mut Budget,
    ) -> Result<SlotId<T>, PoolError> {
        self.acquire_with(key, budget, Victims::Clean, |_, _| Ok(()))
    }

    /// [`Pool::acquire`] and [`Pool::acquire_clean`]: the clock takes its victim among `victims`.
    ///
    /// # Errors
    ///
    /// As [`Pool::acquire`].
    fn acquire_with<W>(
        &mut self,
        key: TileKey,
        budget: &mut Budget,
        victims: Victims,
        write_back: impl FnMut(TileKey, &T) -> Result<(), W>,
    ) -> Result<SlotId<T>, PoolError<W>> {
        if self.index.contains_key(&key) {
            return Err(PoolError::Present);
        }
        let pos = match self.free.pop() {
            Some(pos) => pos,
            None if self.map_chunk(budget) => self.free.pop().ok_or(PoolError::Full)?,
            None => self.victim(victims, write_back)?,
        };
        let generation = {
            let meta = self.meta_mut(pos).ok_or(PoolError::Full)?;
            *meta = SlotMeta {
                key: Some(key),
                generation: meta.generation,
                pins: 0,
                referenced: true,
                dirty: false,
                stale: false,
            };
            meta.generation
        };
        if let Some(slot) = self.slot_mut(pos) {
            slot.clear();
        }
        self.index.insert(key, pos);
        Ok(SlotId::new(pos, generation))
    }

    /// Pins the slot: the clock and [`Pool::evict_all`] leave it alone until it is unpinned as
    /// many times.
    ///
    /// # Errors
    ///
    /// [`PoolError::Stale`].
    pub(crate) fn pin(&mut self, id: SlotId<T>) -> Result<(), PoolError> {
        let meta = self.live_mut(id).ok_or(PoolError::Stale)?;
        meta.pins += 1;
        if meta.pins == 1 {
            self.pinned += 1;
        }
        Ok(())
    }

    /// Points the clock's hand at slot `id`, so the next sweep looks at it first: for tests
    /// that need a particular victim.
    #[cfg(test)]
    pub(crate) fn aim_hand_at(&mut self, id: SlotId<T>) {
        self.hand = u64::from(id.chunk) * u64::from(self.per_chunk) + u64::from(id.index);
    }

    /// Undoes one [`Pool::pin`]. A [`Freshness::Stale`] slot loses its `referenced` mark, so it
    /// is the clock's first victim once nothing pins it.
    ///
    /// # Errors
    ///
    /// [`PoolError::Stale`], [`PoolError::NotPinned`].
    pub(crate) fn unpin(&mut self, id: SlotId<T>) -> Result<(), PoolError> {
        let meta = self.live_mut(id).ok_or(PoolError::Stale)?;
        meta.pins = meta.pins.checked_sub(1).ok_or(PoolError::NotPinned)?;
        if meta.stale {
            meta.referenced = false;
        }
        if meta.pins == 0 {
            self.pinned -= 1;
        }
        Ok(())
    }

    /// Marks the slot's contents current again, after recomputing a [`Freshness::Stale`] slot in
    /// place.
    ///
    /// # Errors
    ///
    /// [`PoolError::Stale`] for a stale handle (the slot was freed since).
    pub(crate) fn mark_fresh(&mut self, id: SlotId<T>) -> Result<(), PoolError> {
        self.live_mut(id).ok_or(PoolError::Stale)?.stale = false;
        Ok(())
    }

    /// Invalidates `key`'s tile: its contents no longer follow from its inputs.
    ///
    /// An unpinned, clean slot is freed (its handles go stale). A pinned slot can't be, since a
    /// reader holds it, and a dirty one isn't silently, since its changes would be lost: either
    /// stays, keeping its key and generation, marked [`Freshness::Stale`] for [`Pool::lookup`] to
    /// report, so the caller recomputes it in place (and writes back or discards a dirty one's
    /// changes).
    pub(crate) fn invalidate(&mut self, key: &TileKey) -> Invalidated {
        let Some(&pos) = self.index.get(key) else {
            return Invalidated::Absent;
        };
        let Some(meta) = self.meta_mut(pos) else {
            return Invalidated::Absent;
        };
        if meta.pins > 0 || meta.dirty {
            meta.stale = true;
            return Invalidated::Marked;
        }
        self.release(pos);
        self.free.push(pos);
        Invalidated::Freed
    }

    /// Marks the slot as changed (to be written back before it is reused) or written back.
    ///
    /// # Errors
    ///
    /// [`PoolError::Stale`].
    pub(crate) fn set_dirty(&mut self, id: SlotId<T>, dirty: bool) -> Result<(), PoolError> {
        self.live_mut(id).ok_or(PoolError::Stale)?.dirty = dirty;
        Ok(())
    }

    /// Whether the slot is dirty; `None` for a stale handle.
    pub(crate) fn is_dirty(&self, id: SlotId<T>) -> Option<bool> {
        self.live(id).map(|meta| meta.dirty)
    }

    /// Frees `key`'s slot without writing it back; whether it had one.
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
    pub(crate) fn evict_all<W>(
        &mut self,
        mut write_back: impl FnMut(TileKey, &T) -> Result<(), W>,
    ) -> Result<(), PoolError<W>> {
        for chunk in 0..self.chunk_count() {
            self.evict_chunk(chunk, &mut write_back)?;
        }
        Ok(())
    }

    /// Gives chunks back until `budget.used() <= bytes` or nothing more can go, the cheapest to
    /// lose first (see [`Pool::give_back_cost`]): each chunk is emptied (dirty slots written back
    /// first) and unmapped. Chunks holding pinned slots stay.
    ///
    /// # Errors
    ///
    /// [`PoolError::WriteBack`]: that slot keeps its tile and its chunk stays mapped.
    pub(crate) fn shrink_to<W>(
        &mut self,
        budget: &mut Budget,
        bytes: usize,
        mut write_back: impl FnMut(TileKey, &T) -> Result<(), W>,
    ) -> Result<(), PoolError<W>> {
        while budget.used() > bytes {
            // Ties go to the highest chunk, so the pool keeps its low chunks.
            let cheapest = (0..self.chunk_count())
                .filter_map(|chunk| Some((self.give_back_cost(chunk)?, Reverse(chunk))))
                .min();
            let Some((_, Reverse(chunk))) = cheapest else {
                break;
            };
            self.evict_chunk(chunk, &mut write_back)?;
            self.free.retain(|&(c, _)| c != chunk);
            if let (Some(storage), Some(charge)) = (
                self.chunks.get_mut(chunk as usize),
                self.charges.get_mut(chunk as usize),
            ) {
                *storage = None;
                budget.give(std::mem::take(charge));
            }
        }
        Ok(())
    }

    /// What giving `chunk` back would cost, to compare chunks by: first its dirty slots (each a
    /// write now, and a read later if the tile is wanted again), then its slots in use (holding
    /// a current tile, or referenced since the clock last passed). `None` if the chunk isn't
    /// mapped or holds a pinned slot, so it can't go.
    fn give_back_cost(&self, chunk: u32) -> Option<(usize, usize)> {
        if !self.is_mapped(chunk) {
            return None;
        }
        let slots = self.meta.get(chunk as usize)?;
        if slots.iter().any(|m| m.pins > 0) {
            return None;
        }
        let dirty = slots.iter().filter(|m| m.dirty).count();
        let in_use = slots
            .iter()
            .filter(|m| m.key.is_some() && (m.referenced || !m.stale))
            .count();
        Some((dirty, in_use))
    }

    /// Maps a chunk if the budget allows one, reusing the lowest unmapped chunk's place, and puts
    /// its slots on the free list; whether it did. A pool [`Pool::with_short_chunks`] maps a
    /// short chunk where the budget has room for at least half a chunk but not a whole one.
    fn map_chunk(&mut self, budget: &mut Budget) -> bool {
        let unmapped = self.chunks.iter().position(Option::is_none);
        let Ok(chunk) = u32::try_from(unmapped.unwrap_or(self.chunks.len())) else {
            return false;
        };
        let Some((slots, charge)) = self.slots_to_map(budget) else {
            return false;
        };
        let storage = Some(bytemuck::zeroed_vec(slots as usize));
        if let (Some(place), Some(charged)) = (
            self.chunks.get_mut(chunk as usize),
            self.charges.get_mut(chunk as usize),
        ) {
            *place = storage;
            *charged = charge;
        } else {
            self.chunks.push(storage);
            self.charges.push(charge);
            self.meta
                .push(vec![SlotMeta::default(); self.per_chunk as usize]);
        }
        // Reversed, so slots are handed out in order.
        self.free
            .extend((0..slots).rev().map(|index| (chunk, index)));
        true
    }

    /// How many slots the next chunk holds and what it is charged, the charge taken from
    /// `budget`: a whole chunk if the budget has room for one, else, for a pool
    /// [`Pool::with_short_chunks`], the slots [`Pool::short_slots`] finds in the room, charged
    /// exactly their bytes; `None` if nothing is mapped.
    fn slots_to_map(&self, budget: &mut Budget) -> Option<(u32, usize)> {
        if budget.try_take(self.chunk_bytes) {
            return Some((self.per_chunk, self.chunk_bytes));
        }
        let slots = self.short_slots(budget.room());
        let bytes = slots as usize * size_of::<T>();
        (slots > 0 && budget.try_take(bytes)).then_some((slots, bytes))
    }

    /// The slots of a short chunk in `room` bytes, less than a whole chunk: as many whole slots
    /// as fit (at most a whole chunk's), if that is at least half a chunk and the pool maps
    /// short chunks ([`Pool::with_short_chunks`]); else zero.
    pub(crate) fn short_slots(&self, room: usize) -> u32 {
        let size = size_of::<T>().max(1);
        let slots = (room / size).min(self.per_chunk as usize);
        if !self.short_chunks || slots * size < self.chunk_bytes / 2 {
            return 0;
        }
        u32::try_from(slots).unwrap_or(0)
    }

    /// The clock's victim: sweeping from the hand over mapped slots, skipping pinned ones (and
    /// dirty ones, for [`Victims::Clean`]) and clearing `referenced` on the first that have it,
    /// the first such slot unreferenced, written back if dirty and then freed (but not put on
    /// the free list).
    ///
    /// # Errors
    ///
    /// [`PoolError::Full`] after two full sweeps without a victim; [`PoolError::WriteBack`].
    fn victim<W>(
        &mut self,
        victims: Victims,
        mut write_back: impl FnMut(TileKey, &T) -> Result<(), W>,
    ) -> Result<Pos, PoolError<W>> {
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
            if meta.pins > 0 || meta.key.is_none() || (victims == Victims::Clean && meta.dirty) {
                continue;
            }
            if meta.referenced {
                meta.referenced = false;
                continue;
            }
            self.write_back_slot(pos, &mut write_back)?;
            self.log_eviction(pos);
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
    fn evict_chunk<W>(
        &mut self,
        chunk: u32,
        write_back: &mut impl FnMut(TileKey, &T) -> Result<(), W>,
    ) -> Result<(), PoolError<W>> {
        for index in 0..self.per_chunk {
            let pos = (chunk, index);
            let evictable = self
                .meta_of(pos)
                .is_some_and(|m| m.key.is_some() && m.pins == 0);
            if evictable {
                self.write_back_slot(pos, write_back)?;
                self.log_eviction(pos);
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
    fn write_back_slot<W>(
        &mut self,
        pos: Pos,
        write_back: &mut impl FnMut(TileKey, &T) -> Result<(), W>,
    ) -> Result<(), PoolError<W>> {
        let Some(SlotMeta {
            key: Some(key),
            dirty: true,
            ..
        }) = self.meta_of(pos).copied()
        else {
            return Ok(());
        };
        if let Some(slot) = self.slot(pos) {
            write_back(key, slot).map_err(PoolError::WriteBack)?;
        }
        if let Some(meta) = self.meta_mut(pos) {
            meta.dirty = false;
        }
        Ok(())
    }

    /// Adds the key at `pos` to the eviction log, if the pool keeps one.
    fn log_eviction(&mut self, pos: Pos) {
        let key = self.meta_of(pos).and_then(|m| m.key);
        if let (Some(log), Some(key)) = (self.evicted.as_mut(), key) {
            log.push(key);
        }
    }

    /// Frees the slot at `pos`: unkeyed, clean, a new generation. The caller puts it on the free
    /// list or reuses it.
    fn release(&mut self, pos: Pos) {
        let Some(meta) = self.meta_mut(pos) else {
            return;
        };
        let key = meta.key.take();
        *meta = SlotMeta {
            generation: meta.generation.wrapping_add(1),
            ..SlotMeta::default()
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

    /// The bookkeeping of the slot at `pos`, mapped or not; `None` past the chunks ever mapped.
    fn meta_of(&self, (chunk, index): Pos) -> Option<&SlotMeta> {
        self.meta.get(chunk as usize)?.get(index as usize)
    }

    /// As [`Pool::meta_of`], mutably.
    fn meta_mut(&mut self, (chunk, index): Pos) -> Option<&mut SlotMeta> {
        self.meta.get_mut(chunk as usize)?.get_mut(index as usize)
    }

    /// The contents of the slot at `pos`, whatever it holds; `None` if its chunk isn't mapped.
    /// Unlike [`Pool::get`], no handle is checked: for the pool's own passes over its slots.
    fn slot(&self, (chunk, index): Pos) -> Option<&T> {
        self.chunks
            .get(chunk as usize)?
            .as_ref()?
            .get(index as usize)
    }

    /// As [`Pool::slot`], mutably.
    fn slot_mut(&mut self, (chunk, index): Pos) -> Option<&mut T> {
        self.chunks
            .get_mut(chunk as usize)?
            .as_mut()?
            .get_mut(index as usize)
    }

    /// The live slot's bookkeeping, or `None` if `id` is stale.
    fn live(&self, id: SlotId<T>) -> Option<&SlotMeta> {
        self.meta_of((id.chunk, id.index))
            .filter(|m| m.generation == id.generation && m.key.is_some())
    }

    /// As [`Pool::live`], mutably.
    fn live_mut(&mut self, id: SlotId<T>) -> Option<&mut SlotMeta> {
        self.meta_mut((id.chunk, id.index))
            .filter(|m| m.generation == id.generation && m.key.is_some())
    }
}

/// Proofs, for every input, run by `cargo kani -p store`.
#[cfg(kani)]
mod proofs {
    use super::*;

    /// A small slot, of the pool's own tests' kind.
    type Slot = [u64; 3];

    /// For every chunk size, a chunk holds at least one slot and its slots fit what it is
    /// charged.
    ///
    /// [`Pool::short_slots`] and [`Pool::pos_of`] are left out: they need a pool, which these
    /// proofs don't build.
    #[kani::proof]
    fn layouts_hold_their_slots() {
        let chunk_bytes: usize = kani::any();
        let (per_chunk, charged) = Pool::<Slot>::layout(chunk_bytes);
        assert!(per_chunk >= 1, "at least one slot");
        assert!(
            u64::from(per_chunk) * size_of::<Slot>() as u64 <= charged as u64,
            "the slots fit the charge"
        );
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;
