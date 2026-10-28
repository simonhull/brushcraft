//! GPU residency: which tiles have a slot on the GPU, how they get there, and which leave.
//!
//! # Slots
//!
//! Tile data lives in **slot pools**, one per size class (plan decision 2). A class is a tile
//! side: `2^l + 1` at levels up to 6, where a tile is a whole face, and 65 above. Each class is
//! a set of storage buffers of up to 128 MiB, cut into equal slots of [`GpuVertex`]es, created
//! as the budget ([`ResidencyConfig::budget_bytes`], from the capability tier) allows and kept;
//! a [`Slot`] is (class, buffer, index), and stays the tile's while it is resident.
//!
//! # What stays
//!
//! The **floor**, every face's display tiles up to level 5 ([`ResidencyConfig::floor_top`]), is
//! uploaded whole before the first frame and never leaves, so there is always something to
//! draw. Other tiles come when the level choice requests them ([`Residency::want`]) and leave
//! least recently drawn first, when their class needs a slot and the budget allows no new
//! buffer. Never evicted: the floor, a tile in the current cut, a tile whose children are drawn
//! (the cut and what was drawn come from the level choice each frame, [`FrameUse`]), and a tile
//! with an epoch's change pending.
//!
//! # Epochs
//!
//! The store's stale-tile feed names the tiles each operation changed ([`store::Epoch`]). An
//! epoch is applied **whole**: its resident tiles switch to their new data in one frame, so a
//! seam never shows one side new and the other old. A small epoch is written into the tiles'
//! own slots in the frame it is staged; a large one goes into **shadow slots** over as many
//! frames as it takes, the tiles keeping their old slots meanwhile, and switches when the last
//! is staged. A tile arriving while an epoch naming it is pending joins it, and epochs pending
//! together whose tiles overlap merge. A pending epoch that can't make progress (no shadow
//! slot to be had) goes whole in its next frame, in place, through a staging buffer of its
//! own. [`Drained::Resync`] (the feed lost this consumer's epochs) drops every tile that
//! isn't protected and re-reads the rest, the visible ones, as one epoch.
//!
//! # A frame
//!
//! On the owner thread:
//!
//! 1. [`Residency::begin_frame`]: polls the device, takes back staging buffers and request
//!    readbacks the GPU is done with.
//! 2. [`Residency::apply`] the drained feed; [`Residency::use_frame`] with the cut and what was
//!    drawn; [`Residency::take_requests`], the requests a frame's readback gave, to
//!    [`crate::upload::prefetch_requests`] and [`Residency::want`].
//! 3. [`Residency::stage`]: reads tiles from the source until the deadline, writes them and the
//!    tile tree's changed nodes into the staging ring, and returns the frame's [`FrameUploads`] for
//!    the render thread.
//!
//! The render thread encodes [`FrameUploads::encode`] before its passes and
//! [`FrameUploads::encode_requests`] after them, submits, and calls
//! [`FrameUploads::submitted`].

use std::collections::{BTreeMap, BTreeSet};
use std::convert::Infallible;
use std::time::Instant;

use geom::{BaseMesh, Level, TileKey};
use store::{CacheError, Drained, Epoch};
use thiserror::Error;
use wgpu::{Buffer, PollType};

use crate::gpu::{Gpu, GpuError};
use crate::tree::{NODE_BYTES, NODE_FINEST, NODE_FLOOR, Place, TileTree, TreeError};
use crate::upload::{Batch, FrameUploads, GpuVertex, StagingRing, TileSource, VERTEX_BYTES};

mod pool;
mod requests;
#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;

use pool::Pools;
use requests::RequestBuffer;
pub use requests::{GpuRequest, REQUEST_HEADER, REQUESTS_WGSL, Request, Requests};

/// How many size classes there are: tile sides 2, 3, 5, 9, 17, 33 and 65.
pub const CLASSES: usize = 7;

/// A slot size class: the side of the tiles it holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SizeClass(u8);

impl SizeClass {
    /// Class `class`, `0..CLASSES`: tiles of side `2^class + 1`.
    #[must_use]
    pub const fn new(class: u8) -> Option<Self> {
        if (class as usize) < CLASSES {
            Some(Self(class))
        } else {
            None
        }
    }

    /// The class of tiles at `level`: the level itself up to 6, where a tile is a whole face,
    /// then 6 (side 65) for every finer level.
    #[must_use]
    pub const fn of(level: Level) -> Self {
        /// The finest class: tiles of side 65.
        const LAST: u8 = 6;
        let l = level.get();
        Self(if l < LAST { l } else { LAST })
    }

    /// The class number, `0..CLASSES`.
    #[must_use]
    pub const fn get(self) -> u8 {
        self.0
    }

    /// Vertices per side of its tiles.
    #[must_use]
    pub const fn side(self) -> u32 {
        (1 << self.0) + 1
    }

    /// Vertices per tile.
    #[must_use]
    pub const fn vertices(self) -> u32 {
        self.side() * self.side()
    }

    /// Bytes per slot: [`SizeClass::vertices`] [`GpuVertex`]es.
    #[must_use]
    pub const fn slot_bytes(self) -> u64 {
        self.vertices() as u64 * VERTEX_BYTES
    }
}

/// Where a resident tile's vertices are: its class, the buffer within the class, the slot.
///
/// A tile keeps its slot while it stays resident (a pending epoch's data goes to a shadow slot,
/// and the tile moves there when the epoch switches).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Slot {
    /// The size class.
    class: SizeClass,
    /// The buffer, by index among the class's ([`Residency::slot_buffers`]).
    buffer: u16,
    /// The slot within the buffer: its first vertex is `index · class.vertices()`.
    index: u32,
}

impl Slot {
    /// The size class.
    #[must_use]
    pub const fn class(self) -> SizeClass {
        self.class
    }

    /// The buffer, by index among the class's ([`Residency::slot_buffers`]).
    #[must_use]
    pub const fn buffer(self) -> u16 {
        self.buffer
    }

    /// The slot within the buffer.
    #[must_use]
    pub const fn index(self) -> u32 {
        self.index
    }

    /// Slot `index` of `class`'s first buffer: for tests that need a slot without a pool.
    ///
    /// # Panics
    ///
    /// If `index` doesn't fit a `u32`.
    #[cfg(test)]
    pub(crate) fn for_tests(class: SizeClass, index: usize) -> Self {
        Self {
            class,
            buffer: 0,
            index: u32::try_from(index).expect("a small index"),
        }
    }

    /// The slot as the GPU holds it: class in bits 29 to 32, buffer in bits 21 to 29, index in
    /// bits 0 to 21 (the pools keep both within those).
    #[must_use]
    pub const fn packed(self) -> u32 {
        ((self.class.0 as u32) << 29) | ((self.buffer as u32) << 21) | self.index
    }
}

/// A capability tier: what the device is given, per the renderer spec's tier table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Tier {
    /// A desktop GPU (an RTX 4090, an M4): 4 GB of tile slots.
    Desktop,
    /// A high-end tablet (Galaxy Tab S8+, iPad M-series): 1 GB of tile slots.
    MobileHigh,
}

/// How residency is set up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidencyConfig {
    /// The most bytes the slot buffers may take together.
    pub budget_bytes: u64,
    /// Bytes per slot buffer, at most: 128 MiB, inside `wgpu`'s default binding limits (and
    /// never past the device's).
    pub buffer_bytes: u64,
    /// Bytes per staging buffer.
    pub staging_bytes: u64,
    /// The most bytes of staging buffers alive at once; a frame stages no more than the ring
    /// has free, but for an epoch that must go whole.
    pub staging_cap: u64,
    /// The largest epoch, in bytes of tile data, written into its tiles' own slots in the frame
    /// it is staged; a larger one goes through shadow slots.
    pub small_epoch_bytes: u64,
    /// Requests the request buffer holds per frame.
    pub request_capacity: u32,
    /// The finest level of the floor: every face's display tiles up to it are pinned.
    pub floor_top: Level,
}

/// A mebibyte.
const MIB: u64 = 1 << 20;

/// The floor's finest level by default: the spec's "levels ≤ 5".
const FLOOR_TOP: Level = match Level::new(5) {
    Ok(level) => level,
    Err(_) => Level::ZERO,
};

impl ResidencyConfig {
    /// The configuration of `tier`.
    #[must_use]
    pub const fn for_tier(tier: Tier) -> Self {
        let (budget_bytes, staging_bytes) = match tier {
            Tier::Desktop => (4 << 30, 16 * MIB),
            Tier::MobileHigh => (1 << 30, 8 * MIB),
        };
        Self {
            budget_bytes,
            buffer_bytes: 128 * MIB,
            staging_bytes,
            staging_cap: 4 * staging_bytes,
            small_epoch_bytes: 8 * MIB,
            request_capacity: 4096,
            floor_top: FLOOR_TOP,
        }
    }

    /// This configuration with a budget of `budget_bytes`.
    #[must_use]
    pub const fn with_budget(self, budget_bytes: u64) -> Self {
        Self {
            budget_bytes,
            ..self
        }
    }
}

impl Default for ResidencyConfig {
    fn default() -> Self {
        Self::for_tier(Tier::Desktop)
    }
}

/// What the level choice used in a frame: the cut it chose and the tiles it drew. Their tiles,
/// and the parents of either, are never evicted while this is the latest.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FrameUse {
    /// The tiles of the current cut.
    cut: BTreeSet<TileKey>,
    /// The tiles drawn (the cut after culling): their recency is renewed.
    drawn: BTreeSet<TileKey>,
}

impl FrameUse {
    /// A frame's use: the tiles of its `cut`, and those it `drawn`.
    #[must_use]
    pub const fn new(cut: BTreeSet<TileKey>, drawn: BTreeSet<TileKey>) -> Self {
        Self { cut, drawn }
    }
}

/// Why residency failed.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ResidencyError {
    /// The GPU failed: making a buffer, mapping one, polling the device.
    #[error(transparent)]
    Gpu(#[from] GpuError),
    /// The tile tree is full: it is sized for every slot the budget allows, so this is a bug.
    #[error(transparent)]
    Tree(#[from] TreeError),
    /// The store couldn't give a tile.
    #[error("the store couldn't give a tile: {0}")]
    Store(#[from] CacheError),
}

impl From<Infallible> for ResidencyError {
    fn from(never: Infallible) -> Self {
        match never {}
    }
}

/// Counts for the debug overlay and the tests.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResidencyStats {
    /// Tiles resident (drawable).
    pub resident: usize,
    /// Floor tiles resident.
    pub floor: usize,
    /// Bytes of slot buffers.
    pub allocated_bytes: u64,
    /// Tiles requested and not resident yet.
    pub wanted: usize,
    /// Tiles the source had nothing for, waiting for an epoch naming them.
    pub unavailable: usize,
    /// Epochs (merged) pending.
    pub pending_epochs: usize,
    /// Tiles evicted, ever.
    pub evictions: u64,
    /// Requests left waiting for want of a slot, ever (counted each frame).
    pub starved: u64,
    /// Resident tiles dropped because an epoch left the source with nothing for them, ever.
    pub dropped: u64,
    /// Pending epochs forced whole into one frame, ever.
    pub forced: u64,
    /// Resyncs, ever.
    pub resyncs: u64,
    /// Bytes the last frame staged.
    pub staged_bytes: u64,
}

/// A resident tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Entry {
    /// Its slot.
    slot: Slot,
    /// The frame it was last drawn in (or arrived in).
    last_drawn: u64,
    /// Whether it is part of the floor.
    floor: bool,
}

/// A pending tile's new data.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Staged {
    /// Uploaded into `slot` (a shadow slot, or the tile's own for an epoch written in place),
    /// with bound `error`.
    Data {
        /// Where it is.
        slot: Slot,
        /// Its bound `e`.
        error: f32,
    },
    /// The source had nothing for it: it leaves when the epoch switches.
    Missing,
}

/// One or more epochs pending together.
#[derive(Debug, Default)]
struct Group {
    /// Every tile its epochs named, resident or not: a tile arriving that is one of them joins.
    keys: BTreeSet<TileKey>,
    /// Members whose new data isn't staged yet.
    todo: BTreeSet<TileKey>,
    /// Members whose new data is staged.
    staged: BTreeMap<TileKey, Staged>,
    /// Frames it was staged in without progress.
    stalled: u32,
}

impl Group {
    /// Whether `key` is one of its members (pending, staged or not).
    fn holds(&self, key: &TileKey) -> bool {
        self.todo.contains(key) || self.staged.contains_key(key)
    }

    /// Takes `other`'s keys and members.
    fn absorb(&mut self, other: Self) {
        self.keys.extend(other.keys);
        self.todo.extend(other.todo);
        self.staged.extend(other.staged);
        self.stalled = self.stalled.max(other.stalled);
    }
}

/// What reading one tile into staging came to.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Read {
    /// Staged, with its bound `e`.
    Staged(f32),
    /// The source had nothing for it.
    Missing,
    /// The staging ring had no room.
    NoRoom,
}

/// The GPU residency of one project's tiles (see the module notes).
///
/// Owned by the store's owner thread.
#[derive(Debug)]
pub struct Residency {
    /// How it is set up.
    config: ResidencyConfig,
    /// The project's finest level: its tiles are finest tiles, those above display tiles.
    finest: Level,
    /// How many faces the base mesh has.
    faces: u32,
    /// The slot pools.
    pools: Pools,
    /// The resident tile tree.
    tree: TileTree,
    /// Every resident tile.
    entries: BTreeMap<TileKey, Entry>,
    /// Per class, resident tiles outside the floor by the frame they were last drawn in.
    recency: Vec<BTreeSet<(u64, TileKey)>>,
    /// The floor's tiles.
    floor: BTreeSet<TileKey>,
    /// Tiles requested, by screen error.
    wanted: BTreeMap<TileKey, f32>,
    /// Tiles the source had nothing for, until an epoch names them.
    unavailable: BTreeSet<TileKey>,
    /// Pending epochs, oldest first.
    groups: Vec<Group>,
    /// Members of pending epochs as staging began: never evicted.
    pending: BTreeSet<TileKey>,
    /// The frame number: one more each [`Residency::begin_frame`].
    frame: u64,
    /// The current cut.
    cut: BTreeSet<TileKey>,
    /// The places whose children are in the cut or drawn.
    shielded: BTreeSet<Place>,
    /// The staging ring.
    ring: StagingRing,
    /// The request buffer and its readbacks.
    requests: RequestBuffer,
    /// Counts.
    stats: ResidencyStats,
    /// A tile's vertices, as read from the source.
    scratch: Vec<GpuVertex>,
}

impl Residency {
    /// Residency for a project of `mesh` whose finest level is `finest`, with nothing resident
    /// yet: the floor is staged by the first [`Residency::stage`], whole.
    ///
    /// # Errors
    ///
    /// [`ResidencyError::Gpu`] or [`ResidencyError::Tree`] if a buffer can't be created.
    pub fn new(
        gpu: &Gpu,
        mesh: &BaseMesh,
        finest: Level,
        config: ResidencyConfig,
    ) -> Result<Self, ResidencyError> {
        let limits = gpu.device().limits();
        let buffer_bytes = config
            .buffer_bytes
            .min(limits.max_storage_buffer_binding_size)
            .min(limits.max_buffer_size);
        let faces = mesh.face_count();
        let pools = Pools::new(faces, finest, buffer_bytes, config.budget_bytes);
        let capacity = u32::try_from(pools.max_slots()).unwrap_or(u32::MAX);
        let tree = TileTree::new(gpu, mesh, capacity)?;
        let requests = RequestBuffer::new(gpu, config.request_capacity)?;
        let floor_levels = (0..=config.floor_top.get())
            .filter_map(|l| Level::new(l).ok())
            .filter(|&level| level < finest);
        let floor_levels: Vec<Level> = floor_levels.collect();
        let floor: BTreeSet<TileKey> = mesh
            .faces()
            .flat_map(|face| {
                floor_levels.iter().flat_map(move |&level| {
                    let per_side = u16::try_from(level.tiles_per_side()).unwrap_or(0);
                    (0..per_side).flat_map(move |ty| {
                        (0..per_side)
                            .filter_map(move |tx| TileKey::display(face, level, tx, ty).ok())
                    })
                })
            })
            .collect();
        // A staging buffer holds at least one tile of the largest class, or a tile that isn't
        // forced would never find room.
        let staging_bytes = config.staging_bytes.max(SizeClass::of(finest).slot_bytes());
        let staging_cap = config.staging_cap.max(staging_bytes);
        Ok(Self {
            ring: StagingRing::new(staging_bytes, staging_cap),
            config,
            finest,
            faces,
            pools,
            tree,
            entries: BTreeMap::new(),
            recency: vec![BTreeSet::new(); CLASSES],
            floor,
            wanted: BTreeMap::new(),
            unavailable: BTreeSet::new(),
            groups: Vec::new(),
            pending: BTreeSet::new(),
            frame: 0,
            cut: BTreeSet::new(),
            shielded: BTreeSet::new(),
            requests,
            stats: ResidencyStats::default(),
            scratch: Vec::new(),
        })
    }

    /// The project's finest level.
    #[must_use]
    pub const fn finest(&self) -> Level {
        self.finest
    }

    /// The resident tile tree, for the passes that walk it.
    #[must_use]
    pub const fn tree(&self) -> &TileTree {
        &self.tree
    }

    /// `class`'s slot buffers, by buffer index ([`Slot::buffer`]): each holds whole slots of
    /// [`GpuVertex`]es. A class gains buffers as tiles arrive (rebind when the count changes);
    /// a buffer, once made, stays.
    #[must_use]
    pub fn slot_buffers(&self, class: SizeClass) -> &[Buffer] {
        self.pools.buffers(class)
    }

    /// The request buffer the level choice appends wanted tiles to ([`REQUESTS_WGSL`]).
    #[must_use]
    pub const fn request_buffer(&self) -> &Buffer {
        self.requests.buffer()
    }

    /// `key`'s slot, if it is resident.
    #[must_use]
    pub fn slot(&self, key: &TileKey) -> Option<Slot> {
        self.entries.get(key).map(|entry| entry.slot)
    }

    /// The resident tiles, in key order.
    pub fn resident(&self) -> impl Iterator<Item = &TileKey> {
        self.entries.keys()
    }

    /// Whether every floor tile the source could give is resident: true after the first
    /// [`Residency::stage`].
    #[must_use]
    pub fn floor_ready(&self) -> bool {
        self.floor
            .iter()
            .all(|key| self.entries.contains_key(key) || self.unavailable.contains(key))
    }

    /// Counts, as they stand.
    #[must_use]
    pub fn stats(&self) -> ResidencyStats {
        ResidencyStats {
            resident: self.entries.len(),
            floor: self.entries.values().filter(|entry| entry.floor).count(),
            allocated_bytes: self.pools.allocated(),
            wanted: self.wanted.len(),
            unavailable: self.unavailable.len(),
            pending_epochs: self.groups.len(),
            ..self.stats
        }
    }

    /// Starts a frame: polls the device without waiting, takes back the staging buffers the
    /// GPU is done with, and reads the request readbacks that completed.
    ///
    /// # Errors
    ///
    /// [`ResidencyError::Gpu`] if polling fails or a readback can't be read.
    pub fn begin_frame(&mut self, gpu: &Gpu) -> Result<(), ResidencyError> {
        self.frame += 1;
        gpu.device().poll(PollType::Poll).map_err(GpuError::from)?;
        self.ring.collect();
        self.requests.collect(self.finest)?;
        Ok(())
    }

    /// The requests the latest request readbacks gave (merged, if several arrived), if any
    /// arrived since the last call: for [`crate::upload::prefetch_requests`], then
    /// [`Residency::want`].
    pub const fn take_requests(&mut self) -> Option<Requests> {
        self.requests.take()
    }

    /// Wants the tiles of `requests` resident, larger screen error first: they replace the
    /// requests wanted before (the floor stays wanted until resident). Resident tiles, tiles
    /// of no face of the mesh, and tiles the source had nothing for are skipped.
    pub fn want(&mut self, requests: &Requests) {
        let floor = &self.floor;
        self.wanted.retain(|key, _| floor.contains(key));
        let faces = self.faces;
        let finest = self.finest;
        let fresh = requests.iter().filter(|request| {
            request.key.face().index() < faces
                && request.key.level() <= finest
                && !self.entries.contains_key(&request.key)
                && !self.unavailable.contains(&request.key)
        });
        let fresh: Vec<(TileKey, f32)> = fresh.map(|r| (r.key, r.screen_error)).collect();
        self.wanted.extend(fresh);
    }

    /// Takes what the level choice used this frame: `frame.drawn`'s tiles are renewed as most
    /// recently drawn, and until the next call `frame.cut`'s tiles and the parents of either's
    /// are never evicted.
    pub fn use_frame(&mut self, frame: FrameUse) {
        let FrameUse { cut, drawn } = frame;
        for key in &drawn {
            if let Some(entry) = self.entries.get_mut(key) {
                if !entry.floor
                    && let Some(recency) = self.recency.get_mut(usize::from(entry.slot.class.0))
                {
                    recency.remove(&(entry.last_drawn, *key));
                    recency.insert((self.frame, *key));
                }
                entry.last_drawn = self.frame;
            }
        }
        let parents = cut
            .iter()
            .chain(&drawn)
            .filter_map(|key| Place::of(key).parent());
        self.shielded = parents.collect();
        self.cut = cut;
    }

    /// Applies what the store's stale-tile feed gave: each epoch becomes pending, to switch
    /// whole; [`Drained::Resync`] re-reads every tile kept.
    pub fn apply(&mut self, drained: Drained) {
        match drained {
            Drained::Epochs(epochs) => {
                for epoch in epochs {
                    self.apply_epoch(epoch);
                }
            }
            Drained::Resync => self.resync(),
        }
    }

    /// Makes `epoch` pending: its resident tiles, merged with the pending epochs that share a
    /// member with it. A member already staged for an earlier epoch is staged again.
    fn apply_epoch(&mut self, epoch: Epoch) {
        let keys = epoch.keys;
        for key in &keys {
            if self.unavailable.remove(key) && self.floor.contains(key) {
                self.wanted.insert(*key, f32::INFINITY);
            }
        }
        let live: BTreeSet<TileKey> = keys
            .iter()
            .filter(|key| self.entries.contains_key(key))
            .copied()
            .collect();
        let (touching, rest): (Vec<Group>, Vec<Group>) = std::mem::take(&mut self.groups)
            .into_iter()
            .partition(|group| keys.iter().any(|key| group.holds(key)));
        self.groups = rest;
        if live.is_empty() && touching.is_empty() {
            return;
        }
        let mut merged = Group {
            todo: live,
            ..Group::default()
        };
        for group in touching {
            merged.absorb(group);
        }
        for key in &keys {
            if let Some(staged) = merged.staged.remove(key) {
                self.release_staged(key, staged);
                merged.todo.insert(*key);
            }
        }
        merged.keys.extend(keys);
        self.groups.push(merged);
    }

    /// Gives back a staged tile's slot, unless it is the tile's own (written in place).
    fn release_staged(&mut self, key: &TileKey, staged: Staged) {
        if let Staged::Data { slot, .. } = staged
            && self.entries.get(key).is_none_or(|entry| entry.slot != slot)
        {
            self.pools.release(slot);
        }
    }

    /// After the feed lost this consumer's epochs: drops every pending epoch and every
    /// resident tile that isn't protected, and makes the rest (the floor and what the frame
    /// uses) one pending epoch, re-read whole.
    fn resync(&mut self) {
        self.stats.resyncs += 1;
        for group in std::mem::take(&mut self.groups) {
            for (key, staged) in group.staged {
                self.release_staged(&key, staged);
            }
        }
        let unprotected: Vec<TileKey> = self
            .entries
            .keys()
            .filter(|key| !self.protected(key))
            .copied()
            .collect();
        for key in unprotected {
            self.remove_live(&key);
        }
        let kept: BTreeSet<TileKey> = self.entries.keys().copied().collect();
        let retry = std::mem::take(&mut self.unavailable);
        let floor_retry = retry.into_iter().filter(|key| self.floor.contains(key));
        let floor_retry: Vec<(TileKey, f32)> =
            floor_retry.map(|key| (key, f32::INFINITY)).collect();
        self.wanted.extend(floor_retry);
        if !kept.is_empty() {
            self.groups.push(Group {
                keys: kept.clone(),
                todo: kept,
                ..Group::default()
            });
        }
    }

    /// Whether `key` must stay: the floor, the current cut, or a tile whose children are in
    /// the cut or drawn.
    fn protected(&self, key: &TileKey) -> bool {
        self.floor.contains(key)
            || self.cut.contains(key)
            || self.shielded.contains(&Place::of(key))
    }

    /// Stages the frame's uploads from `source`: the floor first if it isn't resident (whole,
    /// whatever the deadline), then the pending epochs oldest first, then requested tiles,
    /// largest screen error first, until `deadline` or until the staging ring is full; then
    /// the tile tree's changed nodes. Returns them for the render thread.
    ///
    /// # Errors
    ///
    /// [`ResidencyError::Gpu`] if a buffer can't be made or written; the source's errors; the
    /// tiles staged until then are lost with the frame (and staged again later).
    pub fn stage<S: TileSource>(
        &mut self,
        gpu: &Gpu,
        source: &mut S,
        deadline: Instant,
    ) -> Result<FrameUploads, ResidencyError>
    where
        S::Error: Into<ResidencyError>,
    {
        let mut batch = Batch::default();
        self.pending = self
            .groups
            .iter()
            .flat_map(|group| group.todo.iter().chain(group.staged.keys()))
            .copied()
            .collect();
        self.stage_floor(gpu, &mut batch, source)?;
        for group in std::mem::take(&mut self.groups) {
            if let Some(group) = self.stage_group(gpu, &mut batch, source, group, deadline)? {
                self.groups.push(group);
            }
        }
        self.stage_wanted(gpu, &mut batch, source, deadline)?;
        for (start, nodes) in self.tree.take_dirty() {
            let offset = u64::from(start) * NODE_BYTES;
            batch.write(
                gpu,
                &mut self.ring,
                bytemuck::cast_slice(&nodes),
                self.tree.buffer(),
                offset,
                true,
            )?;
        }
        self.stats.staged_bytes = batch.bytes();
        let requests = self.requests.copy(gpu)?;
        Ok(batch.finish(&self.ring, requests))
    }

    /// Stages every floor tile not resident, whole, past the ring's cap if need be.
    ///
    /// # Errors
    ///
    /// As [`Residency::stage`].
    fn stage_floor<S: TileSource>(
        &mut self,
        gpu: &Gpu,
        batch: &mut Batch,
        source: &mut S,
    ) -> Result<(), ResidencyError>
    where
        S::Error: Into<ResidencyError>,
    {
        let missing: Vec<TileKey> = self
            .floor
            .iter()
            .filter(|key| {
                !self.entries.contains_key(key)
                    && !self.unavailable.contains(key)
                    && !self.pending.contains(key)
            })
            .copied()
            .collect();
        if missing.is_empty() {
            return Ok(());
        }
        source.prepare(&missing).map_err(Into::into)?;
        for key in missing {
            self.wanted.remove(&key);
            self.arrive(gpu, batch, source, key, true)?;
        }
        Ok(())
    }

    /// Stages the requested tiles, largest screen error first, until `deadline` or the ring
    /// is full.
    ///
    /// # Errors
    ///
    /// As [`Residency::stage`].
    fn stage_wanted<S: TileSource>(
        &mut self,
        gpu: &Gpu,
        batch: &mut Batch,
        source: &mut S,
        deadline: Instant,
    ) -> Result<(), ResidencyError>
    where
        S::Error: Into<ResidencyError>,
    {
        let mut by_error: Vec<(TileKey, f32)> = self.wanted.iter().map(|(k, e)| (*k, *e)).collect();
        by_error.sort_by(|a, b| b.1.total_cmp(&a.1));
        let waiting = by_error
            .iter()
            .filter(|(key, _)| !self.is_pending_arrival(key));
        let order: Vec<TileKey> = waiting.map(|(key, _)| *key).collect();
        let room = batch.room(&self.ring);
        let mut planned = 0;
        let fits = order.iter().take_while(|key| {
            planned += SizeClass::of(key.level()).slot_bytes();
            planned <= room
        });
        let ahead: Vec<TileKey> = fits.copied().collect();
        source.prepare(&ahead).map_err(Into::into)?;
        for key in order {
            if Instant::now() >= deadline {
                break;
            }
            if self.entries.contains_key(&key) || self.unavailable.contains(&key) {
                self.wanted.remove(&key);
                continue;
            }
            if batch.room(&self.ring) < SizeClass::of(key.level()).slot_bytes() {
                break;
            }
            match self.arrive(gpu, batch, source, key, false)? {
                Read::NoRoom => break,
                Read::Staged(_) | Read::Missing => {
                    self.wanted.remove(&key);
                }
            }
        }
        Ok(())
    }

    /// Whether `key` is arriving as a member of a pending epoch, staged already.
    fn is_pending_arrival(&self, key: &TileKey) -> bool {
        self.groups
            .iter()
            .any(|group| group.staged.contains_key(key))
    }

    /// Stages `key`, not resident, into a slot of its own: it becomes resident now, or with
    /// the pending epoch naming it when that switches. With `forced`, past the ring's cap.
    /// Without a slot to be had, it waits (counted as starved).
    ///
    /// # Errors
    ///
    /// As [`Residency::stage`].
    fn arrive<S: TileSource>(
        &mut self,
        gpu: &Gpu,
        batch: &mut Batch,
        source: &mut S,
        key: TileKey,
        forced: bool,
    ) -> Result<Read, ResidencyError>
    where
        S::Error: Into<ResidencyError>,
    {
        let Some(slot) = self.acquire(gpu, SizeClass::of(key.level()))? else {
            self.stats.starved += 1;
            return Ok(Read::NoRoom);
        };
        let read = self.read_into(gpu, batch, source, key, slot, forced)?;
        match read {
            Read::Staged(error) => match self.joining(&key) {
                Some(group) => {
                    group.staged.insert(key, Staged::Data { slot, error });
                }
                None => self.insert_live(key, slot, error)?,
            },
            Read::Missing => {
                self.pools.release(slot);
                self.unavailable.insert(key);
            }
            Read::NoRoom => self.pools.release(slot),
        }
        Ok(read)
    }

    /// The pending epoch a tile arriving now joins: the one naming it, the pending epochs
    /// naming it merged into one.
    fn joining(&mut self, key: &TileKey) -> Option<&mut Group> {
        let (naming, rest): (Vec<Group>, Vec<Group>) = std::mem::take(&mut self.groups)
            .into_iter()
            .partition(|group| group.keys.contains(key));
        self.groups = rest;
        let mut naming = naming.into_iter();
        let mut merged = naming.next()?;
        for group in naming {
            merged.absorb(group);
        }
        self.groups.push(merged);
        self.groups.last_mut()
    }

    /// Stages what it can of pending `group`, and switches it if it is complete: whole in place
    /// if small enough for this frame (or stalled), else into shadow slots until `deadline`.
    /// Returns the group if it is still pending.
    ///
    /// # Errors
    ///
    /// As [`Residency::stage`].
    fn stage_group<S: TileSource>(
        &mut self,
        gpu: &Gpu,
        batch: &mut Batch,
        source: &mut S,
        mut group: Group,
        deadline: Instant,
    ) -> Result<Option<Group>, ResidencyError>
    where
        S::Error: Into<ResidencyError>,
    {
        let todo: Vec<TileKey> = group.todo.iter().copied().collect();
        let remaining: u64 = todo
            .iter()
            .map(|key| SizeClass::of(key.level()).slot_bytes())
            .sum();
        let forced = group.stalled > 0;
        let in_place = forced
            || (remaining <= self.config.small_epoch_bytes && remaining <= batch.room(&self.ring));
        source.prepare(&todo).map_err(Into::into)?;
        if in_place {
            if forced {
                self.stats.forced += 1;
            }
            for key in todo {
                group.todo.remove(&key);
                let own = self.entries.get(&key).map(|entry| entry.slot);
                let target = match own {
                    Some(slot) => slot,
                    None => match self.acquire(gpu, SizeClass::of(key.level()))? {
                        Some(slot) => slot,
                        // An arrival with no slot to be had: it isn't drawn yet, so it waits
                        // to be requested again.
                        None => continue,
                    },
                };
                match self.read_into(gpu, batch, source, key, target, true)? {
                    Read::Staged(error) => {
                        group.staged.insert(
                            key,
                            Staged::Data {
                                slot: target,
                                error,
                            },
                        );
                    }
                    Read::Missing | Read::NoRoom => {
                        if own.is_none() {
                            self.pools.release(target);
                        }
                        group.staged.insert(key, Staged::Missing);
                    }
                }
            }
            self.switch(group)?;
            return Ok(None);
        }
        let mut progressed = false;
        for key in todo {
            if Instant::now() >= deadline {
                break;
            }
            let Some(slot) = self.acquire(gpu, SizeClass::of(key.level()))? else {
                continue;
            };
            match self.read_into(gpu, batch, source, key, slot, false)? {
                Read::Staged(error) => {
                    group.staged.insert(key, Staged::Data { slot, error });
                }
                Read::Missing => {
                    self.pools.release(slot);
                    group.staged.insert(key, Staged::Missing);
                }
                Read::NoRoom => {
                    self.pools.release(slot);
                    break;
                }
            }
            group.todo.remove(&key);
            progressed = true;
        }
        if group.todo.is_empty() {
            self.switch(group)?;
            return Ok(None);
        }
        if !progressed {
            group.stalled += 1;
        }
        Ok(Some(group))
    }

    /// Switches a complete pending epoch: every staged member takes its new slot (giving its
    /// old one back) or arrives, and every member the source had nothing for leaves.
    ///
    /// # Errors
    ///
    /// [`ResidencyError::Tree`] if an arriving tile finds the tree full (it is sized for every
    /// slot, so it doesn't).
    fn switch(&mut self, group: Group) -> Result<(), ResidencyError> {
        for (key, staged) in group.staged {
            let place = Place::of(&key);
            match staged {
                Staged::Data { slot, error } => {
                    if let Some(entry) = self.entries.get_mut(&key) {
                        let old = entry.slot;
                        entry.slot = slot;
                        self.tree.move_to(place, slot, error);
                        if old != slot {
                            self.pools.release(old);
                        }
                    } else {
                        self.insert_live(key, slot, error)?;
                    }
                }
                Staged::Missing => {
                    if self.entries.contains_key(&key) {
                        self.remove_live(&key);
                        self.stats.dropped += 1;
                    }
                    self.unavailable.insert(key);
                }
            }
        }
        Ok(())
    }

    /// Reads `key` from `source` and stages it for `slot`. With `forced`, past the ring's cap.
    ///
    /// # Errors
    ///
    /// As [`Residency::stage`].
    fn read_into<S: TileSource>(
        &mut self,
        gpu: &Gpu,
        batch: &mut Batch,
        source: &mut S,
        key: TileKey,
        slot: Slot,
        forced: bool,
    ) -> Result<Read, ResidencyError>
    where
        S::Error: Into<ResidencyError>,
    {
        let bytes = slot.class.slot_bytes();
        if !forced && batch.room(&self.ring) < bytes {
            return Ok(Read::NoRoom);
        }
        let vertices = usize::try_from(slot.class.vertices()).unwrap_or(usize::MAX);
        self.scratch.clear();
        self.scratch.resize(vertices, GpuVertex::default());
        let Some(error) = source.read(key, &mut self.scratch).map_err(Into::into)? else {
            return Ok(Read::Missing);
        };
        let Some((buffer, offset)) = self.pools.locate(slot) else {
            return Ok(Read::NoRoom);
        };
        let data: &[u8] = bytemuck::cast_slice(&self.scratch);
        if batch.write(gpu, &mut self.ring, data, buffer, offset, forced)? {
            Ok(Read::Staged(error))
        } else {
            Ok(Read::NoRoom)
        }
    }

    /// A free slot of `class`, evicting the least recently drawn tile of the class that may
    /// leave if the pools have none; `None` if none may.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`], making a buffer.
    fn acquire(&mut self, gpu: &Gpu, class: SizeClass) -> Result<Option<Slot>, ResidencyError> {
        if let Some(slot) = self.pools.acquire(gpu, class)? {
            return Ok(Some(slot));
        }
        let candidates = self
            .recency
            .get(usize::from(class.0))
            .into_iter()
            .flatten()
            .map(|(_, key)| key);
        let victim = candidates
            .filter(|key| !self.protected(key) && !self.pending.contains(key))
            .find(|key| !self.groups.iter().any(|group| group.holds(key)))
            .copied();
        let Some(victim) = victim else {
            return Ok(None);
        };
        self.remove_live(&victim);
        self.stats.evictions += 1;
        Ok(self.pools.acquire(gpu, class)?)
    }

    /// Makes `key` resident in `slot` with bound `error`.
    ///
    /// # Errors
    ///
    /// [`ResidencyError::Tree`] if the tree is full.
    fn insert_live(&mut self, key: TileKey, slot: Slot, error: f32) -> Result<(), ResidencyError> {
        let floor = self.floor.contains(&key);
        let mut flags = 0;
        if floor {
            flags |= NODE_FLOOR;
        }
        if key.level() == self.finest {
            flags |= NODE_FINEST;
        }
        self.tree.insert(Place::of(&key), slot, error, flags)?;
        self.entries.insert(
            key,
            Entry {
                slot,
                last_drawn: self.frame,
                floor,
            },
        );
        if !floor && let Some(recency) = self.recency.get_mut(usize::from(slot.class.0)) {
            recency.insert((self.frame, key));
        }
        Ok(())
    }

    /// Makes `key` not resident, giving back its slot.
    fn remove_live(&mut self, key: &TileKey) {
        let Some(entry) = self.entries.remove(key) else {
            return;
        };
        if let Some(recency) = self.recency.get_mut(usize::from(entry.slot.class.0)) {
            recency.remove(&(entry.last_drawn, *key));
        }
        self.tree.remove(Place::of(key));
        self.pools.release(entry.slot);
    }
}
