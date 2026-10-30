//! GPU residency: which tiles have a slot on the GPU, how they get there, and which leave.
//!
//! # Slots
//!
//! Tile data lives in **slot pools**, one per size class (plan decision 2). A class is a tile
//! side: `2^l + 1` at levels up to 6, where a tile is a whole face, and 65 above. Each class is
//! a few storage buffers, the first 64 MiB and each next twice the last, up to as large as the
//! device binds (`max_storage_buffer_binding_size`; see [`Tier`]), cut into equal slots of
//! [`GpuVertex`](crate::upload::GpuVertex)es, created as the budget
//! ([`ResidencyConfig::budget_bytes`], from the capability tier) allows and kept; a [`Slot`] is
//! (class, buffer, index). A tile keeps its slot until an epoch large enough for shadow slots
//! moves it; what stays the tile's while it is resident is its node index in the tile tree
//! ([`crate::tree`]), and that is what the visibility buffer's cluster id derives from.
//!
//! A finest tile's slot is like any other of its side: its one-vertex ring isn't stored. The tile
//! is staged as its positions with the ring (a ring record), and the frame's normal pass writes
//! its slot's vertices from them (the crate's `normals` module). Room and the forced
//! bytes are counted by the slot, a little over twice the record, so on the safe side.
//!
//! # What stays
//!
//! The **floor**, every face's display tiles up to level 5 ([`ResidencyConfig::floor_top`]), is
//! uploaded whole before the first frame and never leaves, so there is always something to
//! draw. Other tiles come when the level choice requests them ([`Residency::want`]) and leave
//! least recently drawn first, when their class needs a slot and the budget allows no new
//! buffer. Never evicted: the floor, a tile in the current cut, a tile whose children are drawn
//! (the cut and what was drawn come from the level choice each frame, [`FrameUse`]), a tile
//! with an epoch's change pending, and a tile uploaded in the frame being staged.
//!
//! # Epochs
//!
//! The store's stale-tile feed names the tiles each operation changed ([`store::Epoch`]). An
//! epoch is applied **whole**: its resident tiles switch to their new data in one frame, so a
//! seam never shows one side new and the other old. A small epoch is written into the tiles'
//! own slots in the frame it is staged; a large one goes into **shadow slots** over as many
//! frames as it takes, the tiles keeping their old slots meanwhile, and switches when the last
//! is staged. A tile arriving while an epoch naming it is pending joins it, and epochs pending
//! together whose tiles overlap merge.
//!
//! A pending epoch goes whole into one frame, in place (**forced**), when it stalled (no shadow
//! slot to be had), or when a small epoch (a brush step) merged into it, so that the step shows
//! now; forced bytes per frame are capped ([`ResidencyConfig::forced_bytes`]), past the cap only
//! for an epoch stalled for [`LONG_STALL`] frames. [`Drained::Resync`] (the feed lost this
//! consumer's epochs) drops every tile that isn't protected and re-reads the rest as one epoch,
//! which every tile arriving meanwhile joins.
//!
//! # A frame
//!
//! On the owner thread:
//!
//! 1. [`Residency::begin_frame`]: polls the device, takes back staging buffers and request
//!    readbacks the GPU is done with, and hears which frames were submitted or lost.
//! 2. [`Residency::apply`] the drained feed; [`Residency::use_frame`] with the cut and what was
//!    drawn; [`Residency::take_requests`], the requests a frame's readback gave, to
//!    [`crate::upload::prefetch_requests`] and [`Residency::want`] (staged from the next frame on,
//!    so the page-in or the read ahead has a frame to land; a display tile whose data is at hand
//!    already, from this one).
//! 3. [`Residency::stage`]: reads tiles from the source until the deadline, writes them and the
//!    tile tree's changed nodes into the staging ring, and returns the frame's [`FrameUploads`] for
//!    the render thread.
//!
//! The render thread hands its passes to [`FrameUploads::submit`], which encodes them after
//! the uploads' copies and submits them together. A frame that fails to stage, or is dropped
//! unsubmitted, drops the tiles it wrote (their data never reached the GPU) and the whole tile tree
//! is written again with the next frame; the floor comes back whole.

use std::collections::{BTreeMap, BTreeSet};
use std::convert::Infallible;
use std::ops::Bound;
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::Instant;

use geom::{BaseMesh, Channel, Level, TileKey};
use store::{CacheError, Drained, Epoch};
use thiserror::Error;
use wgpu::{Buffer, MAP_ALIGNMENT, PollType};

use crate::gpu::{Gpu, GpuError};
use crate::normals::NormalPass;
use crate::tree::{NODE_BYTES, NODE_FINEST, NODE_FLOOR, Place, TileTree, TreeError};
use crate::upload::{
    Batch, FrameOutcome, FrameUploads, StagingRing, TileData, TileDetail, TileSource, Upload,
    VERTEX_BYTES,
};

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

    /// Bytes per slot: [`SizeClass::vertices`] [`GpuVertex`](crate::upload::GpuVertex)es.
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
///
/// The device is opened with its adapter's limits ([`Gpu::open`]). A class's slot buffers start
/// at 64 MiB and double up to as large as `max_storage_buffer_binding_size` allows (`pool.rs`),
/// so a class is a few buffers. As measured (`cargo run -p render --example adapters`, wgpu
/// 30.0.1), with the buffers the benchmark project's finest class and all classes can have at
/// the budget (counted in `pool.rs`'s tests):
///
/// | Tier | Adapter | Storage binding | Buffer | Storage buffers per stage | Finest class's buffers | All classes' |
/// |---|---|---|---|---|---|---|
/// | Desktop | RTX 4090 (Vulkan) | 2047 MiB | 4095 MiB | 524,288 | 7 | 14 |
/// | Desktop | Apple M4 (Metal) | 4095 MiB | 4095 MiB | not recorded | 7 | 14 |
/// | (gate) | lavapipe (Vulkan) | 128 MiB | 2047 MiB | 48 | 33 | 40 |
/// | Mobile high | Adreno 730 | pending (M1c) | | | |
///
/// The bounds pass binds one slot buffer per dispatch. The raster's one draw a phase binds every
/// slot buffer at once, so [`Residency::new`] refuses a configuration whose most buffers
/// ([`SlotLayout`]) don't fit `max_storage_buffers_per_shader_stage` beside the raster's own
/// ([`ConfigError::TooManySlotBuffers`]; binding arrays are not used, not being on every
/// backend). A tier past that would draw one indirect draw per slot buffer, from cluster lists
/// the level choice splits by buffer; not built, no tier measured needs it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Tier {
    /// A desktop GPU (an RTX 4090, an M4): 4 GiB of tile slots.
    Desktop,
    /// A high-end tablet (Galaxy Tab S8+, iPad M-series): 1 GiB of tile slots.
    MobileHigh,
}

/// Why a residency configuration was refused.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ConfigError {
    /// A size isn't a positive multiple of [`MAP_ALIGNMENT`], as mapping and copying need.
    #[error("{what} is {bytes} bytes: it must be a positive multiple of {MAP_ALIGNMENT}")]
    Unaligned {
        /// Which size.
        what: &'static str,
        /// Its value.
        bytes: u64,
    },
    /// The staging cap is below one staging buffer.
    #[error("the staging cap ({cap} bytes) is below one staging buffer ({buffer} bytes)")]
    StagingCap {
        /// The cap.
        cap: u64,
        /// Bytes per staging buffer.
        buffer: u64,
    },
    /// The request buffer holds no request, or more than [`MAX_REQUESTS`].
    #[error("the request buffer holds {0} requests: it must hold 1 to {MAX_REQUESTS}")]
    Requests(u32),
    /// A slot buffer can't hold one tile of the largest class.
    #[error("slot buffers of {bytes} bytes can't hold one tile of {tile} bytes")]
    BufferTooSmall {
        /// The bytes per slot buffer.
        bytes: u64,
        /// The bytes of the largest class's tile.
        tile: u64,
    },
    /// The budget allows more resident tiles than the tile tree can index, or bind.
    #[error("the budget allows {tiles} resident tiles, more than a tile tree can hold here")]
    TooManyTiles {
        /// The tiles the budget allows.
        tiles: u64,
    },
    /// The most slot buffers the budget allows, every class together, don't fit one draw's
    /// bindings beside the raster's own.
    #[error("the budget allows {buffers} slot buffers; one draw binds at most {limit} of them")]
    TooManySlotBuffers {
        /// The most slot buffers, every class together.
        buffers: u32,
        /// The most one draw binds here.
        limit: u32,
    },
    /// The budget can't hold the floor: some of its tiles would never get a slot, and the floor
    /// is pinned and wanted until it is all resident.
    #[error("the budget holds {slots} tiles of side {side}, but the floor has {tiles} of them")]
    FloorTooBig {
        /// The side of the tiles of the class that can't hold its floor tiles.
        side: u32,
        /// The floor tiles of that class.
        tiles: u64,
        /// The most of them the budget allows at once.
        slots: u64,
    },
}

/// The most slot buffers each size class can ever have: what the raster's pipeline is built to
/// bind, once ([`crate::raster::Raster::new`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SlotLayout([u16; CLASSES]);

impl SlotLayout {
    /// `buffers[c]` buffers of each class `c`.
    #[must_use]
    pub const fn new(buffers: [u16; CLASSES]) -> Self {
        Self(buffers)
    }

    /// The most buffers of `class`.
    #[must_use]
    pub fn buffers(&self, class: SizeClass) -> u16 {
        self.0.get(usize::from(class.get())).copied().unwrap_or(0)
    }

    /// Per class, its most buffers.
    #[must_use]
    pub const fn per_class(&self) -> &[u16; CLASSES] {
        &self.0
    }

    /// Every class's together.
    #[must_use]
    pub fn total(&self) -> u32 {
        self.0.iter().map(|&b| u32::from(b)).sum()
    }
}

/// The most requests a request buffer may hold.
pub const MAX_REQUESTS: u32 = 1 << 20;

/// How residency is set up: built by [`ResidencyConfig::for_tier`] or [`ResidencyConfig::new`],
/// and changed through its checked `with_` methods, so every configuration is a valid one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidencyConfig {
    /// The most bytes the slot buffers may take together.
    budget_bytes: u64,
    /// Bytes per slot buffer, at most; `None` for as large as the device binds.
    buffer_bytes: Option<u64>,
    /// Bytes per staging buffer.
    staging_bytes: u64,
    /// The most bytes of staging buffers alive at once.
    staging_cap: u64,
    /// The largest epoch, in bytes of tile data, written into its tiles' own slots in the frame
    /// it is staged; a larger one goes through shadow slots.
    small_epoch_bytes: u64,
    /// The most bytes of pending epochs forced whole into one frame.
    forced_bytes: u64,
    /// Requests the request buffer holds per frame.
    request_capacity: u32,
    /// The finest level of the floor: every face's display tiles up to it are pinned.
    floor_top: Level,
}

/// A mebibyte.
const MIB: u64 = 1 << 20;

/// The floor's finest level by default: the spec's "levels ≤ 5".
const FLOOR_TOP: Level = match Level::new(5) {
    Ok(level) => level,
    Err(_) => Level::ZERO,
};

/// Frames a pending epoch may stall before it is forced whole past
/// [`ResidencyConfig::forced_bytes`]: so a tiny budget can't keep an epoch from ever applying.
pub const LONG_STALL: u32 = 8;

/// Keys readied at once by [`TileSource::prepare`]: the deadline is checked between such
/// batches, so a frame overruns it by at most one batch of tiles.
const PREPARE_BATCH: usize = 16;

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
            buffer_bytes: None,
            staging_bytes,
            staging_cap: 4 * staging_bytes,
            small_epoch_bytes: 8 * MIB,
            forced_bytes: 4 * staging_bytes,
            request_capacity: 4096,
            floor_top: FLOOR_TOP,
        }
    }

    /// A configuration: a budget of `budget_bytes`; staging buffers of `staging_bytes`, at
    /// most `staging_cap` of them alive; epochs up to `small_epoch_bytes` written in place,
    /// forced epochs up to the staging cap a frame; `request_capacity` requests a frame; the
    /// floor up to `floor_top`. Slot buffers grow to as large as the device binds.
    ///
    /// # Errors
    ///
    /// [`ConfigError::Unaligned`] for a size that isn't a positive multiple of
    /// [`MAP_ALIGNMENT`]; [`ConfigError::StagingCap`] for a cap below one staging buffer;
    /// [`ConfigError::Requests`] for a request capacity of 0 or above [`MAX_REQUESTS`].
    pub fn new(
        budget_bytes: u64,
        staging_bytes: u64,
        staging_cap: u64,
        small_epoch_bytes: u64,
        request_capacity: u32,
        floor_top: Level,
    ) -> Result<Self, ConfigError> {
        let config = Self {
            budget_bytes,
            buffer_bytes: None,
            staging_bytes,
            staging_cap,
            small_epoch_bytes,
            forced_bytes: staging_cap,
            request_capacity,
            floor_top,
        };
        config.check()?;
        Ok(config)
    }

    /// This configuration with a budget of `budget_bytes`.
    ///
    /// # Errors
    ///
    /// [`ConfigError::Unaligned`] if it isn't a positive multiple of [`MAP_ALIGNMENT`].
    pub fn with_budget(self, budget_bytes: u64) -> Result<Self, ConfigError> {
        let config = Self {
            budget_bytes,
            ..self
        };
        config.check()?;
        Ok(config)
    }

    /// This configuration with slot buffers of at most `buffer_bytes` (still no larger than
    /// the device binds).
    ///
    /// # Errors
    ///
    /// [`ConfigError::Unaligned`] if it isn't a positive multiple of [`MAP_ALIGNMENT`].
    pub fn with_buffer_bytes(self, buffer_bytes: u64) -> Result<Self, ConfigError> {
        let config = Self {
            buffer_bytes: Some(buffer_bytes),
            ..self
        };
        config.check()?;
        Ok(config)
    }

    /// This configuration forcing at most `forced_bytes` of pending epochs whole into a frame.
    ///
    /// # Errors
    ///
    /// [`ConfigError::Unaligned`] if it isn't a positive multiple of [`MAP_ALIGNMENT`].
    pub fn with_forced_bytes(self, forced_bytes: u64) -> Result<Self, ConfigError> {
        let config = Self {
            forced_bytes,
            ..self
        };
        config.check()?;
        Ok(config)
    }

    /// The most bytes the slot buffers may take together.
    #[must_use]
    pub const fn budget_bytes(&self) -> u64 {
        self.budget_bytes
    }

    /// The finest level of the floor.
    #[must_use]
    pub const fn floor_top(&self) -> Level {
        self.floor_top
    }

    /// The most bytes of pending epochs forced whole into one frame.
    #[must_use]
    pub const fn forced_bytes(&self) -> u64 {
        self.forced_bytes
    }

    /// Checks the sizes against each other.
    ///
    /// # Errors
    ///
    /// As [`ResidencyConfig::new`].
    fn check(&self) -> Result<(), ConfigError> {
        let sizes = [
            ("the budget", Some(self.budget_bytes)),
            ("a slot buffer", self.buffer_bytes),
            ("a staging buffer", Some(self.staging_bytes)),
            ("the staging cap", Some(self.staging_cap)),
            ("the small epoch size", Some(self.small_epoch_bytes)),
            ("the forced bytes", Some(self.forced_bytes)),
        ];
        let given = sizes
            .into_iter()
            .filter_map(|(what, bytes)| Some((what, bytes?)));
        for (what, bytes) in given {
            if bytes == 0 || !bytes.is_multiple_of(MAP_ALIGNMENT) {
                return Err(ConfigError::Unaligned { what, bytes });
            }
        }
        if self.staging_cap < self.staging_bytes {
            return Err(ConfigError::StagingCap {
                cap: self.staging_cap,
                buffer: self.staging_bytes,
            });
        }
        if self.request_capacity == 0 || self.request_capacity > MAX_REQUESTS {
            return Err(ConfigError::Requests(self.request_capacity));
        }
        Ok(())
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
    /// The tile tree is full: it is sized for every tile the budget allows, so this is a bug.
    #[error(transparent)]
    Tree(#[from] TreeError),
    /// The store couldn't give a tile.
    #[error("the store couldn't give a tile: {0}")]
    Store(#[from] CacheError),
    /// The configuration doesn't fit this project or device.
    #[error(transparent)]
    Config(#[from] ConfigError),
    /// Residency's own bookkeeping broke: a bug.
    #[error("residency bookkeeping broke: {0}")]
    Invariant(&'static str),
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
    /// Bytes of slot buffers made: every slot, handed out or not.
    pub allocated_bytes: u64,
    /// Bytes of the slots handed out: resident tiles and pending epochs' shadow copies. The
    /// floor's classes are made whole up front with twice their slots (headroom for a resync of
    /// the whole floor: 219 MiB on the benchmark project), so half of those stay allocated and
    /// unused.
    pub used_bytes: u64,
    /// Bytes the normal pass keeps between frames, outside the slot budget: its record chunks
    /// (as many as hold the staging ring's cap) and its job list.
    pub record_bytes: u64,
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
    /// Tiles staged into shadow slots, ever.
    pub shadowed: u64,
    /// Pending epochs forced whole into one frame, ever.
    pub forced: u64,
    /// Frames that failed to stage or were dropped unsubmitted, ever.
    pub lost_frames: u64,
    /// Resyncs, ever.
    pub resyncs: u64,
    /// Bytes the last frame staged.
    pub staged_bytes: u64,
    /// Tiles the last frame staged.
    pub staged_tiles: u64,
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

/// A tile requested.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Want {
    /// Its screen error: larger is staged first.
    error: f32,
    /// The frame it was first requested in: it is staged from the next one on (a display tile
    /// whose data is at hand, from this one).
    since: u64,
}

/// A pending tile's new data.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Staged {
    /// Uploaded into `slot` (a shadow slot, or the tile's own for an epoch written in place),
    /// with its bound `e` and finest normals' cone.
    Data {
        /// Where it is.
        slot: Slot,
        /// Its bound `e` and finest normals' cone.
        detail: TileDetail,
    },
    /// The source had nothing for it: it leaves when the epoch switches.
    Missing,
}

/// One or more epochs pending together.
#[derive(Debug, Default)]
struct Group {
    /// Every tile its epochs named, resident or not: a tile arriving that is one of them joins.
    keys: BTreeSet<TileKey>,
    /// Whether it names every tile: a resync's, which every arrival joins.
    all: bool,
    /// Members whose new data isn't staged yet.
    todo: BTreeSet<TileKey>,
    /// Members whose new data is staged.
    staged: BTreeMap<TileKey, Staged>,
    /// Frames it found no shadow slot to be had.
    stalled: u32,
    /// Whether a small epoch merged into it: it goes whole in its next frame, so the small
    /// one shows.
    urgent: bool,
}

impl Group {
    /// Whether `key` is one of its members (pending, staged or not).
    fn holds(&self, key: &TileKey) -> bool {
        self.todo.contains(key) || self.staged.contains_key(key)
    }

    /// Whether a tile arriving as `key` joins it.
    fn names(&self, key: &TileKey) -> bool {
        self.all || self.keys.contains(key)
    }

    /// Takes `other`'s keys and members.
    fn absorb(&mut self, other: Self) {
        self.keys.extend(other.keys);
        self.all |= other.all;
        self.todo.extend(other.todo);
        self.staged.extend(other.staged);
        self.stalled = self.stalled.max(other.stalled);
        self.urgent |= other.urgent;
    }
}

/// What reading one tile into staging came to.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Read {
    /// Staged, with its bound `e` and finest normals' cone.
    Staged(TileDetail),
    /// The source had nothing for it.
    Missing,
    /// The staging ring had no room.
    NoRoom,
    /// No slot of its class could be had.
    NoSlot,
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
    /// Per face, the valence of the base vertex at each of its corners, by
    /// [`corner_index`]: what a finest tile touching it uploads of its ring.
    corner_valences: Vec<[u32; 4]>,
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
    /// How many of them are resident: so a frame with the whole floor up skips scanning it.
    floor_resident: usize,
    /// How many of them the source had nothing for (they are in `unavailable`): with
    /// `floor_resident`, whether the floor has anything left to look for.
    floor_unavailable: usize,
    /// Tiles requested.
    wanted: BTreeMap<TileKey, Want>,
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
    /// The places with a descendant in the cut or drawn: every ancestor of either.
    shielded: BTreeSet<Place>,
    /// Whether `shielded` came from the cut alone (every tile drawn was in it).
    shielded_from_cut: bool,
    /// The tiles the frame being staged wrote, with the slot each was written to.
    touched: Vec<(TileKey, Slot)>,
    /// The same tiles, as a set: never evicted by the frame that wrote them.
    fresh: BTreeSet<TileKey>,
    /// Bytes of pending epochs forced into the frame being staged.
    forced_now: u64,
    /// Per class, where the frame's last eviction scan stopped: the tiles before it may not
    /// leave this frame, so the next scan starts past them.
    scanned: Vec<Option<(u64, TileKey)>>,
    /// Per frame staged and not yet heard of as submitted, the tiles it wrote.
    in_flight: BTreeMap<u64, Vec<(TileKey, Slot)>>,
    /// The next staged frame's number.
    next_upload: u64,
    /// Given to each frame's uploads, to tell what became of them.
    outcome_sender: Sender<FrameOutcome>,
    /// Where frames' outcomes arrive.
    outcomes: Receiver<FrameOutcome>,
    /// The staging ring.
    ring: StagingRing,
    /// The request buffer and its readbacks.
    requests: RequestBuffer,
    /// The most slot buffers each class can have.
    layout: SlotLayout,
    /// Counts.
    stats: ResidencyStats,
    /// A tile as read from the source: its vertices, or a finest tile's ring.
    data: TileData,
    /// The normal pass: the frame's finest tiles' records, and the work that turns them into
    /// their slots' vertices.
    normals: NormalPass,
}

impl Residency {
    /// Residency for a project of `mesh` whose finest level is `finest`, with nothing resident
    /// yet: the floor's slot buffers are made now (with their shadow headroom), and the floor is
    /// staged by the first [`Residency::stage`], whole.
    ///
    /// Slot buffers grow to as large as the device binds, or the configuration's smaller size;
    /// staging buffers hold at least one tile of the finest class.
    ///
    /// # Errors
    ///
    /// [`ResidencyError::Config`] if a slot buffer can't hold a tile of the finest class, or
    /// the budget allows more resident tiles than the tile tree can index or the device bind, or
    /// can't hold the floor ([`ConfigError::FloorTooBig`]: its tiles would be wanted forever), or
    /// allows more slot buffers than the raster's one draw binds
    /// ([`ConfigError::TooManySlotBuffers`]); [`ResidencyError::Gpu`] or
    /// [`ResidencyError::Tree`] if a buffer can't be created.
    pub fn new(
        gpu: &Gpu,
        mesh: &BaseMesh,
        finest: Level,
        config: ResidencyConfig,
    ) -> Result<Self, ResidencyError> {
        let limits = gpu.device().limits();
        let binding = limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size);
        let buffer_bytes = config.buffer_bytes.map_or(binding, |b| b.min(binding));
        let largest = SizeClass::of(finest).slot_bytes();
        if buffer_bytes < largest {
            return Err(ConfigError::BufferTooSmall {
                bytes: buffer_bytes,
                tile: largest,
            }
            .into());
        }
        let faces = mesh.face_count();
        let mut pools = Pools::new(faces, finest, buffer_bytes, config.budget_bytes);
        let classes = (0..CLASSES).filter_map(|c| u8::try_from(c).ok().and_then(SizeClass::new));
        let mut most = [0_u16; CLASSES];
        for (slot, class) in most.iter_mut().zip(classes) {
            *slot = pools.most_buffers(class);
        }
        let layout = SlotLayout::new(most);
        let limit = crate::raster::slot_buffer_limit(gpu);
        if layout.total() > limit {
            return Err(ConfigError::TooManySlotBuffers {
                buffers: layout.total(),
                limit,
            }
            .into());
        }
        let tiles = pools.max_resident();
        let capacity = u32::try_from(tiles)
            .ok()
            .filter(|&nodes| u64::from(nodes) * NODE_BYTES <= binding)
            .ok_or(ConfigError::TooManyTiles { tiles })?;
        let tree = TileTree::new(gpu, mesh, capacity)?;
        let requests = RequestBuffer::new(gpu, config.request_capacity)?;
        let floor_levels = (0..=config.floor_top.get())
            .filter_map(|l| Level::new(l).ok())
            .filter(|&level| level < finest);
        let floor_levels: Vec<Level> = floor_levels.collect();
        for &level in &floor_levels {
            let class = SizeClass::of(level);
            // A class the floor fills alone (every class but the last): its tiles all pinned,
            // and tiny.
            if usize::from(class.get()) + 1 < CLASSES {
                pools.reserve(gpu, class)?;
            }
        }
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
        check_floor_fits(&pools, &floor)?;
        let staging_bytes = config.staging_bytes.max(largest);
        let staging_cap = config.staging_cap.max(staging_bytes);
        let (outcome_sender, outcomes) = mpsc::channel();
        // A record chunk holds what a staging buffer does; as many are kept as hold the ring.
        let normals = NormalPass::new(gpu, staging_bytes, staging_cap)?;
        Ok(Self {
            ring: StagingRing::new(staging_bytes, staging_cap),
            config,
            finest,
            faces,
            corner_valences: corner_valences(mesh, finest),
            pools,
            tree,
            entries: BTreeMap::new(),
            recency: vec![BTreeSet::new(); CLASSES],
            floor,
            floor_resident: 0,
            floor_unavailable: 0,
            wanted: BTreeMap::new(),
            unavailable: BTreeSet::new(),
            groups: Vec::new(),
            pending: BTreeSet::new(),
            frame: 0,
            cut: BTreeSet::new(),
            shielded: BTreeSet::new(),
            shielded_from_cut: true,
            touched: Vec::new(),
            fresh: BTreeSet::new(),
            forced_now: 0,
            scanned: vec![None; CLASSES],
            in_flight: BTreeMap::new(),
            next_upload: 0,
            outcome_sender,
            outcomes,
            requests,
            layout,
            stats: ResidencyStats::default(),
            data: TileData::default(),
            normals,
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
    /// [`GpuVertex`](crate::upload::GpuVertex)es. A class gains buffers as tiles arrive (rebind
    /// when the count changes); a buffer, once made, stays.
    #[must_use]
    pub fn slot_buffers(&self, class: SizeClass) -> &[Buffer] {
        self.pools.buffers(class)
    }

    /// The most slot buffers each class can ever have, which the raster is built to bind.
    #[must_use]
    pub const fn slot_layout(&self) -> SlotLayout {
        self.layout
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

    /// The cut the last [`Residency::use_frame`] gave (empty before the first): what a newer
    /// frame's cut is compared with, to tell whether it changed.
    #[must_use]
    pub const fn cut(&self) -> &BTreeSet<TileKey> {
        &self.cut
    }

    /// The floor: every face's display tiles up to [`ResidencyConfig::floor_top`], pinned.
    #[must_use]
    pub const fn floor(&self) -> &BTreeSet<TileKey> {
        &self.floor
    }

    /// Whether every floor tile the source could give is resident: true after the first
    /// [`Residency::stage`].
    #[must_use]
    pub fn floor_ready(&self) -> bool {
        self.floor_settled()
    }

    /// Counts, as they stand.
    #[must_use]
    pub fn stats(&self) -> ResidencyStats {
        ResidencyStats {
            resident: self.entries.len(),
            floor: self.floor_resident,
            allocated_bytes: self.pools.allocated(),
            used_bytes: self.pools.used(),
            record_bytes: self.normals.kept_bytes(),
            wanted: self.wanted.len(),
            unavailable: self.unavailable.len(),
            pending_epochs: self.groups.len(),
            ..self.stats
        }
    }

    /// Starts a frame: polls the device without waiting, takes back the staging buffers the
    /// GPU is done with, reads the request readbacks that completed, and hears which staged
    /// frames were submitted or lost (a lost frame's tiles are dropped).
    ///
    /// # Errors
    ///
    /// [`ResidencyError::Gpu`] if polling fails or a readback can't be read.
    pub fn begin_frame(&mut self, gpu: &Gpu) -> Result<(), ResidencyError> {
        self.frame += 1;
        gpu.device().poll(PollType::Poll).map_err(GpuError::from)?;
        self.ring.collect();
        self.requests.collect(self.finest)?;
        while let Ok(outcome) = self.outcomes.try_recv() {
            match outcome {
                FrameOutcome::Submitted(frame) => {
                    self.in_flight.remove(&frame);
                }
                FrameOutcome::Lost(frame) => {
                    if let Some(touched) = self.in_flight.remove(&frame) {
                        self.recover(&touched, false);
                    }
                }
            }
        }
        Ok(())
    }

    /// The frame number: one more each [`Residency::begin_frame`], so the current frame's
    /// after it. A frame's request readback carries it ([`Residency::requests_heard`]).
    #[must_use]
    pub const fn frame(&self) -> u64 {
        self.frame
    }

    /// The newest frame ([`Residency::frame`]) whose request readback has been read, if any
    /// has: the level choice's requests as of that frame's uploads are in hand. A frame dropped
    /// unsubmitted never makes one.
    #[must_use]
    pub const fn requests_heard(&self) -> Option<u64> {
        self.requests.heard()
    }

    /// The requests the latest request readbacks gave (merged, if several arrived), if any
    /// arrived since the last call: for [`crate::upload::prefetch_requests`], then
    /// [`Residency::want`].
    pub const fn take_requests(&mut self) -> Option<Requests> {
        self.requests.take()
    }

    /// Wants the tiles of `requests` resident, larger screen error first: they replace the
    /// requests wanted before (the floor stays wanted until resident). A tile is staged from
    /// the frame after it was first requested on, so a page-in or a read ahead started now
    /// ([`crate::upload::prefetch_requests`]) has a frame to land; a display tile whose data the
    /// source has at hand ([`TileSource::at_hand`]) from this frame on. Resident tiles, tiles of no
    /// face of the mesh or below its finest level, and tiles the source had nothing for are
    /// skipped.
    pub fn want(&mut self, requests: &Requests) {
        let old = std::mem::take(&mut self.wanted);
        let (faces, finest, frame) = (self.faces, self.finest, self.frame);
        let floor = old
            .iter()
            .filter(|(key, _)| self.floor.contains(key))
            .map(|(key, want)| (*key, *want));
        let floor: Vec<(TileKey, Want)> = floor.collect();
        let fresh = requests.iter().filter(|request| {
            request.key.face().index() < faces
                && request.key.level() <= finest
                && !self.entries.contains_key(&request.key)
                && !self.unavailable.contains(&request.key)
        });
        let fresh = fresh.map(|request| {
            let since = old.get(&request.key).map_or(frame, |want| want.since);
            let want = Want {
                error: request.screen_error,
                since,
            };
            (request.key, want)
        });
        let fresh: Vec<(TileKey, Want)> = fresh.collect();
        self.wanted.extend(floor);
        self.wanted.extend(fresh);
    }

    /// Takes what the level choice used this frame: `frame.drawn`'s tiles are renewed as most
    /// recently drawn, and until the next call `frame.cut`'s tiles and every ancestor of
    /// either's are never evicted.
    ///
    /// The ancestors are the refined set the cut hangs from (and the tiles the relaxation and
    /// its sweep refined beside it): were one evicted, everything under it would be
    /// unreachable and drawn by the next ancestor up, two or more levels coarser.
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
        // The drawn tiles are the cut's after culling: with the cut as it was, and the shielded
        // set made from it alone, the set is the same, and isn't made again.
        let cut_only = drawn.is_subset(&cut);
        if cut_only && self.shielded_from_cut && cut == self.cut {
            return;
        }
        self.shielded_from_cut = cut_only;
        let mut shielded = BTreeSet::new();
        for key in cut.iter().chain(&drawn) {
            let mut above = Place::of(key).parent();
            while let Some(place) = above {
                // Already in: so are all its ancestors.
                if !shielded.insert(place) {
                    break;
                }
                above = place.parent();
            }
        }
        self.shielded = shielded;
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
    /// member with it. A member already staged for an earlier epoch is staged again. A small
    /// epoch merged into a pending one makes it urgent: it goes whole in the next frame.
    fn apply_epoch(&mut self, epoch: Epoch) {
        let keys = epoch.keys;
        for key in &keys {
            if self.forget_unavailable(key) && self.floor.contains(key) {
                self.wanted.insert(
                    *key,
                    Want {
                        error: f32::INFINITY,
                        since: 0,
                    },
                );
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
        let bytes: u64 = live.iter().map(|key| self.upload_bytes(key)).sum();
        let mut merged = Group {
            urgent: !touching.is_empty() && bytes <= self.config.small_epoch_bytes,
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
    /// uses) one pending epoch, re-read whole, which every tile arriving meanwhile joins (an
    /// epoch the feed lost may name it).
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
        self.floor_unavailable = 0;
        let floor_retry = retry.into_iter().filter(|key| self.floor.contains(key));
        let floor_retry = floor_retry.map(|key| {
            let want = Want {
                error: f32::INFINITY,
                since: 0,
            };
            (key, want)
        });
        let floor_retry: Vec<(TileKey, Want)> = floor_retry.collect();
        self.wanted.extend(floor_retry);
        self.groups.push(Group {
            keys: kept.clone(),
            all: true,
            todo: kept,
            ..Group::default()
        });
    }

    /// Whether `key` must stay: the floor, the current cut, or a tile whose children are in
    /// the cut or drawn.
    fn protected(&self, key: &TileKey) -> bool {
        self.floor.contains(key)
            || self.cut.contains(key)
            || self.shielded.contains(&Place::of(key))
    }

    /// After a frame failed to stage (`staged_now`) or was dropped unsubmitted: drops every
    /// tile whose slot holds data the frame wrote (it never reached the GPU), unstages the
    /// shadow data it wrote, and marks the whole tile tree to be written again.
    ///
    /// A slot the frame wrote that nothing holds any more was given back since (a later frame
    /// evicted its tile), unless the frame failed while staging (`staged_now`): then it was
    /// taken and never handed to its tile, and is given back here.
    fn recover(&mut self, touched: &[(TileKey, Slot)], staged_now: bool) {
        self.stats.lost_frames += 1;
        for &(key, slot) in touched {
            let live_here = self
                .entries
                .get(&key)
                .is_some_and(|entry| entry.slot == slot);
            let mut unstaged = false;
            for group in &mut self.groups {
                if matches!(group.staged.get(&key), Some(Staged::Data { slot: s, .. }) if *s == slot)
                {
                    group.staged.remove(&key);
                    group.todo.insert(key);
                    unstaged = true;
                }
            }
            if live_here {
                let mut shadows = Vec::new();
                for group in &mut self.groups {
                    group.todo.remove(&key);
                    shadows.extend(group.staged.remove(&key));
                }
                for staged in shadows {
                    self.release_staged(&key, staged);
                }
                self.remove_live(&key);
            } else if unstaged || staged_now {
                self.pools.release(slot);
            }
        }
        // Members that aren't resident any more and were never staged wait to arrive anew.
        let entries = &self.entries;
        for group in &mut self.groups {
            group.todo.retain(|key| entries.contains_key(key));
        }
        self.tree.mark_all_dirty();
    }

    /// Stages the frame's uploads from `source`: the floor first if it isn't resident (whole,
    /// whatever the deadline), then the pending epochs oldest first, then requested tiles,
    /// largest screen error first, until `deadline` (overrun by at most one batch of tiles
    /// being readied) or until the staging ring is full; then the tile tree's changed nodes.
    /// Returns them for the render thread.
    ///
    /// # Errors
    ///
    /// [`ResidencyError::Gpu`] if a buffer can't be made or written, the source's errors, and
    /// [`ResidencyError::Tree`] or [`ResidencyError::Invariant`] for a bug. The frame is then
    /// lost: its staging buffers go back to the ring, the tiles it wrote are dropped (none
    /// reached the GPU), and the next frame writes the whole tile tree again.
    pub fn stage<S: TileSource>(
        &mut self,
        gpu: &Gpu,
        source: &mut S,
        deadline: Instant,
    ) -> Result<FrameUploads, ResidencyError>
    where
        S::Error: Into<ResidencyError>,
    {
        self.touched.clear();
        self.fresh.clear();
        self.forced_now = 0;
        self.scanned.fill(None);
        self.normals.begin();
        self.pending = self
            .groups
            .iter()
            .flat_map(|group| group.todo.iter().chain(group.staged.keys()))
            .copied()
            .collect();
        let mut batch = Batch::new(&self.ring);
        match self.stage_into(gpu, &mut batch, source, deadline) {
            Ok(()) => {
                let normals = match self.normals.finish(gpu, &mut batch, &mut self.ring) {
                    Ok(normals) => normals,
                    Err(error) => return Err(self.fail(batch, error.into())),
                };
                let requests = match self.requests.copy(gpu, self.frame) {
                    Ok(requests) => requests,
                    Err(error) => return Err(self.fail(batch, error.into())),
                };
                self.stats.staged_bytes = batch.bytes();
                self.stats.staged_tiles = self.touched.len() as u64;
                let frame = self.next_upload;
                self.next_upload += 1;
                self.in_flight
                    .insert(frame, std::mem::take(&mut self.touched));
                Ok(batch.finish(frame, normals, requests, self.outcome_sender.clone()))
            }
            Err(error) => Err(self.fail(batch, error)),
        }
    }

    /// Ends a frame that failed to stage: its batch goes (its buffers back to the ring) and
    /// the tiles it wrote are dropped. Returns `error`.
    fn fail(&mut self, batch: Batch, error: ResidencyError) -> ResidencyError {
        drop(batch);
        let touched = std::mem::take(&mut self.touched);
        self.recover(&touched, true);
        error
    }

    /// [`Residency::stage`]'s work, into `batch`.
    ///
    /// # Errors
    ///
    /// As [`Residency::stage`].
    fn stage_into<S: TileSource>(
        &mut self,
        gpu: &Gpu,
        batch: &mut Batch,
        source: &mut S,
        deadline: Instant,
    ) -> Result<(), ResidencyError>
    where
        S::Error: Into<ResidencyError>,
    {
        self.stage_floor(gpu, batch, source)?;
        let mut groups = std::mem::take(&mut self.groups);
        let mut at = 0;
        let mut failed = None;
        while at < groups.len() {
            let Some(group) = groups.get_mut(at) else {
                break;
            };
            match self.stage_group(gpu, batch, source, group, deadline) {
                Ok(true) => {
                    groups.remove(at);
                }
                Ok(false) => at += 1,
                Err(error) => {
                    failed = Some(error);
                    break;
                }
            }
        }
        // Groups made meanwhile (none: only arrivals make them) go after the older ones.
        groups.append(&mut self.groups);
        self.groups = groups;
        if let Some(error) = failed {
            return Err(error);
        }
        self.stage_wanted(gpu, batch, source, deadline)?;
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
        Ok(())
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
        // Nothing to look for while every floor tile is up or has nothing to give, as it is but
        // after a lost frame, a dropped tile or an epoch naming one: the scan is for those.
        if self.floor_settled() {
            return Ok(());
        }
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
            self.forced_now += self.upload_bytes(&key);
            if self.arrive(gpu, batch, source, key, true)? != Read::NoSlot {
                self.wanted.remove(&key);
            }
        }
        Ok(())
    }

    /// Stages the requested tiles, largest screen error first, readied a batch at a time,
    /// until `deadline` or the ring is full. Tiles first requested this frame wait for the
    /// next, but for display tiles the source has at hand ([`TileSource::at_hand`]): a finest
    /// tile's page-in, or a display record's read ahead, has that frame to land.
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
        let frame = self.frame;
        // A finest tile waits a frame for its page-in; a display tile only if its record isn't
        // at hand yet (asked of the source as its turn comes, so reads landing meanwhile count).
        let could = self
            .wanted
            .iter()
            .filter(|(key, want)| want.since < frame || key.channel() == Channel::DISPLAY);
        let mut by_error: Vec<(TileKey, f32, bool)> = could
            .map(|(k, want)| (*k, want.error, want.since < frame))
            .collect();
        by_error.sort_by(|a, b| b.1.total_cmp(&a.1));
        let waiting = by_error
            .iter()
            .filter(|(key, _, _)| !self.is_pending_arrival(key));
        let order: Vec<(TileKey, bool)> = waiting.map(|(key, _, due)| (*key, *due)).collect();
        for chunk in order.chunks(PREPARE_BATCH) {
            if Instant::now() >= deadline {
                break;
            }
            let mut ready = Vec::with_capacity(chunk.len());
            for &(key, due) in chunk {
                if self.entries.contains_key(&key) || self.unavailable.contains(&key) {
                    self.wanted.remove(&key);
                } else if due || source.at_hand(key) {
                    ready.push(key);
                }
            }
            let Some(first) = ready.first() else {
                continue;
            };
            if batch.room(&self.ring) < self.upload_bytes(first) {
                break;
            }
            source.prepare(&ready).map_err(Into::into)?;
            for key in ready {
                match self.arrive(gpu, batch, source, key, false)? {
                    Read::NoRoom => return Ok(()),
                    Read::NoSlot => {}
                    Read::Staged(_) | Read::Missing => {
                        self.wanted.remove(&key);
                    }
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
            return Ok(Read::NoSlot);
        };
        let read = self.read_fresh(gpu, batch, source, key, slot, forced)?;
        match read {
            Read::Staged(detail) => match self.joining(&key) {
                Some(group) => {
                    group.staged.insert(key, Staged::Data { slot, detail });
                }
                None => self.insert_live(key, slot, detail)?,
            },
            Read::Missing => {
                self.pools.release(slot);
                self.mark_unavailable(key);
            }
            Read::NoRoom | Read::NoSlot => self.pools.release(slot),
        }
        Ok(read)
    }

    /// The pending epoch a tile arriving now joins: the one naming it, the pending epochs
    /// naming it merged into one.
    fn joining(&mut self, key: &TileKey) -> Option<&mut Group> {
        let (naming, rest): (Vec<Group>, Vec<Group>) = std::mem::take(&mut self.groups)
            .into_iter()
            .partition(|group| group.names(key));
        self.groups = rest;
        let mut naming = naming.into_iter();
        let mut merged = naming.next()?;
        for group in naming {
            merged.absorb(group);
        }
        self.groups.push(merged);
        self.groups.last_mut()
    }

    /// Stages what it can of pending `group`, and switches it if it is complete (returning
    /// true): whole in place if small enough for this frame, or forced (stalled or urgent,
    /// within the frame's forced bytes); else into shadow slots, a batch at a time, until
    /// `deadline`. A stall is a frame that found no shadow slot to be had.
    ///
    /// # Errors
    ///
    /// As [`Residency::stage`].
    fn stage_group<S: TileSource>(
        &mut self,
        gpu: &Gpu,
        batch: &mut Batch,
        source: &mut S,
        group: &mut Group,
        deadline: Instant,
    ) -> Result<bool, ResidencyError>
    where
        S::Error: Into<ResidencyError>,
    {
        let todo: Vec<TileKey> = group.todo.iter().copied().collect();
        let remaining: u64 = todo.iter().map(|key| self.upload_bytes(key)).sum();
        let small =
            remaining <= self.config.small_epoch_bytes && remaining <= batch.room(&self.ring);
        let within = self.forced_now + remaining <= self.config.forced_bytes;
        let rescue = group.stalled >= LONG_STALL && self.forced_now == 0;
        let forced = !small && (group.stalled > 0 || group.urgent) && (within || rescue);
        if small || forced {
            if forced {
                self.stats.forced += 1;
                self.forced_now += remaining;
            }
            source.prepare(&todo).map_err(Into::into)?;
            self.stage_in_place(gpu, batch, source, group, &todo)?;
            self.switch(group)?;
            return Ok(true);
        }
        let mut no_slot = false;
        'chunks: for chunk in todo.chunks(PREPARE_BATCH) {
            if Instant::now() >= deadline {
                break;
            }
            source.prepare(chunk).map_err(Into::into)?;
            for &key in chunk {
                let Some(slot) = self.acquire(gpu, SizeClass::of(key.level()))? else {
                    no_slot = true;
                    continue;
                };
                match self.read_fresh(gpu, batch, source, key, slot, false)? {
                    Read::Staged(detail) => {
                        group.staged.insert(key, Staged::Data { slot, detail });
                        self.stats.shadowed += 1;
                    }
                    Read::Missing => {
                        self.pools.release(slot);
                        group.staged.insert(key, Staged::Missing);
                    }
                    Read::NoRoom | Read::NoSlot => {
                        self.pools.release(slot);
                        break 'chunks;
                    }
                }
                group.todo.remove(&key);
            }
        }
        if group.todo.is_empty() {
            self.switch(group)?;
            return Ok(true);
        }
        if no_slot {
            group.stalled += 1;
        }
        Ok(false)
    }

    /// Writes every member of `group` in `todo` in place (a resident tile into its own slot, an
    /// arriving one into a new slot), past the ring's cap if need be: the group then switches
    /// in this frame.
    ///
    /// # Errors
    ///
    /// As [`Residency::stage`]; [`ResidencyError::Invariant`] if a forced write found no room.
    fn stage_in_place<S: TileSource>(
        &mut self,
        gpu: &Gpu,
        batch: &mut Batch,
        source: &mut S,
        group: &mut Group,
        todo: &[TileKey],
    ) -> Result<(), ResidencyError>
    where
        S::Error: Into<ResidencyError>,
    {
        for &key in todo {
            let own = self.entries.get(&key).map(|entry| entry.slot);
            let target = if let Some(slot) = own {
                slot
            } else {
                let fresh = self.acquire(gpu, SizeClass::of(key.level()))?;
                // An arrival with no slot to be had isn't drawn yet: it waits to be requested
                // again.
                let Some(slot) = fresh else {
                    group.todo.remove(&key);
                    continue;
                };
                slot
            };
            // Out of the todo only once read: a read that fails leaves it there, to be staged
            // again (and the tiles this frame wrote are dropped).
            let read = if own.is_some() {
                self.read_into(gpu, batch, source, key, target, true)?
            } else {
                self.read_fresh(gpu, batch, source, key, target, true)?
            };
            group.todo.remove(&key);
            match read {
                Read::Staged(detail) => {
                    group.staged.insert(
                        key,
                        Staged::Data {
                            slot: target,
                            detail,
                        },
                    );
                }
                Read::Missing => {
                    if own.is_none() {
                        self.pools.release(target);
                    }
                    group.staged.insert(key, Staged::Missing);
                }
                Read::NoRoom | Read::NoSlot => {
                    return Err(ResidencyError::Invariant("a forced write found no room"));
                }
            }
        }
        Ok(())
    }

    /// Switches a complete pending epoch: every staged member takes its new slot (giving its
    /// old one back) or arrives, and every member the source had nothing for leaves.
    ///
    /// # Errors
    ///
    /// [`ResidencyError::Tree`] if an arriving tile finds the tree full (it is sized for every
    /// tile the budget allows, so it doesn't).
    fn switch(&mut self, group: &mut Group) -> Result<(), ResidencyError> {
        for (key, staged) in std::mem::take(&mut group.staged) {
            let place = Place::of(&key);
            match staged {
                Staged::Data { slot, detail } => {
                    if let Some(entry) = self.entries.get_mut(&key) {
                        let old = entry.slot;
                        entry.slot = slot;
                        self.tree.move_to(place, slot, detail);
                        if old != slot {
                            self.pools.release(old);
                        }
                    } else {
                        self.insert_live(key, slot, detail)?;
                    }
                }
                Staged::Missing => {
                    if self.entries.contains_key(&key) {
                        self.remove_live(&key);
                        self.stats.dropped += 1;
                    }
                    self.mark_unavailable(key);
                }
            }
        }
        Ok(())
    }

    /// [`Residency::read_into`] for a slot just taken: given back if the read fails.
    ///
    /// # Errors
    ///
    /// As [`Residency::read_into`].
    fn read_fresh<S: TileSource>(
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
        let read = self.read_into(gpu, batch, source, key, slot, forced);
        if read.is_err() {
            self.pools.release(slot);
        }
        read
    }

    /// Reads `key` from `source` and stages it for `slot`. With `forced`, past the ring's cap.
    ///
    /// # Errors
    ///
    /// As [`Residency::stage`]; [`ResidencyError::Invariant`] if `slot` has no buffer.
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
        let bytes = self.upload_bytes(&key);
        if !forced && batch.room(&self.ring) < bytes {
            return Ok(Read::NoRoom);
        }
        let vertices = usize::try_from(slot.class.vertices()).unwrap_or(usize::MAX);
        self.data.reset(vertices);
        let Some(detail) = source.read(key, &mut self.data).map_err(Into::into)? else {
            return Ok(Read::Missing);
        };
        let (buffer, offset) = self
            .pools
            .locate(slot)
            .ok_or(ResidencyError::Invariant("a slot without a buffer"))?;
        let written = match self.data.upload() {
            Upload::Ring(record) => {
                // A finest tile's positions with its ring: the normal pass writes the slot.
                if record.quads() + 1 != slot.class.side() {
                    return Err(ResidencyError::Invariant(
                        "a ring of another side than its slot",
                    ));
                }
                self.normals.stage(
                    gpu,
                    batch,
                    &mut self.ring,
                    record,
                    slot,
                    (buffer, offset),
                    forced,
                )?
            }
            Upload::Vertices(data) => {
                let written = batch.write(gpu, &mut self.ring, data, buffer, offset, forced)?;
                if written {
                    // Copied in, not computed: an earlier record for the slot mustn't overwrite
                    // it.
                    self.normals.forget(slot);
                }
                written
            }
        };
        if written {
            self.touched.push((key, slot));
            self.fresh.insert(key);
            Ok(Read::Staged(detail))
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
        let index = usize::from(class.0);
        let after = self.scanned.get(index).copied().flatten();
        let from = after.map_or(Bound::Unbounded, Bound::Excluded);
        let candidates = self
            .recency
            .get(index)
            .into_iter()
            .flat_map(|recency| recency.range((from, Bound::Unbounded)));
        let victim = candidates
            .filter(|(_, key)| {
                !self.protected(key) && !self.pending.contains(key) && !self.fresh.contains(key)
            })
            .find(|(_, key)| !self.groups.iter().any(|group| group.holds(key)))
            .copied();
        let Some(victim) = victim else {
            return Ok(None);
        };
        // Every tile older than the victim stays protected for the rest of the frame (what
        // protects a tile is fixed while it is staged), so the next scan starts past it.
        if let Some(scanned) = self.scanned.get_mut(index) {
            *scanned = Some(victim);
        }
        let (_, victim) = victim;
        self.remove_live(&victim);
        self.stats.evictions += 1;
        Ok(self.pools.acquire(gpu, class)?)
    }

    /// Makes `key` resident in `slot` with its bound `e` and finest normals' cone (`detail`).
    ///
    /// # Errors
    ///
    /// [`ResidencyError::Tree`] if the tree is full.
    fn insert_live(
        &mut self,
        key: TileKey,
        slot: Slot,
        detail: TileDetail,
    ) -> Result<(), ResidencyError> {
        let floor = self.floor.contains(&key);
        let mut flags = 0;
        if floor {
            flags |= NODE_FLOOR;
        }
        if key.level() == self.finest {
            flags |= NODE_FINEST;
        }
        self.tree.insert(Place::of(&key), slot, detail, flags)?;
        let entry = Entry {
            slot,
            last_drawn: self.frame,
            floor,
        };
        // Only a tile not resident arrives, so this counts each floor tile once.
        if self.entries.insert(key, entry).is_none() && floor {
            self.floor_resident += 1;
        }
        if !floor && let Some(recency) = self.recency.get_mut(usize::from(slot.class.0)) {
            recency.insert((self.frame, key));
        }
        Ok(())
    }

    /// Whether every floor tile is resident or one the source had nothing for: what
    /// [`Residency::floor_ready`] says, from the counts (a tile is never both).
    fn floor_settled(&self) -> bool {
        self.floor_resident + self.floor_unavailable == self.floor.len()
    }

    /// Notes that the source had nothing for `key`, counting it if it is a floor tile.
    fn mark_unavailable(&mut self, key: TileKey) {
        if self.unavailable.insert(key) && self.floor.contains(&key) {
            self.floor_unavailable += 1;
        }
    }

    /// Forgets that the source had nothing for `key` (an epoch named it); whether it was so.
    fn forget_unavailable(&mut self, key: &TileKey) -> bool {
        let was = self.unavailable.remove(key);
        if was && self.floor.contains(key) {
            self.floor_unavailable -= 1;
        }
        was
    }

    /// Makes `key` not resident, giving back its slot.
    fn remove_live(&mut self, key: &TileKey) {
        let Some(entry) = self.entries.remove(key) else {
            return;
        };
        if entry.floor {
            self.floor_resident -= 1;
        }
        if let Some(recency) = self.recency.get_mut(usize::from(entry.slot.class.0)) {
            recency.remove(&(entry.last_drawn, *key));
        }
        self.tree.remove(Place::of(key));
        self.pools.release(entry.slot);
    }

    /// The bytes staging `key` takes: a finest tile's ring record and its job (exact for one
    /// read with its ring, as [`crate::upload::StoreSource`] reads them), any other tile's slot.
    ///
    /// What the staging ring's room, the forced bytes and the small-epoch size are counted in.
    /// A source writing a finest tile's vertices instead stages its slot, more: its write then
    /// finds no room where the count said there was, and the tile waits for the next frame.
    pub(crate) fn upload_bytes(&self, key: &TileKey) -> u64 {
        let class = SizeClass::of(key.level());
        if key.level() != self.finest || key.channel() == Channel::DISPLAY {
            return class.slot_bytes();
        }
        let q = class.side() - 1;
        let n = self.finest.face_quads();
        let (x0, y0) = (u32::from(key.tx()) * q, u32::from(key.ty()) * q);
        let corners = [(x0, y0), (x0 + q, y0), (x0 + q, y0 + q), (x0, y0 + q)];
        let valences = self
            .corner_valences
            .get(usize::try_from(key.face().index()).unwrap_or(usize::MAX));
        let ring_points: u32 = corners
            .into_iter()
            .filter_map(|(fi, fj)| corner_index(fi, fj, n))
            .filter_map(|c| valences.and_then(|v| v.get(c)).copied())
            .sum();
        crate::normals::record_bytes(q, ring_points) + crate::normals::JOB_BYTES
    }
}

/// Which corner of its face face-grid vertex `(fi, fj)` is, `n` quads a side, if it is one:
/// `(0, 0)`, `(n, 0)`, `(0, n)`, `(n, n)` are 0 to 3.
fn corner_index(fi: u32, fj: u32, n: u32) -> Option<usize> {
    let on = |x: u32| x == 0 || x == n;
    (on(fi) && on(fj)).then(|| usize::from(fi == n) + 2 * usize::from(fj == n))
}

/// Per face of `mesh`, the valence of the base vertex at each corner (by [`corner_index`]): the
/// faces holding a copy of it at `level`.
fn corner_valences(mesh: &BaseMesh, level: Level) -> Vec<[u32; 4]> {
    let n = level.face_quads();
    let topology = mesh.topology();
    let per_face = mesh.faces().map(|face| {
        [(0, 0), (n, 0), (0, n), (n, n)].map(|(fi, fj)| {
            let faces: BTreeSet<u32> = topology
                .copies(face, level, fi, fj)
                .iter()
                .map(|copy| copy.face.index())
                .collect();
            u32::try_from(faces.len()).unwrap_or(u32::MAX)
        })
    });
    per_face.collect()
}

/// Checks that `pools` can hold every tile of `floor` at once.
///
/// # Errors
///
/// [`ConfigError::FloorTooBig`] for the first size class that can't.
fn check_floor_fits(pools: &Pools, floor: &BTreeSet<TileKey>) -> Result<(), ConfigError> {
    let mut per_class = [0_u64; CLASSES];
    for key in floor {
        if let Some(count) = per_class.get_mut(usize::from(SizeClass::of(key.level()).get())) {
            *count += 1;
        }
    }
    let classes = (0..CLASSES).filter_map(|c| u8::try_from(c).ok().and_then(SizeClass::new));
    let counted = classes.zip(per_class);
    for (class, tiles) in counted {
        let slots = pools.capacity(class);
        if tiles > slots {
            return Err(ConfigError::FloorTooBig {
                side: class.side(),
                tiles,
                slots,
            });
        }
    }
    Ok(())
}
