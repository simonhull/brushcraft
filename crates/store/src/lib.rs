//! Project file, out-of-core tile paging, tile pool, cache budget and spatial index.
//!
//! No GPU or windowing code: everything here runs headless.
//!
//! Positional reads and writes use Unix's `FileExt` (`read_exact_at`, `write_all_at`) for now, so
//! the crate builds on Unix only. Windows needs `seek_read`/`seek_write` later.

mod activity;
mod bounds;
mod bytes;
mod codec;
mod crc;
mod deps;
mod feed;
mod file;
#[cfg(test)]
mod fixture;
#[cfg(any(feature = "fuzzing", test))]
#[doc(hidden)]
pub mod fuzzing;
mod hash;
mod index;
mod kept;
mod meta;
mod op;
mod pool;
mod prefetch;
mod project;
mod record;
mod recovery;
mod shared_map;
mod slot;
mod stamp;
#[cfg(test)]
mod test_dir;
mod tile_bounds;
mod tiles;
mod workers;

pub use activity::Activity;
pub use codec::CodecError;
pub use feed::{ChangeFeed, ConsumerId};
pub use file::{
    FileError, MAX_MESH_RAW, MAX_RAW, ProjectFile, TileBytes, compact, compact_in_place,
};
pub use index::{
    Aabb, Bvh, Hit, IndexError, LoadStats, MAX_TILES, NormalCone, Ray, SpatialIndex, SphereQuery,
    TileBounds, WaveBounds,
};
pub use meta::{Meta, ProjectId, Role, SavePoint};
pub use op::{MAX_SEQ, OpId, RESERVED_SITE};
pub use project::{
    CandidateStatus, KeptRecovery, Project, ProjectConfig, ProjectError, Protection,
    ROLL_OVER_RETRY, RecoveryCandidate, SaveOutcome, default_recovery_dir, find_recoveries,
};
pub use recovery::{FLUSH_INTERVAL, RecoveryError, RecoveryWriter};
pub use tiles::{
    CacheError, Evaluation, Invariant, PinGuard, StoreConfig, StoreId, TileEdit, TileStore,
};
pub use workers::{ParseWorkersError, StartWorkersError, Workers};
