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
mod display;
mod feed;
mod file;
#[cfg(test)]
mod fixture;
#[cfg(any(feature = "fuzzing", test))]
#[doc(hidden)]
pub mod fuzzing;
mod hash;
mod index;
mod jobs;
mod kept;
mod meta;
mod op;
mod pool;
mod prefetch;
mod project;
mod queued;
mod record;
mod recovery;
mod shared_map;
mod slot;
mod stale;
mod stamp;
mod stroke_record;
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
    TileBounds, TriangleHit, WaveBounds,
};
pub use jobs::JobsReport;
pub use meta::{Meta, ProjectId, Role, SavePoint};
pub use op::{MAX_SEQ, OpId, RESERVED_SITE};
pub use project::{
    CandidateStatus, KeptRecovery, Project, ProjectConfig, ProjectError, Protection,
    ROLL_OVER_RETRY, RecoveryCandidate, SaveAhead, SaveOutcome, default_recovery_dir,
    find_recoveries,
};
pub use queued::EvaluatedTile;
pub use recovery::{FLUSH_INTERVAL, RecoveryError, RecoveryWriter};
pub use stale::{DEFAULT_STALE_CAP, Drained, Epoch, EpochKind, StaleCursor, Unrefreshed};
pub use stroke_record::{
    MAX_STROKE_SAMPLES, MAX_STROKE_SETTINGS, PenSample, PenTilt, SAMPLE_BYTES, StrokeCamera,
    StrokeRecord, StrokeRecordError,
};
pub use tiles::{
    CacheError, DisplayPreview, Evaluation, Invariant, MadePenDown, ParentSlots, PenDownMaker,
    PenDownShape, PenDownTaker, PinGuard, Predictor, StoreConfig, StoreId, TakenShape, TileEdit,
    TileStore,
};
pub use workers::{ParseWorkersError, StartWorkersError, Workers};
