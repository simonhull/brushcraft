//! Project file, out-of-core tile paging, tile pool, cache budget and spatial index.
//!
//! No GPU or windowing code: everything here runs headless.
//!
//! Positional reads and writes use Unix's `FileExt` (`read_exact_at`, `write_all_at`) for now, so
//! the crate builds on Unix only. Windows needs `seek_read`/`seek_write` later.

mod bytes;
mod codec;
mod crc;
mod deps;
mod feed;
mod file;
#[cfg(test)]
mod fixture;
mod meta;
mod op;
mod pool;
mod prefetch;
mod record;
mod slot;
#[cfg(test)]
mod test_dir;
mod tiles;

pub use codec::CodecError;
pub use feed::{ChangeFeed, ConsumerId};
pub use file::{
    BaseFile, MAX_MESH_RAW, MAX_RAW, ProjectFile, StoreError, TileBytes, compact, compact_in_place,
};
pub use meta::{Meta, ProjectId, Role, SavePoint};
pub use op::{MAX_SEQ, OpId, RESERVED_SITE};
pub use tiles::{CacheError, Invariant, PinGuard, StoreConfig, TileEdit, TileStore};
