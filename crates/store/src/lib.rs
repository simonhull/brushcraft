//! Project file, out-of-core tile paging, tile pool, cache budget and spatial index.
//!
//! No GPU or windowing code: everything here runs headless.

mod bytes;
mod codec;
mod crc;
mod feed;
mod file;
mod op;
mod record;
#[cfg(test)]
mod test_dir;

pub use codec::CodecError;
pub use feed::{ChangeFeed, ConsumerId};
pub use file::{ProjectFile, StoreError, TileBytes, compact};
pub use op::{MAX_SEQ, OpId};
