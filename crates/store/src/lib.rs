//! Project file, out-of-core tile paging, tile pool, cache budget and spatial index.
//!
//! No GPU or windowing code: everything here runs headless.

#[expect(
    dead_code,
    reason = "scaffolding for the record and file modules, added in the next commits"
)]
mod bytes;
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "used by record framing, added in the next commit")
)]
mod crc;
mod op;

pub use op::{MAX_SEQ, OpId};
