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
    expect(
        dead_code,
        reason = "used by the project file, added in the next commit"
    )
)]
mod codec;
mod crc;
mod op;
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "used by the project file, added in a later commit"
    )
)]
mod record;

pub use codec::CodecError;
pub use op::{MAX_SEQ, OpId};
