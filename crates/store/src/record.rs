//! Record framing: `kind: u8, reserved: [u8; 3], len: u32, payload, crc32: u32` (little-endian),
//! the checksum covering every byte before it. Readers skip kinds they don't know.

use crate::crc::crc32;

/// Head bytes before the payload.
pub(crate) const HEAD: usize = 8;
/// Checksum bytes after the payload.
pub(crate) const TAIL: usize = 4;
/// Framing bytes around a payload: head and checksum.
const OVERHEAD: u64 = (HEAD + TAIL) as u64;
/// A tile record.
pub(crate) const KIND_TILE: u8 = 1;
/// An index record (a save point).
pub(crate) const KIND_INDEX: u8 = 2;
/// A meta record: the file's identity (see `meta`). Always the first record.
pub(crate) const KIND_META: u8 = 3;
/// A mesh record: the base mesh, compressed. Always the second record.
pub(crate) const KIND_MESH: u8 = 4;
/// A bounds record: one tile's bounds for the spatial index, or that they are unknown (see
/// `bounds`).
pub(crate) const KIND_BOUNDS: u8 = 5;

/// A record's head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Head {
    /// The record kind.
    pub(crate) kind: u8,
    /// Payload bytes.
    pub(crate) len: u32,
}

impl Head {
    /// The whole record's length: head, payload and checksum.
    pub(crate) fn total(self) -> u64 {
        u64::from(self.len) + OVERHEAD
    }
}

/// Frames `payload` as a record, or `None` if it is longer than `u32::MAX` bytes.
pub(crate) fn frame(kind: u8, payload: &[u8]) -> Option<Vec<u8>> {
    let len = u32::try_from(payload.len()).ok()?;
    let mut out = Vec::with_capacity(HEAD + payload.len() + TAIL);
    out.push(kind);
    out.extend_from_slice(&[0; 3]);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(payload);
    let crc = crc32(&out);
    out.extend_from_slice(&crc.to_le_bytes());
    Some(out)
}

/// Parses a head; `None` if its reserved bytes aren't zero.
pub(crate) const fn parse_head(bytes: [u8; HEAD]) -> Option<Head> {
    let [kind, r0, r1, r2, l0, l1, l2, l3] = bytes;
    if r0 == 0 && r1 == 0 && r2 == 0 {
        Some(Head {
            kind,
            len: u32::from_le_bytes([l0, l1, l2, l3]),
        })
    } else {
        None
    }
}

/// Checks a whole framed record: its head, its length and its checksum. Returns the kind and
/// payload.
pub(crate) fn verify(record: &[u8]) -> Option<(u8, &[u8])> {
    let (body, crc) = record.split_at_checked(record.len().checked_sub(TAIL)?)?;
    let stored = u32::from_le_bytes(crc.try_into().ok()?);
    if crc32(body) != stored {
        return None;
    }
    let (head, payload) = body.split_at_checked(HEAD)?;
    let head = parse_head(head.try_into().ok()?)?;
    (usize::try_from(head.len).ok()? == payload.len()).then_some((head.kind, payload))
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    #[test]
    fn framed_records_verify() {
        let rec = frame(KIND_TILE, b"hello").unwrap();
        assert_eq!(rec.len(), HEAD + 5 + TAIL, "framing overhead");
        assert_eq!(verify(&rec), Some((KIND_TILE, &b"hello"[..])), "round trip");
        let head = parse_head(rec[..HEAD].try_into().unwrap()).unwrap();
        assert_eq!((head.kind, head.len), (KIND_TILE, 5), "head");
        assert_eq!(
            head.total(),
            u64::try_from(rec.len()).unwrap(),
            "total length"
        );
    }

    #[test]
    fn any_flipped_bit_fails_verification() {
        let rec = frame(KIND_INDEX, &[1, 2, 3, 4, 5, 6]).unwrap();
        for byte in 0..rec.len() {
            for bit in 0..8 {
                let mut bad = rec.clone();
                bad[byte] ^= 1 << bit;
                assert_eq!(verify(&bad), None, "flip at byte {byte} bit {bit}");
            }
        }
    }

    #[test]
    fn nonzero_reserved_bytes_are_not_a_head() {
        let mut rec = frame(KIND_TILE, b"x").unwrap();
        rec[2] = 1;
        assert!(
            parse_head(rec[..HEAD].try_into().unwrap()).is_none(),
            "reserved must be zero"
        );
    }
}
