//! Little-endian reading and writing for the file format.

use geom::{Channel, FaceId, LayerId, Level, TileKey};

/// Encoded size of a [`TileKey`]: layer u16, channel u8, face u32, level u8, tx u16, ty u16.
pub(crate) const KEY_LEN: usize = 12;

/// Appends `key`'s [`KEY_LEN`]-byte encoding.
pub(crate) fn put_key(out: &mut Vec<u8>, key: &TileKey) {
    out.extend_from_slice(&key.layer().get().to_le_bytes());
    out.push(key.channel().raw());
    out.extend_from_slice(&key.face().index().to_le_bytes());
    out.push(key.level().get());
    out.extend_from_slice(&key.tx().to_le_bytes());
    out.extend_from_slice(&key.ty().to_le_bytes());
}

/// A cursor over a byte slice; every read returns `None` past the end.
#[derive(Debug, Clone)]
pub(crate) struct Reader<'a> {
    /// The bytes not read yet.
    rest: &'a [u8],
}

impl<'a> Reader<'a> {
    /// A reader at the start of `bytes`.
    pub(crate) const fn new(bytes: &'a [u8]) -> Self {
        Self { rest: bytes }
    }

    /// The next `n` bytes.
    pub(crate) fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let (head, rest) = self.rest.split_at_checked(n)?;
        self.rest = rest;
        Some(head)
    }

    /// The next `N` bytes as an array.
    fn array<const N: usize>(&mut self) -> Option<[u8; N]> {
        self.take(N)?.try_into().ok()
    }

    /// A byte.
    pub(crate) fn u8(&mut self) -> Option<u8> {
        self.array::<1>().map(|[b]| b)
    }

    /// A little-endian `u16`.
    pub(crate) fn u16(&mut self) -> Option<u16> {
        self.array().map(u16::from_le_bytes)
    }

    /// A little-endian `u32`.
    pub(crate) fn u32(&mut self) -> Option<u32> {
        self.array().map(u32::from_le_bytes)
    }

    /// A little-endian `u64`.
    pub(crate) fn u64(&mut self) -> Option<u64> {
        self.array().map(u64::from_le_bytes)
    }

    /// A little-endian `f32`, bit for bit.
    pub(crate) fn f32(&mut self) -> Option<f32> {
        self.array().map(f32::from_le_bytes)
    }

    /// Everything not yet read.
    pub(crate) const fn rest(&self) -> &'a [u8] {
        self.rest
    }

    /// A key, validated (level range, tile range).
    pub(crate) fn key(&mut self) -> Option<TileKey> {
        let layer = LayerId::new(self.u16()?);
        let channel = Channel::from_raw(self.u8()?);
        let face = FaceId::new(self.u32()?);
        let level = Level::new(self.u8()?).ok()?;
        let (tx, ty) = (self.u16()?, self.u16()?);
        TileKey::new(layer, channel, face, level, tx, ty).ok()
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use geom::{Channel, FaceId, LayerId, Level, TileKey};

    use super::*;

    #[test]
    fn keys_round_trip() {
        let key = TileKey::new(
            LayerId::new(3),
            Channel::from_raw(7),
            FaceId::new(2399),
            Level::new(9).unwrap(),
            7,
            5,
        )
        .unwrap();
        let mut out = Vec::new();
        put_key(&mut out, &key);
        assert_eq!(out.len(), KEY_LEN, "fixed width");
        assert_eq!(Reader::new(&out).key(), Some(key), "round trip");
        let mut bad = out.clone();
        bad[7] = 13; // level 13 > MAX_LEVEL
        assert_eq!(Reader::new(&bad).key(), None, "invalid level rejected");
    }
}
