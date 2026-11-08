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

/// `rows` of `W` bytes transposed into `W` byte planes: byte `j` of row `i` at `j × n + i`, for
/// `n` rows.
///
/// Rows of numbers that change little from one to the next become long runs of alike bytes,
/// which compress well (the stroke record's samples, the journal's dabs). The rows are taken as
/// they come, so a caller that makes them on the fly holds no copy of them. Every caller's rows
/// are of a record held to its limits, far below `usize::MAX` bytes; rows too many to count in
/// a `usize` would give no planes rather than panic. Rows past the count the iterator states are
/// left out, and rows it states but doesn't give are zero.
pub(crate) fn transpose<const W: usize>(rows: impl ExactSizeIterator<Item = [u8; W]>) -> Vec<u8> {
    let n = rows.len();
    let mut out = vec![0; n.checked_mul(W).unwrap_or(0)];
    let numbered = rows.enumerate();
    for (i, row) in numbered {
        let targets = out.iter_mut().skip(i).step_by(n.max(1)).zip(row);
        for (slot, byte) in targets {
            *slot = byte;
        }
    }
    out
}

/// Undoes [`transpose`]: the `count` rows of `W` bytes `planes` holds, made as they are taken;
/// `None` unless it is exactly `W × count` bytes.
pub(crate) fn untranspose<const W: usize>(
    planes: &[u8],
    count: usize,
) -> Option<impl ExactSizeIterator<Item = [u8; W]> + '_> {
    if planes.len() != count.checked_mul(W)? {
        return None;
    }
    let rows = (0..count).map(move |i| {
        let mut row = [0; W];
        let sources = row
            .iter_mut()
            .zip(planes.iter().skip(i).step_by(count.max(1)));
        for (slot, &byte) in sources {
            *slot = byte;
        }
        row
    });
    Some(rows)
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

    /// A little-endian `u128`.
    pub(crate) fn u128(&mut self) -> Option<u128> {
        self.array().map(u128::from_le_bytes)
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

/// Proofs, for every input, run by `cargo kani -p store`.
#[cfg(kani)]
mod proofs {
    use super::*;

    /// [`untranspose`] never panics or reads out of range, whatever the planes' length and the
    /// count, and gives back exactly the rows [`transpose`] was given.
    #[kani::proof]
    #[kani::unwind(9)]
    fn untranspose_is_in_range_and_undoes_transpose() {
        let planes: [u8; 8] = kani::any();
        let len: usize = kani::any();
        kani::assume(len <= planes.len());
        let count: usize = kani::any();
        let Some(prefix) = planes.get(..len) else {
            return;
        };
        if let Some(rows) = untranspose::<2>(prefix, count) {
            assert_eq!(len, 2 * count, "only exactly 2 × count bytes");
            assert_eq!(rows.len(), count, "count rows");
            let back = transpose(rows);
            assert_eq!(back.as_slice(), prefix, "transposed back, the same planes");
        }
    }

    /// Every key [`TileKey::new`] accepts reads back from its encoding.
    #[kani::proof]
    #[kani::unwind(13)]
    fn keys_round_trip() {
        let Ok(level) = Level::new(kani::any()) else {
            return;
        };
        let made = TileKey::new(
            LayerId::new(kani::any()),
            Channel::from_raw(kani::any()),
            FaceId::new(kani::any()),
            level,
            kani::any(),
            kani::any(),
        );
        let Ok(key) = made else {
            return;
        };
        let mut out = Vec::with_capacity(KEY_LEN);
        put_key(&mut out, &key);
        assert_eq!(out.len(), KEY_LEN, "fixed width");
        assert_eq!(Reader::new(&out).key(), Some(key), "the same key");
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

    #[test]
    fn rows_transpose_into_planes_and_back() {
        let rows: Vec<[u8; 3]> = vec![[1, 2, 3], [4, 5, 6]];
        let planes = transpose(rows.iter().copied());
        assert_eq!(planes, [1, 4, 2, 5, 3, 6], "byte j of row i at j × n + i");
        let back: Vec<[u8; 3]> = untranspose(&planes, 2).unwrap().collect();
        assert_eq!(back, rows, "and back");
        assert!(untranspose::<3>(&planes, 3).is_none(), "too few bytes");
        assert!(untranspose::<3>(&planes[..5], 2).is_none(), "a short plane");
        assert!(
            untranspose::<3>(&planes, usize::MAX).is_none(),
            "a count past usize"
        );
        assert!(transpose::<3>(std::iter::empty()).is_empty(), "no rows");
        assert_eq!(untranspose::<3>(&[], 0).unwrap().count(), 0, "no rows back");
        assert_eq!(
            untranspose::<0>(&[], 4).unwrap().count(),
            4,
            "rows of no bytes"
        );
    }
}
