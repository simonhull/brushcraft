//! A fast hasher for the store's maps keyed by tile.
//!
//! The standard maps hash with `SipHash`, keyed at random, so that keys an attacker chooses can't
//! pile into one bucket. The store's tile keys are its own (tiles of its mesh, found by its
//! own walks), never chosen from outside, so its maps take a multiply-and-rotate hash instead,
//! which reads a [`TileKey`] in two steps (its [`Hash`](std::hash::Hash) writes two words).
//! Should tile keys ever come from untrusted input into a long-lived map, this choice must be
//! revisited.
//!
//! Nothing may depend on the order such a map iterates in: it differs from `SipHash`'s, and
//! nothing of it is part of what the store promises.

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasher, Hasher};

use geom::TileKey;

/// A map from tiles, with [`TileHash`].
pub(crate) type TileMap<V> = HashMap<TileKey, V, TileHash>;

/// A set of tiles, with [`TileHash`].
pub(crate) type TileSet = HashSet<TileKey, TileHash>;

/// Builds [`TileHasher`]s: the hash of the store's tile maps (see the module notes).
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct TileHash;

impl BuildHasher for TileHash {
    type Hasher = TileHasher;

    fn build_hasher(&self) -> TileHasher {
        TileHasher(0)
    }
}

/// A word-at-a-time hash: a rotate, an exclusive or and a multiply per word, and a rotate at
/// the end.
///
/// Each word is folded into the state with the first three, and the result is rotated so that
/// the multiply's best-mixed high bits land where the map's bucket index reads.
/// A multiply carries each bit of its input into every higher bit only, so the state's low
/// bits know little and its high bits know everything; the map takes its bucket from the low
/// bits of [`Hasher::finish`] and its tag from the top seven, so `finish` turns the top bits
/// down. Bytes written other than as words are taken eight to a word.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TileHasher(u64);

impl TileHasher {
    /// The multiplier: an odd constant with its bits spread (2⁶⁴ over the golden ratio, made
    /// odd), as in Fibonacci hashing.
    const K: u64 = 0x9E37_79B9_7F4A_7C15;

    /// Folds `word` into the state.
    const fn add(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(Self::K);
    }
}

impl Hasher for TileHasher {
    fn write(&mut self, bytes: &[u8]) {
        let (words, rest) = bytes.as_chunks::<8>();
        for &word in words {
            self.add(u64::from_le_bytes(word));
        }
        if !rest.is_empty() {
            let mut word = [0; 8];
            word.iter_mut().zip(rest).for_each(|(w, &b)| *w = b);
            self.add(u64::from_le_bytes(word));
        }
    }

    fn write_u8(&mut self, i: u8) {
        self.add(u64::from(i));
    }

    fn write_u16(&mut self, i: u16) {
        self.add(u64::from(i));
    }

    fn write_u32(&mut self, i: u32) {
        self.add(u64::from(i));
    }

    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }

    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }

    fn finish(&self) -> u64 {
        self.0.rotate_left(26)
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use std::collections::BTreeSet;

    use geom::{Channel, FaceId, LayerId, Level};

    use super::*;

    /// The hash of `key`.
    fn hash(key: &TileKey) -> u64 {
        TileHash.hash_one(key)
    }

    /// A key at level 9, the level a brush edits.
    fn key(layer: u16, channel: u8, face: u32, tx: u16, ty: u16) -> TileKey {
        TileKey::new(
            LayerId::new(layer),
            Channel::from_raw(channel),
            FaceId::new(face),
            Level::new(9).unwrap(),
            tx,
            ty,
        )
        .unwrap()
    }

    /// Keys that differ in any one field hash differently, for a sample of keys.
    #[test]
    fn keys_differing_in_one_field_hash_differently() {
        for (layer, channel, face, tx, ty) in [(0, 0, 0, 0, 0), (1, 2, 17, 3, 5), (7, 1, 999, 7, 6)]
        {
            let base = key(layer, channel, face, tx, ty);
            let others = [
                key(layer + 1, channel, face, tx, ty),
                key(layer, channel + 1, face, tx, ty),
                key(layer, channel, face + 1, tx, ty),
                key(layer, channel, face, tx ^ 1, ty),
                key(layer, channel, face, tx, ty ^ 1),
                TileKey::new(
                    base.layer(),
                    base.channel(),
                    base.face(),
                    Level::new(8).unwrap(),
                    tx % 4,
                    ty % 4,
                )
                .unwrap(),
            ];
            for other in others {
                assert_ne!(other, base);
                assert_ne!(hash(&other), hash(&base), "{other:?} and {base:?}");
            }
        }
    }

    /// The tiles of a brush-sized patch spread over a small table's buckets and tags.
    ///
    /// The 64 tiles fill a table of 64 buckets about as random hashes would (those fill 40 on
    /// average), and so do their tags (the top seven bits, 50 of 128 on average): the column and
    /// row reach the bits the map reads.
    #[test]
    fn a_patch_of_tiles_spreads_over_buckets() {
        let patch: Vec<TileKey> = (0..8)
            .flat_map(|ty| (0..8).map(move |tx| key(0, 0, 42, tx, ty)))
            .collect();
        let buckets: BTreeSet<u64> = patch.iter().map(|k| hash(k) % 64).collect();
        assert!(buckets.len() >= 32, "{} buckets of 64", buckets.len());
        let tags: BTreeSet<u64> = patch.iter().map(|k| hash(k) >> 57).collect();
        assert!(tags.len() >= 40, "{} tags of 128", tags.len());
    }

    /// Bytes written loose hash as the words they make, zero-padded.
    #[test]
    fn loose_bytes_hash_as_words() {
        let mut loose = TileHash.build_hasher();
        loose.write(&[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let mut words = TileHash.build_hasher();
        words.write_u64(u64::from_le_bytes([1, 2, 3, 4, 5, 6, 7, 8]));
        words.write_u64(9);
        assert_eq!(loose.finish(), words.finish());
    }
}
