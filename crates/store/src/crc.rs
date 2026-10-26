//! CRC-32 (IEEE 802.3: reflected, polynomial 0xEDB88320), the checksum on every record.

/// The lookup tables for slicing by eight: `TABLES[0]` is the classic byte-at-a-time table,
/// and `TABLES[k][b]` is the CRC register after feeding byte `b` followed by `k` zero bytes into
/// a zero register.
///
/// With them, eight bytes are folded into the register at once: each of the eight bytes (the
/// register's four, xored with the first four of the input, then the next four) is looked up in
/// the table for how many bytes still follow it, and the eight entries xored together. That is
/// the same register the byte loop reaches, as CRC is linear over GF(2), in a quarter of the
/// dependent steps.
const TABLES: [[u32; 256]; 8] = tables();

/// One reflected CRC step of a whole byte: the register after shifting `c` eight times with the
/// polynomial.
const fn byte_step(mut c: u32) -> u32 {
    let mut k = 0;
    while k < 8 {
        c = if c & 1 == 1 {
            0xEDB8_8320 ^ (c >> 1)
        } else {
            c >> 1
        };
        k += 1;
    }
    c
}

#[expect(
    clippy::indexing_slicing,
    reason = "const evaluation: k < 8 and i < 256 are checked by the loops and at compile time"
)]
/// The tables of [`TABLES`], built at compile time: the byte table first, then each further
/// table from the one before it by one more zero byte.
const fn tables() -> [[u32; 256]; 8] {
    let mut tables = [[0u32; 256]; 8];
    let mut i = 0;
    while i < 256 {
        #[expect(clippy::cast_possible_truncation, reason = "i < 256")]
        let byte = i as u32;
        tables[0][i] = byte_step(byte);
        i += 1;
    }
    let mut k = 1;
    while k < 8 {
        let mut b = 0;
        while b < 256 {
            let previous = tables[k - 1][b];
            tables[k][b] = (previous >> 8) ^ tables[0][(previous & 0xFF) as usize];
            b += 1;
        }
        k += 1;
    }
    tables
}

/// Entry `byte` of table `k` of [`TABLES`].
#[inline]
fn entry(k: usize, byte: u8) -> u32 {
    // Always `Some`: `k` is a constant below 8 at every call, and a `u8` indexes 256 entries.
    TABLES
        .get(k)
        .and_then(|table| table.get(usize::from(byte)))
        .copied()
        .unwrap_or(0)
}

/// A CRC-32 computed over bytes fed in pieces: [`Crc32::update`] with each piece in order, then
/// [`Crc32::finish`] gives what [`crc32`] gives of them all at once. A long record is so checked
/// without holding it whole.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Crc32(u32);

impl Crc32 {
    /// The checksum of no bytes yet.
    pub(crate) const fn new() -> Self {
        Self(!0)
    }

    /// Feeds `bytes`, after those fed before.
    ///
    /// Eight bytes at a time through [`TABLES`], then the last few one at a time.
    pub(crate) fn update(&mut self, bytes: &[u8]) {
        let (whole, rest) = bytes.as_chunks::<8>();
        let mut crc = self.0;
        for chunk in whole {
            let [b0, b1, b2, b3, b4, b5, b6, b7] = *chunk;
            let [r0, r1, r2, r3] = (crc ^ u32::from_le_bytes([b0, b1, b2, b3])).to_le_bytes();
            crc = entry(7, r0)
                ^ entry(6, r1)
                ^ entry(5, r2)
                ^ entry(4, r3)
                ^ entry(3, b4)
                ^ entry(2, b5)
                ^ entry(1, b6)
                ^ entry(0, b7);
        }
        self.0 = rest.iter().fold(crc, |crc, &b| {
            let [low, ..] = crc.to_le_bytes();
            entry(0, low ^ b) ^ (crc >> 8)
        });
    }

    /// The CRC-32 of every byte fed.
    pub(crate) const fn finish(self) -> u32 {
        !self.0
    }
}

impl Default for Crc32 {
    fn default() -> Self {
        Self::new()
    }
}

/// The CRC-32 of `bytes`.
pub(crate) fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = Crc32::new();
    crc.update(bytes);
    crc.finish()
}

/// Proofs, for every input, run by `cargo kani -p store`: every table entry, and the tail step.
///
/// The eight-byte step itself (the xor of eight table entries against 64 bit steps) is left to
/// the tests: CBMC did not finish it, for one chunk from any register, in 30 minutes.
#[cfg(kani)]
mod proofs {
    use super::*;

    /// The byte-at-a-time CRC step: register `crc` after feeding `bytes`, one bit at a time,
    /// the definition [`Crc32::update`] must match.
    fn bytewise(mut crc: u32, bytes: &[u8]) -> u32 {
        for &b in bytes {
            crc ^= u32::from(b);
            for _ in 0..8 {
                crc = if crc & 1 == 1 {
                    0xEDB8_8320 ^ (crc >> 1)
                } else {
                    crc >> 1
                };
            }
        }
        crc
    }

    /// Every entry of every table is what its definition says: the register after feeding its
    /// byte and then as many zero bytes as its table's number, into a zero register.
    #[kani::proof]
    #[kani::unwind(9)]
    fn every_table_entry_is_its_definition() {
        let byte: u8 = kani::any();
        let zeros: usize = kani::any();
        kani::assume(zeros < 8);
        let mut fed = [0_u8; 8];
        fed[0] = byte;
        let Some(fed) = fed.get(..=zeros) else {
            return;
        };
        assert_eq!(entry(zeros, byte), bytewise(0, fed), "the defined register");
    }

    /// From every register, a tail byte through the byte table gives the register the bitwise
    /// loop gives.
    #[kani::proof]
    #[kani::unwind(9)]
    fn a_tail_byte_matches_bytewise() {
        let start: u32 = kani::any();
        let byte: u8 = kani::any();
        let mut crc = Crc32(start);
        crc.update(&[byte]);
        assert_eq!(crc.0, bytewise(start, &[byte]), "the same register");
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    #[test]
    fn known_vectors() {
        assert_eq!(crc32(b""), 0, "empty");
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926, "the standard check value");
        assert_eq!(
            crc32(b"The quick brown fox jumps over the lazy dog"),
            0x414F_A339,
            "pangram"
        );
    }

    /// The byte-at-a-time CRC, the definition [`crc32`] must match.
    fn bytewise(bytes: &[u8]) -> u32 {
        !bytes.iter().fold(!0u32, |crc, &b| {
            byte_step((crc ^ u32::from(b)) & 0xFF) ^ (crc >> 8)
        })
    }

    #[test]
    fn chunked_updates_match_the_whole() {
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            state >> 33
        };
        for _ in 0..200 {
            let len = usize::try_from(next() % 300).unwrap();
            let bytes: Vec<u8> = (0..len).map(|_| next().to_le_bytes()[0]).collect();
            let mut splits: Vec<usize> = (0..next() % 5)
                .map(|_| usize::try_from(next()).unwrap() % (len + 1))
                .collect();
            splits.sort_unstable();
            let mut crc = Crc32::new();
            let mut from = 0;
            for &at in splits.iter().chain([&len]) {
                crc.update(&bytes[from..at]);
                from = at;
            }
            assert_eq!(
                crc.finish(),
                crc32(&bytes),
                "length {len}, splits {splits:?}"
            );
        }
    }

    #[test]
    fn slicing_matches_the_byte_loop_at_every_length_and_offset() {
        let data: Vec<u8> = (0..1031_u32)
            .map(|i| u8::try_from(i.wrapping_mul(2_654_435_761) >> 24).unwrap())
            .collect();
        for start in 0..9 {
            for len in (0..=data.len() - start)
                .step_by(7)
                .chain([0, 1, 7, 8, 9, 15, 16, 17])
            {
                let bytes = &data[start..start + len];
                assert_eq!(crc32(bytes), bytewise(bytes), "start {start}, length {len}");
            }
        }
    }
}
