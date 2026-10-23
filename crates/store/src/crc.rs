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

/// The CRC-32 of `bytes`.
///
/// Eight bytes at a time through [`TABLES`], then the last few one at a time.
pub(crate) fn crc32(bytes: &[u8]) -> u32 {
    let (whole, rest) = bytes.as_chunks::<8>();
    let mut crc = !0u32;
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
    !rest.iter().fold(crc, |crc, &b| {
        let [low, ..] = crc.to_le_bytes();
        entry(0, low ^ b) ^ (crc >> 8)
    })
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
