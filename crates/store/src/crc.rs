//! CRC-32 (IEEE 802.3: reflected, polynomial 0xEDB88320), the checksum on every record.

/// The byte-at-a-time lookup table.
const TABLE: [u32; 256] = table();

#[expect(
    clippy::indexing_slicing,
    reason = "const evaluation: i < 256 is checked by the loop and at compile time"
)]
/// The lookup table, built at compile time: entry `b` is the CRC register after feeding byte
/// `b` into a zero register, eight reflected shifts with the polynomial.
const fn table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        #[expect(clippy::cast_possible_truncation, reason = "i < 256")]
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 == 1 {
                0xEDB8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
            k += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
}

/// The CRC-32 of `bytes`.
pub(crate) fn crc32(bytes: &[u8]) -> u32 {
    !bytes.iter().fold(!0u32, |crc, &b| {
        let [low, ..] = crc.to_le_bytes();
        // Always `Some`: a `u8` indexes a 256-entry table.
        TABLE.get(usize::from(low ^ b)).copied().unwrap_or(0) ^ (crc >> 8)
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
}
