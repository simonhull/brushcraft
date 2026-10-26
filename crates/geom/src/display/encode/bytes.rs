//! Byte-level helpers for the display codecs: a bounds-checked reader, byte planes, delta
//! coding and zigzag signs.

use crate::display::DisplayError;

/// Each value less the one before it (the first less zero), wrapping.
pub(super) fn deltas_u16(values: &[u16]) -> Vec<u16> {
    let mut prev = 0_u16;
    values
        .iter()
        .map(|&v| {
            let d = v.wrapping_sub(prev);
            prev = v;
            d
        })
        .collect()
}

/// [`deltas_u16`]'s inverse.
pub(super) fn undeltas_u16(deltas: &[u16]) -> Vec<u16> {
    let mut prev = 0_u16;
    deltas
        .iter()
        .map(|&d| {
            prev = prev.wrapping_add(d);
            prev
        })
        .collect()
}

/// Each byte less the one before it (the first less zero), wrapping.
pub(super) fn deltas_u8(values: &[u8]) -> Vec<u8> {
    let mut prev = 0_u8;
    values
        .iter()
        .map(|&v| {
            let d = v.wrapping_sub(prev);
            prev = v;
            d
        })
        .collect()
}

/// [`deltas_u8`]'s inverse.
pub(super) fn undeltas_u8(deltas: &[u8]) -> Vec<u8> {
    let mut prev = 0_u8;
    deltas
        .iter()
        .map(|&d| {
            prev = prev.wrapping_add(d);
            prev
        })
        .collect()
}

/// `values` as two byte planes: every low byte, then every high byte.
pub(super) fn put_u16_planes(out: &mut Vec<u8>, values: &[u16]) {
    for k in 0..2 {
        out.extend(
            values
                .iter()
                .map(|v| v.to_le_bytes().get(k).copied().unwrap_or(0)),
        );
    }
}

/// `count` values from [`put_u16_planes`]'s planes.
///
/// # Errors
///
/// [`DisplayError::Malformed`] if the reader holds fewer.
pub(super) fn take_u16_planes(
    reader: &mut Reader<'_>,
    count: usize,
) -> Result<Vec<u16>, DisplayError> {
    let lo = reader.take(count)?;
    let hi = reader.take(count)?;
    Ok(lo
        .iter()
        .zip(hi)
        .map(|(&l, &h)| u16::from_le_bytes([l, h]))
        .collect())
}

/// `values` as four byte planes, lowest byte first.
pub(super) fn put_u32_planes(out: &mut Vec<u8>, values: &[u32]) {
    for k in 0..4 {
        out.extend(
            values
                .iter()
                .map(|v| v.to_le_bytes().get(k).copied().unwrap_or(0)),
        );
    }
}

/// `count` values from [`put_u32_planes`]'s planes.
///
/// # Errors
///
/// [`DisplayError::Malformed`] if the reader holds fewer.
pub(super) fn take_u32_planes(
    reader: &mut Reader<'_>,
    count: usize,
) -> Result<Vec<u32>, DisplayError> {
    let planes = [
        reader.take(count)?,
        reader.take(count)?,
        reader.take(count)?,
        reader.take(count)?,
    ];
    let [b0, b1, b2, b3] = planes;
    Ok(b0
        .iter()
        .zip(b1)
        .zip(b2)
        .zip(b3)
        .map(|(((&a, &b), &c), &d)| u32::from_le_bytes([a, b, c, d]))
        .collect())
}

/// `v`'s sign folded into its lowest bit, so small magnitudes of either sign are small.
pub(super) const fn zigzag(v: i32) -> u32 {
    u32::from_le_bytes(((v << 1) ^ (v >> 31)).to_le_bytes())
}

/// [`zigzag`]'s inverse.
pub(super) const fn unzigzag(z: u32) -> i32 {
    let half = i32::from_le_bytes((z >> 1).to_le_bytes());
    let sign = i32::from_le_bytes((z & 1).to_le_bytes());
    half ^ -sign
}

/// Reads a byte slice front to back, failing on a short read.
pub(super) struct Reader<'a> {
    /// What is left to read.
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    /// A reader over `bytes`.
    pub(super) const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes }
    }

    /// The next `n` bytes.
    ///
    /// # Errors
    ///
    /// [`DisplayError::Malformed`] if fewer are left.
    pub(super) fn take(&mut self, n: usize) -> Result<&'a [u8], DisplayError> {
        let (head, rest) = self
            .bytes
            .split_at_checked(n)
            .ok_or(DisplayError::Malformed)?;
        self.bytes = rest;
        Ok(head)
    }

    /// The next byte.
    ///
    /// # Errors
    ///
    /// As [`Reader::take`].
    pub(super) fn u8(&mut self) -> Result<u8, DisplayError> {
        let [b] = *self.take_array::<1>()?;
        Ok(b)
    }

    /// The next little-endian `u32`.
    ///
    /// # Errors
    ///
    /// As [`Reader::take`].
    pub(super) fn u32(&mut self) -> Result<u32, DisplayError> {
        Ok(u32::from_le_bytes(*self.take_array::<4>()?))
    }

    /// The next little-endian `f32`.
    ///
    /// # Errors
    ///
    /// As [`Reader::take`].
    pub(super) fn f32(&mut self) -> Result<f32, DisplayError> {
        Ok(f32::from_le_bytes(*self.take_array::<4>()?))
    }

    /// The next `N` bytes as an array.
    ///
    /// # Errors
    ///
    /// As [`Reader::take`].
    pub(super) fn take_array<const N: usize>(&mut self) -> Result<&'a [u8; N], DisplayError> {
        self.take(N)?
            .try_into()
            .map_err(|_short| DisplayError::Malformed)
    }

    /// Everything left.
    pub(super) const fn rest(&self) -> &'a [u8] {
        self.bytes
    }

    /// Checks nothing is left.
    ///
    /// # Errors
    ///
    /// [`DisplayError::Malformed`] if something is.
    pub(super) const fn finish(&self) -> Result<(), DisplayError> {
        if self.bytes.is_empty() {
            Ok(())
        } else {
            Err(DisplayError::Malformed)
        }
    }
}
