//! Normals and roughness: octahedral normals, two signed fractions per vertex, and one
//! roughness byte per vertex.

use super::bytes::{
    Reader, deltas_u8, deltas_u16, put_u16_planes, take_u16_planes, undeltas_u8, undeltas_u16,
};
use crate::Vec3;
use crate::display::DisplayError;

/// How a display tile's normals and roughness are stored: octahedral normals, two components
/// per vertex, then one roughness byte per vertex.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalEncoding {
    /// Two `i16` components.
    Oct16,
    /// Two `i8` components.
    Oct8,
}

/// The unit normal and roughness byte of each of `means`: the normalized mean (`(0, 0, 1)` where
/// it is too short to normalize), and its length, clamped to `0..=1`, in 255ths.
pub(in crate::display) fn shade(means: &[Vec3]) -> (Vec<Vec3>, Vec<u8>) {
    means
        .iter()
        .map(|m| {
            let unit = m.normalized().unwrap_or(Vec3::new(0.0, 0.0, 1.0));
            let len = m.length().clamp(0.0, 1.0);
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "a value in 0..=255 after the clamp, rounded"
            )]
            let rough = (len * 255.0).round() as u8;
            (unit, rough)
        })
        .unzip()
}

impl NormalEncoding {
    /// `normals` (unit) and `roughness`, one each per vertex, encoded.
    ///
    /// # Errors
    ///
    /// [`DisplayError::Shape`] if the lengths differ; [`DisplayError::NotFinite`] for a normal
    /// that isn't finite.
    pub fn encode(self, normals: &[Vec3], roughness: &[u8]) -> Result<Vec<u8>, DisplayError> {
        if normals.len() != roughness.len() {
            return Err(DisplayError::Shape);
        }
        if !normals.iter().all(|n| n.is_finite()) {
            return Err(DisplayError::NotFinite);
        }
        let us: Vec<f32> = normals.iter().map(|&n| octahedral(n).0).collect();
        let vs: Vec<f32> = normals.iter().map(|&n| octahedral(n).1).collect();
        let mut out = Vec::new();
        for plane in [&us, &vs] {
            match self {
                Self::Oct16 => {
                    let q: Vec<u16> = plane.iter().map(|&c| snorm16(c)).collect();
                    put_u16_planes(&mut out, &deltas_u16(&q));
                }
                Self::Oct8 => {
                    let q: Vec<u8> = plane.iter().map(|&c| snorm8(c)).collect();
                    out.extend(deltas_u8(&q));
                }
            }
        }
        out.extend(deltas_u8(roughness));
        Ok(out)
    }

    /// `count` unit normals and roughness bytes from `bytes` that [`NormalEncoding::encode`]
    /// gave.
    ///
    /// # Errors
    ///
    /// [`DisplayError::Malformed`] if the bytes don't hold `count` of each.
    pub fn decode(self, count: usize, bytes: &[u8]) -> Result<(Vec<Vec3>, Vec<u8>), DisplayError> {
        let mut reader = Reader::new(bytes);
        let us = self.read_plane(&mut reader, count)?;
        let vs = self.read_plane(&mut reader, count)?;
        let roughness = undeltas_u8(reader.take(count)?);
        reader.finish()?;
        let normals = us
            .into_iter()
            .zip(vs)
            .map(|(u, v)| from_octahedral(u, v))
            .collect();
        Ok((normals, roughness))
    }

    /// One plane of `count` octahedral components, read from `reader`.
    ///
    /// # Errors
    ///
    /// [`DisplayError::Malformed`] if the reader holds fewer.
    fn read_plane(self, reader: &mut Reader<'_>, count: usize) -> Result<Vec<f32>, DisplayError> {
        Ok(match self {
            Self::Oct16 => undeltas_u16(&take_u16_planes(reader, count)?)
                .into_iter()
                .map(unsnorm16)
                .collect(),
            Self::Oct8 => undeltas_u8(reader.take(count)?)
                .into_iter()
                .map(unsnorm8)
                .collect(),
        })
    }
}

/// Unit vector `n` mapped onto the octahedron and unfolded into `[−1, 1]²`.
fn octahedral(n: Vec3) -> (f32, f32) {
    let l1 = n.x.abs() + n.y.abs() + n.z.abs();
    if l1 <= 0.0 {
        return (0.0, 0.0);
    }
    let (u, v) = (n.x / l1, n.y / l1);
    if n.z >= 0.0 {
        (u, v)
    } else {
        ((1.0 - v.abs()) * sign(u), (1.0 - u.abs()) * sign(v))
    }
}

/// [`octahedral`]'s inverse: the unit vector at `(u, v)`.
fn from_octahedral(across: f32, up: f32) -> Vec3 {
    let height = 1.0 - across.abs() - up.abs();
    let (x, y) = if height >= 0.0 {
        (across, up)
    } else {
        (
            (1.0 - up.abs()) * sign(across),
            (1.0 - across.abs()) * sign(up),
        )
    };
    Vec3::new(x, y, height)
        .normalized()
        .unwrap_or(Vec3::new(0.0, 0.0, 1.0))
}

/// `1` for zero and up, `−1` below: the octahedral fold's sign (`f32::signum` gives `−1` for
/// `−0.0`, which would fold a vector on the equator the wrong way).
fn sign(v: f32) -> f32 {
    if v >= 0.0 { 1.0 } else { -1.0 }
}

/// `c` in `[−1, 1]` as a signed 16-bit fraction, its bits as a `u16`.
fn snorm16(c: f32) -> u16 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "within ±32767 after the clamp, rounded"
    )]
    let q = (c.clamp(-1.0, 1.0) * 32767.0).round() as i16;
    u16::from_le_bytes(q.to_le_bytes())
}

/// [`snorm16`]'s inverse.
fn unsnorm16(q: u16) -> f32 {
    (f32::from(i16::from_le_bytes(q.to_le_bytes())) / 32767.0).clamp(-1.0, 1.0)
}

/// `c` in `[−1, 1]` as a signed 8-bit fraction, its bits as a `u8`.
fn snorm8(c: f32) -> u8 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "within ±127 after the clamp, rounded"
    )]
    let q = (c.clamp(-1.0, 1.0) * 127.0).round() as i8;
    q.to_le_bytes()[0]
}

/// [`snorm8`]'s inverse.
fn unsnorm8(q: u8) -> f32 {
    (f32::from(i8::from_le_bytes([q])) / 127.0).clamp(-1.0, 1.0)
}
