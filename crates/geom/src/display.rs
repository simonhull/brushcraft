//! The display layer: what a tile below the finest level draws, and how wrong it can be.
//!
//! A brush moves the finest level only, so a coarse level of the correction pyramid shows none
//! of what was sculpted finer. A display tile instead holds, for one tile at level `l` below the
//! finest, the finest surface **subsampled** at that level (vertex `(i, j)` of the level-`l` face
//! grid is the finest vertex `(i·2ᵏ, j·2ᵏ)`, `k = finest − l`), its **averaged normals** with a
//! roughness term, and `e`, a **true bound** on the distance from the surface it draws to every
//! finest vertex under it.
//!
//! # The surface a tile draws
//!
//! Every quad of every level is split along the diagonal from face-grid vertex `(i, j)` to
//! `(i + 1, j + 1)`. A finer level's triangles then refine the coarser level's, so the difference
//! between two levels' surfaces is linear on each finer triangle and largest at a finer vertex.
//! The error a level adds, `δ` ([`step_error`]), is therefore the largest distance from a
//! level-`(l + 1)` vertex to the point of level `l` at the same grid parameter: the vertex itself
//! for an even vertex, the midpoint of its edge's ends for an edge vertex, or the midpoint of its
//! quad's diagonal for a quad centre. The bound chains up the levels:
//! `e_l(tile) = δ_l(tile) + max e_{l+1}` over the level-`(l + 1)` tiles under it, `e = 0` at the
//! finest level. Every position in it is a **decoded** one, so the codec's quantization is
//! inside the bound.
//!
//! # Normals
//!
//! A vertex-centred mip of *unnormalized* mean normals, a 3×3 tent per level (weights ¼ centre,
//! ⅛ edge neighbours, 1/16 diagonal ones, renormalized where a face border cuts the tent off;
//! [`mip_normals`]), so each level is computed from the one below. The stored normal is the
//! normalized mean, and the mean's length is the roughness term (Toksvig), as a `u8`.
//!
//! # Encoding
//!
//! [`DisplayCodec`] is how a tile's positions are stored; [`NormalEncoding`] how its normals are.
//! [`build_face`] and [`DisplayTile::decode`] use one fixed choice of both (see the encoding
//! module's notes for the candidates and why).

use thiserror::Error;

use crate::{EvalError, Level, TileKey, Vec3};

mod build;
mod encode;

pub use build::{MeanNormals, build_face, build_face_with, mip_normals, step_error, subsample};
#[cfg(any(test, feature = "display-study"))]
pub use encode::{BaseOffsets, FixedPoint};
pub use encode::{DisplayCodec, Interpolating, NormalEncoding, Precision};

/// Why a display tile couldn't be built, encoded or decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum DisplayError {
    /// A grid doesn't have the shape the operation needs: a finest grid that isn't its level's
    /// whole face, a finer grid that isn't twice a coarser one, or a tile's positions that
    /// aren't its key's side.
    #[error("a grid has the wrong shape for its level or tile")]
    Shape,
    /// A position or normal isn't finite.
    #[error("a display tile's input is not finite")]
    NotFinite,
    /// Encoded bytes have the wrong length for their tile, or a header value is out of range.
    #[error("display tile bytes are malformed")]
    Malformed,
    /// A value is too large for the encoding to store.
    #[error("a display tile's values are too large to encode")]
    TooLarge,
    /// A codec decoded a vertex shared by two tiles of a level to different bits in each, the
    /// second of them this tile: seams wouldn't be exact, and the bound would miss the gap.
    #[error("display tile {0:?} decodes a shared vertex differently from its neighbour")]
    Seam(TileKey),
    /// The level is the finest, or above it: it has no display tiles.
    #[error("level {0} has no display tiles under the finest level")]
    NoDisplayLevel(Level),
}

impl From<EvalError> for DisplayError {
    fn from(_: EvalError) -> Self {
        Self::Shape
    }
}

/// One display tile: the finest surface subsampled at a coarser tile, as decoded from its
/// encoding, with its averaged normals, roughness and error bound.
///
/// Built by [`build_face`] (all of a face's tiles at once, bound chained), or read back with
/// [`DisplayTile::decode`] from the bytes [`DisplayTile::encoded`] gives.
#[derive(Debug, Clone, PartialEq)]
pub struct DisplayTile {
    /// The tile: channel [`Channel::DISPLAY`](crate::Channel::DISPLAY), level below the finest.
    key: TileKey,
    /// The decoded positions, row-major `side²`: what the renderer draws.
    positions: Vec<Vec3>,
    /// The decoded unit normals, row-major `side²`.
    normals: Vec<Vec3>,
    /// The roughness term per vertex: the length of the mean normal, `0..=255` for `0..=1`.
    roughness: Vec<u8>,
    /// `e`: a bound, in world units, on the distance from the surface the tile draws to every
    /// finest vertex under it.
    error: f32,
    /// The encoded positions and normals ([`DisplayTile::encoded`]).
    encoded: Vec<u8>,
}

impl DisplayTile {
    /// The tile.
    #[must_use]
    pub const fn key(&self) -> &TileKey {
        &self.key
    }

    /// The decoded positions, row-major `side²`.
    #[must_use]
    pub fn positions(&self) -> &[Vec3] {
        &self.positions
    }

    /// The decoded unit normals, row-major `side²`.
    #[must_use]
    pub fn normals(&self) -> &[Vec3] {
        &self.normals
    }

    /// The roughness term per vertex, row-major `side²`: the mean normal's length, `0..=255`
    /// standing for `0..=1` (255 where the finest normals under the vertex all agree).
    #[must_use]
    pub fn roughness(&self) -> &[u8] {
        &self.roughness
    }

    /// `e`: a bound, in world units, on the distance from the surface the tile draws (its
    /// decoded positions, triangulated along each quad's `(i, j)`–`(i + 1, j + 1)` diagonal) to
    /// every finest vertex under it, at the same face-grid parameter.
    #[must_use]
    pub const fn error(&self) -> f32 {
        self.error
    }

    /// The tile's encoded positions and normals, as [`DisplayTile::decode`] reads them: a
    /// `u32` length of the positions' part, the positions as the codec stored them, then the
    /// normals and roughness.
    #[must_use]
    pub fn encoded(&self) -> &[u8] {
        &self.encoded
    }

    /// The tile `key` from bytes [`DisplayTile::encoded`] gave for a tile [`build_face`] built,
    /// with its bound `error` (stored beside them).
    ///
    /// # Errors
    ///
    /// [`DisplayError::Malformed`] if the bytes don't hold such a tile of `key`'s side, or the
    /// bound isn't finite and non-negative; [`DisplayError::NotFinite`] if a position decodes
    /// to a value that isn't finite.
    pub fn decode(key: TileKey, error: f32, bytes: &[u8]) -> Result<Self, DisplayError> {
        Self::decode_with(&encode::CHOSEN, encode::CHOSEN_NORMALS, key, error, bytes)
    }

    /// [`DisplayTile::decode`] for bytes encoded with `codec` and `normals`.
    ///
    /// # Errors
    ///
    /// As [`DisplayTile::decode`].
    pub fn decode_with(
        codec: &impl DisplayCodec,
        normals: NormalEncoding,
        key: TileKey,
        error: f32,
        bytes: &[u8],
    ) -> Result<Self, DisplayError> {
        if !(error.is_finite() && error >= 0.0) {
            return Err(DisplayError::Malformed);
        }
        let (positions, rest) = encode::split_parts(bytes)?;
        let positions = codec.decode(&key, positions)?.as_slice().to_vec();
        let (normals, roughness) = normals.decode(positions.len(), rest)?;
        Ok(Self {
            key,
            positions,
            normals,
            roughness,
            error,
            encoded: bytes.to_vec(),
        })
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;
