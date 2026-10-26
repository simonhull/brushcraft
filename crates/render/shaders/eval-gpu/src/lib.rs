//! One level step of tile evaluation as compute shaders: `geom`'s Catmull–Clark prediction,
//! tangent frames, corrections and normals, for every tile of a dispatch at once.
//!
//! The host gathers each tile's inputs with `geom::gather_block` and uploads them; three entry
//! points then run, each one invocation per vertex (`x`) and tile (`y`):
//!
//! 1. [`predict`]: the tile's predicted block with its one-vertex ring, from the parent block;
//! 2. [`apply`]: each vertex's tangent frame from the predicted block, and its correction applied
//!    in it: the evaluated positions;
//! 3. [`normals`]: each vertex's normal from the evaluated positions, the tile's and its
//!    neighbours', read through a host-built index of owner copies.
//!
//! Every tile has [`QUADS`] quads a side (the finest level's tiles). Positions and corrections
//! are `[f32; 3]`, tightly packed. A tile's header is [`HEADER_WORDS`] `u32`s; see [`Header`].
//!
//! The arithmetic is `geom`'s, operation for operation and in the same grouping (the modules
//! say which function each twins), so a GPU that rounds like the CPU gives the same bits, and
//! every tile computes a vertex it shares with another from the same inputs in the same order.

#![no_std]

mod frame;
mod math;
mod subdiv;

use spirv_std::glam::UVec3;
use spirv_std::spirv;

use crate::frame::{Frame, Local, NO_CORNER, OWN_AXES, corner_local, stencil};
use crate::math::Vec3;
use crate::subdiv::{edge_point, extraordinary, face_point, ring_edge_point, vertex_point};

/// Quads per side of every tile.
pub const QUADS: u32 = 64;

/// Vertices per side of a tile.
const SIDE: u32 = QUADS + 1;

/// Slots per side of the parent block with its ring: `QUADS / 2 + 3`.
const PARENT_SIDE: u32 = QUADS / 2 + 3;

/// Slots per side of a tile's block with its ring: `QUADS + 3`.
const PADDED_SIDE: u32 = QUADS + 3;

/// `u32`s per tile header.
pub const HEADER_WORDS: u32 = 40;

/// A tile header's word offsets: one tile's place on its face and its corners' rings.
///
/// | words | meaning |
/// |---|---|
/// | 0–3 | `n` (face quads at this level), `x0`, `y0`, the bits of a NaN |
/// | 4–7 | per block corner, the offset of its parent-level ring in the rings buffer |
/// | 8–11 | per block corner, its ring's valence (0: not a base vertex) |
/// | 12–15 | per block corner, the owner face's corner there |
/// | 16–31 | per face edge `k`, the owner axes `u.0, u.1, v.0, v.1` (as `i32`) |
/// | 32–35 | per block corner, the offset of its this-level ring's index list |
#[derive(Debug, Clone, Copy)]
pub struct Header {
    /// The tile's first word in the headers buffer.
    base: u32,
}

/// A base-vertex ring in the rings buffer: the centre at `offset`, then `valence` edge
/// neighbours, then `valence` diagonal neighbours, owner first.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Ring {
    /// Where the centre is.
    pub(crate) offset: u32,
    /// How many edge (and diagonal) neighbours.
    pub(crate) valence: u32,
    /// The owner face's corner at the vertex.
    pub(crate) owner_corner: u32,
}

impl Ring {
    /// Edge neighbour `k`.
    pub(crate) fn edge(self, rings: &[[f32; 3]], k: u32) -> Vec3 {
        load(rings, self.offset + 1 + k)
    }

    /// Diagonal neighbour `k`.
    pub(crate) fn diagonal(self, rings: &[[f32; 3]], k: u32) -> Vec3 {
        load(rings, self.offset + 1 + self.valence + k)
    }
}

/// Element `k` of a buffer of positions.
///
/// rust-gpu can't compile `slice::get` (an `Option<&T>` needs a pointer-to-integer cast, which
/// SPIR-V lacks), so this indexes. Every index comes from the dispatch and the host's headers,
/// which the host sizes the buffers for; and `wgpu` bounds-checks every access, so a host bug
/// reads zero rather than past the buffer.
#[expect(
    clippy::indexing_slicing,
    reason = "rust-gpu can't compile slice::get; indices are host-sized (see above)"
)]
pub(crate) fn load(buffer: &[[f32; 3]], k: u32) -> Vec3 {
    Vec3::from_array(buffer[k as usize])
}

/// Writes `v` to element `k` of a buffer of positions (see [`load`] for the indexing).
#[expect(
    clippy::indexing_slicing,
    reason = "rust-gpu can't compile slice::get_mut; indices are host-sized (see `load`)"
)]
fn store(buffer: &mut [[f32; 3]], k: u32, v: Vec3) {
    buffer[k as usize] = v.to_array();
}

/// Word `k` of a buffer of `u32`s (see [`load`] for the indexing).
#[expect(
    clippy::indexing_slicing,
    reason = "rust-gpu can't compile slice::get; indices are host-sized (see `load`)"
)]
fn word(buffer: &[u32], k: u32) -> u32 {
    buffer[k as usize]
}

impl Header {
    /// Tile `tile`'s header.
    const fn of(tile: u32) -> Self {
        Self {
            base: tile * HEADER_WORDS,
        }
    }

    /// Word `k` of the header.
    fn at(self, headers: &[u32], k: u32) -> u32 {
        word(headers, self.base + k)
    }

    /// `[n, x0, y0, quads]`: where the tile lies on its face.
    fn place(self, headers: &[u32]) -> [u32; 4] {
        [
            self.at(headers, 0),
            self.at(headers, 1),
            self.at(headers, 2),
            QUADS,
        ]
    }

    /// A NaN, from the host: a NaN literal in the shader would be rejected by `naga`.
    fn nan(self, headers: &[u32]) -> Vec3 {
        Vec3::splat(f32::from_bits(self.at(headers, 3)))
    }

    /// The parent-level ring at block corner `c`, if it is a base vertex.
    fn ring(self, headers: &[u32], c: u32) -> Ring {
        Ring {
            offset: self.at(headers, 4 + c),
            valence: self.at(headers, 8 + c),
            owner_corner: self.at(headers, 12 + c),
        }
    }

    /// The axes a vertex with `stencil`'s `edge` takes: the owner axes of that face edge, or
    /// the block's own.
    fn axes(self, headers: &[u32], edge: u32) -> Axes {
        if edge == OWN_AXES {
            return Axes {
                u0: 1,
                u1: 0,
                v0: 0,
                v1: 1,
            };
        }
        let first = 16 + 4 * edge;
        Axes {
            u0: self.at(headers, first).cast_signed(),
            u1: self.at(headers, first + 1).cast_signed(),
            v0: self.at(headers, first + 2).cast_signed(),
            v1: self.at(headers, first + 3).cast_signed(),
        }
    }

    /// Where block corner `c`'s this-level ring index list starts in the index buffer: the
    /// centre's index, then its `valence` edge neighbours'.
    fn ring_indices(self, headers: &[u32], c: u32) -> u32 {
        self.at(headers, 32 + c)
    }
}

/// The grid steps along which a vertex's central differences are taken: the owner's `+i` and
/// `+j`, in this block's grid.
#[derive(Debug, Clone, Copy)]
struct Axes {
    /// The owner's `+i`: its column step.
    u0: i32,
    /// The owner's `+i`: its row step.
    u1: i32,
    /// The owner's `+j`: its column step.
    v0: i32,
    /// The owner's `+j`: its row step.
    v1: i32,
}

/// Parent block slot `(i, j)` (ring included, `-1 ..= QUADS / 2 + 1`) of the tile whose block
/// starts at `base`.
fn parent(parents: &[[f32; 3]], base: u32, i: i32, j: i32) -> Vec3 {
    let k = (j + 1).cast_unsigned() * PARENT_SIDE + (i + 1).cast_unsigned();
    load(parents, base + k)
}

/// Predicts one slot of each tile's block with its ring: `geom`'s `predict`, slot by slot.
///
/// `x` is the slot, `(QUADS + 3)²` of them row-major from `(-1, -1)`; `y` the tile. An even row
/// and column is a vertex point, odd ones a face point, a mix an edge point (the parity rules
/// of `geom`'s `vertex_row` and `face_row`); base vertices at block corners get the
/// extraordinary rule, and the four slots diagonally off the block's corners hold NaN.
///
/// Bindings: 0, the headers; 1, the parent blocks (`(QUADS / 2 + 3)²` each); 2, the rings;
/// 3, the predicted blocks (`(QUADS + 3)²` each), written.
#[spirv(compute(threads(64)))]
pub fn predict(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] headers: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] parents: &[[f32; 3]],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] rings: &[[f32; 3]],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] predicted: &mut [[f32; 3]],
) {
    let (slot, tile) = (id.x, id.y);
    if slot >= PADDED_SIDE * PADDED_SIDE {
        return;
    }
    let header = Header::of(tile);
    let base = tile * PARENT_SIDE * PARENT_SIDE;
    let ci = (slot % PADDED_SIDE).cast_signed() - 1;
    let cj = (slot / PADDED_SIDE).cast_signed() - 1;
    let big = QUADS.cast_signed();
    // A macro, not a closure: rust-gpu can't compile a closure that captures a buffer.
    macro_rules! p {
        ($i:expr, $j:expr) => {
            parent(parents, base, $i, $j)
        };
    }
    let corner = match (ci, cj) {
        (0, 0) => 0,
        (x, 0) if x == big => 1,
        (x, y) if x == big && y == big => 2,
        (0, y) if y == big => 3,
        _ => 4,
    };
    let value = if (ci == -1 || ci == big + 1) && (cj == -1 || cj == big + 1) {
        header.nan(headers)
    } else if corner < 4 && header.ring(headers, corner).valence > 0 {
        extraordinary(rings, header.ring(headers, corner))
    } else if cj % 2 == 0 {
        let j = cj / 2;
        if ci % 2 != 0 {
            let i = (ci + 1) / 2;
            edge_point(
                p!(i - 1, j),
                p!(i, j),
                p!(i - 1, j - 1),
                p!(i, j - 1),
                p!(i - 1, j + 1),
                p!(i, j + 1),
            )
        } else {
            let i = ci / 2;
            vertex_point(
                p!(i, j),
                [p!(i + 1, j), p!(i, j + 1), p!(i - 1, j), p!(i, j - 1)],
                [
                    p!(i + 1, j + 1),
                    p!(i - 1, j + 1),
                    p!(i - 1, j - 1),
                    p!(i + 1, j - 1),
                ],
            )
        }
    } else {
        let b = (cj - 1) / 2;
        if ci % 2 != 0 {
            let i = (ci + 1) / 2;
            face_point(p!(i - 1, b), p!(i, b), p!(i, b + 1), p!(i - 1, b + 1))
        } else {
            let i = ci / 2;
            edge_point(
                p!(i, b),
                p!(i, b + 1),
                p!(i - 1, b),
                p!(i - 1, b + 1),
                p!(i + 1, b),
                p!(i + 1, b + 1),
            )
        }
    };
    store(predicted, tile * PADDED_SIDE * PADDED_SIDE + slot, value);
}

/// Slot `(i, j)` (ring included) of the padded block of `QUADS + 3` a side starting at `base`.
fn padded(block: &[[f32; 3]], base: u32, i: i32, j: i32) -> Vec3 {
    let k = (j + 1).cast_unsigned() * PADDED_SIDE + (i + 1).cast_unsigned();
    load(block, base + k)
}

/// Evaluates one vertex of each tile: its tangent frame from the predicted block (`geom`'s
/// `frames`), and its correction applied in that frame (`geom`'s `Frame::apply`).
///
/// `x` is the vertex, `(QUADS + 1)²` row-major; `y` the tile.
///
/// Bindings: 0, the headers; 2, the rings; 3, the predicted blocks; 4, the corrections
/// (`(QUADS + 1)²` each); 5, the evaluated positions (as many), written.
#[spirv(compute(threads(64)))]
pub fn apply(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] headers: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] rings: &[[f32; 3]],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] predicted: &[[f32; 3]],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] corrections: &[[f32; 3]],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] positions: &mut [[f32; 3]],
) {
    let (vertex, tile) = (id.x, id.y);
    if vertex >= SIDE * SIDE {
        return;
    }
    let header = Header::of(tile);
    let base = tile * PADDED_SIDE * PADDED_SIDE;
    let (i, j) = (vertex % SIDE, vertex / SIDE);
    // A macro, not a closure: rust-gpu can't compile a closure that captures a buffer.
    macro_rules! at {
        ($a:expr, $b:expr) => {
            padded(predicted, base, $a, $b)
        };
    }
    let (si, sj) = (i.cast_signed(), j.cast_signed());
    let stencil = stencil(header.place(headers), i, j);
    let local = if stencil.corner == NO_CORNER {
        let Axes { u0, u1, v0, v1 } = header.axes(headers, stencil.edge);
        Local::central(
            at!(si + u0, sj + u1),
            at!(si - u0, sj - u1),
            at!(si + v0, sj + v1),
            at!(si - v0, sj - v1),
        )
    } else {
        let ring = header.ring(headers, stencil.corner);
        if ring.valence == 0 {
            Local::unknown()
        } else {
            corner_local!(
                extraordinary(rings, ring),
                ring.valence,
                ring.owner_corner,
                |k| ring_edge_point(rings, ring, k)
            )
        }
    };
    let frame = Frame::from_local(local, header.nan(headers));
    let k = tile * SIDE * SIDE + vertex;
    let position = at!(si, sj);
    store(positions, k, frame.apply(position, load(corrections, k)));
}

/// The evaluated position a tile's index slot names, or `nan` where it names none
/// (`u32::MAX`, diagonally off a face corner).
fn indexed(positions: &[[f32; 3]], index: u32, nan: Vec3) -> Vec3 {
    if index == u32::MAX {
        nan
    } else {
        load(positions, index)
    }
}

/// One vertex's normal of each tile, from the evaluated positions: `geom`'s `normals`.
///
/// A tile's normals read one vertex past it (and base vertices their whole ring): the
/// neighbouring tiles' positions, evaluated by [`apply`] in the same dispatch. The host's index
/// says which: per tile, `(QUADS + 3)²` indices into the positions buffer, row-major from
/// `(-1, -1)`, each the owner copy `geom` reads (`u32::MAX` for none); and per base-vertex
/// corner, a list of the centre's and its edge neighbours' indices.
///
/// `x` is the vertex, `(QUADS + 1)²` row-major; `y` the tile.
///
/// Bindings: 0, the headers; 5, the evaluated positions; 6, the index; 7, the normals
/// (`(QUADS + 1)²` each), written.
#[spirv(compute(threads(64)))]
pub fn normals(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] headers: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] positions: &[[f32; 3]],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] index: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] normals: &mut [[f32; 3]],
) {
    let (vertex, tile) = (id.x, id.y);
    if vertex >= SIDE * SIDE {
        return;
    }
    let header = Header::of(tile);
    let nan = header.nan(headers);
    let base = tile * PADDED_SIDE * PADDED_SIDE;
    let (i, j) = (vertex % SIDE, vertex / SIDE);
    // A macro, not a closure: rust-gpu can't compile a closure that captures a buffer.
    macro_rules! at {
        ($a:expr, $b:expr) => {{
            let k = ($b + 1).cast_unsigned() * PADDED_SIDE + ($a + 1).cast_unsigned();
            indexed(positions, word(index, base + k), nan)
        }};
    }
    let (si, sj) = (i.cast_signed(), j.cast_signed());
    let stencil = stencil(header.place(headers), i, j);
    let local = if stencil.corner == NO_CORNER {
        let Axes { u0, u1, v0, v1 } = header.axes(headers, stencil.edge);
        Local::central(
            at!(si + u0, sj + u1),
            at!(si - u0, sj - u1),
            at!(si + v0, sj + v1),
            at!(si - v0, sj - v1),
        )
    } else {
        let ring = header.ring(headers, stencil.corner);
        let list = header.ring_indices(headers, stencil.corner);
        if ring.valence == 0 {
            Local::unknown()
        } else {
            corner_local!(
                indexed(positions, word(index, list), nan),
                ring.valence,
                ring.owner_corner,
                |k| indexed(positions, word(index, list + 1 + k), nan)
            )
        }
    };
    store(
        normals,
        tile * SIDE * SIDE + vertex,
        frame::normal(local, nan),
    );
}
