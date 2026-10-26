//! Spike 2a: one level step of tile evaluation on the GPU.
//!
//! The question (renderer spec, "Spikes"): what does moving tile evaluation to the GPU buy, and
//! does the GPU agree with itself at seams? For a set of tiles of one level, the CPU gathers each
//! tile's parent block and corner rings with `geom::gather_block` (as `geom`'s evaluation does,
//! through owner copies) and its decoded corrections; the GPU then predicts, builds frames,
//! applies the corrections and computes normals, every tile at once (`shaders/eval-gpu`,
//! compiled by rust-gpu).
//!
//! Seams are exact on the CPU because every read goes to a vertex's owner copy, so the GPU need
//! not match the CPU bit for bit: what must hold is that a vertex several tiles compute comes out
//! of every one of them with the same bits. [`seam_consistency`] checks it; [`VecDifference`]
//! measures the GPU against `geom` for the record.

use std::collections::HashMap;
use std::fmt;
use std::time::Duration;

use geom::{
    BaseMesh, Channel, EvalError, LayerId, Level, PositionSource, Region, TileKey, Vec3,
    VertexStencil, gather_block, vertex_stencil,
};
use thiserror::Error;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, Buffer, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ShaderSource,
};

use crate::gpu::{Gpu, GpuError};
use crate::spike::Timer;

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;

/// The SPIR-V rust-gpu compiled from `shaders/eval-gpu`.
///
/// Rebuilt by `shaders/build.sh`; checked in with the stamp of its sources (`spirv.sources`)
/// the gate checks.
const SPIRV: &[u8] = include_bytes!("eval_gpu.spv");

// SPIR-V is a stream of 32-bit words; `make_spirv_raw` would panic on a partial one.
const _: () = assert!(SPIRV.len().is_multiple_of(4), "eval_gpu.spv is whole words");

/// Quads per side of every tile the shaders evaluate: the finest level's tiles.
pub const QUADS: u32 = 64;

/// Vertices per side of a tile.
const SIDE: u32 = QUADS + 1;

/// Vertices per tile.
const VERTICES: u32 = SIDE * SIDE;

/// Slots per side of a tile's block with its one-vertex ring.
const PADDED_SIDE: u32 = QUADS + 3;

/// Slots per side of a tile's parent block with its ring.
const PARENT_SIDE: u32 = QUADS / 2 + 3;

/// `u32`s per tile header (`shaders/eval-gpu`'s `Header` lays them out).
const HEADER_WORDS: usize = 40;

/// Invocations per workgroup in every entry point.
const WORKGROUP: u32 = 64;

/// The bits of the NaN the shaders write where `geom` writes NaN: a quiet NaN. The host hands
/// it over because `naga` rejects NaN literals in a shader.
const NAN_BITS: u32 = 0x7FC0_0000;

/// Marks an index slot with no vertex (diagonally off a face corner).
const NO_VERTEX: u32 = u32::MAX;

/// Why the evaluation spike couldn't run.
#[derive(Debug, Error)]
pub enum EvalSpikeError {
    /// Gathering a tile's inputs failed: a position the source lacks.
    #[error(transparent)]
    Eval(#[from] EvalError),
    /// The GPU failed.
    #[error(transparent)]
    Gpu(#[from] GpuError),
    /// A tile isn't [`QUADS`] quads a side, or its level has no parent.
    #[error("tile {0:?} isn't a {QUADS}-quad tile with a parent level")]
    TileSize(TileKey),
    /// The tiles' vertices don't all have an index exact in `f32` (`2^24`): too many tiles.
    #[error("{0} tiles are more than one evaluation indexes")]
    TooManyTiles(usize),
    /// The tiles aren't all of one level.
    #[error("the tiles span more than one level")]
    MixedLevels,
}

/// One level step's inputs for a set of tiles, in the shaders' layouts.
#[derive(Debug, Clone)]
pub struct Inputs {
    /// The tiles, in dispatch order.
    tiles: Vec<TileKey>,
    /// Per tile, [`HEADER_WORDS`] words: its place on the face, its corners' rings, its face's
    /// owner axes.
    headers: Vec<u32>,
    /// Per tile, its parent block with its ring, `(QUADS / 2 + 3)²` positions.
    parents: Vec<[f32; 3]>,
    /// The parent-level rings of the tiles' base-vertex corners: centre, edges, diagonals.
    rings: Vec<[f32; 3]>,
    /// Per tile, its decoded corrections, `(QUADS + 1)²`.
    corrections: Vec<[f32; 3]>,
    /// Per tile, `(QUADS + 3)²` indices into the evaluated positions of the owner copy each
    /// normal reads; then the tiles' this-level rings, as lists of indices.
    index: Vec<u32>,
}

/// The bytes of each kind of input, for the upload comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InputBytes {
    /// The parent blocks and rings: what must be resident on the GPU for a cold tile.
    pub parents: usize,
    /// The decoded corrections: what an edit changes.
    pub corrections: usize,
    /// The headers and the normals' index: structural, the same until the tile set changes.
    pub structure: usize,
}

/// Answers every read of one level with the vertex's index among a set of tiles' positions.
///
/// The index is encoded as the position's `x`: `gather_block` through this source gives the
/// index of each owner copy a block reads, as the GPU must read it.
struct IndexSource<'a> {
    /// Each tile's place in the set.
    slots: &'a HashMap<TileKey, u32>,
}

impl PositionSource for IndexSource<'_> {
    fn position(&self, face: geom::FaceId, level: Level, i: u32, j: u32) -> Option<Vec3> {
        let (key, li, lj) =
            TileKey::containing(LayerId::BASE, Channel::DISPLACEMENT, face, level, i, j)?;
        let slot = *self.slots.get(&key)?;
        let index = slot * VERTICES + lj * SIDE + li;
        #[expect(
            clippy::cast_precision_loss,
            reason = "indices stay below 2^24 (checked by Inputs::gather), exact in f32"
        )]
        Some(Vec3::new(index as f32, 0.0, 0.0))
    }
}

/// The index an [`IndexSource`] encoded in `v`, or [`NO_VERTEX`] for a slot without a vertex.
fn decode(v: Vec3) -> u32 {
    if v.x.is_nan() {
        NO_VERTEX
    } else {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "an encoded index: a non-negative integer below 2^24, exact"
        )]
        let index = v.x as u32;
        index
    }
}

/// Writes `values` into `words` from `at` on; what doesn't fit is dropped (the callers' layouts
/// fit).
fn put(words: &mut [u32], at: usize, values: &[u32]) {
    for (word, &value) in words.iter_mut().skip(at).zip(values) {
        *word = value;
    }
}

/// The owner axes of each edge of `face` at `level`, as the shaders' header holds them: probed
/// with `geom::vertex_stencil` at each edge's midpoint.
fn edge_axes(mesh: &BaseMesh, face: geom::FaceId, level: Level) -> [i32; 16] {
    let n = level.face_quads();
    let region = Region::whole_face(face, level);
    let probes = [(n / 2, 0), (n, n / 2), (n / 2, n), (0, n / 2)];
    let mut words = [0; 16];
    for (chunk, (i, j)) in words.as_chunks_mut::<4>().0.iter_mut().zip(probes) {
        let [[u0, u1], [v0, v1]] = match vertex_stencil(mesh, region, i, j) {
            VertexStencil::Axes(axes) => axes,
            VertexStencil::Corner(_) | VertexStencil::Outside => [[1, 0], [0, 1]],
        };
        *chunk = [u0, u1, v0, v1];
    }
    words
}

impl Inputs {
    /// Gathers the inputs of `tiles` (all [`QUADS`] quads a side, of one level): each tile's
    /// parent block and corner rings from `parents` (positions of the level above), its
    /// corrections from `corrections(tile, i, j)` (tile-local, `0..=QUADS`), and the index of
    /// the owner copies its normals read among the tiles themselves.
    ///
    /// A tile's normals read one vertex past it, and a base vertex's whole ring, so those reads
    /// must land in `tiles` too: pass every tile of the area evaluated.
    ///
    /// # Errors
    ///
    /// [`EvalSpikeError::TileSize`] for a tile of another size or at level 0;
    /// [`EvalSpikeError::MixedLevels`] for tiles of several levels;
    /// [`EvalSpikeError::TooManyTiles`] beyond what `f32` indexes exactly;
    /// [`EvalSpikeError::Eval`] if `parents` lacks a position, or a normal reads a tile not in
    /// `tiles`.
    pub fn gather(
        mesh: &BaseMesh,
        parents: &impl PositionSource,
        tiles: &[TileKey],
        corrections: impl Fn(&TileKey, u32, u32) -> Vec3,
    ) -> Result<Self, EvalSpikeError> {
        let count = u32::try_from(tiles.len())
            .ok()
            .filter(|&n| u64::from(n) * u64::from(VERTICES) < 1 << 24)
            .ok_or(EvalSpikeError::TooManyTiles(tiles.len()))?;
        let level = tiles.first().map(TileKey::level);
        if tiles.iter().any(|t| Some(t.level()) != level) {
            return Err(EvalSpikeError::MixedLevels);
        }
        let slots: HashMap<TileKey, u32> = tiles.iter().copied().zip(0..count).collect();
        let index_source = IndexSource { slots: &slots };
        let mut inputs = Self {
            tiles: tiles.to_vec(),
            headers: Vec::with_capacity(tiles.len() * HEADER_WORDS),
            parents: Vec::new(),
            // Offset 0 is a placeholder, so the buffer is never empty.
            rings: vec![[0.0; 3]],
            corrections: Vec::new(),
            index: Vec::new(),
        };
        let mut this_level_rings: Vec<u32> = Vec::new();
        let mut axes_by_face: HashMap<geom::FaceId, [i32; 16]> = HashMap::new();
        let mut ring_lists = Vec::with_capacity(tiles.len());
        for tile in tiles {
            let region = Region::of_tile(tile);
            let parent_region = region
                .parent()
                .filter(|_| region.quads() == QUADS)
                .ok_or(EvalSpikeError::TileSize(*tile))?;
            let parent = gather_block(mesh, parents, parent_region)?;
            if parent.padded().len() != (PARENT_SIDE * PARENT_SIDE) as usize {
                return Err(EvalSpikeError::TileSize(*tile));
            }
            inputs
                .parents
                .extend(parent.padded().iter().map(|p| [p.x, p.y, p.z]));
            let own = gather_block(mesh, &index_source, region)?;
            inputs.index.extend(own.padded().iter().map(|&v| decode(v)));
            let mut header = [0_u32; HEADER_WORDS];
            let place = [
                tile.level().face_quads(),
                region.x0(),
                region.y0(),
                NAN_BITS,
            ];
            put(&mut header, 0, &place);
            let mut lists = [0_u32; 4];
            for c in 0..4 {
                let (Some(ring), Some(own_ring)) = (parent.corner(c), own.corner(c)) else {
                    continue;
                };
                let valence = u32::try_from(ring.edges().len()).unwrap_or(0);
                let offset = u32::try_from(inputs.rings.len()).unwrap_or(u32::MAX);
                put(&mut header, 4 + c, &[offset]);
                put(&mut header, 8 + c, &[valence]);
                put(&mut header, 12 + c, &[u32::from(ring.owner_corner())]);
                let point = |p: Vec3| [p.x, p.y, p.z];
                inputs.rings.push(point(ring.center()));
                inputs.rings.extend(ring.edges().iter().copied().map(point));
                inputs
                    .rings
                    .extend(ring.diagonals().iter().copied().map(point));
                put(
                    &mut lists,
                    c,
                    &[u32::try_from(this_level_rings.len()).unwrap_or(u32::MAX)],
                );
                this_level_rings.push(decode(own_ring.center()));
                this_level_rings.extend(own_ring.edges().iter().map(|&v| decode(v)));
            }
            ring_lists.push(lists);
            let axes = *axes_by_face
                .entry(tile.face())
                .or_insert_with(|| edge_axes(mesh, tile.face(), tile.level()));
            put(&mut header, 16, &axes.map(i32::cast_unsigned));
            inputs.headers.extend_from_slice(&header);
            for j in 0..SIDE {
                for i in 0..SIDE {
                    let c = corrections(tile, i, j);
                    inputs.corrections.push([c.x, c.y, c.z]);
                }
            }
        }
        // The this-level ring lists go after every tile's padded index block.
        let lists_start = u32::try_from(inputs.index.len()).unwrap_or(u32::MAX);
        for (header, lists) in inputs.headers.chunks_mut(HEADER_WORDS).zip(&ring_lists) {
            put(header, 32, &lists.map(|list| lists_start + list));
        }
        inputs.index.extend(this_level_rings);
        Ok(inputs)
    }

    /// The tiles, in dispatch order: the positions and normals come back in this order.
    #[must_use]
    pub fn tiles(&self) -> &[TileKey] {
        &self.tiles
    }

    /// How many bytes each kind of input is.
    #[must_use]
    pub const fn bytes(&self) -> InputBytes {
        InputBytes {
            parents: (self.parents.len() + self.rings.len()) * size_of::<[f32; 3]>(),
            corrections: self.corrections.len() * size_of::<[f32; 3]>(),
            structure: (self.headers.len() + self.index.len()) * size_of::<u32>(),
        }
    }

    /// Whether tile `slot` has a base vertex of valence other than 4 at a corner.
    #[must_use]
    pub fn has_extraordinary_corner(&self, slot: usize) -> bool {
        self.headers
            .get(slot * HEADER_WORDS + 8..slot * HEADER_WORDS + 12)
            .is_some_and(|valences| valences.iter().any(|&v| v != 0 && v != 4))
    }
}

/// The three compute pipelines of one level step: predict, apply, normals.
#[derive(Debug)]
pub struct GpuEval {
    /// Predicts each tile's block with its ring.
    predict: ComputePipeline,
    /// Builds frames and applies corrections: the evaluated positions.
    apply: ComputePipeline,
    /// Computes normals from the evaluated positions.
    normals: ComputePipeline,
}

impl GpuEval {
    /// Builds the pipelines from the rust-gpu SPIR-V.
    ///
    /// # Errors
    ///
    /// As [`Gpu::compute_pipeline`].
    pub fn new(gpu: &Gpu) -> Result<Self, GpuError> {
        let source = || ShaderSource::SpirV(wgpu::util::make_spirv_raw(SPIRV));
        Ok(Self {
            predict: gpu.compute_pipeline("eval predict", source(), "predict")?,
            apply: gpu.compute_pipeline("eval apply", source(), "apply")?,
            normals: gpu.compute_pipeline("eval normals", source(), "normals")?,
        })
    }

    /// Uploads `inputs` and binds the pipelines to them.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`].
    pub fn prepare(&self, gpu: &Gpu, inputs: &Inputs) -> Result<Run, GpuError> {
        let tiles = u32::try_from(inputs.tiles.len()).unwrap_or(u32::MAX);
        let device = gpu.device();
        gpu.scoped("eval prepare", || {
            let init = |label, contents: &[u8], usage| init_buffer(gpu, label, contents, usage);
            let empty = |label, size: u64, usage| {
                device.create_buffer(&BufferDescriptor {
                    label: Some(label),
                    // A zero-sized binding is invalid.
                    size: size.max(4),
                    usage,
                    mapped_at_creation: false,
                })
            };
            let storage = BufferUsages::STORAGE;
            let point = size_of::<[f32; 3]>() as u64;
            let vertices = u64::from(tiles) * u64::from(VERTICES);
            let headers = init(
                "eval headers",
                bytemuck::cast_slice(&inputs.headers),
                storage,
            );
            let parents = init(
                "eval parents",
                bytemuck::cast_slice(&inputs.parents),
                storage | BufferUsages::COPY_DST,
            );
            let rings = init("eval rings", bytemuck::cast_slice(&inputs.rings), storage);
            let corrections = init(
                "eval corrections",
                bytemuck::cast_slice(&inputs.corrections),
                storage | BufferUsages::COPY_DST,
            );
            let index = init("eval index", bytemuck::cast_slice(&inputs.index), storage);
            let predicted = empty(
                "eval predicted",
                u64::from(tiles) * u64::from(PADDED_SIDE * PADDED_SIDE) * point,
                storage,
            );
            let positions = empty(
                "eval positions",
                vertices * point,
                storage | BufferUsages::COPY_SRC,
            );
            let normals = empty(
                "eval normals",
                vertices * point,
                storage | BufferUsages::COPY_SRC,
            );
            let bind = |pipeline: &ComputePipeline, entries: &[(u32, &Buffer)]| {
                bind_group(gpu, pipeline, entries)
            };
            let groups = [
                bind(
                    &self.predict,
                    &[(0, &headers), (1, &parents), (2, &rings), (3, &predicted)],
                ),
                bind(
                    &self.apply,
                    &[
                        (0, &headers),
                        (2, &rings),
                        (3, &predicted),
                        (4, &corrections),
                        (5, &positions),
                    ],
                ),
                bind(
                    &self.normals,
                    &[(0, &headers), (5, &positions), (6, &index), (7, &normals)],
                ),
            ];
            Run {
                tiles,
                pipelines: [
                    self.predict.clone(),
                    self.apply.clone(),
                    self.normals.clone(),
                ],
                groups,
                parents,
                corrections,
                positions,
                normals,
                timer: gpu.timestamp_period().map(|period| Timer::new(gpu, period)),
            }
        })
    }
}

/// A buffer holding `contents` (one word of zero if empty: a zero-sized binding is invalid).
fn init_buffer(gpu: &Gpu, label: &str, contents: &[u8], usage: BufferUsages) -> Buffer {
    gpu.device().create_buffer_init(&BufferInitDescriptor {
        label: Some(label),
        contents: if contents.is_empty() {
            &[0; 4]
        } else {
            contents
        },
        usage,
    })
}

/// A bind group of `pipeline`'s group 0, binding each `(binding, buffer)` of `entries`.
fn bind_group(gpu: &Gpu, pipeline: &ComputePipeline, entries: &[(u32, &Buffer)]) -> BindGroup {
    let entries: Vec<BindGroupEntry<'_>> = entries
        .iter()
        .map(|&(binding, buffer)| BindGroupEntry {
            binding,
            resource: buffer.as_entire_binding(),
        })
        .collect();
    gpu.device().create_bind_group(&BindGroupDescriptor {
        label: Some("eval"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    })
}

/// A set of tiles' inputs on the GPU, ready to evaluate: made by [`GpuEval::prepare`].
#[derive(Debug)]
pub struct Run {
    /// How many tiles.
    tiles: u32,
    /// Predict, apply, normals.
    pipelines: [ComputePipeline; 3],
    /// Their bindings to the buffers.
    groups: [BindGroup; 3],
    /// The parent blocks: what a cold tile needs resident.
    parents: Buffer,
    /// The decoded corrections: what an edit uploads.
    corrections: Buffer,
    /// The evaluated positions, written by apply.
    positions: Buffer,
    /// The normals, written by normals.
    normals: Buffer,
    /// Times the step on the GPU, if the device has timestamp queries.
    timer: Option<Timer>,
}

/// The evaluated positions and normals of every tile of a [`Run`], tile after tile, each
/// `(QUADS + 1)²` row-major.
#[derive(Debug, Clone)]
pub struct Evaluated {
    /// The positions.
    pub positions: Vec<[f32; 3]>,
    /// The normals.
    pub normals: Vec<[f32; 3]>,
}

impl Run {
    /// Uploads new corrections (`inputs` of the same tiles): the per-step upload of an edit.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`].
    pub fn upload_corrections(&self, gpu: &Gpu, inputs: &Inputs) -> Result<(), GpuError> {
        gpu.scoped("eval corrections upload", || {
            gpu.queue().write_buffer(
                &self.corrections,
                0,
                bytemuck::cast_slice(&inputs.corrections),
            );
        })
    }

    /// Uploads new parent blocks and corrections (`inputs` of the same tiles): what the CPU
    /// sends when the tiles' parents changed too.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`].
    pub fn upload_parents(&self, gpu: &Gpu, inputs: &Inputs) -> Result<(), GpuError> {
        gpu.scoped("eval parents upload", || {
            gpu.queue()
                .write_buffer(&self.parents, 0, bytemuck::cast_slice(&inputs.parents));
        })?;
        self.upload_corrections(gpu, inputs)
    }

    /// Evaluates every tile: predict, apply, normals, in one compute pass, and submits it
    /// without waiting.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`].
    pub fn dispatch(&self, gpu: &Gpu) -> Result<(), GpuError> {
        self.submit(gpu, None)
    }

    /// Evaluates every tile, waits, and returns how long the pass took on the GPU; `None`
    /// if the device has no timestamp queries.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`] and [`Gpu::read_buffer`].
    pub fn time(&self, gpu: &Gpu) -> Result<Option<Duration>, GpuError> {
        let Some(timer) = &self.timer else {
            self.dispatch(gpu)?;
            return Ok(None);
        };
        self.submit(gpu, Some(timer))?;
        timer.elapsed(gpu).map(Some)
    }

    /// Reads the positions and normals back, waiting for every dispatch submitted before.
    ///
    /// # Errors
    ///
    /// As [`Gpu::read_buffer`].
    pub fn read(&self, gpu: &Gpu) -> Result<Evaluated, GpuError> {
        let tiles = self.tiles as usize * VERTICES as usize;
        let mut positions: Vec<[f32; 3]> = gpu.read_buffer(&self.positions)?;
        let mut normals: Vec<[f32; 3]> = gpu.read_buffer(&self.normals)?;
        positions.truncate(tiles);
        normals.truncate(tiles);
        Ok(Evaluated { positions, normals })
    }

    /// Records the three dispatches, in one pass timed by `timer` if given, and submits them.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`].
    fn submit(&self, gpu: &Gpu, timer: Option<&Timer>) -> Result<(), GpuError> {
        let sizes = [
            (PADDED_SIDE * PADDED_SIDE).div_ceil(WORKGROUP),
            VERTICES.div_ceil(WORKGROUP),
            VERTICES.div_ceil(WORKGROUP),
        ];
        gpu.scoped("eval dispatch", || {
            let mut encoder = gpu
                .device()
                .create_command_encoder(&CommandEncoderDescriptor::default());
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("eval"),
                    timestamp_writes: timer.map(Timer::writes),
                });
                for ((pipeline, group), x) in self.pipelines.iter().zip(&self.groups).zip(sizes) {
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, group, &[]);
                    pass.dispatch_workgroups(x, self.tiles, 1);
                }
            }
            if let Some(timer) = timer {
                timer.resolve(&mut encoder);
            }
            gpu.queue().submit([encoder.finish()]);
        })
    }
}

/// How far one set of vectors is from another, vector by vector.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct VecDifference {
    /// The largest distance in units in the last place, over every component.
    max_ulps: u64,
    /// The largest Euclidean distance.
    max_distance: f32,
    /// Vectors whose bits differ at all.
    differing: usize,
    /// Vectors compared.
    compared: usize,
}

impl VecDifference {
    /// Adds the comparison of `a[k]` against `b[k]` for every `k` (the shorter prefix if their
    /// lengths differ).
    pub fn add(&mut self, a: &[[f32; 3]], b: &[[f32; 3]]) {
        for (x, y) in a.iter().zip(b) {
            self.compared += 1;
            let mut differs = false;
            let mut squared = 0.0_f32;
            for (&p, &q) in x.iter().zip(y) {
                let ulps = ordered(p).abs_diff(ordered(q));
                self.max_ulps = self.max_ulps.max(ulps);
                differs |= ulps != 0;
                squared += (p - q) * (p - q);
            }
            self.differing += usize::from(differs);
            self.max_distance = self.max_distance.max(squared.sqrt());
        }
    }

    /// The largest distance in units in the last place, over every component.
    #[must_use]
    pub const fn max_ulps(&self) -> u64 {
        self.max_ulps
    }

    /// The largest Euclidean distance.
    #[must_use]
    pub const fn max_distance(&self) -> f32 {
        self.max_distance
    }

    /// Vectors whose bits differ at all.
    #[must_use]
    pub const fn differing(&self) -> usize {
        self.differing
    }

    /// Vectors compared.
    #[must_use]
    pub const fn compared(&self) -> usize {
        self.compared
    }
}

impl fmt::Display for VecDifference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} vectors: max {} ulps, max distance {:e}, {} differ",
            self.compared, self.max_ulps, self.max_distance, self.differing
        )
    }
}

/// `x`'s bits as an integer that orders like the float, so the difference of two is their
/// distance in units in the last place, across zero too.
fn ordered(x: f32) -> i64 {
    let bits = i64::from(x.to_bits());
    if bits & 0x8000_0000 == 0 {
        bits
    } else {
        0x8000_0000 - bits
    }
}

/// Whether the vertices several tiles compute come out of each with the same bits.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Consistency {
    /// Vertices computed by more than one tile of the set.
    pub shared: usize,
    /// Pairs of copies compared (each copy against the first).
    pub pairs: usize,
    /// Of those, pairs whose positions' bits differ.
    pub position_mismatches: usize,
    /// Of those, pairs whose normals' bits differ.
    pub normal_mismatches: usize,
    /// The most tiles that computed one vertex.
    pub most_copies: usize,
}

impl fmt::Display for Consistency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} shared vertices, {} pairs of copies (up to {} copies of one vertex): {} \
             position and {} normal mismatches",
            self.shared,
            self.pairs,
            self.most_copies,
            self.position_mismatches,
            self.normal_mismatches
        )
    }
}

/// Checks `evaluated` (of `tiles`, in order) for seam self-consistency.
///
/// Every vertex on a tile's border, which `geom::Topology::copies` says other tiles of the set
/// hold too, must have the same position and normal bits in every one of them.
#[must_use]
pub fn seam_consistency(mesh: &BaseMesh, tiles: &[TileKey], evaluated: &Evaluated) -> Consistency {
    let slots: HashMap<TileKey, usize> = tiles.iter().copied().zip(0..).collect();
    let at = |slot: usize, i: u32, j: u32| slot * VERTICES as usize + (j * SIDE + i) as usize;
    let bits = |values: &[[f32; 3]], k: usize| values.get(k).map(|v| v.map(f32::to_bits));
    let mut result = Consistency::default();
    for (slot, tile) in tiles.iter().enumerate() {
        let region = Region::of_tile(tile);
        for (i, j) in border(QUADS) {
            let copies =
                mesh.topology()
                    .copies(tile.face(), tile.level(), region.x0() + i, region.y0() + j);
            let held: Vec<usize> = copies
                .iter()
                .filter_map(|copy| {
                    let other = *slots.get(&copy.key(tile.level()))?;
                    Some(at(other, copy.i, copy.j))
                })
                .collect();
            // Each shared vertex is counted once: from the tile that holds its first copy.
            if held.len() < 2 || held.first() != Some(&at(slot, i, j)) {
                continue;
            }
            result.shared += 1;
            result.most_copies = result.most_copies.max(held.len());
            let first = held.first().copied().unwrap_or_default();
            for &other in held.iter().skip(1) {
                result.pairs += 1;
                let same = |values: &[[f32; 3]]| bits(values, first) == bits(values, other);
                result.position_mismatches += usize::from(!same(&evaluated.positions));
                result.normal_mismatches += usize::from(!same(&evaluated.normals));
            }
        }
    }
    result
}

/// The border vertices of a block of `q` quads per side, each once.
fn border(q: u32) -> impl Iterator<Item = (u32, u32)> {
    let rows = [0, q]
        .into_iter()
        .flat_map(move |j| (0..=q).map(move |i| (i, j)));
    let columns = (1..q).flat_map(move |j| [(0, j), (q, j)]);
    rows.chain(columns)
}
