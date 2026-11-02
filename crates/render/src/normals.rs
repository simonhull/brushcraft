//! Finest tiles' normals, computed on the GPU from positions alone (spec "Positions only").
//!
//! A finest tile is uploaded as a [`RingRecord`]: its positions with the one-vertex ring taken
//! from its neighbours, the rings of the base vertices at its corners, and which stencil each
//! vertex's normal takes. The normal pass (`NormalPass`, `normals.wgsl`) runs where the frame's
//! uploads land, right after their copies and before any pass reads a slot: it writes each such
//! tile's vertices, positions and unit normals, into the tile's slot as [`GpuVertex`]es. So a
//! tile's normals never wait on a neighbouring tile being resident, and both sides of a seam
//! compute them from the same bits with the same instructions, so they agree bit for bit.
//!
//! # Where the ring lives
//!
//! A finest tile of 65² vertices comes with 67² positions. Its **slot stays 65² [`GpuVertex`]es**,
//! in the size class of every other tile of side 65, read by the raster, the bounds pass and the
//! shading pass as before (`side = 2^class + 1`). The ring travels apart from the slot, in a
//! record staged into a **record chunk** (a storage buffer the pass reads) that lives for the
//! frame: the pass reads the record and writes the whole slot. A slot of side 67 would have been
//! the other way, with every pass that reads slots learning a second stride and an offset; the
//! ring is only the normal pass's input, so only the normal pass reads it.
//!
//! Uploads shrink with it: a record is positions only, 12 bytes a point, so a finest tile is
//! 20 header words, 67² points and its corners' rings: 53,952 bytes with no base vertex at a
//! corner (53,984 with one of valence 3), and 8 bytes of job, against the 135,200 bytes of its 65²
//! [`GpuVertex`]es uploaded before (the spec's estimate of ≈ 107 KB took 24 bytes a point). The
//! GPU writes those 135,200 bytes itself, a slot to a slot.
//!
//! # The arithmetic
//!
//! [`geom::GatheredBlock`]'s notes spell out what the CPU computes, in `f32` without fused
//! multiply-adds; the shader writes the same expressions in the same order. Bit equality with the
//! CPU isn't promised: a GPU's square root and division are within a few ULP rather than correctly
//! rounded, its compiler may fuse a product into a subtraction, and it may flush denormals to zero
//! (WGSL allows it): where the CPU normalizes a normal whose squared length is denormal (a cross
//! product below about 1e-19), such a GPU takes it as zero and gives the fallback `(0, 0, 1)`.
//! Matching it would mean changing the CPU's formula in `geom` too; at the scales a sculpt has
//! (positions of order 1, finest steps of order 1e-3, squared lengths of order 1e-12) it doesn't
//! arise. What is promised is the seam:
//! every vertex runs the same code, its stencil read as data, so two tiles holding a vertex compute
//! it with the same instructions from the same bits (the store gives every copy of a vertex its
//! owner's bits, and every face's copy of a base vertex's ring the owner's order).
//!
//! # Stencils
//!
//! Which points a vertex's normal reads is structural ([`geom::vertex_stencil`]): inside the
//! tile, the block's own axes; on a tile edge, one stencil for every vertex strictly inside that
//! edge (the edge lies inside the face, with the block's own axes, or along one face edge, with
//! the axes of the face owning it); at a block corner, its own (a base vertex's ring, or a face
//! edge's axes). So a record carries nine stencils, the four corners' and one per edge, taken from
//! `vertex_stencil` at the corners and the edges' middle vertices, and the inside's; a vertex picks
//! one by its place and reads it from the record, so every vertex runs the same code on data.

use std::collections::BTreeMap;

use geom::{BaseMesh, GatheredBlock, Vec3, VertexStencil, vertex_stencil};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBinding, BufferBindingType,
    BufferSize, CommandEncoder, ComputePassDescriptor, ComputePipeline, ShaderStages,
};

use crate::cull::{MAX_GROUPS, entry, pipeline, storage_buffer};
use crate::gpu::{Gpu, GpuError};
use crate::residency::{SizeClass, Slot};
use crate::upload::{Batch, GpuVertex, StagingRing, VERTEX_WGSL};

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
pub(crate) mod tests;

/// The normal pass's own source, after [`VERTEX_WGSL`].
const NORMALS_WGSL: &str = include_str!("normals.wgsl");

/// Threads per workgroup: `normals.wgsl`'s `GROUP`.
const GROUP: u32 = 64;

/// Header word: quads per side.
const HEADER_QUADS: usize = 0;

/// Header words: the stencils, by the index `normals.wgsl`'s `stencil_index` gives a vertex.
///
/// Block corners 0 to 3, then tile edges 0 to 3 (`j = 0`, `i = q`, `j = q`, `i = 0`), then the
/// vertices inside the tile ([`OWN_AXES`]).
const HEADER_STENCILS: usize = 1;

/// How many stencils a record holds.
const STENCILS: usize = 9;

/// Header words: per block corner, its ring's first edge point, in words from the record's start.
const HEADER_RINGS: usize = HEADER_STENCILS + STENCILS;

/// Header words: per block corner, its ring's valence (0 where the corner isn't a base vertex).
const HEADER_VALENCES: usize = HEADER_RINGS + 4;

/// Words before the positions: the header, padded.
const HEADER_WORDS: usize = 20;

/// A stencil reading the ring of block corner `stencil & 3`.
const STENCIL_CORNER: u32 = 1 << 31;

/// A stencil off the face: the normal is NaN.
const STENCIL_OUTSIDE: u32 = 1 << 30;

/// The block's own axes, `u = (1, 0)` and `v = (0, 1)`: the stencil of every vertex inside the
/// tile, and of a tile edge too short to have a vertex inside it.
const OWN_AXES: VertexStencil = VertexStencil::Axes([[1, 0], [0, 1]]);

/// One finest tile as the normal pass reads it: positions with the one-vertex ring, the rings of
/// the base vertices at its corners, and its stencils, as `u32` words (`f32`s by their bits).
///
/// Layout, in words:
///
/// - `0`: `q`, the quads per side (64 for a tile of 65² vertices);
/// - `1..10`: the stencils of block corners `(0, 0)`, `(q, 0)`, `(q, q)`, `(0, q)`, then of the
///   vertices strictly inside tile edges `j = 0`, `i = q`, `j = q`, `i = 0`, then of the vertices
///   inside the tile (the block's own axes): `STENCIL_CORNER` with the corner, `STENCIL_OUTSIDE`,
///   or the axes `[[ui, uj], [vi, vj]]`, each step plus one in two bits from bit 0. Every vertex
///   reads its stencil from here, none takes one the shader knows;
/// - `10..14`: per block corner, the word its ring's edge points start at;
/// - `14..18`: per block corner, its ring's valence, 0 where it isn't a base vertex;
/// - `20..`: the `(q + 3)²` positions, row-major from `(-1, -1)`, three words each (as
///   [`GatheredBlock::padded`]), then each base vertex's edge points, owner first, three words
///   each; padded to a whole number of 8 bytes, as staging copies need.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct RingRecord {
    /// The words.
    words: Vec<u32>,
}

impl RingRecord {
    /// Makes this the record of `block`, gathered over `mesh` (whose topology gives the
    /// stencils), reusing its memory.
    pub(crate) fn write(&mut self, mesh: &BaseMesh, block: &GatheredBlock) {
        let region = block.region();
        let q = region.quads();
        let corners = [(0, 0), (q, 0), (q, q), (0, q)];
        let middle = q / 2;
        let edges = [(middle, 0), (q, middle), (middle, q), (0, middle)];
        let mut stencils = [encode(OWN_AXES); STENCILS];
        let at = corners.into_iter().chain(edges).enumerate();
        for ((k, (i, j)), stencil) in at.zip(&mut stencils) {
            // Every vertex of a tile edge is one of its corners when the tile is one quad wide.
            if k < 4 || q >= 2 {
                *stencil = encode(vertex_stencil(mesh, region, i, j));
            }
        }
        let rings = [0, 1, 2, 3].map(|c| block.corner(c).map(geom::GatheredRing::edges));
        self.assemble(q, &stencils, block.padded(), rings);
    }

    /// Makes this the record of a tile of `q` quads a side: its stencils (by `stencil_index`), its
    /// `(q + 3)²` positions with the ring, and per block corner the edge points of the base
    /// vertex's ring there, if it is one.
    pub(crate) fn assemble(
        &mut self,
        q: u32,
        stencils: &[u32; STENCILS],
        padded: &[Vec3],
        rings: [Option<&[Vec3]>; 4],
    ) {
        let words = &mut self.words;
        words.clear();
        words.resize(HEADER_WORDS, 0);
        if let Some(first) = words.get_mut(HEADER_QUADS) {
            *first = q;
        }
        let header = words.iter_mut().skip(HEADER_STENCILS).zip(stencils);
        for (slot, stencil) in header {
            *slot = *stencil;
        }
        // `Vec3` is three `f32`s, `repr(C)`: its words are its components' bits.
        words.extend_from_slice(bytemuck::cast_slice(padded));
        for (c, ring) in rings.into_iter().enumerate() {
            let Some(edges) = ring else {
                continue;
            };
            let start = u32::try_from(words.len()).unwrap_or(u32::MAX);
            let valence = u32::try_from(edges.len()).unwrap_or(0);
            words.extend_from_slice(bytemuck::cast_slice(edges));
            if let Some(slot) = words.get_mut(HEADER_RINGS + c) {
                *slot = start;
            }
            if let Some(slot) = words.get_mut(HEADER_VALENCES + c) {
                *slot = valence;
            }
        }
        if !words.len().is_multiple_of(2) {
            words.push(0);
        }
    }

    /// Quads per side of the tile; 0 for a record never written.
    #[must_use]
    pub(crate) fn quads(&self) -> u32 {
        self.words.get(HEADER_QUADS).copied().unwrap_or(0)
    }

    /// Appends the tile's own `(q + 1)²` positions, without the ring, to `out`: `f32` bits,
    /// three words a vertex, row-major.
    pub(crate) fn tile_positions(&self, out: &mut Vec<u32>) {
        let q = usize::try_from(self.quads()).unwrap_or(0);
        let padded = q + 3;
        out.reserve(3 * (q + 1) * (q + 1));
        for j in 1..=q + 1 {
            let start = HEADER_WORDS + 3 * (j * padded + 1);
            if let Some(row) = self.words.get(start..start + 3 * (q + 1)) {
                out.extend_from_slice(row);
            }
        }
    }

    /// The record's bytes, as staged.
    #[must_use]
    pub(crate) fn bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.words)
    }

    /// The words, for tests that change one side of a seam.
    #[cfg(test)]
    pub(crate) fn words_mut(&mut self) -> &mut [u32] {
        &mut self.words
    }
}

/// The bytes of the record of a tile of `q` quads a side whose corners' base vertices have
/// `ring_points` edge points together: what [`RingRecord::bytes`] is for it.
pub(crate) fn record_bytes(q: u32, ring_points: u32) -> u64 {
    let padded = u64::from(q) + 3;
    let header = HEADER_WORDS as u64;
    let words = header + 3 * padded * padded + 3 * u64::from(ring_points);
    words.next_multiple_of(2) * 4
}

/// `stencil` as a record's header word.
pub(crate) fn encode(stencil: VertexStencil) -> u32 {
    /// A grid step (-1, 0 or 1), plus one, in two bits.
    fn step(x: i32) -> u32 {
        (x.clamp(-1, 1) + 1).unsigned_abs()
    }
    match stencil {
        VertexStencil::Corner(c) => STENCIL_CORNER | u32::from(c & 3),
        VertexStencil::Axes([[ui, uj], [vi, vj]]) => {
            step(ui) | (step(uj) << 2) | (step(vi) << 4) | (step(vj) << 6)
        }
        VertexStencil::Outside => STENCIL_OUTSIDE,
    }
}

/// One tile the frame's normal pass computes.
#[derive(Debug, Clone)]
struct Job {
    /// The record chunk its record is in, by index.
    chunk: usize,
    /// That chunk.
    records: Buffer,
    /// The record's first word in the chunk.
    record: u32,
    /// The slot buffer its vertices go to.
    target: Buffer,
    /// Its first vertex in the slot buffer.
    base: u32,
    /// Quads per side.
    quads: u32,
}

/// A job as the shader reads it: WGSL's `Job`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuJob {
    /// The record's first word in its chunk.
    record: u32,
    /// The tile's first vertex in its slot buffer.
    base: u32,
}

/// The bytes of one [`GpuJob`].
pub(crate) const JOB_BYTES: u64 = size_of::<GpuJob>() as u64;

/// The normal pass: the records a frame stages for its finest tiles, and the pipeline that turns
/// them into the tiles' vertices.
///
/// Owned by residency, on the owner thread: [`NormalPass::begin`] each frame,
/// [`NormalPass::stage`] per finest tile read with its ring, [`NormalPass::finish`] when the
/// frame's uploads are staged, which gives the frame's [`NormalWork`].
///
/// **Record chunks** are storage buffers of a fixed size (a staging buffer's, at most what the
/// device binds), made as a frame first needs them and reused by every frame after, from the
/// first: a frame's copies into them follow the previous frame's pass in queue order. At most
/// `kept` are kept, as many as hold the staging ring's cap, so what a frame the ring caps stages
/// always fits them. A frame forced past the cap (the floor, a resync's epoch rescued whole)
/// stages the rest into **one-off chunks**, held by the frame's [`NormalWork`] and dropped with it
/// once submitted, as a one-off staging buffer is. The kept chunks and the job list are GPU memory
/// outside the slot budget ([`NormalPass::kept_bytes`]).
#[derive(Debug)]
pub(crate) struct NormalPass {
    /// The pass.
    pipeline: ComputePipeline,
    /// Its bind group layout.
    layout: BindGroupLayout,
    /// Bytes per record chunk.
    chunk_bytes: u64,
    /// The most record chunks kept.
    kept: usize,
    /// The record chunks kept.
    chunks: Vec<Buffer>,
    /// The frame's one-off chunks: by index, past the kept ones.
    one_off: Vec<Buffer>,
    /// The chunk the frame is filling, by index (kept, then one-off).
    chunk: usize,
    /// Bytes the frame used in it.
    cursor: u64,
    /// The job list the frame's dispatches read, grown as a frame needs.
    jobs: Buffer,
    /// Dispatches' job lists start at multiples of this many bytes (the device's storage offset
    /// alignment).
    job_alignment: u64,
    /// The frame's jobs, by the slot they write: a slot written twice in a frame keeps the last.
    pending: BTreeMap<Slot, Job>,
}

impl NormalPass {
    /// The pass, with record chunks of `chunk_bytes` (at most what the device binds), keeping as
    /// many as hold `kept_bytes` (the staging ring's cap).
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`], making the pipeline or the job list.
    pub(crate) fn new(gpu: &Gpu, chunk_bytes: u64, kept_bytes: u64) -> Result<Self, GpuError> {
        let device = gpu.device();
        let limits = device.limits();
        let largest = limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size);
        let storage = |binding, read_only| BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = gpu.scoped("normals layout", || {
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("normals"),
                entries: &[storage(0, true), storage(1, false), storage(2, true)],
            })
        })?;
        let source = format!("{VERTEX_WGSL}{NORMALS_WGSL}");
        let pipeline = pipeline(gpu, "normals", &source, &[&layout], "finest_normals")?;
        let jobs = storage_buffer(gpu, "normal jobs", 4096)?;
        let chunk_bytes = chunk_bytes.min(largest).max(8);
        let kept = usize::try_from(kept_bytes.div_ceil(chunk_bytes)).unwrap_or(usize::MAX);
        Ok(Self {
            pipeline,
            layout,
            chunk_bytes,
            kept,
            chunks: Vec::new(),
            one_off: Vec::new(),
            chunk: 0,
            cursor: 0,
            jobs,
            job_alignment: u64::from(limits.min_storage_buffer_offset_alignment).max(JOB_BYTES),
            pending: BTreeMap::new(),
        })
    }

    /// Starts a frame: no jobs, the record chunks empty, the last frame's one-off chunks let go
    /// (its work holds them until it is submitted).
    pub(crate) fn begin(&mut self) {
        self.pending.clear();
        self.one_off.clear();
        self.chunk = 0;
        self.cursor = 0;
    }

    /// Bytes of GPU memory it keeps between frames, outside the slot budget: the record chunks
    /// kept and the job list.
    pub(crate) fn kept_bytes(&self) -> u64 {
        let chunks: u64 = self.chunks.iter().map(Buffer::size).sum();
        chunks + self.jobs.size()
    }

    /// How many record chunks it keeps.
    #[cfg(test)]
    pub(crate) const fn kept_chunks(&self) -> usize {
        self.chunks.len()
    }

    /// Chunk `index`: a kept one, then the frame's one-offs.
    fn chunk_at(&self, index: usize) -> Option<&Buffer> {
        match index.checked_sub(self.chunks.len()) {
            None => self.chunks.get(index),
            Some(past) => self.one_off.get(past),
        }
    }

    /// Forgets the frame's job for `slot`, written since with vertices of their own (which the
    /// pass would overwrite).
    pub(crate) fn forget(&mut self, slot: Slot) {
        self.pending.remove(&slot);
    }

    /// Stages `record` for the tile whose vertices go to `slot`, the slot at `slot_offset` bytes
    /// into `target`; whether it was. Without `forced`, `false` when the staging ring has no
    /// room, as [`Batch::write`].
    ///
    /// # Errors
    ///
    /// As [`Batch::write`]; as [`Gpu::scoped`], making a record chunk.
    #[expect(
        clippy::too_many_arguments,
        reason = "a staging write's arguments and the slot it computes into"
    )]
    pub(crate) fn stage(
        &mut self,
        gpu: &Gpu,
        batch: &mut Batch,
        ring: &mut StagingRing,
        record: &RingRecord,
        slot: Slot,
        target: (&Buffer, u64),
        forced: bool,
    ) -> Result<bool, GpuError> {
        let data = record.bytes();
        let size = data.len() as u64;
        let (chunk, offset) = self.room(gpu, size)?;
        let Some(buffer) = self.chunk_at(chunk) else {
            return Ok(false);
        };
        if !batch.write(gpu, ring, data, buffer, offset, forced)? {
            return Ok(false);
        }
        let records = buffer.clone();
        self.chunk = chunk;
        self.cursor = offset + size;
        let (target, slot_offset) = target;
        let base = slot_offset / size_of::<GpuVertex>() as u64;
        self.pending.insert(
            slot,
            Job {
                chunk,
                records,
                record: u32::try_from(offset / 4).unwrap_or(u32::MAX),
                target: target.clone(),
                base: u32::try_from(base).unwrap_or(u32::MAX),
                quads: record.quads(),
            },
        );
        Ok(true)
    }

    /// Where `size` bytes of record go: the chunk and the offset in it, the chunk made if need
    /// be: kept while fewer than `kept` are and the record fits one, else one-off (as large as
    /// the record, if it is larger).
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`], making a chunk.
    fn room(&mut self, gpu: &Gpu, size: u64) -> Result<(usize, u64), GpuError> {
        let mut chunk = self.chunk;
        let mut cursor = self.cursor;
        loop {
            match self.chunk_at(chunk) {
                Some(buffer) if buffer.size() - cursor >= size => return Ok((chunk, cursor)),
                Some(_) => {
                    chunk += 1;
                    cursor = 0;
                }
                None => {
                    // Kept chunks come first: one is added only while there is no one-off.
                    let keep = self.one_off.is_empty()
                        && self.chunks.len() < self.kept
                        && size <= self.chunk_bytes;
                    if keep {
                        let buffer = storage_buffer(gpu, "normal records", self.chunk_bytes)?;
                        self.chunks.push(buffer);
                    } else {
                        let bytes = self.chunk_bytes.max(size);
                        let buffer = storage_buffer(gpu, "one-off normal records", bytes)?;
                        self.one_off.push(buffer);
                    }
                    return Ok((self.chunks.len() + self.one_off.len() - 1, 0));
                }
            }
        }
    }

    /// Ends the frame: stages its job list into `batch` (past the ring's cap: the records are
    /// staged already) and returns the pass's dispatches, or `None` if no finest tile came with
    /// its ring.
    ///
    /// # Errors
    ///
    /// As [`Batch::write`]; as [`Gpu::scoped`], growing the job list or making a bind group.
    pub(crate) fn finish(
        &mut self,
        gpu: &Gpu,
        batch: &mut Batch,
        ring: &mut StagingRing,
    ) -> Result<Option<NormalWork>, GpuError> {
        if self.pending.is_empty() {
            return Ok(None);
        }
        // One dispatch per record chunk and slot buffer (a slot buffer holds one size class, so
        // one tile side), each at most MAX_GROUPS jobs.
        let mut runs: BTreeMap<(usize, SizeClass, u16), Run> = BTreeMap::new();
        for (slot, job) in std::mem::take(&mut self.pending) {
            let gpu_job = GpuJob {
                record: job.record,
                base: job.base,
            };
            runs.entry((job.chunk, slot.class(), slot.buffer()))
                .or_insert_with(|| Run {
                    records: job.records,
                    target: job.target,
                    quads: job.quads,
                    jobs: Vec::new(),
                })
                .jobs
                .push(gpu_job);
        }
        let max = usize::try_from(MAX_GROUPS).unwrap_or(usize::MAX);
        let align = usize::try_from(self.job_alignment / JOB_BYTES).unwrap_or(1);
        let mut list: Vec<GpuJob> = Vec::new();
        // Per dispatch: its run, its first job in the list, how many.
        let mut spans: Vec<(&Run, u64, u32)> = Vec::new();
        for run in runs.values() {
            for jobs in run.jobs.chunks(max) {
                let padded = list.len().next_multiple_of(align);
                list.resize(padded, GpuJob { record: 0, base: 0 });
                let start = list.len() as u64 * JOB_BYTES;
                spans.push((run, start, u32::try_from(jobs.len()).unwrap_or(MAX_GROUPS)));
                list.extend_from_slice(jobs);
            }
        }
        let list_bytes = list.len() as u64 * JOB_BYTES;
        if self.jobs.size() < list_bytes {
            self.jobs = storage_buffer(gpu, "normal jobs", list_bytes.next_power_of_two())?;
        }
        batch.write(gpu, ring, bytemuck::cast_slice(&list), &self.jobs, 0, true)?;
        let dispatches = spans.iter().map(|&(run, start, count)| {
            let group = gpu.scoped("normals bind group", || {
                gpu.device().create_bind_group(&BindGroupDescriptor {
                    label: Some("normals"),
                    layout: &self.layout,
                    entries: &[
                        entry(0, &run.records),
                        entry(1, &run.target),
                        BindGroupEntry {
                            binding: 2,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: &self.jobs,
                                offset: start,
                                size: BufferSize::new(u64::from(count) * JOB_BYTES),
                            }),
                        },
                    ],
                })
            })?;
            let side = run.quads + 1;
            Ok(Dispatch {
                group,
                runs: (side * side).div_ceil(GROUP),
                jobs: count,
            })
        });
        let dispatches: Result<Vec<Dispatch>, GpuError> = dispatches.collect();
        Ok(Some(NormalWork {
            pipeline: self.pipeline.clone(),
            dispatches: dispatches?,
        }))
    }
}

/// The frame's jobs of one record chunk and one slot buffer.
#[derive(Debug)]
struct Run {
    /// The record chunk.
    records: Buffer,
    /// The slot buffer.
    target: Buffer,
    /// Quads per side of its tiles (a slot buffer's tiles are all of one side).
    quads: u32,
    /// The jobs.
    jobs: Vec<GpuJob>,
}

/// One dispatch of the normal pass.
#[derive(Debug)]
struct Dispatch {
    /// Its record chunk, slot buffer and job list.
    group: BindGroup,
    /// Workgroups per tile: runs of [`GROUP`] vertices.
    runs: u32,
    /// Tiles.
    jobs: u32,
}

/// A frame's normal pass, carried by its [`crate::upload::FrameUploads`] to the render thread and
/// encoded right after the uploads' copies.
#[derive(Debug)]
pub(crate) struct NormalWork {
    /// The pass.
    pipeline: ComputePipeline,
    /// Its dispatches.
    dispatches: Vec<Dispatch>,
}

impl NormalWork {
    /// Encodes the pass into `encoder`: after the copies that bring its records, before any pass
    /// that reads the slots it writes.
    pub(crate) fn encode(&self, encoder: &mut CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("finest normals"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        for dispatch in &self.dispatches {
            pass.set_bind_group(0, &dispatch.group, &[]);
            pass.dispatch_workgroups(dispatch.runs, dispatch.jobs, 1);
        }
    }
}
