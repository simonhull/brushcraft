//! Stand-ins: a stroke seen from a distance, shown in the frame of its step (spec, "A stroke
//! seen from a distance", the contract of 2026-10-31).
//!
//! When a brush step edits finest tiles whose display ancestors are on the GPU, those display
//! tiles still hold what the store last built for them, and the CPU refresh rebuilds them only
//! later. A **stand-in** is such a resident display tile, rewritten on the GPU in the frame of
//! the step: every vertex of it that lies inside an edited finest tile takes the exact bits of
//! the finest vertex it subsamples; every other vertex keeps its stored position, and the
//! stored normals stay. It is never saved, read back or reused: the refresh's epoch replaces it
//! (and its arrival morphs in, [`crate::tree::NODE_MORPH_IN`]).
//!
//! # Seams
//!
//! Every display tile's border vertices are the finest vertex's exact bits, stored or stood in:
//! the store keeps them lossless, and a stand-in takes them from the finest tile. A finest
//! vertex that moved changed every finest tile holding it, on every face (each holds its owner's
//! bits, so all of them are in the step's epoch); so every resident display tile with that
//! vertex on its border is an ancestor of an edited finest tile, and stands in too. Both sides
//! of every seam then hold the vertex's new bits, at any pair of levels.
//!
//! # The bound
//!
//! Write `F` for the finest surface the tile's bound `e` was a bound against, `S` for the
//! tile's surface (its triangles over its vertices), `F'` and `S'` after the step, `u` for a
//! place on the face grid. The bound is pointwise: `|F(u) − S(u)| ≤ e` at every finest vertex
//! `u` under the tile (each level's error is measured at the same grid place, and the finer
//! triangles nest in the coarser). A stand-in moves each replaced vertex `v` by
//! `m_v = F'(v) − S(v)` and no other vertex, and `S'(u) − S(u)` is the triangles' blend of the
//! `m_v` (weights at least 0, summing to at most 1), so
//!
//! `|F'(u) − S'(u)| ≤ |F'(u) − F(u)| + |F(u) − S(u)| + max |m_v| ≤ D + e + M`,
//!
//! with `D` the step's displacement bound (how far any finest vertex moved) and `M` the largest
//! `|m_v|`, which the pass measures. The new bound is `e + D + max(D, M)`: at least the spec's
//! `e + 2D`, and a bound whatever the replaced vertices' quantization was (where `M > D`). It is
//! summed in `f32` and raised by `SLACK`, more than the rounding of the sum and of `M` can
//! have taken off. A tile that stands in again bounds against its stand-in, `D` the next step's.
//! The cone becomes the one that covers everything: no displacement bounds how normals turned.
//!
//! The bound lives on the GPU, in the stand-in's **row**: the pass sets it there, and settles
//! it into the tile's node every frame after the frame's tree upload (which writes the node's
//! stored `e` whenever the CPU's tree marks it), until the row is freed.
//!
//! # A frame
//!
//! Residency plans the frame's stand-ins ([`crate::residency`]), stages the finest tiles they
//! read into the **scratch** (`StandInPass::stage_tile`) and the rows' changes
//! (`StandInPass::write_row`), and `StandInPass::finish` makes the frame's
//! `StandInWork`, which [`crate::upload::FrameUploads`] encodes after the copies and the
//! normal pass, before any pass reads a slot or the tree.

use std::collections::BTreeMap;

use bytemuck::{Pod, Zeroable};
use geom::{FinestNormalCone, Level, TILE_QUADS};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBinding, BufferBindingType,
    BufferSize, CommandEncoder, ComputePassDescriptor, ComputePipeline, ShaderStages,
};

use crate::cull::{MAX_GROUPS, entry, pipeline, storage_buffer};
use crate::gpu::{Gpu, GpuError};
use crate::residency::{SizeClass, Slot};
use crate::tree::{NODE_WGSL, NONE, Place};
use crate::upload::{Batch, GpuVertex, StagingRing, VERTEX_WGSL};

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;

/// The passes' shared declarations.
const STANDIN_WGSL: &str = include_str!("standin.wgsl");

/// The replace pass's own source.
const REPLACE_WGSL: &str = include_str!("standin_replace.wgsl");

/// The settle pass's own source.
const SETTLE_WGSL: &str = include_str!("standin_settle.wgsl");

/// Threads per workgroup of both passes.
const GROUP: u32 = 64;

/// A row's flag: a step replaced its tile's vertices this frame, and its bound is to be raised.
const PENDING: u32 = 1;

/// Frames the one-off scratch is kept without a frame using it: a few seconds.
const ONE_OFF_FRAMES: u32 = 240;

/// What the settled bound is multiplied by: `1 + 2⁻¹⁶`, more than the rounding of its sum and
/// of the distances measured (a few `f32` steps, `2⁻²⁴` each, of each) can have taken off.
const SLACK: f32 = 1.0 + 1.0 / 65_536.0;

/// The vertices a side of the largest finest tile.
const FINEST_SIDE: u32 = TILE_QUADS + 1;

/// Words of one finest tile in the scratch: three a vertex, and one to keep a tile's bytes a
/// multiple of 8, as staging copies need.
pub(crate) const TILE_WORDS: u32 = FINEST_SIDE * FINEST_SIDE * 3 + 1;

/// Bytes of one finest tile in the scratch.
const TILE_BYTES: u64 = TILE_WORDS as u64 * 4;

/// A bound on how far a brush step moved any finest vertex, in world units: what a stand-in's
/// bound grows by ([module notes](self), "The bound").
///
/// The store's step epoch carries it ([`store::EpochKind::Step`]): the edit path computes it
/// for the store's in-stroke bound.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct Displacement(f32);

/// A displacement bound that isn't one: not finite, or negative.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
#[error("{0} is no displacement bound: it must be finite and not negative")]
pub struct NotADisplacement(pub f32);

impl TryFrom<f32> for Displacement {
    type Error = NotADisplacement;

    /// A displacement bound of `bound` world units.
    ///
    /// # Errors
    ///
    /// [`NotADisplacement`] unless it is finite and not negative.
    fn try_from(bound: f32) -> Result<Self, NotADisplacement> {
        if bound.is_finite() && bound >= 0.0 {
            Ok(Self(bound))
        } else {
            Err(NotADisplacement(bound))
        }
    }
}

impl Displacement {
    /// No movement.
    pub const ZERO: Self = Self(0.0);

    /// The bound, world units.
    #[must_use]
    pub const fn get(self) -> f32 {
        self.0
    }

    /// A bound on two movements one after the other: the sum, rounded up (the largest finite
    /// `f32` past it).
    #[must_use]
    pub fn plus(self, other: Self) -> Self {
        if other.0 == 0.0 {
            return self;
        }
        if self.0 == 0.0 {
            return other;
        }
        let sum = (self.0 + other.0).next_up();
        Self(if sum.is_finite() { sum } else { f32::MAX })
    }
}

/// One job of the replace pass as the shader reads it: WGSL's `Job`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
struct GpuJob {
    /// The display tile's first vertex in its slot buffer.
    base: u32,
    /// The display tile's vertices per side.
    side: u32,
    /// Its row.
    row: u32,
    /// The finest tile's place in the scratch, in tiles.
    scratch: u32,
    /// The first display vertex the finest tile holds: `i | j << 16` in the display tile.
    lo: u32,
    /// How many display vertices it holds: `along i | along j << 16`.
    count: u32,
    /// The first one's finest vertex: `i | j << 16` in the finest tile.
    fine: u32,
    /// Finest vertices between display vertices, `| ` the finest tile's side `<< 16`.
    step: u32,
}

/// The bytes of one [`GpuJob`].
const JOB_BYTES: u64 = size_of::<GpuJob>() as u64;

/// One stand-in's row as the shaders read it: WGSL's `Row`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub(crate) struct GpuRow {
    /// Its tile's node, or [`NONE`] for a free row.
    pub(crate) node: u32,
    /// Its tile's face, as the node holds it.
    pub(crate) face: u32,
    /// Its tile's place, as the node holds it ([`Place::packed`]).
    pub(crate) place: u32,
    /// Zero.
    pub(crate) reserved: u32,
    /// [`PENDING`].
    pub(crate) flags: u32,
    /// The step's displacement bound, `f32` bits.
    pub(crate) displacement: u32,
    /// The tile's settled bound, `f32` bits.
    pub(crate) error: u32,
    /// The largest distance a replaced vertex moved this frame, `f32` bits.
    pub(crate) moved: u32,
}

impl GpuRow {
    /// A free row.
    pub(crate) const FREE: Self = Self {
        node: NONE,
        face: 0,
        place: 0,
        reserved: 0,
        flags: 0,
        displacement: 0,
        error: 0,
        moved: 0,
    };
}

/// The bytes of one [`GpuRow`].
const ROW_BYTES: u64 = size_of::<GpuRow>() as u64;

/// Where a row's step fields ([`GpuRow::flags`] and [`GpuRow::displacement`]) start: a step on
/// a row writes those alone, leaving the bound the GPU settled.
const ROW_STEP_OFFSET: u64 = 16;

/// What a frame writes into one row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum RowWrite {
    /// The row is the tile at `place`'s, at `node`, made now: its bound before the step is
    /// `stored`, the bound the tile's data was stored with (not a stroke's raise of it, which
    /// would count the stroke's steps twice).
    Fresh {
        /// The tile's node.
        node: u32,
        /// The tile.
        place: Place,
        /// Its stored bound.
        stored: f32,
        /// The step's displacement bound.
        displacement: Displacement,
    },
    /// A step on the row's tile: its bound before it is the row's.
    Step {
        /// The step's displacement bound.
        displacement: Displacement,
    },
    /// The tile at `place`, at `node`, keeps the row, with no step this frame: what a frame
    /// that lost its writes needs written again.
    Kept {
        /// The tile's node.
        node: u32,
        /// The tile.
        place: Place,
    },
    /// The row is free.
    Free,
}

/// The stand-ins' rows, as the CPU keeps them: which tile holds which row.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct RowBook {
    /// Each stand-in's row, by its tile.
    by_place: BTreeMap<Place, u32>,
    /// Rows freed, to reuse before growing.
    free: Vec<u32>,
    /// Rows ever handed out: every row at or past it is free.
    high: u32,
    /// The most rows there are.
    capacity: u32,
}

impl RowBook {
    /// No rows taken, out of `capacity`.
    pub(crate) const fn new(capacity: u32) -> Self {
        Self {
            by_place: BTreeMap::new(),
            free: Vec::new(),
            high: 0,
            capacity,
        }
    }

    /// The row of the tile at `place`, if it stands in.
    pub(crate) fn get(&self, place: Place) -> Option<u32> {
        self.by_place.get(&place).copied()
    }

    /// A new row for the tile at `place`, which has none; `None` if every row is taken.
    pub(crate) fn take(&mut self, place: Place) -> Option<u32> {
        let row = if let Some(row) = self.free.pop() {
            row
        } else if self.high < self.capacity {
            self.high += 1;
            self.high - 1
        } else {
            return None;
        };
        self.by_place.insert(place, row);
        Some(row)
    }

    /// Frees the row of the tile at `place`, returning it, if it had one.
    pub(crate) fn release(&mut self, place: Place) -> Option<u32> {
        let row = self.by_place.remove(&place)?;
        self.free.push(row);
        Some(row)
    }

    /// Every row taken, by its tile.
    pub(crate) fn taken(&self) -> impl Iterator<Item = (Place, u32)> + '_ {
        self.by_place.iter().map(|(place, row)| (*place, *row))
    }

    /// Rows ever handed out: the settle pass runs over these.
    pub(crate) const fn high(&self) -> u32 {
        self.high
    }

    /// Whether no tile stands in.
    pub(crate) fn is_empty(&self) -> bool {
        self.by_place.is_empty()
    }
}

/// One display tile standing in this frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StandIn {
    /// The tile.
    pub(crate) place: Place,
    /// Its slot.
    pub(crate) slot: Slot,
    /// Its row.
    pub(crate) row: u32,
    /// The edited finest tiles under it, by their index in the scratch, with their places.
    pub(crate) under: Vec<(u32, Place)>,
}

/// The stand-in pass: the scratch the frame's edited finest tiles are staged into, the rows,
/// and the two pipelines.
///
/// Owned by residency, on the owner thread. The scratch and the rows are GPU memory outside the
/// slot budget ([`StandInPass::kept_bytes`]).
#[derive(Debug)]
pub(crate) struct StandInPass {
    /// The replace pipeline.
    replace: ComputePipeline,
    /// Its bind group layout.
    replace_layout: BindGroupLayout,
    /// The settle pipeline.
    settle: ComputePipeline,
    /// Its bind group layout.
    settle_layout: BindGroupLayout,
    /// The scratch kept between frames: [`TILE_WORDS`] words per finest tile.
    scratch: Buffer,
    /// How many finest tiles it holds.
    scratch_tiles: u32,
    /// The most finest tiles a scratch can hold: as many as the device binds.
    most_tiles: u32,
    /// A larger scratch for frames that read more finest tiles than the kept one holds: made
    /// as one needs it, grown by half again past the need, dropped after
    /// [`ONE_OFF_FRAMES`] frames without one.
    one_off: Option<Buffer>,
    /// Whether the frame being staged uses the one-off scratch.
    use_one_off: bool,
    /// Frames since one last used the one-off scratch.
    one_off_idle: u32,
    /// The rows.
    rows: Buffer,
    /// The job list the frame's dispatches read, grown as a frame needs.
    jobs: Buffer,
    /// Dispatches' job lists start at multiples of this many bytes (the device's storage offset
    /// alignment).
    job_alignment: u64,
}

impl StandInPass {
    /// The pass, its scratch holding `scratch_tiles` finest tiles and `rows` rows.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`], making the pipelines or the buffers.
    pub(crate) fn new(gpu: &Gpu, scratch_tiles: u32, rows: u32) -> Result<Self, GpuError> {
        Self::build(gpu, scratch_tiles, rows, &[])
    }

    /// [`StandInPass::new`] with each `(from, to)` of `patches` replaced in both passes'
    /// sources, for the tests' bites.
    ///
    /// # Errors
    ///
    /// As [`StandInPass::new`].
    #[cfg(test)]
    pub(crate) fn with_patches(
        gpu: &Gpu,
        scratch_tiles: u32,
        rows: u32,
        patches: &[(&str, &str)],
    ) -> Result<Self, GpuError> {
        Self::build(gpu, scratch_tiles, rows, patches)
    }

    /// The most finest tiles a frame's stand-ins can read, made smaller for the tests of what
    /// happens past it.
    #[cfg(test)]
    pub(crate) const fn set_most_tiles(&mut self, most: u32) {
        self.most_tiles = most;
    }

    /// The pass, its sources with each `(from, to)` of `patches` replaced (none but in the
    /// tests' bites).
    ///
    /// # Errors
    ///
    /// As [`StandInPass::new`].
    fn build(
        gpu: &Gpu,
        scratch_tiles: u32,
        rows: u32,
        patches: &[(&str, &str)],
    ) -> Result<Self, GpuError> {
        let device = gpu.device();
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
        let replace_layout = gpu.scoped("stand-in replace layout", || {
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("stand-in replace"),
                entries: &[
                    storage(0, false),
                    storage(1, true),
                    storage(2, true),
                    storage(3, false),
                ],
            })
        })?;
        let settle_layout = gpu.scoped("stand-in settle layout", || {
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("stand-in settle"),
                entries: &[storage(0, false), storage(1, false)],
            })
        })?;
        let prefix = shared_source();
        let mut replace_source = format!("{prefix}{REPLACE_WGSL}");
        for (from, to) in patches {
            replace_source = replace_source.replace(from, to);
        }
        let replace = pipeline(
            gpu,
            "stand-in replace",
            &replace_source,
            &[&replace_layout],
            "replace",
        )?;
        let mut settle_source = format!("{prefix}{SETTLE_WGSL}");
        for (from, to) in patches {
            settle_source = settle_source.replace(from, to);
        }
        let settle = pipeline(
            gpu,
            "stand-in settle",
            &settle_source,
            &[&settle_layout],
            "settle",
        )?;
        let scratch_tiles = scratch_tiles.max(1);
        let scratch = storage_buffer(
            gpu,
            "stand-in scratch",
            u64::from(scratch_tiles) * TILE_BYTES,
        )?;
        let row_buffer = storage_buffer(gpu, "stand-in rows", u64::from(rows.max(1)) * ROW_BYTES)?;
        // Every row starts free.
        let free = vec![GpuRow::FREE; usize::try_from(rows.max(1)).unwrap_or(1)];
        gpu.queue()
            .write_buffer(&row_buffer, 0, bytemuck::cast_slice(&free));
        let limits = device.limits();
        Ok(Self {
            replace,
            replace_layout,
            settle,
            settle_layout,
            scratch,
            scratch_tiles,
            most_tiles: u32::try_from(
                limits
                    .max_storage_buffer_binding_size
                    .min(limits.max_buffer_size)
                    / TILE_BYTES,
            )
            .unwrap_or(u32::MAX),
            one_off: None,
            use_one_off: false,
            one_off_idle: 0,
            rows: row_buffer,
            jobs: storage_buffer(gpu, "stand-in jobs", 4096)?,
            job_alignment: u64::from(limits.min_storage_buffer_offset_alignment).max(JOB_BYTES),
        })
    }

    /// The most finest tiles a frame's stand-ins can read: as many as a scratch the device
    /// binds holds.
    pub(crate) const fn most_tiles(&self) -> u32 {
        self.most_tiles
    }

    /// Starts a frame whose stand-ins read `tiles` finest tiles (at most
    /// [`StandInPass::most_tiles`]): the kept scratch if it holds them, else the one-off
    /// scratch, made or grown (by half again past the need, up to what the device binds) if
    /// it is too small.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`], making the one-off scratch.
    pub(crate) fn begin(&mut self, gpu: &Gpu, tiles: u32) -> Result<(), GpuError> {
        self.use_one_off = tiles > self.scratch_tiles;
        if !self.use_one_off {
            self.one_off_idle = self.one_off_idle.saturating_add(1);
            if self.one_off_idle > ONE_OFF_FRAMES {
                self.one_off = None;
            }
            return Ok(());
        }
        self.one_off_idle = 0;
        let need = u64::from(tiles) * TILE_BYTES;
        if self
            .one_off
            .as_ref()
            .is_none_or(|buffer| buffer.size() < need)
        {
            let grown = (need + need / 2).min(u64::from(self.most_tiles) * TILE_BYTES);
            let bytes = grown.max(need);
            self.one_off = Some(storage_buffer(gpu, "one-off stand-in scratch", bytes)?);
        }
        Ok(())
    }

    /// Bytes of the one-off scratch, while it is kept.
    pub(crate) fn one_off_bytes(&self) -> u64 {
        self.one_off.as_ref().map_or(0, Buffer::size)
    }

    /// The frame's scratch: the one-off one, if the frame uses it, else the kept one.
    fn frame_scratch(&self) -> &Buffer {
        match &self.one_off {
            Some(one_off) if self.use_one_off => one_off,
            _ => &self.scratch,
        }
    }

    /// Bytes of GPU memory it keeps, outside the slot budget: the scratch, the rows and the job
    /// list.
    pub(crate) fn kept_bytes(&self) -> u64 {
        self.scratch.size() + self.rows.size() + self.jobs.size()
    }

    /// Stages finest tile `index` of the scratch: `positions`, its vertices' positions as
    /// `f32` bits, three words a vertex, row-major, padded to a whole number of 8-byte words as
    /// staging copies need (at most [`TILE_WORDS`] words).
    ///
    /// # Errors
    ///
    /// As [`Batch::write`]; [`GpuError::MapRange`] never (the batch's buffers are mapped).
    pub(crate) fn stage_tile(
        &self,
        gpu: &Gpu,
        batch: &mut Batch,
        ring: &mut StagingRing,
        index: u32,
        positions: &[u32],
    ) -> Result<(), GpuError> {
        batch.write(
            gpu,
            ring,
            bytemuck::cast_slice(positions),
            self.frame_scratch(),
            u64::from(index) * TILE_BYTES,
            true,
        )?;
        Ok(())
    }

    /// Stages `write` into row `row`.
    ///
    /// # Errors
    ///
    /// As [`Batch::write`].
    pub(crate) fn write_row(
        &self,
        gpu: &Gpu,
        batch: &mut Batch,
        ring: &mut StagingRing,
        row: u32,
        write: RowWrite,
    ) -> Result<(), GpuError> {
        let at = u64::from(row) * ROW_BYTES;
        match write {
            RowWrite::Fresh {
                node,
                place,
                stored,
                displacement,
            } => {
                let fresh = GpuRow {
                    node,
                    face: place.face().index(),
                    place: place.packed(),
                    flags: PENDING,
                    displacement: displacement.get().to_bits(),
                    error: stored.to_bits(),
                    ..GpuRow::FREE
                };
                batch.write(gpu, ring, bytemuck::bytes_of(&fresh), &self.rows, at, true)?;
            }
            RowWrite::Step { displacement } => {
                let fields = [PENDING, displacement.get().to_bits()];
                let target = at + ROW_STEP_OFFSET;
                let data = bytemuck::cast_slice(&fields);
                batch.write(gpu, ring, data, &self.rows, target, true)?;
            }
            RowWrite::Kept { node, place } => {
                // Its node, face, place and step fields: not the bound the GPU settled.
                let fields = [node, place.face().index(), place.packed(), 0, 0, 0];
                let data = bytemuck::cast_slice(&fields);
                batch.write(gpu, ring, data, &self.rows, at, true)?;
            }
            RowWrite::Free => {
                let free = bytemuck::bytes_of(&GpuRow::FREE);
                batch.write(gpu, ring, free, &self.rows, at, true)?;
            }
        }
        Ok(())
    }

    /// Ends the frame: stages the job list of `stand_ins` (each tile's slot found by `locate`,
    /// in a project whose finest level is `finest`) into `batch` and returns the frame's work,
    /// the replace pass's dispatches and the settle over the first `rows` rows; `None` if there
    /// is neither.
    ///
    /// # Errors
    ///
    /// As [`Batch::write`]; as [`Gpu::scoped`], growing the job list or making a bind group.
    #[expect(
        clippy::too_many_arguments,
        reason = "a staging write's arguments and what the jobs are built from"
    )]
    pub(crate) fn finish(
        &mut self,
        gpu: &Gpu,
        batch: &mut Batch,
        ring: &mut StagingRing,
        tree: &Buffer,
        finest: Level,
        stand_ins: &[StandIn],
        locate: impl Fn(Slot) -> Option<(Buffer, u64)>,
        rows: u32,
    ) -> Result<Option<StandInWork>, GpuError> {
        if stand_ins.is_empty() && rows == 0 {
            return Ok(None);
        }
        // One dispatch per slot buffer, each at most MAX_GROUPS jobs.
        let mut runs: BTreeMap<(SizeClass, u16), (Buffer, Vec<GpuJob>)> = BTreeMap::new();
        for stand_in in stand_ins {
            let Some((buffer, offset)) = locate(stand_in.slot) else {
                continue;
            };
            let base = offset / size_of::<GpuVertex>() as u64;
            let base = u32::try_from(base).unwrap_or(u32::MAX);
            let jobs = stand_in.under.iter().filter_map(|&(index, finest_place)| {
                job(stand_in, base, index, finest_place, finest)
            });
            let run = runs
                .entry((stand_in.slot.class(), stand_in.slot.buffer()))
                .or_insert_with(|| (buffer, Vec::new()));
            run.1.extend(jobs);
        }
        let max = usize::try_from(MAX_GROUPS).unwrap_or(usize::MAX);
        let align = usize::try_from(self.job_alignment / JOB_BYTES).unwrap_or(1);
        let mut list: Vec<GpuJob> = Vec::new();
        // Per dispatch: its slot buffer, its first job in the list, how many.
        let mut spans: Vec<(&Buffer, u64, u32)> = Vec::new();
        for (target, jobs) in runs.values() {
            for chunk in jobs.chunks(max) {
                let padded = list.len().next_multiple_of(align);
                list.resize(padded, GpuJob::zeroed());
                let start = list.len() as u64 * JOB_BYTES;
                spans.push((
                    target,
                    start,
                    u32::try_from(chunk.len()).unwrap_or(MAX_GROUPS),
                ));
                list.extend_from_slice(chunk);
            }
        }
        let list_bytes = list.len() as u64 * JOB_BYTES;
        if self.jobs.size() < list_bytes {
            self.jobs = storage_buffer(gpu, "stand-in jobs", list_bytes.next_power_of_two())?;
        }
        if !list.is_empty() {
            batch.write(gpu, ring, bytemuck::cast_slice(&list), &self.jobs, 0, true)?;
        }
        let scratch = self.frame_scratch();
        let replaces = spans.iter().map(|&(target, start, count)| {
            let group = gpu.scoped("stand-in replace bind group", || {
                gpu.device().create_bind_group(&BindGroupDescriptor {
                    label: Some("stand-in replace"),
                    layout: &self.replace_layout,
                    entries: &[
                        entry(0, target),
                        entry(1, scratch),
                        BindGroupEntry {
                            binding: 2,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: &self.jobs,
                                offset: start,
                                size: BufferSize::new(u64::from(count) * JOB_BYTES),
                            }),
                        },
                        entry(3, &self.rows),
                    ],
                })
            })?;
            Ok((group, count))
        });
        let replaces: Result<Vec<(BindGroup, u32)>, GpuError> = replaces.collect();
        let settle = gpu.scoped("stand-in settle bind group", || {
            gpu.device().create_bind_group(&BindGroupDescriptor {
                label: Some("stand-in settle"),
                layout: &self.settle_layout,
                entries: &[entry(0, tree), entry(1, &self.rows)],
            })
        })?;
        Ok(Some(StandInWork {
            replace: self.replace.clone(),
            replaces: replaces?,
            settle: self.settle.clone(),
            settle_group: settle,
            settle_groups: rows.div_ceil(GROUP).max(1),
        }))
    }
}

/// The shared declarations with the constants they need, before each pass's own source.
fn shared_source() -> String {
    let everything = FinestNormalCone::EVERYTHING.to_bits();
    format!(
        "{VERTEX_WGSL}{NODE_WGSL}\
         const TILE_WORDS: u32 = {TILE_WORDS}u;\n\
         const PENDING: u32 = {PENDING}u;\n\
         const EVERYTHING: u32 = {everything}u;\n\
         const SLACK: f32 = {SLACK:?};\n\
         {STANDIN_WGSL}"
    )
}

/// The job of `stand_in` and the finest tile at `finest_place`: the display vertices the finest
/// tile holds.
///
/// The display tile's first vertex is at `base` in its slot buffer, and the finest tile is at
/// `index` of the scratch. `None` if the finest tile holds none of its vertices (it isn't under
/// the tile).
fn job(
    stand_in: &StandIn,
    base: u32,
    index: u32,
    finest_place: Place,
    finest: Level,
) -> Option<GpuJob> {
    let place = stand_in.place;
    let level = place.level();
    let below = finest.get().checked_sub(level.get())?;
    let step = 1_u32.checked_shl(u32::from(below))?;
    let quads = level.tile_quads();
    let fine_quads = finest.tile_quads();
    // Each axis: the display vertices (in the tile) and finest vertices (in the finest tile)
    // both hold.
    let axis = |tile: u16, fine_tile: u16| -> Option<(u32, u32, u32)> {
        let x0 = u32::from(tile) * quads;
        let f0 = u32::from(fine_tile) * fine_quads;
        let lo = f0.div_ceil(step).max(x0);
        let hi = ((f0 + fine_quads) / step).min(x0 + quads);
        (lo <= hi).then(|| (lo - x0, hi - lo + 1, lo * step - f0))
    };
    let (lx, nx, fi) = axis(place.tx(), finest_place.tx())?;
    let (ly, ny, fj) = axis(place.ty(), finest_place.ty())?;
    Some(GpuJob {
        base,
        side: quads + 1,
        row: stand_in.row,
        scratch: index,
        lo: lx | (ly << 16),
        count: nx | (ny << 16),
        fine: fi | (fj << 16),
        step: step | ((fine_quads + 1) << 16),
    })
}

/// A frame's stand-in pass, carried by its [`crate::upload::FrameUploads`] to the render thread
/// and encoded after the uploads' copies and the normal pass.
#[derive(Debug)]
pub(crate) struct StandInWork {
    /// The replace pipeline.
    replace: ComputePipeline,
    /// Its dispatches: a bind group and its jobs, one workgroup each.
    replaces: Vec<(BindGroup, u32)>,
    /// The settle pipeline.
    settle: ComputePipeline,
    /// Its bind group.
    settle_group: BindGroup,
    /// Its workgroups.
    settle_groups: u32,
}

impl StandInWork {
    /// Whether it makes a stand-in (else it only settles bounds into the tree).
    pub(crate) fn replaces(&self) -> bool {
        !self.replaces.is_empty()
    }

    /// Encodes the pass into `encoder`: after the copies that bring the scratch, the rows and
    /// the tree, before any pass that reads the slots or the tree.
    pub(crate) fn encode(&self, encoder: &mut CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("stand-ins"),
            timestamp_writes: None,
        });
        if !self.replaces.is_empty() {
            pass.set_pipeline(&self.replace);
            for (group, jobs) in &self.replaces {
                pass.set_bind_group(0, group, &[]);
                pass.dispatch_workgroups(*jobs, 1, 1);
            }
        }
        pass.set_pipeline(&self.settle);
        pass.set_bind_group(0, &self.settle_group, &[]);
        pass.dispatch_workgroups(self.settle_groups, 1, 1);
    }
}
