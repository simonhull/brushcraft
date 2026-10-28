//! Uploads: tile data read on the store's owner thread, written into a staging ring it owns,
//! and handed to the render thread once per frame (plan decision 5).
//!
//! The owner thread reads each tile from a [`TileSource`] (the store, through [`StoreSource`])
//! as [`GpuVertex`]es and writes them, with the tile tree's changed nodes, into staging buffers
//! mapped for writing. When the frame's uploads are staged, the buffers are unmapped and the
//! copies they need become a [`FrameUploads`], which the render thread encodes into the frame's
//! command buffer ahead of every pass ([`FrameUploads::encode`]) and submits
//! ([`FrameUploads::submit`]): each staging buffer is then mapped for writing again and returns
//! to the ring when the GPU is done with it, without the owner thread ever waiting on the GPU.
//! A frame dropped instead (staging failed, or the render thread gave up on it) gives its
//! buffers back too.
//!
//! Everything a frame stages goes into one submission, so a group of tiles staged together
//! (an epoch, [`crate::residency`]) and the tree nodes pointing at them switch at once.

use std::collections::BTreeMap;
use std::sync::mpsc::{self, Receiver, Sender};

use bytemuck::{Pod, Zeroable};
use geom::{Channel, DisplayTile, TileKey, Vec3};
use store::{CacheError, Evaluation, TileStore};
use wgpu::{Buffer, BufferDescriptor, BufferUsages, CommandEncoder, MapMode, Queue};

use crate::gpu::{Gpu, GpuError};
use crate::residency::Requests;

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;

/// One vertex of a resident tile as the GPU reads it: 32 bytes, two `vec4`s.
///
/// Positions are decoded `f32` (plan decision 1). Normals are unit normals: a display tile's
/// averaged normal, unpacked; a finest tile's normal as the store computes it.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default, Pod, Zeroable)]
pub struct GpuVertex {
    /// The position, world units.
    pub position: [f32; 3],
    /// The roughness term: the mean normal's length, `0..=1` (1 for a finest tile).
    pub roughness: f32,
    /// The unit normal.
    pub normal: [f32; 3],
    /// Zero.
    pub reserved: u32,
}

/// The bytes of one [`GpuVertex`].
pub const VERTEX_BYTES: u64 = size_of::<GpuVertex>() as u64;

/// The WGSL declaration of [`GpuVertex`], for the passes that read slots.
pub const VERTEX_WGSL: &str = r"
// One vertex of a resident tile (render::upload::GpuVertex): 32 bytes.
struct TileVertex {
    // The position, world units.
    position: vec3<f32>,
    // The roughness term, 0..1 (1 for a finest tile).
    roughness: f32,
    // The unit normal.
    normal: vec3<f32>,
    // Zero.
    reserved: u32,
}
";

/// Where the data of resident tiles comes from: the store, or a model of it in tests.
///
/// Read on the owner thread, a tile at a time, while the frame's uploads are staged.
pub trait TileSource {
    /// Why a tile couldn't be read.
    type Error;

    /// Readies `keys`, about to be read in this order: evaluates finest tiles together, reads
    /// display tiles in one batch. Optional: [`TileSource::read`] must work without it.
    ///
    /// # Errors
    ///
    /// As the source reports them; staging stops there.
    #[expect(
        unused_variables,
        reason = "the default readies nothing; a source that batches reads overrides it"
    )]
    fn prepare(&mut self, keys: &[TileKey]) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Writes `key`'s vertices into `out` (row-major `side²`, as many as `out` holds) and
    /// returns its bound `e`, or `None` if the source has nothing to draw for it yet (a display
    /// tile not built, or made unknown by edits): it is then drawn finer, or by an ancestor.
    ///
    /// # Errors
    ///
    /// As the source reports them; staging stops there.
    fn read(&mut self, key: TileKey, out: &mut [GpuVertex]) -> Result<Option<f32>, Self::Error>;
}

/// [`TileSource`] over the store: display tiles from [`TileStore::display_for_draw`], finest
/// tiles from [`TileStore::normals`] and [`TileStore::evaluate`].
///
/// A finest tile uploads its positions and the store's normals. Plan decision 1 wants its
/// positions with the one-vertex ring instead, the normals computed on the GPU, but the store
/// has no read of a tile with its ring (corner rings at base vertices included); until it does,
/// the store's normals are what both sides of every seam agree on (it gives every copy of a
/// shared vertex the same bits).
#[derive(Debug)]
pub struct StoreSource<'a> {
    /// The store read.
    store: &'a mut TileStore,
    /// Display tiles read ahead by [`TileSource::prepare`], until [`TileSource::read`] takes
    /// them.
    display: BTreeMap<TileKey, Option<DisplayTile>>,
}

impl<'a> StoreSource<'a> {
    /// Reads tiles from `store`.
    pub const fn new(store: &'a mut TileStore) -> Self {
        Self {
            store,
            display: BTreeMap::new(),
        }
    }
}

impl TileSource for StoreSource<'_> {
    type Error = CacheError;

    fn prepare(&mut self, keys: &[TileKey]) -> Result<(), CacheError> {
        let (display, finest): (Vec<TileKey>, Vec<TileKey>) = keys
            .iter()
            .partition(|key| key.channel() == Channel::DISPLAY);
        if !finest.is_empty() {
            self.store.evaluate_many(&finest, Evaluation::Normals)?;
        }
        if !display.is_empty() {
            let tiles = self.store.display_for_draw(&display)?;
            self.display.extend(display.into_iter().zip(tiles));
        }
        Ok(())
    }

    fn read(&mut self, key: TileKey, out: &mut [GpuVertex]) -> Result<Option<f32>, CacheError> {
        if key.channel() == Channel::DISPLAY {
            let tile = match self.display.remove(&key) {
                Some(tile) => tile,
                None => self.store.display_for_draw(&[key])?.pop().flatten(),
            };
            let Some(tile) = tile else {
                return Ok(None);
            };
            let vertices = out
                .iter_mut()
                .zip(tile.positions())
                .zip(tile.normals())
                .zip(tile.roughness());
            for (((vertex, position), normal), roughness) in vertices {
                *vertex = GpuVertex {
                    position: components(*position),
                    roughness: f32::from(*roughness) / f32::from(u8::MAX),
                    normal: components(*normal),
                    reserved: 0,
                };
            }
            return Ok(Some(tile.error()));
        }
        let normals = self.store.normals(key)?;
        let with_normals = out.iter_mut().zip(normals);
        for (vertex, normal) in with_normals {
            vertex.normal = components(*normal);
            vertex.roughness = 1.0;
            vertex.reserved = 0;
        }
        let positions = self.store.evaluate(key)?;
        let with_positions = out.iter_mut().zip(positions);
        for (vertex, position) in with_positions {
            vertex.position = components(*position);
        }
        Ok(Some(0.0))
    }
}

/// `v`'s components.
const fn components(v: Vec3) -> [f32; 3] {
    [v.x, v.y, v.z]
}

/// Asks the store to page in the correction tiles of the finest tiles among `requests`.
///
/// Reading them later then finds them resident: what the owner thread does with the
/// requests a frame's readback gave ([`crate::residency::Residency::take_requests`]) before
/// handing them to [`crate::residency::Residency::want`]. Returns how many reads were queued.
///
/// # Errors
///
/// As [`TileStore::prefetch`].
pub fn prefetch_requests(store: &mut TileStore, requests: &Requests) -> Result<usize, CacheError> {
    let finest: Vec<TileKey> = requests
        .iter()
        .map(|request| request.key)
        .filter(|key| key.channel() != Channel::DISPLAY)
        .collect();
    store.prefetch(&finest)
}

/// A staging buffer coming back to the ring: from the render thread's map callback once the
/// GPU is done with it, or from a frame dropped before submission.
#[derive(Debug)]
struct Returned {
    /// The buffer.
    buffer: Buffer,
    /// Whether it is mapped for writing (else mapping failed, and it is dropped).
    mapped: bool,
}

/// The staging ring: buffers mapped for writing, owned by the owner thread between frames,
/// lent to the render thread for one submission each.
#[derive(Debug)]
pub(crate) struct StagingRing {
    /// Bytes per ring buffer.
    buffer_bytes: u64,
    /// The most bytes of ring buffers alive at once.
    cap: u64,
    /// Bytes of ring buffers alive: free, being written, or in flight.
    alive: u64,
    /// Ring buffers mapped and free.
    ready: Vec<Buffer>,
    /// Where buffers come back.
    returns: Receiver<Returned>,
    /// Given to each frame, to send its buffers back.
    sender: Sender<Returned>,
}

impl StagingRing {
    /// An empty ring of `buffer_bytes` buffers, at most `cap` bytes of them; buffers are created
    /// as they are first needed.
    pub(crate) fn new(buffer_bytes: u64, cap: u64) -> Self {
        let (sender, returns) = mpsc::channel();
        Self {
            buffer_bytes,
            cap,
            alive: 0,
            ready: Vec::new(),
            returns,
            sender,
        }
    }

    /// Takes back the buffers that came back (the device is polled by the caller, for the map
    /// callbacks to run).
    pub(crate) fn collect(&mut self) {
        while let Ok(returned) = self.returns.try_recv() {
            if returned.mapped {
                self.ready.push(returned.buffer);
            } else {
                self.alive = self.alive.saturating_sub(returned.buffer.size());
            }
        }
    }

    /// Bytes of ring buffers alive: free, being written, or in flight.
    #[cfg(test)]
    pub(crate) const fn alive(&self) -> u64 {
        self.alive
    }

    /// Whether every ring buffer alive is back and free (none in flight, none lost).
    #[cfg(test)]
    pub(crate) fn idle(&self) -> bool {
        self.ready.len() as u64 * self.buffer_bytes == self.alive
    }

    /// A free ring buffer, mapped: one that came back, else a new one if the cap allows;
    /// `None` if all are in use.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`], creating one.
    fn take(&mut self, gpu: &Gpu) -> Result<Option<Buffer>, GpuError> {
        if let Some(buffer) = self.ready.pop() {
            return Ok(Some(buffer));
        }
        if self.alive + self.buffer_bytes > self.cap {
            return Ok(None);
        }
        let buffer = create_staging(gpu, self.buffer_bytes, "staging")?;
        self.alive += self.buffer_bytes;
        Ok(Some(buffer))
    }

    /// The bytes the frame could still take from the ring: free buffers and those the cap
    /// allows creating.
    fn room(&self) -> u64 {
        let free = self.ready.len() as u64 * self.buffer_bytes;
        free + self.cap.saturating_sub(self.alive)
    }
}

/// A new staging buffer of `size` bytes, mapped for writing.
///
/// # Errors
///
/// As [`Gpu::scoped`].
fn create_staging(gpu: &Gpu, size: u64, label: &str) -> Result<Buffer, GpuError> {
    gpu.scoped(label, || {
        gpu.device().create_buffer(&BufferDescriptor {
            label: Some(label),
            size,
            usage: BufferUsages::MAP_WRITE | BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        })
    })
}

/// A staging buffer a frame writes into.
#[derive(Debug)]
struct Staging {
    /// The buffer, mapped while the frame is staged.
    buffer: Buffer,
    /// Made for this frame alone, past the ring's cap (a group too large for the ring that
    /// must go whole): dropped after the submission, never returned.
    one_off: bool,
}

/// One copy from a staging buffer to a slot or the tree.
#[derive(Debug)]
struct CopyOp {
    /// The staging buffer, by index in the frame's.
    staging: usize,
    /// Where in it.
    offset: u64,
    /// The buffer written.
    target: Buffer,
    /// Where in it.
    target_offset: u64,
    /// Bytes.
    size: u64,
}

/// What the frame being staged has written so far.
///
/// Dropped before [`Batch::finish`] (staging failed), it gives its ring buffers back, still
/// mapped, and nothing it staged reaches the GPU.
#[derive(Debug)]
pub(crate) struct Batch {
    /// Its staging buffers, mapped, the last one being filled.
    staging: Vec<Staging>,
    /// Bytes used in the last staging buffer.
    cursor: u64,
    /// Its copies, in the order written.
    copies: Vec<CopyOp>,
    /// Bytes staged.
    bytes: u64,
    /// Where its ring buffers go back if it is dropped.
    returns: Sender<Returned>,
}

impl Batch {
    /// An empty batch drawing on `ring`.
    pub(crate) fn new(ring: &StagingRing) -> Self {
        Self {
            staging: Vec::new(),
            cursor: 0,
            copies: Vec::new(),
            bytes: 0,
            returns: ring.sender.clone(),
        }
    }

    /// Bytes the frame could still stage without going past the ring's cap: what's left in its
    /// current staging buffer and what the ring could give.
    pub(crate) fn room(&self, ring: &StagingRing) -> u64 {
        let current = self
            .staging
            .last()
            .map_or(0, |s| s.buffer.size().saturating_sub(self.cursor));
        current + ring.room()
    }

    /// Bytes staged so far.
    pub(crate) const fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Stages `data` to be copied into `target` at `target_offset`; whether it was. Without
    /// `forced`, `false` when the ring has no room; with it, a buffer is made past the cap.
    ///
    /// `data` and both offsets are whole multiples of 8 bytes (every tile slot and tree node
    /// is), as mapping and copying need.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`], making a staging buffer; [`GpuError::MapRange`] if the mapped
    /// range can't be viewed (a buffer not mapped: it always is here).
    pub(crate) fn write(
        &mut self,
        gpu: &Gpu,
        ring: &mut StagingRing,
        data: &[u8],
        target: &Buffer,
        target_offset: u64,
        forced: bool,
    ) -> Result<bool, GpuError> {
        let size = data.len() as u64;
        let Some((staging, offset)) = self.reserve(gpu, ring, size, forced)? else {
            return Ok(false);
        };
        let Some(chosen) = self.staging.get(staging) else {
            return Ok(false);
        };
        chosen
            .buffer
            .get_mapped_range_mut(offset..offset + size)?
            .copy_from_slice(data);
        self.copies.push(CopyOp {
            staging,
            offset,
            target: target.clone(),
            target_offset,
            size,
        });
        self.bytes += size;
        Ok(true)
    }

    /// Room for `size` bytes: the staging buffer's index and the offset in it.
    ///
    /// # Errors
    ///
    /// As [`Batch::write`].
    fn reserve(
        &mut self,
        gpu: &Gpu,
        ring: &mut StagingRing,
        size: u64,
        forced: bool,
    ) -> Result<Option<(usize, u64)>, GpuError> {
        if let Some(last) = self.staging.last()
            && last.buffer.size() - self.cursor >= size
        {
            let at = self.cursor;
            self.cursor += size;
            return Ok(Some((self.staging.len() - 1, at)));
        }
        let fresh = if size <= ring.buffer_bytes {
            ring.take(gpu)?.map(|buffer| Staging {
                buffer,
                one_off: false,
            })
        } else {
            None
        };
        let fresh = match fresh {
            Some(staging) => staging,
            None if forced => Staging {
                buffer: create_staging(gpu, size.max(ring.buffer_bytes), "one-off staging")?,
                one_off: true,
            },
            None => return Ok(None),
        };
        self.staging.push(fresh);
        self.cursor = size;
        Ok(Some((self.staging.len() - 1, 0)))
    }

    /// Ends the frame's staging: unmaps its buffers and hands them, with their copies and the
    /// request readback, to the render thread as frame `frame`, whose outcome goes to
    /// `outcomes`.
    pub(crate) fn finish(
        mut self,
        frame: u64,
        requests: Option<RequestCopy>,
        outcomes: Sender<FrameOutcome>,
    ) -> FrameUploads {
        let staging = std::mem::take(&mut self.staging);
        for s in &staging {
            s.buffer.unmap();
        }
        FrameUploads {
            frame: Some(frame),
            staging,
            copies: std::mem::take(&mut self.copies),
            bytes: self.bytes,
            requests,
            returns: self.returns.clone(),
            outcomes,
        }
    }
}

impl Drop for Batch {
    fn drop(&mut self) {
        // Not finished: its ring buffers go back as they are, still mapped.
        for staging in self.staging.drain(..) {
            if !staging.one_off {
                drop(self.returns.send(Returned {
                    buffer: staging.buffer,
                    mapped: true,
                }));
            }
        }
    }
}

/// A readback buffer coming back to the request buffer.
#[derive(Debug)]
pub(crate) enum Readback {
    /// Mapped for reading, its frame's requests in it.
    Mapped(Buffer),
    /// Unmapped and unused: its frame was dropped before submission.
    Unused(Buffer),
    /// Mapping failed; the buffer is gone.
    Failed,
}

/// The request readback a frame makes: the buffer the level choice writes its wanted tiles
/// into, copied to a readback buffer the owner thread reads a frame later.
#[derive(Debug)]
pub(crate) struct RequestCopy {
    /// The GPU's request buffer.
    pub(crate) source: Buffer,
    /// The readback buffer, unmapped, for this frame alone.
    pub(crate) target: Buffer,
    /// Where the readback buffer goes back.
    pub(crate) done: Sender<Readback>,
}

/// What became of a frame's uploads, told to the residency that staged them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FrameOutcome {
    /// Frame `0` was submitted.
    Submitted(u64),
    /// Frame `0` was dropped before submission: nothing it staged reached the GPU.
    Lost(u64),
}

/// One frame's uploads, staged by the owner thread, for the render thread to encode and submit.
///
/// It is [`Send`]: made on the owner thread, consumed on the render thread. Frames are
/// submitted in the order they were staged. One dropped without [`FrameUploads::submit`] gives
/// its buffers back and tells its residency, which drops the tiles the frame wrote (they never
/// reached the GPU) and rewrites the tile tree whole.
#[derive(Debug)]
#[must_use = "the frame's uploads must be encoded and submitted, or the tiles never arrive"]
pub struct FrameUploads {
    /// The frame number the residency staged it as; `None` once submitted.
    frame: Option<u64>,
    /// Its staging buffers, unmapped.
    staging: Vec<Staging>,
    /// Its copies, in order.
    copies: Vec<CopyOp>,
    /// Bytes staged.
    bytes: u64,
    /// The request readback, if a readback buffer was free.
    requests: Option<RequestCopy>,
    /// Where staging buffers go back to the ring.
    returns: Sender<Returned>,
    /// Where the frame's outcome goes.
    outcomes: Sender<FrameOutcome>,
}

impl FrameUploads {
    /// Bytes staged: tile data and tree nodes.
    #[must_use]
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Whether nothing was staged.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.copies.is_empty()
    }

    /// Encodes the copies into tile slots and the tile tree: first in the frame's command
    /// buffer, before any pass reads them.
    pub fn encode(&self, encoder: &mut CommandEncoder) {
        for copy in &self.copies {
            if let Some(staging) = self.staging.get(copy.staging) {
                encoder.copy_buffer_to_buffer(
                    &staging.buffer,
                    copy.offset,
                    &copy.target,
                    copy.target_offset,
                    copy.size,
                );
            }
        }
    }

    /// Ends the frame: encodes the request readback after the passes already in `encoder` (the
    /// request buffer is copied out, then its count cleared for the next frame), submits the
    /// command buffer, and only then maps the staging buffers for writing again (they return
    /// to the ring when the GPU is done with them) and the readback buffer for reading. The
    /// callbacks run when the device is polled ([`crate::residency::Residency::begin_frame`]
    /// does).
    ///
    /// `encoder` must hold [`FrameUploads::encode`]'s copies first.
    pub fn submit(mut self, queue: &Queue, mut encoder: CommandEncoder) {
        if let Some(requests) = &self.requests {
            encoder.copy_buffer_to_buffer(
                &requests.source,
                0,
                &requests.target,
                0,
                requests.source.size(),
            );
            encoder.clear_buffer(&requests.source, 0, Some(crate::residency::REQUEST_HEADER));
        }
        queue.submit([encoder.finish()]);
        for staging in std::mem::take(&mut self.staging) {
            if !staging.one_off {
                remap(&staging.buffer, self.returns.clone());
            }
        }
        if let Some(requests) = self.requests.take() {
            let RequestCopy { target, done, .. } = requests;
            let buffer = target.clone();
            target.map_async(MapMode::Read, .., move |result| {
                let back = if result.is_ok() {
                    Readback::Mapped(buffer)
                } else {
                    Readback::Failed
                };
                drop(done.send(back));
            });
        }
        if let Some(frame) = self.frame.take() {
            // A residency that's gone has nothing to hear.
            self.outcomes
                .send(FrameOutcome::Submitted(frame))
                .unwrap_or(());
        }
    }
}

/// Maps `buffer` for writing again, to go back through `returns` once mapped.
fn remap(buffer: &Buffer, returns: Sender<Returned>) {
    let back = buffer.clone();
    buffer.map_async(MapMode::Write, .., move |result| {
        // A ring that's gone (the residency dropped) has nothing to take it back.
        drop(returns.send(Returned {
            buffer: back,
            mapped: result.is_ok(),
        }));
    });
}

impl Drop for FrameUploads {
    fn drop(&mut self) {
        let Some(frame) = self.frame.take() else {
            // Submitted: everything went with it.
            return;
        };
        // Never submitted: its buffers go back unused, and its residency hears of the loss.
        for staging in self.staging.drain(..) {
            if !staging.one_off {
                remap(&staging.buffer, self.returns.clone());
            }
        }
        if let Some(requests) = self.requests.take() {
            drop(requests.done.send(Readback::Unused(requests.target)));
        }
        // A residency that's gone has nothing to hear.
        self.outcomes.send(FrameOutcome::Lost(frame)).unwrap_or(());
    }
}
