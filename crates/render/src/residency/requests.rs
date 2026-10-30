//! Requests: the tiles the level choice wanted but found missing, with their screen error,
//! written on the GPU into a request buffer, read back a frame later on the owner thread.

use std::collections::BTreeMap;
use std::sync::mpsc::{self, Receiver, Sender};

use bytemuck::{Pod, Zeroable};
use geom::{Level, TileKey};
use wgpu::{Buffer, BufferDescriptor, BufferUsages};

use crate::gpu::{Gpu, GpuError};
use crate::tree::Place;
use crate::upload::{Readback, RequestCopy};

/// Bytes before the request buffer's entries: the count, the dropped count, two reserved
/// words. Cleared every frame after the readback copy.
pub const REQUEST_HEADER: u64 = 16;

/// Readback buffers in flight at most: a readback is read a frame or two after its frame.
const READBACKS: usize = 3;

/// The WGSL declarations of the request buffer, for the pass that writes it (the level choice).
///
/// A request is appended as
/// `let i = atomicAdd(&requests.count, 1u); if i < arrayLength(&requests.entries) {
/// requests.entries[i] = TileRequest(face, place, screen_error, 0u); } else {
/// atomicAdd(&requests.dropped, 1u); }`, with `place` packed as `render::tree::Place::packed`.
pub const REQUESTS_WGSL: &str = r"
// One tile wanted but not resident (render::residency::GpuRequest): 16 bytes.
struct TileRequest {
    // The base-mesh face.
    face: u32,
    // Level (bits 28..32), column (bits 14..28) and row (bits 0..14), as TileNode.place.
    place: u32,
    // How far, in pixels, what is drawn in its place is from it: larger is fetched first.
    screen_error: f32,
    // Zero.
    reserved: u32,
}

// The request buffer: a count, then the entries. The count and dropped are cleared every frame.
struct TileRequests {
    // Requests appended this frame (may exceed the entries: those past the end are dropped).
    count: atomic<u32>,
    // Requests past the end of the entries.
    dropped: atomic<u32>,
    // Zero.
    reserved0: u32,
    // Zero.
    reserved1: u32,
    // The requests, in no order.
    entries: array<TileRequest>,
}
";

/// One request as the GPU writes it ([`REQUESTS_WGSL`]'s `TileRequest`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct GpuRequest {
    /// The base-mesh face.
    pub face: u32,
    /// The place on it ([`Place::packed`]).
    pub place: u32,
    /// The screen error, pixels.
    pub screen_error: f32,
    /// Zero.
    pub reserved: u32,
}

/// The bytes of one [`GpuRequest`].
const REQUEST_BYTES: u64 = size_of::<GpuRequest>() as u64;

/// One tile wanted but missing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Request {
    /// The tile.
    pub key: TileKey,
    /// How far, in pixels, what is drawn in its place is from it; larger is fetched first.
    pub screen_error: f32,
}

/// The requests of a frame's readback: one per tile (its largest screen error), largest first.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Requests {
    /// The requests, largest screen error first, ties in key order.
    requests: Vec<Request>,
    /// Requests the GPU couldn't fit into the buffer.
    dropped: u32,
}

impl Requests {
    /// Requests from any list: one per tile, with its largest screen error, largest first.
    #[must_use]
    pub fn new(requests: impl IntoIterator<Item = Request>, dropped: u32) -> Self {
        let mut by_key: BTreeMap<TileKey, f32> = BTreeMap::new();
        for request in requests {
            let error = by_key.entry(request.key).or_insert(request.screen_error);
            *error = error.max(request.screen_error);
        }
        let mut sorted: Vec<Request> = by_key
            .into_iter()
            .map(|(key, screen_error)| Request { key, screen_error })
            .collect();
        sorted.sort_by(|a, b| b.screen_error.total_cmp(&a.screen_error));
        Self {
            requests: sorted,
            dropped,
        }
    }

    /// The requests, largest screen error first.
    pub fn iter(&self) -> impl Iterator<Item = &Request> {
        self.requests.iter()
    }

    /// How many tiles are requested.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.requests.len()
    }

    /// Whether no tile is requested.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    /// Requests the GPU had no room for: the buffer is too small for what the frame wanted.
    #[must_use]
    pub const fn dropped(&self) -> u32 {
        self.dropped
    }

    /// These requests and `other`'s together.
    fn merged(self, other: Self) -> Self {
        let dropped = self.dropped.saturating_add(other.dropped);
        Self::new(self.requests.into_iter().chain(other.requests), dropped)
    }
}

/// The request buffer and its readback buffers.
#[derive(Debug)]
pub(crate) struct RequestBuffer {
    /// The buffer the GPU appends requests to.
    buffer: Buffer,
    /// Readback buffers unmapped and free.
    free: Vec<Buffer>,
    /// Readback buffers made.
    made: usize,
    /// Given to each frame's readback, to send its buffer back.
    sender: Sender<Readback>,
    /// Where readback buffers come back: mapped, unused, or lost.
    mapped: Receiver<Readback>,
    /// Requests read back and not taken yet.
    latest: Option<Requests>,
    /// The newest frame whose readback has been read, if any has.
    heard: Option<u64>,
}

impl RequestBuffer {
    /// A request buffer of `capacity` entries, empty.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`].
    pub(crate) fn new(gpu: &Gpu, capacity: u32) -> Result<Self, GpuError> {
        let size = REQUEST_HEADER + u64::from(capacity.max(1)) * REQUEST_BYTES;
        let buffer = gpu.scoped("tile requests", || {
            gpu.device().create_buffer(&BufferDescriptor {
                label: Some("tile requests"),
                size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        })?;
        let (sender, mapped) = mpsc::channel();
        Ok(Self {
            buffer,
            free: Vec::new(),
            made: 0,
            sender,
            mapped,
            latest: None,
            heard: None,
        })
    }

    /// Readback buffers alive: free or in flight.
    #[cfg(test)]
    pub(crate) const fn made(&self) -> usize {
        self.made
    }

    /// Whether every readback buffer made is back and free.
    #[cfg(test)]
    pub(crate) const fn idle(&self) -> bool {
        self.free.len() == self.made
    }

    /// The buffer the GPU appends requests to ([`REQUESTS_WGSL`]'s `TileRequests`).
    pub(crate) const fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// The readback this frame makes, if a readback buffer is free (else the requests wait in
    /// the buffer for the next frame's).
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`], making a readback buffer.
    pub(crate) fn copy(&mut self, gpu: &Gpu, frame: u64) -> Result<Option<RequestCopy>, GpuError> {
        let target = match self.free.pop() {
            Some(buffer) => buffer,
            None if self.made < READBACKS => {
                let size = self.buffer.size();
                let buffer = gpu.scoped("tile request readback", || {
                    gpu.device().create_buffer(&BufferDescriptor {
                        label: Some("tile request readback"),
                        size,
                        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    })
                })?;
                self.made += 1;
                buffer
            }
            None => return Ok(None),
        };
        Ok(Some(RequestCopy {
            source: self.buffer.clone(),
            target,
            done: self.sender.clone(),
            frame,
        }))
    }

    /// Reads the readback buffers whose maps completed (the device is polled by the caller),
    /// takes back those of frames dropped before submission,
    /// keeping their requests for [`RequestBuffer::take`], and frees them. Places past
    /// `finest`, or off their face, are skipped.
    ///
    /// # Errors
    ///
    /// [`GpuError::MapRange`] if a mapped buffer can't be viewed.
    pub(crate) fn collect(&mut self, finest: Level) -> Result<(), GpuError> {
        while let Ok(back) = self.mapped.try_recv() {
            let buffer = match back {
                Readback::Mapped(buffer, frame) => {
                    self.heard = self.heard.max(Some(frame));
                    buffer
                }
                Readback::Unused(buffer) => {
                    self.free.push(buffer);
                    continue;
                }
                Readback::Skipped(buffer, frame) => {
                    // Heard of, so settling can end, but nothing asked for: the tiles wanted
                    // stay as the last frame with a level choice left them.
                    self.heard = self.heard.max(Some(frame));
                    self.free.push(buffer);
                    continue;
                }
                Readback::Failed => {
                    self.made -= 1;
                    continue;
                }
            };
            let read = read_requests(&buffer, finest);
            buffer.unmap();
            self.free.push(buffer);
            let read = read?;
            self.latest = Some(match self.latest.take() {
                Some(earlier) => earlier.merged(read),
                None => read,
            });
        }
        Ok(())
    }

    /// The newest frame whose readback has been read, if any has.
    pub(crate) const fn heard(&self) -> Option<u64> {
        self.heard
    }

    /// The requests read back since the last call, if any readback arrived.
    pub(crate) const fn take(&mut self) -> Option<Requests> {
        self.latest.take()
    }
}

/// The requests in a mapped readback buffer.
///
/// # Errors
///
/// [`GpuError::MapRange`] if it can't be viewed.
fn read_requests(buffer: &Buffer, finest: Level) -> Result<Requests, GpuError> {
    let view = buffer.get_mapped_range(..)?;
    let words: &[u32] = bytemuck::try_cast_slice(&view).unwrap_or(&[]);
    let count = words.first().copied().unwrap_or(0);
    let dropped = words.get(1).copied().unwrap_or(0);
    let entries: &[GpuRequest] = view
        .get(usize::try_from(REQUEST_HEADER).unwrap_or(usize::MAX)..)
        .and_then(|bytes| bytemuck::try_cast_slice(bytes).ok())
        .unwrap_or(&[]);
    let written = entries
        .iter()
        .take(usize::try_from(count).unwrap_or(usize::MAX));
    let requests = written.filter_map(|entry| {
        let key = Place::from_packed(entry.face, entry.place)?.key(finest)?;
        let screen_error = entry.screen_error;
        screen_error
            .is_finite()
            .then_some(Request { key, screen_error })
    });
    Ok(Requests::new(requests, dropped))
}
