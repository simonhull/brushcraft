//! Uploads: tile data read on the store's owner thread, written into a staging ring it owns,
//! and handed to the render thread once per frame (plan decision 5).
//!
//! The owner thread reads each tile from a [`TileSource`] (the store, through [`StoreSource`])
//! into a [`TileData`]: a display tile as [`GpuVertex`]es, a finest tile as its positions with
//! the one-vertex ring (a ring record, from which the GPU computes its vertices: the crate's
//! `normals` module). A requested finest tile is read from the copy the store's queued
//! evaluation made in an earlier frame's jobs ([`FinestEvaluations`]), or evaluated while
//! staging only if the time left before the staging deadline holds it
//! ([`StoreSource::evaluating_until`]). The owner writes the tiles, with the tile tree's changed
//! nodes, into staging buffers mapped for writing. When the frame's uploads are staged, the buffers
//! are unmapped and the copies they need become a [`FrameUploads`], which the render thread encodes
//! into the frame's command buffer ahead of every pass, the normal pass right after them, and
//! submits ([`FrameUploads::submit`], which takes the passes, so they can only follow the copies
//! and the normal pass): each staging buffer is then mapped for writing
//! again and returns to the ring when the GPU is done with it, without the owner thread ever
//! waiting on the GPU. A frame dropped instead (staging failed, or the render thread gave up on it)
//! gives its buffers back too.
//!
//! Everything a frame stages goes into one submission, so a group of tiles staged together
//! (an epoch, [`crate::residency`]) and the tree nodes pointing at them switch at once.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use geom::{Channel, DisplayTile, FinestNormalCone, TileKey, Vec3};
use store::{CacheError, EvaluatedTile, Evaluation, TileStore};
use wgpu::{Buffer, BufferDescriptor, BufferUsages, CommandEncoder, MapMode, Queue};

use crate::gpu::{Gpu, GpuError};
use crate::normals::{NormalWork, RingRecord};
use crate::residency::{Request, Requests};

mod finest;
#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;

pub use finest::FinestEvaluations;

/// One vertex of a resident tile as the GPU reads it: 32 bytes, two `vec4`s.
///
/// Positions are decoded `f32` (plan decision 1). Normals are unit normals: a display tile's
/// averaged normal, unpacked; a finest tile's computed on the GPU from its positions and ring
/// (the crate's `normals` module), by the formula the store's normals use.
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

/// What a tile read gives the level choice beside its vertices: how far what the tile draws may
/// be from the finest surface under it, and which way that surface faces.
///
/// The two change together, so they travel together from the read to the tile tree's node
/// ([`crate::tree::GpuNode::error`] and [`crate::tree::GpuNode::finest_cone`]): a dab raises
/// `e` and makes the cone cover everything, and the refresh rebuilds both, each in an epoch
/// of the stale-tile feed naming the tile, so residency reads it again and switches both.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TileDetail {
    /// `e`: a bound, in world units, on the distance from what the tile draws to the finest
    /// surface under it; 0 for a finest tile.
    pub error: f32,
    /// The cone holding every finest normal under the tile, as its display record stores it
    /// (covering everything while a dab has raised its bound).
    /// [`FinestNormalCone::EVERYTHING`] for a finest tile (the level choice tests a finest
    /// tile by its own triangles, which are the finest surface) and for any tile whose finest
    /// normals aren't known: never facing away, so refined whenever its error asks.
    pub cone: FinestNormalCone,
}

impl TileDetail {
    /// A finest tile's: `e` 0, the cone that covers everything.
    pub const FINEST: Self = Self {
        error: 0.0,
        cone: FinestNormalCone::EVERYTHING,
    };

    /// A display tile's: its `e` and its finest normals' cone.
    #[must_use]
    pub const fn of_display(tile: &DisplayTile) -> Self {
        Self {
            error: tile.error(),
            cone: tile.cone(),
        }
    }
}

/// What a [`TileSource`] writes one tile into: its vertices, or a finest tile's positions with
/// its ring.
///
/// Residency hands it over for a tile of `side²` vertices; a source fills them
/// ([`TileData::vertices`]) or writes the ring instead ([`TileData::ring`]), and whichever it
/// did last is what is uploaded (zeroed vertices if it did neither). The vertices are zeroed only
/// when they are asked for, so a finest tile's read never touches them.
#[derive(Debug, Default)]
pub struct TileData {
    /// The vertices, row-major: `side²` of them once asked for in this read.
    vertices: Vec<GpuVertex>,
    /// How many vertices the tile has.
    count: usize,
    /// The ring's record.
    ring: RingRecord,
    /// What the source wrote in this read.
    written: Written,
}

/// What a source wrote into a [`TileData`] in one read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum Written {
    /// Nothing yet: the tile uploads zeroed vertices.
    #[default]
    Nothing,
    /// The vertices.
    Vertices,
    /// A finest tile's ring.
    Ring,
}

/// What one read of a [`TileData`] uploads.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Upload<'a> {
    /// Vertices, copied into the slot as they are: their bytes.
    Vertices(&'a [u8]),
    /// A finest tile's record, which the normal pass turns into the slot's vertices.
    Ring(&'a RingRecord),
}

impl TileData {
    /// Starts a read of a tile of `count` vertices: nothing written.
    pub(crate) const fn reset(&mut self, count: usize) {
        self.count = count;
        self.written = Written::Nothing;
    }

    /// The tile's vertices, row-major `side²`, zeroed at the first call of a read, to fill: they
    /// are what is uploaded.
    pub fn vertices(&mut self) -> &mut [GpuVertex] {
        if self.written != Written::Vertices {
            self.vertices.clear();
            self.vertices.resize(self.count, GpuVertex::default());
            self.written = Written::Vertices;
        }
        &mut self.vertices
    }

    /// Writes a finest tile as `block`, its positions with its one-vertex ring and its
    /// base-vertex rings, gathered over `mesh`: that is what is uploaded, and the GPU computes the
    /// tile's vertices from it. `block` must be of the tile's side.
    pub fn ring(&mut self, mesh: &geom::BaseMesh, block: &geom::GatheredBlock) {
        self.ring.write(mesh, block);
        self.written = Written::Ring;
    }

    /// What this read uploads: the ring's record if the ring was written last, else the
    /// vertices (zeroed if nothing was written).
    pub(crate) fn upload(&mut self) -> Upload<'_> {
        if self.written == Written::Ring {
            return Upload::Ring(&self.ring);
        }
        Upload::Vertices(bytemuck::cast_slice(self.vertices()))
    }

    /// The ring's record, if the ring is what was written, for tests that change it.
    #[cfg(test)]
    pub(crate) fn ring_record_mut(&mut self) -> Option<&mut RingRecord> {
        (self.written == Written::Ring).then_some(&mut self.ring)
    }
}

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

    /// Whether display tile `key` reads now without waiting on a file: its data is in memory,
    /// or was read ahead ([`prefetch_requests`]). Such a tile is staged in the frame it is first
    /// requested in; any other waits a frame for its read ahead to land. The default says no:
    /// every tile waits.
    #[expect(
        unused_variables,
        reason = "the default knows of no data in memory; a source that does overrides it"
    )]
    fn at_hand(&mut self, key: TileKey) -> bool {
        false
    }

    /// Whether requested tile `key` is being made elsewhere and isn't ready yet, so reading it
    /// now would block on that work: a finest tile in the store's queued evaluation
    /// ([`FinestEvaluations`]). Such a tile is never read for its request; it stays wanted, and
    /// is staged in a later frame, once this says no. Tiles a pending epoch names are read
    /// whatever this says, since the epoch switches whole. The default says no: nothing is made
    /// elsewhere.
    #[expect(
        unused_variables,
        reason = "the default makes nothing elsewhere; a source that does overrides it"
    )]
    fn deferred(&mut self, key: TileKey) -> bool {
        false
    }

    /// Writes `key` into `out`, its vertices or (a finest tile) its positions with the ring, and
    /// returns its bound `e` and its finest normals' cone, or `None` if the source has nothing
    /// to draw for it yet (a display tile not built, or made unknown by edits): it is then drawn
    /// finer, or by an ancestor.
    ///
    /// # Errors
    ///
    /// As the source reports them; staging stops there.
    fn read(&mut self, key: TileKey, out: &mut TileData)
    -> Result<Option<TileDetail>, Self::Error>;
}

/// [`TileSource`] over the store: display tiles from [`TileStore::display_for_draw`], finest
/// tiles from the copies the store's queued evaluation made ([`FinestEvaluations`]) or from
/// [`TileStore::positions_with_ring`].
///
/// A finest tile uploads its positions with the one-vertex ring and its corners' base-vertex
/// rings (plan decision 1), every vertex its owner's bits, and the GPU computes its normals.
///
/// Built [`with_evaluations`](StoreSource::with_evaluations), a requested finest tile is read
/// from its evaluated copy if one is held, and else deferred until there is one: the owner
/// never evaluates it while staging. Given a deadline as well
/// ([`evaluating_until`](StoreSource::evaluating_until)), a requested finest tile without a
/// copy is evaluated here while the time left before the deadline holds it, at the cost
/// measured lately (with [`STAGE_MARGIN`] kept free), and withdrawn from the queue; the rest
/// are deferred. Finest tiles a pending epoch names (resident tiles an edit changed) are still
/// evaluated here, together, since an epoch switches whole in the frame its last tile is
/// staged; after a brush step they are mostly cached already. Built [`new`](StoreSource::new),
/// every finest tile is evaluated here.
#[derive(Debug)]
pub struct StoreSource<'a> {
    /// The store read.
    store: &'a mut TileStore,
    /// Display tiles read ahead by [`TileSource::prepare`], until [`TileSource::read`] takes
    /// them.
    display: BTreeMap<TileKey, Option<DisplayTile>>,
    /// The finest tiles the store's queued evaluation made, if requested finest tiles are read
    /// from them first.
    evaluations: Option<&'a mut FinestEvaluations>,
    /// The deadline requested finest tiles without a copy may be evaluated here until, if any
    /// may.
    until: Option<Instant>,
    /// Requested finest tiles without a copy that [`TileSource::deferred`] let through since
    /// the last [`TileSource::prepare`]: what that prepare evaluates here, counted against the
    /// time left.
    reserved: BTreeSet<TileKey>,
    /// Finest tiles read through the store's own evaluation on this thread, not from a copy.
    evaluated_here: usize,
    /// A delay added to each requested finest tile evaluated here, for the tests of a slow
    /// evaluation.
    #[cfg(test)]
    slow: std::time::Duration,
}

impl<'a> StoreSource<'a> {
    /// Reads tiles from `store`, evaluating finest tiles as they are prepared.
    pub const fn new(store: &'a mut TileStore) -> Self {
        Self {
            store,
            display: BTreeMap::new(),
            evaluations: None,
            until: None,
            reserved: BTreeSet::new(),
            evaluated_here: 0,
            #[cfg(test)]
            slow: std::time::Duration::ZERO,
        }
    }

    /// Reads tiles from `store`, a finest tile from its copy in `evaluations` if one is held;
    /// requested finest tiles without one are deferred ([`TileSource::deferred`]), unless
    /// [`StoreSource::evaluating_until`] gives them time.
    pub const fn with_evaluations(
        store: &'a mut TileStore,
        evaluations: &'a mut FinestEvaluations,
    ) -> Self {
        Self {
            store,
            display: BTreeMap::new(),
            evaluations: Some(evaluations),
            until: None,
            reserved: BTreeSet::new(),
            evaluated_here: 0,
            #[cfg(test)]
            slow: std::time::Duration::ZERO,
        }
    }

    /// Lets requested finest tiles without a copy be evaluated here while the frame has the
    /// time, as staging without the queue does.
    ///
    /// Each is let through ([`TileSource::deferred`] says no) while the tiles let through since
    /// the last [`TileSource::prepare`], at the cost per tile measured lately
    /// ([`FinestEvaluations::cost`]) and a quarter more, fit before `deadline` less
    /// [`STAGE_MARGIN`]; with no cost measured yet, one tile at a time, if [`PROBE_COST`] fits.
    /// Each tile evaluated here is withdrawn from the queue, so none is evaluated twice.
    /// Without evaluations ([`StoreSource::new`]), it changes nothing.
    #[must_use]
    pub const fn evaluating_until(mut self, deadline: Instant) -> Self {
        self.until = Some(deadline);
        self
    }

    /// How many finest tiles were read through the store's evaluation on this thread, not from
    /// a copy the queued evaluation made: tiles a pending epoch names, or every finest tile
    /// for a source built [`new`](StoreSource::new). Each may have evaluated (and read from the
    /// file) while staging.
    #[must_use]
    pub const fn evaluated_here(&self) -> usize {
        self.evaluated_here
    }

    /// The copy of finest tile `key` the queued evaluation made, if one is held.
    fn evaluated(&self, key: &TileKey) -> Option<&EvaluatedTile> {
        self.evaluations
            .as_deref()
            .and_then(|evaluations| evaluations.get(key))
    }

    /// Whether requested finest tile `key`, without a copy, fits in the time left to be
    /// evaluated here: see [`StoreSource::evaluating_until`].
    fn fits_here(&self, key: &TileKey) -> bool {
        let (Some(until), Some(evaluations)) = (self.until, self.evaluations.as_deref()) else {
            return false;
        };
        if self.reserved.contains(key) {
            return true;
        }
        let room = until
            .saturating_duration_since(Instant::now())
            .saturating_sub(STAGE_MARGIN);
        match evaluations.cost() {
            None => self.reserved.is_empty() && room >= PROBE_COST,
            Some(cost) => {
                let tiles = u32::try_from(self.reserved.len() + 1).unwrap_or(u32::MAX);
                cost.saturating_add(cost / 4).saturating_mul(tiles) <= room
            }
        }
    }
}

/// The time kept free before the staging deadline when requested finest tiles are evaluated
/// while staging ([`StoreSource::evaluating_until`]).
///
/// It is for the rest of staging after them: writing their records and the tile tree's nodes.
pub const STAGE_MARGIN: std::time::Duration = std::time::Duration::from_millis(1);

/// What a requested finest tile is taken to cost when no cost has been measured.
///
/// The time left must hold it before the first is evaluated while staging, alone, to measure
/// one (a cold tile, its cone read from the file, measured about 0.8 ms on the benchmark
/// project).
pub const PROBE_COST: std::time::Duration = std::time::Duration::from_millis(1);

impl TileSource for StoreSource<'_> {
    type Error = CacheError;

    fn prepare(&mut self, keys: &[TileKey]) -> Result<(), CacheError> {
        let (display, finest): (Vec<TileKey>, Vec<TileKey>) = keys
            .iter()
            .partition(|key| key.channel() == Channel::DISPLAY);
        // A tile with a copy reads it: nothing to evaluate.
        let finest: Vec<TileKey> = finest
            .into_iter()
            .filter(|key| self.evaluated(key).is_none())
            .collect();
        // The requested tiles let through to be evaluated here, timed for the next.
        let reserved = std::mem::take(&mut self.reserved);
        let requested: Vec<TileKey> = finest
            .iter()
            .copied()
            .filter(|key| reserved.contains(key))
            .collect();
        if !finest.is_empty() {
            let began = Instant::now();
            // Positions only, of every tile the rings read: the tiles, their aprons and the
            // tiles round their corners' base vertices.
            let mut inputs = BTreeSet::new();
            for key in &finest {
                inputs.extend(self.store.normal_inputs(key)?.iter().copied());
            }
            let inputs: Vec<TileKey> = inputs.into_iter().collect();
            self.store.evaluate_many(&inputs, Evaluation::Positions)?;
            #[cfg(test)]
            std::thread::sleep(self.slow * u32::try_from(requested.len()).unwrap_or(0));
            if let Some(evaluations) = self.evaluations.as_deref_mut()
                && let Ok(count) = u32::try_from(requested.len())
                && count > 0
            {
                evaluations.note_cost(began.elapsed() / count);
            }
        }
        if !requested.is_empty() {
            // Evaluated here: the queue lets them go, so none is evaluated twice.
            self.store.retain_evaluation(|key| !requested.contains(key));
        }
        if !display.is_empty() {
            let tiles = self.store.display_for_draw(&display)?;
            self.display.extend(display.into_iter().zip(tiles));
        }
        Ok(())
    }

    fn at_hand(&mut self, key: TileKey) -> bool {
        self.display.contains_key(&key) || !self.store.display_needs_read(&key)
    }

    fn deferred(&mut self, key: TileKey) -> bool {
        if self.evaluations.is_none()
            || key.channel() == Channel::DISPLAY
            || self.evaluated(&key).is_some()
        {
            return false;
        }
        if self.fits_here(&key) {
            self.reserved.insert(key);
            return false;
        }
        true
    }

    fn read(&mut self, key: TileKey, out: &mut TileData) -> Result<Option<TileDetail>, CacheError> {
        if key.channel() == Channel::DISPLAY {
            let tile = match self.display.remove(&key) {
                Some(tile) => tile,
                None => self.store.display_for_draw(&[key])?.pop().flatten(),
            };
            // Staged now: its record read ahead, if any, has done its work. A tile prepared but
            // never read here (no room or slot this frame) keeps its record for the next.
            self.store.forget_drawn(&[key]);
            let Some(tile) = tile else {
                return Ok(None);
            };
            let vertices = out
                .vertices()
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
            return Ok(Some(TileDetail::of_display(&tile)));
        }
        if let Some(tile) = self.evaluated(&key) {
            out.ring(self.store.mesh(), tile.ring());
            return Ok(Some(TileDetail::FINEST));
        }
        self.evaluated_here += 1;
        // Prepared tiles are cached and current: the gather reads them, evaluating nothing.
        let block = match self.store.cached_positions_with_ring(&key)? {
            Some(block) => block,
            None => self.store.positions_with_ring(key)?,
        };
        out.ring(self.store.mesh(), &block);
        Ok(Some(TileDetail::FINEST))
    }
}

/// `v`'s components.
const fn components(v: Vec3) -> [f32; 3] {
    [v.x, v.y, v.z]
}

/// Asks the store to page in the requested tiles' data on its workers.
///
/// The correction tiles of the finest tiles among `requests` are paged in, and the records of
/// the display tiles among them read ahead, larger screen error first
/// ([`TileStore::read_display_ahead`]).
///
/// Reading them later then finds them in memory: what the owner thread does with the
/// requests a frame's readback gave ([`crate::residency::Residency::take_requests`]) before
/// handing them to [`crate::residency::Residency::want`]. Returns how many reads were queued.
///
/// # Errors
///
/// As [`TileStore::prefetch`].
pub fn prefetch_requests(store: &mut TileStore, requests: &Requests) -> Result<usize, CacheError> {
    let (mut display, finest): (Vec<&Request>, Vec<&Request>) = requests
        .iter()
        .partition(|request| request.key.channel() == Channel::DISPLAY);
    let finest: Vec<TileKey> = finest.iter().map(|request| request.key).collect();
    display.sort_by(|a, b| b.screen_error.total_cmp(&a.screen_error));
    let display: Vec<TileKey> = display.iter().map(|request| request.key).collect();
    let read_ahead = store.read_display_ahead(&display);
    Ok(store.prefetch(&finest)? + read_ahead)
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

    /// Ends the frame's staging: unmaps its buffers and hands them, with their copies, the
    /// normal pass and the request readback, to the render thread as frame `frame`, whose
    /// outcome goes to `outcomes`.
    pub(crate) fn finish(
        mut self,
        frame: u64,
        normals: Option<NormalWork>,
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
            normals,
            bytes: self.bytes,
            requests,
            level_choice: true,
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
    /// Mapped for reading, the requests of frame `1` (residency's frame number) in it.
    Mapped(Buffer, u64),
    /// Unmapped and unused: its frame was dropped before submission.
    Unused(Buffer),
    /// Unmapped and unread: frame `1` was submitted without a level choice, which writes the
    /// requests ([`FrameUploads::without_level_choice`]). It is heard of, with no requests.
    Skipped(Buffer, u64),
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
    /// The residency frame the copy is made in: the level choice's requests of that frame.
    pub(crate) frame: u64,
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
    /// The normal pass computing its finest tiles' vertices from their records, if it staged
    /// any.
    normals: Option<NormalWork>,
    /// Bytes staged.
    bytes: u64,
    /// The request readback, if a readback buffer was free.
    requests: Option<RequestCopy>,
    /// Whether the frame's passes run the level choice, which writes the request buffer the
    /// readback copies.
    level_choice: bool,
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

    /// Whether nothing was staged: no copies and no normal pass.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.copies.is_empty() && self.normals.is_none()
    }

    /// The same uploads, for a frame whose passes don't run the level choice (a frame with no
    /// image, or no scene): the request buffer isn't copied out or cleared, since nothing wrote
    /// it, and the frame's readback is heard of with no requests, so it neither replaces the
    /// tiles wanted nor keeps residency waiting for it
    /// ([`crate::residency::Residency::requests_heard`]).
    pub const fn without_level_choice(mut self) -> Self {
        self.level_choice = false;
        self
    }

    /// Encodes the copies into tile slots and the tile tree: first in the frame's command
    /// buffer, before any pass reads them.
    fn encode(&self, encoder: &mut CommandEncoder) {
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

    /// Encodes and submits the frame: into `encoder`, the copies first, then the normal pass
    /// (which writes the finest tiles' vertices from the records the copies bring), then
    /// `passes` (which read the tiles they bring), then the request readback (the request buffer is
    /// copied out, then its count cleared for the next frame); submits the command buffer, and
    /// only then maps the staging buffers for writing again (they return to the ring when the
    /// GPU is done with them) and the readback buffer for reading. The callbacks run when the
    /// device is polled ([`crate::residency::Residency::begin_frame`] does).
    ///
    /// Taking the passes keeps the order fixed: no pass can be encoded before the copies.
    pub fn submit(
        mut self,
        queue: &Queue,
        mut encoder: CommandEncoder,
        passes: impl FnOnce(&mut CommandEncoder),
    ) {
        self.encode(&mut encoder);
        if let Some(normals) = &self.normals {
            normals.encode(&mut encoder);
        }
        passes(&mut encoder);
        if !self.level_choice
            && let Some(RequestCopy {
                target,
                done,
                frame,
                ..
            }) = self.requests.take()
        {
            // A residency that's gone has nothing to hear.
            drop(done.send(Readback::Skipped(target, frame)));
        }
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
            let RequestCopy {
                target,
                done,
                frame,
                ..
            } = requests;
            let buffer = target.clone();
            target.map_async(MapMode::Read, .., move |result| {
                let back = if result.is_ok() {
                    Readback::Mapped(buffer, frame)
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
