//! GPU residency in the frame loop: the [`Uploads`] seam over [`render::residency::Residency`],
//! and the scene the render thread draws from it.
//!
//! Each frame, on the owner thread, [`GpuResidency`] runs residency's frame
//! ([`Residency::begin_frame`], the drained feed, the requests and the cut readbacks gave, then
//! [`Residency::stage`] from the store until the uploads' deadline) and returns the staged
//! [`FrameUploads`] with the frame's [`ResidentScene`]: what the scene's passes read (the tile
//! tree, the request buffer, the slot buffers, the bounds to recompute). Both travel to the
//! render thread in the frame's request, which encodes the uploads, then the scene's passes
//! ([`render::raster::FramePasses`], in [`render::Renderer::draw`]), and submits them together;
//! a frame whose uploads never reach the GPU is recovered by the next [`Residency::begin_frame`].
//!
//! # The passes
//!
//! The level choice and raster are built here, beside residency, for its tile tree and slot
//! layout, so a device that can't take them fails as the app starts ([`GpuResidency::new`]), never
//! mid-session. They go to the render thread with the first frame (beside its scene, so a frame
//! with no scene still takes them) and stay there; the level choice's readbacks come back here
//! ([`CutReader`]).
//!
//! # Readbacks
//!
//! A frame's level choice writes the tiles it wants and finds missing into residency's request
//! buffer, and its cut into its own readback; both are copied out in the frame's submission and
//! mapped when the GPU finishes it. The device poll in a later frame's
//! [`Residency::begin_frame`] hears of both at once: the render thread registers both maps (the
//! requests' in `FrameUploads::submit`, the cut's right after it) before it reports the frame
//! shown, the owner polls only at the start of a frame, and only one frame is in flight, so no
//! poll falls between the two, and maps of one submission complete in the same poll. Each says
//! which frame it is of (residency's frame number, [`Residency::requests_heard`],
//! [`render::lod::LodOutput::frame`]):
//!
//! - **the requests** are paged in and wanted ([`Residency::take_requests`], [`Residency::want`]);
//! - **the cut** is the frame's use ([`Residency::use_frame`]): the tiles it drew renewed as
//!   recently drawn, its tiles and their ancestors never evicted while it is the latest. A frame
//!   whose readback hasn't arrived is covered by residency's rules as they are (the latest use
//!   holds; the floor is never evicted). A cut that differs from the one before
//!   ([`Residency::cut`]) reports its frame's time, if the camera was still in that frame, so the
//!   scheduler keeps frames coming until the morph it started is over. A moving frame's level
//!   changes don't morph by time (the raster stamps them a whole duration ago), so they want no
//!   morph frames.
//!
//! Frames keep coming after a change (the view moved, tiles arrived, the requests changed) until
//! a request readback from then on is read, and a cut readback too once a frame from then on has
//! drawn its scene ([`settle`]): so the last view's requests and cut are never left unread until
//! the next event. A request readback that overflowed the request buffer
//! ([`render::residency::Requests::dropped`]) holds whichever requests the GPU's atomics let in
//! first, so it isn't compared with the last: it says only that more is wanted than fits, which
//! the pending requests already keep frames coming for.
//!
//! # Bounds
//!
//! The bounds pass recomputes the bounds of every tile whose data changed since the bounds the
//! GPU has: the frame gathers them from the tile tree after its uploads are staged
//! ([`BoundsUpdate::since`]), so the pass sees each tile in the frame its data lands. The GPU has
//! them once the frame's scene is drawn ([`Uploads::shown`]); a frame that wasn't leaves them to
//! the next.
//!
//! All of a frame's GPU work here runs in one error scope on the owner thread
//! ([`render::gpu::Gpu::scoped`]), so an error it raises is the owner's, never blamed on the
//! render thread's frame.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context as _, Result};
use geom::TileKey;
use render::cull::{BoundsUpdate, SlotBuffers};
use render::gpu::Gpu;
use render::lod::{CutReader, LodConfig, LodOutput, LodStats};
use render::raster::{FramePasses, MORPH};
use render::residency::{FrameUse, Residency, ResidencyConfig, ResidencyStats, Tier};
use render::shade::ShadePath;
use render::upload::{FrameUploads, StoreSource, prefetch_requests};
use store::TileStore;

use self::jumps::LevelJumps;
use crate::frame::{CutHeard, FrameUpdate, UploadStatus, Uploads};
use crate::record::SceneRecord;
use crate::render_thread::ResidentScene;

mod jumps;

/// A mebibyte, for the overlay's sizes.
const MIB: f64 = 1024.0 * 1024.0;

/// The capability tier the app runs at.
const TIER: Tier = Tier::Desktop;

/// The most frames whose times are kept for their cut readbacks.
///
/// A few more than can be in flight (one frame at a time, its readbacks a frame or two behind),
/// so frames whose scene was never drawn, or whose readback was skipped, don't pile up.
const TIMES_KEPT: usize = 8;

/// A staged frame's time and whether its camera moved, kept until its cut is heard of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Stamp {
    /// Its time on the frame loop's clock.
    time: Duration,
    /// Whether the camera moved in it.
    moving: bool,
}

/// The project's tiles on the GPU: residency, the scene's passes until the render thread has
/// them, and what the frame loop needs around them.
#[derive(Debug)]
pub(crate) struct GpuResidency {
    /// The GPU, shared with the render thread.
    gpu: Arc<Gpu>,
    /// Which tiles are on it.
    residency: Residency,
    /// The scene's passes, until the first frame takes them to the render thread.
    passes: Option<FramePasses>,
    /// Where the level choice's cut readbacks arrive.
    reader: CutReader,
    /// How long a level change morphs: the raster's.
    morph: Duration,
    /// Every tile the latest request readback asked for.
    requested: BTreeSet<TileKey>,
    /// [`ResidencyStats::starved`] as the last frame left it: the next frame's increase is how
    /// many requests found no slot then.
    starved: u64,
    /// The first frame ([`Residency::frame`]) whose request readback must be read before the
    /// requests are settled; `None` once they are. Set by a frame whose view moved, whose
    /// uploads changed what is resident, or whose readback changed what is requested: what the
    /// level choice asks for may differ from then on, and only that frame's readback (or a
    /// later one) says.
    settle_from: Option<u64>,
    /// Like `settle_from`, for the cut readback: the first frame whose cut must be heard of before
    /// the cut is settled, waited for only once a frame from then on has drawn its scene.
    cut_from: Option<u64>,
    /// The newest frame whose cut readback has been heard of.
    cut_heard: Option<u64>,
    /// The newest frame whose scene was drawn.
    drawn: Option<u64>,
    /// The last frame staged: the one [`Uploads::shown`] reports on.
    staged: Option<u64>,
    /// The stamp of each frame staged whose cut hasn't been heard of yet, by its number: at most
    /// [`TIMES_KEPT`].
    times: BTreeMap<u64, Stamp>,
    /// The tile tree's generation the GPU's bounds are at.
    bounds_at: u64,
    /// The generation the last frame's bounds update brings them to, once its scene is drawn.
    bounds_sent: Option<u64>,
    /// The latest cut readback's counts.
    scene: Option<LodStats>,
    /// How many cut readbacks have been heard.
    cuts_heard: u64,
    /// The current stroke's tiles, for the level jumps of the cuts drawing them.
    jumps: LevelJumps,
}

impl GpuResidency {
    /// Residency for `store`'s project on `gpu`, configured for the desktop tier
    /// ([`ResidencyConfig::for_tier`]), and the scene's passes for it; nothing is resident until
    /// the first frame stages the floor.
    ///
    /// # Errors
    ///
    /// If the store keeps no display layer (the app always opens one with one), or residency
    /// can't be set up on this device (among others, a budget whose slot buffers one draw can't
    /// bind: `render::residency::ConfigError::TooManySlotBuffers`), or the level choice or the
    /// raster can't be built for it.
    pub(crate) fn new(gpu: Arc<Gpu>, store: &TileStore) -> Result<Self> {
        Self::with_config(gpu, store, ResidencyConfig::for_tier(TIER))
    }

    /// [`GpuResidency::new`], residency set up by `config`: for tests of a tiny budget.
    ///
    /// # Errors
    ///
    /// As [`GpuResidency::new`].
    pub(crate) fn with_config(
        gpu: Arc<Gpu>,
        store: &TileStore,
        config: ResidencyConfig,
    ) -> Result<Self> {
        let finest = store
            .display_finest()
            .context("the store keeps no display layer to draw")?;
        let residency = gpu
            .scoped("setting up GPU residency", || {
                Residency::new(&gpu, store.mesh(), finest, config)
            })?
            .context("setting up GPU residency: this GPU can't hold the tiles as configured")?;
        let mut passes = gpu
            .scoped("building the level choice and the raster", || {
                FramePasses::new(
                    &gpu,
                    residency.tree().capacity(),
                    residency.slot_layout(),
                    LodConfig::for_tier(TIER),
                    MORPH,
                    ShadePath::for_tier(TIER),
                )
            })?
            .context("building the level choice and the raster for this GPU")?;
        let reader = passes
            .lod_mut()
            .take_reader()
            .context("the level choice's readbacks were taken already")?;
        let morph = passes.morph();
        Ok(Self {
            gpu,
            residency,
            passes: Some(passes),
            reader,
            morph,
            requested: BTreeSet::new(),
            starved: 0,
            settle_from: None,
            cut_from: None,
            cut_heard: None,
            drawn: None,
            staged: None,
            times: BTreeMap::new(),
            bounds_at: 0,
            bounds_sent: None,
            scene: None,
            cuts_heard: 0,
            jumps: LevelJumps::default(),
        })
    }

    /// Residency's counts, as they stand.
    pub(crate) fn stats(&self) -> ResidencyStats {
        self.residency.stats()
    }

    /// The residency, for tests that look inside it.
    #[cfg(test)]
    pub(crate) const fn residency(&self) -> &Residency {
        &self.residency
    }

    /// How many cut readbacks have been heard.
    #[cfg(test)]
    pub(crate) const fn cuts_heard(&self) -> u64 {
        self.cuts_heard
    }

    /// How many tiles the floor has, resident or not.
    pub(crate) fn floor_tiles(&self) -> usize {
        self.residency.floor().len()
    }

    /// One frame of residency's work on the owner thread, inside the caller's error scope:
    /// see the module notes. Returns the uploads, the scene, and the time of the newest frame
    /// heard of whose cut changed.
    ///
    /// # Errors
    ///
    /// If polling the GPU, reading a readback, the store's page-ins or the staging fail.
    fn stage(&mut self, update: FrameUpdate, store: &mut TileStore) -> Result<Staged> {
        let FrameUpdate {
            drained,
            deadline,
            moved,
            time,
            stroke_open,
        } = update;
        let gpu = &*self.gpu;
        let residency = &mut self.residency;
        residency
            .begin_frame(gpu)
            .context("hearing what became of earlier frames")?;
        let frame = residency.frame();
        self.times.insert(
            frame,
            Stamp {
                time,
                moving: moved,
            },
        );
        while self.times.len() > TIMES_KEPT {
            self.times.pop_first();
        }
        let mut changed = moved;
        self.jumps
            .drained(stroke_open, &drained, residency.finest());
        residency.apply(drained);
        if let Some(requests) = residency.take_requests() {
            prefetch_requests(store, &requests).context("paging in requested tiles")?;
            residency.want(&requests);
            let requested: BTreeSet<TileKey> = requests.iter().map(|r| r.key).collect();
            // An overflowed readback's set is the GPU's atomic order: not compared (see above).
            changed |= requests.dropped() == 0 && requested != self.requested;
            self.requested = requested;
        }
        let mut cut_changed = None;
        // Oldest first: each is the frame's use in turn, as if heard of one at a time.
        let cuts = self
            .reader
            .collect_all(residency.finest())
            .context("reading the level choice's cut back")?;
        let mut heard_cuts: Vec<CutHeard> = Vec::with_capacity(cuts.len());
        for output in cuts {
            let jump = self
                .jumps
                .jump(residency.cut(), &output.cut, residency.finest());
            heard_cuts.push(heard_of(&output, jump));
            if residency.cut() != &output.cut {
                let still = self.times.get(&output.frame).filter(|stamp| !stamp.moving);
                cut_changed = still.map(|stamp| stamp.time).or(cut_changed);
            }
            // Frames up to this one have nothing more to hear.
            self.times = self.times.split_off(&(output.frame + 1));
            self.cut_heard = self.cut_heard.max(Some(output.frame));
            self.cuts_heard += 1;
            self.scene = Some(output.stats);
            residency.use_frame(FrameUse::new(output.cut, output.drawn));
        }
        let uploads = residency
            .stage(gpu, &mut StoreSource::new(store), deadline)
            .context("staging tiles")?;
        // Tiles arriving or leaving change what the level choice asks for too.
        changed |= uploads.changes_scene();
        let heard = residency.requests_heard();
        self.settle_from = settle(self.settle_from, changed.then_some(frame), heard);
        self.cut_from = settle(self.cut_from, changed.then_some(frame), self.cut_heard);
        self.staged = Some(frame);
        let bounds = BoundsUpdate::since(residency.tree(), self.bounds_at);
        self.bounds_sent = Some(bounds.generation());
        let scene = ResidentScene {
            tree: residency.tree().buffer().clone(),
            requests: residency.request_buffer().clone(),
            slots: SlotBuffers::of(residency),
            bounds,
            finest: residency.finest(),
            frame,
        };
        Ok(Staged {
            uploads,
            scene,
            cut_changed,
            frame,
            heard: heard_cuts,
        })
    }

    /// Turns on timing the level choice's stages on the GPU, read back with each cut
    /// ([`render::lod::Lod::enable_timing`]), if the device can; whether it does. Call it before
    /// the first frame, which takes the passes to the render thread; after, it does nothing.
    ///
    /// # Errors
    ///
    /// If the queries can't be made.
    pub(crate) fn enable_timing(&mut self) -> Result<bool> {
        let Some(passes) = self.passes.as_mut() else {
            return Ok(false);
        };
        passes
            .lod_mut()
            .enable_timing(&self.gpu)
            .context("timing the level choice's stages")
    }
}

/// What a scripted run's record keeps of the cut `output`, whose stroked tiles' largest level
/// change since the cut heard before it was `level_jump` ([`LevelJumps::jump`]).
fn heard_of(output: &LodOutput, level_jump: Option<u8>) -> CutHeard {
    let mut levels: Vec<u64> = Vec::new();
    for key in &output.cut {
        let level = usize::from(key.level().get());
        if levels.len() <= level {
            levels.resize(level + 1, 0);
        }
        if let Some(count) = levels.get_mut(level) {
            *count += 1;
        }
    }
    CutHeard {
        frame: output.frame,
        scene: SceneRecord {
            stats: output.stats.clone(),
            levels,
            stages: output.timings,
            level_jump,
        },
    }
}

/// What [`GpuResidency::stage`] made of a frame.
#[derive(Debug)]
struct Staged {
    /// The frame's uploads.
    uploads: FrameUploads,
    /// What the scene's passes read.
    scene: ResidentScene,
    /// The time of the newest frame heard of whose cut changed, if one did.
    cut_changed: Option<Duration>,
    /// Residency's number for the frame.
    frame: u64,
    /// The cuts heard this frame, oldest first.
    heard: Vec<CutHeard>,
}

impl Uploads for GpuResidency {
    fn apply(&mut self, update: FrameUpdate, store: &mut TileStore) -> Result<UploadStatus> {
        let gpu = Arc::clone(&self.gpu);
        let staged = gpu.scoped("the owner thread's GPU work", || self.stage(update, store))??;
        let stats = self.residency.stats();
        let starved_now = stats.starved.saturating_sub(self.starved);
        self.starved = stats.starved;
        let work = Work {
            floor_ready: self.residency.floor_ready(),
            starved_now,
            settling: self.settle_from.is_some() || awaits_cut(self.cut_from, self.drawn),
        };
        let pending = pending(&stats, work);
        Ok(UploadStatus {
            uploading: pending.uploads,
            requesting: pending.requests,
            cut_changed: staged.cut_changed,
            summary: Some(summary(&stats, self.floor_tiles())),
            frame: Some(staged.uploads),
            scene: Some(staged.scene),
            passes: self.passes.take(),
            staged_frame: Some(staged.frame),
            stats: Some(stats),
            heard: staged.heard,
        })
    }

    fn shown(&mut self, drawn: bool) {
        let sent = self.bounds_sent.take();
        if drawn {
            if let Some(generation) = sent {
                self.bounds_at = generation;
            }
            self.drawn = self.drawn.max(self.staged);
        } else if let Some(frame) = self.staged {
            // No scene, no cut readback: its time has nothing to wait for.
            self.times.remove(&frame);
        }
    }

    fn morph(&self) -> Duration {
        self.morph
    }

    fn scene(&self) -> Option<&LodStats> {
        self.scene.as_ref()
    }

    fn draws(&self, key: &TileKey) -> bool {
        self.residency.holds_or_wants(key)
    }

    fn waiting_for_previews(&self) -> Vec<TileKey> {
        self.residency.waiting_for_previews().copied().collect()
    }
}

/// What residency's counts don't say about the work left, for [`pending`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Work {
    /// Whether every floor tile the source can give is resident.
    pub(crate) floor_ready: bool,
    /// How many of the wanted tiles found no slot this frame.
    pub(crate) starved_now: u64,
    /// Whether a readback is still to be read before the requests and the cut are settled.
    pub(crate) settling: bool,
}

/// What residency still has to do, each of which wants the next frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Pending {
    /// Uploads in flight: the floor isn't all resident, or an epoch is pending (it switches
    /// whole, over as many frames as it takes, forced within a few).
    pub(crate) uploads: bool,
    /// Requests pending: a requested tile is still to come that a slot could be found for, or the
    /// requests are settling ([`settle`]).
    pub(crate) requests: bool,
}

/// What residency still has to do.
///
/// Settling: a frame changed what the level choice may ask for (the view moved, tiles arrived
/// or left, the requests changed) and no readback that recent has been read. Only frames bring
/// readbacks, so frames come until one does and agrees with what is requested. A frame dropped
/// unsubmitted makes no readback, so frames keep coming and the next one recovers its tiles.
///
/// Wanted tiles that found no slot (`work.starved_now`) wait for one, which only a change of
/// use frees, so they alone want no frame, and an idle residency never spins.
pub(crate) const fn pending(stats: &ResidencyStats, work: Work) -> Pending {
    Pending {
        uploads: !work.floor_ready || stats.pending_epochs > 0,
        requests: stats.wanted as u64 > work.starved_now || work.settling,
    }
}

/// The frame whose request readback must be read next; `None` once the requests are settled.
///
/// Given the one before (`from`), the frame that just changed what may be requested
/// (`changed`, if it did) and the newest frame whose readback has been read (`heard`): settled
/// once a readback that recent is in hand.
pub(crate) fn settle(from: Option<u64>, changed: Option<u64>, heard: Option<u64>) -> Option<u64> {
    from.max(changed).filter(|&frame| heard < Some(frame))
}

/// Whether a cut readback is still awaited.
///
/// One from frame `from` on (the last change's, if one is unheard), once a frame from then on
/// (`drawn`, the newest whose scene was drawn) has made one. Until a frame draws its scene, none
/// will come, so none is waited for.
pub(crate) fn awaits_cut(from: Option<u64>, drawn: Option<u64>) -> bool {
    from.is_some_and(|from| drawn >= Some(from))
}

/// The overlay's line for what the GPU holds.
///
/// Resident tiles (the floor's of `floor_tiles`), slot memory used of that allocated (the
/// floor's classes are allocated twice over, for a resync, so used is below allocated even when
/// every tile is resident) and the finest tiles' record chunks beside it, what is still wanted
/// and pending, and what the last frame staged.
pub(crate) fn summary(stats: &ResidencyStats, floor_tiles: usize) -> String {
    #[expect(
        clippy::cast_precision_loss,
        reason = "byte counts shown to two places in MiB: far below f64's exact range"
    )]
    let (used, allocated, records, staged) = (
        stats.used_bytes as f64 / MIB,
        stats.allocated_bytes as f64 / MIB,
        stats.record_bytes as f64 / MIB,
        stats.staged_bytes as f64 / MIB,
    );
    format!(
        "{} resident (floor {}/{floor_tiles})  {used:.1} of {allocated:.1} MiB used  \
         + {records:.1} MiB records  {} wanted  {} epochs \
         pending  staged {staged:.2} MiB  {} evicted  {} lost frames",
        stats.resident,
        stats.floor,
        stats.wanted,
        stats.pending_epochs,
        stats.evictions,
        stats.lost_frames
    )
}

/// The overlay's line for what the latest frame read back drew: its cut, the tiles and clusters
/// each phase drew, what it requested, and what is left over a pixel; `None` before the first.
pub(crate) fn scene_summary(stats: Option<&LodStats>) -> String {
    stats.map_or_else(
        || "no cut read back yet".to_owned(),
        |stats| {
            format!(
                "cut {}  drawn {}+{} tiles  {}+{} clusters  {} triangles  {} requested  {} over \
                 1 px  {} shortfall",
                stats.cut,
                stats.drawn.0,
                stats.drawn.1,
                stats.clusters.0,
                stats.clusters.1,
                stats.drawn_triangles,
                stats.requested,
                stats.over_pixel,
                stats.shortfall
            )
        },
    )
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    /// Stats with `wanted` tiles wanted and `pending` epochs pending.
    fn stats(wanted: usize, pending: usize) -> ResidencyStats {
        ResidencyStats {
            wanted,
            pending_epochs: pending,
            ..ResidencyStats::default()
        }
    }

    #[test]
    fn residency_wants_frames_while_it_has_work_and_idles_without() {
        let ready = Work {
            floor_ready: true,
            ..Work::default()
        };
        let starved = |starved_now| Work {
            starved_now,
            ..ready
        };
        let uploads = Pending {
            uploads: true,
            requests: false,
        };
        let requests = Pending {
            uploads: false,
            requests: true,
        };
        assert_eq!(
            pending(&stats(0, 0), Work::default()),
            uploads,
            "the floor isn't up"
        );
        assert_eq!(pending(&stats(0, 1), ready), uploads, "an epoch pending");
        assert_eq!(pending(&stats(3, 0), ready), requests, "tiles to stage");
        assert_eq!(
            pending(&stats(3, 0), starved(2)),
            requests,
            "one could be staged"
        );
        assert_eq!(
            pending(&stats(3, 0), starved(3)),
            Pending::default(),
            "all starved: no spinning"
        );
        let settling = Work {
            settling: true,
            ..ready
        };
        assert_eq!(
            pending(&stats(0, 0), settling),
            requests,
            "a readback to read"
        );
        assert_eq!(pending(&stats(0, 0), ready), Pending::default(), "idle");
    }

    #[test]
    fn requests_settle_once_a_readback_from_after_the_last_change_is_read() {
        assert_eq!(settle(None, None, None), None, "nothing changed");
        assert_eq!(
            settle(None, Some(5), Some(4)),
            Some(5),
            "frame 5 changed things"
        );
        assert_eq!(
            settle(Some(5), None, Some(4)),
            Some(5),
            "its readback not read yet"
        );
        assert_eq!(
            settle(Some(5), Some(6), Some(5)),
            Some(6),
            "it changed them again"
        );
        assert_eq!(
            settle(Some(6), None, Some(6)),
            None,
            "frame 6's agreed: settled"
        );
        assert_eq!(
            settle(Some(6), None, Some(9)),
            None,
            "a later readback will do"
        );
    }

    #[test]
    fn a_cut_is_awaited_once_a_frame_since_the_change_has_drawn_its_scene() {
        assert!(
            !awaits_cut(None, Some(9)),
            "nothing changed, or heard already"
        );
        assert!(
            !awaits_cut(Some(5), None),
            "no scene drawn: no cut will come"
        );
        assert!(
            !awaits_cut(Some(5), Some(4)),
            "drawn only before the change"
        );
        assert!(
            awaits_cut(Some(5), Some(5)),
            "the changed frame drew: its cut comes"
        );
        assert!(awaits_cut(Some(5), Some(7)), "a later one did");
    }

    #[test]
    fn the_tiles_line_shows_slot_memory_used_and_allocated() {
        let stats = ResidencyStats {
            resident: 7,
            floor: 5,
            used_bytes: 3 << 20,
            allocated_bytes: 10 << 20,
            record_bytes: 1 << 19,
            ..ResidencyStats::default()
        };
        let line = summary(&stats, 5);
        assert!(
            line.starts_with(
                "7 resident (floor 5/5)  3.0 of 10.0 MiB used  + 0.5 MiB records  0 wanted"
            ),
            "{line}"
        );
    }

    #[test]
    fn the_scene_line_says_what_the_cut_drew() {
        assert_eq!(scene_summary(None), "no cut read back yet");
        let stats = LodStats {
            cut: 40,
            drawn: (30, 2),
            clusters: (120, 4),
            requested: 6,
            ..LodStats::default()
        };
        let line = scene_summary(Some(&stats));
        assert!(
            line.starts_with("cut 40  drawn 30+2 tiles  120+4 clusters"),
            "{line}"
        );
        assert!(line.contains("6 requested"), "{line}");
    }
}
