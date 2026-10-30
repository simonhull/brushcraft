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
//! mid-session. They go to the render thread with the first frame's scene and stay there; the
//! level choice's readbacks come back here ([`CutReader`]).
//!
//! # Readbacks
//!
//! A frame's level choice writes the tiles it wants and finds missing into residency's request
//! buffer, and its cut into its own readback; both are copied out in the frame's submission and
//! mapped when the GPU finishes it, so the device poll in a later frame's
//! [`Residency::begin_frame`] hears of both at once. Each says which frame it is of (residency's
//! frame number, [`Residency::requests_heard`], [`render::lod::LodOutput::frame`]):
//!
//! - **the requests** are paged in and wanted ([`Residency::take_requests`], [`Residency::want`]);
//! - **the cut** is the frame's use ([`Residency::use_frame`]): the tiles it drew renewed as
//!   recently drawn, its tiles and their ancestors never evicted while it is the latest. A frame
//!   whose readback hasn't arrived is covered by residency's rules as they are (the latest use
//!   holds; the floor is never evicted). A cut that differs from the one before
//!   ([`Residency::cut`]) reports its frame's time, so the scheduler keeps frames coming until the
//!   morph it started is over.
//!
//! Frames keep coming after a change (the view moved, tiles arrived, the requests changed) until
//! a request readback from then on is read ([`settle`]): so the last view's requests and cut are
//! never left unread until the next event.
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
use render::lod::{CutReader, LodConfig, LodStats};
use render::raster::{FramePasses, MORPH};
use render::residency::{FrameUse, Residency, ResidencyConfig, ResidencyStats, Tier};
use render::upload::{FrameUploads, StoreSource, prefetch_requests};
use store::TileStore;

use crate::frame::{FrameUpdate, UploadStatus, Uploads};
use crate::render_thread::ResidentScene;

/// A mebibyte, for the overlay's sizes.
const MIB: f64 = 1024.0 * 1024.0;

/// The capability tier the app runs at.
const TIER: Tier = Tier::Desktop;

/// The project's tiles on the GPU: residency, the scene's passes until the render thread has
/// them, and what the frame loop needs around them.
#[derive(Debug)]
pub(crate) struct GpuResidency {
    /// The GPU, shared with the render thread.
    gpu: Arc<Gpu>,
    /// Which tiles are on it.
    residency: Residency,
    /// The scene's passes, until the first frame's scene takes them to the render thread.
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
    /// The time of each frame staged whose cut hasn't been heard of yet, by its number.
    times: BTreeMap<u64, Duration>,
    /// The tile tree's generation the GPU's bounds are at.
    bounds_at: u64,
    /// The generation the last frame's bounds update brings them to, once its scene is drawn.
    bounds_sent: Option<u64>,
    /// The latest cut readback's counts.
    scene: Option<LodStats>,
    /// How many cut readbacks have been heard.
    cuts_heard: u64,
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
        let finest = store
            .display_finest()
            .context("the store keeps no display layer to draw")?;
        let config = ResidencyConfig::for_tier(TIER);
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
            times: BTreeMap::new(),
            bounds_at: 0,
            bounds_sent: None,
            scene: None,
            cuts_heard: 0,
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
        } = update;
        let gpu = &*self.gpu;
        let residency = &mut self.residency;
        residency
            .begin_frame(gpu)
            .context("hearing what became of earlier frames")?;
        let frame = residency.frame();
        self.times.insert(frame, time);
        let mut changed = moved;
        residency.apply(drained);
        if let Some(requests) = residency.take_requests() {
            prefetch_requests(store, &requests).context("paging in requested tiles")?;
            residency.want(&requests);
            let requested: BTreeSet<TileKey> = requests.iter().map(|r| r.key).collect();
            if requested != self.requested {
                self.requested = requested;
                changed = true;
            }
        }
        let mut cut_changed = None;
        let cut = self
            .reader
            .collect(residency.finest())
            .context("reading the level choice's cut back")?;
        if let Some(output) = cut {
            if residency.cut() != &output.cut {
                cut_changed = self.times.get(&output.frame).copied();
            }
            // Frames up to this one have nothing more to hear.
            self.times = self.times.split_off(&(output.frame + 1));
            self.cuts_heard += 1;
            self.scene = Some(output.stats);
            residency.use_frame(FrameUse::new(output.cut, output.drawn));
        }
        let uploads = residency
            .stage(gpu, &mut StoreSource::new(store), deadline)
            .context("staging tiles")?;
        // Tiles arriving or leaving change what the level choice asks for too.
        changed |= !uploads.is_empty();
        let heard = residency.requests_heard();
        self.settle_from = settle(self.settle_from, changed.then_some(frame), heard);
        let bounds = BoundsUpdate::since(residency.tree(), self.bounds_at);
        self.bounds_sent = Some(bounds.generation());
        let scene = ResidentScene {
            tree: residency.tree().buffer().clone(),
            requests: residency.request_buffer().clone(),
            slots: SlotBuffers::of(residency),
            bounds,
            finest: residency.finest(),
            frame,
            passes: self.passes.take(),
        };
        Ok(Staged {
            uploads,
            scene,
            cut_changed,
        })
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
            settling: self.settle_from.is_some(),
        };
        let pending = pending(&stats, work);
        Ok(UploadStatus {
            uploading: pending.uploads,
            requesting: pending.requests,
            cut_changed: staged.cut_changed,
            summary: Some(summary(&stats, self.floor_tiles())),
            frame: Some(staged.uploads),
            scene: Some(staged.scene),
        })
    }

    fn shown(&mut self, drawn: bool) {
        let sent = self.bounds_sent.take();
        if drawn && let Some(generation) = sent {
            self.bounds_at = generation;
        }
    }

    fn morph(&self) -> Duration {
        self.morph
    }

    fn scene(&self) -> Option<&LodStats> {
        self.scene.as_ref()
    }
}

/// What residency's counts don't say about the work left, for [`pending`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Work {
    /// Whether every floor tile the source can give is resident.
    pub(crate) floor_ready: bool,
    /// How many of the wanted tiles found no slot this frame.
    pub(crate) starved_now: u64,
    /// Whether a request readback is still to be read before the requests are settled.
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

/// The overlay's line for what the GPU holds.
///
/// Resident tiles (the floor's of `floor_tiles`), slot memory used of that allocated (the
/// floor's classes are allocated twice over, for a resync, so used is below allocated even when
/// every tile is resident), what is still wanted and pending, and what the last frame staged.
pub(crate) fn summary(stats: &ResidencyStats, floor_tiles: usize) -> String {
    #[expect(
        clippy::cast_precision_loss,
        reason = "byte counts shown to two places in MiB: far below f64's exact range"
    )]
    let (used, allocated, staged) = (
        stats.used_bytes as f64 / MIB,
        stats.allocated_bytes as f64 / MIB,
        stats.staged_bytes as f64 / MIB,
    );
    format!(
        "{} resident (floor {}/{floor_tiles})  {used:.1} of {allocated:.1} MiB used  {} \
         wanted  {} epochs \
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
    fn the_tiles_line_shows_slot_memory_used_and_allocated() {
        let stats = ResidencyStats {
            resident: 7,
            floor: 5,
            used_bytes: 3 << 20,
            allocated_bytes: 10 << 20,
            ..ResidencyStats::default()
        };
        let line = summary(&stats, 5);
        assert!(
            line.starts_with("7 resident (floor 5/5)  3.0 of 10.0 MiB used  0 wanted"),
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
