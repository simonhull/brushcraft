//! GPU residency in the frame loop: the [`Uploads`] seam over [`render::residency::Residency`].
//!
//! Each frame, on the owner thread, [`GpuResidency`] runs residency's frame
//! ([`Residency::begin_frame`], the drained feed, the requests a readback gave, the frame's
//! use, then [`Residency::stage`] from the store until the uploads' deadline) and returns the
//! staged [`FrameUploads`], which travel to the render thread in the frame's request. The
//! render thread encodes them ahead of its passes and submits them with the frame; a frame
//! whose uploads never reach the GPU is recovered by the next [`Residency::begin_frame`].
//!
//! The tile requests a frame's readback gives come back through residency's own readback
//! ([`Residency::take_requests`]), not through the render thread's report.
//!
//! Until the level choice (M1a3 Task 3) says what each frame cut and drew, the frame's use is
//! a placeholder: the floor plus every tile requested, as both the cut and the drawn set
//! ([`placeholder_use`]).
//!
//! All of a frame's GPU work here runs in one error scope on the owner thread
//! ([`render::gpu::Gpu::scoped`]), so an error it raises is the owner's, never blamed on the
//! render thread's frame.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context as _, Result};
use geom::TileKey;
use render::gpu::Gpu;
use render::residency::{FrameUse, Residency, ResidencyConfig, ResidencyStats, Tier};
use render::upload::{FrameUploads, StoreSource, prefetch_requests};
use store::{Drained, TileStore};

use crate::frame::{UploadStatus, Uploads};

/// A mebibyte, for the overlay's sizes.
const MIB: f64 = 1024.0 * 1024.0;

/// The project's tiles on the GPU: residency, and what the frame loop needs around it.
#[derive(Debug)]
pub(crate) struct GpuResidency {
    /// The GPU, shared with the render thread.
    gpu: Arc<Gpu>,
    /// Which tiles are on it.
    residency: Residency,
    /// Every tile the latest readback requested: the placeholder use's tiles beyond the floor.
    requested: BTreeSet<TileKey>,
    /// Whether the placeholder use changed since residency was last told it.
    use_changed: bool,
    /// [`ResidencyStats::starved`] as the last frame left it: the next frame's increase is how
    /// many requests found no slot then.
    starved: u64,
}

impl GpuResidency {
    /// Residency for `store`'s project on `gpu`, configured for the desktop tier
    /// ([`ResidencyConfig::for_tier`]); nothing is resident until the first frame stages the
    /// floor.
    ///
    /// # Errors
    ///
    /// If the store keeps no display layer (the app always opens one with one), or residency
    /// can't be set up on this device.
    pub(crate) fn new(gpu: Arc<Gpu>, store: &TileStore) -> Result<Self> {
        let finest = store
            .display_finest()
            .context("the store keeps no display layer to draw")?;
        let config = ResidencyConfig::for_tier(Tier::Desktop);
        let residency = gpu
            .scoped("setting up GPU residency", || {
                Residency::new(&gpu, store.mesh(), finest, config)
            })
            .context("setting up GPU residency")?
            .context("setting up GPU residency")?;
        Ok(Self {
            gpu,
            residency,
            requested: BTreeSet::new(),
            use_changed: true,
            starved: 0,
        })
    }

    /// Residency's counts, as they stand.
    pub(crate) fn stats(&self) -> ResidencyStats {
        self.residency.stats()
    }

    /// How many tiles the floor has, resident or not.
    pub(crate) fn floor_tiles(&self) -> usize {
        self.residency.floor().len()
    }

    /// One frame of residency's work on the owner thread, inside the caller's error scope:
    /// see the module notes.
    ///
    /// # Errors
    ///
    /// If polling the GPU, the store's page-ins or the staging fail.
    fn stage(
        &mut self,
        drained: Drained,
        store: &mut TileStore,
        deadline: Instant,
    ) -> Result<FrameUploads> {
        let gpu = &*self.gpu;
        let residency = &mut self.residency;
        residency
            .begin_frame(gpu)
            .context("hearing what became of earlier frames")?;
        residency.apply(drained);
        if let Some(requests) = residency.take_requests() {
            prefetch_requests(store, &requests).context("paging in requested tiles")?;
            residency.want(&requests);
            let requested: BTreeSet<TileKey> = requests.iter().map(|r| r.key).collect();
            if requested != self.requested {
                self.requested = requested;
                self.use_changed = true;
            }
        }
        // The use holds until the next call, so it is told only when it changes.
        if std::mem::take(&mut self.use_changed) {
            residency.use_frame(placeholder_use(residency.floor(), &self.requested));
        }
        let uploads = residency
            .stage(gpu, &mut StoreSource::new(store), deadline)
            .context("staging tiles")?;
        Ok(uploads)
    }
}

impl Uploads for GpuResidency {
    fn apply(
        &mut self,
        drained: Drained,
        store: &mut TileStore,
        deadline: Instant,
    ) -> Result<UploadStatus> {
        let gpu = Arc::clone(&self.gpu);
        let uploads = gpu
            .scoped("the owner thread's uploads", || {
                self.stage(drained, store, deadline)
            })
            .context("uploading tiles on the owner thread")??;
        let stats = self.residency.stats();
        let starved_now = stats.starved.saturating_sub(self.starved);
        self.starved = stats.starved;
        Ok(UploadStatus {
            converging: converging(&stats, self.residency.floor_ready(), starved_now),
            summary: Some(summary(&stats, self.floor_tiles())),
            frame: Some(uploads),
        })
    }
}

/// The frame's use until the level choice gives a real one (Task 3).
///
/// The `floor` and every tile `requested`, as both the cut and the drawn set: so nothing a
/// request brought in is evicted while it is still requested, and the floor (pinned anyway)
/// counts as drawn.
pub(crate) fn placeholder_use(
    floor: &BTreeSet<TileKey>,
    requested: &BTreeSet<TileKey>,
) -> FrameUse {
    let cut: BTreeSet<TileKey> = floor.union(requested).copied().collect();
    FrameUse::new(cut.clone(), cut)
}

/// Whether residency wants the next frame.
///
/// It does while the floor isn't all resident, while an epoch is pending (it switches whole,
/// over as many frames as it takes, forced within a few), and while a requested tile is still
/// to come that a slot could be found for. `starved_now` of the wanted found none this frame:
/// they wait for a slot, which only a change of use frees, so they alone want no frame, and an
/// idle residency never spins.
pub(crate) const fn converging(
    stats: &ResidencyStats,
    floor_ready: bool,
    starved_now: u64,
) -> bool {
    !floor_ready || stats.pending_epochs > 0 || stats.wanted as u64 > starved_now
}

/// The overlay's line for what the GPU holds: resident tiles (the floor's of
/// `floor_tiles`), slot memory, what is still wanted and pending, and what the last frame
/// staged.
pub(crate) fn summary(stats: &ResidencyStats, floor_tiles: usize) -> String {
    #[expect(
        clippy::cast_precision_loss,
        reason = "byte counts shown to two places in MiB: far below f64's exact range"
    )]
    let (allocated, staged) = (
        stats.allocated_bytes as f64 / MIB,
        stats.staged_bytes as f64 / MIB,
    );
    format!(
        "{} resident (floor {}/{floor_tiles})  {allocated:.1} MiB  {} wanted  {} epochs \
         pending  staged {staged:.2} MiB  {} evicted  {} lost frames",
        stats.resident,
        stats.floor,
        stats.wanted,
        stats.pending_epochs,
        stats.evictions,
        stats.lost_frames
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
    fn residency_converges_while_it_has_work_and_idles_without() {
        assert!(converging(&stats(0, 0), false, 0), "the floor isn't up");
        assert!(converging(&stats(0, 1), true, 0), "an epoch pending");
        assert!(converging(&stats(3, 0), true, 0), "tiles to stage");
        assert!(
            converging(&stats(3, 0), true, 2),
            "one of them could be staged"
        );
        assert!(
            !converging(&stats(3, 0), true, 3),
            "all starved: no spinning"
        );
        assert!(!converging(&stats(0, 0), true, 0), "idle");
    }

    #[test]
    fn the_placeholder_use_is_the_floor_and_the_requests() {
        let face = geom::FaceId::new(0);
        let level = |l| geom::Level::new(l).expect("a level");
        let floor = BTreeSet::from([TileKey::display(face, level(0), 0, 0).expect("a key")]);
        let requested = BTreeSet::from([TileKey::display(face, level(7), 1, 1).expect("a key")]);
        let used = placeholder_use(&floor, &requested);
        let both: BTreeSet<TileKey> = floor.union(&requested).copied().collect();
        assert_eq!(used, FrameUse::new(both.clone(), both));
    }
}
