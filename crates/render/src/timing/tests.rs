//! A frame's phases from its passes' ticks, with no GPU; the GPU run is in `draw`'s tests.

use super::*;

/// Nanoseconds.
const fn ns(nanos: u64) -> Duration {
    Duration::from_nanos(nanos)
}

/// The frame's own ticks `own`, and none of the level choice's.
fn own(own: &[u64]) -> Ticks<'_> {
    Ticks { own, lod: &[] }
}

#[test]
fn a_phase_is_the_sum_of_its_passes_and_the_gaps_its_copies_sit_in() {
    use GpuPhase::{Raster, Readback, Shade, Uploads};
    let spans = [
        Span::Start(Pair::Own(0)),
        Span::Between(Uploads),
        Span::Pass(Raster, Pair::Own(2)),
        Span::Pass(Raster, Pair::Own(4)),
        Span::Between(Readback),
        Span::Pass(Shade, Pair::Own(6)),
    ];
    // Start 100–110; raster 130–160 and 170–200; shade 230–300.
    let ticks = [100, 110, 130, 160, 170, 200, 230, 300];
    let times = GpuTimes::from_ticks(9, &spans, own(&ticks), 1.0, None).unwrap();
    assert_eq!(times.frame(), 9);
    assert_eq!(
        times.phase(Uploads),
        Some(ns(20)),
        "start's end to raster's beginning"
    );
    assert_eq!(
        times.phase(Raster),
        Some(ns(60)),
        "both its passes, not the gap"
    );
    assert_eq!(
        times.phase(Readback),
        Some(ns(30)),
        "raster's end to shade's beginning"
    );
    assert_eq!(times.phase(Shade), Some(ns(70)));
    assert_eq!(times.phase(GpuPhase::Tone), None, "not run");
    assert_eq!(times.total(), ns(200), "first stamp to last");
    let sum: Duration = GpuPhase::ALL.iter().filter_map(|&p| times.phase(p)).sum();
    let left = times.total().checked_sub(sum);
    assert_eq!(
        left,
        Some(ns(20)),
        "the start and the gap between the raster passes"
    );
}

#[test]
fn ticks_scale_by_the_period_and_borrowed_stamps_read_the_level_choice_s_ticks() {
    let spans = [
        Span::Start(Pair::Own(0)),
        Span::Pass(GpuPhase::LevelChoice, Pair::Lod(2)),
        Span::Pass(GpuPhase::LevelChoice, Pair::Lod(4)),
        Span::Pass(GpuPhase::Tone, Pair::Own(2)),
    ];
    let ticks = Ticks {
        own: &[10, 12, 40, 44],
        lod: &[1, 1, 20, 25, 30, 38],
    };
    let times = GpuTimes::from_ticks(1, &spans, ticks, 2.5, None).unwrap();
    assert_eq!(
        times.phase(GpuPhase::LevelChoice),
        Some(ns(32)),
        "(5 + 8) × 2.5, rounded down"
    );
    assert_eq!(times.phase(GpuPhase::Tone), Some(ns(10)));
    assert_eq!(times.total(), ns(85));
}

#[test]
fn an_unwritten_stamp_leaves_its_phase_and_the_gaps_beside_it_absent() {
    use GpuPhase::{Raster, Readback, Shade, Uploads};
    let spans = [
        Span::Start(Pair::Own(0)),
        Span::Between(Uploads),
        Span::Pass(Raster, Pair::Own(2)),
        Span::Pass(Raster, Pair::Own(4)),
        Span::Between(Readback),
        Span::Pass(Shade, Pair::Own(6)),
    ];
    // The second raster pass's end is Metal's error value: never written.
    let ticks = [100, 110, 130, 160, 170, u64::MAX, 230, 300];
    let times = GpuTimes::from_ticks(1, &spans, own(&ticks), 1.0, None).unwrap();
    assert_eq!(times.phase(Uploads), Some(ns(20)));
    assert_eq!(times.phase(Raster), None, "half its passes isn't its time");
    assert_eq!(times.phase(Readback), None, "a gap needs both sides");
    assert_eq!(times.phase(Shade), Some(ns(70)));
    assert_eq!(times.total(), ns(200));
    // Zero is no time either, and a frame with no stamp written isn't timed at all.
    let zeros = [0; 8];
    assert_eq!(
        GpuTimes::from_ticks(1, &spans, own(&zeros), 1.0, None),
        None
    );
    assert_eq!(
        GpuTimes::from_ticks(1, &spans, own(&[1, 2]), 1.0, None)
            .unwrap()
            .total(),
        ns(1)
    );
    assert_eq!(GpuTimes::from_ticks(1, &[], own(&ticks), 1.0, None), None);
}

#[test]
fn a_pass_that_ends_before_it_begins_is_unwritten_and_a_gap_never_goes_negative() {
    let spans = [
        Span::Pass(GpuPhase::Raster, Pair::Own(0)),
        Span::Between(GpuPhase::Readback),
        Span::Pass(GpuPhase::Shade, Pair::Own(2)),
        Span::Pass(GpuPhase::Tone, Pair::Own(4)),
    ];
    // Shade begins before raster ends (passes overlap on a tile-based GPU); tone runs backwards.
    let ticks = [100, 200, 150, 260, 300, 290];
    let times = GpuTimes::from_ticks(1, &spans, own(&ticks), 1.0, None).unwrap();
    assert_eq!(times.phase(GpuPhase::Readback), Some(Duration::ZERO));
    assert_eq!(times.phase(GpuPhase::Shade), Some(ns(110)));
    assert_eq!(times.phase(GpuPhase::Tone), None);
    assert_eq!(times.total(), ns(160), "earliest stamp to latest written");
}

#[test]
fn stamps_off_give_and_note_nothing() {
    let mut stamps = Stamps::off();
    assert!(stamps.compute(GpuPhase::Shade).is_none());
    assert!(stamps.render(GpuPhase::Tone).is_none());
    stamps.between(GpuPhase::Readback);
    assert_eq!(stamps.spans, []);
}

#[test]
fn phase_names_are_distinct_record_words() {
    let names: std::collections::BTreeSet<&str> = GpuPhase::ALL.map(GpuPhase::name).into();
    assert_eq!(names.len(), GpuPhase::ALL.len());
    assert!(
        names
            .iter()
            .all(|name| name.bytes().all(|b| b.is_ascii_lowercase()))
    );
}

#[test]
fn the_level_choice_s_ticks_start_at_a_resolve_s_alignment_after_the_frame_s_own() {
    assert_eq!(LOD_AT % wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT, 0);
    assert!(LOD_AT >= u64::from(CAPACITY) * u64::from(QUERY_SIZE));
    assert_eq!(BYTES % 8, 0);
}

#[test]
fn a_start_run_beside_the_frame_before_is_taken_from_where_that_frame_ended() {
    use GpuPhase::{Bounds, Shade, Uploads};
    let spans = [
        Span::Start(Pair::Own(0)),
        Span::Between(Uploads),
        Span::Pass(Bounds, Pair::Own(2)),
        Span::Pass(Shade, Pair::Own(4)),
    ];
    // The start ran at 100, beside the frame before, which ended at 1000; the uploads' copies
    // then took 1000–1020, bounds 1020–1030, shade 1030–1100.
    let ticks = [100, 101, 1020, 1030, 1030, 1100];
    let late = GpuTimes::from_ticks(2, &spans, own(&ticks), 1.0, Some(1000)).unwrap();
    assert_eq!(
        late.phase(Uploads),
        Some(ns(20)),
        "not the frame before's tail"
    );
    assert_eq!(late.total(), ns(100), "from where the frame before ended");
    assert_eq!(late.phase(Shade), Some(ns(70)), "the passes as they are");
    let alone = GpuTimes::from_ticks(2, &spans, own(&ticks), 1.0, None).unwrap();
    assert_eq!(
        alone.phase(Uploads),
        Some(ns(919)),
        "with no frame before: as stamped"
    );
    let kept_up = GpuTimes::from_ticks(2, &spans, own(&ticks), 1.0, Some(50)).unwrap();
    assert_eq!(
        kept_up.phase(Uploads),
        Some(ns(919)),
        "a start after it changes nothing"
    );
}

#[test]
fn a_frame_s_last_tick_is_its_latest_written_one() {
    use GpuPhase::{Raster, Tone};
    let spans = [
        Span::Start(Pair::Own(0)),
        Span::Pass(Raster, Pair::Lod(0)),
        Span::Pass(Tone, Pair::Own(2)),
    ];
    // The tone pass's end unwritten (Metal's all ones): its beginning still counts.
    let own_ticks = [10, 20, 300, u64::MAX];
    let lod_ticks = [50, 400];
    let ticks = Ticks {
        own: &own_ticks,
        lod: &lod_ticks,
    };
    assert_eq!(last_tick(&spans, ticks), Some(400));
    assert_eq!(
        last_tick(&spans, own(&[0, 0, 0, 0])),
        None,
        "nothing written"
    );
}

/// Where a slot of the timer's ring is in its life, in the model of the ring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotLife {
    /// Its frame's queries, written or being written, not resolved yet.
    Pending(usize),
    /// Resolved by the encoding of frame `0`; mapped once the GPU has finished that frame.
    Resolving(usize),
}

#[test]
fn a_timing_slot_is_resolved_by_the_first_frame_after_its_own_and_freed_once_that_one_finishes() {
    // A model of the ring under the app's bound: a GPU taking 40 ms a frame, a 60 Hz display,
    // each frame starting once the GPU has finished all but `FRAMES_IN_FLIGHT - 1` before it.
    // The owner's poll at a frame's start runs the callbacks of every frame finished by then:
    // the timer's (which sets a frame's done flag and maps the slots it resolved) before the
    // app's, as the timer registers its own first.
    let bound = crate::FRAMES_IN_FLIGHT;
    let (interval, per_frame) = (16_667_u64, 40_000_u64);
    let frames = 200;
    let mut starts: Vec<u64> = Vec::new();
    let mut done: Vec<u64> = Vec::new();
    let mut slots: Vec<Option<SlotLife>> = vec![None; RING];
    let mut resolved_by = vec![0; frames];
    let mut peak = 0;
    for frame in 0..frames {
        let paced = starts.last().map_or(0, |&s| s + interval);
        let start = frame
            .checked_sub(bound)
            .and_then(|k| done.get(k))
            .map_or(paced, |&d| paced.max(d));
        let finished = done.iter().take_while(|&&d| d <= start).count();
        // `begin`: slots whose resolving frame has finished are mapped and taken back.
        for slot in &mut slots {
            if let Some(SlotLife::Resolving(by)) = *slot
                && by < finished
            {
                *slot = None;
            }
        }
        let free = slots.iter_mut().find(|slot| slot.is_none());
        *free.expect("a free slot") = Some(SlotLife::Pending(frame));
        peak = peak.max(slots.iter().flatten().count());
        // `finish`: the earlier frames the GPU has finished are resolved by this one.
        for slot in &mut slots {
            if let Some(SlotLife::Pending(own)) = *slot
                && own < finished
            {
                *slot = Some(SlotLife::Resolving(frame));
                if let Some(by) = resolved_by.get_mut(own) {
                    *by = frame;
                }
            }
        }
        starts.push(start);
        let gpu_free = done.last().copied().unwrap_or(0);
        done.push(start.max(gpu_free) + per_frame);
    }
    for (own, &by) in resolved_by.iter().enumerate().take(frames - bound - 1) {
        let first_after = starts.iter().position(|&s| s >= done[own]);
        assert_eq!(
            Some(by),
            first_after,
            "frame {own}: resolved by the first frame encoded after it finished"
        );
    }
    assert!(
        peak < RING,
        "at most RING − 1 slots busy, one to spare: {peak} of {RING}"
    );
    assert_eq!(
        peak,
        bound + 2,
        "the bound's frames, one resolving, and the new one"
    );
}
