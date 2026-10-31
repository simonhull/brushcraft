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
