//! A frame's phases from its marks' ticks, with no GPU; the GPU run is in `draw`'s tests.

use super::*;

/// The marks of a frame that drew a scene, in the order `Renderer::draw` writes them.
fn scene_marks() -> Vec<Mark> {
    let ends = GpuPhase::ALL.map(Mark::End);
    std::iter::once(Mark::Start).chain(ends).collect()
}

#[test]
fn each_phase_runs_from_the_mark_before_its_own() {
    let marks = scene_marks();
    let ticks = [100, 110, 130, 160, 200, 250, 310, 380, 460];
    let times = GpuTimes::from_ticks(9, &marks, &ticks, 1.0).unwrap();
    assert_eq!(times.frame(), 9);
    let ns = Duration::from_nanos;
    let wanted = [10, 20, 30, 40, 50, 60, 70, 80];
    for (phase, took) in GpuPhase::ALL.into_iter().zip(wanted) {
        assert_eq!(times.phase(phase), Some(ns(took)), "{phase:?}");
    }
    assert_eq!(times.total(), ns(360), "first to last");
}

#[test]
fn ticks_scale_by_the_period_and_phases_not_run_are_absent() {
    // A frame with no scene: its uploads, then the overlay.
    let marks = [
        Mark::Start,
        Mark::End(GpuPhase::Uploads),
        Mark::End(GpuPhase::Tone),
    ];
    let times = GpuTimes::from_ticks(1, &marks, &[0, 4, 10], 2.5).unwrap();
    assert_eq!(
        times.phase(GpuPhase::Uploads),
        Some(Duration::from_nanos(10))
    );
    assert_eq!(times.phase(GpuPhase::Tone), Some(Duration::from_nanos(15)));
    assert_eq!(times.phase(GpuPhase::Raster), None, "not run");
    assert_eq!(times.total(), Duration::from_nanos(25));
}

#[test]
fn too_few_marks_or_mismatched_ticks_time_nothing() {
    assert_eq!(GpuTimes::from_ticks(1, &[Mark::Start], &[5], 1.0), None);
    let marks = scene_marks();
    assert_eq!(GpuTimes::from_ticks(1, &marks, &[1, 2, 3], 1.0), None);
    // A timestamp that goes backwards (a reset between them) reads as zero, never wraps.
    let back = [Mark::Start, Mark::End(GpuPhase::Shade)];
    let times = GpuTimes::from_ticks(1, &back, &[50, 40], 1.0).unwrap();
    assert_eq!(times.phase(GpuPhase::Shade), Some(Duration::ZERO));
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
