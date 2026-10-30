//! The render metrics from a recorded run, and from hand-made ones for the edges.
//!
//! The recorded run (`lavapipe.jsonl`) is a scripted run at 4× of a small project on lavapipe,
//! thinned to the open segment's first frames, every eighth orbit frame and the frames around the
//! flick's end.
//!
//! The expected values for the recorded run were worked out apart from this code, with `jq` over
//! the fixture: the orbit's GPU and CPU times sorted and taken at the nearest rank above, the
//! first correct frame of the open segment (frame 5), the first caught-up frame after the flick
//! (frame 227).

use super::*;
use crate::perf::metric::Measured;

/// The recorded run.
const RECORDED: &str = include_str!("lavapipe.jsonl");

/// Whether `a` and `b` agree to rounding.
fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

#[test]
fn a_recorded_run_reads_back_and_gives_every_metric() {
    let recorded = Recorded::parse(RECORDED).unwrap();
    assert!(close(recorded.open_ms, 192.719_746));
    let names: Vec<&str> = recorded.segments.iter().map(|s| s.0.as_str()).collect();
    assert_eq!(names, ["open", "orbit", "zoom", "flick", "settle"]);
    assert_eq!(recorded.frames.len(), 8 + 15 + 7);
    let metrics = derive(&recorded);
    let (gpu50, gpu99) = metrics.orbit_gpu.unwrap();
    assert!(
        close(gpu50, 6.768_356) && close(gpu99, 7.810_756),
        "{metrics:?}"
    );
    let (cpu50, cpu99) = metrics.orbit_cpu.unwrap();
    assert!(
        close(cpu50, 0.683_362) && close(cpu99, 0.987_497),
        "{metrics:?}"
    );
    assert_eq!(metrics.orbit_over_budget, Some(0.0));
    assert!(
        close(metrics.open_to_correct_ms.unwrap(), 288.247_405),
        "{metrics:?}"
    );
    assert!(
        close(metrics.spin_catch_up_ms.unwrap(), 9.081_083),
        "{metrics:?}"
    );
}

#[test]
fn the_metrics_are_named_for_the_scale_and_sampled_once_a_run() {
    let metrics = derive(&Recorded::parse(RECORDED).unwrap());
    let mut samples = Samples::default();
    for _ in 0..3 {
        let missing = metrics.sample(Scale::Medium, &mut samples).unwrap();
        assert_eq!(missing, Vec::<&str>::new(), "every metric");
    }
    let measured = samples.measure().unwrap();
    let names: Vec<&str> = measured.iter().map(|m| m.name.as_str()).collect();
    assert_eq!(
        names,
        [
            "medium.render.orbit_gpu_p50_ms",
            "medium.render.orbit_gpu_p99_ms",
            "medium.render.orbit_cpu_p50_ms",
            "medium.render.orbit_cpu_p99_ms",
            "medium.render.orbit_over_budget",
            "medium.render.spin_catch_up_ms",
            "medium.render.open_to_correct_ms",
        ]
    );
    assert!(measured.iter().all(|m| m.value.samples().len() == 3));
    assert_eq!(measured[4].unit, Unit::Tiles);
}

#[test]
fn only_the_benchmark_project_is_held_to_the_spec_targets() {
    let metric = |name: &str| Metric {
        name: name.parse().unwrap(),
        unit: Unit::Ms,
        value: Measured::of(vec![1.0]).unwrap(),
        ceiling: None,
    };
    let names = [
        "full.render.orbit_gpu_p99_ms",
        "full.render.orbit_over_budget",
        "full.render.spin_catch_up_ms",
        "full.render.open_to_correct_ms",
    ];
    let full = with_targets(names.map(metric).to_vec(), Scale::Full);
    let ceilings: Vec<Option<f64>> = full.iter().map(|m| m.ceiling).collect();
    assert_eq!(ceilings, [None, Some(0.0), Some(100.0), Some(1000.0)]);
    let medium = with_targets(
        vec![metric("medium.render.spin_catch_up_ms")],
        Scale::Medium,
    );
    assert_eq!(medium[0].ceiling, None, "compared with history only");
}

/// A frame at `time_ms` in `segment`, shown 1 ms later, its cut back with `over` tiles over
/// 1 px, `requested` requested, and residency wanting `wanted`.
fn frame(time_ms: f64, segment: &str, (over, requested, wanted): (f64, f64, f64)) -> Frame {
    Frame {
        time_ms,
        presented_ms: Some(time_ms + 1.0),
        segment: Some(segment.to_owned()),
        gpu_ms: None,
        cpu_ms: Some(2.0),
        drawn: true,
        over_budget: Some(over),
        requested: Some(requested),
        wanted: Some(wanted),
    }
}

/// A run opening in `open_ms`, its segments as the benchmark's at 1×, with `frames`.
fn run(open_ms: f64, frames: Vec<Frame>) -> Recorded {
    let segments = [
        ("open", 0.0, 3000.0),
        ("orbit", 3000.0, 11_000.0),
        ("zoom", 11_000.0, 15_000.0),
        ("flick", 15_000.0, 15_150.0),
        ("settle", 15_150.0, 18_150.0),
    ];
    Recorded {
        open_ms,
        segments: segments
            .iter()
            .map(|&(name, start, end)| (name.to_owned(), start, end))
            .collect(),
        frames,
    }
}

#[test]
fn a_run_with_no_gpu_times_and_never_correct_lacks_those_metrics_only() {
    let busy = (2.0, 5.0, 3.0);
    let frames = vec![
        frame(0.0, "open", busy),
        frame(2000.0, "open", busy),
        // Correct, but only once the orbit began: not open-to-correct.
        frame(3000.0, "orbit", (0.0, 0.0, 0.0)),
        frame(4000.0, "orbit", (4.0, 1.0, 0.0)),
        // Before the flick ends: not a catch-up, however quiet.
        frame(15_100.0, "flick", (0.0, 0.0, 0.0)),
        frame(15_160.0, "settle", (0.0, 2.0, 1.0)),
        frame(15_190.0, "settle", (0.0, 0.0, 1.0)),
        frame(15_220.0, "settle", (0.0, 0.0, 0.0)),
    ];
    let metrics = derive(&run(400.0, frames));
    assert_eq!(metrics.orbit_gpu, None, "no timestamp queries");
    assert_eq!(metrics.orbit_cpu, Some((2.0, 2.0)));
    assert_eq!(
        metrics.orbit_over_budget,
        Some(4.0),
        "the most in any orbit frame"
    );
    assert_eq!(
        metrics.open_to_correct_ms, None,
        "never correct while opening"
    );
    assert_eq!(
        metrics.spin_catch_up_ms,
        Some(15_221.0 - 15_150.0),
        "the first still frame with nothing wanted or requested, as shown"
    );
    let mut samples = Samples::default();
    let missing = metrics.sample(Scale::Tiny, &mut samples).unwrap();
    assert_eq!(
        missing,
        [
            "render.orbit_gpu_p50_ms",
            "render.orbit_gpu_p99_ms",
            "render.open_to_correct_ms"
        ]
    );
}

#[test]
fn open_to_correct_counts_the_open_itself_and_skips_frames_whose_cut_never_came() {
    let mut lost = frame(100.0, "open", (0.0, 0.0, 0.0));
    lost.drawn = false;
    let frames = vec![
        frame(0.0, "open", (3.0, 9.0, 0.0)),
        lost,
        frame(200.0, "open", (0.0, 0.0, 4.0)),
    ];
    let metrics = derive(&run(250.0, frames));
    assert_eq!(
        metrics.open_to_correct_ms,
        Some(250.0 + 201.0),
        "the open, then until frame 3 was shown (wanted tiles off screen don't matter)"
    );
    assert_eq!(metrics.spin_catch_up_ms, None, "no frame after the flick");
}

#[test]
fn percentiles_take_the_nearest_rank_above() {
    let values: Vec<f64> = (1..=10).map(f64::from).collect();
    assert_eq!(percentile(&values, 50), Some(6.0));
    assert_eq!(percentile(&values, 99), Some(10.0));
    assert_eq!(percentile(&values, 0), Some(1.0));
    assert_eq!(percentile(&[3.0], 99), Some(3.0));
    assert_eq!(percentile(&[], 50), None);
}

#[test]
fn records_that_dont_read_are_refused() {
    let frame_line = r#"{"bench":"render","result":"frame","time_ms":1}"#;
    assert!(Recorded::parse(frame_line).is_err(), "no run line");
    let run_line = r#"{"bench":"render","result":"run","open_ms":5}"#;
    assert!(
        Recorded::parse(&format!("{run_line}\n{{oops")).is_err(),
        "not JSON"
    );
    let no_time = r#"{"bench":"render","result":"frame","gpu_ms":1}"#;
    assert!(
        Recorded::parse(&format!("{run_line}\n{no_time}")).is_err(),
        "a frame needs its time"
    );
    let read = Recorded::parse(&format!("{run_line}\n\n{frame_line}\n")).unwrap();
    assert_eq!(read.frames.len(), 1, "blank lines skipped");
    assert_eq!(read.frames[0].gpu_ms, None);
}

#[test]
fn the_measured_scales_run_on_a_hardware_gpu_at_4k_and_the_tiny_check_on_lavapipe() {
    for scale in [Scale::Medium, Scale::Full] {
        let setting = Setting::of(scale);
        assert_eq!(
            (setting.adapter, setting.size, setting.speed),
            ("hardware", "3840x2160", "1")
        );
        assert!(setting.runs >= 3);
    }
    assert_eq!(Setting::of(Scale::Tiny).adapter, "software");
}
