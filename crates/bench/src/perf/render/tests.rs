//! The render metrics from a recorded run, and from hand-made ones for the edges.
//!
//! The recorded run (`lavapipe.jsonl`) is a scripted run at 4× of a small project on lavapipe,
//! thinned to the open segment's first frames, every eighth orbit frame and the frames around the
//! flick's end.
//!
//! The expected values for the recorded run were worked out apart from this code, with `jq` over
//! the fixture: the orbit's GPU and CPU times sorted and taken at the nearest rank above, the
//! first correct frame of the open segment (frame 5), the first caught-up frame after the flick
//! (frame 227: shown with nothing over 1 px, requested or wanted; the run's frames after the
//! flick are all at their chosen levels, so the hand-made runs below cover a frame quiet in
//! residency but over 1 px).

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
    assert_eq!(
        metrics.orbit_gpu_full_clock, metrics.orbit_gpu,
        "no clocks sampled: every orbit frame"
    );
    assert_eq!(metrics.clocks, Clocks::Unknown(NoClocks::NotSampled));
    assert_eq!(
        metrics.orbit_frames_over_refresh,
        Some(0.0),
        "none over 1/60 s"
    );
    let (cpu50, cpu99) = metrics.orbit_cpu.unwrap();
    assert!(
        close(cpu50, 0.683_362) && close(cpu99, 0.987_497),
        "{metrics:?}"
    );
    assert_eq!(metrics.orbit_over_budget, Some(0.0));
    let after = |reached: Option<Reached>| match reached {
        Some(Reached::After(ms)) => ms,
        other => panic!("reached: {other:?}"),
    };
    assert!(
        close(after(metrics.open_to_correct_ms), 288.247_405),
        "{metrics:?}"
    );
    assert!(
        close(after(metrics.spin_catch_up_ms), 9.081_083),
        "{metrics:?}"
    );
}

#[test]
fn the_metrics_are_named_for_the_scale_and_sampled_once_a_run() {
    let metrics = derive(&Recorded::parse(RECORDED).unwrap());
    let mut samples = Samples::default();
    for _ in 0..3 {
        let gaps = metrics.sample(Scale::Medium, &mut samples).unwrap();
        assert_eq!(gaps, Vec::<Gap>::new(), "every metric");
    }
    let measured = samples.measure().unwrap();
    let names: Vec<&str> = measured.iter().map(|m| m.name.as_str()).collect();
    assert_eq!(
        names,
        [
            "medium.render.orbit_gpu_p50_ms",
            "medium.render.orbit_gpu_p99_ms",
            "medium.render.orbit_gpu_p50_full_clock_ms",
            "medium.render.orbit_gpu_p99_full_clock_ms",
            "medium.render.orbit_frames_over_refresh",
            "medium.render.orbit_cpu_p50_ms",
            "medium.render.orbit_cpu_p99_ms",
            "medium.render.orbit_over_budget",
            "medium.render.spin_catch_up_ms",
            "medium.render.open_to_correct_ms",
        ]
    );
    assert!(
        measured
            .iter()
            .all(|m| m.value.measured().unwrap().samples().len() == 3)
    );
    assert_eq!(measured[4].unit, Unit::Frames);
    assert_eq!(measured[7].unit, Unit::Tiles);
}

#[test]
fn only_the_benchmark_project_is_held_to_the_spec_targets() {
    let metric = |name: &str| Metric {
        name: name.parse().unwrap(),
        unit: Unit::Ms,
        value: Measured::of(vec![1.0]).unwrap().into(),
        ceiling: None,
    };
    let names = [
        "full.render.orbit_gpu_p99_ms",
        "full.render.orbit_gpu_p50_full_clock_ms",
        "full.render.orbit_gpu_p99_full_clock_ms",
        "full.render.orbit_frames_over_refresh",
        "full.render.orbit_over_budget",
        "full.render.spin_catch_up_ms",
        "full.render.open_to_correct_ms",
    ];
    let full = with_targets(names.map(metric).to_vec(), Scale::Full);
    let ceilings: Vec<Option<f64>> = full.iter().map(|m| m.ceiling).collect();
    assert_eq!(
        ceilings,
        [
            None,
            None,
            Some(3.0),
            Some(0.0),
            Some(0.0),
            Some(100.0),
            Some(1000.0)
        ]
    );
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
        ..Frame::default()
    }
}

/// A frame of a stroke segment at `time_ms`.
///
/// It applied `applied` dabs and left `carried`, its stroke `open` or not, its latency
/// `latency_ms` (with dabs applied), submitted 3 ms after its start, its brush 1 ms, its
/// preparation 0.1 ms.
fn stroke_frame(
    time_ms: f64,
    (applied, carried, open): (f64, f64, bool),
    latency_ms: f64,
) -> Frame {
    Frame {
        time_ms,
        segment: Some("stroke_near".to_owned()),
        submitted_ms: Some(time_ms + 3.0),
        sample_to_present_ms: (applied > 0.0).then_some(latency_ms),
        brush_ms: Some(1.0),
        prepare_ms: Some(0.1),
        dabs_applied: Some(applied),
        dabs_carried: Some(carried),
        above_f: Some(false),
        stroke_open: Some(open),
        ..Frame::default()
    }
}

/// A strokes run of `frames`, all in `stroke_near`.
fn stroke_run(frames: Vec<Frame>) -> Recorded {
    Recorded {
        segments: vec![("stroke_near".to_owned(), 0.0, 1000.0)],
        frames,
        ..Recorded::default()
    }
}

#[test]
fn a_stroke_gives_its_latency_its_queue_and_its_drain() {
    let frames = vec![
        stroke_frame(0.0, (0.0, 0.0, false), 0.0),
        stroke_frame(6.0, (2.0, 1.0, true), 5.0),
        stroke_frame(12.0, (3.0, 0.0, true), 7.5),
        stroke_frame(18.0, (2.0, 3.0, true), 6.0),
        stroke_frame(24.0, (2.0, 1.0, false), 9.0),
        stroke_frame(30.0, (1.0, 0.0, false), 4.0),
        stroke_frame(36.0, (0.0, 0.0, false), 0.0),
    ];
    let [near, far, long, heavy] = strokes::derive(&stroke_run(frames));
    assert_eq!(near.sample_to_present_p99, Some(9.0), "the nearest rank");
    assert_eq!(near.start_to_submitted_p99, Some(3.0));
    assert_eq!(near.brush_p99, Some(1.0));
    assert_eq!(near.most_carried, Some(3.0));
    assert_eq!(
        near.drain,
        Some(strokes::Drain::Frames(2.0)),
        "the second frame after pen-up carries none"
    );
    assert_eq!(near.above_f, Some(0.0));
    assert_eq!(near.prepare_p99, Some(0.1));
    assert_eq!(far, strokes::StrokeMetrics::default(), "no such segment");
    assert_eq!(long, strokes::StrokeMetrics::default());
    assert_eq!(heavy, strokes::StrokeMetrics::default());
}

#[test]
fn a_queue_never_drained_within_its_segment_is_not_reached_and_a_miss_on_the_benchmark() {
    let frames = vec![
        stroke_frame(0.0, (1.0, 5.0, true), 5.0),
        stroke_frame(6.0, (1.0, 9.0, false), 30.0),
        stroke_frame(12.0, (1.0, 8.0, false), 40.0),
    ];
    let mut samples = Samples::default();
    let gaps = sample_strokes(&stroke_run(frames), Scale::Full, &mut samples).unwrap();
    assert!(
        gaps.contains(&Gap::NotReached(
            "render.stroke_near.drain_frames",
            Unit::Frames
        )),
        "{gaps:?}"
    );
    let metrics = with_targets(samples.measure().unwrap(), Scale::Full);
    let ceiling = |name: &str| {
        metrics
            .iter()
            .find(|m| m.name.as_str() == name)
            .and_then(|m| m.ceiling)
    };
    assert_eq!(
        ceiling("full.render.stroke_near.sample_to_present_p99_ms"),
        Some(8.0),
        "the spec's pen latency"
    );
    assert_eq!(
        ceiling("full.render.stroke_near.dabs_carried_max"),
        Some(strokes::MOST_CARRIED)
    );
    assert_eq!(ceiling("full.render.stroke_near.brush_p99_ms"), None);
    let carried = metrics
        .iter()
        .find(|m| m.name.as_str() == "full.render.stroke_near.dabs_carried_max")
        .expect("measured");
    assert_eq!(carried.unit, Unit::Dabs);
    let name: MetricName = "full.render.stroke_far.drain_frames".parse().unwrap();
    assert_eq!(ceiling_of(&name, Scale::Full), Some(2.0));
    assert_eq!(ceiling_of(&name, Scale::Medium), None);
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
        ..Recorded::default()
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
        // Residency quiet, but a tile held over 1 px: not every tile at its chosen level.
        frame(15_220.0, "settle", (3.0, 0.0, 0.0)),
        frame(15_250.0, "settle", (0.0, 0.0, 0.0)),
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
        metrics.open_to_correct_ms,
        Some(Reached::Not),
        "never correct while opening"
    );
    assert_eq!(
        metrics.spin_catch_up_ms,
        Some(Reached::After(15_251.0 - 15_150.0)),
        "the first still frame with nothing over 1 px, wanted or requested, as shown"
    );
    let mut samples = Samples::default();
    let gaps = metrics.sample(Scale::Tiny, &mut samples).unwrap();
    let said: Vec<String> = gaps.iter().map(Gap::to_string).collect();
    assert_eq!(
        said,
        [
            "no render.orbit_gpu_p50_ms",
            "no render.orbit_gpu_p99_ms",
            "no render.orbit_gpu_p50_full_clock_ms",
            "no render.orbit_gpu_p99_full_clock_ms",
            "no render.orbit_frames_over_refresh",
            "render.open_to_correct_ms not reached"
        ]
    );
}

#[test]
fn a_run_that_never_catches_up_says_so_rather_than_timing_its_end() {
    let frames = vec![
        frame(0.0, "open", (0.0, 0.0, 0.0)),
        frame(14_000.0, "zoom", (0.0, 0.0, 0.0)),
        // Residency quiet to the end, but tiles held over 1 px the whole settle.
        frame(15_160.0, "settle", (12.0, 0.0, 0.0)),
        frame(16_500.0, "settle", (12.0, 0.0, 0.0)),
        frame(18_140.0, "settle", (12.0, 0.0, 0.0)),
    ];
    let metrics = derive(&run(100.0, frames));
    assert_eq!(metrics.spin_catch_up_ms, Some(Reached::Not));
    let mut samples = Samples::default();
    let gaps = metrics.sample(Scale::Tiny, &mut samples).unwrap();
    assert!(
        gaps.contains(&Gap::NotReached("render.spin_catch_up_ms", Unit::Ms)),
        "{gaps:?}"
    );
    assert_eq!(
        Gap::NotReached("render.spin_catch_up_ms", Unit::Ms).to_string(),
        "render.spin_catch_up_ms not reached"
    );
    let measured = samples.measure().unwrap();
    assert!(
        measured
            .iter()
            .all(|m| m.name.as_str() != "tiny.render.spin_catch_up_ms"),
        "no number stands in for it"
    );
    // A run with no flick lacks the metric; it isn't "not reached".
    let mut flickless = run(100.0, Vec::new());
    flickless.segments.retain(|(name, _, _)| name != "flick");
    assert_eq!(derive(&flickless).spin_catch_up_ms, None);
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
        Some(Reached::After(250.0 + 201.0)),
        "the open, then until frame 3 was shown (wanted tiles off screen don't matter)"
    );
    assert_eq!(
        metrics.spin_catch_up_ms,
        Some(Reached::Not),
        "no frame after the flick"
    );
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

/// Three runs of a project at `scale`, the spin catch-up reached in the first `reached` (at 50
/// and 60 ms) and never in the rest: the part's metrics and notes for it.
fn three_runs(scale: Scale, reached: usize) -> (Vec<Metric>, Vec<String>) {
    let mut samples = Samples::default();
    let mut unreached: Vec<Unreached> = Vec::new();
    for run in 0..3 {
        let catch_up = [50.0, 60.0]
            .get(run)
            .filter(|_| run < reached)
            .map_or(Reached::Not, |&ms| Reached::After(ms));
        let metrics = RenderMetrics {
            spin_catch_up_ms: Some(catch_up),
            ..RenderMetrics::default()
        };
        for gap in metrics.sample(scale, &mut samples).unwrap() {
            if let Gap::NotReached(name, unit) = gap {
                match unreached.iter_mut().find(|u| u.name == name) {
                    Some(known) => known.runs += 1,
                    None => unreached.push(Unreached {
                        name,
                        unit,
                        runs: 1,
                    }),
                }
            }
        }
    }
    let metrics = with_targets(samples.measure().unwrap(), scale);
    judge_unreached(metrics, &unreached, 3, scale).unwrap()
}

#[test]
fn on_the_benchmark_project_a_catch_up_never_reached_is_a_miss_even_if_other_runs_reached_it() {
    let (metrics, notes) = three_runs(Scale::Full, 2);
    let catch_up: Vec<&Metric> = metrics
        .iter()
        .filter(|m| m.name.as_str() == "full.render.spin_catch_up_ms")
        .collect();
    assert_eq!(catch_up.len(), 1, "{metrics:?}");
    assert_eq!(catch_up[0].value, Outcome::NotReached, "the stricter call");
    assert_eq!(catch_up[0].ceiling, Some(100.0));
    assert_eq!(
        notes,
        [
            "render.spin_catch_up_ms not reached in 1 of 3 runs: judged not reached against its \
             ceiling (100 ms), the stricter call"
        ]
    );
    let (none, none_notes) = three_runs(Scale::Full, 0);
    assert!(
        none.iter()
            .any(|m| m.name.as_str() == "full.render.spin_catch_up_ms"
                && m.value == Outcome::NotReached),
        "recorded though no run gave a number: {none:?}"
    );
    assert!(
        none_notes[0].contains("not reached in any of 3 runs"),
        "{none_notes:?}"
    );
}

#[test]
fn without_a_ceiling_a_catch_up_never_reached_stays_a_note() {
    let (metrics, notes) = three_runs(Scale::Medium, 2);
    let catch_up = metrics
        .iter()
        .find(|m| m.name.as_str() == "medium.render.spin_catch_up_ms")
        .unwrap();
    assert_eq!(catch_up.ceiling, None);
    assert_eq!(
        catch_up.value.measured().map(Measured::samples),
        Some(&[50.0, 60.0][..]),
        "measured from the runs that reached it"
    );
    assert_eq!(
        notes,
        [
            "render.spin_catch_up_ms not reached in 1 of 3 runs: no ceiling, measured from the others"
        ]
    );
}

/// The system clock's time, in ms since the epoch, of the hand-made clocked runs' frame loop's
/// clock starting.
const ORIGIN_MS: f64 = 1_793_401_426_000.0;

/// A sample of GPU 0's clocks `at` ms after [`ORIGIN_MS`]: graphics clock `mhz`, power state
/// `state`.
fn sample(at: f64, mhz: f64, state: &str) -> ClockSample {
    ClockSample {
        at_unix_ms: ORIGIN_MS + at,
        gpu: 0,
        graphics_mhz: mhz,
        memory_mhz: Some(10_501.0),
        pstate: Some(state.parse().unwrap()),
    }
}

/// A run on an NVIDIA GPU placed at [`ORIGIN_MS`], its orbit frames at `(time_ms, gpu_ms)`.
fn clocked_run(frames: &[(f64, f64)]) -> Recorded {
    let orbit = frames.iter().map(|&(time_ms, gpu_ms)| Frame {
        gpu_ms: Some(gpu_ms),
        ..frame(time_ms, "orbit", (0.0, 0.0, 0.0))
    });
    Recorded {
        adapter: "NVIDIA GeForce RTX 4090 (Vulkan)".to_owned(),
        clock_unix_ms: Some(ORIGIN_MS),
        ..run(100.0, orbit.collect())
    }
}

/// The orbit frames of the clocked runs: start and GPU time, in ms.
const ORBIT_FRAMES: [(f64, f64); 6] = [
    (3000.0, 0.25),
    (3040.0, 0.30),
    (3090.0, 0.70),
    (3140.0, 0.60),
    (3210.0, 20.0),
    // Nothing sampled within 100 ms: its clock is unknown.
    (3500.0, 0.90),
];

/// Samples beside the clocked runs: the highest graphics clock 2700 MHz, so full clocks are
/// P0 at 2430 MHz or more.
fn orbit_samples() -> Vec<ClockSample> {
    vec![
        sample(3200.0, 2700.0, "P0"),
        sample(3000.0, 2700.0, "P0"),
        sample(3050.0, 2650.0, "P0"),
        // A low clock state.
        sample(3100.0, 1005.0, "P3"),
        // P0, but below 90 % of the highest.
        sample(3150.0, 2400.0, "P0"),
    ]
}

#[test]
fn each_frame_takes_the_clocks_nearest_its_start_and_only_full_clocks_count() {
    let mut recorded = clocked_run(&ORBIT_FRAMES);
    recorded.join_clocks(Ok(orbit_samples()));
    let Clocks::Joined(full) = recorded.clocks else {
        panic!("joined: {:?}", recorded.clocks);
    };
    assert!((full.floor_mhz - 2430.0).abs() < 1e-9, "{full:?}");
    let joined: Vec<Option<f64>> = recorded
        .frames
        .iter()
        .map(|f| f.clock.map(|c| c.graphics_mhz))
        .collect();
    assert_eq!(
        joined,
        [
            Some(2700.0),
            Some(2650.0),
            Some(1005.0),
            Some(2400.0),
            Some(2700.0),
            None
        ]
    );
    let metrics = derive(&recorded);
    assert_eq!(
        metrics.orbit_gpu_full_clock,
        Some((0.30, 20.0)),
        "frames 1, 2 and 5: p50 and p99 by the nearest rank above"
    );
    assert_eq!(metrics.full_clock_frames, (3, 5), "of five with a clock");
    let (p50, p99) = metrics.orbit_gpu.unwrap();
    assert!(
        (p50 - 0.70).abs() < 1e-12 && (p99 - 20.0).abs() < 1e-12,
        "every frame as before"
    );
    assert_eq!(
        metrics.orbit_frames_over_refresh,
        Some(1.0),
        "20 ms over 1/60 s, at full clocks or not"
    );
    assert_eq!(
        metrics.clock_note(),
        "3 of 5 timed orbit frames at full clocks (P0, graphics at least 2430 MHz)"
    );
}

#[test]
fn the_refresh_is_the_records_where_it_gives_one() {
    let mut recorded = clocked_run(&[(3000.0, 20.0), (3020.0, 7.5), (3040.0, 6.5)]);
    recorded.refresh_ms = Some(1000.0 / 144.0);
    assert_eq!(derive(&recorded).orbit_frames_over_refresh, Some(2.0));
    recorded.refresh_ms = Some(25.0);
    assert_eq!(derive(&recorded).orbit_frames_over_refresh, Some(0.0));
}

#[test]
fn without_clocks_the_full_clock_metrics_fall_back_to_every_frame_and_say_why() {
    let other = Recorded {
        adapter: "llvmpipe (LLVM 23.1.3, 256 bits) (Vulkan)".to_owned(),
        ..clocked_run(&ORBIT_FRAMES)
    };
    let placeless = Recorded {
        clock_unix_ms: None,
        ..clocked_run(&ORBIT_FRAMES)
    };
    let mut two_gpus = orbit_samples();
    two_gpus.push(ClockSample {
        gpu: 1,
        ..sample(3000.0, 2100.0, "P0")
    });
    let not_found = NoClocks::NoSampler(std::io::ErrorKind::NotFound);
    let cases = [
        (clocked_run(&ORBIT_FRAMES), Err(not_found), not_found),
        (
            clocked_run(&ORBIT_FRAMES),
            Ok(Vec::new()),
            NoClocks::NoSamples,
        ),
        (
            clocked_run(&ORBIT_FRAMES),
            Ok(two_gpus),
            NoClocks::SeveralGpus,
        ),
        (other, Ok(orbit_samples()), NoClocks::OtherAdapter),
        (placeless, Ok(orbit_samples()), NoClocks::NoClockOrigin),
    ];
    for (mut recorded, sampled, why) in cases {
        recorded.join_clocks(sampled);
        assert_eq!(recorded.clocks, Clocks::Unknown(why));
        assert!(recorded.frames.iter().all(|f| f.clock.is_none()), "{why}");
        let metrics = derive(&recorded);
        assert_eq!(metrics.orbit_gpu_full_clock, metrics.orbit_gpu, "{why}");
        assert_eq!(metrics.orbit_frames_over_refresh, Some(1.0), "{why}");
        assert_eq!(
            metrics.clock_note(),
            format!("no GPU clocks ({why}): the full-clock metrics are over every orbit frame")
        );
    }
    assert_eq!(
        not_found.to_string(),
        "nvidia-smi didn't start: entity not found"
    );
}

#[test]
fn the_run_line_places_the_clock_and_gives_the_refresh() {
    let text = r#"{"bench":"render","result":"run","adapter":"NVIDIA GeForce RTX 4090 (Vulkan)","open_ms":12.5,"clock_unix_us":1793401426346500,"refresh_ms":6.944}"#;
    let recorded = Recorded::parse(text).unwrap();
    assert_eq!(recorded.adapter, "NVIDIA GeForce RTX 4090 (Vulkan)");
    assert_eq!(recorded.clock_unix_ms, Some(1_793_401_426_346.5));
    assert_eq!(recorded.refresh_ms, Some(6.944));
    let older = Recorded::parse(RECORDED).unwrap();
    assert_eq!((older.clock_unix_ms, older.refresh_ms), (None, None));
}
