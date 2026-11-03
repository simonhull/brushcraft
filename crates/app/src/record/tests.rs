//! The recorder files each fact under its frame, whenever it arrives, and writes every field,
//! `null` for those that never came.

use super::*;
use crate::script::Speed;

/// Every field of a frame line, in order: what `brushcraft-bench perf` reads.
const FRAME_FIELDS: [&str; 72] = [
    "bench",
    "result",
    "frame",
    "time_ms",
    "presented_ms",
    "segment",
    "moving",
    "cpu_ms",
    "gpu_wait_ms",
    "input_ms",
    "brush_ms",
    "between_ms",
    "feed_ms",
    "uploads_ms",
    "upkeep_ms",
    "prepare_ms",
    "jobs_ms",
    "acquire_ms",
    "encode_ms",
    "present_ms",
    "gpu_ms",
    "gpu_uploads_ms",
    "gpu_bounds_ms",
    "gpu_lod_ms",
    "gpu_raster_ms",
    "gpu_readback_ms",
    "gpu_shade_ms",
    "gpu_accumulate_ms",
    "gpu_tone_ms",
    "lod_bounds_ms",
    "lod_choose_ms",
    "lod_relax_ms",
    "lod_sweep_ms",
    "lod_emit_ms",
    "lod_post_ms",
    "drawn",
    "triangles",
    "cut",
    "tiles_per_level",
    "over_budget",
    "waiting",
    "budget_limited",
    "held_by_neighbours",
    "shortfall",
    "cone_culled",
    "cone_exempt",
    "requested",
    "relax_passes",
    "level_jump",
    "resident",
    "wanted",
    "evicted",
    "uploaded_bytes",
    "uploaded_tiles",
    "standing_in",
    "pending_epochs",
    "cut_readback_dropped",
    "request_readback_dropped",
    "timing_readback_dropped",
    "submitted_ms",
    "sample_to_present_ms",
    "received_to_present_ms",
    "dabs_applied",
    "dabs_shown",
    "dabs_carried",
    "dabs_catching_up",
    "tiles_edited",
    "above_f",
    "stroke_open",
    "kept_tiles",
    "kept_bytes",
    "kept_beyond_budget",
];

/// The names of `line`'s fields, in order.
fn names(line: &Line) -> Vec<&str> {
    line.fields.iter().map(|(name, _)| name.as_str()).collect()
}

/// `line`'s value of `name`.
fn value<'a>(line: &'a Line, name: &str) -> &'a Value {
    &line
        .fields
        .iter()
        .find(|(n, _)| n == name)
        .unwrap_or_else(|| panic!("no {name}"))
        .1
}

/// `ms` milliseconds.
fn ms(ms: u64) -> Duration {
    Duration::from_millis(ms)
}

/// Residency's counts with `evictions` evictions ever.
fn stats(evictions: u64) -> ResidencyStats {
    ResidencyStats {
        resident: 40,
        wanted: 3,
        evictions,
        staged_bytes: 4096,
        staged_tiles: 2,
        ..ResidencyStats::default()
    }
}

#[test]
fn a_frame_line_names_every_field_and_nulls_what_never_came() {
    let mut recorder = Recorder::default();
    recorder.started(1, ms(0), Some(Segment::Open), true);
    let line = recorder.frames[&1].line();
    assert_eq!(names(&line), FRAME_FIELDS);
    for name in [
        "presented_ms",
        "cpu_ms",
        "gpu_wait_ms",
        "gpu_ms",
        "cut_readback_dropped",
        "request_readback_dropped",
        "timing_readback_dropped",
        "lod_post_ms",
        "over_budget",
        "wanted",
        "submitted_ms",
        "sample_to_present_ms",
        "dabs_applied",
        "above_f",
        "stroke_open",
        "kept_bytes",
        "between_ms",
    ] {
        assert_eq!(value(&line, name), &Value::Null, "{name}");
    }
    assert_eq!(value(&line, "drawn"), &Value::Bool(false));
    assert_eq!(value(&line, "segment"), &Value::Str("open".to_owned()));
    let json = line.to_json();
    assert!(
        json.starts_with(r#"{"bench":"render","result":"frame","frame":1,"time_ms":0,"#),
        "{json}"
    );
    assert!(json.contains(r#""gpu_ms":null"#), "{json}");
}

#[test]
fn facts_arriving_late_land_on_their_own_frame() {
    let mut recorder = Recorder::default();
    for frame in 1..=3 {
        recorder.started(frame, ms(frame * 16), Some(Segment::Orbit), true);
        // Residency numbers its frames from 10.
        recorder.staged(frame, Some(frame + 9), Some(&stats(frame * 5)));
    }
    recorder.owner(
        2,
        OwnerPhases {
            input: ms(1),
            feed: ms(2),
            uploads: ms(3),
            jobs: ms(9),
            ..OwnerPhases::default()
        },
    );
    let render = DrawTimes {
        encode: ms(4),
        ..DrawTimes::default()
    };
    recorder.shown(2, render, ms(40));
    recorder.brush(
        2,
        &BrushReport {
            dabs_applied: 3,
            dabs_carried: 1,
            tiles_edited: 7,
            ..BrushReport::default()
        },
    );
    let latency = Latency {
        taken: ms(6),
        received: ms(5),
    };
    recorder.latency(2, ms(39), Some(latency));
    // Frame 2's cut, as residency's frame 11, heard during frame 3.
    let cut = CutHeard {
        frame: 11,
        scene: SceneRecord {
            stats: LodStats {
                over_pixel: 2,
                requested: 5,
                drawn_triangles: 1000,
                raises: vec![3, 1, 0],
                ..LodStats::default()
            },
            levels: vec![0, 6, 12],
            stages: None,
            level_jump: Some(2),
        },
    };
    recorder.cut(cut);
    let line = recorder.frames[&2].line();
    let cpu = value(&line, "cpu_ms");
    assert_eq!(cpu, &Value::Float(10.0), "input + feed + uploads + encode");
    assert_eq!(value(&line, "jobs_ms"), &Value::Float(9.0));
    assert_eq!(value(&line, "presented_ms"), &Value::Float(40.0));
    assert_eq!(value(&line, "over_budget"), &Value::Int(2));
    assert_eq!(value(&line, "requested"), &Value::Int(5));
    assert_eq!(value(&line, "relax_passes"), &Value::Int(2));
    assert_eq!(
        value(&line, "level_jump"),
        &Value::Int(2),
        "the stroke's jump"
    );
    assert_eq!(
        value(&line, "tiles_per_level"),
        &Value::Ints(vec![0, 6, 12])
    );
    assert_eq!(
        value(&line, "evicted"),
        &Value::Int(5),
        "this frame's: 10 − 5"
    );
    assert_eq!(value(&line, "uploaded_tiles"), &Value::Int(2));
    assert_eq!(value(&line, "submitted_ms"), &Value::Float(39.0));
    assert_eq!(value(&line, "sample_to_present_ms"), &Value::Float(6.0));
    assert_eq!(value(&line, "received_to_present_ms"), &Value::Float(5.0));
    assert_eq!(value(&line, "dabs_applied"), &Value::Int(3));
    assert_eq!(value(&line, "dabs_carried"), &Value::Int(1));
    assert_eq!(value(&line, "tiles_edited"), &Value::Int(7));
    assert_eq!(value(&line, "above_f"), &Value::Bool(false));
    assert!(
        line.to_json().contains(r#""tiles_per_level":[0,6,12]"#),
        "{}",
        line.to_json()
    );
    let first = recorder.frames[&1].line();
    assert_eq!(
        value(&first, "drawn"),
        &Value::Bool(false),
        "frame 1's cut never came"
    );
}

#[test]
fn a_run_writes_its_setting_its_segments_then_its_frames() {
    let mut recorder = Recorder::default();
    recorder.started(1, ms(0), Some(Segment::Open), false);
    recorder.started(2, ms(17), Some(Segment::Open), false);
    let info = RunInfo {
        project: "/p/sphere \"x\".bcrf".to_owned(),
        adapter: "llvmpipe (Vulkan)".to_owned(),
        size: (320, 200),
        gpu_timing: true,
        lod_timing: false,
        open: ms(250),
        clock_unix: Some(Duration::from_micros(1_793_401_426_346_500)),
        refresh: Duration::from_micros(6944),
    };
    let script = Script::benchmark(Speed::ONE);
    let lines = recorder.lines(&info, Some(&script));
    assert_eq!(lines.len(), 1 + 5 + 2);
    assert!(
        lines[0]
            .starts_with(r#"{"bench":"render","result":"run","project":"/p/sphere \"x\".bcrf","#),
        "{}",
        lines[0]
    );
    assert!(
        lines[0].contains(r#""open_ms":250"#) && lines[0].contains(r#""frames":2"#),
        "{}",
        lines[0]
    );
    assert!(
        lines[0].contains(r#""gpu_timing":true,"lod_timing":false"#),
        "{}",
        lines[0]
    );
    assert!(
        lines[0].contains(r#""clock_unix_us":1793401426346500,"refresh_ms":6.944,"#),
        "{}",
        lines[0]
    );
    assert_eq!(
        lines[2],
        r#"{"bench":"render","result":"segment","segment":"orbit","start_ms":3000,"end_ms":11000}"#
    );
    assert!(lines[6].contains(r#""frame":1,"#) && lines[7].contains(r#""frame":2,"#));
    assert!(
        lines[0].contains(r#""readbacks_dropped":0"#),
        "{}",
        lines[0]
    );
}

#[test]
fn dropped_readbacks_are_marked_on_their_frame_and_totalled_on_the_run_line() {
    let mut recorder = Recorder::default();
    for frame in 1..=3 {
        recorder.started(frame, ms(frame * 16), Some(Segment::Zoom), true);
    }
    // Residency's request readbacks dropped ever: none, then one as frame 2 was staged.
    let dropping = |dropped| ResidencyStats {
        request_readbacks_dropped: dropped,
        ..stats(0)
    };
    recorder.staged(1, Some(1), Some(&dropping(0)));
    recorder.staged(2, Some(2), Some(&dropping(1)));
    recorder.staged(3, Some(3), Some(&dropping(1)));
    recorder.dropped(1, DroppedReadbacks::default());
    recorder.dropped(2, DroppedReadbacks { cut: 1, timing: 0 });
    recorder.dropped(3, DroppedReadbacks { cut: 1, timing: 1 });
    let flags = |frame| {
        let line = recorder.frames[&frame].line();
        [
            "cut_readback_dropped",
            "request_readback_dropped",
            "timing_readback_dropped",
        ]
        .map(|name| value(&line, name).clone())
    };
    let (yes, no) = (Value::Bool(true), Value::Bool(false));
    assert_eq!(flags(1), [no.clone(), no.clone(), no.clone()]);
    assert_eq!(flags(2), [yes.clone(), yes.clone(), no.clone()]);
    assert_eq!(
        flags(3),
        [yes.clone(), no, yes],
        "this frame's request readback, not every one ever"
    );
    let info = RunInfo {
        project: "p".to_owned(),
        adapter: "a".to_owned(),
        size: (8, 8),
        gpu_timing: true,
        lod_timing: true,
        open: ms(0),
        clock_unix: None,
        refresh: ms(16),
    };
    let lines = recorder.lines(&info, Some(&Script::benchmark(Speed::ONE)));
    assert!(
        lines[0].contains(r#""readbacks_dropped":4"#),
        "{}",
        lines[0]
    );
}

#[test]
fn a_moment_by_the_frame_loop_clock_is_placed_on_the_system_clock() {
    let start = Instant::now();
    let before = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap();
    std::thread::sleep(Duration::from_millis(20));
    let placed = since_epoch(start).unwrap();
    let off = placed.abs_diff(before);
    assert!(
        off < Duration::from_millis(5),
        "{off:?} from the clock read beside it"
    );
}
