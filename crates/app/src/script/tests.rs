//! The path is a pure function of time: the same at every call, continuous across its segments,
//! and each segment does what it says.

use super::*;

/// The benchmark path at real speed.
fn path() -> Script {
    Script::benchmark(Speed::ONE)
}

/// `seconds` as a duration.
fn at(seconds: f64) -> Duration {
    Duration::from_secs_f64(seconds)
}

/// Halfway from `start` to `end`.
fn middle(start: Duration, end: Duration) -> Duration {
    start + end.checked_sub(start).unwrap() / 2
}

/// Whether two poses agree to rounding.
fn close(a: Pose, b: Pose) -> bool {
    (a.yaw - b.yaw).abs() < 1e-9
        && (a.pitch - b.pitch).abs() < 1e-9
        && (a.distance - b.distance).abs() < 1e-9
}

#[test]
fn the_path_is_the_same_at_every_call_and_in_every_copy() {
    let (a, b) = (path(), path());
    assert_eq!(a, b);
    let steps = (0..2000).map(|i| at(f64::from(i) * 0.01));
    for time in steps {
        assert_eq!(a.at(time), a.at(time), "{time:?}");
        assert_eq!(a.at(time), b.at(time), "{time:?}");
    }
}

#[test]
fn the_segments_run_in_order_back_to_back() {
    let script = path();
    let spans = script.segments();
    let names: Vec<Segment> = spans.iter().map(|span| span.0).collect();
    assert_eq!(
        names,
        [
            Segment::Open,
            Segment::Orbit,
            Segment::Zoom,
            Segment::Flick,
            Segment::Settle
        ]
    );
    assert_eq!(spans[0].1, Duration::ZERO, "from the start");
    for pair in spans.windows(2) {
        assert_eq!(pair[0].2, pair[1].1, "{:?} then {:?}", pair[0], pair[1]);
    }
    assert_eq!(spans[4].2, script.duration(), "to the end");
    for &(segment, start, end) in &spans {
        let middle = middle(start, end);
        assert_eq!(script.at(middle).segment, segment, "{segment}");
        assert!(!script.at(middle).over);
    }
    let end = script.at(script.duration());
    assert!(end.over && end.segment == Segment::Settle, "{end:?}");
}

#[test]
fn the_pose_is_continuous_across_every_boundary() {
    let script = path();
    for (segment, _, end) in script.segments() {
        let before = script.at(end.saturating_sub(Duration::from_nanos(1))).pose;
        let after = script.at(end).pose;
        assert!(
            (before.yaw - after.yaw).abs() < 1e-6
                && (before.distance - after.distance).abs() < 1e-6,
            "{segment}: {before:?} then {after:?}"
        );
    }
}

#[test]
fn each_segment_moves_the_camera_as_it_says() {
    let script = path();
    let spans = script.segments();
    let pose = |time: Duration| script.at(time).pose;
    let [open, orbit, zoom, flick, settle] = [0, 1, 2, 3, 4].map(|i| spans[i]);
    assert!(
        close(pose(open.1), START) && close(pose(open.2.checked_sub(at(1e-3)).unwrap()), START),
        "still"
    );
    assert!(
        close(pose(orbit.1), START),
        "the orbit starts where the camera was"
    );
    let half = pose(middle(orbit.1, orbit.2));
    assert!(
        (half.yaw - PI).abs() < 1e-6,
        "half a turn halfway: {half:?}"
    );
    assert!(close(pose(orbit.2), ORBITED), "a whole turn");
    assert!((pose(zoom.2).distance - 1.1).abs() < 1e-9, "close up");
    let zoom_mid = pose(middle(zoom.1, zoom.2)).distance;
    assert!(
        (zoom_mid - (3.0_f64 * 1.1).sqrt()).abs() < 1e-9,
        "geometric: halfway in log-distance at the middle: {zoom_mid}"
    );
    assert_eq!(
        flick.2.checked_sub(flick.1),
        Some(at(0.15)),
        "a flick is fast"
    );
    assert!(
        (pose(flick.2).yaw - pose(flick.1).yaw - PI).abs() < 1e-9,
        "half a turn"
    );
    assert!(
        close(pose(settle.1), FLICKED) && close(pose(settle.2), FLICKED),
        "still"
    );
}

#[test]
fn a_faster_speed_runs_the_same_path_in_less_time() {
    let real = path();
    let fast = Script::benchmark("4".parse().unwrap());
    assert_eq!(fast.duration(), real.duration().div_f64(4.0));
    for i in 0..200 {
        let time = at(f64::from(i) * 0.02);
        let (a, b) = (fast.at(time), real.at(time * 4));
        assert!(close(a.pose, b.pose), "{time:?}: {a:?} {b:?}");
        assert_eq!(a.segment, b.segment, "{time:?}");
    }
    for bad in ["0", "-1", "nan", "inf", "fast", ""] {
        assert!(bad.parse::<Speed>().is_err(), "{bad:?}");
    }
}

#[test]
fn a_pose_puts_the_eye_where_its_angles_say() {
    let pose = Pose {
        yaw: 0.7,
        pitch: 0.3,
        distance: 2.0,
    };
    let eye = pose.camera().eye();
    let want = [
        0.7_f64.sin() * 0.3_f64.cos() * 2.0,
        0.3_f64.sin() * 2.0,
        0.7_f64.cos() * 0.3_f64.cos() * 2.0,
    ];
    for (got, wanted) in [eye.x, eye.y, eye.z].into_iter().zip(want) {
        assert!((got - wanted).abs() < 1e-12, "{eye:?} {want:?}");
    }
    let up = pose.camera().axis(Vec3::new(0.0, 1.0, 0.0));
    assert!(up.y > 0.0, "upright: {up:?}");
}

#[test]
fn the_strokes_path_closes_in_draws_backs_off_and_draws_again() {
    let script = Script::strokes(Speed::ONE);
    let names: Vec<Segment> = script.segments().iter().map(|span| span.0).collect();
    assert_eq!(
        names,
        [
            Segment::Open,
            Segment::Approach,
            Segment::StrokeNear,
            Segment::Retreat,
            Segment::StrokeFar,
            Segment::Turn,
            Segment::StrokeLong,
            Segment::Turn,
            Segment::StrokeHeavy,
            Segment::Settle
        ]
    );
    assert!(
        close(script.at(at(5.5)).pose, NEAR),
        "close for the first stroke"
    );
    assert!(close(script.at(at(9.0)).pose, FAR), "far for the second");
    assert!(
        close(script.at(at(12.0)).pose, ROUND),
        "half a turn round for the long one"
    );
    assert!(
        close(script.at(at(15.0)).pose, LAST),
        "three quarters round for the heavy one"
    );
    assert_eq!(Segment::StrokeLong.name(), "stroke_long");
    assert_eq!(Segment::StrokeHeavy.name(), "stroke_heavy");
    assert_eq!(Segment::Turn.name(), "turn");
    assert_eq!(
        Script::benchmark(Speed::ONE).pen(at(0.0), at(30.0)),
        &[],
        "no pen on the benchmark path"
    );
    for (text, path) in [
        ("benchmark", ScriptPath::Benchmark),
        ("strokes", ScriptPath::Strokes),
    ] {
        assert_eq!(text.parse::<ScriptPath>(), Ok(path));
    }
    "orbit".parse::<ScriptPath>().unwrap_err();
}

#[test]
fn a_scripted_pen_hovers_touches_drags_lifts_and_leaves_at_its_rate() {
    let script = Script::strokes(Speed::ONE);
    let all = script.pen(at(0.0), at(20.0));
    // Cut anywhere, the same samples.
    let mut halves = script.pen(at(0.0), at(5.37)).to_vec();
    halves.extend_from_slice(script.pen(at(5.37), at(20.0)));
    assert_eq!(all, halves, "the same samples however the run is cut");
    let near: Vec<&ScriptPen> = all.iter().filter(|s| s.time < at(7.0)).collect();
    let phases: Vec<Phase> = near.iter().map(|s| s.phase).collect();
    assert_eq!(phases.first(), Some(&Phase::Enter));
    assert_eq!(phases.last(), Some(&Phase::Leave));
    let down = phases
        .iter()
        .position(|&p| p == Phase::Down)
        .expect("a down");
    let up = phases.iter().position(|&p| p == Phase::Up).expect("an up");
    assert!(phases[1..down].iter().all(|&p| p == Phase::Hover));
    assert!(phases[down + 1..up].iter().all(|&p| p == Phase::Drag));
    assert_eq!(up + 2, phases.len(), "then the leave");
    // 0.3 s of hover, 1.2 s of stroke, at 173 Hz.
    assert!((51..=53).contains(&down), "{down}");
    assert!((257..=261).contains(&up), "{up}");
    let gaps = near[..up]
        .windows(2)
        .map(|w| w[1].time.checked_sub(w[0].time).unwrap().as_secs_f64());
    for gap in gaps {
        assert!((gap * PEN_RATE - 1.0).abs() < 1e-3, "{gap}");
    }
    assert_eq!(
        near[0].radius,
        Some(40.0),
        "the stroke's size, as it enters"
    );
    assert!(near[1..].iter().all(|s| s.radius.is_none()));
    let mut pressing = near[down..up].iter().map(|s| s.pressure.get());
    assert!(pressing.all(|p| (0.34..=0.81).contains(&p)), "pressed");
    assert!(near[up].pressure.get() == 0.0 && near[0].pressure.get() == 0.0);
    // Each point lies on the unit sphere, on the side facing the camera.
    for sample in all {
        let p = sample.point;
        assert!((p.length() - 1.0).abs() < 1e-9, "{p:?}");
        let eye = sample.camera.eye();
        assert!(p.x * eye.x + p.y * eye.y + p.z * eye.z > 0.0, "facing");
    }
    let far = all
        .iter()
        .find(|s| s.time > at(7.0))
        .expect("the far stroke");
    assert_eq!(far.radius, Some(6.0));
}

#[test]
fn the_long_stroke_runs_fast_across_the_whole_sphere_at_forty_pixels() {
    let script = Script::strokes(Speed::ONE);
    let long: Vec<&ScriptPen> = script
        .pen(at(11.0), at(13.0))
        .iter()
        .filter(|s| s.phase.in_stroke())
        .collect();
    let (first, last) = (long.first().expect("a down"), long.last().expect("an up"));
    assert_eq!(first.phase, Phase::Down);
    assert_eq!(last.phase, Phase::Up);
    let lasted = last.time.checked_sub(first.time).expect("in order");
    assert!(
        (lasted.as_secs_f64() - 0.8).abs() < 0.01,
        "fast: {lasted:?}"
    );
    let entered = script
        .pen(at(11.0), at(13.0))
        .first()
        .expect("the pen enters");
    assert_eq!(entered.radius, Some(40.0), "the default brush");
    // From about 39° one side of the point facing the camera to 39° the other.
    let facing = |sample: &ScriptPen| {
        let eye = sample.camera.eye();
        let p = sample.point;
        (p.x * eye.x + p.y * eye.y + p.z * eye.z) / eye.dot(eye).sqrt()
    };
    for end in [first, last] {
        let angle = facing(end).acos().to_degrees();
        assert!((38.0..40.0).contains(&angle), "{angle}");
    }
}

#[test]
fn the_next_scripted_sample_is_found_after_any_time() {
    let script = Script::strokes(Speed::ONE);
    let all = script.pen(at(0.0), at(30.0));
    let first = all.first().expect("samples").time;
    assert_eq!(script.next_pen_after(Duration::ZERO), Some(first));
    assert_eq!(
        script.next_pen_after(first),
        Some(all[1].time),
        "strictly after"
    );
    let end = all.last().expect("samples").time;
    assert_eq!(script.next_pen_after(end), None, "none after the last");
    assert_eq!(Script::benchmark(Speed::ONE).next_pen_after(at(1.0)), None);
}

#[test]
fn the_heavy_stroke_is_the_long_stroke_pressed_at_full_pressure() {
    let script = Script::strokes(Speed::ONE);
    let heavy: Vec<&ScriptPen> = script
        .pen(at(14.0), at(16.0))
        .iter()
        .filter(|s| s.phase.in_stroke())
        .collect();
    let (first, last) = (heavy.first().expect("a down"), heavy.last().expect("an up"));
    assert_eq!(first.phase, Phase::Down);
    assert_eq!(last.phase, Phase::Up);
    let lasted = last.time.checked_sub(first.time).expect("in order");
    assert!(
        (lasted.as_secs_f64() - 0.8).abs() < 0.01,
        "as fast as the long one: {lasted:?}"
    );
    let pressing = heavy.iter().filter(|s| s.phase != Phase::Up);
    assert!(
        pressing.map(|s| s.pressure).all(|p| p == Unit::ONE),
        "full pressure all the way"
    );
}
