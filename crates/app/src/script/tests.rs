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
    let up = pose.camera().axis(crate::camera::Vec3::new(0.0, 1.0, 0.0));
    assert!(up.y > 0.0, "upright: {up:?}");
}
