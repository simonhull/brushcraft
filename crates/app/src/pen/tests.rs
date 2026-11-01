//! The pen stream without a window.
//!
//! A recorded stroke through the fallback's de-jitter filter, the fallback chosen when our
//! backend can't start, the backend stopped before the window, and the stream as navigation's
//! input.

use std::cell::RefCell;
use std::rc::Rc;
use std::time::{Duration, Instant};

use anyhow::bail;
use input::{Buttons, Phase, Tool, Unit};
use winit::event::{Force, TabletToolData, TabletToolKind, TabletToolTilt};

use super::fallback::WinitButton;
use super::*;
use crate::camera::{Arcball, Camera};
use crate::navigate::{Button, Gesture, Input, Navigator};

/// What happened to the test's backend and window, in order.
type Log = Rc<RefCell<Vec<&'static str>>>;

/// A stand-in for our backend: hands out what it was given, in logical pixels, and logs being
/// stopped and dropped.
#[derive(Debug)]
struct Mock {
    /// Samples it will hand out, positions in logical pixels.
    queued: Vec<PenSample>,
    /// Samples it hands out only once stopped: they arrived after the last drain.
    late: Vec<PenSample>,
    /// Whether its thread still reads; shared so a test can stop it.
    running: Rc<RefCell<bool>>,
    /// Why it stopped, if on an error.
    error: Option<&'static str>,
    /// Where it logs.
    log: Log,
}

impl Mock {
    /// A running backend that logs to `log`.
    fn new(log: &Log) -> Self {
        Self {
            queued: Vec::new(),
            late: Vec::new(),
            running: Rc::new(RefCell::new(true)),
            error: None,
            log: Rc::clone(log),
        }
    }
}

/// `sample` with its position scaled by `scale`, as our backend turns logical pixels physical.
fn scaled(sample: PenSample, scale: f64) -> PenSample {
    PenSample {
        at: Pixel {
            x: sample.at.x * scale,
            y: sample.at.y * scale,
        },
        ..sample
    }
}

impl Backend for Mock {
    fn drain(&mut self, scale: f64, out: &mut Vec<PenSample>) {
        let drained = self.queued.drain(..).map(|sample| scaled(sample, scale));
        out.extend(drained);
    }

    fn running(&self) -> bool {
        *self.running.borrow()
    }

    fn losses(&self) -> Losses {
        Losses {
            dropped: 3,
            lost: 0,
        }
    }

    fn stop(mut self, scale: f64, out: &mut Vec<PenSample>) -> (Losses, Result<()>) {
        self.log.borrow_mut().push("backend stopped");
        self.drain(scale, out);
        let late = self.late.drain(..).map(|sample| scaled(sample, scale));
        out.extend(late);
        // More lost after the last look: the final count is the stop's.
        let losses = Losses {
            dropped: 5,
            lost: 1,
        };
        let result = match self.error {
            Some(error) => Err(anyhow::anyhow!("{error}")),
            None => Ok(()),
        };
        (losses, result)
    }
}

impl Drop for Mock {
    fn drop(&mut self) {
        self.log.borrow_mut().push("backend dropped");
    }
}

/// A stand-in for the window: logs being dropped (the surface destroyed).
#[derive(Debug)]
struct Window(Log);

impl Drop for Window {
    fn drop(&mut self) {
        self.0.borrow_mut().push("window dropped");
    }
}

/// A sample in `phase` at `(x, y)`, taken and received at `at`.
fn sample(phase: Phase, (x, y): (f64, f64), at: Instant) -> PenSample {
    PenSample {
        phase,
        tool: Tool::Pen,
        at: Pixel { x, y },
        pressure: if phase.in_stroke() {
            Unit::ONE
        } else {
            Unit::ZERO
        },
        distance: None,
        tilt: None,
        roll: None,
        buttons: Buttons::default(),
        timestamp: at,
        received: at,
    }
}

/// The stream over a [`Window`], with the backend `spawn` gives.
fn stream(log: &Log, spawn: impl FnOnce(&Window) -> Result<Mock>) -> PenStream<Mock, Window> {
    PenStream::start(Window(Rc::clone(log)), 2.0, spawn)
}

#[test]
fn a_backend_that_cannot_start_falls_back_to_winit_and_says_why() {
    let log = Log::default();
    let mut pen = stream(&log, |_| bail!("not a Wayland display or window"));
    let status = pen.status();
    assert_eq!(
        status.source,
        PenSource::Fallback {
            why: "not a Wayland display or window".into()
        }
    );
    assert!(
        status
            .to_string()
            .contains("fallback: not a Wayland display"),
        "the overlay says so: {status}"
    );
    let t0 = Instant::now();
    pen.winit(
        WinitEvent::Entered,
        Some(Pixel { x: 5.0, y: 6.0 }),
        None,
        (t0, 1),
    );
    let mut out = Vec::new();
    pen.drain(&mut out);
    assert_eq!(out.len(), 1, "winit's events are the stream");
    let first = out.first().expect("one sample");
    assert_eq!(first.phase, Phase::Enter);
    assert_eq!(first.at, Pixel { x: 5.0, y: 6.0 }, "physical already");
    assert!(log.borrow().is_empty(), "no backend ran");
}

#[test]
fn a_running_backend_is_the_stream_and_winits_events_are_ignored() {
    let log = Log::default();
    let t0 = Instant::now();
    let mut pen = stream(&log, |_| {
        let mut mock = Mock::new(&log);
        mock.queued.push(sample(Phase::Enter, (10.0, 20.0), t0));
        Ok(mock)
    });
    assert_eq!(pen.status().source, PenSource::Ours);
    assert_eq!(
        pen.status().dropped,
        3,
        "the backend's losses reach the overlay"
    );
    pen.winit(WinitEvent::Moved, Some(Pixel::default()), None, (t0, 1));
    let mut out = Vec::new();
    pen.drain(&mut out);
    let positions: Vec<Pixel> = out.iter().map(|sample| sample.at).collect();
    assert_eq!(
        positions,
        [Pixel { x: 20.0, y: 40.0 }],
        "ours only, logical pixels made physical at the scale"
    );
}

#[test]
fn a_backend_that_stops_on_its_own_hands_over_to_the_fallback() {
    let log = Log::default();
    let t0 = Instant::now();
    let running = Rc::new(RefCell::new(true));
    let mut pen = stream(&log, |_| {
        let mut mock = Mock::new(&log);
        mock.running = Rc::clone(&running);
        mock.late.push(sample(Phase::Leave, (1.0, 1.0), t0));
        mock.error = Some("the Wayland connection failed");
        Ok(mock)
    });
    *running.borrow_mut() = false;
    let mut out = Vec::new();
    pen.drain(&mut out);
    let phases: Vec<Phase> = out.iter().map(|sample| sample.phase).collect();
    assert_eq!(
        phases,
        [Phase::Leave],
        "what it queued before stopping is kept"
    );
    assert_eq!(*log.borrow(), ["backend stopped", "backend dropped"]);
    let status = pen.status();
    assert_eq!(
        status.source,
        PenSource::Fallback {
            why: "our backend stopped: the Wayland connection failed".into()
        }
    );
    assert_eq!(
        (status.dropped, status.lost),
        (5, 1),
        "its final losses, counted at the stop, stay counted"
    );
    pen.winit(WinitEvent::Entered, Some(Pixel::default()), None, (t0, 2));
    pen.drain(&mut out);
    assert_eq!(out.len(), 2, "winit's events follow");
}

#[test]
fn dropping_the_stream_stops_the_backend_before_the_window_goes() {
    let log = Log::default();
    let pen = stream(&log, |_| Ok(Mock::new(&log)));
    drop(pen);
    assert_eq!(*log.borrow(), ["backend dropped", "window dropped"]);
}

#[test]
fn stopping_the_stream_stops_the_backend_then_hands_back_the_window() {
    let log = Log::default();
    let t0 = Instant::now();
    let pen = stream(&log, |_| {
        let mut mock = Mock::new(&log);
        mock.late.push(sample(Phase::Up, (3.0, 4.0), t0));
        Ok(mock)
    });
    let mut out = Vec::new();
    let (window, stopped) = pen.stop(&mut out);
    stopped.expect("stopped cleanly");
    assert_eq!(
        *log.borrow(),
        ["backend stopped", "backend dropped"],
        "the backend is gone before the window is handed back"
    );
    assert_eq!(out.len(), 1, "what arrived since the last drain");
    drop(window);
    assert_eq!(
        log.borrow().last().copied(),
        Some("window dropped"),
        "the window last"
    );
}

#[test]
fn a_backends_error_at_the_stop_is_reported() {
    let log = Log::default();
    let pen = stream(&log, |_| {
        let mut mock = Mock::new(&log);
        mock.error = Some("the input thread panicked");
        Ok(mock)
    });
    let (_window, stopped) = pen.stop(&mut Vec::new());
    let error = stopped.expect_err("the error comes back");
    assert!(error.to_string().contains("panicked"), "{error:#}");
}

/// A pen's sample period at the Kamvas's measured 173 Hz.
const PERIOD: Duration = Duration::from_nanos(5_780_347);

/// Delivery delays recorded on the event-loop thread, in microseconds: the spread of a
/// compositor and a busy main thread, up to most of a period, repeating.
const DELAYS_US: [u64; 12] = [
    410, 3_900, 1_200, 5_100, 700, 2_600, 4_400, 300, 1_900, 3_300, 900, 4_800,
];

/// A recorded fallback session: the pen enters, hovers, draws, and leaves.
///
/// It presses `strokes` strokes of `length` samples each. Every event comes in a wakeup of its
/// own, received at its delay after the pen's clock; each comes with its receipt and the pen's
/// true time.
fn recorded(strokes: u32, length: u32) -> Vec<(WinitEvent, Pixel, Instant, Instant)> {
    let t0 = Instant::now();
    let mut events = Vec::new();
    let mut slot = 0_u32;
    let mut push = |event: WinitEvent, into: &mut Vec<_>| {
        let delay = DELAYS_US
            .get(usize::try_from(slot).expect("small") % DELAYS_US.len())
            .copied()
            .unwrap_or(0);
        let truth = t0 + PERIOD * slot;
        let at = Pixel {
            x: f64::from(slot) * 3.0,
            y: 50.0,
        };
        into.push((event, at, truth + Duration::from_micros(delay), truth));
        slot += 1;
    };
    push(WinitEvent::Entered, &mut events);
    for _ in 0..4 {
        push(WinitEvent::Moved, &mut events);
    }
    for _ in 0..strokes {
        push(
            WinitEvent::Button {
                button: WinitButton::Contact,
                pressed: true,
            },
            &mut events,
        );
        for _ in 0..length {
            push(WinitEvent::Moved, &mut events);
        }
        push(
            WinitEvent::Button {
                button: WinitButton::Contact,
                pressed: false,
            },
            &mut events,
        );
        for _ in 0..3 {
            push(WinitEvent::Moved, &mut events);
        }
    }
    push(WinitEvent::Left, &mut events);
    events
}

/// The standard deviation of the gaps between `times`, in seconds, skipping the first `skip`.
fn gap_sd(times: &[Instant], skip: usize) -> f64 {
    let gaps: Vec<f64> = times
        .windows(2)
        .filter_map(|pair| match pair {
            [a, b] => Some(b.duration_since(*a).as_secs_f64()),
            _ => None,
        })
        .skip(skip)
        .collect();
    let n = f64::from(u32::try_from(gaps.len()).expect("few"));
    let total: f64 = gaps.iter().sum();
    let mean = total / n;
    let squares: f64 = gaps.iter().map(|gap| (gap - mean).powi(2)).sum();
    (squares / n).sqrt()
}

#[test]
fn a_recorded_fallback_stroke_keeps_its_order_and_runs_on_the_pens_clock() {
    let log = Log::default();
    let mut pen = stream(&log, |_| bail!("no Wayland tablet support"));
    let events = recorded(2, 300);
    let numbered = events.iter().zip(1_u64..);
    for (&(event, at, received, _), wakeup) in numbered {
        pen.winit(event, Some(at), None, (received, wakeup));
    }
    let mut out = Vec::new();
    pen.drain(&mut out);
    assert_eq!(out.len(), events.len(), "one sample per event, none lost");

    // Order: as received, positions and phases in sequence.
    let in_order = out
        .iter()
        .zip(&events)
        .all(|(sample, &(_, at, received, _))| sample.at == at && sample.received == received);
    assert!(in_order, "samples keep the events' order");
    let mut runs: Vec<Phase> = out.iter().map(|sample| sample.phase).collect();
    runs.dedup();
    assert_eq!(
        runs,
        [
            Phase::Enter,
            Phase::Hover,
            Phase::Down,
            Phase::Drag,
            Phase::Up,
            Phase::Hover,
            Phase::Down,
            Phase::Drag,
            Phase::Up,
            Phase::Hover,
            Phase::Leave
        ]
    );

    // Timing: a stroke's samples are never stamped after their receipt, never go backwards,
    // and the second stroke (seeded by the first's period) runs far steadier than receipt.
    assert!(
        out.iter().all(|sample| sample.timestamp <= sample.received),
        "never after the receipt"
    );
    let strokes: Vec<&[PenSample]> = out
        .split(|sample| !sample.phase.in_stroke())
        .filter(|stroke| !stroke.is_empty())
        .collect();
    assert_eq!(strokes.len(), 2);
    let second = strokes.get(1).expect("two strokes");
    let stamped: Vec<Instant> = second.iter().map(|sample| sample.timestamp).collect();
    let received: Vec<Instant> = second.iter().map(|sample| sample.received).collect();
    assert!(
        stamped.windows(2).all(|pair| pair.first() <= pair.last()),
        "never backwards"
    );
    let (raw, filtered) = (gap_sd(&received, 0), gap_sd(&stamped, 16));
    assert!(raw > 1e-3, "the receipts jitter: sd {raw}");
    assert!(
        filtered < raw / 10.0,
        "de-jittered gap sd {filtered} against the receipts' {raw}"
    );
    let truths: Vec<Instant> = events
        .iter()
        .zip(&out)
        .filter(|(_, sample)| sample.phase.in_stroke())
        .map(|(&(_, _, _, truth), _)| truth)
        .collect();
    let second_truths = truths.get(302..).expect("the second stroke's truths");
    let worst = stamped
        .iter()
        .zip(second_truths)
        .skip(16)
        .map(|(stamp, truth)| stamp.duration_since(*truth).as_secs_f64())
        .fold(0.0, f64::max);
    assert!(
        worst < 1e-3,
        "within a millisecond of the pen's clock once settled: {worst}"
    );

    // Hover and leave samples carry their receipt time, and no pressure.
    let hovering = out
        .iter()
        .filter(|sample| !sample.phase.in_stroke())
        .all(|sample| sample.timestamp == sample.received && sample.pressure == Unit::ZERO);
    assert!(hovering);
}

#[test]
fn the_fallback_reads_winits_tool_data() {
    let data = TabletToolData {
        force: Some(Force::Normalized(0.25)),
        tilt: Some(TabletToolTilt { x: 30, y: -10 }),
        twist: Some(90),
        ..TabletToolData::default()
    };
    let tool = ToolData::from_winit(TabletToolKind::Eraser, &data);
    assert_eq!(tool.tool, Tool::Eraser);
    assert!((tool.pressure.get() - 0.25).abs() < 1e-6);
    let tilt = tool.tilt.expect("tilt");
    assert_eq!((tilt.x(), tilt.y()), (30.0, -10.0));
    assert_eq!(tool.roll, Some(90.0));
    let over = TabletToolData {
        force: Some(Force::Normalized(1.7)),
        ..TabletToolData::default()
    };
    assert_eq!(
        ToolData::from_winit(TabletToolKind::Pen, &over).pressure,
        Unit::ONE,
        "clamped"
    );
    let none = ToolData::from_winit(TabletToolKind::Pen, &TabletToolData::default());
    assert_eq!((none.pressure, none.tilt), (Unit::ZERO, None));
}

#[test]
fn on_the_fallback_barrel_one_is_the_tip_and_barrel_two_a_button() {
    let mut winit = WinitPen::default();
    let t0 = Instant::now();
    let at = Some(Pixel::default());
    let tool = Some(ToolData {
        tool: Tool::Pen,
        pressure: Unit::ONE,
        tilt: None,
        roll: None,
    });
    let contact = |pressed| WinitEvent::Button {
        button: WinitButton::Contact,
        pressed,
    };
    let barrel = winit.apply(
        WinitEvent::Button {
            button: WinitButton::Barrel,
            pressed: true,
        },
        at,
        tool,
        (t0, 1),
    );
    assert_eq!(
        (barrel.phase, barrel.buttons),
        (Phase::Hover, Buttons::SECONDARY)
    );
    let down = winit.apply(contact(true), at, tool, (t0, 2));
    assert_eq!(
        down.phase,
        Phase::Down,
        "winit's contact: the tip or barrel one"
    );
    assert_eq!(down.pressure, Unit::ONE);
    let again = winit.apply(contact(true), at, tool, (t0, 3));
    assert_eq!(again.phase, Phase::Drag, "a second contact changes nothing");
    let up = winit.apply(contact(false), at, tool, (t0, 4));
    assert_eq!((up.phase, up.pressure), (Phase::Up, Unit::ZERO));
    let left = winit.apply(WinitEvent::Left, None, None, (t0, 5));
    assert_eq!(left.phase, Phase::Leave);
    assert_eq!(
        left.buttons,
        Buttons::default(),
        "leaving releases the barrel"
    );
}

/// Starts `frame` over with `samples`, routed by `router`, the brush asking for `claims`.
fn take(frame: &mut PenFrame, router: &mut PenRouter, samples: &[PenSample], claims: Route) {
    frame.clear();
    for sample in samples {
        let route = router.route(sample, |_| claims);
        frame.push(*sample, route);
    }
}

#[test]
fn a_frame_takes_its_samples_in_order_and_keeps_the_hover() {
    let t0 = Instant::now();
    let mut frame = PenFrame::default();
    let mut router = PenRouter::default();
    let brush = Route::Brush;
    take(
        &mut frame,
        &mut router,
        &[
            sample(Phase::Enter, (1.0, 1.0), t0),
            sample(Phase::Hover, (2.0, 1.0), t0),
        ],
        brush,
    );
    assert_eq!(frame.samples().len(), 2);
    assert!(frame.strokes().is_empty(), "out of a stroke: both");
    let hover = frame.hover().expect("hovering");
    assert_eq!(hover.at, Pixel { x: 2.0, y: 1.0 });

    take(&mut frame, &mut router, &[], brush);
    assert!(
        frame.samples().is_empty(),
        "a frame with no samples has none"
    );
    assert_eq!(
        frame.hover().map(|still| still.at),
        Some(Pixel { x: 2.0, y: 1.0 }),
        "a pen hovering still still hovers"
    );

    take(
        &mut frame,
        &mut router,
        &[
            sample(Phase::Hover, (3.0, 1.0), t0),
            sample(Phase::Down, (3.0, 1.0), t0),
            sample(Phase::Drag, (4.0, 1.0), t0),
        ],
        brush,
    );
    assert_eq!(frame.hover(), None, "touching isn't hovering");
    assert_eq!(frame.strokes(), [(1..3, Route::Brush)]);

    take(
        &mut frame,
        &mut router,
        &[
            sample(Phase::Drag, (5.0, 1.0), t0),
            sample(Phase::Up, (5.0, 1.0), t0),
            sample(Phase::Hover, (6.0, 1.0), t0),
        ],
        Route::Navigate,
    );
    assert_eq!(
        frame.strokes(),
        [(0..2, Route::Brush)],
        "a stroke keeps the route its Down chose, across frames"
    );
    assert!(frame.hover().is_some(), "lifted: in range again");
    take(
        &mut frame,
        &mut router,
        &[sample(Phase::Leave, (6.0, 1.0), t0)],
        brush,
    );
    assert_eq!(frame.hover(), None, "gone");
}

#[test]
fn the_brush_reads_its_strokes_and_the_hover_never_a_navigation_stroke() {
    let t0 = Instant::now();
    let mut frame = PenFrame::default();
    let mut router = PenRouter::default();
    let samples = [
        sample(Phase::Hover, (1.0, 1.0), t0),
        sample(Phase::Down, (1.0, 1.0), t0),
        sample(Phase::Drag, (2.0, 1.0), t0),
        sample(Phase::Up, (3.0, 1.0), t0),
        sample(Phase::Hover, (4.0, 1.0), t0),
    ];
    take(&mut frame, &mut router, &samples, Route::Navigate);
    let read: Vec<Phase> = frame.for_brush().map(|sample| sample.phase).collect();
    assert_eq!(read, [Phase::Hover, Phase::Hover]);
    take(&mut frame, &mut router, &samples, Route::Brush);
    assert_eq!(
        frame.for_brush().count(),
        5,
        "the brush's stroke, and the hover"
    );
}

#[test]
fn a_barrel_held_at_the_down_navigates_whatever_the_brush_asks() {
    let t0 = Instant::now();
    let down = holding(sample(Phase::Down, (0.0, 0.0), t0), Buttons::SECONDARY);
    assert_eq!(route(&down, |_| Route::Brush), Route::Navigate);
    let bare = sample(Phase::Down, (0.0, 0.0), t0);
    assert_eq!(route(&bare, |_| Route::Brush), Route::Brush);
}

#[test]
fn a_lost_up_ends_the_routed_stroke() {
    let t0 = Instant::now();
    let mut router = PenRouter::default();
    let decided: Vec<Option<Route>> = [
        sample(Phase::Down, (0.0, 0.0), t0),
        sample(Phase::Drag, (1.0, 0.0), t0),
        sample(Phase::Hover, (2.0, 0.0), t0),
        sample(Phase::Drag, (3.0, 0.0), t0),
    ]
    .iter()
    .map(|sample| router.route(sample, |_| Route::Brush))
    .collect();
    assert_eq!(
        decided,
        [
            Some(Route::Brush),
            Some(Route::Brush),
            None,
            Some(Route::Navigate)
        ],
        "a drag with no down before it navigates, where it does nothing"
    );
}

/// The navigation input `samples` make, in order.
fn navigation(samples: &[PenSample]) -> Vec<Input> {
    let mut nav = PenNavigation::default();
    let mut inputs = Vec::new();
    for sample in samples {
        nav.inputs(sample, |input| inputs.push(input));
    }
    inputs
}

/// `sample` with `buttons` held.
const fn holding(sample: PenSample, buttons: Buttons) -> PenSample {
    PenSample { buttons, ..sample }
}

#[test]
fn the_tip_drags_and_hovering_moves_the_pointer() {
    let t0 = Instant::now();
    let at = |x: f64| Pixel { x, y: 0.0 };
    let inputs = navigation(&[
        sample(Phase::Enter, (1.0, 0.0), t0),
        sample(Phase::Hover, (2.0, 0.0), t0),
        sample(Phase::Down, (3.0, 0.0), t0),
        sample(Phase::Drag, (4.0, 0.0), t0),
        sample(Phase::Up, (5.0, 0.0), t0),
        sample(Phase::Leave, (5.0, 0.0), t0),
    ]);
    assert_eq!(
        inputs,
        [
            Input::Move { at: at(1.0) },
            Input::Move { at: at(2.0) },
            Input::Move { at: at(3.0) },
            Input::Press {
                button: Button::PenTip,
                at: at(3.0)
            },
            Input::Move { at: at(4.0) },
            Input::Move { at: at(5.0) },
            Input::Release {
                button: Button::PenTip
            },
        ]
    );
}

#[test]
fn barrel_buttons_are_modifiers_pressed_before_the_tip() {
    let t0 = Instant::now();
    let origin = (100.0, 50.0);
    let ball = Arcball::new(render::surface::Size::new(200, 100).expect("non-zero"));
    let gesture = |buttons: Buttons| {
        let samples = [
            sample(Phase::Hover, origin, t0),
            holding(sample(Phase::Down, origin, t0), buttons),
            holding(sample(Phase::Drag, (120.0, 30.0), t0), buttons),
        ];
        let mut navigator = Navigator::default();
        let mut camera = Camera::default();
        let inputs = navigation(&samples);
        for input in inputs {
            navigator.apply(input, &mut camera, &ball, 100.0);
        }
        (navigator.gesture(), camera)
    };
    assert_eq!(gesture(Buttons::default()).0, Some(Gesture::Orbit));
    let (pan, panned) = gesture(Buttons::SECONDARY);
    assert_eq!(pan, Some(Gesture::Pan), "barrel two pans");
    assert_ne!(panned.target(), Camera::default().target());
    let (zoom, zoomed) = gesture(Buttons::PRIMARY);
    assert_eq!(zoom, Some(Gesture::Zoom), "barrel one zooms");
    assert!(
        zoomed.distance() < Camera::default().distance(),
        "dragged up: closer"
    );
    assert_eq!(
        gesture(Buttons::PRIMARY.with(Buttons::SECONDARY, true)).0,
        Some(Gesture::Zoom),
        "zoom wins"
    );
}

#[test]
fn leaving_releases_whatever_the_pen_held() {
    let t0 = Instant::now();
    let both = Buttons::PRIMARY.with(Buttons::SECONDARY, true);
    let inputs = navigation(&[
        holding(sample(Phase::Down, (0.0, 0.0), t0), both),
        holding(sample(Phase::Leave, (0.0, 0.0), t0), both),
    ]);
    let releases: Vec<Button> = inputs
        .iter()
        .filter_map(|input| match input {
            Input::Release { button } => Some(*button),
            Input::Press { .. }
            | Input::Move { .. }
            | Input::Wheel { .. }
            | Input::Modifiers(_)
            | Input::Cancel => None,
        })
        .collect();
    assert_eq!(
        releases,
        [Button::PenFirstBarrel, Button::PenBarrel, Button::PenTip]
    );
}

#[test]
fn a_lost_up_ends_the_old_drag_before_the_next() {
    let t0 = Instant::now();
    let inputs = navigation(&[
        sample(Phase::Down, (0.0, 0.0), t0),
        sample(Phase::Down, (9.0, 0.0), t0),
    ]);
    let tip: Vec<&Input> = inputs
        .iter()
        .filter(|input| !matches!(input, Input::Move { .. }))
        .collect();
    assert_eq!(tip.len(), 3, "press, release, press: {inputs:?}");
    assert!(matches!(
        tip.get(1),
        Some(Input::Release {
            button: Button::PenTip
        })
    ));
}

#[test]
fn a_stroke_left_half_done_at_the_hand_over_ends_with_a_leave() {
    let log = Log::default();
    let t0 = Instant::now();
    let running = Rc::new(RefCell::new(true));
    let mut pen = stream(&log, |_| {
        let mut mock = Mock::new(&log);
        mock.running = Rc::clone(&running);
        mock.late.push(sample(Phase::Down, (10.0, 10.0), t0));
        mock.late.push(sample(Phase::Drag, (12.0, 10.0), t0));
        Ok(mock)
    });
    *running.borrow_mut() = false;
    let mut out = Vec::new();
    pen.drain(&mut out);
    let phases: Vec<Phase> = out.iter().map(|sample| sample.phase).collect();
    assert_eq!(phases, [Phase::Down, Phase::Drag, Phase::Leave]);
    let leave = out.last().expect("a leave");
    assert_eq!(
        leave.at,
        Pixel { x: 24.0, y: 20.0 },
        "where the stroke stopped"
    );
    assert_eq!(leave.pressure, Unit::ZERO);
    let inputs = navigation(&out);
    assert_eq!(
        inputs.last(),
        Some(&Input::Release {
            button: Button::PenTip
        }),
        "navigation's drag ends: {inputs:?}"
    );
}

#[test]
fn a_backend_that_stops_between_strokes_adds_no_leave() {
    let log = Log::default();
    let t0 = Instant::now();
    let running = Rc::new(RefCell::new(true));
    let mut pen = stream(&log, |_| {
        let mut mock = Mock::new(&log);
        mock.running = Rc::clone(&running);
        mock.queued.push(sample(Phase::Down, (1.0, 1.0), t0));
        mock.late.push(sample(Phase::Up, (1.0, 1.0), t0));
        Ok(mock)
    });
    let mut out = Vec::new();
    pen.drain(&mut out);
    *running.borrow_mut() = false;
    pen.drain(&mut out);
    let phases: Vec<Phase> = out.iter().map(|sample| sample.phase).collect();
    assert_eq!(phases, [Phase::Down, Phase::Up], "the up ended it already");
}

#[test]
fn a_lost_up_then_hover_releases_the_tip() {
    let t0 = Instant::now();
    let inputs = navigation(&[
        sample(Phase::Down, (0.0, 0.0), t0),
        sample(Phase::Drag, (5.0, 0.0), t0),
        sample(Phase::Hover, (6.0, 0.0), t0),
    ]);
    assert_eq!(
        inputs.get(3..),
        Some(
            &[
                Input::Release {
                    button: Button::PenTip
                },
                Input::Move {
                    at: Pixel { x: 6.0, y: 0.0 }
                },
            ][..]
        ),
        "{inputs:?}"
    );
}

#[test]
fn winit_taking_the_pen_from_a_silent_backend_hands_over_to_the_fallback() {
    let log = Log::default();
    let t0 = Instant::now();
    let mut pen = stream(&log, |_| Ok(Mock::new(&log)));
    let mut out = Vec::new();
    let held = STARVED - 1;
    let events = (0..held).map(|i| {
        (
            i,
            if i == 0 {
                WinitEvent::Entered
            } else {
                WinitEvent::Moved
            },
        )
    });
    for (i, event) in events {
        let x = f64::from(u32::try_from(i).expect("small"));
        pen.winit(event, Some(Pixel { x, y: 0.0 }), None, (t0, 1));
    }
    pen.drain(&mut out);
    assert!(out.is_empty(), "held back while ours may still deliver");
    assert_eq!(pen.status().source, PenSource::Ours);
    pen.winit(WinitEvent::Moved, Some(Pixel::default()), None, (t0, 2));
    assert!(matches!(pen.status().source, PenSource::Fallback { .. }));
    assert_eq!(*log.borrow(), ["backend stopped", "backend dropped"]);
    pen.drain(&mut out);
    assert_eq!(out.len(), STARVED, "every held sample is kept, in order");
    assert_eq!(out.first().map(|sample| sample.phase), Some(Phase::Enter));
}

#[test]
fn winits_duplicates_are_dropped_while_ours_delivers() {
    let log = Log::default();
    let t0 = Instant::now();
    let mut pen = stream(&log, |_| {
        let mut mock = Mock::new(&log);
        mock.queued.push(sample(Phase::Enter, (1.0, 1.0), t0));
        Ok(mock)
    });
    pen.winit(WinitEvent::Entered, Some(Pixel::default()), None, (t0, 1));
    let mut out = Vec::new();
    pen.drain(&mut out);
    assert_eq!(out.len(), 1, "ours only");
    assert_eq!(pen.status().source, PenSource::Ours);
}

#[test]
fn a_barrel_pressed_and_released_mid_drag_leaves_the_orbit_alone() {
    let t0 = Instant::now();
    let ball = Arcball::new(render::surface::Size::new(200, 100).expect("non-zero"));
    let samples = [
        sample(Phase::Down, (100.0, 50.0), t0),
        holding(sample(Phase::Drag, (110.0, 50.0), t0), Buttons::SECONDARY),
        holding(sample(Phase::Drag, (120.0, 50.0), t0), Buttons::SECONDARY),
        sample(Phase::Drag, (130.0, 50.0), t0),
    ];
    let mut navigator = Navigator::default();
    let mut camera = Camera::default();
    let inputs = navigation(&samples);
    let mut before_last = camera;
    let count = inputs.len();
    let numbered = inputs.into_iter().enumerate();
    for (i, input) in numbered {
        if i + 1 == count {
            before_last = camera;
        }
        navigator.apply(input, &mut camera, &ball, 100.0);
    }
    assert_eq!(navigator.gesture(), Some(Gesture::Orbit), "still orbiting");
    assert_ne!(
        camera, before_last,
        "the drag lives: the last motion orbited"
    );
    assert_eq!(camera.target(), Camera::default().target(), "never panned");
}
