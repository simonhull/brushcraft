//! Navigation about the pivot in the frame loop, on the picked sphere.
//!
//! A navigation drag started on the model makes the surface under it the pivot (one started off
//! it keeps it), the grabbed point stays on its pixel through an orbit and under the pointer
//! through a pan, F frames the model about its centre, and the wheel closes on the surface up to
//! the boundary, magnifies past it and backs out the same way, the eye never inside the model.

use input::{Buttons, Phase, Tool, Unit};

use super::*;
use crate::camera::{self, Boundary};
use crate::navigate::{Button, Modifiers};
use crate::pen::PenSample;

/// The frame loop over a fresh test sphere (no display layer: it goes idle at once), picking.
fn owner(dir: &TestDir) -> Owner<NoBrush, NoUploads> {
    let mut project = project(dir, false);
    let picker = picker(&mut project);
    let setup = Setup {
        size: Some(size()),
        interval: Duration::from_millis(1),
        presentation: Presentation::Offscreen,
        adapter: "test".to_owned(),
        frames: None,
        script: None,
        record: None,
    };
    let mut owner =
        Owner::new(project, NoBrush, NoUploads, setup, Instant::now()).with_picker(picker);
    settle(&mut owner);
    owner
}

/// A second sphere like the frame loop's, and its picker: where the loop's surface is, asked
/// without borrowing the loop.
struct Probe {
    /// Its directory.
    _dir: TestDir,
    /// The project.
    project: Project,
    /// Its picker.
    picker: Picker,
}

impl Probe {
    /// A fresh probe.
    fn new() -> Self {
        let dir = TestDir::new();
        let mut project = project(&dir, false);
        let picker = picker(&mut project);
        Self {
            _dir: dir,
            project,
            picker,
        }
    }

    /// The surface `camera` shows at `pixel`, which must be on the sphere.
    fn under(&mut self, camera: &Camera, pixel: Pixel) -> Vec3 {
        let view = camera.view(size()).expect("a view");
        self.picker
            .pick(self.project.store(), &view, pixel)
            .expect("a pick")
            .expect("the sphere is under it")
            .position
    }

    /// How far ahead of `camera`'s eye, along its line to the pivot, the surface first lies.
    fn ahead(&mut self, camera: &Camera) -> f64 {
        let pivot = shown(camera, camera.pivot());
        let view = camera.view(size()).expect("a view");
        self.picker
            .pick(self.project.store(), &view, pivot)
            .expect("a pick")
            .expect("the surface ahead")
            .distance
    }
}

/// The app's `f64` vector as the renderer's.
const fn render_point(v: camera::Vec3) -> Vec3 {
    Vec3::new(v.x, v.y, v.z)
}

/// Where `camera` shows `point`, in [`size`].
fn shown(camera: &Camera, point: camera::Vec3) -> Pixel {
    let (x, y, _) = camera
        .view(size())
        .expect("a view")
        .pixel(render_point(point))
        .expect("in front");
    Pixel { x, y }
}

/// How close a grabbed point shows to the pixel it should: a pick's hit is rounded to `f32`, as
/// the mesh is, which moves it a few millionths of a pixel off the ray it was picked along.
const GRABBED: f64 = 1e-4;

/// Whether two pixels agree to `tolerance`.
fn same(a: Pixel, b: Pixel, tolerance: f64) -> bool {
    (a.x - b.x).abs() < tolerance && (a.y - b.y).abs() < tolerance
}

/// A drag of `button` from `from` to `to` in `steps` moves, then its release, each step a frame.
fn drag(owner: &mut Owner<NoBrush, NoUploads>, button: Button, from: Pixel, to: Pixel) {
    owner.input(Input::Press { button, at: from });
    settle(owner);
    let steps = 8;
    let path = (1..=steps).map(|step| {
        let f = f64::from(step) / f64::from(steps);
        at(from.x + (to.x - from.x) * f, from.y + (to.y - from.y) * f)
    });
    for point in path {
        owner.input(Input::Move { at: point });
        settle(owner);
    }
    owner.input(Input::Release { button });
    settle(owner);
}

#[test]
fn a_drag_started_on_the_model_makes_the_surface_there_the_pivot_and_one_off_it_keeps_it() {
    let dir = TestDir::new();
    let mut owner = owner(&dir);
    let mut probe = Probe::new();
    let grabbed = at(205.0, 70.0);
    let surface = probe.under(owner.camera(), grabbed);
    owner.input(Input::Press {
        button: Button::MouseLeft,
        at: grabbed,
    });
    let before = owner.camera().view(size());
    settle(&mut owner);
    assert_eq!(
        owner.camera().view(size()),
        before,
        "the press moves no pixel"
    );
    let pivot = render_point(owner.camera().pivot());
    assert!(close(pivot, surface, 1e-9), "{pivot:?} {surface:?}");
    owner.input(Input::Release {
        button: Button::MouseLeft,
    });
    settle(&mut owner);
    // Off the sphere: the drag orbits, and the pivot stays where it was.
    drag(&mut owner, Button::MouseLeft, at(4.0, 4.0), at(40.0, 30.0));
    assert_ne!(owner.camera().view(size()), before, "it orbited");
    assert_eq!(render_point(owner.camera().pivot()), pivot, "kept");
    owner.close().expect("closed");
}

#[test]
fn a_pen_drag_with_a_barrel_button_on_the_model_makes_the_surface_there_the_pivot() {
    let dir = TestDir::new();
    let mut owner = owner(&dir);
    let mut probe = Probe::new();
    let grabbed = at(120.0, 125.0);
    let surface = probe.under(owner.camera(), grabbed);
    let now = Instant::now();
    let pen = |phase, at| PenSample {
        phase,
        tool: Tool::Pen,
        at,
        pressure: Unit::ONE,
        distance: None,
        tilt: None,
        roll: None,
        buttons: Buttons::SECONDARY,
        timestamp: now,
        received: now,
    };
    owner.pen(pen(Phase::Down, grabbed));
    owner.pen(pen(Phase::Drag, at(150.0, 110.0)));
    settle(&mut owner);
    let pivot = render_point(owner.camera().pivot());
    assert!(close(pivot, surface, 1e-9), "{pivot:?} {surface:?}");
    assert!(
        same(
            shown(owner.camera(), owner.camera().pivot()),
            at(150.0, 110.0),
            GRABBED
        ),
        "the barrel pans: the grabbed point follows the tip"
    );
    owner.pen(pen(Phase::Up, at(150.0, 110.0)));
    settle(&mut owner);
    owner.close().expect("closed");
}

#[test]
fn orbiting_keeps_the_grabbed_point_on_its_pixel() {
    let dir = TestDir::new();
    let mut owner = owner(&dir);
    let grabbed = at(200.0, 120.0);
    drag(&mut owner, Button::MouseLeft, grabbed, at(90.0, 60.0));
    let camera = owner.camera();
    assert!(camera.eye().z < 2.9, "it turned: {:?}", camera.eye());
    let now = shown(camera, camera.pivot());
    assert!(same(now, grabbed, GRABBED), "{now:?}");
    owner.close().expect("closed");
}

#[test]
fn panning_keeps_the_grabbed_point_under_the_pointer() {
    let dir = TestDir::new();
    let mut owner = owner(&dir);
    let grabbed = at(150.0, 90.0);
    let to = at(230.0, 150.0);
    for button in [Button::MouseRight, Button::MouseMiddle] {
        drag(&mut owner, button, grabbed, to);
        let camera = owner.camera();
        let now = shown(camera, camera.pivot());
        assert!(same(now, to, GRABBED), "{button:?}: {now:?}");
        drag(&mut owner, button, to, grabbed);
    }
    // Shift and the left button pan too, a grab on the surface the drag starts on.
    owner.input(Input::Modifiers(Modifiers {
        shift: true,
        ctrl: false,
    }));
    drag(
        &mut owner,
        Button::MouseLeft,
        at(170.0, 110.0),
        at(120.0, 60.0),
    );
    let camera = owner.camera();
    assert!(same(
        shown(camera, camera.pivot()),
        at(120.0, 60.0),
        GRABBED
    ));
    owner.close().expect("closed");
}

#[test]
fn framing_fits_the_model_about_its_centre() {
    let dir = TestDir::new();
    let mut owner = owner(&dir);
    drag(
        &mut owner,
        Button::MouseRight,
        at(150.0, 90.0),
        at(60.0, 40.0),
    );
    for _ in 0..60 {
        owner.input(Input::Wheel { notches: 1.0 });
    }
    settle(&mut owner);
    assert!(owner.camera().magnification() > 1.0, "magnified");
    owner.frame_model();
    settle(&mut owner);
    let camera = *owner.camera();
    let (centre, radius) = owner
        .picker()
        .expect("a picker")
        .model_sphere()
        .expect("bounds");
    assert_eq!(camera.pivot(), centre, "the pivot is the model's centre");
    assert!((camera.magnification() - 1.0).abs() < 1e-12, "unmagnified");
    let view = camera.view(size()).expect("a view");
    assert_eq!(view.principal(), [160.0, 100.0]);
    // The model's sphere just fits the view's height (the narrower field of a 320 × 200 view).
    let distance = (camera.eye() - centre).length();
    assert!(
        (radius / distance - (camera.fov_y() / 2.0).sin()).abs() < 1e-9,
        "{radius} at {distance}"
    );
    let boundary = camera.boundary().expect("a boundary");
    assert!((boundary.radius() - radius * (1.0 + Boundary::MARGIN)).abs() < 1e-9);
    owner.close().expect("closed");
}

#[test]
fn the_wheel_closes_on_the_surface_magnifies_at_the_boundary_and_backs_out_the_same_way() {
    let dir = TestDir::new();
    let mut owner = owner(&dir);
    let mut probe = Probe::new();
    // Grab a point of the surface off the view's centre, with a click.
    let grabbed = at(190.0, 80.0);
    owner.input(Input::Press {
        button: Button::MouseLeft,
        at: grabbed,
    });
    owner.input(Input::Release {
        button: Button::MouseLeft,
    });
    settle(&mut owner);
    let start = *owner.camera();
    let boundary = start.boundary().expect("a boundary");
    let outside = |camera: &Camera| (camera.eye() - boundary.centre()).length() / boundary.radius();
    let pinned = shown(&start, start.pivot());
    assert!(same(pinned, grabbed, GRABBED), "grabbed: {pinned:?}");
    let mut magnified = 0;
    for notch in 0..60 {
        owner.input(Input::Wheel { notches: 1.0 });
        settle(&mut owner);
        let camera = *owner.camera();
        let now = shown(&camera, camera.pivot());
        assert!(
            same(now, pinned, 1e-6),
            "notch {notch}: the pivot moved to {now:?}"
        );
        assert!(
            outside(&camera) >= 1.0 - 1e-12,
            "notch {notch}: inside the boundary"
        );
        let ahead = probe.ahead(&camera);
        let to_pivot = (camera.eye() - camera.pivot()).length();
        assert!(
            (ahead - to_pivot).abs() < 1e-4 && ahead > 0.0,
            "notch {notch}: the pivot is the first surface ahead ({ahead} vs {to_pivot})"
        );
        let view = camera.view(size()).expect("a view");
        assert!(
            view.near() < ahead,
            "notch {notch}: the surface is beyond the near plane"
        );
        if camera.magnification() > 1.0 {
            magnified += 1;
            assert!(
                (outside(&camera) - 1.0).abs() < 1e-9,
                "notch {notch}: on the boundary"
            );
        }
    }
    assert!(
        magnified > 20,
        "it reached the boundary: {magnified} notches magnified"
    );
    for _ in 0..60 {
        owner.input(Input::Wheel { notches: -1.0 });
        settle(&mut owner);
    }
    let back = *owner.camera();
    assert!(
        close(render_point(back.eye()), render_point(start.eye()), 1e-9),
        "as many notches out: back where it started, {:?} → {:?}",
        start.eye(),
        back.eye()
    );
    assert!((back.magnification() - 1.0).abs() < 1e-12);
    owner.close().expect("closed");
}

/// Makes frames of `owner` until it wants none, as [`settle`] does; returns the overlay lines of
/// the last frame made, if any.
fn settle_shown(owner: &mut Owner<NoBrush, NoUploads>) -> Option<Vec<String>> {
    let mut last = None;
    for _ in 0..50 {
        let now = Instant::now();
        match owner.next(now) {
            Next::Frame { deadline } => {
                let mut number = 0;
                owner
                    .frame(now, deadline, |request| {
                        number = request.number;
                        last = Some(request.overlay);
                        Ok(())
                    })
                    .expect("a frame");
                owner.shown(Presented {
                    number,
                    outcome: Shown::Presented,
                    times: DrawTimes::default(),
                    scene: false,
                    shown: false,
                    aa: render::aa::AaReport::default(),
                    gpu: Vec::new(),
                    dropped: render::DroppedReadbacks::default(),
                    gpu_finished: number,
                    submitted: Instant::now(),
                });
            }
            Next::Wait { until } => std::thread::sleep(until.saturating_duration_since(now)),
            Next::Render | Next::Gpu | Next::Idle { .. } => break,
        }
    }
    last
}

#[test]
fn a_press_that_moves_the_pivot_shows_it_in_the_overlay_without_moving_the_view() {
    let dir = TestDir::new();
    let mut owner = owner(&dir);
    owner.input(Input::Move {
        at: at(205.0, 70.0),
    });
    settle(&mut owner);
    let view = owner.camera().view(size());
    owner.input(Input::Press {
        button: Button::MouseRight,
        at: at(205.0, 70.0),
    });
    let lines = settle_shown(&mut owner).expect("a frame for the overlay");
    let pivot = owner.camera().pivot();
    assert_ne!(
        pivot,
        camera::Vec3::default(),
        "the press grabbed the surface"
    );
    let said = format!("pivot ({:.2}, {:.2}, {:.2})", pivot.x, pivot.y, pivot.z);
    assert!(
        lines.iter().any(|line| line.contains(&said)),
        "{said} in {lines:?}"
    );
    assert_eq!(owner.camera().view(size()), view, "no pixel moved");
    owner.close().expect("closed");
}
