//! The brush and navigation together: strokes through a magnified view, the pivot a stroke sets
//! against one a navigation drag sets, and the camera's boundary as the brush raises the surface.

use super::*;
use crate::camera::Boundary;
use crate::navigate::{Button, Input};

/// The renderer's vector of the app's.
const fn wide(v: crate::camera::Vec3) -> render::camera::Vec3 {
    render::camera::Vec3::new(v.x, v.y, v.z)
}

/// Where `view` shows `point`.
fn shown(view: &View, point: crate::camera::Vec3) -> (f64, f64) {
    let (x, y, _) = view.pixel(wide(point)).expect("in front");
    (x, y)
}

/// A click of the right button at `at`: a navigation drag that starts and ends there, making the
/// surface under it the pivot.
fn click(owner: &mut Owner<Sculpting, NoUploads>, at: (f64, f64)) {
    let at = Pixel { x: at.0, y: at.1 };
    owner.input(Input::Press {
        button: Button::MouseRight,
        at,
    });
    owner.input(Input::Release {
        button: Button::MouseRight,
    });
}

/// Whether `boundary` holds `point` as it holds the model's points: a margin inside it.
fn holds(boundary: Boundary, point: crate::camera::Vec3) -> bool {
    let model = boundary.radius() / (1.0 + Boundary::MARGIN);
    (point - boundary.centre()).length() <= model * (1.0 + 1e-12)
}

#[test]
fn a_stroke_drawn_through_a_view_magnified_off_its_centre_lands_under_the_pen_and_is_recorded_so() {
    let dir = TestDir::new();
    let mut owner = owner(&dir);
    run(&mut owner, 10_000);
    let grabbed = (205.0, 75.0);
    click(&mut owner, grabbed);
    run(&mut owner, 10_000);
    for _ in 0..200 {
        if owner.camera().magnification() >= 4.0 {
            break;
        }
        owner.input(Input::Wheel { notches: 1.0 });
        run(&mut owner, 10_000);
    }
    let camera = *owner.camera();
    assert!(camera.magnification() >= 4.0, "magnified: {camera:?}");
    assert_ne!(camera.shift(), [0.0, 0.0], "off its centre");
    let view = camera.view(size()).expect("a view");
    let down = shown(&view, camera.pivot());
    assert!(
        (down.0 - grabbed.0).abs() < 1e-3 && (down.1 - grabbed.1).abs() < 1e-3,
        "the grabbed point stayed under the click: {down:?}"
    );
    queue(
        &mut owner,
        stroke((down, (down.0 + 30.0, down.1 + 20.0)), 8),
    );
    run(&mut owner, 10_000);
    assert!(owner.brush().last().dabs_applied > 0, "the stroke landed");
    // Its first dab, where it began: the pivot it set, under the pen's Down in the view it was
    // drawn through.
    let start = owner.camera().pivot();
    assert_ne!(start, camera.pivot(), "the stroke set the pivot");
    let (x, y) = shown(&view, start);
    assert!(
        (x - down.0).abs() < 0.5 && (y - down.1).abs() < 0.5,
        "the stroke began under the pen: ({x}, {y}) for {down:?}"
    );
    let records = saved_strokes(owner, &dir, "magnified.bcrf");
    let recorded = records.last().expect("a record").camera();
    assert_eq!(recorded.magnification, camera.magnification());
    assert_eq!(recorded.shift, camera.shift());
    let rebuilt = recorded_view(recorded).expect("a view");
    assert_eq!(
        rebuilt.principal(),
        view.principal(),
        "the same principal point"
    );
    assert!((rebuilt.pixels_per_unit() / view.pixels_per_unit() - 1.0).abs() < 1e-12);
    for point in [
        start,
        camera.pivot(),
        crate::camera::Vec3::new(0.1, 0.2, 0.3),
    ] {
        let (a, b) = (shown(&view, point), shown(&rebuilt, point));
        assert!(
            (a.0 - b.0).abs() < 1e-6 && (a.1 - b.1).abs() < 1e-6,
            "{point:?}: {a:?} {b:?}"
        );
    }
}

#[test]
fn a_grab_after_a_tap_wins_over_the_taps_start_noted_at_its_tail() {
    let dir = TestDir::new();
    let mut owner = owner(&dir);
    run(&mut owner, 10_000);
    // A tap: a Down and an Up, no drag, its only dab placed at the tail as the pen lifts.
    let tap = (150.0, 95.0);
    owner.pen(sample(Phase::Down, tap, 0.8));
    owner.pen(sample(Phase::Up, tap, 0.0));
    // Then, later, a navigation drag elsewhere on the sphere, heard in the same frame.
    let grabbed = (190.0, 120.0);
    let view = owner.camera().view(size()).expect("a view");
    click(&mut owner, grabbed);
    run(&mut owner, 10_000);
    assert!(
        owner.project().has_unsaved_edits(),
        "the tap placed its dab"
    );
    let pivot = owner.camera().pivot();
    let (x, y) = shown(&view, pivot);
    assert!(
        (x - grabbed.0).abs() < 0.01 && (y - grabbed.1).abs() < 0.01,
        "the later grab is the pivot: ({x}, {y})"
    );
    owner.close().expect("closed");
}

#[test]
fn the_boundary_grows_after_a_stroke_and_a_grab_in_the_frame_after_its_up_lies_inside_it() {
    let dir = TestDir::new();
    let mut owner = owner(&dir);
    for _ in 0..8 {
        owner.change_brush(BrushChange::Stronger);
        owner.change_brush(BrushChange::Larger);
    }
    run(&mut owner, 10_000);
    let before = owner.camera().boundary().expect("fitted at open");
    // A strong stroke across the middle, and in the frame after its Up, a grab on what it raised.
    queue(&mut owner, stroke(ACROSS, 12));
    run(&mut owner, 1);
    click(&mut owner, (f64::midpoint(ACROSS.0.0, ACROSS.1.0), 105.0));
    run(&mut owner, 1);
    let camera = *owner.camera();
    let boundary = camera.boundary().expect("a boundary");
    assert!(
        holds(boundary, camera.pivot()),
        "the grabbed pivot is inside"
    );
    assert!(
        (camera.eye() - boundary.centre()).length() >= boundary.radius() * (1.0 - 1e-12),
        "the eye is outside"
    );
    run(&mut owner, 10_000);
    // Once the stroke has landed, the boundary holds the model as its index now bounds it.
    let (centre, radius) = owner
        .picker()
        .expect("a picker")
        .model_sphere()
        .expect("bounds");
    let grown = owner.camera().boundary().expect("a boundary");
    assert!(
        (centre - grown.centre()).length() + radius * (1.0 + Boundary::MARGIN)
            <= grown.radius() * (1.0 + 1e-9),
        "the grown model: {before:?} → {grown:?}, the model {centre:?} {radius}"
    );
    assert!(holds(grown, owner.camera().pivot()));
    owner.close().expect("closed");
}

#[test]
fn a_down_the_brush_finds_off_the_surface_says_so_and_one_on_it_doesnt() {
    let dir = TestDir::new();
    let mut project = project(&dir);
    let mut picker = Picker::load(&mut project)
        .expect("the index loads")
        .expect("a display layer");
    let mut brush = Sculpting::new(&mut project);
    let camera = Camera::default();
    for (at, off) in [((3.0, 3.0), true), ((160.0, 100.0), false)] {
        let context = BrushContext {
            project: &mut project,
            index: Some(picker.index_mut()),
            camera: &camera,
            size: size(),
        };
        let route = brush.route(context, &sample(Phase::Down, at, 0.5));
        assert_eq!(route == Route::Navigate, off, "{at:?}");
        assert_eq!(brush.routed_off_surface(), off, "{at:?}");
    }
    let context = BrushContext {
        project: &mut project,
        index: None,
        camera: &camera,
        size: size(),
    };
    assert_eq!(
        brush.route(context, &sample(Phase::Down, (3.0, 3.0), 0.5)),
        Route::Navigate
    );
    assert!(!brush.routed_off_surface(), "no index: it didn't look");
    project.close(false).expect("closed");
}
