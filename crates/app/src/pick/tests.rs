//! Picking's tests.
//!
//! The ray through a pixel (the centre, the corners, a high-DPI window), picks on a small
//! generated sphere (on the surface, the near side, a miss), the frame loop picking only when the
//! pointer or the view changed, the overlay's line, a pick's time, and picking through a
//! magnified view; navigation about the pivot the frame loop picks under, in [`navigation`].

use std::time::{Duration, Instant};

use geom::{Level, cube_sphere};
use render::surface::{Presentation, Size};
use render::{DrawTimes, Outcome as Shown};
use store::{ProjectConfig, StoreConfig};

use super::*;
use crate::camera::Camera;
use crate::frame::schedule::Next;
use crate::frame::{NoBrush, NoUploads, Owner, Setup};
use crate::navigate::Input;
use crate::render_thread::Presented;
use crate::test_dir::TestDir;

mod navigation;

/// The level the test projects are picked at: a 24-face sphere's 96 tiles.
const LEVEL: u8 = 7;

/// The viewport the tests pick in.
fn size() -> Size {
    Size::new(320, 200).expect("non-zero")
}

/// The default camera's view of `size`.
fn view_of(size: Size) -> View {
    Camera::default().view(size).expect("a view")
}

/// The pixel `(x, y)`.
const fn at(x: f64, y: f64) -> Pixel {
    Pixel { x, y }
}

/// Whether `a` and `b` agree to within `tolerance` in every component.
fn close(a: Vec3, b: Vec3, tolerance: f64) -> bool {
    (a - b).length() < tolerance
}

/// A never-saved project over a 24-face cube sphere (radius about 1, unsculpted), its recovery
/// file in `dir`; with a display layer under [`LEVEL`] if `display`.
fn project(dir: &TestDir, display: bool) -> Project {
    let mut store = StoreConfig::new(512 << 20, 1);
    if display {
        store = store.with_display(Level::new(LEVEL).expect("a level"));
    }
    let config = ProjectConfig {
        store,
        recovery_dir: dir.path().join("recovery"),
    };
    Project::new(cube_sphere(2).expect("a cube sphere"), config).expect("a new project")
}

/// A picker over `project` at [`LEVEL`], its index built.
fn picker(project: &mut Project) -> Picker {
    let level = Level::new(LEVEL).expect("a level");
    let (index, _) = project.load_index(level, |_, _| {}).expect("the index");
    Picker::new(index)
}

#[test]
fn the_centre_pixel_looks_along_the_view() {
    let view = view_of(size());
    let ray = PickRay::through(&view, at(160.0, 100.0)).expect("a ray");
    assert!(
        close(ray.direction(), view.forward(), 1e-12),
        "{:?}",
        ray.direction()
    );
    assert!(
        close(
            ray.origin(),
            view.eye() + view.forward() * view.near(),
            1e-12
        ),
        "it starts on the near plane: {:?}",
        ray.origin()
    );
}

#[test]
fn the_corner_pixels_run_along_the_frustum_edges() {
    let view = view_of(size());
    let [left, right, bottom, top, near, _] = view.planes();
    let corners = [
        (at(0.0, 0.0), [left, top]),
        (at(320.0, 0.0), [right, top]),
        (at(0.0, 200.0), [left, bottom]),
        (at(320.0, 200.0), [right, bottom]),
    ];
    for (corner, planes) in corners {
        let ray = PickRay::through(&view, corner).expect("a ray");
        let far = ray.origin() + ray.direction() * 10.0;
        for [a, b, c, d] in planes {
            let normal = Vec3::new(a, b, c);
            assert!(
                normal.dot(ray.direction()).abs() < 1e-12,
                "{corner:?} runs along the plane {:?}",
                [a, b, c, d]
            );
            assert!(
                (normal.dot(far) + d).abs() < 1e-9,
                "{corner:?}: a far point lies on the plane"
            );
        }
        let [a, b, c, d] = near;
        assert!(
            (Vec3::new(a, b, c).dot(ray.origin()) + d).abs() < 1e-12,
            "{corner:?} starts on the near plane"
        );
    }
}

#[test]
fn points_along_a_ray_project_back_to_its_pixel() {
    let mut camera = Camera::default();
    camera.pan(37.0, -12.0, 200.0);
    camera.zoom(0.4);
    let view = camera.view(size()).expect("a view");
    for pixel in [
        at(3.25, 191.5),
        at(160.0, 100.0),
        at(301.75, 7.0),
        at(0.5, 0.5),
    ] {
        let ray = PickRay::through(&view, pixel).expect("a ray");
        for depth in [0.0, 0.5, 3.0] {
            let point = ray.origin() + ray.direction() * depth;
            let (x, y, _) = view.pixel(point).expect("in front of the eye");
            assert!(
                (x - pixel.x).abs() < 1e-9 && (y - pixel.y).abs() < 1e-9,
                "{pixel:?} at depth {depth} projects to ({x}, {y})"
            );
        }
    }
}

#[test]
fn a_hidpi_window_gives_the_same_ray_in_physical_pixels() {
    // A 320 × 200 window at scale 2 is 640 × 400 physical pixels, and winit reports the pointer
    // in physical pixels too: the same point of the window is the same ray.
    let logical = view_of(size());
    let physical = view_of(Size::new(640, 400).expect("non-zero"));
    for (x, y) in [(10.0, 20.0), (160.0, 100.0), (319.0, 1.5)] {
        let one = PickRay::through(&logical, at(x, y)).expect("a ray");
        let two = PickRay::through(&physical, at(2.0 * x, 2.0 * y)).expect("a ray");
        assert!(
            close(one.direction(), two.direction(), 1e-12),
            "({x}, {y}) at scale 1 and 2"
        );
        assert!(close(one.origin(), two.origin(), 1e-12));
    }
}

#[test]
fn a_point_that_isnt_finite_makes_no_ray() {
    let view = view_of(size());
    assert_eq!(PickRay::through(&view, at(f64::NAN, 1.0)), None);
    assert_eq!(PickRay::through(&view, at(1.0, f64::INFINITY)), None);
}

#[test]
fn a_pick_hits_the_near_side_of_the_surface_under_the_pixel() {
    let dir = TestDir::new();
    let mut project = project(&dir, false);
    let picker = picker(&mut project);
    let mut camera = Camera::default();
    camera.pan(-23.0, 17.0, 200.0);
    let view = camera.view(size()).expect("a view");
    let eye = view.eye();
    for pixel in [at(160.0, 100.0), at(130.5, 80.25), at(200.0, 140.0)] {
        let hit = picker
            .pick(project.store(), &view, pixel)
            .expect("a pick")
            .expect("the sphere is under the pixel");
        assert_eq!(hit.key.level(), picker.index.level(), "the finest level");
        // On the surface: within the radii of the tile's own points (a flat quad dips below
        // its corners by far less than the slack).
        let points = project.store().evaluate(hit.key).expect("the tile");
        let radii = points.iter().map(|p| f64::from(p.length()));
        let (low, high) = radii.fold((f64::MAX, 0.0_f64), |(low, high), r| {
            (low.min(r), high.max(r))
        });
        let radius = hit.position.length();
        assert!(
            radius > low - 1e-4 && radius < high + 1e-4,
            "{pixel:?}: radius {radius} outside the tile's {low}..{high}"
        );
        // Where the pixel shows it, to well within the 1 px rule.
        let (x, y, _) = view.pixel(hit.position).expect("in front");
        assert!(
            (x - pixel.x).abs() < 0.01 && (y - pixel.y).abs() < 0.01,
            "{pixel:?}: the hit projects to ({x}, {y})"
        );
        // On the near side: facing the eye, and nearer than the centre.
        assert!(
            hit.position.dot(eye - hit.position) > 0.0,
            "{pixel:?}: the hit faces the eye: {hit:?}"
        );
        assert!(
            hit.distance < eye.length(),
            "{pixel:?}: nearer than the centre: {hit:?}"
        );
        assert!(
            ((hit.position - eye).length() - hit.distance).abs() < 1e-5,
            "the distance is from the eye: {hit:?}"
        );
    }
}

#[test]
fn a_ray_past_the_silhouette_hits_nothing() {
    let dir = TestDir::new();
    let mut project = project(&dir, false);
    let picker = picker(&mut project);
    let view = view_of(size());
    for corner in [at(0.0, 0.0), at(319.0, 199.0), at(5.0, 100.0)] {
        assert_eq!(
            picker.pick(project.store(), &view, corner).expect("a pick"),
            None,
            "{corner:?} is off the sphere"
        );
    }
}

#[test]
fn a_pick_is_wanted_only_when_the_pointer_or_the_view_changed() {
    let mut since = Since::default();
    let view = view_of(size());
    let asked = Asked {
        view: Some(view),
        at: Some(at(10.0, 20.0)),
    };
    assert!(since.changed(&asked), "the first pick");
    assert!(!since.changed(&asked), "nothing changed");
    let moved = Asked {
        at: Some(at(11.0, 20.0)),
        ..asked
    };
    assert!(since.changed(&moved), "the pointer moved");
    let mut camera = Camera::default();
    camera.zoom(0.9);
    let zoomed = Asked {
        view: camera.view(size()),
        ..moved
    };
    assert!(since.changed(&zoomed), "the camera moved");
    let resized = Asked {
        view: camera.view(Size::new(321, 200).expect("non-zero")),
        ..zoomed
    };
    assert!(since.changed(&resized), "the window resized");
    let left = Asked {
        at: None,
        ..resized
    };
    assert!(since.changed(&left), "the pointer left");
    assert!(!since.changed(&left), "and stayed out");
}

/// Makes frames of `owner` until it wants none (up to 50), each shown at once; returns how many.
fn settle(owner: &mut Owner<NoBrush, NoUploads>) -> u32 {
    let mut made = 0;
    while made < 50 {
        let now = Instant::now();
        match owner.next(now) {
            Next::Frame { deadline } => {
                let mut number = 0;
                owner
                    .frame(now, deadline, |request| {
                        number = request.number;
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
                made += 1;
            }
            Next::Wait { until } => std::thread::sleep(until.saturating_duration_since(now)),
            Next::Render | Next::Gpu | Next::Idle { .. } => break,
        }
    }
    made
}

/// The frame loop's picks so far.
fn picks(owner: &Owner<NoBrush, NoUploads>) -> u64 {
    owner.picker().expect("a picker").picks()
}

/// The frame loop's last pick.
fn outcome(owner: &Owner<NoBrush, NoUploads>) -> Outcome {
    owner.picker().expect("a picker").report().outcome
}

#[test]
fn the_frame_loop_picks_only_when_the_pointer_or_the_view_changed() {
    let dir = TestDir::new();
    // No display layer: no store jobs, so the loop goes idle as soon as nothing wants a frame.
    let mut project = project(&dir, false);
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
    assert_eq!(settle(&mut owner), 1, "the first frame");
    assert_eq!(picks(&owner), 1, "picked once: no pointer yet");
    assert_eq!(outcome(&owner), Outcome::NoPointer);

    owner.input(Input::Move {
        at: at(160.0, 100.0),
    });
    assert_eq!(settle(&mut owner), 1, "hovering wants a frame for the pick");
    assert_eq!(picks(&owner), 2);
    assert!(
        matches!(outcome(&owner), Outcome::Hit(_)),
        "{:?}",
        outcome(&owner)
    );

    owner.input(Input::Move {
        at: at(160.0, 100.0),
    });
    assert_eq!(settle(&mut owner), 0, "the pointer didn't move: no frame");
    owner.expose();
    assert_eq!(settle(&mut owner), 1, "a frame for another reason");
    assert_eq!(picks(&owner), 2, "nothing changed: no pick");

    owner.input(Input::Wheel { notches: 1.0 });
    assert_eq!(settle(&mut owner), 1);
    assert_eq!(picks(&owner), 3, "the view changed: picked again");

    owner.pointer_left();
    assert_eq!(settle(&mut owner), 1, "leaving the window wants a frame");
    assert_eq!(picks(&owner), 4);
    assert_eq!(outcome(&owner), Outcome::NoPointer);
    owner.close().expect("closed");
}

#[test]
fn the_overlay_line_says_what_was_hit() {
    let key = TileKey::displacement(geom::FaceId::new(3), Level::new(9).expect("a level"), 5, 7)
        .expect("a tile");
    let hit = Report {
        number: 7,
        outcome: Outcome::Hit(Hit {
            position: Vec3::new(0.125, -0.5, 0.857),
            key,
            distance: 2.142_857,
        }),
        took: Duration::from_micros(210),
    };
    assert_eq!(
        summary(Some(&hit)),
        "#7  f3 L9 tile (5, 7)  at (0.125, -0.500, 0.857)  distance 2.143  (0.21 ms)"
    );
    let miss = Report {
        outcome: Outcome::Miss,
        ..hit
    };
    assert_eq!(summary(Some(&miss)), "#7  no hit  (0.21 ms)");
    assert_eq!(
        summary(Some(&Report::default())),
        "#0  no pointer over the window"
    );
    assert_eq!(summary(None), "off (no spatial index)");
}

#[test]
fn picks_are_timed_on_a_reopened_project() {
    // Saved and opened again, the index loads from stored bounds and no tile is evaluated, so the
    // first picks are cold, as on the benchmark project after it opens.
    let dir = TestDir::new();
    let path = dir.path().join("sphere.bcrf");
    {
        let mut made = project(&dir, true);
        drop(picker(&mut made));
        let saved = made.save_as(&path).expect("saved");
        assert!(saved.unprotected.is_empty(), "{saved:?}");
        made.close(false).expect("closed");
    }
    let config = ProjectConfig {
        store: StoreConfig::new(512 << 20, 1).with_display(Level::new(LEVEL).expect("a level")),
        recovery_dir: dir.path().join("recovery"),
    };
    let mut project = Project::open(&path, config).expect("opened");
    let (index, loaded) = project
        .load_index(Level::new(LEVEL).expect("a level"), |_, _| {})
        .expect("the index");
    assert_eq!(loaded.evaluated, 0, "loaded from stored bounds: {loaded:?}");
    let picker = Picker::new(index);
    let view = view_of(size());
    let pixels: Vec<Pixel> = (0..8)
        .map(|i| at(120.0 + 10.0 * f64::from(i), 70.0 + 8.0 * f64::from(i)))
        .collect();
    let time = |opened: &mut Project| -> Vec<String> {
        pixels
            .iter()
            .map(|&pixel| {
                let started = Instant::now();
                let hit = picker.pick(opened.store(), &view, pixel).expect("a pick");
                let took = started.elapsed();
                assert!(hit.is_some(), "{pixel:?} is over the sphere");
                format!("{:.3}", took.as_secs_f64() * 1e3)
            })
            .collect()
    };
    let cold = time(&mut project);
    let warm = time(&mut project);
    println!(
        "picks at level {LEVEL}, ms: first pass (cold) {}; second (warm) {}",
        cold.join(" "),
        warm.join(" ")
    );
    project.close(false).expect("closed");
}

/// A camera on the test sphere's boundary, its image magnified about `pivot`'s pixel at least
/// `factor` times; with the pivot on the line of sight (the origin), magnified about the centre.
fn magnified_about(pivot: camera::Vec3, factor: f64) -> Camera {
    let mut camera = Camera::default();
    let boundary = camera::Boundary::around(camera::Vec3::default(), 1.0);
    camera.grow_boundary(boundary.expect("a boundary"));
    camera.set_pivot(pivot);
    for _ in 0..200 {
        if camera.magnification() >= factor {
            return camera;
        }
        camera.zoom(1.0 / 1.1);
    }
    panic!("never magnified {factor} times");
}

/// `camera`'s view of [`size`] unmagnified: the same eye and axes, the plain field.
fn plain(camera: &Camera) -> View {
    let point = |v: camera::Vec3| Vec3::new(v.x, v.y, v.z);
    View::look_at(
        point(camera.eye()),
        point(camera.target()),
        point(camera.axis(camera::Vec3::new(0.0, 1.0, 0.0))),
        camera.fov_y(),
        size(),
        camera.distance() * render::camera::NEAR_FRACTION,
    )
    .expect("a view")
}

#[test]
fn picking_through_a_magnified_view_matches_the_narrowed_view_of_the_same_image() {
    let dir = TestDir::new();
    let mut project = project(&dir, false);
    let picker = picker(&mut project);
    let camera = magnified_about(camera::Vec3::default(), 6.0);
    let magnified = camera.view(size()).expect("a view");
    assert_eq!(magnified.principal(), [160.0, 100.0], "about the centre");
    // The view of the same image: the same eye, its field narrowed by the magnification.
    let plain = plain(&camera);
    let narrowed = View::look_at(
        plain.eye(),
        plain.eye() + plain.forward(),
        plain.up(),
        camera.magnified_fov_y(),
        size(),
        plain.near(),
    )
    .expect("a view");
    assert!(
        (magnified.pixels_per_unit() / narrowed.pixels_per_unit() - 1.0).abs() < 1e-12,
        "the same pixels"
    );
    for pixel in [
        at(160.0, 100.0),
        at(31.5, 12.25),
        at(300.0, 180.0),
        at(90.0, 150.0),
    ] {
        let (a, b) = (
            PickRay::through(&magnified, pixel).expect("a ray"),
            PickRay::through(&narrowed, pixel).expect("a ray"),
        );
        assert!(close(a.origin(), b.origin(), 1e-12), "{pixel:?}");
        assert!(close(a.direction(), b.direction(), 1e-12), "{pixel:?}");
        let hits = [
            picker
                .pick(project.store(), &magnified, pixel)
                .expect("a pick"),
            picker
                .pick(project.store(), &narrowed, pixel)
                .expect("a pick"),
        ];
        let [Some(seen), Some(also)] = hits else {
            panic!("{pixel:?}: the sphere fills a magnified view: {hits:?}");
        };
        assert!(close(seen.position, also.position, 1e-9), "{pixel:?}");
    }
}

#[test]
fn picking_through_a_view_magnified_off_centre_hits_what_its_pixels_show() {
    let dir = TestDir::new();
    let mut project = project(&dir, false);
    let picker = picker(&mut project);
    let pivot = picker
        .pick(project.store(), &view_of(size()), at(215.0, 60.0))
        .expect("a pick")
        .expect("the sphere is there")
        .position;
    let camera = magnified_about(camera::Vec3::new(pivot.x, pivot.y, pivot.z), 8.0);
    let magnified = camera.view(size()).expect("a view");
    let [px, py] = magnified.principal();
    assert!((px - 160.0).abs() > 100.0, "off the centre: ({px}, {py})");
    let unmagnified = plain(&camera);
    let factor = camera.magnification();
    for pixel in [at(215.0, 60.0), at(180.0, 40.0), at(250.0, 90.0)] {
        let hit = picker
            .pick(project.store(), &magnified, pixel)
            .expect("a pick")
            .expect("the sphere is under it");
        let (x, y, _) = magnified.pixel(hit.position).expect("in front");
        assert!(
            (x - pixel.x).abs() < 0.01 && (y - pixel.y).abs() < 0.01,
            "{pixel:?}: the hit shows at ({x}, {y})"
        );
        // The unmagnified view shows it where the magnification carries back from.
        let back = at(
            160.0 + (pixel.x - px) / factor,
            100.0 + (pixel.y - py) / factor,
        );
        let plain_hit = picker
            .pick(project.store(), &unmagnified, back)
            .expect("a pick")
            .expect("the sphere is under it");
        assert!(
            close(hit.position, plain_hit.position, 1e-5),
            "{pixel:?}: {hit:?} {plain_hit:?}"
        );
    }
}
