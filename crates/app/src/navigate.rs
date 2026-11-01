//! Orbit, pan and zoom from the mouse and the pen: which button and modifier make which
//! gesture, and what each pointer motion does to the camera.
//!
//! The bindings, the same for the mouse's left button and the pen's tip:
//!
//! | Input | Gesture |
//! |---|---|
//! | left button or pen tip, dragged | orbit (the arcball) |
//! | the same with Shift, or the pen's second barrel button held | pan |
//! | the same with Ctrl, or the pen's first barrel button held | zoom (drag up to come closer) |
//! | middle or right button, dragged | pan |
//! | wheel | zoom, a notch at a time |
//!
//! A gesture is chosen when its button goes down and lasts until that button comes up, or the
//! window loses focus (when the release may never come); modifiers changing meanwhile don't
//! switch it. Only motion in a drag and the wheel ask for a frame. Input arrives as [`Input`], the
//! window's events in our terms (the pen's through its stream, [`crate::pen`]), so the rules here
//! are tested without a window. The pen's tip navigates only off the model: a stroke started on it
//! is the brush's ([`crate::sculpting`]), unless a barrel button is held.

use crate::camera::{Arcball, Camera, Pixel};

/// A button that can start a gesture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Button {
    /// The mouse's left button.
    MouseLeft,
    /// The mouse's middle button (or wheel press).
    MouseMiddle,
    /// The mouse's right button.
    MouseRight,
    /// The pen tip touching the tablet. On winit's fallback the pen's first barrel button
    /// arrives this way too (winit maps evdev's `BTN_STYLUS` to the tip, as SDL does), so there
    /// pressing it while hovering orbits as well.
    PenTip,
    /// The pen's first barrel button (`BTN_STYLUS`): held, the tip zooms. Only our pen backend
    /// tells it from the tip.
    PenFirstBarrel,
    /// The pen's second barrel button (`BTN_STYLUS2`): held, the tip pans.
    PenBarrel,
}

/// The modifier keys navigation reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Modifiers {
    /// Shift is held: the left button and pen tip pan.
    pub(crate) shift: bool,
    /// Ctrl is held: the left button and pen tip zoom.
    pub(crate) ctrl: bool,
}

/// A pointer event, in navigation's terms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Input {
    /// `button` went down with the pointer at `at`.
    Press {
        /// The button.
        button: Button,
        /// Where the pointer was.
        at: Pixel,
    },
    /// `button` came up.
    Release {
        /// The button.
        button: Button,
    },
    /// The pointer moved to `at`.
    Move {
        /// Where it is now.
        at: Pixel,
    },
    /// The wheel turned by `notches` (positive away from the user, which zooms in).
    Wheel {
        /// Notches turned: whole for a mouse wheel, fractional for a touchpad.
        notches: f64,
    },
    /// The modifier keys changed.
    Modifiers(Modifiers),
    /// The window lost the keyboard and pointer focus: a drag in progress ends, and the pen's
    /// barrel button counts as released, since their releases may never arrive.
    Cancel,
}

/// What a drag does to the camera.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Gesture {
    /// Turns the view with the arcball.
    Orbit,
    /// Moves the view in its plane.
    Pan,
    /// Moves the eye towards the target or away.
    Zoom,
}

/// A drag in progress.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Drag {
    /// The button that started it; its release ends it.
    button: Button,
    /// What it does.
    gesture: Gesture,
    /// Where the pointer was at the last motion applied.
    last: Pixel,
}

/// How much one wheel notch scales the distance: 10 % closer per notch in.
const NOTCH_ZOOM: f64 = 1.1;
/// How much dragging one pixel scales the distance when zooming by drag: 100 pixels up is
/// about 2.7× closer (`e`).
const DRAG_ZOOM_PER_PIXEL: f64 = 0.01;

/// Turns pointer input into camera motion.
#[derive(Debug, Clone, Default)]
pub(crate) struct Navigator {
    /// The drag in progress, if any.
    drag: Option<Drag>,
    /// The modifier keys held.
    modifiers: Modifiers,
    /// Whether the pen's second barrel button is held: the tip pans.
    barrel: bool,
    /// Whether the pen's first barrel button is held: the tip zooms.
    first_barrel: bool,
}

impl Navigator {
    /// The gesture a press of `button` starts, given the modifiers and the barrel buttons;
    /// `None` for a button that starts none. Zoom wins over pan when both are asked for.
    pub(crate) const fn gesture_for(&self, button: Button) -> Option<Gesture> {
        let pen = matches!(button, Button::PenTip);
        match button {
            Button::MouseLeft | Button::PenTip => {
                Some(if self.modifiers.ctrl || (pen && self.first_barrel) {
                    Gesture::Zoom
                } else if self.modifiers.shift || (pen && self.barrel) {
                    Gesture::Pan
                } else {
                    Gesture::Orbit
                })
            }
            Button::MouseMiddle | Button::MouseRight => Some(Gesture::Pan),
            Button::PenBarrel | Button::PenFirstBarrel => None,
        }
    }

    /// The gesture in progress, if a drag is.
    #[cfg(test)]
    pub(crate) fn gesture(&self) -> Option<Gesture> {
        self.drag.map(|drag| drag.gesture)
    }

    /// Whether `input` may move the camera once applied after the inputs `queued` before it:
    /// motion during a drag (one in progress, or started by a queued press and not ended by a
    /// queued release), or a wheel turn. What asks for a frame; hovering, a press alone (the
    /// barrel's included) and motion after a release ask for none.
    pub(crate) fn may_move(&self, queued: &[Input], input: &Input) -> bool {
        match input {
            Input::Move { .. } => {
                let mut ahead = self.clone();
                for &earlier in queued {
                    ahead.track(earlier);
                }
                ahead.drag.is_some()
            }
            Input::Wheel { notches } => *notches != 0.0 && notches.is_finite(),
            Input::Press { .. } | Input::Release { .. } | Input::Modifiers(_) | Input::Cancel => {
                false
            }
        }
    }

    /// Applies `input` to the navigator's own state (the drag, the barrel button, the
    /// modifiers), leaving the camera alone; motion only moves the drag's last point.
    fn track(&mut self, input: Input) {
        match input {
            Input::Press { button, at } => {
                self.hold(button, true);
                if self.drag.is_none()
                    && let Some(gesture) = self.gesture_for(button)
                {
                    self.drag = Some(Drag {
                        button,
                        gesture,
                        last: at,
                    });
                }
            }
            Input::Release { button } => {
                self.hold(button, false);
                if self.drag.is_some_and(|drag| drag.button == button) {
                    self.drag = None;
                }
            }
            Input::Move { at } => {
                if let Some(drag) = &mut self.drag {
                    drag.last = at;
                }
            }
            Input::Modifiers(modifiers) => self.modifiers = modifiers,
            Input::Cancel => {
                self.drag = None;
                self.barrel = false;
                self.first_barrel = false;
            }
            Input::Wheel { .. } => {}
        }
    }

    /// Notes `button` held or released, if it is a pen barrel button (a modifier).
    const fn hold(&mut self, button: Button, held: bool) {
        match button {
            Button::PenBarrel => self.barrel = held,
            Button::PenFirstBarrel => self.first_barrel = held,
            Button::MouseLeft | Button::MouseMiddle | Button::MouseRight | Button::PenTip => {}
        }
    }

    /// Applies `input` to `camera`, whose viewport the `arcball` covers and is `height` pixels
    /// tall; returns whether the camera moved.
    pub(crate) fn apply(
        &mut self,
        input: Input,
        camera: &mut Camera,
        arcball: &Arcball,
        height: f64,
    ) -> bool {
        match input {
            Input::Move { at } => {
                let Some(drag) = &mut self.drag else {
                    return false;
                };
                let from = std::mem::replace(&mut drag.last, at);
                if from == at {
                    return false;
                }
                match drag.gesture {
                    Gesture::Orbit => camera.orbit(arcball.drag(from, at)),
                    Gesture::Pan => camera.pan(at.x - from.x, at.y - from.y, height),
                    Gesture::Zoom => camera.zoom((DRAG_ZOOM_PER_PIXEL * (at.y - from.y)).exp()),
                }
                true
            }
            Input::Wheel { notches } => {
                if notches == 0.0 || !notches.is_finite() {
                    return false;
                }
                camera.zoom(NOTCH_ZOOM.powf(-notches));
                true
            }
            Input::Press { .. } | Input::Release { .. } | Input::Modifiers(_) | Input::Cancel => {
                self.track(input);
                false
            }
        }
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use render::surface::Size;

    use super::*;

    /// A 200 × 100 viewport's arcball.
    fn arcball() -> Arcball {
        Arcball::new(Size::new(200, 100).expect("non-zero"))
    }

    /// The pixel `(x, y)`.
    const fn at(x: f64, y: f64) -> Pixel {
        Pixel { x, y }
    }

    /// Applies `inputs` in turn to a fresh navigator and the default camera; returns the
    /// camera and the navigator.
    fn run(inputs: &[Input]) -> (Camera, Navigator) {
        let mut camera = Camera::default();
        let mut navigator = Navigator::default();
        let ball = arcball();
        for &input in inputs {
            navigator.apply(input, &mut camera, &ball, 100.0);
        }
        (camera, navigator)
    }

    #[test]
    fn buttons_and_modifiers_choose_the_gesture() {
        let mut navigator = Navigator::default();
        assert_eq!(
            navigator.gesture_for(Button::MouseLeft),
            Some(Gesture::Orbit)
        );
        assert_eq!(navigator.gesture_for(Button::PenTip), Some(Gesture::Orbit));
        assert_eq!(
            navigator.gesture_for(Button::MouseMiddle),
            Some(Gesture::Pan)
        );
        assert_eq!(
            navigator.gesture_for(Button::MouseRight),
            Some(Gesture::Pan)
        );
        assert_eq!(navigator.gesture_for(Button::PenBarrel), None);
        navigator.modifiers.shift = true;
        assert_eq!(navigator.gesture_for(Button::MouseLeft), Some(Gesture::Pan));
        navigator.modifiers.ctrl = true;
        assert_eq!(
            navigator.gesture_for(Button::PenTip),
            Some(Gesture::Zoom),
            "ctrl wins"
        );
        navigator.modifiers = Modifiers::default();
        navigator.barrel = true;
        assert_eq!(navigator.gesture_for(Button::PenTip), Some(Gesture::Pan));
        assert_eq!(
            navigator.gesture_for(Button::MouseLeft),
            Some(Gesture::Orbit),
            "the barrel is the pen's"
        );
    }

    #[test]
    fn a_drag_orbits_and_the_release_ends_it() {
        let press = Input::Press {
            button: Button::MouseLeft,
            at: at(100.0, 50.0),
        };
        let (camera, navigator) = run(&[
            press,
            Input::Move {
                at: at(150.0, 50.0),
            },
        ]);
        assert_eq!(navigator.gesture(), Some(Gesture::Orbit));
        let eye = camera.eye();
        assert!(
            (eye.x + 3.0).abs() < 1e-9,
            "a quarter turn puts the eye on −x: {eye:?}"
        );
        let (after, released) = run(&[
            press,
            Input::Move {
                at: at(150.0, 50.0),
            },
            Input::Release {
                button: Button::MouseLeft,
            },
            Input::Move {
                at: at(100.0, 50.0),
            },
        ]);
        assert_eq!(released.gesture(), None);
        assert_eq!(after, camera, "moving after the release does nothing");
    }

    #[test]
    fn hovering_moves_nothing() {
        let (camera, _) = run(&[
            Input::Move { at: at(10.0, 10.0) },
            Input::Move { at: at(90.0, 70.0) },
        ]);
        assert_eq!(camera, Camera::default());
    }

    #[test]
    fn the_pen_pans_with_its_barrel_held() {
        let (camera, navigator) = run(&[
            Input::Press {
                button: Button::PenBarrel,
                at: at(100.0, 50.0),
            },
            Input::Press {
                button: Button::PenTip,
                at: at(100.0, 50.0),
            },
            Input::Move {
                at: at(120.0, 50.0),
            },
        ]);
        assert_eq!(navigator.gesture(), Some(Gesture::Pan));
        assert!(camera.target().x < 0.0, "panned: {:?}", camera.target());
        assert!(
            (camera.distance() - Camera::default().distance()).abs() < 1e-12,
            "not zoomed"
        );
    }

    #[test]
    fn a_gesture_keeps_its_kind_while_modifiers_change() {
        let (camera, navigator) = run(&[
            Input::Press {
                button: Button::MouseLeft,
                at: at(100.0, 50.0),
            },
            Input::Modifiers(Modifiers {
                shift: true,
                ctrl: false,
            }),
            Input::Move {
                at: at(100.0, 40.0),
            },
        ]);
        assert_eq!(navigator.gesture(), Some(Gesture::Orbit));
        assert_eq!(camera.target(), Camera::default().target(), "not panned");
    }

    #[test]
    fn only_the_starting_button_ends_a_drag() {
        let (_, navigator) = run(&[
            Input::Press {
                button: Button::MouseRight,
                at: at(0.0, 0.0),
            },
            Input::Press {
                button: Button::MouseLeft,
                at: at(0.0, 0.0),
            },
            Input::Release {
                button: Button::MouseLeft,
            },
        ]);
        assert_eq!(
            navigator.gesture(),
            Some(Gesture::Pan),
            "the right button's pan goes on"
        );
    }

    #[test]
    fn only_motion_in_a_drag_and_the_wheel_may_move_the_camera() {
        let motion = Input::Move { at: at(1.0, 1.0) };
        let press = Input::Press {
            button: Button::MouseLeft,
            at: at(0.0, 0.0),
        };
        let release = Input::Release {
            button: Button::MouseLeft,
        };
        let idle = Navigator::default();
        assert!(!idle.may_move(&[], &motion), "hovering");
        assert!(
            idle.may_move(&[press], &motion),
            "a press is queued before it"
        );
        assert!(
            !idle.may_move(&[press, release], &motion),
            "and released again: hovering"
        );
        let barrel = Input::Press {
            button: Button::PenBarrel,
            at: at(0.0, 0.0),
        };
        assert!(
            !idle.may_move(&[barrel], &motion),
            "the barrel alone drags nothing"
        );
        assert!(idle.may_move(&[], &Input::Wheel { notches: -1.0 }));
        assert!(!idle.may_move(&[], &Input::Wheel { notches: 0.0 }));
        assert!(!idle.may_move(&[], &Input::Wheel { notches: f64::NAN }));
        assert!(!idle.may_move(&[], &press), "a press alone moves nothing");
        let (_, dragging) = run(&[press]);
        assert!(dragging.may_move(&[], &motion), "motion in a drag");
        assert!(
            !dragging.may_move(&[release], &motion),
            "after its queued release"
        );
    }

    #[test]
    fn losing_focus_ends_a_drag_whose_release_never_came() {
        let (camera, navigator) = run(&[
            Input::Press {
                button: Button::PenBarrel,
                at: at(100.0, 50.0),
            },
            Input::Press {
                button: Button::MouseLeft,
                at: at(100.0, 50.0),
            },
            Input::Cancel,
            Input::Move {
                at: at(150.0, 50.0),
            },
        ]);
        assert_eq!(navigator.gesture(), None, "the drag ended");
        assert!(!navigator.barrel, "the barrel counts as released");
        assert_eq!(
            camera,
            Camera::default(),
            "the motion after it moves nothing"
        );
    }

    #[test]
    fn dragging_up_with_ctrl_and_the_wheel_zoom_in() {
        let ctrl = Input::Modifiers(Modifiers {
            shift: false,
            ctrl: true,
        });
        let (camera, _) = run(&[
            ctrl,
            Input::Press {
                button: Button::MouseLeft,
                at: at(100.0, 50.0),
            },
            Input::Move { at: at(100.0, 0.0) },
        ]);
        let expected = 3.0 * (-0.5_f64).exp();
        assert!(
            (camera.distance() - expected).abs() < 1e-9,
            "{}",
            camera.distance()
        );
        let (wheeled, _) = run(&[Input::Wheel { notches: 2.0 }]);
        assert!(
            (wheeled.distance() - 3.0 / 1.21).abs() < 1e-9,
            "two notches in: {}",
            wheeled.distance()
        );
        let (still, _) = run(&[Input::Wheel { notches: f64::NAN }]);
        assert_eq!(still, Camera::default());
    }
}
