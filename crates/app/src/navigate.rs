//! Orbit, pan and zoom from the mouse and the pen: which button and modifier make which
//! gesture, and what each pointer motion does to the camera.
//!
//! The bindings, the same for the mouse's left button and the pen's tip:
//!
//! | Input | Gesture |
//! |---|---|
//! | left button or pen tip, dragged | orbit (the arcball) |
//! | the same with Shift, or the pen's barrel button held | pan |
//! | the same with Ctrl | zoom (drag up to come closer) |
//! | middle or right button, dragged | pan |
//! | wheel | zoom, a notch at a time |
//!
//! A gesture is chosen when its button goes down and lasts until that button comes up;
//! modifiers changing meanwhile don't switch it. Input arrives as [`Input`], the window's
//! events in our terms, so the rules here are tested without a window. The brush takes the pen
//! tip over the model in M1a4; until then the tip navigates everywhere.

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
    /// The pen tip touching the tablet. winit reports the pen's first barrel button this way
    /// too (it maps evdev's `BTN_STYLUS` to the tip, as SDL does), so pressing it while
    /// hovering orbits as well.
    PenTip,
    /// The pen's second barrel button: held, the tip pans.
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
    /// Whether the pen's barrel button is held.
    barrel: bool,
}

impl Navigator {
    /// The gesture a press of `button` starts, given the modifiers and the barrel button; `None`
    /// for a button that starts none.
    pub(crate) const fn gesture_for(&self, button: Button) -> Option<Gesture> {
        match button {
            Button::MouseLeft | Button::PenTip => Some(if self.modifiers.ctrl {
                Gesture::Zoom
            } else if self.modifiers.shift || (self.barrel && matches!(button, Button::PenTip)) {
                Gesture::Pan
            } else {
                Gesture::Orbit
            }),
            Button::MouseMiddle | Button::MouseRight => Some(Gesture::Pan),
            Button::PenBarrel => None,
        }
    }

    /// The gesture in progress, if a drag is.
    #[cfg(test)]
    pub(crate) fn gesture(&self) -> Option<Gesture> {
        self.drag.map(|drag| drag.gesture)
    }

    /// Whether `input` may move the camera once applied after everything queued before it:
    /// motion while a drag is in progress or about to start (`pending_press`: a press is
    /// queued, not yet applied), or a wheel turn. What asks for a frame; hovering asks for
    /// none.
    pub(crate) fn may_move(&self, input: &Input, pending_press: bool) -> bool {
        match input {
            Input::Move { .. } => self.drag.is_some() || pending_press,
            Input::Wheel { notches } => *notches != 0.0 && notches.is_finite(),
            Input::Press { .. } | Input::Release { .. } | Input::Modifiers(_) => false,
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
            Input::Press { button, at } => {
                if button == Button::PenBarrel {
                    self.barrel = true;
                }
                if self.drag.is_none()
                    && let Some(gesture) = self.gesture_for(button)
                {
                    self.drag = Some(Drag {
                        button,
                        gesture,
                        last: at,
                    });
                }
                false
            }
            Input::Release { button } => {
                if button == Button::PenBarrel {
                    self.barrel = false;
                }
                if self.drag.is_some_and(|drag| drag.button == button) {
                    self.drag = None;
                }
                false
            }
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
            Input::Modifiers(modifiers) => {
                self.modifiers = modifiers;
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
        let idle = Navigator::default();
        assert!(!idle.may_move(&motion, false), "hovering");
        assert!(idle.may_move(&motion, true), "a press is queued before it");
        assert!(idle.may_move(&Input::Wheel { notches: -1.0 }, false));
        assert!(!idle.may_move(&Input::Wheel { notches: 0.0 }, false));
        assert!(!idle.may_move(&Input::Wheel { notches: f64::NAN }, false));
        let press = Input::Press {
            button: Button::MouseLeft,
            at: at(0.0, 0.0),
        };
        assert!(!idle.may_move(&press, false), "a press alone moves nothing");
        let (_, dragging) = run(&[press]);
        assert!(dragging.may_move(&motion, false), "motion in a drag");
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
