//! The pen stream as navigation's input: the tip drags, the barrel buttons are modifiers, and
//! hovering moves the pointer that picking follows.
//!
//! | Pen | Navigation |
//! |---|---|
//! | in range, moving | the pointer moves ([`Input::Move`]): picking follows it |
//! | tip down, up | [`Button::PenTip`] pressed, released |
//! | first barrel button held (`BTN_STYLUS`) | [`Button::PenFirstBarrel`]: the tip zooms |
//! | second barrel button held (`BTN_STYLUS2`) | [`Button::PenBarrel`]: the tip pans |
//! | out of range | everything held is released |
//!
//! A lost `Up` (a `Down` while down, or `Enter` or `Hover` while down) releases the tip first,
//! so a drag never outlives the stroke that started it.
//!
//! A sample's button changes come before its motion and its tip, so a barrel pressed in the
//! frame the tip touches already chooses the gesture. On the fallback, winit reports the first
//! barrel button as the tip ([`super::fallback`]), so there it orbits as the tip does.

use input::{Buttons, Phase};

use super::PenSample;
use crate::navigate::{Button, Input};

/// The barrel buttons navigation reads, and the button each one is.
const BARRELS: [(Buttons, Button); 2] = [
    (Buttons::PRIMARY, Button::PenFirstBarrel),
    (Buttons::SECONDARY, Button::PenBarrel),
];

/// Follows the pen stream into navigation's input: what it has pressed so far.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct PenNavigation {
    /// Whether the tip is down, as navigation has heard.
    touching: bool,
    /// The barrel buttons navigation has heard pressed.
    buttons: Buttons,
}

impl PenNavigation {
    /// Hands `emit` navigation's input for `sample`, in order: barrel buttons changed first,
    /// then the motion, then the tip.
    pub(crate) fn inputs(&mut self, sample: &PenSample, mut emit: impl FnMut(Input)) {
        let at = sample.at;
        let held = if sample.phase == Phase::Leave {
            Buttons::default()
        } else {
            sample.buttons
        };
        for (bit, button) in BARRELS {
            let (was, is) = (self.buttons.contains(bit), held.contains(bit));
            if was != is {
                emit(if is {
                    Input::Press { button, at }
                } else {
                    Input::Release { button }
                });
            }
        }
        self.buttons = held;
        match sample.phase {
            Phase::Enter | Phase::Hover => {
                if std::mem::replace(&mut self.touching, false) {
                    // In range and not touching while down: the up between was lost.
                    emit(Input::Release {
                        button: Button::PenTip,
                    });
                }
                emit(Input::Move { at });
            }
            Phase::Drag => emit(Input::Move { at }),
            Phase::Down => {
                emit(Input::Move { at });
                if std::mem::replace(&mut self.touching, true) {
                    // A down while down: the up between was lost. End the old drag first.
                    emit(Input::Release {
                        button: Button::PenTip,
                    });
                }
                emit(Input::Press {
                    button: Button::PenTip,
                    at,
                });
            }
            Phase::Up | Phase::Leave => {
                if sample.phase == Phase::Up {
                    emit(Input::Move { at });
                }
                if std::mem::replace(&mut self.touching, false) {
                    emit(Input::Release {
                        button: Button::PenTip,
                    });
                }
            }
        }
    }
}
