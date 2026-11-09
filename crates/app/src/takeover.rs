//! Taking over from a finisher (the dab journal spec, "Taking over").
//!
//! The close bar's Save while dabs still land has the window go at once, and the process finishes
//! unseen ([`crate::frame::Owner::finish_unseen_unless`]), holding the project file's lock and its
//! recovery file's until it has landed and saved. Opening the project meanwhile finds it locked.
//! Rather than wait for the landing, the new instance asks the finisher to stop, and restores what
//! it leaves: the recovery file holds every dab placed, so nothing is lost.
//!
//! **The protocol** ([`socket`], Linux only): the finisher listens on a Unix socket named by its
//! project file's identity, `$XDG_RUNTIME_DIR/brushcraft/<device>-<inode>.sock`, bound before its
//! window goes (so an opener never finds a finisher that can't hear it) and removed as it exits.
//! The opener writes `stop\n`. The finisher looks between batches; once asked, it flushes, waits
//! for that save point to be durable, closes the project keeping the recovery file, writes
//! `stopped\n`, removes the socket and exits with success: a takeover is no failure, and nobody
//! is notified. The opener then waits for the locks to be free and opens the project as usual,
//! restoring the recovery file, the stroke still landing resumed with it
//! ([`crate::frame::Owner::resume_stroke`]).
//!
//! **Before any window** ([`make_way`]): the opener asks and waits before it opens the project, so
//! a takeover that works never shows anything but the project. It waits [`Patience::answer`] (5 s)
//! for `stopped`, then [`Patience::release`] (5 s more) for the locks.
//!
//! **While the window waits** ([`Waiter`]): it looks at the lock every [`LOCK_POLL`] and asks again
//! every [`ASK_AGAIN`], on a thread of its own so the window never blocks, so a finisher that
//! binds late, or a window that becomes a finisher meanwhile, is taken over at once. Its bar says
//! who holds the project ([`Holder`]).
//!
//! **Only the project file's lock is probed** ([`is_locked`]). A finisher holds both files, so that
//! is enough to find one; a recovery file held alone, for a moment (an instance just closing),
//! makes the open fail as held, and the window waits and opens about [`LOCK_POLL`] later.
//!
//! **Failure modes:**
//!
//! - *The project isn't locked:* nothing to take over ([`Way::Clear`]).
//! - *A finisher that doesn't answer in time,* or that can't be reached as there is no channel at
//!   all (no `XDG_RUNTIME_DIR`, a runtime directory too deep for a socket's address, or not Linux):
//!   [`Way::Busy`] with [`Holder::Finisher`], and the window waits with the "Finishing your last
//!   session" bar and the time it has waited.
//! - *Nobody answers on a working channel* (no socket, a stale socket nothing listens on, left for
//!   the next finisher to replace, or one that isn't ours): the project is open in another window,
//!   as a finisher always listens. [`Holder::Elsewhere`]: the bar says "`<name>` is open
//!   elsewhere".
//! - *It said `stopped` and the locks stay held:* [`Holder::Finisher`], after
//!   [`Patience::release`].
//! - Either way the project opens once it is free: the finisher has saved by then, or stopped, or
//!   died, and the recovery file holds whatever it didn't save.
//! - *The finisher couldn't make its flush durable* (crash protection paused or the writer failed):
//!   it still closes, keeping the recovery file at its last durable save point, answers `stopped`,
//!   and says on stderr that recent work may not be in it. The opener restores what is there, as
//!   after a crash.
//! - *A killed or crashed finisher:* its locks go with it, its socket file is stale, and the next
//!   open restores the recovery file as after any crash.
//! - *Garbage or a second `stop`:* garbage is dropped unanswered; another `stop` is one more asker
//!   answered. Neither changes what the finisher does.
//!
//! Elsewhere than Linux there is no socket yet: the opener goes straight to the waiting bar.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::time::{Duration, Instant};

#[cfg(target_os = "linux")]
mod socket;

#[cfg(target_os = "linux")]
pub(crate) use self::socket::{Answer, FileIdentity, Listener, SocketDir, ask_to_stop};

/// Where the takeover's sockets would live: there are none here.
#[cfg(not(target_os = "linux"))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SocketDir {}

#[cfg(not(target_os = "linux"))]
impl SocketDir {
    /// None: this system has no takeover socket yet.
    pub(crate) const fn from_env() -> Option<Self> {
        None
    }
}

/// A finisher's ears: there are none here, so none is ever made.
#[cfg(not(target_os = "linux"))]
#[derive(Debug)]
pub(crate) enum Listener {}

#[cfg(not(target_os = "linux"))]
impl Listener {
    /// Never called: there is no listener.
    pub(crate) const fn stop_asked(&mut self) -> bool {
        match *self {}
    }

    /// Never called: there is no listener.
    pub(crate) const fn answer(self) {
        match self {}
    }
}

/// How often the waiting window looks at the project's lock.
pub(crate) const LOCK_POLL: Duration = Duration::from_millis(200);

/// How often the waiting window asks the holder to stop again, once an ask has come back.
pub(crate) const ASK_AGAIN: Duration = Duration::from_secs(1);

/// How often the waiting window looks whether an ask in flight has come back.
const ASK_POLL: Duration = Duration::from_millis(20);

/// How often the opener looks at the lock once the finisher has said `stopped`: it is released
/// by then, so this is a few looks at most.
const RELEASE_POLL: Duration = Duration::from_millis(2);

/// How long the opener waits before it gives up and shows the waiting window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Patience {
    /// For the finisher's `stopped`, connecting included.
    pub(crate) answer: Duration,
    /// After `stopped`, for the locks to be free.
    pub(crate) release: Duration,
}

impl Patience {
    /// The spec's: 5 s for the answer, 5 s more for the locks.
    pub(crate) const DEFAULT: Self = Self {
        answer: Duration::from_secs(5),
        release: Duration::from_secs(5),
    };
}

/// Who holds a project the opener waits for, as far as asking found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Holder {
    /// A finisher: it listens and hasn't let go, or it can't be reached at all. "Finishing your
    /// last session".
    Finisher,
    /// Another window: nobody listens on a working channel. "`<name>` is open elsewhere".
    Elsewhere,
}

/// What the opener found on its way to the project.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Way {
    /// The project wasn't locked.
    Clear,
    /// A finisher held it, and has stopped and let it go.
    TookOver,
    /// It is held, and nothing let it go in time: the waiting window, from `since`.
    Busy {
        /// When the opener first found it held.
        since: Instant,
        /// Who holds it.
        holder: Holder,
    },
}

/// What one ask came to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Asked {
    /// A finisher said it has stopped.
    Stopped,
    /// A finisher listens, and didn't say so in time.
    Finisher,
    /// Nobody listens on a working channel: another window holds the project.
    Nobody,
    /// There is no channel to ask on.
    Unreachable,
}

impl Asked {
    /// Who holds the project, as this says.
    const fn holder(self) -> Holder {
        match self {
            Self::Stopped | Self::Finisher | Self::Unreachable => Holder::Finisher,
            Self::Nobody => Holder::Elsewhere,
        }
    }
}

/// Whether the project file at `path` is locked for writing: a shared lock can't be taken on it.
///
/// A file that can't be opened, or a lock that fails otherwise, isn't locked: the open says what
/// is wrong with it.
pub(crate) fn is_locked(path: &Path) -> bool {
    let Ok(file) = File::open(path) else {
        return false;
    };
    // The shared lock, if taken, goes with the file, at once.
    matches!(
        file.try_lock_shared(),
        Err(std::fs::TryLockError::WouldBlock)
    )
}

/// Clears the way to the project at `path` before it opens: if it is locked, asks its finisher,
/// listening in `sockets`, to stop, and waits for the locks to be free, as `patience` says.
pub(crate) fn make_way(path: &Path, sockets: Option<&SocketDir>, patience: Patience) -> Way {
    if !is_locked(path) {
        return Way::Clear;
    }
    let since = Instant::now();
    let asked = ask(path, sockets, patience.answer);
    if asked == Asked::Stopped {
        let deadline = Instant::now() + patience.release;
        while Instant::now() < deadline {
            if !is_locked(path) {
                return Way::TookOver;
            }
            std::thread::sleep(RELEASE_POLL);
        }
    }
    // Let go meanwhile, by itself: nothing to wait for.
    if !is_locked(path) {
        return if asked == Asked::Stopped {
            Way::TookOver
        } else {
            Way::Clear
        };
    }
    Way::Busy {
        since,
        holder: asked.holder(),
    }
}

/// Asks the finisher of the file at `path`, through `sockets`, to stop, waiting `within` at most.
#[cfg(target_os = "linux")]
fn ask(path: &Path, sockets: Option<&SocketDir>, within: Duration) -> Asked {
    let (Some(sockets), Ok(file)) = (sockets, FileIdentity::of(path)) else {
        return Asked::Unreachable;
    };
    match ask_to_stop(sockets, file, within) {
        Answer::Stopped => Asked::Stopped,
        Answer::Unanswered => Asked::Finisher,
        Answer::NoFinisher => Asked::Nobody,
        Answer::NoChannel => Asked::Unreachable,
    }
}

/// Nothing to ask on here: there is no socket yet.
#[cfg(not(target_os = "linux"))]
fn ask(path: &Path, sockets: Option<&SocketDir>, within: Duration) -> Asked {
    let _unasked = (path, sockets, within);
    Asked::Unreachable
}

/// The waiting window's patience: when to look at the lock, and when to ask the holder again, on
/// a thread of its own, so nothing here blocks.
#[derive(Debug)]
pub(crate) struct Waiter {
    /// The project file.
    path: PathBuf,
    /// Where finishers listen, if anywhere.
    sockets: Option<SocketDir>,
    /// How long an ask waits for its answer.
    within: Duration,
    /// When the project was first found held.
    since: Instant,
    /// Who holds it, as far as asking found.
    holder: Holder,
    /// When to look at the lock next.
    next_look: Instant,
    /// When to ask next, with no ask in flight.
    next_ask: Instant,
    /// The ask in flight, if one is: its answer comes here.
    asking: Option<Receiver<Asked>>,
}

impl Waiter {
    /// Waiting for the project file at `path`, held by `holder` since `since`, at `now`; asking
    /// through `sockets`, each ask waiting `patience.answer` at most.
    pub(crate) fn new(
        path: PathBuf,
        sockets: Option<SocketDir>,
        patience: Patience,
        (since, holder): (Instant, Holder),
        now: Instant,
    ) -> Self {
        Self {
            path,
            sockets,
            within: patience.answer,
            since,
            holder,
            next_look: now + LOCK_POLL,
            next_ask: now + ASK_AGAIN,
            asking: None,
        }
    }

    /// A pass at `now`: hears the ask in flight if it has come back, or starts the next one when
    /// it is due. Returns whether the lock is to be looked at now: when due, and at once after a
    /// finisher says it has stopped.
    ///
    /// A finisher found listening makes the holder [`Holder::Finisher`]; finding nobody leaves it
    /// as it was, as a finisher known to hold the project may be mid-exit.
    pub(crate) fn step(&mut self, now: Instant) -> bool {
        if let Some(asking) = &self.asking {
            let heard = match asking.try_recv() {
                Ok(asked) => Some(asked),
                Err(TryRecvError::Empty) => None,
                Err(TryRecvError::Disconnected) => Some(Asked::Unreachable),
            };
            if let Some(asked) = heard {
                self.asking = None;
                self.next_ask = now + ASK_AGAIN;
                match asked {
                    Asked::Stopped => {
                        self.holder = Holder::Finisher;
                        self.next_look = now;
                    }
                    Asked::Finisher => self.holder = Holder::Finisher,
                    Asked::Nobody | Asked::Unreachable => {}
                }
            }
        } else if now >= self.next_ask && self.sockets.is_some() {
            self.asking = Some(self.ask());
        }
        now >= self.next_look
    }

    /// Starts an ask on a thread of its own: its answer comes back on what is returned. A thread
    /// that can't be spawned answers that nobody could be reached, and is tried again later.
    fn ask(&self) -> Receiver<Asked> {
        let (answer, asked) = mpsc::channel();
        let (path, sockets, within) = (self.path.clone(), self.sockets.clone(), self.within);
        let spawned = std::thread::Builder::new()
            .name("takeover-ask".to_owned())
            .spawn(move || {
                // The window may have opened and stopped listening: then nobody hears.
                answer
                    .send(ask(&path, sockets.as_ref(), within))
                    .unwrap_or_default();
            });
        // A thread not spawned drops its sender: the next step hears it as unreachable.
        drop(spawned);
        asked
    }

    /// The lock was looked at, at `now`, and is still held: the next look is due a
    /// [`LOCK_POLL`] on.
    pub(crate) fn looked(&mut self, now: Instant) {
        self.next_look = now + LOCK_POLL;
    }

    /// When the window should come back, at the latest, from `now`: for the next look, the next
    /// ask, or an ask in flight.
    pub(crate) fn wake(&self, now: Instant) -> Instant {
        let ask = if self.asking.is_some() {
            now + ASK_POLL
        } else if self.sockets.is_some() {
            self.next_ask
        } else {
            self.next_look
        };
        self.next_look.min(ask)
    }

    /// When the project was first found held.
    pub(crate) const fn since(&self) -> Instant {
        self.since
    }

    /// Who holds it, as far as asking found.
    pub(crate) const fn holder(&self) -> Holder {
        self.holder
    }
}

#[cfg(test)]
#[cfg(target_os = "linux")]
mod tests;
