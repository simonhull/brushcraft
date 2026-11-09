//! Taking over from a finisher (the dab journal spec, "Taking over").
//!
//! The close bar's Save while dabs still land has the window go at once, and the process finishes
//! unseen ([`crate::frame::Owner::finish_unseen_unless`]), holding the project file's lock and its
//! recovery file's until it has landed and saved. Opening the project meanwhile finds it locked.
//! Rather than wait for the landing, the new instance asks the finisher to stop, and restores what
//! it leaves: the recovery file holds every dab placed, so nothing is lost.
//!
//! **The protocol** ([`socket`], Linux only): the finisher listens on a Unix socket named by its
//! project file's identity, `$XDG_RUNTIME_DIR/brushcraft/<device>-<inode>.sock`, from before its
//! landing until it exits. The opener writes `stop\n`. The finisher looks between batches; once
//! asked, it flushes, waits for that save point to be durable, closes the project keeping the
//! recovery file, writes `stopped\n`, removes the socket and exits with success: a takeover is no
//! failure, and nobody is notified. The opener then waits for the locks to be free and opens the
//! project as usual, restoring the recovery file, the stroke still landing resumed with it
//! ([`crate::frame::Owner::resume_stroke`]).
//!
//! **Before any window** ([`make_way`]): the opener asks and waits before it opens the project, so
//! a takeover that works never shows anything but the project. It waits [`Patience::answer`] (5 s)
//! for `stopped`, then [`Patience::release`] (5 s more) for the locks.
//!
//! **Failure modes:**
//!
//! - *The project isn't locked:* nothing to take over ([`Way::Clear`]).
//! - *No finisher answers in time,* or none can be reached (no `XDG_RUNTIME_DIR`, no socket, a
//!   socket nothing listens on, which is removed as stale, or one that isn't ours): [`Way::Busy`].
//!   The window then opens with the "Finishing your last session" bar and the time it has waited,
//!   polling the lock every [`LOCK_POLL`], and opens the project once it is free: the finisher has
//!   saved by then, or stopped, or died, and the recovery file holds whatever it didn't save. A
//!   project held by another window, not a finisher, looks the same, and opens once that window
//!   closes.
//! - *It said `stopped` and the locks stay held:* [`Way::Busy`] too, after [`Patience::release`].
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
use std::path::Path;
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

/// How often the waiting window looks at the project's lock.
pub(crate) const LOCK_POLL: Duration = Duration::from_millis(200);

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
    },
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
    if asked_to_stop(path, sockets, patience) {
        let deadline = Instant::now() + patience.release;
        while Instant::now() < deadline {
            if !is_locked(path) {
                return Way::TookOver;
            }
            std::thread::sleep(RELEASE_POLL);
        }
    }
    Way::Busy { since }
}

/// Whether the finisher of the file at `path` said it has stopped, asked through `sockets`.
#[cfg(target_os = "linux")]
fn asked_to_stop(path: &Path, sockets: Option<&SocketDir>, patience: Patience) -> bool {
    let (Some(sockets), Ok(file)) = (sockets, FileIdentity::of(path)) else {
        return false;
    };
    ask_to_stop(sockets, file, patience.answer) == Answer::Stopped
}

/// Nothing to ask here: there is no socket yet, so no finisher says it has stopped.
#[cfg(not(target_os = "linux"))]
fn asked_to_stop(path: &Path, sockets: Option<&SocketDir>, patience: Patience) -> bool {
    let _unasked = (path, sockets, patience.answer);
    false
}

#[cfg(test)]
#[cfg(target_os = "linux")]
mod tests;
