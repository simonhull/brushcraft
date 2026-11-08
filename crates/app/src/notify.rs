//! Desktop notifications: the freedesktop `Notify` call over the session D-Bus, through a client
//! of our own ([`bus`]).
//!
//! What the app needs of D-Bus is one method call, so this is no general client: [`send`] reads
//! the session bus's address, connects to its Unix socket, authenticates with EXTERNAL (the
//! kernel vouches for our uid), and sends `org.freedesktop.DBus.Hello`, which every connection
//! must, then `org.freedesktop.Notifications.Notify`. It waits for Notify's answer, so a
//! notification server's error is reported, not lost, and gives up after [`TIMEOUT`] all told,
//! connecting included: a bus that stalls costs the caller two seconds, never a hang.
//!
//! Linux only, for now: elsewhere [`send`] says it is unsupported.

use std::io;
use std::time::Duration;

use thiserror::Error;

#[cfg(target_os = "linux")]
mod bus;

/// How long a notification may take, from connecting to the notification server's answer.
const TIMEOUT: Duration = Duration::from_secs(2);

/// Why a notification wasn't sent.
#[derive(Debug, Error)]
pub(crate) enum NotifyError {
    /// There is no session bus to send it on: `DBUS_SESSION_BUS_ADDRESS` is unset or empty.
    #[error("there is no session bus (DBUS_SESSION_BUS_ADDRESS is unset or empty)")]
    NoBus,
    /// None of the session bus's addresses is a Unix socket this client can connect to.
    #[error("no address of the session bus is a Unix socket: {0:?}")]
    NoUsableAddress(String),
    /// The text holds a NUL, which a D-Bus string can't carry.
    #[error("the notification's text holds a NUL")]
    Nul,
    /// A string too long for a D-Bus message.
    #[error("a string too long for a D-Bus message ({length} bytes)")]
    TooLong {
        /// Its length, in bytes.
        length: usize,
    },
    /// The bus's socket couldn't be connected to.
    #[error("connecting to the session bus")]
    Connect(#[source] io::Error),
    /// Reading from the bus or writing to it failed.
    #[error("talking to the session bus")]
    Io(#[source] io::Error),
    /// The bus closed the connection before the notification server answered.
    #[error("the session bus closed the connection")]
    Closed,
    /// The bus didn't let us connect or didn't answer in time ([`TIMEOUT`]).
    #[error("the session bus didn't answer in time")]
    TimedOut,
    /// The bus wouldn't let us in. It names the mechanisms it would take.
    #[error("the session bus refused to authenticate us (it takes {0:?})")]
    AuthRejected(String),
    /// The bus said something this client doesn't understand.
    #[error("the session bus broke the protocol: {0}")]
    Protocol(&'static str),
    /// The bus or the notification server answered with an error: no server is running, say.
    #[error("the notification was refused: {name}: {message}")]
    Refused {
        /// The error's D-Bus name, such as `org.freedesktop.DBus.Error.ServiceUnknown`.
        name: String,
        /// Its text, if it came with one.
        message: String,
    },
    /// This system has no session bus client here yet.
    #[error("desktop notifications aren't supported on this system")]
    #[cfg(not(target_os = "linux"))]
    Unsupported,
}

impl NotifyError {
    /// The error for a failed connect: [`NotifyError::TimedOut`] if it ran out of time.
    fn connect(error: io::Error) -> Self {
        if is_timeout(&error) {
            Self::TimedOut
        } else {
            Self::Connect(error)
        }
    }
}

impl From<io::Error> for NotifyError {
    /// A timeout's error is [`NotifyError::TimedOut`], an early end [`NotifyError::Closed`].
    fn from(error: io::Error) -> Self {
        if is_timeout(&error) {
            Self::TimedOut
        } else if error.kind() == io::ErrorKind::UnexpectedEof {
            Self::Closed
        } else {
            Self::Io(error)
        }
    }
}

/// Whether an I/O error is a timeout: a socket's timeout is `WouldBlock` on Unix, the client's
/// own deadline `TimedOut`.
fn is_timeout(error: &io::Error) -> bool {
    matches!(
        error.kind(),
        io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
    )
}

/// Shows a desktop notification: `summary` its title, `body` its text, from "BrushCraft".
///
/// Blocks until the notification server has answered, for [`TIMEOUT`] at most.
///
/// # Errors
///
/// If there is no session bus, it can't be reached or doesn't let us in, the server refuses the
/// notification or none is running, or nothing answers in time. Then nothing was shown.
#[cfg(target_os = "linux")]
pub(crate) fn send(summary: &str, body: &str) -> Result<(), NotifyError> {
    let address = std::env::var_os("DBUS_SESSION_BUS_ADDRESS");
    let address = address.as_deref().map(std::ffi::OsStr::to_string_lossy);
    let uid = rustix::process::getuid().as_raw();
    bus::send_to(address.as_deref(), uid, summary, body, TIMEOUT)
}

/// Shows a desktop notification, where there is a client for the system's bus: none here yet.
///
/// # Errors
///
/// Always: [`NotifyError::Unsupported`].
#[cfg(not(target_os = "linux"))]
pub(crate) fn send(_summary: &str, _body: &str) -> Result<(), NotifyError> {
    Err(NotifyError::Unsupported)
}
