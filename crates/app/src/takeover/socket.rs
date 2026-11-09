//! The takeover's channel on Linux: a Unix socket per project file, in a directory of our own
//! under `$XDG_RUNTIME_DIR`, and the line each side writes on it.
//!
//! **Where:** `$XDG_RUNTIME_DIR/brushcraft/<device>-<inode>.sock` ([`SocketDir`],
//! [`FileIdentity`]), both numbers in lower-case hex, of the project file as its path names it
//! (links followed). The directory is made mode 0700 if it is missing, and refused unless it is a
//! real directory (no link), owned by us, with no bits for group or others: anything else could be
//! someone else's, and nothing in it is used or removed. With no `XDG_RUNTIME_DIR` there is no
//! socket, and the opener waits ([`super::Way::Busy`]).
//!
//! **Who:** the finisher binds the socket as it starts, before it lands anything
//! ([`Listener::bind`]), and removes it as it goes, on every path, a panic's unwinding included
//! ([`Listener`]'s drop). A socket file a crashed or killed finisher left is stale: connecting is
//! refused (`ECONNREFUSED`), so whoever finds it (the next finisher binding, or an opener asking)
//! removes it, if it is still the same file and ours. Only a socket file owned by us is connected
//! to, and only a peer whose credentials are ours is listened to or believed.
//!
//! **What:** the opener writes `stop\n` ([`ask_to_stop`]). The finisher reads it at its next look
//! between batches ([`Listener::stop_asked`]), stops (the frame loop's
//! [`crate::frame::Owner::finish_unseen_unless`]), and once its project is closed, its locks
//! released, writes `stopped\n` to everyone who asked ([`Listener::answer`]) and removes the
//! socket. A finisher that finishes on its own answers those who asked meanwhile all the same:
//! its locks are free.
//!
//! **Bounded:** a request is at most [`MAX_LINE`] bytes and must arrive within
//! [`REQUEST_TIMEOUT`] of its connection, and at most [`MAX_PENDING`] connections wait to be read;
//! anything longer, slower, more, or not exactly `stop` is dropped, unanswered, and changes
//! nothing. A second `stop`, on the same connection or another, is one more asker answered. The
//! opener's connect, write and read share one deadline, so a finisher that hangs costs it that and
//! no more. Answers are written with [`ANSWER_TIMEOUT`] each, and a reader gone is no error.

use std::fs::{self, DirBuilder, Metadata};
use std::io::{self, Read as _, Write as _};
use std::os::unix::fs::{DirBuilderExt as _, FileTypeExt as _, MetadataExt as _};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use rustix::net::sockopt::{Timeout, set_socket_timeout, socket_peercred};
use rustix::net::{AddressFamily, SocketAddrUnix, SocketFlags, SocketType};
use thiserror::Error;

/// The longest request line read, its newline included: `stop\n` and room to spare.
pub(crate) const MAX_LINE: usize = 64;

/// How long a connection may take to send its request before it is dropped.
pub(crate) const REQUEST_TIMEOUT: Duration = Duration::from_secs(1);

/// The most connections waiting for their request at once: one more is dropped at once.
pub(crate) const MAX_PENDING: usize = 8;

/// How long writing `stopped` to one asker may take.
const ANSWER_TIMEOUT: Duration = Duration::from_secs(1);

/// What the opener writes.
const STOP: &[u8] = b"stop\n";

/// What the finisher answers.
const STOPPED: &[u8] = b"stopped\n";

/// The bits of a mode that give anyone but the owner access.
const OTHERS: u32 = 0o077;

/// Why there is no socket, or one can't be used.
#[derive(Debug, Error)]
pub(crate) enum SocketError {
    /// The directory isn't ours alone: a link, not a directory, another user's, or open to others.
    #[error("{0} isn't a directory of ours alone")]
    Untrusted(PathBuf),
    /// A live finisher of this file already listens there.
    #[error("another finisher of this file listens at {0}")]
    InUse(PathBuf),
    /// Making, checking or binding failed.
    #[error("the takeover socket")]
    Io(#[from] io::Error),
}

/// The identity of a project file: its device and inode, as its path names it, links followed.
///
/// What names its finisher's socket: two copies of a project share a project id, but never a
/// file, and the identity of the file a finisher holds can't change under it, as replacing it
/// (compacting in place) needs the lock the finisher holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FileIdentity {
    /// The device the file is on.
    device: u64,
    /// Its inode there.
    inode: u64,
}

impl FileIdentity {
    /// The identity of the file at `path`, links followed.
    ///
    /// # Errors
    ///
    /// The I/O error of `stat`.
    pub(crate) fn of(path: &Path) -> io::Result<Self> {
        let meta = fs::metadata(path)?;
        Ok(Self {
            device: meta.dev(),
            inode: meta.ino(),
        })
    }

    /// The name of its finisher's socket.
    fn socket_name(self) -> String {
        format!("{:x}-{:x}.sock", self.device, self.inode)
    }
}

/// Our effective uid: what owns a directory or socket we may trust, and a peer we may listen to.
pub(super) fn our_uid() -> u32 {
    rustix::process::geteuid().as_raw()
}

/// Whether `meta`, of something found at a path, is owned by `uid` with no bits for anyone else.
pub(super) fn ours_alone(meta: &Metadata, uid: u32) -> bool {
    meta.uid() == uid && meta.mode() & OTHERS == 0
}

/// Whether `meta` is a socket file owned by `uid`: one we may connect to, or remove as stale.
pub(super) fn our_socket(meta: &Metadata, uid: u32) -> bool {
    meta.file_type().is_socket() && meta.uid() == uid
}

/// Whether the peer on `stream` runs as us.
fn peer_is_us(stream: &UnixStream) -> bool {
    socket_peercred(stream).is_ok_and(|cred| cred.uid.as_raw() == our_uid())
}

/// The directory the takeover's sockets live in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SocketDir(PathBuf);

impl SocketDir {
    /// `$XDG_RUNTIME_DIR/brushcraft`; `None` if `XDG_RUNTIME_DIR` is unset, empty or relative.
    pub(crate) fn from_env() -> Option<Self> {
        let runtime = PathBuf::from(std::env::var_os("XDG_RUNTIME_DIR")?);
        runtime
            .is_absolute()
            .then(|| Self(runtime.join("brushcraft")))
    }

    /// The directory at `path`: a test's, never the real one.
    #[cfg(test)]
    pub(crate) const fn at(path: PathBuf) -> Self {
        Self(path)
    }

    /// Where the finisher of `file` listens.
    fn socket(&self, file: FileIdentity) -> PathBuf {
        self.0.join(file.socket_name())
    }

    /// Makes the directory, mode 0700, if it is missing, then checks it as [`SocketDir::trusted`].
    ///
    /// # Errors
    ///
    /// [`SocketError::Untrusted`] as [`SocketDir::trusted`]; [`SocketError::Io`] if it can't be
    /// made or looked at.
    fn ensure(&self) -> Result<(), SocketError> {
        match DirBuilder::new().mode(0o700).create(&self.0) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error.into()),
        }
        self.trusted()
    }

    /// Checks the directory is ours alone: a directory, not a link to one, owned by us, with no
    /// bits for group or others.
    ///
    /// # Errors
    ///
    /// [`SocketError::Untrusted`] if it isn't; [`SocketError::Io`] if it can't be looked at (a
    /// missing one included).
    fn trusted(&self) -> Result<(), SocketError> {
        let meta = fs::symlink_metadata(&self.0)?;
        if meta.file_type().is_dir() && ours_alone(&meta, our_uid()) {
            Ok(())
        } else {
            Err(SocketError::Untrusted(self.0.clone()))
        }
    }
}

/// Removes the socket file at `path` if it is still the one `identity` names (device, inode) and
/// ours: never a file someone put there since. Best effort; a file gone already is what was
/// wanted.
fn remove_if_same(path: &Path, identity: (u64, u64)) {
    let same = fs::symlink_metadata(path)
        .is_ok_and(|meta| our_socket(&meta, our_uid()) && (meta.dev(), meta.ino()) == identity);
    if same {
        drop(fs::remove_file(path));
    }
}

/// Connects to the socket at `path` by `deadline` at most.
///
/// A Unix socket's connect waits only while the listener's backlog is full, and then no longer
/// than the socket's send timeout, so the socket is made by hand with that timeout set first, as
/// the notification client's is.
///
/// # Errors
///
/// [`io::ErrorKind::ConnectionRefused`] or [`io::ErrorKind::NotFound`] if nothing listens there,
/// [`io::ErrorKind::TimedOut`] (or `WouldBlock`) if the deadline passes, and any other I/O error.
fn connect_by(path: &Path, deadline: Instant) -> io::Result<UnixStream> {
    let address = SocketAddrUnix::new(path)?;
    let socket = rustix::net::socket_with(
        AddressFamily::UNIX,
        SocketType::STREAM,
        SocketFlags::CLOEXEC,
        None,
    )?;
    set_socket_timeout(&socket, Timeout::Send, Some(left(deadline)?))?;
    rustix::net::connect(&socket, &address)?;
    Ok(UnixStream::from(socket))
}

/// The time left until `deadline`.
///
/// # Errors
///
/// [`io::ErrorKind::TimedOut`] if none is.
fn left(deadline: Instant) -> io::Result<Duration> {
    let left = deadline.saturating_duration_since(Instant::now());
    if left.is_zero() {
        return Err(io::ErrorKind::TimedOut.into());
    }
    Ok(left)
}

/// Whether `error`, of a connect, says nothing listens: the socket is stale, or gone.
fn nothing_listens(error: &io::Error) -> bool {
    matches!(
        error.kind(),
        io::ErrorKind::ConnectionRefused | io::ErrorKind::NotFound
    )
}

/// A connection read for its request, until it sends one or runs out of time.
#[derive(Debug)]
struct Pending {
    /// The connection, non-blocking.
    stream: UnixStream,
    /// What it has sent so far.
    line: Vec<u8>,
    /// When it was accepted.
    since: Instant,
}

/// What a read of a pending connection came to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Request {
    /// `stop`: it is to be answered.
    Stop,
    /// Nothing whole yet: read it again later.
    Waiting,
    /// Anything else, too long, too late, or a connection gone: dropped.
    Dropped,
}

impl Pending {
    /// Reads what the connection has sent by `now`, up to [`MAX_LINE`] in all.
    fn read(&mut self, now: Instant) -> Request {
        let mut chunk = [0_u8; MAX_LINE];
        loop {
            let room = MAX_LINE.saturating_sub(self.line.len());
            let Some(buffer) = chunk.get_mut(..room).filter(|buffer| !buffer.is_empty()) else {
                return Request::Dropped;
            };
            match self.stream.read(buffer) {
                Ok(0) => return Request::Dropped,
                Ok(read) => {
                    self.line.extend(buffer.iter().take(read));
                    if let Some(end) = self.line.iter().position(|&byte| byte == b'\n') {
                        let first = self.line.get(..=end).unwrap_or_default();
                        return if first == STOP {
                            Request::Stop
                        } else {
                            Request::Dropped
                        };
                    }
                }
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    let late = now.saturating_duration_since(self.since) >= REQUEST_TIMEOUT;
                    return if late {
                        Request::Dropped
                    } else {
                        Request::Waiting
                    };
                }
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(_) => return Request::Dropped,
            }
        }
    }
}

/// The finisher's end of the socket: listening from before its landing until it has stopped.
///
/// Dropped, on any path, it removes its socket file, if that is still its own.
#[derive(Debug)]
pub(crate) struct Listener {
    /// The socket, non-blocking.
    socket: UnixListener,
    /// Its file.
    path: PathBuf,
    /// That file's device and inode as bound: what may be removed.
    identity: (u64, u64),
    /// Connections waiting for their request.
    pending: Vec<Pending>,
    /// Connections that asked to stop, to answer once stopped.
    asked: Vec<UnixStream>,
}

impl Listener {
    /// Listens for openers of `file` in `dir`, which is made if it is missing.
    ///
    /// A stale socket file at the name (nothing listens) is removed first, if it is ours.
    ///
    /// # Errors
    ///
    /// [`SocketError::Untrusted`] if `dir` isn't ours alone, [`SocketError::InUse`] if a live
    /// finisher of the file listens there already, [`SocketError::Io`] otherwise.
    pub(crate) fn bind(dir: &SocketDir, file: FileIdentity) -> Result<Self, SocketError> {
        dir.ensure()?;
        let path = dir.socket(file);
        let socket = match UnixListener::bind(&path) {
            Ok(socket) => socket,
            Err(error) if error.kind() == io::ErrorKind::AddrInUse => {
                let found = fs::symlink_metadata(&path)?;
                if !our_socket(&found, our_uid()) {
                    return Err(SocketError::InUse(path));
                }
                let deadline = Instant::now() + REQUEST_TIMEOUT;
                match connect_by(&path, deadline) {
                    Err(refused) if nothing_listens(&refused) => {
                        remove_if_same(&path, (found.dev(), found.ino()));
                        UnixListener::bind(&path)?
                    }
                    _ => return Err(SocketError::InUse(path)),
                }
            }
            Err(error) => return Err(error.into()),
        };
        socket.set_nonblocking(true)?;
        let bound = fs::symlink_metadata(&path)?;
        Ok(Self {
            socket,
            path,
            identity: (bound.dev(), bound.ino()),
            pending: Vec::new(),
            asked: Vec::new(),
        })
    }

    /// Whether anyone has asked the finisher to stop, by now: takes the connections waiting,
    /// reads what they have sent, and drops what isn't a request of ours. Never blocks.
    pub(crate) fn stop_asked(&mut self) -> bool {
        self.accept();
        let now = Instant::now();
        let pending = std::mem::take(&mut self.pending);
        for mut connection in pending {
            match connection.read(now) {
                Request::Stop => self.asked.push(connection.stream),
                Request::Waiting => self.pending.push(connection),
                Request::Dropped => {}
            }
        }
        !self.asked.is_empty()
    }

    /// Takes the connections waiting to be accepted, keeping those of our own user, up to
    /// [`MAX_PENDING`] unread.
    fn accept(&mut self) {
        loop {
            let stream = match self.socket.accept() {
                Ok((stream, _)) => stream,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                // `WouldBlock`, or anything else: look again at the next batch.
                Err(_) => return,
            };
            let room = self.pending.len() < MAX_PENDING;
            if room && peer_is_us(&stream) && stream.set_nonblocking(true).is_ok() {
                self.pending.push(Pending {
                    stream,
                    line: Vec::new(),
                    since: Instant::now(),
                });
            }
        }
    }

    /// Says `stopped` to everyone who asked, those asking just now included, and removes the
    /// socket: for once the project is closed, its locks released. Best effort: an asker gone is
    /// no error.
    pub(crate) fn answer(mut self) {
        self.stop_asked();
        remove_if_same(&self.path, self.identity);
        let asked = std::mem::take(&mut self.asked);
        for mut stream in asked {
            let ready = stream
                .set_nonblocking(false)
                .and_then(|()| stream.set_write_timeout(Some(ANSWER_TIMEOUT)));
            if ready.is_ok() {
                drop(stream.write_all(STOPPED));
            }
        }
    }

    /// Where it listens.
    #[cfg(test)]
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for Listener {
    /// Removes the socket file, if it is still this one's. Never panics.
    fn drop(&mut self) {
        remove_if_same(&self.path, self.identity);
    }
}

/// What came of asking a finisher to stop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Answer {
    /// It said `stopped`: its project is closed, its locks released.
    Stopped,
    /// There's no finisher to ask: no socket, a stale one (removed), or one we must not use.
    NoFinisher,
    /// One listens, but didn't say `stopped` in time.
    Unanswered,
}

/// Asks the finisher of `file`, listening in `dir`, to stop, and waits `within` at most for it to
/// say it has: connecting, writing and reading all count.
///
/// Nothing is used that isn't ours: a directory open to others, a socket file of another user or
/// not a socket at all, or a peer running as someone else is [`Answer::NoFinisher`], left as it
/// is. A socket file nothing listens on is stale, and removed.
pub(crate) fn ask_to_stop(dir: &SocketDir, file: FileIdentity, within: Duration) -> Answer {
    let deadline = Instant::now() + within;
    if dir.trusted().is_err() {
        return Answer::NoFinisher;
    }
    let path = dir.socket(file);
    let Ok(found) = fs::symlink_metadata(&path) else {
        return Answer::NoFinisher;
    };
    if !our_socket(&found, our_uid()) {
        return Answer::NoFinisher;
    }
    let mut stream = match connect_by(&path, deadline) {
        Ok(stream) => stream,
        Err(error) if nothing_listens(&error) => {
            remove_if_same(&path, (found.dev(), found.ino()));
            return Answer::NoFinisher;
        }
        Err(_) => return Answer::Unanswered,
    };
    if !peer_is_us(&stream) {
        return Answer::NoFinisher;
    }
    let asked = left(deadline)
        .and_then(|left| stream.set_write_timeout(Some(left)))
        .and_then(|()| stream.write_all(STOP));
    if asked.is_err() {
        return Answer::Unanswered;
    }
    match read_answer(&mut stream, deadline) {
        Ok(true) => Answer::Stopped,
        Ok(false) | Err(_) => Answer::Unanswered,
    }
}

/// Reads one line from `stream` by `deadline`, at most [`MAX_LINE`] bytes: whether it is
/// `stopped`.
///
/// # Errors
///
/// If the deadline passes, or reading fails.
fn read_answer(stream: &mut UnixStream, deadline: Instant) -> io::Result<bool> {
    let mut line = Vec::new();
    let mut chunk = [0_u8; MAX_LINE];
    while line.len() < MAX_LINE {
        stream.set_read_timeout(Some(left(deadline)?))?;
        let read = match stream.read(&mut chunk) {
            Ok(0) => return Ok(false),
            Ok(read) => read,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        };
        line.extend(chunk.iter().take(read));
        if let Some(end) = line.iter().position(|&byte| byte == b'\n') {
            return Ok(line.get(..=end) == Some(STOPPED));
        }
    }
    Ok(false)
}
