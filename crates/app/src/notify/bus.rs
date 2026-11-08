//! The session bus client: connecting, the EXTERNAL handshake, and the two calls in D-Bus's wire
//! format.
//!
//! The marshalling follows the D-Bus specification: little-endian, the header's fields an array
//! of `(byte, variant)` structs, each 8-aligned, the body after the header padded to 8, every
//! value aligned to its size. Only the types these calls use are written: bytes, `u32`, `i32`,
//! strings, object paths, signatures and arrays. Replies are read only as far as the client needs:
//! their type, the call they answer, and an error's name and text. A big-endian reply is refused:
//! a bus answers in its own byte order, and no Linux machine this app runs on is big-endian.
//!
//! Every step is bounded by one deadline. Connecting waits, if the bus's listening backlog is
//! full, for at most the socket's send timeout, set to the time left; every read and write after
//! it re-arms the socket's timeout with what is left ([`Deadlined`]).

use std::io::{self, BufRead as _, BufReader, Read, Write};
use std::os::unix::ffi::OsStringExt as _;
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use rustix::net::sockopt::{Timeout, set_socket_timeout};
use rustix::net::{AddressFamily, SocketAddrUnix, SocketFlags, SocketType};

use super::NotifyError;

/// The app's name, as notifications give it.
const APP_NAME: &str = "BrushCraft";

/// The serial of the `Hello` call: the first message on the connection.
const HELLO_SERIAL: u32 = 1;

/// The serial of the `Notify` call, sent right after `Hello`.
const NOTIFY_SERIAL: u32 = 2;

/// The longest line the bus may answer the handshake with: its replies are a few dozen bytes.
const MAX_LINE: u64 = 1024;

/// The largest message read from the bus: what comes back is a reply or two and a signal, each
/// some dozens of bytes.
const MAX_MESSAGE: u32 = 64 * 1024;

/// The header field holding the object a call is to (an object path).
const PATH: u8 = 1;
/// The header field holding the member's interface.
const INTERFACE: u8 = 2;
/// The header field holding the method called.
const MEMBER: u8 = 3;
/// The header field holding an error's name.
const ERROR_NAME: u8 = 4;
/// The header field holding the serial of the call a reply answers.
const REPLY_SERIAL: u8 = 5;
/// The header field holding the connection a message is for.
const DESTINATION: u8 = 6;
/// The header field holding the body's signature.
const SIGNATURE: u8 = 8;

/// [`super::send`] to the bus at `address` (a `DBUS_SESSION_BUS_ADDRESS`, if there is one), as
/// `uid`, giving up after `timeout`.
///
/// # Errors
///
/// As [`super::send`].
pub(super) fn send_to(
    address: Option<&str>,
    uid: u32,
    summary: &str,
    body: &str,
    timeout: Duration,
) -> Result<(), NotifyError> {
    let deadline = Deadline(Instant::now() + timeout);
    let addresses = address
        .filter(|text| !text.is_empty())
        .ok_or(NotifyError::NoBus)?;
    let socket = parse_address(addresses)?;
    // Marshalled before connecting: text the bus would refuse never gets that far.
    let mut calls = hello_call()?;
    calls.extend(notify_call(NOTIFY_SERIAL, summary, body)?);
    let stream = socket.connect(deadline)?;
    let mut bus = BufReader::new(Deadlined {
        stream: &stream,
        deadline,
    });
    authenticate(&mut bus, uid)?;
    // Pipelined: the bus answers in order, and a failed Hello fails the Notify after it too.
    bus.get_mut().write_all(&calls)?;
    loop {
        let reply = read_reply(&mut bus)?;
        let ours = reply
            .answers
            .is_some_and(|serial| serial == HELLO_SERIAL || serial == NOTIFY_SERIAL);
        match reply.kind {
            Some(Kind::Error) if ours => return Err(reply.refusal()),
            Some(Kind::MethodReturn) if reply.answers == Some(NOTIFY_SERIAL) => return Ok(()),
            // Hello's reply, and the signal that tells a connection its name.
            _ => {}
        }
    }
}

/// When the whole exchange gives up.
#[derive(Debug, Clone, Copy)]
struct Deadline(Instant);

impl Deadline {
    /// The time left.
    ///
    /// # Errors
    ///
    /// [`io::ErrorKind::TimedOut`] if none is.
    fn left(self) -> io::Result<Duration> {
        let left = self.0.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Err(io::ErrorKind::TimedOut.into());
        }
        Ok(left)
    }
}

/// Where the session bus listens: one of the transports a bus address can name.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Address {
    /// A Unix socket at this path (`unix:path=`).
    Path(PathBuf),
    /// A Unix socket in the abstract namespace, by this name (`unix:abstract=`).
    Abstract(Vec<u8>),
}

impl Address {
    /// Connects to the socket, waiting until `deadline` at most.
    ///
    /// A Unix socket's connect waits only while the listener's backlog is full, and then for no
    /// longer than the socket's send timeout: so the socket is made by hand, its send timeout set
    /// to the time left, before connecting.
    ///
    /// # Errors
    ///
    /// [`NotifyError::TimedOut`] if the deadline passes, [`NotifyError::Connect`] if nothing
    /// listens there, we may not connect, or the address is no socket's.
    fn connect(&self, deadline: Deadline) -> Result<UnixStream, NotifyError> {
        let address = match self {
            Self::Path(path) => SocketAddrUnix::new(path.as_path()),
            Self::Abstract(name) => SocketAddrUnix::new_abstract_name(name),
        };
        let address = address.map_err(|errno| NotifyError::Connect(errno.into()))?;
        let socket = rustix::net::socket_with(
            AddressFamily::UNIX,
            SocketType::STREAM,
            SocketFlags::CLOEXEC,
            None,
        )
        .map_err(|errno| NotifyError::Connect(errno.into()))?;
        set_socket_timeout(&socket, Timeout::Send, Some(deadline.left()?))
            .map_err(|errno| NotifyError::Connect(errno.into()))?;
        rustix::net::connect(&socket, &address)
            .map_err(|errno| NotifyError::connect(errno.into()))?;
        Ok(UnixStream::from(socket))
    }
}

/// The first address in a D-Bus address list (`;`-separated `transport:key=value,…`) that is a
/// Unix socket by path or abstract name, its value unescaped.
///
/// # Errors
///
/// [`NotifyError::NoUsableAddress`] if none is, or a value's escapes are malformed.
fn parse_address(addresses: &str) -> Result<Address, NotifyError> {
    let mut sockets = addresses.split(';').filter_map(|entry| {
        let params = entry.strip_prefix("unix:")?;
        params.split(',').find_map(|param| {
            let (key, value) = param.split_once('=')?;
            match key {
                "path" => {
                    let path = std::ffi::OsString::from_vec(unescape(value)?);
                    Some(Address::Path(PathBuf::from(path)))
                }
                "abstract" => Some(Address::Abstract(unescape(value)?)),
                _ => None,
            }
        })
    });
    sockets
        .next()
        .ok_or_else(|| NotifyError::NoUsableAddress(addresses.to_owned()))
}

/// An address value with its `%xx` escapes decoded, or `None` if one is malformed.
fn unescape(value: &str) -> Option<Vec<u8>> {
    let mut bytes = value.bytes();
    let mut unescaped = Vec::with_capacity(value.len());
    while let Some(byte) = bytes.next() {
        if byte == b'%' {
            let high = char::from(bytes.next()?).to_digit(16)?;
            let low = char::from(bytes.next()?).to_digit(16)?;
            unescaped.push(u8::try_from(high * 16 + low).ok()?);
        } else {
            unescaped.push(byte);
        }
    }
    Some(unescaped)
}

/// The handshake before any message: the credentials byte, `AUTH EXTERNAL` with our uid's
/// decimal digits in hex, the bus's `OK`, then `BEGIN`.
///
/// # Errors
///
/// [`NotifyError::AuthRejected`] if the bus answers `REJECTED`, [`NotifyError::Protocol`] if it
/// answers anything else but `OK`, and as [`read_line`].
fn authenticate(bus: &mut BufReader<Deadlined<'_>>, uid: u32) -> Result<(), NotifyError> {
    // An ASCII digit's hex is a 3 and the digit itself: 1000 is 31303030.
    let decimal = uid.to_string();
    let hex_digits = decimal.chars().flat_map(|digit| ['3', digit]);
    let hex: String = hex_digits.collect();
    let auth = format!("\0AUTH EXTERNAL {hex}\r\n");
    bus.get_mut().write_all(auth.as_bytes())?;
    let answer = read_line(bus)?;
    if let Some(mechanisms) = answer.strip_prefix("REJECTED") {
        return Err(NotifyError::AuthRejected(mechanisms.trim().to_owned()));
    }
    if !answer.starts_with("OK ") {
        return Err(NotifyError::Protocol("an answer to AUTH that isn't OK"));
    }
    bus.get_mut().write_all(b"BEGIN\r\n")?;
    Ok(())
}

/// One line of the handshake from the bus, without its CRLF.
///
/// # Errors
///
/// [`NotifyError::Closed`] if the bus hangs up first, [`NotifyError::Protocol`] if the line is
/// longer than [`MAX_LINE`], isn't ASCII or doesn't end in CRLF, and as [`NotifyError::from`] an
/// [`io::Error`].
fn read_line(bus: &mut BufReader<Deadlined<'_>>) -> Result<String, NotifyError> {
    let mut line = Vec::new();
    if bus.take(MAX_LINE).read_until(b'\n', &mut line)? == 0 {
        return Err(NotifyError::Closed);
    }
    let text = line
        .strip_suffix(b"\r\n")
        .filter(|text| text.is_ascii())
        .ok_or(NotifyError::Protocol("a handshake line that isn't one"))?;
    Ok(String::from_utf8_lossy(text).into_owned())
}

/// The connection, every read and write bounded by what is left of the deadline.
#[derive(Debug)]
struct Deadlined<'a> {
    /// The connection to the bus.
    stream: &'a UnixStream,
    /// When the whole exchange gives up.
    deadline: Deadline,
}

impl Read for Deadlined<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.stream.set_read_timeout(Some(self.deadline.left()?))?;
        let mut stream = self.stream;
        stream.read(buf)
    }
}

impl Write for Deadlined<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.stream.set_write_timeout(Some(self.deadline.left()?))?;
        let mut stream = self.stream;
        stream.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        let mut stream = self.stream;
        stream.flush()
    }
}

/// A message's type, the header's second byte: those this client writes or reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// A method call.
    MethodCall = 1,
    /// A method's reply.
    MethodReturn = 2,
    /// An error reply.
    Error = 3,
}

impl Kind {
    /// The kind a header's byte names, if it is one of these: not for a signal, or a type the
    /// specification may add (which it says to ignore).
    const fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            1 => Some(Self::MethodCall),
            2 => Some(Self::MethodReturn),
            3 => Some(Self::Error),
            _ => None,
        }
    }
}

/// A header field a call carries, with its value.
#[derive(Debug, Clone, Copy)]
enum Field<'a> {
    /// The object called (an object path).
    Path(&'a str),
    /// The interface of the method.
    Interface(&'a str),
    /// The method.
    Member(&'a str),
    /// The connection called.
    Destination(&'a str),
    /// The body's signature.
    Signature(&'a str),
}

impl Field<'_> {
    /// Writes the field: its code, its variant's signature and its value, as one 8-aligned
    /// struct.
    ///
    /// # Errors
    ///
    /// As [`Marshal::string`].
    fn write(self, out: &mut Marshal) -> Result<(), NotifyError> {
        out.align(8);
        let (code, signature, value) = match self {
            Self::Path(path) => (PATH, "o", path),
            Self::Interface(interface) => (INTERFACE, "s", interface),
            Self::Member(member) => (MEMBER, "s", member),
            Self::Destination(destination) => (DESTINATION, "s", destination),
            Self::Signature(signature) => (SIGNATURE, "g", signature),
        };
        out.byte(code);
        out.signature(signature)?;
        if matches!(self, Self::Signature(_)) {
            out.signature(value)
        } else {
            out.string(value)
        }
    }
}

/// Bytes in D-Bus's little-endian wire format, each value aligned to its size from the start.
#[derive(Debug, Default)]
struct Marshal {
    /// What is written so far.
    bytes: Vec<u8>,
}

impl Marshal {
    /// Pads with zeros to a multiple of `align` bytes.
    fn align(&mut self, align: usize) {
        let aligned = self.bytes.len().next_multiple_of(align);
        self.bytes.resize(aligned, 0);
    }

    /// A byte.
    fn byte(&mut self, value: u8) {
        self.bytes.push(value);
    }

    /// A `u32`, 4-aligned.
    fn u32(&mut self, value: u32) {
        self.align(4);
        self.bytes.extend(value.to_le_bytes());
    }

    /// An `i32`, 4-aligned.
    fn i32(&mut self, value: i32) {
        self.align(4);
        self.bytes.extend(value.to_le_bytes());
    }

    /// A string or object path: its length as a 4-aligned `u32`, its bytes, a NUL.
    ///
    /// # Errors
    ///
    /// [`NotifyError::Nul`] if it holds one, [`NotifyError::TooLong`] if its length doesn't fit
    /// a `u32`.
    fn string(&mut self, text: &str) -> Result<(), NotifyError> {
        if text.contains('\0') {
            return Err(NotifyError::Nul);
        }
        let Ok(length) = u32::try_from(text.len()) else {
            return Err(NotifyError::TooLong { length: text.len() });
        };
        self.u32(length);
        self.bytes.extend(text.as_bytes());
        self.bytes.push(0);
        Ok(())
    }

    /// A signature: its length as a byte, its bytes, a NUL.
    ///
    /// # Errors
    ///
    /// [`NotifyError::TooLong`] if it is longer than 255 bytes.
    fn signature(&mut self, text: &str) -> Result<(), NotifyError> {
        let Ok(length) = u8::try_from(text.len()) else {
            return Err(NotifyError::TooLong { length: text.len() });
        };
        self.byte(length);
        self.bytes.extend(text.as_bytes());
        self.bytes.push(0);
        Ok(())
    }

    /// An array of elements aligned to `align` and written by `elements`: its length in bytes
    /// (a 4-aligned `u32`, not counting the padding after it to the first element), the padding,
    /// the elements. An empty array still has the padding.
    ///
    /// # Errors
    ///
    /// As `elements`, and [`NotifyError::TooLong`] if the elements' length doesn't fit a `u32`.
    fn array(
        &mut self,
        align: usize,
        elements: impl FnOnce(&mut Self) -> Result<(), NotifyError>,
    ) -> Result<(), NotifyError> {
        self.u32(0);
        let length_at = self.bytes.len() - 4;
        self.align(align);
        let start = self.bytes.len();
        elements(self)?;
        let length = self.bytes.len() - start;
        let Ok(length) = u32::try_from(length) else {
            return Err(NotifyError::TooLong { length });
        };
        // The four bytes written as the length's placeholder above.
        let slot = self
            .bytes
            .iter_mut()
            .skip(length_at)
            .zip(length.to_le_bytes());
        for (byte, value) in slot {
            *byte = value;
        }
        Ok(())
    }
}

/// A whole message: the header (endianness, `kind`, no flags, protocol version 1, the body's
/// length, `serial`, then `fields`), padded to 8, then `body`, marshalled from an 8-aligned start.
///
/// # Errors
///
/// As [`Marshal::string`], for a field's text, and [`NotifyError::TooLong`] for a body whose
/// length doesn't fit a `u32`.
fn message(
    kind: Kind,
    serial: u32,
    fields: &[Field<'_>],
    body: &[u8],
) -> Result<Vec<u8>, NotifyError> {
    let Ok(body_length) = u32::try_from(body.len()) else {
        return Err(NotifyError::TooLong { length: body.len() });
    };
    let mut out = Marshal::default();
    out.byte(b'l');
    out.byte(kind as u8);
    out.byte(0);
    out.byte(1);
    out.u32(body_length);
    out.u32(serial);
    out.array(8, |out| {
        fields.iter().try_for_each(|field| field.write(out))
    })?;
    out.align(8);
    out.bytes.extend(body);
    Ok(out.bytes)
}

/// The `org.freedesktop.DBus.Hello` call, which a connection must make before any other.
///
/// # Errors
///
/// As [`message`].
fn hello_call() -> Result<Vec<u8>, NotifyError> {
    message(
        Kind::MethodCall,
        HELLO_SERIAL,
        &[
            Field::Path("/org/freedesktop/DBus"),
            Field::Interface("org.freedesktop.DBus"),
            Field::Destination("org.freedesktop.DBus"),
            Field::Member("Hello"),
        ],
        &[],
    )
}

/// The `org.freedesktop.Notifications.Notify` call, as `serial`, showing `summary` and `body`.
///
/// From [`APP_NAME`], replacing no notification, with no icon, no actions and no hints, expiring
/// when the server decides. The header fields are in the order `GDBusMessage` writes them, so
/// the bytes can be compared with an implementation of D-Bus that isn't ours.
///
/// # Errors
///
/// As [`Marshal::string`], for the summary or the body.
fn notify_call(serial: u32, summary: &str, body: &str) -> Result<Vec<u8>, NotifyError> {
    let mut args = Marshal::default();
    args.string(APP_NAME)?;
    args.u32(0);
    args.string("")?;
    args.string(summary)?;
    args.string(body)?;
    // The actions, `as`, and the hints, `a{sv}`, whose entries are 8-aligned.
    args.array(4, |_| Ok(()))?;
    args.array(8, |_| Ok(()))?;
    args.i32(-1);
    message(
        Kind::MethodCall,
        serial,
        &[
            Field::Path("/org/freedesktop/Notifications"),
            Field::Interface("org.freedesktop.Notifications"),
            Field::Destination("org.freedesktop.Notifications"),
            Field::Signature("susssasa{sv}i"),
            Field::Member("Notify"),
        ],
        &args.bytes,
    )
}

/// One message read whole from the bus, checked as far as its framing.
#[derive(Debug)]
struct Frame {
    /// The message's bytes.
    bytes: Vec<u8>,
    /// Where its header fields end, before the padding to the body.
    fields_end: usize,
}

impl Frame {
    /// Its type, if it is one this client knows.
    fn kind(&self) -> Option<Kind> {
        self.bytes.get(1).copied().and_then(Kind::from_byte)
    }

    /// Hands each header field to `each`, by its code and value, and returns the body.
    ///
    /// # Errors
    ///
    /// As [`Reader::variant`].
    fn fields(&self, mut each: impl FnMut(u8, Value)) -> Result<&[u8], NotifyError> {
        let mut reader = Reader::new(&self.bytes);
        reader.take(16)?;
        while reader.at() < self.fields_end {
            reader.align(8)?;
            let code = reader.byte()?;
            each(code, reader.variant()?);
        }
        reader.align(8)?;
        Ok(reader.rest())
    }
}

/// Reads one whole message: the fixed part of its header, then the rest its lengths give.
///
/// # Errors
///
/// [`NotifyError::Protocol`] if it is big-endian, of another protocol version than 1, or larger
/// than [`MAX_MESSAGE`], and as [`NotifyError::from`] an [`io::Error`].
fn read_frame(reader: &mut impl Read) -> Result<Frame, NotifyError> {
    let mut fixed = [0; 16];
    reader.read_exact(&mut fixed)?;
    let mut header = Reader::new(&fixed);
    let &[endianness, _kind, _flags, version] = header.take(4)? else {
        return Err(NotifyError::Protocol("a header cut short"));
    };
    if endianness != b'l' {
        return Err(NotifyError::Protocol("a message that isn't little-endian"));
    }
    if version != 1 {
        return Err(NotifyError::Protocol(
            "a message of another protocol version",
        ));
    }
    let body_length = header.u32()?;
    header.u32()?;
    let fields_length = header.u32()?;
    let too_large = NotifyError::Protocol("a message too large");
    if fields_length > MAX_MESSAGE || body_length > MAX_MESSAGE {
        return Err(too_large);
    }
    // Both bounded, so no sum overflows. The fields array starts right after its length, already
    // 8-aligned, and the body is 8-aligned.
    let fields_end = 16 + fields_length;
    let size = fields_end.next_multiple_of(8) + body_length;
    if size > MAX_MESSAGE {
        return Err(too_large);
    }
    let (Ok(fields_end), Ok(size)) = (usize::try_from(fields_end), usize::try_from(size)) else {
        return Err(too_large);
    };
    let mut bytes = fixed.to_vec();
    bytes.resize(size, 0);
    reader.read_exact(bytes.get_mut(fixed.len()..).unwrap_or_default())?;
    Ok(Frame { bytes, fields_end })
}

/// A reply from the bus, as far as the client reads it.
#[derive(Debug)]
struct Reply {
    /// Its type, if it is one this client knows.
    kind: Option<Kind>,
    /// An error's name.
    error_name: Option<String>,
    /// The serial of the call it answers.
    answers: Option<u32>,
    /// Its body's signature.
    signature: Option<String>,
    /// Its body.
    body: Vec<u8>,
}

impl Reply {
    /// The error an error reply stands for: its name, and its text when the body starts with
    /// one.
    fn refusal(&self) -> NotifyError {
        let starts_with_text = self
            .signature
            .as_deref()
            .is_some_and(|signature| signature.starts_with('s'));
        let message = starts_with_text
            .then(|| Reader::new(&self.body).string().ok())
            .flatten()
            .unwrap_or_default();
        NotifyError::Refused {
            name: self.error_name.clone().unwrap_or_default(),
            message,
        }
    }
}

/// Reads one message from the bus, as a [`Reply`].
///
/// # Errors
///
/// As [`read_frame`] and [`Frame::fields`].
fn read_reply(reader: &mut impl Read) -> Result<Reply, NotifyError> {
    let frame = read_frame(reader)?;
    let mut reply = Reply {
        kind: frame.kind(),
        error_name: None,
        answers: None,
        signature: None,
        body: Vec::new(),
    };
    let body = frame.fields(|code, value| match (code, value) {
        (ERROR_NAME, Value::Text(name)) => reply.error_name = Some(name),
        (REPLY_SERIAL, Value::U32(serial)) => reply.answers = Some(serial),
        (SIGNATURE, Value::Text(signature)) => reply.signature = Some(signature),
        // The sender, the destination, and fields the client doesn't need.
        _ => {}
    })?;
    reply.body = body.to_vec();
    Ok(reply)
}

/// A header field's value, of the types header fields have.
#[derive(Debug)]
enum Value {
    /// A string, object path or signature.
    Text(String),
    /// A `u32`.
    U32(u32),
}

/// Reads D-Bus's little-endian wire format from bytes that begin 8-aligned.
#[derive(Debug)]
struct Reader<'a> {
    /// The bytes.
    bytes: &'a [u8],
    /// Where the next value is read.
    at: usize,
}

impl<'a> Reader<'a> {
    /// A reader at the start of `bytes`.
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, at: 0 }
    }

    /// Where the next value is read.
    const fn at(&self) -> usize {
        self.at
    }

    /// What is left.
    fn rest(&self) -> &'a [u8] {
        self.bytes.get(self.at..).unwrap_or_default()
    }

    /// The next `count` bytes.
    ///
    /// # Errors
    ///
    /// [`NotifyError::Protocol`] if there aren't that many.
    fn take(&mut self, count: usize) -> Result<&'a [u8], NotifyError> {
        let end = self
            .at
            .checked_add(count)
            .ok_or(NotifyError::Protocol("a length past the end"))?;
        let taken = self
            .bytes
            .get(self.at..end)
            .ok_or(NotifyError::Protocol("a message cut short"))?;
        self.at = end;
        Ok(taken)
    }

    /// Skips the padding to a multiple of `align`.
    ///
    /// # Errors
    ///
    /// As [`Reader::take`].
    fn align(&mut self, align: usize) -> Result<(), NotifyError> {
        let padding = self.at.next_multiple_of(align) - self.at;
        self.take(padding)?;
        Ok(())
    }

    /// A byte.
    ///
    /// # Errors
    ///
    /// As [`Reader::take`].
    fn byte(&mut self) -> Result<u8, NotifyError> {
        let &[byte] = self.take(1)? else {
            return Err(NotifyError::Protocol("a byte that is two"));
        };
        Ok(byte)
    }

    /// A 4-aligned `u32`.
    ///
    /// # Errors
    ///
    /// As [`Reader::take`].
    fn u32(&mut self) -> Result<u32, NotifyError> {
        self.align(4)?;
        let &[a, b, c, d] = self.take(4)? else {
            return Err(NotifyError::Protocol("a u32 that isn't four bytes"));
        };
        Ok(u32::from_le_bytes([a, b, c, d]))
    }

    /// A string or object path: a `u32` length, the bytes, a NUL.
    ///
    /// # Errors
    ///
    /// [`NotifyError::Protocol`] if it is cut short, isn't UTF-8 or isn't ended by a NUL.
    fn string(&mut self) -> Result<String, NotifyError> {
        let Ok(length) = usize::try_from(self.u32()?) else {
            return Err(NotifyError::Protocol("a string longer than memory"));
        };
        self.text(length)
    }

    /// A signature: a byte's length, the bytes, a NUL.
    ///
    /// # Errors
    ///
    /// As [`Reader::string`].
    fn signature(&mut self) -> Result<String, NotifyError> {
        let length = usize::from(self.byte()?);
        self.text(length)
    }

    /// `length` bytes of UTF-8 and the NUL after them.
    ///
    /// # Errors
    ///
    /// As [`Reader::string`].
    fn text(&mut self, length: usize) -> Result<String, NotifyError> {
        let bytes = self.take(length)?;
        if self.byte()? != 0 {
            return Err(NotifyError::Protocol("a string not ended by a NUL"));
        }
        let text = std::str::from_utf8(bytes).ok();
        let text = text.ok_or(NotifyError::Protocol("a string that isn't UTF-8"))?;
        Ok(text.to_owned())
    }

    /// A header field's variant: its signature, then a value of the types header fields have.
    ///
    /// # Errors
    ///
    /// [`NotifyError::Protocol`] for another type, and as [`Reader::take`].
    fn variant(&mut self) -> Result<Value, NotifyError> {
        match self.signature()?.as_str() {
            "s" | "o" => Ok(Value::Text(self.string()?)),
            "g" => Ok(Value::Text(self.signature()?)),
            "u" => Ok(Value::U32(self.u32()?)),
            _ => Err(NotifyError::Protocol(
                "a header field of an unexpected type",
            )),
        }
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;
