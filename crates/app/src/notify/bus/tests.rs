//! The client against a fake bus: a Unix socket in a test directory speaking the bus's side of
//! the handshake, so no test touches the real session bus (but the ignored one, which does).

use std::io::{BufRead, BufReader};
use std::os::linux::net::SocketAddrExt as _;
use std::os::unix::net::{SocketAddr, UnixListener};
use std::thread::{self, JoinHandle};

use super::*;
use crate::notify::{TIMEOUT, send};
use crate::test_dir::TestDir;

/// The uid the tests authenticate as: its decimal digits are what EXTERNAL sends, in hex.
const UID: u32 = 1000;

/// What the fake bus saw of a client.
#[derive(Debug, Default)]
struct Seen {
    /// The first byte the client sent: the credentials byte, a NUL.
    first: Option<u8>,
    /// The AUTH line, without its CRLF.
    auth: String,
    /// The line after OK, without its CRLF.
    begin: String,
    /// The messages after BEGIN, in order.
    calls: Vec<Call>,
}

/// How the fake bus answers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Answer {
    /// As a bus does: OK, Hello's reply, a signal, then Notify's reply.
    Notified,
    /// OK and Hello's reply, then an error for Notify: nobody owns the name.
    NoServer,
    /// REJECTED to the AUTH line.
    Reject,
    /// Nothing at all, holding the connection open until the client goes.
    Silent,
    /// Nothing, hanging up after the AUTH line.
    HangUp,
}

/// A fake bus listening at a path in a test directory, serving one client.
#[derive(Debug)]
struct FakeBus {
    /// Holds the socket's directory.
    _dir: TestDir,
    /// The bus's address, as `DBUS_SESSION_BUS_ADDRESS` would give it.
    address: String,
    /// The bus's thread, which ends with what it saw once the client has gone.
    server: JoinHandle<Seen>,
}

impl FakeBus {
    /// Listens at a fresh path and serves one client in a thread, answering so.
    fn new(answer: Answer) -> Self {
        let dir = TestDir::new();
        let path: PathBuf = dir.path().join("bus");
        let listener = UnixListener::bind(&path).unwrap();
        let address = format!("unix:path={},guid=0123456789abcdef", path.display());
        let server = thread::spawn(move || serve(&listener, answer));
        Self {
            _dir: dir,
            address,
            server,
        }
    }

    /// What the bus saw, once its client has gone.
    fn seen(self) -> Seen {
        self.server.join().unwrap()
    }
}

/// Serves one client from `listener`, answering as `answer` says, and returns what it saw.
fn serve(listener: &UnixListener, answer: Answer) -> Seen {
    let (stream, _) = listener.accept().unwrap();
    // A client that never hangs up fails the test rather than holding it.
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .unwrap();
    let mut reader = BufReader::new(&stream);
    let mut writer = &stream;
    let mut seen = Seen::default();
    let mut first = [0; 1];
    reader.read_exact(&mut first).unwrap();
    seen.first = Some(first[0]);
    seen.auth = line(&mut reader);
    match answer {
        Answer::Reject => {
            writer.write_all(b"REJECTED EXTERNAL\r\n").unwrap();
            // Wait for the client to hang up.
            drop(reader.read_to_end(&mut Vec::new()));
            return seen;
        }
        Answer::Silent => {
            drop(reader.read_to_end(&mut Vec::new()));
            return seen;
        }
        Answer::HangUp => return seen,
        Answer::Notified | Answer::NoServer => {}
    }
    writer
        .write_all(b"OK 0123456789abcdef0123456789abcdef\r\n")
        .unwrap();
    seen.begin = line(&mut reader);
    let hello = read_call(&mut reader);
    let mut name = Marshal::default();
    name.string(":1.42").unwrap();
    let hello_reply = reply(
        2,
        1,
        &[
            (REPLY_SERIAL, Header::U32(hello.serial)),
            (DESTINATION, Header::Text("s", ":1.42")),
            (SIGNATURE, Header::Text("g", "s")),
        ],
        &name.bytes,
    );
    writer.write_all(&hello_reply).unwrap();
    // The bus tells a new connection its name in a signal before anything else.
    let acquired = reply(
        4,
        2,
        &[
            (PATH, Header::Text("o", "/org/freedesktop/DBus")),
            (INTERFACE, Header::Text("s", "org.freedesktop.DBus")),
            (MEMBER, Header::Text("s", "NameAcquired")),
            (DESTINATION, Header::Text("s", ":1.42")),
            (SIGNATURE, Header::Text("g", "s")),
        ],
        &name.bytes,
    );
    writer.write_all(&acquired).unwrap();
    let notify = read_call(&mut reader);
    let answer = if answer == Answer::Notified {
        let mut id = Marshal::default();
        id.u32(7);
        reply(
            2,
            3,
            &[
                (REPLY_SERIAL, Header::U32(notify.serial)),
                (SIGNATURE, Header::Text("g", "u")),
            ],
            &id.bytes,
        )
    } else {
        let mut text = Marshal::default();
        text.string("The name is not activatable").unwrap();
        reply(
            3,
            3,
            &[
                (
                    ERROR_NAME,
                    Header::Text("s", "org.freedesktop.DBus.Error.ServiceUnknown"),
                ),
                (REPLY_SERIAL, Header::U32(notify.serial)),
                (SIGNATURE, Header::Text("g", "s")),
            ],
            &text.bytes,
        )
    };
    writer.write_all(&answer).unwrap();
    seen.calls = vec![hello, notify];
    drop(reader.read_to_end(&mut Vec::new()));
    seen
}

/// One CRLF-ended line from the client, without its CRLF.
fn line(reader: &mut impl BufRead) -> String {
    let mut text = String::new();
    reader.read_line(&mut text).unwrap();
    text.strip_suffix("\r\n").unwrap().to_owned()
}

/// A header field's value, as the fake bus writes it.
#[derive(Debug, Clone, Copy)]
enum Header<'a> {
    /// A string, object path or signature: its variant's signature (`s`, `o` or `g`), and it.
    Text(&'a str, &'a str),
    /// A `u32`.
    U32(u32),
}

/// A message from the fake bus: of type `kind` (2 a reply, 3 an error, 4 a signal), as
/// `serial`, with these header fields and this body.
fn reply(kind: u8, serial: u32, fields: &[(u8, Header<'_>)], body: &[u8]) -> Vec<u8> {
    let mut out = Marshal::default();
    out.bytes.extend([b'l', kind, 0, 1]);
    out.u32(u32::try_from(body.len()).unwrap());
    out.u32(serial);
    out.array(8, |out| {
        for &(code, value) in fields {
            out.align(8);
            out.byte(code);
            match value {
                Header::Text("g", signature) => {
                    out.signature("g")?;
                    out.signature(signature)?;
                }
                Header::Text(variant, text) => {
                    out.signature(variant)?;
                    out.string(text)?;
                }
                Header::U32(value) => {
                    out.signature("u")?;
                    out.u32(value);
                }
            }
        }
        Ok(())
    })
    .unwrap();
    out.align(8);
    out.bytes.extend(body);
    out.bytes
}

/// A call from the client, as the fake bus reads it.
#[derive(Debug)]
struct Call {
    /// Its type.
    kind: Option<Kind>,
    /// Its serial.
    serial: u32,
    /// The object called.
    path: Option<String>,
    /// The method's interface.
    interface: Option<String>,
    /// The method.
    member: Option<String>,
    /// The connection called.
    destination: Option<String>,
    /// The body's signature.
    signature: Option<String>,
    /// The body.
    body: Vec<u8>,
}

/// Reads one call from the client, every header field.
fn read_call(reader: &mut impl Read) -> Call {
    let frame = read_frame(reader).unwrap();
    let serial = u32::from_le_bytes(frame.bytes[8..12].try_into().unwrap());
    let mut call = Call {
        kind: frame.kind(),
        serial,
        path: None,
        interface: None,
        member: None,
        destination: None,
        signature: None,
        body: Vec::new(),
    };
    let body = frame
        .fields(|code, value| {
            let Value::Text(text) = value else {
                panic!("field {code} isn't text");
            };
            match code {
                PATH => call.path = Some(text),
                INTERFACE => call.interface = Some(text),
                MEMBER => call.member = Some(text),
                DESTINATION => call.destination = Some(text),
                SIGNATURE => call.signature = Some(text),
                _ => panic!("a field a call doesn't carry: {code}"),
            }
        })
        .unwrap();
    call.body = body.to_vec();
    call
}

/// The arguments of a `Notify` call's body, in order: everything but the empty array and dict.
#[derive(Debug, PartialEq, Eq)]
struct NotifyArgs {
    /// The app's name.
    app_name: String,
    /// The notification it replaces.
    replaces_id: u32,
    /// The icon.
    app_icon: String,
    /// The summary.
    summary: String,
    /// The body.
    body: String,
    /// How many actions (the array's length in bytes).
    actions: u32,
    /// How many hints (the dict's length in bytes).
    hints: u32,
    /// When it expires.
    expire_timeout: i32,
}

/// Reads a `Notify` call's body, checking that the dict's padding is there and nothing follows.
fn notify_args(body: &[u8]) -> NotifyArgs {
    let mut reader = Reader::new(body);
    let app_name = reader.string().unwrap();
    let replaces_id = reader.u32().unwrap();
    let app_icon = reader.string().unwrap();
    let summary = reader.string().unwrap();
    let text = reader.string().unwrap();
    let actions = reader.u32().unwrap();
    let hints = reader.u32().unwrap();
    // A dict's entries are 8-aligned: the padding follows the length even when there are none.
    reader.align(8).unwrap();
    let expire_timeout = i32::from_le_bytes(reader.take(4).unwrap().try_into().unwrap());
    assert!(reader.rest().is_empty(), "nothing follows the timeout");
    NotifyArgs {
        app_name,
        replaces_id,
        app_icon,
        summary,
        body: text,
        actions,
        hints,
        expire_timeout,
    }
}

#[test]
fn it_authenticates_with_external_and_calls_notify() {
    let bus = FakeBus::new(Answer::Notified);
    let sent = send_to(
        Some(&bus.address),
        UID,
        "BrushCraft couldn't save first-sculpt",
        "The disk is full.",
        TIMEOUT,
    );
    let seen = bus.seen();
    sent.unwrap();
    assert_eq!(seen.first, Some(0), "the credentials byte");
    assert_eq!(
        seen.auth, "AUTH EXTERNAL 31303030",
        "the uid's digits, in hex"
    );
    assert_eq!(seen.begin, "BEGIN", "after OK");
    let [hello, notify] = seen.calls.as_slice() else {
        panic!("two calls: {:?}", seen.calls);
    };
    assert_eq!(hello.kind, Some(Kind::MethodCall), "Hello is a call");
    assert_eq!(hello.serial, HELLO_SERIAL, "the first message");
    assert_eq!(hello.member.as_deref(), Some("Hello"), "the first call");
    assert_eq!(
        hello.interface.as_deref(),
        Some("org.freedesktop.DBus"),
        "Hello's"
    );
    assert_eq!(
        hello.path.as_deref(),
        Some("/org/freedesktop/DBus"),
        "Hello's"
    );
    assert_eq!(
        hello.destination.as_deref(),
        Some("org.freedesktop.DBus"),
        "the bus"
    );
    assert_eq!(hello.signature, None, "Hello takes nothing");
    assert!(hello.body.is_empty(), "Hello takes nothing");
    assert_eq!(notify.kind, Some(Kind::MethodCall), "Notify is a call");
    assert_eq!(notify.serial, NOTIFY_SERIAL, "the second message");
    assert_eq!(notify.member.as_deref(), Some("Notify"), "the second call");
    assert_eq!(
        notify.interface.as_deref(),
        Some("org.freedesktop.Notifications"),
        "the notification server's interface"
    );
    assert_eq!(
        notify.path.as_deref(),
        Some("/org/freedesktop/Notifications"),
        "its object"
    );
    assert_eq!(
        notify.destination.as_deref(),
        Some("org.freedesktop.Notifications"),
        "its name"
    );
    assert_eq!(
        notify.signature.as_deref(),
        Some("susssasa{sv}i"),
        "Notify's signature"
    );
    assert_eq!(
        notify_args(&notify.body),
        NotifyArgs {
            app_name: "BrushCraft".to_owned(),
            replaces_id: 0,
            app_icon: String::new(),
            summary: "BrushCraft couldn't save first-sculpt".to_owned(),
            body: "The disk is full.".to_owned(),
            actions: 0,
            hints: 0,
            expire_timeout: -1,
        },
        "the arguments"
    );
}

#[test]
fn no_bus_address_sends_nothing_and_says_so() {
    let unset = send_to(None, UID, "summary", "body", TIMEOUT).unwrap_err();
    assert!(matches!(unset, NotifyError::NoBus), "unset: {unset:?}");
    let empty = send_to(Some(""), UID, "summary", "body", TIMEOUT).unwrap_err();
    assert!(matches!(empty, NotifyError::NoBus), "empty: {empty:?}");
    assert!(
        unset.to_string().contains("no session bus"),
        "says so: {unset}"
    );
    let tcp = send_to(Some("tcp:host=localhost,port=1"), UID, "s", "b", TIMEOUT).unwrap_err();
    assert!(
        matches!(tcp, NotifyError::NoUsableAddress(_)),
        "only Unix sockets: {tcp:?}"
    );
}

#[test]
fn a_bus_that_refuses_auth_is_an_error_not_a_hang() {
    let rejecting = FakeBus::new(Answer::Reject);
    let asked = Instant::now();
    let rejected = send_to(Some(&rejecting.address), UID, "s", "b", TIMEOUT).unwrap_err();
    assert!(asked.elapsed() < TIMEOUT, "at once: {:?}", asked.elapsed());
    assert!(
        matches!(&rejected, NotifyError::AuthRejected(mechanisms) if mechanisms == "EXTERNAL"),
        "rejected: {rejected:?}"
    );
    drop(rejecting.seen());

    let silent = FakeBus::new(Answer::Silent);
    let waiting = Instant::now();
    let unanswered = send_to(Some(&silent.address), UID, "s", "b", TIMEOUT).unwrap_err();
    let waited = waiting.elapsed();
    assert!(
        (TIMEOUT..TIMEOUT + Duration::from_millis(500)).contains(&waited),
        "waits the timeout, no longer: {waited:?}"
    );
    assert!(
        matches!(unanswered, NotifyError::TimedOut),
        "timed out: {unanswered:?}"
    );
    drop(silent.seen());
}

#[test]
fn a_bus_that_hangs_up_in_the_handshake_has_closed() {
    let bus = FakeBus::new(Answer::HangUp);
    let closed = send_to(Some(&bus.address), UID, "s", "b", TIMEOUT).unwrap_err();
    drop(bus.seen());
    assert!(matches!(closed, NotifyError::Closed), "closed: {closed:?}");
}

#[test]
fn a_bus_that_takes_no_connections_times_out_connecting() {
    let dir = TestDir::new();
    let path: PathBuf = dir.path().join("full");
    let listener = rustix::net::socket(AddressFamily::UNIX, SocketType::STREAM, None).unwrap();
    rustix::net::bind(&listener, &SocketAddrUnix::new(path.as_path()).unwrap()).unwrap();
    // A backlog of none takes one connection that is never accepted; a second waits.
    rustix::net::listen(&listener, 0).unwrap();
    let _queued = UnixStream::connect(&path).unwrap();
    // The connect itself, not the handshake after it, is what runs out of time.
    let timeout = Duration::from_millis(300);
    let connecting = Instant::now();
    let unconnected = Address::Path(path)
        .connect(Deadline(connecting + timeout))
        .unwrap_err();
    let waited = connecting.elapsed();
    assert!(
        matches!(unconnected, NotifyError::TimedOut),
        "timed out: {unconnected:?}"
    );
    assert!(
        (timeout..timeout + Duration::from_millis(500)).contains(&waited),
        "waits the timeout, no longer: {waited:?}"
    );
}

#[test]
fn an_error_reply_says_who_refused_and_why() {
    let bus = FakeBus::new(Answer::NoServer);
    let refused = send_to(Some(&bus.address), UID, "s", "b", TIMEOUT).unwrap_err();
    drop(bus.seen());
    assert!(
        matches!(
            &refused,
            NotifyError::Refused { name, message }
                if name == "org.freedesktop.DBus.Error.ServiceUnknown"
                    && message == "The name is not activatable"
        ),
        "refused: {refused:?}"
    );
}

#[test]
fn an_abstract_socket_is_reached_too() {
    let name = format!("brushcraft-notify-test-{}", std::process::id());
    let listener =
        UnixListener::bind_addr(&SocketAddr::from_abstract_name(name.as_bytes()).unwrap()).unwrap();
    let server = thread::spawn(move || serve(&listener, Answer::Notified));
    // The first address isn't a Unix socket, and the name comes escaped.
    let escaped = name.replace('-', "%2d");
    let address = format!("tcp:host=localhost,port=1;unix:abstract={escaped},guid=00");
    send_to(Some(&address), UID, "s", "b", TIMEOUT).unwrap();
    assert_eq!(server.join().unwrap().calls.len(), 2, "Hello and Notify");
}

#[test]
fn addresses_parse_as_the_specification_writes_them() {
    assert_eq!(
        parse_address("unix:path=/run/user/1000/bus").unwrap(),
        Address::Path(PathBuf::from("/run/user/1000/bus")),
        "the usual one"
    );
    assert_eq!(
        parse_address("unix:guid=00,path=/tmp/a%20b%2Cc").unwrap(),
        Address::Path(PathBuf::from("/tmp/a b,c")),
        "escaped, after another key"
    );
    assert_eq!(
        parse_address("unix:abstract=/tmp/dbus-XYZ,guid=00").unwrap(),
        Address::Abstract(b"/tmp/dbus-XYZ".to_vec()),
        "abstract"
    );
    assert_eq!(
        parse_address("unixexec:path=/bin/true;unix:path=/b").unwrap(),
        Address::Path(PathBuf::from("/b")),
        "the first usable one"
    );
    let malformed = [
        "unix:path=/a%2",
        "unix:path=/a%zz",
        "unix:runtime=yes",
        "path=/a",
        "",
    ];
    for address in malformed {
        assert!(
            matches!(parse_address(address), Err(NotifyError::NoUsableAddress(_))),
            "{address:?} isn't usable"
        );
    }
}

/// One `Notify` call, summary "Saved" and body "Done", as `GDBusMessage` marshals it.
///
/// An implementation independent of ours, through `new_method_call`, `set_body`,
/// `set_serial(2)`, little-endian, and `to_blob`. A line per header field or argument, each
/// then its padding.
const NOTIFY_GOLDEN: &str = "\
    6c 01 00 01  44000000  02000000  9f000000\
    01 01 6f 00  1e000000  2f6f72672f667265656465736b746f702f4e6f74696669636174696f6e7300  00\
    02 01 73 00  1d000000  6f72672e667265656465736b746f702e4e6f74696669636174696f6e7300  0000\
    06 01 73 00  1d000000  6f72672e667265656465736b746f702e4e6f74696669636174696f6e7300  0000\
    08 01 67 00  0d 73757373736173617b73767d69 00  0000000000\
    03 01 73 00  06000000  4e6f7469667900  00\
    0a000000  4272757368437261667400  00\
    00000000\
    00000000 00  000000\
    05000000  536176656400  0000\
    04000000  446f6e6500  000000\
    00000000\
    00000000  00000000\
    ffffffff";

#[test]
fn the_message_is_valid_dbus_wire_format() {
    let golden: Vec<u8> = hex(NOTIFY_GOLDEN);
    assert_eq!(
        notify_call(2, "Saved", "Done").unwrap(),
        golden,
        "byte for byte"
    );
    // And it reads back.
    let parsed = read_call(&mut golden.as_slice());
    assert_eq!(parsed.serial, 2, "its serial");
    assert_eq!(parsed.member.as_deref(), Some("Notify"), "its member");
    assert_eq!(notify_args(&parsed.body).summary, "Saved", "its summary");
}

#[test]
fn messages_out_of_the_format_are_refused() {
    let golden: Vec<u8> = hex(NOTIFY_GOLDEN);
    let mut big_endian = golden.clone();
    big_endian[0] = b'B';
    let mut version_2 = golden.clone();
    version_2[3] = 2;
    let mut too_large = golden.clone();
    too_large[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
    for (message, what) in [
        (big_endian, "big-endian"),
        (version_2, "version 2"),
        (too_large, "a body of 4 GiB"),
    ] {
        let refused = read_frame(&mut message.as_slice()).unwrap_err();
        assert!(
            matches!(refused, NotifyError::Protocol(_)),
            "{what}: {refused:?}"
        );
    }
    // The path's NUL, replaced.
    let mut unended = golden;
    unended[16 + 8 + 30] = b'x';
    let frame = read_frame(&mut unended.as_slice()).unwrap();
    let refused = frame.fields(|_, _| {}).unwrap_err();
    assert!(
        matches!(
            refused,
            NotifyError::Protocol("a string not ended by a NUL")
        ),
        "a string without its NUL: {refused:?}"
    );
}

#[test]
fn text_d_bus_cannot_carry_is_refused_before_connecting() {
    let nul = send_to(Some("unix:path=/nonexistent"), UID, "a\0b", "body", TIMEOUT).unwrap_err();
    assert!(matches!(nul, NotifyError::Nul), "NUL: {nul:?}");
}

/// Bytes from hex digits, ignoring whitespace.
fn hex(text: &str) -> Vec<u8> {
    let digits: Vec<u8> = text
        .bytes()
        .filter(|byte| !byte.is_ascii_whitespace())
        .collect();
    let bytes = digits
        .chunks(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap());
    bytes.collect()
}

/// Sends one notification to the real session bus, which shows on the desktop. Run by hand:
/// `cargo nextest run -p app --run-ignored only a_real_notification_shows`.
#[test]
#[ignore = "sends a notification to the desktop"]
fn a_real_notification_shows() {
    send("BrushCraft test", "Notification check from development").unwrap();
}
