//! The takeover's socket and the opener's way to a locked project, with no window and no GPU.
//!
//! Every socket lives in a test directory, never the real `$XDG_RUNTIME_DIR/brushcraft`.

#![expect(clippy::missing_panics_doc, reason = "tests panic to fail")]

use std::io::{Read as _, Write as _};
use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::thread;

use super::socket::{MAX_LINE, our_socket, our_uid, ours_alone};
use super::*;
use crate::test_dir::TestDir;

/// A short patience for the tests: their finishers answer at once, or never.
const QUICK: Patience = Patience {
    answer: Duration::from_millis(300),
    release: Duration::from_millis(300),
};

/// A file in `dir` called `name`, standing for a project file, and its identity.
fn file_in(dir: &TestDir, name: &str) -> (PathBuf, FileIdentity) {
    let path = dir.path().join(name);
    std::fs::write(&path, b"a project").expect("written");
    let identity = FileIdentity::of(&path).expect("its identity");
    (path, identity)
}

/// The sockets' directory in `dir`, not yet made.
fn sockets_in(dir: &TestDir) -> SocketDir {
    SocketDir::at(dir.path().join("brushcraft"))
}

/// Where the finisher of `file` would listen in `dir`'s sockets' directory, as a listener bound
/// there says.
fn socket_path(sockets: &SocketDir, file: FileIdentity) -> PathBuf {
    let listener = Listener::bind(sockets, file).expect("bound");
    listener.path().to_owned()
}

/// Asks `listener` until it has been asked to stop, for a second at most: whether it was.
fn heard(listener: &mut Listener) -> bool {
    let began = Instant::now();
    while began.elapsed() < Duration::from_secs(1) {
        if listener.stop_asked() {
            return true;
        }
        thread::sleep(Duration::from_millis(1));
    }
    false
}

#[test]
fn a_finisher_asked_to_stop_says_stopped_once_it_answers() {
    let dir = TestDir::new();
    let (_path, file) = file_in(&dir, "first-sculpt.bcrf");
    let sockets = sockets_in(&dir);
    let mut listener = Listener::bind(&sockets, file).expect("bound");
    let socket = listener.path().to_owned();
    let asking = thread::spawn(move || ask_to_stop(&sockets, file, Duration::from_secs(5)));
    assert!(heard(&mut listener), "the request arrives");
    listener.answer();
    assert_eq!(asking.join().expect("asked"), Answer::Stopped);
    assert!(!socket.exists(), "the socket goes with the answer");
}

#[test]
fn the_directory_is_made_ours_alone() {
    let dir = TestDir::new();
    let (_path, file) = file_in(&dir, "first-sculpt.bcrf");
    let sockets = sockets_in(&dir);
    let listener = Listener::bind(&sockets, file).expect("bound");
    let made = std::fs::symlink_metadata(dir.path().join("brushcraft")).expect("made");
    assert!(made.is_dir());
    assert_eq!(made.mode() & 0o777, 0o700, "{:o}", made.mode());
    drop(listener);
}

#[test]
fn a_finisher_removes_its_socket_however_it_goes() {
    let dir = TestDir::new();
    let (_path, file) = file_in(&dir, "first-sculpt.bcrf");
    let sockets = sockets_in(&dir);
    let socket = socket_path(&sockets, file);
    assert!(!socket.exists(), "dropped, it is removed");
    let unwound = thread::spawn(move || {
        let _listener = Listener::bind(&sockets, file).expect("bound");
        panic!("a finisher's panic");
    });
    assert!(unwound.join().is_err(), "it panicked");
    assert!(!socket.exists(), "unwinding removes it too");
}

#[test]
fn a_finisher_never_removes_a_socket_put_there_since() {
    let dir = TestDir::new();
    let (_path, file) = file_in(&dir, "first-sculpt.bcrf");
    let sockets = sockets_in(&dir);
    let listener = Listener::bind(&sockets, file).expect("bound");
    let socket = listener.path().to_owned();
    std::fs::remove_file(&socket).expect("removed");
    let other = UnixListener::bind(&socket).expect("another socket there");
    drop(listener);
    assert!(socket.exists(), "not its own: left alone");
    drop(other);
}

#[test]
fn a_stale_socket_is_no_finisher_and_the_next_finisher_replaces_it() {
    let dir = TestDir::new();
    let (_path, file) = file_in(&dir, "first-sculpt.bcrf");
    let sockets = sockets_in(&dir);
    let socket = socket_path(&sockets, file);
    // A finisher killed: its socket file stays, and nothing listens.
    drop(UnixListener::bind(&socket).expect("bound"));
    assert!(socket.exists(), "left behind");
    let began = Instant::now();
    let answer = ask_to_stop(&sockets, file, Duration::from_secs(5));
    assert_eq!(answer, Answer::NoFinisher);
    assert!(
        began.elapsed() < Duration::from_secs(1),
        "at once, not after the patience: {:?}",
        began.elapsed()
    );
    assert!(
        socket.exists(),
        "the opener leaves it: a finisher may rename over it any moment"
    );

    // A finisher binding renames its own over it, in one step.
    let stale = std::fs::symlink_metadata(&socket).expect("stale").ino();
    let mut listener = Listener::bind(&sockets, file).expect("over a stale socket");
    let placed = std::fs::symlink_metadata(&socket).expect("placed").ino();
    assert_ne!(placed, stale, "replaced");
    let asking = thread::spawn(move || ask_to_stop(&sockets, file, Duration::from_secs(5)));
    assert!(heard(&mut listener), "and heard");
    listener.answer();
    assert_eq!(asking.join().expect("asked"), Answer::Stopped);
}

#[test]
fn a_staged_socket_a_killed_finisher_left_is_swept() {
    let dir = TestDir::new();
    let (_path, file) = file_in(&dir, "first-sculpt.bcrf");
    let (_other_path, other) = file_in(&dir, "other.bcrf");
    let sockets = sockets_in(&dir);
    let socket = socket_path(&sockets, file);
    // Killed between its bind and its rename: its staged socket stays, nothing listening.
    let staged = socket.with_file_name(format!(
        "{}.4242.tmp",
        socket.file_name().expect("a name").to_string_lossy()
    ));
    drop(UnixListener::bind(&staged).expect("bound"));
    // One still binding, listening: not ours to sweep.
    let binding = socket.with_file_name(format!(
        "{}.4243.tmp",
        socket.file_name().expect("a name").to_string_lossy()
    ));
    let live = UnixListener::bind(&binding).expect("bound");
    // Another file's: not this finisher's to look at.
    let theirs = socket_path(&sockets, other);
    let theirs = theirs.with_file_name(format!(
        "{}.4244.tmp",
        theirs.file_name().expect("a name").to_string_lossy()
    ));
    drop(UnixListener::bind(&theirs).expect("bound"));
    let listener = Listener::bind(&sockets, file).expect("bound");
    assert!(!staged.exists(), "swept");
    assert!(binding.exists(), "a live one stays");
    assert!(theirs.exists(), "another file's stays");
    drop((listener, live));
}

#[test]
fn a_live_finisher_s_socket_is_never_taken() {
    let dir = TestDir::new();
    let (_path, file) = file_in(&dir, "first-sculpt.bcrf");
    let sockets = sockets_in(&dir);
    let first = Listener::bind(&sockets, file).expect("bound");
    let second = Listener::bind(&sockets, file);
    assert!(
        matches!(second, Err(socket::SocketError::InUse(_))),
        "{second:?}"
    );
    assert!(first.path().exists(), "the first still listens");
}

#[test]
fn a_socket_of_another_user_or_another_file_is_never_used() {
    let dir = TestDir::new();
    let sockets = sockets_in(&dir);
    // Another file: a copy of the project, which has its project id, gets a socket of its own.
    let (original, file) = file_in(&dir, "first-sculpt.bcrf");
    let copy = dir.path().join("copy.bcrf");
    std::fs::copy(&original, &copy).expect("copied");
    let copied = FileIdentity::of(&copy).expect("its identity");
    assert_ne!(file, copied, "two files");
    let mut copys_finisher = Listener::bind(&sockets, copied).expect("bound");
    let answer = ask_to_stop(&sockets, file, QUICK.answer);
    assert_eq!(answer, Answer::NoFinisher, "the original has none");
    assert!(
        !copys_finisher.stop_asked(),
        "the copy's finisher wasn't asked"
    );
    let mut finisher = Listener::bind(&sockets, file).expect("both listen at once");
    let asking = thread::spawn(move || ask_to_stop(&sockets, file, Duration::from_secs(5)));
    assert!(heard(&mut finisher), "the original's is asked");
    finisher.answer();
    assert_eq!(asking.join().expect("asked"), Answer::Stopped);
    assert!(!copys_finisher.stop_asked(), "and the copy's still isn't");

    // Another user: what isn't ours is never trusted. (A test can't make another user's file;
    // the checks it would meet are these.)
    let made = std::fs::symlink_metadata(copys_finisher.path()).expect("a socket");
    assert!(our_socket(&made, our_uid()));
    assert!(!our_socket(&made, our_uid() + 1), "another user's socket");
    let ours = std::fs::symlink_metadata(dir.path().join("brushcraft")).expect("a directory");
    assert!(ours_alone(&ours, our_uid()));
    assert!(
        !ours_alone(&ours, our_uid() + 1),
        "another user's directory"
    );
}

#[test]
fn a_directory_open_to_others_is_never_used() {
    let dir = TestDir::new();
    let (_path, file) = file_in(&dir, "first-sculpt.bcrf");
    let sockets = sockets_in(&dir);
    let socket = socket_path(&sockets, file);
    let shared = dir.path().join("brushcraft");
    let open = std::fs::Permissions::from_mode(0o755);
    std::fs::set_permissions(&shared, open).expect("opened up");
    // Someone's socket file there, nothing listening: not removed, as nothing there is trusted.
    drop(UnixListener::bind(&socket).expect("bound"));
    assert_eq!(
        ask_to_stop(&sockets, file, QUICK.answer),
        Answer::NoFinisher
    );
    assert!(socket.exists(), "left alone");
    let bound = Listener::bind(&sockets, file);
    assert!(
        matches!(bound, Err(socket::SocketError::Untrusted(_))),
        "{bound:?}"
    );
}

#[test]
fn something_else_at_the_socket_s_name_is_never_used_or_removed() {
    let dir = TestDir::new();
    let (_path, file) = file_in(&dir, "first-sculpt.bcrf");
    let sockets = sockets_in(&dir);
    let socket = socket_path(&sockets, file);
    std::fs::write(&socket, b"not a socket").expect("written");
    assert_eq!(
        ask_to_stop(&sockets, file, QUICK.answer),
        Answer::NoFinisher
    );
    assert!(socket.exists(), "left alone");
    let bound = Listener::bind(&sockets, file);
    assert!(
        matches!(bound, Err(socket::SocketError::Occupied(_))),
        "{bound:?}"
    );
    assert_eq!(
        std::fs::read(&socket).expect("still there"),
        b"not a socket",
        "not renamed over"
    );
    assert_eq!(staged_files(&dir), 0, "nothing left of the try");
}

/// The files the sockets' directory in `dir` holds at a staging name (`.tmp`).
fn staged_files(dir: &TestDir) -> usize {
    let entries = std::fs::read_dir(dir.path().join("brushcraft")).expect("listed");
    let names = entries.map(|entry| entry.expect("an entry").file_name());
    let staged = names.filter(|name| name.to_string_lossy().ends_with(".tmp"));
    staged.count()
}

#[test]
fn a_socket_has_its_name_only_once_it_listens_and_is_ours_alone() {
    let dir = TestDir::new();
    let (_path, file) = file_in(&dir, "first-sculpt.bcrf");
    let sockets = sockets_in(&dir);
    let listener = Listener::bind(&sockets, file).expect("bound");
    let made = std::fs::symlink_metadata(listener.path()).expect("in place");
    assert_eq!(made.mode() & 0o777, 0o600, "{:o}", made.mode());
    assert_eq!(staged_files(&dir), 0, "renamed into place");
    // Connecting to it the moment it has its name works: it listened before.
    drop(UnixStream::connect(listener.path()).expect("it listens"));
}

#[test]
fn a_runtime_directory_too_deep_for_an_address_is_no_channel() {
    let dir = TestDir::new();
    let (path, file) = file_in(&dir, "first-sculpt.bcrf");
    let deep = dir.path().join("d".repeat(120));
    let sockets = SocketDir::at(deep.join("brushcraft"));
    assert!(Listener::bind(&sockets, file).is_err(), "no socket there");
    let began = Instant::now();
    assert_eq!(
        ask_to_stop(&sockets, file, Duration::from_secs(5)),
        Answer::NoChannel,
        "neither nobody listening nor a finisher that doesn't answer"
    );
    assert!(began.elapsed() < Duration::from_secs(1), "at once");
    let held = File::open(&path).expect("opened");
    held.lock().expect("locked");
    let way = make_way(&path, Some(&sockets), QUICK);
    assert!(
        matches!(
            way,
            Way::Busy {
                holder: Holder::Finisher,
                ..
            }
        ),
        "a finisher that can't be reached: {way:?}"
    );
}

#[test]
fn garbage_and_a_second_stop_are_ignored_safely() {
    let dir = TestDir::new();
    let (_path, file) = file_in(&dir, "first-sculpt.bcrf");
    let sockets = sockets_in(&dir);
    let mut listener = Listener::bind(&sockets, file).expect("bound");
    let socket = listener.path().to_owned();
    let talk = |bytes: &[u8]| {
        let mut stream = UnixStream::connect(&socket).expect("connected");
        stream.write_all(bytes).expect("written");
        stream
    };
    let garbage = [
        talk(b"hello\n"),
        talk(b"stopx\n"),
        talk(b"STOP\n"),
        talk(&[b's'; MAX_LINE * 4]),
        talk(b"st"),
    ];
    thread::sleep(Duration::from_millis(50));
    assert!(!listener.stop_asked(), "nothing asked");
    // Twice on one connection, and once more on another: two askers, both answered.
    let twice = talk(b"stop\nstop\n");
    let again = talk(b"stop\n");
    assert!(heard(&mut listener), "asked");
    listener.answer();
    for mut stream in [twice, again] {
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .expect("a timeout");
        let mut answer = String::new();
        stream.read_to_string(&mut answer).expect("read");
        assert_eq!(answer, "stopped\n", "once each");
    }
    for mut stream in garbage {
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .expect("a timeout");
        let mut answer = String::new();
        // Dropped unanswered: the connection ends with nothing said, or is reset.
        drop(stream.read_to_string(&mut answer));
        assert_eq!(answer, "", "no answer to garbage");
    }
}

#[test]
fn an_unanswered_ask_gives_up_within_its_patience() {
    let dir = TestDir::new();
    let (_path, file) = file_in(&dir, "first-sculpt.bcrf");
    let sockets = sockets_in(&dir);
    let socket = socket_path(&sockets, file);
    // A finisher that never answers: it listens, and never reads.
    let hung = UnixListener::bind(&socket).expect("bound");
    let began = Instant::now();
    assert_eq!(
        ask_to_stop(&sockets, file, QUICK.answer),
        Answer::Unanswered
    );
    let took = began.elapsed();
    assert!(took >= QUICK.answer, "it waited: {took:?}");
    assert!(took < Duration::from_secs(5), "and gave up: {took:?}");
    assert!(socket.exists(), "a live socket isn't stale");
    drop(hung);
}

#[test]
fn no_socket_directory_means_no_finisher() {
    let dir = TestDir::new();
    let (_path, file) = file_in(&dir, "first-sculpt.bcrf");
    let sockets = sockets_in(&dir);
    assert_eq!(
        ask_to_stop(&sockets, file, QUICK.answer),
        Answer::NoFinisher
    );
    assert!(
        !dir.path().join("brushcraft").exists(),
        "an opener makes nothing"
    );
}

#[test]
fn a_project_nobody_holds_is_clear_at_once() {
    let dir = TestDir::new();
    let (path, _) = file_in(&dir, "first-sculpt.bcrf");
    assert!(!is_locked(&path));
    assert_eq!(make_way(&path, Some(&sockets_in(&dir)), QUICK), Way::Clear);
    assert!(
        !is_locked(&dir.path().join("missing.bcrf")),
        "the open says"
    );
}

#[test]
fn a_held_project_with_no_finisher_is_busy() {
    let dir = TestDir::new();
    let (path, _) = file_in(&dir, "first-sculpt.bcrf");
    let held = File::open(&path).expect("opened");
    held.lock().expect("locked, as a running instance holds it");
    assert!(is_locked(&path));
    let began = Instant::now();
    let way = make_way(&path, Some(&sockets_in(&dir)), Patience::DEFAULT);
    assert!(
        matches!(
            way,
            Way::Busy {
                holder: Holder::Elsewhere,
                ..
            }
        ),
        "a working channel, nobody on it: another window: {way:?}"
    );
    assert!(
        began.elapsed() < Duration::from_secs(1),
        "nobody to wait for"
    );
    let unreachable = make_way(&path, None, QUICK);
    assert!(
        matches!(
            unreachable,
            Way::Busy {
                holder: Holder::Finisher,
                ..
            }
        ),
        "no channel at all: a finisher that can't be reached: {unreachable:?}"
    );
    drop(held);
    assert!(!is_locked(&path), "the lock goes with the file");
}

#[test]
fn a_finisher_that_doesnt_answer_holds_it_as_a_finisher() {
    let dir = TestDir::new();
    let (path, file) = file_in(&dir, "first-sculpt.bcrf");
    let sockets = sockets_in(&dir);
    let held = File::open(&path).expect("opened");
    held.lock().expect("locked");
    // Bound, and never read: a finisher stuck in a long batch.
    let stuck = Listener::bind(&sockets, file).expect("bound");
    let way = make_way(&path, Some(&sockets), QUICK);
    assert!(
        matches!(
            way,
            Way::Busy {
                holder: Holder::Finisher,
                ..
            }
        ),
        "{way:?}"
    );
    drop((stuck, held));
}

/// A finisher on a thread of its own holding `held`, listening on `listener`: once asked, it
/// lets the lock go and says it has stopped.
fn finisher_on_a_thread(held: File, mut listener: Listener) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let began = Instant::now();
        while !listener.stop_asked() {
            assert!(began.elapsed() < Duration::from_secs(30), "asked");
            thread::sleep(Duration::from_millis(1));
        }
        drop(held);
        listener.answer();
    })
}

/// Steps `waiter` as the waiting window does until the lock, looked at when due, is free: how
/// long that took. Bounded by `most`.
fn waited_out(waiter: &mut Waiter, path: &Path, most: Duration) -> Duration {
    let began = Instant::now();
    loop {
        let now = Instant::now();
        if waiter.step(now) {
            if !is_locked(path) {
                return began.elapsed();
            }
            waiter.looked(now);
        }
        assert!(began.elapsed() < most, "it opened within {most:?}");
        thread::sleep(
            waiter
                .wake(now)
                .saturating_duration_since(Instant::now())
                .min(Duration::from_millis(5)),
        );
    }
}

#[test]
fn a_finisher_that_binds_after_the_first_ask_is_taken_over() {
    let dir = TestDir::new();
    let (path, file) = file_in(&dir, "first-sculpt.bcrf");
    let sockets = sockets_in(&dir);
    let held = File::open(&path).expect("opened");
    held.lock().expect("locked");
    let Way::Busy { since, holder } = make_way(&path, Some(&sockets), QUICK) else {
        panic!("held");
    };
    assert_eq!(holder, Holder::Elsewhere, "nobody listened yet");
    let now = Instant::now();
    let mut waiter = Waiter::new(
        path.clone(),
        Some(sockets.clone()),
        QUICK,
        (since, holder),
        now,
    );
    // It binds now: the waiting window's next ask finds it.
    let finisher = finisher_on_a_thread(held, Listener::bind(&sockets, file).expect("bound"));
    let took = waited_out(&mut waiter, &path, Duration::from_secs(10));
    finisher.join().expect("the finisher");
    assert_eq!(waiter.holder(), Holder::Finisher, "found as one");
    assert!(
        took < ASK_AGAIN * 3,
        "taken over at its next ask, not at the end of the landing: {took:?}"
    );
}

#[test]
fn a_window_that_becomes_a_finisher_while_another_waits_is_said_so() {
    let dir = TestDir::new();
    let (path, file) = file_in(&dir, "first-sculpt.bcrf");
    let sockets = sockets_in(&dir);
    let held = File::open(&path).expect("opened");
    held.lock().expect("locked, by a window");
    let now = Instant::now();
    let held_by = (now, Holder::Elsewhere);
    let mut waiter = Waiter::new(path.clone(), Some(sockets.clone()), QUICK, held_by, now);
    // The window's close bar Save: it binds as its window goes, and is busy landing a batch.
    let stuck = Listener::bind(&sockets, file).expect("bound");
    let began = Instant::now();
    while waiter.holder() == Holder::Elsewhere {
        let at = Instant::now();
        if waiter.step(at) {
            waiter.looked(at);
        }
        assert!(
            began.elapsed() < ASK_AGAIN * 3 + QUICK.answer,
            "the bar changes"
        );
        thread::sleep(Duration::from_millis(5));
    }
    // Its landing ends, and it lets go: the waiting window opens.
    drop((stuck, held));
    waited_out(&mut waiter, &path, Duration::from_secs(5));
}

#[test]
fn a_finisher_that_stops_clears_the_way() {
    let dir = TestDir::new();
    let (path, file) = file_in(&dir, "first-sculpt.bcrf");
    let sockets = sockets_in(&dir);
    let held = File::open(&path).expect("opened");
    held.lock().expect("locked");
    let mut listener = Listener::bind(&sockets, file).expect("bound");
    let finisher = thread::spawn(move || {
        assert!(heard(&mut listener), "asked");
        // Its project closes, its lock released, before it answers.
        drop(held);
        listener.answer();
    });
    let way = make_way(&path, Some(&sockets), Patience::DEFAULT);
    finisher.join().expect("the finisher");
    assert_eq!(way, Way::TookOver);
}
