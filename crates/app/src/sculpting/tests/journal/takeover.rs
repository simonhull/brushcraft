//! Taking over from a finisher (the dab journal spec, "Taking over"), end to end with the brush.
//!
//! A finisher stopped mid-landing leaves its recovery file holding every dab placed and the count
//! of those landed; the next instance restores it, resumes the stroke, and lands the rest, ending
//! where a landing never interrupted does, within the restore's bound. Once in process, and once
//! with the finisher a process of its own, asked through its socket as the app asks.

use std::mem::ManuallyDrop;

use super::resume::{Corrections, corrections, steps_apart};
use super::*;
use crate::frame::{CloseAnswer, Finished, UiKey, WindowRequest};
use crate::saving::SaveStep;
use crate::sculpting::tests::saving::{backlogged_in, finish_save, path_in, recovery_files};
use crate::takeover::{Patience, SocketDir, Way};
use crate::{ProjectToOpen, held_elsewhere};

/// Passes of the stroke to and fro: a backlog of more than 4,000 dabs.
const PASSES: u32 = 512;

/// Passes of the stroke in the test that stops a finisher in this process: enough to stop midway,
/// fewer to land again.
const IN_PROCESS_PASSES: u32 = 32;

/// The variable holding the directory the finishing child works in: the parent's [`TestDir`].
/// The test binary is a child only when it and [`SOCKETS`] are set.
const CHILD_DIR: &str = "BRUSHCRAFT_TAKEOVER_DIR";

/// The variable holding the directory the child's takeover socket goes in: a test's, never the
/// real `$XDG_RUNTIME_DIR/brushcraft`.
const SOCKETS: &str = "BRUSHCRAFT_TAKEOVER_SOCKETS";

/// The child test's full path, as libtest names it.
const CHILD_TEST: &str = "sculpting::tests::journal::takeover::finishing_child";

/// The line the child prints once it listens, as its landing begins, its backlog after it.
const LISTENING: &str = "brushcraft takeover child: listening, backlog";

/// The line the child prints as it lands, the batches it has landed after it: a dab each.
const LANDED: &str = "brushcraft takeover child: landed";

/// How many batches the child lands before the parent asks it to stop: well into the landing,
/// and far from its end.
const LANDED_BEFORE_ASKING: usize = 40;

/// How long the parent waits for the child to get there, or to end.
const CHILD_TIMEOUT: Duration = Duration::from_secs(300);

/// The close bar's Save on `owner`, its backlog landing, then the window's passes until it is
/// asked to go unseen ([`WindowRequest::Finish`]).
///
/// The brush goes on landing a dab a batch, so the finisher looks for a takeover between every
/// dab: the app's batches land a 4,000-dab backlog on these small projects in a fifth of a second,
/// too soon to be caught midway.
fn close_and_save_to_finish(owner: &mut Owner<Sculpting, NoUploads>) {
    assert_eq!(owner.ask_close(Instant::now()), CloseAnswer::Asked);
    let saved = Instant::now();
    assert!(owner.key(UiKey::Enter, saved), "Save");
    let mut asked = None;
    while asked.is_none() {
        if owner.save_step() == SaveStep::Waiting {
            std::thread::sleep(Duration::from_millis(1));
        }
        run(owner, 1);
        asked = owner.take_request();
        assert!(saved.elapsed() < Duration::from_secs(60), "it ends");
    }
    assert_eq!(asked, Some(WindowRequest::Finish));
}

/// What the app opens in `dir`: its project, restoring.
fn to_open(dir: &TestDir) -> ProjectToOpen {
    ProjectToOpen {
        path: path_in(dir),
        config: config(dir),
    }
}

/// The frame loop the next instance starts in `dir`, as the app's start does: the project opened
/// restoring, and the stroke a restore found still landing resumed.
fn reopened(dir: &TestDir) -> Owner<Sculpting, NoUploads> {
    let (project, note) = to_open(dir).open_restoring().expect("opened");
    assert!(
        matches!(note, Some(crate::OpenNote::Restored { .. })),
        "{note:?}"
    );
    let mut owner = owner_of(project, Sculpting::new, setup());
    owner.resume_stroke();
    owner
}

/// Lands what `owner`'s brush holds, saves, and returns every finest tile's corrections.
fn landed_and_saved(owner: &mut Owner<Sculpting, NoUploads>) -> BTreeMap<TileKey, Corrections> {
    let began = Instant::now();
    while owner.brush().stroking() {
        run(owner, 1_000);
        assert!(began.elapsed() < Duration::from_secs(300), "it lands");
    }
    assert_eq!(owner.brush().error(), None);
    owner.request_save(Instant::now());
    finish_save(owner);
    assert_eq!(owner.saving_succeeded(), Some(true), "saved");
    corrections(owner.project())
}

/// A landing never interrupted, from the same stroke of `passes` passes: every finest tile's
/// corrections.
fn never_interrupted(passes: u32) -> BTreeMap<TileKey, Corrections> {
    let (dir, mut owner) = backlogged_in(TestDir::new(), passes, Instant::now());
    owner.brush_mut().cost = DabCost::default();
    let expected = landed_and_saved(&mut owner);
    owner.close().expect("closed");
    drop(dir);
    expected
}

/// The dabs `owner`'s resumed stroke still has to land.
fn left_to_land(owner: &Owner<Sculpting, NoUploads>) -> usize {
    let error = owner.brush().error();
    let stroke = owner
        .brush()
        .strokes
        .first()
        .unwrap_or_else(|| panic!("the stroke resumed: {error:?}"));
    stroke.dabs.len()
}

#[test]
fn the_new_window_restores_what_the_finisher_left() {
    let (dir, mut finisher) = backlogged_in(TestDir::new(), IN_PROCESS_PASSES, Instant::now());
    let backlog = finisher.landing();
    assert!(backlog > 64, "{backlog} dabs to land");
    close_and_save_to_finish(&mut finisher);
    let mut looks = 0;
    let outcome = finisher.finish_unseen_unless(
        |summary, body| panic!("{summary}: {body}"),
        || {
            looks += 1;
            looks > 3
        },
    );
    assert_eq!(
        outcome.expect("stopped"),
        Finished::Stopped { flushed: true }
    );

    let mut owner = reopened(&dir);
    let left = left_to_land(&owner);
    assert!(
        0 < left && left < backlog,
        "it resumes where the finisher stopped: {left} of {backlog} left"
    );
    let got = landed_and_saved(&mut owner);
    owner.close().expect("closed");
    assert_eq!(recovery_files(&dir), 0, "saved: nothing left to restore");
    let apart = steps_apart(&got, &never_interrupted(IN_PROCESS_PASSES));
    assert!(
        apart <= 1.5,
        "{apart} steps from a landing never interrupted"
    );
}

/// The child: a finisher of its own, until it is asked to stop.
///
/// It builds the backlog in the parent's directory, Saves on the close bar, binds its socket in
/// [`SOCKETS`] as the app does when its window goes ([`crate::listen`]), prints [`LISTENING`],
/// and finishes unseen as [`crate::finish_heard`] does, saying how many batches it has landed
/// ([`LANDED`]) as it goes. Without its variables it does nothing.
#[test]
#[ignore = "reopening_while_finishing_takes_over_and_loses_nothing runs it as its child"]
fn finishing_child() {
    let (Some(dir), Some(sockets)) = (std::env::var_os(CHILD_DIR), std::env::var_os(SOCKETS))
    else {
        return;
    };
    let (dir, mut owner) = backlogged_in(TestDir::adopt(dir.into()), PASSES, Instant::now());
    // The parent's directory: it stays when the child is done.
    let _parents = ManuallyDrop::new(dir);
    let backlog = owner.landing();
    close_and_save_to_finish(&mut owner);
    let sockets = SocketDir::at(sockets.into());
    let listener = crate::listen(owner.project_path(), Some(&sockets));
    assert!(listener.is_some(), "it listens");
    println!("{LISTENING} {backlog}");
    let mut looks = 0_usize;
    let notify = |summary: &str, body: &str| panic!("{summary}: {body}");
    // A look before each batch: those before it have landed.
    let landed = || {
        println!("{LANDED} {looks}");
        looks += 1;
    };
    crate::finish_heard(owner, listener, notify, landed).expect("stopped, which is no failure");
}

/// What the child says, as the parent hears it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Said {
    /// It listens, with this many dabs to land.
    Listening(usize),
    /// It has landed this many batches.
    Landed(usize),
}

/// What `line` of the child's output says, if anything: a [`LISTENING`] or [`LANDED`] line,
/// perhaps after libtest's `test <name> ... ` on the same line.
fn heard(line: &str) -> Option<Said> {
    let number = |marker: &str| {
        let (_, after) = line.split_once(marker)?;
        after.trim().parse().ok()
    };
    number(LISTENING)
        .map(Said::Listening)
        .or_else(|| number(LANDED).map(Said::Landed))
}

/// The finishing child, running in `dir` with its socket in `sockets`, its output read on
/// threads of its own; returns it once it is [`LISTENING`], with its backlog.
fn finishing(dir: &TestDir, sockets: &std::path::Path) -> (Child, usize) {
    use std::io::{BufRead as _, BufReader};
    use std::process::{Command, Stdio};
    let mut child = Command::new(std::env::current_exe().expect("the test binary"))
        .args([CHILD_TEST, "--exact", "--ignored", "--nocapture"])
        .args(["--test-threads", "1"])
        .env(CHILD_DIR, dir.path())
        .env(SOCKETS, sockets)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("the child starts");
    let stdout = child.stdout.take().expect("its output");
    let mut stderr = child.stderr.take().expect("its errors");
    let errors = std::thread::spawn(move || {
        let mut text = String::new();
        drop(std::io::Read::read_to_string(&mut stderr, &mut text));
        text
    });
    let (says, said) = std::sync::mpsc::channel();
    let output = std::thread::spawn(move || {
        let lines = BufReader::new(stdout).lines();
        let mut text = String::new();
        let read = lines.map_while(Result::ok);
        for line in read {
            if let Some(heard) = heard(&line) {
                // The parent may have stopped listening: then nobody hears.
                says.send(heard).unwrap_or_default();
            } else {
                text.push_str(&line);
                text.push('\n');
            }
        }
        text
    });
    let mut child = Child {
        process: child,
        output: Some(output),
        errors: Some(errors),
        said,
    };
    let Some(Said::Listening(backlog)) = child.next(|_| true) else {
        let words = child.end();
        panic!("the child never listened:\n{words}");
    };
    (child, backlog)
}

/// A finishing child, and what it says.
struct Child {
    /// The process.
    process: std::process::Child,
    /// The thread reading its output, which returns it.
    output: Option<std::thread::JoinHandle<String>>,
    /// The thread reading its errors, which returns them.
    errors: Option<std::thread::JoinHandle<String>>,
    /// What it says, as heard.
    said: std::sync::mpsc::Receiver<Said>,
}

impl Child {
    /// The next thing the child says that `wanted` takes, within [`CHILD_TIMEOUT`]; `None` if it
    /// says nothing of the sort in time.
    fn next(&self, wanted: impl Fn(Said) -> bool) -> Option<Said> {
        let deadline = Instant::now() + CHILD_TIMEOUT;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            let said = self.said.recv_timeout(left).ok()?;
            if wanted(said) {
                return Some(said);
            }
        }
    }

    /// Waits for the child to end, killing it after [`CHILD_TIMEOUT`]: what it said, and whether
    /// it succeeded.
    fn wait(&mut self) -> (bool, String) {
        let began = Instant::now();
        let succeeded = loop {
            if let Some(status) = self.process.try_wait().expect("its status") {
                break status.success();
            }
            if began.elapsed() > CHILD_TIMEOUT {
                drop(self.process.kill());
                drop(self.process.wait());
                break false;
            }
            std::thread::sleep(Duration::from_millis(10));
        };
        let take = |thread: Option<std::thread::JoinHandle<String>>| {
            thread
                .map(|thread| thread.join().expect("read"))
                .unwrap_or_default()
        };
        let said = format!("{}{}", take(self.output.take()), take(self.errors.take()));
        (succeeded, said)
    }

    /// Kills the child and waits for it: what it said.
    fn end(&mut self) -> String {
        drop(self.process.kill());
        self.wait().1
    }
}

impl Drop for Child {
    /// A child left running is killed: a failing test leaves nothing behind.
    fn drop(&mut self) {
        if self.output.is_some() {
            self.end();
        }
    }
}

#[test]
fn reopening_while_finishing_takes_over_and_loses_nothing() {
    let dir = TestDir::new();
    let sockets = TestDir::new();
    let socket_dir = sockets.path().join("brushcraft");
    let (mut child, backlog) = finishing(&dir, &socket_dir);
    assert!(backlog >= 4_000, "{backlog} dabs to land");
    let into = |said| matches!(said, Said::Landed(batches) if batches >= LANDED_BEFORE_ASKING);
    if child.next(into).is_none() {
        let said = child.end();
        panic!("the child never got into its landing:\n{said}");
    }
    let project = path_in(&dir);
    let held = to_open(&dir).open().expect_err("the finisher holds it");
    assert!(held_elsewhere(&held), "{held:#}");

    let asked = Instant::now();
    let way = crate::takeover::make_way(
        &project,
        Some(&SocketDir::at(socket_dir)),
        Patience::DEFAULT,
    );
    let stopped = asked.elapsed();
    assert_eq!(way, Way::TookOver);
    let mut owner = reopened(&dir);
    let sculpting = asked.elapsed();
    let left = left_to_land(&owner);
    println!(
        "takeover: the finisher stopped {stopped:?} after the ask; sculpting {sculpting:?} after \
         it, {left} dabs left to land"
    );
    assert!(
        0 < left && left < backlog,
        "it took over mid-landing: {left} of {backlog} left"
    );
    assert!(
        sculpting < Duration::from_secs(5),
        "sculpting soon after asking: {sculpting:?}"
    );
    let (succeeded, said) = child.wait();
    assert!(succeeded, "the finisher exits with success:\n{said}");

    let got = landed_and_saved(&mut owner);
    owner.close().expect("closed");
    assert_eq!(recovery_files(&dir), 0, "saved: nothing left to restore");
    let apart = steps_apart(&got, &never_interrupted(PASSES));
    println!("takeover: {apart} quantization steps from a landing never interrupted");
    assert!(
        apart <= 1.5,
        "{apart} steps from a landing never interrupted"
    );
}
