//! A count of the other processes' active threads, with this process and its descendants left
//! out.
//!
//! Active means running, waiting to run, or in uninterruptible sleep (`R` or `D` in `/proc`):
//! the tasks the Linux load average counts.
//!
//! The quiet guard judges the machine on this count instead of on the load average less a model
//! of the suite's own share: the load average is a moving average of one snapshot every 5 s,
//! and the suite's parts swing between 0 and 16 runnable threads within milliseconds, so the
//! snapshots carry noise of their own (about ±1.3 at one standard deviation during a page-in)
//! that no smooth model of the suite's share can take back off. Counting only the others leaves
//! the suite's bursts out of the figure altogether.

use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::{ErrorKind, Read as _};
use std::num::ParseIntError;
use std::path::Path;
use std::str::FromStr;

use anyhow::{Context as _, Result};
use rustix::io::Errno;

/// A process id, as `/proc` names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Pid(u32);

impl Pid {
    /// This process's id.
    pub(crate) fn this() -> Self {
        Self(std::process::id())
    }
}

impl FromStr for Pid {
    type Err = ParseIntError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        text.parse().map(Self)
    }
}

impl fmt::Display for Pid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// One process as the census sees it: its parent, and how many of its threads are active.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Process {
    /// Its id.
    pub(crate) pid: Pid,
    /// Its parent's id.
    pub(crate) parent: Pid,
    /// Its threads in state `R` or `D`.
    pub(crate) active: u32,
}

/// What the census takes from a thread's `stat` line: whether the thread counts in the load
/// average, and its process's parent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Task {
    /// Its state (the field after the name) is `R` (running or runnable) or `D`
    /// (uninterruptible sleep, usually a wait for the disk). An idle kernel thread shows `I` and
    /// doesn't count, as in the load average.
    active: bool,
    /// Its process's parent (the second field after the name).
    parent: Pid,
}

impl Task {
    /// The task in `stat`, the start of a `stat` line: enough of it to reach the parent field.
    /// `None` if it has no name, state or parent.
    ///
    /// The command name may itself hold spaces and parentheses (and bytes that aren't UTF-8),
    /// so it ends at the last `)`; everything after it is ASCII.
    fn parse(stat: &[u8]) -> Option<Self> {
        let close = stat.iter().rposition(|&byte| byte == b')')?;
        let rest = std::str::from_utf8(stat.get(close + 1..)?).ok()?;
        let mut fields = rest.split_ascii_whitespace();
        let active = matches!(fields.next()?, "R" | "D");
        let parent = fields.next()?.parse().ok()?;
        Some(Self { active, parent })
    }
}

/// How much of a `stat` line is read: the name is at most 64 bytes, so the parent field ends
/// well within this.
const STAT_HEAD: usize = 512;

/// The task whose `stat` file is `path`, read into `buffer`; `None` if it doesn't exist any
/// more (it ended during the scan, as tasks may at any time) or can't be parsed (never seen
/// from Linux).
///
/// One read of the head into a reused buffer: a census reads ~2,500 of these, and the kernel's
/// work to produce each dominates its cost.
///
/// # Errors
///
/// For any other failure to read it.
fn read_task(path: &Path, buffer: &mut [u8; STAT_HEAD]) -> Result<Option<Task>> {
    let read = File::open(path).and_then(|mut file| file.read(buffer));
    match read {
        Ok(length) => Ok(buffer.get(..length).and_then(Task::parse)),
        // ESRCH: the task was reaped between opening and reading.
        Err(error)
            if error.kind() == ErrorKind::NotFound
                || error.raw_os_error() == Some(Errno::SRCH.raw_os_error()) =>
        {
            Ok(None)
        }
        Err(error) => Err(error).with_context(|| format!("reading {}", path.display())),
    }
}

/// Every process under `proc` (`/proc`, or a stand-in laid out the same way:
/// `<pid>/task/<tid>/stat`), with its parent and its active threads.
///
/// The parent comes from the main thread's line (its id is the process's), so each thread's
/// `stat` is read once and no process-wide one is. Processes and threads come and go while the
/// scan runs: one that vanishes is left out.
///
/// # Errors
///
/// If `proc` can't be listed, or a file in it can't be read for a reason other than its task
/// having ended.
pub(crate) fn scan(proc: &Path) -> Result<Vec<Process>> {
    let mut buffer = [0; STAT_HEAD];
    let mut processes = Vec::new();
    let entries = std::fs::read_dir(proc).with_context(|| format!("listing {}", proc.display()))?;
    for entry in entries {
        let entry = entry.with_context(|| format!("listing {}", proc.display()))?;
        let Some(pid) = entry
            .file_name()
            .to_str()
            .and_then(|name| name.parse().ok())
        else {
            continue;
        };
        let dir = entry.path().join("task");
        let tasks = match std::fs::read_dir(&dir) {
            Ok(tasks) => tasks,
            Err(error) if error.kind() == ErrorKind::NotFound => continue,
            Err(error) => {
                return Err(error).with_context(|| format!("listing {}", dir.display()));
            }
        };
        let (mut parent, mut active) = (None, 0);
        for task in tasks {
            // A thread that ended while being listed.
            let Ok(task) = task else { continue };
            let Some(read) = read_task(&task.path().join("stat"), &mut buffer)? else {
                continue;
            };
            active += u32::from(read.active);
            if task.file_name().to_str().and_then(|name| name.parse().ok()) == Some(pid) {
                parent = Some(read.parent);
            }
        }
        if let Some(parent) = parent {
            processes.push(Process {
                pid,
                parent,
                active,
            });
        }
    }
    Ok(processes)
}

/// The active threads of the processes in `processes` other than `ours` and its descendants:
/// the other processes' contribution to the load average at this instant.
///
/// `ours`'s descendants are found by following each process's parents up.
/// A parent missing from `processes` (it ended, or it is pid 0, the root's parent) ends the
/// walk; so does a loop, which `/proc` never shows but a scan racing with pid reuse could.
pub(crate) fn others_active(processes: &[Process], ours: Pid) -> u32 {
    let parents: HashMap<Pid, Pid> = processes.iter().map(|p| (p.pid, p.parent)).collect();
    let is_ours = |pid: Pid| {
        let mut at = pid;
        for _ in 0..=processes.len() {
            if at == ours {
                return true;
            }
            match parents.get(&at) {
                Some(&parent) => at = parent,
                None => return false,
            }
        }
        false
    };
    processes
        .iter()
        .filter(|p| !is_ours(p.pid))
        .map(|p| p.active)
        .sum()
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use std::path::PathBuf;

    use super::*;

    /// A process `pid` under `parent` with `active` active threads.
    fn process(pid: u32, parent: u32, active: u32) -> Process {
        Process {
            pid: Pid(pid),
            parent: Pid(parent),
            active,
        }
    }

    #[test]
    fn stat_lines_give_the_state_and_the_parent() {
        let line = |state| format!("42 (a (b) c) {state} 7 42 42 0 -1 4194560 1234");
        let task = |text: &str| Task::parse(text.as_bytes());
        for (state, active) in [("R", true), ("D", true), ("S", false), ("I", false)] {
            assert_eq!(
                task(&line(state)),
                Some(Task {
                    active,
                    parent: Pid(7)
                }),
                "{state}"
            );
        }
        assert_eq!(
            Task::parse(b"42 (\xff) Z 1"),
            Some(Task {
                active: false,
                parent: Pid(1)
            }),
            "a name that isn't UTF-8"
        );
        assert_eq!(task("42 no name"), None, "no name");
        assert_eq!(task("42 (x)"), None, "no state");
        assert_eq!(task("42 (x) S"), None, "no parent");
        assert_eq!(task("42 (x) S y"), None, "not a number");
    }

    /// The suite (10), its child (11) and grandchild (12) are left out wherever they are in
    /// the list; its parent (1, the shell) and an unrelated process (20) count.
    #[test]
    fn our_tree_is_left_out_of_the_count() {
        let processes = [
            process(12, 11, 16),
            process(1, 0, 1),
            process(10, 1, 1),
            process(11, 10, 4),
            process(20, 1, 2),
            process(21, 99, 1),
        ];
        assert_eq!(others_active(&processes, Pid(10)), 4, "1 + 2 + 1");
        assert_eq!(
            others_active(&processes, Pid(11)),
            5,
            "the suite counts too"
        );
        assert_eq!(others_active(&processes, Pid(77)), 25, "no tree: all");
        assert_eq!(others_active(&[], Pid(10)), 0, "nothing");
        let looped = [process(5, 6, 1), process(6, 5, 1)];
        assert_eq!(others_active(&looped, Pid(10)), 2, "a loop ends the walk");
    }

    /// A stand-in `/proc` in a directory of its own, removed on drop.
    struct FakeProc(PathBuf);

    impl FakeProc {
        /// An empty one, named for this test process and `name`.
        fn new(name: &str) -> Self {
            let root = std::env::temp_dir().join(format!(
                "brushcraft-bench-census-{name}-{}",
                std::process::id()
            ));
            let _gone = std::fs::remove_dir_all(&root);
            std::fs::create_dir_all(&root).unwrap();
            Self(root)
        }

        /// Adds process `pid` under `parent`, with a thread in each state of `states`, the
        /// first its main thread.
        fn add(&self, pid: u32, parent: u32, states: &[&str]) {
            let dir = self.0.join(pid.to_string());
            std::fs::create_dir_all(dir.join("task")).unwrap();
            for (n, state) in (0_u32..).zip(states) {
                let tid = if n == 0 { pid } else { pid * 100 + n };
                let task = dir.join("task").join(tid.to_string());
                std::fs::create_dir_all(&task).unwrap();
                std::fs::write(task.join("stat"), format!("{pid} (x y) {state} {parent}")).unwrap();
            }
        }
    }

    impl Drop for FakeProc {
        fn drop(&mut self) {
            let _gone = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn a_proc_layout_is_scanned() {
        let fake = FakeProc::new("scan");
        fake.add(1, 0, &["S"]);
        fake.add(10, 1, &["R", "R", "D", "S"]);
        fake.add(11, 10, &["D"]);
        fake.add(20, 1, &["R", "I", "D"]);
        std::fs::create_dir_all(fake.0.join("self")).unwrap();
        std::fs::write(fake.0.join("loadavg"), "1 1 1 1/1 1").unwrap();
        let mut processes = scan(&fake.0).unwrap();
        processes.sort_by_key(|p| p.pid.0);
        assert_eq!(
            processes,
            [
                process(1, 0, 0),
                process(10, 1, 3),
                process(11, 10, 1),
                process(20, 1, 2)
            ],
            "numbered directories only"
        );
        assert_eq!(others_active(&processes, Pid(10)), 2, "20's R and D");
    }

    /// This machine's `/proc` scans, and this process's own threads (the test's, at least, is
    /// running) are left out.
    #[test]
    fn this_machines_proc_is_scanned() {
        let processes = scan(Path::new("/proc")).unwrap();
        let this = processes.iter().find(|p| p.pid == Pid::this()).unwrap();
        assert!(this.active >= 1, "this test is running: {this:?}");
        let all: u32 = processes.iter().map(|p| p.active).sum();
        assert!(
            others_active(&processes, Pid::this()) <= all - this.active,
            "ours left out"
        );
        assert_eq!(Pid::this().to_string(), std::process::id().to_string());
    }
}
