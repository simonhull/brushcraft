//! What a run records about where and on what it ran: the machine, the commit, the time.

use std::fmt;
use std::path::Path;
use std::process::Command;
use std::str::FromStr;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context as _, Result, anyhow, ensure};
use store::Workers;
use thiserror::Error;

/// The machine a run measured: its name, and a fingerprint of the hardware that sets its
/// speed.
///
/// History is kept per machine name (`<name>.jsonl`); a baseline only counts for a run with the
/// same fingerprint, so new hardware under an old name starts afresh instead of alarming.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Machine {
    /// The host name, reduced to `a-z`, `0-9` and `-` for a file name.
    pub(crate) name: String,
    /// CPU model, logical CPUs, physical cores and memory, as in `AMD Ryzen 9 5950X 16-Core
    /// Processor; 32 CPUs; 16 cores; 62 GiB`.
    pub(crate) fingerprint: Fingerprint,
    /// The kernel release, recorded beside the fingerprint but not part of it: a kernel update
    /// shouldn't orphan every baseline, and a change it makes should show as a change.
    pub(crate) kernel: String,
}

impl Machine {
    /// This machine, from `/proc`.
    ///
    /// # Errors
    ///
    /// If `/proc` can't be read or lacks what is needed.
    pub(crate) fn this() -> Result<Self> {
        let read =
            |path: &str| std::fs::read_to_string(path).with_context(|| format!("reading {path}"));
        let name = file_name(read("/proc/sys/kernel/hostname")?.trim())?;
        let kernel = read("/proc/sys/kernel/osrelease")?.trim().to_owned();
        let cpuinfo = read("/proc/cpuinfo")?;
        let model = field(&cpuinfo, "model name").unwrap_or("unknown CPU");
        let cpus = std::thread::available_parallelism()?.get();
        let cores = Workers::PhysicalCores.count().get();
        let memory = field(&read("/proc/meminfo")?, "MemTotal")
            .and_then(|kib| kib.trim_end_matches("kB").trim().parse::<u64>().ok())
            .ok_or_else(|| anyhow!("no MemTotal in /proc/meminfo"))?;
        // Rounded to the GiB: firmware reserves a little more or less after an update.
        let gib = (memory + (1 << 19)) >> 20;
        Ok(Self {
            name,
            fingerprint: Fingerprint(format!("{model}; {cpus} CPUs; {cores} cores; {gib} GiB")),
            kernel,
        })
    }
}

/// The value of the first line `name: value` in `text` (as in `/proc/cpuinfo`), trimmed.
fn field<'a>(text: &'a str, name: &str) -> Option<&'a str> {
    text.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        (key.trim() == name).then(|| value.trim())
    })
}

/// `host` reduced to a file name: lowercase, with every character but `a-z`, `0-9` and `-`
/// dropped.
///
/// # Errors
///
/// If nothing is left.
fn file_name(host: &str) -> Result<String> {
    let name: String = host
        .to_ascii_lowercase()
        .chars()
        .filter(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || *c == '-')
        .collect();
    ensure!(
        !name.is_empty(),
        "the host name {host:?} makes no file name"
    );
    Ok(name)
}

/// The commit a run measured, and whether the working tree differed from it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Revision {
    /// The commit, unknown outside a git checkout.
    pub(crate) commit: Commit,
    /// Whether tracked files had uncommitted changes (untracked files don't count). True when
    /// the commit is unknown: nothing vouches for the code.
    pub(crate) dirty: bool,
}

impl Revision {
    /// The revision this binary was built from, as `build.rs` recorded it: the commit (unknown
    /// if the recorded text isn't a hash) and whether tracked files differed from it then.
    pub(crate) fn built() -> Self {
        match env!("BRUSHCRAFT_BUILD_COMMIT").parse() {
            Ok(commit) => Self {
                commit,
                dirty: env!("BRUSHCRAFT_BUILD_DIRTY") != "false",
            },
            Err(_not_a_hash) => Self::unknown(),
        }
    }

    /// Why this binary's revision (`self`) doesn't vouch for the tree's (`tree`), if it
    /// doesn't: a different commit, or changes in either. The stage benches build from the
    /// tree, the rest is this binary, so a run with a reason measures a mix.
    pub(crate) fn mismatch(&self, tree: &Self) -> Option<String> {
        if self.commit != tree.commit {
            Some(format!(
                "built from {}, the tree is at {}",
                self.commit, tree.commit
            ))
        } else if self.dirty || tree.dirty {
            Some(format!(
                "{} with uncommitted changes (built {}, tree {})",
                self.commit,
                if self.dirty { "dirty" } else { "clean" },
                if tree.dirty { "dirty" } else { "clean" }
            ))
        } else {
            None
        }
    }

    /// An unknown commit, counted dirty: nothing vouches for the code.
    fn unknown() -> Self {
        Self {
            commit: Commit::unknown(),
            dirty: true,
        }
    }

    /// The revision of the checkout at `repo`, from `git`; unknown and dirty if `git` can't
    /// tell (no git, or not a checkout).
    pub(crate) fn of(repo: &Path) -> Self {
        let git = |args: &[&str]| {
            let output = Command::new("git")
                // No index refresh: a status that rewrote the index would make the next build rerun
                // build.rs, whose triggers include the index.
                .env("GIT_OPTIONAL_LOCKS", "0")
                .arg("-C")
                .arg(repo)
                .args(args)
                .output()
                .ok()?;
            output
                .status
                .success()
                .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        };
        match (
            git(&["rev-parse", "HEAD"]),
            git(&["status", "--porcelain", "--untracked-files=no"]),
        ) {
            (Some(commit), Some(status)) => match commit.parse() {
                Ok(commit) => Self {
                    commit,
                    dirty: !status.is_empty(),
                },
                Err(_not_a_hash) => Self::unknown(),
            },
            _ => Self::unknown(),
        }
    }
}

/// Why a text isn't a run detail ([`Commit`], [`Fingerprint`] or [`Timestamp`]).
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub(crate) enum BadDetail {
    /// Not 40 lowercase hex digits, nor `unknown`.
    #[error("{0:?} isn't a commit hash")]
    Commit(String),
    /// Empty.
    #[error("an empty hardware fingerprint")]
    Fingerprint,
    /// Not `YYYY-MM-DDTHH:MM:SSZ`.
    #[error("{0:?} isn't a UTC timestamp")]
    Timestamp(String),
}

/// A commit: its full 40-digit hash, or unknown (a build outside a git checkout).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Commit(String);

impl Commit {
    /// The commit of a build outside a git checkout.
    pub(crate) fn unknown() -> Self {
        Self("unknown".to_owned())
    }

    /// The hash, or `unknown`.
    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

impl FromStr for Commit {
    type Err = BadDetail;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let hash = text.len() == 40
            && text
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
        if hash || text == "unknown" {
            Ok(Self(text.to_owned()))
        } else {
            Err(BadDetail::Commit(text.to_owned()))
        }
    }
}

impl fmt::Display for Commit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A hardware fingerprint (see [`Machine::fingerprint`]): free text, never empty. Baselines
/// count only for runs with the same one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Fingerprint(String);

impl Fingerprint {
    /// The fingerprint as text.
    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

impl FromStr for Fingerprint {
    type Err = BadDetail;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        if text.trim().is_empty() {
            Err(BadDetail::Fingerprint)
        } else {
            Ok(Self(text.to_owned()))
        }
    }
}

impl fmt::Display for Fingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// An RFC 3339 UTC timestamp to the second, `YYYY-MM-DDTHH:MM:SSZ`: when a run started, which
/// with the machine names the run.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Timestamp(String);

impl Timestamp {
    /// `time` as a timestamp.
    ///
    /// # Errors
    ///
    /// If `time` is before 1970.
    pub(crate) fn of(time: SystemTime) -> Result<Self> {
        let seconds = time.duration_since(UNIX_EPOCH)?.as_secs();
        let (days, rest) = (seconds / 86_400, seconds % 86_400);
        let (year, month, day) = civil(i64::try_from(days)?);
        Ok(Self(format!(
            "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
            rest / 3600,
            rest % 3600 / 60,
            rest % 60
        )))
    }

    /// The timestamp as text.
    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

impl FromStr for Timestamp {
    type Err = BadDetail;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        const PATTERN: &[u8] = b"dddd-dd-ddTdd:dd:ddZ";
        let fits = text.len() == PATTERN.len()
            && text.bytes().zip(PATTERN).all(|(b, &p)| match p {
                b'd' => b.is_ascii_digit(),
                literal => b == literal,
            });
        if fits {
            Ok(Self(text.to_owned()))
        } else {
            Err(BadDetail::Timestamp(text.to_owned()))
        }
    }
}

impl fmt::Display for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// The proleptic Gregorian date `days` after 1970-01-01: year, month (1–12), day (1–31).
///
/// Howard Hinnant's `civil_from_days`: counts in 400-year eras starting on 1 March, so the leap
/// day falls at the end of each counted year.
const fn civil(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097); // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11], March first
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + if month <= 2 { 1 } else { 0 };
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "month is 1–12 and day 1–31 by construction"
    )]
    (year, month as u32, day as u32)
}

/// The CPU time this process's waited-for children have taken, from `/proc/self/stat`
/// (`cutime` + `cstime`): read before and after running a child, the difference is its CPU
/// time.
///
/// # Errors
///
/// If `/proc/self/stat` can't be read or parsed.
pub(crate) fn children_cpu() -> Result<Duration> {
    let stat = std::fs::read_to_string("/proc/self/stat").context("reading /proc/self/stat")?;
    let ticks = parse_children_ticks(&stat)?;
    let per_second = rustix::param::clock_ticks_per_second();
    ensure!(per_second > 0, "no clock ticks per second");
    Ok(Duration::from_secs(ticks / per_second)
        + Duration::from_nanos(ticks % per_second * 1_000_000_000 / per_second))
}

/// The children's CPU ticks in `stat` (the text of `/proc/self/stat`): its 16th and 17th fields
/// summed, counted after the command name's closing parenthesis as
/// [`crate::sys::minor_faults`] counts.
///
/// # Errors
///
/// If there is no `)`, too few fields, or they aren't numbers.
fn parse_children_ticks(stat: &str) -> Result<u64> {
    let (_, after_name) = stat
        .rsplit_once(')')
        .ok_or_else(|| anyhow!("no command name in /proc/self/stat"))?;
    // After the name, the 3rd field is the first: the 16th and 17th are the 14th and 15th.
    let mut fields = after_name.split_whitespace().skip(13);
    let mut tick = || -> Result<u64> {
        let field = fields
            .next()
            .ok_or_else(|| anyhow!("too few fields in /proc/self/stat"))?;
        field
            .parse()
            .with_context(|| format!("children's CPU time in /proc/self/stat: {field:?}"))
    };
    Ok(tick()? + tick()?)
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    #[test]
    fn this_machine_is_described() {
        let machine = Machine::this().unwrap();
        assert!(!machine.name.is_empty(), "{machine:?}");
        assert!(
            machine.fingerprint.as_str().contains(" CPUs; "),
            "{machine:?}"
        );
        assert_eq!(
            file_name("Omarchy.local_1").unwrap(),
            "omarchylocal1",
            "filtered"
        );
        assert!(file_name("..").is_err(), "nothing left");
        assert_eq!(
            field("model name\t: AMD Ryzen\nflags : x\n", "model name"),
            Some("AMD Ryzen"),
            "cpuinfo's spacing"
        );
    }

    /// Commits, fingerprints and timestamps refuse what they can't be.
    #[test]
    fn run_details_are_checked() {
        let hash = "0123456789abcdef0123456789abcdef01234567";
        assert_eq!(hash.parse::<Commit>().unwrap().as_str(), hash, "a hash");
        assert_eq!(
            "unknown".parse::<Commit>().unwrap(),
            Commit::unknown(),
            "unknown"
        );
        for bad in ["", "0123", &hash.to_uppercase(), &format!("{hash}0")] {
            assert!(bad.parse::<Commit>().is_err(), "{bad:?}");
        }
        assert!(" ".parse::<Fingerprint>().is_err(), "empty fingerprint");
        assert_eq!(
            "cpu".parse::<Fingerprint>().unwrap().as_str(),
            "cpu",
            "fingerprint"
        );
        let stamp = "2026-10-24T13:05:09Z";
        assert_eq!(
            stamp.parse::<Timestamp>().unwrap().as_str(),
            stamp,
            "a timestamp"
        );
        for bad in [
            "r1",
            "2026-10-24 13:05:09Z",
            "2026-10-24T13:05:09",
            "2026-1x-24T13:05:09Z",
        ] {
            assert!(bad.parse::<Timestamp>().is_err(), "{bad:?}");
        }
    }

    #[test]
    fn timestamps_are_utc_dates() {
        let at = |secs| {
            Timestamp::of(UNIX_EPOCH + Duration::from_secs(secs))
                .unwrap()
                .to_string()
        };
        assert_eq!(at(0), "1970-01-01T00:00:00Z", "the epoch");
        assert_eq!(at(951_782_400), "2000-02-29T00:00:00Z", "a leap day");
        assert_eq!(at(1_792_847_109), "2026-10-24T13:05:09Z", "today");
        assert_eq!(at(4_107_542_399), "2100-02-28T23:59:59Z", "2100 isn't leap");
        assert_eq!(civil(-1), (1969, 12, 31), "before the epoch");
    }

    #[test]
    fn childrens_cpu_ticks_parse() {
        let stat = "42 (a (b) c) S 1 42 42 0 -1 4194560 1234 7 5 0 10 3 25 17 20 0";
        assert_eq!(
            parse_children_ticks(stat).unwrap(),
            42,
            "cutime 25 + cstime 17"
        );
        assert!(
            parse_children_ticks("42 (x) S 1 2").is_err(),
            "too few fields"
        );
        assert!(children_cpu().is_ok(), "this process's");
    }

    /// The build's revision is recorded, and a mismatch with the tree, or changes in either,
    /// is named.
    #[test]
    fn the_build_vouches_for_the_tree_only_when_both_are_clean_at_one_commit() {
        let built = Revision::built();
        assert!(
            built.commit == Commit::unknown() || built.commit.as_str().len() == 40,
            "{built:?}"
        );
        let at = |hash: &str, dirty| Revision {
            commit: hash.repeat(40).parse().unwrap(),
            dirty,
        };
        assert_eq!(
            at("a", false).mismatch(&at("a", false)),
            None,
            "clean, one commit"
        );
        assert!(
            at("a", false).mismatch(&at("b", false)).is_some(),
            "two commits"
        );
        assert!(
            at("a", true).mismatch(&at("a", false)).is_some(),
            "a dirty build"
        );
        assert!(
            at("a", false).mismatch(&at("a", true)).is_some(),
            "a dirty tree"
        );
    }

    #[test]
    fn this_checkout_has_a_revision() {
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
        let revision = Revision::of(repo);
        assert!(
            revision.commit.as_str() == "unknown" || revision.commit.as_str().len() == 40,
            "{revision:?}"
        );
        let nowhere = Revision::of(Path::new("/nonexistent/brushcraft"));
        assert_eq!(nowhere.commit, Commit::unknown(), "no checkout");
        assert!(nowhere.dirty, "unknown counts as dirty");
    }
}
