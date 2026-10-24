//! What a run records about where and on what it ran: the machine, the commit, the time.

use std::path::Path;
use std::process::Command;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context as _, Result, anyhow, ensure};
use store::Workers;

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
    pub(crate) fingerprint: String,
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
            fingerprint: format!("{model}; {cpus} CPUs; {cores} cores; {gib} GiB"),
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
    /// The commit's full hash, or `unknown` outside a git checkout.
    pub(crate) commit: String,
    /// Whether tracked files had uncommitted changes (untracked files don't count). True when
    /// the commit is unknown: nothing vouches for the code.
    pub(crate) dirty: bool,
}

impl Revision {
    /// The revision of the checkout at `repo`, from `git`; unknown and dirty if `git` can't
    /// tell (no git, or not a checkout).
    pub(crate) fn of(repo: &Path) -> Self {
        let git = |args: &[&str]| {
            let output = Command::new("git")
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
            (Some(commit), Some(status)) => Self {
                commit,
                dirty: !status.is_empty(),
            },
            _ => Self {
                commit: "unknown".to_owned(),
                dirty: true,
            },
        }
    }
}

/// `time` as an RFC 3339 UTC timestamp to the second, as in `2026-10-24T13:05:09Z`.
///
/// # Errors
///
/// If `time` is before 1970.
pub(crate) fn utc(time: SystemTime) -> Result<String> {
    let seconds = time.duration_since(UNIX_EPOCH)?.as_secs();
    let (days, rest) = (seconds / 86_400, seconds % 86_400);
    let (year, month, day) = civil(i64::try_from(days)?);
    Ok(format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        rest / 3600,
        rest % 3600 / 60,
        rest % 60
    ))
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
        assert!(machine.fingerprint.contains(" CPUs; "), "{machine:?}");
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

    #[test]
    fn timestamps_are_utc_dates() {
        let at = |secs| utc(UNIX_EPOCH + Duration::from_secs(secs)).unwrap();
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

    #[test]
    fn this_checkout_has_a_revision() {
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
        let revision = Revision::of(repo);
        assert!(
            revision.commit == "unknown" || revision.commit.len() == 40,
            "{revision:?}"
        );
        let nowhere = Revision::of(Path::new("/nonexistent/brushcraft"));
        assert_eq!(nowhere.commit, "unknown", "no checkout");
        assert!(nowhere.dirty, "unknown counts as dirty");
    }
}
