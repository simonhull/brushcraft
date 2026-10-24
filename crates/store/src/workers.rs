//! How many evaluation workers a store starts: by default one per physical core this process
//! may run on.
//!
//! A brush step is a handful of short parallel waves with serial work between them. On a machine
//! with simultaneous multithreading, a worker per hardware thread puts a worker on the sibling of
//! every core, including the owner thread's: between waves the workers spin, and the owner's
//! serial work and every wave run slower. On the development box (16 cores, 32 threads) a brush
//! step took p50 1.95 / p99 2.59 ms with 32 workers and 1.55 / 2.00 ms with 16, and paging in
//! and building the index were no faster on 32 either. So the default is one worker per
//! physical core: efficient on modest hardware, and as wide as the machine's cores on a big one.
//! [`Workers::LogicalCores`] asks for every hardware thread.
//!
//! Counting physical cores is the platform's business. On Linux and Android it is the distinct
//! cores among the CPUs the process's affinity mask allows, grouped by the kernel's topology
//! files, and never more than [`std::thread::available_parallelism`] (which also counts cgroup
//! CPU quotas). Anywhere else, or if anything can't be read or parsed, it falls back to
//! `available_parallelism`, as [`Workers::LogicalCores`] does: the count is a performance
//! choice, never a reason to fail.

use std::num::NonZeroUsize;

/// How many evaluation workers a [`crate::TileStore`] starts (see the module notes).
///
/// Results are the same bits whatever the count; only speed depends on it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum Workers {
    /// One per physical core the process may run on, at most one per hardware thread
    /// available to it; one per hardware thread where the cores can't be counted.
    #[default]
    PhysicalCores,
    /// One per hardware thread available to the process
    /// ([`std::thread::available_parallelism`]), one if that can't be told.
    LogicalCores,
    /// Exactly this many: for tests that compare thread counts, or an app's own choice.
    Fixed(NonZeroUsize),
}

impl Workers {
    /// The number of workers this asks for on this machine, now. Never fails: see the variants
    /// for the fallbacks.
    #[must_use]
    pub fn count(self) -> NonZeroUsize {
        match self {
            Self::PhysicalCores => physical_cores().unwrap_or_else(logical_cores),
            Self::LogicalCores => logical_cores(),
            Self::Fixed(n) => n,
        }
    }
}

/// The names [`Workers`] goes by in text: `physical`, `logical`, or a positive count.
impl std::fmt::Display for Workers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PhysicalCores => f.write_str("physical"),
            Self::LogicalCores => f.write_str("logical"),
            Self::Fixed(n) => write!(f, "{n}"),
        }
    }
}

/// Why a [`Workers`] setting didn't parse.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0:?} is not a worker setting: expected physical, logical or a positive count")]
pub struct ParseWorkersError(String);

/// Why the evaluation worker threads couldn't be started (see [`crate::CacheError::Workers`]):
/// the operating system refused a thread. The thread library's own error is its source.
#[derive(Debug, thiserror::Error)]
#[error("the evaluation worker threads couldn't be started")]
pub struct StartWorkersError(#[source] rayon::ThreadPoolBuildError);

impl StartWorkersError {
    /// The error building the thread pool gave.
    pub(crate) const fn new(source: rayon::ThreadPoolBuildError) -> Self {
        Self(source)
    }
}

/// Parses what [`Workers`]' `Display` writes: `physical`, `logical`, or a positive count.
impl std::str::FromStr for Workers {
    type Err = ParseWorkersError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text {
            "physical" => Ok(Self::PhysicalCores),
            "logical" => Ok(Self::LogicalCores),
            _ => text
                .parse()
                .ok()
                .map(Self::Fixed)
                .ok_or_else(|| ParseWorkersError(text.to_owned())),
        }
    }
}

/// The hardware threads available to the process, one if that can't be told.
fn logical_cores() -> NonZeroUsize {
    std::thread::available_parallelism().unwrap_or(NonZeroUsize::MIN)
}

/// The physical cores this process may run on, at most [`logical_cores`]; `None` where they
/// can't be counted (see the module notes).
#[cfg(any(target_os = "linux", target_os = "android"))]
fn physical_cores() -> Option<NonZeroUsize> {
    linux::physical_cores()
}

/// The physical cores this process may run on: not counted on this platform.
#[cfg(not(any(target_os = "linux", target_os = "android")))]
const fn physical_cores() -> Option<NonZeroUsize> {
    None
}

/// Counting physical cores where the kernel describes them in `/sys`: Linux and Android.
///
/// A pure part (parsing the kernel's CPU lists and grouping CPUs into cores) and a thin shell
/// that reads the affinity mask and the topology files.
#[cfg(any(target_os = "linux", target_os = "android"))]
mod linux {
    use std::collections::BTreeSet;
    use std::num::NonZeroUsize;

    use super::logical_cores;

    /// The most CPUs a list may name: lists past it are taken as malformed, so a corrupt range
    /// can't make the parser build a huge set.
    const MAX_CPUS: usize = 1 << 16;

    /// The CPUs a kernel CPU list names, as in `0-3,8,10-11` (whitespace around it ignored),
    /// or `None` if it is empty or malformed: an item that isn't a number or a range `a-b`
    /// with `a ≤ b`, or a CPU past [`MAX_CPUS`].
    pub(super) fn parse_cpu_list(text: &str) -> Option<BTreeSet<usize>> {
        let mut cpus = BTreeSet::new();
        // An empty list is one empty item, which doesn't parse.
        for item in text.trim().split(',') {
            let (low, high): (usize, usize) = if let Some((low, high)) = item.split_once('-') {
                (low.parse().ok()?, high.parse().ok()?)
            } else {
                let cpu = item.parse().ok()?;
                (cpu, cpu)
            };
            if low > high || high >= MAX_CPUS {
                return None;
            }
            cpus.extend(low..=high);
        }
        Some(cpus)
    }

    /// The distinct physical cores among the `allowed` CPUs, given each CPU's `siblings`: the
    /// CPU list of the core it belongs to (the kernel's `core_cpus_list`), `None` if it can't
    /// be read.
    ///
    /// Two CPUs are one core when their sibling lists name the same least CPU. `None` if
    /// `allowed` is empty, or a CPU's list is missing, malformed or doesn't name the CPU
    /// itself: then the cores can't be told.
    pub(super) fn count_cores(
        allowed: &BTreeSet<usize>,
        siblings: impl Fn(usize) -> Option<String>,
    ) -> Option<NonZeroUsize> {
        let mut cores = BTreeSet::new();
        for &cpu in allowed {
            let core = parse_cpu_list(&siblings(cpu)?)?;
            if !core.contains(&cpu) {
                return None;
            }
            cores.insert(core.first().copied()?);
        }
        NonZeroUsize::new(cores.len())
    }

    /// The physical cores this process may run on, at most [`logical_cores`].
    ///
    /// They are the distinct cores among the CPUs its affinity mask allows, grouped by each
    /// CPU's `topology/core_cpus_list` (else `thread_siblings_list`, the older name). `None` if
    /// the mask or a file can't be read or parsed.
    pub(super) fn physical_cores() -> Option<NonZeroUsize> {
        let mask = rustix::thread::sched_getaffinity(None).ok()?;
        let allowed: BTreeSet<usize> = (0..rustix::thread::CpuSet::MAX_CPU)
            .filter(|&cpu| mask.is_set(cpu))
            .collect();
        let topology = |cpu: usize, file: &str| {
            std::fs::read_to_string(format!("/sys/devices/system/cpu/cpu{cpu}/topology/{file}"))
                .ok()
        };
        let cores = count_cores(&allowed, |cpu| {
            topology(cpu, "core_cpus_list").or_else(|| topology(cpu, "thread_siblings_list"))
        })?;
        Some(cores.min(logical_cores()))
    }

    #[cfg(test)]
    #[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
    mod tests {
        use std::collections::BTreeMap;

        use super::*;

        /// A machine's topology files: each online CPU's sibling list, as the kernel writes it.
        fn machine(lists: &[(usize, &str)]) -> BTreeMap<usize, String> {
            lists
                .iter()
                .map(|&(cpu, list)| (cpu, format!("{list}\n")))
                .collect()
        }

        /// The cores among `allowed` (a CPU list, well formed) on `machine`.
        fn cores(machine: &BTreeMap<usize, String>, allowed: &str) -> Option<usize> {
            let allowed = parse_cpu_list(allowed)?;
            count_cores(&allowed, |cpu| machine.get(&cpu).cloned()).map(NonZeroUsize::get)
        }

        /// Four cores of two hardware threads, numbered as Linux numbers a desktop part: CPU
        /// `n` and `n + 4` share a core.
        fn smt() -> BTreeMap<usize, String> {
            machine(&[
                (0, "0,4"),
                (1, "1,5"),
                (2, "2,6"),
                (3, "3,7"),
                (4, "0,4"),
                (5, "1,5"),
                (6, "2,6"),
                (7, "3,7"),
            ])
        }

        /// Lists parse as the kernel writes them (ranges, single CPUs, a trailing newline), and
        /// anything else is refused rather than guessed at.
        #[test]
        fn cpu_lists_parse_as_the_kernel_writes_them() {
            let want: BTreeSet<usize> = [0, 1, 2, 3, 8, 10, 11].into();
            assert_eq!(parse_cpu_list("0-3,8,10-11\n"), Some(want));
            assert_eq!(parse_cpu_list(" 5 "), Some([5].into()));
            assert_eq!(parse_cpu_list("2-2"), Some([2].into()));
            for malformed in [
                "", "\n", "a", "1-", "-1", "3-1", "1,,2", "1-2-3", "0-70000", "1;2",
            ] {
                assert_eq!(parse_cpu_list(malformed), None, "{malformed:?}");
            }
        }

        /// A core's hardware threads count once, whichever of them the mask allows, with SMT on
        /// or off.
        #[test]
        fn hardware_threads_of_one_core_count_once() {
            let smt = smt();
            assert_eq!(cores(&smt, "0-7"), Some(4), "SMT on: two threads a core");
            assert_eq!(
                cores(&smt, "0-3"),
                Some(4),
                "one thread of each core allowed"
            );
            assert_eq!(cores(&smt, "0,4"), Some(1), "both threads of one core");
            assert_eq!(cores(&smt, "0,1,4"), Some(2), "a restricted affinity mask");
            let no_smt = machine(&[(0, "0"), (1, "1"), (2, "2"), (3, "3")]);
            assert_eq!(cores(&no_smt, "0-3"), Some(4), "SMT off: a thread a core");
        }

        /// Offline CPUs have no topology files and aren't in the mask: they are neither
        /// counted nor read, even when an online CPU's core still lists one.
        #[test]
        fn offline_cpus_are_neither_counted_nor_needed() {
            // CPUs 2, 6 and 7 are offline. CPU 3's core lists its offline sibling 7.
            let partly = machine(&[(0, "0,4"), (1, "1,5"), (3, "3,7"), (4, "0,4"), (5, "1,5")]);
            assert_eq!(cores(&partly, "0-1,3-5"), Some(3));
        }

        /// A missing or malformed topology file, or an empty mask, means the cores can't be
        /// told: the caller then falls back instead of guessing.
        #[test]
        fn anything_unreadable_means_the_cores_cant_be_told() {
            let smt = smt();
            assert_eq!(cores(&smt, "0-8"), None, "CPU 8 has no topology file");
            let garbled = machine(&[(0, "0,x"), (1, "1")]);
            assert_eq!(cores(&garbled, "0-1"), None, "a malformed list");
            let elsewhere = machine(&[(0, "1,2"), (1, "1,2")]);
            assert_eq!(
                cores(&elsewhere, "0-1"),
                None,
                "a list without the CPU itself"
            );
            assert_eq!(
                count_cores(&BTreeSet::new(), |_| Some("0".to_owned())),
                None,
                "nothing allowed"
            );
        }

        /// On the machine running the tests, the cores are counted and are at most its
        /// hardware threads.
        #[test]
        fn this_machine_s_cores_are_counted() {
            let cores = physical_cores().expect("Linux describes its cores in /sys");
            assert!(cores <= logical_cores(), "{cores} of {}", logical_cores());
        }
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    /// Every setting asks for at least one worker: physical cores never more than hardware
    /// threads, a fixed count exactly that count, and physical cores the default.
    #[test]
    fn every_setting_counts_at_least_one_worker() {
        let logical = logical_cores();
        let physical = Workers::PhysicalCores.count();
        assert!(physical <= logical, "{physical} cores of {logical} threads");
        assert_eq!(Workers::LogicalCores.count(), logical);
        let three = NonZeroUsize::new(3).unwrap();
        assert_eq!(Workers::Fixed(three).count(), three);
        assert_eq!(Workers::default(), Workers::PhysicalCores);
    }

    /// A setting reads back as it is written, and anything else is refused.
    #[test]
    fn settings_read_back_as_written() {
        let three = Workers::Fixed(NonZeroUsize::new(3).unwrap());
        for workers in [Workers::PhysicalCores, Workers::LogicalCores, three] {
            assert_eq!(workers.to_string().parse(), Ok(workers), "{workers}");
        }
        assert_eq!(three.to_string(), "3");
        for text in ["", "0", "-1", "Physical", "physical ", "3.5", "many"] {
            assert!(text.parse::<Workers>().is_err(), "{text:?}");
        }
    }
}
