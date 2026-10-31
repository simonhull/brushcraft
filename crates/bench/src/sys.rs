//! What the harness asks of the operating system: the process's resident memory and page
//! faults, and dropping a file's pages from the page cache.
//!
//! Linux in full. Elsewhere it builds, but the memory and faults come from `/proc`, so reading
//! them fails, and a file's pages aren't dropped ([`drop_page_cache`] says so).

use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, anyhow};

/// The process's resident memory, from `/proc/self/status`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Rss {
    /// Resident now (`VmRSS`), in bytes.
    pub(crate) now: u64,
    /// The most ever resident (`VmHWM`, the high-water mark), in bytes.
    pub(crate) peak: u64,
}

impl Rss {
    /// Reads the process's resident memory now.
    ///
    /// # Errors
    ///
    /// If `/proc/self/status` can't be read or lacks either line.
    pub(crate) fn read() -> Result<Self> {
        let status =
            std::fs::read_to_string("/proc/self/status").context("reading /proc/self/status")?;
        Self::parse(&status)
    }

    /// The memory `status` (the text of `/proc/self/status`) gives.
    ///
    /// # Errors
    ///
    /// If either line is missing or doesn't read `NAME: N kB`.
    fn parse(status: &str) -> Result<Self> {
        /// The value of the line `name`, in bytes.
        ///
        /// # Errors
        ///
        /// If the line is missing or malformed.
        fn kib(status: &str, name: &str) -> Result<u64> {
            let line = status
                .lines()
                .find_map(|line| line.strip_prefix(name)?.strip_prefix(':'))
                .ok_or_else(|| anyhow!("no {name} in /proc/self/status"))?;
            let value = line
                .trim()
                .strip_suffix("kB")
                .ok_or_else(|| anyhow!("{name} isn't in kB: {line:?}"))?
                .trim()
                .parse::<u64>()
                .with_context(|| format!("{name}: {line:?}"))?;
            Ok(value * 1024)
        }
        Ok(Self {
            now: kib(status, "VmRSS")?,
            peak: kib(status, "VmHWM")?,
        })
    }
}

/// The minor page faults the process (every thread of it) has taken so far: faults served
/// without reading the disk, such as the first touch of a page of freshly mapped memory. Read
/// from `/proc/self/stat`.
///
/// # Errors
///
/// If `/proc/self/stat` can't be read or doesn't parse.
pub(crate) fn minor_faults() -> Result<u64> {
    let stat = std::fs::read_to_string("/proc/self/stat").context("reading /proc/self/stat")?;
    parse_minor_faults(&stat)
}

/// The minor faults `stat` (the text of `/proc/self/stat`) gives: its 10th field.
///
/// The 2nd field, the command name, is in parentheses and may hold spaces and parentheses
/// itself, so the fields are counted from after its last `)`: the 3rd field (the state) is the
/// first there, and the 10th the eighth.
///
/// # Errors
///
/// If there is no `)`, too few fields, or the field isn't a number.
fn parse_minor_faults(stat: &str) -> Result<u64> {
    let (_, after_name) = stat
        .rsplit_once(')')
        .ok_or_else(|| anyhow!("no command name in /proc/self/stat"))?;
    let field = after_name
        .split_whitespace()
        .nth(7)
        .ok_or_else(|| anyhow!("too few fields in /proc/self/stat"))?;
    field
        .parse()
        .with_context(|| format!("minor faults in /proc/self/stat: {field:?}"))
}

/// Resets the process's resident high-water mark (`VmHWM`) to what is resident now.
///
/// It writes `5` to `/proc/self/clear_refs`, so the next [`Rss::read`]'s `peak` is the most
/// resident since this call, exactly: no peak between two samples is missed.
///
/// # Errors
///
/// If `/proc/self/clear_refs` can't be written (a kernel before 4.0, or one without
/// `CONFIG_PROC_PAGE_MONITOR`).
pub(crate) fn reset_peak() -> Result<()> {
    std::fs::write("/proc/self/clear_refs", "5").context("resetting VmHWM in /proc/self/clear_refs")
}

/// Whether [`drop_page_cache`] dropped a file's pages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CacheDrop {
    /// Dropped: the next reads come from the disk.
    #[cfg_attr(
        not(any(target_os = "linux", test)),
        expect(dead_code, reason = "only Linux can drop a file's pages")
    )]
    Dropped,
    /// Not dropped, on a system with no way to: the next reads may come from the cache, so a
    /// "cold" measurement is not guaranteed cold.
    NotGuaranteed,
}

/// Drops `path`'s pages from the page cache, so the next reads of it come from the disk: a cold
/// read.
///
/// On Linux it is `posix_fadvise(DONTNEED)` over the whole file; the kernel drops only clean
/// pages, and the harness calls it on files it has finished writing.
///
/// Elsewhere nothing is dropped ([`CacheDrop::NotGuaranteed`]). macOS has no call that evicts
/// a file's cached pages short of `purge` (all of them, as root) or `msync(MS_INVALIDATE)` over
/// a mapping (`unsafe` here). `fcntl(F_NOCACHE)` only stops caching what is read through the
/// descriptor that sets it, and `F_GLOBAL_NOCACHE` changes how the file is read afterwards
/// rather than emptying the cache, so neither makes the store's next read cold.
///
/// # Errors
///
/// If the file can't be opened or the advice is refused.
pub(crate) fn drop_page_cache(path: &Path) -> Result<CacheDrop> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    drop_pages(&file).with_context(|| format!("dropping {}'s pages", path.display()))
}

/// Drops `file`'s pages from the page cache: `posix_fadvise(DONTNEED)`.
///
/// # Errors
///
/// If the advice is refused.
#[cfg(target_os = "linux")]
fn drop_pages(file: &File) -> rustix::io::Result<CacheDrop> {
    rustix::fs::fadvise(file, 0, None, rustix::fs::Advice::DontNeed)?;
    Ok(CacheDrop::Dropped)
}

/// Can't drop `file`'s pages on this system ([`drop_page_cache`]).
///
/// # Errors
///
/// Never: an `Ok` the Linux version's signature asks for.
#[cfg(not(target_os = "linux"))]
#[expect(
    clippy::unnecessary_wraps,
    reason = "the same signature as Linux's, which can fail"
)]
fn drop_pages(_file: &File) -> rustix::io::Result<CacheDrop> {
    Ok(CacheDrop::NotGuaranteed)
}

/// Samples the process's resident memory on a thread of its own, every [`RssSampler::EVERY`],
/// until stopped.
#[derive(Debug)]
pub(crate) struct RssSampler {
    /// Tells the thread to stop.
    stop: Arc<AtomicBool>,
    /// The thread, which returns its samples: seconds since the start, and bytes resident.
    thread: JoinHandle<Result<Vec<(f64, u64)>>>,
}

impl RssSampler {
    /// How often the sampler reads the resident memory.
    pub(crate) const EVERY: Duration = Duration::from_millis(100);

    /// Starts sampling.
    ///
    /// # Errors
    ///
    /// If the thread can't be started.
    pub(crate) fn start() -> Result<Self> {
        let stop = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&stop);
        let thread = std::thread::Builder::new()
            .name("rss-sampler".to_owned())
            .spawn(move || {
                let start = Instant::now();
                let mut samples = Vec::new();
                loop {
                    samples.push((start.elapsed().as_secs_f64(), Rss::read()?.now));
                    if flag.load(Ordering::Relaxed) {
                        return Ok(samples);
                    }
                    std::thread::sleep(Self::EVERY);
                }
            })
            .context("starting the RSS sampler")?;
        Ok(Self { stop, thread })
    }

    /// Stops sampling (after one last sample) and returns the samples: seconds since the
    /// start, and bytes resident.
    ///
    /// # Errors
    ///
    /// If a read failed, or the thread panicked.
    pub(crate) fn stop(self) -> Result<Vec<(f64, u64)>> {
        self.stop.store(true, Ordering::Relaxed);
        self.thread
            .join()
            .map_err(|panic| anyhow!("the RSS sampler panicked: {panic:?}"))?
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    #[test]
    fn status_lines_parse_into_bytes() {
        let status = "Name:\tx\nVmHWM:\t  2048 kB\nVmRSS:\t   1024 kB\nThreads:\t1\n";
        assert_eq!(
            Rss::parse(status).unwrap(),
            Rss {
                now: 1 << 20,
                peak: 2 << 20
            },
            "kB to bytes"
        );
        assert!(Rss::parse("VmRSS: 1 kB\n").is_err(), "no high-water mark");
        let own = Rss::read().unwrap();
        assert!(own.now > 0 && own.peak >= own.now, "this process: {own:?}");
    }

    #[test]
    fn minor_faults_parse_and_count_first_touches() {
        let stat = "42 (a (b) c) S 1 42 42 0 -1 4194560 1234 0 5 0 10 3";
        assert_eq!(parse_minor_faults(stat).unwrap(), 1234, "the 10th field");
        assert!(parse_minor_faults("42 x S 1").is_err(), "no command name");
        assert!(
            parse_minor_faults("42 (x) S 1 2").is_err(),
            "too few fields"
        );
        let before = minor_faults().unwrap();
        // 16 MiB never touched, then one byte written in each page: at least one minor fault
        // (one a page, or one per huge page where transparent huge pages back it).
        let mut fresh = vec![0_u8; 16 << 20];
        fresh.iter_mut().step_by(4096).for_each(|b| *b = 1);
        let after = minor_faults().unwrap();
        drop(fresh);
        assert!(after > before, "first touches fault: {before} to {after}");
    }

    #[test]
    fn resetting_the_peak_brings_it_down_to_now() {
        // A peak well above what stays resident: 64 MiB touched, then freed.
        let mut big = vec![1_u8; 64 << 20];
        big.iter_mut().step_by(4096).for_each(|b| *b = 2);
        drop(big);
        let before = Rss::read().unwrap();
        reset_peak().unwrap();
        let after = Rss::read().unwrap();
        assert!(
            after.peak < before.peak,
            "the high-water mark came down: {before:?} to {after:?}"
        );
        assert!(
            after.peak >= after.now,
            "still at least what is resident: {after:?}"
        );
    }

    #[test]
    fn the_sampler_samples_until_stopped() {
        let sampler = RssSampler::start().unwrap();
        std::thread::sleep(Duration::from_millis(250));
        let taken = sampler.stop().unwrap();
        assert!(taken.len() >= 2, "a few samples: {taken:?}");
        assert!(taken.iter().all(|&(_, rss)| rss > 0), "all resident");
    }

    #[test]
    fn dropping_a_files_pages_works() {
        let path =
            std::env::temp_dir().join(format!("brushcraft-bench-fadvise-{}", std::process::id()));
        std::fs::write(&path, vec![7_u8; 1 << 16]).unwrap();
        let dropped = drop_page_cache(&path).unwrap();
        let wanted = if cfg!(target_os = "linux") {
            CacheDrop::Dropped
        } else {
            CacheDrop::NotGuaranteed
        };
        assert_eq!(dropped, wanted);
        assert_eq!(
            std::fs::read(&path).unwrap().len(),
            1 << 16,
            "still readable"
        );
        std::fs::remove_file(&path).unwrap();
    }
}
