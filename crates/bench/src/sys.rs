//! What the harness asks of the operating system: the process's resident memory, and dropping
//! a file's pages from the page cache. Linux only for now (macOS comes with the `kit` runs).

use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, anyhow};
use rustix::fs::{Advice, fadvise};

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

/// Drops `path`'s pages from the page cache (`posix_fadvise(DONTNEED)` over the whole file),
/// so the next reads of it come from the disk: a cold read.
///
/// The kernel drops only clean pages; the harness calls it on files it has finished writing.
///
/// # Errors
///
/// If the file can't be opened or the advice is refused.
pub(crate) fn drop_page_cache(path: &Path) -> Result<()> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    fadvise(&file, 0, None, Advice::DontNeed)
        .with_context(|| format!("dropping {}'s pages", path.display()))?;
    Ok(())
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
        drop_page_cache(&path).unwrap();
        assert_eq!(
            std::fs::read(&path).unwrap().len(),
            1 << 16,
            "still readable"
        );
        std::fs::remove_file(&path).unwrap();
    }
}
