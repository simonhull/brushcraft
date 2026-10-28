//! What the load ([`crate::load`]) did each frame, written beside the samples' CSV as
//! `<stem>-load.csv` and read back by `--report`.

use std::fmt::{self, Write as _};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context as _, Result};

/// The load CSV's header.
const HEADER: &str = "start_s,interval_ms,dabs,skipped,brush_ms,stroke_end_ms,feed_ms,upkeep_ms,\
                      jobs_ms,slices,longest_slice_ms,busy_ms";

/// One frame of the load, on the main thread.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct FrameRecord {
    /// When it started, since the probe started.
    pub(crate) start: Duration,
    /// The frame interval it was paced to.
    pub(crate) interval: Duration,
    /// Draw steps it ran.
    pub(crate) dabs: u32,
    /// Draw steps due that it dropped, being too far behind.
    pub(crate) skipped: u32,
    /// The draw steps, stroke ends excluded.
    pub(crate) brush: Duration,
    /// Ending a stroke and beginning the next (`end_stroke` hands the stroke's edits to the
    /// recovery writer), if one ended.
    pub(crate) stroke_end: Duration,
    /// Draining the stale-tile feed.
    pub(crate) feed: Duration,
    /// Ticking and polling the project: the recovery flushes.
    pub(crate) upkeep: Duration,
    /// The store's jobs: display refresh, and the write-through of display records.
    pub(crate) jobs: Duration,
    /// Job slices run.
    pub(crate) slices: usize,
    /// The longest job slice.
    pub(crate) longest_slice: Duration,
    /// The whole frame, start to end: how long the main thread was away from its events.
    pub(crate) busy: Duration,
}

/// Where the load CSV of the samples' CSV `csv` goes: `<stem>-load.csv` beside it.
pub(crate) fn path_for(csv: &Path) -> PathBuf {
    let stem = csv.file_stem().map_or_else(
        || "pen-samples".into(),
        |s| s.to_string_lossy().into_owned(),
    );
    csv.with_file_name(format!("{stem}-load.csv"))
}

/// Milliseconds, for the CSV.
fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// The CSV text for `frames`.
///
/// # Errors
///
/// Only if formatting fails, which writing to a `String` never does.
pub(crate) fn to_csv(frames: &[FrameRecord]) -> Result<String, fmt::Error> {
    let mut out = format!("{HEADER}\n");
    for f in frames {
        writeln!(
            out,
            "{:.6},{:.4},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{},{:.4},{:.4}",
            f.start.as_secs_f64(),
            ms(f.interval),
            f.dabs,
            f.skipped,
            ms(f.brush),
            ms(f.stroke_end),
            ms(f.feed),
            ms(f.upkeep),
            ms(f.jobs),
            f.slices,
            ms(f.longest_slice),
            ms(f.busy),
        )?;
    }
    Ok(out)
}

/// Writes `frames` to `path`.
///
/// # Errors
///
/// If the file can't be written.
pub(crate) fn write(frames: &[FrameRecord], path: &Path) -> Result<()> {
    std::fs::write(path, to_csv(frames)?).with_context(|| format!("writing {}", path.display()))
}

/// Reads the load CSV at `path`, if there is one.
///
/// # Errors
///
/// If it exists but can't be read or parsed.
pub(crate) fn read(path: &Path) -> Result<Option<Vec<FrameRecord>>> {
    if !path.exists() {
        return Ok(None);
    }
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    parse(&text)
        .with_context(|| format!("parsing {}", path.display()))
        .map(Some)
}

/// Parses load CSV text.
///
/// # Errors
///
/// If the header isn't the load CSV's, or a row doesn't parse; the message names the line.
pub(crate) fn parse(text: &str) -> Result<Vec<FrameRecord>> {
    let mut lines = text.lines();
    let header = lines.next().context("the file is empty")?;
    anyhow::ensure!(
        header.trim() == HEADER,
        "not a load CSV: expected the header {HEADER}"
    );
    lines
        .zip(2_usize..)
        .filter(|(row, _)| !row.trim().is_empty())
        .map(|(row, line)| parse_row(row).with_context(|| format!("line {line}")))
        .collect()
}

/// Parses one row.
///
/// # Errors
///
/// If it has the wrong number of fields or one doesn't parse.
fn parse_row(row: &str) -> Result<FrameRecord> {
    let fields: Vec<&str> = row.split(',').map(str::trim).collect();
    let [
        start,
        interval,
        dabs,
        skipped,
        brush,
        stroke_end,
        feed,
        upkeep,
        jobs,
        slices,
        longest,
        busy,
    ] = fields.as_slice()
    else {
        anyhow::bail!("expected 12 fields, got {}", fields.len());
    };
    let secs = |f: &str, scale: f64| -> Result<Duration> {
        let v: f64 = f.parse().with_context(|| format!("bad number {f:?}"))?;
        Duration::try_from_secs_f64(v / scale).with_context(|| format!("bad duration {f:?}"))
    };
    Ok(FrameRecord {
        start: secs(start, 1.0)?,
        interval: secs(interval, 1000.0)?,
        dabs: dabs.parse().with_context(|| format!("bad dabs {dabs:?}"))?,
        skipped: skipped
            .parse()
            .with_context(|| format!("bad skipped {skipped:?}"))?,
        brush: secs(brush, 1000.0)?,
        stroke_end: secs(stroke_end, 1000.0)?,
        feed: secs(feed, 1000.0)?,
        upkeep: secs(upkeep, 1000.0)?,
        jobs: secs(jobs, 1000.0)?,
        slices: slices
            .parse()
            .with_context(|| format!("bad slices {slices:?}"))?,
        longest_slice: secs(longest, 1000.0)?,
        busy: secs(busy, 1000.0)?,
    })
}

#[cfg(test)]
#[expect(
    clippy::missing_panics_doc,
    reason = "test functions are documented by their names; panicking is how they fail"
)]
mod tests {
    use super::*;

    #[test]
    fn the_load_csv_sits_beside_the_samples() {
        assert_eq!(
            path_for(Path::new("/x/winit.csv")),
            PathBuf::from("/x/winit-load.csv")
        );
    }

    #[test]
    fn frames_round_trip() {
        let frames = vec![
            FrameRecord {
                start: Duration::from_millis(1500),
                interval: Duration::from_micros(16_667),
                dabs: 3,
                skipped: 0,
                brush: Duration::from_micros(4_250),
                stroke_end: Duration::ZERO,
                feed: Duration::from_micros(10),
                upkeep: Duration::from_micros(800),
                jobs: Duration::from_micros(7_000),
                slices: 9,
                longest_slice: Duration::from_micros(1_100),
                busy: Duration::from_micros(12_500),
            },
            FrameRecord::default(),
        ];
        let parsed = parse(&to_csv(&frames).expect("a String")).expect("parses");
        assert_eq!(parsed.len(), 2);
        for (a, b) in parsed.iter().zip(&frames) {
            assert_eq!((a.dabs, a.slices), (b.dabs, b.slices));
            assert!(a.busy.abs_diff(b.busy) < Duration::from_nanos(200), "{a:?}");
            assert!(
                a.start.abs_diff(b.start) < Duration::from_micros(2),
                "{a:?}"
            );
        }
        parse("start_s\n1\n").unwrap_err();
        parse(&format!("{HEADER}\n1,2,3\n")).unwrap_err();
    }
}
