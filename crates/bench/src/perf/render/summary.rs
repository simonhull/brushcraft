//! `strokes-summary`: the pen's numbers from recorded runs of the app, one row a stroke.
//!
//! The measured runs (`scripts/measure-strokes.sh`) leave a record (`brushcraft --record`) a run:
//! the strokes script's, offscreen or in a window, at a size, or a live session drawn by hand.
//! Each record's strokes are summarised as the performance suite derives them ([`strokes`]), plus
//! what a person comparing runs wants beside them:
//!
//! - **the level jump after pen-up**: the largest level change of the stroke's tiles over the
//!   segment's frames after its last open one (the coarsening as its refresh lands), which the
//!   suite's metric, during the stroke, leaves out;
//! - **tiles a dab**: the tiles the segment's dabs edited over the dabs applied, for checking a
//!   brush's size (the huge stroke's million vertices are about 1,165 tiles a dab on the benchmark
//!   project);
//! - **the pen's frames**, and the dabs applied and shown ahead.
//!
//! A scripted record gives a row for each stroke segment it has ([`strokes::SEGMENTS`]); a live
//! session's (no segments) gives one row, `live`, over all its frames: its drain, time to exact
//! and level jump after pen-up are its last stroke's, and it has no settle frames.
//!
//! Each row is a JSON line (`{"bench":"strokes-summary","result":"stroke",…}`, a number not
//! known `null`) and a line of the human summary, with each figure the spec sets a target for
//! marked as meeting or missing it.

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};

use super::strokes::{self, Drain, HUGE_EXACT_MS, MOST_CARRIED, StrokeMetrics};
use super::{Frame, Recorded};
use crate::report::{Output, Record};

/// The pen's latency target, sample to present submitted at p99, in ms.
const LATENCY_MS: f64 = 8.0;

/// The far stroke's level-jump target: none bigger than one.
const MOST_JUMP: f64 = 1.0;

/// The name of a live session's one row.
const LIVE: &str = "live";

/// One stroke of one record, summarised.
#[derive(Debug, Clone, PartialEq)]
struct Row {
    /// The stroke's segment, or [`LIVE`].
    stroke: &'static str,
    /// The metrics the suite derives.
    metrics: StrokeMetrics,
    /// The largest level jump of the stroke's tiles after its pen-up, within its frames.
    jump_after_up: Option<f64>,
    /// Tiles edited over dabs applied, over its frames; `None` with no dab applied.
    tiles_a_dab: Option<f64>,
    /// Frames that showed a pen sample's dabs first.
    pen_frames: usize,
    /// Dabs applied over its frames: finest work, a catch-up's included.
    dabs_applied: f64,
    /// Dabs shown ahead of their finest work over its frames.
    dabs_shown: f64,
}

/// Every row of `recorded`: one for each stroke segment it has, or one [`LIVE`] row for a live
/// session (no segments).
fn rows(recorded: &Recorded) -> Vec<Row> {
    if recorded.segments.is_empty() {
        let frames: Vec<&Frame> = recorded.frames.iter().collect();
        return vec![row(LIVE, strokes::stroke(&frames), &frames)];
    }
    let derived = strokes::derive(recorded);
    let segments = strokes::SEGMENTS.iter().zip(derived);
    let kept = segments.filter(|(segment, _)| {
        recorded
            .segments
            .iter()
            .any(|(name, _, _)| name == *segment)
    });
    kept.map(|(&segment, metrics)| {
        let frames: Vec<&Frame> = recorded.in_segment(segment).collect();
        row(segment, metrics, &frames)
    })
    .collect()
}

/// The row of stroke `stroke`, its suite metrics `metrics`, from its `frames`.
fn row(stroke: &'static str, metrics: StrokeMetrics, frames: &[&Frame]) -> Row {
    let after_up = frames
        .iter()
        .rposition(|frame| frame.stroke_open == Some(true))
        .and_then(|last| frames.get(last + 1..))
        .unwrap_or_default();
    let jumps = after_up.iter().filter_map(|frame| frame.level_jump);
    let sum =
        |pick: fn(&Frame) -> Option<f64>| -> f64 { frames.iter().filter_map(|f| pick(f)).sum() };
    let applied = sum(|f| f.dabs_applied);
    let edited = sum(|f| f.tiles_edited);
    let pens = frames.iter().filter(|f| f.sample_to_present_ms.is_some());
    Row {
        stroke,
        metrics,
        jump_after_up: jumps.reduce(f64::max),
        tiles_a_dab: (applied > 0.0).then(|| edited / applied),
        pen_frames: pens.count(),
        dabs_applied: applied,
        dabs_shown: sum(|f| f.dabs_shown),
    }
}

/// A timed figure as a number of its unit: never reached is `∞`.
const fn drained(drain: Drain) -> f64 {
    match drain {
        Drain::Frames(n) => n,
        Drain::Not => f64::INFINITY,
    }
}

impl Row {
    /// The row as a JSON record of the record at `file`, run on `recorded`'s adapter and size.
    fn record(&self, file: &Path, recorded: &Recorded) -> Record {
        let m = &self.metrics;
        let number = |value: Option<f64>| value.unwrap_or(f64::NAN);
        let reached = |drain: Option<Drain>| match drain {
            Some(Drain::Frames(n)) => n,
            Some(Drain::Not) | None => f64::NAN,
        };
        let (width, height) = recorded.size.unwrap_or((f64::NAN, f64::NAN));
        Record::new("strokes-summary", "stroke")
            .with("file", file.display().to_string())
            .with("project", recorded.project.as_str())
            .with("adapter", recorded.adapter.as_str())
            .with("width", width)
            .with("height", height)
            .with("stroke", self.stroke)
            .with("pen_frames", self.pen_frames)
            .with("sample_to_present_p50_ms", number(m.sample_to_present_p50))
            .with("sample_to_present_p99_ms", number(m.sample_to_present_p99))
            .with(
                "start_to_submitted_p99_ms",
                number(m.start_to_submitted_p99),
            )
            .with("brush_p99_ms", number(m.brush_p99))
            .with("dabs_carried_max", number(m.most_carried))
            .with("drain_frames", reached(m.drain))
            .with("drained", matches!(m.drain, Some(Drain::Frames(_))))
            .with("frames_above_f", number(m.above_f))
            .with("catch_up_p99_ms", number(m.catch_up_p99))
            .with("catching_up_max", number(m.most_catching_up))
            .with("exact_after_up_ms", reached(m.exact_after_up))
            .with("exact", matches!(m.exact_after_up, Some(Drain::Frames(_))))
            .with("settle_frames", reached(m.settle))
            .with("level_jump_max", number(m.most_level_jump))
            .with("level_jump_after_up_max", number(self.jump_after_up))
            .with("tiles_a_dab", number(self.tiles_a_dab))
            .with("dabs_applied", self.dabs_applied)
            .with("dabs_shown", self.dabs_shown)
    }

    /// The row as a line of the human summary, each figure with a target marked.
    fn line(&self) -> String {
        let m = &self.metrics;
        let ms = |value: Option<f64>| value.map_or_else(|| "–".to_owned(), |v| format!("{v:.2}"));
        let whole = |value: Option<f64>| {
            value.map_or_else(
                || "–".to_owned(),
                |v| {
                    if v.is_finite() {
                        format!("{v:.0}")
                    } else {
                        "never".to_owned()
                    }
                },
            )
        };
        let judged = |value: Option<f64>, ceiling: f64| match value {
            Some(v) if v <= ceiling => " (meets)",
            Some(_) => " (MISSES)",
            None => "",
        };
        let huge = if self.stroke == "stroke_huge" {
            judged(m.exact_after_up.map(drained), HUGE_EXACT_MS)
        } else {
            ""
        };
        let far = if self.stroke == "stroke_far" {
            judged(m.most_level_jump, MOST_JUMP)
        } else {
            ""
        };
        let tiles = self
            .tiles_a_dab
            .map_or_else(|| "–".to_owned(), |t| format!("{t:.1}"));
        let parts = [
            format!(
                "  {:<13} {:>4} pen frames  sample to present p50 {} p99 {} ms{}",
                self.stroke,
                self.pen_frames,
                ms(m.sample_to_present_p50),
                ms(m.sample_to_present_p99),
                judged(m.sample_to_present_p99, LATENCY_MS),
            ),
            format!(
                "frame p99 {} ms, brush p99 {} ms, catch-up p99 {} ms",
                ms(m.start_to_submitted_p99),
                ms(m.brush_p99),
                ms(m.catch_up_p99),
            ),
            format!(
                "carried ≤ {}{}, drained in {} frames",
                whole(m.most_carried),
                judged(m.most_carried, MOST_CARRIED),
                whole(m.drain.map(drained)),
            ),
            format!("above F {}", whole(m.above_f)),
            format!(
                "catching up ≤ {}, exact {} ms after up{huge}, settled in {} frames",
                whole(m.most_catching_up),
                whole(m.exact_after_up.map(drained)),
                whole(m.settle.map(drained)),
            ),
            format!(
                "level jump ≤ {}{far} ({} after up)",
                whole(m.most_level_jump),
                whole(self.jump_after_up),
            ),
            format!(
                "{tiles} tiles a dab, {} dabs applied, {} shown ahead",
                whole(Some(self.dabs_applied)),
                whole(Some(self.dabs_shown)),
            ),
        ];
        parts.join("; ")
    }
}

/// The records in `dir`: every `.jsonl` file in it, in name order.
///
/// # Errors
///
/// If the directory can't be read.
fn records_in(dir: &Path) -> Result<Vec<PathBuf>> {
    let entries = std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))?;
    let mut files: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "jsonl"))
        .collect();
    files.sort();
    Ok(files)
}

/// Summarises every record in `dir` ([`records_in`]) to `out`: a heading a record, then its rows
/// ([`rows`]).
///
/// # Errors
///
/// If the directory can't be read, holds no record, a record doesn't read, or writing fails.
pub(crate) fn run<J: Write, H: Write>(dir: &Path, out: &mut Output<J, H>) -> Result<()> {
    let files = records_in(dir)?;
    anyhow::ensure!(
        !files.is_empty(),
        "no records (*.jsonl) in {}",
        dir.display()
    );
    for file in files {
        let text = std::fs::read_to_string(&file)
            .with_context(|| format!("reading {}", file.display()))?;
        let recorded = Recorded::parse(&text).with_context(|| format!("{}", file.display()))?;
        let size = recorded
            .size
            .map_or_else(String::new, |(w, h)| format!(", {w}×{h}"));
        out.say(&format!(
            "{}: {}{size}, {}",
            file.display(),
            recorded.adapter,
            recorded.project
        ))?;
        for row in rows(&recorded) {
            out.record(&row.record(&file, &recorded))?;
            out.say(&row.line())?;
        }
    }
    Ok(())
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    /// A frame of `segment` at `time_ms`, its stroke `open`, showing dabs with `latency` (if
    /// any), applying `applied` dabs over `tiles` tiles, its cut's jump `jump`.
    fn frame(
        segment: Option<&str>,
        time_ms: f64,
        open: bool,
        latency: Option<f64>,
        (applied, tiles): (f64, f64),
        jump: Option<f64>,
    ) -> Frame {
        Frame {
            time_ms,
            segment: segment.map(str::to_owned),
            submitted_ms: Some(time_ms + 2.0),
            sample_to_present_ms: latency,
            brush_ms: Some(1.0),
            dabs_applied: Some(applied),
            dabs_shown: Some(0.0),
            dabs_carried: Some(0.0),
            dabs_catching_up: Some(0.0),
            tiles_edited: Some(tiles),
            stroke_open: Some(open),
            level_jump: jump,
            ..Frame::default()
        }
    }

    #[test]
    fn a_scripted_record_gives_a_row_a_stroke_it_has() {
        let far = Some("stroke_far");
        let recorded = Recorded {
            segments: vec![
                ("open".to_owned(), 0.0, 10.0),
                ("stroke_far".to_owned(), 10.0, 100.0),
            ],
            frames: vec![
                frame(far, 10.0, true, Some(4.0), (2.0, 30.0), Some(1.0)),
                frame(far, 20.0, true, Some(6.0), (2.0, 10.0), Some(0.0)),
                frame(far, 30.0, false, None, (0.0, 0.0), Some(3.0)),
            ],
            ..Recorded::default()
        };
        let rows = rows(&recorded);
        assert_eq!(rows.len(), 1, "only the far stroke is in the record");
        let row = &rows[0];
        assert_eq!(row.stroke, "stroke_far");
        assert_eq!(row.pen_frames, 2);
        assert_eq!(
            row.metrics.sample_to_present_p50,
            Some(6.0),
            "nearest rank above"
        );
        assert_eq!(row.metrics.most_level_jump, Some(1.0), "while open");
        assert_eq!(row.jump_after_up, Some(3.0), "after pen-up");
        assert_eq!(row.tiles_a_dab, Some(10.0), "40 tiles over 4 dabs");
        let line = row.line();
        assert!(
            line.contains("level jump ≤ 1 (meets) (3 after up)"),
            "{line}"
        );
        assert!(line.contains("p99 6.00 ms (meets)"), "{line}");
        let json = row.record(Path::new("r.jsonl"), &recorded).to_json();
        assert!(json.contains(r#""stroke":"stroke_far""#), "{json}");
        assert!(
            json.contains(r#""settle_frames":null"#),
            "never settled: {json}"
        );
    }

    #[test]
    fn a_live_session_is_one_row_over_all_its_frames() {
        let recorded = Recorded {
            frames: vec![
                frame(None, 0.0, true, Some(9.0), (1.0, 5.0), None),
                frame(None, 10.0, true, Some(3.0), (1.0, 5.0), None),
                frame(None, 20.0, false, None, (0.0, 0.0), None),
            ],
            ..Recorded::default()
        };
        let rows = rows(&recorded);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].stroke, LIVE);
        assert_eq!(rows[0].metrics.sample_to_present_p99, Some(9.0));
        assert!(rows[0].line().contains("p99 9.00 ms (MISSES)"), "over 8 ms");
    }
}
