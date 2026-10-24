//! What a run reports: one JSON line per result, a human summary, and progress.
//!
//! The JSON is written by hand: a record is a flat object of named numbers, strings and number
//! arrays, all the harness needs. Every record carries the subcommand that wrote it (`"bench"`)
//! and what it measured (`"result"`), so the lines of several runs can be concatenated and
//! still be told apart.

use std::io::Write;
use std::time::{Duration, Instant};

use anyhow::Result;

/// A value in a JSON record.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Value {
    /// A string.
    Str(String),
    /// An unsigned integer.
    Int(u64),
    /// A float; written as `null` if it isn't finite (JSON has no infinities or NaN).
    Float(f64),
    /// An array of floats, each written as [`Value::Float`] is.
    Floats(Vec<f64>),
    /// `true` or `false`.
    Bool(bool),
}

impl From<&str> for Value {
    fn from(s: &str) -> Self {
        Self::Str(s.to_owned())
    }
}

impl From<String> for Value {
    fn from(s: String) -> Self {
        Self::Str(s)
    }
}

impl From<u64> for Value {
    fn from(n: u64) -> Self {
        Self::Int(n)
    }
}

impl From<usize> for Value {
    fn from(n: usize) -> Self {
        Self::Int(u64::try_from(n).unwrap_or(u64::MAX))
    }
}

impl From<f64> for Value {
    fn from(x: f64) -> Self {
        Self::Float(x)
    }
}

impl From<bool> for Value {
    fn from(b: bool) -> Self {
        Self::Bool(b)
    }
}

impl From<Vec<f64>> for Value {
    fn from(xs: Vec<f64>) -> Self {
        Self::Floats(xs)
    }
}

/// `s` as a JSON string literal, quotes included.
fn quoted(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if u32::from(c) < 0x20 => {
                let code = u32::from(c);
                out.push_str("\\u00");
                out.extend(
                    [code >> 4, code & 0xF]
                        .into_iter()
                        .filter_map(|digit| char::from_digit(digit, 16)),
                );
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// `x` as a JSON number, or `null` if it isn't finite.
fn number(x: f64) -> String {
    if x.is_finite() {
        format!("{x}")
    } else {
        "null".to_owned()
    }
}

/// One result: a flat JSON object, its fields in the order they were added.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Record {
    /// The fields.
    fields: Vec<(String, Value)>,
}

impl Record {
    /// A record of `bench`'s result `result`.
    pub(crate) fn new(bench: &str, result: &str) -> Self {
        Self {
            fields: vec![
                ("bench".to_owned(), bench.into()),
                ("result".to_owned(), result.into()),
            ],
        }
    }

    /// The record with `name` set to `value` too.
    #[must_use]
    pub(crate) fn with(mut self, name: impl Into<String>, value: impl Into<Value>) -> Self {
        self.fields.push((name.into(), value.into()));
        self
    }

    /// The record as one line of JSON, without the newline.
    pub(crate) fn to_json(&self) -> String {
        let fields: Vec<String> = self
            .fields
            .iter()
            .map(|(name, value)| {
                let value = match value {
                    Value::Str(s) => quoted(s),
                    Value::Int(n) => n.to_string(),
                    Value::Bool(b) => b.to_string(),
                    Value::Float(x) => number(*x),
                    Value::Floats(xs) => {
                        let xs: Vec<String> = xs.iter().map(|&x| number(x)).collect();
                        format!("[{}]", xs.join(","))
                    }
                };
                format!("{}:{value}", quoted(name))
            })
            .collect();
        format!("{{{}}}", fields.join(","))
    }
}

/// Where a run reports: JSON lines to one writer, the human summary and progress to another.
#[derive(Debug)]
pub(crate) struct Output<J, H> {
    /// The JSON lines: stdout in the binary.
    json: J,
    /// Fields every JSON line ends with, whatever its result: the run's setting (see
    /// [`Output::stamp`]).
    stamps: Vec<(String, Value)>,
    /// The human summary and progress: stderr in the binary.
    human: H,
    /// When the last progress line was written, to write at most one every
    /// [`PROGRESS_EVERY`].
    last_progress: Option<Instant>,
}

/// The least time between two progress lines.
const PROGRESS_EVERY: Duration = Duration::from_secs(2);

impl<J: Write, H: Write> Output<J, H> {
    /// Reports to `json` and `human`.
    pub(crate) const fn new(json: J, human: H) -> Self {
        Self {
            json,
            human,
            stamps: Vec::new(),
            last_progress: None,
        }
    }

    /// Ends every JSON line written from now on with `name` set to `value`: for what holds for
    /// the whole run (its evaluation workers), so no result can leave it out.
    pub(crate) fn stamp(&mut self, name: impl Into<String>, value: impl Into<Value>) {
        self.stamps.push((name.into(), value.into()));
    }

    /// Writes `record` as a JSON line.
    ///
    /// # Errors
    ///
    /// If writing fails.
    pub(crate) fn record(&mut self, record: &Record) -> Result<()> {
        let stamped = self
            .stamps
            .iter()
            .fold(record.clone(), |line, (name, value)| {
                line.with(name.clone(), value.clone())
            });
        writeln!(self.json, "{}", stamped.to_json())?;
        self.json.flush()?;
        Ok(())
    }

    /// Writes a line of the human summary.
    ///
    /// # Errors
    ///
    /// If writing fails.
    pub(crate) fn say(&mut self, line: &str) -> Result<()> {
        writeln!(self.human, "{line}")?;
        Ok(())
    }

    /// Writes a progress line of `what`: `done` of `total`, if the last one was written at least
    /// [`PROGRESS_EVERY`] ago or this is the last.
    ///
    /// # Errors
    ///
    /// If writing fails.
    pub(crate) fn progress(
        &mut self,
        what: &str,
        done: usize,
        total: usize,
        since: Instant,
    ) -> Result<()> {
        let now = Instant::now();
        let due = self
            .last_progress
            .is_none_or(|last| now.duration_since(last) >= PROGRESS_EVERY);
        if due || done == total {
            self.last_progress = Some(now);
            let percent = if total == 0 {
                100.0
            } else {
                to_f64(done) * 100.0 / to_f64(total)
            };
            writeln!(
                self.human,
                "  {what}: {done}/{total} ({percent:.1} %), {:.1} s",
                since.elapsed().as_secs_f64()
            )?;
        }
        Ok(())
    }

    /// The writers, for tests to read back.
    #[cfg(test)]
    pub(crate) fn into_parts(self) -> (J, H) {
        (self.json, self.human)
    }
}

/// `n` as a float, for statistics and rates.
#[expect(
    clippy::cast_precision_loss,
    reason = "counts past 2^53 round, which only perturbs a reported statistic"
)]
pub(crate) const fn to_f64(n: usize) -> f64 {
    n as f64
}

/// `d` in milliseconds.
pub(crate) fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// Bytes in MiB.
pub(crate) fn mib(bytes: u64) -> f64 {
    // Up to 2^53 bytes are exact, far beyond any file or RSS here.
    #[expect(
        clippy::cast_precision_loss,
        reason = "byte counts below 2^53 convert exactly"
    )]
    let bytes = bytes as f64;
    bytes / f64::from(1_u32 << 20)
}

/// The order statistics of a set of durations: what a measurement with several samples reports.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Spread {
    /// How many samples.
    pub(crate) samples: usize,
    /// The least, in milliseconds.
    pub(crate) min: f64,
    /// The median (the mean of the middle two for an even count), in milliseconds.
    pub(crate) median: f64,
    /// The 99th percentile (nearest rank), in milliseconds.
    pub(crate) p99: f64,
    /// The greatest, in milliseconds.
    pub(crate) max: f64,
    /// The mean, in milliseconds.
    pub(crate) mean: f64,
}

impl Spread {
    /// The spread of `samples`; `None` if there are none.
    pub(crate) fn of(samples: &[Duration]) -> Option<Self> {
        let mut sorted: Vec<f64> = samples.iter().map(|&d| ms(d)).collect();
        sorted.sort_by(f64::total_cmp);
        let n = sorted.len();
        let (&min, &max) = (sorted.first()?, sorted.last()?);
        let middle = sorted.get((n - 1) / 2..=n / 2)?;
        let median = middle.iter().sum::<f64>() / to_f64(middle.len());
        // Nearest rank: the least sample with at least 99 % of them at or below it.
        let rank = (99 * n).div_ceil(100).max(1);
        let p99 = *sorted.get(rank - 1)?;
        let mean = sorted.iter().sum::<f64>() / to_f64(n);
        Some(Self {
            samples: n,
            min,
            median,
            p99,
            max,
            mean,
        })
    }

    /// `record` with this spread's fields added: `{prefix}_min_ms`, `{prefix}_median_ms`,
    /// `{prefix}_p99_ms`, `{prefix}_max_ms`, `{prefix}_mean_ms` and `{prefix}_samples`.
    pub(crate) fn add_to(&self, record: Record, prefix: &str) -> Record {
        record
            .with(format!("{prefix}_min_ms"), self.min)
            .with(format!("{prefix}_median_ms"), self.median)
            .with(format!("{prefix}_p99_ms"), self.p99)
            .with(format!("{prefix}_max_ms"), self.max)
            .with(format!("{prefix}_mean_ms"), self.mean)
            .with(format!("{prefix}_samples"), self.samples)
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    #[test]
    fn records_are_one_line_of_json() {
        let r = Record::new("brush", "step")
            .with("name", "a \"b\"\n")
            .with("n", 3_u64)
            .with("x", 1.5)
            .with("bad", f64::NAN)
            .with("xs", vec![1.0, 2.25])
            .with("yes", true);
        assert_eq!(
            r.to_json(),
            r#"{"bench":"brush","result":"step","name":"a \"b\"\n","n":3,"x":1.5,"bad":null,"xs":[1,2.25],"yes":true}"#,
            "fields in order, strings escaped, NaN as null"
        );
    }

    /// A stamp ends every JSON line written after it, whatever the result, and never the
    /// human summary.
    #[test]
    fn stamps_end_every_json_line() {
        let mut out = Output::new(Vec::new(), Vec::new());
        out.stamp("workers", "physical");
        out.stamp("worker_threads", 16_usize);
        out.record(&Record::new("brush", "index")).unwrap();
        out.record(&Record::new("brush", "stroke").with("n", 3_u64))
            .unwrap();
        out.say("a summary").unwrap();
        let (json, human) = out.into_parts();
        assert_eq!(
            String::from_utf8(json).unwrap(),
            "{\"bench\":\"brush\",\"result\":\"index\",\"workers\":\"physical\",\"worker_threads\":16}\n\
             {\"bench\":\"brush\",\"result\":\"stroke\",\"n\":3,\"workers\":\"physical\",\"worker_threads\":16}\n"
        );
        assert_eq!(String::from_utf8(human).unwrap(), "a summary\n");
    }

    #[test]
    fn spreads_are_order_statistics() {
        let ms = |v: u64| Duration::from_millis(v);
        let s = Spread::of(&[ms(4), ms(1), ms(3), ms(2)]).unwrap();
        assert!((s.median - 2.5).abs() < 1e-9, "even count: {s:?}");
        assert!(
            (s.min - 1.0).abs() < 1e-9 && (s.max - 4.0).abs() < 1e-9,
            "{s:?}"
        );
        assert!((s.p99 - 4.0).abs() < 1e-9, "p99 of four is the top: {s:?}");
        let many: Vec<Duration> = (1..=200).map(ms).collect();
        let hundreds = Spread::of(&many).unwrap();
        assert!(
            (hundreds.p99 - 198.0).abs() < 1e-9,
            "nearest rank: {hundreds:?}"
        );
        assert!((hundreds.median - 100.5).abs() < 1e-9, "{hundreds:?}");
        assert!(Spread::of(&[]).is_none(), "no samples");
    }
}
