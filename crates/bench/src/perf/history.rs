//! The suite's history: one JSON line per metric per run, appended to
//! `research/platform/perf/<machine>.jsonl` in the docs repository.
//!
//! Each line is a whole record of its metric's run (when, which commit, which machine, how
//! loaded, the value), so a line can be read, compared or plotted alone. Baselines are lines
//! marked `"baseline":true`; the latest baseline line of a metric, on the same hardware, is what
//! a run is compared against.

use std::collections::BTreeMap;
use std::fmt;
use std::io::Write as _;
use std::path::Path;

use anyhow::{Context as _, Result, anyhow, bail};
use store::Workers;

use crate::perf::json::{Json, Object};
use crate::perf::machine::{Commit, Fingerprint, Timestamp};
use crate::perf::metric::{Measured, Metric, MetricName, Unit};
use crate::perf::quiet::{Load, Settled};
use crate::report::Record;

/// How quiet the machine was for a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Label {
    /// The quiet guard found the others' load below its threshold before every part.
    Quiet,
    /// Before at least one part it never did: the numbers carry other work's noise, and the
    /// run can't be a baseline.
    Loaded,
    /// The guard was off (the tiny project's check that the suite works).
    Unchecked,
}

impl Label {
    /// Every label.
    const ALL: [Self; 3] = [Self::Quiet, Self::Loaded, Self::Unchecked];

    /// The label's name in the records.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Quiet => "quiet",
            Self::Loaded => "loaded",
            Self::Unchecked => "unchecked",
        }
    }

    /// The label named `name`.
    fn named(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|label| label.name() == name)
    }

    /// The run's label once the guard settled `settled` before its next part: a quiet run
    /// becomes loaded if the machine never settled, and stays so whatever later parts find.
    #[must_use]
    pub(crate) const fn after(self, settled: Settled) -> Self {
        match (self, settled) {
            (Self::Quiet, Settled::Loaded { .. }) => Self::Loaded,
            (
                Self::Quiet | Self::Loaded | Self::Unchecked,
                Settled::Quiet { .. } | Settled::Loaded { .. },
            ) => self,
        }
    }

    /// Whether a run so labelled may become a baseline: only a quiet one, so a baseline never
    /// carries other work's noise.
    pub(crate) const fn allows_baseline(self) -> bool {
        matches!(self, Self::Quiet)
    }
}

impl fmt::Display for Label {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// What every metric of a run shares: when, what code, where, how loaded.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RunInfo {
    /// When the run started; with the machine, it names the run.
    pub(crate) run: Timestamp,
    /// The commit measured.
    pub(crate) commit: Commit,
    /// Whether tracked files differed from it.
    pub(crate) dirty: bool,
    /// The machine's name.
    pub(crate) machine: String,
    /// Its hardware fingerprint (see [`crate::perf::machine::Machine`]).
    pub(crate) fingerprint: Fingerprint,
    /// Its kernel release.
    pub(crate) kernel: String,
    /// The evaluation workers setting.
    pub(crate) workers: Workers,
    /// The worker threads it gave.
    pub(crate) worker_threads: usize,
    /// How quiet the machine was.
    pub(crate) label: Label,
    /// Whether the run is a baseline.
    pub(crate) baseline: bool,
}

/// The load around one measured part, as the quiet guard and the load log saw it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PartLoad {
    /// The 1-minute load average as the part started.
    pub(crate) start: Load,
    /// That, less the suite's own share: the other processes' load the guard waited on.
    pub(crate) others: Load,
    /// The other processes' highest load during the part and as it ended, judged after it
    /// ([`crate::perf::quiet::others_peak`]); at or above 2 it labels the run loaded.
    pub(crate) others_peak: Load,
    /// The highest 1-minute load average the log sampled during the part.
    pub(crate) max: Load,
}

/// One history line: a metric of a run, with the run's details.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Entry {
    /// The run.
    pub(crate) info: RunInfo,
    /// The project measured: `medium-<cache key>`, the full project's file and size, or `none`
    /// for the stage benches.
    pub(crate) project: String,
    /// The load around the part that measured the metric.
    pub(crate) load: PartLoad,
    /// The metric.
    pub(crate) metric: Metric,
}

impl Entry {
    /// The entry as a JSON record (`{"bench":"perf","result":"metric",…}`).
    pub(crate) fn to_record(&self) -> Record {
        let info = &self.info;
        let metric = &self.metric;
        let record = Record::new("perf", "metric")
            .with("metric", metric.name.as_str())
            .with("unit", metric.unit.name())
            .with("median", metric.value.median())
            .with("spread", metric.value.spread())
            .with("samples", metric.value.samples().to_vec())
            .with("run", info.run.as_str())
            .with("label", info.label.name())
            .with("baseline", info.baseline)
            .with("commit", info.commit.as_str())
            .with("dirty", info.dirty)
            .with("machine", info.machine.as_str())
            .with("fingerprint", info.fingerprint.as_str())
            .with("kernel", info.kernel.as_str())
            .with("workers", info.workers.to_string())
            .with("worker_threads", info.worker_threads)
            .with("project", self.project.as_str())
            .with("load_start", self.load.start.get())
            .with("load_others", self.load.others.get())
            .with("load_others_peak", self.load.others_peak.get())
            .with("load_max", self.load.max.get());
        match metric.ceiling {
            Some(ceiling) => record.with("ceiling", ceiling),
            None => record,
        }
    }

    /// The entry a history line holds.
    ///
    /// # Errors
    ///
    /// If the line isn't a flat JSON object, isn't a `perf` metric record, or lacks a field or
    /// has one of the wrong type. Fields it doesn't know are ignored, so later versions can add
    /// some.
    pub(crate) fn parse(line: &str) -> Result<Self> {
        let object = Object::parse(line)?;
        let get = |name: &str| object.get(name).ok_or_else(|| anyhow!("no {name:?}"));
        let text = |name: &str| -> Result<String> {
            Ok(get(name)?
                .as_str()
                .ok_or_else(|| anyhow!("{name:?} isn't a string"))?
                .to_owned())
        };
        let number = |name: &str| -> Result<f64> {
            get(name)?
                .as_number()
                .ok_or_else(|| anyhow!("{name:?} isn't a number"))
        };
        let load = |name: &str| -> Result<Load> { Ok(Load::new(number(name)?)?) };
        let flag = |name: &str| -> Result<bool> {
            get(name)?
                .as_bool()
                .ok_or_else(|| anyhow!("{name:?} isn't true or false"))
        };
        if (text("bench")?, text("result")?) != ("perf".to_owned(), "metric".to_owned()) {
            bail!("not a perf metric record");
        }
        let samples = match get("samples")? {
            Json::Array(items) => items
                .iter()
                .map(|x| {
                    x.as_number()
                        .ok_or_else(|| anyhow!("a sample isn't a number"))
                })
                .collect::<Result<Vec<_>>>()?,
            Json::Null | Json::Bool(_) | Json::Number(_) | Json::String(_) => {
                bail!("\"samples\" isn't an array")
            }
        };
        let unit = text("unit")?;
        let label = text("label")?;
        let threads = number("worker_threads")?;
        Ok(Self {
            info: RunInfo {
                run: text("run")?.parse()?,
                commit: text("commit")?.parse()?,
                dirty: flag("dirty")?,
                machine: text("machine")?,
                fingerprint: text("fingerprint")?.parse()?,
                kernel: text("kernel")?,
                workers: text("workers")?.parse()?,
                worker_threads: if threads.fract() == 0.0 && (0.0..1e9).contains(&threads) {
                    // Checked whole and small just above.
                    #[expect(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        reason = "a whole number in [0, 1e9)"
                    )]
                    let threads = threads as usize;
                    threads
                } else {
                    bail!("\"worker_threads\" isn't a count")
                },
                label: Label::named(&label).ok_or_else(|| anyhow!("unknown label {label:?}"))?,
                baseline: flag("baseline")?,
            },
            project: text("project")?,
            load: PartLoad {
                start: load("load_start")?,
                others: load("load_others")?,
                // Records written before the peak was kept have none: their start stands in.
                others_peak: if object.get("load_others_peak").is_some() {
                    load("load_others_peak")?
                } else {
                    load("load_others")?
                },
                max: load("load_max")?,
            },
            metric: Metric {
                name: text("metric")?.parse()?,
                unit: Unit::named(&unit).ok_or_else(|| anyhow!("unknown unit {unit:?}"))?,
                value: Measured::recorded(number("median")?, number("spread")?, samples)?,
                ceiling: match object.get("ceiling") {
                    Some(_) => Some(number("ceiling")?),
                    None => None,
                },
            },
        })
    }
}

/// A history file as read: its entries, and the torn last line left out, if there was one.
#[derive(Debug, Default)]
pub(crate) struct History {
    /// Every whole line's entry, in file order.
    pub(crate) entries: Vec<Entry>,
    /// The number of a last line without its newline, left out: a write cut short (the run
    /// killed mid-append, the disk full). [`append`] drops it before writing.
    pub(crate) torn: Option<usize>,
}

/// Reads every entry of the history file `path`; none if it doesn't exist yet.
///
/// A last line without its newline is a write cut short, not a record: it is left out and
/// reported in [`History::torn`], so one interrupted run can't block every later one.
///
/// # Errors
///
/// If it can't be read, or a whole line (blank ones aside) isn't an entry: history is never
/// silently skipped, since a dropped baseline would turn every comparison into "new".
pub(crate) fn read(path: &Path) -> Result<History> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(History::default()),
        Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
    };
    let (whole, torn) = match text.rfind('\n') {
        _ if text.is_empty() || text.ends_with('\n') => (text.as_str(), None),
        Some(end) => (
            text.get(..=end).unwrap_or_default(),
            Some(text.lines().count()),
        ),
        None => ("", Some(1)),
    };
    let entries = whole
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(n, line)| {
            Entry::parse(line).with_context(|| format!("{}, line {}", path.display(), n + 1))
        })
        .collect::<Result<_>>()?;
    Ok(History { entries, torn })
}

/// The latest baseline of each metric among `entries` (in file order) measured as `run` is.
///
/// That is on hardware with the same fingerprint and with the same evaluation workers setting:
/// a run at 32 workers says nothing about one at 16.
pub(crate) fn baselines<'a>(
    entries: &'a [Entry],
    run: &RunInfo,
) -> BTreeMap<&'a MetricName, &'a Entry> {
    entries
        .iter()
        .filter(|e| {
            e.info.baseline
                && e.info.fingerprint == run.fingerprint
                && e.info.workers == run.workers
        })
        .map(|e| (&e.metric.name, e))
        .collect()
}

/// Appends `entries` to the history file `path`, created if missing, in one write.
///
/// Returns how many bytes of a torn last line (one without its newline, see
/// [`History::torn`]) it dropped first, so the new records start on a line of their own.
///
/// # Errors
///
/// If the file can't be opened, read, cut, written or synced.
pub(crate) fn append(path: &Path, entries: &[Entry]) -> Result<u64> {
    let text: String = entries
        .iter()
        .map(|e| e.to_record().to_json() + "\n")
        .collect();
    let old = match std::fs::read(path) {
        Ok(old) => old,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
    };
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("opening {}", path.display()))?;
    let whole = old
        .iter()
        .rposition(|&b| b == b'\n')
        .map_or(0, |newline| newline + 1);
    let dropped = u64::try_from(old.len() - whole)?;
    if dropped > 0 {
        file.set_len(u64::try_from(whole)?)
            .with_context(|| format!("dropping {}'s torn last line", path.display()))?;
    }
    file.write_all(text.as_bytes())
        .with_context(|| format!("appending to {}", path.display()))?;
    file.sync_all()
        .with_context(|| format!("syncing {}", path.display()))?;
    Ok(dropped)
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
pub(crate) mod tests {
    use super::*;

    /// An entry of `metric` with median `median`, from run `run`, a baseline or not, on
    /// hardware `fingerprint`.
    pub(crate) fn entry(
        metric: &str,
        median: f64,
        run: &str,
        baseline: bool,
        fingerprint: &str,
    ) -> Entry {
        Entry {
            info: RunInfo {
                run: run.parse().unwrap(),
                commit: "0123456789abcdef0123456789abcdef01234567".parse().unwrap(),
                dirty: false,
                machine: "omarchy".to_owned(),
                fingerprint: fingerprint.parse().unwrap(),
                kernel: "7.2.8".to_owned(),
                workers: Workers::PhysicalCores,
                worker_threads: 16,
                label: Label::Quiet,
                baseline,
            },
            project: "medium-0123456789abcdef".to_owned(),
            load: PartLoad {
                start: Load::new(1.5).unwrap(),
                others: Load::new(0.75).unwrap(),
                others_peak: Load::new(1.25).unwrap(),
                max: Load::new(3.25).unwrap(),
            },
            metric: Metric {
                name: metric.parse().unwrap(),
                unit: Unit::Ms,
                value: Measured::of(vec![median * 0.99, median, median * 1.02]).unwrap(),
                ceiling: None,
            },
        }
    }

    /// Only a quiet run makes a baseline, and once loaded a run stays loaded.
    #[test]
    fn labels_follow_the_guard() {
        use std::time::Duration;

        use crate::perf::quiet::Load;
        let others = Load::new(1.0).unwrap();
        let quiet = Settled::Quiet {
            waited: Duration::ZERO,
            others,
        };
        let loaded = Settled::Loaded { others };
        assert_eq!(Label::Quiet.after(quiet), Label::Quiet, "still quiet");
        assert_eq!(Label::Quiet.after(loaded), Label::Loaded, "never settled");
        assert_eq!(Label::Loaded.after(quiet), Label::Loaded, "stays loaded");
        assert_eq!(
            Label::Unchecked.after(loaded),
            Label::Unchecked,
            "unguarded"
        );
        assert!(Label::Quiet.allows_baseline(), "quiet");
        assert!(!Label::Loaded.allows_baseline(), "loaded");
        assert!(!Label::Unchecked.allows_baseline(), "unchecked");
    }

    #[test]
    fn entries_round_trip_through_a_line() {
        let mut e = entry(
            "medium.brush.p99_ms",
            2.0,
            "2026-10-24T13:05:09Z",
            true,
            "cpu",
        );
        e.info.label = Label::Loaded;
        e.info.dirty = true;
        let line = e.to_record().to_json();
        assert!(
            line.starts_with(r#"{"bench":"perf","result":"metric","metric":"medium.brush.p99_ms""#),
            "{line}"
        );
        assert_eq!(Entry::parse(&line).unwrap(), e, "read back as written");
        let extra = line.replacen('{', r#"{"later":[1,{"#, 1);
        assert!(
            Entry::parse(&extra).is_err(),
            "nested objects aren't entries"
        );
        let mut capped = e.clone();
        capped.metric.ceiling = Some(1536.0);
        let capped_line = capped.to_record().to_json();
        assert!(
            capped_line.ends_with(r#","ceiling":1536}"#),
            "{capped_line}"
        );
        assert_eq!(
            Entry::parse(&capped_line).unwrap(),
            capped,
            "a ceiling read back"
        );
        let older = line.replacen(r#","load_others_peak":1.25"#, "", 1);
        assert_ne!(older, line, "the edit applies");
        let read_older = Entry::parse(&older).unwrap();
        assert_eq!(
            read_older.load.others_peak, e.load.others,
            "a record from before the peak was kept"
        );
        let added = line.replacen('{', r#"{"later":"field","#, 1);
        assert_eq!(
            Entry::parse(&added).unwrap(),
            e,
            "unknown fields are ignored"
        );
    }

    #[test]
    fn malformed_entries_are_refused() {
        let line = entry(
            "medium.brush.p99_ms",
            2.0,
            "2026-10-24T00:00:00Z",
            false,
            "cpu",
        )
        .to_record()
        .to_json();
        for (from, to, what) in [
            (
                r#""result":"metric""#,
                r#""result":"stroke""#,
                "another record",
            ),
            (r#""unit":"ms""#, r#""unit":"parsecs""#, "an unknown unit"),
            (
                r#""label":"quiet""#,
                r#""label":"calm""#,
                "an unknown label",
            ),
            (
                r#""baseline":false"#,
                r#""baseline":0"#,
                "a number for a flag",
            ),
            (
                r#""metric":"medium.brush.p99_ms""#,
                r#""metric":"Brush""#,
                "a bad name",
            ),
            (
                r#""worker_threads":16"#,
                r#""worker_threads":1.5"#,
                "a fractional count",
            ),
            (r#","dirty":false"#, "", "a missing field"),
        ] {
            let broken = line.replacen(from, to, 1);
            assert_ne!(broken, line, "{what}: the edit applies");
            assert!(Entry::parse(&broken).is_err(), "{what}: {broken}");
        }
    }

    #[test]
    fn history_files_append_and_read_back_with_the_latest_baselines() {
        let path = std::env::temp_dir().join(format!(
            "brushcraft-perf-history-{}.jsonl",
            std::process::id()
        ));
        drop(std::fs::remove_file(&path));
        assert!(
            read(&path).unwrap().entries.is_empty(),
            "no file yet: no history"
        );
        let first = [
            entry(
                "medium.brush.p99_ms",
                2.0,
                "2026-10-24T00:00:01Z",
                true,
                "cpu",
            ),
            entry(
                "medium.brush.p50_ms",
                1.5,
                "2026-10-24T00:00:01Z",
                true,
                "cpu",
            ),
        ];
        assert_eq!(append(&path, &first).unwrap(), 0, "nothing torn");
        let second = [
            entry(
                "medium.brush.p99_ms",
                2.2,
                "2026-10-24T00:00:02Z",
                false,
                "cpu",
            ),
            entry(
                "medium.brush.p99_ms",
                9.9,
                "2026-10-24T00:00:03Z",
                true,
                "other cpu",
            ),
            entry(
                "medium.brush.p99_ms",
                1.9,
                "2026-10-24T00:00:04Z",
                true,
                "cpu",
            ),
            entry(
                "medium.brush.p99_ms",
                2.4,
                "2026-10-24T00:00:05Z",
                false,
                "cpu",
            ),
        ];
        append(&path, &second).unwrap();
        let history = read(&path).unwrap();
        assert_eq!(history.torn, None, "whole lines");
        let all = history.entries;
        assert_eq!(all.len(), 6, "both appends");
        let this = entry(
            "medium.brush.p99_ms",
            2.0,
            "2026-10-24T00:00:09Z",
            false,
            "cpu",
        )
        .info;
        let base = baselines(&all, &this);
        let p99: MetricName = "medium.brush.p99_ms".parse().unwrap();
        let p50: MetricName = "medium.brush.p50_ms".parse().unwrap();
        assert_eq!(
            base.get(&p99).map(|e| e.info.run.as_str()),
            Some("2026-10-24T00:00:04Z"),
            "the latest baseline"
        );
        assert_eq!(
            base.get(&p50).map(|e| e.info.run.as_str()),
            Some("2026-10-24T00:00:01Z"),
            "per metric"
        );

        std::fs::write(&path, "\n{\"bench\":\"perf\"}\n").unwrap();
        let err = read(&path).unwrap_err();
        assert!(
            format!("{err:#}").contains("line 2"),
            "the line is named: {err:#}"
        );
        std::fs::remove_file(&path).unwrap();
    }

    /// A baseline counts only for runs on the same hardware with the same workers setting.
    #[test]
    fn baselines_match_hardware_and_workers() {
        let all = [entry(
            "medium.brush.p99_ms",
            2.0,
            "2026-10-24T00:00:01Z",
            true,
            "cpu",
        )];
        let this = entry(
            "medium.brush.p99_ms",
            2.0,
            "2026-10-24T00:00:09Z",
            false,
            "cpu",
        )
        .info;
        assert_eq!(baselines(&all, &this).len(), 1, "the same setting");
        let elsewhere = entry("x", 1.0, "2026-10-24T00:00:09Z", false, "new cpu").info;
        assert!(
            baselines(&all, &elsewhere).is_empty(),
            "other hardware has none"
        );
        let logical = RunInfo {
            workers: Workers::LogicalCores,
            ..this
        };
        assert!(
            baselines(&all, &logical).is_empty(),
            "another workers setting has none"
        );
    }

    /// A last line cut short is left out with a warning, and the next append drops it; a
    /// malformed whole line is still refused.
    #[test]
    fn a_torn_last_line_is_left_out_and_dropped() {
        let path =
            std::env::temp_dir().join(format!("brushcraft-perf-torn-{}.jsonl", std::process::id()));
        let whole = entry(
            "medium.brush.p99_ms",
            2.0,
            "2026-10-24T00:00:01Z",
            true,
            "cpu",
        )
        .to_record()
        .to_json();
        std::fs::write(&path, format!("{whole}\n{}", &whole[..40])).unwrap();
        let history = read(&path).unwrap();
        assert_eq!(history.entries.len(), 1, "the whole line");
        assert_eq!(history.torn, Some(2), "line 2 is torn");
        let next = [entry(
            "medium.brush.p99_ms",
            2.1,
            "2026-10-24T00:00:02Z",
            false,
            "cpu",
        )];
        assert_eq!(append(&path, &next).unwrap(), 40, "the torn bytes dropped");
        let after = read(&path).unwrap();
        assert_eq!(after.torn, None, "nothing torn now");
        assert_eq!(after.entries.len(), 2, "the old line and the new one");
        std::fs::write(&path, &whole[..40]).unwrap();
        assert_eq!(read(&path).unwrap().torn, Some(1), "a torn only line");
        std::fs::write(&path, format!("{}\n{whole}\n", &whole[..40])).unwrap();
        assert!(read(&path).is_err(), "a malformed interior line is refused");
        std::fs::remove_file(&path).unwrap();
    }
}
