//! The stage benches: `geom`'s divan benchmarks of each subdivision stage on one level-9 tile,
//! interior and face-corner (`crates/geom/benches/stages.rs`), read back from divan's table.
//!
//! Divan prints a tree, one group per stage and one leaf per tile:
//!
//! ```text
//! stages              fastest       │ slowest       │ median        │ mean          │ samples │ iters
//! ├─ apply                          │               │               │               │         │
//! │  ├─ face corner   11.79 µs      │ 20.28 µs      │ 11.82 µs      │ 12.07 µs      │ 100     │ 100
//! │  ╰─ interior      11.79 µs      │ 17.11 µs      │ 11.82 µs      │ 11.88 µs      │ 100     │ 100
//! ```
//!
//! Each leaf's median becomes a sample of `stage.<stage>.<tile>_us`, the tile's spaces as
//! underscores (`stage.apply.face_corner_us`).

use anyhow::{Result, anyhow, bail, ensure};

use crate::perf::metric::MetricName;

/// One leaf of divan's table: a stage on a tile, and its median time.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct StageTime {
    /// `stage.<stage>.<tile>_us`.
    pub(crate) name: MetricName,
    /// The median time, in microseconds.
    pub(crate) median_us: f64,
}

/// Every leaf's median in divan's output `text` (lines outside the table are skipped).
///
/// # Errors
///
/// If a table row is malformed (a time without a unit, a name that makes no metric name), or
/// there is no row at all: divan's format changed, and the parser must follow it.
pub(crate) fn parse(text: &str) -> Result<Vec<StageTime>> {
    let mut stage: Option<String> = None;
    let mut times = Vec::new();
    for line in text.lines() {
        // A row is the tree's lead (`│  ` or three spaces per level above), a branch (`├─` or
        // `╰─`), then the columns, split by the same `│` the lead may hold.
        let Some((lead, row)) = line.split_once(['├', '╰']) else {
            continue;
        };
        let depth = lead.chars().count() / 3;
        // Name and fastest, slowest, median, …
        let mut columns = row.trim_start_matches('─').split('│');
        let (Some(first), Some(median)) = (columns.next(), columns.nth(1)) else {
            continue;
        };
        let body = first.trim();
        let (name, fastest) = split_name(body);
        match (depth, fastest.is_empty()) {
            (0, true) => stage = Some(body.to_owned()),
            (1, false) => {
                let group = stage
                    .as_deref()
                    .ok_or_else(|| anyhow!("a tile before any stage: {line:?}"))?;
                let tile = name.replace(' ', "_");
                times.push(StageTime {
                    name: MetricName::join("stage", &format!("{group}.{tile}_us"))?,
                    median_us: micros(median.trim())
                        .ok_or_else(|| anyhow!("no median time in {line:?}"))?,
                });
            }
            _ => bail!("an unexpected row in divan's table: {line:?}"),
        }
    }
    ensure!(!times.is_empty(), "no benchmark rows in divan's output");
    Ok(times)
}

/// A leaf's first column split into its name and its fastest time: the name runs up to the
/// first word that starts with a digit.
fn split_name(body: &str) -> (&str, &str) {
    let at = body
        .char_indices()
        .find(|&(i, c)| {
            c.is_ascii_digit() && (i == 0 || body.get(..i).is_some_and(|s| s.ends_with(' ')))
        })
        .map_or(body.len(), |(i, _)| i);
    (
        body.get(..at).unwrap_or(body).trim(),
        body.get(at..).unwrap_or_default().trim(),
    )
}

/// A divan time (`11.82 µs`, `856.2 ns`, `1.5 ms`, `2 s`) in microseconds.
fn micros(text: &str) -> Option<f64> {
    let (value, unit) = text.split_once(' ')?;
    let value: f64 = value.parse().ok()?;
    let scale = match unit.trim() {
        "ps" => 1e-6,
        "ns" => 1e-3,
        "µs" | "us" => 1.0,
        "ms" => 1e3,
        "s" => 1e6,
        _ => return None,
    };
    Some(value * scale).filter(|x| x.is_finite())
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    /// Divan's output for the stage benches, as `geom`'s bench printed it on 2026-10-24.
    const OUTPUT: &str = "\
Timer precision: 10 ns
stages              fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ apply                          │               │               │               │         │
│  ├─ face corner   11.79 µs      │ 20.28 µs      │ 11.82 µs      │ 12.07 µs      │ 100     │ 100
│  ╰─ interior      11.79 µs      │ 17.11 µs      │ 11.82 µs      │ 11.88 µs      │ 100     │ 100
├─ gather                         │               │               │               │         │
│  ├─ face corner   1.512 µs      │ 2.083 µs      │ 1.532 µs      │ 1.543 µs      │ 100     │ 100
│  ╰─ interior      836.2 ns      │ 876.7 ns      │ 856.2 ns      │ 855.6 ns      │ 100     │ 200
╰─ subdivide                      │               │               │               │         │
   ├─ face corner   8.535 µs      │ 12.54 µs      │ 8.556 µs      │ 8.645 µs      │ 100     │ 100
   ╰─ interior      8.505 µs      │ 10.46 µs      │ 1.2 ms        │ 8.586 µs      │ 100     │ 100

";

    #[test]
    fn divans_table_reads_as_one_median_per_stage_and_tile() {
        let times = parse(OUTPUT).unwrap();
        let got: Vec<(&str, f64)> = times
            .iter()
            .map(|t| (t.name.as_str(), t.median_us))
            .collect();
        let want = [
            ("stage.apply.face_corner_us", 11.82),
            ("stage.apply.interior_us", 11.82),
            ("stage.gather.face_corner_us", 1.532),
            ("stage.gather.interior_us", 0.8562),
            ("stage.subdivide.face_corner_us", 8.556),
            ("stage.subdivide.interior_us", 1200.0),
        ];
        assert_eq!(got.len(), want.len(), "{got:?}");
        for ((name, x), (want_name, want_x)) in got.into_iter().zip(want) {
            assert_eq!(name, want_name, "in divan's order");
            assert!((x - want_x).abs() < 1e-9, "{name}: {x} µs, not {want_x}");
        }
    }

    #[test]
    fn surprises_in_the_table_are_refused() {
        assert!(parse("Timer precision: 10 ns\n").is_err(), "no rows");
        let no_stage = "│  ├─ interior      1 µs      │ 1 µs      │ 1 µs      │ 1 µs      │ 1 │ 1";
        assert!(parse(no_stage).is_err(), "a tile outside a stage");
        let bad_unit = OUTPUT.replace("11.82 µs      │ 12.07", "11.82 fortnights │ 12.07");
        assert!(parse(&bad_unit).is_err(), "an unknown unit");
        let deeper = "├─ a                          │               │               │               │         │
│  ├─ b                       │               │               │               │         │
│  │  ╰─ c   1 µs      │ 1 µs      │ 1 µs      │ 1 µs      │ 1 │ 1";
        assert!(
            parse(deeper).is_err(),
            "a deeper tree than the benches make"
        );
    }

    #[test]
    fn divan_times_convert_to_microseconds() {
        let ns = micros("856.2 ns").unwrap();
        assert!((ns - 0.8562).abs() < 1e-12, "ns: {ns}");
        assert_eq!(micros("2 s"), Some(2e6), "s");
        assert_eq!(micros("12"), None, "no unit");
        assert_eq!(
            split_name("face corner   11.79 µs"),
            ("face corner", "11.79 µs"),
            "split"
        );
        assert_eq!(split_name("apply"), ("apply", ""), "a group");
        assert_eq!(
            split_name("level 9 tile   1 µs"),
            ("level", "9 tile   1 µs"),
            "a digit word ends the name"
        );
    }
}
