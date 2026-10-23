//! Every subcommand end to end on a tiny project: `cube_sphere(2)` sculpted to level 6, 24
//! whole-face tiles at the finest level, in a temporary directory.

#![expect(clippy::missing_panics_doc, reason = "tests panic to fail")]

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use super::*;

/// A temporary directory for one test's project, removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    /// A fresh directory under the system's temporary directory (outside the repository).
    fn new() -> Self {
        /// Distinguishes this process's directories.
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "brushcraft-bench-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    /// The project file's path in it.
    fn project(&self) -> PathBuf {
        self.0.join("tiny.bcrf")
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        drop(std::fs::remove_dir_all(&self.0));
    }
}

/// What a run printed: its JSON lines and its human output.
struct Printed {
    /// The JSON lines.
    json: Vec<String>,
    /// The summary and progress.
    human: String,
}

/// Runs the command line `args` (split on spaces) with `--project project` appended.
///
/// # Errors
///
/// Whatever the command fails with.
fn run_on(project: &Path, args: &str) -> Result<Printed> {
    let args = args
        .split_whitespace()
        .map(OsString::from)
        .chain([OsString::from("--project"), project.as_os_str().to_owned()]);
    let command = Command::parse(args)?;
    let mut out = Output::new(Vec::new(), Vec::new());
    run(&command, &mut out)?;
    let (json, human) = out.into_parts();
    Ok(Printed {
        json: String::from_utf8(json)?
            .lines()
            .map(str::to_owned)
            .collect(),
        human: String::from_utf8(human)?,
    })
}

/// The tiny project, generated in `dir`.
fn generate(dir: &TempDir) -> Printed {
    run_on(&dir.project(), "generate --splits 2 --levels 6").unwrap()
}

/// The number field `name` of the JSON line `line`.
fn number(line: &str, name: &str) -> f64 {
    let at = line
        .find(&format!("\"{name}\":"))
        .unwrap_or_else(|| panic!("no {name} in {line}"));
    let rest = &line[at + name.len() + 3..];
    let end = rest.find([',', '}']).unwrap();
    rest[..end]
        .parse()
        .unwrap_or_else(|_| panic!("{name} in {line}"))
}

/// The one JSON line of `printed` whose result is `result`.
fn line<'a>(printed: &'a Printed, bench: &str, result: &str) -> &'a str {
    let head = format!("{{\"bench\":\"{bench}\",\"result\":\"{result}\"");
    let found: Vec<&String> = printed
        .json
        .iter()
        .filter(|l| l.starts_with(&head))
        .collect();
    assert_eq!(
        found.len(),
        1,
        "one {bench} {result} line in {:?}",
        printed.json
    );
    found[0]
}

#[test]
fn generate_writes_a_project_and_reports_it() {
    let dir = TempDir::new();
    let printed = generate(&dir);
    let project = line(&printed, "generate", "project");
    assert!(
        (number(project, "tiles") - 144.0).abs() < 0.5,
        "24 faces at 6 levels: {project}"
    );
    assert!(
        number(project, "vertices_written") > 90_000.0,
        "every vertex: {project}"
    );
    assert!(
        number(project, "max_rss_exact_mib") > 0.0,
        "the peak: {project}"
    );
    let bytes = std::fs::metadata(dir.project()).unwrap().len();
    assert!(
        (number(project, "file_bytes") - f64::from(u32::try_from(bytes).unwrap())).abs() < 0.5,
        "the file's size"
    );
    assert!(
        printed.human.contains("level 6"),
        "progress: {}",
        printed.human
    );
    assert!(
        !dir.0
            .join("recovery")
            .read_dir()
            .unwrap()
            .any(|e| e.is_ok()),
        "no recovery file left behind"
    );
    let again = run_on(&dir.project(), "generate --splits 2 --levels 6");
    assert!(again.is_err(), "an existing project isn't replaced");
    run_on(&dir.project(), "generate --splits 2 --levels 3 --force").unwrap();
}

#[test]
fn page_in_times_both_regions_cold_and_warm() {
    let dir = TempDir::new();
    generate(&dir);
    let printed = run_on(
        &dir.project(),
        "page-in --level 6 --samples 2 --footprint 5000 --region-quads 20000",
    )
    .unwrap();
    let footprint = line(&printed, "page-in", "footprint");
    assert!(
        (number(footprint, "tiles") - 2.0).abs() < 0.5,
        "5,000 vertices are two tiles of 65²: {footprint}"
    );
    let region = line(&printed, "page-in", "region");
    assert!(
        (number(region, "tiles") - 5.0).abs() < 0.5,
        "20,000 quads are five tiles of 64²: {region}"
    );
    for l in [footprint, region] {
        for field in ["cold_median_ms", "warm_median_ms"] {
            assert!(number(l, field) > 0.0, "{field} in {l}");
        }
        assert!(
            (number(l, "cold_samples") - 2.0).abs() < 0.5,
            "two samples: {l}"
        );
    }
}

#[test]
fn page_in_measures_a_warm_pool_beside_the_cold_one() {
    let dir = TempDir::new();
    generate(&dir);
    let printed = run_on(
        &dir.project(),
        "page-in --level 6 --samples 2 --footprint 5000 --region-quads 20000 --pool both",
    )
    .unwrap();
    for region in ["footprint", "region"] {
        let cold = line(&printed, "page-in", region);
        assert!(cold.contains("\"pool\":\"cold\""), "the cold pool: {cold}");
        assert!(
            !cold.contains("pool_growth_mib"),
            "a cold pool isn't warmed: {cold}"
        );
        let warm = line(&printed, "page-in", &format!("{region}-warm-pool"));
        assert!(warm.contains("\"pool\":\"warm\""), "the warm pool: {warm}");
        assert!(
            (number(warm, "tiles") - number(cold, "tiles")).abs() < 0.5,
            "the same regions: {warm}"
        );
        for field in ["cold_median_ms", "warm_median_ms"] {
            assert!(number(warm, field) > 0.0, "{field} in {warm}");
        }
        for field in ["cold_faults_median", "warm_faults_median"] {
            assert!(number(warm, field) >= 0.0, "{field} in {warm}");
        }
        assert!(number(warm, "pool_mib") > 0.0, "chunks kept mapped: {warm}");
        assert!(
            number(warm, "pool_growth_mib").abs() < 1e-9,
            "every sample reused the warmed chunks: {warm}"
        );
    }

    // Alone, from a store just opened (no chunk mapped): the warm-up maps the chunks.
    let alone = run_on(
        &dir.project(),
        "page-in --level 6 --samples 2 --footprint 5000 --region-quads 20000 --pool warm",
    )
    .unwrap();
    assert_eq!(alone.json.len(), 2, "two records: {:?}", alone.json);
    for region in ["footprint-warm-pool", "region-warm-pool"] {
        let warm = line(&alone, "page-in", region);
        assert!(
            number(warm, "pool_growth_mib").abs() < 1e-9,
            "warmed before the first sample: {warm}"
        );
    }
}

#[test]
fn brush_times_a_stroke_after_loading_the_index() {
    let dir = TempDir::new();
    generate(&dir);
    let printed = run_on(&dir.project(), "brush --level 6 --steps 5 --footprint 3000").unwrap();
    let index = line(&printed, "brush", "index");
    assert!(
        (number(index, "tiles") - 24.0).abs() < 0.5,
        "every level-6 tile: {index}"
    );
    assert!(
        number(index, "stored") > 23.5 && number(index, "evaluated") < 0.5,
        "every bound saved by generate, none evaluated: {index}"
    );
    let stroke = line(&printed, "brush", "stroke");
    assert!(
        (number(stroke, "steps") - 5.0).abs() < 0.5,
        "five steps: {stroke}"
    );
    assert!(number(stroke, "step_p99_ms") > 0.0, "a latency: {stroke}");
    assert!(
        number(stroke, "cold_first_dab_ms") > 0.0,
        "a cold first dab: {stroke}"
    );
    let vertices = number(stroke, "vertices_mean");
    assert!(
        (1500.0..6000.0).contains(&vertices),
        "about the footprint asked for: {stroke}"
    );
}

#[test]
fn save_reports_growth_per_edited_vertex() {
    let dir = TempDir::new();
    generate(&dir);
    let printed = run_on(&dir.project(), "save --level 6 --steps 3 --footprint 3000").unwrap();
    let save = line(&printed, "save", "save");
    let edited = number(save, "edited_vertices");
    assert!(edited > 1000.0, "the stroke edited vertices: {save}");
    assert!(number(save, "growth_bytes") > 0.0, "the file grew: {save}");
    assert!(number(save, "bytes_per_edited_vertex") > 0.0, "{save}");
    // The save is a save point of the project: it opens, and brushing it again works.
    run_on(&dir.project(), "brush --level 6 --steps 1 --footprint 3000").unwrap();
}

#[test]
fn budget_walks_and_samples_rss() {
    let dir = TempDir::new();
    generate(&dir);
    let printed = run_on(
        &dir.project(),
        "budget --level 6 --moves 3 --steps-per-move 2 --view-tiles 6 --footprint 3000",
    )
    .unwrap();
    let walk = line(&printed, "budget", "walk");
    assert!(
        (number(walk, "steps") - 6.0).abs() < 0.5,
        "three moves of two steps: {walk}"
    );
    assert!(number(walk, "max_rss_mib") > 0.0, "sampled: {walk}");
    // Tests share the process, and another's reset may come mid-walk: only that it is there.
    assert!(
        number(walk, "max_rss_exact_mib") > 0.0,
        "the exact peak: {walk}"
    );
    assert!(walk.contains("\"rss_samples_mib\":["), "the series: {walk}");
}

#[test]
fn a_missing_project_is_reported() {
    let dir = TempDir::new();
    let err = run_on(&dir.project(), "brush --level 6").err().unwrap();
    assert!(err.to_string().contains("run `generate` first"), "{err}");
}
