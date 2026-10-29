//! The command line: which subcommand to run, and its options.
//!
//! Hand-parsed from `std::env::args`, as `pen-probe` does: `brushcraft-bench SUBCOMMAND
//! [--flag VALUE | --flag=VALUE | --switch]…`. Every option has a default; each subcommand
//! accepts only the options it uses, and refuses the rest, so a typo never runs a benchmark
//! with a setting silently ignored.

use std::ffi::{OsStr, OsString};
use std::path::PathBuf;
use std::str::FromStr;

use anyhow::{Context as _, Result, anyhow, bail, ensure};
use store::Workers;

use crate::session::Detail;

/// What the harness prints when its arguments don't parse.
pub(crate) const USAGE: &str = "\
usage: brushcraft-bench SUBCOMMAND [OPTIONS]

subcommands:
  generate   build the benchmark sphere with fBm corrections at every level and save it
  page-in    time evaluating and pinning a brush footprint and a large region, cold and warm
  brush      time a stroke of draw steps over a footprint along a path
  save       brush, then save: time, file growth and bytes per edited vertex
             (the save goes into the project: it gains a save point each run)
  budget     a random walk (evaluate, brush, pan) under a budget, sampling RSS every 100 ms
  perf       the tracked performance suite: stage benches and a cached medium project (and
             with --full the benchmark project), compared with the history's baseline
  display    the display layer kept: rebuilt level by level, saved (file growth), then the
             brush stroke with its refresh after each step (saves into the project)
  display-save  the display layer rebuilt and saved, nothing else: a project that opens with
             its layer (saves into the project, so --project must be given)
  display-study  every display-tile encoding measured on sampled faces: file growth, the
             error bound, decode cost and seam exactness (reads the project, saves nothing;
             needs the display-study feature)

options (each subcommand takes the ones it uses):
  --project PATH          the project file (default $XDG_CACHE_HOME/brushcraft/bench/sphere.bcrf);
                          never inside the repository. Recovery files go to `recovery/` beside it
  --budget-mib N          the tile store's budget (default: generate 2048, page-in 4096,
                          brush 2048, save 2048, budget 1024)
  --seed N                the random seed (default 1)
  --workers W             the store's evaluation workers: physical (default, one per physical
                          core), logical (one per hardware thread) or a count; every JSON line
                          records the setting and the count it gave
  generate: --splits N (20)  --levels N (9)  --amplitude F (0.01)  --force (replace the file)
  page-in, brush, save, budget: --level N, the finest level, the one measured (9)
  display-study: --samples N (200), the faces measured, evenly spaced
  page-in:  --samples N (5)  --footprint VERTICES (50000)  --region-quads N (42000000)
            --pool cold|warm|both (cold): between samples the store is emptied and its memory
            given back (cold: every pin maps fresh memory), or emptied with its memory kept
            (warm: as in a running app), or each in turn (both)
  brush:    --steps N (200)  --footprint VERTICES (50000)  --strength F (0.0001)
            --display (keep the display layer: each step raises its display tiles' bounds)
  display:  --steps N (200)  --footprint VERTICES (50000)  --strength F (0.0001)
            --per-step (a JSON line per step besides the summary: its time and each phase's,
            its footprint, the tiles the store loaded, evaluated and evicted, page faults)
  save:     --steps N (50)   --footprint VERTICES (50000)  --strength F (0.0001)
  budget:   --moves N (20)  --steps-per-move N (10)  --view-tiles N (2000)
            --footprint VERTICES (50000)  --strength F (0.0001)
  perf:     --history DIR (append to DIR/<machine>.jsonl and compare with its baselines)
            --baseline (mark this run the baseline; needs --history, a clean tree, a quiet run)
            --full (also the benchmark project at --project, through a reflinked copy)
            --tiny (a tiny project, no guard, nothing recorded: checks the suite runs)
            --no-stages (skip the stage benches)  --settle-secs N (300: the quiet guard's
            longest wait for the other processes' 1-minute load to fall below 1.5; it
            reaching 2 during a part labels the run loaded)
            Exit status 3 when a metric got worse beyond its noise; else 4 when one was too
            noisy to compare (its noise band over 25 %).

Results go to stdout as JSON lines; progress and a human summary go to stderr.";

/// A subcommand: one benchmark.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Subcommand {
    /// Build the benchmark sphere and save it.
    Generate,
    /// Time evaluating and pinning regions from a fresh cache.
    PageIn,
    /// Time a stroke of draw steps.
    Brush,
    /// Brush, then time a save.
    Save,
    /// A random walk under a memory budget, sampling RSS.
    Budget,
    /// The tracked performance suite.
    Perf,
    /// The display-tile encoding study.
    DisplayStudy,
    /// The display layer's costs.
    Display,
    /// Rebuild the display layer and save it.
    DisplaySave,
}

impl Subcommand {
    /// Every subcommand.
    const ALL: [Self; 9] = [
        Self::Generate,
        Self::PageIn,
        Self::Brush,
        Self::Save,
        Self::Budget,
        Self::Perf,
        Self::DisplayStudy,
        Self::Display,
        Self::DisplaySave,
    ];

    /// The subcommand's name on the command line and in the JSON records.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Generate => "generate",
            Self::PageIn => "page-in",
            Self::Brush => "brush",
            Self::Save => "save",
            Self::Budget => "budget",
            Self::Perf => "perf",
            Self::DisplayStudy => "display-study",
            Self::Display => "display",
            Self::DisplaySave => "display-save",
        }
    }

    /// The store budget the subcommand runs with unless told otherwise, in MiB.
    ///
    /// `page-in` pins a whole 42M-quad region at once (about 10,000 level-9 tiles, 1.3 GB of
    /// evaluated and correction slots), so it needs more than the others; `budget` is the
    /// spec's 1 GB workload. `perf` doesn't use one: each part runs with its subcommand's.
    const fn default_budget_mib(self) -> usize {
        match self {
            Self::Generate
            | Self::Brush
            | Self::Save
            | Self::Perf
            | Self::DisplayStudy
            | Self::Display
            | Self::DisplaySave => 2048,
            Self::PageIn => 4096,
            Self::Budget => 1024,
        }
    }

    /// Whether the subcommand takes the option `flag`.
    fn takes(self, flag: Flag) -> bool {
        use Flag::{
            Amplitude, Baseline, BudgetMib, Display, Footprint, Force, Full, History, Level,
            Levels, Moves, NoStages, PerStep, Pool, Project, RegionQuads, Samples, Seed,
            SettleSecs, Splits, Steps, StepsPerMove, Strength, Tiny, ViewTiles, Workers,
        };
        match flag {
            Project | Workers => true,
            BudgetMib | Seed => self != Self::Perf,
            Splits | Levels | Amplitude | Force => self == Self::Generate,
            Level => !matches!(self, Self::Generate | Self::Perf),
            Footprint => !matches!(
                self,
                Self::Generate | Self::Perf | Self::DisplayStudy | Self::DisplaySave
            ),
            History | Baseline | Full | Tiny | NoStages | SettleSecs => self == Self::Perf,
            Samples => matches!(self, Self::PageIn | Self::DisplayStudy),
            RegionQuads | Pool => self == Self::PageIn,
            Steps => matches!(self, Self::Brush | Self::Save | Self::Display),
            PerStep | Display => self == Self::Brush,
            Strength => matches!(
                self,
                Self::Brush | Self::Save | Self::Budget | Self::Display
            ),
            Moves | StepsPerMove | ViewTiles => self == Self::Budget,
        }
    }
}

/// `page-in`: the state of the store's slot pool when a sample's pin starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Pool {
    /// Emptied with its memory given back between samples (`evict_all`, then `shrink_to(0)`):
    /// every pin maps fresh chunks and faults their pages in, zeroed by the kernel, as the
    /// first pin after the app starts does.
    Cold,
    /// Emptied with its memory kept between samples (`evict_all` alone): every pin reuses
    /// chunks mapped and touched earlier, as a running app panning to a new region does.
    Warm,
}

impl Pool {
    /// The pool state's name on the command line and in the JSON records.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Cold => "cold",
            Self::Warm => "warm",
        }
    }
}

/// `--pool`: which pool states `page-in` measures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Pools {
    /// Only [`Pool::Cold`], the default: the measure the harness has always taken.
    Cold,
    /// Only [`Pool::Warm`].
    Warm,
    /// [`Pool::Cold`], then [`Pool::Warm`], each over the same places.
    Both,
}

impl Pools {
    /// The choice named `name` on the command line (`cold`, `warm` or `both`).
    fn named(name: &str) -> Option<Self> {
        match name {
            "cold" => Some(Self::Cold),
            "warm" => Some(Self::Warm),
            "both" => Some(Self::Both),
            _ => None,
        }
    }

    /// The pool states to measure, in the order they are measured.
    pub(crate) const fn each(self) -> &'static [Pool] {
        match self {
            Self::Cold => &[Pool::Cold],
            Self::Warm => &[Pool::Warm],
            Self::Both => &[Pool::Cold, Pool::Warm],
        }
    }
}

/// `perf`: the projects measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Projects {
    /// The cached medium project, the default.
    Medium,
    /// The medium project, then the full benchmark project (`--full`).
    MediumAndFull,
    /// A tiny project instead (`--tiny`), unguarded and unrecorded.
    Tiny,
}

/// An option's flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Flag {
    /// `--project`.
    Project,
    /// `--budget-mib`.
    BudgetMib,
    /// `--seed`.
    Seed,
    /// `--splits`.
    Splits,
    /// `--levels`.
    Levels,
    /// `--amplitude`.
    Amplitude,
    /// `--force`, a switch.
    Force,
    /// `--level`.
    Level,
    /// `--samples`.
    Samples,
    /// `--footprint`.
    Footprint,
    /// `--region-quads`.
    RegionQuads,
    /// `--pool`.
    Pool,
    /// `--steps`.
    Steps,
    /// `--per-step`, a switch.
    PerStep,
    /// `--strength`.
    Strength,
    /// `--moves`.
    Moves,
    /// `--steps-per-move`.
    StepsPerMove,
    /// `--view-tiles`.
    ViewTiles,
    /// `--workers`.
    Workers,
    /// `--history`.
    History,
    /// `--baseline`, a switch.
    Baseline,
    /// `--full`, a switch.
    Full,
    /// `--tiny`, a switch.
    Tiny,
    /// `--no-stages`, a switch.
    NoStages,
    /// `--settle-secs`.
    SettleSecs,
    /// `--display`, a switch.
    Display,
}

impl Flag {
    /// Every flag with its name, without the leading `--`.
    const NAMES: [(Self, &'static str); 26] = [
        (Self::Project, "project"),
        (Self::BudgetMib, "budget-mib"),
        (Self::Seed, "seed"),
        (Self::Splits, "splits"),
        (Self::Levels, "levels"),
        (Self::Amplitude, "amplitude"),
        (Self::Force, "force"),
        (Self::Level, "level"),
        (Self::Samples, "samples"),
        (Self::Footprint, "footprint"),
        (Self::RegionQuads, "region-quads"),
        (Self::Pool, "pool"),
        (Self::Steps, "steps"),
        (Self::PerStep, "per-step"),
        (Self::Strength, "strength"),
        (Self::Moves, "moves"),
        (Self::StepsPerMove, "steps-per-move"),
        (Self::ViewTiles, "view-tiles"),
        (Self::Workers, "workers"),
        (Self::History, "history"),
        (Self::Baseline, "baseline"),
        (Self::Full, "full"),
        (Self::Tiny, "tiny"),
        (Self::NoStages, "no-stages"),
        (Self::SettleSecs, "settle-secs"),
        (Self::Display, "display"),
    ];

    /// Whether the flag is a switch, taking no value.
    const fn is_switch(self) -> bool {
        matches!(
            self,
            Self::Force
                | Self::Baseline
                | Self::Full
                | Self::Tiny
                | Self::NoStages
                | Self::PerStep
                | Self::Display
        )
    }

    /// The flag named `name` (without the leading `--`).
    fn named(name: &str) -> Option<Self> {
        Self::NAMES
            .iter()
            .find(|&&(_, n)| n == name)
            .map(|&(flag, _)| flag)
    }
}

/// Every option of every subcommand, with its value or its default.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Options {
    /// The project file; `None` for the default under the cache directory (see
    /// [`crate::paths`]).
    pub(crate) project: Option<PathBuf>,
    /// The tile store's budget, in MiB.
    pub(crate) budget_mib: usize,
    /// The seed of every random choice.
    pub(crate) seed: u64,
    /// `generate`: the base mesh's `cube_sphere` splits.
    pub(crate) splits: u32,
    /// `generate`: the finest level given corrections, from level 1.
    pub(crate) levels: u8,
    /// `generate`: the fBm noise's amplitude at level 1, halving every level after.
    pub(crate) amplitude: f32,
    /// `generate`: replace an existing project file.
    pub(crate) force: bool,
    /// The finest level of the project: the one evaluated, indexed and brushed.
    pub(crate) level: u8,
    /// `page-in`: samples of each measurement.
    pub(crate) samples: usize,
    /// The brush footprint, in vertices.
    pub(crate) footprint: usize,
    /// `page-in`: the large region, in quads of the measured level.
    pub(crate) region_quads: u64,
    /// `page-in`: whether the pool is measured cold, warm or both.
    pub(crate) pools: Pools,
    /// `brush`, `save`: draw steps in the stroke.
    pub(crate) steps: usize,
    /// `brush`: how closely each step of the stroke is watched and reported.
    pub(crate) detail: Detail,
    /// How far the draw brush pushes at its centre, along the frame normal.
    pub(crate) strength: f32,
    /// `budget`: places the walk pans to.
    pub(crate) moves: usize,
    /// `budget`: draw steps at each place.
    pub(crate) steps_per_move: usize,
    /// `budget`: tiles evaluated and pinned at each place, as a view would.
    pub(crate) view_tiles: usize,
    /// How many evaluation workers the tile store starts.
    pub(crate) workers: Workers,
    /// `perf`: the history directory to append to and compare with.
    pub(crate) history: Option<PathBuf>,
    /// `perf`: mark the run as the new baseline.
    pub(crate) baseline: bool,
    /// `perf`: the projects measured.
    pub(crate) projects: Projects,
    /// `perf`: run the stage benches.
    pub(crate) stages: bool,
    /// `perf`: the quiet guard's longest wait before a part, in seconds.
    pub(crate) settle_secs: u64,
    /// Whether the store keeps a display layer under the level (`--display`; always for
    /// `display`).
    pub(crate) display: DisplayLayer,
}

/// Whether a store keeps a display layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DisplayLayer {
    /// It keeps none: what the M0 subcommands measure.
    Off,
    /// It keeps one under the level measured.
    Kept,
}

impl Options {
    /// The defaults of `subcommand`.
    fn defaults(subcommand: Subcommand) -> Self {
        Self {
            project: None,
            budget_mib: subcommand.default_budget_mib(),
            seed: 1,
            splits: 20,
            levels: 9,
            amplitude: 0.01,
            force: false,
            level: 9,
            samples: if subcommand == Subcommand::DisplayStudy {
                200
            } else {
                5
            },
            footprint: 50_000,
            region_quads: 42_000_000,
            pools: Pools::Cold,
            steps: if subcommand == Subcommand::Save {
                50
            } else {
                200
            },
            detail: Detail::Summary,
            strength: 1e-4,
            moves: 20,
            steps_per_move: 10,
            view_tiles: 2000,
            workers: Workers::PhysicalCores,
            history: None,
            baseline: false,
            projects: Projects::Medium,
            stages: true,
            settle_secs: 300,
            display: if matches!(subcommand, Subcommand::Display | Subcommand::DisplaySave) {
                DisplayLayer::Kept
            } else {
                DisplayLayer::Off
            },
        }
    }

    /// The store budget in bytes.
    pub(crate) const fn budget_bytes(&self) -> usize {
        self.budget_mib.saturating_mul(1 << 20)
    }

    /// Sets `flag` from `value` (`None` for a switch).
    ///
    /// # Errors
    ///
    /// If the value doesn't parse, is out of range, or is missing (or given to a switch).
    fn set(&mut self, flag: Flag, value: Option<&OsStr>) -> Result<()> {
        if flag.is_switch() {
            ensure!(value.is_none(), "{} takes no value", flag_name(flag));
        }
        let given = || value.ok_or_else(|| anyhow!("{} needs a value", flag_name(flag)));
        let path = || given().map(PathBuf::from);
        let text = || {
            given()?
                .to_str()
                .ok_or_else(|| anyhow!("{}'s value isn't text", flag_name(flag)))
        };
        match flag {
            Flag::Force => self.force = true,
            Flag::Baseline => self.baseline = true,
            Flag::Full => {
                self.projects = match self.projects {
                    Projects::Tiny => bail!("--tiny and --full don't go together"),
                    Projects::Medium | Projects::MediumAndFull => Projects::MediumAndFull,
                };
            }
            Flag::Tiny => {
                self.projects = match self.projects {
                    Projects::MediumAndFull => bail!("--tiny and --full don't go together"),
                    Projects::Medium | Projects::Tiny => Projects::Tiny,
                };
            }
            Flag::NoStages => self.stages = false,
            Flag::PerStep => self.detail = Detail::PerStep,
            Flag::Display => self.display = DisplayLayer::Kept,
            Flag::Project => self.project = Some(path()?),
            Flag::History => self.history = Some(path()?),
            Flag::BudgetMib => self.budget_mib = parse(flag, text()?)?,
            Flag::Seed => self.seed = parse(flag, text()?)?,
            Flag::Splits => self.splits = positive(flag, text()?)?,
            Flag::Levels => self.levels = positive(flag, text()?)?,
            Flag::Amplitude => self.amplitude = finite(flag, text()?)?,
            Flag::Level => self.level = positive(flag, text()?)?,
            Flag::Samples => self.samples = positive(flag, text()?)?,
            Flag::Footprint => self.footprint = positive(flag, text()?)?,
            Flag::RegionQuads => self.region_quads = positive(flag, text()?)?,
            Flag::Pool => {
                let text = text()?;
                self.pools = Pools::named(text).ok_or_else(|| {
                    anyhow!("--pool {text:?}: expected one of cold, warm or both")
                })?;
            }
            Flag::Steps => self.steps = positive(flag, text()?)?,
            Flag::Strength => self.strength = finite(flag, text()?)?,
            Flag::Moves => self.moves = positive(flag, text()?)?,
            Flag::StepsPerMove => self.steps_per_move = positive(flag, text()?)?,
            Flag::ViewTiles => self.view_tiles = positive(flag, text()?)?,
            Flag::Workers => self.workers = parse(flag, text()?)?,
            Flag::SettleSecs => self.settle_secs = parse(flag, text()?)?,
        }
        Ok(())
    }
}

/// `flag` as typed: `--name`.
fn flag_name(flag: Flag) -> String {
    let name = Flag::NAMES
        .iter()
        .find(|&&(f, _)| f == flag)
        .map_or("?", |&(_, n)| n);
    format!("--{name}")
}

/// `text` parsed as a `T`, for `flag`.
///
/// # Errors
///
/// If it doesn't parse.
fn parse<T: FromStr>(flag: Flag, text: &str) -> Result<T>
where
    T::Err: std::error::Error + Send + Sync + 'static,
{
    text.parse()
        .with_context(|| format!("{} {text:?}", flag_name(flag)))
}

/// `text` parsed as a positive integer, for `flag`.
///
/// # Errors
///
/// If it doesn't parse, or is zero.
fn positive<T: FromStr + Default + PartialEq>(flag: Flag, text: &str) -> Result<T>
where
    T::Err: std::error::Error + Send + Sync + 'static,
{
    let value: T = parse(flag, text)?;
    ensure!(
        value != T::default(),
        "{} must be positive",
        flag_name(flag)
    );
    Ok(value)
}

/// `text` parsed as a finite float, for `flag`.
///
/// # Errors
///
/// If it doesn't parse, or isn't finite.
fn finite(flag: Flag, text: &str) -> Result<f32> {
    let value: f32 = parse(flag, text)?;
    ensure!(value.is_finite(), "{} must be finite", flag_name(flag));
    Ok(value)
}

/// What the command line asks for: a subcommand and its options.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Command {
    /// The benchmark to run.
    pub(crate) subcommand: Subcommand,
    /// Its options.
    pub(crate) options: Options,
}

impl Command {
    /// Parses the arguments after the program name.
    ///
    /// # Errors
    ///
    /// On a missing or unknown subcommand, an unknown flag or one the subcommand doesn't take,
    /// a flag without its value, a value that doesn't parse, a stray argument, or
    /// `display-save` without `--project`.
    pub(crate) fn parse(args: impl IntoIterator<Item = OsString>) -> Result<Self> {
        let mut args = args.into_iter();
        let first = args.next().ok_or_else(|| anyhow!("no subcommand"))?;
        let subcommand = Subcommand::ALL
            .into_iter()
            .find(|s| first.to_str() == Some(s.name()))
            .ok_or_else(|| anyhow!("unknown subcommand {}", first.display()))?;
        let mut options = Options::defaults(subcommand);
        while let Some(arg) = args.next() {
            let text = arg
                .to_str()
                .ok_or_else(|| anyhow!("argument {} isn't text", arg.display()))?;
            let Some(body) = text.strip_prefix("--") else {
                bail!("unexpected argument {text:?}");
            };
            let (name, inline) = match body.split_once('=') {
                Some((name, value)) => (name, Some(OsString::from(value))),
                None => (body, None),
            };
            let flag = Flag::named(name).ok_or_else(|| anyhow!("unknown option --{name}"))?;
            ensure!(
                subcommand.takes(flag),
                "{} doesn't take --{name}",
                subcommand.name()
            );
            let value = match inline {
                Some(value) => Some(value),
                None if flag.is_switch() => None,
                None => Some(
                    args.next()
                        .ok_or_else(|| anyhow!("--{name} needs a value"))?,
                ),
            };
            options.set(flag, value.as_deref())?;
        }
        // It writes into the project, and the default is the benchmark sphere `perf` measures
        // as it is.
        ensure!(
            subcommand != Subcommand::DisplaySave || options.project.is_some(),
            "display-save writes into the project: name it with --project (a reflinked copy of \
             the benchmark project, never the one perf measures)"
        );
        Ok(Self {
            subcommand,
            options,
        })
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    /// Parses `args`, split on spaces.
    ///
    /// # Errors
    ///
    /// As [`Command::parse`].
    fn parse(args: &str) -> Result<Command> {
        Command::parse(args.split_whitespace().map(OsString::from))
    }

    #[test]
    fn defaults_and_values_parse() {
        let brush = parse("brush").unwrap();
        assert_eq!(brush.subcommand, Subcommand::Brush, "the subcommand");
        assert_eq!(brush.options.steps, 200, "brush's default steps");
        assert_eq!(brush.options.budget_mib, 2048, "brush's default budget");
        assert_eq!(
            brush.options.detail,
            Detail::Summary,
            "a summary by default"
        );
        assert_eq!(
            parse("brush --per-step").unwrap().options.detail,
            Detail::PerStep,
            "a record per step asked for"
        );
        let save = parse("save --steps 7 --level=6 --project /tmp/x.bcrf").unwrap();
        assert_eq!(save.options.steps, 7, "a spaced value");
        assert_eq!(save.options.level, 6, "an inline value");
        assert_eq!(
            save.options.project,
            Some(PathBuf::from("/tmp/x.bcrf")),
            "a path"
        );
        let page_in = parse("page-in").unwrap();
        assert_eq!(page_in.options.pools, Pools::Cold, "page-in's default pool");
        let both = parse("page-in --pool both").unwrap();
        assert_eq!(
            both.options.pools.each(),
            [Pool::Cold, Pool::Warm],
            "both, cold first"
        );
        let generate = parse("generate --force --splits 2").unwrap();
        assert!(generate.options.force, "a switch");
        assert_eq!(generate.options.splits, 2, "after a switch");
        assert_eq!(
            generate.options.workers,
            Workers::PhysicalCores,
            "one worker a physical core by default"
        );
        let perf =
            parse("perf --full --baseline --no-stages --history /d --settle-secs 9").unwrap();
        assert_eq!(perf.subcommand, Subcommand::Perf, "perf");
        let o = &perf.options;
        assert!(o.baseline && !o.stages, "switches: {o:?}");
        assert_eq!(o.projects, Projects::MediumAndFull, "--full");
        assert_eq!(
            o.history,
            Some(PathBuf::from("/d")),
            "the history directory"
        );
        assert_eq!(o.settle_secs, 9, "the guard's bound");
        let plain = parse("perf --tiny").unwrap().options;
        assert_eq!(plain.projects, Projects::Tiny, "--tiny");
        assert!(
            plain.stages && plain.history.is_none(),
            "defaults: {plain:?}"
        );
        assert_eq!(plain.settle_secs, 300, "five minutes by default");
        for (args, workers) in [
            ("brush --workers logical", Workers::LogicalCores),
            ("page-in --workers physical", Workers::PhysicalCores),
            (
                "generate --workers 3",
                Workers::Fixed(std::num::NonZeroUsize::new(3).unwrap()),
            ),
        ] {
            assert_eq!(parse(args).unwrap().options.workers, workers, "{args}");
        }
    }

    #[test]
    fn bad_command_lines_are_refused() {
        for (args, what) in [
            ("", "no subcommand"),
            ("frobnicate", "an unknown subcommand"),
            ("brush --splits 3", "an option of another subcommand"),
            ("brush --nope 3", "an unknown option"),
            ("brush --steps", "a missing value"),
            ("brush --steps 0", "zero steps"),
            ("brush --steps x", "a value that doesn't parse"),
            ("brush --strength inf", "an infinite strength"),
            ("page-in --pool lukewarm", "an unknown pool state"),
            ("brush --pool warm", "a pool state for another subcommand"),
            ("brush --workers 0", "no workers"),
            ("brush --workers some", "an unknown worker setting"),
            ("generate --force=yes", "a value for a switch"),
            ("brush stray", "a stray argument"),
            ("brush --history /d", "a perf option for another subcommand"),
            ("perf --seed 3", "a seed for perf, which fixes its own"),
            ("perf --level 6", "a level for perf, which fixes its own"),
            ("perf --tiny=yes", "a value for a perf switch"),
            (
                "save --per-step",
                "per-step records for a subcommand without them",
            ),
            ("brush --per-step=yes", "a value for the per-step switch"),
            ("perf --tiny --full", "a tiny and a full project"),
            ("perf --full --tiny", "a full and a tiny project"),
            ("perf --history", "a history without its directory"),
            ("perf --settle-secs x", "a wait that doesn't parse"),
            ("display-save", "display-save on the default project"),
            (
                "display-save --project /tmp/x.bcrf --steps 3",
                "steps for display-save, which doesn't brush",
            ),
        ] {
            assert!(parse(args).is_err(), "{what}: {args:?}");
        }
    }
}
