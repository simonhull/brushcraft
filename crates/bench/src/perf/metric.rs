//! What the suite measures: named metrics, each the median of a few repeated measurements and
//! their spread.

use std::fmt;
use std::str::FromStr;

use thiserror::Error;

/// A metric's name: lowercase words of `a-z`, `0-9` and `_`, joined by dots, the unit last, as
/// in `medium.brush.p99_ms` or `stage.frames.interior_us`.
///
/// The first word says where it was measured (`stage`, `tiny`, `medium` or `full`); history
/// matches metrics by this name, so a renamed metric starts a new series.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct MetricName(String);

/// Why a text isn't a [`MetricName`].
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{0:?} isn't a metric name (lowercase words of a-z, 0-9 and _, joined by dots)")]
pub(crate) struct BadMetricName(String);

impl MetricName {
    /// The name `prefix.rest`, with `rest` already words joined by dots.
    ///
    /// # Errors
    ///
    /// If the joined text isn't a metric name.
    pub(crate) fn join(prefix: &str, rest: &str) -> Result<Self, BadMetricName> {
        format!("{prefix}.{rest}").parse()
    }

    /// The name as text.
    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

impl FromStr for MetricName {
    type Err = BadMetricName;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let word = |w: &str| {
            !w.is_empty()
                && w.bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
        };
        if text.split('.').all(word) {
            Ok(Self(text.to_owned()))
        } else {
            Err(BadMetricName(text.to_owned()))
        }
    }
}

impl fmt::Display for MetricName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// The unit a metric is in. Every metric is better lower: times, memory, bytes, tiles over
/// budget, frames over the refresh, levels jumped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Unit {
    /// Milliseconds.
    Ms,
    /// Microseconds.
    Us,
    /// Mebibytes.
    Mib,
    /// Bytes.
    Bytes,
    /// Tiles: a count of them.
    Tiles,
    /// Frames: a count of them.
    Frames,
    /// Brush dabs: a count of them.
    Dabs,
    /// Subdivision levels: how many a tile's drawn level changed by.
    Levels,
}

impl Unit {
    /// Every unit.
    const ALL: [Self; 8] = [
        Self::Ms,
        Self::Us,
        Self::Mib,
        Self::Bytes,
        Self::Tiles,
        Self::Frames,
        Self::Dabs,
        Self::Levels,
    ];

    /// The unit's name in history records and tables (ASCII, so `us` for microseconds).
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Ms => "ms",
            Self::Us => "us",
            Self::Mib => "MiB",
            Self::Bytes => "B",
            Self::Tiles => "tiles",
            Self::Frames => "frames",
            Self::Dabs => "dabs",
            Self::Levels => "levels",
        }
    }

    /// The unit named `name`, as [`Unit::name`] gives it.
    pub(crate) fn named(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|unit| unit.name() == name)
    }
}

/// A metric's value: the median of its samples and their spread.
///
/// A sample is one repetition's figure (a run's p99, a round's median time, a round's fitted
/// fixed cost), never a single event: the spread is the noise between repetitions, which is
/// what a later run's figure will be compared against.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Measured {
    /// The median sample (the mean of the middle two for an even count).
    median: f64,
    /// Half the range of the samples kept for it (see [`Measured::of`]) over the median's
    /// magnitude: `(max − min) / (2 |median|)`, or 0 for a single sample or a zero median.
    spread: f64,
    /// The samples, in the order they were taken.
    samples: Vec<f64>,
}

/// Why samples can't make a [`Measured`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub(crate) enum BadSamples {
    /// No sample at all.
    #[error("no samples")]
    Empty,
    /// A sample that isn't a finite number.
    #[error("a sample isn't finite")]
    NotFinite,
}

/// The sample count from which [`Measured::of`] leaves the lowest and highest samples out of
/// the spread.
pub(crate) const TRIM_FROM: usize = 5;

impl Measured {
    /// The median and spread of `samples`.
    ///
    /// The spread is half the range over the median, not a standard deviation: the suite takes
    /// five samples, too few for a deviation to mean much, and half the range bounds how far from
    /// the median the samples fell. With [`TRIM_FROM`] samples or more, the lowest and the
    /// highest are left out of it first: one sample disturbed by a burst of other work would
    /// otherwise widen the noise band enough to hide any change.
    ///
    /// # Errors
    ///
    /// If there are none, or one isn't finite.
    pub(crate) fn of(samples: Vec<f64>) -> Result<Self, BadSamples> {
        if samples.iter().any(|x| !x.is_finite()) {
            return Err(BadSamples::NotFinite);
        }
        let mut sorted = samples.clone();
        sorted.sort_by(f64::total_cmp);
        let n = sorted.len();
        let (Some(&min), Some(&max)) = (sorted.first(), sorted.last()) else {
            return Err(BadSamples::Empty);
        };
        let middle = sorted.get((n - 1) / 2..=n / 2).unwrap_or_default();
        #[expect(
            clippy::cast_precision_loss,
            reason = "the middle of a list is one or two samples"
        )]
        let median = middle.iter().sum::<f64>() / middle.len() as f64;
        let kept = if n >= TRIM_FROM {
            sorted.get(1..n - 1).unwrap_or_default()
        } else {
            &sorted
        };
        let (low, high) = (
            kept.first().copied().unwrap_or(min),
            kept.last().copied().unwrap_or(max),
        );
        let spread = if median == 0.0 {
            0.0
        } else {
            (high - low) / (2.0 * median.abs())
        };
        Ok(Self {
            median,
            spread,
            samples,
        })
    }

    /// A value recorded earlier, as history gives it back: its median and spread as they were
    /// computed then, and its samples.
    ///
    /// # Errors
    ///
    /// If the median or spread isn't finite, or the spread is negative.
    pub(crate) fn recorded(
        median: f64,
        spread: f64,
        samples: Vec<f64>,
    ) -> Result<Self, BadSamples> {
        if median.is_finite() && spread.is_finite() && spread >= 0.0 {
            Ok(Self {
                median,
                spread,
                samples,
            })
        } else {
            Err(BadSamples::NotFinite)
        }
    }

    /// The median sample.
    pub(crate) const fn median(&self) -> f64 {
        self.median
    }

    /// The spread: half the samples' range over the median's magnitude.
    pub(crate) const fn spread(&self) -> f64 {
        self.spread
    }

    /// The samples, in the order they were taken.
    pub(crate) fn samples(&self) -> &[f64] {
        &self.samples
    }
}

/// What a run's metric came to: a value, or, for a timed metric (from an event until a
/// condition holds), the condition never holding.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Outcome {
    /// Measured.
    Measured(Measured),
    /// The condition the metric times until never held: no value, and against a ceiling a miss
    /// (see [`crate::perf::compare::Target`]).
    NotReached,
}

impl Outcome {
    /// The value, if it was measured.
    pub(crate) const fn measured(&self) -> Option<&Measured> {
        match self {
            Self::Measured(measured) => Some(measured),
            Self::NotReached => None,
        }
    }
}

impl From<Measured> for Outcome {
    fn from(measured: Measured) -> Self {
        Self::Measured(measured)
    }
}

/// One metric of one run: its name, unit and value, and the spec's ceiling on it if it has one.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Metric {
    /// What was measured.
    pub(crate) name: MetricName,
    /// Its unit.
    pub(crate) unit: Unit,
    /// Its value, or that it was never reached.
    pub(crate) value: Outcome,
    /// A target the median must stay at or under, in the metric's unit, judged on every run
    /// whatever the baseline (see [`crate::perf::compare::Target`]); `None` for most metrics.
    pub(crate) ceiling: Option<f64>,
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    #[test]
    fn metric_names_are_dotted_lowercase_words() {
        for good in ["medium.brush.p99_ms", "stage.frames.face_corner_us", "x"] {
            assert_eq!(good.parse::<MetricName>().unwrap().as_str(), good, "{good}");
        }
        for bad in ["", "a..b", ".a", "a.", "Medium.brush", "a b", "a-b", "a.µs"] {
            assert!(bad.parse::<MetricName>().is_err(), "{bad:?}");
        }
        assert_eq!(
            MetricName::join("medium", "brush.p50_ms")
                .unwrap()
                .to_string(),
            "medium.brush.p50_ms",
            "joined"
        );
    }

    #[test]
    fn units_round_trip_their_names() {
        for unit in Unit::ALL {
            assert_eq!(Unit::named(unit.name()), Some(unit), "{unit:?}");
        }
        assert_eq!(Unit::named("µs"), None, "names are ASCII");
    }

    #[test]
    fn the_spread_is_half_the_range_over_the_median() {
        let m = Measured::of(vec![10.0, 9.0, 12.0]).unwrap();
        assert!((m.median() - 10.0).abs() < 1e-12, "odd count: {m:?}");
        assert!((m.spread() - 0.15).abs() < 1e-12, "(12 − 9) / 20: {m:?}");
        assert_eq!(m.samples(), [10.0, 9.0, 12.0], "in the order taken");
        let even = Measured::of(vec![4.0, 1.0, 3.0, 2.0]).unwrap();
        assert!((even.median() - 2.5).abs() < 1e-12, "even count: {even:?}");
        let five = Measured::of(vec![10.0, 30.0, 9.0, 11.0, 1.0]).unwrap();
        assert!((five.median() - 10.0).abs() < 1e-12, "{five:?}");
        assert!(
            (five.spread() - 0.1).abs() < 1e-12,
            "five: the extremes left out, (11 − 9) / 20: {five:?}"
        );
        let one = Measured::of(vec![7.0]).unwrap();
        assert!(one.spread().abs() < 1e-12, "one sample has no spread");
        let negative = Measured::of(vec![-10.0, -12.0, -8.0]).unwrap();
        assert!(
            (negative.spread() - 0.2).abs() < 1e-12,
            "over |median|: {negative:?}"
        );
        let zero = Measured::of(vec![0.0, 0.0, 0.0]).unwrap();
        assert!(zero.spread().abs() < 1e-12, "a zero median: {zero:?}");
        assert_eq!(Measured::of(Vec::new()), Err(BadSamples::Empty), "none");
        assert_eq!(
            Measured::of(vec![1.0, f64::NAN]),
            Err(BadSamples::NotFinite),
            "NaN"
        );
        assert!(
            Measured::recorded(1.0, -0.1, Vec::new()).is_err(),
            "negative spread"
        );
    }
}
