//! The noise rule: whether a metric moved beyond its own noise since the baseline.
//!
//! A change counts when it exceeds `max(5 %, 3 × spread)`, the spread being the larger of the
//! baseline's and this run's (see [`crate::perf::metric::Measured::spread`]). Five percent is
//! the floor: below it, a quiet desktop's drift between runs would alarm on every metric. Three
//! spreads covers the noisier metrics: when one run's repetitions of a metric land ±10 % around
//! its median, two runs 12 % apart haven't shown it moved. The larger of the two spreads is used
//! because either run's noise can make the difference.
//!
//! The band has a ceiling: when it would pass [`MAX_THRESHOLD`] (25 %), the comparison can't
//! tell a 30 % slowdown from noise, so a change inside it is [`Verdict::Noisy`], inconclusive,
//! rather than "same". A spread of 0.34 would otherwise call a doubling unchanged. 25 % is five
//! times the floor, and about the smallest slowdown nobody would want to merge unseen. A change
//! beyond even a wide band is still flagged.

use std::fmt;

use crate::perf::metric::{Measured, Metric};

/// The least relative change ever flagged.
pub(crate) const MIN_CHANGE: f64 = 0.05;

/// How many spreads a change must exceed to be flagged.
pub(crate) const SPREADS: f64 = 3.0;

/// The widest noise band a comparison may conclude "same" within.
pub(crate) const MAX_THRESHOLD: f64 = 0.25;

/// A relative change against a baseline, and the threshold it was held to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Change {
    /// `(now − baseline) / |baseline|`: positive is higher, which is worse for every metric.
    /// Infinite if the baseline was 0 and now isn't.
    pub(crate) ratio: f64,
    /// `max(MIN_CHANGE, SPREADS × spread)`.
    pub(crate) threshold: f64,
}

impl fmt::Display for Change {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:+.1} % (noise ±{:.1} %)",
            self.ratio * 100.0,
            self.threshold * 100.0
        )
    }
}

/// What a metric did against its baseline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Verdict {
    /// No baseline to compare with.
    New,
    /// Within the noise, and the noise narrow enough to say so.
    Same(Change),
    /// Within the noise, but the noise is wider than [`MAX_THRESHOLD`]: inconclusive.
    Noisy(Change),
    /// Lower beyond the noise.
    Better(Change),
    /// Higher beyond the noise: flagged.
    Worse(Change),
}

impl Verdict {
    /// The verdict on `now` against `baseline` (see the module notes); [`Verdict::New`] without
    /// one.
    pub(crate) fn judge(baseline: Option<&Measured>, now: &Measured) -> Self {
        let Some(baseline) = baseline else {
            return Self::New;
        };
        let threshold = MIN_CHANGE.max(SPREADS * baseline.spread().max(now.spread()));
        let difference = now.median() - baseline.median();
        let ratio = if baseline.median() == 0.0 {
            if difference == 0.0 {
                0.0
            } else {
                difference.signum() * f64::INFINITY
            }
        } else {
            difference / baseline.median().abs()
        };
        let change = Change { ratio, threshold };
        if ratio > threshold {
            Self::Worse(change)
        } else if ratio < -threshold {
            Self::Better(change)
        } else if threshold > MAX_THRESHOLD {
            Self::Noisy(change)
        } else {
            Self::Same(change)
        }
    }

    /// The verdict's word in the comparison table.
    pub(crate) const fn word(&self) -> &'static str {
        match self {
            Self::New => "new",
            Self::Same(_) => "same",
            Self::Noisy(_) => "NOISY",
            Self::Better(_) => "better",
            Self::Worse(_) => "WORSE",
        }
    }

    /// The change, if there was a baseline.
    pub(crate) const fn change(&self) -> Option<Change> {
        match self {
            Self::New => None,
            Self::Same(change)
            | Self::Noisy(change)
            | Self::Better(change)
            | Self::Worse(change) => Some(*change),
        }
    }
}

/// A metric against its ceiling (the spec's target, such as peak RSS within budget + 512 MB):
/// judged on every run, independent of any baseline, since a baseline can miss it too.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Target {
    /// The metric has no ceiling.
    Unset,
    /// The median is at or under it.
    Met,
    /// The median is over it: flagged.
    Missed {
        /// How far over, in the metric's unit.
        by: f64,
    },
}

impl Target {
    /// `metric` against its ceiling.
    pub(crate) fn check(metric: &Metric) -> Self {
        match metric.ceiling {
            None => Self::Unset,
            Some(ceiling) if metric.value.median() <= ceiling => Self::Met,
            Some(ceiling) => Self::Missed {
                by: metric.value.median() - ceiling,
            },
        }
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    /// A value with median `median` and spread `spread` (no samples).
    fn value(median: f64, spread: f64) -> Measured {
        Measured::recorded(median, spread, Vec::new()).unwrap()
    }

    #[test]
    fn without_a_baseline_a_metric_is_new() {
        assert_eq!(Verdict::judge(None, &value(1.0, 0.0)), Verdict::New, "new");
        assert_eq!(Verdict::New.change(), None, "no change");
    }

    /// Quiet metrics are held to 5 %: 4.9 % is noise, 5.1 % is flagged, either way.
    #[test]
    fn quiet_metrics_are_held_to_five_percent() {
        let base = value(100.0, 0.001);
        assert!(
            matches!(
                Verdict::judge(Some(&base), &value(104.9, 0.0)),
                Verdict::Same(_)
            ),
            "+4.9 %"
        );
        assert!(
            matches!(
                Verdict::judge(Some(&base), &value(105.1, 0.0)),
                Verdict::Worse(_)
            ),
            "+5.1 %"
        );
        assert!(
            matches!(
                Verdict::judge(Some(&base), &value(95.1, 0.0)),
                Verdict::Same(_)
            ),
            "−4.9 %"
        );
        assert!(
            matches!(
                Verdict::judge(Some(&base), &value(94.9, 0.0)),
                Verdict::Better(_)
            ),
            "−5.1 %"
        );
    }

    /// Noisy metrics are held to three spreads, the larger of the two runs'.
    #[test]
    fn noisy_metrics_are_held_to_three_spreads() {
        let noisy = value(100.0, 0.04);
        let change = |now: &Measured| Verdict::judge(Some(&noisy), now);
        assert!(
            matches!(change(&value(111.0, 0.0)), Verdict::Same(_)),
            "+11 % within 12 %"
        );
        assert!(
            matches!(change(&value(113.0, 0.0)), Verdict::Worse(_)),
            "+13 % beyond 12 %"
        );
        let quiet = value(100.0, 0.0);
        assert!(
            matches!(
                Verdict::judge(Some(&quiet), &value(111.0, 0.04)),
                Verdict::Same(_)
            ),
            "this run's noise counts too"
        );
        let Verdict::Worse(c) = change(&value(113.0, 0.01)) else {
            panic!("flagged");
        };
        assert!((c.ratio - 0.13).abs() < 1e-12, "{c:?}");
        assert!((c.threshold - 0.12).abs() < 1e-12, "{c:?}");
        assert_eq!(c.to_string(), "+13.0 % (noise ±12.0 %)", "shown");
    }

    /// Past the 25 % ceiling a change inside the band is inconclusive, not "same": the
    /// reviewer's case, a run at spread 0.34 doubling from 31 to 62, is no longer passed.
    #[test]
    fn wide_bands_are_inconclusive() {
        let base = value(31.0, 0.0);
        assert!(
            matches!(
                Verdict::judge(Some(&base), &value(62.0, 0.34)),
                Verdict::Noisy(_)
            ),
            "a doubling inside a ±102 % band"
        );
        assert!(
            matches!(
                Verdict::judge(Some(&base), &value(31.0, 0.084)),
                Verdict::Noisy(_)
            ),
            "a band of 25.2 %"
        );
        assert!(
            matches!(
                Verdict::judge(Some(&base), &value(31.0, 0.083)),
                Verdict::Same(_)
            ),
            "a band of 24.9 %"
        );
        assert!(
            matches!(
                Verdict::judge(Some(&base), &value(93.0, 0.34)),
                Verdict::Worse(_)
            ),
            "beyond even a wide band"
        );
        assert_eq!(
            Verdict::judge(Some(&base), &value(62.0, 0.34)).word(),
            "NOISY",
            "shown"
        );
    }

    /// A ceiling is judged on the run alone: over it is missed, at it is met.
    #[test]
    fn ceilings_are_judged_without_a_baseline() {
        let metric = |median: f64, ceiling| Metric {
            name: "medium.budget.peak_rss_mib".parse().unwrap(),
            unit: crate::perf::metric::Unit::Mib,
            value: value(median, 0.0),
            ceiling,
        };
        assert_eq!(
            Target::check(&metric(2000.0, None)),
            Target::Unset,
            "no ceiling"
        );
        assert_eq!(
            Target::check(&metric(1536.0, Some(1536.0))),
            Target::Met,
            "at it"
        );
        assert_eq!(
            Target::check(&metric(1600.0, Some(1536.0))),
            Target::Missed { by: 64.0 },
            "over it"
        );
    }

    #[test]
    fn negative_and_zero_baselines_compare_by_magnitude() {
        let below = value(-100.0, 0.0);
        assert!(
            matches!(
                Verdict::judge(Some(&below), &value(-80.0, 0.0)),
                Verdict::Worse(_)
            ),
            "higher is worse below zero too"
        );
        let zero = value(0.0, 0.0);
        assert!(
            matches!(
                Verdict::judge(Some(&zero), &value(0.0, 0.0)),
                Verdict::Same(_)
            ),
            "0 to 0"
        );
        assert!(
            matches!(
                Verdict::judge(Some(&zero), &value(1.0, 0.0)),
                Verdict::Worse(_)
            ),
            "0 to 1"
        );
        assert!(
            matches!(
                Verdict::judge(Some(&zero), &value(-1.0, 0.0)),
                Verdict::Better(_)
            ),
            "0 to −1"
        );
    }
}
