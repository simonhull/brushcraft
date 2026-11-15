//! A stroke over one still landing, from the app's over script read back.
//!
//! The script (`brushcraft --script --script-path over`, its record written with `--record`)
//! draws pairs of strokes, the later along the earlier while that one's dabs still land (the
//! stroke-while-landing spec, "Cost and latency"). The earlier stroke is of two weights:
//! `ordinary` (the suite's 50,000-vertex brush run fast across the sphere: some tens of dabs not
//! landed on each tile under the later stroke) and `heavy` (a brush at the footprint limit
//! scribbled over one span: hundreds, piled on the same tiles); each is met three ways, a pair each
//! on ground of its own, the later stroke's frames in a segment of its own (`over_ordinary_cold` to
//! `over_heavy_landed`):
//!
//! - **cold**: the pen comes down at once with no hover before, so its `Down` makes every future it
//!   reads;
//! - **warmed**: it hovers where it comes down first, long enough for the futures and the seams
//!   there to be made ahead;
//! - **landed**: it comes down once the earlier stroke has landed: a stroke alone, the reference
//!   the other two are read against, in the same run.
//!
//! Each weight gives ([`OverMetrics`]):
//!
//! - **`Down` to first dab shown**, cold, warmed and landed: from the later stroke's `Down` (the
//!   segment's `down_ms`) to the present submitted of the first frame begun at or after it that
//!   showed a pen sample's dabs first. A `Down` places no dab itself (the stroke's first comes with
//!   its first drag, a pen report later), so this is that report's interval more than the frame's
//!   own `sample_to_present_ms`; the spec's target is the pen's latency, 8 ms, once warmed;
//! - **a steady frame**, over the stroke still landing (the warmed pair's later stroke) and once it
//!   has landed: the median sample to present of the later stroke's pen frames after its first, the
//!   pen drawing on along the earlier stroke;
//! - **the open strokes' peak**: the most the open strokes' pen-down data held (the record's
//!   `kept_bytes`: futures, previews and the seam tables included) over the cold and the warmed
//!   pairs, the earlier stroke's frames included. The part's notes say what share of the open
//!   strokes' allowance that is (the run line's `open_strokes_share`).
//!
//! The script's last ground is a **chain**: an ordinary stroke, a second at once along it, and a
//! third at once over both (`over_chain`): the third's `Down` to its first dab shown, and the
//! three strokes' peak.
//!
//! **What a sample needs.** The cold and the warmed `Down` are samples only if the earlier stroke
//! still had dabs to land as the pen came down (the last frame begun before it touched says how
//! many, `dabs_catching_up`), the landed one only if it had none: a machine that lands the
//! earlier stroke before the warmed `Down`, or not before the landed one, gives no sample of that
//! metric, and the part's note says why ([`OverMetrics::notes`]). A record that doesn't say when
//! the pen touched (an older app's) gives none at all.
//!
//! The frame taken as the later stroke's first is the first begun once its pen had touched that
//! showed any sample's dabs first: the earlier stroke's last dabs are shown by a frame begun
//! before that (its lift is half the script's gap before the segment, a whole gap before the
//! `Down`), or in the same one. A `Down` that waits for the strokes before it (past the open
//! strokes' share of the budget) shows its first dab only once they have landed: its time is
//! that wait, as it should be.
//!
//! No ceiling yet: the metrics are compared with their history only (the first runs put the
//! warmed `Down` far above the pen's 8 ms on the medium project; see the stroke-while-landing
//! spec's "As built").

use crate::perf::metric::Unit;
use crate::perf::render::{Frame, Recorded, Value, percentile};

/// The script's segment in which the earlier strokes are drawn.
const FIRST: &str = "over_first";

/// The script's segment in which the chain's third stroke is drawn.
const CHAIN: &str = "over_chain";

/// One weight's segments and metric names after the scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Case {
    /// The later stroke's segment with no hover before its `Down`.
    cold: &'static str,
    /// Its segment after a hover.
    warmed: &'static str,
    /// Its segment once the earlier stroke has landed.
    landed: &'static str,
    /// The metrics' names, in [`OverMetrics::named`]'s order.
    names: [&'static str; 6],
}

/// One weight's [`Case`], by its name in segments and metrics.
macro_rules! case {
    ($weight:literal) => {
        Case {
            cold: concat!("over_", $weight, "_cold"),
            warmed: concat!("over_", $weight, "_warmed"),
            landed: concat!("over_", $weight, "_landed"),
            names: [
                concat!("render.over_", $weight, ".cold_down_ms"),
                concat!("render.over_", $weight, ".warmed_down_ms"),
                concat!("render.over_", $weight, ".landed_down_ms"),
                concat!("render.over_", $weight, ".steady_p50_ms"),
                concat!("render.over_", $weight, ".landed_steady_p50_ms"),
                concat!("render.over_", $weight, ".open_strokes_peak_mib"),
            ],
        }
    };
}

/// The weights, in the script's order.
pub(crate) const CASES: [Case; 2] = [case!("ordinary"), case!("heavy")];

/// The chain's third `Down` to its first dab shown, by its name after the scale.
const CHAIN_DOWN: &str = "render.over_chain.third_down_ms";

/// The chain's open strokes' peak, by its name after the scale.
const CHAIN_PEAK: &str = "render.over_chain.open_strokes_peak_mib";

/// A later stroke as its segment's frames show it.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct Later {
    /// Its `Down` to its first dab shown, in ms; `None` if the record doesn't say when that
    /// frame's present was submitted.
    pub(crate) down_ms: Option<f64>,
    /// The median sample to present of its pen frames after the first, in ms; `None` with none.
    pub(crate) steady_ms: Option<f64>,
    /// The earlier strokes' dabs still to land as the pen came down: the `dabs_catching_up` of
    /// the last frame begun before it touched; `None` if the record doesn't say.
    pub(crate) unlanded: Option<f64>,
}

impl Later {
    /// The later stroke drawn in `segment` of `recorded`; `None` if the record doesn't say when
    /// the pen touched in such a segment, or no frame begun in it since showed a pen sample's
    /// dabs.
    fn of(recorded: &Recorded, segment: &str) -> Option<Self> {
        let (_, touched) = recorded.downs.iter().find(|(name, _)| name == segment)?;
        let shows = |frame: &Frame| {
            frame.segment.as_deref() == Some(segment)
                && frame.time_ms >= *touched
                && frame.sample_to_present_ms.is_some()
        };
        let first = recorded.frames.iter().position(shows)?;
        let down = recorded.frames.get(first)?;
        let after = recorded.frames.get(first + 1..).unwrap_or_default();
        let steady: Vec<f64> = after
            .iter()
            .filter(|frame| frame.segment.as_deref() == Some(segment))
            .filter_map(|frame| frame.sample_to_present_ms)
            .collect();
        let mut earlier = recorded.frames.iter().rev();
        let before = earlier.find(|frame| frame.time_ms < *touched);
        Some(Self {
            down_ms: down.submitted_ms.map(|submitted| submitted - touched),
            steady_ms: percentile(&steady, 50),
            unlanded: before.and_then(|frame| frame.dabs_catching_up),
        })
    }

    /// Whether the earlier strokes still had dabs to land as the pen came down; `None` if the
    /// record doesn't say.
    fn over_landing(&self) -> Option<bool> {
        self.unlanded.map(|dabs| dabs > 0.0)
    }
}

/// The open strokes' peak about `segment` of `recorded`, in MiB.
///
/// The most their pen-down data held over the frames from the start of the [`FIRST`] segment
/// before `segment` to `segment`'s end; `None` if the record lacks the segment or no frame says.
fn peak_mib(recorded: &Recorded, segment: &str) -> Option<f64> {
    let at = recorded
        .segments
        .iter()
        .position(|(name, _, _)| name == segment)?;
    let (_, _, end) = recorded.segments.get(at)?;
    let earlier = recorded.segments.get(..at).unwrap_or_default();
    let (_, start, _) = earlier.iter().rev().find(|(name, _, _)| name == FIRST)?;
    let within = recorded
        .frames
        .iter()
        .filter(|frame| (*start..*end).contains(&frame.time_ms));
    let peak = within.filter_map(|frame| frame.kept_bytes).reduce(f64::max);
    peak.map(|bytes| bytes / (1024.0 * 1024.0))
}

/// One weight's metrics from one run; `None` for one the record can't give, or whose pair
/// wasn't what it measures (the module notes, "What a sample needs").
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct OverMetrics {
    /// The later stroke's `Down` to its first dab shown with no hover before, in ms.
    pub(crate) cold_down: Option<f64>,
    /// The same after a hover, in ms.
    pub(crate) warmed_down: Option<f64>,
    /// The same once the earlier stroke has landed, in ms.
    pub(crate) landed_down: Option<f64>,
    /// A steady frame of the warmed pair's later stroke, in ms.
    pub(crate) steady: Option<f64>,
    /// A steady frame once the earlier stroke has landed, in ms.
    pub(crate) landed_steady: Option<f64>,
    /// The open strokes' peak over the cold and the warmed pairs, in MiB.
    pub(crate) peak_mib: Option<f64>,
    /// What the pairs were, and why any gave no sample: one line each.
    pub(crate) notes: Vec<String>,
}

impl OverMetrics {
    /// `case`'s metrics from `recorded`.
    fn of(recorded: &Recorded, case: &Case) -> Self {
        let mut notes = Vec::new();
        // A pair drawn over a stroke still landing, if it was one.
        let mut over = |segment: &str| -> Option<Later> {
            let later = Later::of(recorded, segment)?;
            match later.over_landing() {
                Some(true) => {
                    let dabs = later.unlanded.unwrap_or_default();
                    notes.push(format!("{segment} over {dabs:.0} dabs still to land"));
                    Some(later)
                }
                Some(false) => {
                    notes.push(format!(
                        "{segment}: the earlier stroke had landed as the pen came down, no sample"
                    ));
                    None
                }
                None => None,
            }
        };
        let cold = over(case.cold);
        let warmed = over(case.warmed);
        let landed = Later::of(recorded, case.landed).filter(|later| {
            let alone = later.over_landing() == Some(false);
            if !alone {
                notes.push(format!(
                    "{}: the earlier stroke hadn't landed as the pen came down, no sample",
                    case.landed
                ));
            }
            alone
        });
        let peaks = [case.cold, case.warmed].map(|segment| peak_mib(recorded, segment));
        Self {
            cold_down: cold.and_then(|later| later.down_ms),
            warmed_down: warmed.and_then(|later| later.down_ms),
            landed_down: landed.and_then(|later| later.down_ms),
            steady: warmed.and_then(|later| later.steady_ms),
            landed_steady: landed.and_then(|later| later.steady_ms),
            peak_mib: peaks.into_iter().flatten().reduce(f64::max),
            notes,
        }
    }

    /// Each metric as `(name after the scale, unit, value)`, for `case`.
    pub(crate) fn named(&self, case: &Case) -> [(&'static str, Unit, Value); 6] {
        let [cold, warmed, landed, steady, landed_steady, peak] = case.names;
        [
            (cold, Unit::Ms, self.cold_down.into()),
            (warmed, Unit::Ms, self.warmed_down.into()),
            (landed, Unit::Ms, self.landed_down.into()),
            (steady, Unit::Ms, self.steady.into()),
            (landed_steady, Unit::Ms, self.landed_steady.into()),
            (peak, Unit::Mib, self.peak_mib.into()),
        ]
    }
}

/// The chain's metrics from one run; `None` for one the record can't give.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct ChainMetrics {
    /// The third stroke's `Down` to its first dab shown, over two strokes still landing, in ms.
    pub(crate) third_down: Option<f64>,
    /// The three strokes' peak, in MiB.
    pub(crate) peak_mib: Option<f64>,
    /// What the chain was, or why it gave no sample.
    pub(crate) notes: Vec<String>,
}

impl ChainMetrics {
    /// The chain's metrics from `recorded`.
    fn of(recorded: &Recorded) -> Self {
        let mut notes = Vec::new();
        let third = Later::of(recorded, CHAIN).filter(|later| match later.over_landing() {
            Some(true) => {
                let dabs = later.unlanded.unwrap_or_default();
                notes.push(format!("{CHAIN} over {dabs:.0} dabs still to land"));
                true
            }
            Some(false) => {
                notes.push(format!(
                    "{CHAIN}: the strokes before had landed as the pen came down, no sample"
                ));
                false
            }
            None => false,
        });
        Self {
            third_down: third.and_then(|later| later.down_ms),
            peak_mib: peak_mib(recorded, CHAIN),
            notes,
        }
    }

    /// Each metric as `(name after the scale, unit, value)`.
    pub(crate) fn named(&self) -> [(&'static str, Unit, Value); 2] {
        [
            (CHAIN_DOWN, Unit::Ms, self.third_down.into()),
            (CHAIN_PEAK, Unit::Mib, self.peak_mib.into()),
        ]
    }
}

/// An over run's metrics: each weight's, in [`CASES`]' order, and the chain's.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct Derived {
    /// Each weight's metrics.
    pub(crate) cases: [OverMetrics; 2],
    /// The chain's.
    pub(crate) chain: ChainMetrics,
    /// The open strokes' allowance in the run, in MiB, if the record gives it.
    pub(crate) share_mib: Option<f64>,
}

/// The over metrics of `recorded`.
pub(crate) fn derive(recorded: &Recorded) -> Derived {
    Derived {
        cases: CASES.map(|case| OverMetrics::of(recorded, &case)),
        chain: ChainMetrics::of(recorded),
        share_mib: recorded
            .open_strokes_share
            .map(|bytes| bytes / (1024.0 * 1024.0)),
    }
}

impl Derived {
    /// Every metric as `(name after the scale, unit, value)`.
    pub(crate) fn named(&self) -> Vec<(&'static str, Unit, Value)> {
        let cases = CASES.iter().zip(&self.cases);
        let mut named: Vec<(&'static str, Unit, Value)> = cases
            .flat_map(|(case, metrics)| metrics.named(case))
            .collect();
        named.extend(self.chain.named());
        named
    }

    /// What the run's pairs were, for the part's notes: each pair's line, and the largest peak
    /// against the open strokes' allowance.
    pub(crate) fn notes(&self) -> Vec<String> {
        let cases = self.cases.iter().flat_map(|metrics| metrics.notes.iter());
        let mut notes: Vec<String> = cases.chain(&self.chain.notes).cloned().collect();
        let peaks = self.cases.iter().map(|metrics| metrics.peak_mib);
        let peak = peaks
            .chain([self.chain.peak_mib])
            .flatten()
            .reduce(f64::max);
        if let Some((peak, share)) = peak.zip(self.share_mib).filter(|&(_, share)| share > 0.0) {
            notes.push(format!(
                "open strokes held at most {peak:.1} MiB of their {share:.0} MiB ({:.1} %)",
                100.0 * peak / share
            ));
        }
        notes
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;
    use crate::perf::metric::MetricName;

    /// A frame of `segment` at `time_ms`, its present submitted 3 ms later, with `catching_up`
    /// dabs still to land and the open strokes holding `kept_mib`; showing a pen sample's dabs
    /// first, `latency` ms old, if given.
    fn frame(
        segment: &str,
        time_ms: f64,
        latency: Option<f64>,
        catching_up: f64,
        kept_mib: f64,
    ) -> Frame {
        Frame {
            time_ms,
            segment: Some(segment.to_owned()),
            submitted_ms: Some(time_ms + 3.0),
            sample_to_present_ms: latency,
            dabs_catching_up: Some(catching_up),
            kept_bytes: Some(kept_mib * 1024.0 * 1024.0),
            ..Frame::default()
        }
    }

    /// Adds a pair's frames to `recorded`, from `start`.
    ///
    /// The earlier stroke's (100 ms, the open strokes holding 10 MiB), then the later stroke's in
    /// `segment`, its pen touching 25 ms into it with `unlanded` dabs of the earlier stroke still
    /// to land. Its first dab is shown by a frame begun `first` ms after the touch, then three
    /// more pen frames 4, 6 and 5 ms behind their samples; the open strokes hold `kept_mib` at
    /// their most.
    fn pair(
        recorded: &mut Recorded,
        segment: &str,
        start: f64,
        (unlanded, first, kept_mib): (f64, f64, f64),
    ) {
        let edge = start + 100.0;
        let touched = edge + 25.0;
        let segments = [(FIRST, start, edge), (segment, edge, edge + 400.0)];
        let named = segments.map(|(name, from, to)| (name.to_owned(), from, to));
        recorded.segments.extend(named);
        recorded.downs.push((FIRST.to_owned(), start + 10.0));
        recorded.downs.push((segment.to_owned(), touched));
        recorded.frames.extend([
            frame(FIRST, start + 20.0, Some(3.0), 0.0, 10.0),
            frame(FIRST, start + 90.0, Some(3.0), unlanded + 5.0, 10.0),
            // The earlier stroke's last dabs, shown by a frame begun in the later one's segment.
            frame(segment, edge + 5.0, Some(30.0), unlanded, 10.0),
            frame(
                segment,
                touched + first,
                Some(first + 3.0),
                unlanded,
                kept_mib,
            ),
            frame(segment, touched + 60.0, Some(4.0), 0.0, kept_mib / 2.0),
            frame(segment, touched + 70.0, Some(6.0), 0.0, 1.0),
            frame(segment, touched + 80.0, Some(5.0), 0.0, 1.0),
            frame(segment, touched + 300.0, None, 0.0, 0.0),
        ]);
    }

    /// An over run: both weights, each cold, warmed and landed, and the chain.
    fn run() -> Recorded {
        let mut recorded = Recorded {
            open_strokes_share: Some(1024.0 * 1024.0 * 1024.0),
            ..Recorded::default()
        };
        pair(&mut recorded, "over_ordinary_cold", 0.0, (40.0, 20.0, 64.0));
        pair(
            &mut recorded,
            "over_ordinary_warmed",
            1000.0,
            (30.0, 2.0, 96.0),
        );
        pair(
            &mut recorded,
            "over_ordinary_landed",
            2000.0,
            (0.0, 1.0, 12.0),
        );
        pair(
            &mut recorded,
            "over_heavy_cold",
            3000.0,
            (400.0, 120.0, 128.0),
        );
        pair(
            &mut recorded,
            "over_heavy_warmed",
            4000.0,
            (300.0, 9.0, 100.0),
        );
        pair(&mut recorded, "over_heavy_landed", 5000.0, (0.0, 1.5, 12.0));
        pair(&mut recorded, CHAIN, 6000.0, (90.0, 50.0, 256.0));
        recorded
    }

    /// Whether `value` is a sample within rounding of `wanted`.
    fn is(value: Option<f64>, wanted: f64) -> bool {
        value.is_some_and(|value| (value - wanted).abs() < 1e-9)
    }

    #[test]
    fn a_run_gives_each_weights_downs_its_steady_frames_and_its_peak() {
        let derived = derive(&run());
        let [ordinary, heavy] = &derived.cases;
        // From the touch to the present submitted of the first frame begun since that showed
        // dabs: its start after the touch, and 3 ms.
        assert!(is(ordinary.cold_down, 23.0), "{ordinary:?}");
        assert!(is(ordinary.warmed_down, 5.0), "{ordinary:?}");
        assert!(is(ordinary.landed_down, 4.0), "{ordinary:?}");
        assert!(is(heavy.cold_down, 123.0), "{heavy:?}");
        assert!(is(heavy.warmed_down, 12.0), "{heavy:?}");
        assert!(is(heavy.landed_down, 4.5), "{heavy:?}");
        // The median of the pen frames after the first: 4, 6 and 5 ms.
        assert!(is(ordinary.steady, 5.0), "the warmed pair's");
        assert!(is(ordinary.landed_steady, 5.0), "the landed pair's");
        // The most of the cold and the warmed pairs, the earlier stroke's frames included.
        assert!(is(ordinary.peak_mib, 96.0), "{ordinary:?}");
        assert!(is(heavy.peak_mib, 128.0), "{heavy:?}");
        assert!(is(derived.chain.third_down, 53.0), "{:?}", derived.chain);
        assert!(is(derived.chain.peak_mib, 256.0), "{:?}", derived.chain);
        assert!(is(derived.share_mib, 1024.0), "the run's allowance");
        let notes = derived.notes();
        assert!(
            notes.contains(&"over_heavy_cold over 400 dabs still to land".to_owned()),
            "{notes:?}"
        );
        assert_eq!(
            notes.last().map(String::as_str),
            Some("open strokes held at most 256.0 MiB of their 1024 MiB (25.0 %)"),
            "the largest peak against the allowance"
        );
    }

    #[test]
    fn every_metric_is_named_once_for_its_scale() {
        let named = derive(&run()).named();
        assert_eq!(named.len(), 14, "six a weight and the chain's two");
        for &(name, unit, value) in &named {
            let joined = MetricName::join("medium", name).unwrap();
            assert!(joined.as_str().starts_with("medium.render.over_"), "{name}");
            assert!(matches!(value, Value::Of(_)), "{name}: {value:?}");
            let wanted = if name.ends_with("_mib") {
                Unit::Mib
            } else {
                Unit::Ms
            };
            assert_eq!(unit, wanted, "{name}");
        }
        let mut unique: Vec<&str> = named.iter().map(|&(name, _, _)| name).collect();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), 14, "each named once");
    }

    #[test]
    fn a_pair_that_isnt_what_it_measures_gives_no_sample_and_says_why() {
        // The earlier stroke had landed by the warmed `Down`, and hadn't by the landed one.
        let mut recorded = Recorded::default();
        pair(&mut recorded, "over_ordinary_cold", 0.0, (40.0, 20.0, 64.0));
        pair(
            &mut recorded,
            "over_ordinary_warmed",
            1000.0,
            (0.0, 2.0, 96.0),
        );
        pair(
            &mut recorded,
            "over_ordinary_landed",
            2000.0,
            (7.0, 1.0, 12.0),
        );
        let derived = derive(&recorded);
        let [ordinary, heavy] = &derived.cases;
        assert!(ordinary.cold_down.is_some(), "the cold pair stands");
        assert_eq!(
            ordinary.warmed_down, None,
            "landed before the pen came down"
        );
        assert_eq!(ordinary.steady, None, "nor its steady frame");
        assert_eq!(
            ordinary.landed_down, None,
            "not landed as the pen came down"
        );
        assert_eq!(ordinary.landed_steady, None, "nor its steady frame");
        assert!(is(ordinary.peak_mib, 96.0), "what they held all the same");
        assert_eq!(
            ordinary.notes,
            [
                "over_ordinary_cold over 40 dabs still to land",
                "over_ordinary_warmed: the earlier stroke had landed as the pen came down, no \
                 sample",
                "over_ordinary_landed: the earlier stroke hadn't landed as the pen came down, no \
                 sample",
            ]
        );
        // The other weight, and the chain, weren't in the record.
        assert_eq!(heavy, &OverMetrics::default(), "no heavy pair");
        assert_eq!(derived.chain, ChainMetrics::default(), "no chain");
        let named = derived.named();
        let lacking = named
            .iter()
            .filter(|(_, _, value)| *value == Value::Lacking);
        assert_eq!(lacking.count(), 4 + 6 + 2, "{named:?}");
        assert_eq!(derived.notes().len(), 3, "no share, so no line of it");
    }

    #[test]
    fn a_record_that_doesnt_say_when_the_pen_touched_gives_no_down() {
        // An older app's record: the segments, without their `down_ms`.
        let mut recorded = run();
        recorded.downs.clear();
        let derived = derive(&recorded);
        let timed = derived
            .named()
            .into_iter()
            .filter(|(name, _, _)| name.ends_with("_ms"));
        for (name, _, value) in timed {
            assert_eq!(value, Value::Lacking, "{name}");
        }
        let [ordinary, _] = &derived.cases;
        assert!(is(ordinary.peak_mib, 96.0), "the peak needs none");
    }

    #[test]
    fn a_strokes_record_has_no_over_metric() {
        let recorded = Recorded {
            segments: vec![("stroke_near".to_owned(), 0.0, 1000.0)],
            downs: vec![("stroke_near".to_owned(), 500.0)],
            frames: vec![frame("stroke_near", 600.0, Some(3.0), 0.0, 1.0)],
            ..Recorded::default()
        };
        let derived = derive(&recorded);
        assert_eq!(derived, Derived::default(), "nothing derived");
        assert!(derived.notes().is_empty(), "nothing said");
    }
}
