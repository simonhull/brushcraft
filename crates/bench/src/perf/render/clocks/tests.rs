//! Reading `nvidia-smi`'s lines, finding the sample nearest a moment, and judging full clocks, on
//! hand-made samples.

use super::*;

/// A sample of GPU `gpu` at `at` ms: graphics clock `mhz`, power state `state` if any.
fn sample(at: f64, gpu: u32, mhz: f64, state: Option<u8>) -> ClockSample {
    ClockSample {
        at_unix_ms: at,
        gpu,
        graphics_mhz: mhz,
        memory_mhz: None,
        pstate: state.map(PState),
    }
}

#[test]
fn a_line_reads_as_its_clocks_and_power_state() {
    let full = "0, 2745, 10501, P0";
    assert_eq!(
        ClockSample::parse(full, 12.5),
        Some(ClockSample {
            at_unix_ms: 12.5,
            gpu: 0,
            graphics_mhz: 2745.0,
            memory_mhz: Some(10_501.0),
            pstate: Some(PState::P0),
        })
    );
    let unreported = ClockSample::parse("1, 1005, [N/A], [N/A]", 0.0).unwrap();
    assert_eq!(
        (unreported.gpu, unreported.memory_mhz, unreported.pstate),
        (1, None, None)
    );
    let not_samples = [
        "",
        "index, clocks.current.graphics [MHz], clocks.current.memory [MHz], pstate",
        "0, [N/A], 10501, P0",
        "0, 2745, 10501",
        "0, 2745, 10501, P0, extra",
        "No devices were found",
    ];
    for line in not_samples {
        assert_eq!(ClockSample::parse(line, 0.0), None, "{line:?}");
    }
}

#[test]
fn power_states_run_from_p0_to_p15() {
    assert_eq!("P0".parse(), Ok(PState::P0));
    assert_eq!("P15".parse(), Ok(PState(15)));
    assert_eq!(PState(3).to_string(), "P3");
    for bad in ["P16", "P", "p0", "0", "P-1", "[N/A]"] {
        assert_eq!(bad.parse::<PState>(), Err(BadPState), "{bad:?}");
    }
    assert!(PState::P0 < PState(5), "P0 is the most performance");
}

#[test]
fn a_moment_takes_the_nearest_sample_within_two_spacings() {
    let samples = [
        sample(1000.0, 0, 2700.0, Some(0)),
        sample(1050.0, 0, 2600.0, Some(0)),
        sample(1300.0, 0, 900.0, Some(5)),
    ];
    let at = |ms: f64| nearest(&samples, ms).map(|s| s.graphics_mhz);
    assert_eq!(at(1000.0), Some(2700.0), "on a sample");
    assert_eq!(at(1020.0), Some(2700.0), "nearer the one before");
    assert_eq!(at(1030.0), Some(2600.0), "nearer the one after");
    assert_eq!(at(1150.0), Some(2600.0), "100 ms after the last before");
    assert_eq!(at(1175.0), None, "a gap in the sampling");
    assert_eq!(at(900.0), Some(2700.0), "before the first, within reach");
    assert_eq!(at(899.0), None, "before the first, out of reach");
    assert_eq!(at(1400.0), Some(900.0), "after the last, within reach");
    assert_eq!(nearest(&[], 1000.0), None);
}

#[test]
fn full_clocks_are_p0_within_a_tenth_of_the_runs_highest_clock() {
    let samples = [
        sample(0.0, 0, 2700.0, Some(0)),
        sample(50.0, 0, 1005.0, Some(3)),
        sample(100.0, 0, 2500.0, Some(2)),
    ];
    let full = FullClocks::of(&samples).unwrap();
    assert!((full.floor_mhz - 2430.0).abs() < 1e-9, "{full:?}");
    assert!(full.holds(&sample(0.0, 0, 2430.0, Some(0))), "at the floor");
    assert!(!full.holds(&sample(0.0, 0, 2429.0, Some(0))), "below it");
    assert!(
        !full.holds(&sample(0.0, 0, 2700.0, Some(2))),
        "fast, but not P0"
    );
    assert!(
        full.holds(&sample(0.0, 0, 2600.0, None)),
        "no power state reported: the clock alone"
    );
    // A run that never left a low state doesn't call it full.
    let idle = [
        sample(0.0, 0, 555.0, Some(5)),
        sample(50.0, 0, 555.0, Some(5)),
    ];
    let idle_full = FullClocks::of(&idle).unwrap();
    assert!(!idle.iter().any(|s| idle_full.holds(s)));
    assert_eq!(FullClocks::of(&[]), Err(NoClocks::NoSamples));
    let two = [
        sample(0.0, 0, 2700.0, Some(0)),
        sample(0.0, 1, 2100.0, Some(0)),
    ];
    assert_eq!(FullClocks::of(&two), Err(NoClocks::SeveralGpus));
}

#[test]
fn every_clock_state_without_samples_says_why() {
    let reasons = [
        NoClocks::NotSampled,
        NoClocks::NoSampler(std::io::ErrorKind::NotFound),
        NoClocks::NoSamples,
        NoClocks::SeveralGpus,
        NoClocks::OtherAdapter,
        NoClocks::NoClockOrigin,
    ];
    let said: Vec<String> = reasons.iter().map(ToString::to_string).collect();
    assert!(said.iter().all(|why| !why.is_empty()), "{said:?}");
    assert_eq!(Clocks::default(), Clocks::Unknown(NoClocks::NotSampled));
}
