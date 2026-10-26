//! Both GPU draw shaders against `brush::draw` on the CPU, on lavapipe.
//!
//! The differences are measured and printed (`cargo nextest run -p render --features spike
//! --no-capture spike::brush`) for the spike report; the assertions bound them loosely, so a
//! shader that computes something else fails while rounding differences pass.

use super::*;
use crate::gpu::GpuTest;

/// Positions per comparison, the M0 benchmark's 50k-vertex brush step.
const POSITIONS: usize = 50_000;

/// Random brushes each comparison runs over the same positions.
const BRUSHES: usize = 20;

/// A seeded random generator (splitmix64), so every run compares the same values.
struct Rng(u64);

impl Rng {
    /// The next 64 random bits.
    const fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A float uniform in `[lo, hi]`.
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        // The top 16 bits, scaled to [0, 1].
        let unit =
            f32::from(u16::try_from(self.next_u64() >> 48).expect("16 bits")) / f32::from(u16::MAX);
        lo + (hi - lo) * unit
    }
}

/// `n` random positions in the cube `[-1, 1]³`.
fn positions(rng: &mut Rng, n: usize) -> Vec<[f32; 3]> {
    (0..n)
        .map(|_| {
            [
                rng.uniform(-1.0, 1.0),
                rng.uniform(-1.0, 1.0),
                rng.uniform(-1.0, 1.0),
            ]
        })
        .collect()
}

/// A random brush: centre in the cube, radius in `[0.05, 1.5]`, strength in `[-1, 1]`.
fn random_brush(rng: &mut Rng) -> DrawBrush {
    let center = [
        rng.uniform(-1.0, 1.0),
        rng.uniform(-1.0, 1.0),
        rng.uniform(-1.0, 1.0),
    ];
    DrawBrush::new(center, rng.uniform(0.05, 1.5), rng.uniform(-1.0, 1.0)).expect("a valid radius")
}

/// Runs [`BRUSHES`] random brushes over [`POSITIONS`] random positions through `shader`, and
/// measures the deltas against the CPU's.
fn compare(gpu: &Gpu, shader: Shader) -> Difference {
    let draw = GpuDraw::new(gpu, shader).expect("the shader builds");
    let mut rng = Rng(0x5EED_B0A5);
    let points = positions(&mut rng, POSITIONS);
    let brushes: Vec<DrawBrush> = (0..BRUSHES).map(|_| random_brush(&mut rng)).collect();
    let inside: usize = brushes
        .iter()
        .map(|brush| brush.cpu(&points).iter().filter(|&&d| d != 0.0).count())
        .sum();
    assert!(
        inside > POSITIONS && inside < POSITIONS * BRUSHES,
        "the brushes cover some vertices and not others: {inside}"
    );
    against_cpu(gpu, &draw, &points, &brushes).expect("the draws")
}

#[test]
fn the_params_block_is_laid_out_as_the_shaders_declare_it() {
    assert_eq!(size_of::<Params>(), 32, "two 16-byte rows");
    assert_eq!(
        std::mem::offset_of!(Params, radius_squared),
        16,
        "radius fills the centre's fourth lane"
    );
}

#[test]
fn a_radius_that_is_not_positive_and_finite_is_rejected() {
    for radius in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(
            matches!(
                DrawBrush::new([0.0; 3], radius, 1.0),
                Err(DrawError::Radius(_))
            ),
            "radius {radius}"
        );
    }
}

/// Asserts `difference` is what this adapter allows: bit-identical on a software adapter
/// (lavapipe, where the gate runs; measured), within rounding of the strength on any other.
fn assert_matches_cpu(gpu: &Gpu, shader: Shader, difference: &Difference) {
    if gpu.adapter_info().device_type == wgpu::DeviceType::Cpu {
        assert_eq!(
            difference.differing(),
            0,
            "{shader:?} matches the CPU bit for bit on a software adapter: {difference}"
        );
    } else {
        // Strengths are below 1, so this is a relative bound too: rounding, not a different
        // computation.
        assert!(
            difference.max_abs() < 1e-5,
            "{shader:?} follows the CPU: {difference}"
        );
    }
}

#[test]
fn both_shaders_match_the_cpu_on_random_brushes() {
    GpuTest::run(|gpu| {
        for shader in Shader::ALL {
            let difference = compare(gpu, shader);
            println!("{shader:?} on {}: {difference}", gpu.adapter_info().name);
            assert_eq!(
                difference.compared(),
                POSITIONS * BRUSHES,
                "every vertex of every brush"
            );
            assert_matches_cpu(gpu, shader, &difference);
        }
    });
}

#[test]
fn the_two_shaders_agree_with_each_other_bit_for_bit() {
    GpuTest::run(|gpu| {
        let mut rng = Rng(0xD1FF);
        let points = positions(&mut rng, 10_000);
        let brush = random_brush(&mut rng);
        let [wgsl, rust_gpu] = Shader::ALL.map(|shader| {
            GpuDraw::new(gpu, shader)
                .expect("the shader builds")
                .draw(gpu, &brush, &points)
                .expect("the draw")
        });
        let mut difference = Difference::default();
        difference.add(&rust_gpu, &wgsl, &brush);
        assert_eq!(difference.differing(), 0, "{difference}");
    });
}

#[test]
fn full_strength_at_the_centre_and_nothing_from_the_rim_on() {
    GpuTest::run(|gpu| {
        for shader in Shader::ALL {
            let draw = GpuDraw::new(gpu, shader).expect("the shader builds");
            let brush = DrawBrush::new([0.0; 3], 3.0, 0.25).expect("a valid radius");
            let points = [
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, -3.0, 0.0],
                [0.0, 0.0, 3.0],
                [3.5, 0.0, 0.0],
                [2.999, 0.0, 0.0],
            ];
            let deltas = draw.draw(gpu, &brush, &points).expect("the draw");
            assert_eq!(
                deltas.first().map(|d| d.to_bits()),
                Some(0.25_f32.to_bits()),
                "{shader:?}: weight(0) = 1, so the full strength at the centre: {deltas:?}"
            );
            assert_eq!(
                deltas.get(1..5),
                Some([0.0; 4].as_slice()),
                "{shader:?}: weight(1) = 0, so nothing on the rim or past it: {deltas:?}"
            );
            assert!(
                deltas.get(5).is_some_and(|&d| d > 0.0 && d < 1e-6),
                "{shader:?}: just inside the rim, a little: {deltas:?}"
            );
        }
    });
}

#[test]
fn differences_are_counted_in_ulps_where_both_sides_are_inside() {
    let mut difference = Difference::default();
    let one_up = f32::from_bits(1.0_f32.to_bits() + 1);
    let tiny = f32::from_bits(1);
    let brush = DrawBrush::new([0.0; 3], 1.0, -2.0).expect("a valid radius");
    difference.add(
        &[1.0, -0.0, tiny, 2.0, 0.0],
        &[one_up, 0.0, -tiny, 2.0, 1e-9],
        &brush,
    );
    assert_eq!(difference.compared(), 5, "five vertices");
    assert_eq!(
        difference.max_ulps(),
        2,
        "the smallest subnormals are two ulps apart across zero; the vertex inside on one side \
         only doesn't count"
    );
    assert_eq!(
        difference.max_abs(),
        one_up - 1.0,
        "the largest absolute difference, the one-sided vertex included"
    );
    assert!(
        (difference.max_strength_ulps() - 0.5).abs() < 1e-6,
        "one ulp of 1 is half an ulp of a strength of 2: {difference}"
    );
    assert_eq!(
        difference.to_string(),
        format!(
            "5 vertices: max 2 ulps where both are inside; max |Δ| {:e}, 0.50 ulps of the \
             strength; 3 differ, 1 inside on one side only",
            one_up - 1.0
        ),
        "-0 and 0 are the same"
    );
}

#[test]
fn a_footprint_takes_new_positions_between_dispatches() {
    GpuTest::run(|gpu| {
        let draw = GpuDraw::new(gpu, Shader::Wgsl).expect("the shader builds");
        let mut rng = Rng(7);
        let first = positions(&mut rng, 1000);
        let second = positions(&mut rng, 1000);
        let brush = DrawBrush::new([0.1, 0.2, 0.3], 0.8, 0.5).expect("a valid radius");
        let footprint = draw.footprint(gpu, &first).expect("1000 vertices fit");
        footprint.upload(gpu, &second).expect("the upload");
        footprint.dispatch(gpu, &brush).expect("the dispatch");
        let deltas = footprint.read_deltas(gpu).expect("the readback");
        let expected = brush.cpu(&second);
        let mut difference = Difference::default();
        difference.add(&deltas, &expected, &brush);
        assert_matches_cpu(gpu, Shader::Wgsl, &difference);
        assert_ne!(
            brush.cpu(&first),
            expected,
            "and the two sets differ, so the test could tell"
        );
    });
}

#[test]
fn an_empty_footprint_draws_nothing() {
    GpuTest::run(|gpu| {
        let draw = GpuDraw::new(gpu, Shader::Wgsl).expect("the shader builds");
        let brush = DrawBrush::new([0.0; 3], 1.0, 1.0).expect("a valid radius");
        assert_eq!(
            draw.draw(gpu, &brush, &[]).expect("the draw"),
            Vec::<f32>::new(),
            "no vertices, no deltas"
        );
    });
}

#[test]
fn a_dispatch_is_timed_where_the_device_has_timestamps() {
    GpuTest::run(|gpu| {
        let draw = GpuDraw::new(gpu, Shader::Wgsl).expect("the shader builds");
        let mut rng = Rng(11);
        let points = positions(&mut rng, POSITIONS);
        let brush = random_brush(&mut rng);
        let footprint = draw.footprint(gpu, &points).expect("50k vertices fit");
        let timed = footprint.time_dispatch(gpu, &brush).expect("the timing");
        assert_eq!(
            timed.is_some(),
            gpu.timestamp_period().is_some(),
            "timed exactly when the device has timestamps: {timed:?}"
        );
        let deltas = footprint.read_deltas(gpu).expect("the readback");
        let mut difference = Difference::default();
        difference.add(&deltas, &brush.cpu(&points), &brush);
        assert_matches_cpu(gpu, Shader::Wgsl, &difference);
    });
}

#[test]
fn more_vertices_than_one_dispatch_covers_is_an_error() {
    GpuTest::run(|gpu| {
        let draw = GpuDraw::new(gpu, Shader::Wgsl).expect("the shader builds");
        let limits = gpu.device().limits();
        let max = (limits.max_compute_workgroups_per_dimension as usize * WORKGROUP as usize).min(
            usize::try_from(limits.max_storage_buffer_binding_size).expect("fits")
                / size_of::<[f32; 3]>(),
        );
        let points = vec![[0.0; 3]; max + 1];
        assert!(
            matches!(
                draw.footprint(gpu, &points),
                Err(DrawError::TooManyVertices { count, .. }) if count == max + 1
            ),
            "one too many"
        );
    });
}
