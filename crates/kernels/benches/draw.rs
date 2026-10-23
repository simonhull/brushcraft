//! The draw kernel over one brush footprint: 50,000 positions, the M0 benchmark's 50k-vertex
//! brush step. The kernel is the innermost part of that step's "≤ 2 ms p99" target.

use kernels::{Smoothstep, draw};

fn main() {
    divan::main();
}

/// Columns of the benchmark's grid of positions.
const COLUMNS: u16 = 250;

/// Rows of the benchmark's grid of positions: `COLUMNS × ROWS` = 50,000.
const ROWS: u16 = 200;

/// A `COLUMNS × ROWS` grid of positions in the unit square of the plane `z = 0`, with a gentle
/// bump in `z` so distances aren't planar.
fn footprint() -> Vec<[f32; 3]> {
    (0..ROWS)
        .flat_map(|j| {
            (0..COLUMNS).map(move |i| {
                let x = f32::from(i) / f32::from(COLUMNS);
                let y = f32::from(j) / f32::from(ROWS);
                [x, y, 0.05 * (x * y)]
            })
        })
        .collect()
}

/// `draw` with a smoothstep falloff over 50,000 positions, the brush (radius 0.5, centred on
/// the grid) covering about three quarters of them.
#[divan::bench]
fn draw_50k(bencher: divan::Bencher<'_, '_>) {
    let positions = footprint();
    let mut out = vec![0.0_f32; positions.len()];
    bencher
        .counter(divan::counter::ItemsCount::new(positions.len()))
        .bench_local(|| {
            draw(
                &Smoothstep,
                divan::black_box([0.5, 0.5, 0.0]),
                divan::black_box(0.5),
                divan::black_box(0.01),
                divan::black_box(&positions),
                &mut out,
            )
        });
}
