//! The draw brush on the GPU against the CPU, over the M0 brush step's 50,000 positions
//! (`brush`'s own `draw` bench uses the same footprint).
//!
//! `cargo bench -p render --features spike --bench brush_gpu`. Wall-clock benches, by divan:
//! the CPU's `brush::draw`; the GPU's whole step (upload the positions, dispatch, read the deltas
//! back); and the dispatch alone with a wait for it. Before them, the GPU time of the dispatch
//! itself, from timestamp queries, is printed as a median of [`TIMED_RUNS`] runs per shader.
//!
//! The GPU is the default adapter, or the first whose name contains `BRUSHCRAFT_ADAPTER`.

use std::io::{self, Write as _};
use std::sync::OnceLock;
use std::time::Duration;

use render::gpu::{AdapterRequest, Gpu};
use render::spike::brush::{DrawBrush, DrawError, GpuDraw, Shader, against_cpu};
use wgpu::PollType;

/// Columns of the benchmark's grid of positions.
const COLUMNS: u16 = 250;

/// Rows of the benchmark's grid of positions: `COLUMNS × ROWS` = 50,000.
const ROWS: u16 = 200;

/// Dispatches timed with timestamp queries per shader; the median is reported.
const TIMED_RUNS: usize = 51;

/// The GPU every bench runs on, opened once by `main`.
static GPU: OnceLock<Gpu> = OnceLock::new();

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let request =
        std::env::var("BRUSHCRAFT_ADAPTER").map_or(AdapterRequest::Default, AdapterRequest::Named);
    let opened = Gpu::open(&request)?;
    let gpu = GPU.get_or_init(move || opened);
    report_dispatch_times(gpu)?;
    divan::main();
    Ok(())
}

/// Prints the adapter, then each shader's dispatch time on the GPU: the median, minimum and
/// maximum of [`TIMED_RUNS`] timestamped dispatches, after one untimed warm-up.
///
/// # Errors
///
/// If a shader doesn't build, the GPU fails, or stdout can't be written.
fn report_dispatch_times(gpu: &Gpu) -> Result<(), Box<dyn std::error::Error>> {
    let mut out = io::stdout().lock();
    let info = gpu.adapter_info();
    writeln!(
        out,
        "adapter: {} ({:?}, {:?}), driver {} {}",
        info.name, info.backend, info.device_type, info.driver, info.driver_info
    )?;
    let positions = footprint();
    let brush = brush()?;
    let brushes = comparison_brushes()?;
    for shader in Shader::ALL {
        let draw = match GpuDraw::new(gpu, shader) {
            Ok(draw) => draw,
            Err(error) => {
                writeln!(out, "{shader:?}: doesn't build here: {error}")?;
                continue;
            }
        };
        let difference = against_cpu(gpu, &draw, &positions, &brushes)?;
        writeln!(
            out,
            "{shader:?} against the CPU, {} brushes: {difference}",
            brushes.len()
        )?;
        let footprint = draw.footprint(gpu, &positions)?;
        footprint.time_dispatch(gpu, &brush)?;
        let mut times = Vec::with_capacity(TIMED_RUNS);
        for _ in 0..TIMED_RUNS {
            match footprint.time_dispatch(gpu, &brush)? {
                Some(time) => times.push(time),
                None => break,
            }
        }
        times.sort_unstable();
        match (times.first(), times.get(times.len() / 2), times.last()) {
            (Some(min), Some(median), Some(max)) => writeln!(
                out,
                "{shader:?} dispatch on the GPU, 50k vertices, {} runs: median {}, min {}, max {}",
                times.len(),
                micros(*median),
                micros(*min),
                micros(*max)
            )?,
            _ => writeln!(out, "{shader:?}: the device has no timestamp queries")?,
        }
    }
    Ok(())
}

/// Brushes over the grid for the comparison with the CPU: centres across it, radii from 0.1 to
/// 1.0, strengths of both signs and several magnitudes.
///
/// # Errors
///
/// None in practice: every radius is valid.
fn comparison_brushes() -> Result<Vec<DrawBrush>, DrawError> {
    (1_u8..=10)
        .map(|k| {
            let f = f32::from(k) / 10.0;
            let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
            DrawBrush::new([f, 1.0 - f, 0.03 * f], f, sign * 0.37 * f)
        })
        .collect()
}

/// `time` in microseconds, to two decimals.
fn micros(time: Duration) -> String {
    format!("{:.2} µs", time.as_secs_f64() * 1e6)
}

/// A `COLUMNS × ROWS` grid of positions in the unit square of the plane `z = 0`, with a gentle
/// bump in `z`: the footprint `brush`'s own bench draws on.
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

/// The brush `brush`'s own bench draws with: radius 0.5, centred on the grid, covering about
/// three quarters of it.
///
/// # Errors
///
/// None in practice: the radius is valid.
fn brush() -> Result<DrawBrush, DrawError> {
    DrawBrush::new([0.5, 0.5, 0.0], 0.5, 0.01)
}

/// `brush::draw` on the CPU, one thread: the reference the GPU replaces.
#[divan::bench]
fn cpu_draw(bencher: divan::Bencher<'_, '_>) {
    let positions = footprint();
    let Ok(brush) = brush() else { return };
    bencher
        .counter(divan::counter::ItemsCount::new(positions.len()))
        .bench_local(|| brush.cpu(divan::black_box(&positions)));
}

/// The whole GPU step as a brush would pay it each dab: write the positions into the resident
/// buffer, dispatch, wait, and read the deltas back.
#[divan::bench(args = Shader::ALL)]
fn gpu_upload_dispatch_readback(bencher: divan::Bencher<'_, '_>, shader: Shader) {
    let Some(gpu) = GPU.get() else { return };
    let positions = footprint();
    let (Ok(brush), Ok(draw)) = (brush(), GpuDraw::new(gpu, shader)) else {
        return;
    };
    let Ok(footprint) = draw.footprint(gpu, &positions) else {
        return;
    };
    bencher
        .counter(divan::counter::ItemsCount::new(positions.len()))
        .bench_local(|| {
            footprint
                .upload(gpu, &positions)
                .and_then(|()| footprint.dispatch(gpu, &brush))
                .and_then(|()| footprint.read_deltas(gpu))
        });
}

/// The dispatch alone, positions resident, measured from the host: submit and wait.
#[divan::bench(args = Shader::ALL)]
fn gpu_dispatch_and_wait(bencher: divan::Bencher<'_, '_>, shader: Shader) {
    let Some(gpu) = GPU.get() else { return };
    let positions = footprint();
    let (Ok(brush), Ok(draw)) = (brush(), GpuDraw::new(gpu, shader)) else {
        return;
    };
    let Ok(footprint) = draw.footprint(gpu, &positions) else {
        return;
    };
    bencher
        .counter(divan::counter::ItemsCount::new(positions.len()))
        .bench_local(|| {
            footprint
                .dispatch(gpu, &brush)
                .map(|()| gpu.device().poll(PollType::wait_indefinitely()))
        });
}
