//! Spike 2b: what moving one level step of tile evaluation to the GPU buys, measured.
//!
//! `cargo bench -p render --features spike --bench eval_gpu` (`BRUSHCRAFT_ADAPTER` picks the
//! adapter by name, else the default one). Not a divan bench: the spike wants medians and p99s
//! of whole steps and a fit over tile counts, printed as the report's tables.
//!
//! Three meshes at level 7 (every face 2 × 2 tiles of 64 quads), random corrections at every
//! level: a cube sphere split once (24 tiles) and twice (96), and a subdivided pentagonal prism
//! (120). For each, after checking the GPU against `geom` and for seam self-consistency:
//!
//! - **CPU path**: `geom` evaluates every tile (positions and normals, in parallel on every core),
//!   then the evaluated positions and normals are uploaded;
//! - **GPU path**: the decoded corrections are uploaded, then the GPU evaluates every tile from
//!   resident parent blocks (and, as a cold variant, with the parent blocks uploaded too).
//!
//! Each is timed over [`STEPS`] steps (wall clock to the GPU's completion; the GPU path's pass
//! also by timestamp queries), and a line through the three sizes projects the M0 brush step's
//! 47 tiles and the 1M-vertex brush's 1,165. Finally it counts what a cold finest tile of the
//! benchmark project needs resident: its ancestry through `geom::parent_tiles`, the walk
//! `store`'s dependency graph records.

use std::collections::BTreeSet;
use std::io::{self, Write as _};
use std::time::{Duration, Instant};

use geom::fixtures::subdivided_pentagonal_prism;
use geom::{
    BaseMesh, CorrectionMap, DenseEvaluation, Level, Region, TileKey, Vec3, cube_sphere,
    evaluate_region, normals_into, parent_tiles,
};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use render::gpu::{AdapterRequest, Gpu};
use render::spike::eval::{Evaluated, GpuEval, Inputs, Run, VecDifference, seam_consistency};
use wgpu::{BufferDescriptor, BufferUsages, PollType};

/// Steps timed per path and mesh.
const STEPS: usize = 200;

/// The level evaluated.
const LEVEL: u8 = 7;

/// Vertices per tile.
const VERTICES: usize = 65 * 65;

/// The M0 brush step's tiles updated on the full project.
const BRUSH_STEP_TILES: f64 = 47.0;

/// The 1M-vertex brush's tiles.
const BIG_BRUSH_TILES: f64 = 1165.0;

/// The benchmark project's base mesh: `cube_sphere(20)`, 2,400 faces (`brushcraft-bench`'s
/// generator), sculpted to level 9.
const BENCHMARK_SPLITS: u32 = 20;

/// The benchmark project's finest level.
const BENCHMARK_FINEST: u8 = 9;

/// The benchmark project file, for its size per tile.
const BENCHMARK_FILE: &str = ".cache/brushcraft/bench/sphere.bcrf";

/// What the bench reports through, line by line.
type Out<'a> = io::StdoutLock<'a>;

/// The bench's result: any error ends it.
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() -> Result<()> {
    let request =
        std::env::var("BRUSHCRAFT_ADAPTER").map_or(AdapterRequest::Default, AdapterRequest::Named);
    let gpu = Gpu::open(&request)?;
    let mut out = io::stdout().lock();
    let info = gpu.adapter_info();
    writeln!(
        out,
        "adapter: {} ({:?}, {:?}), driver {} {}; CPU threads: {}",
        info.name,
        info.backend,
        info.device_type,
        info.driver,
        info.driver_info,
        rayon::current_num_threads()
    )?;
    let mut fits = Vec::new();
    for (name, mesh) in [
        ("cube sphere x1", cube_sphere(1)?),
        ("cube sphere x2", cube_sphere(2)?),
        ("pentagonal prism", subdivided_pentagonal_prism()?),
    ] {
        fits.push(measure(&gpu, &mut out, name, &mesh)?);
    }
    project(&mut out, &fits)?;
    residency(&mut out)?;
    Ok(())
}

/// A splitmix64 step: the next random bits of `state`.
const fn next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A float in `[-scale, scale]` from `state`.
fn signed(state: &mut u64, scale: f32) -> f32 {
    let unit = f32::from(u16::try_from(next(state) >> 48).unwrap_or(0)) / f32::from(u16::MAX);
    scale * (2.0 * unit - 1.0)
}

/// Random corrections of up to `0.002` on every vertex of every level up to [`LEVEL`].
///
/// # Errors
///
/// None in practice: every correction is valid.
fn corrections(mesh: &BaseMesh) -> Result<CorrectionMap> {
    let mut map = CorrectionMap::new();
    let mut state = 0x5EED;
    for l in 1..=LEVEL {
        let level = Level::new(l)?;
        let n = level.face_quads();
        for face in mesh.faces() {
            for j in 0..=n {
                for i in 0..=n {
                    let c = Vec3::new(
                        signed(&mut state, 0.002),
                        signed(&mut state, 0.002),
                        signed(&mut state, 0.002),
                    );
                    map.set(mesh.topology(), face, level, i, j, c)?;
                }
            }
        }
    }
    Ok(map)
}

/// Every tile of `level` of `mesh`.
///
/// # Errors
///
/// None in practice: the tiles are the level's.
fn tiles(mesh: &BaseMesh, level: Level) -> Result<Vec<TileKey>> {
    let per_side = u16::try_from(level.tiles_per_side())?;
    let mut keys = Vec::new();
    for face in mesh.faces() {
        for ty in 0..per_side {
            for tx in 0..per_side {
                keys.push(TileKey::displacement(face, level, tx, ty)?);
            }
        }
    }
    Ok(keys)
}

/// The median and 99th percentile of `times`.
fn stats(mut times: Vec<Duration>) -> (Duration, Duration) {
    times.sort_unstable();
    let at = |q: f64| {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss,
            reason = "an index into a few hundred samples"
        )]
        let k = ((times.len().saturating_sub(1)) as f64 * q).round() as usize;
        times.get(k).copied().unwrap_or_default()
    };
    (at(0.5), at(0.99))
}

/// `time` in milliseconds, to three decimals.
fn ms(time: Duration) -> String {
    format!("{:.3} ms", time.as_secs_f64() * 1e3)
}

/// One mesh's medians: tiles, CPU path, GPU path (warm), GPU pass alone.
#[derive(Debug, Clone, Copy)]
struct Fit {
    /// Tiles per step.
    tiles: f64,
    /// The CPU path's median, in milliseconds.
    cpu: f64,
    /// The GPU path's median (corrections upload and evaluation), in milliseconds.
    gpu: f64,
    /// The GPU pass alone (timestamps), in milliseconds; NaN without timestamps.
    pass: f64,
}

/// One mesh with its corrections, evaluated by `geom`, and its tiles at [`LEVEL`].
struct Scene<'a> {
    /// The base mesh.
    mesh: &'a BaseMesh,
    /// The corrections at every level.
    map: CorrectionMap,
    /// Every level, evaluated by `geom`.
    dense: DenseEvaluation,
    /// Every tile of [`LEVEL`].
    keys: Vec<TileKey>,
}

/// A tile's positions and normals, as the CPU path computes them.
type Evaluation = (Vec<Vec3>, Vec<Vec3>);

impl<'a> Scene<'a> {
    /// `mesh` with random corrections, evaluated.
    ///
    /// # Errors
    ///
    /// If `geom` fails, which it doesn't for these meshes.
    fn new(mesh: &'a BaseMesh) -> Result<Self> {
        let level = Level::new(LEVEL)?;
        let map = corrections(mesh)?;
        let dense = DenseEvaluation::new(mesh, level, &map)?;
        let keys = tiles(mesh, level)?;
        Ok(Self {
            mesh,
            map,
            dense,
            keys,
        })
    }

    /// The correction at tile-local vertex `(i, j)` of `tile`.
    fn correction(&self, tile: &TileKey, i: u32, j: u32) -> Vec3 {
        let region = Region::of_tile(tile);
        self.map
            .get(tile.face(), tile.level(), region.x0() + i, region.y0() + j)
    }

    /// The GPU's inputs for every tile.
    ///
    /// # Errors
    ///
    /// As [`Inputs::gather`].
    fn inputs(&self) -> Result<Inputs> {
        Ok(Inputs::gather(
            self.mesh,
            &self.dense,
            &self.keys,
            |tile, i, j| self.correction(tile, i, j),
        )?)
    }

    /// The CPU path's evaluation: every tile's positions and normals, in parallel.
    ///
    /// # Errors
    ///
    /// As `geom::evaluate_region`.
    fn evaluate(&self) -> Result<Evaluated> {
        let tiles: Vec<Evaluation> = self
            .keys
            .par_iter()
            .map(|tile| {
                let region = Region::of_tile(tile);
                let grid = evaluate_region(self.mesh, &self.dense, region, |i, j| {
                    self.correction(tile, i, j)
                })?;
                let mut normals = vec![Vec3::ZERO; VERTICES];
                normals_into(self.mesh, &self.dense, region, &mut normals)?;
                Ok((grid.as_slice().to_vec(), normals))
            })
            .collect::<std::result::Result<_, geom::EvalError>>()?;
        let flat = |pick: fn(&Evaluation) -> &[Vec3]| {
            tiles
                .iter()
                .flat_map(|t| pick(t).iter().map(|v| [v.x, v.y, v.z]))
                .collect()
        };
        Ok(Evaluated {
            positions: flat(|t| &t.0),
            normals: flat(|t| &t.1),
        })
    }
}

/// Checks and times one mesh; prints its rows.
///
/// # Errors
///
/// If `geom` or the GPU fails, or stdout can't be written.
fn measure(gpu: &Gpu, out: &mut Out<'_>, name: &str, mesh: &BaseMesh) -> Result<Fit> {
    let scene = Scene::new(mesh)?;
    let inputs = scene.inputs()?;
    let run = GpuEval::new(gpu)?.prepare(gpu, &inputs)?;
    run.dispatch(gpu)?;
    let got = run.read(gpu)?;
    let expected = scene.evaluate()?;
    let mut positions = VecDifference::default();
    let mut normals = VecDifference::default();
    positions.add(&got.positions, &expected.positions);
    normals.add(&got.normals, &expected.normals);
    let consistency = seam_consistency(mesh, inputs.tiles(), &got);
    writeln!(out, "\n{name}: {} tiles at level {LEVEL}", scene.keys.len())?;
    writeln!(
        out,
        "  GPU against geom: positions {positions}; normals {normals}"
    )?;
    writeln!(out, "  seams: {consistency}")?;
    let bytes = inputs.bytes();
    writeln!(
        out,
        "  inputs: parent blocks and rings {} B, decoded corrections {} B, structure {} B",
        bytes.parents, bytes.corrections, bytes.structure
    )?;
    let [cpu_eval, cpu_upload, cpu_step] = time_cpu(gpu, &scene)?;
    let [gpu_step, gpu_pass, gpu_cold] = time_gpu(gpu, &run, &inputs)?;
    let rows = [
        ("CPU path: evaluate (parallel)", cpu_eval),
        ("CPU path: upload evaluated", cpu_upload),
        ("CPU path: step", cpu_step),
        ("GPU path: step (upload corrections, evaluate)", gpu_step),
        ("GPU path: the pass alone (timestamps)", gpu_pass),
        ("GPU path: step with parent blocks uploaded", gpu_cold),
    ];
    let mut medians = [f64::NAN; 6];
    for ((label, times), median_ms) in rows.into_iter().zip(&mut medians) {
        if times.is_empty() {
            writeln!(out, "  {label}: no timestamps")?;
            continue;
        }
        let count = times.len();
        let (median, p99) = stats(times);
        *median_ms = median.as_secs_f64() * 1e3;
        writeln!(
            out,
            "  {label}: median {}, p99 {} ({count} steps)",
            ms(median),
            ms(p99)
        )?;
    }
    #[expect(clippy::cast_precision_loss, reason = "a few hundred tiles")]
    let tiles = scene.keys.len() as f64;
    let [_, _, cpu, gpu_median, pass, _] = medians;
    Ok(Fit {
        tiles,
        cpu,
        gpu: gpu_median,
        pass,
    })
}

/// [`STEPS`] steps of the CPU path: evaluate every tile, then upload the positions and normals
/// and wait for the GPU to have them. Returns the evaluation, the upload and the step times.
///
/// # Errors
///
/// If `geom` or the GPU fails.
fn time_cpu(gpu: &Gpu, scene: &Scene<'_>) -> Result<[Vec<Duration>; 3]> {
    let half = u64::try_from(scene.keys.len() * VERTICES * 12)?;
    let target = gpu.device().create_buffer(&BufferDescriptor {
        label: Some("evaluated upload"),
        size: 2 * half,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut times = [
        Vec::with_capacity(STEPS),
        Vec::with_capacity(STEPS),
        Vec::with_capacity(STEPS),
    ];
    for _ in 0..STEPS {
        let start = Instant::now();
        let evaluated = scene.evaluate()?;
        let evaluated_at = Instant::now();
        gpu.queue()
            .write_buffer(&target, 0, bytemuck::cast_slice(&evaluated.positions));
        gpu.queue()
            .write_buffer(&target, half, bytemuck::cast_slice(&evaluated.normals));
        gpu.queue().submit([]);
        gpu.device().poll(PollType::wait_indefinitely())?;
        let done = Instant::now();
        let [eval, upload, step] = &mut times;
        eval.push(evaluated_at - start);
        upload.push(done - evaluated_at);
        step.push(done - start);
    }
    Ok(times)
}

/// [`STEPS`] steps of the GPU path, warm and cold.
///
/// Warm: upload the corrections and evaluate, timing the pass with timestamps too. Cold: the
/// parent blocks uploaded as well. Returns the step, the pass and the cold step times.
///
/// # Errors
///
/// If the GPU fails.
fn time_gpu(gpu: &Gpu, run: &Run, inputs: &Inputs) -> Result<[Vec<Duration>; 3]> {
    let mut times = [
        Vec::with_capacity(STEPS),
        Vec::with_capacity(STEPS),
        Vec::with_capacity(STEPS),
    ];
    let [step, pass, cold] = &mut times;
    for _ in 0..STEPS {
        let warm = Instant::now();
        run.upload_corrections(gpu, inputs)?;
        if let Some(time) = run.time(gpu)? {
            pass.push(time);
        }
        step.push(warm.elapsed());
        let parents = Instant::now();
        run.upload_parents(gpu, inputs)?;
        run.dispatch(gpu)?;
        gpu.device().poll(PollType::wait_indefinitely())?;
        cold.push(parents.elapsed());
    }
    Ok(times)
}

/// Picks one path's median from a [`Fit`].
type Pick = fn(&Fit) -> f64;

/// A least-squares line `a + b·x` through `points`.
fn line(points: &[(f64, f64)]) -> (f64, f64) {
    #[expect(clippy::cast_precision_loss, reason = "three points")]
    let n = points.len() as f64;
    let (sx, sy) = points
        .iter()
        .fold((0.0, 0.0), |(x, y), &(px, py)| (x + px, y + py));
    let (mx, my) = (sx / n, sy / n);
    let (sxy, sxx) = points.iter().fold((0.0, 0.0), |(xy, xx), &(px, py)| {
        (xy + (px - mx) * (py - my), xx + (px - mx) * (px - mx))
    });
    let b = if sxx > 0.0 { sxy / sxx } else { 0.0 };
    (my - b * mx, b)
}

/// Prints the fixed and per-tile cost of each path, and its projection to the M0 brush step's
/// tiles and the 1M-vertex brush's.
///
/// # Errors
///
/// If stdout can't be written.
fn project(out: &mut Out<'_>, fits: &[Fit]) -> Result<()> {
    writeln!(
        out,
        "\nper-tile costs (least squares over the three meshes' medians):"
    )?;
    let paths: [(&str, Pick); 3] = [
        ("CPU path step", |f| f.cpu),
        ("GPU path step", |f| f.gpu),
        ("GPU pass alone", |f| f.pass),
    ];
    for (label, pick) in paths {
        let points: Vec<(f64, f64)> = fits.iter().map(|f| (f.tiles, pick(f))).collect();
        let (a, b) = line(&points);
        writeln!(
            out,
            "  {label}: {a:.3} ms + {:.1} µs per tile; {BRUSH_STEP_TILES} tiles {:.3} ms; \
             {BIG_BRUSH_TILES} tiles {:.1} ms",
            b * 1e3,
            a + b * BRUSH_STEP_TILES,
            a + b * BIG_BRUSH_TILES
        )?;
    }
    Ok(())
}

/// Prints what a cold finest tile of the benchmark project needs resident on the GPU.
///
/// That is its ancestry through `geom::parent_tiles`, level by level; and the file's bytes per
/// tile, what compressed corrections cost to upload.
///
/// # Errors
///
/// If stdout can't be written.
fn residency(out: &mut Out<'_>) -> Result<()> {
    let mesh = cube_sphere(BENCHMARK_SPLITS)?;
    let finest = Level::new(BENCHMARK_FINEST)?;
    let face = mesh.faces().next().ok_or("no face")?;
    let per_side = u16::try_from(finest.tiles_per_side())?;
    writeln!(
        out,
        "\ncold finest tile of the benchmark mesh (cube_sphere({BENCHMARK_SPLITS}), level {BENCHMARK_FINEST}):"
    )?;
    for (label, tx, ty) in [
        ("inside a face", per_side / 2, per_side / 2),
        ("at a valence-3 corner", 0, 0),
    ] {
        let mut frontier: BTreeSet<TileKey> =
            BTreeSet::from([TileKey::displacement(face, finest, tx, ty)?]);
        let mut levels = Vec::new();
        let mut bytes = 0_usize;
        while let Some(level) = frontier.first().map(TileKey::level).and_then(Level::parent) {
            if level == Level::ZERO {
                break;
            }
            let mut parents = BTreeSet::new();
            for tile in &frontier {
                parents.extend(parent_tiles(&mesh, Region::of_tile(tile))?);
            }
            let side = level.tile_quads() as usize + 1;
            bytes += parents.len() * side * side * 12;
            levels.push(format!("L{}: {}", level.get(), parents.len()));
            frontier = parents;
        }
        writeln!(
            out,
            "  {label}: ancestor tiles {}; their evaluated positions {} KB",
            levels.join(", "),
            bytes / 1000
        )?;
    }
    let home = std::env::var("HOME").unwrap_or_default();
    match std::fs::metadata(std::path::Path::new(&home).join(BENCHMARK_FILE)) {
        Ok(metadata) => {
            // Every level's tiles, 1 to 9: 2,400 faces × 4^(L − 6) tiles at L ≥ 6, one per
            // face below.
            let tiles: u64 = (1..=u32::from(BENCHMARK_FINEST))
                .map(|l| 2_400_u64 * if l >= 6 { 4_u64.pow(l - 6) } else { 1 })
                .sum();
            writeln!(
                out,
                "  benchmark project: {} B over {tiles} tiles: {:.1} KB per tile on disk \
                 (compressed corrections and their share of index and history)",
                metadata.len(),
                f64::from(u32::try_from(metadata.len() / tiles)?) / 1000.0
            )?;
        }
        Err(error) => writeln!(out, "  benchmark project not found: {error}")?,
    }
    Ok(())
}
