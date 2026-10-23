//! `generate`: builds the benchmark sphere, sculpts every level with fBm noise, and saves it.
//!
//! The base mesh is `cube_sphere(splits)` (2,400 faces for the benchmark). Every level from 1
//! to `levels` gets one octave of value noise along the normal: level `L`'s corrections push
//! each vertex by `amplitude · 2^−(L−1) · noise(p · f_L)`, with `p` the vertex's predicted
//! position (the levels below already sculpted) and `f_L = splits · 2^L / 2π`, which puts a
//! feature about four quads of level `L` across. Each level halves the amplitude and doubles
//! the frequency: fractal Brownian motion, one octave per level, as detail is sculpted.
//!
//! Everything goes through [`Project`], as the app's edits do: a stroke per level, each tile
//! evaluated, then edited with [`store::TileStore::edit_tile`], each vertex written once
//! through the tile that owns it (which writes its copies in the neighbouring tiles too), and the
//! project ticked every [`TICK_EVERY`] tiles, as a frame loop would, so the recovery writer
//! streams the edits as they are made. Then [`Project::save_as`] writes the project file and
//! the recovery file is deleted.

use std::io::Write;
use std::time::Instant;

use anyhow::{Context as _, Result, bail};
use geom::{Level, TileKey, Vec3, cube_sphere};
use store::Project;

use crate::cli::Options;
use crate::noise::ValueNoise;
use crate::paths::Paths;
use crate::report::{Output, Record, mib, ms};
use crate::session;
use crate::surface::owns;
use crate::sys::Rss;

/// How often, in tiles, the generator ticks the project ([`Project::tick`]).
///
/// A tick, as the app's frame loop makes one, advances the recovery flush in flight and starts
/// a new one once a second has passed, so a level's edits stream into the recovery file while
/// it is sculpted, rather than all waiting for the flush at its stroke's end.
const TICK_EVERY: usize = 64;

/// Every tile of `level` on a mesh of `faces` faces, face by face, row by row.
///
/// # Errors
///
/// Never for a valid level: the tile coordinates are in range.
fn tiles_of(faces: u32, level: Level) -> Result<Vec<TileKey>> {
    let per_side = u16::try_from(level.tiles_per_side())?;
    let mut keys = Vec::new();
    for face in 0..faces {
        for ty in 0..per_side {
            for tx in 0..per_side {
                keys.push(TileKey::displacement(
                    geom::FaceId::new(face),
                    level,
                    tx,
                    ty,
                )?);
            }
        }
    }
    Ok(keys)
}

/// One level's octave: how far it pushes, and how fine its features are.
#[derive(Debug, Clone, Copy)]
struct Octave {
    /// The level.
    level: Level,
    /// The push at the noise's extremes: `amplitude · 2^−(L−1)`.
    amplitude: f32,
    /// The noise's frequency: `splits · 2^L / 2π`.
    frequency: f32,
}

impl Octave {
    /// Level `level`'s octave for `options`.
    fn new(options: &Options, level: Level) -> Self {
        let l = i32::from(level.get());
        #[expect(
            clippy::cast_precision_loss,
            reason = "splits are at most tens of thousands, exact in an f32"
        )]
        let splits = options.splits as f32;
        Self {
            level,
            amplitude: options.amplitude * 0.5_f32.powi(l - 1),
            frequency: splits * 2.0_f32.powi(l) / std::f32::consts::TAU,
        }
    }

    /// The push along the normal at predicted position `p`.
    fn push(&self, noise: ValueNoise, p: Vec3) -> f32 {
        self.amplitude * noise.at(p * self.frequency, u32::from(self.level.get()))
    }
}

/// Runs `generate` (see the module notes).
///
/// # Errors
///
/// If the project file exists (without `--force`), or making, sculpting or saving the project
/// fails, or writing the report does.
pub(crate) fn run<J: Write, H: Write>(
    options: &Options,
    paths: &Paths,
    out: &mut Output<J, H>,
) -> Result<()> {
    let start = Instant::now();
    if paths.project.exists() {
        if !options.force {
            bail!(
                "{} exists: pass --force to replace it",
                paths.project.display()
            );
        }
        std::fs::remove_file(&paths.project)
            .with_context(|| format!("removing {}", paths.project.display()))?;
    }
    let mesh = cube_sphere(options.splits)?;
    let faces = mesh.face_count();
    out.say(&format!(
        "generating cube_sphere({}) ({faces} faces), levels 1 to {}, into {}",
        options.splits,
        options.levels,
        paths.project.display()
    ))?;
    let mut project =
        Project::new(mesh, session::config(options, paths)).context("making the project")?;
    let levels = (1..=options.levels)
        .map(Level::new)
        .collect::<Result<Vec<_>, _>>()?;
    let total: usize = levels
        .iter()
        .map(|l| faces as usize * (l.tiles_per_side() as usize).pow(2))
        .sum();
    let noise = ValueNoise::new(options.seed);
    let (mut done, mut vertices) = (0, 0_u64);
    let mut scratch = Scratch::default();
    for &level in &levels {
        let octave = Octave::new(options, level);
        project.begin_stroke()?;
        for key in tiles_of(faces, level)? {
            sculpt_tile(&mut project, noise, octave, key, &mut scratch)?;
            vertices += u64::try_from(scratch.writes.len())?;
            done += 1;
            if done % TICK_EVERY == 0 {
                project.tick(Instant::now())?;
            }
            out.progress(&format!("level {level}"), done, total, start)?;
        }
        project.end_stroke()?;
    }
    let sculpted = start.elapsed();
    out.say("saving")?;
    let saving = Instant::now();
    let outcome = project
        .save_as(&paths.project)
        .with_context(|| format!("saving as {}", paths.project.display()))?;
    let saved = saving.elapsed();
    let unprotected = outcome.unprotected.len();
    project.close(false).context("closing the project")?;
    let bytes = std::fs::metadata(&paths.project)?.len();
    let rss = Rss::read()?;
    let total_time = start.elapsed();
    out.record(
        &Record::new("generate", "project")
            .with("splits", u64::from(options.splits))
            .with("faces", u64::from(faces))
            .with("levels", u64::from(options.levels))
            .with("tiles", total)
            .with("vertices_written", vertices)
            .with("sculpt_ms", ms(sculpted))
            .with("save_ms", ms(saved))
            .with("total_ms", ms(total_time))
            .with("file_bytes", bytes)
            .with("file_mib", mib(bytes))
            .with("peak_rss_mib", mib(rss.peak))
            .with("budget_mib", options.budget_mib)
            .with("unprotected_tiles", unprotected),
    )?;
    out.say(&format!(
        "generate: {total} tiles, {vertices} vertices written in {:.1} s, saved in {:.1} s; \
         file {:.1} MiB, peak RSS {:.1} MiB (budget {} MiB)",
        sculpted.as_secs_f64(),
        saved.as_secs_f64(),
        mib(bytes),
        mib(rss.peak),
        options.budget_mib
    ))?;
    Ok(())
}

/// Buffers [`sculpt_tile`] reuses from tile to tile.
#[derive(Debug, Default)]
struct Scratch {
    /// The tile's evaluated positions, copied out of the store.
    positions: Vec<Vec3>,
    /// The writes: each owned vertex's `(i, j)` and its push.
    writes: Vec<(u32, u32, f32)>,
}

/// Sculpts one tile with `octave`: evaluates it, works out the push of every vertex it owns
/// (into `scratch.writes`), and writes them into its corrections' `z`.
///
/// # Errors
///
/// If the tile can't be evaluated or edited.
fn sculpt_tile(
    project: &mut Project,
    noise: ValueNoise,
    octave: Octave,
    key: TileKey,
    scratch: &mut Scratch,
) -> Result<()> {
    let Scratch { positions, writes } = scratch;
    let side = key.side();
    let store = project.store();
    // The positions are copied out: the topology is borrowed from the store too.
    positions.clear();
    positions.extend_from_slice(store.evaluate(key)?);
    writes.clear();
    let topology = store.mesh().topology();
    let grid = (0..side).flat_map(|j| (0..side).map(move |i| (i, j)));
    for ((i, j), &p) in grid.zip(positions.iter()) {
        if owns(topology, &key, i, j) {
            writes.push((i, j, octave.push(noise, p)));
        }
    }
    store.edit_tile(key, |edit| {
        writes
            .iter()
            .try_for_each(|&(i, j, z)| edit.set(i, j, Vec3::new(0.0, 0.0, z)))
    })??;
    Ok(())
}
