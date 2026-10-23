//! `brush` and `save`: a stroke of draw steps over a footprint along a path, then, for `save`,
//! a save of what it did.
//!
//! The stroke starts at a random place on the surface (from the seed). The radius is set there
//! to cover about `footprint` vertices (50,000 for the benchmark), and each dab moves a tenth
//! of a radius of arc east along a great circle (see [`crate::surface::path`]), its centre put
//! on the surface by a raycast from the origin. Every dab is one [`sculpt::draw_step`], timed
//! as a whole (the latency reported, p50 and p99) and by its phases (the means reported).
//!
//! The stroke runs warm: planning it ([`Plan::cold`]) raycasts every dab's centre first, then
//! [`Plan::warm`] evaluates every tile the dabs' spheres reach, positions and normals, so the steps
//! find their tiles resident and evaluated (as an app's do, the view having drawn them). The index
//! is loaded from the project's stored bounds, which evaluates nothing, so this warm-up is the
//! stroke's own (since 38eab39; before, building the index on open warmed it). Before the warm-up,
//! `brush` times one dab where the stroke starts, cold, and reports it beside the warm stroke
//! (`cold_first_dab_ms`): what a stroke's first dab costs over a region the view hasn't drawn.
//! What a cold region costs in full is `page-in`'s measure.
//!
//! Loading the spatial index comes first, and is reported apart: from the bounds `generate`
//! saved with the tiles, it evaluates no tile (a project without them has all 153,600 level-9
//! tiles evaluated).
//!
//! `save` then counts the vertices the stroke changed (each once, through the tile that owns
//! it), saves the project with [`store::Project::save`], and reports the save's time and how
//! much the file grew per edited vertex. The save goes into the project file, which gains a
//! save point every run.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::time::Instant;

use anyhow::{Context as _, Result};
use geom::{TileKey, Vec3};
use store::{Evaluation, Project, SpatialIndex};

use crate::cli::Options;
use crate::noise::Rng;
use crate::paths::Paths;
use crate::report::{Output, Record, mib, ms};
use crate::session;
use crate::surface::{calibrated_radius, on_surface, path};

/// How far the brush moves each dab, in radii.
const STEP_RADII: f32 = 0.1;

/// How big a planned stroke's brush is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Size {
    /// Big enough to cover about this many vertices where the stroke starts.
    Footprint(usize),
    /// This radius.
    Radius(f32),
}

/// A stroke ready to run: its radius and the centres of its dabs.
#[derive(Debug, Clone)]
pub(crate) struct Plan {
    /// The brush radius, covering about the footprint asked for where the stroke starts.
    pub(crate) radius: f32,
    /// The dabs' centres, on the surface.
    pub(crate) centres: Vec<Vec3>,
}

impl Plan {
    /// A stroke of `steps` dabs starting in `direction` from the origin, with a brush of
    /// `size` (see the module notes), warm: [`Plan::cold`], then [`Plan::warm`].
    ///
    /// # Errors
    ///
    /// As [`Plan::cold`] and [`Plan::warm`].
    pub(crate) fn new(
        project: &mut Project,
        index: &SpatialIndex,
        direction: Vec3,
        size: Size,
        steps: usize,
    ) -> Result<Self> {
        let plan = Self::cold(project, index, direction, size, steps)?;
        plan.warm(project, index)?;
        Ok(plan)
    }

    /// The stroke [`Plan::new`] plans, without the warm-up: finding the dabs' centres raycasts
    /// the path, which evaluates the positions of the tiles the rays reach, and nothing more.
    ///
    /// # Errors
    ///
    /// If a tile can't be evaluated, or a raycast finds no surface.
    pub(crate) fn cold(
        project: &mut Project,
        index: &SpatialIndex,
        direction: Vec3,
        size: Size,
        steps: usize,
    ) -> Result<Self> {
        let store = project.store();
        let radius = match size {
            Size::Radius(radius) => radius,
            Size::Footprint(footprint) => {
                let start = on_surface(index, store, direction)?;
                calibrated_radius(index, store, start, footprint)?
            }
        };
        let centres = path(direction, STEP_RADII * radius, steps)
            .map(|d| on_surface(index, store, d))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { radius, centres })
    }

    /// Evaluates every tile a dab's sphere reaches, positions and normals, together: the stroke
    /// then runs warm, as an app's does over tiles its view has drawn.
    ///
    /// # Errors
    ///
    /// If a tile can't be evaluated.
    pub(crate) fn warm(&self, project: &mut Project, index: &SpatialIndex) -> Result<()> {
        let reached: BTreeSet<TileKey> = self
            .centres
            .iter()
            .flat_map(|&centre| index.query_sphere(centre, self.radius))
            .collect();
        let reached: Vec<TileKey> = reached.into_iter().collect();
        project
            .store()
            .evaluate_many(&reached, Evaluation::Normals)
            .context("warming the stroke's tiles")
    }
}

/// Opens the project, loads its index and plans the stroke `options` asks for, reporting as
/// `bench`, without warming it ([`Plan::cold`]).
///
/// # Errors
///
/// As [`session::open`], [`session::load_index`] and [`Plan::cold`].
fn prepare<J: Write, H: Write>(
    options: &Options,
    paths: &Paths,
    bench: &str,
    out: &mut Output<J, H>,
) -> Result<(Project, SpatialIndex, Plan)> {
    let level = session::level(options)?;
    let mut project = session::open(options, paths)?;
    let index = session::load_index(&mut project, level, bench, out)?;
    let direction = Rng::new(options.seed).direction();
    let plan = Plan::cold(
        &mut project,
        &index,
        direction,
        Size::Footprint(options.footprint),
        options.steps,
    )?;
    Ok((project, index, plan))
}

/// Runs `brush` (see the module notes).
///
/// # Errors
///
/// If the project can't be opened, the index loaded or a step run, or writing the report fails.
pub(crate) fn run<J: Write, H: Write>(
    options: &Options,
    paths: &Paths,
    out: &mut Output<J, H>,
) -> Result<()> {
    let (mut project, mut index, plan) = prepare(options, paths, "brush", out)?;
    out.say(&format!(
        "brush: {} draw steps, radius {:.5}",
        plan.centres.len(),
        plan.radius
    ))?;
    // A dab where the stroke starts, before its tiles are evaluated: what the first dab of a
    // stroke over a region the view hasn't drawn costs. Then the warm-up, and the stroke.
    let cold = session::stroke(
        &mut project,
        &mut index,
        plan.centres.get(..1).unwrap_or_default(),
        plan.radius,
        options.strength,
    )?;
    let cold_ms = cold.first().map_or(0.0, |dab| ms(dab.wall));
    plan.warm(&mut project, &index)?;
    let dabs = session::stroke(
        &mut project,
        &mut index,
        &plan.centres,
        plan.radius,
        options.strength,
    )?;
    out.record(&session::add_dabs(
        Record::new("brush", "stroke")
            .with("level", u64::from(options.level))
            .with("steps", dabs.len())
            .with("radius", f64::from(plan.radius))
            .with("footprint_target", options.footprint)
            .with("budget_mib", options.budget_mib)
            .with("cold_first_dab_ms", cold_ms),
        &dabs,
    ))?;
    out.say(&format!(
        "brush: {}; cold first dab {cold_ms:.2} ms",
        session::describe(&dabs)
    ))?;
    project.close(false).context("closing the project")?;
    Ok(())
}

/// Every correction of each of `keys`, for comparing before and after a stroke.
///
/// # Errors
///
/// If a tile's corrections can't be loaded.
fn corrections(project: &mut Project, keys: &[TileKey]) -> Result<BTreeMap<TileKey, Vec<Vec3>>> {
    let store = project.store();
    let mut all = BTreeMap::new();
    for &key in keys {
        let view = store.correction(key)?;
        let side = view.side();
        let values = (0..side)
            .flat_map(|j| (0..side).map(move |i| (i, j)))
            .filter_map(|(i, j)| view.get(i, j))
            .collect();
        all.insert(key, values);
    }
    Ok(all)
}

/// The vertices whose corrections differ between `before` and `after`, each counted once:
/// through the tile that owns it.
fn changed_vertices(
    project: &mut Project,
    before: &BTreeMap<TileKey, Vec<Vec3>>,
    after: &BTreeMap<TileKey, Vec<Vec3>>,
) -> usize {
    let topology = project.store().mesh().topology();
    before
        .iter()
        .filter_map(|(key, old)| Some((key, old, after.get(key)?)))
        .map(|(key, old, new)| {
            let side = key.side();
            let grid = (0..side).flat_map(|j| (0..side).map(move |i| (i, j)));
            grid.zip(old.iter().zip(new))
                .filter(|&((i, j), (a, b))| a != b && topology.owns(key, i, j))
                .count()
        })
        .sum()
}

/// Runs `save` (see the module notes).
///
/// # Errors
///
/// If the project can't be opened, the index built, a step run or the save made, or writing the
/// report fails.
pub(crate) fn run_save<J: Write, H: Write>(
    options: &Options,
    paths: &Paths,
    out: &mut Output<J, H>,
) -> Result<()> {
    let (mut project, mut index, plan) = prepare(options, paths, "save", out)?;
    plan.warm(&mut project, &index)?;
    // Every tile a dab can reach: the corrections that may change.
    let mut reached: Vec<TileKey> = plan
        .centres
        .iter()
        .flat_map(|&c| index.query_sphere(c, plan.radius).collect::<Vec<_>>())
        .collect();
    reached.sort_unstable();
    reached.dedup();
    let before = corrections(&mut project, &reached)?;
    let size_before = std::fs::metadata(&paths.project)?.len();
    out.say(&format!(
        "save: {} draw steps, radius {:.5}, then a save",
        plan.centres.len(),
        plan.radius
    ))?;
    let dabs = session::stroke(
        &mut project,
        &mut index,
        &plan.centres,
        plan.radius,
        options.strength,
    )?;
    let after = corrections(&mut project, &reached)?;
    let edited = changed_vertices(&mut project, &before, &after);
    let start = Instant::now();
    let outcome = project.save().context("saving")?;
    let took = start.elapsed();
    let unprotected = outcome.unprotected.len();
    let size_after = std::fs::metadata(&paths.project)?.len();
    let growth = size_after.saturating_sub(size_before);
    #[expect(
        clippy::cast_precision_loss,
        reason = "byte and vertex counts far below 2^53 convert exactly"
    )]
    let per_vertex = growth as f64 / edited.max(1) as f64;
    out.record(
        &Record::new("save", "save")
            .with("level", u64::from(options.level))
            .with("steps", dabs.len())
            .with("radius", f64::from(plan.radius))
            .with("edited_vertices", edited)
            .with("save_ms", ms(took))
            .with("file_bytes_before", size_before)
            .with("file_bytes_after", size_after)
            .with("growth_bytes", growth)
            .with("growth_mib", mib(growth))
            .with("bytes_per_edited_vertex", per_vertex)
            .with("unprotected_tiles", unprotected),
    )?;
    out.say(&format!(
        "save: {edited} vertices edited in {} steps; saved in {:.1} ms; file grew {growth} \
         bytes, {per_vertex:.2} bytes per edited vertex",
        dabs.len(),
        ms(took)
    ))?;
    project.close(false).context("closing the project")?;
    Ok(())
}
