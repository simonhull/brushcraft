//! `display-study`: the display-tile encoding, chosen by measurement.
//!
//! Display tiles (`geom::DisplayTile`) are derived records in the project file, so their
//! encoding sets how much the file grows. This builds every display tile of the faces sampled
//! (`--samples`, evenly spaced) with each candidate position codec and each normal encoding, and
//! measures, per combination:
//!
//! - **bytes** of each tile as the store would write it: the encoded tile compressed with the
//!   store's zstd level, plus a record's framing, key, encoding byte and bound
//!   ([`RECORD_OVERHEAD`]); projected over every face and taken as a fraction of the project file's
//!   current size (all levels, and the levels up to `finest − 2` alone, the design note's lever);
//! - **`e`** per level (it includes the codec's quantization), in world units and in quad edges of
//!   its level;
//! - the **decode** cost of a `finest − 1` tile (positions and normals);
//! - the decision-4 **constraints**: whether every border vertex decodes to the finest position it
//!   subsamples, bit for bit (so every tile holding it agrees: the store's copies of a finest
//!   vertex agree bit for bit), and whether a tile decodes on its own (a property of the codec: the
//!   base-offset one needs the base level);
//! - the largest **position** error (decoded against subsampled, in quad edges) and **normal**
//!   error (decoded against the normalized mean, in degrees).
//!
//! One JSON line per combination (`{"bench": "display-study", "result": "codec", …}`), one for
//! the run, and a table on stderr. The project is opened read-only in effect (nothing is edited
//! or saved); measure a reflinked copy all the same.

use std::collections::BTreeMap;
use std::io::Write;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, ensure};
use geom::{
    BaseOffsets, DisplayTile, FaceId, FixedPoint, Grid, Interpolating, Level, MeanNormals,
    NormalEncoding, PositionSource, Precision, Region, TileKey, Vec3, build_face_with, mip_normals,
    subsample,
};
use store::{Evaluation, TileStore};

use crate::cli::Options;
use crate::paths::Paths;
use crate::report::{Output, Record, ms, to_f64};
use crate::session;

/// The zstd level the store compresses tile payloads with.
const ZSTD_LEVEL: i32 = 3;

/// Bytes a display record adds around its compressed tile: record framing (8 head, 4
/// checksum), the tile key (12), an encoding byte and the bound `e` (4).
const RECORD_OVERHEAD: usize = 8 + 4 + 12 + 1 + 4;

/// A candidate position codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Candidate {
    /// Offsets from the base level in its frame, as correction tiles are stored (design note).
    BaseOffsets,
    /// Midpoint-interpolating residuals at a precision.
    Interpolating(Precision),
    /// 16-bit fixed point in the tile's bounding box.
    FixedPoint,
}

impl Candidate {
    /// Every candidate, in the table's order.
    const ALL: [Self; 8] = [
        Self::BaseOffsets,
        Self::Interpolating(Precision::Lossless),
        Self::Interpolating(Precision::Bits16),
        Self::Interpolating(Precision::QuadFraction(10)),
        Self::Interpolating(Precision::QuadFraction(8)),
        Self::Interpolating(Precision::QuadFraction(6)),
        Self::Interpolating(Precision::QuadFraction(4)),
        Self::FixedPoint,
    ];

    /// The candidate's name in the records.
    const fn name(self) -> &'static str {
        match self {
            Self::BaseOffsets => "A base offsets",
            Self::Interpolating(Precision::Lossless) => "B lossless",
            Self::Interpolating(Precision::Bits16) => "B 16-bit",
            Self::Interpolating(Precision::QuadFraction(10)) => "B quad/1024",
            Self::Interpolating(Precision::QuadFraction(8)) => "B quad/256",
            Self::Interpolating(Precision::QuadFraction(6)) => "B quad/64",
            Self::Interpolating(Precision::QuadFraction(4)) => "B quad/16",
            Self::Interpolating(Precision::QuadFraction(_)) => "B quad/2^n",
            Self::FixedPoint => "C fixed point",
        }
    }

    /// Whether a tile decodes on its own (decision 4's second constraint): all but the
    /// base-offset codec, which reads the base level's positions and frames.
    const fn standalone(self) -> bool {
        !matches!(self, Self::BaseOffsets)
    }

    /// `face`'s display tiles with this codec and `normals`; `base` holds the face's base grids
    /// by level, for the base-offset codec.
    ///
    /// # Errors
    ///
    /// The build's.
    fn build(
        self,
        base: &[Grid],
        normals: NormalEncoding,
        face: FaceId,
        finest: (&Grid, &Grid, Level),
    ) -> Result<Vec<DisplayTile>> {
        let (positions, unit, level) = finest;
        Ok(match self {
            Self::BaseOffsets => build_face_with(
                &BaseOffsets::new(base),
                normals,
                face,
                positions,
                unit,
                level,
            )?,
            Self::Interpolating(p) => build_face_with(
                &Interpolating::new(p),
                normals,
                face,
                positions,
                unit,
                level,
            )?,
            Self::FixedPoint => {
                build_face_with(&FixedPoint, normals, face, positions, unit, level)?
            }
        })
    }

    /// `tile` read back from its bytes with this codec and `normals`.
    ///
    /// # Errors
    ///
    /// The decode's.
    fn decode(self, base: &[Grid], normals: NormalEncoding, tile: &DisplayTile) -> Result<()> {
        let (key, error, bytes) = (*tile.key(), tile.error(), tile.encoded());
        match self {
            Self::BaseOffsets => {
                DisplayTile::decode_with(&BaseOffsets::new(base), normals, key, error, bytes)
            }
            Self::Interpolating(p) => {
                DisplayTile::decode_with(&Interpolating::new(p), normals, key, error, bytes)
            }
            Self::FixedPoint => DisplayTile::decode_with(&FixedPoint, normals, key, error, bytes),
        }?;
        Ok(())
    }
}

/// The name of a normal encoding in the records.
const fn normals_name(normals: NormalEncoding) -> &'static str {
    match normals {
        NormalEncoding::Oct16 => "oct16",
        NormalEncoding::Oct8 => "oct8",
    }
}

/// What one combination measured over the faces so far.
#[derive(Debug, Clone, Default)]
struct Tally {
    /// Record bytes, by level.
    bytes: BTreeMap<u8, u64>,
    /// Bytes of the positions' part and of the normals' part, each compressed alone.
    parts: (u64, u64),
    /// Each tile's `e`, by level.
    errors: BTreeMap<u8, Vec<f32>>,
    /// Each tile's `e` in quad edges of its level, by level.
    relative: BTreeMap<u8, Vec<f32>>,
    /// Time decoding `finest − 1` tiles, and how many.
    decode: (Duration, usize),
    /// Time building faces, and how many.
    build: (Duration, usize),
    /// Border vertices decoded to other bits than the finest vertex they subsample, and how many
    /// were compared.
    border: (u64, u64),
    /// The largest position error, in quad edges of the level.
    position_error: f64,
    /// The largest normal error, in degrees.
    normal_error: f64,
}

/// One face's reference data: the finest positions subsampled to each level, the normalized
/// mean normals at each level, and each level's mean quad edge.
struct Truth {
    /// By level: the subsampled positions (face grid).
    positions: BTreeMap<u8, Grid>,
    /// By level: the unit mean normals (face grid), row-major.
    normals: BTreeMap<u8, Vec<Vec3>>,
    /// By level: the mean quad edge.
    edge: BTreeMap<u8, f64>,
}

/// The finest positions and normals of `face`, put together from its tiles at `level`.
///
/// # Errors
///
/// If a tile can't be evaluated.
fn finest_grids(store: &mut TileStore, face: FaceId, level: Level) -> Result<(Grid, Grid)> {
    let keys = face_tiles(face, level)?;
    store.evaluate_many(&keys, Evaluation::Normals)?;
    let side = level.face_quads() + 1;
    let count = usize::try_from(side)?.pow(2);
    let (mut positions, mut normals) = (vec![Vec3::ZERO; count], vec![Vec3::ZERO; count]);
    for key in keys {
        let region = Region::of_tile(&key);
        let tile_side = key.side();
        let tile_positions = store.evaluate(key)?.to_vec();
        let tile_normals = store.normals(key)?.to_vec();
        for (index, (p, n)) in tile_positions.into_iter().zip(tile_normals).enumerate() {
            let index = u32::try_from(index)?;
            let (x, y) = (
                region.x0() + index % tile_side,
                region.y0() + index / tile_side,
            );
            let at = usize::try_from(y * side + x)?;
            *positions.get_mut(at).context("inside the face")? = p;
            *normals.get_mut(at).context("inside the face")? = n;
        }
    }
    Ok((Grid::new(side, positions)?, Grid::new(side, normals)?))
}

/// `face`'s base grids at every level below `finest`, evaluated (level 0 from the base mesh).
///
/// # Errors
///
/// If a tile can't be evaluated.
fn base_grids(store: &mut TileStore, face: FaceId, finest: Level) -> Result<Vec<Grid>> {
    let mut grids = Vec::new();
    let corners = [(0, 0), (1, 0), (0, 1), (1, 1)]
        .map(|(i, j)| store.mesh().position(face, Level::ZERO, i, j));
    let corners: Vec<Vec3> = corners
        .into_iter()
        .collect::<Option<_>>()
        .context("corners")?;
    grids.push(Grid::new(2, corners)?);
    for l in 1..finest.get() {
        let level = Level::new(l)?;
        let side = level.face_quads() + 1;
        let mut data = vec![Vec3::ZERO; usize::try_from(side)?.pow(2)];
        let keys = face_tiles(face, level)?;
        store.evaluate_many(&keys, Evaluation::Positions)?;
        for key in keys {
            let region = Region::of_tile(&key);
            let tile_side = key.side();
            for (index, &p) in store.evaluate(key)?.iter().enumerate() {
                let index = u32::try_from(index)?;
                let (x, y) = (
                    region.x0() + index % tile_side,
                    region.y0() + index / tile_side,
                );
                *data
                    .get_mut(usize::try_from(y * side + x)?)
                    .context("inside the face")? = p;
            }
        }
        grids.push(Grid::new(side, data)?);
    }
    Ok(grids)
}

/// Every base displacement tile of `face` at `level`.
///
/// # Errors
///
/// Never for a level whose tiles per side fit a `u16`.
fn face_tiles(face: FaceId, level: Level) -> Result<Vec<TileKey>> {
    let per_side = u16::try_from(level.tiles_per_side())?;
    (0..per_side)
        .flat_map(|ty| (0..per_side).map(move |tx| (tx, ty)))
        .map(|(tx, ty)| Ok(TileKey::displacement(face, level, tx, ty)?))
        .collect()
}

/// The reference data for a face with finest `positions` and `unit` normals at `finest`.
///
/// # Errors
///
/// If a grid doesn't halve.
fn truth(positions: &Grid, unit: &Grid, finest: Level) -> Result<Truth> {
    let mut out = Truth {
        positions: BTreeMap::new(),
        normals: BTreeMap::new(),
        edge: BTreeMap::new(),
    };
    let (mut p, mut m) = (positions.clone(), MeanNormals::from_unit(unit.clone()));
    for l in (0..finest.get()).rev() {
        p = subsample(&p)?;
        m = mip_normals(&m)?;
        out.edge.insert(l, mean_edge(&p));
        out.normals.insert(
            l,
            m.as_slice()
                .iter()
                .map(|v| v.normalized().unwrap_or(Vec3::new(0.0, 0.0, 1.0)))
                .collect(),
        );
        out.positions.insert(l, p.clone());
    }
    Ok(out)
}

/// The mean length of a grid's row and column edges.
fn mean_edge(grid: &Grid) -> f64 {
    let (mut sum, mut count) = (0.0, 0_usize);
    for j in 0..grid.side() {
        for i in 0..grid.side() {
            let Some(p) = grid.get(i, j) else { continue };
            for q in [grid.get(i + 1, j), grid.get(i, j + 1)]
                .into_iter()
                .flatten()
            {
                sum += f64::from((q - p).length());
                count += 1;
            }
        }
    }
    sum / to_f64(count.max(1))
}

/// The angle between unit vectors `a` and `b`, in degrees (`atan2` of the cross and dot
/// products in `f64`, which keeps small angles).
fn angle_degrees(a: Vec3, b: Vec3) -> f64 {
    let wide = |v: Vec3| [f64::from(v.x), f64::from(v.y), f64::from(v.z)];
    let ([ax, ay, az], [bx, by, bz]) = (wide(a), wide(b));
    let cross = [ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx];
    let sin = cross.iter().map(|c| c * c).sum::<f64>().sqrt();
    sin.atan2(ax * bx + ay * by + az * bz).to_degrees()
}

/// Adds what `tiles` (one face's, one combination's) measure to `tally`.
///
/// # Errors
///
/// If compressing a tile fails or a tile doesn't match its truth.
fn measure(tally: &mut Tally, tiles: &[DisplayTile], truth: &Truth) -> Result<()> {
    for tile in tiles {
        let key = tile.key();
        let level = key.level().get();
        let packed = zstd::bulk::compress(tile.encoded(), ZSTD_LEVEL)?;
        let (head, rest) = tile.encoded().split_at_checked(4).context("a length")?;
        let length = usize::try_from(u32::from_le_bytes(head.try_into()?))?;
        let (position_part, normal_part) =
            rest.split_at_checked(length).context("the positions")?;
        tally.parts.0 += u64::try_from(zstd::bulk::compress(position_part, ZSTD_LEVEL)?.len())?;
        tally.parts.1 += u64::try_from(zstd::bulk::compress(normal_part, ZSTD_LEVEL)?.len())?;
        *tally.bytes.entry(level).or_default() += u64::try_from(packed.len() + RECORD_OVERHEAD)?;
        let edge = truth
            .edge
            .get(&level)
            .copied()
            .context("an edge per level")?;
        tally.errors.entry(level).or_default().push(tile.error());
        #[expect(
            clippy::cast_possible_truncation,
            reason = "a ratio for a statistic: f32 is plenty"
        )]
        let relative = (f64::from(tile.error()) / edge) as f32;
        tally.relative.entry(level).or_default().push(relative);
        let region = Region::of_tile(key);
        let want = truth.positions.get(&level).context("positions per level")?;
        let normals = truth.normals.get(&level).context("normals per level")?;
        let side = key.side();
        let last = side - 1;
        for (index, (&got, &normal)) in tile.positions().iter().zip(tile.normals()).enumerate() {
            let index = u32::try_from(index)?;
            let (i, j) = (index % side, index / side);
            let (x, y) = (region.x0() + i, region.y0() + j);
            let exact = want.get(x, y).context("inside the face")?;
            if i == 0 || j == 0 || i == last || j == last {
                tally.border.1 += 1;
                if got.to_bits() != exact.to_bits() {
                    tally.border.0 += 1;
                }
            }
            let off = f64::from((got - exact).length()) / edge;
            tally.position_error = tally.position_error.max(off);
            let mean = normals
                .get(usize::try_from(y * want.side() + x)?)
                .copied()
                .context("inside the face")?;
            tally.normal_error = tally.normal_error.max(angle_degrees(normal, mean));
        }
    }
    Ok(())
}

/// The value at quantile `q` (nearest rank) of `values`, or NaN if there are none.
fn quantile(values: &[f32], q: f64) -> f64 {
    let mut sorted: Vec<f32> = values.to_vec();
    sorted.sort_by(f32::total_cmp);
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a rank within the vector, from a quantile in [0, 1]"
    )]
    let rank = ((q * to_f64(n)).ceil() as usize).clamp(1, n);
    sorted.get(rank - 1).map_or(f64::NAN, |&v| f64::from(v))
}

/// `count` faces of `faces`, evenly spaced (all of them if `count` is at least `faces`).
fn sample_faces(faces: u32, count: usize) -> Vec<FaceId> {
    let count = u32::try_from(count).unwrap_or(u32::MAX).min(faces).max(1);
    (0..count)
        .map(|n| {
            FaceId::new(
                u32::try_from(u64::from(n) * u64::from(faces) / u64::from(count)).unwrap_or(0),
            )
        })
        .collect()
}

/// Runs the study over the project at `paths`, reporting to `out`.
///
/// # Errors
///
/// If the project can't be opened, a tile evaluated, built or decoded, or writing the report
/// fails.
pub(crate) fn run<J: Write, H: Write>(
    options: &Options,
    paths: &Paths,
    out: &mut Output<J, H>,
) -> Result<()> {
    let finest = session::level(options)?;
    ensure!(
        finest.get() >= 2,
        "the study needs at least two display levels"
    );
    let file_bytes = std::fs::metadata(&paths.project)
        .with_context(|| format!("reading {}", paths.project.display()))?
        .len();
    let mut project = session::open(options, paths)?;
    let store = project.store();
    let face_count = store.mesh().face_count();
    let faces = sample_faces(face_count, options.samples);
    out.say(&format!(
        "display-study: {} of {face_count} faces, finest level {finest}, file {:.1} MiB",
        faces.len(),
        crate::report::mib(file_bytes)
    ))?;
    let encodings = [NormalEncoding::Oct16, NormalEncoding::Oct8];
    let mut tallies: Vec<((Candidate, NormalEncoding), Tally)> = Candidate::ALL
        .iter()
        .flat_map(|&c| encodings.map(|n| ((c, n), Tally::default())))
        .collect();
    let started = Instant::now();
    for (done, &face) in faces.iter().enumerate() {
        let (positions, unit) = finest_grids(store, face, finest)?;
        let base = base_grids(store, face, finest)?;
        let reference = truth(&positions, &unit, finest)?;
        for ((candidate, normals), tally) in &mut tallies {
            let building = Instant::now();
            let tiles = candidate.build(&base, *normals, face, (&positions, &unit, finest))?;
            tally.build.0 += building.elapsed();
            tally.build.1 += 1;
            let below = finest.parent().context("a level below the finest")?;
            let decoding = Instant::now();
            let mut decoded = 0;
            for tile in tiles.iter().filter(|t| t.key().level() == below) {
                candidate.decode(&base, *normals, tile)?;
                decoded += 1;
            }
            tally.decode.0 += decoding.elapsed();
            tally.decode.1 += decoded;
            measure(tally, &tiles, &reference)?;
        }
        out.progress("faces", done + 1, faces.len(), started)?;
    }
    let scale = f64::from(face_count) / to_f64(faces.len());
    report(&tallies, finest, (faces.len(), scale), file_bytes, out)?;
    out.record(
        &Record::new("display-study", "run")
            .with("faces", faces.len())
            .with("face_count", u64::from(face_count))
            .with("finest", u64::from(finest.get()))
            .with("file_bytes", file_bytes)
            .with("seconds", started.elapsed().as_secs_f64()),
    )?;
    Ok(())
}

/// Writes each combination's record and table row to `out`: `faces` is how many were measured
/// and the factor that projects them onto every face.
///
/// # Errors
///
/// If writing fails, or a count doesn't fit.
fn report<J: Write, H: Write>(
    tallies: &[((Candidate, NormalEncoding), Tally)],
    finest: Level,
    (measured, scale): (usize, f64),
    file_bytes: u64,
    out: &mut Output<J, H>,
) -> Result<()> {
    let file = to_f64(usize::try_from(file_bytes)?);
    out.say(&format!(
        "{:<15} {:>6} {:>9} {:>9} {:>11} {:>8} {:>9} {:>9} {:>10} {:>9}",
        "codec",
        "normal",
        "growth",
        "to f-2",
        "decode µs",
        "borders",
        "alone",
        "pos err",
        "normal °",
        "e f-1 p50"
    ))?;
    for ((candidate, normals), tally) in tallies {
        let total: u64 = tally.bytes.values().sum();
        let lower: u64 = tally
            .bytes
            .iter()
            .filter(|&(&l, _)| l + 2 <= finest.get())
            .map(|(_, &b)| b)
            .sum();
        let growth = to_f64(usize::try_from(total)?) * scale / file;
        let growth_lower = to_f64(usize::try_from(lower)?) * scale / file;
        let decode_us = ms(tally.decode.0) * 1000.0 / to_f64(tally.decode.1.max(1));
        let build_ms = ms(tally.build.0) / to_f64(tally.build.1.max(1));
        let borders_exact = tally.border.0 == 0;
        let mut record = Record::new("display-study", "codec")
            .with("codec", candidate.name())
            .with("normals", normals_name(*normals))
            .with("faces", measured)
            .with("bytes_sampled", total)
            .with("growth", growth)
            .with("growth_to_finest_minus_2", growth_lower)
            .with(
                "positions_growth",
                to_f64(usize::try_from(tally.parts.0)?) * scale / file,
            )
            .with(
                "normals_growth",
                to_f64(usize::try_from(tally.parts.1)?) * scale / file,
            )
            .with("decode_us_per_tile", decode_us)
            .with("build_ms_per_face", build_ms)
            .with("border_mismatches", tally.border.0)
            .with("border_vertices", tally.border.1)
            .with("borders_exact", borders_exact)
            .with("standalone", candidate.standalone())
            .with("position_error_max_edges", tally.position_error)
            .with("normal_error_max_degrees", tally.normal_error);
        for (&l, errors) in &tally.errors {
            let relative = tally.relative.get(&l).map_or(&[][..], Vec::as_slice);
            record = record
                .with(format!("e_l{l}_p50"), quantile(errors, 0.5))
                .with(format!("e_l{l}_p99"), quantile(errors, 0.99))
                .with(format!("e_l{l}_max"), quantile(errors, 1.0))
                .with(format!("e_l{l}_edges_p50"), quantile(relative, 0.5))
                .with(format!("e_l{l}_edges_max"), quantile(relative, 1.0))
                .with(
                    format!("bytes_l{l}"),
                    tally.bytes.get(&l).copied().unwrap_or(0),
                );
        }
        out.record(&record)?;
        let below = finest.get() - 1;
        let e_below = tally
            .errors
            .get(&below)
            .map_or(f64::NAN, |e| quantile(e, 0.5));
        out.say(&format!(
            "{:<15} {:>6} {:>8.1}% {:>8.1}% {:>11.1} {:>8} {:>9} {:>9.2e} {:>10.4} {:>9.2e}",
            candidate.name(),
            normals_name(*normals),
            growth * 100.0,
            growth_lower * 100.0,
            decode_us,
            if borders_exact { "exact" } else { "differ" },
            if candidate.standalone() { "yes" } else { "no" },
            tally.position_error,
            tally.normal_error,
            e_below,
        ))?;
    }
    Ok(())
}
