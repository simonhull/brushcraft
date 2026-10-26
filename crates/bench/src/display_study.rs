//! `display-study`: the display-tile encoding, chosen by measurement.
//!
//! Display tiles (`geom::DisplayTile`) are derived records in the project file, so their
//! encoding sets how much the file grows. This builds every display tile of the faces sampled
//! (`--samples`, evenly spaced; all of them for the normals, whose mip reads across faces), in
//! two parts.
//!
//! **Positions**, a face at a time, with each candidate codec ([`Candidate`]):
//!
//! - **bytes** of each tile's positions, compressed with the store's zstd level, plus a record's
//!   framing, key, encoding byte and bound ([`RECORD_OVERHEAD`]); projected over every face and
//!   taken as a fraction of the project file's current size (all levels, and the levels up to
//!   `finest − 2` alone, the design note's lever);
//! - **`e`** per level (it includes the codec's quantization), in world units and in quad edges;
//! - the **decode** cost of a `finest − 1` tile's positions;
//! - the decision-4 **constraints**: whether every border vertex decodes to the finest position it
//!   subsamples, bit for bit (so every tile holding it agrees: the store's copies of a finest
//!   vertex agree bit for bit), and whether a tile decodes on its own (a property of the codec: the
//!   base-offset one needs the base level);
//! - the largest **position** error (decoded against subsampled, in quad edges of the level).
//!
//! **Normals**, a level at a time over every face, with the chosen positions and each candidate
//! normal encoding ([`ENCODINGS`]): the normals' compressed bytes (roughness included), the total
//! growth with the chosen positions, the largest and 99th-percentile angle between each decoded
//! normal and the mean it encodes, the decode cost of a `finest − 1` tile's normals, and the
//! roughness bytes raw against delta-coded.
//!
//! One JSON line per position codec (`"result": "positions"`) and per normal encoding
//! (`"result": "normals"`), one for the run, and tables on stderr. The project is opened
//! read-only in effect (nothing is edited or saved); measure a reflinked copy all the same.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, ensure};
use geom::{
    BaseOffsets, Channel, DisplayCodec, DisplayTile, FaceGrids, FaceId, FacePositions, FixedPoint,
    Grid, Interpolating, LayerId, Level, MeanNormals, NormalEncoding, PositionSource, Precision,
    Region, TileKey, Vec3, build_level_with, build_positions_with, mip_normals,
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

/// The position codec the normals are measured with: the one chosen by the position study.
const POSITIONS: Candidate = Candidate::Interpolating(Precision::QuadFraction(6));

/// The normal encodings measured.
const ENCODINGS: [NormalEncoding; 8] = [
    NormalEncoding::Octahedral(6),
    NormalEncoding::Octahedral(8),
    NormalEncoding::Octahedral(10),
    NormalEncoding::Octahedral(12),
    NormalEncoding::Octahedral(16),
    NormalEncoding::Predicted(8),
    NormalEncoding::Predicted(10),
    NormalEncoding::Predicted(12),
];

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

    /// `face`'s display positions with this codec; `base` holds the face's base grids by level,
    /// for the base-offset codec.
    ///
    /// # Errors
    ///
    /// The build's.
    fn build(self, base: &[Grid], face: FaceId, finest: (&Grid, Level)) -> Result<FacePositions> {
        let (positions, level) = finest;
        Ok(match self {
            Self::BaseOffsets => {
                build_positions_with(&BaseOffsets::new(base), face, positions, level)?
            }
            Self::Interpolating(p) => {
                build_positions_with(&Interpolating::new(p), face, positions, level)?
            }
            Self::FixedPoint => build_positions_with(&FixedPoint, face, positions, level)?,
        })
    }

    /// Tile `key`'s positions decoded from `bytes` with this codec.
    ///
    /// # Errors
    ///
    /// The decode's.
    fn decode(self, base: &[Grid], key: &TileKey, bytes: &[u8]) -> Result<Grid> {
        Ok(match self {
            Self::BaseOffsets => BaseOffsets::new(base).decode(key, bytes)?,
            Self::Interpolating(p) => Interpolating::new(p).decode(key, bytes)?,
            Self::FixedPoint => FixedPoint.decode(key, bytes)?,
        })
    }
}

/// The name of a normal encoding in the records.
fn encoding_name(encoding: NormalEncoding) -> String {
    match encoding {
        NormalEncoding::Octahedral(bits) => format!("oct 2x{bits}"),
        NormalEncoding::Predicted(bits) => format!("predicted 2x{bits}"),
    }
}

/// Angles in a histogram of [`ANGLE_BIN`]-degree bins, for percentiles of a few hundred million
/// of them without keeping each.
#[derive(Debug, Clone)]
struct Angles {
    /// Counts per bin; the last counts everything past the range.
    bins: Vec<u64>,
    /// The largest angle seen.
    max: f64,
}

/// Width of an [`Angles`] bin, in degrees.
const ANGLE_BIN: f64 = 0.0005;

/// Bins of [`Angles`]: up to 10°, then one for the rest.
const ANGLE_BINS: usize = 20_001;

impl Angles {
    /// An empty histogram.
    fn new() -> Self {
        Self {
            bins: vec![0; ANGLE_BINS],
            max: 0.0,
        }
    }

    /// Counts `degrees`.
    fn add(&mut self, degrees: f64) {
        self.max = self.max.max(degrees);
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a non-negative bin index, clamped to the last bin"
        )]
        let bin = ((degrees / ANGLE_BIN) as usize).min(ANGLE_BINS - 1);
        if let Some(count) = self.bins.get_mut(bin) {
            *count += 1;
        }
    }

    /// The angle at or below which a fraction `q` of the counted ones lie: the upper edge of the
    /// bin it falls in (the maximum for the last).
    fn quantile(&self, q: f64) -> f64 {
        let total: u64 = self.bins.iter().sum();
        let mut seen = 0_u64;
        for (index, &count) in self.bins.iter().enumerate() {
            seen += count;
            if to_f64(usize::try_from(seen).unwrap_or(usize::MAX))
                >= q * to_f64(usize::try_from(total).unwrap_or(usize::MAX))
            {
                return if index + 1 == ANGLE_BINS {
                    self.max
                } else {
                    to_f64(index + 1) * ANGLE_BIN
                };
            }
        }
        self.max
    }
}

/// What one position codec measured over the faces.
#[derive(Debug, Clone, Default)]
struct PositionTally {
    /// Record bytes (positions compressed, plus the record), by level.
    bytes: BTreeMap<u8, u64>,
    /// Each tile's `e`, by level.
    errors: BTreeMap<u8, Vec<f32>>,
    /// Each tile's `e` in quad edges of its level, by level.
    relative: BTreeMap<u8, Vec<f32>>,
    /// Time decoding `finest − 1` tiles' positions, and how many.
    decode: (Duration, usize),
    /// Time building faces, and how many.
    build: (Duration, usize),
    /// Border vertices decoded to other bits than the finest vertex they subsample, and how many
    /// were compared.
    border: (u64, u64),
    /// The largest position error, in quad edges of the level.
    position_error: f64,
}

/// What one normal encoding measured over the faces.
#[derive(Debug, Clone)]
struct NormalTally {
    /// Normals' bytes (roughness included), compressed.
    bytes: u64,
    /// Tiles measured.
    tiles: u64,
    /// The angle between each decoded normal and the normalized mean it encodes.
    angles: Angles,
    /// Time decoding `finest − 1` tiles' normals, and how many.
    decode: (Duration, usize),
    /// Roughness bytes compressed alone: raw, and delta-coded along each row.
    roughness: (u64, u64),
}

/// One face's reference data: the finest positions subsampled to each level and each level's
/// mean quad edge.
struct Truth {
    /// By level: the subsampled positions (face grid).
    positions: BTreeMap<u8, Grid>,
    /// By level: the mean quad edge.
    edge: BTreeMap<u8, f64>,
}

/// The store's finest unit normals, read as a [`PositionSource`] for [`mip_normals`]: each tile
/// evaluated on first read and kept.
struct StoreNormals<'a> {
    /// The store, evaluated through.
    store: RefCell<&'a mut TileStore>,
    /// Tiles read so far, their normals row-major.
    cache: RefCell<HashMap<TileKey, Vec<Vec3>>>,
}

impl PositionSource for StoreNormals<'_> {
    fn position(&self, face: FaceId, level: Level, i: u32, j: u32) -> Option<Vec3> {
        let (key, ti, tj) =
            TileKey::containing(LayerId::BASE, Channel::DISPLACEMENT, face, level, i, j)?;
        let index = usize::try_from(tj * key.side() + ti).ok()?;
        if let Some(normals) = self.cache.borrow().get(&key) {
            return normals.get(index).copied();
        }
        let normals = self.store.borrow_mut().normals(key).ok()?.to_vec();
        let found = normals.get(index).copied();
        self.cache.borrow_mut().insert(key, normals);
        found
    }
}

/// The finest positions of `face`, put together from its tiles at `level`; its normals are
/// evaluated too, for the [`StoreNormals`] read next.
///
/// # Errors
///
/// If a tile can't be evaluated.
fn finest_positions(store: &mut TileStore, face: FaceId, level: Level) -> Result<Grid> {
    let keys = face_tiles(face, level)?;
    store.evaluate_many(&keys, Evaluation::Normals)?;
    face_grid(store, &keys, level)
}

/// `level`'s face grid put together from `keys`, a face's tiles there, evaluated.
///
/// # Errors
///
/// If a tile can't be evaluated.
fn face_grid(store: &mut TileStore, keys: &[TileKey], level: Level) -> Result<Grid> {
    let side = level.face_quads() + 1;
    let mut data = vec![Vec3::ZERO; usize::try_from(side)?.pow(2)];
    for &key in keys {
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
    Ok(Grid::new(side, data)?)
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
        let keys = face_tiles(face, level)?;
        store.evaluate_many(&keys, Evaluation::Positions)?;
        grids.push(face_grid(store, &keys, level)?);
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

/// Every display tile key of `face` at `level`, row by row.
///
/// # Errors
///
/// Never for a level whose tiles per side fit a `u16`.
fn display_keys(face: FaceId, level: Level) -> Result<Vec<TileKey>> {
    let per_side = u16::try_from(level.tiles_per_side())?;
    (0..per_side)
        .flat_map(|ty| (0..per_side).map(move |tx| (tx, ty)))
        .map(|(tx, ty)| Ok(TileKey::display(face, level, tx, ty)?))
        .collect()
}

/// The reference data for a face with finest `positions` at `finest`.
///
/// # Errors
///
/// If a grid doesn't halve.
fn truth(positions: &Grid, finest: Level) -> Result<Truth> {
    let mut out = Truth {
        positions: BTreeMap::new(),
        edge: BTreeMap::new(),
    };
    let mut p = positions.clone();
    for l in (0..finest.get()).rev() {
        p = subsample(&p)?;
        out.edge.insert(l, mean_edge(&p));
        out.positions.insert(l, p.clone());
    }
    Ok(out)
}

/// The even vertices of `finer` (`(2i, 2j)`), one level coarser.
///
/// # Errors
///
/// If `finer`'s quads per side aren't even.
fn subsample(finer: &Grid) -> Result<Grid> {
    let quads = finer.side() - 1;
    ensure!(
        quads >= 2 && quads.is_multiple_of(2),
        "the grid doesn't halve"
    );
    let side = quads / 2 + 1;
    let data = (0..side)
        .flat_map(|j| (0..side).map(move |i| (i, j)))
        .map(|(i, j)| finer.get(2 * i, 2 * j).context("inside"))
        .collect::<Result<Vec<_>>>()?;
    Ok(Grid::new(side, data)?)
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

/// `bytes` compressed as the store compresses a payload, in bytes.
///
/// # Errors
///
/// If compression fails.
fn packed(bytes: &[u8]) -> Result<u64> {
    Ok(u64::try_from(
        zstd::bulk::compress(bytes, ZSTD_LEVEL)?.len(),
    )?)
}

/// Adds what `positions` (one face's, one codec's) measure to `tally`.
///
/// # Errors
///
/// If compressing or decoding a tile fails, or a tile doesn't match its truth.
fn measure_positions(
    tally: &mut PositionTally,
    candidate: Candidate,
    base: &[Grid],
    positions: &FacePositions,
    truth: &Truth,
) -> Result<()> {
    let below = positions.finest().get() - 1;
    for l in 0..positions.finest().get() {
        let level = Level::new(l)?;
        let decoded = positions.decoded(level).context("a decoded level")?;
        let want = truth.positions.get(&l).context("positions per level")?;
        let edge = truth.edge.get(&l).copied().context("an edge per level")?;
        for key in display_keys(positions.face(), level)? {
            let bytes = positions.encoded(&key).context("a tile")?;
            *tally.bytes.entry(l).or_default() += packed(bytes)? + u64::try_from(RECORD_OVERHEAD)?;
            let error = positions.error(&key).context("a bound")?;
            tally.errors.entry(l).or_default().push(error);
            #[expect(
                clippy::cast_possible_truncation,
                reason = "a ratio for a statistic: f32 is plenty"
            )]
            let relative = (f64::from(error) / edge) as f32;
            tally.relative.entry(l).or_default().push(relative);
            if l == below {
                let decoding = Instant::now();
                std::hint::black_box(candidate.decode(base, &key, bytes)?);
                tally.decode.0 += decoding.elapsed();
                tally.decode.1 += 1;
            }
            let region = Region::of_tile(&key);
            let last = key.side() - 1;
            for j in 0..key.side() {
                for i in 0..key.side() {
                    let (x, y) = (region.x0() + i, region.y0() + j);
                    let got = decoded.get(x, y).context("inside the face")?;
                    let exact = want.get(x, y).context("inside the face")?;
                    if i == 0 || j == 0 || i == last || j == last {
                        tally.border.1 += 1;
                        if got.to_bits() != exact.to_bits() {
                            tally.border.0 += 1;
                        }
                    }
                    let off = f64::from((got - exact).length()) / edge;
                    tally.position_error = tally.position_error.max(off);
                }
            }
        }
    }
    Ok(())
}

/// Adds what `tiles` (one face's at one level, one encoding's, built from `means`) measure to
/// `tally`, timing the normals' decode if `timed`.
///
/// # Errors
///
/// If compressing or decoding a tile fails.
fn measure_normals(
    tally: &mut NormalTally,
    encoding: NormalEncoding,
    tiles: &[DisplayTile],
    means: &MeanNormals,
    timed: bool,
) -> Result<()> {
    let side = u32::try_from(means.as_slice().len().isqrt())?;
    for tile in tiles {
        let key = tile.key();
        let (head, rest) = tile.encoded().split_at_checked(4).context("a length")?;
        let length = usize::try_from(u32::from_le_bytes(head.try_into()?))?;
        let (_, normal_part) = rest.split_at_checked(length).context("the positions")?;
        tally.bytes += packed(normal_part)?;
        tally.tiles += 1;
        let rough = tile.roughness();
        let mut previous = 0_u8;
        let delta: Vec<u8> = rough
            .iter()
            .map(|&r| {
                let d = r.wrapping_sub(previous);
                previous = r;
                d
            })
            .collect();
        tally.roughness.0 += packed(rough)?;
        tally.roughness.1 += packed(&delta)?;
        if timed {
            let positions = Grid::new(key.side(), tile.positions().to_vec())?;
            let decoding = Instant::now();
            std::hint::black_box(encoding.decode(&positions, normal_part)?);
            tally.decode.0 += decoding.elapsed();
            tally.decode.1 += 1;
        }
        let region = Region::of_tile(key);
        for (index, &normal) in tile.normals().iter().enumerate() {
            let index = u32::try_from(index)?;
            let (x, y) = (
                region.x0() + index % key.side(),
                region.y0() + index / key.side(),
            );
            let mean = means
                .as_slice()
                .get(usize::try_from(y * side + x)?)
                .copied()
                .context("inside the face")?;
            let unit = mean.normalized().unwrap_or(Vec3::new(0.0, 0.0, 1.0));
            tally.angles.add(angle_degrees(normal, unit));
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

/// What the positions part keeps for the normals part: every face's positions with the chosen
/// codec, and its exact mean normals one level below the finest.
type Kept = Vec<(FacePositions, MeanNormals)>;

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
    let mesh = store.mesh().clone();
    let face_count = mesh.face_count();
    let faces = sample_faces(face_count, options.samples);
    ensure!(
        faces.len() == usize::try_from(face_count)?,
        "the normals' mip reads across faces: measure every face (--samples {face_count})"
    );
    out.say(&format!(
        "display-study: {} faces, finest level {finest}, file {:.1} MiB",
        faces.len(),
        crate::report::mib(file_bytes)
    ))?;
    let started = Instant::now();
    let (position_tallies, kept) = positions_part(store, &mesh, &faces, finest, out)?;
    drop(project);
    let scale = f64::from(face_count) / to_f64(faces.len());
    let file = to_f64(usize::try_from(file_bytes)?);
    report_positions(&position_tallies, finest, (faces.len(), scale), file, out)?;
    let chosen_bytes: u64 = position_tallies
        .iter()
        .find(|(c, _)| *c == POSITIONS)
        .map(|(_, t)| t.bytes.values().sum())
        .context("the chosen position codec was measured")?;
    normals_part(&mesh, &kept, finest, (chosen_bytes, scale, file), out)?;
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

/// The positions part: every candidate codec over `faces`, a face at a time. Keeps the chosen
/// codec's positions and the exact mean normals one level below the finest, for the normals.
///
/// # Errors
///
/// If a tile can't be evaluated, built or decoded.
fn positions_part<J: Write, H: Write>(
    store: &mut TileStore,
    mesh: &geom::BaseMesh,
    faces: &[FaceId],
    finest: Level,
    out: &mut Output<J, H>,
) -> Result<(Vec<(Candidate, PositionTally)>, Kept)> {
    let below = finest.parent().context("a level below the finest")?;
    let mut tallies: Vec<(Candidate, PositionTally)> = Candidate::ALL
        .iter()
        .map(|&c| (c, PositionTally::default()))
        .collect();
    let mut kept = Vec::with_capacity(faces.len());
    let started = Instant::now();
    for (done, &face) in faces.iter().enumerate() {
        let positions = finest_positions(store, face, finest)?;
        let base = base_grids(store, face, finest)?;
        let reference = truth(&positions, finest)?;
        let mut chosen = None;
        for (candidate, tally) in &mut tallies {
            let building = Instant::now();
            let built = candidate.build(&base, face, (&positions, finest))?;
            tally.build.0 += building.elapsed();
            tally.build.1 += 1;
            measure_positions(tally, *candidate, &base, &built, &reference)?;
            if *candidate == POSITIONS {
                chosen = Some(built);
            }
        }
        let normals = StoreNormals {
            store: RefCell::new(&mut *store),
            cache: RefCell::new(HashMap::new()),
        };
        let means = mip_normals(mesh, &normals, face, below)?;
        kept.push((chosen.context("the chosen codec built")?, means));
        out.progress("positions, faces", done + 1, faces.len(), started)?;
    }
    Ok((tallies, kept))
}

/// The normals part: every candidate encoding over every face, a level at a time, with the
/// chosen positions (`chosen_bytes` of them over the faces measured), and its report.
///
/// # Errors
///
/// If a tile can't be built or decoded, or writing the report fails.
fn normals_part<J: Write, H: Write>(
    mesh: &geom::BaseMesh,
    kept: &Kept,
    finest: Level,
    (chosen_bytes, scale, file): (u64, f64, f64),
    out: &mut Output<J, H>,
) -> Result<()> {
    out.say(&format!(
        "{:<16} {:>8} {:>8} {:>9} {:>9} {:>10} {:>11} {:>11}",
        "normals", "total", "normals", "max °", "p99 °", "decode µs", "rough raw", "rough delta"
    ))?;
    let below = finest.parent().context("a level below the finest")?;
    for encoding in ENCODINGS {
        let started = Instant::now();
        let mut tally = NormalTally {
            bytes: 0,
            tiles: 0,
            angles: Angles::new(),
            decode: (Duration::ZERO, 0),
            roughness: (0, 0),
        };
        let mut finer: Option<FaceGrids> = None;
        let mut level = finest;
        while let Some(coarser) = level.parent() {
            let mut decoded = FaceGrids::new(coarser);
            for (positions, exact) in kept {
                let face = positions.face();
                let means = match &finer {
                    None => exact.clone(),
                    Some(source) => mip_normals(mesh, source, face, coarser)?,
                };
                let (tiles, decoded_means) =
                    build_level_with(encoding, positions, coarser, &means)?;
                measure_normals(&mut tally, encoding, &tiles, &means, coarser == below)?;
                decoded.insert(face, decoded_means.into_grid())?;
            }
            finer = Some(decoded);
            level = coarser;
        }
        let bytes = |b: u64| to_f64(usize::try_from(b).unwrap_or(usize::MAX)) * scale / file;
        let overhead = tally.tiles * u64::try_from(RECORD_OVERHEAD)?;
        let total = bytes(chosen_bytes + tally.bytes);
        let normals = bytes(tally.bytes);
        let decode_us = ms(tally.decode.0) * 1000.0 / to_f64(tally.decode.1.max(1));
        let (max, p99) = (tally.angles.max, tally.angles.quantile(0.99));
        let (raw, delta) = (bytes(tally.roughness.0), bytes(tally.roughness.1));
        out.record(
            &Record::new("display-study", "normals")
                .with("encoding", encoding_name(encoding))
                .with("positions", POSITIONS.name())
                .with("growth", total)
                .with("normals_growth", normals)
                .with("record_overhead_bytes", overhead)
                .with("angle_max_degrees", max)
                .with("angle_p99_degrees", p99)
                .with("decode_us_per_tile", decode_us)
                .with("roughness_raw_growth", raw)
                .with("roughness_delta_growth", delta)
                .with("seconds", started.elapsed().as_secs_f64()),
        )?;
        out.say(&format!(
            "{:<16} {:>7.1}% {:>7.1}% {:>9.4} {:>9.4} {:>10.1} {:>10.2}% {:>10.2}%",
            encoding_name(encoding),
            total * 100.0,
            normals * 100.0,
            max,
            p99,
            decode_us,
            raw * 100.0,
            delta * 100.0
        ))?;
    }
    Ok(())
}

/// Writes each position codec's record and table row to `out`: `measured` faces, `scale` the
/// factor that projects them onto every face, `file` the project's bytes.
///
/// # Errors
///
/// If writing fails, or a count doesn't fit.
fn report_positions<J: Write, H: Write>(
    tallies: &[(Candidate, PositionTally)],
    finest: Level,
    (measured, scale): (usize, f64),
    file: f64,
    out: &mut Output<J, H>,
) -> Result<()> {
    out.say(&format!(
        "{:<15} {:>9} {:>9} {:>11} {:>8} {:>9} {:>9} {:>10}",
        "positions", "growth", "to f-2", "decode µs", "borders", "alone", "pos err", "e f-1 p50"
    ))?;
    for (candidate, tally) in tallies {
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
        let mut record = Record::new("display-study", "positions")
            .with("codec", candidate.name())
            .with("faces", measured)
            .with("growth", growth)
            .with("growth_to_finest_minus_2", growth_lower)
            .with("decode_us_per_tile", decode_us)
            .with("build_ms_per_face", build_ms)
            .with("border_mismatches", tally.border.0)
            .with("border_vertices", tally.border.1)
            .with("borders_exact", borders_exact)
            .with("standalone", candidate.standalone())
            .with("position_error_max_edges", tally.position_error);
        for (&l, errors) in &tally.errors {
            let relative = tally.relative.get(&l).map_or(&[][..], Vec::as_slice);
            record = record
                .with(format!("e_l{l}_p50"), quantile(errors, 0.5))
                .with(format!("e_l{l}_p99"), quantile(errors, 0.99))
                .with(format!("e_l{l}_max"), quantile(errors, 1.0))
                .with(format!("e_l{l}_edges_p50"), quantile(relative, 0.5))
                .with(format!("e_l{l}_edges_max"), quantile(relative, 1.0));
        }
        out.record(&record)?;
        let e_below = tally
            .errors
            .get(&(finest.get() - 1))
            .map_or(f64::NAN, |e| quantile(e, 0.5));
        out.say(&format!(
            "{:<15} {:>8.1}% {:>8.1}% {:>11.1} {:>8} {:>9} {:>9.2e} {:>10.2e}",
            candidate.name(),
            growth * 100.0,
            growth_lower * 100.0,
            decode_us,
            if borders_exact { "exact" } else { "differ" },
            if candidate.standalone() { "yes" } else { "no" },
            tally.position_error,
            e_below,
        ))?;
    }
    Ok(())
}
