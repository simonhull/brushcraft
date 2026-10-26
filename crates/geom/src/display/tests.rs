//! Display tiles: subsampling, the error a level adds, the normals' mip, every codec's round
//! trip, seams across tiles and faces, and the bound itself as a property over random sculpts.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::*;
use crate::test_rng::Rng;
use crate::{
    BaseMesh, CorrectionMap, DenseEvaluation, FaceId, Grid, LayerId, Region, cube_sphere, normals,
};

/// The finest level the tests build display tiles under: its parent, level 7, has 2×2 tiles a
/// face, so the bound chains across tile borders as well as levels.
const FINEST: u8 = 8;

/// Level `n`.
fn level(n: u8) -> Level {
    Level::new(n).unwrap()
}

/// A `side × side` grid of `f(i, j)`.
fn grid(side: u32, f: impl Fn(u32, u32) -> Vec3) -> Grid {
    let data = (0..side)
        .flat_map(|j| (0..side).map(move |i| (i, j)))
        .map(|(i, j)| f(i, j))
        .collect();
    Grid::new(side, data).unwrap()
}

/// A gently curved patch: what a smooth surface looks like to a codec.
fn patch(i: u32, j: u32) -> Vec3 {
    let (x, y) = (
        f32::from(u16::try_from(i).unwrap()),
        f32::from(u16::try_from(j).unwrap()),
    );
    Vec3::new(
        0.3 + x * 0.01,
        -0.2 + y * 0.01,
        0.8 + (x * 0.07).sin() * 0.002 + (y * 0.05).cos() * 0.003,
    )
}

/// A codec's encode then decode of a tile: its bytes, and the positions they decode to.
type RoundTrip = Box<dyn Fn(&TileKey, &Grid) -> (Vec<u8>, Grid)>;

/// Every codec the study measures, without [`BaseOffsets`] (which needs a face's base grids),
/// with its name for messages.
fn standalone_codecs() -> Vec<(&'static str, RoundTrip)> {
    /// One codec's encode and decode, boxed.
    fn boxed(codec: impl DisplayCodec + 'static) -> RoundTrip {
        Box::new(move |key, positions| {
            let bytes = codec.encode(key, positions).unwrap();
            let decoded = codec.decode(key, &bytes).unwrap();
            (bytes, decoded)
        })
    }
    vec![
        ("lossless", boxed(Interpolating::new(Precision::Lossless))),
        ("bits16", boxed(Interpolating::new(Precision::Bits16))),
        (
            "quad/1024",
            boxed(Interpolating::new(Precision::QuadFraction(10))),
        ),
        (
            "quad/64",
            boxed(Interpolating::new(Precision::QuadFraction(6))),
        ),
        ("fixed", boxed(FixedPoint)),
    ]
}

/// The border vertices of a tile with `side` vertices per side.
fn border(side: u32) -> impl Iterator<Item = (u32, u32)> {
    let last = side - 1;
    (0..side)
        .flat_map(move |j| (0..side).map(move |i| (i, j)))
        .filter(move |&(i, j)| i == 0 || j == 0 || i == last || j == last)
}

#[test]
fn subsampling_keeps_the_even_vertices_bit_for_bit() {
    let fine = grid(9, patch);
    let coarse = subsample(&fine).unwrap();
    assert_eq!(coarse.side(), 5, "8 quads halve to 4");
    for j in 0..5 {
        for i in 0..5 {
            assert_eq!(
                coarse.get(i, j).unwrap().to_bits(),
                fine.get(2 * i, 2 * j).unwrap().to_bits(),
                "({i}, {j})"
            );
        }
    }
    assert_eq!(
        subsample(&grid(2, patch)),
        Err(DisplayError::Shape),
        "one quad has nothing to halve"
    );
    assert_eq!(
        subsample(&grid(4, patch)),
        Err(DisplayError::Shape),
        "three quads don't halve"
    );
}

/// A finer grid lying on the coarser one's triangles (each odd vertex at the midpoint the
/// triangulation draws there) adds no error; moving one vertex of either kind by `h` adds `h`.
#[test]
fn step_error_is_the_largest_distance_to_the_triangulated_coarser_surface() {
    let coarse = grid(3, |i, j| patch(i * 8, j * 8));
    let on_triangles = grid(5, |i, j| {
        let at = |a: u32, b: u32| coarse.get(a, b).unwrap();
        let (ci, cj) = (i / 2, j / 2);
        match (i % 2, j % 2) {
            (0, 0) => at(ci, cj),
            (1, 0) => (at(ci, cj) + at(ci + 1, cj)) * 0.5,
            (0, _) => (at(ci, cj) + at(ci, cj + 1)) * 0.5,
            _ => (at(ci, cj) + at(ci + 1, cj + 1)) * 0.5,
        }
    });
    let flat = step_error(&on_triangles, &coarse).unwrap();
    assert!(flat < 1e-7, "on the triangles: {flat}");
    let h = 0.001_f32;
    // A quad centre, an edge vertex, and an even vertex (where the coarser grid draws its own
    // vertex), each moved by `h`.
    for (mi, mj) in [(1, 1), (3, 2), (2, 2)] {
        let moved = grid(5, |i, j| {
            let p = on_triangles.get(i, j).unwrap();
            if (i, j) == (mi, mj) {
                p + Vec3::new(0.0, 0.0, h)
            } else {
                p
            }
        });
        let delta = step_error(&moved, &coarse).unwrap();
        assert!(
            (delta - h).abs() < 1e-6,
            "({mi}, {mj}) moved by {h}: {delta}"
        );
    }
    // The diagonal is the `(i, j)`–`(i + 1, j + 1)` one: a quad centre on the other diagonal's
    // midpoint is off the drawn surface where the quad isn't planar.
    let twisted = grid(2, |i, j| {
        Vec3::new(
            f32::from(u8::try_from(i).unwrap()),
            f32::from(u8::try_from(j).unwrap()),
            if i == j { 0.0 } else { 1.0 },
        )
    });
    let other_diagonal = grid(3, |i, j| {
        if (i, j) == (1, 1) {
            Vec3::new(0.5, 0.5, 1.0)
        } else {
            Vec3::new(
                f32::from(u8::try_from(i).unwrap()) * 0.5,
                f32::from(u8::try_from(j).unwrap()) * 0.5,
                0.0,
            )
        }
    });
    let centre_gap = step_error(&other_diagonal, &twisted).unwrap();
    assert!(
        centre_gap >= 1.0,
        "the centre is drawn at the (0,0)–(1,1) midpoint: {centre_gap}"
    );
    assert_eq!(
        step_error(&grid(4, patch), &coarse),
        Err(DisplayError::Shape),
        "sides that don't match"
    );
}

/// Unit normals that all agree mip to unit means at every level, face border included (where
/// the tent is cut off and renormalized); normals that alternate shorten the mean.
#[test]
fn the_normals_mip_keeps_agreement_and_shortens_disagreement() {
    let up = Vec3::new(0.0, 0.0, 1.0);
    let mut means = MeanNormals::from_unit(grid(17, |_, _| up));
    while means.side() > 2 {
        means = mip_normals(&means).unwrap();
        for m in means.as_slice() {
            assert!(
                (m.length() - 1.0).abs() < 1e-6,
                "agreeing normals stay unit: {m:?}"
            );
        }
    }
    let tilted = |i: u32| {
        if i.is_multiple_of(2) {
            Vec3::new(0.6, 0.0, 0.8)
        } else {
            Vec3::new(-0.6, 0.0, 0.8)
        }
    };
    let rough = mip_normals(&MeanNormals::from_unit(grid(9, |i, _| tilted(i)))).unwrap();
    for m in rough.as_slice() {
        assert!(
            m.length() < 0.95,
            "alternating normals average shorter: {m:?}"
        );
        assert!(m.z > 0.79, "but still point up: {m:?}");
    }
}

/// Every standalone codec decodes what it encoded to the same bits each time, with exact borders.
///
/// Each also stays within its quantization of the input (the lossless one exact everywhere),
/// and decoding rejects truncated or padded bytes.
#[test]
fn codecs_round_trip_with_exact_borders() {
    let key = TileKey::display(FaceId::new(0), level(7), 1, 0).unwrap();
    let positions = grid(key.side(), patch);
    for (name, codec) in standalone_codecs() {
        let (bytes, decoded) = codec(&key, &positions);
        let (again, decoded_again) = codec(&key, &positions);
        assert_eq!(bytes, again, "{name}: encoding is deterministic");
        assert_eq!(decoded, decoded_again, "{name}: so is decoding");
        for (i, j) in border(key.side()) {
            assert_eq!(
                decoded.get(i, j).unwrap().to_bits(),
                positions.get(i, j).unwrap().to_bits(),
                "{name}: border ({i}, {j}) exact"
            );
        }
        let worst = decoded
            .as_slice()
            .iter()
            .zip(positions.as_slice())
            .map(|(d, p)| (*d - *p).length())
            .fold(0.0_f32, f32::max);
        if name == "lossless" {
            assert_eq!(worst, 0.0, "lossless is exact");
        } else {
            // Quad edges here are 0.01: every precision is well inside a 64th of one.
            assert!(worst < 0.01 / 64.0, "{name}: off by {worst}");
        }
    }
    let codec = Interpolating::new(Precision::QuadFraction(10));
    let bytes = codec.encode(&key, &positions).unwrap();
    let short = &bytes[..bytes.len() - 1];
    assert_eq!(
        codec.decode(&key, short),
        Err(DisplayError::Malformed),
        "short"
    );
    let mut long = bytes.clone();
    long.push(0);
    assert_eq!(
        codec.decode(&key, &long),
        Err(DisplayError::Malformed),
        "long"
    );
    let mut bad_mode = bytes;
    *bad_mode.first_mut().unwrap() = 9;
    assert_eq!(
        codec.decode(&key, &bad_mode),
        Err(DisplayError::Malformed),
        "mode"
    );
    // Precisions finer than an `f32` can step fall back to the finest step that fits.
    let finest_step = Interpolating::new(Precision::QuadFraction(u8::MAX));
    let finest_bytes = finest_step.encode(&key, &positions).unwrap();
    let back = finest_step.decode(&key, &finest_bytes).unwrap();
    let worst = back
        .as_slice()
        .iter()
        .zip(positions.as_slice())
        .map(|(d, p)| (*d - *p).length())
        .fold(0.0_f32, f32::max);
    assert!(worst < 1e-6, "2^-255 of a quad: {worst}");
    let wrong_side = grid(17, patch);
    assert_eq!(
        codec.encode(&key, &wrong_side),
        Err(DisplayError::Shape),
        "side"
    );
}

/// The angle between `a` and `b`, in degrees, from `atan2(|a × b|, a · b)` in `f64` (an
/// `acos` of the dot product loses small angles to rounding).
pub(crate) fn angle_degrees(a: Vec3, b: Vec3) -> f64 {
    let wide = |v: Vec3| [f64::from(v.x), f64::from(v.y), f64::from(v.z)];
    let ([ax, ay, az], [bx, by, bz]) = (wide(a), wide(b));
    let cross = [ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx];
    let sin = cross.iter().map(|c| c * c).sum::<f64>().sqrt();
    sin.atan2(ax * bx + ay * by + az * bz).to_degrees()
}

/// Octahedral normals come back within their precision's angle, roughness exactly.
#[test]
fn normals_round_trip_within_their_precision() {
    let mut rng = Rng::new(7);
    let normals: Vec<Vec3> = (0..2000)
        .map(|_| {
            rng.vec3(1.0)
                .normalized()
                .unwrap_or(Vec3::new(0.0, 0.0, -1.0))
        })
        .collect();
    let roughness: Vec<u8> = (0..2000).map(|n| u8::try_from(n % 256).unwrap()).collect();
    for (encoding, limit_degrees) in [
        (NormalEncoding::Oct16, 0.01_f64),
        (NormalEncoding::Oct8, 1.5),
    ] {
        let bytes = encoding.encode(&normals, &roughness).unwrap();
        let (back, rough) = encoding.decode(normals.len(), &bytes).unwrap();
        assert_eq!(rough, roughness, "{encoding:?}: roughness exact");
        let worst = normals
            .iter()
            .zip(&back)
            .map(|(a, b)| angle_degrees(*a, *b))
            .fold(0.0_f64, f64::max);
        assert!(worst < limit_degrees, "{encoding:?}: {worst}°");
        assert_eq!(
            encoding.decode(normals.len() + 1, &bytes).map(|_| ()),
            Err(DisplayError::Malformed),
            "{encoding:?}: too few bytes"
        );
    }
}

/// A sculpt: the mesh, its evaluation to [`FINEST`] with the corrections, and a face's finest
/// grid and normals.
struct Sculpt {
    /// The mesh.
    mesh: BaseMesh,
    /// Every level, every face.
    dense: DenseEvaluation,
}

impl Sculpt {
    /// `cube_sphere(1)` with random corrections from `seed`: single vertices at random levels
    /// with amplitudes from a micron to a fifth of the sphere's radius, and smooth bumps at the
    /// finest level, as brush strokes leave.
    fn random(seed: u64) -> Self {
        let mesh = cube_sphere(1).unwrap();
        let mut rng = Rng::new(seed);
        let mut map = CorrectionMap::new();
        let faces = u32::try_from(mesh.faces().count()).unwrap();
        for _ in 0..rng.below(40) {
            let l = level(1 + u8::try_from(rng.below(u32::from(FINEST))).unwrap());
            let n = l.face_quads();
            let amplitude = 10_f32.powf(-6.0 + 5.3 * rng.unit());
            let (face, i, j) = (
                FaceId::new(rng.below(faces)),
                rng.below(n + 1),
                rng.below(n + 1),
            );
            map.set(mesh.topology(), face, l, i, j, rng.vec3(amplitude))
                .unwrap();
        }
        for _ in 0..rng.below(4) {
            let l = level(FINEST);
            let n = l.face_quads();
            let face = FaceId::new(rng.below(faces));
            let (ci, cj) = (rng.below(n + 1), rng.below(n + 1));
            let radius = 2 + rng.below(40);
            let height = 0.02 * rng.signed();
            for j in cj.saturating_sub(radius)..=(cj + radius).min(n) {
                for i in ci.saturating_sub(radius)..=(ci + radius).min(n) {
                    let d = f32::from(u16::try_from(i.abs_diff(ci).max(j.abs_diff(cj))).unwrap())
                        / f32::from(u16::try_from(radius).unwrap());
                    let falloff = (1.0 - d * d).max(0.0);
                    let c = map.get(face, l, i, j) + Vec3::new(0.0, 0.0, height * falloff);
                    map.set(mesh.topology(), face, l, i, j, c).unwrap();
                }
            }
        }
        let dense = DenseEvaluation::new(&mesh, level(FINEST), &map).unwrap();
        Self { mesh, dense }
    }

    /// `face`'s finest positions and unit normals.
    fn finest(&self, face: FaceId) -> (Grid, Grid) {
        let positions = self.dense.grid(face, level(FINEST)).unwrap().clone();
        let unit = normals(
            &self.mesh,
            &self.dense,
            Region::whole_face(face, level(FINEST)),
        )
        .unwrap();
        (positions, unit)
    }

    /// `face`'s base grids at every level below the finest, for [`BaseOffsets`].
    fn base(&self, face: FaceId) -> Vec<Grid> {
        (0..FINEST)
            .map(|l| self.dense.grid(face, level(l)).unwrap().clone())
            .collect()
    }
}

/// `face`'s display tiles built with every codec, by codec name.
fn every_build(sculpt: &Sculpt, face: FaceId) -> Vec<(&'static str, Vec<DisplayTile>)> {
    let (positions, unit) = sculpt.finest(face);
    let base = sculpt.base(face);
    let with = |codec: &dyn ErasedCodec| codec.build(face, &positions, &unit);
    vec![
        ("lossless", with(&Interpolating::new(Precision::Lossless))),
        ("bits16", with(&Interpolating::new(Precision::Bits16))),
        (
            "quad/1024",
            with(&Interpolating::new(Precision::QuadFraction(10))),
        ),
        (
            "quad/64",
            with(&Interpolating::new(Precision::QuadFraction(6))),
        ),
        ("fixed", with(&FixedPoint)),
        ("base offsets", with(&BaseOffsets::new(&base))),
    ]
}

/// [`build_face_with`] behind a trait object, so the tests can list codecs of different types.
trait ErasedCodec {
    /// `face`'s display tiles with this codec and 16-bit normals.
    fn build(&self, face: FaceId, positions: &Grid, unit: &Grid) -> Vec<DisplayTile>;
}

impl<C: DisplayCodec> ErasedCodec for C {
    fn build(&self, face: FaceId, positions: &Grid, unit: &Grid) -> Vec<DisplayTile> {
        build_face_with(
            self,
            NormalEncoding::Oct16,
            face,
            positions,
            unit,
            level(FINEST),
        )
        .unwrap()
    }
}

/// The surface tile `tile` draws at the point `at` of its own grid (in quads, fractional).
///
/// That is, in `f64`, its decoded positions on the two triangles of the quad holding the point,
/// split along the `(i, j)`–`(i + 1, j + 1)` diagonal.
fn drawn(tile: &DisplayTile, (along, up): (f64, f64)) -> [f64; 3] {
    let side = tile.key().side();
    let quads = f64::from(side - 1);
    let qi = along.floor().min(quads - 1.0);
    let qj = up.floor().min(quads - 1.0);
    let (fu, fv) = (along - qi, up - qj);
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a quad index inside the tile"
    )]
    let (qi, qj) = (qi as u32, qj as u32);
    let at = |i: u32, j: u32| {
        let p = *tile
            .positions()
            .get(usize::try_from(j * side + i).unwrap())
            .unwrap();
        [f64::from(p.x), f64::from(p.y), f64::from(p.z)]
    };
    let (p00, p10, p01, p11) = (
        at(qi, qj),
        at(qi + 1, qj),
        at(qi, qj + 1),
        at(qi + 1, qj + 1),
    );
    let mut out = [0.0; 3];
    for (k, o) in out.iter_mut().enumerate() {
        let (c00, c10, c01, c11) = (p00[k], p10[k], p01[k], p11[k]);
        *o = if fu >= fv {
            c00 + fu * (c10 - c00) + fv * (c11 - c10)
        } else {
            c00 + fu * (c11 - c01) + fv * (c01 - c00)
        };
    }
    out
}

/// Checks the bound for every tile of `tiles` (one face's, from one codec), and returns how many
/// vertex checks ran.
///
/// Every finest vertex under a tile lies within its `e` of the surface it draws, `e` never grows
/// toward finer levels, and the finest level's parents' `e` is their own step's error alone (the
/// finest level is exact).
fn check_bound(name: &str, tiles: &[DisplayTile], finest: &Grid) -> usize {
    let by_key: BTreeMap<TileKey, &DisplayTile> = tiles.iter().map(|t| (*t.key(), t)).collect();
    let mut checks = 0;
    for tile in tiles {
        let key = tile.key();
        let shift = FINEST - key.level().get();
        let scale = f64::from(1_u32 << shift);
        let region = Region::of_tile(key);
        let bound = f64::from(tile.error());
        // The test's own `f64` rounding, far below an `f32` step of `e`.
        let allowed = bound + bound * 1e-12;
        for y in (region.y0() << shift)..=((region.y0() + region.quads()) << shift) {
            for x in (region.x0() << shift)..=((region.x0() + region.quads()) << shift) {
                let point = finest.get(x, y).unwrap();
                let at = (
                    f64::from(x) / scale - f64::from(region.x0()),
                    f64::from(y) / scale - f64::from(region.y0()),
                );
                let surface = drawn(tile, at);
                let gap = [
                    f64::from(point.x) - surface[0],
                    f64::from(point.y) - surface[1],
                    f64::from(point.z) - surface[2],
                ];
                let dist = gap.iter().map(|g| g * g).sum::<f64>().sqrt();
                assert!(
                    dist <= allowed,
                    "{name}: {key:?} finest ({x}, {y}) is {dist} from the drawn surface, e = {bound}"
                );
                checks += 1;
            }
        }
        if let Some(finer) = key.level().child().filter(|l| l.get() < FINEST) {
            let ratio = finer.tiles_per_side() / key.level().tiles_per_side();
            for dy in 0..ratio {
                for dx in 0..ratio {
                    let tx = u16::try_from(u32::from(key.tx()) * ratio + dx).unwrap();
                    let ty = u16::try_from(u32::from(key.ty()) * ratio + dy).unwrap();
                    let child = by_key[&TileKey::display(key.face(), finer, tx, ty).unwrap()];
                    assert!(
                        tile.error() >= child.error(),
                        "{name}: e grows from {key:?} to {:?}",
                        child.key()
                    );
                }
            }
        } else {
            let grid = Grid::new(key.side(), tile.positions().to_vec()).unwrap();
            let own = step_error_at_tile(finest, &region, &grid);
            assert_eq!(
                tile.error(),
                own,
                "{name}: {key:?} above the finest is its step alone"
            );
        }
    }
    checks
}

/// [`step_error`] of the finest grid's block under `region` against `coarser`.
fn step_error_at_tile(finest: &Grid, region: &Region, coarser: &Grid) -> f32 {
    let side = 2 * region.quads() + 1;
    let block = grid(side, |i, j| {
        finest
            .get(2 * region.x0() + i, 2 * region.y0() + j)
            .unwrap()
    });
    step_error(&block, coarser).unwrap()
}

/// Bolero cases `the_bound_holds_for_random_sculpts` runs.
const BOUND_CASES: usize = 64;

/// Cases run, counted so the test can say how many it checked.
static BOUND_RUN: AtomicUsize = AtomicUsize::new(0);

/// **The bound**: every finest vertex lies within `e` of the surface each display tile over it
/// draws, at every level, for random sculpts and every codec.
///
/// The sculpts are random corrections at random levels, and smooth bumps at the finest, on a
/// random face. Also: `e` never grows toward finer levels, and the finest level's parents carry
/// their own step's error alone.
#[test]
fn the_bound_holds_for_random_sculpts() {
    bolero::check!()
        .with_type::<(u64, u8)>()
        .with_iterations(BOUND_CASES)
        .for_each(|&(seed, face)| {
            let sculpt = Sculpt::random(seed);
            let face = FaceId::new(u32::from(face % 6));
            let (finest, _) = sculpt.finest(face);
            for (name, tiles) in every_build(&sculpt, face) {
                let checks = check_bound(name, &tiles, &finest);
                assert!(checks > 0, "{name}: something was checked");
            }
            BOUND_RUN.fetch_add(1, Ordering::Relaxed);
        });
    assert!(BOUND_RUN.load(Ordering::Relaxed) >= 1, "cases ran");
}

/// The chosen encoding and [`FixedPoint`] decode every vertex two display tiles share to the
/// same bits in both.
///
/// That holds across tile borders and across faces: seams are exact without reading a
/// neighbour.
#[test]
fn shared_vertices_decode_identically_across_tiles_and_faces() {
    let sculpt = Sculpt::random(11);
    let faces: Vec<FaceId> = sculpt.mesh.faces().collect();
    let builds: [(&str, &dyn ErasedCodec); 2] =
        [("chosen", &encode::CHOSEN), ("fixed", &FixedPoint)];
    for (name, codec) in builds {
        let mut tiles: BTreeMap<TileKey, DisplayTile> = BTreeMap::new();
        for &face in &faces {
            let (positions, unit) = sculpt.finest(face);
            for tile in codec.build(face, &positions, &unit) {
                tiles.insert(*tile.key(), tile);
            }
        }
        let mut shared = 0;
        for tile in tiles.values() {
            let key = tile.key();
            let region = Region::of_tile(key);
            for (i, j) in border(key.side()) {
                let mine = tile.positions()[usize::try_from(j * key.side() + i).unwrap()];
                let copies = sculpt.mesh.topology().copies(
                    key.face(),
                    key.level(),
                    region.x0() + i,
                    region.y0() + j,
                );
                for copy in &copies {
                    let other = &tiles[&copy.key_in(LayerId::BASE, Channel::DISPLAY, key.level())];
                    let side = other.key().side();
                    let theirs =
                        other.positions()[usize::try_from(copy.j * side + copy.i).unwrap()];
                    assert_eq!(
                        mine.to_bits(),
                        theirs.to_bits(),
                        "{name}: {key:?} ({i}, {j}) against {:?}",
                        other.key()
                    );
                    shared += 1;
                }
            }
        }
        assert!(shared > 0, "{name}: shared vertices were compared");
    }
}

/// A tile [`build_face`] built reads back from its bytes bit for bit, and its bytes say the same
/// again: the store can keep only the bytes and the bound.
#[test]
fn a_built_tile_reads_back_from_its_bytes() {
    let sculpt = Sculpt::random(3);
    let face = FaceId::new(2);
    let (positions, unit) = sculpt.finest(face);
    let tiles = build_face(face, &positions, &unit, level(FINEST)).unwrap();
    assert_eq!(
        tiles.len(),
        usize::from(FINEST - 1) + 4,
        "levels 0–6 a tile each, level 7 four"
    );
    for tile in &tiles {
        let back = DisplayTile::decode(*tile.key(), tile.error(), tile.encoded()).unwrap();
        assert_eq!(&back, tile, "{:?}", tile.key());
    }
    let first = tiles.first().unwrap();
    assert_eq!(
        DisplayTile::decode(*first.key(), f32::NAN, first.encoded()).map(|_| ()),
        Err(DisplayError::Malformed),
        "a bound that isn't one"
    );
    let displacement = TileKey::displacement(face, first.key().level(), 0, 0).unwrap();
    assert_eq!(
        DisplayTile::decode(displacement, first.error(), first.encoded()).map(|_| ()),
        Err(DisplayError::NotDisplay(displacement)),
        "a key of another channel"
    );
    assert_eq!(
        build_face(face, &unit, &positions, level(FINEST - 1)).map(|_| ()),
        Err(DisplayError::Shape),
        "grids of the wrong level"
    );
}

/// A codec that breaks the shared-vertex contract: the chosen one, but every tile in an odd
/// column decodes its corner `(0, 0)` a step off, so it disagrees with its left neighbour.
struct BrokenSeams;

impl DisplayCodec for BrokenSeams {
    fn encode(&self, key: &TileKey, positions: &Grid) -> Result<Vec<u8>, DisplayError> {
        encode::CHOSEN.encode(key, positions)
    }

    fn decode(&self, key: &TileKey, bytes: &[u8]) -> Result<Grid, DisplayError> {
        let grid = encode::CHOSEN.decode(key, bytes)?;
        if key.tx().is_multiple_of(2) {
            return Ok(grid);
        }
        let mut data = grid.as_slice().to_vec();
        data[0].x = data[0].x.next_up();
        Ok(Grid::new(grid.side(), data)?)
    }
}

/// A codec that decodes a shared vertex to different bits in two tiles is refused, naming the
/// second tile, rather than built with a bound that covers one copy only.
#[test]
fn a_codec_breaking_seams_is_refused() {
    let sculpt = Sculpt::random(5);
    let face = FaceId::new(1);
    let (positions, unit) = sculpt.finest(face);
    let built = build_face_with(
        &BrokenSeams,
        NormalEncoding::Oct16,
        face,
        &positions,
        &unit,
        level(FINEST),
    );
    assert_eq!(
        built.map(|_| ()),
        Err(DisplayError::Seam(
            TileKey::display(face, level(FINEST - 1), 1, 0).unwrap()
        )),
        "the level-7 tile in column 1 disagrees with column 0"
    );
}

/// A parent's bound takes the largest of its own children's, wherever the parent lies.
///
/// At level 7 under a level-9 finest, each tile has four level-8 children, at twice its column
/// and row. (The bound property's finest level, 8, gives level 7 only zero-bound children.)
#[test]
fn a_parent_takes_its_own_childrens_largest_bound() {
    // Level 8 has 4×4 tiles: each one's bound is its row-major index, so every parent's
    // children have a distinct largest bound.
    let finer: Vec<f32> = (0..16_u8).map(f32::from).collect();
    let cases = [((0, 0), 5.0), ((1, 0), 7.0), ((0, 1), 13.0), ((1, 1), 15.0)];
    for ((tx, ty), want) in cases {
        let key = TileKey::display(FaceId::new(0), level(7), tx, ty).unwrap();
        assert_eq!(
            build::children_bound(&key, level(8), level(9), &finer),
            want,
            "({tx}, {ty})"
        );
    }
    let whole = TileKey::display(FaceId::new(0), level(6), 0, 0).unwrap();
    assert_eq!(
        build::children_bound(&whole, level(7), level(9), &[1.0, 4.0, 2.0, 3.0]),
        4.0,
        "a whole-face parent over 2×2 children"
    );
    let below_finest = TileKey::display(FaceId::new(0), level(8), 0, 0).unwrap();
    assert_eq!(
        build::children_bound(&below_finest, level(9), level(9), &finer),
        0.0,
        "the finest level is exact"
    );
}

/// The bound survives `f64` rounding at extreme magnitudes.
///
/// An edge from `x = 1` to `x = 10⁻³⁰` has its exact midpoint `10⁻³⁰ / 2` above `0.5`, which
/// `f64` rounds to `0.5` exactly; a finer vertex at `0.5` is that far off, and the step error
/// must not say 0.
#[test]
fn step_error_bounds_what_f64_rounding_hides() {
    let corner = |i: u32, j: u32| {
        if (i, j) == (1, 0) {
            Vec3::new(1e-30, 0.0, 0.0)
        } else {
            Vec3::new(1.0, 0.0, 0.0)
        }
    };
    let coarse = grid(2, corner);
    let finer = grid(3, |i, j| match (i, j) {
        (1, 0) | (2, 1) => Vec3::new(0.5, 0.0, 0.0),
        _ if i % 2 == 0 && j % 2 == 0 => corner(i / 2, j / 2),
        _ => Vec3::new(1.0, 0.0, 0.0),
    });
    let delta = step_error(&finer, &coarse).unwrap();
    assert!(
        f64::from(delta) >= 0.5e-30,
        "true distance 5e-31, got {delta}"
    );
    assert_eq!(
        step_error(&grid(5, patch), &coarse),
        Err(DisplayError::Shape),
        "a finer grid larger than the coarser one's refinement"
    );
}
