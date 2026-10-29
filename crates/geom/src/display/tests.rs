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

/// Unit normals that all agree mip to unit means, at every vertex, face corners of valence 3
/// included (the tent's weights sum to 1 at any valence); normals that alternate shorten the
/// mean.
#[test]
fn the_normals_mip_keeps_agreement_and_shortens_disagreement() {
    let mesh = cube_sphere(1).unwrap();
    let up = Vec3::new(0.0, 0.0, 1.0);
    let all = |f: &dyn Fn(u32, u32) -> Vec3| {
        let mut grids = FaceGrids::new(level(4));
        for face in mesh.faces() {
            grids.insert(face, grid(17, f)).unwrap();
        }
        grids
    };
    let agreeing = all(&|_, _| up);
    for face in mesh.faces() {
        let means = mip_normals(&mesh, &agreeing, face, level(3)).unwrap();
        for m in means.as_slice() {
            assert!(
                (m.length() - 1.0).abs() < 1e-6,
                "agreeing normals stay unit: {m:?}"
            );
        }
    }
    let tilted = |i: u32, _: u32| {
        if i.is_multiple_of(2) {
            Vec3::new(0.6, 0.0, 0.8)
        } else {
            Vec3::new(-0.6, 0.0, 0.8)
        }
    };
    let alternating = all(&tilted);
    let means = mip_normals(&mesh, &alternating, FaceId::new(0), level(3)).unwrap();
    for j in 1..8 {
        for i in 1..8 {
            let m = means.as_slice()[usize::try_from(j * 9 + i).unwrap()];
            assert!(
                m.length() < 0.95,
                "alternating normals average shorter: {m:?}"
            );
            assert!(m.z > 0.79, "but still point up: {m:?}");
        }
    }
    assert_eq!(
        mip_normals(&mesh, &FaceGrids::new(level(4)), FaceId::new(0), level(3)).map(|_| ()),
        Err(DisplayError::Unavailable),
        "no means one level finer"
    );
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

/// Every normal encoding comes back within its precision's angle, roughness exactly, and
/// rejects bytes for another tile size.
///
/// Octahedral normals are random; predicted ones lie within 5° of the tile's own surface
/// normal, as averaged normals do (a tile whose normals stray past 60° falls back to octahedral,
/// checked with the random normals).
#[test]
fn normals_round_trip_within_their_precision() {
    let positions = grid(45, patch);
    let count = positions.as_slice().len();
    let mut rng = Rng::new(7);
    let random: Vec<Vec3> = (0..count)
        .map(|_| {
            rng.vec3(1.0)
                .normalized()
                .unwrap_or(Vec3::new(0.0, 0.0, -1.0))
        })
        .collect();
    let surface = |i: u32, j: u32| {
        let at = |x: u32, y: u32| positions.get(x.min(44), y.min(44)).unwrap();
        let du = at(i + 1, j) - at(i.saturating_sub(1), j);
        let dv = at(i, j + 1) - at(i, j.saturating_sub(1));
        du.cross(dv).normalized().unwrap()
    };
    let near: Vec<Vec3> = (0..45)
        .flat_map(|j| (0..45).map(move |i| (i, j)))
        .map(|(i, j)| (surface(i, j) + rng.vec3(0.04)).normalized().unwrap())
        .collect();
    let roughness: Vec<u8> = (0..count).map(|n| u8::try_from(n % 256).unwrap()).collect();
    let cases = [
        (NormalEncoding::Octahedral(6), &random, 5.0_f64),
        (NormalEncoding::Octahedral(8), &random, 1.3),
        (NormalEncoding::Octahedral(10), &random, 0.33),
        (NormalEncoding::Octahedral(12), &random, 0.085),
        (NormalEncoding::Octahedral(16), &random, 0.006),
        (NormalEncoding::Predicted(8), &near, 0.1),
        (NormalEncoding::Predicted(10), &near, 0.085),
        (NormalEncoding::Predicted(12), &near, 0.085),
        (NormalEncoding::Predicted(10), &random, 0.085),
    ];
    for (encoding, normals, limit_degrees) in cases {
        let bytes = encoding.encode(normals, &roughness, &positions).unwrap();
        let (back, rough) = encoding.decode(&positions, &bytes).unwrap();
        assert_eq!(rough, roughness, "{encoding:?}: roughness exact");
        let worst = normals
            .iter()
            .zip(&back)
            .map(|(a, b)| angle_degrees(*a, *b))
            .fold(0.0_f64, f64::max);
        assert!(worst < limit_degrees, "{encoding:?}: {worst}°");
        assert_eq!(
            encoding.decode(&grid(46, patch), &bytes).map(|_| ()),
            Err(DisplayError::Malformed),
            "{encoding:?}: bytes for a smaller tile"
        );
    }
    assert_eq!(
        NormalEncoding::Octahedral(17)
            .encode(&random, &roughness, &positions)
            .map(|_| ()),
        Err(DisplayError::Shape),
        "more bits than an i16"
    );
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

    /// `cube_sphere(1)` without corrections: the smooth limit surface.
    fn smooth() -> Self {
        let mesh = cube_sphere(1).unwrap();
        let dense = DenseEvaluation::new(&mesh, level(FINEST), &CorrectionMap::new()).unwrap();
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

    /// Every face's finest positions and unit normals.
    fn finest_grids(&self) -> (FaceGrids, FaceGrids) {
        let (mut positions, mut normals) =
            (FaceGrids::new(level(FINEST)), FaceGrids::new(level(FINEST)));
        for face in self.mesh.faces() {
            let (p, n) = self.finest(face);
            positions.insert(face, p).unwrap();
            normals.insert(face, n).unwrap();
        }
        (positions, normals)
    }

    /// `face`'s base grids at every level below the finest, for [`BaseOffsets`].
    fn base(&self, face: FaceId) -> Vec<Grid> {
        (0..FINEST)
            .map(|l| self.dense.grid(face, level(l)).unwrap().clone())
            .collect()
    }
}

/// `face`'s display positions built with every codec, by codec name.
fn every_build(sculpt: &Sculpt, face: FaceId) -> Vec<(&'static str, FacePositions)> {
    let (positions, _) = sculpt.finest(face);
    let base = sculpt.base(face);
    let with = |codec: &dyn ErasedCodec| codec.positions(face, &positions);
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

/// [`build_positions_with`] behind a trait object, so the tests can list codecs of different
/// types.
trait ErasedCodec {
    /// `face`'s display positions with this codec.
    fn positions(&self, face: FaceId, positions: &Grid) -> FacePositions;
}

impl<C: DisplayCodec> ErasedCodec for C {
    fn positions(&self, face: FaceId, positions: &Grid) -> FacePositions {
        build_positions_with(self, face, positions, level(FINEST)).unwrap()
    }
}

/// The surface a tile with decoded positions `tile` draws at the point `at` of its own grid (in
/// quads, fractional).
///
/// That is, in `f64`, its decoded positions on the two triangles of the quad holding the point,
/// split along the `(i, j)`–`(i + 1, j + 1)` diagonal.
fn drawn(tile: &Grid, (along, up): (f64, f64)) -> [f64; 3] {
    let side = tile.side();
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
        let p = tile.get(i, j).unwrap();
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

/// Checks the bound for every display tile of a face's `positions` (from one codec), and
/// returns how many vertex checks ran.
///
/// Every finest vertex under a tile lies within its `e` of the surface it draws, `e` never grows
/// toward finer levels, and the finest level's parents' `e` is their own step's error alone (the
/// finest level is exact).
fn check_bound(name: &str, positions: &FacePositions, finest: &Grid) -> usize {
    let mut checks = 0;
    for l in 0..FINEST {
        let decoded = positions.decoded(level(l)).unwrap();
        let per_side = u16::try_from(level(l).tiles_per_side()).unwrap();
        for (tx, ty) in (0..per_side).flat_map(|ty| (0..per_side).map(move |tx| (tx, ty))) {
            let key = TileKey::display(positions.face(), level(l), tx, ty).unwrap();
            let region = Region::of_tile(&key);
            let tile = grid(key.side(), |i, j| {
                decoded.get(region.x0() + i, region.y0() + j).unwrap()
            });
            let error = positions.error(&key).unwrap();
            checks += check_tile(name, &key, &tile, error, finest);
            if let Some(finer) = key.level().child().filter(|f| f.get() < FINEST) {
                let ratio = finer.tiles_per_side() / key.level().tiles_per_side();
                for dy in 0..ratio {
                    for dx in 0..ratio {
                        let cx = u16::try_from(u32::from(tx) * ratio + dx).unwrap();
                        let cy = u16::try_from(u32::from(ty) * ratio + dy).unwrap();
                        let child = TileKey::display(key.face(), finer, cx, cy).unwrap();
                        let below = positions.error(&child).unwrap();
                        assert!(error >= below, "{name}: e grows from {key:?} to {child:?}");
                    }
                }
            } else {
                let own = step_error_at_tile(finest, &region, &tile);
                assert_eq!(
                    error, own,
                    "{name}: {key:?} above the finest is its step alone"
                );
            }
        }
    }
    checks
}

/// Checks that every finest vertex under tile `key` (decoded positions `tile`, bound `error`)
/// lies within `error` of the surface it draws; returns how many vertices it checked.
fn check_tile(name: &str, key: &TileKey, tile: &Grid, error: f32, finest: &Grid) -> usize {
    let shift = FINEST - key.level().get();
    let scale = f64::from(1_u32 << shift);
    let region = Region::of_tile(key);
    let bound = f64::from(error);
    // The test's own `f64` rounding, far below an `f32` step of `e`.
    let allowed = bound + bound * 1e-12;
    let mut checks = 0;
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
            for (name, positions) in every_build(&sculpt, face) {
                let checks = check_bound(name, &positions, &finest);
                assert!(checks > 0, "{name}: something was checked");
            }
            BOUND_RUN.fetch_add(1, Ordering::Relaxed);
        });
    assert!(BOUND_RUN.load(Ordering::Relaxed) >= 1, "cases ran");
}

/// Bolero cases `shared_vertices_decode_identically_across_tiles_and_faces` runs.
const SEAM_CASES: usize = 12;

/// Every normal encoding the study measures, with the chosen positions.
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

/// Every vertex two display tiles share decodes to the same position, normal and roughness
/// bits in both.
///
/// Across tile borders and across faces, for random sculpts and every normal encoding: seams
/// are exact in shape and in shading, without reading a neighbour at draw time.
#[test]
fn shared_vertices_decode_identically_across_tiles_and_faces() {
    bolero::check!()
        .with_type::<u64>()
        .with_iterations(SEAM_CASES)
        .for_each(|&seed| {
            let sculpt = Sculpt::random(seed);
            let (positions, normals) = sculpt.finest_grids();
            for encoding in ENCODINGS {
                let tiles: BTreeMap<TileKey, DisplayTile> = build_mesh_with(
                    &encode::CHOSEN,
                    encoding,
                    &sculpt.mesh,
                    &positions,
                    &normals,
                )
                .unwrap()
                .into_iter()
                .map(|t| (*t.key(), t))
                .collect();
                let shared = compare_shared(&sculpt.mesh, &tiles);
                assert!(shared > 0, "{encoding:?}: shared vertices were compared");
            }
        });
}

/// Compares every shared vertex of `tiles` (every display tile of `mesh`) across the tiles that
/// hold it; returns how many pairs it compared.
fn compare_shared(mesh: &BaseMesh, tiles: &BTreeMap<TileKey, DisplayTile>) -> usize {
    let mut shared = 0;
    for tile in tiles.values() {
        let key = tile.key();
        let region = Region::of_tile(key);
        for (i, j) in border(key.side()) {
            let mine = usize::try_from(j * key.side() + i).unwrap();
            let copies =
                mesh.topology()
                    .copies(key.face(), key.level(), region.x0() + i, region.y0() + j);
            for copy in &copies {
                let other = &tiles[&copy.key_in(LayerId::BASE, Channel::DISPLAY, key.level())];
                let theirs = usize::try_from(copy.j * other.key().side() + copy.i).unwrap();
                let what = || format!("{key:?} ({i}, {j}) against {:?}", other.key());
                assert_eq!(
                    tile.positions()[mine].to_bits(),
                    other.positions()[theirs].to_bits(),
                    "position of {}",
                    what()
                );
                assert_eq!(
                    tile.normals()[mine].to_bits(),
                    other.normals()[theirs].to_bits(),
                    "normal of {}",
                    what()
                );
                assert_eq!(
                    tile.roughness()[mine],
                    other.roughness()[theirs],
                    "roughness of {}",
                    what()
                );
                shared += 1;
            }
        }
    }
    shared
}

/// A tile [`build_mesh`] built reads back from its bytes bit for bit, and its bytes say the same
/// again: the store can keep only the bytes and the bound.
#[test]
fn a_built_tile_reads_back_from_its_bytes() {
    let sculpt = Sculpt::random(3);
    let (positions, normals) = sculpt.finest_grids();
    let tiles = build_mesh(&sculpt.mesh, &positions, &normals).unwrap();
    assert_eq!(
        tiles.len(),
        6 * (usize::from(FINEST - 1) + 4),
        "per face, levels 0–6 a tile each, level 7 four"
    );
    for tile in &tiles {
        let back =
            DisplayTile::decode(*tile.key(), tile.error(), tile.cone(), tile.encoded()).unwrap();
        assert_eq!(&back, tile, "{:?}", tile.key());
    }
    let first = tiles.first().unwrap();
    assert_eq!(
        DisplayTile::decode(*first.key(), f32::NAN, first.cone(), first.encoded()).map(|_| ()),
        Err(DisplayError::Malformed),
        "a bound that isn't one"
    );
    let face = first.key().face();
    let displacement = TileKey::displacement(face, first.key().level(), 0, 0).unwrap();
    assert_eq!(
        DisplayTile::decode(displacement, first.error(), first.cone(), first.encoded()).map(|_| ()),
        Err(DisplayError::NotDisplay(displacement)),
        "a key of another channel"
    );
    let finest = positions.get(face).unwrap();
    assert_eq!(
        build_positions(face, finest, level(FINEST - 1)).map(|_| ()),
        Err(DisplayError::Shape),
        "a grid of the wrong level"
    );
    assert_eq!(
        build_mesh(&sculpt.mesh, &positions, &FaceGrids::new(level(FINEST))).map(|_| ()),
        Err(DisplayError::Unavailable),
        "no normals"
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
    let (positions, _) = sculpt.finest(face);
    let built = build_positions_with(&BrokenSeams, face, &positions, level(FINEST));
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

/// Bolero cases `decoded_means_stay_near_the_exact_ones` runs.
const DRIFT_CASES: usize = 16;

/// The largest error, by level, of a decoded display mean against the exact mean of the finest
/// normals under it, over every case so far, with the bound it was held to.
static DRIFT: std::sync::Mutex<[(f64, f64); FINEST as usize]> =
    std::sync::Mutex::new([(0.0, 0.0); FINEST as usize]);

/// The largest angle the chosen normal encoding puts between a unit normal and its decoding.
///
/// The limit it was chosen under (0.1°), above the worst measured over every face of the
/// benchmark project (0.059°, `research/platform/m1a2-display-encoding.md`).
const NORMAL_ANGLE_DEGREES: f64 = 0.1;

/// The least length of an exact mean at which the angle is checked too.
///
/// Below it the mean is mostly cancelled detail, a short vector whose direction a small
/// absolute error turns a long way (see `decoded_means_stay_near_the_exact_ones`).
const ANGLE_FLOOR: f64 = 0.5;

/// The largest angle, in degrees, between a decoded display normal and the exact mean's
/// direction, where that mean is at least [`ANGLE_FLOOR`] long.
const ANGLE_LIMIT_DEGREES: f64 = 0.5;

/// The decoded display means of a random sculpt stay within their bound of the exact means.
///
/// Each level is mipped from the decoded means one level finer, so encoding error builds up
/// over levels; this checks it builds up no faster than the encoding allows. At vertex `v` of
/// level `l`, write `M` for the exact mean (the tent mip of the finest unit normals, never
/// encoded), `M̃` for the mean the build mips there from the decoded means one level finer, and
/// `D` for the decoded mean (the decoded unit normal at its roughness's length); `E_l` is the
/// largest `|D − M|` over level `l`, and `θ` is [`NORMAL_ANGLE_DEGREES`].
///
/// **One level's encoding.** `D = n̂ q`, where `q` is `|M̃|` rounded to 255ths and `n̂` is the
/// direction `u` of `M̃` encoded, at most `θ` from it; the chord between them is at most
/// `2 sin(θ/2)`. A mean too short to normalize has `q = 0`, which meets the same bound.
///
/// ```text
/// |q − |M̃||  ≤ 1/510
/// |n̂ − u|    ≤ 2 sin(θ/2)
/// |D − M̃|    ≤ |n̂ q − n̂ |M̃|| + |M̃| |n̂ − u|  ≤  1/510 + 2 sin(θ/2) |M̃|
/// ```
///
/// **The mip.** `M̃` and `M` are the same tent of the decoded and the exact means one level
/// finer, at the same vertices, its weights non-negative and summing to 1: ¼, ⅛ and 1/16 inside
/// a face, renormalized at its borders; ¼, `1/(2n)` and `1/(4n)` around a base vertex of
/// valence `n`. So the tent's error is at most the largest error it reads:
///
/// ```text
/// |M̃ − M|  ≤ E_{l+1}
/// |M̃|      ≤ |M| + E_{l+1}
/// ```
///
/// **Together**, with `E` zero one level finer than the finest display level (whose means are
/// mipped from the finest normals themselves) and a margin of 1e-5 a level for the `f32` sums:
///
/// ```text
/// |D − M|  ≤  E_{l+1}  +  1/510  +  2 sin(θ/2) (|M| + E_{l+1})
/// ```
///
/// The bound is checked at every vertex; the angle to the exact mean's direction only where the
/// exact mean is at least [`ANGLE_FLOOR`] long (within [`ANGLE_LIMIT_DEGREES`]). A short mean's
/// direction is what tiny absolute errors move: at seed 5642940964296373039, a mean 0.0965 long
/// with an absolute error of 0.002 points 0.78° off, which an angle limit took for a fault. The
/// largest error and bound per level are printed (`--nocapture`).
#[test]
fn decoded_means_stay_near_the_exact_ones() {
    bolero::check!()
        .with_type::<u64>()
        .with_iterations(DRIFT_CASES)
        .for_each(|&seed| means_within_their_bound(seed));
    println!(
        "largest error and bound by level: {:?}",
        DRIFT.lock().unwrap()
    );
}

/// `decoded_means_stay_near_the_exact_ones` at the seed that found its old angle limit wrong:
/// short means, whose directions drift though their errors stay within the bound.
#[test]
fn decoded_short_means_stay_within_their_bound() {
    means_within_their_bound(5_642_940_964_296_373_039);
}

/// Checks the decoded display means of the sculpt `seed` makes against the exact means, level
/// by level (see `decoded_means_stay_near_the_exact_ones`).
fn means_within_their_bound(seed: u64) {
    let chord = 2.0 * (NORMAL_ANGLE_DEGREES.to_radians() / 2.0).sin();
    let sculpt = Sculpt::random(seed);
    let (positions, normals) = sculpt.finest_grids();
    let tiles = build_mesh(&sculpt.mesh, &positions, &normals).unwrap();
    let mut exact = normals;
    let mut finer_bound = 0.0_f64;
    for l in (0..FINEST).rev() {
        let mut coarser = FaceGrids::new(level(l));
        for face in sculpt.mesh.faces() {
            let means = mip_normals(&sculpt.mesh, &exact, face, level(l)).unwrap();
            coarser.insert(face, means.into_grid()).unwrap();
        }
        exact = coarser;
        let (mut worst, mut level_bound) = (0.0_f64, 0.0_f64);
        let level_tiles = tiles.iter().filter(|t| t.key().level() == level(l));
        for tile in level_tiles {
            let key = tile.key();
            let region = Region::of_tile(key);
            let grid = exact.get(key.face()).unwrap();
            let decoded = tile.means();
            let pairs = decoded.iter().zip(tile.normals()).enumerate();
            for (index, (&mean, &normal)) in pairs {
                let index = u32::try_from(index).unwrap();
                let (i, j) = (index % key.side(), index / key.side());
                let want = grid.get(region.x0() + i, region.y0() + j).unwrap();
                let length = f64::from(want.length());
                let bound = finer_bound + 1.0 / 510.0 + chord * (length + finer_bound) + 1e-5;
                let error = f64::from((mean - want).length());
                assert!(
                    error <= bound,
                    "seed {seed}, level {l}, {key:?} ({i}, {j}): {error} past its bound {bound}"
                );
                if length >= ANGLE_FLOOR {
                    let unit = want.normalized().unwrap();
                    let angle = angle_degrees(normal, unit);
                    assert!(
                        angle <= ANGLE_LIMIT_DEGREES,
                        "seed {seed}, level {l}: {angle}° from a mean {length} long"
                    );
                }
                worst = worst.max(error);
                level_bound = level_bound.max(bound);
            }
        }
        if let Some(at) = DRIFT.lock().unwrap().get_mut(usize::from(l)) {
            *at = (at.0.max(worst), at.1.max(level_bound));
        }
        finer_bound = level_bound;
    }
}

/// Every tile [`build_tile`] builds from its inputs alone is bit for bit the tile
/// [`build_mesh`] built.
///
/// The inputs are the finest positions under it, the decoded tiles one level finer and their
/// means: a store can bring tiles up to date one at a time.
#[test]
fn a_tile_built_alone_is_the_tile_built_with_its_mesh() {
    let sculpt = Sculpt::random(21);
    let (positions, normals) = sculpt.finest_grids();
    let tiles = build_mesh(&sculpt.mesh, &positions, &normals).unwrap();
    let by_key: BTreeMap<TileKey, &DisplayTile> = tiles.iter().map(|t| (*t.key(), t)).collect();
    for l in (0..FINEST).rev() {
        let finer_level = level(l + 1);
        // Every face's grid one level finer: decoded positions and means of the tiles there, or
        // the finest positions and normals.
        let (mut finer_positions, mut finer_means) =
            (FaceGrids::new(finer_level), FaceGrids::new(finer_level));
        for face in sculpt.mesh.faces() {
            if l + 1 == FINEST {
                finer_positions
                    .insert(face, positions.get(face).unwrap().clone())
                    .unwrap();
                finer_means
                    .insert(face, normals.get(face).unwrap().clone())
                    .unwrap();
                continue;
            }
            let side = finer_level.face_quads() + 1;
            let count = usize::try_from(side * side).unwrap();
            let (mut ps, mut ms) = (vec![Vec3::ZERO; count], vec![Vec3::ZERO; count]);
            for tile in tiles
                .iter()
                .filter(|t| t.key().face() == face && t.key().level() == finer_level)
            {
                let region = Region::of_tile(tile.key());
                for (index, (pp, mm)) in tile.positions().iter().zip(tile.means()).enumerate() {
                    let index = u32::try_from(index).unwrap();
                    let (x, y) = (
                        region.x0() + index % tile.key().side(),
                        region.y0() + index / tile.key().side(),
                    );
                    let at = usize::try_from(y * side + x).unwrap();
                    ps[at] = *pp;
                    ms[at] = mm;
                }
            }
            finer_positions
                .insert(face, Grid::new(side, ps).unwrap())
                .unwrap();
            finer_means
                .insert(face, Grid::new(side, ms).unwrap())
                .unwrap();
        }
        for tile in tiles.iter().filter(|t| t.key().level() == level(l)) {
            let key = *tile.key();
            let region = Region::of_tile(&key);
            let finer_grid = finer_positions.get(key.face()).unwrap();
            let finer = grid(2 * region.quads() + 1, |i, j| {
                finer_grid
                    .get(2 * region.x0() + i, 2 * region.y0() + j)
                    .unwrap()
            });
            let (children, cone) = if l + 1 == FINEST {
                let finest_normals = normals.get(key.face()).unwrap();
                let under = grid(2 * region.quads() + 1, |i, j| {
                    finest_normals
                        .get(2 * region.x0() + i, 2 * region.y0() + j)
                        .unwrap()
                });
                (0.0, FinestNormalCone::covering(under.as_slice()).unwrap())
            } else {
                let ratio = finer_level.tiles_per_side() / level(l).tiles_per_side();
                // Handed over in reverse, as a store might gather them: the order doesn't
                // matter.
                let under: Vec<&DisplayTile> = (0..ratio * ratio)
                    .rev()
                    .map(|n| {
                        let tx = u16::try_from(u32::from(key.tx()) * ratio + n % ratio).unwrap();
                        let ty = u16::try_from(u32::from(key.ty()) * ratio + n / ratio).unwrap();
                        by_key[&TileKey::display(key.face(), finer_level, tx, ty).unwrap()]
                    })
                    .collect();
                let cones: Vec<FinestNormalCone> = under.iter().map(|t| t.cone()).collect();
                (
                    under.iter().map(|t| t.error()).fold(0.0, f32::max),
                    FinestNormalCone::holding(&cones).unwrap(),
                )
            };
            let means = mip_normals_tile(&sculpt.mesh, &finer_means, &key).unwrap();
            let alone = build_tile(key, &finer, children, cone, &means).unwrap();
            assert_eq!(&alone, tile, "{key:?}");
        }
    }
}

/// The finest normals under display tile `key`: its block of the face's finest grid, borders
/// included.
fn finest_under(normals: &FaceGrids, key: &TileKey) -> Vec<Vec3> {
    let region = Region::of_tile(key);
    let shift = u32::from(FINEST - key.level().get());
    let face = normals.get(key.face()).unwrap();
    let side = (region.quads() << shift) + 1;
    let (x0, y0) = (region.x0() << shift, region.y0() << shift);
    (0..side)
        .flat_map(|j| (0..side).map(move |i| (i, j)))
        .map(|(i, j)| face.get(x0 + i, y0 + j).unwrap())
        .collect()
}

/// The display tiles under `key` one level finer.
fn children_of<'a>(
    tiles: &'a BTreeMap<TileKey, DisplayTile>,
    key: &TileKey,
) -> impl Iterator<Item = &'a DisplayTile> {
    let region = Region::of_tile(key);
    let finer = key.level().child().unwrap();
    tiles.values().filter(move |t| {
        let under = Region::of_tile(t.key());
        t.key().face() == key.face()
            && t.key().level() == finer
            && under.x0() >= 2 * region.x0()
            && under.y0() >= 2 * region.y0()
            && under.x0() + under.quads() <= 2 * (region.x0() + region.quads())
            && under.y0() + under.quads() <= 2 * (region.y0() + region.quads())
    })
}

/// Every display tile of `sculpt`, by key, with the finest normals they were built from.
fn built(sculpt: &Sculpt) -> (BTreeMap<TileKey, DisplayTile>, FaceGrids) {
    let (positions, normals) = sculpt.finest_grids();
    let tiles = build_mesh(&sculpt.mesh, &positions, &normals).unwrap();
    (tiles.into_iter().map(|t| (*t.key(), t)).collect(), normals)
}

/// The smooth sphere and random sculpts the cone tests build, by name.
fn cone_sculpts() -> impl Iterator<Item = (String, Sculpt)> {
    let random = [3_u64, 8, 21, 34, 55]
        .into_iter()
        .map(|seed| (format!("seed {seed}"), Sculpt::random(seed)));
    std::iter::once(("smooth".to_owned(), Sculpt::smooth())).chain(random)
}

/// Every finest normal under every display tile lies within its cone as decoded, at every level,
/// on the smooth sphere and on rough random sculpts; and a tile's cone holds each of its
/// children's.
#[test]
fn cones_hold_every_finest_normal_and_their_childrens_cones() {
    for (name, sculpt) in cone_sculpts() {
        let (tiles, normals) = built(&sculpt);
        for (key, tile) in &tiles {
            let cone = tile.cone();
            for normal in finest_under(&normals, key) {
                assert!(
                    cone.contains(normal),
                    "{name}, {key:?}: {normal:?} outside {cone:?} ({}° from its axis)",
                    angle_degrees(normal, cone.axis())
                );
            }
            if key.level().get() + 1 == FINEST {
                continue;
            }
            let children: Vec<&DisplayTile> = children_of(&tiles, key).collect();
            assert!(!children.is_empty(), "{name}, {key:?}: no children");
            for child in children {
                assert!(
                    cone.holds(child.cone()),
                    "{name}, {key:?}: {:?} of {:?} outside {cone:?}",
                    child.cone(),
                    child.key()
                );
            }
        }
    }
}

/// The half-angle of `tile`'s cone and how far it lies over the exact cone around the same
/// axis (the widest angle from it to a finest normal under the tile), in degrees.
fn width_and_slack(normals: &FaceGrids, tile: &DisplayTile) -> (f64, f64) {
    let cone = tile.cone();
    let half = f64::from(cone.half_angle().to_degrees());
    let exact = finest_under(normals, tile.key())
        .into_iter()
        .map(|n| angle_degrees(n, cone.axis()))
        .fold(0.0_f64, f64::max);
    (half, half - exact)
}

/// On the smooth sphere, no cone covers everything, and each is as tight as its inputs allow
/// around its own axis.
///
/// That is, within a step of the widest finest normal one level below the finest,
/// within a step of the widest child cone above, and a tile holding one child has the child's
/// very cone.
#[test]
fn cones_on_a_smooth_sphere_are_tight() {
    // A step, in degrees, with room for the slack and the rounding of the degrees.
    let step = 180.0 / 4095.0 + 1e-6;
    let (tiles, normals) = built(&Sculpt::smooth());
    for (key, tile) in &tiles {
        let cone = tile.cone();
        assert!(!cone.covers_everything(), "{key:?}");
        let half = f64::from(cone.half_angle().to_degrees());
        if key.level().get() + 1 == FINEST {
            let (_, slack) = width_and_slack(&normals, tile);
            assert!(
                (0.0..=step).contains(&slack),
                "{key:?}: {half}°, {slack}° over its normals"
            );
            continue;
        }
        let children: Vec<&DisplayTile> = children_of(&tiles, key).collect();
        if let [only] = children.as_slice() {
            assert_eq!(cone, only.cone(), "{key:?}: one child");
            continue;
        }
        let widest = children
            .iter()
            .map(|c| {
                angle_degrees(cone.axis(), c.cone().axis())
                    + f64::from(c.cone().half_angle().to_degrees())
            })
            .fold(0.0_f64, f64::max);
        assert!(
            (0.0..=step).contains(&(half - widest)),
            "{key:?}: {half}° around children reaching {widest}°"
        );
    }
}

/// The `q` quantile of sorted `values`, by nearest rank.
fn quantile(values: &[f64], q: f64) -> f64 {
    let last = values.len().saturating_sub(1);
    let rank = (q * f64::from(u32::try_from(last).unwrap())).round();
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a rank within the values, rounded"
    )]
    let rank = rank as usize;
    values[rank.min(last)]
}

/// The finest level of the deep cone fixture ([`deep_normals`]): its level 9 has 8×8 tiles a
/// face, so levels 8, 7 and 6 each take the union of several children's cones, unions of unions.
const DEEP_FINEST: u8 = 10;

/// One face's unit normals at [`DEEP_FINEST`], made up rather than evaluated.
///
/// A whole mesh that deep is slow to build. These are a curved face's, as a cube face's projected
/// onto a sphere, with bumps at a few scales and per-vertex noise of length `rough`, from `seed`.
fn deep_normals(seed: u64, rough: f32) -> Grid {
    let side = level(DEEP_FINEST).face_quads() + 1;
    let step = 2.0 / f32::from(u16::try_from(side - 1).unwrap());
    let mut rng = Rng::new(seed);
    let data = (0..side)
        .flat_map(|j| (0..side).map(move |i| (i, j)))
        .map(|(i, j)| {
            let u = -1.0 + f32::from(u16::try_from(i).unwrap()) * step;
            let v = -1.0 + f32::from(u16::try_from(j).unwrap()) * step;
            let bump = 0.2 * (13.0 * u).sin() * (9.0 * v).cos() + 0.05 * (70.0 * (u + v)).sin();
            let n = Vec3::new(u + bump, v - bump, 1.0) + rng.vec3(rough);
            n.normalized().unwrap()
        })
        .collect();
    Grid::new(side, data).unwrap()
}

/// Cones chain through unions of unions without losing a finest normal.
///
/// The cones [`build_cones`] gives a face deeper than the meshes' fixtures hold every finest
/// normal under each tile, and each parent holds its children's, through three levels of unions:
/// on a smooth face and on a rough one.
#[test]
fn deep_cones_chain_through_unions_of_unions() {
    let finest = level(DEEP_FINEST);
    let face = FaceId::new(0);
    for (seed, rough) in [(1_u64, 0.0_f32), (2, 0.3)] {
        let normals = deep_normals(seed, rough);
        let cones = build_cones(face, &normals, finest).unwrap();
        let mut merging_levels = 0;
        for l in 0..DEEP_FINEST {
            let at = level(l);
            let per_side = u16::try_from(at.tiles_per_side()).unwrap();
            let keys: Vec<TileKey> = (0..per_side)
                .flat_map(|ty| (0..per_side).map(move |tx| (tx, ty)))
                .map(|(tx, ty)| TileKey::display(face, at, tx, ty).unwrap())
                .collect();
            for key in &keys {
                let cone = cones.cone(key).unwrap();
                let region = Region::of_tile(key);
                let shift = u32::from(DEEP_FINEST - l);
                let side = (region.quads() << shift) + 1;
                let (x0, y0) = (region.x0() << shift, region.y0() << shift);
                let under = (0..side).flat_map(|j| (0..side).map(move |i| (x0 + i, y0 + j)));
                for (i, j) in under {
                    let n = normals.get(i, j).unwrap();
                    assert!(
                        cone.contains(n),
                        "rough {rough}, {key:?}: {n:?} outside {cone:?}"
                    );
                }
            }
            let Some(finer) = at.child().filter(|f| *f < finest) else {
                continue;
            };
            let ratio = finer.tiles_per_side() / at.tiles_per_side();
            merging_levels += usize::from(ratio > 1);
            for key in &keys {
                let cone = cones.cone(key).unwrap();
                let children = (0..ratio * ratio).map(|n| {
                    let tx = u16::try_from(u32::from(key.tx()) * ratio + n % ratio).unwrap();
                    let ty = u16::try_from(u32::from(key.ty()) * ratio + n / ratio).unwrap();
                    cones
                        .cone(&TileKey::display(face, finer, tx, ty).unwrap())
                        .unwrap()
                });
                for child in children {
                    assert!(
                        cone.holds(child),
                        "rough {rough}, {key:?}: {child:?} in {cone:?}"
                    );
                }
            }
        }
        assert_eq!(merging_levels, 3, "levels 8, 7 and 6 take unions");
    }
}

/// Prints the median and 99th percentile half-angle of the cones at each level, and of how far
/// they lie over the exact cone around the same axis.
///
/// On the smooth sphere and the random sculpts, then the half-angles alone on the deep smooth
/// and rough faces: `cargo nextest run -p geom --run-ignored only cone_widths --no-capture`.
#[test]
#[ignore = "a measurement to read, not a check"]
fn cone_widths() {
    let face = FaceId::new(0);
    for (seed, rough) in [(1_u64, 0.0_f32), (2, 0.3)] {
        let cones = build_cones(face, &deep_normals(seed, rough), level(DEEP_FINEST)).unwrap();
        println!("deep face, rough {rough}: level, median / p99 half-angle °");
        for l in 0..DEEP_FINEST {
            let per_side = u16::try_from(level(l).tiles_per_side()).unwrap();
            let mut halves: Vec<f64> = (0..per_side)
                .flat_map(|ty| (0..per_side).map(move |tx| (tx, ty)))
                .map(|(tx, ty)| {
                    let key = TileKey::display(face, level(l), tx, ty).unwrap();
                    f64::from(cones.cone(&key).unwrap().half_angle().to_degrees())
                })
                .collect();
            halves.sort_by(f64::total_cmp);
            println!(
                "  {l}: {:7.3} / {:7.3}   ({} tiles)",
                quantile(&halves, 0.5),
                quantile(&halves, 0.99),
                halves.len()
            );
        }
    }
    for (name, sculpt) in cone_sculpts() {
        let (tiles, normals) = built(&sculpt);
        let mut by_level: BTreeMap<u8, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
        for tile in tiles.values() {
            let (half, slack) = width_and_slack(&normals, tile);
            let entry = by_level.entry(tile.key().level().get()).or_default();
            entry.0.push(half);
            entry.1.push(slack);
        }
        println!("{name}: level, median / p99 half-angle °, median / p99 over the exact °");
        for (l, (mut halves, mut slacks)) in by_level {
            halves.sort_by(f64::total_cmp);
            slacks.sort_by(f64::total_cmp);
            println!(
                "  {l}: {:7.3} / {:7.3}   {:6.3} / {:6.3}   ({} tiles)",
                quantile(&halves, 0.5),
                quantile(&halves, 0.99),
                quantile(&slacks, 0.5),
                quantile(&slacks, 0.99),
                halves.len()
            );
        }
    }
}
