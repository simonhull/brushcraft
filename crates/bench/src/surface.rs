//! Where things are on the benchmark's surface: tiles near a direction, points on the surface,
//! a brush radius that covers a footprint, and the path a stroke follows.

use anyhow::{Context as _, Result, anyhow};
use geom::{BaseMesh, Level, TileKey, Vec3};
use store::{Ray, SpatialIndex, TileStore};

/// Every tile of `level` on `mesh` with an estimate of the direction of its centre from the
/// mesh's centre, worked out from the base mesh alone (nothing is evaluated).
///
/// The estimate is the point of the tile's centre on the bilinear patch through its face's
/// four corners (corners 0–3 at face-grid `(0, 0)`, `(n, 0)`, `(n, n)`, `(0, n)`), normalised.
/// On the benchmark sphere that is within a fraction of a tile of the true centre, which is
/// all choosing a region of nearby tiles needs.
///
/// # Errors
///
/// If a face has no corner, which a valid mesh never lacks.
pub(crate) fn tile_directions(mesh: &BaseMesh, level: Level) -> Result<Vec<(TileKey, Vec3)>> {
    let per_side = level.tiles_per_side();
    let side = f32::from(u16::try_from(per_side)?);
    let mut tiles = Vec::new();
    for face in mesh.faces() {
        let corner = |c| {
            mesh.corner_position(face, c)
                .ok_or_else(|| anyhow!("{face} has no corner {c}"))
        };
        let [c0, c1, c2, c3] = [corner(0)?, corner(1)?, corner(2)?, corner(3)?];
        for ty in 0..u16::try_from(per_side)? {
            for tx in 0..u16::try_from(per_side)? {
                let u = (f32::from(tx) + 0.5) / side;
                let v = (f32::from(ty) + 0.5) / side;
                let p = c0 * ((1.0 - u) * (1.0 - v))
                    + c1 * (u * (1.0 - v))
                    + c2 * (u * v)
                    + c3 * ((1.0 - u) * v);
                let key = TileKey::displacement(face, level, tx, ty)?;
                tiles.push((key, p.normalized().unwrap_or(p)));
            }
        }
    }
    Ok(tiles)
}

/// The `n` tiles of `tiles` (from [`tile_directions`]) nearest `direction`, nearest first; all
/// of them if there are fewer.
pub(crate) fn nearest_tiles(tiles: &[(TileKey, Vec3)], direction: Vec3, n: usize) -> Vec<TileKey> {
    let mut by_distance: Vec<(f32, TileKey)> = tiles
        .iter()
        .map(|&(key, d)| ((d - direction).length(), key))
        .collect();
    by_distance.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    by_distance
        .into_iter()
        .take(n)
        .map(|(_, key)| key)
        .collect()
}

/// Where the surface lies in `direction` from the origin, the centre of the closed benchmark
/// surface: where a ray from the origin leaves it.
///
/// # Errors
///
/// If a tile can't be evaluated, or the ray leaves through no tile (the surface isn't closed
/// around the origin).
pub(crate) fn on_surface(
    index: &SpatialIndex,
    store: &mut TileStore,
    direction: Vec3,
) -> Result<Vec3> {
    let ray = Ray::new(Vec3::ZERO, direction).ok_or_else(|| anyhow!("no direction"))?;
    let hit = index
        .raycast(store, &ray)
        .context("raycasting")?
        .ok_or_else(|| anyhow!("the ray along {direction:?} leaves through no tile"))?;
    Ok(ray.at(hit.t))
}

/// The radius that covers about `footprint` vertices around `center`.
///
/// The brushes' own search ([`sculpt::radius_covering`]), which counts a vertex on a tile border
/// once per tile holding it, a few percent more than the vertices themselves; the draw step's own
/// count, which the harness reports, is exact.
///
/// # Errors
///
/// If a tile can't be evaluated.
pub(crate) fn calibrated_radius(
    index: &SpatialIndex,
    store: &mut TileStore,
    center: Vec3,
    footprint: usize,
) -> Result<f32> {
    Ok(sculpt::radius_covering(index, store, center, footprint)?)
}

/// The unit directions of a stroke's `steps` dabs from the origin: along the great circle
/// through `start` heading east (anticlockwise about the `z` axis), each `step` radians of arc
/// on from the one before.
///
/// A great circle, not the circle of latitude through `start`: on the unit sphere each step
/// is then `step` long wherever the stroke starts, where a turn about `z` by `step` covers
/// only `step · cos(latitude)`, dabs bunching up towards the poles. At a pole, where east is
/// undefined, the stroke heads along `y`.
pub(crate) fn path(start: Vec3, step: f32, steps: usize) -> impl Iterator<Item = Vec3> {
    let z = Vec3::new(0.0, 0.0, 1.0);
    let from = start.normalized().unwrap_or(z);
    // The circle turns about the axis at right angles to `from` in the plane of `from` and `z`,
    // which moves `from` along `axis × from`: east.
    let axis = (z - from * from.dot(z))
        .normalized()
        .unwrap_or_else(|| Vec3::new(-1.0, 0.0, 0.0));
    let heading = axis.cross(from);
    (0_u32..).take(steps).map(move |k| {
        // Up to 2^24 steps convert exactly; strokes are hundreds.
        #[expect(clippy::cast_precision_loss, reason = "step counts are far below 2^24")]
        let angle = step * k as f32;
        let (sin, cos) = angle.sin_cos();
        from * cos + heading * sin
    })
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use geom::cube_sphere;

    use super::*;

    #[test]
    fn tile_directions_cover_the_sphere_and_nearest_finds_neighbours() {
        let mesh = cube_sphere(2).unwrap();
        let level = Level::new(8).unwrap();
        let tiles = tile_directions(&mesh, level).unwrap();
        assert_eq!(tiles.len(), 24 * 16, "every tile");
        assert!(
            tiles.iter().all(|(_, d)| (d.length() - 1.0).abs() < 1e-5),
            "unit directions"
        );
        let up = Vec3::new(0.0, 0.0, 1.0);
        let near = nearest_tiles(&tiles, up, 4);
        assert_eq!(near.len(), 4, "as many as asked");
        for key in &near {
            let (_, d) = tiles.iter().find(|(k, _)| k == key).unwrap();
            assert!(d.dot(up) > 0.95, "{key:?} is near the pole: {d:?}");
        }
        assert_eq!(
            nearest_tiles(&tiles, up, 10_000).len(),
            tiles.len(),
            "at most all"
        );
    }

    #[test]
    fn the_path_heads_east_along_a_great_circle() {
        let points: Vec<Vec3> =
            path(Vec3::new(2.0, 0.0, 0.0), std::f32::consts::FRAC_PI_2, 3).collect();
        assert_eq!(points.len(), 3, "three dabs");
        assert!(
            (points[0] - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-6,
            "unit directions"
        );
        assert!(
            (points[1] - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-6,
            "a quarter turn east, along the equator"
        );
        assert!(
            (points[2] - Vec3::new(-1.0, 0.0, 0.0)).length() < 1e-6,
            "a half turn"
        );
    }

    #[test]
    fn dabs_advance_evenly_at_any_latitude() {
        let step = 0.01;
        for start in [
            Vec3::new(1.0, 0.0, 0.1),
            Vec3::new(0.3, -0.2, 0.9),
            Vec3::new(0.01, 0.0, -1.0),
            Vec3::new(0.0, 0.0, 1.0),
        ] {
            let points: Vec<Vec3> = path(start, step, 50).collect();
            for pair in points.windows(2) {
                let (a, b) = (pair[0], pair[1]);
                assert!((b.length() - 1.0).abs() < 1e-5, "{start:?}: unit {b:?}");
                let arc = a.cross(b).length().atan2(a.dot(b));
                assert!(
                    (arc - step).abs() < 1e-4,
                    "{start:?}: each dab {step} on, not {arc}"
                );
            }
            let east = Vec3::new(0.0, 0.0, 1.0).cross(points[0]);
            if east.length() > 1e-3 {
                assert!(
                    (points[1] - points[0]).dot(east) > 0.0,
                    "{start:?}: heading east"
                );
            }
        }
    }
}
