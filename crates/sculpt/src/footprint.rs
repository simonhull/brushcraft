//! A brush's footprint by its vertex count: the radius about a point whose ball holds a given
//! number of the finest level's evaluated positions.
//!
//! Brush sizes are given in pixels or world units, but a brush's cost follows the vertices it
//! moves, and the M1 spec's big brush is "a brush of 1M vertices". [`radius_covering`] finds the
//! radius that holds that many about a point on the surface as it is, so a measured run can ask
//! for the same work on any project, whatever its density there.

use geom::Vec3;
use store::{CacheError, SpatialIndex, TileStore};

/// The radius the search starts from, in world units: well below any brush on a unit-sized
/// model, so the first ball holds fewer vertices than asked for and the search only grows it.
const START: f32 = 1.0 / 1024.0;

/// The radius the search gives up growing at, in world units.
///
/// Far past any model's size (the app's models are about a unit across), so a ball this big
/// holds the whole model, a count still short is all the model has, and the answer is the whole
/// model.
const LARGEST: f32 = 1024.0;

/// How many halvings narrow the bracket, once it holds the answer: from at most its upper end
/// wide to 2⁻²⁴ of it, about one `f32` step, where no halving changes the count any more.
const NARROWING: u32 = 24;

/// The smallest radius about `center` whose ball holds at least `vertices` of the finest level's
/// evaluated positions, to about an `f32` step; [`LARGEST`] if the whole model holds fewer.
///
/// The count is of evaluated positions tile by tile, so a vertex on a border between tiles is
/// counted once for each tile holding it: a few percent more than the distinct vertices, which
/// a footprint of hundreds of thousands hardly notices. A position counts when its distance
/// from `center` is below the radius.
///
/// The search doubles the radius from [`START`] until the ball holds enough, then halves the
/// bracket between the last radius short of it and the first enough [`NARROWING`] times, each step
/// evaluating the tiles whose boxes touch the ball through `store` (its pool keeps them for the
/// next). Its cost follows the footprint: the largest ball it evaluates is at most twice the
/// answer's radius, four times its vertices. A million-vertex footprint on the benchmark project is
/// tens of milliseconds once its tiles are in the pool; a tool's or a scripted run's call, never a
/// stroke's.
///
/// # Errors
///
/// [`CacheError`] if a tile can't be evaluated.
pub fn radius_covering(
    index: &SpatialIndex,
    store: &mut TileStore,
    center: Vec3,
    vertices: usize,
) -> Result<f32, CacheError> {
    // `low` always holds fewer than asked for (none at radius 0), `high` enough.
    let (mut low, mut high) = (0.0, START);
    while positions_within(index, store, center, high)? < vertices {
        if high >= LARGEST {
            return Ok(LARGEST);
        }
        low = high;
        high *= 2.0;
    }
    for _ in 0..NARROWING {
        let middle = f32::midpoint(low, high);
        if positions_within(index, store, center, middle)? < vertices {
            low = middle;
        } else {
            high = middle;
        }
    }
    Ok(high)
}

/// How many of the finest level's evaluated positions lie closer to `center` than `radius`,
/// counted tile by tile (see [`radius_covering`]).
///
/// # Errors
///
/// [`CacheError`] if a tile can't be evaluated.
fn positions_within(
    index: &SpatialIndex,
    store: &mut TileStore,
    center: Vec3,
    radius: f32,
) -> Result<usize, CacheError> {
    let squared = radius * radius;
    let touched = index.query_sphere(center, radius);
    let keys: Vec<_> = touched.collect();
    let mut count = 0;
    for key in keys {
        let inside = store.evaluate(key)?.iter().filter(|&&p| {
            let d = p - center;
            d.dot(d) < squared
        });
        count += inside.count();
    }
    Ok(count)
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use geom::Level;

    use super::*;
    use crate::testing::{TestDir, cube, fixed};

    /// The radius found holds the vertices asked for, and a hair less holds fewer: the smallest
    /// such radius, to about an `f32` step.
    #[test]
    fn the_radius_found_is_the_smallest_holding_the_vertices_asked_for() {
        let dir = TestDir::new();
        let level = Level::new(5).unwrap();
        let (mut project, index) = cube(&dir, level, fixed(2));
        let store = project.store();
        let center = Vec3::new(0.0, 0.0, 1.0);
        for vertices in [1, 100, 2_000] {
            let radius = radius_covering(&index, store, center, vertices).unwrap();
            let held = positions_within(&index, store, center, radius).unwrap();
            assert!(held >= vertices, "{vertices}: {held} at {radius}");
            let less = radius * (1.0 - 1e-6);
            let fewer = positions_within(&index, store, center, less).unwrap();
            assert!(fewer < vertices, "{vertices}: {fewer} at {less}");
        }
    }

    /// More vertices than the whole model holds: the largest radius, the whole model.
    #[test]
    fn more_vertices_than_the_model_holds_is_the_whole_model() {
        let dir = TestDir::new();
        let (mut project, index) = cube(&dir, Level::new(2).unwrap(), fixed(1));
        let center = Vec3::new(0.0, 0.0, 1.0);
        let radius = radius_covering(&index, project.store(), center, usize::MAX).unwrap();
        assert!((radius - LARGEST).abs() < f32::EPSILON, "{radius}");
    }
}
