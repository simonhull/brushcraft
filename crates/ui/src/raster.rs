//! Outline to coverage: signed-area accumulation.
//!
//! Each edge adds, to every pixel it crosses, how much the coverage changes there, so a running sum
//! along a row gives the coverage itself and the fill needs no sorting and no winding bookkeeping.

use ttf_parser::OutlineBuilder;

/// The most a flattened curve may stray from the true one, in pixels.
///
/// Pieces cut inside a curve, so a loose tolerance thins everything round, and a small dot the
/// most. Each glyph rasterises once, so being fine costs next to nothing.
const TOLERANCE: f32 = 1.0 / 256.0;

/// A size as a coordinate.
pub(crate) fn coordinate(size: usize) -> f32 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "sizes here are at most an atlas's, far inside f32's whole numbers"
    )]
    let coordinate = size as f32;
    coordinate
}

/// A coordinate known to be whole and on the grid, as an index.
fn index(coordinate: f32) -> usize {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "callers pass whole coordinates, clamped to the grid"
    )]
    let index = coordinate as usize;
    index
}

/// How many straight pieces keep a curve within [`TOLERANCE`], given how far it bends: `bend` is
/// the largest second difference of its control points, weighted for the curve's degree.
fn pieces(bend: f32) -> usize {
    index((bend / TOLERANCE).sqrt().ceil().clamp(1.0, 256.0))
}

/// How far three control points are from lying evenly on a line.
fn second_difference(one: (f32, f32), two: (f32, f32), three: (f32, f32)) -> f32 {
    let x = one.0 - 2.0 * two.0 + three.0;
    let y = one.1 - 2.0 * two.1 + three.1;
    x.hypot(y)
}

/// The point `along` a quadratic, 0 to 1.
fn quadratic(from: (f32, f32), control: (f32, f32), to: (f32, f32), along: f32) -> (f32, f32) {
    let rest = 1.0 - along;
    let (first, second, third) = (rest * rest, 2.0 * rest * along, along * along);
    (
        first * from.0 + second * control.0 + third * to.0,
        first * from.1 + second * control.1 + third * to.1,
    )
}

/// The point `along` a cubic, 0 to 1.
fn cubic(
    from: (f32, f32),
    one: (f32, f32),
    two: (f32, f32),
    to: (f32, f32),
    along: f32,
) -> (f32, f32) {
    let rest = 1.0 - along;
    let (first, second) = (rest * rest * rest, 3.0 * rest * rest * along);
    let (third, fourth) = (3.0 * rest * along * along, along * along * along);
    (
        first * from.0 + second * one.0 + third * two.0 + fourth * to.0,
        first * from.1 + second * one.1 + third * two.1 + fourth * to.1,
    )
}

/// A grid of pixels being filled from straight edges in pixel coordinates, y down, or from an
/// outline through [`OutlineBuilder`].
#[derive(Debug, Clone)]
pub(crate) struct Raster {
    /// Pixels across.
    width: usize,
    /// Each row's change in coverage at each pixel, then two more cells for what falls on the
    /// right edge, so a row's sum never spills into the next.
    deltas: Vec<f32>,
    /// Pixels an outline unit.
    scale: f32,
    /// Where the outline's origin lies, in pixels.
    origin: (f32, f32),
    /// The outline's current point, in pixels.
    pen: (f32, f32),
    /// Where the current contour began, in pixels.
    start: (f32, f32),
}

impl Raster {
    /// An empty grid for edges in pixels.
    #[cfg(test)]
    pub(crate) fn new(width: usize, height: usize) -> Self {
        Self::placed(width, height, 1.0, (0.0, 0.0))
    }

    /// An empty grid for an outline: y up, `scale` pixels a unit, its origin at the pixel `origin`.
    pub(crate) fn placed(width: usize, height: usize, scale: f32, origin: (f32, f32)) -> Self {
        Self {
            width,
            deltas: vec![0.0; (width + 2) * height],
            scale,
            origin,
            pen: origin,
            start: origin,
        }
    }

    /// The cells a row holds.
    const fn stride(&self) -> usize {
        self.width + 2
    }

    /// The pixel an outline's point falls on.
    fn pixel(&self, x: f32, y: f32) -> (f32, f32) {
        (
            self.origin.0 + x * self.scale,
            self.origin.1 - y * self.scale,
        )
    }

    /// Adds a straight edge, in pixels. A contour's edges must close it. What lies above or below
    /// the grid is dropped, and what lies beside it counts as on its edge.
    pub(crate) fn line(&mut self, from: (f32, f32), to: (f32, f32)) {
        let (sign, top, bottom) = if from.1 < to.1 {
            (1.0, from, to)
        } else if to.1 < from.1 {
            (-1.0, to, from)
        } else {
            return;
        };
        let slope = (bottom.0 - top.0) / (bottom.1 - top.1);
        let (stride, right) = (self.stride(), coordinate(self.width));
        let rows_on_grid = coordinate(self.deltas.len() / stride);
        let first = index(top.1.clamp(0.0, rows_on_grid).floor());
        let last = index(bottom.1.clamp(0.0, rows_on_grid).ceil());
        let rows = self
            .deltas
            .chunks_exact_mut(stride)
            .enumerate()
            .take(last)
            .skip(first);
        for (y, row) in rows {
            let entered = coordinate(y).max(top.1);
            let left = coordinate(y + 1).min(bottom.1);
            let x_in = (top.0 + (entered - top.1) * slope).clamp(0.0, right);
            let x_out = (top.0 + (left - top.1) * slope).clamp(0.0, right);
            cross_row(row, x_in, x_out, sign * (left - entered));
        }
    }

    /// The coverage, a byte a pixel, row by row. Overlapping contours count as one.
    pub(crate) fn finish(self) -> Vec<u8> {
        let stride = self.stride();
        let mut coverage = Vec::with_capacity(self.deltas.len() / stride * self.width);
        for row in self.deltas.chunks_exact(stride) {
            let mut sum = 0.0;
            let summed = row.iter().take(self.width).map(|delta| {
                sum += delta;
                to_byte(sum)
            });
            coverage.extend(summed);
        }
        coverage
    }

    /// Draws to a pixel from the current point.
    fn advance(&mut self, to: (f32, f32)) {
        self.line(self.pen, to);
        self.pen = to;
    }
}

/// Adds one edge's crossing of a row, from `x_in` to `x_out`, over a signed `height` of the row.
///
/// Each column the edge passes through gets its share, split between that column and the next by
/// how far along the column the share lies.
fn cross_row(row: &mut [f32], x_in: f32, x_out: f32, height: f32) {
    let (left, right) = (x_in.min(x_out), x_in.max(x_out));
    let first = left.floor();
    let span = right - left;
    let columns = index(first)..index(right.ceil().max(first + 1.0));
    for column in columns {
        let cell = coordinate(column);
        let (start, end) = (left.max(cell), right.min(cell + 1.0));
        let share = if span > 0.0 {
            (end - start) / span
        } else {
            1.0
        };
        let middle = f32::midpoint(start, end) - cell;
        let delta = height * share;
        add(row, column, delta * (1.0 - middle));
        add(row, column + 1, delta * middle);
    }
}

/// Adds to a cell. Rows are wide enough for every crossing, so none is ever missed.
fn add(row: &mut [f32], column: usize, delta: f32) {
    if let Some(cell) = row.get_mut(column) {
        *cell += delta;
    }
}

/// A running sum as coverage: either winding fills, and overlaps saturate.
fn to_byte(sum: f32) -> u8 {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped to 0 to 1 before scaling to 0 to 255"
    )]
    let byte = (sum.abs().min(1.0) * 255.0).round() as u8;
    byte
}

impl OutlineBuilder for Raster {
    fn move_to(&mut self, x: f32, y: f32) {
        self.pen = self.pixel(x, y);
        self.start = self.pen;
    }

    fn line_to(&mut self, x: f32, y: f32) {
        self.advance(self.pixel(x, y));
    }

    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        let (from, control, to) = (self.pen, self.pixel(x1, y1), self.pixel(x, y));
        // Uniform steps leave an error of at most |p0 - 2 p1 + p2| / (4 n^2).
        let steps = pieces(second_difference(from, control, to) / 4.0);
        for step in 1..=steps {
            let along = coordinate(step) / coordinate(steps);
            self.advance(quadratic(from, control, to, along));
        }
    }

    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        let (from, to) = (self.pen, self.pixel(x, y));
        let (one, two) = (self.pixel(x1, y1), self.pixel(x2, y2));
        // Uniform steps leave an error of at most 3/4 of the larger second difference over n^2.
        let bend = second_difference(from, one, two).max(second_difference(one, two, to));
        let steps = pieces(0.75 * bend);
        for step in 1..=steps {
            let along = coordinate(step) / coordinate(steps);
            self.advance(cubic(from, one, two, to, along));
        }
    }

    fn close(&mut self) {
        self.advance(self.start);
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use ttf_parser::OutlineBuilder;

    use super::*;

    /// The total coverage of a finished raster, in whole pixels.
    fn area(coverage: &[u8]) -> f32 {
        coverage.iter().map(|&value| f32::from(value) / 255.0).sum()
    }

    #[test]
    fn a_square_covers_its_inside_fully_and_splits_a_shared_pixel() {
        // A 4 x 4 square from (0.5, 0.5) on a 5 x 5 grid.
        let mut raster = Raster::new(5, 5);
        raster.line((0.5, 0.5), (4.5, 0.5));
        raster.line((4.5, 0.5), (4.5, 4.5));
        raster.line((4.5, 4.5), (0.5, 4.5));
        raster.line((0.5, 4.5), (0.5, 0.5));
        let coverage = raster.finish();
        assert_eq!(coverage[2 * 5 + 2], 255, "inside");
        assert_eq!(coverage[0], 64, "a quarter of the corner pixel");
        assert_eq!(coverage[2 * 5], 128, "half of an edge pixel");
    }

    #[test]
    fn a_shape_on_the_right_edge_spills_nothing_into_the_next_row() {
        let mut raster = Raster::new(3, 3);
        raster.line((1.0, 0.0), (3.0, 0.0));
        raster.line((3.0, 0.0), (3.0, 2.0));
        raster.line((3.0, 2.0), (1.0, 2.0));
        raster.line((1.0, 2.0), (1.0, 0.0));
        let coverage = raster.finish();
        assert_eq!(
            coverage,
            [0, 255, 255, 0, 255, 255, 0, 0, 0],
            "two columns of two rows"
        );
    }

    #[test]
    fn a_slanted_edge_across_many_columns_keeps_its_area() {
        let mut raster = Raster::new(8, 2);
        raster.line((0.0, 0.0), (8.0, 0.0));
        raster.line((8.0, 0.0), (0.0, 2.0));
        raster.line((0.0, 2.0), (0.0, 0.0));
        let coverage = raster.finish();
        assert!((area(&coverage) - 8.0).abs() < 0.05, "half of 8 x 2");
        assert_eq!(coverage[0], 255, "the triangle's fat end");
        assert_eq!(coverage[15], 0, "past its thin end");
    }

    #[test]
    fn what_lies_off_the_grid_is_clipped_and_still_fills_what_is_on_it() {
        let mut raster = Raster::new(2, 2);
        raster.line((-5.0, -5.0), (9.0, -5.0));
        raster.line((9.0, -5.0), (9.0, 9.0));
        raster.line((9.0, 9.0), (-5.0, 9.0));
        raster.line((-5.0, 9.0), (-5.0, -5.0));
        assert_eq!(raster.finish(), [255; 4], "the whole grid is inside");
    }

    #[test]
    fn an_outline_is_scaled_flipped_and_offset() {
        // A unit square at the outline's origin, y up, at 2 px a unit, its origin at (1, 3).
        let mut raster = Raster::placed(4, 4, 2.0, (1.0, 3.0));
        raster.move_to(0.0, 0.0);
        raster.line_to(1.0, 0.0);
        raster.line_to(1.0, 1.0);
        raster.line_to(0.0, 1.0);
        raster.close();
        let coverage = raster.finish();
        let expected = [
            0, 0, 0, 0, //
            0, 255, 255, 0, //
            0, 255, 255, 0, //
            0, 0, 0, 0,
        ];
        assert_eq!(coverage, expected, "rows 1 and 2, columns 1 and 2");
    }

    #[test]
    fn close_joins_the_contour_back_to_its_start() {
        let mut raster = Raster::placed(2, 2, 1.0, (0.0, 2.0));
        raster.move_to(0.0, 0.0);
        raster.line_to(2.0, 0.0);
        raster.line_to(2.0, 2.0);
        raster.line_to(0.0, 2.0);
        raster.close();
        assert_eq!(raster.finish(), [255; 4], "the left edge came from close");
    }

    /// A curve cut too coarsely loses area.
    ///
    /// The area between a quadratic and its chord is 2/3 of the triangle its control points make,
    /// here half of 16 x 16. Straying 1/256 px along a curve 26 px long loses at most
    /// 2/3 x 26 / 256, about 0.07 px, and pieces cut inside the curve, so never gain any.
    #[test]
    fn a_quadratic_loses_only_what_the_tolerance_allows() {
        let mut raster = Raster::placed(16, 16, 1.0, (0.0, 16.0));
        raster.move_to(0.0, 0.0);
        raster.quad_to(16.0, 0.0, 16.0, 16.0);
        raster.close();
        let exact = 2.0 / 3.0 * 128.0;
        let found = area(&raster.finish());
        assert!(
            found <= exact + 0.1 && exact - found < 0.1,
            "{found} against {exact}"
        );
    }

    /// A dot 3 px across, as a full stop is at small sizes: four cubics of the usual circle.
    #[test]
    fn a_small_round_dot_keeps_its_area() {
        let radius = 1.5;
        let handle = radius * 0.552_284_8;
        let mut raster = Raster::placed(4, 4, 1.0, (2.0, 2.0));
        raster.move_to(radius, 0.0);
        raster.curve_to(radius, handle, handle, radius, 0.0, radius);
        raster.curve_to(-handle, radius, -radius, handle, -radius, 0.0);
        raster.curve_to(-radius, -handle, -handle, -radius, 0.0, -radius);
        raster.curve_to(handle, -radius, radius, -handle, radius, 0.0);
        raster.close();
        let exact = std::f32::consts::PI * radius * radius;
        let found = area(&raster.finish());
        assert!(
            (found - exact).abs() / exact < 0.01,
            "{found} against {exact}"
        );
    }

    /// That quadratic as a cubic, which must fill what it fills.
    #[test]
    fn a_cubic_loses_only_what_the_tolerance_allows() {
        let mut raster = Raster::placed(16, 16, 1.0, (0.0, 16.0));
        let third = 16.0 * 2.0 / 3.0;
        raster.move_to(0.0, 0.0);
        raster.curve_to(third, 0.0, 16.0, 16.0 - third, 16.0, 16.0);
        raster.close();
        let exact = 2.0 / 3.0 * 128.0;
        let found = area(&raster.finish());
        assert!(
            found <= exact + 0.1 && exact - found < 0.1,
            "{found} against {exact}"
        );
    }
}
