//! Normals of evaluated positions: in full, into a buffer, and brought up to date where only
//! some inputs changed.

use super::gather::Reader;
use super::{EvalError, Grid, PositionSource, block_of, check_buffer, gather};
use crate::frame::{self, CornerAxes, FrameRows, Scattered, Stencil};
use crate::{BaseMesh, Region, TileKey, Vec3};

/// Unit normals of `region`'s evaluated positions in `src` (same level).
///
/// Every copy of a shared vertex gets the same bits. Away from base vertices the normal is
/// `du × dv` from central differences in the owner's orientation. At a base vertex `v` (a face
/// corner) with edge neighbours `e_0 … e_{n−1}` at this level, it is the ring normal
/// `Σ_k (e_k − v) × (e_{k+1} − v)`, summed over the whole ring from the owner face, wrapping.
/// [`Prediction`](super::Prediction)'s frames use the same rules.
///
/// Non-finite positions give NaN normals. Finite but degenerate ones (a zero cross product or
/// ring sum) give the fallback normal `(0, 0, 1)`, as frames do.
///
/// Like prediction, this reads the region, its one-vertex apron and the rings of the base
/// vertices at its corners, so all must be resident in `src`. At valence 5 or more, a ring
/// includes faces that touch only the vertex, outside the apron.
///
/// # Errors
///
/// [`EvalError::Missing`] if `src` lacks a position of the region, its apron or a corner's ring.
pub fn normals(
    mesh: &BaseMesh,
    src: &impl PositionSource,
    region: Region,
) -> Result<Grid, EvalError> {
    let side = region.quads() + 1;
    let count = usize::try_from(side * side).unwrap_or(usize::MAX);
    let mut data = vec![Vec3::ZERO; count];
    normals_into(mesh, src, region, &mut data)?;
    Ok(Grid::from_parts(side, data))
}

/// [`normals`] into `out`, row-major `(quads + 1)²`.
///
/// # Errors
///
/// [`EvalError::BufferSize`] unless `out` holds exactly `(quads + 1)²` values, or
/// [`EvalError::Missing`] as [`normals`].
pub fn normals_into(
    mesh: &BaseMesh,
    src: &impl PositionSource,
    region: Region,
    out: &mut [Vec3],
) -> Result<(), EvalError> {
    let side = region.quads() + 1;
    check_buffer(out, usize::try_from(side * side).unwrap_or(usize::MAX))?;
    let gathered = gather(mesh, src, region)?;
    // The level's own ring: the base vertex (the owner's copy) and its edge neighbours.
    let axes = gathered.corners.each_ref().map(|c| {
        c.as_ref().map(|ring| CornerAxes {
            center: ring.center,
            points: ring.edges(),
            owner_corner: ring.owner_corner,
        })
    });
    let block = block_of(mesh, region);
    frame::normals(&block, &gathered.padded, &axes, out);
    Ok(())
}

/// Brings `region`'s normals in `out` up to date after positions outside the region changed.
///
/// Recomputes, in place, the normal of every vertex whose inputs include a position read from a
/// tile `stale` names, and returns how many it recomputed.
///
/// `out` must hold the normals [`normals_into`] gave for the region from earlier positions
/// that differ from `src`'s only in tiles `stale` names, and never at a vertex of the region
/// itself. That is the case after an edit that moved vertices of neighbouring tiles but none of
/// the region's own: a brush's neighbours, whose positions didn't change. Every normal is then
/// the same bits a full [`normals_into`] from `src` gives.
///
/// Only the region's border can read a changed position. Inside, a vertex's central
/// differences read its four grid neighbours, all vertices of the region; on the border they
/// reach one vertex past it, into the apron, and at a base vertex (a face corner) the normal
/// reads the vertex's whole ring, which at valence 5 or more reaches faces that touch the
/// region only at the vertex. So the border vertices are visited, and each is recomputed if a
/// read outside the region, or a ring point, resolves (through its owner copy, as every read
/// does) to a tile `stale` names. Reads are resolved to [`TileKey::containing`]'s tile, as
/// [`PositionSource`] answers them.
///
/// Only the positions of the vertices recomputed are read: their four neighbours, or their
/// ring. Those must be resident in `src`.
///
/// # Errors
///
/// [`EvalError::BufferSize`] unless `out` holds exactly `(quads + 1)²` values, or
/// [`EvalError::Missing`] if `src` lacks a position a recomputed normal reads.
pub fn refresh_normals_into(
    mesh: &BaseMesh,
    src: &impl PositionSource,
    region: Region,
    stale: impl Fn(&TileKey) -> bool,
    out: &mut [Vec3],
) -> Result<usize, EvalError> {
    let side = region.quads() + 1;
    check_buffer(out, usize::try_from(side * side).unwrap_or(usize::MAX))?;
    let reader = Reader::new(mesh, src, region.level());
    let block = block_of(mesh, region);
    let face = region.face();
    let q = i64::from(region.quads());
    let (x0, y0) = (i64::from(region.x0()), i64::from(region.y0()));
    let inside = |(i, j): (i64, i64)| (0..=q).contains(&i) && (0..=q).contains(&j);
    // The stale vertices, computed together at the end as the full normals compute theirs.
    let mut stale_vertices = Scattered::with_capacity(4 * usize::try_from(side).unwrap_or(0));
    for (i, j) in border(q) {
        let local = match frame::stencil(&block, i, j) {
            Stencil::Axes([u, v]) => {
                let reads = [
                    (i + u.0, j + u.1),
                    (i - u.0, j - u.1),
                    (i + v.0, j + v.1),
                    (i - v.0, j - v.1),
                ];
                let mut changed = false;
                for (a, b) in reads.into_iter().filter(|&read| !inside(read)) {
                    if reader
                        .tile_at(face, x0 + a, y0 + b)?
                        .is_some_and(|key| stale(&key))
                    {
                        changed = true;
                        break;
                    }
                }
                if !changed {
                    continue;
                }
                let mut values = [Vec3::NAN; 4];
                for (value, (a, b)) in values.iter_mut().zip(reads) {
                    *value = reader.at(face, x0 + a, y0 + b)?;
                }
                frame::Local::Central(values)
            }
            Stencil::Corner(_) => {
                let at = (x0 + i, y0 + j);
                if !reader.ring_reads(face, at, &stale)? {
                    continue;
                }
                let center = reader.at(face, at.0, at.1)?;
                match reader.corner_ring(face, at, center)? {
                    Some(ring) => CornerAxes {
                        center: ring.center,
                        points: ring.edges(),
                        owner_corner: ring.owner_corner,
                    }
                    .local(),
                    None => frame::Local::NonFinite,
                }
            }
            Stencil::Outside => continue,
        };
        let index = usize::try_from(j * i64::from(side) + i).unwrap_or(usize::MAX);
        if index < out.len() {
            stale_vertices.push(index, local);
        }
    }
    stale_vertices.normals_into(&mut FrameRows::default(), out);
    Ok(stale_vertices.len())
}

/// The border vertices of a block of `q` quads per side, each once: the bottom and top rows,
/// then the left and right columns between them.
fn border(q: i64) -> impl Iterator<Item = (i64, i64)> {
    let rows = [0, q]
        .into_iter()
        .flat_map(move |j| (0..=q).map(move |i| (i, j)));
    let columns = (1..q).flat_map(move |j| [(0, j), (q, j)]);
    rows.chain(columns)
}
