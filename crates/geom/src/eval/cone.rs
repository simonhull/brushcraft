//! Which tiles an evaluation reads: the parent tiles a prediction gathers ([`parent_tiles`]),
//! the tiles normals read ([`normal_tiles`]), and the most tiles one level of a tile's cone
//! holds ([`cone_width`]).

use std::collections::BTreeSet;

use super::gather::Reader;
use super::{EvalError, NoPositions, parent_region};
use crate::topology::edge_to_grid;
use crate::{BaseMesh, Region, TILE_QUADS, TileKey};

/// The tiles [`gather`] reads for `region`: every tile it resolves a read to, through the
/// owner copy and [`TileKey::containing`], without reading.
///
/// Only the reads that can leave the face are resolved one by one, exactly as [`gather`]
/// resolves them (with the same [`Reader`]): the points of the padded block on or past the
/// face's border, and the rings of base vertices at the block's corners. The rest, the points
/// strictly inside the face, are each their own owner copy, and [`TileKey::containing`] picks
/// a point's tile column from its column alone and its tile row from its row alone; so their
/// tiles are every pairing of the tile columns their columns fall in with the tile rows their
/// rows fall in, found from one point of each. That makes the listing cost the block's
/// perimeter, not its area: a level-9 tile's normal inputs are about 270 lookups instead of
/// 4,500. The tests check it against a recording of [`gather`]'s reads.
///
/// # Errors
///
/// [`EvalError::Missing`] where [`gather`] would miss a read with a source holding every tile
/// (a point that resolves to no tile), which no [`Region`] causes.
pub(super) fn gathered_tiles(
    mesh: &BaseMesh,
    region: Region,
) -> Result<BTreeSet<TileKey>, EvalError> {
    let reader = Reader::new(mesh, &NoPositions, region.level());
    let face = region.face();
    let n = i64::from(reader.n);
    let q = i64::from(region.quads());
    let (x0, y0) = (i64::from(region.x0()), i64::from(region.y0()));
    let mut tiles = BTreeSet::new();
    // The padded block is `-1..=q + 1` on each axis; the face's inside is `1..n`.
    let (xs, ys) = ((x0 - 1, x0 + q + 1), (y0 - 1, y0 + q + 1));
    let inner = |(a, b): (i64, i64)| (a.max(1), b.min(n - 1));
    let ((xa, xb), (ya, yb)) = (inner(xs), inner(ys));
    if xa <= xb && ya <= yb {
        let mut columns: Vec<(i64, TileKey)> = Vec::new();
        for x in xa..=xb {
            let key = reader.tile(face, x, ya)?;
            if columns
                .last()
                .is_none_or(|&(_, last)| last.tx() != key.tx())
            {
                columns.push((x, key));
            }
        }
        let mut rows: Vec<(i64, TileKey)> = Vec::new();
        for y in ya..=yb {
            let key = reader.tile(face, xa, y)?;
            if rows.last().is_none_or(|&(_, last)| last.ty() != key.ty()) {
                rows.push((y, key));
            }
        }
        for &(x, _) in &columns {
            for &(y, _) in &rows {
                tiles.insert(reader.tile(face, x, y)?);
            }
        }
    }
    // Everything else of the padded block: whole rows outside the face's inside, and in the
    // rows inside it, the columns outside.
    for y in ys.0..=ys.1 {
        let row_inside = (1..n).contains(&y);
        for x in xs.0..=xs.1 {
            if row_inside && (1..n).contains(&x) {
                continue;
            }
            if let Some(key) = reader.tile_at(face, x, y)? {
                tiles.insert(key);
            }
        }
    }
    // The rings of base vertices at the block's corners, as `Reader::corner_ring` reads them.
    for (ci, cj) in [(0, 0), (q, 0), (q, q), (0, q)] {
        let Some(ring) = reader.ring(face, (x0 + ci, y0 + cj)) else {
            continue;
        };
        for &(g, c) in &ring.entries {
            for (s, d) in [(1, 0), (1, 1)] {
                let (gi, gj) = edge_to_grid(c, n, s, d);
                tiles.insert(reader.tile(g, gi, gj)?);
            }
        }
    }
    Ok(tiles)
}

/// The tiles of `region`'s parent level that predicting `region` reads.
///
/// They are base-layer displacement keys naming each tile's position (the parent block, its
/// apron, and every face around a base vertex at the block's corners). Reads are resolved to
/// owner copies and then to [`TileKey::containing`], exactly as
/// [`predict_region`](super::predict_region) resolves them, so a source holding just these tiles
/// can predict `region`. At level 1 the parent is the base mesh: the keys are level-0 tiles, which
/// `store` serves from the mesh.
///
/// # Errors
///
/// [`EvalError::LevelZero`] or [`EvalError::NoExactParent`].
pub fn parent_tiles(mesh: &BaseMesh, region: Region) -> Result<BTreeSet<TileKey>, EvalError> {
    gathered_tiles(mesh, parent_region(region)?)
}

/// The tiles of `region`'s own level that [`normals`](super::normals) reads.
///
/// Like [`parent_tiles`], these are base-layer displacement keys naming tile positions (the block,
/// its one-vertex apron, and every face around a base vertex at the block's corners), the tiles
/// of the gather [`normals`](super::normals) runs, so a source holding just these tiles can compute
/// `region`'s normals. At level 0 the keys are level-0 tiles, which `store` serves from the mesh.
///
/// # Errors
///
/// [`EvalError::Missing`] only if a read falls outside every tile, which no [`Region`] causes.
pub fn normal_tiles(mesh: &BaseMesh, region: Region) -> Result<BTreeSet<TileKey>, EvalError> {
    gathered_tiles(mesh, region)
}

/// The widest level of any tile's dependency cone: the most tiles of one level that evaluating a
/// tile's positions, or its normals, reads, directly or through the tiles it is predicted from.
///
/// The cone of a tile is its own level's tiles to evaluate (the tile, or for its normals the
/// tiles of [`normal_tiles`]), then level by level down to level 1 every tile [`parent_tiles`]
/// lists for a tile of the level above. A cache that evaluates a tile coarse to fine, holding
/// one level's tiles while it builds the next, needs room for two such levels.
///
/// The bound is the larger of two counts, one per kind of level, both derived from the reads:
///
/// - **Levels holding several tiles per face** (finer than `log2(TILE_QUADS)`): at most
///   `max(valence, 4) + 5`. A tile's region starts and ends on even vertices, and the level below
///   reads its half, plus a one-vertex apron and the rings of the base vertices at its corners: an
///   odd range of vertices, which stays inside the tiles next to the region's own. So each level of
///   a cone lies in a 3 × 3 block of tiles around the tile, across face edges where it reaches
///   them. The reads reach at most one face corner, so at most one base vertex's ring: a face is at
///   least two tiles wide at these levels, and the range reaches only one vertex past the tile's
///   own. At a corner of valence `V`, the block's diagonal tile gives way to the corner tiles of
///   all `V` faces around the vertex: those `V`, three more of the tile's own face, and one more
///   along each of the two edges at the corner, so `V + 5` tiles. Away from corners, and at valence
///   3, the block's nine is the most.
/// - **Levels where a tile is a whole face**: [`parent_tiles`] of a face is every face sharing a
///   vertex with it, so each coarser level reaches one ring of faces further. A cone enters these
///   levels at most one ring out from its tile's face, at level `log2(TILE_QUADS)`, and reaches one
///   ring further at each level down to level 1: `log2(TILE_QUADS)` rings in all. Its widest level
///   is so at most the most faces within `log2(TILE_QUADS)` rings of any face (never more than the
///   mesh).
///
/// Both counts are reached: the tests find cones exactly this wide on meshes of valence 3 to
/// [`MAX_VALENCE`](crate::MAX_VALENCE), and on a mesh large enough that the rings don't cover it.
///
/// Computed with a breadth-first search from every face: run it once per mesh.
#[must_use]
pub fn cone_width(mesh: &BaseMesh) -> usize {
    let topology = mesh.topology();
    let face_levels = TILE_QUADS.trailing_zeros();
    let tiled = topology.max_valence().max(4) + 5;
    let whole_faces = topology.widest_neighbourhood(face_levels);
    tiled.max(whole_faces)
}
