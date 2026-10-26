//! Reading one level's positions from a source as evaluation does, from owner copies only:
//! [`Reader`], and [`gather`], which reads a region with its apron and the rings of the base
//! vertices at its corners.

use super::{EvalError, Parent, PositionSource, missing};
use crate::subdiv::{CornerRing, Padded};
use crate::topology::{Ring, Topology, corner_at, edge_to_grid};
use crate::{BaseMesh, Channel, FaceId, LayerId, Level, Region, TileKey, Vec3};

/// Reads one level's positions from a source the way evaluation does.
///
/// Every read goes to the vertex's owner copy, so all faces compute a shared vertex from the
/// same inputs even if the source's twins disagree.
///
/// Both the full gather ([`gather`]) and the partial normal update ([`refresh_normals_into`])
/// read through it, so a vertex's inputs are the same bits whichever computes it.
pub(super) struct Reader<'a, S> {
    /// The mesh's topology: owner copies, neighbours across edges, base-vertex rings.
    topology: &'a Topology,
    /// Where positions come from.
    src: &'a S,
    /// The level read.
    level: Level,
    /// Quads per face side at `level`.
    pub(super) n: u32,
}

impl<'a, S: PositionSource> Reader<'a, S> {
    /// A reader of `level` of `src` over `mesh`.
    pub(super) fn new(mesh: &'a BaseMesh, src: &'a S, level: Level) -> Self {
        Self {
            topology: mesh.topology(),
            src,
            level,
            n: level.face_quads(),
        }
    }

    /// The owner copy of face-grid vertex `(i, j)` of `f`.
    ///
    /// # Errors
    ///
    /// [`EvalError::Missing`] if `(i, j)` is off the face.
    fn owner(&self, f: FaceId, i: i64, j: i64) -> Result<(FaceId, u32, u32), EvalError> {
        let (Ok(ui), Ok(uj)) = (u32::try_from(i), u32::try_from(j)) else {
            return Err(missing(f, self.level, i, j));
        };
        self.topology
            .owner_copy(f, self.n, ui, uj)
            .ok_or_else(|| missing(f, self.level, i, j))
    }

    /// The position of face-grid vertex `(i, j)` of `f`, read from its owner copy.
    ///
    /// # Errors
    ///
    /// [`EvalError::Missing`] if `(i, j)` is off the face or the source lacks the position.
    fn fetch(&self, f: FaceId, i: i64, j: i64) -> Result<Vec3, EvalError> {
        let (g, gi, gj) = self.owner(f, i, j)?;
        self.src
            .position(g, self.level, gi, gj)
            .ok_or_else(|| missing(g, self.level, i64::from(gi), i64::from(gj)))
    }

    /// The positions of a run of row `j` of `face`, from column `i`, into `out`: vertices the
    /// caller knows are their own owner copies (strictly inside a face of the mesh), read as one
    /// [`PositionSource::row`] without the owner lookup.
    ///
    /// # Errors
    ///
    /// [`EvalError::Missing`] for the run's first vertex the source lacks (or that is off the
    /// face), as reading them one by one would report.
    fn own_row(&self, face: FaceId, i: i64, j: i64, out: &mut [Vec3]) -> Result<(), EvalError> {
        let missing = |x: i64| missing(face, self.level, x, j);
        let (Ok(ui), Ok(uj)) = (u32::try_from(i), u32::try_from(j)) else {
            return Err(missing(i));
        };
        if self.src.row(face, self.level, ui, uj, out).is_some() {
            return Ok(());
        }
        // Find which vertex was missing, in order.
        for (x, slot) in (i..).zip(out) {
            *slot = u32::try_from(x)
                .ok()
                .and_then(|ux| self.src.position(face, self.level, ux, uj))
                .ok_or_else(|| missing(x))?;
        }
        Ok(())
    }

    /// The tile a read of face-grid vertex `(i, j)` of `f` resolves to: [`TileKey::containing`]
    /// of its owner copy, the tile a source answers it from.
    ///
    /// # Errors
    ///
    /// [`EvalError::Missing`] if `(i, j)` is off the face or no tile holds it.
    pub(super) fn tile(&self, f: FaceId, i: i64, j: i64) -> Result<TileKey, EvalError> {
        let (g, gi, gj) = self.owner(f, i, j)?;
        TileKey::containing(LayerId::BASE, Channel::DISPLACEMENT, g, self.level, gi, gj)
            .map(|(key, _, _)| key)
            .ok_or_else(|| missing(g, self.level, i64::from(gi), i64::from(gj)))
    }

    /// The position at point `(i, j)` of `face`'s grid, which may lie one step off the face
    /// across an edge, or NaN where no vertex is there (diagonally off a corner): what a
    /// [`Padded`] block holds at the point.
    ///
    /// # Errors
    ///
    /// As [`Reader::fetch`].
    pub(super) fn at(&self, face: FaceId, i: i64, j: i64) -> Result<Vec3, EvalError> {
        match self.topology.locate(face, self.n, i, j) {
            Some((g, gi, gj)) => self.fetch(g, i64::from(gi), i64::from(gj)),
            None => Ok(Vec3::NAN),
        }
    }

    /// The tile [`Reader::at`] reads point `(i, j)` of `face`'s grid from, or `None` where it
    /// reads nothing.
    ///
    /// # Errors
    ///
    /// As [`Reader::tile`].
    pub(super) fn tile_at(
        &self,
        face: FaceId,
        i: i64,
        j: i64,
    ) -> Result<Option<TileKey>, EvalError> {
        self.topology
            .locate(face, self.n, i, j)
            .map(|(g, gi, gj)| self.tile(g, i64::from(gi), i64::from(gj)))
            .transpose()
    }

    /// The ring around face-grid point `(fi, fj)` of `face` if it is a base vertex (a corner of
    /// the face), with `center` as the vertex's own position, else `None`.
    ///
    /// # Errors
    ///
    /// As [`Reader::fetch`], for a ring point.
    pub(super) fn corner_ring(
        &self,
        face: FaceId,
        (fi, fj): (i64, i64),
        center: Vec3,
    ) -> Result<Option<CornerRing>, EvalError> {
        let Some(ring) = self.ring(face, (fi, fj)) else {
            return Ok(None);
        };
        let Some(&(_, owner_corner)) = ring.entries.first() else {
            return Ok(None);
        };
        let n64 = i64::from(self.n);
        let mut corner = CornerRing::new(center, owner_corner);
        for &(g, c) in &ring.entries {
            let point = |s: i64, d: i64| {
                let (gi, gj) = edge_to_grid(c, n64, s, d);
                self.fetch(g, gi, gj)
            };
            corner.push(point(1, 0)?, point(1, 1)?);
        }
        Ok(Some(corner))
    }

    /// Whether a normal at base vertex `(fi, fj)` of `face` reads a tile `stale` names: the
    /// tiles of its ring's edge points, which with the vertex itself are all a ring normal
    /// reads. `false` if the point isn't a base vertex.
    ///
    /// # Errors
    ///
    /// As [`Reader::tile`], for a ring point.
    pub(super) fn ring_reads(
        &self,
        face: FaceId,
        (fi, fj): (i64, i64),
        stale: impl Fn(&TileKey) -> bool,
    ) -> Result<bool, EvalError> {
        let Some(ring) = self.ring(face, (fi, fj)) else {
            return Ok(false);
        };
        let n64 = i64::from(self.n);
        for &(g, c) in &ring.entries {
            let (gi, gj) = edge_to_grid(c, n64, 1, 0);
            if stale(&self.tile(g, gi, gj)?) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// The topology's ring around face-grid point `(fi, fj)` of `face`, if it is a corner of
    /// the face.
    pub(super) fn ring(&self, face: FaceId, (fi, fj): (i64, i64)) -> Option<&'a Ring> {
        let n64 = i64::from(self.n);
        let on = |x: i64| x == 0 || x == n64;
        if !(on(fi) && on(fj)) {
            return None;
        }
        let (ui, uj) = (u32::try_from(fi).ok()?, u32::try_from(fj).ok()?);
        self.topology.ring(face, corner_at(ui, uj, self.n))
    }
}

/// Gathers `region` (at its own level) with its one-vertex ring and base-vertex rings.
///
/// Every position is read through one [`Reader`], as the owner's copy. The part of each row
/// strictly inside the face, its own owner copies, is read as one [`PositionSource::row`]; the
/// points on or past the face's border are located one by one. [`gathered_tiles`] lists the
/// tiles these reads resolve to, and the tests check it against a recording of them.
///
/// # Errors
///
/// [`EvalError::Missing`] if `src` lacks a needed position.
pub(crate) fn gather(
    mesh: &BaseMesh,
    src: &impl PositionSource,
    region: Region,
) -> Result<Parent, EvalError> {
    let reader = Reader::new(mesh, src, region.level());
    let q = i64::from(region.quads());
    let (x0, y0) = (i64::from(region.x0()), i64::from(region.y0()));
    let face = region.face();
    let mut padded = Padded::filled(region.quads(), Vec3::NAN);
    // A point on or past the face's border is located and read from its owner's copy.
    let located = |slots: &mut [Vec3], first_x: i64, y: i64| -> Result<(), EvalError> {
        for (x, slot) in (first_x..).zip(slots) {
            if let Some((g, gi, gj)) = reader.topology.locate(face, reader.n, x, y) {
                *slot = reader.fetch(g, i64::from(gi), i64::from(gj))?;
            }
        }
        Ok(())
    };
    // Strictly inside a face of the mesh, a point is its own owner copy: those of a row are
    // read as one run.
    let n = i64::from(reader.n);
    let (first, last) = (x0 - 1, x0 + q + 1);
    let (inner_first, inner_last) = (first.max(1), last.min(n - 1));
    let on_mesh = reader.topology.contains(face);
    for j in -1..=q + 1 {
        let y = y0 + j;
        let row = padded.row_mut(j);
        let run = usize::try_from(inner_first - first)
            .ok()
            .zip(usize::try_from(inner_last - inner_first + 1).ok())
            .filter(|_| on_mesh && (1..n).contains(&y));
        let Some((start, len)) = run else {
            located(row, first, y)?;
            continue;
        };
        let (before, rest) = row.split_at_mut(start.min(row.len()));
        let (inside, after) = rest.split_at_mut(len.min(rest.len()));
        located(before, first, y)?;
        reader.own_row(face, inner_first, y, inside)?;
        located(after, inner_last + 1, y)?;
    }
    let mut corners: [Option<CornerRing>; 4] = [None, None, None, None];
    for (slot, (ci, cj)) in corners.iter_mut().zip([(0, 0), (q, 0), (q, q), (0, q)]) {
        // The centre was already read from the owner's copy, like every other input.
        *slot = reader.corner_ring(face, (x0 + ci, y0 + cj), padded.get(ci, cj))?;
    }
    Ok(Parent { padded, corners })
}
