//! Base-mesh adjacency, and the maps between neighbouring faces' grids.

use std::collections::HashMap;

use crate::mesh::fit;
use crate::{FaceId, Level, MeshError, TileKey};

/// What lies across one edge of a face: the neighbour and the index of the same edge in it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Across {
    pub(crate) face: FaceId,
    pub(crate) edge: u8,
}

/// The faces around one base vertex, in rotation order, starting at the owner (lowest face id).
/// Each entry is a face and the corner of it at this vertex.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Ring {
    pub(crate) entries: Vec<(FaceId, u8)>,
}

/// How a closed quad mesh's faces fit together.
#[derive(Debug, Clone)]
pub struct Topology {
    across: Vec<[Across; 4]>,
    corner_vertex: Vec<[u32; 4]>,
    rings: Vec<Ring>,
}

/// One stored copy of a grid vertex: the tile holding it and the vertex's coordinates within that
/// tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VertexCopy {
    /// The face.
    pub face: FaceId,
    /// Tile column.
    pub tx: u16,
    /// Tile row.
    pub ty: u16,
    /// Column within the tile.
    pub i: u32,
    /// Row within the tile.
    pub j: u32,
}

impl VertexCopy {
    /// The base-layer displacement tile holding this copy.
    #[must_use]
    pub const fn key(&self, level: Level) -> TileKey {
        TileKey::new_unchecked(
            crate::LayerId::BASE,
            crate::Channel::DISPLACEMENT,
            self.face,
            level,
            self.tx,
            self.ty,
        )
    }
}

impl Topology {
    /// Builds adjacency for quads given as vertex indices.
    ///
    /// # Errors
    ///
    /// [`MeshError::DuplicateEdge`], [`MeshError::OpenEdge`], [`MeshError::IsolatedVertex`],
    /// [`MeshError::NonManifoldVertex`] or [`MeshError::TooLarge`].
    pub(crate) fn build(faces: &[[u32; 4]], vertex_count: u32) -> Result<Self, MeshError> {
        let mut directed: HashMap<(u32, u32), (u32, u8)> = HashMap::with_capacity(faces.len() * 4);
        for (f, quad) in faces.iter().enumerate() {
            let f: u32 = fit(f)?;
            for (k, (a, b)) in (0u8..).zip(edges(quad)) {
                if directed.insert((a, b), (f, k)).is_some() {
                    return Err(MeshError::DuplicateEdge { from: a, to: b });
                }
            }
        }
        let across = faces
            .iter()
            .map(|quad| {
                let mut out = [Across {
                    face: FaceId::new(0),
                    edge: 0,
                }; 4];
                for (slot, (a, b)) in out.iter_mut().zip(edges(quad)) {
                    let &(g, m) = directed
                        .get(&(b, a))
                        .ok_or(MeshError::OpenEdge { from: a, to: b })?;
                    *slot = Across {
                        face: FaceId::new(g),
                        edge: m,
                    };
                }
                Ok(out)
            })
            .collect::<Result<Vec<_>, MeshError>>()?;

        let vertices: usize = fit(vertex_count)?;
        let mut incident: Vec<Vec<(FaceId, u8)>> = vec![Vec::new(); vertices];
        for (f, quad) in faces.iter().enumerate() {
            let f = FaceId::new(fit(f)?);
            for (c, v) in (0u8..).zip(quad) {
                incident
                    .get_mut(fit::<usize, _>(*v)?)
                    .ok_or(MeshError::TooLarge)?
                    .push((f, c));
            }
        }
        let rings = (0..vertex_count)
            .zip(&incident)
            .map(|(v, inc)| ring(&across, v, inc))
            .collect::<Result<Vec<_>, MeshError>>()?;
        Ok(Self {
            across,
            corner_vertex: faces.to_vec(),
            rings,
        })
    }

    /// Number of faces.
    #[must_use]
    pub fn face_count(&self) -> usize {
        self.across.len()
    }

    /// What lies across edge `k` of `face`, or `None` if `face` is not in this mesh.
    pub(crate) fn across(&self, face: FaceId, k: u8) -> Option<Across> {
        self.across
            .get(usize::try_from(face.index()).ok()?)?
            .get(usize::from(k & 3))
            .copied()
    }

    /// The ring around corner `corner` of `face`.
    pub(crate) fn ring(&self, face: FaceId, corner: u8) -> Option<&Ring> {
        let v = self
            .corner_vertex
            .get(usize::try_from(face.index()).ok()?)?
            .get(usize::from(corner & 3))?;
        self.rings.get(usize::try_from(*v).ok()?)
    }

    /// The face and grid position of face-grid point `(i, j)` of `face`, where the point may be one
    /// step outside the face across one edge. `None` for points outside on both axes (diagonal
    /// of a corner) or further out, and for faces not in this mesh.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "used by owner-oriented frames (M0a Tasks 6–8)")
    )]
    #[expect(
        clippy::many_single_char_names,
        reason = "grid math keeps the conventions' names: n, (i, j) and (s, d)"
    )]
    pub(crate) fn locate(
        &self,
        face: FaceId,
        n: u32,
        i: i64,
        j: i64,
    ) -> Option<(FaceId, u32, u32)> {
        let n = i64::from(n);
        let inside = |x: i64| (0..=n).contains(&x);
        match (inside(i), inside(j)) {
            (true, true) => {
                self.across(face, 0)?;
                Some((face, u32::try_from(i).ok()?, u32::try_from(j).ok()?))
            }
            (false, false) => None,
            _ => {
                let edge = if j < 0 {
                    0
                } else if i > n {
                    1
                } else if j > n {
                    2
                } else {
                    3
                };
                let (s, d) = grid_to_edge(edge, n, i, j);
                if d != -1 {
                    return None;
                }
                let across = self.across(face, edge)?;
                let (gi, gj) = edge_to_grid(across.edge, n, n - s, -d);
                Some((
                    across.face,
                    u32::try_from(gi).ok()?,
                    u32::try_from(gj).ok()?,
                ))
            }
        }
    }

    /// Every stored copy of face-grid vertex `(i, j)` of `face` at `level`: one per tile holding
    /// it, in every face that shares it. Empty if `(i, j)` is outside the face or `face` is not
    /// in this mesh.
    #[must_use]
    pub fn copies(&self, face: FaceId, level: Level, i: u32, j: u32) -> Vec<VertexCopy> {
        let n = level.face_quads();
        if i > n || j > n {
            return Vec::new();
        }
        self.face_twins(face, n, i, j)
            .unwrap_or_default()
            .into_iter()
            .flat_map(|(f, fi, fj)| {
                tile_copies(level, fi, fj).map(move |(tx, ty, li, lj)| VertexCopy {
                    face: f,
                    tx,
                    ty,
                    i: li,
                    j: lj,
                })
            })
            .collect()
    }

    /// The same vertex in every face that holds it (itself first, unless it is a base vertex, whose
    /// ring starts at the owner). `None` if `face` is not in this mesh.
    #[expect(
        clippy::many_single_char_names,
        reason = "grid math keeps the conventions' names: n, (i, j) and (s, d)"
    )]
    fn face_twins(&self, face: FaceId, n: u32, i: u32, j: u32) -> Option<Vec<(FaceId, u32, u32)>> {
        let on = |x: u32| x == 0 || x == n;
        let n64 = i64::from(n);
        if on(i) && on(j) {
            let ring = self.ring(face, corner_at(i, j, n))?;
            return ring
                .entries
                .iter()
                .map(|&(g, c)| {
                    let (gi, gj) = corner_coords(c, n64);
                    Some((g, u32::try_from(gi).ok()?, u32::try_from(gj).ok()?))
                })
                .collect();
        }
        if on(i) || on(j) {
            let k = border_edge(i, j, n);
            let (s, _) = grid_to_edge(k, n64, i64::from(i), i64::from(j));
            let across = self.across(face, k)?;
            let (gi, gj) = edge_to_grid(across.edge, n64, n64 - s, 0);
            return Some(vec![
                (face, i, j),
                (
                    across.face,
                    u32::try_from(gi).ok()?,
                    u32::try_from(gj).ok()?,
                ),
            ]);
        }
        self.across(face, 0)?;
        Some(vec![(face, i, j)])
    }
}

/// The four directed edges of a quad, edge `k` from corner `k` to `k + 1`.
fn edges(quad: &[u32; 4]) -> impl Iterator<Item = (u32, u32)> + '_ {
    quad.iter()
        .zip(quad.iter().cycle().skip(1))
        .map(|(a, b)| (*a, *b))
}

/// Walks the fan around vertex `v` and stores it from its owner.
///
/// # Errors
///
/// [`MeshError::IsolatedVertex`] or [`MeshError::NonManifoldVertex`].
fn ring(across: &[[Across; 4]], v: u32, incident: &[(FaceId, u8)]) -> Result<Ring, MeshError> {
    let &start = incident.first().ok_or(MeshError::IsolatedVertex(v))?;
    let mut entries = Vec::with_capacity(incident.len());
    let mut at = start;
    for _ in 0..incident.len() {
        entries.push(at);
        let incoming = (at.1 + 3) & 3;
        let next = usize::try_from(at.0.index())
            .ok()
            .and_then(|f| across.get(f))
            .and_then(|a| a.get(usize::from(incoming)))
            .ok_or(MeshError::NonManifoldVertex(v))?;
        at = (next.face, next.edge);
        if at == start {
            break;
        }
    }
    if at != start || entries.len() != incident.len() {
        return Err(MeshError::NonManifoldVertex(v));
    }
    let owner = entries
        .iter()
        .enumerate()
        .min_by_key(|(_, (f, _))| *f)
        .map_or(0, |(k, _)| k);
    entries.rotate_left(owner);
    Ok(Ring { entries })
}

/// Grid position of edge parameters `(s, d)` on edge `k` of a face with `n` quads per side.
pub(crate) const fn edge_to_grid(k: u8, n: i64, s: i64, d: i64) -> (i64, i64) {
    match k & 3 {
        0 => (s, d),
        1 => (n - d, s),
        2 => (n - s, n - d),
        _ => (d, n - s),
    }
}

/// Edge parameters `(s, d)` of grid position `(i, j)` relative to edge `k`.
pub(crate) const fn grid_to_edge(k: u8, n: i64, i: i64, j: i64) -> (i64, i64) {
    match k & 3 {
        0 => (i, j),
        1 => (j, n - i),
        2 => (n - i, n - j),
        _ => (n - j, i),
    }
}

/// Grid position of corner `c`.
pub(crate) const fn corner_coords(c: u8, n: i64) -> (i64, i64) {
    match c & 3 {
        0 => (0, 0),
        1 => (n, 0),
        2 => (n, n),
        _ => (0, n),
    }
}

/// The corner at grid position `(i, j)`, which must be a corner.
pub(crate) const fn corner_at(i: u32, j: u32, n: u32) -> u8 {
    match (i == n, j == n) {
        (false, false) => 0,
        (true, false) => 1,
        (true, true) => 2,
        (false, true) => 3,
    }
}

/// The edge a non-corner border position lies on.
pub(crate) const fn border_edge(i: u32, j: u32, n: u32) -> u8 {
    if j == 0 {
        0
    } else if i == n {
        1
    } else if j == n {
        2
    } else {
        3
    }
}

/// Grid steps, in the grid of the face whose edge `k` this is, that match the neighbour's `+i` and
/// `+j`; the neighbour's edge is `m`. Used to take derivatives in the owner's orientation.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "used by owner-oriented frames (M0a Task 7)")
)]
pub(crate) const fn owner_axes(k: u8, m: u8) -> [(i64, i64); 2] {
    // The neighbour's +i and +j as (Δs', Δd') along its edge m.
    let (u_s, u_d) = match m & 3 {
        0 => (1, 0),
        1 => (0, -1),
        2 => (-1, 0),
        _ => (0, 1),
    };
    let (v_s, v_d) = match m & 3 {
        0 => (0, 1),
        1 => (1, 0),
        2 => (0, -1),
        _ => (-1, 0),
    };
    // Across the edge, s' = n − s and d' = −d, so the steps negate.
    [edge_step(k, -u_s, -u_d), edge_step(k, -v_s, -v_d)]
}

/// A step `(Δs, Δd)` along edge `k`, as a grid step.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "used by owner-oriented frames (M0a Task 7)")
)]
const fn edge_step(k: u8, ds: i64, dd: i64) -> (i64, i64) {
    match k & 3 {
        0 => (ds, dd),
        1 => (-dd, ds),
        2 => (-ds, -dd),
        _ => (dd, -ds),
    }
}

/// The tiles of `level` holding face-grid vertex `(i, j)` and its coordinates in each: one, two (on
/// a tile border) or four (on a tile corner).
fn tile_copies(level: Level, i: u32, j: u32) -> impl Iterator<Item = (u16, u16, u32, u32)> {
    let tq = level.tile_quads();
    let n = level.face_quads();
    let last = level.tiles_per_side() - 1;
    let along = move |x: u32| -> [Option<u32>; 2] {
        if x.is_multiple_of(tq) && x > 0 && x < n {
            [Some(x / tq - 1), Some(x / tq)]
        } else {
            [Some((x / tq).min(last)), None]
        }
    };
    let (xs, ys) = (along(i), along(j));
    ys.into_iter().flatten().flat_map(move |ty| {
        xs.into_iter().flatten().filter_map(move |tx| {
            Some((
                u16::try_from(tx).ok()?,
                u16::try_from(ty).ok()?,
                i - tx * tq,
                j - ty * tq,
            ))
        })
    })
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;
    use crate::{FaceId, Level, cube_sphere};

    /// Whether two face-grid positions name the same vertex. A vertex on a face border has one name
    /// per face holding it, so positions are compared through their twins, never as tuples.
    fn same_vertex(t: &Topology, n: u32, a: (FaceId, u32, u32), b: (FaceId, u32, u32)) -> bool {
        t.face_twins(a.0, n, a.1, a.2).unwrap().contains(&b)
    }

    #[test]
    fn edge_maps_round_trip() {
        for k in 0..4u8 {
            for (s, d) in [(0, 0), (3, 1), (7, 0), (2, 5)] {
                let (i, j) = edge_to_grid(k, 8, s, d);
                assert_eq!(grid_to_edge(k, 8, i, j), (s, d), "edge {k}");
            }
        }
    }

    #[test]
    fn locate_crosses_edges_and_comes_back() {
        let mesh = cube_sphere(2).unwrap();
        let t = mesh.topology();
        let n = 8;
        for face in mesh.faces() {
            for (i, j) in [(-1, 3), (9, 5), (4, -1), (2, 9)] {
                let (g, gi, gj) = t
                    .locate(face, n, i, j)
                    .expect("one step out lands in a neighbour");
                assert_ne!(g, face, "a different face");
                // One step back from the located vertex must reach f's border (f is one of its
                // faces).
                let back = (-1i64..=1)
                    .flat_map(|di| (-1i64..=1).map(move |dj| (di, dj)))
                    .filter(|(di, dj)| (di + dj).abs() == 1)
                    .filter_map(|(di, dj)| t.locate(g, n, i64::from(gi) + di, i64::from(gj) + dj))
                    .filter(|&(h, hi, hj)| {
                        t.face_twins(h, n, hi, hj)
                            .unwrap()
                            .iter()
                            .any(|(f, _, _)| *f == face)
                    })
                    .count();
                assert!(
                    back > 0,
                    "a neighbour of the located vertex is back on {face:?}"
                );
            }
            assert_eq!(
                t.locate(face, n, -1, -1),
                None,
                "diagonal corner: no single answer"
            );
        }
    }

    #[test]
    fn copies_agree_across_faces_and_tiles() {
        let mesh = cube_sphere(1).unwrap();
        let t = mesh.topology();
        let level = Level::new(7).unwrap(); // 2×2 tiles of 64 per face
        let face = FaceId::new(0);
        assert_eq!(t.copies(face, level, 30, 40).len(), 1, "interior: one copy");
        assert_eq!(
            t.copies(face, level, 64, 40).len(),
            2,
            "tile border inside a face: two tiles"
        );
        assert_eq!(
            t.copies(face, level, 64, 64).len(),
            4,
            "tile corner: four tiles"
        );
        assert_eq!(
            t.copies(face, level, 0, 40).len(),
            2,
            "face edge: two faces"
        );
        assert_eq!(
            t.copies(face, level, 0, 64).len(),
            4,
            "face edge on a tile border: 2 faces × 2 tiles"
        );
        assert_eq!(
            t.copies(face, level, 0, 0).len(),
            3,
            "cube corner: valence 3"
        );
    }

    #[test]
    #[expect(
        clippy::many_single_char_names,
        reason = "grid math keeps the conventions' names: n, (i, j), (s, d) and the axes u, v"
    )]
    fn owner_axes_point_where_the_owner_steps() {
        let mesh = cube_sphere(2).unwrap();
        let t = mesh.topology();
        let n: u32 = 8;
        for face in mesh.faces() {
            for k in 0..4u8 {
                let across = t.across(face, k).unwrap();
                let (s, d) = (3, 0);
                let (i, j) = edge_to_grid(k, i64::from(n), s, d);
                let (gi, gj) = edge_to_grid(across.edge, i64::from(n), i64::from(n) - s, -d);
                let [u, v] = owner_axes(k, across.edge);
                // f's step along the owner's +i lands on g's (gi+1, gj), and +j on (gi, gj+1).
                let step_u = t.locate(face, n, i + u.0, j + u.1).unwrap();
                let step_v = t.locate(face, n, i + v.0, j + v.1).unwrap();
                let want_u = t.locate(across.face, n, gi + 1, gj).unwrap();
                let want_v = t.locate(across.face, n, gi, gj + 1).unwrap();
                assert!(
                    same_vertex(t, n, step_u, want_u),
                    "owner +i, face {face:?} edge {k}: {step_u:?} vs {want_u:?}"
                );
                assert!(
                    same_vertex(t, n, step_v, want_v),
                    "owner +j, face {face:?} edge {k}: {step_v:?} vs {want_v:?}"
                );
            }
        }
    }
}
