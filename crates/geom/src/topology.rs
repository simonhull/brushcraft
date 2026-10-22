//! Base-mesh adjacency, and the maps between neighbouring faces' grids.

use std::collections::HashMap;

use crate::mesh::fit;
use crate::{Channel, FaceId, LayerId, Level, MeshError, TileKey, VertexId};

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
    /// The tile of `layer`'s `channel` at `level` holding this copy.
    #[must_use]
    pub const fn key_in(&self, layer: LayerId, channel: Channel, level: Level) -> TileKey {
        TileKey::new_unchecked(layer, channel, self.face, level, self.tx, self.ty)
    }

    /// The base-layer displacement tile holding this copy.
    #[must_use]
    pub const fn key(&self, level: Level) -> TileKey {
        self.key_in(LayerId::BASE, Channel::DISPLACEMENT, level)
    }
}

impl Topology {
    /// Builds adjacency for quads given as vertex indices.
    ///
    /// # Errors
    ///
    /// [`MeshError::DuplicateEdge`], [`MeshError::OpenEdge`], [`MeshError::VertexOutOfRange`],
    /// [`MeshError::IsolatedVertex`], [`MeshError::NonManifoldVertex`],
    /// [`MeshError::LowValence`] or [`MeshError::TooLarge`].
    pub(crate) fn build(faces: &[[u32; 4]], vertex_count: u32) -> Result<Self, MeshError> {
        let mut directed: HashMap<(u32, u32), (u32, u8)> = HashMap::with_capacity(faces.len() * 4);
        for (f, quad) in faces.iter().enumerate() {
            let f: u32 = fit(f)?;
            for (k, (a, b)) in (0u8..).zip(edges(quad)) {
                if directed.insert((a, b), (f, k)).is_some() {
                    return Err(MeshError::DuplicateEdge {
                        from: VertexId::new(a),
                        to: VertexId::new(b),
                    });
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
                    let &(g, m) = directed.get(&(b, a)).ok_or_else(|| MeshError::OpenEdge {
                        from: VertexId::new(a),
                        to: VertexId::new(b),
                    })?;
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
        for (face, quad) in faces.iter().enumerate() {
            let f = FaceId::new(fit(face)?);
            for (c, &v) in (0u8..).zip(quad) {
                // `BaseMesh::from_polygons` already checked the indices; this guards direct
                // callers.
                incident
                    .get_mut(fit::<usize, _>(v)?)
                    .ok_or_else(|| MeshError::VertexOutOfRange {
                        face,
                        vertex: VertexId::new(v),
                        count: vertices,
                    })?
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
    pub fn face_count(&self) -> u32 {
        // `build` checked that every face index fits a `u32`.
        u32::try_from(self.across.len()).unwrap_or(u32::MAX)
    }

    /// Whether `face` is a face of this mesh.
    #[must_use]
    pub fn contains(&self, face: FaceId) -> bool {
        face.index() < self.face_count()
    }

    /// What lies across edge `k` (0–3) of `face`, or `None` if either is out of range.
    pub(crate) fn across(&self, face: FaceId, k: u8) -> Option<Across> {
        self.across
            .get(usize::try_from(face.index()).ok()?)?
            .get(usize::from(k))
            .copied()
    }

    /// The ring around corner `corner` (0–3) of `face`, or `None` if either is out of range.
    pub(crate) fn ring(&self, face: FaceId, corner: u8) -> Option<&Ring> {
        let v = self
            .corner_vertex
            .get(usize::try_from(face.index()).ok()?)?
            .get(usize::from(corner))?;
        self.rings.get(usize::try_from(*v).ok()?)
    }

    /// The face and grid position of face-grid point `(i, j)` of `face`, where the point may be one
    /// step outside the face across one edge. `None` for points outside on both axes (diagonal
    /// of a corner) or further out, and for faces not in this mesh.
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
            (true, true) => self.contains(face).then_some((
                face,
                u32::try_from(i).ok()?,
                u32::try_from(j).ok()?,
            )),
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

    /// The owner's copy of face-grid vertex `(i, j)` of `face` (`n` quads per side): the lowest
    /// face id holding it, and the lowest position within that face if the face holds it twice.
    /// Every name of a vertex maps to the same copy, so reading positions through it never
    /// depends on a source storing identical twins. `None` if `(i, j)` is outside the face or
    /// `face` is not in this mesh.
    #[expect(
        clippy::many_single_char_names,
        reason = "grid math keeps the conventions' names: n, (i, j) and (s, d)"
    )]
    pub(crate) fn owner_copy(
        &self,
        face: FaceId,
        n: u32,
        i: u32,
        j: u32,
    ) -> Option<(FaceId, u32, u32)> {
        if i > n || j > n || !self.contains(face) {
            return None;
        }
        let on = |x: u32| x == 0 || x == n;
        let n64 = i64::from(n);
        if on(i) && on(j) {
            // Rings start at the owner.
            let &(g, c) = self.ring(face, corner_at(i, j, n))?.entries.first()?;
            let (gi, gj) = corner_coords(c, n64);
            return Some((g, u32::try_from(gi).ok()?, u32::try_from(gj).ok()?));
        }
        if on(i) || on(j) {
            let k = border_edge(i, j, n);
            let across = self.across(face, k)?;
            if across.face > face {
                return Some((face, i, j));
            }
            let (s, _) = grid_to_edge(k, n64, i64::from(i), i64::from(j));
            let (gi, gj) = edge_to_grid(across.edge, n64, n64 - s, 0);
            let twin = (
                across.face,
                u32::try_from(gi).ok()?,
                u32::try_from(gj).ok()?,
            );
            return Some(twin.min((face, i, j)));
        }
        Some((face, i, j))
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
        self.contains(face).then(|| vec![(face, i, j)])
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
/// [`MeshError::IsolatedVertex`], [`MeshError::NonManifoldVertex`] or [`MeshError::LowValence`].
fn ring(across: &[[Across; 4]], v: u32, incident: &[(FaceId, u8)]) -> Result<Ring, MeshError> {
    let vertex = VertexId::new(v);
    let &start = incident.first().ok_or(MeshError::IsolatedVertex(vertex))?;
    let mut entries = Vec::with_capacity(incident.len());
    let mut at = start;
    for _ in 0..incident.len() {
        entries.push(at);
        let incoming = (at.1 + 3) & 3;
        let next = usize::try_from(at.0.index())
            .ok()
            .and_then(|f| across.get(f))
            .and_then(|a| a.get(usize::from(incoming)))
            .ok_or(MeshError::NonManifoldVertex(vertex))?;
        at = (next.face, next.edge);
        if at == start {
            break;
        }
    }
    if at != start || entries.len() != incident.len() {
        return Err(MeshError::NonManifoldVertex(vertex));
    }
    if entries.len() < 3 {
        return Err(MeshError::LowValence(vertex));
    }
    let owner = entries
        .iter()
        .enumerate()
        .min_by_key(|(_, (f, _))| *f)
        .map_or(0, |(k, _)| k);
    // `incident` is filled in face order, so the walk already starts at the owner and this is a
    // no-op today; it keeps the canonical order independent of how `incident` was built.
    entries.rotate_left(owner);
    Ok(Ring { entries })
}

/// Grid position of edge parameters `(s, d)` on edge `k` of a face with `n` quads per side.
pub(crate) const fn edge_to_grid(k: u8, n: i64, s: i64, d: i64) -> (i64, i64) {
    debug_assert!(k < 4, "an edge index is 0–3");
    match k & 3 {
        0 => (s, d),
        1 => (n - d, s),
        2 => (n - s, n - d),
        _ => (d, n - s),
    }
}

/// Edge parameters `(s, d)` of grid position `(i, j)` relative to edge `k`.
pub(crate) const fn grid_to_edge(k: u8, n: i64, i: i64, j: i64) -> (i64, i64) {
    debug_assert!(k < 4, "an edge index is 0–3");
    match k & 3 {
        0 => (i, j),
        1 => (j, n - i),
        2 => (n - i, n - j),
        _ => (n - j, i),
    }
}

/// Grid position of corner `c`.
pub(crate) const fn corner_coords(c: u8, n: i64) -> (i64, i64) {
    debug_assert!(c < 4, "a corner index is 0–3");
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
pub(crate) const fn owner_axes(k: u8, m: u8) -> [(i64, i64); 2] {
    debug_assert!(k < 4 && m < 4, "edge indices are 0–3");
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
const fn edge_step(k: u8, ds: i64, dd: i64) -> (i64, i64) {
    debug_assert!(k < 4, "an edge index is 0–3");
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
    use crate::{BaseMesh, FaceId, Level, cube_sphere};

    /// Whether two face-grid positions name the same vertex. A vertex on a face border has one name
    /// per face holding it, so positions are compared through their twins, never as tuples.
    fn same_vertex(t: &Topology, n: u32, a: (FaceId, u32, u32), b: (FaceId, u32, u32)) -> bool {
        t.face_twins(a.0, n, a.1, a.2).unwrap().contains(&b)
    }

    #[test]
    fn every_twin_has_the_same_owner_copy() {
        let mesh = cube_sphere(2).unwrap();
        let t = mesh.topology();
        let n = 4;
        for face in mesh.faces() {
            for j in 0..=n {
                for i in 0..=n {
                    let twins = t.face_twins(face, n, i, j).unwrap();
                    let lowest = twins.iter().copied().min().unwrap();
                    for (g, gi, gj) in twins {
                        assert_eq!(
                            t.owner_copy(g, n, gi, gj),
                            Some(lowest),
                            "{face:?} ({i}, {j}) via {g:?}"
                        );
                    }
                }
            }
        }
        assert_eq!(
            t.owner_copy(FaceId::new(0), n, n + 1, 0),
            None,
            "outside the face"
        );
        assert_eq!(
            t.owner_copy(FaceId::new(24), n, 1, 1),
            None,
            "not a face of this mesh"
        );
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

    /// Where face-grid point `(i, j)` of `face` lies on the bilinear patch through its corners.
    ///
    /// An `f64` oracle independent of the edge maps: two names of one vertex land on the same
    /// point (within rounding), because adjacent faces share their edge's two corners.
    #[expect(
        clippy::many_single_char_names,
        reason = "grid math keeps the conventions' names: n, (i, j) and (u, v)"
    )]
    fn bilinear(mesh: &BaseMesh, n: u32, face: FaceId, i: u32, j: u32) -> [f64; 3] {
        let corner = |c: u8| {
            let p = mesh.corner_position(face, c).unwrap();
            [p.x, p.y, p.z].map(f64::from)
        };
        let (u, v) = (f64::from(i) / f64::from(n), f64::from(j) / f64::from(n));
        let weights = [(1.0 - u) * (1.0 - v), u * (1.0 - v), u * v, (1.0 - u) * v];
        (0u8..).zip(weights).fold([0.0; 3], |acc, (c, w)| {
            let p = corner(c);
            [acc[0] + w * p[0], acc[1] + w * p[1], acc[2] + w * p[2]]
        })
    }

    fn same_point(a: [f64; 3], b: [f64; 3]) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-12)
    }

    #[test]
    fn locate_lands_one_step_inside_the_neighbour() {
        let mesh = cube_sphere(2).unwrap();
        let t = mesh.topology();
        let n = 8;
        for face in mesh.faces() {
            for (i, j) in [(-1, 3), (9, 5), (4, -1), (2, 9)] {
                let (g, gi, gj) = t
                    .locate(face, n, i, j)
                    .expect("one step out lands in a neighbour");
                assert_ne!(g, face, "a different face");
                // The point on f's border that (i, j) steps out from.
                let clamp = |x: i64| u32::try_from(x.clamp(0, i64::from(n))).unwrap();
                let border = bilinear(&mesh, n, face, clamp(i), clamp(j));
                let located = bilinear(&mesh, n, g, gi, gj);
                assert!(
                    !same_point(located, border),
                    "{face:?} ({i}, {j}): not on the shared edge"
                );
                // Exactly one grid step back from the located vertex reaches that border point.
                let back = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    .into_iter()
                    .filter_map(|(di, dj)| {
                        Some((gi.checked_add_signed(di)?, gj.checked_add_signed(dj)?))
                    })
                    .filter(|&(hi, hj)| hi <= n && hj <= n)
                    .filter(|&(hi, hj)| same_point(bilinear(&mesh, n, g, hi, hj), border))
                    .count();
                assert_eq!(
                    back, 1,
                    "{face:?} ({i}, {j}) → {g:?} ({gi}, {gj}): one step back to f's border"
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
        let (n, tq) = (level.face_quads(), level.tile_quads());
        let cases = [
            ((30, 40), 1, "interior: one copy"),
            ((64, 40), 2, "tile border inside a face: two tiles"),
            ((64, 64), 4, "tile corner: four tiles"),
            ((0, 40), 2, "face edge: two faces"),
            ((0, 64), 4, "face edge on a tile border: 2 faces × 2 tiles"),
            (
                (128, 64),
                4,
                "far face edge on a tile border: 2 faces × 2 tiles",
            ),
            ((0, 0), 3, "cube corner: valence 3"),
            ((128, 128), 3, "far cube corner: valence 3"),
        ];
        for face in mesh.faces() {
            for ((i, j), count, what) in cases {
                let want = bilinear(&mesh, n, face, i, j);
                let copies = t.copies(face, level, i, j);
                assert_eq!(copies.len(), count, "{face:?} ({i}, {j}): {what}");
                let mut tiles: Vec<_> = copies.iter().map(|c| (c.face, c.tx, c.ty)).collect();
                tiles.sort_unstable();
                tiles.dedup();
                assert_eq!(tiles.len(), count, "{face:?} ({i}, {j}): one copy per tile");
                for c in &copies {
                    assert!(c.i <= tq && c.j <= tq, "{c:?}: inside its tile");
                    let (fi, fj) = (u32::from(c.tx) * tq + c.i, u32::from(c.ty) * tq + c.j);
                    assert!(
                        same_point(bilinear(&mesh, n, c.face, fi, fj), want),
                        "{face:?} ({i}, {j}): {c:?} is a different point ({what})"
                    );
                }
            }
        }
        assert!(
            t.copies(FaceId::new(0), level, 129, 0).is_empty(),
            "outside the face"
        );
        assert!(
            t.copies(FaceId::new(6), level, 1, 1).is_empty(),
            "not a face of this mesh"
        );
    }

    #[test]
    fn copy_keys_name_any_layer_and_channel() {
        let mesh = cube_sphere(1).unwrap();
        let level = Level::new(8).unwrap();
        let (layer, channel) = (LayerId::new(3), Channel::from_raw(5));
        let copies = mesh.topology().copies(FaceId::new(2), level, 64, 0);
        assert_eq!(
            copies.len(),
            4,
            "face edge on a tile border: 2 faces × 2 tiles"
        );
        for c in copies {
            let base = c.key(level);
            assert_eq!(
                base,
                c.key_in(LayerId::BASE, Channel::DISPLACEMENT, level),
                "key is key_in for the base layer's displacement"
            );
            let other = c.key_in(layer, channel, level);
            assert_eq!(
                (other.layer(), other.channel()),
                (layer, channel),
                "{c:?}: layer and channel"
            );
            assert_eq!(
                (other.face(), other.level(), other.tx(), other.ty()),
                (base.face(), base.level(), base.tx(), base.ty()),
                "{c:?}: same tile position"
            );
        }
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
