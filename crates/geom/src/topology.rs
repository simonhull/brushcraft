//! Base-mesh adjacency, and the maps between neighbouring faces' grids.

use std::collections::HashMap;

use crate::mesh::fit;
use crate::{Channel, FaceId, LayerId, Level, MAX_VALENCE, MeshError, Region, TileKey, VertexId};

/// What lies across one edge of a face: the neighbour and the index of the same edge in it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Across {
    /// The face across the edge.
    pub(crate) face: FaceId,
    /// The index (0–3) of the shared edge in that face.
    pub(crate) edge: u8,
}

/// The faces around one base vertex, in rotation order, starting at the owner (lowest face id).
/// Each entry is a face and the corner of it at this vertex.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Ring {
    /// The faces around the vertex with their corners at it, in rotation order, the owner
    /// first: as many as the vertex's valence.
    pub(crate) entries: Vec<(FaceId, u8)>,
}

/// How a closed quad mesh's faces fit together.
#[derive(Debug, Clone)]
pub struct Topology {
    /// Per face, what lies across each of its edges (edge `k` runs from corner `k` to
    /// `k + 1`).
    across: Vec<[Across; 4]>,
    /// Per face, the base vertex at each of its corners.
    corner_vertex: Vec<[u32; 4]>,
    /// Per base vertex, the faces around it.
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
    /// [`MeshError::LowValence`], [`MeshError::HighValence`] or [`MeshError::TooLarge`].
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

    /// The most faces around any one base vertex.
    pub(crate) fn max_valence(&self) -> usize {
        self.rings
            .iter()
            .map(|ring| ring.entries.len())
            .max()
            .unwrap_or(0)
    }

    /// The most faces any face has within `steps` steps of it, itself included, where one step
    /// reaches every face sharing a vertex with those reached so far.
    ///
    /// A breadth-first search from every face, stopping early once a search reaches the whole
    /// mesh: `O(faces × neighbourhood)`.
    pub(crate) fn widest_neighbourhood(&self, steps: u32) -> usize {
        let faces = self.corner_vertex.len();
        // Which search last reached each face and vertex, so the marks needn't be cleared
        // between searches.
        let mut face_seen = vec![usize::MAX; faces];
        let mut vertex_seen = vec![usize::MAX; self.rings.len()];
        let (mut frontier, mut next) = (Vec::new(), Vec::new());
        let mut widest = 0;
        for start in 0..faces {
            if let Some(mark) = face_seen.get_mut(start) {
                *mark = start;
            }
            frontier.clear();
            frontier.push(start);
            let mut reached = 1;
            for _ in 0..steps {
                if reached == faces {
                    break;
                }
                next.clear();
                let corners = frontier
                    .iter()
                    .filter_map(|&f| self.corner_vertex.get(f))
                    .flatten();
                for &v in corners {
                    let Ok(v) = usize::try_from(v) else { continue };
                    match vertex_seen.get_mut(v) {
                        Some(mark) if *mark != start => *mark = start,
                        _ => continue,
                    }
                    let around = self.rings.get(v).map_or(&[][..], |r| &r.entries);
                    for &(g, _) in around {
                        let Ok(g) = usize::try_from(g.index()) else {
                            continue;
                        };
                        if let Some(mark) = face_seen.get_mut(g)
                            && *mark != start
                        {
                            *mark = start;
                            next.push(g);
                            reached += 1;
                        }
                    }
                }
                std::mem::swap(&mut frontier, &mut next);
            }
            widest = widest.max(reached);
            if widest == faces {
                break;
            }
        }
        widest
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
    ///
    /// The copies are returned inline, so finding them never allocates: an edit sets every copy
    /// of each vertex it touches, so this runs once per edited vertex.
    #[must_use]
    pub fn copies(&self, face: FaceId, level: Level, i: u32, j: u32) -> Copies {
        let mut copies = Copies::default();
        let n = level.face_quads();
        if i > n || j > n {
            return copies;
        }
        for (f, fi, fj) in self.face_twins(face, n, i, j).unwrap_or_default() {
            for (tx, ty, li, lj) in tile_copies(level, fi, fj) {
                copies.0.push(VertexCopy {
                    face: f,
                    tx,
                    ty,
                    i: li,
                    j: lj,
                });
            }
        }
        copies
    }

    /// The copies of every border vertex of tile `key`, found once for the tile
    /// ([`TileBorder`]): [`Topology::copies`] for each, without its cost per call. `None` for a
    /// tile holding one of its border vertices twice (only a face glued to itself could), whose
    /// vertices then each need [`Topology::copies`].
    ///
    /// It looks up the copies of the tile's four corners and of three vertices of each edge: a
    /// few dozen lookups, against the hundreds of border vertices of a finest tile.
    #[must_use]
    pub fn tile_border(&self, key: &TileKey) -> Option<TileBorder> {
        let (level, side) = (key.level(), key.side());
        let last = side - 1;
        let region = Region::of_tile(key);
        let at =
            |(i, j): (u32, u32)| self.copies(key.face(), level, region.x0() + i, region.y0() + j);
        let held_once = |copy: &VertexCopy, (i, j): (u32, u32)| {
            copy.face != key.face()
                || copy.tx != key.tx()
                || copy.ty != key.ty()
                || (copy.i, copy.j) == (i, j)
        };
        let corners = CORNERS.map(|corner| (corner, at(corner.at(last))));
        if !corners
            .iter()
            .all(|(corner, copies)| copies.iter().all(|copy| held_once(copy, corner.at(last))))
        {
            return None;
        }
        let mut edges = [EdgeRun::EMPTY; 4];
        for (edge, run) in EDGES.into_iter().zip(&mut edges) {
            *run = EdgeRun::along(side, |t| at(edge.at(last, t)))?;
            let ends = [0, run.vertices.saturating_sub(1)];
            for t in ends.into_iter().take(run.vertices) {
                let copies = run.at(t)?;
                if !copies.iter().all(|copy| held_once(copy, edge.at(last, t))) {
                    return None;
                }
            }
        }
        Some(TileBorder {
            side,
            corners: corners.map(|(_, copies)| copies),
            edges,
        })
    }

    /// Whether tile `key` owns its vertex `(i, j)` (in the tile's own coordinates): whether, of
    /// the vertex's copies (one per tile holding it, see [`Topology::copies`]), this is the one
    /// that counts it, so that a vertex on a tile border is counted, or edited, exactly once.
    ///
    /// An interior vertex has one copy, and its tile owns it. A border vertex is owned by the
    /// least of its copies, ordered by face, tile row, tile column, then row and column within
    /// the tile: an order every copy computes alike, whichever tile asks. A vertex with no
    /// copies (outside the tile, or of a face not in this mesh) is the asking tile's own.
    ///
    /// # Examples
    ///
    /// A cube corner is held by a tile of each of the three faces meeting there, and exactly
    /// one of them owns it:
    ///
    /// ```
    /// use geom::{FaceId, Level, TileKey, cube_sphere};
    ///
    /// let mesh = cube_sphere(1)?;
    /// let topology = mesh.topology();
    /// let level = Level::new(6)?;
    /// let key = TileKey::displacement(FaceId::new(0), level, 0, 0)?;
    /// assert!(
    ///     topology.owns(&key, 5, 5),
    ///     "an interior vertex is its tile's"
    /// );
    ///
    /// let corner = topology.copies(FaceId::new(0), level, 0, 0);
    /// assert_eq!(corner.len(), 3, "three faces meet at a cube corner");
    /// let owners = corner
    ///     .iter()
    ///     .filter(|copy| topology.owns(&copy.key(level), copy.i, copy.j))
    ///     .count();
    /// assert_eq!(owners, 1, "one copy owns it");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn owns(&self, key: &TileKey, i: u32, j: u32) -> bool {
        let inner = 1..key.side() - 1;
        if inner.contains(&i) && inner.contains(&j) {
            return true;
        }
        let region = Region::of_tile(key);
        let copies = self.copies(key.face(), key.level(), region.x0() + i, region.y0() + j);
        let least = copies
            .iter()
            .map(|copy| (copy.face.index(), copy.ty, copy.tx, copy.j, copy.i))
            .min();
        least.is_none_or(|least| least == (key.face().index(), key.ty(), key.tx(), j, i))
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
    fn face_twins(&self, face: FaceId, n: u32, i: u32, j: u32) -> Option<Twins> {
        let on = |x: u32| x == 0 || x == n;
        let n64 = i64::from(n);
        let mut twins = Twins::new((face, i, j));
        if on(i) && on(j) {
            let ring = self.ring(face, corner_at(i, j, n))?;
            for &(g, c) in &ring.entries {
                let (gi, gj) = corner_coords(c, n64);
                twins.push((g, u32::try_from(gi).ok()?, u32::try_from(gj).ok()?));
            }
            return Some(twins);
        }
        if on(i) || on(j) {
            let k = border_edge(i, j, n);
            let (s, _) = grid_to_edge(k, n64, i64::from(i), i64::from(j));
            let across = self.across(face, k)?;
            let (gi, gj) = edge_to_grid(across.edge, n64, n64 - s, 0);
            twins.push((face, i, j));
            twins.push((
                across.face,
                u32::try_from(gi).ok()?,
                u32::try_from(gj).ok()?,
            ));
            return Some(twins);
        }
        self.contains(face).then(|| {
            twins.push((face, i, j));
            twins
        })
    }
}

/// The most copies one grid vertex has: one per face around a base vertex (at most
/// [`MAX_VALENCE`]), or four where a vertex lies on a tile corner (inside a face, or on a face
/// edge: two faces × two tiles).
pub const MAX_COPIES: usize = if MAX_VALENCE > 4 { MAX_VALENCE } else { 4 };

/// At most `N` values of `T`, held inline: the fixed-capacity list behind [`Copies`] and the
/// face twins it is built from.
///
/// `T` is plain data, so the unused tail is filled with a copy of a placeholder value instead of
/// being left uninitialized.
#[derive(Debug, Clone, Copy)]
struct Inline<T: Copy, const N: usize> {
    /// Values in use: a prefix of `items`.
    len: usize,
    /// The values, then placeholders.
    items: [T; N],
}

impl<T: Copy, const N: usize> Inline<T, N> {
    /// An empty list whose unused places hold `placeholder`.
    const fn new(placeholder: T) -> Self {
        Self {
            len: 0,
            items: [placeholder; N],
        }
    }

    /// Appends `value`.
    ///
    /// The capacities here are bounds the mesh enforces ([`MAX_VALENCE`] is checked when a mesh
    /// is built), so a full list is a broken invariant: debug builds panic, release builds drop
    /// the value.
    fn push(&mut self, value: T) {
        debug_assert!(self.len < N, "more than {N} copies of one vertex");
        if let Some(place) = self.items.get_mut(self.len) {
            *place = value;
            self.len += 1;
        }
    }

    /// The values in use.
    fn as_slice(&self) -> &[T] {
        self.items.get(..self.len).unwrap_or_default()
    }
}

/// The faces and face-grid positions holding one vertex, as [`Topology::face_twins`] finds them.
type Twins = Inline<(FaceId, u32, u32), MAX_VALENCE>;

impl Default for Twins {
    fn default() -> Self {
        Self::new((FaceId::new(0), 0, 0))
    }
}

impl IntoIterator for Twins {
    type IntoIter = std::iter::Take<std::array::IntoIter<(FaceId, u32, u32), MAX_VALENCE>>;
    type Item = (FaceId, u32, u32);

    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter().take(self.len)
    }
}

/// Every stored copy of one grid vertex, as [`Topology::copies`] returns them: at most
/// [`MAX_COPIES`], held inline so that finding them never allocates.
///
/// It dereferences to a slice of [`VertexCopy`], and iterates by value or by reference.
#[derive(Clone, Copy)]
pub struct Copies(Inline<VertexCopy, MAX_COPIES>);

impl Default for Copies {
    /// No copies: what a vertex outside its face has.
    fn default() -> Self {
        let placeholder = VertexCopy {
            face: FaceId::new(0),
            tx: 0,
            ty: 0,
            i: 0,
            j: 0,
        };
        Self(Inline::new(placeholder))
    }
}

impl std::ops::Deref for Copies {
    type Target = [VertexCopy];

    fn deref(&self) -> &[VertexCopy] {
        self.0.as_slice()
    }
}

impl std::fmt::Debug for Copies {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl PartialEq for Copies {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl Eq for Copies {}

impl IntoIterator for Copies {
    type IntoIter = std::iter::Take<std::array::IntoIter<VertexCopy, MAX_COPIES>>;
    type Item = VertexCopy;

    fn into_iter(self) -> Self::IntoIter {
        self.0.items.into_iter().take(self.0.len)
    }
}

impl<'a> IntoIterator for &'a Copies {
    type IntoIter = std::slice::Iter<'a, VertexCopy>;
    type Item = &'a VertexCopy;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// A corner of a tile, by which of its axes are at their last coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TileCorner {
    /// `(0, 0)`.
    First,
    /// `(last, 0)`.
    LastColumn,
    /// `(0, last)`.
    LastRow,
    /// `(last, last)`.
    Last,
}

/// The tile's corners in the order [`TileBorder`] keeps them: `(0, 0)`, `(last, 0)`,
/// `(0, last)`, `(last, last)`, so corner `(i, j)` is at `[(i == last) + 2 (j == last)]`.
const CORNERS: [TileCorner; 4] = [
    TileCorner::First,
    TileCorner::LastColumn,
    TileCorner::LastRow,
    TileCorner::Last,
];

impl TileCorner {
    /// The corner's coordinates in a tile whose last coordinate is `last`.
    const fn at(self, last: u32) -> (u32, u32) {
        match self {
            Self::First => (0, 0),
            Self::LastColumn => (last, 0),
            Self::LastRow => (0, last),
            Self::Last => (last, last),
        }
    }
}

/// An edge of a tile, between two of its corners.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TileEdge {
    /// The first row, `j = 0`.
    Bottom,
    /// The last column, `i = last`.
    Right,
    /// The last row, `j = last`.
    Top,
    /// The first column, `i = 0`.
    Left,
}

/// The tile's edges in the order [`TileBorder`] keeps them.
const EDGES: [TileEdge; 4] = [
    TileEdge::Bottom,
    TileEdge::Right,
    TileEdge::Top,
    TileEdge::Left,
];

impl TileEdge {
    /// The edge's place in [`EDGES`].
    const fn index(self) -> usize {
        match self {
            Self::Bottom => 0,
            Self::Right => 1,
            Self::Top => 2,
            Self::Left => 3,
        }
    }

    /// The coordinates of the `t`-th vertex between the edge's corners (from 0, counted from
    /// the corner with the lower coordinate along it), in a tile whose last coordinate is
    /// `last`.
    const fn at(self, last: u32, t: usize) -> (u32, u32) {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "a tile edge has fewer vertices than u32 counts"
        )]
        let along = 1 + t as u32;
        match self {
            Self::Bottom => (along, 0),
            Self::Right => (last, along),
            Self::Top => (along, last),
            Self::Left => (0, along),
        }
    }
}

/// The copies of the vertices between two corners of a tile: copy `k` of the `t`-th vertex is
/// `first[k]` moved `t` steps of `step[k]` within its tile.
///
/// Along an open tile edge nothing changes but the position: the faces holding the vertices
/// are the same (a face corner is a tile corner), and so are the tiles (tile borders lie at the
/// same multiples on both sides of a face edge, as neighbouring faces' grids meet end to end).
/// So the copies' coordinates move linearly, which [`EdgeRun::along`] checks at both ends.
#[derive(Debug, Clone, Copy)]
struct EdgeRun {
    /// The vertices between the corners: the tile's side less two.
    vertices: usize,
    /// How many copies each vertex has: at most two (two tiles of one face, or one tile of each
    /// of two faces).
    copies: usize,
    /// The copies of the first vertex; past `copies`, placeholders.
    first: [VertexCopy; 2],
    /// How each copy's coordinates move from one vertex to the next, `(di, dj)`.
    step: [(i64, i64); 2],
}

impl EdgeRun {
    /// No vertices: what the run of each edge is until it is found.
    const EMPTY: Self = Self {
        vertices: 0,
        copies: 0,
        first: [VertexCopy {
            face: FaceId::new(0),
            tx: 0,
            ty: 0,
            i: 0,
            j: 0,
        }; 2],
        step: [(0, 0); 2],
    };

    /// The run of an edge of a tile with `side` vertices a side, from `copies(t)`, the copies of
    /// the edge's `t`-th vertex. `None` unless every vertex has the same number of copies (at most
    /// two), each in the same tile for every vertex and moving linearly: checked at the first,
    /// second and last vertices.
    fn along(side: u32, copies: impl Fn(usize) -> Copies) -> Option<Self> {
        let vertices = usize::try_from(side.saturating_sub(2)).ok()?;
        if vertices == 0 {
            return Some(Self::EMPTY);
        }
        let first = copies(0);
        if first.len() > 2 {
            return None;
        }
        let mut run = Self {
            vertices,
            copies: first.len(),
            ..Self::EMPTY
        };
        for (place, copy) in run.first.iter_mut().zip(first.iter()) {
            *place = *copy;
        }
        if vertices == 1 {
            return Some(run);
        }
        let (second, end) = (copies(1), copies(vertices - 1));
        if second.len() != run.copies || end.len() != run.copies {
            return None;
        }
        let span = i64::try_from(vertices - 1).ok()?;
        for ((step, (a, b)), z) in run
            .step
            .iter_mut()
            .zip(first.iter().zip(second.iter()))
            .zip(end.iter())
        {
            let same_tile = |c: &VertexCopy| (c.face, c.tx, c.ty) == (a.face, a.tx, a.ty);
            if !same_tile(b) || !same_tile(z) {
                return None;
            }
            *step = (
                i64::from(b.i) - i64::from(a.i),
                i64::from(b.j) - i64::from(a.j),
            );
            let reaches =
                |from: u32, by: i64, to: u32| i64::from(from) + by * span == i64::from(to);
            if !reaches(a.i, step.0, z.i) || !reaches(a.j, step.1, z.j) {
                return None;
            }
        }
        Some(run)
    }

    /// The copies of the `t`-th vertex; `None` past the run.
    fn at(&self, t: usize) -> Option<BorderCopies<'_>> {
        if t >= self.vertices {
            return None;
        }
        let t = i64::try_from(t).ok()?;
        let mut copies = self.first;
        for (copy, (di, dj)) in copies.iter_mut().zip(self.step).take(self.copies) {
            copy.i = u32::try_from(i64::from(copy.i) + di * t).ok()?;
            copy.j = u32::try_from(i64::from(copy.j) + dj * t).ok()?;
        }
        Some(BorderCopies(Held::Edge {
            copies,
            len: self.copies,
        }))
    }
}

/// The copies of every border vertex of one tile, as [`Topology::tile_border`] found them.
///
/// [`TileBorder::copies`] gives each vertex's, the same as [`Topology::copies`] in the same
/// order, from a few values kept per corner and edge.
///
/// An edit sets every copy of each border vertex it moves, so finding them per vertex (each a
/// walk of the face's neighbours and two lists of inline copies) costs more than writing them;
/// a tile edited across its border finds them all here at once.
#[derive(Debug, Clone)]
pub struct TileBorder {
    /// The tile's vertices a side.
    side: u32,
    /// The copies of each corner, in [`CORNERS`] order.
    corners: [Copies; 4],
    /// The copies along each edge, in [`EDGES`] order.
    edges: [EdgeRun; 4],
}

impl TileBorder {
    /// The copies of the tile's vertex `(i, j)`, in [`Topology::copies`]'s order; `None` if it
    /// isn't on the tile's border.
    #[must_use]
    pub fn copies(&self, i: u32, j: u32) -> Option<BorderCopies<'_>> {
        let last = self.side.checked_sub(1)?;
        if i > last || j > last {
            return None;
        }
        let (end_i, end_j) = (i == 0 || i == last, j == 0 || j == last);
        if end_i && end_j {
            let corner = usize::from(i == last) + 2 * usize::from(j == last);
            return self
                .corners
                .get(corner)
                .map(|c| BorderCopies(Held::Corner(c)));
        }
        let (edge, along) = match (i, j) {
            (_, 0) => (TileEdge::Bottom, i),
            (_, j) if j == last => (TileEdge::Top, i),
            (0, _) => (TileEdge::Left, j),
            (i, _) if i == last => (TileEdge::Right, j),
            _ => return None,
        };
        self.edges
            .get(edge.index())?
            .at(usize::try_from(along.checked_sub(1)?).ok()?)
    }
}

/// The copies of one border vertex, from a [`TileBorder`]: a slice of [`VertexCopy`] through
/// `Deref`, as [`Copies`] is.
#[derive(Debug, Clone, Copy)]
pub struct BorderCopies<'a>(Held<'a>);

/// Where a [`BorderCopies`] holds its copies.
#[derive(Debug, Clone, Copy)]
enum Held<'a> {
    /// A corner's, kept by the border.
    Corner(&'a Copies),
    /// An edge vertex's, worked out from its edge's run: the first `len` of `copies`.
    Edge {
        /// The copies, then placeholders.
        copies: [VertexCopy; 2],
        /// How many are copies.
        len: usize,
    },
}

impl std::ops::Deref for BorderCopies<'_> {
    type Target = [VertexCopy];

    fn deref(&self) -> &[VertexCopy] {
        match &self.0 {
            Held::Corner(copies) => copies,
            Held::Edge { copies, len } => copies.get(..*len).unwrap_or_default(),
        }
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
/// [`MeshError::IsolatedVertex`], [`MeshError::NonManifoldVertex`], [`MeshError::LowValence`] or
/// [`MeshError::HighValence`].
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
    if entries.len() > MAX_VALENCE {
        return Err(MeshError::HighValence(vertex));
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
        t.face_twins(a.0, n, a.1, a.2)
            .unwrap()
            .as_slice()
            .contains(&b)
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
                    let lowest = twins.as_slice().iter().copied().min().unwrap();
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
    fn a_corner_of_the_highest_valence_has_every_copy_inline() {
        // Subdividing a prism with MAX_VALENCE-gon caps puts a vertex of that valence at each
        // cap's centre.
        let mesh = crate::reference::prism(MAX_VALENCE)
            .subdivide()
            .to_base_mesh();
        let t = mesh.topology();
        let level = Level::new(8).unwrap();
        let n = level.face_quads();
        let most = mesh
            .faces()
            .flat_map(|face| [(0, 0), (n, 0), (0, n), (n, n)].map(|(i, j)| (face, i, j)))
            .map(|(face, i, j)| t.copies(face, level, i, j).len())
            .max()
            .unwrap();
        assert_eq!(
            most, MAX_COPIES,
            "one copy per face around the cap's centre"
        );
        let edge = t.copies(FaceId::new(0), level, 64, 0);
        assert_eq!(
            edge.len(),
            4,
            "a face edge on a tile border: 2 faces × 2 tiles"
        );
        assert_eq!(
            edge.iter().copied().collect::<Vec<_>>(),
            edge.into_iter().collect::<Vec<_>>(),
            "by reference and by value alike"
        );
    }

    /// Asserts that [`Topology::tile_border`] of every tile of `mesh` at each of `levels` gives
    /// every border vertex exactly [`Topology::copies`]'s copies, in its order, and nothing for a
    /// vertex inside the tile.
    fn assert_tile_borders(mesh: &BaseMesh, levels: std::ops::RangeInclusive<u8>, what: &str) {
        let t = mesh.topology();
        for level in levels.map(|n| Level::new(n).unwrap()) {
            let tiles = u16::try_from(level.tiles_per_side()).unwrap();
            for face in mesh.faces() {
                for (tx, ty) in (0..tiles).flat_map(|ty| (0..tiles).map(move |tx| (tx, ty))) {
                    let key = TileKey::displacement(face, level, tx, ty).unwrap();
                    let border = t.tile_border(&key).unwrap();
                    let (side, region) = (key.side(), Region::of_tile(&key));
                    for (i, j) in (0..side).flat_map(|j| (0..side).map(move |i| (i, j))) {
                        let inside = (1..side - 1).contains(&i) && (1..side - 1).contains(&j);
                        let got = border.copies(i, j);
                        if inside {
                            assert!(got.is_none(), "{what}: {key:?} ({i}, {j}) is inside");
                            continue;
                        }
                        let want = t.copies(face, level, region.x0() + i, region.y0() + j);
                        assert_eq!(got.as_deref(), Some(&*want), "{what}: {key:?} ({i}, {j})");
                    }
                    assert!(border.copies(side, 0).is_none(), "{what}: outside the tile");
                }
            }
        }
    }

    #[test]
    fn a_tile_border_gives_each_border_vertex_its_copies() {
        assert_tile_borders(&cube_sphere(1).unwrap(), 1..=9, "the cube");
        assert_tile_borders(&cube_sphere(2).unwrap(), 1..=7, "a split cube");
        assert_tile_borders(
            &crate::fixtures::subdivided_pentagonal_prism().unwrap(),
            1..=7,
            "valence 3, 4 and 5",
        );
        assert_tile_borders(
            &crate::reference::prism(MAX_VALENCE)
                .subdivide()
                .to_base_mesh(),
            1..=7,
            "the highest valence",
        );
    }

    #[test]
    fn an_edge_run_refuses_copies_that_dont_move_alike() {
        let copy = |tx: u16, i: u32, j: u32| VertexCopy {
            face: FaceId::new(0),
            tx,
            ty: 0,
            i,
            j,
        };
        let copies = |list: &[VertexCopy]| {
            let mut copies = Copies::default();
            for &c in list {
                copies.0.push(c);
            }
            copies
        };
        // Five vertices along an edge, two copies each, moving one step a vertex.
        let linear = |t: usize| {
            let t = u32::try_from(t).unwrap();
            copies(&[copy(0, 1 + t, 0), copy(1, 1 + t, 64)])
        };
        let run = EdgeRun::along(7, linear).unwrap();
        assert_eq!(&*run.at(3).unwrap(), &*linear(3), "worked out alike");
        assert!(run.at(5).is_none(), "past the run");
        let bent = |t: usize| {
            let t = u32::try_from(t).unwrap();
            copies(&[copy(0, 1 + t * t, 0)])
        };
        assert!(EdgeRun::along(7, bent).is_none(), "not linear at its end");
        let moving = |t: usize| copies(&[copy(u16::from(t > 0), 1, 0)]);
        assert!(EdgeRun::along(7, moving).is_none(), "a copy changing tiles");
        let growing = |t: usize| {
            let list = [copy(0, 1, 0), copy(1, 1, 64)];
            copies(list.get(..1 + usize::from(t > 0)).unwrap())
        };
        assert!(
            EdgeRun::along(7, growing).is_none(),
            "a copy more on the way"
        );
        let three = |_: usize| copies(&[copy(0, 1, 0), copy(1, 1, 64), copy(2, 1, 64)]);
        assert!(EdgeRun::along(7, three).is_none(), "more than two copies");
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
