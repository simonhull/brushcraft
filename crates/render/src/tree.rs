//! The tree of resident tiles, kept on the CPU and mirrored into a GPU buffer the level choice
//! walks (plan decision 3).
//!
//! Every tile resident on the GPU, display tile or finest tile, is one **node** of the tree, at
//! a node index that stays the same while the tile stays resident. A node links to its parent,
//! its children and its four edge neighbours by topology (across face borders too), each by
//! node index, or [`NONE`] where that tile isn't resident. The links follow the tiles' places on
//! their faces ([`Place`]), whichever channel holds them: a display tile's children one level
//! below the finest are finest tiles.
//!
//! The tree is updated incrementally: a tile arriving, leaving or moving to another slot marks
//! its own node and the nodes that link to it, and only those are written to the GPU, through
//! the frame's uploads ([`crate::upload`]), in the same submission as the tile data they point
//! at.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use bytemuck::{Pod, Zeroable};
use geom::{BaseMesh, FaceId, Level, TileKey, Topology};
use thiserror::Error;
use wgpu::{Buffer, BufferDescriptor, BufferUsages};

use crate::gpu::{Gpu, GpuError};
use crate::residency::Slot;

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;

/// The node index that stands for "no node": a link to a tile that isn't resident.
pub const NONE: u32 = u32::MAX;

/// [`GpuNode::flags`]: the node holds a resident tile. A node without it is empty, and every
/// other field of it is meaningless.
pub const NODE_LIVE: u32 = 1;
/// [`GpuNode::flags`]: the tile is part of the pinned floor, never evicted.
pub const NODE_FLOOR: u32 = 1 << 1;
/// [`GpuNode::flags`]: the tile is a finest tile (evaluated positions), not a display tile.
pub const NODE_FINEST: u32 = 1 << 2;

/// The WGSL declaration of [`GpuNode`], for the passes that walk the tree: `TileNode`, with the
/// same fields in the same order, and the constants above.
pub const NODE_WGSL: &str = r"
// One resident tile of the tile tree (render::tree::GpuNode): 64 bytes.
struct TileNode {
    // The base-mesh face the tile lies on.
    face: u32,
    // Its level (bits 28..32), column (bits 14..28) and row (bits 0..14) on the face.
    place: u32,
    // Its slot: size class (bits 29..32), buffer (bits 21..29), index in the buffer (0..21).
    slot: u32,
    // NODE_LIVE, NODE_FLOOR, NODE_FINEST.
    flags: u32,
    // The parent's node index, or NONE.
    parent: u32,
    // e: a bound, in world units, on the distance from what the tile draws to the finest
    // surface under it (0 for a finest tile).
    error: f32,
    // Zero.
    reserved0: u32,
    // Zero.
    reserved1: u32,
    // The children's node indices, NONE where not resident: (2tx, 2ty), (2tx + 1, 2ty),
    // (2tx, 2ty + 1), (2tx + 1, 2ty + 1); a whole-face tile's one child first, the rest NONE.
    children: array<u32, 4>,
    // The edge neighbours' node indices, NONE where not resident: across the tile's edge at
    // j = 0, at i = side - 1, at j = side - 1 and at i = 0, in that order.
    neighbours: array<u32, 4>,
}

// A link to a tile that isn't resident.
const NONE: u32 = 0xffffffffu;
// TileNode.flags: the node holds a resident tile.
const NODE_LIVE: u32 = 1u;
// TileNode.flags: the tile is part of the pinned floor.
const NODE_FLOOR: u32 = 2u;
// TileNode.flags: the tile is a finest tile.
const NODE_FINEST: u32 = 4u;
";

/// Bits of [`Place::packed`] holding the row; the column sits above them, the level above that.
const PLACE_AXIS_BITS: u32 = 14;

/// Where a tile sits in the hierarchy of a face: its face, level, column and row, whichever
/// channel holds it (the display layer below the finest level, evaluated positions at it).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Place {
    /// The face.
    face: FaceId,
    /// The level.
    level: Level,
    /// The column in the face's tile grid at `level`.
    tx: u16,
    /// The row in the face's tile grid at `level`.
    ty: u16,
}

impl Place {
    /// Where `key` sits.
    #[must_use]
    pub const fn of(key: &TileKey) -> Self {
        Self {
            face: key.face(),
            level: key.level(),
            tx: key.tx(),
            ty: key.ty(),
        }
    }

    /// The tile at this place in a project whose finest level is `finest`: a display tile
    /// below it, the finest tile at it, `None` above it (or off the face's grid).
    #[must_use]
    pub fn key(self, finest: Level) -> Option<TileKey> {
        match self.level.cmp(&finest) {
            Ordering::Less => TileKey::display(self.face, self.level, self.tx, self.ty).ok(),
            Ordering::Equal => TileKey::displacement(self.face, self.level, self.tx, self.ty).ok(),
            Ordering::Greater => None,
        }
    }

    /// The face.
    #[must_use]
    pub const fn face(self) -> FaceId {
        self.face
    }

    /// The level.
    #[must_use]
    pub const fn level(self) -> Level {
        self.level
    }

    /// The next coarser tile covering this one, or `None` at level 0.
    #[must_use]
    pub const fn parent(self) -> Option<Self> {
        let Some(level) = self.level.parent() else {
            return None;
        };
        let split = level.tiles_per_side() != self.level.tiles_per_side();
        Some(Self {
            face: self.face,
            level,
            tx: if split { self.tx / 2 } else { self.tx },
            ty: if split { self.ty / 2 } else { self.ty },
        })
    }

    /// The tiles one level finer covering this one, in the order of [`GpuNode::children`]: four
    /// where the finer level has twice the tiles per side, else the one tile first and `None`
    /// after it; all `None` at [`geom::MAX_LEVEL`].
    #[must_use]
    pub fn children(self) -> [Option<Self>; 4] {
        let Some(level) = self.level.child() else {
            return [None; 4];
        };
        let at = |tx: u16, ty: u16| {
            Some(Self {
                face: self.face,
                level,
                tx,
                ty,
            })
        };
        if level.tiles_per_side() == self.level.tiles_per_side() {
            return [at(self.tx, self.ty), None, None, None];
        }
        let (x, y) = (self.tx * 2, self.ty * 2);
        [at(x, y), at(x + 1, y), at(x, y + 1), at(x + 1, y + 1)]
    }

    /// The four tiles across this tile's edges at the same level, in the order of
    /// [`GpuNode::neighbours`], found through `topology` (across face borders too); `None`
    /// across an open border.
    ///
    /// Each is the other tile holding the midpoint of that edge (at level 1 for a level-0 tile,
    /// whose edges have no midpoint vertex of their own).
    #[must_use]
    pub fn neighbours(self, topology: &Topology) -> [Option<Self>; 4] {
        // The level the midpoints are looked up at, and the tile's extent in its face grid.
        let (at, scale) = match self.level.child() {
            Some(finer) if self.level.tile_quads() < 2 => (finer, 2),
            _ => (self.level, 1),
        };
        let size = self.level.tile_quads() * scale;
        let (x0, y0) = (u32::from(self.tx) * size, u32::from(self.ty) * size);
        let half = size / 2;
        let midpoints = [
            (x0 + half, y0),
            (x0 + size, y0 + half),
            (x0 + half, y0 + size),
            (x0, y0 + half),
        ];
        midpoints.map(|(i, j)| {
            let copies = topology.copies(self.face, at, i, j);
            let mut others = copies.iter().filter(|c| {
                // At `at` a whole-face level's tile is (0, 0) either way, so the copy's tile
                // is this one's exactly when it is this place's.
                !(c.face == self.face && c.tx == self.tx && c.ty == self.ty)
            });
            others.next().map(|c| Self {
                face: c.face,
                level: self.level,
                tx: c.tx,
                ty: c.ty,
            })
        })
    }

    /// The place as the GPU holds it beside the face ([`GpuNode::place`]): level in bits 28 to
    /// 32, column in bits 14 to 28, row in bits 0 to 14.
    #[must_use]
    pub fn packed(self) -> u32 {
        (u32::from(self.level.get()) << (2 * PLACE_AXIS_BITS))
            | (u32::from(self.tx) << PLACE_AXIS_BITS)
            | u32::from(self.ty)
    }

    /// The place a GPU pass wrote as `face` and [`Place::packed`] bits, if it is one: a level up
    /// to [`geom::MAX_LEVEL`] and a column and row inside its face's grid.
    #[must_use]
    pub fn from_packed(face: u32, packed: u32) -> Option<Self> {
        let axis = (1 << PLACE_AXIS_BITS) - 1;
        let level = Level::new(u8::try_from(packed >> (2 * PLACE_AXIS_BITS)).ok()?).ok()?;
        let tx = u16::try_from((packed >> PLACE_AXIS_BITS) & axis).ok()?;
        let ty = u16::try_from(packed & axis).ok()?;
        let per_side = level.tiles_per_side();
        (u32::from(tx) < per_side && u32::from(ty) < per_side).then_some(Self {
            face: FaceId::new(face),
            level,
            tx,
            ty,
        })
    }
}

/// One node of the tree as the GPU reads it: 64 bytes, laid out as [`NODE_WGSL`]'s `TileNode`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct GpuNode {
    /// The base-mesh face the tile lies on.
    pub face: u32,
    /// Its place on the face ([`Place::packed`]).
    pub place: u32,
    /// Its slot ([`Slot::packed`]).
    pub slot: u32,
    /// [`NODE_LIVE`], [`NODE_FLOOR`], [`NODE_FINEST`]; 0 for an empty node.
    pub flags: u32,
    /// The parent's node index, or [`NONE`].
    pub parent: u32,
    /// `e`: a bound, in world units, on the distance from what the tile draws to the finest
    /// surface under it; 0 for a finest tile.
    pub error: f32,
    /// Zero.
    pub reserved: [u32; 2],
    /// The children's node indices in [`Place::children`]'s order, [`NONE`] where not resident.
    pub children: [u32; 4],
    /// The edge neighbours' node indices in [`Place::neighbours`]'s order, [`NONE`] where not
    /// resident.
    pub neighbours: [u32; 4],
}

/// The bytes of one [`GpuNode`].
pub const NODE_BYTES: u64 = size_of::<GpuNode>() as u64;

/// Why the tree couldn't take a node.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum TreeError {
    /// Every node index is taken: more tiles resident than the tree was sized for.
    #[error("the tile tree is full: all {capacity} nodes are taken")]
    Full {
        /// The nodes the tree holds.
        capacity: u32,
    },
    /// The place already has a node.
    #[error("{0:?} is already in the tile tree")]
    Present(Place),
    /// Creating the tree's GPU buffer failed.
    #[error("the tile tree's buffer couldn't be created: {0}")]
    Gpu(#[from] GpuError),
}

/// One resident tile, as the CPU side of the tree keeps it.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Node {
    /// Where the tile sits.
    place: Place,
    /// Its slot.
    slot: Slot,
    /// Its bound `e`.
    error: f32,
    /// [`NODE_FLOOR`] and [`NODE_FINEST`] as they apply ([`NODE_LIVE`] is implied).
    flags: u32,
    /// Its edge neighbours' places, found once on arrival.
    neighbours: [Option<Place>; 4],
}

/// The resident tile tree: node indices by place, each node's tile, and the nodes to write to
/// the GPU next.
#[derive(Debug)]
pub struct TileTree {
    /// The topology neighbours are found through.
    topology: Topology,
    /// Per node index, its tile, `None` for an empty node.
    nodes: Vec<Option<Node>>,
    /// Empty node indices below `nodes.len()`, to reuse before growing it.
    free: Vec<u32>,
    /// The node index of every resident place.
    by_place: BTreeMap<Place, u32>,
    /// Node indices whose GPU copy is out of date.
    dirty: BTreeSet<u32>,
    /// The most nodes the tree holds: the GPU buffer's length.
    capacity: u32,
    /// The GPU buffer, `capacity` [`GpuNode`]s.
    buffer: Buffer,
}

impl TileTree {
    /// An empty tree for `mesh`'s tiles, holding up to `capacity` nodes, with its GPU buffer
    /// (every node empty: zeroed).
    ///
    /// # Errors
    ///
    /// [`TreeError::Gpu`] if the buffer can't be created.
    pub fn new(gpu: &Gpu, mesh: &BaseMesh, capacity: u32) -> Result<Self, TreeError> {
        let size = u64::from(capacity.max(1)) * NODE_BYTES;
        let buffer = gpu.scoped("tile tree", || {
            gpu.device().create_buffer(&BufferDescriptor {
                label: Some("tile tree"),
                size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        })?;
        Ok(Self {
            topology: mesh.topology().clone(),
            nodes: Vec::new(),
            free: Vec::new(),
            by_place: BTreeMap::new(),
            dirty: BTreeSet::new(),
            capacity,
            buffer,
        })
    }

    /// The GPU buffer: [`TileTree::capacity`] [`GpuNode`]s, read as `array<TileNode>`
    /// ([`NODE_WGSL`]). It stays the same buffer for the tree's life.
    #[must_use]
    pub const fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// The most nodes the tree holds.
    #[must_use]
    pub const fn capacity(&self) -> u32 {
        self.capacity
    }

    /// How many tiles the tree holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.by_place.len()
    }

    /// Whether the tree holds no tile.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_place.is_empty()
    }

    /// The node index of the tile at `place`, if it is resident.
    #[must_use]
    pub fn node(&self, place: Place) -> Option<u32> {
        self.by_place.get(&place).copied()
    }

    /// Adds the tile at `place`, in `slot` with bound `error`, and returns its node index.
    /// `flags` adds [`NODE_FLOOR`] or [`NODE_FINEST`]; [`NODE_LIVE`] is implied.
    ///
    /// # Errors
    ///
    /// [`TreeError::Present`] if the place already has a node; [`TreeError::Full`] if every
    /// node index is taken.
    pub(crate) fn insert(
        &mut self,
        place: Place,
        slot: Slot,
        error: f32,
        flags: u32,
    ) -> Result<u32, TreeError> {
        if self.by_place.contains_key(&place) {
            return Err(TreeError::Present(place));
        }
        let index = if let Some(index) = self.free.pop() {
            index
        } else {
            let index = u32::try_from(self.nodes.len())
                .ok()
                .filter(|&index| index < self.capacity)
                .ok_or(TreeError::Full {
                    capacity: self.capacity,
                })?;
            self.nodes.push(None);
            index
        };
        let node = Node {
            place,
            slot,
            error,
            flags: flags & (NODE_FLOOR | NODE_FINEST),
            neighbours: place.neighbours(&self.topology),
        };
        if let Some(entry) = usize::try_from(index)
            .ok()
            .and_then(|i| self.nodes.get_mut(i))
        {
            *entry = Some(node);
        }
        self.by_place.insert(place, index);
        self.mark_around(&node, index);
        Ok(index)
    }

    /// Removes the tile at `place`, if it is resident, and returns its slot.
    pub(crate) fn remove(&mut self, place: Place) -> Option<Slot> {
        let index = self.by_place.remove(&place)?;
        let node = usize::try_from(index)
            .ok()
            .and_then(|i| self.nodes.get_mut(i))
            .and_then(Option::take)?;
        self.free.push(index);
        self.mark_around(&node, index);
        Some(node.slot)
    }

    /// Points the resident tile at `place` to `slot`, with bound `error`; returns the slot it
    /// held, or `None` if `place` isn't resident.
    pub(crate) fn move_to(&mut self, place: Place, slot: Slot, error: f32) -> Option<Slot> {
        let index = *self.by_place.get(&place)?;
        let node = usize::try_from(index)
            .ok()
            .and_then(|i| self.nodes.get_mut(i))
            .and_then(Option::as_mut)?;
        let old = std::mem::replace(&mut node.slot, slot);
        node.error = error;
        self.dirty.insert(index);
        Some(old)
    }

    /// Marks `node` (at `index`) and every node that links to it as out of date on the GPU.
    fn mark_around(&mut self, node: &Node, index: u32) {
        self.dirty.insert(index);
        let parent = node.place.parent();
        let linked = node
            .neighbours
            .into_iter()
            .chain(node.place.children())
            .chain([parent]);
        for place in linked.flatten() {
            if let Some(&other) = self.by_place.get(&place) {
                self.dirty.insert(other);
            }
        }
    }

    /// The node index at `place`, or [`NONE`].
    fn link(&self, place: Option<Place>) -> u32 {
        place
            .and_then(|place| self.by_place.get(&place).copied())
            .unwrap_or(NONE)
    }

    /// The GPU form of node `index`: its tile and links as they stand, or zeroed if empty.
    #[must_use]
    pub fn gpu_node(&self, index: u32) -> GpuNode {
        let node = usize::try_from(index)
            .ok()
            .and_then(|i| self.nodes.get(i))
            .and_then(Option::as_ref);
        let Some(node) = node else {
            return GpuNode::zeroed();
        };
        GpuNode {
            face: node.place.face.index(),
            place: node.place.packed(),
            slot: node.slot.packed(),
            flags: NODE_LIVE | node.flags,
            parent: self.link(node.place.parent()),
            error: node.error,
            reserved: [0; 2],
            children: node.place.children().map(|child| self.link(child)),
            neighbours: node.neighbours.map(|neighbour| self.link(neighbour)),
        }
    }

    /// The nodes out of date on the GPU, as runs of consecutive node indices: each run's first
    /// index and its nodes as they stand now. Clears the marks.
    pub(crate) fn take_dirty(&mut self) -> Vec<(u32, Vec<GpuNode>)> {
        let dirty = std::mem::take(&mut self.dirty);
        let mut runs: Vec<(u32, Vec<GpuNode>)> = Vec::new();
        for index in dirty {
            let node = self.gpu_node(index);
            match runs.last_mut() {
                Some((start, nodes))
                    if u64::from(*start) + nodes.len() as u64 == u64::from(index) =>
                {
                    nodes.push(node);
                }
                _ => runs.push((index, vec![node])),
            }
        }
        runs
    }
}
