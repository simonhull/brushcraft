//! Pen-down shapes: each tile a brush stroke changes, as it was when the stroke began, kept
//! compactly for the whole stroke.
//!
//! A stroke reads the surface as it was at pen-down (the brush-engine spec, "The surface a stroke
//! reads is its pen-down surface"): placement's raycasts, walks and area normals, the hit
//! normals, and the distances its action's falloff is measured over. A tile no edit of the stroke
//! has changed is its own pen-down shape: its positions as the store evaluates them. A tile the
//! stroke changes keeps its **pen-down shape** ([`PenDownShape`]) from just before its first
//! write ([`TileStore::keep_pen_down`]), and keeps it until the stroke ends, however often the
//! stroke comes back to it: a stroke never reads its own dabs.
//!
//! The shapes go when the next stroke's op begins ([`TileStore::begin_op`]) or the stroke ends
//! ([`TileStore::release_pen_down`], which [`Project::end_stroke`](crate::Project::end_stroke)
//! calls).
//!
//! # The shape
//!
//! A tile's positions are its prediction, from the level below, which a brush never edits, with
//! its own corrections applied vertex by vertex. So a pen-down shape is the tile's **corrections**
//! as they were before the stroke's first write, a copy of the store's own block (its interior
//! quantized per plane, its border exact: [`geom::CorrectionTile`]), and its positions are made
//! again from the tile's prediction (the one the store keeps, or one made for the call) with them
//! applied: the bits the store's evaluation gave at pen-down, exactly. So a tile reads the same at
//! pen-down before the stroke first writes it and after, and every tile holding a vertex has its
//! bits: pen-down shapes meet their neighbours, kept or not, bit for bit.
//!
//! A shape of a tile of 65² vertices holds 63² interior vertices of three `i16` and 256 border
//! vertices of three `f32`: 26,886 bytes and a few dozen more ([`PenDownShape::bytes`]), about
//! half its positions in `f32`. An ordinary stroke changes 30 to 50 tiles, 1 to 2 MiB; one dab of
//! a brush of a million vertices about 1,165, some 30 MiB.
//!
//! # Positions made, kept for the next reads
//!
//! Making a changed tile's positions applies its corrections to its prediction, and a raycast
//! needs blocks built over them too: tens of microseconds a tile, where the store's own tiles are
//! read from its cache. Placement reads the same few changed tiles near the pen many times a dab
//! (about a dozen raycasts, the walks, the area normal), so the positions of the
//! [`MADE_TILES`] changed tiles read last are kept made, each with its raycast blocks once built,
//! the least recently read going first. Reads through `&self` (the brush's workers,
//! [`TileStore::made_pen_down`]) find them there but don't count as reads; a brush has the store
//! make its tiles' first ([`TileStore::make_pen_down_many`]), or makes them on the workers that
//! read them ([`TileStore::pen_down_maker`]) and has the store keep them after
//! ([`TileStore::keep_brush_pen_down`]).
//!
//! # Shapes taken on the workers
//!
//! A brush's workers read a tile the stroke hasn't changed from the store's cache, and can take
//! its pen-down shape there ([`TileStore::pen_down_taker`]), for the store to keep before the
//! stroke's first write ([`TileStore::keep_pen_down_taken`]): the shapes of a dab's new tiles made
//! beside their brushing, not one by one after it. A taken shape names the store and its count of
//! writes when it was taken, and is kept only while they are the same, so it is the shape
//! [`TileStore::keep_pen_down`] makes.
//!
//! # The budget
//!
//! The shapes are counted in the store's memory budget, with the positions and normals made of
//! them and the previews' pen-down blocks ([`TileStore::pen_down_bytes`]): the pools' limit is
//! lowered by their bytes, never below the pools' minimum, and raised back when they go. The
//! lowering is settled as a frame pins its tiles ([`TileStore::pin`]), before it counts its room,
//! and never below what those pins need: so the data a stroke keeps never makes a frame's pins
//! fail, and a frame that has pinned its tiles is never short of the room it counted. There is
//! no cap: a stroke keeps every tile's shape whatever it costs, and the bytes the floor leaves no
//! room for are reported as held beyond the budget ([`TileStore::pen_down_beyond_budget`]).
//! Lowering the limit unmaps nothing at once: the pools take no chunk while they hold more than
//! it, and give chunks back as they next make room.
//!
//! # Reads
//!
//! [`TileStore::pen_down_positions`] gives a tile's pen-down positions (made from its shape and
//! kept, or as the store evaluates it); [`TileStore::pen_down_block`] its block with its one-vertex
//! ring and base-vertex rings gathered from its neighbours' pen-down positions, as the store
//! gathers its own ([`TileStore::positions_with_ring`]); [`TileStore::pen_down_normals`] the
//! normals of that block, the bits [`TileStore::normals`] gives where nothing the normals read was
//! changed. The spatial index's
//! [`SpatialIndex::raycast_pen_down`](crate::SpatialIndex::raycast_pen_down) raycasts the pen-down
//! surface, and its refits keep a changed tile's box over its pen-down shape
//! too (see [`SpatialIndex::refit`](crate::SpatialIndex::refit)).

use std::cell::{Cell, OnceCell};
use std::collections::{BTreeMap, BTreeSet};

use geom::{
    BaseMesh, Channel, CorrectionTile, CorrectionView, FaceId, GatheredBlock, LayerId, Level,
    PositionSource, Prediction, Region, TileKey, Vec3, gather_block,
};

use super::error::{CacheError, Invariant};
use super::{StoreId, TileStore};
use crate::Aabb;
use crate::index::TileBlocks;
use crate::tiles::config::Evaluation;

/// A tile's shape at pen-down: its corrections as they were, and the box of its positions then
/// (see the module notes).
#[derive(Debug, Clone, PartialEq)]
pub struct PenDownShape {
    /// The tile's corrections before the stroke's first write.
    corrections: CorrectionTile,
    /// The box of the tile's finite positions at pen-down; `None` for none.
    bounds: Option<Aabb>,
}

impl PenDownShape {
    /// The shape of a tile whose corrections are `corrections` and whose positions, made from
    /// them, are `positions`: for the tests (the store makes its own from a tile it holds).
    #[cfg(test)]
    #[must_use]
    pub(crate) fn new(corrections: CorrectionTile, positions: &[Vec3]) -> Self {
        let finite = positions.iter().copied().filter(|p| p.is_finite());
        Self {
            corrections,
            bounds: Aabb::from_points(finite),
        }
    }

    /// The tile's pen-down positions, row-major: `prediction` (the tile's) with the pen-down
    /// corrections applied, the bits the store's evaluation gave at pen-down.
    #[must_use]
    pub fn positions(&self, prediction: &Prediction) -> Vec<Vec3> {
        let view = self.corrections.view();
        prediction
            .apply(|i, j| view.get(i, j).unwrap_or(Vec3::ZERO))
            .into_vec()
    }

    /// The box of the tile's finite positions at pen-down; `None` for a tile of none.
    #[must_use]
    pub const fn bounds(&self) -> Option<Aabb> {
        self.bounds
    }

    /// The bytes the shape holds: itself, its interior's three `i16` a vertex and its border's
    /// three `f32`.
    #[must_use]
    pub fn bytes(&self) -> usize {
        let side = usize::try_from(self.corrections.side()).unwrap_or(0);
        let inner = side.saturating_sub(2);
        size_of::<Self>()
            + inner * inner * size_of::<[i16; 3]>()
            + 4 * side.saturating_sub(1) * size_of::<Vec3>()
    }
}

/// What a worker needs to make a changed tile's pen-down positions ([`TileStore::pen_down_maker`]).
///
/// Its pen-down shape and the prediction the store keeps of it, neither of which changes while the
/// stroke lasts.
#[derive(Debug, Clone, Copy)]
pub struct PenDownMaker<'a> {
    /// The tile.
    key: TileKey,
    /// Its pen-down shape.
    shape: &'a PenDownShape,
    /// The prediction the store keeps of it.
    prediction: &'a Prediction,
}

impl PenDownMaker<'_> {
    /// The tile's pen-down positions, made ([`PenDownShape::positions`]): for the worker to read,
    /// and the store to keep after ([`TileStore::keep_brush_pen_down`]).
    #[must_use]
    pub fn make(&self) -> MadePenDown {
        MadePenDown {
            key: self.key,
            positions: self.shape.positions(self.prediction),
        }
    }
}

/// A changed tile's pen-down positions made on a worker ([`PenDownMaker::make`]).
///
/// Only the store's [`PenDownMaker`] makes one, from the tile's pen-down shape and its kept
/// prediction: so the positions are the tile's pen-down positions for as long as the stroke keeps
/// its shape.
#[derive(Debug)]
pub struct MadePenDown {
    /// The tile.
    key: TileKey,
    /// Its pen-down positions, row-major.
    positions: Vec<Vec3>,
}

impl MadePenDown {
    /// The tile whose positions these are.
    #[must_use]
    pub const fn key(&self) -> TileKey {
        self.key
    }

    /// The tile's pen-down positions, row-major.
    #[must_use]
    pub fn positions(&self) -> &[Vec3] {
        &self.positions
    }
}

/// What a worker needs to read a tile the stroke hasn't changed and take its pen-down shape: its
/// positions and corrections as the store has them cached ([`TileStore::pen_down_taker`]).
#[derive(Debug, Clone, Copy)]
pub struct PenDownTaker<'a> {
    /// The tile.
    key: TileKey,
    /// Its positions as the store evaluates them: its pen-down positions.
    positions: &'a [Vec3],
    /// Its corrections.
    corrections: CorrectionView<'a>,
    /// The store they were read from.
    store: StoreId,
    /// That store's count of writes ([`TileStore::edits`]) when they were read.
    edits: u64,
}

impl<'a> PenDownTaker<'a> {
    /// The tile's positions as the store evaluates them, row-major: its pen-down positions, as the
    /// stroke hasn't changed it.
    #[must_use]
    pub const fn positions(&self) -> &'a [Vec3] {
        self.positions
    }

    /// The tile's pen-down shape, from its positions and corrections, for
    /// [`TileStore::keep_pen_down_taken`] to keep.
    #[must_use]
    pub fn take(&self) -> TakenShape {
        let PositionsBox(bounds) = PositionsBox::of(self.positions);
        TakenShape {
            key: self.key,
            shape: PenDownShape {
                corrections: self.corrections.to_tile(),
                bounds,
            },
            store: self.store,
            edits: self.edits,
        }
    }
}

/// A tile's pen-down shape taken on a worker ([`PenDownTaker::take`]), for
/// [`TileStore::keep_pen_down_taken`] to keep.
///
/// It names the store it was taken from and that store's count of writes then, and is kept only
/// while they are the same: so it is the shape [`TileStore::keep_pen_down`] would make, bit for
/// bit.
#[derive(Debug)]
pub struct TakenShape {
    /// The tile.
    key: TileKey,
    /// Its shape as the store had it.
    shape: PenDownShape,
    /// The store it was taken from.
    store: StoreId,
    /// That store's count of writes ([`TileStore::edits`]) when it was taken.
    edits: u64,
}

impl TakenShape {
    /// The tile whose shape this is.
    #[must_use]
    pub const fn key(&self) -> TileKey {
        self.key
    }
}

/// The box of a tile's finite positions: `None` inside for a tile of none.
#[derive(Debug, Clone, Copy)]
struct PositionsBox(Option<Aabb>);

impl PositionsBox {
    /// The box of `positions`' finite points.
    fn of(positions: &[Vec3]) -> Self {
        Self(Aabb::from_points(
            positions.iter().copied().filter(|p| p.is_finite()),
        ))
    }
}

/// The pen-down shapes of the tiles the current stroke changed, and their bytes.
#[derive(Debug, Default)]
pub(crate) struct PenDown {
    /// Each changed tile's shape.
    shapes: BTreeMap<TileKey, PenDownShape>,
    /// The bytes the shapes hold, each counted as [`PenDownShape::bytes`] and its map entry.
    bytes: usize,
    /// The bytes the pools' minimum leaves no room for: held beyond the budget.
    beyond: usize,
    /// The positions made of the changed tiles read last, at most [`MADE_TILES`] (see "Positions
    /// made, kept for the next reads" in the module notes).
    made: BTreeMap<TileKey, Made>,
    /// Reads of `made` so far: each entry's last read is one of them.
    reads: u64,
    /// The positions made of the changed tiles a brush is working on: the tiles of its last dab
    /// or run, whatever their number (see "Positions made, kept for the next reads" in the module
    /// notes).
    brush: BTreeMap<TileKey, Vec<Vec3>>,
    /// The pen-down normals of the tiles whose normals read a changed tile, made once each: they
    /// can't change while the stroke lasts.
    normals: BTreeMap<TileKey, Vec<Vec3>>,
    /// The pen-down blocks a brush's previews weigh their points at, one a previewed tile (see
    /// [`TileStore::keep_preview_pen_down`]).
    previews: BTreeMap<TileKey, GatheredBlock>,
    /// The bytes `previews` holds.
    preview_bytes: usize,
}

/// A changed tile's pen-down positions, made, and its raycast blocks once built.
#[derive(Debug)]
struct Made {
    /// The positions, row-major.
    positions: Vec<Vec3>,
    /// The blocks a raycast tests them by, built at the first raycast that needs them.
    blocks: OnceCell<TileBlocks>,
    /// The read it was last read at.
    read: u64,
}

/// The most changed tiles whose pen-down positions are kept made: about the tiles placement reads
/// around the pen, several dabs' worth. With [`MADE_BYTES`] each, about 3.5 MiB.
pub(crate) const MADE_TILES: usize = 64;

/// The bytes a tile's positions or normals made are counted at: a tile of 65² of them and its map
/// entry.
const GRID_BYTES: usize = 65 * 65 * size_of::<Vec3>() + ENTRY_BYTES;

/// The bytes a tile's positions kept made for placement are counted at: [`GRID_BYTES`] and its
/// raycast blocks (at most about 4.3 KB).
const MADE_BYTES: usize = GRID_BYTES + 4_352;

/// The bytes a map entry adds to a shape's own: its key, and the map's node share.
const ENTRY_BYTES: usize = size_of::<TileKey>() + 2 * size_of::<usize>();

impl TileStore {
    /// Keeps the pen-down shape of each of `keys` the current stroke hasn't kept yet, from its
    /// positions as the store evaluates them now: call it before the stroke's first write to a
    /// tile (see the module notes). The tiles not cached are evaluated together first; the
    /// shapes are made on the store's workers.
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate_many`] and [`TileStore::evaluate`]; the shapes made before the
    /// error are kept.
    pub fn keep_pen_down(&mut self, keys: &[TileKey]) -> Result<(), CacheError> {
        let new: Vec<TileKey> = keys
            .iter()
            .filter(|key| !self.pen_down.shapes.contains_key(key))
            .copied()
            .collect();
        if new.is_empty() {
            return Ok(());
        }
        let cold: Vec<TileKey> = new
            .iter()
            .filter(|key| self.cached_positions(key).is_none())
            .copied()
            .collect();
        if !cold.is_empty() {
            self.evaluate_many(&cold, Evaluation::Positions)?;
        }
        let boxes: Vec<Option<PositionsBox>> = {
            let cached: Vec<Option<&[Vec3]>> =
                new.iter().map(|key| self.cached_positions(key)).collect();
            self.par_map(&cached, |positions| positions.map(PositionsBox::of))
        };
        let mut outcome = Ok(());
        for (key, bounds) in new.into_iter().zip(boxes) {
            let made = self.pen_down_shape_of(key, bounds);
            match made {
                Ok(shape) => {
                    self.pen_down.bytes += shape.bytes() + ENTRY_BYTES;
                    self.pen_down.shapes.insert(key, shape);
                }
                Err(error) => {
                    outcome = Err(error);
                    break;
                }
            }
        }
        outcome
    }

    /// Keeps the pen-down shape of each of `keys` the current stroke hasn't kept yet, as
    /// [`TileStore::keep_pen_down`] does, taking those of `taken` that are still the store's as it
    /// is (no write since they were taken) instead of making them again: the same shapes, made on
    /// the workers that read the tiles before. A shape of `taken` for a tile not of `keys`, or one
    /// kept already, is dropped.
    ///
    /// # Errors
    ///
    /// As [`TileStore::keep_pen_down`].
    pub fn keep_pen_down_taken(
        &mut self,
        keys: &[TileKey],
        taken: impl IntoIterator<Item = TakenShape>,
    ) -> Result<(), CacheError> {
        let (store, edits) = (self.id(), self.edits());
        let fresh = taken
            .into_iter()
            .filter(|taken| taken.store == store && taken.edits == edits);
        for TakenShape { key, shape, .. } in fresh {
            if keys.contains(&key) && !self.pen_down.shapes.contains_key(&key) {
                self.pen_down.bytes += shape.bytes() + ENTRY_BYTES;
                self.pen_down.shapes.insert(key, shape);
            }
        }
        self.keep_pen_down(keys)
    }

    /// What a worker needs to read tile `key` and take its pen-down shape
    /// ([`PenDownTaker::take`]), if the current stroke hasn't kept one for it and its positions and
    /// corrections are cached: for [`TileStore::keep_pen_down_taken`] to keep before the stroke's
    /// first write to it.
    #[must_use]
    pub fn pen_down_taker(&self, key: &TileKey) -> Option<PenDownTaker<'_>> {
        if self.pen_down.shapes.contains_key(key) {
            return None;
        }
        Some(PenDownTaker {
            key: *key,
            positions: self.cached_positions(key)?,
            corrections: self.cached_correction(key)?,
            store: self.id(),
            edits: self.edits(),
        })
    }

    /// Tile `key`'s shape as it is now, `bounds` its positions' box if at hand (else taken from
    /// its evaluation).
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate`] and [`TileStore::correction`].
    fn pen_down_shape_of(
        &mut self,
        key: TileKey,
        bounds: Option<PositionsBox>,
    ) -> Result<PenDownShape, CacheError> {
        let PositionsBox(bounds) = match bounds {
            Some(bounds) => bounds,
            // Pushed out of the cache by a later tile's evaluation: on its own.
            None => PositionsBox::of(self.evaluate(key)?),
        };
        let corrections = self.correction(key)?.to_tile();
        Ok(PenDownShape {
            corrections,
            bounds,
        })
    }

    /// Tile `key`'s pen-down shape, if the current stroke changed it; `None` for a tile that is
    /// its own pen-down shape.
    #[must_use]
    pub fn pen_down_shape(&self, key: &TileKey) -> Option<&PenDownShape> {
        self.pen_down.shapes.get(key)
    }

    /// Tile `key`'s pen-down positions, row-major: if the stroke changed it, made from its shape
    /// and its prediction (the one the store keeps, else one made now and not kept), and kept made
    /// for the next reads; else as the store evaluates it ([`TileStore::evaluate`]). The same bits
    /// either way for a tile the stroke has yet to write.
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate`], for the tile or its parents.
    pub fn pen_down_positions(&mut self, key: TileKey) -> Result<&[Vec3], CacheError> {
        if !self.pen_down.shapes.contains_key(&key) {
            return self.evaluate(key);
        }
        Ok(&self.made(key)?.positions)
    }

    /// Changed tile `key`'s pen-down positions if they are kept made, for reads through `&self`
    /// (a brush's workers): not counted as a read. `None` for a tile the stroke hasn't changed, or
    /// one whose positions aren't kept made now.
    #[must_use]
    pub fn made_pen_down(&self, key: &TileKey) -> Option<&[Vec3]> {
        let pen_down = &self.pen_down;
        pen_down
            .made
            .get(key)
            .map(|made| made.positions.as_slice())
            .or_else(|| pen_down.brush.get(key).map(Vec::as_slice))
    }

    /// Changed tile `key`'s pen-down positions and raycast blocks, made and kept for the next
    /// reads; `None` for a tile the stroke hasn't changed.
    ///
    /// # Errors
    ///
    /// As [`TileStore::pen_down_positions`].
    pub(crate) fn pen_down_blocks(
        &mut self,
        key: TileKey,
    ) -> Result<Option<(&[Vec3], &TileBlocks)>, CacheError> {
        if !self.pen_down.shapes.contains_key(&key) {
            return Ok(None);
        }
        let made = self.made(key)?;
        let blocks = made
            .blocks
            .get_or_init(|| TileBlocks::new(&made.positions, key.side()));
        Ok(Some((&made.positions, blocks)))
    }

    /// Changed tile `key`'s positions kept made: made now if they aren't, the least recently read
    /// let go past [`MADE_TILES`]; counted as a read.
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate`], for the tile's parents.
    fn made(&mut self, key: TileKey) -> Result<&Made, CacheError> {
        self.pen_down.reads += 1;
        let read = self.pen_down.reads;
        if !self.pen_down.made.contains_key(&key) {
            // A brush's, if it made them; else made now.
            let positions = match self.pen_down.brush.get(&key) {
                Some(brush) => brush.clone(),
                None => self.made_afresh(key)?,
            };
            self.keep_made(key, positions, read);
        }
        let made = self
            .pen_down
            .made
            .get_mut(&key)
            .ok_or(Invariant::NoPenDownShape(key))?;
        made.read = read;
        Ok(made)
    }

    /// Makes the pen-down positions of the changed tiles of `keys` a brush is about to read
    /// through `&self` on its workers ([`TileStore::made_pen_down`]): its dab's or run's tiles.
    ///
    /// They are kept apart from placement's, as the brush's working set: those of `keys` it holds
    /// already stay, the others' go, and the changed tiles of `keys` with a kept prediction and no
    /// positions made yet have them made, together on the store's workers. However many they are
    /// (a big brush's dab reaches over a thousand tiles), the next dab over the same tiles finds
    /// them made. Their bytes are counted with the pen-down data.
    pub fn make_pen_down_many(&mut self, keys: &[TileKey]) {
        self.keep_brush_pen_down(keys, []);
        let todo: Vec<PenDownMaker<'_>> = keys
            .iter()
            .filter_map(|key| self.pen_down_maker(key))
            .collect();
        if todo.is_empty() {
            return;
        }
        let made: Vec<MadePenDown> = self.par_map(&todo, PenDownMaker::make);
        self.keep_brush_pen_down(keys, made);
    }

    /// What a worker needs to make changed tile `key`'s pen-down positions
    /// ([`PenDownMaker::make`]), for the store to keep after ([`TileStore::keep_brush_pen_down`]):
    /// `None` for a tile the stroke hasn't changed, one whose positions are kept made already
    /// ([`TileStore::made_pen_down`]), or one whose prediction isn't kept.
    #[must_use]
    pub fn pen_down_maker(&self, key: &TileKey) -> Option<PenDownMaker<'_>> {
        if self.made_pen_down(key).is_some() {
            return None;
        }
        Some(PenDownMaker {
            key: *key,
            shape: self.pen_down.shapes.get(key)?,
            prediction: self.kept.get(key)?,
        })
    }

    /// Sets the brush's working set to the changed tiles of `keys`, its dab's or run's tiles, as
    /// [`TileStore::make_pen_down_many`] does, keeping `made` (made on the workers that brushed
    /// them, [`PenDownMaker::make`]): those of `keys` held already stay, the others go, and
    /// each of `made` of a tile of `keys` without positions kept made is kept. Their bytes are
    /// counted with the pen-down data.
    pub fn keep_brush_pen_down(
        &mut self,
        keys: &[TileKey],
        made: impl IntoIterator<Item = MadePenDown>,
    ) {
        let wanted: BTreeSet<TileKey> = keys
            .iter()
            .filter(|key| self.pen_down.shapes.contains_key(key))
            .copied()
            .collect();
        let pen_down = &mut self.pen_down;
        pen_down.brush.retain(|key, _| wanted.contains(key));
        for MadePenDown { key, positions } in made {
            if wanted.contains(&key) && !pen_down.made.contains_key(&key) {
                pen_down.brush.entry(key).or_insert(positions);
            }
        }
    }

    /// Keeps `positions`, changed tile `key`'s pen-down positions made, as read at `read`, the
    /// least recently read let go past [`MADE_TILES`].
    fn keep_made(&mut self, key: TileKey, positions: Vec<Vec3>, read: u64) {
        let made = Made {
            positions,
            blocks: OnceCell::new(),
            read,
        };
        self.pen_down.made.insert(key, made);
        if self.pen_down.made.len() > MADE_TILES {
            let oldest = self
                .pen_down
                .made
                .iter()
                .min_by_key(|(_, kept)| kept.read)
                .map(|(oldest, _)| *oldest);
            if let Some(oldest) = oldest {
                self.pen_down.made.remove(&oldest);
            }
        }
    }

    /// Changed tile `key`'s pen-down positions, made from its shape and its prediction (the one
    /// the store keeps, else one made now and not kept), not kept.
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate`], for the tile's parents.
    fn made_afresh(&mut self, key: TileKey) -> Result<Vec<Vec3>, CacheError> {
        let made = match self.kept.get(&key) {
            Some(_) => None,
            None => Some(self.prediction_of_one(key)?),
        };
        let prediction = made.as_ref().or_else(|| self.kept.get(&key));
        let shape = self.pen_down.shapes.get(&key);
        match (shape, prediction) {
            (Some(shape), Some(prediction)) => Ok(shape.positions(prediction)),
            _ => Err(Invariant::NoPenDownShape(key).into()),
        }
    }

    /// Tile `key`'s pen-down positions with its one-vertex ring and its base-vertex rings, as
    /// [`TileStore::positions_with_ring`] gathers the store's, every tile read at its pen-down
    /// shape: the block itself if nothing it reads was changed by the stroke.
    ///
    /// # Errors
    ///
    /// As [`TileStore::positions_with_ring`].
    pub fn pen_down_block(&mut self, key: TileKey) -> Result<GatheredBlock, CacheError> {
        let inputs: Vec<TileKey> = self.normal_inputs(&key)?.iter().copied().collect();
        if !inputs
            .iter()
            .any(|input| self.pen_down.shapes.contains_key(input))
        {
            return self.positions_with_ring(key);
        }
        // The changed tiles first: making one's prediction may evaluate, and push the others
        // out of the cache. Kept made, unless they are more than that keeps.
        let changed: Vec<TileKey> = inputs
            .iter()
            .filter(|input| self.pen_down.shapes.contains_key(input))
            .copied()
            .collect();
        let mut owned: Vec<(TileKey, Vec<Vec3>)> = Vec::new();
        if changed.len() <= MADE_TILES {
            for &input in &changed {
                self.made(input)?;
            }
        } else {
            for &input in &changed {
                owned.push((input, self.made_afresh(input)?));
            }
        }
        let unchanged: Vec<TileKey> = inputs
            .iter()
            .filter(|input| !self.pen_down.shapes.contains_key(input))
            .copied()
            .collect();
        self.evaluate_many(&unchanged, Evaluation::Positions)?;
        if unchanged
            .iter()
            .any(|input| self.cached_positions(input).is_none())
        {
            // One pushed out by the others' evaluation: each copied on its own.
            for input in &unchanged {
                owned.push((*input, self.evaluate(*input)?.to_vec()));
            }
        }
        let mut tiles: Vec<(TileKey, &[Vec3])> = inputs
            .iter()
            .map(|input| {
                let positions = owned
                    .iter()
                    .find(|(k, _)| k == input)
                    .map(|(_, copy)| copy.as_slice())
                    .or_else(|| self.made_pen_down(input))
                    .or_else(|| self.cached_positions(input))
                    .unwrap_or_default();
                (*input, positions)
            })
            .collect();
        tiles.sort_unstable_by_key(|(input, _)| *input);
        let source = PenDownSource {
            mesh: &self.mesh,
            tiles: &tiles,
            last: Cell::new(None),
        };
        Ok(gather_block(&self.mesh, &source, Region::of_tile(&key))?)
    }

    /// [`TileStore::pen_down_block`] of each of `keys`, in order, gathered on the store's
    /// workers: the same blocks, bit for bit.
    ///
    /// The tiles they read are made current first: the changed ones' positions made and kept
    /// (more than the 64 kept made, or one pushed out meanwhile, and the blocks are gathered
    /// one by one), the others evaluated together.
    ///
    /// # Errors
    ///
    /// As [`TileStore::pen_down_block`].
    pub fn pen_down_blocks_many(
        &mut self,
        keys: &[TileKey],
    ) -> Result<Vec<GatheredBlock>, CacheError> {
        let mut inputs: BTreeSet<TileKey> = BTreeSet::new();
        for key in keys {
            inputs.extend(self.normal_inputs(key)?.iter().copied());
        }
        let (changed, unchanged): (Vec<TileKey>, Vec<TileKey>) = inputs
            .iter()
            .partition(|input| self.pen_down.shapes.contains_key(input));
        if changed.len() <= MADE_TILES {
            for &input in &changed {
                self.made(input)?;
            }
            self.evaluate_many(&unchanged, Evaluation::Positions)?;
            let at_hand: Option<Vec<(TileKey, &[Vec3])>> = inputs
                .iter()
                .map(|input| {
                    let positions = self
                        .made_pen_down(input)
                        .or_else(|| self.cached_positions(input))?;
                    Some((*input, positions))
                })
                .collect();
            if let Some(tiles) = at_hand {
                let mesh = &*self.mesh;
                let gathered = self.par_map(keys, |key| {
                    let source = PenDownSource {
                        mesh,
                        tiles: &tiles,
                        last: Cell::new(None),
                    };
                    gather_block(mesh, &source, Region::of_tile(key))
                });
                return gathered
                    .into_iter()
                    .map(|block| block.map_err(CacheError::from))
                    .collect();
            }
        }
        keys.iter().map(|&key| self.pen_down_block(key)).collect()
    }

    /// Tile `key`'s pen-down unit normals, row-major: the normals of its pen-down block
    /// ([`TileStore::pen_down_block`]), which are [`TileStore::normals`]' bits where nothing they
    /// read was changed by the stroke (then read from the cache). Where something was, they are
    /// made once and kept until the stroke's pen-down data goes: the tiles they read keep their
    /// pen-down positions while the stroke lasts.
    ///
    /// # Errors
    ///
    /// As [`TileStore::normals`] and [`TileStore::pen_down_block`].
    pub fn pen_down_normals(&mut self, key: TileKey) -> Result<&[Vec3], CacheError> {
        if !self.pen_down.normals.contains_key(&key) {
            let changed = self
                .normal_inputs(&key)?
                .iter()
                .any(|input| self.pen_down.shapes.contains_key(input));
            if !changed {
                return self.normals(key);
            }
            let block = self.pen_down_block(key)?;
            let normals = block.normals(&self.mesh).as_slice().to_vec();
            self.pen_down.normals.insert(key, normals);
        }
        Ok(self
            .pen_down
            .normals
            .get(&key)
            .map_or(&[][..], Vec::as_slice))
    }

    /// Keeps `block`, previewed tile `key`'s pen-down block, for the preview that weighs its
    /// points at it ([`TileStore::take_preview_pen_down`] gives it back), its bytes counted with
    /// the pen-down data: a big brush's previews hold one for each of over a thousand tiles. It
    /// goes when the stroke's pen-down data does, or when the preview drops it
    /// ([`TileStore::drop_preview_pen_down`]).
    pub fn keep_preview_pen_down(&mut self, key: TileKey, block: GatheredBlock) {
        self.pen_down.preview_bytes += block_bytes(&block);
        if let Some(old) = self.pen_down.previews.insert(key, block) {
            self.pen_down.preview_bytes -= block_bytes(&old);
        }
    }

    /// Takes previewed tile `key`'s pen-down block out, kept with
    /// [`TileStore::keep_preview_pen_down`], to weigh more dabs at; `None` if none is kept (the
    /// stroke's pen-down data went since).
    pub fn take_preview_pen_down(&mut self, key: &TileKey) -> Option<GatheredBlock> {
        let block = self.pen_down.previews.remove(key)?;
        self.pen_down.preview_bytes -= block_bytes(&block);
        Some(block)
    }

    /// Drops previewed tile `key`'s pen-down block, if one is kept: its preview is cleared.
    pub fn drop_preview_pen_down(&mut self, key: &TileKey) {
        drop(self.take_preview_pen_down(key));
    }

    /// How many tiles' pen-down shapes the current stroke keeps.
    #[must_use]
    pub fn pen_down_tiles(&self) -> usize {
        self.pen_down.shapes.len()
    }

    /// The bytes the current stroke's pen-down data holds: its shapes, the positions and normals
    /// made of them, and its previews' pen-down blocks; withheld from the pools' share of the
    /// budget (see the module notes).
    #[must_use]
    pub fn pen_down_bytes(&self) -> usize {
        let pen_down = &self.pen_down;
        pen_down.bytes
            + pen_down.made.len() * MADE_BYTES
            + (pen_down.brush.len() + pen_down.normals.len()) * GRID_BYTES
            + pen_down.preview_bytes
    }

    /// The bytes of the pen-down shapes the pools' minimum leaves no room for in the budget: held
    /// beyond it (zero while they fit).
    #[must_use]
    pub const fn pen_down_beyond_budget(&self) -> usize {
        self.pen_down.beyond
    }

    /// Lets go of the current stroke's pen-down shapes, their bytes given back to the pools: the
    /// stroke has ended. [`TileStore::begin_op`] does it too, as the next stroke begins.
    pub fn release_pen_down(&mut self) {
        if self.pen_down_bytes() == 0 {
            return;
        }
        self.pen_down = PenDown::default();
        self.withhold_pen_down(0);
    }

    /// Sets the pools' limit to the budget's less the pen-down data's bytes, never below the
    /// pools' minimum nor `beside` (the bytes the pins about to be taken need), and notes what
    /// that leaves beyond the budget: called by [`TileStore::pin`] before it counts its room, so
    /// the data a stroke keeps never starves its next frame's pins.
    pub(super) fn withhold_pen_down(&mut self, beside: usize) {
        let floor = self
            .headroom
            .minimum_budget(self.corrections.chunk_bytes())
            .max(beside);
        let bytes = self.pen_down_bytes();
        self.pen_down.beyond = self.budget.withhold(bytes, floor);
    }
}

/// The bytes `block` holds: itself and its positions with the ring.
fn block_bytes(block: &GatheredBlock) -> usize {
    size_of::<GatheredBlock>() + size_of_val(block.padded())
}

/// The pen-down positions of the tiles a gather reads, and the base mesh at level 0: what
/// [`TileStore::pen_down_block`] gathers from.
///
/// Each vertex is read from [`TileKey::containing`]'s tile, as [`PositionSource`] requires, so a
/// vertex has its owner's bits wherever it is read.
struct PenDownSource<'a> {
    /// Level 0.
    mesh: &'a BaseMesh,
    /// Each tile read, with its pen-down positions, in key order.
    tiles: &'a [(TileKey, &'a [Vec3])],
    /// The tile the last read resolved to, and its place in `tiles`.
    last: Cell<Option<(TileKey, usize)>>,
}

impl PenDownSource<'_> {
    /// Tile `key`'s pen-down positions, if the gather reads it.
    fn positions(&self, key: TileKey) -> Option<&[Vec3]> {
        let at = match self.last.get() {
            Some((last, at)) if last == key => at,
            _ => {
                let at = self.tiles.binary_search_by_key(&key, |(k, _)| *k).ok()?;
                self.last.set(Some((key, at)));
                at
            }
        };
        Some(self.tiles.get(at)?.1)
    }
}

impl PositionSource for PenDownSource<'_> {
    fn position(&self, face: FaceId, level: Level, i: u32, j: u32) -> Option<Vec3> {
        let (key, li, lj) =
            TileKey::containing(LayerId::BASE, Channel::DISPLACEMENT, face, level, i, j)?;
        if level == Level::ZERO {
            return self.mesh.position(face, level, li, lj);
        }
        let index = usize::try_from(lj * key.side() + li).ok()?;
        self.positions(key)?.get(index).copied()
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;
