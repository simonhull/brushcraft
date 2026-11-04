//! A preview on a display tile ([`DisplayState`]).
//!
//! The vertices a renderer draws at a display level, moved by the dabs still to land, with
//! normals of the moved surface and a bound against the finest result (the parent module's
//! "Display tiles").
//!
//! # The pen-down tile
//!
//! Every display vertex `v` subsamples one finest vertex `u`. A display tile's **pen-down tile**
//! `P0` is its record ([`TileStore::display_for_draw`], as stored before any stroke since its
//! last refresh) with every vertex inside a finest tile that changed since replaced by the
//! finest vertex's exact bits as the stroke began: from the tile's pen-down block where a tile the
//! stroke changed is among those its block reads ([`TileStore::pen_down_blocks_many`]), else as
//! the store holds it now (unchanged since the stroke began). Its **base** is the same with the
//! store's current bits: what a renderer's stand-in draws of it. A border vertex of either is the
//! finest vertex's exact bits (stored losslessly, or replaced), so every tile holding it, at any
//! level and on any face, holds the same bits.
//!
//! # Moves
//!
//! A dab moves `v` by the draw action's own move ([`brush::draw_imprint_vertex`],
//! [`brush::draw_move`]) weighed at `P0(v)`, as the finest work weighs `u` at its pen-down
//! position. What is drawn is `D(v) = base(v) + Σ m_d(P0(v))` over the dabs not yet in the base,
//! in order: a vertex the store has moved since (inside a finest tile a landed dab changed) is
//! **rebased** on the store's bits and takes the dabs still to land; any other takes every dab
//! since the tile was made. A dab that moves no finest vertex of a finest tile moves no border
//! vertex of it either (border vertices are finest bits), so a border vertex's `D` is the same
//! function of the same bits in every tile holding it: previews meet at their seams bit for bit,
//! with each other, with the finest previews (which move the same bits by the same dabs) and with
//! a renderer's stand-ins where nothing still to land moves them.
//!
//! # Normals
//!
//! A vertex's pen-down normal `n0` is its record's (averaged, the same bits in every tile holding
//! it at the level), or for a vertex inside a finest tile changed since the record, made from
//! the pen-down positions as a stand-in makes it: over the tile's own grid inside it, over the
//! finest grid around it on its border (the four finest neighbours, the same in every tile
//! holding it), turned to the record's side. The drawn normal is that of the pen-down tangent
//! plane carried by every dab of the stroke: with `t₁, t₂` a frame of `n0`, each tangent moves by
//! `Σ (∇w_d · tᵢ) n_d` ([`brush::draw_imprint_gradient`]), and the normal is their cross product,
//! normalised. A function of the vertex's bits and the dabs alone, so the same in every tile at a
//! level; a vertex no dab moved keeps `n0`'s bits.
//!
//! # The bound
//!
//! The finest result once every dab of the stroke lands is `F'(u) = F₀(u) + Σ m_d(F₀(u)) + ε`,
//! `F₀` the finest surface at pen-down and `ε` the finest work's own rounding into its stored
//! corrections. Write `G(v) = P0(v) + Σ m_d(P0(v))` over every dab of the stroke. At a finest
//! vertex `u` under a triangle of the tile, `λ` its blend of the corners:
//!
//! `|F'(u) − Σλ D(v)| ≤ |F₀(u) − Σλ P0(v)| + Σ_d |m_d(F₀(u)) − Σλ m_d(P0(v))| + max |D(v) − G(v)| +
//! ε`.
//!
//! - The first term is at most `e₀`, the pen-down tile's bound: the record's own where nothing was
//!   replaced; else measured, as a stand-in measures its bound: the farthest finest vertex of every
//!   finest tile changed since the record (all at hand, gathered for the replacing) from the
//!   pen-down tile's triangles, and for the others, unchanged, the record's bound plus `M`, the
//!   largest move of a replaced vertex (the triangles' blend of the corners' moves). The stand-in
//!   formula `e + D + max(D, M)` (`D` the steps since the record) is kept for a place that couldn't
//!   be measured.
//! - Each dab's term is at most `s · min(1, L e₀ + K h² / 2)`, `s` its strength, `L = steepest / r`
//!   and `K = curvature / r²` its falloff's bounds ([`brush::Falloff`]), `h` the quad's longest
//!   pen-down edge: the move at `F₀(u)` against the move at the point of the triangle (Lipschitz),
//!   and that against the triangle's blend of the corners' moves (linear interpolation's error);
//!   both moves lie in `[0, s]` along the dab's normal, so never more than `s`. A dab counts only
//!   in the quads whose pen-down box, grown by `e₀` (where the finest vertices under the quad lie),
//!   its sphere touches (cut into cells of a few finest quads for a coarse quad); the tile's bound
//!   takes the largest cell's sum. Where `e₀` is far larger than the dabs (the coarsest levels),
//!   every cell's grown box takes every dab near it, so the sum is capped by another bound of the
//!   same part: `|M(F₀(u)) − Σλ M(P0(v))| ≤ |M(F₀(u))| + max |M(P0(v))| ≤ 2 max_p |M(p)|`, and
//!   `|M(p)|` is at most the strengths of the dabs overlapping any one dab (every dab holding `p`
//!   overlaps the one nearest it): local to the stroke, not its length.
//! - `max |D(v) − G(v)|` is measured, in `f64`, over every vertex: the rebased vertices' store bits
//!   against the sum, and every vertex's rounding.
//! - `ε`, the finest work's rounding into its stored corrections, lands on every vertex of a finest
//!   tile the stroke writes (a plane's scale that grows re-rounds every interior value), quads no
//!   dab touches included: it is bounded per finest tile the stroke's dabs reach by its planes'
//!   final steps, from their scales now and the dabs' strengths over it ([`rounding_bound`]), the
//!   largest of them under the tile counting everywhere on it.
//!
//! The sum is raised for its own rounding as the stand-ins' is (`render`'s stand-in notes,
//! "Rounding"): times `1 + 2⁻¹⁹`, plus `2⁻¹⁷` of the largest coordinate. The cone is unknown (a
//! renderer takes the one that covers everything).

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use brush::{
    Falloff, Hardened, Imprint, Smoothstep, draw_imprint_gradient, draw_imprint_vertex, draw_move,
};
use geom::{DisplayTile, GatheredBlock, Level, TileKey, Vec3};
use store::{Aabb, CacheError, DisplayPreview, Evaluation, TileStore};

use super::{bounds_of_points, touches};

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;

/// What a sum of bounds is multiplied by: `1 + 2⁻¹⁹`, more than its roundings can have taken off.
const SLACK: f64 = 1.0 + 1.0 / 524_288.0;

/// What a bound is raised by per unit of the largest coordinate it read: `2⁻¹⁷`.
const MAGNITUDE_SLACK: f64 = 1.0 / 131_072.0;

/// A quad of a display tile's pen-down tile, as its bound reads it (module notes, "The bound").
///
/// Its finest vertices lie within `e₀` of its two triangles, so a dab counts at one only if its
/// sphere touches the triangles' part over the vertex, grown by `e₀`, or holds a corner (the
/// interpolation blends the corners' moves). A coarse quad spans many finest tiles: it is cut
/// into cells of [`CELL_FINEST_QUADS`] finest quads a side, and a dab counts in the cells it
/// touches, so a long stroke across a coarse quad adds, at any place, only the dabs over that
/// place (and those holding a corner of the quad).
#[derive(Debug)]
struct Quad {
    /// Its pen-down corners: `(i, j)`, `(i+1, j)`, `(i, j+1)`, `(i+1, j+1)`.
    corners: [Vec3; 4],
    /// Half the square of the longest edge of its two triangles, `h² / 2`.
    half_edge_squared: f64,
    /// A box holding its corners and its cells' boxes: a dab that misses it counts in none.
    bounds: Option<Aabb>,
    /// Per cell, row-major, the dabs' terms of the bound, summed; empty until a dab counts at the
    /// quad, as most of a coarse tile's never do. The cells' boxes are worked out as a dab comes
    /// near ([`cell_box`]), not kept.
    terms: Vec<f64>,
}

/// One display tile's preview as the brush keeps it while its stroke lasts (see the module
/// notes).
#[derive(Debug)]
pub(super) struct DisplayState {
    /// The tile.
    key: TileKey,
    /// The pen-down tile's positions, `P0`.
    p0: Vec<Vec3>,
    /// The pen-down normals, `n0`.
    n0: Vec<Vec3>,
    /// Each vertex's tangent frame of `n0`.
    frames: Vec<[Vec3; 2]>,
    /// How far the dabs have turned each tangent of the frame.
    turn: Vec<[Vec3; 2]>,
    /// The drawn normals as last made ([`DisplayState::publish`]).
    normals: Vec<Vec3>,
    /// The vertices whose tangents a dab turned since the normals were made.
    turned: Vec<bool>,
    /// What is drawn, `D`.
    drawn: Vec<Vec3>,
    /// `G`: the pen-down positions moved by every dab of the stroke, summed in `f64`.
    ideal: Vec<[f64; 3]>,
    /// The record's roughness.
    roughness: Vec<u8>,
    /// A box over its quads' ([`Quad::bounds`]): where a dab can count at all.
    bounds: Option<Aabb>,
    /// `e₀`, the pen-down tile's bound.
    pen_down_bound: f64,
    /// Its quads, row-major: where a dab counts in its bound, and how much.
    quads: Vec<Quad>,
    /// A box over each row of quads: a dab that misses it counts in none of them.
    quad_rows: Vec<Option<Aabb>>,
    /// How many cells a side each quad is cut into ([`cells_per_side`]).
    cells_per_side: usize,
    /// How far a cell's box is grown: `e₀`, rounded up.
    cell_growth: f32,
    /// The largest of the cells' sums: the moves' part of the bound.
    deepest: f64,
    /// How many cells' sums are held ([`Quad::terms`]).
    cells_held: usize,
    /// Whether it was last handed to the store held.
    pub(super) published_held: Option<bool>,
    /// Whether a dab moved a vertex of it, or a landing rebased one: else what it draws is its
    /// record's, and once released and unchanged under it, the record is drawn again.
    pub(super) moved: bool,
}

/// Where display vertex `(i, j)` of `key` lies on the finest level `finest`: the finest tile
/// under `key` holding it and its place there (a vertex on a finest tile's border is in the one
/// on `key`'s side of it).
fn finest_place(key: &TileKey, finest: Level, i: u32, j: u32) -> Option<(TileKey, u32, u32)> {
    let quads = key.side() - 1;
    let step = 1_u32 << (finest.get() - key.level().get());
    let finest_quads = finest.tile_quads().max(1);
    let span = (quads * step / finest_quads).max(1);
    let place = |t: u16, v: u32| {
        let g = (u32::from(t) * quads + v) * step;
        let first = u32::from(t) * span;
        let tile = (g / finest_quads).clamp(first, first + span - 1);
        (tile, g - tile * finest_quads)
    };
    let (fx, a) = place(key.tx(), i);
    let (fy, b) = place(key.ty(), j);
    let tile = TileKey::displacement(
        key.face(),
        finest,
        u16::try_from(fx).ok()?,
        u16::try_from(fy).ok()?,
    )
    .ok()?;
    Some((tile, a, b))
}

/// The vertices of display tile `key` inside finest tile `tile` (at level `finest`), its border
/// included: each one's index in the tile and its place in the finest tile. None for a finest
/// tile not under `key`.
fn vertices_in(key: &TileKey, finest: Level, tile: &TileKey) -> Vec<(usize, u32, u32)> {
    let quads = key.side() - 1;
    let step = 1_u32 << (finest.get() - key.level().get());
    let finest_quads = finest.tile_quads().max(1);
    let span = (quads * step / finest_quads).max(1);
    // Along one axis: the display vertices whose finest coordinate lies in the finest tile, and
    // where.
    let along = |t: u16, f: u16| -> Vec<(u32, u32)> {
        let (t, f) = (u32::from(t), u32::from(f));
        if f < t * span || f >= (t + 1) * span {
            return Vec::new();
        }
        let origin = t * quads * step;
        let (low, high) = (f * finest_quads - origin, (f + 1) * finest_quads - origin);
        let first = low.div_ceil(step);
        let last = (high / step).min(quads);
        (first..=last)
            .map(|i| (i, origin + i * step - f * finest_quads))
            .collect()
    };
    if tile.face() != key.face() || tile.level() != finest {
        return Vec::new();
    }
    let (columns, rows) = (along(key.tx(), tile.tx()), along(key.ty(), tile.ty()));
    let side = usize::try_from(key.side()).unwrap_or(0);
    let mut out = Vec::with_capacity(columns.len() * rows.len());
    for &(j, row) in &rows {
        for &(i, col) in &columns {
            let v = usize::try_from(j).unwrap_or(0) * side + usize::try_from(i).unwrap_or(0);
            out.push((v, col, row));
        }
    }
    out
}

/// The finest tiles under display tile `key`, at `finest`.
pub(super) fn finest_under(key: &TileKey, finest: Level) -> Vec<TileKey> {
    let quads = key.side() - 1;
    let step = 1_u32 << (finest.get() - key.level().get());
    let span = (quads * step / finest.tile_quads().max(1)).max(1);
    let (x0, y0) = (u32::from(key.tx()) * span, u32::from(key.ty()) * span);
    let grid = (y0..y0 + span).flat_map(|y| (x0..x0 + span).map(move |x| (x, y)));
    grid.filter_map(|(x, y)| {
        let (x, y) = (u16::try_from(x).ok()?, u16::try_from(y).ok()?);
        TileKey::displacement(key.face(), finest, x, y).ok()
    })
    .collect()
}

/// A frame of tangents of unit normal `n` (Duff et al., "Building an orthonormal basis,
/// revisited"): a function of `n`'s bits alone, `t₁ × t₂ = n`.
fn frame_of(n: Vec3) -> [Vec3; 2] {
    let sign = if n.z >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    [
        Vec3::new(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x),
        Vec3::new(b, sign + n.y * n.y * a, -n.y),
    ]
}

/// What a dab's terms of the bound are made of (module notes, "The bound"): worked out once a dab,
/// as every quad near it takes a term.
struct Term {
    /// The falloff's steepest slope over the radius, `L`.
    steep: f64,
    /// The falloff's largest curvature over the radius squared, `K`.
    curve: f64,
    /// The dab's strength, `s`.
    strength: f64,
}

impl Term {
    /// `dab`'s.
    fn of(dab: &Imprint) -> Self {
        let hardened = Hardened::new(&Smoothstep, dab.hardness);
        let r = f64::from(dab.radius);
        Self {
            steep: f64::from(hardened.steepest()) / r,
            curve: f64::from(hardened.curvature()) / (r * r),
            strength: f64::from(dab.strength.abs()),
        }
    }

    /// The term for a tile of pen-down bound `e0` and half squared edge `half_h2`.
    fn at(&self, e0: f64, half_h2: f64) -> f64 {
        let share = (self.steep * e0 + self.curve * half_h2).min(1.0);
        // A NaN share (an infinite slope times a zero bound) is the whole move.
        let share = if share.is_nan() { 1.0 } else { share };
        self.strength * share
    }
}

/// How far the finest work's rounding can move a vertex of a finest tile whose correction scales
/// are `scales` (`(t, b, n)`) now, once dabs of strengths summing to `strength` land on it.
///
/// A plane holds every interior value within one final step of its exact value, however often
/// it grew (geom's `fitted_scale`). The dabs move a vertex's correction by at most `strength`
/// along any axis of its (orthonormal) frame, so each plane ends at most `2 (s · 32767 +
/// strength) / 32767` (grown by doubling to hold its largest value), the tangential planes at
/// least 8 times the normal one's; a vertex is within the three final steps' length.
pub(super) fn rounding_bound(scales: [f32; 3], strength: f64) -> f64 {
    const STEPS: f64 = 32_767.0;
    let grown = |scale: f32| {
        let scale = f64::from(scale);
        scale.max(2.0 * scale.mul_add(STEPS, strength) / STEPS)
    };
    let [t, b, n] = scales;
    let normal = grown(n);
    let (tangent, bitangent) = (grown(t).max(8.0 * normal), grown(b).max(8.0 * normal));
    tangent
        .mul_add(tangent, bitangent.mul_add(bitangent, normal * normal))
        .sqrt()
}

/// `v` as `f64` components.
fn wide(v: Vec3) -> [f64; 3] {
    [f64::from(v.x), f64::from(v.y), f64::from(v.z)]
}

/// The tiles' gathered blocks a display preview reads: the store's current blocks of the finest
/// tiles changed since a record, and the pen-down blocks of those the stroke changed.
#[derive(Debug, Default)]
pub(super) struct Blocks {
    /// The store's blocks, with their rings.
    pub(super) now: BTreeMap<TileKey, GatheredBlock>,
    /// The pen-down blocks.
    pub(super) pen_down: BTreeMap<TileKey, GatheredBlock>,
}

impl Blocks {
    /// The finest tiles gathered here under display tile `key`, each once.
    fn under(&self, key: &TileKey, finest: Level) -> impl Iterator<Item = &TileKey> {
        let quads = key.side() - 1;
        let step = 1_u32 << (finest.get() - key.level().get());
        let span = (quads * step / finest.tile_quads().max(1)).max(1);
        let inside = move |t: u16, f: u16| {
            let (t, f) = (u32::from(t), u32::from(f));
            f >= t * span && f < (t + 1) * span
        };
        let (face, tx, ty) = (key.face(), key.tx(), key.ty());
        let only_now = self
            .now
            .keys()
            .filter(|tile| !self.pen_down.contains_key(tile));
        self.pen_down.keys().chain(only_now).filter(move |tile| {
            tile.face() == face && inside(tx, tile.tx()) && inside(ty, tile.ty())
        })
    }

    /// The store's current blocks of `keys` (gathered with their rings), added to these.
    ///
    /// # Errors
    ///
    /// [`CacheError`] if a tile can't be evaluated or gathered.
    pub(super) fn add_now(
        &mut self,
        store: &mut TileStore,
        keys: &BTreeSet<TileKey>,
    ) -> Result<(), CacheError> {
        let keys: Vec<TileKey> = keys
            .iter()
            .filter(|key| !self.now.contains_key(key))
            .copied()
            .collect();
        if keys.is_empty() {
            return Ok(());
        }
        let mut inputs: BTreeSet<TileKey> = BTreeSet::new();
        for key in &keys {
            inputs.extend(store.normal_inputs(key)?.iter().copied());
        }
        let inputs: Vec<TileKey> = inputs.into_iter().collect();
        store.evaluate_many(&inputs, Evaluation::Positions)?;
        let cached = store.cached_positions_with_ring_many(&keys)?;
        for (key, block) in keys.iter().zip(cached) {
            let block = match block {
                Some(block) => block,
                None => store.positions_with_ring(*key)?,
            };
            self.now.insert(*key, block);
        }
        Ok(())
    }

    /// The pen-down blocks of `keys`, added to these.
    ///
    /// # Errors
    ///
    /// As [`TileStore::pen_down_blocks_many`].
    fn add_pen_down(
        &mut self,
        store: &mut TileStore,
        keys: &BTreeSet<TileKey>,
    ) -> Result<(), CacheError> {
        let keys: Vec<TileKey> = keys
            .iter()
            .filter(|key| !self.pen_down.contains_key(key))
            .copied()
            .collect();
        if keys.is_empty() {
            return Ok(());
        }
        let blocks = store.pen_down_blocks_many(&keys)?;
        self.pen_down.extend(keys.into_iter().zip(blocks));
        Ok(())
    }
}

impl DisplayState {
    /// Makes the previews of display tiles `keys` over the store as it holds it now: each its
    /// pen-down tile and base (module notes), moved by `pending` (the dabs shown and not landed,
    /// a suffix of `stroke`), its normals and bound over every dab of `stroke`. A tile the store
    /// has no record of to draw (nor does a renderer) is left out.
    ///
    /// The records are decoded together and the finest blocks gathered once for all of them;
    /// the previews are made on the store's workers.
    ///
    /// # Errors
    ///
    /// [`CacheError`] if a record or a finest tile can't be read.
    pub(super) fn make_many(
        store: &mut TileStore,
        keys: &[TileKey],
        stroke: &[Imprint],
        pending: &[Imprint],
    ) -> Result<Vec<Self>, CacheError> {
        let Some(finest_level) = store.display_finest() else {
            return Ok(Vec::new());
        };
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        let tiles = store.display_for_draw(keys)?;
        let mut inputs = Vec::new();
        let mut changed: BTreeSet<TileKey> = BTreeSet::new();
        let mut at_pen_down = BTreeSet::new();
        for (key, tile) in keys.iter().zip(tiles) {
            let Some(tile) = tile else {
                continue;
            };
            let recorded_bound = store
                .display_stored_error(key)
                .unwrap_or_else(|| tile.error());
            let displacement = store.display_changes(key).map_or(0.0, |(d, under)| {
                changed.extend(under.iter().copied());
                d
            });
            inputs.push((tile, recorded_bound, displacement));
        }
        // The finest tiles whose blocks read one the stroke changed: their pen-down bits are its
        // pen-down blocks', whatever their records hold now (one rebuilt mid-stroke holds bits the
        // stroke moved). The readers of a changed tile are among its normal inputs, as
        // neighbourhood goes both ways.
        let wanted: BTreeSet<TileKey> = keys.iter().copied().collect();
        let changed_by_stroke: Vec<TileKey> = store.pen_down_keys().copied().collect();
        for tile_key in changed_by_stroke {
            for reader in store.normal_inputs(&tile_key)?.iter() {
                let mut above = super::display_ancestors(reader);
                if above.any(|ancestor| wanted.contains(&ancestor)) {
                    at_pen_down.insert(*reader);
                }
            }
        }
        let mut blocks = Blocks::default();
        blocks.add_now(store, &changed)?;
        blocks.add_pen_down(store, &at_pen_down)?;
        let blocks = &blocks;
        Ok(
            store.par_map_mut(&mut inputs, |(tile, recorded_bound, displacement)| {
                Self::build(
                    tile,
                    *recorded_bound,
                    *displacement,
                    finest_level,
                    blocks,
                    stroke,
                    pending,
                )
            }),
        )
    }

    /// Display tile `tile`'s preview (see [`DisplayState::make_many`]), its record's own bound
    /// `recorded_bound`, the steps since moving a finest vertex under it `displacement` at most.
    fn build(
        tile: &DisplayTile,
        recorded_bound: f32,
        displacement: f32,
        finest_level: Level,
        blocks: &Blocks,
        stroke: &[Imprint],
        pending: &[Imprint],
    ) -> Self {
        let key = *tile.key();
        let side = usize::try_from(key.side()).unwrap_or(0);
        let made = Replaced::of(&key, finest_level, tile.positions(), blocks);
        let mut n0 = tile.normals().to_vec();
        let normals = made.normals(&key, finest_level, tile.normals(), blocks);
        for (v, n) in normals {
            if let Some(slot) = n0.get_mut(v) {
                *slot = n;
            }
        }
        let pen_down_bound = if made.flags.iter().any(|&r| r) {
            // Measured where the finest tiles are at hand (every one changed since the record),
            // the record's own bound and the replaced vertices' largest move elsewhere.
            let elsewhere = f64::from(recorded_bound) + made.moved_most;
            farthest_under(&key, finest_level, &made.p0, blocks).map_or_else(
                || {
                    let d = f64::from(displacement);
                    f64::from(recorded_bound) + d + d.max(made.moved_most)
                },
                |measured| measured.max(elsewhere),
            )
        } else {
            f64::from(tile.error())
        };
        let Replaced { base, p0, .. } = made;
        let frames: Vec<[Vec3; 2]> = n0.iter().copied().map(frame_of).collect();
        let per_side = cells_per_side(&key, finest_level);
        let cell_growth = growth_of(pen_down_bound);
        let quads = quads_of(&p0, side, cell_growth);
        let ideal: Vec<[f64; 3]> = p0.iter().copied().map(wide).collect();
        let quad_rows: Vec<Option<Aabb>> = quads
            .chunks(side.saturating_sub(1).max(1))
            .map(|row| {
                row.iter()
                    .filter_map(|quad| quad.bounds)
                    .reduce(Aabb::union)
            })
            .collect();
        let bounds = quad_rows.iter().flatten().copied().reduce(Aabb::union);
        let mut state = Self {
            key,
            bounds,
            turn: vec![[Vec3::ZERO; 2]; p0.len()],
            normals: n0.clone(),
            turned: vec![false; p0.len()],
            drawn: base,
            p0,
            n0,
            frames,
            ideal,
            roughness: tile.roughness().to_vec(),
            pen_down_bound,
            quads,
            quad_rows,
            cells_per_side: per_side,
            cell_growth,
            deepest: 0.0,
            cells_held: 0,
            published_held: None,
            moved: false,
        };
        let landed = stroke.len().saturating_sub(pending.len());
        for (n, dab) in stroke.iter().enumerate() {
            state.take(dab, n >= landed);
        }
        state
    }

    /// The bytes it holds.
    pub(super) fn bytes(&self) -> usize {
        size_of::<Self>()
            + size_of_val(self.p0.as_slice())
            + size_of_val(self.n0.as_slice())
            + size_of_val(self.frames.as_slice())
            + size_of_val(self.turn.as_slice())
            + size_of_val(self.normals.as_slice())
            + size_of_val(self.turned.as_slice())
            + size_of_val(self.drawn.as_slice())
            + size_of_val(self.ideal.as_slice())
            + size_of_val(self.roughness.as_slice())
            + size_of_val(self.quads.as_slice())
            + size_of_val(self.quad_rows.as_slice())
            + self.cells_held * size_of::<f64>()
    }

    /// The tile.
    pub(super) const fn key(&self) -> &TileKey {
        &self.key
    }

    /// Adds `dab` to the preview, after every dab it holds: its move to what is drawn if
    /// `drawn` (a dab still to land), and to the normals, `G` and the bound in any case. Returns
    /// whether it touched the preview (counted in its bound, if nothing else): it is to be
    /// handed to the store again.
    pub(super) fn take(&mut self, dab: &Imprint, drawn: bool) -> bool {
        if !touches(self.bounds.as_ref(), dab) {
            return false;
        }
        let mut touched = false;
        let e0 = self.pen_down_bound;
        let term = Term::of(dab);
        let per_side = self.cells_per_side;
        let side = usize::try_from(self.key.side()).unwrap_or(0);
        let across = side.saturating_sub(1).max(1);
        // Per row of vertices, the columns of those a dab can reach: the quads it touches'.
        let mut near: Vec<Option<(usize, usize)>> = vec![None; side];
        let rows = self.quads.chunks_mut(across).zip(&self.quad_rows);
        for (row, (quads, row_box)) in rows.enumerate() {
            if !touches(row_box.as_ref(), dab) {
                continue;
            }
            for (column, quad) in quads.iter_mut().enumerate() {
                if !touches(quad.bounds.as_ref(), dab) {
                    continue;
                }
                let spans = near.get_mut(row..row + 2).unwrap_or_default();
                for span in spans {
                    let (first, last) = span.unwrap_or((column, column + 1));
                    *span = Some((first.min(column), last.max(column + 1)));
                }
                let share = term.at(e0, quad.half_edge_squared);
                // A quad of one cell: the quad's box is the cell's, rounded up.
                let at_corner = per_side == 1
                    || quad
                        .corners
                        .iter()
                        .any(|&corner| touches(point_box(corner).as_ref(), dab));
                let corners = quad.corners;
                let growth = self.cell_growth;
                let counts = |a: usize, b: usize| {
                    at_corner || touches(cell_box(corners, a, b, per_side, growth).as_ref(), dab)
                };
                let cells = (0..per_side).flat_map(|b| (0..per_side).map(move |a| (a, b)));
                for (n, (a, b)) in cells.enumerate() {
                    if !counts(a, b) {
                        continue;
                    }
                    if quad.terms.is_empty() {
                        quad.terms = vec![0.0; per_side * per_side];
                        self.cells_held += per_side * per_side;
                    }
                    if let Some(sum) = quad.terms.get_mut(n) {
                        *sum += share;
                        self.deepest = self.deepest.max(*sum);
                        touched = true;
                    }
                }
            }
        }
        let mut moved = false;
        let spans = near.into_iter().enumerate();
        for (row, span) in spans {
            let Some((first, last)) = span else {
                continue;
            };
            moved |= self.move_vertices(dab, drawn, row * side + first..row * side + last + 1);
        }
        self.moved |= moved;
        touched || moved
    }

    /// Moves the vertices of `range` by `dab`: what it moves and turns of them (see
    /// [`DisplayState::take`]). Returns whether it moved any.
    fn move_vertices(&mut self, dab: &Imprint, drawn: bool, range: Range<usize>) -> bool {
        let mut moved = false;
        let p0 = self.p0.get(range.clone()).unwrap_or_default();
        let drawn_now = self.drawn.get_mut(range.clone()).unwrap_or_default();
        let ideals = self.ideal.get_mut(range.clone()).unwrap_or_default();
        let frames = self.frames.get(range.clone()).unwrap_or_default();
        let turns = self.turn.get_mut(range.clone()).unwrap_or_default();
        let any_turn = self.turned.get_mut(range).unwrap_or_default();
        let vertices = p0
            .iter()
            .zip(drawn_now)
            .zip(ideals)
            .zip(frames.iter().zip(turns))
            .zip(any_turn);
        for ((((p, out), ideal), (frame, turn)), turned) in vertices {
            let at = [p.x, p.y, p.z];
            match draw_imprint_vertex(&Smoothstep, dab, at) {
                Some(w) if w != 0.0 => {
                    moved = true;
                    let [x, y, z] = draw_move(dab, w);
                    if drawn {
                        *out = *out + Vec3::new(x, y, z);
                    }
                    *ideal = [
                        ideal[0] + f64::from(x),
                        ideal[1] + f64::from(y),
                        ideal[2] + f64::from(z),
                    ];
                }
                _ => {}
            }
            if let Some([gx, gy, gz]) = draw_imprint_gradient(&Smoothstep, dab, at) {
                let gradient = Vec3::new(gx, gy, gz);
                if gradient != Vec3::ZERO {
                    let [nx, ny, nz] = dab.normal;
                    let along = Vec3::new(nx, ny, nz);
                    let [t1, t2] = frame;
                    turn[0] = turn[0] + along * gradient.dot(*t1);
                    turn[1] = turn[1] + along * gradient.dot(*t2);
                    *turned = true;
                }
            }
        }
        moved
    }

    /// Rebases the vertices inside finest tiles `changed` (gathered in `blocks.now`) on the
    /// store's bits, moved by `pending`, the dabs still to land: after a landing changed them.
    /// Returns whether any vertex's bits changed.
    pub(super) fn rebase(
        &mut self,
        finest: Level,
        blocks: &Blocks,
        changed: &BTreeSet<TileKey>,
        pending: &[Imprint],
    ) -> bool {
        let mut any = false;
        for under in changed {
            let Some(block) = blocks.now.get(under) else {
                continue;
            };
            for (v, col, row) in vertices_in(&self.key, finest, under) {
                let (Some(out), Some(start)) = (self.drawn.get_mut(v), self.p0.get(v)) else {
                    continue;
                };
                let mut at = block.get(i64::from(col), i64::from(row));
                for dab in pending {
                    if let Some(w) =
                        draw_imprint_vertex(&Smoothstep, dab, [start.x, start.y, start.z])
                        && w != 0.0
                    {
                        let [x, y, z] = draw_move(dab, w);
                        at = at + Vec3::new(x, y, z);
                    }
                }
                if out.to_bits() != at.to_bits() {
                    *out = at;
                    any = true;
                }
            }
        }
        self.moved |= any;
        any
    }

    /// The preview to hand the store: what is drawn, its normals and bound (module notes),
    /// `held` as given.
    pub(super) fn publish(&mut self, held: bool, rounding: f64, moves_cap: f64) -> DisplayPreview {
        // The normals of the vertices a dab turned since the last, made again.
        let vertices = self
            .normals
            .iter_mut()
            .zip(&mut self.turned)
            .zip(self.n0.iter().zip(&self.frames).zip(&self.turn));
        for ((normal, turned), ((&n0, [t1, t2]), [a1, a2])) in vertices {
            if std::mem::take(turned) {
                *normal = (*t1 + *a1).cross(*t2 + *a2).normalized().unwrap_or(n0);
            }
        }
        let mut deviation_squared = 0.0_f64;
        let mut largest = 0.0_f64;
        for (drawn, ideal) in self.drawn.iter().zip(&self.ideal) {
            let d = wide(*drawn);
            let (dx, dy, dz) = (d[0] - ideal[0], d[1] - ideal[1], d[2] - ideal[2]);
            deviation_squared = deviation_squared.max(dz.mul_add(dz, dx.mul_add(dx, dy * dy)));
            largest = largest.max(d[0].abs().max(d[1].abs()).max(d[2].abs()));
        }
        let deviation = deviation_squared.sqrt();
        // The moves' part: the per-cell sums, or twice the most any place moves, the smaller.
        let terms = self.deepest.min(moves_cap);
        let sum = (self.pen_down_bound + terms + rounding + deviation) * SLACK
            + largest * MAGNITUDE_SLACK;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "rounded up to the next f32 below, so still a bound; past f32's range it \
                      saturates to the largest, the store's unknown bound"
        )]
        let narrow = sum as f32;
        let bound = if narrow.is_finite() {
            if f64::from(narrow) < sum {
                narrow.next_up()
            } else {
                narrow
            }
        } else {
            f32::MAX
        };
        DisplayPreview {
            positions: self.drawn.clone(),
            normals: self.normals.clone(),
            roughness: self.roughness.clone(),
            bound,
            held,
        }
    }
}

/// A display tile's record with the vertices inside the finest tiles gathered replaced: its base
/// (the store's bits) and pen-down tile (the bits as the stroke began).
#[derive(Debug)]
struct Replaced {
    /// The base positions.
    base: Vec<Vec3>,
    /// The pen-down positions, `P0`.
    p0: Vec<Vec3>,
    /// Which vertices were replaced.
    flags: Vec<bool>,
    /// How far a pen-down position moved from the record, at most: the stand-in formula's `M`.
    moved_most: f64,
}

impl Replaced {
    /// `record` (display tile `key`'s positions) with every vertex inside a finest tile of
    /// `blocks` replaced.
    fn of(key: &TileKey, finest_level: Level, record: &[Vec3], blocks: &Blocks) -> Self {
        let mut made = Self {
            base: record.to_vec(),
            p0: record.to_vec(),
            flags: vec![false; record.len()],
            moved_most: 0.0,
        };
        let side = usize::try_from(key.side()).unwrap_or(0).max(1);
        for tile in blocks.under(key, finest_level) {
            let now_block = blocks.now.get(tile);
            let then_block = blocks.pen_down.get(tile);
            for (v, col, row) in vertices_in(key, finest_level, tile) {
                // A vertex on a finest tile's border belongs to the one on `key`'s side of it.
                let (i, j) = (u32::try_from(v % side), u32::try_from(v / side));
                let (Ok(i), Ok(j)) = (i, j) else {
                    continue;
                };
                if finest_place(key, finest_level, i, j).is_none_or(|(owner, ..)| owner != *tile) {
                    continue;
                }
                let (col, row) = (i64::from(col), i64::from(row));
                let now = now_block.map(|block| block.get(col, row));
                let then = then_block.map(|block| block.get(col, row));
                let (Some(slot), Some(start), Some(was), Some(flag)) = (
                    made.base.get_mut(v),
                    made.p0.get_mut(v),
                    record.get(v),
                    made.flags.get_mut(v),
                ) else {
                    continue;
                };
                if let Some(now) = now {
                    *slot = now;
                }
                *start = then.or(now).unwrap_or(*was);
                made.moved_most = made.moved_most.max(f64::from((*start - *was).length()));
                *flag = true;
            }
        }
        made
    }

    /// The pen-down normals of the replaced vertices, made from the pen-down positions (module
    /// notes, "Normals"), each turned to the side of the record's normal `recorded`.
    fn normals(
        &self,
        key: &TileKey,
        finest_level: Level,
        recorded: &[Vec3],
        blocks: &Blocks,
    ) -> Vec<(usize, Vec3)> {
        let side = key.side();
        let width = i64::from(side);
        let normal_at = |v: usize, i: u32, j: u32| -> Option<Vec3> {
            let on_border = i == 0 || j == 0 || i + 1 == side || j + 1 == side;
            let (across, along) = if on_border {
                let (under, col, row) = finest_place(key, finest_level, i, j)?;
                let block = blocks
                    .pen_down
                    .get(&under)
                    .or_else(|| blocks.now.get(&under))?;
                let (col, row) = (i64::from(col), i64::from(row));
                (
                    block.get(col + 1, row) - block.get(col - 1, row),
                    block.get(col, row + 1) - block.get(col, row - 1),
                )
            } else {
                let at = |di: i64, dj: i64| -> Option<Vec3> {
                    let k = (i64::from(j) + dj) * width + i64::from(i) + di;
                    self.p0.get(usize::try_from(k).ok()?).copied()
                };
                (at(1, 0)? - at(-1, 0)?, at(0, 1)? - at(0, -1)?)
            };
            let made = across.cross(along).normalized()?;
            let stored = recorded.get(v).copied()?;
            Some(if made.dot(stored) < 0.0 {
                made * -1.0
            } else {
                made
            })
        };
        let side = usize::try_from(side).unwrap_or(0).max(1);
        self.flags
            .iter()
            .enumerate()
            .filter(|&(_, &r)| r)
            .filter_map(|(v, _)| {
                let (i, j) = (u32::try_from(v % side).ok()?, u32::try_from(v / side).ok()?);
                normal_at(v, i, j).map(|n| (v, n))
            })
            .collect()
    }
}

/// The farthest a gathered finest vertex under display tile `key` lies from `p0`'s triangles.
///
/// Each vertex as the stroke began, from the pen-down tile `p0`'s triangles at its place, in
/// `f64`; `None` if a place falls off the tile (never, for tiles under it).
///
/// The point a tile draws at finest grid place `u` is its quad's triangle blend there, the quad
/// split along its `(i, j)`–`(i+1, j+1)` diagonal, as every level is drawn.
fn farthest_under(key: &TileKey, finest: Level, p0: &[Vec3], blocks: &Blocks) -> Option<f64> {
    let quads = key.side() - 1;
    let step = 1_u32 << (finest.get() - key.level().get());
    let finest_quads = finest.tile_quads().max(1);
    let side = usize::try_from(key.side()).ok()?;
    let corner = |i: u32, j: u32| -> Option<[f64; 3]> {
        let v = p0.get(usize::try_from(j).ok()? * side + usize::try_from(i).ok()?)?;
        Some(wide(*v))
    };
    // Finest place `place` of tile `tile` along an axis of `key` at tile `at`: its quad there,
    // and how far across it.
    let along = |tile: u16, place: u32, at: u16| -> (u32, f64) {
        let finest_at = f64::from(u32::from(tile) * finest_quads + place);
        let across = finest_at / f64::from(step) - f64::from(u32::from(at) * quads);
        let quad = across.floor().clamp(0.0, f64::from(quads - 1));
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a quad index within the tile, clamped to it"
        )]
        let index = quad as u32;
        (index, across - quad)
    };
    let mut farthest = 0.0_f64;
    for tile in blocks.under(key, finest) {
        let Some(block) = blocks.pen_down.get(tile).or_else(|| blocks.now.get(tile)) else {
            continue;
        };
        let columns: Vec<(u32, f64)> = (0..=finest_quads)
            .map(|col| along(tile.tx(), col, key.tx()))
            .collect();
        for row in 0..=finest_quads {
            let (j, fy) = along(tile.ty(), row, key.ty());
            for (col, &(i, fx)) in columns.iter().enumerate() {
                let (c00, c10, c01, c11) = (
                    corner(i, j)?,
                    corner(i + 1, j)?,
                    corner(i, j + 1)?,
                    corner(i + 1, j + 1)?,
                );
                let col = i64::try_from(col).ok()?;
                let fine = wide(block.get(col, i64::from(row)));
                let parts = fine
                    .iter()
                    .zip(c00.iter().zip(&c10))
                    .zip(c01.iter().zip(&c11));
                let gap: f64 = parts
                    .map(|((f, (p00, p10)), (p01, p11))| {
                        let drawn = if fx >= fy {
                            fy.mul_add(p11 - p10, fx.mul_add(p10 - p00, *p00))
                        } else {
                            fx.mul_add(p11 - p01, fy.mul_add(p01 - p00, *p00))
                        };
                        (f - drawn).powi(2)
                    })
                    .sum();
                farthest = farthest.max(gap.sqrt());
            }
        }
    }
    Some(farthest)
}

/// `bounds` grown by `by` on every side, rounded outward; `None` past `f32`'s range.
fn grown(bounds: &Aabb, by: f32) -> Option<Aabb> {
    let (lo, hi) = (bounds.min(), bounds.max());
    let low = Vec3::new(
        (lo.x - by).next_down(),
        (lo.y - by).next_down(),
        (lo.z - by).next_down(),
    );
    let high = Vec3::new(
        (hi.x + by).next_up(),
        (hi.y + by).next_up(),
        (hi.z + by).next_up(),
    );
    Aabb::new(low, high)
}

/// The box of `point` alone; `None` for a point not finite.
fn point_box(point: Vec3) -> Option<Aabb> {
    Aabb::from_points([point])
}

/// The finest quads a side of a cell of a coarse quad's bound ([`Quad`]): eight, smaller than a
/// big brush's dab on the screen, so a cell counts about the dabs overlapping one place.
const CELL_FINEST_QUADS: u32 = 8;

/// How many cells a side each quad of display tile `key` is cut into for its bound: of
/// [`CELL_FINEST_QUADS`] finest quads each ([`Quad`]), at least one, at most 64.
fn cells_per_side(key: &TileKey, finest: Level) -> usize {
    let step = 1_u32 << (finest.get() - key.level().get());
    let cells = (step / CELL_FINEST_QUADS).clamp(1, 64);
    usize::try_from(cells).unwrap_or(1)
}

/// The point at fractions `(fx, fy)` of a quad with corners `[c00, c10, c01, c11]`, on its two
/// triangles (split along `(i, j)`–`(i+1, j+1)`, as every level is drawn).
fn on_quad([c00, c10, c01, c11]: [Vec3; 4], fx: f32, fy: f32) -> Vec3 {
    if fx >= fy {
        c00 + (c10 - c00) * fx + (c11 - c10) * fy
    } else {
        c00 + (c01 - c00) * fy + (c11 - c01) * fx
    }
}

/// A cell's growth for a pen-down bound of `e0`: as an `f32`, rounded up past its rounding.
fn growth_of(e0: f64) -> f32 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a bound in world units, rounded up past f32's rounding by next_up"
    )]
    let narrow = e0 as f32;
    narrow.next_up()
}

/// The box of cell `(a, b)` of a quad with corners `corners`, cut into `per_side`² cells.
///
/// Over the cell's part of the two triangles, whose extremes are at its corners and the ends of
/// the diagonal; then grown by `growth`, its rounding outward.
fn cell_box(corners: [Vec3; 4], a: usize, b: usize, per_side: usize, growth: f32) -> Option<Aabb> {
    #[expect(clippy::cast_precision_loss, reason = "at most 64 cells a side")]
    let fraction = |k: usize| k as f32 / per_side as f32;
    let (x0, x1, y0, y1) = (fraction(a), fraction(a + 1), fraction(b), fraction(b + 1));
    let square = [
        on_quad(corners, x0, y0),
        on_quad(corners, x1, y0),
        on_quad(corners, x0, y1),
        on_quad(corners, x1, y1),
    ];
    // Where the diagonal fx = fy crosses the cell.
    let (low, high) = (x0.max(y0), x1.min(y1));
    let [s0, s1, s2, s3] = square;
    let (d0, d1) = if low <= high {
        (on_quad(corners, low, low), on_quad(corners, high, high))
    } else {
        (s0, s0)
    };
    bounds_of_points(&[s0, s1, s2, s3, d0, d1]).and_then(|cell| grown(&cell, growth))
}

/// The box of a whole quad with corners `corners` (its cells' boxes grown by `growth`): the
/// corners' grown by that and a few roundings of [`on_quad`]'s, so it holds every cell's box.
fn quad_box(corners: [Vec3; 4], growth: f32) -> Option<Aabb> {
    let size = corners
        .iter()
        .map(|c| c.x.abs().max(c.y.abs()).max(c.z.abs()))
        .fold(0.0, f32::max);
    let rounding = 8.0 * f32::EPSILON * size;
    bounds_of_points(&corners).and_then(|quad| grown(&quad, growth + rounding))
}

/// The quads of a tile over `positions` (`side²`, row-major), row-major, their cells' boxes grown
/// by `growth` ([`Quad`]).
fn quads_of(positions: &[Vec3], side: usize, growth: f32) -> Vec<Quad> {
    let at = |i: usize, j: usize| positions.get(j * side + i).copied();
    let quads = side.saturating_sub(1);
    let places = (0..quads).flat_map(|j| (0..quads).map(move |i| (i, j)));
    places
        .map(|(i, j)| {
            let (Some(c00), Some(c10), Some(c01), Some(c11)) =
                (at(i, j), at(i + 1, j), at(i, j + 1), at(i + 1, j + 1))
            else {
                return Quad {
                    corners: [Vec3::ZERO; 4],
                    half_edge_squared: 0.0,
                    bounds: None,
                    terms: Vec::new(),
                };
            };
            let corners = [c00, c10, c01, c11];
            let edges = [(c00, c10), (c00, c01), (c00, c11), (c10, c11), (c01, c11)];
            let longest = edges
                .iter()
                .map(|&(from, to)| {
                    let edge = to - from;
                    f64::from(edge.dot(edge))
                })
                .fold(0.0, f64::max);
            Quad {
                corners,
                half_edge_squared: 0.5 * longest,
                bounds: quad_box(corners, growth),
                terms: Vec::new(),
            }
        })
        .collect()
}
