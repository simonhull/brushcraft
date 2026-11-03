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
//!   replaced, else the stand-in formula `e + D + max(D, M)` from it (`D` the steps since the
//!   record, `M` how far a replaced vertex moved).
//! - Each dab's term is at most `s · min(1, L e₀ + K h² / 2)`, `s` its strength, `L = steepest / r`
//!   and `K = curvature / r²` its falloff's bounds ([`brush::Falloff`]), `h` the quad's longest
//!   pen-down edge: the move at `F₀(u)` against the move at the point of the triangle (Lipschitz),
//!   and that against the triangle's blend of the corners' moves (linear interpolation's error);
//!   both moves lie in `[0, s]` along the dab's normal, so never more than `s`. A dab counts only
//!   in the quads whose pen-down box, grown by `e₀` (where the finest vertices under the quad lie),
//!   its sphere touches; the tile's bound takes the largest quad's sum. So the bound follows the
//!   dabs over each place, not the stroke's length: a long stroke over a coarse tile adds only the
//!   few dabs overlapping any one quad.
//! - `max |D(v) − G(v)|` is measured, in `f64`, over every vertex: the rebased vertices' store bits
//!   against the sum, and every vertex's rounding.
//! - `ε` is taken as `2⁻¹⁰` of the dabs' strengths: a correction is stored to within one step of
//!   its tile's plane scale, at most `8 · |c| / 32767` (the tangential floor), `|c|` the stroke's
//!   moves there.
//!
//! The sum is raised for its own rounding as the stand-ins' is (`render`'s stand-in notes,
//! "Rounding"): times `1 + 2⁻¹⁹`, plus `2⁻¹⁷` of the largest coordinate. The cone is unknown (a
//! renderer takes the one that covers everything).

use std::collections::{BTreeMap, BTreeSet};

use brush::{
    Falloff, Hardened, Imprint, Smoothstep, draw_imprint_gradient, draw_imprint_vertex, draw_move,
};
use geom::{GatheredBlock, Level, TileKey, Vec3};
use store::{Aabb, CacheError, DisplayPreview, Evaluation, TileStore};

use super::{bounds_of_points, touches};

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;

/// What a sum of bounds is multiplied by: `1 + 2⁻¹⁹`, more than its roundings can have taken off.
const SLACK: f64 = 1.0 + 1.0 / 524_288.0;

/// What a bound is raised by per unit of the largest coordinate it read: `2⁻¹⁷`.
const MAGNITUDE_SLACK: f64 = 1.0 / 131_072.0;

/// The finest work's own rounding, per unit of a dab's strength: `2⁻¹⁰` (module notes, "The
/// bound").
const STORED_ROUNDING: f64 = 1.0 / 1024.0;

/// One display tile's preview as the brush keeps it while its stroke lasts (see the module
/// notes).
#[derive(Debug)]
pub(super) struct DisplayState {
    /// The tile.
    key: TileKey,
    /// Vertices per side.
    side: usize,
    /// The pen-down tile's positions, `P0`.
    p0: Vec<Vec3>,
    /// The pen-down normals, `n0`.
    n0: Vec<Vec3>,
    /// Each vertex's tangent frame of `n0`.
    frames: Vec<[Vec3; 2]>,
    /// How far the dabs have turned each tangent of the frame.
    turn: Vec<[Vec3; 2]>,
    /// What is drawn, `D`.
    drawn: Vec<Vec3>,
    /// `G`: the pen-down positions moved by every dab of the stroke, summed in `f64`.
    ideal: Vec<[f64; 3]>,
    /// The record's roughness.
    roughness: Vec<u8>,
    /// The box of the pen-down positions.
    bounds: Option<Aabb>,
    /// `e₀`, the pen-down tile's bound.
    pen_down_bound: f64,
    /// Per quad, row-major, the box of its pen-down corners grown by `e₀` and half the square of
    /// its longest pen-down edge (`h² / 2`): where a dab counts in its bound, and how much.
    quads: Vec<(Option<Aabb>, f64)>,
    /// Per quad, the dabs' terms of the bound, summed.
    terms: Vec<f64>,
    /// Whether it was last handed to the store held.
    pub(super) published_held: Option<bool>,
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

/// The finest tiles under display tile `key`, at `finest`.
fn finest_under(key: &TileKey, finest: Level) -> Vec<TileKey> {
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

/// `d`'s term of the bound for a tile of pen-down bound `e0` and half squared edge `half_h2`
/// (module notes, "The bound").
fn term(dab: &Imprint, e0: f64, half_h2: f64) -> f64 {
    let hardened = Hardened::new(&Smoothstep, dab.hardness);
    let r = f64::from(dab.radius);
    let steep = f64::from(hardened.steepest()) / r;
    let curve = f64::from(hardened.curvature()) / (r * r);
    let s = f64::from(dab.strength.abs());
    let share = (steep * e0 + curve * half_h2).min(1.0);
    // A NaN share (an infinite slope times a zero bound) is the whole move.
    let share = if share.is_nan() { 1.0 } else { share };
    s * share + s * STORED_ROUNDING
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
    /// Makes display tile `key`'s preview over the store as it holds it now: its pen-down tile
    /// and base (module notes), moved by `pending` (the dabs shown and not landed, a suffix of
    /// `stroke`), its normals and bound over every dab of `stroke`. `None` if the store has no
    /// record of the tile to draw (nor does a renderer).
    ///
    /// # Errors
    ///
    /// [`CacheError`] if a record or a finest tile can't be read.
    pub(super) fn make(
        store: &mut TileStore,
        key: TileKey,
        stroke: &[Imprint],
        pending: &[Imprint],
    ) -> Result<Option<Self>, CacheError> {
        let Some(finest_level) = store.display_finest() else {
            return Ok(None);
        };
        let Some(tile) = store.display_for_draw(&[key])?.pop().flatten() else {
            return Ok(None);
        };
        let recorded_bound = store
            .display_stored_error(&key)
            .unwrap_or_else(|| tile.error());
        let (displacement, blocks) = gather(store, &key, finest_level)?;
        let side = usize::try_from(key.side()).unwrap_or(0);
        let made = Replaced::of(&key, finest_level, tile.positions(), &blocks);
        let mut n0 = tile.normals().to_vec();
        let normals = made.normals(&key, finest_level, tile.normals(), &blocks);
        for (v, n) in normals {
            if let Some(slot) = n0.get_mut(v) {
                *slot = n;
            }
        }
        let pen_down_bound = if made.flags.iter().any(|&r| r) {
            let d = f64::from(displacement);
            f64::from(recorded_bound) + d + d.max(made.moved_most)
        } else {
            f64::from(tile.error())
        };
        let Replaced { base, p0, .. } = made;
        let frames: Vec<[Vec3; 2]> = n0.iter().copied().map(frame_of).collect();
        let quads = quads_of(&p0, side, pen_down_bound);
        let ideal: Vec<[f64; 3]> = p0.iter().copied().map(wide).collect();
        let mut state = Self {
            key,
            side,
            bounds: bounds_of_points(&p0),
            turn: vec![[Vec3::ZERO; 2]; p0.len()],
            drawn: base,
            p0,
            n0,
            frames,
            ideal,
            roughness: tile.roughness().to_vec(),
            pen_down_bound,
            terms: vec![0.0; quads.len()],
            quads,
            published_held: None,
        };
        let landed = stroke.len().saturating_sub(pending.len());
        for (n, dab) in stroke.iter().enumerate() {
            state.take(dab, n >= landed);
        }
        Ok(Some(state))
    }

    /// The tile.
    pub(super) const fn key(&self) -> &TileKey {
        &self.key
    }

    /// Adds `dab` to the preview, after every dab it holds: its move to what is drawn if
    /// `drawn` (a dab still to land), and to the normals, `G` and the bound in any case. Returns
    /// whether it moved a vertex.
    pub(super) fn take(&mut self, dab: &Imprint, drawn: bool) -> bool {
        if !touches(self.bounds.as_ref(), dab) {
            return false;
        }
        let e0 = self.pen_down_bound;
        for ((bounds, half_h2), sum) in self.quads.iter().zip(&mut self.terms) {
            if touches(bounds.as_ref(), dab) {
                *sum += term(dab, e0, *half_h2);
            }
        }
        let mut moved = false;
        let vertices = self
            .p0
            .iter()
            .zip(&mut self.drawn)
            .zip(&mut self.ideal)
            .zip(self.frames.iter().zip(&mut self.turn));
        for (((p, out), ideal), (frame, turn)) in vertices {
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
                }
            }
        }
        moved
    }

    /// Rebases the vertices inside finest tiles `changed` (gathered in `blocks.now`) on the
    /// store's bits, moved by `pending`, the dabs still to land: after a landing changed them.
    /// Returns whether any vertex was rebased.
    pub(super) fn rebase(
        &mut self,
        finest: Level,
        blocks: &Blocks,
        changed: &BTreeSet<TileKey>,
        pending: &[Imprint],
    ) -> bool {
        let mut any = false;
        let side = u32::try_from(self.side).unwrap_or(0);
        let grid = (0..side).flat_map(|j| (0..side).map(move |i| (i, j)));
        for (v, (i, j)) in grid.enumerate() {
            let Some((under, col, row)) = finest_place(&self.key, finest, i, j) else {
                continue;
            };
            if !changed.contains(&under) {
                continue;
            }
            let Some(block) = blocks.now.get(&under) else {
                continue;
            };
            let (Some(out), Some(start)) = (self.drawn.get_mut(v), self.p0.get(v)) else {
                continue;
            };
            let mut at = block.get(i64::from(col), i64::from(row));
            for dab in pending {
                if let Some(w) = draw_imprint_vertex(&Smoothstep, dab, [start.x, start.y, start.z])
                    && w != 0.0
                {
                    let [x, y, z] = draw_move(dab, w);
                    at = at + Vec3::new(x, y, z);
                }
            }
            *out = at;
            any = true;
        }
        any
    }

    /// The preview to hand the store: what is drawn, its normals and bound (module notes),
    /// `held` as given.
    pub(super) fn publish(&self, held: bool) -> DisplayPreview {
        let normals = self
            .n0
            .iter()
            .zip(&self.frames)
            .zip(&self.turn)
            .map(|((&n0, [t1, t2]), [a1, a2])| {
                if *a1 == Vec3::ZERO && *a2 == Vec3::ZERO {
                    return n0;
                }
                (*t1 + *a1).cross(*t2 + *a2).normalized().unwrap_or(n0)
            })
            .collect();
        let mut deviation = 0.0_f64;
        let mut largest = 0.0_f64;
        for (drawn, ideal) in self.drawn.iter().zip(&self.ideal) {
            let d = wide(*drawn);
            let gap = (d[0] - ideal[0])
                .hypot(d[1] - ideal[1])
                .hypot(d[2] - ideal[2]);
            deviation = deviation.max(gap);
            largest = largest.max(d[0].abs().max(d[1].abs()).max(d[2].abs()));
        }
        let terms = self.terms.iter().copied().fold(0.0, f64::max);
        let sum = (self.pen_down_bound + terms + deviation) * SLACK + largest * MAGNITUDE_SLACK;
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
            normals,
            roughness: self.roughness.clone(),
            bound,
            held,
        }
    }
}

/// The finest blocks display tile `key` is made from, and how far the steps since its record
/// moved a finest vertex under it.
///
/// The store's blocks of the finest tiles changed since the record, and the pen-down blocks of
/// those whose blocks read a tile the stroke changed (module notes, "The pen-down tile"), at the
/// finest level `finest_level`.
///
/// # Errors
///
/// [`CacheError`] if a finest tile can't be listed, evaluated or gathered.
fn gather(
    store: &mut TileStore,
    key: &TileKey,
    finest_level: Level,
) -> Result<(f32, Blocks), CacheError> {
    let (displacement, changed) = store
        .display_changes(key)
        .map_or((0.0, BTreeSet::new()), |(d, under)| (d, under.clone()));
    let mut at_pen_down = BTreeSet::new();
    for tile_key in finest_under(key, finest_level) {
        let read = store.normal_inputs(&tile_key)?;
        if read
            .iter()
            .any(|input| store.pen_down_shape(input).is_some())
        {
            at_pen_down.insert(tile_key);
        }
    }
    let mut blocks = Blocks::default();
    blocks.add_now(store, &changed)?;
    blocks.add_pen_down(store, &at_pen_down)?;
    Ok((displacement, blocks))
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
        let grid = (0..key.side()).flat_map(|j| (0..key.side()).map(move |i| (i, j)));
        for (v, (i, j)) in grid.enumerate() {
            let Some((under, col, row)) = finest_place(key, finest_level, i, j) else {
                continue;
            };
            let (col, row) = (i64::from(col), i64::from(row));
            let now = blocks.now.get(&under).map(|block| block.get(col, row));
            let then = blocks.pen_down.get(&under).map(|block| block.get(col, row));
            if now.is_none() && then.is_none() {
                continue;
            }
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

/// Per quad of a tile over `positions` (`side²`, row-major), row-major: its box and half edge.
///
/// The box of its corners grown by `grow`, and half the square of the longest edge of its two
/// triangles (its four edges and its `(i, j)`–`(i+1, j+1)` diagonal).
fn quads_of(positions: &[Vec3], side: usize, grow: f64) -> Vec<(Option<Aabb>, f64)> {
    let at = |i: usize, j: usize| positions.get(j * side + i).copied();
    let quads = side.saturating_sub(1);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a bound in world units, rounded up past f32's rounding by next_up"
    )]
    let grow = (grow as f32).next_up();
    let cells = (0..quads).flat_map(|j| (0..quads).map(move |i| (i, j)));
    cells
        .map(|(i, j)| {
            let (Some(c00), Some(c10), Some(c01), Some(c11)) =
                (at(i, j), at(i + 1, j), at(i, j + 1), at(i + 1, j + 1))
            else {
                return (None, 0.0);
            };
            let edges = [(c00, c10), (c00, c01), (c00, c11), (c10, c11), (c01, c11)];
            let longest = edges
                .iter()
                .map(|&(from, to)| {
                    let edge = to - from;
                    f64::from(edge.dot(edge))
                })
                .fold(0.0, f64::max);
            let corners = [c00, c10, c01, c11];
            let bounds = bounds_of_points(&corners).and_then(|corners| grown(&corners, grow));
            (bounds, 0.5 * longest)
        })
        .collect()
}
