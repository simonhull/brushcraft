//! The pop a tile's refinement leaves: the level choice refines a tile into its children, whose
//! even vertices stand where the tile's odd vertices were morphing to.

use super::*;
use crate::cull::GpuBounds;
use crate::cull::reference::projected;

/// An interior odd vertex of a drawn tile, probed: where it is drawn, and where it is going.
struct Odd {
    /// The tile.
    tile: Place,
    /// Where the vertex shader drew it.
    drawn: [f32; 3],
    /// Its own position: where the tile's children draw it.
    position: [f32; 3],
    /// Where its morph starts: the midpoint of its parents.
    mid: [f32; 3],
}

/// The interior odd vertices of `cut`'s tiles above the finest level, probed as drawn in the
/// raster's last frame (`clusters`, `list` its cluster list).
fn probe_odd(
    gpu: &Gpu,
    raster: &Raster,
    world: &World,
    clusters: &[GpuCluster],
    list: &Buffer,
) -> Vec<Odd> {
    let mut probes = Vec::new();
    let mut odds = Vec::new();
    for (instance, cluster) in clusters.iter().enumerate() {
        let tile = world.place_of(cluster.node);
        if tile.level() == world.finest {
            continue;
        }
        let quads = tile.level().tile_quads();
        let (cx, cy) = cluster.position();
        let sides = cluster.quads();
        let data = &world.tiles[&tile].vertices;
        let row = quads + 1;
        let at = |i: u32, j: u32| data[usize::try_from(j * row + i).expect("small")].position;
        let grid = (0..sides).flat_map(|v| (0..sides).map(move |u| (u, v)));
        for (u, v) in grid {
            let (column, line) = (cx * 8 + u, cy * 8 + v);
            let interior = column > 0 && line > 0 && column < quads && line < quads;
            if !interior || (column % 2 == 0 && line % 2 == 0) {
                continue;
            }
            let (across, up) = (column % 2, line % 2);
            let low = at(column - across, line - up);
            let high = at(column + across, line + up);
            #[expect(
                clippy::manual_midpoint,
                reason = "the raster's arithmetic, (a + b) · 0.5, bit for bit"
            )]
            let mid = [0, 1, 2].map(|k| (low[k] + high[k]) * 0.5);
            let vertex = GridVertex::new(
                u8::try_from(u).expect("small"),
                u8::try_from(v).expect("small"),
            )
            .expect("on the grid");
            probes.push([
                u32::try_from(instance).expect("small"),
                u32::from(vertex.index()),
            ]);
            odds.push(Odd {
                tile,
                drawn: [0.0; 3],
                position: at(column, line),
                mid,
            });
        }
    }
    if probes.is_empty() {
        return odds;
    }
    let placed = probe(gpu, raster, world, list, &probes);
    for (odd, p) in odds.iter_mut().zip(&placed) {
        odd.drawn = [p[0], p[1], p[2]];
    }
    odds
}

/// `p - q`'s length.
fn distance(p: [f32; 3], q: [f32; 3]) -> f64 {
    let d = [0, 1, 2].map(|k| f64::from(p[k]) - f64::from(q[k]));
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

/// Pixels a length of `e` at `p` covers in `view`, `None` behind the near plane.
fn pixels_at(view: &View, e: f32, p: [f32; 3]) -> Option<f64> {
    let eye = view.eye();
    let forward = view.forward();
    let q = Vec3::new(
        f64::from(p[0]) - eye.x,
        f64::from(p[1]) - eye.y,
        f64::from(p[2]) - eye.z,
    );
    let depth = q.dot(forward);
    (depth > view.near()).then(|| f64::from(e) * view.pixels_per_unit() / depth)
}

/// What a camera's approach found where the level choice refined a drawn tile.
#[derive(Debug, Default)]
struct Pops {
    /// Odd vertices checked.
    checked: usize,
    /// Of which their morph had further to go than the bound allows.
    over: usize,
    /// Tiles refined by their own bound (the level choice's conservative one past the
    /// threshold), whose odd vertices were checked.
    refined: usize,
    /// Tiles refined for another reason (the neighbour rule), not checked.
    pulled: usize,
    /// The largest pop, pixels.
    worst: f64,
}

/// Approaches the sphere from 3 to about 1.2, measuring each pop where a drawn tile refines.
///
/// Wherever the level choice refines a drawn tile between two frames, it measures how far each
/// of the tile's interior odd vertices jumps to its children's even vertex there: the part of
/// its morph left, `1 - t`, against the bound the error-bounded ramp gives it.
///
/// At the frame before, `t = (x - r) / (x - y)` leaves `(r - y) / (x - y)` of the way, `x` and
/// `y` the parent's and the tile's own `e` in pixels at the vertex and `r` the threshold. The
/// level choice refines the tile when its conservative bound `y_c` (over the tile's sphere, in
/// the frame after) passes `r`, so `r - y <= y_c - y`: the part left is at most the gap between
/// the conservative bound and the vertex's projection, over the morph's span `x - y`. Checked
/// to a tolerance of 1e-3 (the f32 arithmetic and the `e` the GPU reads) plus the change of
/// `y` over one camera step (about 3 %, a step being 3 % closer: `0.03 y / (x - y)`).
fn approach(gpu: &Gpu, world: &World, raster: &mut Raster, path: Path) -> Pops {
    let (mut lod, requests) = level_choice(gpu, world);
    let mut pops = Pops::default();
    let direction = Vec3::new(0.21, 0.34, 1.0).normalized().expect("not zero");
    let mut previous: Option<Before> = None;
    let mut distance_from_centre = path.from;
    let mut first = true;
    let mut bounds_of: Vec<GpuBounds> = Vec::new();
    for _ in 0..path.steps {
        let eye = Vec3::new(
            direction.x * distance_from_centre,
            direction.y * distance_from_centre,
            direction.z * distance_from_centre,
        );
        let view = View::look_at(
            eye,
            Vec3::default(),
            Vec3::new(0.0, 1.0, 0.0),
            45_f64.to_radians(),
            size(),
            0.01,
        )
        .expect("a view");
        let frame = RasterFrame {
            view: &view,
            time: Duration::ZERO,
            moving: true,
        };
        let (_, clusters, cut) = lod_frame(gpu, world, &mut lod, &requests, raster, &frame, first);
        if first {
            bounds_of = gpu.read_buffer(lod.bounds().buffer()).expect("read back");
            first = false;
        }
        // Probe what this frame drew, with every cut tile's clusters listed (some culled).
        let all = cut_clusters(world, &cut, true);
        let (list, _) = cluster_buffers(gpu, &all);
        let odds = probe_odd(gpu, raster, world, &all, &list);
        let drawn_now: BTreeSet<u32> = clusters.iter().map(|c| c.node).collect();
        if let Some((before, before_odds, before_view, drawn)) = &previous {
            let refined = before.iter().filter(|t| {
                !cut.contains(t) && t.children().into_iter().flatten().all(|c| cut.contains(&c))
            });
            let refined: BTreeSet<Place> = refined.copied().collect();

            for &tile in &refined {
                let node = world.node(tile);
                if !drawn.contains(&node) {
                    continue;
                }
                let own = world.tiles[&tile].detail.error;
                let parent = world.parent_error(tile);
                let b = bounds_of[usize::try_from(node).expect("small")];
                let conservative = f64::from(projected(&view.gpu(), &b, own, own));
                let r = 1.0;
                if conservative <= r {
                    pops.pulled += 1;
                    continue;
                }
                pops.refined += 1;
                let tile_odds = before_odds.iter().filter(|o| o.tile == tile);
                for odd in tile_odds {
                    let (Some(x), Some(y)) = (
                        pixels_at(before_view, parent, odd.position),
                        pixels_at(before_view, own, odd.position),
                    ) else {
                        continue;
                    };
                    let span = distance(odd.position, odd.mid);
                    if span < 1e-7 || x - y <= 0.0 {
                        continue;
                    }
                    let left = distance(odd.drawn, odd.position) / span;
                    let step = 1.0 / path.factor - 1.0;
                    let bound = ((conservative - y) / (x - y)).max(0.0) + step * y / (x - y) + 1e-3;
                    pops.checked += 1;
                    if left > bound {
                        pops.over += 1;
                    }
                    let jump = pixels_at(before_view, 1.0, odd.position)
                        .map_or(0.0, |unit| unit * distance(odd.drawn, odd.position));
                    pops.worst = pops.worst.max(jump);
                }
            }
        }
        previous = Some((cut, odds, view, drawn_now));
        distance_from_centre *= path.factor;
    }
    pops
}

/// The last frame's cut, its probed odd vertices, its view and the nodes it drew.
type Before = (BTreeSet<Place>, Vec<Odd>, View, BTreeSet<u32>);

/// A camera's approach: from `from` away from the centre, `steps` steps each `factor` of the
/// distance before.
#[derive(Debug, Clone, Copy)]
struct Path {
    /// The first distance.
    from: f64,
    /// Steps.
    steps: u32,
    /// Each step's distance over the last's.
    factor: f64,
}

/// The raster morphing by distance alone, by the shipped ramp or, `old`, the first cut's: by
/// the parent's bound alone, 0 below the threshold and 1 at twice it.
fn ramp_raster(gpu: &Gpu, world: &World, old: bool) -> Raster {
    let config = RasterConfig::default().with_morph(Duration::ZERO);
    let mut raster = Raster::new(gpu, world.tree.capacity(), config).expect("the raster");
    if old {
        raster.patches = vec![(
            "return clamp((x - r) / (x - y), 0.0, 1.0);",
            "return clamp(x / r - 1.0, 0.0, 1.0);",
        )];
    }
    raster
}

#[test]
fn a_tile_refining_into_its_children_jumps_no_further_than_the_level_choice_s_slack() {
    GpuTest::run(|gpu| {
        let mut world = World::new(gpu);
        // The sphere's true bounds, near: the level choice's bound over a tile's whole sphere
        // is loose enough there that tiles refine before their parent's e reaches a pixel at
        // any vertex, so every pop is under a pixel either way.
        let near = Path {
            from: 3.0,
            steps: 30,
            factor: 0.97,
        };
        let pops = approach(gpu, &world, &mut ramp_raster(gpu, &world, false), near);
        println!("true bounds, near: {pops:?}");
        assert!(pops.refined >= 3 && pops.checked > 100, "{pops:?}");
        assert_eq!(pops.over, 0, "every pop within the slack: {pops:?}");
        assert!(pops.worst < 1.0, "under a pixel: {pops:?}");
        // Bounds a parent only half again its child's (as where detail too fine to draw
        // dominates every level's e; the data unchanged), refining far away, where the level
        // choice's bound is within a few per cent of the vertices' projections: the morph must
        // be nearly done when the tile refines.
        let display = world.tiles.keys().filter(|p| p.level() < world.finest);
        let display: Vec<Place> = display.copied().collect();
        for place in display {
            let below = i32::from(world.finest.get() - 1 - place.level().get());
            set_error(gpu, &mut world, place, 0.05 * 1.5_f32.powi(below));
        }
        let far = Path {
            from: 300.0,
            steps: 70,
            factor: 0.95,
        };
        let new = approach(gpu, &world, &mut ramp_raster(gpu, &world, false), far);
        println!("error-bounded ramp, far: {new:?}");
        assert!(new.refined >= 3 && new.checked > 100, "{new:?}");
        assert_eq!(new.over, 0, "every pop within the slack: {new:?}");
        // The bite: the first cut's ramp leaves more of the morph to jump.
        let old = approach(gpu, &world, &mut ramp_raster(gpu, &world, true), far);
        println!("parent-bound ramp, far: {old:?}");
        assert!(old.over > 0, "the old ramp pops past the slack: {old:?}");
    });
}
