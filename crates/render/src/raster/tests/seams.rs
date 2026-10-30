//! Every odd vertex on a seam, mid-morph, is drawn at the same bits from both sides.
//!
//! Probed from each tile's cluster, across face frames rotated against each other and between
//! cousins (tiles of one face with different parents), on a sphere to level 8.

use super::*;

/// One seam vertex as each side's cluster names it: (cluster instance, grid vertex) twice.
struct Pair {
    /// This side.
    ours: [u32; 2],
    /// The other side.
    theirs: [u32; 2],
    /// Whether the two tiles are on different faces.
    across_faces: bool,
    /// Whether they are at one level with different parents.
    cousins: bool,
}

/// Where vertex `(i, j)` of `place` sits in its clusters: the instance of `clusters` and the
/// grid vertex there.
fn cluster_vertex(
    world: &World,
    instances: &BTreeMap<(u32, u32, u32), u32>,
    place: Place,
    (i, j): (u32, u32),
) -> [u32; 2] {
    let quads = place.level().tile_quads();
    let per = (quads / 8).max(1);
    let (cx, cy) = ((i / 8).min(per - 1), (j / 8).min(per - 1));
    let (u, v) = (i - 8 * cx, j - 8 * cy);
    let instance = instances[&(world.node(place), cx, cy)];
    let vertex = GridVertex::new(
        u8::try_from(u).expect("small"),
        u8::try_from(v).expect("small"),
    )
    .expect("on the grid");
    [instance, u32::from(vertex.index())]
}

/// The tile of `cut` at `level` holding the vertex at `(x, y)` of `face`'s grid at that level,
/// and its place in that tile, if one does; any copy across faces.
fn holder(
    world: &World,
    cut: &BTreeSet<Place>,
    face: FaceId,
    at: Level,
    (x, y): (u32, u32),
    not: Place,
) -> Option<(Place, (u32, u32))> {
    let copies = world.mesh.topology().copies(face, at, x, y);
    copies.iter().find_map(|c| {
        let place = Place::of(&TileKey::display(c.face, at, c.tx, c.ty).ok()?);
        (place != not && cut.contains(&place)).then_some((place, (c.i, c.j)))
    })
}

/// Every odd border vertex a cut's tile draws (not one its edge variant replaces) and the
/// other side's name for it: a tile at its level, or a finer tile's even vertex.
fn seam_pairs(
    world: &World,
    cut: &BTreeSet<Place>,
    instances: &BTreeMap<(u32, u32, u32), u32>,
) -> Vec<Pair> {
    let mut pairs = Vec::new();
    for &tile in cut {
        let quads = tile.level().tile_quads();
        let key = tile.key(world.finest).expect("a tile");
        let (x0, y0) = (u32::from(key.tx()) * quads, u32::from(key.ty()) * quads);
        let sides = across(world, cut, tile);
        let border = (0..=quads).flat_map(|t| [(t, 0), (quads, t), (t, quads), (0, t)]);
        for (i, j) in border {
            if i % 2 == 0 && j % 2 == 0 {
                continue;
            }
            let edge = match (i, j) {
                (_, 0) => 0,
                (q, _) if q == quads => 1,
                (_, q) if q == quads => 2,
                _ => 3,
            };
            let (x, y) = (x0 + i, y0 + j);
            let other = match sides[edge] {
                Across::Coarser => continue,
                Across::Same => holder(world, cut, tile.face(), tile.level(), (x, y), tile),
                Across::Finer => {
                    let finer = tile.level().child().expect("a finer level");
                    holder(world, cut, tile.face(), finer, (2 * x, 2 * y), tile)
                }
            };
            let Some((place, at)) = other else { continue };
            let cousins = place.level() == tile.level() && place.parent() != tile.parent();
            pairs.push(Pair {
                ours: cluster_vertex(world, instances, tile, (i, j)),
                theirs: cluster_vertex(world, instances, place, at),
                across_faces: place.face() != tile.face(),
                cousins,
            });
        }
    }
    pairs
}

/// For random cuts of `world` mid-morph, seam vertices compared and those whose bits differ
/// between the two sides, drawn by `raster`; and how many pairs crossed faces and joined
/// cousins.
fn compare_seams(gpu: &Gpu, world: &World, raster: &mut Raster) -> (usize, usize, usize, usize) {
    let mut rng = Rng(29);
    let (mut compared, mut differ, mut faces, mut cousins) = (0, 0, 0, 0);
    for &low in &[3, 4, 5] {
        let (view, _) = random_view(&mut rng, (1.6, 2.6));
        let base: BTreeSet<Place> = world
            .tiles
            .keys()
            .filter(|p| p.level().get() == low)
            .copied()
            .collect();
        let first = grown(&mut rng, world, base.clone(), 3);
        let cut = grown(&mut rng, world, first.clone(), 2);
        let at = |ms: u64| RasterFrame {
            view: &view,
            time: Duration::from_millis(ms),
            moving: false,
        };
        for (stage, ms) in [(&base, 800), (&first, 900), (&cut, 1000), (&cut, 1040)] {
            let stage_clusters = cut_clusters(world, stage, true);
            draw_cut(gpu, raster, world, stage, &stage_clusters, &at(ms));
        }
        let clusters = cut_clusters(world, &cut, true);
        let instances: BTreeMap<(u32, u32, u32), u32> = clusters
            .iter()
            .enumerate()
            .map(|(k, c)| {
                let (cx, cy) = c.position();
                ((c.node, cx, cy), u32::try_from(k).expect("small"))
            })
            .collect();
        let pairs = seam_pairs(world, &cut, &instances);
        let probes: Vec<[u32; 2]> = pairs.iter().flat_map(|p| [p.ours, p.theirs]).collect();
        let (list, _) = cluster_buffers(gpu, &clusters);
        let placed = probe(gpu, raster, world, &list, &probes);
        let compared_pairs = pairs.iter().zip(placed.chunks(2));
        for (pair, both) in compared_pairs {
            compared += 1;
            faces += usize::from(pair.across_faces);
            cousins += usize::from(pair.cousins);
            let bits = |p: [f32; 4]| [p[0], p[1], p[2]].map(f32::to_bits);
            if bits(both[0]) != bits(both[1]) {
                differ += 1;
            }
        }
    }
    (compared, differ, faces, cousins)
}

#[test]
fn every_odd_seam_vertex_mid_morph_is_drawn_at_the_same_bits_from_both_sides() {
    GpuTest::run(|gpu| {
        let mut world = World::with_finest(gpu, FINEST + 1);
        // Each display tile's bound 5 to 20 times the sphere's, at random (its data unchanged):
        // the two sides of a seam then hold different bounds, and the morph by distance stands
        // part way at many seam vertices.
        let mut rng = Rng(31);
        let display = world.tiles.keys().filter(|p| p.level() < world.finest);
        let display: Vec<Place> = display.copied().collect();
        for place in display {
            let error = world.tiles[&place].detail.error * rng.range(5.0, 20.0);
            set_error(gpu, &mut world, place, error);
        }
        let mut raster = Raster::new(
            gpu,
            world.tree.capacity(),
            world.layout(),
            RasterConfig::default(),
        )
        .expect("the raster");
        let (compared, differ, faces, cousins) = compare_seams(gpu, &world, &mut raster);
        println!(
            "{compared} seam vertices ({faces} across faces, {cousins} between cousins), {differ} differ"
        );
        assert!(
            compared > 1000 && faces > 100 && cousins > 100,
            "seams were tested"
        );
        assert_eq!(
            differ, 0,
            "both sides draw every seam vertex at the same bits"
        );
        // The bite: each side by its own factors alone.
        let unshared = [
            (
                "e = select(max(e, other), -1.0, other < 0.0 || e < 0.0);",
                "e = select(e, -1.0, other < 0.0 || e < 0.0);",
            ),
            ("own = max(own, tree[m].error);", "own = own;"),
            ("time = min(time, morph_time(m));", "time = time;"),
        ];
        let mut broken = Raster::patched(
            gpu,
            world.tree.capacity(),
            world.layout(),
            RasterConfig::default(),
            &unshared,
        )
        .expect("the raster");
        let (_, broken_differ, _, _) = compare_seams(gpu, &world, &mut broken);
        println!("each side by its own factors: {broken_differ} differ");
        assert!(
            broken_differ > 0,
            "without the shared factors the sides disagree"
        );
    });
}
