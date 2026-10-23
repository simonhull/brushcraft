//! The BVH against brute force: random scenes of boxes, thousands of sphere queries and rays,
//! before and after refits; the tree's own invariants; and the quality of its splits.

#![expect(clippy::missing_panics_doc, reason = "tests panic to fail")]

use std::collections::HashMap;
use std::convert::Infallible;

use geom::{FaceId, Level};

use super::*;
use crate::index::NormalCone;

/// Deterministic random numbers (splitmix64).
pub(crate) struct Rng(pub(crate) u64);

impl Rng {
    /// The next 64 random bits.
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`, 16 bits of resolution.
    pub(crate) fn unit(&mut self) -> f32 {
        f32::from(u16::try_from(self.next_u64() >> 48).unwrap()) / 65536.0
    }

    /// Uniform in `[low, high)`.
    pub(crate) fn range(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.unit()
    }

    /// Uniform in `0..n`; `n > 0`.
    pub(crate) fn below(&mut self, n: usize) -> usize {
        usize::try_from(self.next_u64() % u64::try_from(n).unwrap()).unwrap()
    }

    /// A point uniform in the cube `[low, high)³`.
    pub(crate) fn point(&mut self, low: f32, high: f32) -> Vec3 {
        Vec3::new(
            self.range(low, high),
            self.range(low, high),
            self.range(low, high),
        )
    }
}

/// A distinct key for item `i` of a scene: level-9 tiles, 64 to a face.
fn key(i: usize) -> TileKey {
    let i = u32::try_from(i).unwrap();
    let (tx, ty) = (i % 8, (i / 8) % 8);
    TileKey::displacement(
        FaceId::new(i / 64),
        Level::new(9).unwrap(),
        u16::try_from(tx).unwrap(),
        u16::try_from(ty).unwrap(),
    )
    .unwrap()
}

/// Bounds of a box around `center` with half-extents `half`, and an all-directions cone.
fn bounds(center: Vec3, half: Vec3) -> TileBounds {
    TileBounds {
        aabb: Aabb::new(center - half, center + half).unwrap(),
        cone: NormalCone::EVERYTHING,
    }
}

/// A random box: centre uniform in `[low, high)³`, half-extents from 0.001 to `size` per axis
/// (some of them flat in one axis, as a tile seen edge-on is).
fn random_bounds(rng: &mut Rng, low: f32, high: f32, size: f32) -> TileBounds {
    let center = rng.point(low, high);
    let mut half = Vec3::new(
        rng.range(0.001, size),
        rng.range(0.001, size),
        rng.range(0.001, size),
    );
    if rng.below(8) == 0 {
        half.z = 0.0;
    }
    bounds(center, half)
}

/// `n` random boxes in the unit cube, keyed `0..n`.
fn random_scene(rng: &mut Rng, n: usize, size: f32) -> Vec<(TileKey, TileBounds)> {
    (0..n)
        .map(|i| (key(i), random_bounds(rng, 0.0, 1.0, size)))
        .collect()
}

/// Every tile of `scene` whose box touches the sphere, sorted: the brute-force answer.
fn brute_sphere(scene: &[(TileKey, TileBounds)], center: Vec3, radius: f32) -> Vec<TileKey> {
    let mut keys: Vec<TileKey> = scene
        .iter()
        .filter(|(_, b)| b.aabb.distance_squared(center) <= radius * radius)
        .map(|&(k, _)| k)
        .collect();
    keys.sort_unstable();
    keys
}

/// The tree's answer to the same query, sorted (duplicates kept, so they show).
fn tree_sphere(bvh: &Bvh, center: Vec3, radius: f32) -> Vec<TileKey> {
    let mut keys: Vec<TileKey> = bvh.query_sphere(center, radius).collect();
    keys.sort_unstable();
    keys
}

/// Where `ray` first meets the ball inscribed in `aabb`: the "precise" surface of a random tile.
///
/// The ball is centred in the box, as large as its narrowest axis allows. `None` for a miss, or
/// for a flat box, whose ball is a point.
fn hit_ball(ray: &Ray, aabb: &Aabb) -> Option<f32> {
    let half = (aabb.max() - aabb.min()) * 0.5;
    let radius = half.x.min(half.y).min(half.z);
    if radius <= 0.0 {
        return None;
    }
    let to = ray.origin() - aabb.centroid();
    let d = ray.direction();
    let a = d.dot(d);
    let b = 2.0 * d.dot(to);
    let c = to.dot(to) - radius * radius;
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return None;
    }
    let root = disc.sqrt();
    [(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)]
        .into_iter()
        .find(|&t| t >= 0.0)
}

/// The nearest ball hit over every tile of `scene`: the brute-force raycast.
fn brute_ray(scene: &[(TileKey, TileBounds)], ray: &Ray) -> Option<Hit> {
    scene
        .iter()
        .filter_map(|&(key, b)| hit_ball(ray, &b.aabb).map(|t| Hit { key, t }))
        .min_by(|a, b| a.t.total_cmp(&b.t))
}

/// The tree's raycast against the same balls, as they are in `scene`; also counts the precise
/// tests it made.
fn tree_ray(bvh: &Bvh, scene: &[(TileKey, TileBounds)], ray: &Ray) -> (Option<Hit>, usize) {
    let lookup: HashMap<TileKey, Aabb> = scene.iter().map(|&(k, b)| (k, b.aabb)).collect();
    let mut tests = 0;
    let hit = bvh
        .raycast(ray, |key| {
            tests += 1;
            Ok::<_, Infallible>(hit_ball(ray, &lookup[&key]))
        })
        .unwrap();
    (hit, tests)
}

/// A random ray from around the unit cube, aimed at a point inside it; every fifth one runs
/// parallel to two axes, which gives the slab test infinite reciprocals.
fn random_ray(rng: &mut Rng, k: usize) -> Ray {
    let origin = rng.point(-0.5, 1.5);
    let target = rng.point(0.0, 1.0);
    let mut direction = target - origin;
    if k.is_multiple_of(5) {
        direction = match rng.below(3) {
            0 => Vec3::new(direction.x, 0.0, 0.0),
            1 => Vec3::new(0.0, direction.y, 0.0),
            _ => Vec3::new(0.0, 0.0, direction.z),
        };
    }
    Ray::new(origin, direction).unwrap()
}

/// Checks the tree's invariants.
///
/// Every node's box is exactly the union of what is below it, children come after their parent
/// as a pair and point back to it, leaves are non-empty and small and cover every item once, and
/// the lookups agree with the items.
fn check(bvh: &Bvh) {
    assert_eq!(bvh.leaf_of.len(), bvh.items.len(), "a leaf per item");
    assert_eq!(bvh.index_of.len(), bvh.items.len(), "an index per item");
    if bvh.nodes.is_empty() {
        assert!(bvh.items.is_empty(), "no nodes, no items");
    } else {
        assert_eq!(bvh.parent(0), None, "the root has no parent");
        assert_eq!(
            bvh.parents.len(),
            (bvh.nodes.len() - 1) / 2,
            "a parent per pair of children"
        );
    }
    let mut covered = vec![0_u32; bvh.items.len()];
    for (n, node) in bvh.nodes.iter().enumerate() {
        let union = match node.kind() {
            NodeKind::Inner { left, right } => {
                assert!(left > n, "node {n}'s children come after it");
                assert_eq!(right, left + 1, "node {n}'s children are a pair");
                for child in [left, right] {
                    assert_eq!(bvh.parent(child), Some(n), "{child}'s parent is {n}");
                }
                bvh.nodes[left].aabb.union(bvh.nodes[right].aabb)
            }
            NodeKind::Leaf { first, end } => {
                assert!(first < end, "leaf {n} isn't empty");
                assert!(end - first <= LEAF_SIZE, "leaf {n} is small");
                for (k, count) in covered.iter_mut().enumerate().take(end).skip(first) {
                    *count += 1;
                    assert_eq!(bvh.leaf_of[k] as usize, n, "item {k} is in leaf {n}");
                }
                bvh.items[first..end]
                    .iter()
                    .map(|(_, b)| b.aabb)
                    .reduce(Aabb::union)
                    .unwrap()
            }
        };
        assert_eq!(node.aabb, union, "node {n}'s box is exactly its contents'");
    }
    assert!(
        covered.iter().all(|&c| c == 1),
        "every item in exactly one leaf"
    );
    for (k, (key, _)) in bvh.items.iter().enumerate() {
        assert_eq!(
            bvh.index_of.get(key),
            Some(k),
            "{key:?} is found where it is"
        );
    }
}

/// Runs `queries` random sphere queries and `rays` random rays on `bvh` and checks each against
/// brute force over `scene`. Returns how many rays hit something.
fn compare(
    bvh: &Bvh,
    scene: &[(TileKey, TileBounds)],
    rng: &mut Rng,
    queries: usize,
    rays: usize,
) -> usize {
    for _ in 0..queries {
        let center = rng.point(-0.2, 1.2);
        let radius = rng.range(0.0, 0.25);
        assert_eq!(
            tree_sphere(bvh, center, radius),
            brute_sphere(scene, center, radius),
            "sphere at {center:?}, radius {radius}"
        );
    }
    let mut hits = 0;
    for k in 0..rays {
        let ray = random_ray(rng, k);
        let (tree, _) = tree_ray(bvh, scene, &ray);
        let brute = brute_ray(scene, &ray);
        assert_eq!(tree, brute, "ray {k}: {ray:?}");
        hits += usize::from(brute.is_some());
    }
    hits
}

#[test]
fn queries_match_brute_force_on_random_scenes() {
    let mut rng = Rng(1);
    for (n, size) in [(1, 0.05), (5, 0.05), (300, 0.08), (4000, 0.02)] {
        let scene = random_scene(&mut rng, n, size);
        let bvh = Bvh::build(scene.clone()).unwrap();
        check(&bvh);
        assert_eq!(bvh.len(), n, "every tile is held");
        let hits = compare(&bvh, &scene, &mut rng, 2000, 2000);
        if n >= 300 {
            assert!(
                hits > 500,
                "{n} tiles: enough rays hit to mean something ({hits})"
            );
        }
    }
}

#[test]
fn refit_after_moving_random_tiles_still_matches_brute_force() {
    let mut rng = Rng(2);
    let mut scene = random_scene(&mut rng, 3000, 0.02);
    let mut bvh = Bvh::build(scene.clone()).unwrap();
    for round in 0..3 {
        // Move a tenth of the tiles, many of them far outside the tree's old box, and grow some.
        let changed: Vec<(TileKey, TileBounds)> = (0..300)
            .map(|_| {
                let k = rng.below(scene.len());
                let moved = random_bounds(&mut rng, -1.0, 2.0, 0.1);
                scene[k].1 = moved;
                (scene[k].0, moved)
            })
            .collect();
        bvh.refit(&changed).unwrap();
        check(&bvh);
        for (key, b) in &scene {
            assert_eq!(
                bvh.bounds(key),
                Some(b),
                "round {round}: {key:?} has its new bounds"
            );
        }
        compare(&bvh, &scene, &mut rng, 2000, 2000);
    }
}

#[test]
fn refit_of_an_unknown_tile_changes_nothing() {
    let mut rng = Rng(3);
    let scene = random_scene(&mut rng, 50, 0.05);
    let mut bvh = Bvh::build(scene.clone()).unwrap();
    let moved = bounds(Vec3::new(5.0, 5.0, 5.0), Vec3::new(0.1, 0.1, 0.1));
    let err = bvh
        .refit(&[(scene[0].0, moved), (key(999), moved)])
        .unwrap_err();
    assert!(
        matches!(err, IndexError::UnknownTile(k) if k == key(999)),
        "names the unknown tile: {err:?}"
    );
    assert_eq!(
        bvh.bounds(&scene[0].0),
        Some(&scene[0].1),
        "the known tile kept its bounds"
    );
    check(&bvh);
}

#[test]
fn a_duplicate_tile_is_refused() {
    let b = bounds(Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0));
    let err = Bvh::build(vec![(key(0), b), (key(1), b), (key(0), b)]).unwrap_err();
    assert!(
        matches!(err, IndexError::DuplicateTile(k) if k == key(0)),
        "names the duplicate: {err:?}"
    );
}

#[test]
fn a_tile_of_a_huge_face_is_refused_not_allocated_for() {
    let b = bounds(Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0));
    let level = Level::new(9).unwrap();
    // 64 tiles a face at level 9: face 2^24 is the first whose faces up to it pass MAX_TILES.
    for face in [u32::MAX, 1 << 24] {
        let far = TileKey::displacement(FaceId::new(face), level, 0, 0).unwrap();
        let err = Bvh::build(vec![(key(0), b), (far, b)]).unwrap_err();
        assert!(
            matches!(err, IndexError::TooManyTiles(n) if n > MAX_TILES),
            "face {face}: every tile of the faces up to it is more than a tree holds: {err:?}"
        );
    }
}

#[test]
fn tiles_of_another_level_layer_or_channel_are_refused() {
    let b = bounds(Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0));
    let other_level = TileKey::displacement(FaceId::new(0), Level::new(8).unwrap(), 0, 0).unwrap();
    let other_layer = TileKey::new(
        geom::LayerId::new(1),
        geom::Channel::DISPLACEMENT,
        FaceId::new(0),
        Level::new(9).unwrap(),
        0,
        0,
    )
    .unwrap();
    for odd in [other_level, other_layer] {
        let err = Bvh::build(vec![(key(0), b), (key(1), b), (odd, b)]).unwrap_err();
        assert!(
            matches!(err, IndexError::MixedTiles(k) if k == odd),
            "names the odd one out: {err:?}"
        );
    }
    let bvh = Bvh::build(vec![(key(0), b), (key(1), b)]).unwrap();
    assert!(
        bvh.bounds(&other_level).is_none() && bvh.bounds(&other_layer).is_none(),
        "and a lookup of one finds nothing"
    );
    assert!(
        bvh.bounds(&key(64 * 5)).is_none(),
        "nor of a tile past the last face"
    );
}

/// The benchmark sphere's level-9 tile count: 2,400 faces of 8 × 8 tiles.
const SPHERE_TILES: usize = 153_600;

#[test]
fn the_tree_takes_at_most_100_bytes_a_tile() {
    let mut rng = Rng(9);
    let scene = random_scene(&mut rng, SPHERE_TILES, 0.002);
    let bvh = Bvh::build(scene).unwrap();
    let tiles = f64::from(u32::try_from(SPHERE_TILES).unwrap());
    let per_tile = f64::from(u32::try_from(bvh.heap_bytes()).unwrap()) / tiles;
    println!(
        "{SPHERE_TILES} tiles: {} bytes, {per_tile:.1} a tile ({} nodes of {} bytes)",
        bvh.heap_bytes(),
        bvh.nodes.len(),
        size_of::<Node>()
    );
    assert_eq!(size_of::<Node>(), 32, "a node is its box and two u32s");
    assert!(per_tile <= 100.0, "{per_tile:.1} bytes a tile");
}

#[test]
fn an_empty_tree_answers_nothing() {
    let bvh = Bvh::build(Vec::new()).unwrap();
    check(&bvh);
    assert!(bvh.is_empty(), "no tiles");
    assert_eq!(bvh.aabb(), None, "no box");
    assert_eq!(
        bvh.query_sphere(Vec3::ZERO, 10.0).count(),
        0,
        "no tiles touched"
    );
    let ray = Ray::new(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0)).unwrap();
    let hit = bvh
        .raycast(&ray, |_| -> Result<Option<f32>, Infallible> {
            panic!("no tile to test")
        })
        .unwrap();
    assert_eq!(hit, None, "no hit");
}

#[test]
fn coincident_tiles_are_halved_into_small_leaves() {
    // Every centroid equal: nothing to bin, so the build halves the tiles as they lie.
    let scene: Vec<_> = (0..1000)
        .map(|i| {
            (
                key(i),
                bounds(Vec3::new(0.5, 0.5, 0.5), Vec3::new(0.1, 0.1, 0.1)),
            )
        })
        .collect();
    let bvh = Bvh::build(scene.clone()).unwrap();
    check(&bvh);
    let depth = (0..bvh.nodes.len())
        .map(|mut n| {
            let mut depth = 0;
            while let Some(p) = bvh.parent(n) {
                n = p;
                depth += 1;
            }
            depth
        })
        .max()
        .unwrap();
    assert!(depth <= 10, "balanced: depth {depth} for 1000 tiles");
    assert_eq!(
        tree_sphere(&bvh, Vec3::new(0.65, 0.5, 0.5), 0.1),
        brute_sphere(&scene, Vec3::new(0.65, 0.5, 0.5), 0.1),
        "all of them touched"
    );
    assert_eq!(
        bvh.query_sphere(Vec3::new(2.0, 2.0, 2.0), 0.1).count(),
        0,
        "none far away"
    );
}

#[test]
fn a_negative_or_nan_radius_touches_nothing() {
    let mut rng = Rng(4);
    let bvh = Bvh::build(random_scene(&mut rng, 100, 0.1)).unwrap();
    assert_eq!(
        bvh.query_sphere(Vec3::new(0.5, 0.5, 0.5), -1.0).count(),
        0,
        "negative"
    );
    assert_eq!(
        bvh.query_sphere(Vec3::new(0.5, 0.5, 0.5), f32::NAN).count(),
        0,
        "NaN"
    );
    assert_eq!(
        bvh.query_sphere(Vec3::new(0.5, 0.5, 0.5), f32::INFINITY)
            .count(),
        100,
        "an infinite radius touches everything"
    );
}

#[test]
fn raycast_stops_testing_once_nothing_can_be_nearer() {
    // A row of 100 unit cubes along x, the ray coming down the row: the first is hit, and the
    // second's box starts past that hit, so nothing behind is tested.
    let scene: Vec<_> = (0..100_u16)
        .map(|i| {
            let x = 3.0 * f32::from(i);
            (
                key(usize::from(i)),
                bounds(Vec3::new(x, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0)),
            )
        })
        .collect();
    let bvh = Bvh::build(scene.clone()).unwrap();
    let ray = Ray::new(Vec3::new(-10.0, 0.01, 0.0), Vec3::new(1.0, 0.0, 0.0)).unwrap();
    let (hit, tests) = tree_ray(&bvh, &scene, &ray);
    assert_eq!(hit.map(|h| h.key), Some(key(0)), "the first cube");
    assert!((hit.unwrap().t - 9.0).abs() < 1e-3, "at its near face");
    assert_eq!(tests, 1, "only the first tile is tested");
}

#[test]
fn an_error_from_the_precise_test_stops_the_search() {
    let mut rng = Rng(5);
    let bvh = Bvh::build(random_scene(&mut rng, 100, 0.2)).unwrap();
    let ray = Ray::new(Vec3::new(-1.0, 0.5, 0.5), Vec3::new(1.0, 0.0, 0.0)).unwrap();
    let mut calls = 0;
    let result = bvh.raycast(&ray, |_| {
        calls += 1;
        Err::<Option<f32>, _>("evaluation failed")
    });
    assert_eq!(result, Err("evaluation failed"), "the error comes back");
    assert_eq!(calls, 1, "after the first call");
}

/// How many nodes a sphere query visits: those whose box touches the sphere, below a parent
/// whose box does too.
fn visited(bvh: &Bvh, center: Vec3, radius: f32) -> usize {
    let mut stack = vec![0];
    let mut count = 0;
    while let Some(n) = stack.pop() {
        let node = &bvh.nodes[n];
        if node.aabb.distance_squared(center) > radius * radius {
            continue;
        }
        count += 1;
        if let NodeKind::Inner { left, right } = node.kind() {
            stack.extend([left, right]);
        }
    }
    count
}

#[test]
fn the_build_splits_where_the_gap_is() {
    // Two slabs, long in x, separated in y: the cheap split is the gap in y, not the middle
    // of the long axis.
    let mut rng = Rng(6);
    let slab = |r: &mut Rng, y: f32| {
        bounds(
            Vec3::new(r.range(0.0, 10.0), y + r.range(0.0, 1.0), r.range(0.0, 1.0)),
            Vec3::new(0.05, 0.05, 0.05),
        )
    };
    let scene: Vec<_> = (0..1000)
        .map(|i| (key(i), slab(&mut rng, if i % 2 == 0 { 0.0 } else { 3.0 })))
        .collect();
    let bvh = Bvh::build(scene).unwrap();
    check(&bvh);
    let NodeKind::Inner { left, right } = bvh.nodes[0].kind() else {
        panic!("the root is split");
    };
    let (a, b) = (bvh.nodes[left].aabb, bvh.nodes[right].aabb);
    assert!(
        a.max().y < b.min().y || b.max().y < a.min().y,
        "the root's children are the two slabs: {a:?}, {b:?}"
    );
}

/// The SAH cost of splitting `entries` into `low` and the rest: each side's count times its
/// box's surface area.
fn split_cost(entries: &[Entry], low: impl Fn(&Entry) -> bool) -> f32 {
    let side = |want: bool| {
        let boxes: Vec<Aabb> = entries
            .iter()
            .filter(|e| low(e) == want)
            .map(|e| e.bounds.aabb)
            .collect();
        count_f32(boxes.len())
            * boxes
                .into_iter()
                .reduce(Aabb::union)
                .map_or(0.0, |b| b.surface_area())
    };
    side(true) + side(false)
}

#[test]
fn split_takes_the_cheapest_binned_plane() {
    // The oracle: every plane between the 16 equal bins of the centroids along each axis,
    // costed from the partition it makes. Boxes of very different sizes, so a split that bins
    // anything other than the centroids picks a plane that costs more.
    let mut rng = Rng(8);
    for trial in 0..300 {
        let n = 5 + rng.below(60);
        let mut entries: Vec<Entry> = (0..n)
            .map(|i| {
                let bounds = random_bounds(&mut rng, 0.0, 1.0, 0.4);
                Entry {
                    key: key(i),
                    bounds,
                    centroid: bounds.aabb.centroid(),
                }
            })
            .collect();
        let centroids = Aabb::from_points(entries.iter().map(|e| e.centroid)).unwrap();
        let cheapest = Axis::ALL
            .iter()
            .flat_map(|&axis| {
                let low = axis.of(centroids.min());
                let scale = BINS_F32 / (axis.of(centroids.max()) - low);
                let entries = &entries;
                (0..BINS - 1).filter_map(move |plane| {
                    let below = |e: &Entry| bin_index(axis.of(e.centroid), low, scale) <= plane;
                    let count = entries.iter().filter(|e| below(e)).count();
                    (count > 0 && count < entries.len()).then(|| split_cost(entries, below))
                })
            })
            .fold(f32::INFINITY, f32::min);
        assert!(
            cheapest.is_finite(),
            "trial {trial}: some plane splits the tiles"
        );
        let mid = split(&mut entries).unwrap();
        let keys: Vec<TileKey> = entries[..mid].iter().map(|e| e.key).collect();
        let chosen = split_cost(&entries, |e| keys.contains(&e.key));
        assert!(
            (chosen - cheapest).abs() <= 1e-5 * cheapest,
            "trial {trial} ({n} tiles): the split costs {chosen}, the cheapest plane {cheapest}"
        );
    }
}

#[test]
fn queries_visit_few_nodes() {
    // 4096 small tiles in the unit cube; a small sphere touches a handful. A tree whose splits
    // ignore the tiles' layout makes the queries wander through most of it. The build gives a
    // mean of 13 visits; the bound is twice that.
    let mut rng = Rng(7);
    let scene = random_scene(&mut rng, 4096, 0.01);
    let bvh = Bvh::build(scene).unwrap();
    let queries = 1000;
    let total: usize = (0..queries)
        .map(|_| visited(&bvh, rng.point(0.0, 1.0), 0.02))
        .sum();
    let mean = total / queries;
    assert!(
        mean <= 26,
        "a query visits {mean} of {} nodes on average",
        bvh.nodes.len()
    );
}

#[test]
fn aabb_rejects_bad_corners_and_skips_bad_points() {
    let (a, b) = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 2.0, 3.0));
    assert!(Aabb::new(a, b).is_some(), "ordered corners");
    assert!(Aabb::new(a, a).is_some(), "a single point");
    assert!(Aabb::new(b, a).is_none(), "inverted corners");
    assert!(
        Aabb::new(a, Vec3::new(f32::NAN, 1.0, 1.0)).is_none(),
        "NaN corner"
    );
    assert!(
        Aabb::new(a, Vec3::new(f32::INFINITY, 1.0, 1.0)).is_none(),
        "infinite corner"
    );
    assert_eq!(Aabb::from_points([]), None, "no points");
    assert_eq!(
        Aabb::from_points([Vec3::new(f32::NAN, 0.0, 0.0)]),
        None,
        "no finite points"
    );
    let boxed = Aabb::from_points([b, Vec3::new(f32::NAN, 9.0, 9.0), a]).unwrap();
    assert_eq!(
        (boxed.min(), boxed.max()),
        (a, b),
        "the NaN point is skipped"
    );
    assert!(boxed.contains(Vec3::new(1.0, 1.0, 1.0)), "inside");
    assert!(!boxed.contains(Vec3::new(1.5, 1.0, 1.0)), "outside");
    assert!((boxed.surface_area() - 22.0).abs() < 1e-6, "2 (2 + 6 + 3)");
    assert!(
        boxed.distance_squared(Vec3::new(0.5, 0.5, 0.5)).abs() < 1e-12,
        "0 inside"
    );
    assert!(
        (boxed.distance_squared(Vec3::new(4.0, 6.0, 3.0)) - 25.0).abs() < 1e-5,
        "3² + 4² to the corner"
    );
}

#[test]
fn ray_entry_handles_inside_parallel_and_behind() {
    let boxed = Aabb::new(Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0)).unwrap();
    let entry = |origin: Vec3, direction: Vec3| {
        let ray = Ray::new(origin, direction).unwrap();
        boxed.entry(&ray)
    };
    let x = Vec3::new(1.0, 0.0, 0.0);
    assert_eq!(entry(Vec3::new(0.5, 0.5, 0.5), x), Some(0.0), "from inside");
    assert_eq!(
        entry(Vec3::new(-2.0, 0.5, 0.5), x),
        Some(2.0),
        "from outside"
    );
    assert_eq!(
        entry(Vec3::new(-2.0, 1.5, 0.5), x),
        None,
        "parallel, outside the slab"
    );
    assert_eq!(
        entry(Vec3::new(2.0, 0.5, 0.5), x),
        None,
        "behind the origin"
    );
    assert_eq!(
        entry(Vec3::new(2.0, 0.5, 0.5), -x),
        Some(1.0),
        "turned around"
    );
    assert!(
        Ray::new(Vec3::ZERO, Vec3::ZERO).is_none(),
        "a ray needs a direction"
    );
    assert!(
        Ray::new(Vec3::new(f32::NAN, 0.0, 0.0), x).is_none(),
        "and a finite origin"
    );
}

#[test]
fn rays_along_a_face_an_edge_or_a_seam_enter_the_box() {
    let boxed = Aabb::new(Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0)).unwrap();
    let entry = |aabb: &Aabb, origin: Vec3, direction: Vec3| {
        aabb.entry(&Ray::new(origin, direction).unwrap())
    };
    let (x, y) = (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
    // Along each face: parallel to the face, the origin on its plane.
    for (origin, direction, what) in [
        (Vec3::new(-2.0, 0.0, 0.5), x, "on the low y face"),
        (Vec3::new(-2.0, 1.0, 0.5), x, "on the high y face"),
        (Vec3::new(-2.0, 0.5, 0.0), x, "on the low z face"),
        (Vec3::new(-2.0, 0.5, 1.0), x, "on the high z face"),
        (Vec3::new(0.0, -3.0, 0.5), y, "on the low x face"),
        (Vec3::new(1.0, -3.0, 0.5), y, "on the high x face"),
    ] {
        let expected = if direction == x { 2.0 } else { 3.0 };
        assert_eq!(entry(&boxed, origin, direction), Some(expected), "{what}");
    }
    // Along the edges, where two of the axes are parallel and on a plane.
    for (origin, what) in [
        (Vec3::new(-2.0, 0.0, 0.0), "the low-low edge"),
        (Vec3::new(-2.0, 1.0, 1.0), "the high-high edge"),
        (Vec3::new(-2.0, 0.0, 1.0), "a mixed edge"),
    ] {
        assert_eq!(entry(&boxed, origin, x), Some(2.0), "along {what}");
    }
    // From a corner, along an edge, and from a point on a face, across the box.
    assert_eq!(
        entry(&boxed, Vec3::ZERO, x),
        Some(0.0),
        "from the corner along an edge"
    );
    assert_eq!(
        entry(&boxed, Vec3::new(0.5, 0.0, 0.5), Vec3::new(1.0, 0.0, 1.0)),
        Some(0.0),
        "diagonally within a face"
    );
    // Just off a face, parallel: a miss.
    let off = 1.0 + f32::EPSILON;
    assert_eq!(
        entry(&boxed, Vec3::new(-2.0, off, 0.5), x),
        None,
        "just outside the high y face"
    );
    assert_eq!(
        entry(&boxed, Vec3::new(-2.0, -1e-30, 0.5), x),
        None,
        "just outside the low y face"
    );
    // A seam: two boxes sharing the plane x = 0, and rays within that plane, which enter both.
    let left = Aabb::new(Vec3::new(-1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 1.0)).unwrap();
    let right = Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0)).unwrap();
    for (origin, direction) in [
        (Vec3::new(0.0, -2.0, 0.5), y),
        (Vec3::new(0.0, -2.0, -2.0), Vec3::new(0.0, 1.0, 1.0)),
        (Vec3::new(0.0, 3.0, 0.25), Vec3::new(0.0, -1.0, 0.1)),
    ] {
        let (a, b) = (
            entry(&left, origin, direction),
            entry(&right, origin, direction),
        );
        assert!(
            a.is_some(),
            "the seam ray from {origin:?} enters the left box"
        );
        assert_eq!(a, b, "and the right one at the same place");
    }
    // A tiny but nonzero direction component, on the plane: no NaN either.
    assert_eq!(
        entry(
            &boxed,
            Vec3::new(-2.0, 0.0, 0.5),
            Vec3::new(1.0, 1e-40, 0.0)
        ),
        Some(2.0),
        "a subnormal component along the low y face"
    );
}
