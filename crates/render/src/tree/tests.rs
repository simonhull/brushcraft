//! The tile tree: places and their links, and its GPU copy kept up to date incrementally.

use geom::{BaseMesh, FaceId, Level, TileKey, cube_sphere};

use super::*;
use crate::gpu::GpuTest;
use crate::residency::{SizeClass, Slot};

/// Every place of `mesh` at `level`.
fn places(mesh: &BaseMesh, level: Level) -> Vec<Place> {
    let per_side = u16::try_from(level.tiles_per_side()).expect("small levels");
    let all = mesh.faces().flat_map(|face| {
        (0..per_side).flat_map(move |ty| {
            (0..per_side).map(move |tx| Place {
                face,
                level,
                tx,
                ty,
            })
        })
    });
    all.collect()
}

/// Level `l`.
fn level(l: u8) -> Level {
    Level::new(l).expect("a level")
}

/// A tile's bound `error` and a cone told apart by `tag` (any bits are a cone).
fn detail(error: f32, tag: usize) -> TileDetail {
    let bits = u32::try_from(tag).expect("small").wrapping_mul(0x9E37_79B9);
    TileDetail {
        stored: None,
        error,
        cone: geom::FinestNormalCone::from_bits(bits),
        held: false,
    }
}

/// A small deterministic generator for the tests' random choices.
struct Rng(u64);

impl Rng {
    /// The next 64 random bits (splitmix64).
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A random index below `n` (`n > 0`).
    fn below(&mut self, n: usize) -> usize {
        usize::try_from(self.next() % n as u64).expect("below n")
    }
}

#[test]
fn every_child_names_its_parent_and_whole_face_levels_have_one_child() {
    let mesh = cube_sphere(2).expect("a mesh");
    for l in 0..8 {
        for place in places(&mesh, level(l)) {
            let children: Vec<Place> = place.children().into_iter().flatten().collect();
            let expected = if level(l + 1).tiles_per_side() == level(l).tiles_per_side() {
                1
            } else {
                4
            };
            assert_eq!(children.len(), expected, "children of {place:?}");
            for child in children {
                assert_eq!(child.parent(), Some(place), "{child:?}'s parent");
            }
        }
    }
    assert_eq!(
        places(&mesh, level(0)).first().and_then(|p| p.parent()),
        None,
        "level 0 has no parent"
    );
}

#[test]
fn neighbours_are_symmetric_and_every_edge_of_a_closed_mesh_has_one() {
    let mesh = cube_sphere(2).expect("a mesh");
    let topology = mesh.topology();
    for l in [0, 1, 3, 6, 7, 8] {
        for place in places(&mesh, level(l)) {
            let neighbours = place.neighbours(topology);
            for (edge, neighbour) in neighbours.iter().enumerate() {
                let neighbour = neighbour.unwrap_or_else(|| panic!("{place:?} edge {edge}"));
                assert_ne!(neighbour, place, "{place:?} is not its own neighbour");
                assert_eq!(neighbour.level, place.level, "same level");
                assert!(
                    neighbour.neighbours(topology).contains(&Some(place)),
                    "{neighbour:?} names {place:?} back"
                );
            }
        }
    }
}

#[test]
fn neighbours_inside_a_face_are_the_next_tiles_along_each_axis() {
    let mesh = cube_sphere(1).expect("a mesh");
    let place = Place {
        face: FaceId::new(0),
        level: level(9),
        tx: 3,
        ty: 4,
    };
    let at = |tx, ty| Some(Place { tx, ty, ..place });
    assert_eq!(
        place.neighbours(mesh.topology()),
        [at(3, 3), at(4, 4), at(3, 5), at(2, 4)],
        "j = 0, i = side - 1, j = side - 1, i = 0"
    );
}

#[test]
fn places_round_trip_through_their_packed_form_and_keys() {
    let mesh = cube_sphere(1).expect("a mesh");
    let finest = level(8);
    for l in 0..=8 {
        for place in places(&mesh, level(l)) {
            assert_eq!(
                Place::from_packed(place.face.index(), place.packed()),
                Some(place),
                "packed"
            );
            let key = place.key(finest).expect("at or above the finest");
            assert_eq!(Place::of(&key), place, "through its key");
            assert_eq!(
                key.channel() == geom::Channel::DISPLAY,
                place.level < finest,
                "display tiles above the finest level"
            );
        }
    }
    let past = Place::of(&TileKey::displacement(FaceId::new(0), level(9), 0, 0).expect("key"));
    assert_eq!(past.key(finest), None, "nothing below the finest level");
    let off_grid = (8 << 28) | (300 << 14);
    assert_eq!(
        Place::from_packed(0, off_grid),
        None,
        "a column off the face"
    );
}

#[test]
fn the_gpu_copy_kept_from_dirty_nodes_matches_the_tree_after_any_arrivals_and_departures() {
    GpuTest::run(|gpu| {
        let mesh = cube_sphere(1).expect("a mesh");
        let all: Vec<Place> = (0..=8).flat_map(|l| places(&mesh, level(l))).collect();
        let capacity = 64;
        let mut tree = TileTree::new(gpu, &mesh, capacity).expect("a tree");
        // The GPU's copy, as the dirty runs alone would leave it (buffers start zeroed).
        let mut mirror = vec![GpuNode::zeroed(); usize::try_from(capacity).expect("small")];
        let mut rng = Rng(7);
        let mut resident: Vec<Place> = Vec::new();
        for step in 0..2000 {
            let arrive = resident.len() < 8 || (resident.len() < 60 && rng.below(2) == 0);
            if arrive {
                let place = all[rng.below(all.len())];
                let slot = Slot::for_tests(SizeClass::of(place.level), rng.below(100));
                match tree.insert(place, slot, detail(0.5, step), 0) {
                    Ok(_) => resident.push(place),
                    Err(TreeError::Present(_)) => {}
                    Err(error) => panic!("step {step}: {error}"),
                }
            } else {
                let place = resident.swap_remove(rng.below(resident.len()));
                assert!(tree.remove(place).is_some(), "step {step}: was resident");
            }
            if rng.below(4) == 0
                && let Some(&place) = resident.first()
            {
                let moved = detail(2.0, step + 1);
                tree.move_to(place, Slot::for_tests(SizeClass::of(place.level), 5), moved);
            }
            for (start, nodes) in tree.take_dirty() {
                for (offset, node) in nodes.into_iter().enumerate() {
                    mirror[usize::try_from(start).expect("small") + offset] = node;
                }
            }
            for index in 0..capacity {
                assert_eq!(
                    mirror[usize::try_from(index).expect("small")],
                    tree.gpu_node(index),
                    "step {step}: node {index} is stale on the GPU"
                );
            }
            check_links(&tree, &resident);
        }
    });
}

/// Every resident place's node links to exactly the resident parent, children and neighbours
/// its place names.
fn check_links(tree: &TileTree, resident: &[Place]) {
    assert_eq!(tree.len(), resident.len(), "one node per resident place");
    for &place in resident {
        let index = tree.node(place).expect("resident");
        let node = tree.gpu_node(index);
        assert_eq!(node.flags & NODE_LIVE, NODE_LIVE, "live");
        assert_eq!(
            Place::from_packed(node.face, node.place),
            Some(place),
            "its place"
        );
        let link = |p: Option<Place>| p.and_then(|p| tree.node(p)).unwrap_or(NONE);
        assert_eq!(node.parent, link(place.parent()), "parent of {place:?}");
        assert_eq!(
            node.children,
            place.children().map(link),
            "children of {place:?}"
        );
        assert_eq!(
            node.neighbours,
            place.neighbours(&tree.topology).map(link),
            "neighbours of {place:?}"
        );
        // A base mesh is closed: every edge has a neighbour, resident or not.
        assert_eq!(node.edges, 0b1111, "edges of {place:?}");
    }
}

#[test]
fn arrivals_and_moves_to_new_data_are_stamped_and_departures_forgotten() {
    GpuTest::run(|gpu| {
        let mesh = cube_sphere(1).expect("a mesh");
        let mut tree = TileTree::new(gpu, &mesh, 8).expect("a tree");
        let slot = |place: Place| Slot::for_tests(SizeClass::of(place.level), 0);
        let [a, b, c] = [0, 1, 2].map(|face| Place {
            face: FaceId::new(face),
            level: level(7),
            tx: 1,
            ty: 0,
        });
        let start = tree.generation();
        let first = tree
            .insert(a, slot(a), detail(1.0, 0), 0)
            .expect("inserted");
        let second = tree
            .insert(b, slot(b), detail(1.0, 0), 0)
            .expect("inserted");
        let all: Vec<u32> = tree.changed_since(start).collect();
        assert_eq!(all, vec![first, second], "both arrivals, in node order");
        let after = tree.generation();
        assert_eq!(tree.changed_since(after).count(), 0, "nothing since");
        let refreshed = detail(0.5, 3);
        tree.move_to(a, slot(a), refreshed);
        let moved: Vec<u32> = tree.changed_since(after).collect();
        assert_eq!(moved, vec![first], "the move to new data");
        let node = tree.gpu_node(first);
        assert_eq!(
            (node.error, node.finest_cone),
            (refreshed.error, refreshed.cone.to_bits()),
            "the new data's e and cone, together"
        );
        assert!(tree.remove(b).is_some(), "resident");
        let third = tree
            .insert(c, slot(c), detail(1.0, 0), 0)
            .expect("inserted");
        assert_eq!(third, second, "the node is reused");
        let since: Vec<u32> = tree.changed_since(after).collect();
        assert_eq!(
            since,
            vec![first, third],
            "the reused node is stamped afresh"
        );
        assert!(tree.remove(c).is_some(), "resident");
        let left: Vec<u32> = tree.changed_since(0).collect();
        assert_eq!(left, vec![first], "a node that left isn't listed");
    });
}
