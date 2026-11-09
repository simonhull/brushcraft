//! Edits: whole tiles and runs of tiles in parallel, border twins, failures part-way, and the
//! ops edits carry.

use super::*;

#[test]
fn editing_a_tile_equals_setting_its_vertices() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut one_by_one = new_store(&dir, "one.bcp", &mesh, roomy());
    let mut batched = new_store(&dir, "batched.bcp", &mesh, roomy());
    let mut map = CorrectionMap::new();
    for (key, edits) in tile_batches(&mesh, 60, 25, 18) {
        let region = Region::of_tile(&key);
        let op = one_by_one.begin_op().unwrap();
        assert_eq!(
            batched.begin_op().unwrap(),
            op,
            "the stores hand out the same ops"
        );
        for &(i, j, c) in &edits {
            let (fi, fj) = (region.x0() + i, region.y0() + j);
            one_by_one
                .set_correction(key.face(), key.level(), fi, fj, c)
                .unwrap();
            map.set(mesh.topology(), key.face(), key.level(), fi, fj, c)
                .unwrap();
        }
        let touched = batched
            .edit_tile(key, |edit| {
                for &(i, j, c) in &edits {
                    edit.set(i, j, c).unwrap();
                }
                assert_eq!(
                    edit.get(0, 0).unwrap(),
                    {
                        let (i, j) = (region.x0(), region.y0());
                        map.get(key.face(), key.level(), i, j)
                    },
                    "reads its own writes"
                );
                edit.touched.len()
            })
            .unwrap();
        assert!(touched >= 1, "{key:?}");
        batched.deps.check_invariants().unwrap();
    }
    let edited: BTreeSet<TileKey> = one_by_one.unsaved.keys().copied().collect();
    assert_eq!(
        edited,
        batched.unsaved.keys().copied().collect(),
        "the same tiles are dirty"
    );
    let feed = |store: &mut TileStore| -> BTreeSet<(TileKey, OpId)> {
        store.feed.drain(store.save_consumer).into_iter().collect()
    };
    assert_eq!(feed(&mut one_by_one), feed(&mut batched), "and on the feed");
    for key in &edited {
        one_by_one.correction(*key).unwrap();
        batched.correction(*key).unwrap();
        assert_eq!(
            slot_resident(&one_by_one, key),
            slot_resident(&batched, key),
            "{key:?}: the same bits"
        );
    }
    let dense = dense(&mesh, &map);
    assert_positions(&mut batched, &dense, &tiles(&mesh, 1, TOP), "batched");
}

/// Applies `change` to every vertex of `key`'s tile in row-major order, through `update` or
/// through `get` and `set` vertex by vertex.
///
/// # Errors
///
/// The edit's or the first write's.
fn change_tile(
    store: &mut TileStore,
    key: TileKey,
    through_update: bool,
    change: &dyn Fn(u32, u32, Vec3) -> Option<Vec3>,
) -> Result<(), CacheError> {
    store
        .edit_tile(key, |edit| {
            if through_update {
                return edit.update(change);
            }
            let side = key.side();
            for (i, j) in (0..side).flat_map(|j| (0..side).map(move |i| (i, j))) {
                if let Some(c) = change(i, j, edit.get(i, j)?) {
                    edit.set(i, j, c)?;
                }
            }
            Ok(())
        })
        .and_then(std::convert::identity)
}

#[test]
fn updating_a_tile_equals_getting_and_setting_each_vertex() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut by_vertex = new_store(&dir, "by_vertex.bcp", &mesh, roomy());
    let mut updated = new_store(&dir, "updated.bcp", &mesh, roomy());
    let all = tiles(&mesh, 1, TOP);
    let mut rng = Rng(0x000B_D7E5);
    for round in 0..24 {
        let key = all[rng.below(u32::try_from(all.len()).unwrap()) as usize];
        let side = key.side();
        let amplitude = 0.5 / f32::from(u16::try_from(key.level().face_quads()).unwrap());
        // About half the vertices change; now and then by a lot, so the scale grows (and the
        // interior is re-quantized) part-way through the tile.
        let deltas: Vec<Option<Vec3>> = (0..side * side)
            .map(|_| match rng.below(40) {
                0..20 => None,
                20 => Some(rng.vec3(amplitude * 40.0)),
                _ => Some(rng.vec3(amplitude)),
            })
            .collect();
        // Every few rounds, a value that isn't finite part-way through a row inside the tile.
        let poisoned = (round % 4 == 3).then_some((side / 2, side / 2));
        let pattern = Pattern::of(round);
        let change = |i: u32, j: u32, c: Vec3| {
            if poisoned == Some((i, j)) {
                return Some(Vec3::new(f32::NAN, 0.0, 0.0));
            }
            if !pattern.allows(side, i, j) {
                return None;
            }
            let d = deltas[(j * side + i) as usize]?;
            Some(Vec3::new(c.x + d.x, c.y + d.y, c.z + d.z))
        };
        let op = by_vertex.begin_op().unwrap();
        assert_eq!(updated.begin_op().unwrap(), op, "the same ops");
        let expected = change_tile(&mut by_vertex, key, false, &change);
        let got = change_tile(&mut updated, key, true, &change);
        assert_eq!(
            format!("{got:?}"),
            format!("{expected:?}"),
            "round {round}: the same outcome"
        );
        if poisoned.is_some() {
            assert!(
                matches!(got, Err(CacheError::Correction(CorrectionError::NotFinite))),
                "round {round}: {got:?}"
            );
        }
    }
    assert_eq!(
        updated.unsaved, by_vertex.unsaved,
        "the same tiles dirty, with the same ops and stamps (written in the same order)"
    );
    let drain = |store: &mut TileStore| store.feed.drain(store.save_consumer);
    assert_eq!(drain(&mut updated), drain(&mut by_vertex), "the same feed");
    let edited: BTreeSet<TileKey> = by_vertex.unsaved.keys().copied().collect();
    for key in &edited {
        assert_eq!(
            slot_resident(&updated, key),
            slot_resident(&by_vertex, key),
            "{key:?}: the same bits"
        );
    }
}

#[test]
fn border_values_are_found_where_border_order_put_them() {
    for side in [2, 3, 4, 65] {
        let order: Vec<(u32, u32)> = border_order(side).collect();
        let last = side - 1;
        let border: Vec<(u32, u32)> = (0..side)
            .flat_map(|j| (0..side).map(move |i| (i, j)))
            .filter(|&(i, j)| i == 0 || i == last || j == 0 || j == last)
            .collect();
        assert_eq!(
            order, border,
            "side {side}: every border vertex once, row-major"
        );
        let values = BorderValues {
            side,
            values: (0_u16..)
                .zip(&order)
                .map(|(n, _)| Vec3::new(f32::from(n), 0.0, 0.0))
                .collect(),
        };
        for (n, &(i, j)) in (0_u16..).zip(&order) {
            assert_eq!(
                values.get(i, j),
                Some(Vec3::new(f32::from(n), 0.0, 0.0)),
                "side {side}: ({i}, {j})"
            );
        }
        for (i, j) in [(1, 1), (side, 0), (0, side)] {
            if !order.contains(&(i, j)) {
                assert_eq!(
                    values.get(i, j),
                    None,
                    "side {side}: ({i}, {j}) isn't on it"
                );
            }
        }
    }
}

/// Updating tiles together ([`TileStore::update_tiles`]) is the same as updating each in turn.
///
/// The same bits, dirty set and feed, whatever the workers, and with or without the tiles'
/// [`TileBorder`]s (the fallback no valid mesh reaches, forced on one store).
#[test]
fn updating_tiles_equals_updating_each_in_turn() {
    let dir = TestDir::new();
    let mesh = mesh();
    // Neighbouring tiles of the finest level, sharing borders; one of them (5) isn't pinned.
    let keys: Vec<TileKey> = tiles(&mesh, TOP, TOP).into_iter().take(12).collect();
    let pinned: Vec<TileKey> = (0..)
        .zip(&keys)
        .filter(|&(n, _)| n != 5)
        .map(|(_, &k)| k)
        .collect();
    // The last store's border passes find each vertex's copies one by one, as for a tile with
    // no `TileBorder`.
    let mut stores: Vec<(String, TileStore)> = [None, Some(1), Some(2), Some(5), Some(3)]
        .into_iter()
        .map(|threads| {
            let name = format!("{threads:?}");
            let config = roomy().with_workers(threads.map_or(Workers::PhysicalCores, fixed));
            let mut store = new_store(&dir, &format!("{name}.bcp"), &mesh, config);
            store.hooks.without_tile_borders = threads == Some(3);
            (name, store)
        })
        .collect();
    let guards: Vec<PinGuard> = stores
        .iter_mut()
        .map(|(_, store)| store.pin(&pinned).unwrap())
        .collect();
    for round in 0..5 {
        for (name, store) in &mut stores {
            assert_eq!(
                update_round(name, store, &keys, round, false),
                "Ok(())",
                "{name}"
            );
        }
    }
    let (reference, others) = stores.split_first_mut().unwrap();
    let reference = &mut reference.1;
    let edited: BTreeSet<TileKey> = reference.unsaved.keys().copied().collect();
    assert!(
        edited.len() > keys.len(),
        "border copies reached other tiles"
    );
    let expected_feed = reference.feed.drain(reference.save_consumer);
    for (name, store) in others.iter_mut() {
        assert_eq!(
            store.unsaved, reference.unsaved,
            "{name} workers: the same tiles dirty, with the same ops and stamps"
        );
        assert_eq!(
            store.feed.drain(store.save_consumer),
            expected_feed,
            "{name} workers: the same feed"
        );
        for key in &edited {
            assert_eq!(
                slot_resident(store, key),
                slot_resident(reference, key),
                "{name} workers, {key:?}: the same bits"
            );
        }
    }

    // A value that isn't finite in tile 3: the same error; the tiles up to it edited alike;
    // any later tile whose interior was written is dirty in the round's op.
    let before: Vec<Vec<u8>> = keys
        .iter()
        .map(|key| slot_resident(reference, key))
        .collect();
    let expected = update_round("None", reference, &keys, 5, true);
    assert!(expected.contains("NotFinite"), "{expected}");
    for (name, store) in others.iter_mut() {
        assert_eq!(
            update_round(name, store, &keys, 5, true),
            expected,
            "{name} workers"
        );
        let op = store.current_op().unwrap();
        for (n, key) in keys.iter().enumerate() {
            let bits = slot_resident(store, key);
            if n <= 3 {
                assert_eq!(
                    bits,
                    slot_resident(reference, key),
                    "{name} workers, tile {n}"
                );
            } else if n == 4 {
                assert_ne!(
                    bits, before[n],
                    "{name} workers: tile 4's interior was written"
                );
                assert_eq!(
                    store.unsaved.get(key).map(|dirty| dirty.op),
                    Some(op),
                    "{name} workers, tile 4: so it is dirty"
                );
            } else if bits != before[n] {
                assert_eq!(
                    store.unsaved.get(key).map(|dirty| dirty.op),
                    Some(op),
                    "{name} workers, tile {n}: written, so dirty"
                );
            }
        }
    }
    drop(guards);
}

/// A tile of a run whose border fails before the row of its first interior write still has its
/// interior pass applied: it is marked written, never left changed and clean.
///
/// Tile 1 of three pinned tiles gets a value that isn't finite at border vertex `(0, 2)` and a
/// write at interior vertex `(4, 5)`, which the parallel pass makes before the border is
/// reached; tile 2, after the failure, an interior write too.
#[test]
fn a_failed_update_still_marks_what_it_wrote() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, roomy());
    let keys: Vec<TileKey> = tiles(&mesh, TOP, TOP).into_iter().take(3).collect();
    let guard = store.pin(&keys).unwrap();
    store.evaluate_many(&keys, Evaluation::Positions).unwrap();
    let before: Vec<Vec<u8>> = keys.iter().map(|key| slot_resident(&store, key)).collect();
    let op = store.begin_op().unwrap();
    let outcome = store.update_tiles(&keys, |n, i, j, c| match (n, i, j) {
        (1, 0, 2) => Some(Vec3::new(f32::NAN, 0.0, 0.0)),
        (1 | 2, 4, 5) => Some(Vec3::new(c.x, c.y, c.z + 0.01)),
        _ => None,
    });
    assert!(
        matches!(
            outcome,
            Err(CacheError::Correction(CorrectionError::NotFinite))
        ),
        "{outcome:?}"
    );
    let fed: Vec<TileKey> = store
        .feed
        .drain(store.save_consumer)
        .into_iter()
        .map(|(key, _)| key)
        .collect();
    for (n, key) in keys.iter().enumerate() {
        if slot_resident(&store, key) == before[n] {
            assert!(
                !store.unsaved.contains_key(key),
                "tile {n}: unchanged, clean"
            );
            continue;
        }
        assert_eq!(
            store.unsaved.get(key).map(|dirty| dirty.op),
            Some(op),
            "tile {n}: changed, so dirty in the op"
        );
        assert!(fed.contains(key), "tile {n}: changed, so on the feed");
        assert_eq!(
            store.cached_positions(key),
            None,
            "tile {n}: changed, so its evaluation is dropped"
        );
    }
    assert_ne!(
        slot_resident(&store, &keys[1]),
        before[1],
        "tile 1's interior pass was applied"
    );
    drop(guard);
}

#[test]
fn the_cache_is_read_only_while_current() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, roomy());
    let key = tiles(&mesh, TOP, TOP)[7];
    assert_eq!(store.cached_positions(&key), None, "nothing evaluated yet");
    let positions = store.evaluate(key).unwrap().to_vec();
    assert_eq!(store.cached_positions(&key), Some(positions.as_slice()));
    assert_eq!(store.cached_normals(&key), None, "positions alone");
    let normals = store.normals(key).unwrap().to_vec();
    assert_eq!(store.cached_normals(&key), Some(normals.as_slice()));
    store.begin_op().unwrap();
    store
        .edit_tile(key, |edit| edit.set(3, 3, Vec3::new(0.0, 0.0, 0.001)))
        .unwrap()
        .unwrap();
    assert_eq!(
        store.cached_positions(&key),
        None,
        "an edit makes them stale"
    );
    assert_eq!(store.cached_normals(&key), None, "and the normals");
    let squares = store.par_map(&[1_u32, 2, 3, 4, 5], |&n| n * n);
    assert_eq!(squares, [1, 4, 9, 16, 25], "in item order");
}

#[test]
fn an_unread_edit_still_reaches_untracked_tiles() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, roomy());
    // A tile recorded after its parent's node went: the parent's ancestry is unknown.
    let (fine, parent) = (tiles(&mesh, 4, 4)[0], tiles(&mesh, 3, 3)[0]);
    store.deps.record_positions(fine, &[parent]);
    assert!(store.deps.has_untracked(), "the parent is untracked");
    // Never evaluated, read by nothing recorded: skipped, but for the untracked node.
    let coarse = tiles(&mesh, 2, 2)[0];
    assert!(
        store.deps.cascade([coarse]).positions.contains(&fine),
        "an edit of it reaches the untracked node's reader"
    );
    store.hooks.log = EvalLog::default();
    store.begin_op().unwrap();
    store
        .edit_tile(coarse, |edit| edit.set(1, 1, Vec3::new(0.001, 0.0, 0.0)))
        .unwrap()
        .unwrap();
    assert_eq!(store.hooks.log.invalidations, 1, "the cascade ran");
    assert_eq!(store.deps.parents(&fine), None, "and dropped the reader");
}

#[test]
fn an_edit_that_panics_releases_its_pins() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, roomy());
    let mut map = CorrectionMap::new();
    let finest = tiles(&mesh, TOP, TOP);
    // A corner tile: the corner vertex's copies lie in tiles of three faces.
    let key = finest[0];
    for tile in &finest {
        store.evaluate(*tile).unwrap();
    }
    store.begin_op().unwrap();
    let c = Vec3::new(0.001, -0.002, 0.0);
    let edited = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        store.edit_tile(key, |edit| {
            edit.set(0, 0, c).unwrap();
            panic!("the brush gave up");
        })
    }));
    let panic = edited.expect_err("the panic came through");
    assert_eq!(
        panic.downcast_ref::<&str>(),
        Some(&"the brush gave up"),
        "the closure's own panic"
    );
    assert_eq!(
        store.corrections.pinned_slots(),
        0,
        "no touched tile stays pinned"
    );
    let region = Region::of_tile(&key);
    map.set(
        mesh.topology(),
        key.face(),
        key.level(),
        region.x0(),
        region.y0(),
        c,
    )
    .unwrap();
    let dense = dense(&mesh, &map);
    assert_positions(
        &mut store,
        &dense,
        &finest,
        "the write before the panic, evaluated",
    );
}

#[test]
fn a_copy_that_fails_to_write_keeps_its_written_twins_invalidated() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, roomy());
    // A vertex on a face edge, on a tile border: four copies.
    let (face, level, i, j) = (FaceId::new(1), level(TOP), 64, 0);
    let copies = mesh.topology().copies(face, level, i, j);
    assert_eq!(copies.len(), 4, "four copies");
    for copy in &copies {
        store.evaluate(copy.key(level)).unwrap();
    }
    let (first, broken) = (copies[0].key(level), copies[2].key(level));
    store.correction(broken).unwrap();
    let id = store.corrections.find(&broken).unwrap();
    store.corrections.get_mut(id).unwrap().side = 0;
    store.begin_op().unwrap();
    let c = Vec3::new(0.002, 0.0, 0.0);
    let failed = store.set_correction(face, level, i, j, c);
    assert!(
        matches!(
            failed,
            Err(CacheError::Internal(Invariant::CorrectionSlot(_)))
        ),
        "the broken copy fails the edit as a broken invariant, not as rejected input: {failed:?}"
    );
    assert_eq!(
        store
            .correction(first)
            .unwrap()
            .get(copies[0].i, copies[0].j),
        Some(c),
        "the first copy was written"
    );
    assert_ne!(
        store.evaluated.freshness(&first),
        Some(Freshness::Fresh),
        "and its evaluation dropped"
    );
}

/// A row the interior pass shouldn't reach is reported as the slot's own error.
///
/// The cause is kept: a row on the border, or past the tile.
#[test]
fn an_interior_row_outside_the_interior_keeps_its_cause() {
    let side = 9;
    let mut slot = CorrectionSlot::from_tile(&CorrectionTile::zero(side).unwrap());
    let mut view = slot.view_mut().unwrap();
    let mut f = |_, _, c: Vec3| Some(c + Vec3::new(0.0, 0.001, 0.0));
    for j in [0, side - 1, side] {
        let (wrote, row) = update_row(&mut view, j, &mut f);
        assert!(!wrote, "row {j}: nothing written");
        assert!(
            matches!(
                row,
                Err(CacheError::Internal(Invariant::CorrectionSlot(
                    CorrectionError::OutOfTile { .. }
                )))
            ),
            "row {j}: {row:?}"
        );
    }
    let (wrote, row) = update_row(&mut view, 1, &mut f);
    assert!(wrote && row.is_ok(), "an interior row is written: {row:?}");
}

/// With no head of its site, a store's ops start at sequence 0, whatever other sites' heads.
#[test]
fn ops_start_at_zero_without_a_head_of_the_site() {
    let zero = OpId::new(SITE, 0).unwrap();
    assert_eq!(first_op(SITE, []).unwrap(), zero, "no heads");
    assert_eq!(first_op(SITE, [None, None]).unwrap(), zero, "no edits");
    let others = [OpId::new(SITE + 1, 900), OpId::new(SITE - 1, 5)];
    assert_eq!(first_op(SITE, others).unwrap(), zero, "other sites");
}

/// Ops start past the newest head of the site, ignoring other sites' heads, however much newer.
#[test]
fn ops_start_past_the_newest_head_of_the_site_only() {
    let heads = [
        OpId::new(SITE, 7),
        None,
        OpId::new(SITE + 1, MAX_SEQ),
        OpId::new(SITE, 41),
        OpId::new(SITE, 3),
    ];
    let next = OpId::new(SITE, 42).unwrap();
    assert_eq!(first_op(SITE, heads).unwrap(), next, "past 41");
}

/// A head at the last sequence of the site leaves no op to start at.
#[test]
fn a_head_at_the_last_op_of_the_site_exhausts_its_ops() {
    let heads = [OpId::new(SITE, 1), OpId::new(SITE, MAX_SEQ)];
    assert!(
        matches!(first_op(SITE, heads), Err(CacheError::OpsExhausted)),
        "the last sequence"
    );
    let below = [OpId::new(SITE, MAX_SEQ - 1)];
    let last = OpId::new(SITE, MAX_SEQ).unwrap();
    assert_eq!(
        first_op(SITE, below).unwrap(),
        last,
        "the last op is still given"
    );
}

/// A correction no tile's scale can hold is the caller's error, `TooLarge`, by every way the
/// store writes one, and nothing is written.
#[test]
fn a_correction_too_large_to_store_is_the_caller_s_error() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, roomy());
    let key = tiles(&mesh, TOP, TOP)[0];
    let region = Region::of_tile(&key);
    // Finite, but past the largest interior value any scale decodes.
    let huge = Vec3::new(f32::MAX, 0.0, 0.0);
    let at = |i: u32, j: u32| ((i, j) == (5, 5)).then_some(huge);
    let too_large = |got: &Result<(), CacheError>| {
        matches!(got, Err(CacheError::Correction(CorrectionError::TooLarge)))
    };
    let (x, y) = (region.x0() + 5, region.y0() + 5);
    let one = store.set_correction(key.face(), key.level(), x, y, huge);
    assert!(too_large(&one), "set_correction: {one:?}");
    let edited = store.edit_tile(key, |edit| edit.set(5, 5, huge)).unwrap();
    assert!(too_large(&edited), "TileEdit::set: {edited:?}");
    let updated = store
        .edit_tile(key, |edit| edit.update(|i, j, _| at(i, j)))
        .unwrap();
    assert!(too_large(&updated), "TileEdit::update: {updated:?}");
    let guard = store.pin(&[key]).unwrap();
    let in_parallel = store.update_tiles(&[key], |_, i, j, _| at(i, j));
    assert!(too_large(&in_parallel), "update_tiles: {in_parallel:?}");
    drop(guard);
    assert!(store.unsaved.is_empty(), "nothing was written");
}

/// The ops of the edits on `store`'s feed since the last call, each once.
fn ops_edited(store: &mut TileStore) -> BTreeSet<OpId> {
    let drained = store.feed.drain(store.save_consumer);
    drained.into_iter().map(|(_, op)| op).collect()
}

/// Ops reserved one after another come in that order, each after the op current as they are
/// reserved, which stays current.
#[test]
fn ops_reserved_in_order_are_ordered_after_the_current_op() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, roomy());
    let first = store.reserve_op().unwrap();
    assert_eq!(store.current_op(), None, "reserving begins nothing");
    store.drop_reserved_op(first);
    let current = store.begin_op().unwrap();
    let reserved: Vec<ReservedOp> = (0..3).map(|_| store.reserve_op().unwrap()).collect();
    let ops: Vec<OpId> = reserved.iter().map(ReservedOp::op).collect();
    assert!(ops.is_sorted(), "{ops:?}");
    assert!(
        ops.windows(2).all(|pair| pair[0] != pair[1]),
        "each its own: {ops:?}"
    );
    assert!(ops[0] > current, "after the current op");
    assert_eq!(store.current_op(), Some(current), "which stays current");
}

/// A reserved op isn't current: edits go to the current op, and once it has ended they are
/// refused, as is a new op, until the oldest reserved op begins, which only it may and only
/// then.
#[test]
fn a_reserved_op_takes_no_edit_until_it_begins_after_the_current_op_ends() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, roomy());
    let key = tiles(&mesh, TOP, TOP)[0];
    let region = Region::of_tile(&key);
    let (column, row) = (region.x0() + 5, region.y0() + 5);
    let set = |on: &mut TileStore, height: f32| {
        let lift = Vec3::new(0.0, 0.0, height);
        on.set_correction(key.face(), key.level(), column, row, lift)
    };
    let a = store.begin_op().unwrap();
    let b = store.reserve_op().unwrap();
    let c = store.reserve_op().unwrap();
    set(&mut store, 0.01).unwrap();
    assert_eq!(ops_edited(&mut store), BTreeSet::from([a]), "edits are A's");

    let early = store.begin_reserved_op(b).unwrap_err();
    assert!(
        matches!(early.error(), CacheError::OpBegun(op) if *op == a),
        "A hasn't ended: {early:?}"
    );
    let b_back = early.into_reserved();
    store.end_op();
    let edit = set(&mut store, 0.02);
    assert!(
        matches!(edit, Err(CacheError::OpReserved(op)) if op == b_back.op()),
        "no op is current, and a new one would pass B: {edit:?}"
    );
    let fresh = store.begin_op();
    assert!(
        matches!(fresh, Err(CacheError::OpReserved(op)) if op == b_back.op()),
        "nor may a new op begin: {fresh:?}"
    );
    let resumed = store.resume_op(a);
    assert!(
        matches!(resumed, Err(CacheError::OpReserved(op)) if op == b_back.op()),
        "nor an op resume: {resumed:?}"
    );
    let out_of_turn = store.begin_reserved_op(c).unwrap_err();
    assert!(
        matches!(out_of_turn.error(), CacheError::OpReserved(op) if *op == b_back.op()),
        "B before C: {out_of_turn:?}"
    );
    let c_back = out_of_turn.into_reserved();
    assert!(ops_edited(&mut store).is_empty(), "nothing was edited");

    let b_op = store.begin_reserved_op(b_back).unwrap();
    assert_eq!(store.current_op(), Some(b_op), "B is current");
    set(&mut store, 0.03).unwrap();
    assert_eq!(
        ops_edited(&mut store),
        BTreeSet::from([b_op]),
        "edits are B's"
    );
    store.end_op();
    let c_op = store.begin_reserved_op(c_back).unwrap();
    assert!(c_op > b_op);
    set(&mut store, 0.04).unwrap();
    assert_eq!(ops_edited(&mut store), BTreeSet::from([c_op]), "then C's");
    store.end_op();
    let next = store.begin_op().unwrap();
    assert!(next > c_op, "a new op once none is reserved, after them");
}

/// A reserved op is refused by a store it wasn't reserved on, given back, and begins on its own.
#[test]
fn a_reserved_op_is_its_own_store_s_only() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, roomy());
    let mut other = new_store(&dir, "other.bcp", &mesh, roomy());
    let b = store.reserve_op().unwrap();
    let elsewhere = other.begin_reserved_op(b).unwrap_err();
    assert!(
        matches!(elsewhere.error(), CacheError::NotReserved(_)),
        "{elsewhere:?}"
    );
    let read = other.read_pen_down_of(elsewhere.reserved());
    assert!(matches!(read, Err(CacheError::NotReserved(_))), "{read:?}");
    let b_back = elsewhere.into_reserved();
    let b_op = store.begin_reserved_op(b_back).unwrap();
    assert_eq!(store.current_op(), Some(b_op));
}
