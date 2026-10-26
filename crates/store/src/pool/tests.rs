//! Tests of the tile pool: its slots, chunks, pins, eviction and budget.

use geom::{FaceId, Level};

use super::*;
use crate::StoreError;
use crate::slot::{CorrectionSlot, EvaluatedSlot};

/// A small slot for the rule tests.
type Small = [u64; 2];

fn key(n: u32) -> TileKey {
    TileKey::displacement(FaceId::new(n), Level::new(9).unwrap(), 0, 0).unwrap()
}

/// A write-back for pools that must have nothing dirty to write.
///
/// # Errors
///
/// Never: it panics instead.
fn no_write_back<T>(key: TileKey, _: &T) -> Result<(), StoreError> {
    panic!("unexpected write-back of {key:?}")
}

/// A pool of `slots` [`Small`] slots per chunk, and a budget of `chunks` of them.
fn small(slots: usize, chunks: usize) -> (Pool<Small>, Budget) {
    let chunk = slots * size_of::<Small>();
    (Pool::new(chunk), Budget::new(chunks * chunk))
}

/// Acquires `key` and stores `value` in its slot.
fn put(pool: &mut Pool<Small>, budget: &mut Budget, n: u32, value: u64) -> SlotId<Small> {
    let id = pool.acquire(key(n), budget, no_write_back).unwrap();
    *pool.get_mut(id).unwrap() = [value, u64::from(n)];
    id
}

/// A pinned handle in [`never_exceeds_the_budget`]'s workload: handles of the two pools are
/// different types.
enum Pinned {
    /// In the correction pool.
    Corrections(SlotId<CorrectionSlot>),
    /// In the evaluated pool.
    Evaluated(SlotId<EvaluatedSlot>),
}

/// A write-back for pools whose dirty slots may be dropped.
///
/// # Errors
///
/// Never.
#[expect(
    clippy::unnecessary_wraps,
    reason = "it is a pool write-back, whose signature returns a result"
)]
const fn write_nothing<T>(_: TileKey, _: &T) -> Result<(), StoreError> {
    Ok(())
}

/// Slots of `pool` with pins, counted one by one: what its counter must say.
fn pinned_by_scan<T: SlotData>(pool: &Pool<T>) -> usize {
    pool.meta.iter().flatten().filter(|m| m.pins > 0).count()
}

/// What holds after every step of [`never_exceeds_the_budget`]: the pools are within the
/// budget, which counts exactly their chunks, pinned slots stay, and the pinned counts are
/// right.
fn check_step(
    step: u32,
    budget: &Budget,
    corrections: &Pool<CorrectionSlot>,
    evaluated: &Pool<EvaluatedSlot>,
    pinned: &[Pinned],
) {
    assert!(
        budget.used() <= budget.limit(),
        "step {step}: {} > {}",
        budget.used(),
        budget.limit()
    );
    assert_eq!(
        budget.used(),
        corrections.bytes() + evaluated.bytes(),
        "step {step}: the budget counts exactly the mapped chunks"
    );
    for handle in pinned {
        let live = match *handle {
            Pinned::Corrections(id) => corrections.get(id).is_some(),
            Pinned::Evaluated(id) => evaluated.get(id).is_some(),
        };
        assert!(live, "step {step}: a pinned slot stays");
    }
    assert_eq!(
        (corrections.pinned_slots(), evaluated.pinned_slots()),
        (pinned_by_scan(corrections), pinned_by_scan(evaluated)),
        "step {step}: the pinned counters"
    );
}

#[test]
fn get_many_mut_borrows_each_slot_once_in_the_order_asked() {
    let (mut pool, mut budget) = small(3, 3);
    let ids: Vec<SlotId<Small>> = (0..8)
        .map(|n| put(&mut pool, &mut budget, n, u64::from(n) * 10))
        .collect();
    // Out of slot order, across chunks.
    let asked = [ids[5], ids[0], ids[7], ids[3], ids[4]];
    let slots = pool.get_many_mut(&asked).unwrap();
    let seen: Vec<u64> = slots.iter().map(|slot| slot[1]).collect();
    assert_eq!(seen, [5, 0, 7, 3, 4], "each slot in the order asked");
    for slot in slots {
        slot[0] += 1;
    }
    for (n, &id) in (0_u32..).zip(&ids) {
        let bumped = u64::from([5, 0, 7, 3, 4].contains(&n));
        assert_eq!(
            pool.get(id).unwrap()[0],
            u64::from(n) * 10 + bumped,
            "slot {n}"
        );
    }
    for twice in [ids[2], ids[4]] {
        assert!(
            pool.get_many_mut(&[twice, ids[6], twice]).is_none(),
            "a slot named twice is refused, at a chunk's end or inside it"
        );
    }
    assert_eq!(
        pool.get_many_mut(&[]).map(|none| none.len()),
        Some(0),
        "none"
    );
    pool.remove(&key(1)).unwrap();
    assert!(
        pool.get_many_mut(&[ids[0], ids[1]]).is_none(),
        "a stale handle is refused"
    );
}

#[test]
fn never_exceeds_the_budget() {
    // Three chunks and three evaluated slots: the evaluated pool maps a short chunk of three
    // slots where a whole chunk doesn't fit.
    let chunk = 4 * size_of::<EvaluatedSlot>();
    let mut budget = Budget::new(3 * chunk + 3 * size_of::<EvaluatedSlot>());
    let mut corrections = Pool::<CorrectionSlot>::new(chunk);
    let mut evaluated = Pool::<EvaluatedSlot>::new(chunk).with_short_chunks();
    let mut short = 0;
    let mut rng = 0x2545_F491_4F6C_DD1D_u64;
    let mut next = |n: u64| {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng % n
    };
    let mut pinned: Vec<Pinned> = Vec::new();
    let (mut fulls, mut grown) = (0, 0);
    for step in 0..20_000 {
        let in_corrections = next(2) == 0;
        let k = key(u32::try_from(next(48)).unwrap());
        macro_rules! with_pool {
            (| $pool:ident | $body:expr) => {
                if in_corrections {
                    let $pool = &mut corrections;
                    $body
                } else {
                    let $pool = &mut evaluated;
                    $body
                }
            };
        }
        match next(16) {
            0..=5 => with_pool!(|pool| {
                if pool.lookup(&k).is_none() {
                    let before = pool.bytes();
                    match pool.acquire(k, &mut budget, write_nothing) {
                        Ok(id) => assert!(pool.get(id).is_some(), "fresh handle is live"),
                        Err(PoolError::Full) => fulls += 1,
                        Err(e) => panic!("acquire: {e}"),
                    }
                    grown += usize::from(pool.bytes() > before);
                }
                short += usize::from(!evaluated.bytes().is_multiple_of(chunk));
            }),
            6 | 7 => with_pool!(|pool| {
                pool.lookup(&k);
            }),
            8 if pinned.len() < 6 => {
                let handle = if in_corrections {
                    corrections.lookup(&k).map(|(id, _)| {
                        corrections.pin(id).unwrap();
                        Pinned::Corrections(id)
                    })
                } else {
                    evaluated.lookup(&k).map(|(id, _)| {
                        evaluated.pin(id).unwrap();
                        Pinned::Evaluated(id)
                    })
                };
                pinned.extend(handle);
            }
            9 => match pinned.pop() {
                Some(Pinned::Corrections(id)) => corrections.unpin(id).unwrap(),
                Some(Pinned::Evaluated(id)) => evaluated.unpin(id).unwrap(),
                None => {}
            },
            10 | 11 => with_pool!(|pool| {
                if let Some((id, _)) = pool.lookup(&k) {
                    pool.set_dirty(id, next(2) == 0).unwrap();
                }
            }),
            12 => with_pool!(|pool| match pool.remove(&k) {
                Ok(_) | Err(PoolError::Pinned | PoolError::Dirty) => {}
                Err(e) => panic!("remove: {e}"),
            }),
            13 => with_pool!(|pool| {
                pool.invalidate(&k);
            }),
            14 if step % 7 == 0 => with_pool!(|pool| pool.evict_all(write_nothing).unwrap()),
            15 if step % 5 == 0 => with_pool!(|pool| {
                let target = usize::try_from(next(3)).unwrap() * chunk;
                pool.shrink_to(&mut budget, target, write_nothing).unwrap();
            }),
            _ => {}
        }
        check_step(step, &budget, &corrections, &evaluated, &pinned);
    }
    assert!(fulls > 0 && grown > 10, "the workload hit the budget");
    assert!(short > 0, "the evaluated pool mapped a short chunk");
}

#[test]
fn everything_pinned_is_full() {
    let (mut pool, mut budget) = small(4, 1);
    let ids: Vec<SlotId<Small>> = (0..4).map(|n| put(&mut pool, &mut budget, n, 7)).collect();
    for &id in &ids {
        pool.pin(id).unwrap();
    }
    assert!(
        matches!(
            pool.acquire(key(99), &mut budget, no_write_back),
            Err(PoolError::Full)
        ),
        "nothing to evict"
    );
    for (n, &id) in (0_u32..).zip(&ids) {
        assert_eq!(
            pool.lookup(&key(n)),
            Some((id, Freshness::Fresh)),
            "tile {n} kept its slot"
        );
        assert_eq!(pool.get(id), Some(&[7, u64::from(n)]), "tile {n} unchanged");
    }
    assert!(
        pool.evict_all(no_write_back).is_ok(),
        "evict_all skips them"
    );
    assert_eq!(
        pool.lookup(&key(0)),
        Some((ids[0], Freshness::Fresh)),
        "still there"
    );
    pool.unpin(ids[0]).unwrap();
    assert!(
        matches!(pool.unpin(ids[0]), Err(PoolError::NotPinned)),
        "pins balance"
    );
    let id = pool.acquire(key(99), &mut budget, no_write_back).unwrap();
    assert_eq!(
        id.index, ids[0].index,
        "the one unpinned slot is the victim"
    );
    assert_eq!(pool.get(id), Some(&[0, 0]), "and comes back zeroed");
}

#[test]
fn dirty_victims_are_written_back_once() {
    let (mut pool, mut budget) = small(4, 1);
    for n in 0..4 {
        let id = put(&mut pool, &mut budget, n, 100 + u64::from(n));
        pool.set_dirty(id, n % 2 == 0).unwrap();
    }
    // A failing write-back leaves the victim dirty and in place.
    let failed = pool.acquire(key(10), &mut budget, |_, _: &Small| {
        Err(StoreError::ReadOnly)
    });
    assert!(
        matches!(failed, Err(PoolError::WriteBack(_))),
        "error returned"
    );
    let (victim, _) = pool.lookup(&key(0)).unwrap();
    assert_eq!(pool.is_dirty(victim), Some(true), "still dirty");
    assert_eq!(pool.get(victim), Some(&[100, 0]), "still there");

    let mut written = Vec::new();
    for n in 10..20 {
        pool.acquire(key(n), &mut budget, |k, v: &Small| {
            written.push((k, *v));
            Ok::<_, StoreError>(())
        })
        .unwrap();
    }
    written.sort_unstable_by_key(|(k, _)| *k);
    assert_eq!(
        written,
        vec![(key(0), [100, 0]), (key(2), [102, 2])],
        "each dirty victim once, clean ones never"
    );
    written.clear();
    let (id, _) = pool.lookup(&key(19)).unwrap();
    pool.set_dirty(id, true).unwrap();
    pool.evict_all(|k, v: &Small| {
        written.push((k, *v));
        Ok::<_, StoreError>(())
    })
    .unwrap();
    assert_eq!(
        written,
        vec![(key(19), [0, 0])],
        "evict_all writes back too"
    );
    assert_eq!(pool.lookup(&key(19)), None, "and frees");
}

#[test]
fn stale_handles_are_rejected() {
    let (mut pool, mut budget) = small(2, 1);
    let a = put(&mut pool, &mut budget, 0, 1);
    put(&mut pool, &mut budget, 1, 2);
    // Both are referenced: the clock clears them, then takes slot 0 for tile 2.
    let c = put(&mut pool, &mut budget, 2, 3);
    assert_eq!((c.chunk, c.index), (a.chunk, a.index), "slot reused");
    assert_eq!(pool.get(a), None, "evicted handle");
    assert!(matches!(pool.pin(a), Err(PoolError::Stale)), "pin");
    assert_eq!(pool.is_dirty(a), None, "is_dirty");
    assert_eq!(pool.get(c), Some(&[3, 2]), "the new tile");

    assert!(pool.remove(&key(2)).unwrap(), "removed");
    assert_eq!(pool.get(c), None, "removed handle");
    assert!(!pool.remove(&key(2)).unwrap(), "already gone");
    let d = put(&mut pool, &mut budget, 3, 4);
    assert_eq!((d.chunk, d.index), (c.chunk, c.index), "slot reused");
    assert_eq!(pool.get(c), None, "still stale after reuse");
    assert!(
        matches!(
            pool.acquire(key(3), &mut budget, no_write_back),
            Err(PoolError::Present)
        ),
        "one slot per tile"
    );
}

#[test]
fn a_clean_acquire_never_writes_back() {
    let (mut pool, mut budget) = small(2, 1);
    let dirty = put(&mut pool, &mut budget, 0, 1);
    pool.set_dirty(dirty, true).unwrap();
    put(&mut pool, &mut budget, 1, 2);
    let id = pool.acquire_clean(key(2), &mut budget).unwrap();
    assert_eq!(pool.lookup(&key(1)), None, "the clean tile was the victim");
    assert_eq!(pool.get(dirty), Some(&[1, 0]), "the dirty one stays");
    pool.set_dirty(id, true).unwrap();
    assert!(
        matches!(
            pool.acquire_clean(key(3), &mut budget),
            Err(PoolError::Full)
        ),
        "every unpinned slot dirty: full, nothing written"
    );
    assert_eq!(pool.is_dirty(dirty), Some(true), "still dirty");
    assert_eq!(pool.is_dirty(id), Some(true), "both");
}

#[test]
fn remove_refuses_pinned_and_dirty_slots() {
    let (mut pool, mut budget) = small(4, 1);
    let dirty = put(&mut pool, &mut budget, 0, 5);
    pool.set_dirty(dirty, true).unwrap();
    assert!(
        matches!(pool.remove(&key(0)), Err(PoolError::Dirty)),
        "a dirty slot's changes would be lost"
    );
    assert_eq!(pool.get(dirty), Some(&[5, 0]), "it stays as it was");
    assert_eq!(pool.is_dirty(dirty), Some(true), "and dirty");
    let pinned = put(&mut pool, &mut budget, 1, 6);
    pool.pin(pinned).unwrap();
    assert!(
        matches!(pool.remove(&key(1)), Err(PoolError::Pinned)),
        "a pinned slot is in use"
    );
    assert_eq!(pool.get(pinned), Some(&[6, 1]), "it stays too");
    pool.set_dirty(dirty, false).unwrap();
    assert!(pool.remove(&key(0)).unwrap(), "clean and unpinned: removed");
}

#[test]
fn invalidating_an_unpinned_clean_slot_frees_it() {
    let (mut pool, mut budget) = small(2, 1);
    assert_eq!(pool.invalidate(&key(0)), Invalidated::Absent, "no slot");
    let a = put(&mut pool, &mut budget, 0, 1);
    put(&mut pool, &mut budget, 1, 2);
    assert_eq!(pool.invalidate(&key(0)), Invalidated::Freed, "freed");
    assert_eq!(pool.lookup(&key(0)), None, "gone");
    assert_eq!(pool.get(a), None, "its handle is stale");
    let b = put(&mut pool, &mut budget, 2, 3);
    assert_eq!(
        (b.chunk, b.index),
        (a.chunk, a.index),
        "back on the free list"
    );
    assert_eq!(
        pool.lookup(&key(1)).map(|(_, f)| f),
        Some(Freshness::Fresh),
        "the other tile wasn't touched"
    );
}

#[test]
fn invalidating_a_pinned_slot_marks_it_stale_in_place() {
    let (mut pool, mut budget) = small(4, 1);
    let ids: Vec<SlotId<Small>> = (0..4).map(|n| put(&mut pool, &mut budget, n, 8)).collect();
    pool.pin(ids[2]).unwrap();
    pool.pin(ids[2]).unwrap();
    assert_eq!(
        pool.invalidate(&key(2)),
        Invalidated::Marked,
        "pinned: marked"
    );
    assert_eq!(
        pool.lookup(&key(2)),
        Some((ids[2], Freshness::Stale)),
        "same key, same handle, reported stale"
    );
    assert_eq!(
        pool.get(ids[2]),
        Some(&[8, 2]),
        "contents kept for the reader"
    );
    *pool.get_mut(ids[2]).unwrap() = [9, 2];
    pool.mark_fresh(ids[2]).unwrap();
    assert_eq!(
        pool.lookup(&key(2)),
        Some((ids[2], Freshness::Fresh)),
        "recomputed in place"
    );

    // Stale again, then unpinned: the clock takes it first, although every slot was
    // referenced and the hand starts at slot 0.
    assert_eq!(
        pool.invalidate(&key(2)),
        Invalidated::Marked,
        "still pinned"
    );
    pool.unpin(ids[2]).unwrap();
    pool.unpin(ids[2]).unwrap();
    let id = pool.acquire(key(9), &mut budget, no_write_back).unwrap();
    assert_eq!(
        (id.chunk, id.index),
        (ids[2].chunk, ids[2].index),
        "the stale slot is the first victim"
    );
    assert_eq!(pool.get(ids[2]), None, "its old handle is stale");
    assert_eq!(
        pool.lookup(&key(9)).map(|(_, f)| f),
        Some(Freshness::Fresh),
        "fresh"
    );
    for n in [0, 1, 3] {
        assert!(pool.lookup(&key(n)).is_some(), "tile {n} kept");
    }
}

#[test]
fn invalidating_a_dirty_slot_never_frees_it() {
    let (mut pool, mut budget) = small(2, 1);
    let id = put(&mut pool, &mut budget, 0, 4);
    pool.set_dirty(id, true).unwrap();
    assert_eq!(
        pool.invalidate(&key(0)),
        Invalidated::Marked,
        "dirty: marked"
    );
    assert_eq!(pool.get(id), Some(&[4, 0]), "its changes kept");
    assert_eq!(pool.is_dirty(id), Some(true), "still dirty");
    assert_eq!(
        pool.lookup(&key(0)),
        Some((id, Freshness::Stale)),
        "reported stale"
    );
    pool.set_dirty(id, false).unwrap();
    assert_eq!(
        pool.invalidate(&key(0)),
        Invalidated::Freed,
        "clean now: freed"
    );
}

#[test]
fn the_eviction_log_names_what_the_pool_freed_on_its_own() {
    let chunk = 2 * size_of::<Small>();
    let (mut pool, mut budget) = (Pool::<Small>::with_eviction_log(chunk), Budget::new(chunk));
    for n in 0..2 {
        pool.acquire(key(n), &mut budget, no_write_back).unwrap();
    }
    assert_eq!(
        pool.invalidate(&key(1)),
        Invalidated::Freed,
        "freed by the caller"
    );
    pool.acquire(key(2), &mut budget, no_write_back).unwrap();
    pool.acquire(key(3), &mut budget, no_write_back).unwrap();
    assert_eq!(
        pool.drain_evicted(),
        vec![key(0)],
        "the clock's victim only"
    );
    assert!(pool.drain_evicted().is_empty(), "drained");
    pool.evict_all(no_write_back).unwrap();
    let mut evicted = pool.drain_evicted();
    evicted.sort_unstable();
    assert_eq!(evicted, vec![key(2), key(3)], "evict_all");
    let (mut quiet, mut quiet_budget) = small(1, 1);
    put(&mut quiet, &mut quiet_budget, 0, 1);
    put(&mut quiet, &mut quiet_budget, 1, 1);
    assert!(quiet.drain_evicted().is_empty(), "no log unless asked for");
}

/// The keys `pool` holds, sorted.
fn resident(pool: &Pool<Small>) -> Vec<TileKey> {
    let mut keys: Vec<TileKey> = pool.index.keys().copied().collect();
    keys.sort_unstable();
    keys
}

#[test]
fn shrinking_gives_back_the_chunk_with_fewest_dirty_tiles_first() {
    let (mut pool, mut budget) = small(2, 3);
    let ids: Vec<SlotId<Small>> = (0..6).map(|n| put(&mut pool, &mut budget, n, 1)).collect();
    // Chunk 0: no dirty tile; chunk 1: one; chunk 2: both. Every slot is in use, so only
    // the dirty tiles tell the chunks apart: the highest goes last, not first.
    for &n in &[2, 4, 5] {
        pool.set_dirty(ids[n], true).unwrap();
    }
    let chunk = pool.chunk_bytes();
    let mut written = Vec::new();
    let mut write = |k: TileKey, _: &Small| {
        written.push(k);
        Ok::<_, StoreError>(())
    };
    pool.shrink_to(&mut budget, 2 * chunk, &mut write).unwrap();
    assert_eq!(
        resident(&pool),
        (2..6).map(key).collect::<Vec<_>>(),
        "chunk 0"
    );
    pool.shrink_to(&mut budget, chunk, &mut write).unwrap();
    assert_eq!(resident(&pool), vec![key(4), key(5)], "then chunk 1");
    pool.shrink_to(&mut budget, 0, &mut write).unwrap();
    assert_eq!(
        written,
        vec![key(2), key(4), key(5)],
        "no write, one, then two"
    );
}

#[test]
fn shrinking_gives_back_the_chunk_with_fewest_tiles_in_use_first() {
    let (mut pool, mut budget) = small(2, 3);
    let ids: Vec<SlotId<Small>> = (0..6).map(|n| put(&mut pool, &mut budget, n, 1)).collect();
    // Chunk 0: two stale tiles, unreferenced once unpinned; chunk 1: one tile and a free
    // slot; chunk 2: two current tiles.
    for &n in &[0, 1] {
        pool.pin(ids[n]).unwrap();
        assert_eq!(
            pool.invalidate(&key(u32::try_from(n).unwrap())),
            Invalidated::Marked
        );
        pool.unpin(ids[n]).unwrap();
    }
    assert_eq!(pool.invalidate(&key(3)), Invalidated::Freed, "a free slot");
    let chunk = pool.chunk_bytes();
    pool.shrink_to(&mut budget, 2 * chunk, no_write_back)
        .unwrap();
    assert_eq!(
        resident(&pool),
        vec![key(2), key(4), key(5)],
        "chunk 0 first"
    );
    pool.shrink_to(&mut budget, chunk, no_write_back).unwrap();
    assert_eq!(resident(&pool), vec![key(4), key(5)], "then chunk 1");
}

/// A pool with short chunks fills the room below a whole chunk with whole slots.
///
/// As many as fit, if that is at least half a chunk, charged exactly their bytes and given
/// exactly them back; a pool without maps whole chunks only.
#[test]
fn short_chunks_fill_the_room_below_a_chunk() {
    let size = size_of::<Small>();
    let chunk = 4 * size;
    for (room, short) in [
        (3 * size, Some(3)),
        (3 * size + 7, Some(3)),
        (2 * size, Some(2)),
        (size, None),
    ] {
        let mut pool = Pool::<Small>::new(chunk).with_short_chunks();
        let mut budget = Budget::new(2 * chunk + room);
        let ids: Vec<SlotId<Small>> = (0..8).map(|n| put(&mut pool, &mut budget, n, 1)).collect();
        assert_eq!(pool.bytes(), 2 * chunk, "{room}: two whole chunks first");
        let more = pool.acquire(key(8), &mut budget, no_write_back);
        let Some(slots) = short else {
            assert!(more.is_ok(), "{room}: a victim instead");
            assert_eq!(pool.bytes(), 2 * chunk, "{room}: no short chunk under half");
            continue;
        };
        more.unwrap();
        assert_eq!(
            pool.bytes(),
            2 * chunk + slots * size,
            "{room}: whole slots"
        );
        assert_eq!(budget.used(), pool.bytes(), "{room}: charged exactly");
        assert!(budget.used() <= budget.limit(), "{room}: within the budget");
        for n in 9..8 + u32::try_from(slots).unwrap() {
            put(&mut pool, &mut budget, n, 1);
        }
        assert_eq!(
            pool.bytes(),
            2 * chunk + slots * size,
            "{room}: filled, no more"
        );
        assert!(pool.get(ids[0]).is_some(), "{room}: nothing evicted yet");
        pool.shrink_to(&mut budget, 0, no_write_back).unwrap();
        assert_eq!(
            (pool.bytes(), budget.used()),
            (0, 0),
            "{room}: all given back"
        );
    }
    let mut whole = Pool::<Small>::new(chunk);
    let mut budget = Budget::new(2 * chunk + 3 * size);
    for n in 0..9 {
        whole.acquire(key(n), &mut budget, no_write_back).unwrap();
    }
    assert_eq!(
        whole.bytes(),
        2 * chunk,
        "without short chunks: whole ones only"
    );
}

/// A short chunk holding as many slots as a whole one is charged its slots' bytes, not the
/// whole chunk's.
///
/// A chunk size that isn't a whole number of slots leaves slack after a whole chunk's last
/// slot; room between those slots' bytes and a whole chunk maps a short chunk of a whole
/// chunk's slots. The budget counts exactly what the pool holds through mapping, giving back
/// and mapping again.
#[test]
fn a_short_chunk_of_a_whole_chunks_slots_is_charged_its_slots() {
    let size = size_of::<Small>();
    let chunk = 4 * size + 7;
    let mut pool = Pool::<Small>::new(chunk).with_short_chunks();
    let mut budget = Budget::new(2 * chunk + 4 * size);
    let check = |mapped: &Pool<Small>, charged: &Budget, what: &str| {
        assert_eq!(
            charged.used(),
            mapped.bytes(),
            "{what}: the budget counts the pool"
        );
        assert!(
            charged.used() <= charged.limit(),
            "{what}: within the limit"
        );
    };
    for round in 0..2 {
        for n in 0..12 {
            put(&mut pool, &mut budget, n, 1);
        }
        assert_eq!(
            pool.bytes(),
            2 * chunk + 4 * size,
            "round {round}: two whole chunks and a short one of four slots"
        );
        check(&pool, &budget, &format!("round {round}: mapped"));
        pool.shrink_to(&mut budget, chunk + 4 * size, no_write_back)
            .unwrap();
        check(&pool, &budget, &format!("round {round}: one given back"));
        pool.shrink_to(&mut budget, 0, no_write_back).unwrap();
        check(&pool, &budget, &format!("round {round}: all given back"));
        assert_eq!(budget.used(), 0, "round {round}: nothing left charged");
    }
}

#[test]
fn shrink_returns_chunks() {
    let (mut pool, mut budget) = small(2, 3);
    let ids: Vec<SlotId<Small>> = (0..6).map(|n| put(&mut pool, &mut budget, n, 9)).collect();
    assert_eq!(pool.bytes(), budget.limit(), "filled");
    pool.set_dirty(ids[5], true).unwrap();
    pool.pin(ids[3]).unwrap();
    let mut written = Vec::new();
    pool.shrink_to(&mut budget, 0, |k, _: &Small| {
        written.push(k);
        Ok::<_, StoreError>(())
    })
    .unwrap();
    assert_eq!(
        written,
        vec![key(5)],
        "dirty slots written back before going"
    );
    assert_eq!(
        pool.bytes(),
        2 * size_of::<Small>(),
        "the pinned slot's chunk stays"
    );
    assert_eq!(budget.used(), pool.bytes(), "budget given back");
    assert_eq!(pool.get(ids[3]), Some(&[9, 3]), "pinned tile kept");
    pool.unpin(ids[3]).unwrap();
    pool.evict_all(no_write_back).unwrap();
    pool.shrink_to(&mut budget, 0, no_write_back).unwrap();
    assert_eq!((pool.bytes(), budget.used()), (0, 0), "all gone");

    // Chunks come back, and handles from before stay stale.
    let id = put(&mut pool, &mut budget, 7, 1);
    assert_eq!(
        (id.chunk, id.index),
        (ids[0].chunk, ids[0].index),
        "same place"
    );
    assert_eq!(pool.get(ids[0]), None, "old handle stale");
    assert_eq!(pool.get(id), Some(&[1, 7]), "fresh slot");
}
