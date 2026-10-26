//! The change feed: which edits its consumers see, and how often.

use super::*;

#[test]
fn a_tile_edited_by_one_op_goes_to_the_feed_once() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, roomy());
    let key = tiles(&mesh, TOP, TOP)[5];
    let region = Region::of_tile(&key);
    store.begin_op().unwrap();
    for n in 0..50 {
        let (i, j) = (region.x0() + 1 + n, region.y0() + 7);
        store
            .set_correction(key.face(), key.level(), i, j, Vec3::new(0.001, 0.0, 0.0))
            .unwrap();
    }
    assert_eq!(
        store.feed.backlog(),
        1,
        "one interior tile, one op: one entry"
    );
    store
        .edit_tile(key, |edit| edit.set(3, 3, Vec3::ZERO))
        .unwrap()
        .unwrap();
    assert_eq!(store.feed.backlog(), 1, "batched or not");
    let next = store.begin_op().unwrap();
    store
        .edit_tile(key, |edit| edit.set(3, 4, Vec3::ZERO))
        .unwrap()
        .unwrap();
    assert_eq!(store.feed.backlog(), 2, "a new op is a new entry");
    store.save().unwrap();
    store
        .edit_tile(key, |edit| edit.set(3, 5, Vec3::ZERO))
        .unwrap()
        .unwrap();
    assert_eq!(
        store.feed.drain(store.save_consumer),
        vec![(key, next)],
        "written by the save, so clean: the same op goes to the feed again"
    );
}

#[test]
fn a_consumer_that_drained_a_tile_sees_its_next_edit_of_the_same_op() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, roomy());
    // A second consumer beside save, which drains nothing here.
    let other = store.feed.subscribe();
    let key = tiles(&mesh, TOP, TOP)[5];
    let op = store.begin_op().unwrap();
    store
        .edit_tile(key, |edit| edit.set(3, 3, Vec3::new(0.001, 0.0, 0.0)))
        .unwrap()
        .unwrap();
    assert_eq!(store.feed.drain(other), vec![(key, op)], "the first dab");
    store
        .edit_tile(key, |edit| edit.set(4, 4, Vec3::new(0.001, 0.0, 0.0)))
        .unwrap()
        .unwrap();
    assert_eq!(
        store.feed.drain(other),
        vec![(key, op)],
        "the second dab of the same op, although the tile was dirty with it already"
    );
    assert_eq!(
        store.feed.drain(store.save_consumer),
        vec![(key, op)],
        "save, which drained neither, sees the tile once"
    );
}
