//! The stale-tile feed alone: epochs delivered whole and in order, and a stalled consumer told
//! to resync with its memory bounded.

use geom::{FaceId, Level};

use super::*;

/// A level-5 tile of face `face`.
fn key(face: u32) -> TileKey {
    TileKey::display(FaceId::new(face), Level::new(5).unwrap(), 0, 0).unwrap()
}

/// An epoch's keys: `count` tiles from `first` on.
fn keys(first: u32, count: u32) -> BTreeSet<TileKey> {
    (first..first + count).map(key).collect()
}

/// Every epoch reaches each consumer once, whole and in order, or the consumer resyncs.
///
/// A model of random closes and drains by a fast, a slow and a stalled consumer: a consumer that
/// falls past the cap gets one `Resync` and goes on from there; no consumer ever holds more than
/// the cap.
#[test]
fn epochs_arrive_once_whole_and_in_order_or_as_a_resync() {
    bolero::check!()
        .with_type::<(u64, Vec<(u8, u8)>)>()
        .for_each(|(cap, ops)| {
            let cap = usize::try_from(cap % 40 + 1).unwrap();
            let mut feed = StaleFeed::new(cap);
            let cursors = [feed.subscribe(), feed.subscribe(), feed.subscribe()];
            let mut closed: Vec<Epoch> = Vec::new();
            // Per consumer: the next epoch it should see, if it hasn't been told to resync since.
            let mut next = [0_u64; 3];
            for &(op, size) in ops {
                let count = u32::from(size % 8);
                feed.close(keys(u32::from(op), count));
                if count > 0 {
                    closed.push(Epoch {
                        seq: u64::try_from(closed.len()).unwrap(),
                        keys: keys(u32::from(op), count),
                    });
                }
                assert!(feed.most_pending() <= cap, "memory stays within the cap");
                for (n, cursor) in cursors.iter().enumerate() {
                    // The fast drains every time, the slow every fourth op, the stalled never.
                    let drains = match n {
                        0 => true,
                        1 => op % 4 == 0,
                        _ => false,
                    };
                    if !drains {
                        continue;
                    }
                    match feed.drain(cursor) {
                        Drained::Epochs(epochs) => {
                            for epoch in epochs {
                                let want = &closed[usize::try_from(epoch.seq).unwrap()];
                                assert_eq!(&epoch, want, "consumer {n}: whole");
                                assert!(epoch.seq >= next[n], "consumer {n}: in order, once");
                                next[n] = epoch.seq + 1;
                            }
                        }
                        Drained::Resync => {
                            next[n] = u64::try_from(closed.len()).unwrap();
                        }
                    }
                }
            }
            // The stalled consumer: everything it was owed, or a resync if that was past the cap.
            let owed: usize = closed.iter().map(|e| e.keys.len()).sum();
            match feed.drain(&cursors[2]) {
                Drained::Epochs(epochs) => {
                    assert!(owed <= cap, "kept only within the cap");
                    assert_eq!(epochs, closed, "the stalled consumer gets every epoch");
                }
                Drained::Resync => assert!(owed > cap, "told to resync only past the cap"),
            }
        });
}

#[test]
fn an_unsubscribed_cursor_is_told_to_resync() {
    let mut feed = StaleFeed::new(10);
    let cursor = feed.subscribe();
    assert!(feed.is_listened(), "listened to");
    let kept = StaleCursor(cursor.0);
    feed.unsubscribe(cursor);
    assert!(!feed.is_listened(), "no one listens");
    feed.close(keys(0, 3));
    assert_eq!(feed.drain(&kept), Drained::Resync, "gone");
}
