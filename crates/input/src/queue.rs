//! The bounded hand-off from a backend's thread to the consumer.
//!
//! The queue keeps a reserve for events that must not be lost: once all but [`RESERVE`] slots are
//! taken, [droppable](Droppable) events (motion that a later sample supersedes) are dropped, so
//! the state changes that frame them (a stroke's start and end, a button release) still fit.
//!
//! The occupancy count only steers that dropping, so its atomics use [`Ordering::Relaxed`]: no
//! other memory is published through it, and the channel itself orders and delivers the events.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};

/// Slots kept for events that are not [droppable](Droppable): once fewer than this are free,
/// droppable events are dropped.
pub(crate) const RESERVE: usize = 64;

/// Whether an event may be dropped when the consumer falls behind.
pub(crate) trait Droppable {
    /// True for samples a later one supersedes (motion); false for state changes.
    fn droppable(&self) -> bool;
}

/// A bounded queue: `capacity` events, of which the last [`RESERVE`] are kept for events that
/// are not [droppable](Droppable). A capacity under [`RESERVE`] drops every droppable event.
pub(crate) fn channel<T>(capacity: usize) -> (Sender<T>, Events<T>) {
    let (tx, rx) = mpsc::sync_channel(capacity);
    let queued = Arc::new(AtomicUsize::new(0));
    let dropped = Arc::new(AtomicU64::new(0));
    let lost_essential = Arc::new(AtomicU64::new(0));
    (
        Sender {
            tx,
            droppable_limit: capacity.saturating_sub(RESERVE),
            queued: Arc::clone(&queued),
            dropped: Arc::clone(&dropped),
            lost_essential: Arc::clone(&lost_essential),
        },
        Events {
            rx,
            queued,
            dropped,
            lost_essential,
        },
    )
}

/// The consumer has gone; the producer should stop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("the event consumer is gone")]
pub(crate) struct Disconnected;

/// The producer's end. Never blocks.
#[derive(Debug)]
pub(crate) struct Sender<T> {
    /// The bounded channel's sending end.
    tx: SyncSender<T>,
    /// Queued events at or above which droppable events are dropped: the capacity less
    /// [`RESERVE`].
    droppable_limit: usize,
    /// Events sent and not yet taken by the consumer, shared with the [`Events`] end. Counted
    /// before the send and taken back if it fails, so it never undercounts what the channel
    /// holds and never goes below zero.
    queued: Arc<AtomicUsize>,
    /// Droppable events dropped because the queue was past its reserve, shared with the
    /// [`Events`] end.
    dropped: Arc<AtomicU64>,
    /// Essential events lost because even the reserve was full, shared with the [`Events`] end.
    lost_essential: Arc<AtomicU64>,
}

impl<T: Droppable> Sender<T> {
    /// Queues `event`. A droppable event is dropped and counted once the queue is past its
    /// reserve; any other event is dropped, and counted apart, only when the whole queue is full.
    ///
    /// # Errors
    ///
    /// [`Disconnected`] once the [`Events`] end is dropped.
    pub(crate) fn send(&self, event: T) -> Result<(), Disconnected> {
        let droppable = event.droppable();
        if droppable && self.queued.load(Ordering::Relaxed) >= self.droppable_limit {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }
        self.queued.fetch_add(1, Ordering::Relaxed);
        let full = match self.tx.try_send(event) {
            Ok(()) => return Ok(()),
            Err(TrySendError::Full(_)) => true,
            Err(TrySendError::Disconnected(_)) => false,
        };
        self.queued.fetch_sub(1, Ordering::Relaxed);
        if !full {
            return Err(Disconnected);
        }
        // Droppable events stop at the reserve and the count never undercounts the channel, so
        // only an essential event finds the channel itself full.
        let counter = if droppable {
            &self.dropped
        } else {
            &self.lost_essential
        };
        counter.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

/// Events from a backend, in the order they happened.
#[derive(Debug)]
pub struct Events<T> {
    /// The bounded channel's receiving end.
    rx: Receiver<T>,
    /// Events sent and not yet taken, shared with the producer, which drops motion by it.
    queued: Arc<AtomicUsize>,
    /// Events the producer dropped because the queue was past its reserve.
    dropped: Arc<AtomicU64>,
    /// Essential events the producer lost because even the reserve was full.
    lost_essential: Arc<AtomicU64>,
}

impl<T> Events<T> {
    /// Takes every event queued so far, without waiting.
    pub fn try_iter(&self) -> impl Iterator<Item = T> + '_ {
        self.rx.try_iter().inspect(|_| {
            self.queued.fetch_sub(1, Ordering::Relaxed);
        })
    }

    /// Events dropped because the queue was full: the consumer fell behind. Only motion (pen
    /// samples between a stroke's phase changes, ring and strip movement) is dropped
    /// this way, keeping room for the state changes around it.
    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// Phase changes or pad state changes lost because even the reserve was full: the stream is
    /// no longer well formed until the next `Enter`.
    #[must_use]
    pub fn lost_essential(&self) -> u64 {
        self.lost_essential.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "test code may panic")]
mod tests {
    use super::*;

    /// Plain numbers stand for motion: droppable.
    impl Droppable for i32 {
        fn droppable(&self) -> bool {
            true
        }
    }

    #[test]
    fn delivers_in_order() {
        let (tx, events) = channel(RESERVE + 8);
        for i in 0..5 {
            tx.send(i).expect("receiver alive");
        }
        assert_eq!(events.try_iter().collect::<Vec<_>>(), [0, 1, 2, 3, 4]);
        assert_eq!(events.dropped(), 0);
    }

    #[test]
    fn full_queue_drops_and_counts() {
        let (tx, events) = channel(RESERVE + 2);
        for i in 0..5 {
            tx.send(i).expect("receiver alive");
        }
        assert_eq!(events.try_iter().collect::<Vec<_>>(), [0, 1]);
        assert_eq!(events.dropped(), 3);
    }

    /// A test event: an id, and whether it is droppable motion.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct Ev(u32, bool);

    impl Droppable for Ev {
        fn droppable(&self) -> bool {
            self.1
        }
    }

    /// Room for 6 droppable events before the reserve.
    const SMALL: usize = RESERVE + 6;

    #[test]
    fn a_full_queue_keeps_room_for_essential_events() {
        let (tx, events) = channel(SMALL);
        for i in 0..100 {
            tx.send(Ev(i, true)).expect("receiver alive");
        }
        for i in 100..103 {
            tx.send(Ev(i, false)).expect("receiver alive");
        }
        let expected: Vec<Ev> = (0..6)
            .map(|i| Ev(i, true))
            .chain((100..103).map(|i| Ev(i, false)))
            .collect();
        assert_eq!(events.try_iter().collect::<Vec<_>>(), expected);
        assert_eq!(events.dropped(), 94);
        assert_eq!(events.lost_essential(), 0);
    }

    #[test]
    fn essential_events_past_the_reserve_are_counted_apart() {
        let (tx, events) = channel(SMALL);
        for i in 0..75 {
            tx.send(Ev(i, false)).expect("receiver alive");
        }
        let delivered: Vec<Ev> = events.try_iter().collect();
        assert_eq!(delivered, (0..70).map(|i| Ev(i, false)).collect::<Vec<_>>());
        assert_eq!(events.lost_essential(), 5);
        assert_eq!(events.dropped(), 0);
    }

    #[test]
    fn draining_frees_room() {
        let (tx, events) = channel(SMALL);
        for i in 0..10 {
            tx.send(Ev(i, true)).expect("receiver alive");
        }
        assert_eq!(events.try_iter().count(), 6);
        assert_eq!(events.dropped(), 4);
        for i in 10..16 {
            tx.send(Ev(i, true)).expect("receiver alive");
        }
        assert_eq!(
            events.try_iter().collect::<Vec<_>>(),
            (10..16).map(|i| Ev(i, true)).collect::<Vec<_>>(),
            "the drained room takes droppable events again"
        );
        assert_eq!(events.dropped(), 4);
    }

    #[test]
    fn send_fails_once_the_receiver_is_gone() {
        let (tx, events) = channel(RESERVE + 2);
        drop(events);
        assert_eq!(tx.send(1), Err(Disconnected));
    }

    #[test]
    fn crosses_threads() {
        let (tx, events) = channel(20_000);
        let producer = std::thread::spawn(move || {
            for i in 0..10_000 {
                tx.send(i).expect("receiver alive");
            }
        });
        producer.join().expect("producer finished");
        assert_eq!(events.try_iter().count(), 10_000);
    }
}
