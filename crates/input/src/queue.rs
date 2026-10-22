//! The bounded hand-off from a backend's thread to the consumer.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};

/// A bounded queue: `capacity` events before the producer starts dropping.
pub(crate) fn channel<T>(capacity: usize) -> (Sender<T>, Events<T>) {
    let (tx, rx) = mpsc::sync_channel(capacity);
    let dropped = Arc::new(AtomicU64::new(0));
    (
        Sender {
            tx,
            dropped: Arc::clone(&dropped),
        },
        Events { rx, dropped },
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
    /// Events dropped because the queue was full, shared with the [`Events`] end.
    dropped: Arc<AtomicU64>,
}

impl<T> Sender<T> {
    /// Queues `event`, or drops and counts it if the queue is full.
    ///
    /// # Errors
    ///
    /// [`Disconnected`] once the [`Events`] end is dropped.
    pub(crate) fn send(&self, event: T) -> Result<(), Disconnected> {
        match self.tx.try_send(event) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(_)) => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            Err(TrySendError::Disconnected(_)) => Err(Disconnected),
        }
    }
}

/// Events from a backend, in the order they happened.
#[derive(Debug)]
pub struct Events<T> {
    /// The bounded channel's receiving end.
    rx: Receiver<T>,
    /// Events the producer dropped because the queue was full.
    dropped: Arc<AtomicU64>,
}

impl<T> Events<T> {
    /// Takes every event queued so far, without waiting.
    pub fn try_iter(&self) -> impl Iterator<Item = T> + '_ {
        self.rx.try_iter()
    }

    /// Events dropped because the queue was full: the consumer fell behind.
    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "test code may panic")]
mod tests {
    use super::*;

    #[test]
    fn delivers_in_order() {
        let (tx, events) = channel(8);
        for i in 0..5 {
            tx.send(i).expect("receiver alive");
        }
        assert_eq!(events.try_iter().collect::<Vec<_>>(), [0, 1, 2, 3, 4]);
        assert_eq!(events.dropped(), 0);
    }

    #[test]
    fn full_queue_drops_and_counts() {
        let (tx, events) = channel(2);
        for i in 0..5 {
            tx.send(i).expect("receiver alive");
        }
        assert_eq!(events.try_iter().collect::<Vec<_>>(), [0, 1]);
        assert_eq!(events.dropped(), 3);
    }

    #[test]
    fn send_fails_once_the_receiver_is_gone() {
        let (tx, events) = channel(2);
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
