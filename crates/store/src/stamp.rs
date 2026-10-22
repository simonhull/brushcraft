//! Write stamps: a tile store's own order of the writes its correction slots take.
//!
//! An op names an edit as the user sees it, a stroke say, and one stroke keeps one op across
//! many dabs. So an op can't tell two writes of one stroke apart, and the store needs to: a
//! recovery flush during a long stroke snapshots some of its dabs and not the rest, and must
//! then know which records and which slots belong to the snapshot. Every write a correction slot
//! takes is stamped with the next value of a counter, and the stamps decide.

/// A write's place in a tile store's order of writes: every write a correction slot takes is
/// stamped with a value greater than every earlier one's.
///
/// Stamps live in memory only, never in a file. A record written from no slot (a copy of
/// another file's record, or a write straight to a [`crate::ProjectFile`]) carries
/// [`Stamp::ZERO`], before every write of the store.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct Stamp(u64);

impl Stamp {
    /// Before every write: what the store's counter starts at, and what records made from no
    /// slot carry.
    pub(crate) const ZERO: Self = Self(0);

    /// The stamp after this one.
    ///
    /// It saturates at the last one, which no store reaches: that takes a write every
    /// nanosecond for over five centuries.
    #[must_use]
    pub(crate) const fn next(self) -> Self {
        Self(self.0.saturating_add(1))
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    #[test]
    fn stamps_grow() {
        let first = Stamp::ZERO.next();
        assert!(first > Stamp::ZERO, "after zero");
        assert!(first.next() > first, "and after each other");
        assert_eq!(Stamp::default(), Stamp::ZERO, "the counter starts at zero");
    }
}
