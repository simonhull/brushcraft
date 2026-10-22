//! Edit ids: which site made an edit, and its sequence number there.

/// Largest sequence number (48 bits).
pub const MAX_SEQ: u64 = (1 << 48) - 1;

/// The site no op may have. Reserving it keeps `u64::MAX`, the file format's "no op", from ever
/// being a real op's bits.
pub const RESERVED_SITE: u16 = u16::MAX;

/// The id of one edit: the site that made it, and its sequence number there.
///
/// The site is the machine or session, any but [`RESERVED_SITE`]; the sequence number has 48
/// bits. M0 uses one site; the site is in the format so streams from several machines can merge
/// later. Ordered by site, then sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OpId {
    site: u16,
    seq: u64,
}

impl OpId {
    /// An op id, if `site` isn't [`RESERVED_SITE`] and `seq` fits in 48 bits.
    #[must_use]
    pub const fn new(site: u16, seq: u64) -> Option<Self> {
        if site != RESERVED_SITE && seq <= MAX_SEQ {
            Some(Self { site, seq })
        } else {
            None
        }
    }

    /// The site.
    #[must_use]
    pub const fn site(self) -> u16 {
        self.site
    }

    /// The sequence number.
    #[must_use]
    pub const fn seq(self) -> u64 {
        self.seq
    }

    /// The next op id of the same site, or `None` past [`MAX_SEQ`].
    #[must_use]
    pub const fn next(self) -> Option<Self> {
        // No overflow: `seq <= MAX_SEQ < u64::MAX`.
        Self::new(self.site, self.seq + 1)
    }

    /// Packed as `site << 48 | seq`, the file format's encoding.
    #[must_use]
    pub fn to_bits(self) -> u64 {
        (u64::from(self.site) << 48) | self.seq
    }

    /// Unpacks [`OpId::to_bits`], or `None` for a packing of [`RESERVED_SITE`].
    #[must_use]
    pub const fn from_bits(bits: u64) -> Option<Self> {
        let [.., hi, lo] = (bits >> 48).to_be_bytes();
        Self::new(u16::from_be_bytes([hi, lo]), bits & MAX_SEQ)
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    #[test]
    fn op_ids_round_trip_through_bits() {
        let op = OpId::new(7, 123_456).unwrap();
        assert_eq!(OpId::from_bits(op.to_bits()), Some(op), "round trip");
        assert!(OpId::new(0, MAX_SEQ + 1).is_none(), "48-bit sequence");
        assert_eq!(
            OpId::new(1, MAX_SEQ).unwrap().next(),
            None,
            "no successor past the max"
        );
        assert!(
            OpId::new(0, 5).unwrap() < OpId::new(0, 6).unwrap(),
            "ordered by sequence within a site"
        );
    }

    #[test]
    fn the_reserved_site_has_no_op_ids() {
        assert_eq!(OpId::new(RESERVED_SITE, 0), None, "reserved site");
        assert_eq!(OpId::new(RESERVED_SITE, MAX_SEQ), None, "at any sequence");
        assert_eq!(
            OpId::from_bits(u64::MAX),
            None,
            "so u64::MAX is never an op"
        );
        let last = OpId::new(RESERVED_SITE - 1, MAX_SEQ).unwrap();
        assert_eq!(
            OpId::from_bits(last.to_bits()),
            Some(last),
            "the site below is usable"
        );
        assert!(last.to_bits() < u64::MAX, "and packs below u64::MAX");
    }
}
