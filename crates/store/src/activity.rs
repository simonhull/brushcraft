//! What a tile store has done: counts of the work behind its evaluations and of the memory
//! policy's moves, for telling what made one call slower than another.

/// Counts of what a [`crate::TileStore`] has done since it opened, as
/// [`crate::TileStore::activity`] reports them. Every counter only grows.
///
/// The counts are kept on the thread that owns the store, as the work is installed, so reading
/// them costs a copy and the work counted on the workers is counted once, whichever ran it.
/// Take one before a call and one after, and [`Activity::since`] gives what the call did: a
/// brush step that read tiles from the file, say, or evicted tiles to make room.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct Activity {
    /// Correction tiles given a slot and loaded into it: read from the working file or the base,
    /// or set to zero for a tile never written. Tiles the prefetch workers read aren't counted.
    pub corrections_loaded: u64,
    /// Tiles whose positions were computed: predicted from the level below and corrected.
    pub positions_computed: u64,
    /// Of those, the tiles predicted again in place from the prediction the store kept for
    /// them, only their corrections applied (see "Kept predictions" on [`crate::TileStore`]).
    pub predictions_reused: u64,
    /// Tiles whose normals were computed, in full or (where they were stale) only for the
    /// vertices that read a changed tile.
    pub normals_computed: u64,
    /// Correction tiles the pool freed on its own: to make room (each written to the working
    /// file first if it was dirty), or emptying it ([`crate::TileStore::evict_all`],
    /// [`crate::TileStore::shrink_to`]). Tiles an edit or a rebase removes aren't counted.
    pub corrections_evicted: u64,
    /// Evaluated tiles the pool freed on its own, as for
    /// [`Activity::corrections_evicted`]; not those an edit invalidated.
    pub evaluated_evicted: u64,
}

impl Activity {
    /// What was done between `earlier` and this count: each counter's difference, or zero
    /// where `earlier` is the later of the two.
    #[must_use]
    pub const fn since(&self, earlier: &Self) -> Self {
        Self {
            corrections_loaded: self
                .corrections_loaded
                .saturating_sub(earlier.corrections_loaded),
            positions_computed: self
                .positions_computed
                .saturating_sub(earlier.positions_computed),
            predictions_reused: self
                .predictions_reused
                .saturating_sub(earlier.predictions_reused),
            normals_computed: self
                .normals_computed
                .saturating_sub(earlier.normals_computed),
            corrections_evicted: self
                .corrections_evicted
                .saturating_sub(earlier.corrections_evicted),
            evaluated_evicted: self
                .evaluated_evicted
                .saturating_sub(earlier.evaluated_evicted),
        }
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    #[test]
    fn since_takes_each_counter_apart() {
        let earlier = Activity {
            corrections_loaded: 1,
            positions_computed: 2,
            predictions_reused: 6,
            normals_computed: 3,
            corrections_evicted: 4,
            evaluated_evicted: 5,
        };
        let later = Activity {
            corrections_loaded: 11,
            positions_computed: 22,
            predictions_reused: 66,
            normals_computed: 33,
            corrections_evicted: 44,
            evaluated_evicted: 55,
        };
        assert_eq!(
            later.since(&earlier),
            Activity {
                corrections_loaded: 10,
                positions_computed: 20,
                predictions_reused: 60,
                normals_computed: 30,
                corrections_evicted: 40,
                evaluated_evicted: 50,
            },
            "each counter's own difference"
        );
        assert_eq!(
            earlier.since(&later),
            Activity::default(),
            "nothing, not a wrap, the wrong way round"
        );
    }
}
