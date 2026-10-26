//! Reading the spatial index's bound records: one record, or many in few large reads.

use std::fs::File;
use std::io;

use geom::TileKey;

use super::Loc;
use super::format::read_exact;
use crate::bounds::{StoredBound, parse_bounds_payload};
use crate::record::{KIND_BOUNDS, verify};
use crate::tile_bounds::TileBounds;

/// The bound in a raw record, if the record verifies and holds a bound of `key`.
pub(super) fn bounds_record(raw: &[u8], key: &TileKey) -> Option<StoredBound> {
    match verify(raw) {
        Some((KIND_BOUNDS, payload)) => parse_bounds_payload(payload),
        _ => None,
    }
    .filter(|(at, _)| at == key)
    .map(|(_, kept)| kept)
}

/// The bounds the records at `entries` hold: for each, the record's bounds, or `None` if they
/// are unknown or the record doesn't verify as a bounds record of its key.
///
/// A bound is derived data: one that doesn't read counts as unknown, and is computed again by
/// whoever wanted it, so damage here is not an error. The records are read in file order, nearby
/// ones together, in windows of up to [`BOUNDS_WINDOW`] bytes: a project's bounds are written a
/// save at a time, so a whole level's usually take a few large reads, not one per tile.
///
/// # Errors
///
/// The I/O error of a read, but for a read past the end, which only damage causes.
pub(super) fn read_bounds_at(
    file: &File,
    entries: &[(TileKey, Loc)],
) -> io::Result<Vec<Option<TileBounds>>> {
    let mut found = vec![None; entries.len()];
    let mut order: Vec<usize> = (0..entries.len()).collect();
    order.sort_unstable_by_key(|&n| entries.get(n).map(|(_, loc)| loc.offset));
    let mut rest = order.as_slice();
    while let Some((&first, _)) = rest.split_first() {
        let Some(&(_, start)) = entries.get(first) else {
            break;
        };
        // The window: every record after the first that starts close enough and ends within it.
        let mut end = start.offset.saturating_add(start.len);
        let mut taken = 1;
        for &n in rest.iter().skip(1) {
            let Some(&(_, loc)) = entries.get(n) else {
                break;
            };
            let until = loc.offset.saturating_add(loc.len);
            if loc.offset > end.saturating_add(BOUNDS_GAP) || until - start.offset > BOUNDS_WINDOW {
                break;
            }
            end = end.max(until);
            taken += 1;
        }
        let (group, after) = rest.split_at(taken);
        rest = after;
        let window = match read_exact(file, start.offset, end - start.offset) {
            Ok(window) => window,
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                // Some record runs past the end: read each on its own.
                for &n in group {
                    if let (Some(&(key, loc)), Some(slot)) = (entries.get(n), found.get_mut(n)) {
                        *slot = match read_exact(file, loc.offset, loc.len) {
                            Ok(raw) => bounds_record(&raw, &key).and_then(StoredBound::bounds),
                            Err(short) if short.kind() == io::ErrorKind::UnexpectedEof => None,
                            Err(failed) => return Err(failed),
                        };
                    }
                }
                continue;
            }
            Err(e) => return Err(e),
        };
        for &n in group {
            if let (Some(&(key, loc)), Some(slot)) = (entries.get(n), found.get_mut(n)) {
                let raw = usize::try_from(loc.offset - start.offset)
                    .ok()
                    .zip(usize::try_from(loc.len).ok())
                    .and_then(|(lo, len)| window.get(lo..lo.checked_add(len)?));
                *slot = raw
                    .and_then(|raw| bounds_record(raw, &key))
                    .and_then(StoredBound::bounds);
            }
        }
    }
    Ok(found)
}

/// The most bytes [`read_bounds_at`] reads at once.
const BOUNDS_WINDOW: u64 = 4 << 20;

/// The widest gap between two bounds records that [`read_bounds_at`] reads through rather than
/// starting a new read.
const BOUNDS_GAP: u64 = 64 << 10;
