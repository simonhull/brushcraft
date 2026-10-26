//! The display records and readers, tested through the store in `tiles::tests::display` and
//! the file harnesses in `file::fuzz`; this holds the unit tests of the pieces alone.

use super::*;

/// A display record header round-trips its state and bound.
#[test]
fn headers_hold_the_state_and_the_bound() {
    let record = DisplayRecord {
        error: 0.25,
        encoded: Vec::new(),
    };
    assert_eq!(display_header(None), [VERSION, UNKNOWN], "unknown");
    let known = display_header(Some(&record));
    assert_eq!(known[..2], [VERSION, KNOWN], "known");
    assert_eq!(
        f32::from_le_bytes(known[2..].try_into().unwrap()),
        0.25,
        "the bound"
    );
    assert!(
        display_body(None).is_empty(),
        "an unknown record has no body"
    );
}
