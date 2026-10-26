//! The bounds record's encoding, the walk from written tiles to the bounds they change, and the
//! book's bookkeeping.

use std::collections::BTreeMap;

use geom::{BaseMesh, Vec3, normal_tiles};

use super::*;
use crate::fixture::{level, mesh, split_prism, tiles};
use crate::tile_bounds::{Aabb, NormalCone};

/// Bounds with every kind of value a tile's bounds can hold: a negative zero, a tight cone.
fn sample() -> TileBounds {
    TileBounds {
        aabb: Aabb::new(Vec3::new(-0.0, -1.5, 2.0), Vec3::new(0.0, 1e-7, 2.5)).unwrap(),
        cone: NormalCone::from_normals(&[Vec3::new(0.0, 0.6, 0.8), Vec3::new(0.0, 0.8, 0.6)]),
    }
}

#[test]
fn a_bound_round_trips_bit_for_bit() {
    let key = tiles(&mesh(), 8, 8)[17];
    let everything = TileBounds {
        cone: NormalCone::EVERYTHING,
        ..sample()
    };
    for bounds in [Some(sample()), Some(everything), None] {
        let payload = bounds_payload(&key, bounds.as_ref());
        let (got_key, got) = parse_bounds_payload(&payload).unwrap();
        assert_eq!(got_key, key, "the key");
        assert_eq!(
            got.bounds().map(TileBounds::to_bits),
            bounds.map(TileBounds::to_bits),
            "the bits"
        );
    }
    assert_eq!(
        bounds_payload(&key, Some(&sample())).len(),
        KEY_LEN + 1 + VALUES_LEN,
        "a known bound"
    );
    assert_eq!(
        bounds_payload(&key, None).len(),
        KEY_LEN + 1,
        "an unknown one"
    );
}

#[test]
fn a_bound_that_is_not_one_does_not_parse() {
    let key = tiles(&mesh(), 8, 8)[3];
    let known = bounds_payload(&key, Some(&sample()));
    let unknown = bounds_payload(&key, None);
    let with = |payload: &[u8], at: usize, value: f32| {
        let mut bad = payload.to_vec();
        let at = KEY_LEN + 1 + 4 * at;
        bad[at..at + 4].copy_from_slice(&value.to_le_bytes());
        bad
    };
    let mut bad_state = unknown.clone();
    bad_state[KEY_LEN] = 2;
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("short", known[..known.len() - 1].to_vec()),
        ("trailing", [known.as_slice(), &[0]].concat()),
        (
            "unknown with values",
            [unknown.as_slice(), &known[KEY_LEN + 1..]].concat(),
        ),
        ("a state that is neither", bad_state),
        ("min above max", with(&known, 0, 9.0)),
        ("not finite", with(&known, 4, f32::INFINITY)),
        ("an axis that isn't a unit", with(&known, 8, 0.0)),
        ("a cosine past 1", with(&known, 9, 1.5)),
        ("a NaN cosine", with(&known, 9, f32::NAN)),
    ];
    for (what, payload) in cases {
        assert_eq!(parse_bounds_payload(&payload), None, "{what}");
    }
}

/// Every tile of `level` with the tiles [`normal_tiles`] lists for it.
fn neighbourhoods(mesh: &BaseMesh, l: u8) -> BTreeMap<TileKey, BTreeSet<TileKey>> {
    tiles(mesh, l, l)
        .into_iter()
        .map(|key| (key, normal_tiles(mesh, Region::of_tile(&key)).unwrap()))
        .collect()
}

/// The meshes the neighbourhood tests walk: valence 3 (the cube), 5 (the subdivided pentagonal
/// prism) and 7 (the split heptagonal prism).
fn meshes() -> Vec<BaseMesh> {
    vec![
        (*mesh()).clone(),
        geom::fixtures::subdivided_pentagonal_prism().unwrap(),
        split_prism(7),
    ]
}

/// Readers are found among neighbours: a tile whose normals read `s` is one of `s`'s normal
/// tiles, and a tile predicted from `s` subdivides one of them. [`readers`] relies on both.
#[test]
fn every_reader_of_a_tile_is_among_its_neighbours() {
    for mesh in meshes() {
        for l in 1..=8 {
            let around = neighbourhoods(&mesh, l);
            for (t, reads) in &around {
                for s in reads {
                    assert!(
                        around[s].contains(t),
                        "level {l}: {t:?}'s normals read {s:?}, which doesn't list it"
                    );
                }
            }
            if l == 8 {
                continue;
            }
            let next = level(l + 1);
            for t in tiles(&mesh, l + 1, l + 1) {
                for s in parent_tiles(&mesh, Region::of_tile(&t)).unwrap() {
                    let found = around[&s]
                        .iter()
                        .any(|n| children(n, next).any(|child| child == t));
                    assert!(
                        found,
                        "level {l}: {t:?} is predicted from {s:?}, far from it"
                    );
                }
            }
        }
    }
}

/// The bounds a write of `edited` changes at `finest`, by brute force: every tile of every level
/// whose parents moved, and at `finest` every tile reading a moved tile's positions.
fn brute_readers(mesh: &BaseMesh, edited: &BTreeSet<TileKey>, finest: u8) -> BTreeSet<TileKey> {
    let mut moved: BTreeSet<TileKey> = BTreeSet::new();
    for l in 1..=finest {
        let at: BTreeSet<TileKey> = tiles(mesh, l, l)
            .into_iter()
            .filter(|t| {
                edited.contains(t)
                    || (l > 1
                        && parent_tiles(mesh, Region::of_tile(t))
                            .unwrap()
                            .iter()
                            .any(|p| moved.contains(p)))
            })
            .collect();
        moved = at;
    }
    let mut answer = moved.clone();
    for (t, reads) in neighbourhoods(mesh, finest) {
        if reads.iter().any(|s| moved.contains(s)) {
            answer.insert(t);
        }
    }
    answer
}

#[test]
fn readers_hold_every_bound_a_write_changes() {
    for mesh in meshes() {
        let all = tiles(&mesh, 1, 8);
        let levels = BTreeSet::from([level(8)]);
        for (n, key) in all.iter().enumerate().step_by(7) {
            let edited = BTreeSet::from([*key, all[(n * 31 + 5) % all.len()]]);
            let got = readers(&mesh, &DepGraph::new(), &edited, &levels).unwrap();
            let want = brute_readers(&mesh, &edited, 8);
            assert!(
                want.is_subset(&got),
                "{edited:?}: missed {:?}",
                want.difference(&got).collect::<Vec<_>>()
            );
            assert!(
                got.iter().all(|k| k.level() == level(8)),
                "the tracked level only"
            );
        }
    }
}

#[test]
fn a_finest_write_reaches_its_neighbourhood_and_nothing_else_does() {
    let mesh = mesh();
    let s = tiles(&mesh, 8, 8)[9];
    let levels = BTreeSet::from([level(8)]);
    let got = readers(&mesh, &DepGraph::new(), &BTreeSet::from([s]), &levels).unwrap();
    let mut want = normal_tiles(&mesh, Region::of_tile(&s)).unwrap();
    want.insert(s);
    assert_eq!(got, want, "the tile and its neighbours");
    let finer = tiles(&mesh, 8, 8)[0];
    let coarse_only = BTreeSet::from([level(7)]);
    assert!(
        readers(
            &mesh,
            &DepGraph::new(),
            &BTreeSet::from([finer]),
            &coarse_only
        )
        .unwrap()
        .is_empty(),
        "a finer write changes no coarser bound"
    );
    let other = TileKey::new(
        LayerId::new(1),
        Channel::DISPLACEMENT,
        s.face(),
        s.level(),
        0,
        0,
    )
    .unwrap();
    assert!(
        readers(&mesh, &DepGraph::new(), &BTreeSet::from([other]), &levels)
            .unwrap()
            .is_empty(),
        "another layer changes nothing evaluated"
    );
    assert!(
        readers(
            &mesh,
            &DepGraph::new(),
            &BTreeSet::from([s]),
            &BTreeSet::new()
        )
        .unwrap()
        .is_empty(),
        "nothing tracked"
    );
}

#[test]
fn the_book_makes_unknown_what_a_write_changes() {
    let mesh = mesh();
    let all = tiles(&mesh, 8, 8);
    let mut book = Book::default();
    book.note_write(all[0]);
    assert!(!book.has_unsettled(), "nothing tracked, nothing noted");
    for key in &all {
        book.set(*key, Some(sample()));
    }
    for key in &all {
        book.flushed(key);
    }
    let changes = book.changes();
    book.note_write(all[9]);
    assert!(book.has_unsettled(), "noted");
    book.settle(&mesh, &DepGraph::new()).unwrap();
    let stale = readers(
        &mesh,
        &DepGraph::new(),
        &BTreeSet::from([all[9]]),
        &BTreeSet::from([level(8)]),
    )
    .unwrap();
    for key in &all {
        let want = if stale.contains(key) {
            None
        } else {
            Some(sample())
        };
        assert_eq!(
            book.changed(key).unwrap().bounds().map(TileBounds::to_bits),
            want.map(TileBounds::to_bits),
            "{key:?}"
        );
    }
    let unflushed: BTreeSet<TileKey> = book.unflushed().map(|(key, _)| *key).collect();
    assert_eq!(unflushed, stale, "the unknown ones are to flush");
    assert!(book.changes() > changes, "counted");
    let forgotten = book.forget(&mesh, &DepGraph::new(), [all[20]]).unwrap();
    assert!(forgotten.contains(&all[20]), "a lost tile's own bound");
    assert_eq!(
        book.changed(&all[20]),
        Some(StoredBound::Unknown),
        "unknown"
    );
    book.rebased();
    assert!(book.changed(&all[0]).is_none(), "a rebase empties the book");
    assert!(!book.has_unflushed(), "and the flush list");
}

#[test]
fn a_recorded_bound_is_flushed_only_over_an_edits_change() {
    let mesh = mesh();
    let all = tiles(&mesh, 8, 8);
    let mut book = Book::default();
    book.record(all[0], sample());
    assert!(!book.has_unflushed(), "derived: no flush writes it");
    assert_eq!(
        book.unsaved().map(|(key, _)| *key).collect::<Vec<_>>(),
        [all[0]],
        "a save does"
    );
    book.set(all[1], None);
    book.record(all[1], sample());
    assert!(
        book.is_unflushed(&all[1]),
        "recorded over an edit's unknown marker: flushed, as the marker would have been"
    );
    book.saved(&all[0]);
    book.saved(&all[1]);
    assert_eq!(book.unsaved().count(), 0, "both saved");
}

/// Settling and recording in one pass leaves the book as a settle followed by a record of each
/// bound does: the same values, the same flush list and the same derived bounds.
///
/// The recorded bounds include tiles the settle makes unknown (recorded over them, to be
/// flushed) and tiles it doesn't (derived); the stale tiles left unrecorded stay unknown.
#[test]
fn settling_while_recording_is_settling_then_recording() {
    let mesh = mesh();
    let all = tiles(&mesh, 8, 8);
    let ready = || {
        let mut book = Book::default();
        for key in &all {
            book.set(*key, Some(sample()));
        }
        for key in &all {
            book.flushed(key);
        }
        book.note_write(all[9]);
        book.note_write(all[40]);
        book
    };
    let stale = readers(
        &mesh,
        &DepGraph::new(),
        &BTreeSet::from([all[9], all[40]]),
        &BTreeSet::from([level(8)]),
    )
    .unwrap();
    let fresh = TileBounds {
        aabb: Aabb::new(Vec3::new(1.0, 1.0, 1.0), Vec3::new(2.0, 2.0, 2.0)).unwrap(),
        ..sample()
    };
    let recorded: Vec<(TileKey, TileBounds)> = stale
        .iter()
        .take(stale.len() / 2)
        .chain(all.iter().filter(|key| !stale.contains(key)).take(5))
        .map(|&key| (key, fresh))
        .collect();
    assert!(recorded.len() > 5, "some of each");

    let mut apart = ready();
    apart.settle(&mesh, &DepGraph::new()).unwrap();
    for &(key, bounds) in &recorded {
        apart.record(key, bounds);
    }
    let mut together = ready();
    together
        .settle_and_record(&mesh, &DepGraph::new(), &recorded)
        .unwrap();

    let contents = |book: &Book| {
        let bits = |kept: Option<&TileBounds>| kept.map(|b| b.to_bits());
        let changed: Vec<_> = book.all_changed().map(|(k, b)| (*k, bits(b))).collect();
        let unflushed: Vec<_> = book.unflushed().map(|(k, b)| (*k, bits(b))).collect();
        let unsaved: Vec<_> = book.unsaved().map(|(k, b)| (*k, bits(b))).collect();
        (changed, unflushed, unsaved, book.has_unsettled())
    };
    assert_eq!(contents(&together), contents(&apart), "the same book");
    for key in stale.iter().skip(stale.len() / 2) {
        assert_eq!(
            together.changed(key),
            Some(StoredBound::Unknown),
            "{key:?} stays unknown"
        );
    }
}
