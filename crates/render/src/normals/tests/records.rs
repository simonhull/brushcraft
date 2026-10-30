//! The normal pass on records built by hand, outside residency: its record chunks, a slot
//! written with vertices after a record in the same frame, and the fallback and NaN normals of
//! degenerate and non-finite inputs.

use std::sync::mpsc;

use wgpu::CommandEncoderDescriptor;

use super::super::*;
use crate::cull::storage_buffer;
use crate::gpu::{Gpu, GpuTest};
use crate::residency::{SizeClass, Slot};
use crate::upload::{Batch, GpuVertex, StagingRing};

/// Quads a side of the hand-built tiles: side 3, size class 1.
pub(super) const Q: u32 = 2;

/// The size class of the hand-built tiles.
fn class() -> SizeClass {
    SizeClass::new(1).expect("class 1")
}

/// The record of a tile of [`Q`] quads whose block vertex `(i, j)` (each `-1 ..= Q + 1`) is at
/// `at(i, j)`, every stencil the block's own axes, no base vertex.
pub(super) fn grid_record(at: impl Fn(i64, i64) -> Vec3) -> RingRecord {
    let q = i64::from(Q);
    let padded: Vec<Vec3> = (-1..=q + 1)
        .flat_map(|j| (-1..=q + 1).map(move |i| (i, j)))
        .map(|(i, j)| at(i, j))
        .collect();
    let mut record = RingRecord::default();
    record.assemble(Q, &[encode(OWN_AXES); STENCILS], &padded, [None; 4]);
    record
}

/// What a frame does to slots: each item of `writes` stages a record or copies vertices into
/// slot `k` of a buffer of `slots` slots of [`class`].
pub(super) enum Write<'a> {
    /// The normal pass computes the slot from the record.
    Record(usize, &'a RingRecord),
    /// The vertices are copied into the slot.
    Vertices(usize, &'a [GpuVertex]),
}

/// Runs one frame of `pass` (record chunks staged past the ring's cap, as a forced frame is)
/// doing `writes`, and returns every slot's vertices as the GPU holds them after it.
pub(super) fn frame(
    gpu: &Gpu,
    pass: &mut NormalPass,
    slots: usize,
    writes: &[Write<'_>],
) -> Vec<Vec<GpuVertex>> {
    let class = class();
    let slot_bytes = class.slot_bytes();
    let target = storage_buffer(gpu, "test slots", slots as u64 * slot_bytes).expect("slots");
    let mut ring = StagingRing::new(1 << 16, 1 << 16);
    let mut batch = Batch::new(&ring);
    pass.begin();
    for write in writes {
        match *write {
            Write::Record(k, record) => {
                let slot = Slot::for_tests(class, k);
                let at = k as u64 * slot_bytes;
                let staged = pass
                    .stage(
                        gpu,
                        &mut batch,
                        &mut ring,
                        record,
                        slot,
                        (&target, at),
                        true,
                    )
                    .expect("staged");
                assert!(staged, "a forced record is staged");
            }
            Write::Vertices(k, vertices) => {
                let at = k as u64 * slot_bytes;
                let data: &[u8] = bytemuck::cast_slice(vertices);
                let staged = batch
                    .write(gpu, &mut ring, data, &target, at, true)
                    .expect("staged");
                assert!(staged, "forced vertices are staged");
                pass.forget(Slot::for_tests(class, k));
            }
        }
    }
    let work = pass.finish(gpu, &mut batch, &mut ring).expect("finished");
    let (outcomes, _heard) = mpsc::channel();
    let uploads = batch.finish(0, work, None, outcomes);
    gpu.scoped("frame", || {
        let encoder = gpu
            .device()
            .create_command_encoder(&CommandEncoderDescriptor::default());
        uploads.submit(gpu.queue(), encoder, |_| {});
    })
    .expect("submits");
    let all: Vec<GpuVertex> = gpu.read_buffer(&target).expect("reads back");
    let per = usize::try_from(class.vertices()).expect("small");
    all.chunks(per).map(<[GpuVertex]>::to_vec).collect()
}

/// A flat grid at height `z`: `(i, j, z)`, whose every normal is exactly `(0, 0, 1)`.
fn flat(z: f32) -> RingRecord {
    #[expect(clippy::cast_precision_loss, reason = "small grid coordinates")]
    grid_record(|i, j| Vec3::new(i as f32, j as f32, z))
}

#[test]
fn a_forced_frame_past_the_cap_keeps_only_the_caps_chunks() {
    GpuTest::run(|gpu| {
        let record_bytes = flat(0.0).bytes().len() as u64;
        // Two records a chunk, two chunks kept; a frame of ten records needs five.
        let mut pass = NormalPass::new(gpu, 2 * record_bytes, 4 * record_bytes).expect("a pass");
        #[expect(clippy::cast_precision_loss, reason = "small counts")]
        let records: Vec<RingRecord> = (0..10).map(|k| flat(k as f32)).collect();
        let writes: Vec<Write<'_>> = records
            .iter()
            .enumerate()
            .map(|(k, r)| Write::Record(k, r))
            .collect();
        let held = frame(gpu, &mut pass, records.len(), &writes);
        assert_eq!(
            pass.kept_chunks(),
            2,
            "the cap's chunks kept, the rest one-off"
        );
        assert_eq!(
            pass.kept_bytes(),
            4 * record_bytes + pass.jobs.size(),
            "and counted"
        );
        for (k, slot) in held.iter().enumerate() {
            #[expect(clippy::cast_precision_loss, reason = "small counts")]
            let z = k as f32;
            for (index, vertex) in slot.iter().enumerate() {
                let (i, j) = (index % 3, index / 3);
                #[expect(clippy::cast_precision_loss, reason = "small grid coordinates")]
                let position = [i as f32, j as f32, z];
                assert_eq!(
                    vertex.position, position,
                    "slot {k} ({i}, {j}): from its record"
                );
                assert_eq!(
                    vertex.normal,
                    [0.0, 0.0, 1.0],
                    "slot {k} ({i}, {j})'s normal"
                );
            }
        }
        // The next frame, within the cap, reuses the kept chunks and makes none.
        let small: Vec<Write<'_>> = records
            .iter()
            .take(3)
            .enumerate()
            .map(|(k, r)| Write::Record(k, r))
            .collect();
        frame(gpu, &mut pass, 3, &small);
        assert_eq!(pass.kept_chunks(), 2, "still the cap's");
    });
}

#[test]
fn a_slot_copied_after_its_record_in_the_same_frame_holds_the_copy() {
    GpuTest::run(|gpu| {
        let record = flat(0.0);
        let copied = vec![
            GpuVertex {
                position: [7.0, 8.0, 9.0],
                roughness: 0.5,
                normal: [1.0, 0.0, 0.0],
                reserved: 0,
            };
            9
        ];
        let mut pass = NormalPass::new(gpu, 4096, 4096).expect("a pass");
        // Slot 0 takes a finest record, then a display tile's vertices; slot 1 the record alone.
        let writes = [
            Write::Record(0, &record),
            Write::Vertices(0, &copied),
            Write::Record(1, &record),
        ];
        let held = frame(gpu, &mut pass, 2, &writes);
        assert_eq!(held[0], copied, "the copy, not the record's vertices");
        assert_eq!(
            held[1][4].normal,
            [0.0, 0.0, 1.0],
            "the other slot is computed"
        );
    });
}

/// A record like [`grid_record`]'s with `stencils` and the base-vertex rings `rings` instead.
fn record_with(
    at: impl Fn(i64, i64) -> Vec3,
    stencils: &[u32; STENCILS],
    rings: [Option<&[Vec3]>; 4],
) -> RingRecord {
    let q = i64::from(Q);
    let padded: Vec<Vec3> = (-1..=q + 1)
        .flat_map(|j| (-1..=q + 1).map(move |i| (i, j)))
        .map(|(i, j)| at(i, j))
        .collect();
    let mut record = RingRecord::default();
    record.assemble(Q, stencils, &padded, rings);
    record
}

/// The one tile's vertices the normal pass computes from `record`.
fn computed(gpu: &Gpu, record: &RingRecord) -> Vec<GpuVertex> {
    let mut pass = NormalPass::new(gpu, 4096, 4096).expect("a pass");
    let mut held = frame(gpu, &mut pass, 1, &[Write::Record(0, record)]);
    held.pop().expect("one slot")
}

/// Asserts `vertices` (a tile of [`Q`] quads) hold NaN normals exactly at `nan` and `(0, 0, 1)`
/// everywhere else.
fn assert_nan_at(vertices: &[GpuVertex], nan: &[(u32, u32)], what: &str) {
    for (index, vertex) in vertices.iter().enumerate() {
        let index = u32::try_from(index).expect("small");
        let (i, j) = (index % (Q + 1), index / (Q + 1));
        if nan.contains(&(i, j)) {
            assert!(
                vertex.normal.iter().all(|c| c.is_nan()),
                "{what}: ({i}, {j}) is NaN, not {:?}",
                vertex.normal
            );
        } else {
            assert_eq!(vertex.normal, [0.0, 0.0, 1.0], "{what}: ({i}, {j})");
        }
    }
}

/// The flat grid `(i, j, 0)`.
#[expect(clippy::cast_precision_loss, reason = "small grid coordinates")]
fn plane(i: i64, j: i64) -> Vec3 {
    Vec3::new(i as f32, j as f32, 0.0)
}

/// The stencils of a tile whose block corners `corners` read their rings, every other vertex the
/// block's own axes.
fn with_corners(corners: &[u8]) -> [u32; STENCILS] {
    let mut stencils = [encode(OWN_AXES); STENCILS];
    for &c in corners {
        stencils[usize::from(c)] = encode(VertexStencil::Corner(c));
    }
    stencils
}

#[test]
fn coincident_positions_give_the_fallback_normal() {
    GpuTest::run(|gpu| {
        // Every point the same, a corner's ring too: every cross product is zero, its inputs
        // finite, so every normal is (0, 0, 1), as the CPU's.
        let point = Vec3::new(0.25, -1.5, 3.0);
        let ring = [point; 3];
        let record = record_with(
            |_, _| point,
            &with_corners(&[0]),
            [Some(&ring[..]), None, None, None],
        );
        let held = computed(gpu, &record);
        assert_nan_at(&held, &[], "coincident");
        assert!(
            held.iter()
                .all(|v| v.position.map(f32::to_bits) == [0.25, -1.5, 3.0_f32].map(f32::to_bits)),
            "the positions are the record's"
        );
    });
}

#[test]
fn a_nan_ring_point_makes_only_the_normals_reading_it_nan() {
    GpuTest::run(|gpu| {
        // Axes: ring point (-1, 1) is read by vertex (0, 1) alone (its u step); the corners read
        // their rings instead.
        let nan = Vec3::new(f32::NAN, 0.0, 0.0);
        let at = |i: i64, j: i64| if (i, j) == (-1, 1) { nan } else { plane(i, j) };
        let flat_ring = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
        ];
        let corner_ring = |c: (f32, f32)| flat_ring.map(|e| Vec3::new(e.x + c.0, e.y + c.1, 0.0));
        let rings = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)].map(corner_ring);
        let all = with_corners(&[0, 1, 2, 3]);
        let left_nan = record_with(
            at,
            &all,
            [
                Some(&rings[0][..]),
                Some(&rings[1][..]),
                Some(&rings[2][..]),
                Some(&rings[3][..]),
            ],
        );
        assert_nan_at(
            &computed(gpu, &left_nan),
            &[(0, 1)],
            "a NaN on the ring's left",
        );

        // Corner: one NaN point in corner 0's ring makes (0, 0) NaN, nothing else.
        let mut bad = rings[0];
        bad[2] = nan;
        let corner_nan = record_with(
            plane,
            &all,
            [
                Some(&bad[..]),
                Some(&rings[1][..]),
                Some(&rings[2][..]),
                Some(&rings[3][..]),
            ],
        );
        assert_nan_at(
            &computed(gpu, &corner_nan),
            &[(0, 0)],
            "a NaN in a corner's ring",
        );

        // A corner reading a ring of one point, and one reading none: no normal, as geom's.
        let one = [rings[2][0]];
        let short_rings = record_with(
            plane,
            &all,
            [
                Some(&rings[0][..]),
                Some(&rings[1][..]),
                Some(&one[..]),
                None,
            ],
        );
        assert_nan_at(
            &computed(gpu, &short_rings),
            &[(2, 2), (0, 2)],
            "rings of one point and of none",
        );
    });
}

#[test]
fn an_infinite_position_makes_its_neighbours_normals_nan() {
    GpuTest::run(|gpu| {
        // (1, 1) at infinity: its own normal reads only its neighbours, so stays (0, 0, 1); the
        // four vertices whose differences read it have a non-finite difference, so NaN.
        let at = |i: i64, j: i64| {
            if (i, j) == (1, 1) {
                Vec3::new(1.0, 1.0, f32::INFINITY)
            } else {
                plane(i, j)
            }
        };
        let held = computed(gpu, &grid_record(at));
        assert_nan_at(
            &held,
            &[(0, 1), (2, 1), (1, 0), (1, 2)],
            "an infinite point",
        );
        assert_eq!(held[4].position, [1.0, 1.0, f32::INFINITY], "kept as it is");
    });
}
