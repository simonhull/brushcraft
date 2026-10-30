//! The normal pass on records built by hand, outside residency: its record chunks, and a slot
//! written with vertices after a record in the same frame.

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
