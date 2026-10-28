//! Uploads from a real store: what lands in a slot is what the store gives.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use geom::{FaceId, Level, TileKey, cube_sphere};
use store::{Project, ProjectConfig, StoreConfig};
use wgpu::CommandEncoderDescriptor;

use super::*;
use crate::gpu::GpuTest;
use crate::residency::{Request, Requests, Residency, ResidencyConfig};

/// A fresh recovery directory under the system temp dir, removed with everything in it on drop.
struct TempDir(PathBuf);

impl TempDir {
    /// Names a directory for this process and `name` (the project creates it).
    fn new(name: &str) -> Self {
        let dir =
            std::env::temp_dir().join(format!("brushcraft-render-{}-{name}", std::process::id()));
        drop(std::fs::remove_dir_all(&dir));
        Self(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        drop(std::fs::remove_dir_all(&self.0));
    }
}

/// Level `l`.
fn level(l: u8) -> Level {
    Level::new(l).expect("a level")
}

/// `key`'s vertices as the GPU holds them.
fn slot_contents(gpu: &Gpu, residency: &Residency, key: &TileKey) -> Vec<GpuVertex> {
    let slot = residency
        .slot(key)
        .unwrap_or_else(|| panic!("{key:?} resident"));
    let class = slot.class();
    let buffer = &residency.slot_buffers(class)[usize::from(slot.buffer())];
    let all: Vec<GpuVertex> = gpu.read_buffer(buffer).expect("reads back");
    let n = usize::try_from(class.vertices()).expect("small");
    let first = usize::try_from(slot.index()).expect("small") * n;
    all[first..first + n].to_vec()
}

/// `v` as an array.
const fn array(v: Vec3) -> [f32; 3] {
    [v.x, v.y, v.z]
}

#[test]
fn the_stores_tiles_land_in_their_slots_bit_for_bit() {
    GpuTest::run(|gpu| {
        let dir = TempDir::new("store-source");
        let mesh = cube_sphere(1).expect("a mesh");
        let finest = level(7);
        let config = ProjectConfig {
            store: StoreConfig::new(256 << 20, 1).with_display(finest),
            recovery_dir: dir.0.clone(),
        };
        let mut project = Project::new(mesh.clone(), config).expect("a project");
        let store = project.store();
        store.queue_display_rebuild(finest);
        while store.has_jobs() {
            store
                .run_jobs(Instant::now() + Duration::from_millis(50))
                .expect("the display layer builds");
        }
        let mut residency =
            Residency::new(gpu, &mesh, finest, ResidencyConfig::default()).expect("residency");
        let face = FaceId::new(2);
        let fine = TileKey::displacement(face, finest, 1, 0).expect("a finest tile");
        let coarse = TileKey::display(face, level(6), 0, 0).expect("a display tile");
        let floor = TileKey::display(face, level(3), 0, 0).expect("a floor tile");
        let requests = Requests::new(
            [fine, coarse].map(|key| Request {
                key,
                screen_error: 2.0,
            }),
            0,
        );
        assert_eq!(
            prefetch_requests(store, &requests).ok(),
            Some(0),
            "nothing to page in"
        );
        // Requests are staged from the frame after they're made: the floor arrives in the
        // first, the requests in the second.
        for _ in 0..2 {
            residency.begin_frame(gpu).expect("begins");
            residency.want(&requests);
            let deadline = Instant::now() + Duration::from_secs(60);
            let uploads = residency
                .stage(gpu, &mut StoreSource::new(store), deadline)
                .expect("staged");
            gpu.scoped("frame", || {
                let encoder = gpu
                    .device()
                    .create_command_encoder(&CommandEncoderDescriptor::default());
                uploads.submit(gpu.queue(), encoder, |_| {});
            })
            .expect("submits");
        }
        assert!(residency.floor_ready(), "the floor arrived");

        let positions: Vec<[f32; 3]> = store
            .evaluate(fine)
            .expect("evaluates")
            .iter()
            .map(|&p| array(p))
            .collect();
        let normals: Vec<[f32; 3]> = store
            .normals(fine)
            .expect("normals")
            .iter()
            .map(|&n| array(n))
            .collect();
        let held = slot_contents(gpu, &residency, &fine);
        let held_positions: Vec<[f32; 3]> = held.iter().map(|v| v.position).collect();
        let held_normals: Vec<[f32; 3]> = held.iter().map(|v| v.normal).collect();
        assert_eq!(held_positions, positions, "the finest tile's positions");
        assert_eq!(held_normals, normals, "and its normals");

        for key in [coarse, floor] {
            let tile = store
                .display_for_draw(&[key])
                .expect("reads")
                .pop()
                .flatten()
                .unwrap_or_else(|| panic!("{key:?} built"));
            let shown = slot_contents(gpu, &residency, &key);
            let shown_positions: Vec<[f32; 3]> = shown.iter().map(|v| v.position).collect();
            let stored: Vec<[f32; 3]> = tile.positions().iter().map(|&p| array(p)).collect();
            assert_eq!(shown_positions, stored, "{key:?}'s positions");
            let roughness = shown.iter().zip(tile.roughness());
            for (vertex, &r) in roughness {
                assert_eq!(
                    vertex.roughness,
                    f32::from(r) / 255.0,
                    "{key:?}'s roughness"
                );
            }
        }
        drop(project.close(false));
    });
}
