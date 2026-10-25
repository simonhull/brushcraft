//! Decodes any bytes as a base mesh: an error or a mesh, never a panic; and a mesh accepted is
//! its own encoding.

#![no_main]

use geom::BaseMesh;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|bytes: &[u8]| {
    if let Ok(mesh) = BaseMesh::from_bytes(bytes) {
        assert!(
            mesh.to_bytes() == bytes,
            "an accepted mesh is its own encoding"
        );
    }
});
