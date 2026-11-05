//! The check on hand-made shaders, each indexing one kind of array, and on the standalone
//! per-pixel shaders.

use super::*;

/// A shader whose `probe` reads `read` (an `f32`) with `k` a run-time index, after `setup`.
fn probe(setup: &str, read: &str) -> String {
    format!(
        "struct Node {{ neighbours: array<u32, 4>, error: f32 }}
@group(0) @binding(0) var<storage, read> tree: array<Node>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
var<private> scratch: array<f32, 4>;
@compute @workgroup_size(1)
fn probe(@builtin(global_invocation_id) id: vec3<u32>) {{
    let k = id.x % 4u;
    {setup}
    out[id.x] = {read};
}}
"
    )
}

#[test]
fn a_local_array_indexed_at_run_time_is_found_in_each_of_its_forms() {
    let found = [
        ("var a: array<f32, 4>;", "a[k]"),
        ("let a = array<f32, 4>(1.0, 2.0, 3.0, 4.0);", "a[k]"),
        ("let node = tree[id.x];", "f32(node.neighbours[k])"),
        ("", "scratch[k]"),
        (
            "var a: array<f32, 4>; for (var i = 0u; i < 4u; i++) { a[i] = 1.0; }",
            "a[0]",
        ),
    ];
    for (setup, read) in found {
        let source = probe(setup, read);
        let indexed = run_time_indexed_locals(&source);
        assert_eq!(indexed.len(), 1, "{setup} {read}: {indexed:?}");
        assert!(indexed[0].starts_with("probe: "), "{indexed:?}");
    }
}

#[test]
fn constant_indices_buffers_and_vectors_pass() {
    let pass = [
        ("var a: array<f32, 4>;", "a[2]"),
        ("let a = array<f32, 4>(1.0, 2.0, 3.0, 4.0);", "a[3] + a[0u]"),
        ("let node = tree[id.x];", "f32(node.neighbours[1])"),
        ("", "f32(tree[k].neighbours[k]) + tree[k].error"),
        ("let v = vec4<f32>(1.0, 2.0, 3.0, 4.0);", "v[k]"),
        ("var v = vec2<f32>(0.0); v[k] = 1.0;", "v.x"),
    ];
    for (setup, read) in pass {
        let source = probe(setup, read);
        assert_eq!(
            run_time_indexed_locals(&source),
            Vec::<String>::new(),
            "{setup} {read}"
        );
    }
}

#[test]
fn the_standalone_per_pixel_shaders_index_no_local_array_at_run_time() {
    let shaders = [
        ("tone", include_str!("../tone.wgsl")),
        ("overlay", include_str!("../overlay.wgsl")),
        ("ui", include_str!("../ui.wgsl")),
        ("visibility_view", include_str!("../visibility_view.wgsl")),
    ];
    for (name, source) in shaders {
        assert_eq!(
            run_time_indexed_locals(source),
            Vec::<String>::new(),
            "{name}"
        );
    }
}
