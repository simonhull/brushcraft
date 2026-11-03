// The relaxation gate (render::lod): after relaxation pass k, sets pass k + 1's dispatch to
// every node if pass k raised anything, else to no workgroup at all, so the passes after the
// relaxation settles cost nothing, with no readback.
//
// Preceded by `const RAISES: u32 = ...;` from lod.rs: the word offset of LodFrame.raises.

// The level choice's frame buffer as words: pass k's raises at RAISES + k.
@group(0) @binding(0) var<storage, read> frame_words: array<u32>;
// Dispatch arguments, three words each: entry 0 is every node (pass 0's), entry k pass k's.
@group(0) @binding(1) var<storage, read_write> args: array<u32>;
// The pass that just ran.
@group(0) @binding(2) var<uniform> cursor: Step;

// The pass index (render::lod::GpuStep).
struct Step {
    // The pass.
    index: u32,
    // Zero.
    reserved0: u32,
    // Zero.
    reserved1: u32,
    // Zero.
    reserved2: u32,
}

// Sets the next pass's dispatch from this pass's raises.
@compute @workgroup_size(1)
fn gate() {
    let k = cursor.index;
    let next = 3u * (k + 1u);
    args[next] = select(0u, args[0], frame_words[RAISES + k] > 0u);
    args[next + 1u] = 1u;
    args[next + 2u] = 1u;
}

// The budget's gate (render::lod), after the relaxation: where the budget binds (word
// BUDGET_WANTED of the frame buffer: threshold couldn't take every bucket, or the level-0 tiles
// are over it, or the relaxation couldn't fit a raise), sets entry REFILL, the budget's passes
// over every node, to every node and entry ONCE, its one-workgroup passes, to one workgroup;
// else both to no workgroup, so a frame the budget doesn't
// bind runs none of them on any workgroup, with no readback deciding.
//
// Preceded by `const BUDGET_WANTED`, `REFILL` and `ONCE` from lod.rs too.
@compute @workgroup_size(1)
fn binds() {
    let binding = frame_words[BUDGET_WANTED] != 0u;
    let every = 3u * REFILL;
    args[every] = select(0u, args[0], binding);
    args[every + 1u] = 1u;
    args[every + 2u] = 1u;
    let once = 3u * ONCE;
    args[once] = select(0u, 1u, binding);
    args[once + 1u] = 1u;
    args[once + 2u] = 1u;
}
