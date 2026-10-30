//! A check of the crate's shader rule ("Shaders" in the crate's notes): no function-local array
//! indexed by a value known only at run time.
//!
//! It reads a shader the way the driver gets it, through naga (the WGSL front end `wgpu` itself
//! uses), so it sees types rather than text: an index into a `let` or `var` array, into an array
//! inside a local copy of a struct (`let node = tree[n]; node.neighbours[k]`), or into a
//! `var<private>` array, by anything but a constant. An index into a buffer's array, or into a
//! vector, is fine. A loop counter counts as run time: the driver may unroll the loop or may not.
//!
//! The shaders' own tests call [`run_time_indexed_locals`] on each per-pixel and per-vertex
//! shader the crate builds, as the pipelines get it (`shade`, `raster`, `aa`'s pixel pass), and
//! this module's tests on the standalone ones (`tone`, `overlay`, `visibility_view`).

use wgpu::naga::valid::{Capabilities, FunctionInfo, ValidationFlags, Validator};
use wgpu::naga::{AddressSpace, Expression, Function, Module, TypeInner};

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;

/// Each place `source` indexes a function-local array at run time, as `function: expression`
/// (the expression's text).
///
/// # Panics
///
/// If `source` doesn't parse or validate: the check is for the crate's own shaders, which do.
pub(crate) fn run_time_indexed_locals(source: &str) -> Vec<String> {
    let module = wgpu::naga::front::wgsl::parse_str(source).expect("the shader parses");
    let info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .expect("the shader validates");
    let functions = module
        .functions
        .iter()
        .map(|(handle, function)| (function, &info[handle]));
    let entry_points = module
        .entry_points
        .iter()
        .enumerate()
        .map(|(index, entry)| (&entry.function, info.get_entry_point(index)));
    functions
        .chain(entry_points)
        .flat_map(|(function, analysis)| indexed_locals(&module, source, function, analysis))
        .collect()
}

/// [`run_time_indexed_locals`] in one function.
fn indexed_locals(
    module: &Module,
    source: &str,
    function: &Function,
    info: &FunctionInfo,
) -> Vec<String> {
    let name = function.name.as_deref().unwrap_or("?");
    let found = function
        .expressions
        .iter()
        .filter_map(|(handle, expression)| {
            let Expression::Access { base, .. } = *expression else {
                return None;
            };
            if !is_local_array(module, info[base].ty.inner_with(&module.types)) {
                return None;
            }
            let span = function.expressions.get_span(handle).to_range();
            let text = span.and_then(|range| source.get(range)).unwrap_or("?");
            Some(format!("{name}: {text}"))
        });
    found.collect()
}

/// Whether a value of type `ty` is an array held by the invocation itself: an array value, or a
/// pointer to one in a function's or the invocation's private memory.
fn is_local_array(module: &Module, ty: &TypeInner) -> bool {
    if let TypeInner::Pointer { base, space } = *ty {
        return matches!(space, AddressSpace::Function | AddressSpace::Private)
            && matches!(module.types[base].inner, TypeInner::Array { .. });
    }
    matches!(*ty, TypeInner::Array { .. })
}
