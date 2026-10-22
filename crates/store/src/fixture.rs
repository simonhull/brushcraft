//! What tests create files with: a small base mesh and a meta made for it.

use std::sync::Arc;

use geom::BaseMesh;

use crate::meta::{Meta, ProjectId, Role};

/// The project id of the tests' files.
pub(crate) const PROJECT: ProjectId =
    ProjectId::from_bits(0x7E57_0000_0000_0000_0000_0000_0000_0001);

/// The tests' default base mesh: the benchmark's sphere at its smallest, a cube of 6 faces.
///
/// # Panics
///
/// Never: the cube is a valid mesh.
pub(crate) fn mesh() -> Arc<BaseMesh> {
    Arc::new(geom::cube_sphere(1).unwrap())
}

/// A project file's meta for `mesh`, of project [`PROJECT`].
pub(crate) fn meta_for(mesh: &BaseMesh) -> Meta {
    Meta::new(PROJECT, Role::Project, mesh)
}

/// A project file's meta for [`mesh`].
pub(crate) fn meta() -> Meta {
    meta_for(&mesh())
}
