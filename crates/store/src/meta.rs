//! A file's identity: which project it belongs to, what role it plays, and which base mesh its
//! tiles displace.
//!
//! Every project or recovery file begins, right after its header, with a `Meta` record (encoded
//! here) and a `Mesh` record holding the base mesh itself. Together they make a file
//! self-contained: it opens on its own, and a recovery file names the project and the save point
//! it extends, so it can't be layered on the wrong project or on a newer save.

use std::fmt;
use std::fs::File;
use std::io::{self, Read};
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::sync::Arc;

use geom::BaseMesh;

use crate::OpId;
use crate::bytes::Reader;

/// "None" for an optional `u64` field: a save point's head op.
const NONE: u64 = u64::MAX;

/// The longest base path a meta holds, in bytes: Linux's `PATH_MAX`. The OS can't open a longer
/// path anyway, so a meta made with one leaves it out (see [`Meta::on_save_point`]).
pub(crate) const MAX_BASE_PATH: usize = 4096;

/// The [`Role`] tag of a project file.
const ROLE_PROJECT: u8 = 0;
/// The [`Role`] tag of a recovery file.
const ROLE_RECOVERY: u8 = 1;

/// A project's identity: 128 random bits, shared by its project file and its recovery files.
///
/// Displayed as 32 lower-case hex digits, the form recovery file names use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ProjectId(u128);

impl ProjectId {
    /// A fresh id: 16 bytes from `/dev/urandom`.
    ///
    /// # Errors
    ///
    /// The I/O error of opening or reading `/dev/urandom`.
    pub fn random() -> io::Result<Self> {
        let mut bytes = [0; 16];
        File::open("/dev/urandom")?.read_exact(&mut bytes)?;
        Ok(Self(u128::from_le_bytes(bytes)))
    }

    /// The id with these bits.
    #[must_use]
    pub const fn from_bits(bits: u128) -> Self {
        Self(bits)
    }

    /// The id's bits.
    #[must_use]
    pub const fn to_bits(self) -> u128 {
        self.0
    }
}

/// 32 lower-case hex digits, zero-padded.
impl fmt::Display for ProjectId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:032x}", self.0)
    }
}

/// A save point of a project file: the state a recovery file's edits are layered on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SavePoint {
    /// The project whose file holds the save point.
    pub project: ProjectId,
    /// The save point's offset in that file (its Index record).
    pub offset: u64,
    /// The file's newest edit as of the save point; `None` before any edit.
    pub head: Option<OpId>,
}

/// What a file is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    /// The project file: where the user saves, holding every save point.
    Project,
    /// A recovery file: the edits since the last save, streamed as the user works.
    Recovery {
        /// The save point the edits are layered on; `None` for a project never saved, whose
        /// recovery file holds all of its work.
        base: Option<SavePoint>,
    },
}

/// A file's `Meta` record: its project, its role, a hash of its base mesh, and for a recovery
/// file on a save point, where that save point's project file was.
///
/// The hash is computed from the mesh when the meta is made, so a meta always describes a real
/// mesh; opening a file checks it against the file's own `Mesh` record.
///
/// The base path lets a recovery file found after a crash be listed with its project and
/// layered back on it (see [`crate::find_recoveries`]). It is only a hint: the project file
/// may have moved or been replaced since, so the identity the recovery file is checked
/// against is its save point, never the path.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Meta {
    /// The project the file belongs to.
    project: ProjectId,
    /// What the file is for.
    role: Role,
    /// [`mesh_hash`] of the base mesh's canonical bytes.
    mesh_hash: u128,
    /// Where the project file holding the role's save point was when the recovery file was
    /// made; `None` for any other role, or when it wasn't known.
    base_path: Option<Arc<Path>>,
}

impl Meta {
    /// The meta of a file of `project` with `role`, over `mesh`.
    #[must_use]
    pub fn new(project: ProjectId, role: Role, mesh: &BaseMesh) -> Self {
        Self {
            project,
            role,
            mesh_hash: mesh_hash(&mesh.to_bytes()),
            base_path: None,
        }
    }

    /// This meta with another role: the same project and mesh, so the mesh isn't hashed again.
    /// A recovery file's meta is its project file's, with the save point it extends. It names
    /// no base path; [`Meta::on_save_point`] makes a recovery file's meta with one.
    #[must_use]
    pub fn with_role(&self, role: Role) -> Self {
        Self {
            role,
            base_path: None,
            ..*self
        }
    }

    /// The meta of a recovery file layered on `point`, a save point of the project file at
    /// `path`: this meta's project and mesh, the role [`Role::Recovery`] with `point` as its
    /// base, and `path` as the base path.
    ///
    /// `path` should be absolute, so the recovery file can be matched with its project from
    /// anywhere. A path longer than 4096 bytes (Linux's `PATH_MAX`), which no file can have, is
    /// left out.
    #[must_use]
    pub fn on_save_point(&self, point: SavePoint, path: &Path) -> Self {
        let fits = path.as_os_str().len() <= MAX_BASE_PATH;
        Self {
            role: Role::Recovery { base: Some(point) },
            base_path: fits.then(|| Arc::from(path)),
            ..*self
        }
    }

    /// This meta with another project id, over the same mesh: a project saved as a new
    /// project keeps its mesh and gets an id of its own.
    #[must_use]
    pub(crate) fn with_project(&self, project: ProjectId) -> Self {
        Self {
            project,
            ..self.clone()
        }
    }

    /// The project the file belongs to.
    #[must_use]
    pub const fn project(&self) -> ProjectId {
        self.project
    }

    /// What the file is for.
    #[must_use]
    pub const fn role(&self) -> Role {
        self.role
    }

    /// The hash of the base mesh's canonical bytes ([`BaseMesh::to_bytes`]): two files with
    /// equal hashes displace the same mesh. A check against mistakes, not against an attacker.
    #[must_use]
    pub const fn mesh_hash(&self) -> u128 {
        self.mesh_hash
    }

    /// Where the project file holding a recovery file's save point was when the recovery file
    /// was made: a hint for finding the project after a crash, not part of its identity (see
    /// the type's notes). `None` for a project file, a recovery file of a project never saved,
    /// or one made without it.
    #[must_use]
    pub fn base_path(&self) -> Option<&Path> {
        self.base_path.as_deref()
    }

    /// The `Meta` record's payload: `project: u128 | role: u8 | base | mesh_hash: u128`,
    /// little-endian, where `base` is present only for a recovery file, as `has_base: u8` then,
    /// if it is 1, `project: u128 | offset: u64 | head: u64 | path_len: u32 | path`: the save
    /// point (head `u64::MAX` for none) and the base path's bytes as the OS has them (length 0
    /// for none).
    pub(crate) fn to_payload(&self) -> Vec<u8> {
        let path = self
            .base_path
            .as_deref()
            .map_or(&[][..], |path| path.as_os_str().as_bytes());
        let mut out = Vec::with_capacity(70 + path.len());
        out.extend_from_slice(&self.project.0.to_le_bytes());
        match self.role {
            Role::Project => out.push(ROLE_PROJECT),
            Role::Recovery { base } => {
                out.push(ROLE_RECOVERY);
                match base {
                    None => out.push(0),
                    Some(point) => {
                        out.push(1);
                        out.extend_from_slice(&point.project.0.to_le_bytes());
                        out.extend_from_slice(&point.offset.to_le_bytes());
                        let head = point.head.map_or(NONE, OpId::to_bits);
                        out.extend_from_slice(&head.to_le_bytes());
                        // At most MAX_BASE_PATH bytes (see `on_save_point`), so it fits.
                        let len = u32::try_from(path.len()).unwrap_or(u32::MAX);
                        out.extend_from_slice(&len.to_le_bytes());
                        out.extend_from_slice(path);
                    }
                }
            }
        }
        out.extend_from_slice(&self.mesh_hash.to_le_bytes());
        out
    }

    /// Parses [`Meta::to_payload`]'s output; `None` on any shortfall, an unknown role or flag,
    /// an invalid op, a base path longer than [`MAX_BASE_PATH`], or trailing bytes.
    pub(crate) fn from_payload(payload: &[u8]) -> Option<Self> {
        let mut r = Reader::new(payload);
        let project = ProjectId(r.u128()?);
        let mut base_path = None;
        let role = match r.u8()? {
            ROLE_PROJECT => Role::Project,
            ROLE_RECOVERY => Role::Recovery {
                base: match r.u8()? {
                    0 => None,
                    1 => {
                        let point = SavePoint {
                            project: ProjectId(r.u128()?),
                            offset: r.u64()?,
                            head: match r.u64()? {
                                NONE => None,
                                bits => Some(OpId::from_bits(bits)?),
                            },
                        };
                        let len = usize::try_from(r.u32()?)
                            .ok()
                            .filter(|&len| len <= MAX_BASE_PATH)?;
                        let path = r.take(len)?;
                        base_path = (!path.is_empty())
                            .then(|| Arc::from(Path::new(std::ffi::OsStr::from_bytes(path))));
                        Some(point)
                    }
                    _ => return None,
                },
            },
            _ => return None,
        };
        let mesh_hash = r.u128()?;
        r.rest().is_empty().then_some(Self {
            project,
            role,
            mesh_hash,
            base_path,
        })
    }

    /// This meta with another mesh hash, for tests that write a file whose meta lies.
    #[cfg(test)]
    pub(crate) fn with_mesh_hash(&self, mesh_hash: u128) -> Self {
        Self {
            mesh_hash,
            ..self.clone()
        }
    }
}

/// Our 128-bit hash of `bytes`: two FNV-1a 64 streams with different offset bases, side by side.
///
/// It catches a mismatched mesh (a file layered on the wrong project, a mesh record that isn't
/// the one its meta was made for); it is not meant to resist anyone forging a collision.
#[must_use]
pub(crate) fn mesh_hash(bytes: &[u8]) -> u128 {
    /// The FNV 64-bit prime.
    const PRIME: u64 = 0x0000_0100_0000_01B3;
    /// The standard FNV-1a 64 offset basis.
    const BASIS_LOW: u64 = 0xCBF2_9CE4_8422_2325;
    /// A second offset basis (the first FNV-1a 64 hash of "BrushCraft"), so the high stream
    /// starts elsewhere and its bits don't repeat the low stream's.
    const BASIS_HIGH: u64 = 0x9D6C_0412_7637_02D7;
    let (low, high) = bytes
        .iter()
        .fold((BASIS_LOW, BASIS_HIGH), |(low, high), &byte| {
            (
                (low ^ u64::from(byte)).wrapping_mul(PRIME),
                (high ^ u64::from(byte)).wrapping_mul(PRIME),
            )
        });
    (u128::from(high) << 64) | u128::from(low)
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    fn op(seq: u64) -> OpId {
        OpId::new(2, seq).unwrap()
    }

    #[test]
    fn ids_display_as_32_hex_digits() {
        assert_eq!(
            ProjectId::from_bits(0xAB).to_string(),
            "000000000000000000000000000000ab",
            "zero-padded, lower case"
        );
        assert_eq!(ProjectId::from_bits(u128::MAX).to_string().len(), 32, "max");
        let (a, b) = (ProjectId::random().unwrap(), ProjectId::random().unwrap());
        assert_ne!(a, b, "two random ids differ");
    }

    #[test]
    fn metas_round_trip_through_their_payload() {
        let mesh = geom::cube_sphere(1).unwrap();
        let project = ProjectId::from_bits(0x0123_4567_89AB_CDEF_FEDC_BA98_7654_3210);
        let point = |head| SavePoint {
            project,
            offset: 4096,
            head,
        };
        let plain = Meta::new(project, Role::Project, &mesh);
        let on_path = plain.on_save_point(point(Some(op(78))), Path::new("/home/ü/p.bcrf"));
        assert_eq!(
            on_path.base_path(),
            Some(Path::new("/home/ü/p.bcrf")),
            "the base path"
        );
        let metas = [
            Role::Project,
            Role::Recovery { base: None },
            Role::Recovery {
                base: Some(point(None)),
            },
            Role::Recovery {
                base: Some(point(Some(op(77)))),
            },
        ]
        .map(|role| Meta::new(project, role, &mesh));
        for meta in metas.into_iter().chain([on_path]) {
            let payload = meta.to_payload();
            assert_eq!(Meta::from_payload(&payload), Some(meta.clone()), "{meta:?}");
            for cut in 0..payload.len() {
                assert_eq!(
                    Meta::from_payload(&payload[..cut]),
                    None,
                    "{meta:?}: cut at {cut}"
                );
            }
            let long = [payload.as_slice(), &[0]].concat();
            assert_eq!(Meta::from_payload(&long), None, "{meta:?}: a trailing byte");
        }
    }

    #[test]
    fn a_base_path_too_long_for_any_file_is_left_out() {
        let mesh = geom::cube_sphere(1).unwrap();
        let project = ProjectId::from_bits(3);
        let point = SavePoint {
            project,
            offset: 512,
            head: None,
        };
        let meta = Meta::new(project, Role::Project, &mesh);
        let longest = format!("/{}", "a".repeat(MAX_BASE_PATH - 1));
        let kept = meta.on_save_point(point, Path::new(&longest));
        assert_eq!(kept.base_path(), Some(Path::new(&longest)), "the longest");
        assert_eq!(
            Meta::from_payload(&kept.to_payload()),
            Some(kept),
            "round trips"
        );
        let too_long = format!("{longest}a");
        let dropped = meta.on_save_point(point, Path::new(&too_long));
        assert_eq!(dropped.base_path(), None, "left out");
        assert_eq!(
            dropped.role(),
            Role::Recovery { base: Some(point) },
            "still on the save point"
        );
        // A payload stating a longer path is refused.
        let mut payload = dropped.to_payload();
        let at = 16 + 1 + 1 + 16 + 8 + 8;
        let stated = u32::try_from(MAX_BASE_PATH + 1).unwrap().to_le_bytes();
        payload[at..at + 4].copy_from_slice(&stated);
        payload.splice(at + 4..at + 4, too_long.bytes());
        assert_eq!(Meta::from_payload(&payload), None, "refused");
    }

    #[test]
    fn a_new_role_drops_the_base_path() {
        let mesh = geom::cube_sphere(1).unwrap();
        let project = ProjectId::from_bits(4);
        let point = SavePoint {
            project,
            offset: 512,
            head: None,
        };
        let on = Meta::new(project, Role::Project, &mesh).on_save_point(point, Path::new("/p"));
        assert_eq!(on.with_role(on.role()).base_path(), None, "with_role");
        assert_eq!(
            on.with_project(ProjectId::from_bits(5)).base_path(),
            Some(Path::new("/p")),
            "with_project keeps it"
        );
    }

    #[test]
    fn a_new_role_or_project_keeps_the_mesh_hash() {
        let mesh = geom::cube_sphere(2).unwrap();
        let project = ProjectId::from_bits(9);
        let meta = Meta::new(project, Role::Project, &mesh);
        let role = Role::Recovery {
            base: Some(SavePoint {
                project,
                offset: 512,
                head: Some(op(3)),
            }),
        };
        assert_eq!(
            meta.with_role(role),
            Meta::new(project, role, &mesh),
            "as if made with the role"
        );
        let other = ProjectId::from_bits(10);
        assert_eq!(
            meta.with_project(other),
            Meta::new(other, Role::Project, &mesh),
            "as if made for the other project"
        );
    }

    #[test]
    fn unknown_roles_and_flags_are_rejected() {
        let mesh = geom::cube_sphere(1).unwrap();
        let project = ProjectId::from_bits(5);
        let mut project_meta = Meta::new(project, Role::Project, &mesh).to_payload();
        project_meta[16] = 2;
        assert_eq!(Meta::from_payload(&project_meta), None, "role 2");
        let mut recovery_meta =
            Meta::new(project, Role::Recovery { base: None }, &mesh).to_payload();
        recovery_meta[17] = 2;
        assert_eq!(Meta::from_payload(&recovery_meta), None, "has_base 2");
    }

    #[test]
    fn the_mesh_hash_tells_meshes_apart() {
        let a = geom::cube_sphere(1).unwrap().to_bytes();
        let b = geom::cube_sphere(2).unwrap().to_bytes();
        assert_eq!(mesh_hash(&a), mesh_hash(&a.clone()), "deterministic");
        assert_ne!(mesh_hash(&a), mesh_hash(&b), "different meshes");
        let mut flipped = a.clone();
        flipped[20] ^= 1;
        let (h, g) = (mesh_hash(&a), mesh_hash(&flipped));
        assert!(
            (h ^ g) & u128::from(u64::MAX) != 0 && (h ^ g) >> 64 != 0,
            "one flipped bit changes both halves"
        );
        // The empty input hashes to the two offset bases.
        assert_eq!(
            mesh_hash(&[]),
            (0x9D6C_0412_7637_02D7_u128 << 64) | 0xCBF2_9CE4_8422_2325,
            "the offset bases"
        );
    }
}
