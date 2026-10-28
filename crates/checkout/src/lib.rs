//! Where the source checkout a binary was built from is, so the binaries that write large files
//! (the app's projects and recovery files, the bench's projects) refuse to write them into it,
//! whatever `.gitignore` says.
//!
//! A binary built in a git worktree protects both that worktree and the main checkout it
//! belongs to: agents build in worktrees under the main checkout's `.claude/worktrees/`, and a
//! project written into the main checkout is as bad as one written into the worktree. The main
//! checkout is found through git's own files, with no git process: a worktree's `.git` is a
//! file naming its git directory (`gitdir: …`), whose `commondir` names the shared one, inside
//! the main checkout.
//!
//! A copy of a binary run away from its checkout (the checkout gone) protects nothing.

use std::io;
use std::path::{Component, Path, PathBuf};

/// The checkout this crate was built from: two levels above its manifest.
#[must_use]
pub fn build_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

/// The checkouts a path must stay out of, symbolic links resolved: the one built from, and,
/// if that is a git worktree, the main checkout it belongs to. Those that no longer exist are
/// left out.
#[must_use]
pub fn protected() -> Vec<PathBuf> {
    roots_of(&build_root())
}

/// The protected checkout `path` lies in, if any: [`protected`]'s first that contains it.
///
/// `path` (absolute, or taken from the working directory) needn't exist: its deepest existing
/// ancestor has its symbolic links resolved and the rest is appended, with `..` and `.` taken
/// lexically, so a path to a file not yet written is checked as it will be.
///
/// # Errors
///
/// If the path can't be made absolute or its existing ancestor resolved.
pub fn containing(path: &Path) -> io::Result<Option<PathBuf>> {
    let resolved = resolve(&std::path::absolute(path)?)?;
    let roots = protected();
    Ok(roots.into_iter().find(|root| resolved.starts_with(root)))
}

/// `root` and, if it is a git worktree, the main checkout it belongs to, canonicalized; those
/// that don't exist are left out.
fn roots_of(root: &Path) -> Vec<PathBuf> {
    let Ok(root) = root.canonicalize() else {
        return Vec::new();
    };
    let main = main_checkout(&root).filter(|main| *main != root);
    let mut roots = vec![root];
    roots.extend(main);
    roots
}

/// The main checkout of the git worktree at `root`; `None` if `root` isn't a worktree (its
/// `.git` is a directory, or missing) or git's files don't say.
fn main_checkout(root: &Path) -> Option<PathBuf> {
    let link = std::fs::read_to_string(root.join(".git")).ok()?;
    let git_dir = root.join(link.trim().strip_prefix("gitdir:")?.trim());
    let common = std::fs::read_to_string(git_dir.join("commondir")).ok()?;
    let common = git_dir.join(common.trim()).canonicalize().ok()?;
    common.parent().map(Path::to_path_buf)
}

/// `path` (absolute) with its deepest existing ancestor's symbolic links resolved, and the rest
/// appended with `.` and `..` applied lexically.
///
/// # Errors
///
/// If canonicalizing that ancestor fails.
fn resolve(path: &Path) -> io::Result<PathBuf> {
    let mut existing = path;
    let mut rest = Vec::new();
    while !existing.exists() {
        let (Some(parent), Some(name)) = (existing.parent(), existing.file_name()) else {
            break;
        };
        rest.push(name);
        existing = parent;
    }
    let mut resolved = existing.canonicalize()?;
    let outward = rest.into_iter().rev();
    for name in outward {
        match Path::new(name).components().next() {
            Some(Component::ParentDir) => {
                resolved.pop();
            }
            Some(Component::CurDir) | None => {}
            Some(_) => resolved.push(name),
        }
    }
    Ok(resolved)
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    /// A fresh directory under the system temp dir for one test, removed on drop.
    struct Scratch(PathBuf);

    impl Scratch {
        /// A directory named for this process and `name`.
        fn new(name: &str) -> Self {
            let dir = std::env::temp_dir()
                .join(format!("brushcraft-checkout-{}-{name}", std::process::id()));
            drop(std::fs::remove_dir_all(&dir));
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir.canonicalize().unwrap())
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            drop(std::fs::remove_dir_all(&self.0));
        }
    }

    #[test]
    fn this_checkout_is_protected_and_paths_in_it_are_found() {
        let root = build_root().canonicalize().unwrap();
        assert_eq!(protected().first(), Some(&root));
        let found = containing(&root.join("target").join("new.bcrf")).unwrap();
        assert_eq!(found, Some(root.clone()), "a file not yet written");
        let sneaky = std::env::temp_dir()
            .join("..")
            .join(root.strip_prefix("/").unwrap())
            .join("x.bcrf");
        assert!(
            containing(&sneaky).unwrap().is_some(),
            "through `..`: {}",
            sneaky.display()
        );
        let outside = std::env::temp_dir().join("brushcraft-checkout-outside.bcrf");
        assert_eq!(containing(&outside).unwrap(), None);
    }

    #[test]
    fn a_worktree_protects_its_main_checkout_too() {
        let scratch = Scratch::new("worktree");
        let main = scratch.0.join("main");
        let git_dir = main.join(".git").join("worktrees").join("wt");
        std::fs::create_dir_all(&git_dir).unwrap();
        std::fs::write(git_dir.join("commondir"), "../..\n").unwrap();
        let worktree = main.join(".claude").join("worktrees").join("wt");
        std::fs::create_dir_all(&worktree).unwrap();
        std::fs::write(
            worktree.join(".git"),
            format!("gitdir: {}\n", git_dir.display()),
        )
        .unwrap();
        assert_eq!(roots_of(&worktree), vec![worktree.clone(), main.clone()]);
        assert_eq!(
            roots_of(&main),
            vec![main.clone()],
            "the main checkout alone"
        );
        assert_eq!(
            roots_of(&scratch.0.join("gone")),
            Vec::<PathBuf>::new(),
            "a checkout that is gone protects nothing"
        );
    }

    #[test]
    fn a_relative_gitdir_is_taken_from_the_worktree() {
        let scratch = Scratch::new("relative");
        let main = scratch.0.join("main");
        let git_dir = main.join(".git").join("worktrees").join("wt");
        std::fs::create_dir_all(&git_dir).unwrap();
        std::fs::write(git_dir.join("commondir"), "../..").unwrap();
        let worktree = scratch.0.join("wt");
        std::fs::create_dir_all(&worktree).unwrap();
        std::fs::write(worktree.join(".git"), "gitdir: ../main/.git/worktrees/wt").unwrap();
        assert_eq!(main_checkout(&worktree), Some(main));
    }
}
