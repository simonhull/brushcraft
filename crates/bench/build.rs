//! Fingerprints the code that decides what `generate` writes, for `perf`'s medium-project
//! cache key.
//!
//! The project `generate` makes depends on the generator (`generate.rs`, its noise, the
//! session's configuration), on how tiles are evaluated (`geom`), on how they are stored and
//! encoded (`store`, and the versions of its dependencies such as zstd in `Cargo.lock`), on the
//! build profiles and features (the root and the crates' `Cargo.toml`s, and Cargo's own
//! configuration, `.cargo/config.toml`, when there is one), on the compiler's flags and target
//! features, on `generate`'s options (`cli.rs`) and on the toolchain (`mise.toml`). This hashes
//! all of those files' contents and the flags, so any change to them gives a new key and `perf`
//! regenerates the project. Changes elsewhere (the brush, `perf`
//! itself, comments in other crates) keep the cached one. A changed comment in `geom` or `store`
//! regenerates needlessly; that costs seconds, and missing a real change would cost a wrong
//! comparison.
//!
//! The hash is FNV-1a (64-bit) over each file's path relative to the repository, its length and
//! its bytes, in sorted path order (an optional file that is absent hashes its path and an
//! absent marker instead), then over each flag variable's name and value, and reaches the crate
//! as `BRUSHCRAFT_GENERATOR_HASH`.
//!
//! It also records the commit the binary is built from and whether tracked files differed from
//! it (`BRUSHCRAFT_BUILD_COMMIT`, `BRUSHCRAFT_BUILD_DIRTY`), so `perf` records what it measured,
//! not what the tree holds when it runs. It reruns on a commit or checkout (git's `HEAD` log and
//! index) and on any change to the code (`crates`, `tools`, the root manifests); a build outside
//! a git checkout says `unknown` and dirty.

use std::io::Write as _;
use std::path::{Path, PathBuf};

/// The files and directories (whole trees) the generated project depends on, relative to the
/// repository.
const INPUTS: [&str; 12] = [
    "Cargo.toml",
    "Cargo.lock",
    "mise.toml",
    "crates/bench/Cargo.toml",
    "crates/geom/Cargo.toml",
    "crates/geom/src",
    "crates/store/Cargo.toml",
    "crates/store/src",
    "crates/bench/src/cli.rs",
    "crates/bench/src/generate.rs",
    "crates/bench/src/noise.rs",
    "crates/bench/src/session.rs",
];

/// Files the generated project depends on when they exist, relative to the repository. One is
/// watched only while it exists: Cargo reruns a build script on every build while a path it
/// watches is missing.
const OPTIONAL_INPUTS: [&str; 1] = [".cargo/config.toml"];

/// The variables that carry the compiler's flags and the target features they enable.
///
/// Cargo gives them to build scripts, from `RUSTFLAGS` or its configuration. Target features can
/// change floating-point results (fused multiply-add, say).
const FLAG_INPUTS: [&str; 2] = ["CARGO_ENCODED_RUSTFLAGS", "CARGO_CFG_TARGET_FEATURE"];

/// What an optional input hashes as its length when it is absent: no file is this long.
const ABSENT: u64 = u64::MAX;

/// FNV-1a's 64-bit offset basis.
const FNV_OFFSET: u64 = 0xCBF2_9CE4_8422_2325;

/// FNV-1a's 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;

/// `hash` continued over `bytes` with FNV-1a.
fn fnv1a(hash: u64, bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .fold(hash, |h, &b| (h ^ u64::from(b)).wrapping_mul(FNV_PRIME))
}

/// Every file under `path` (itself, if it is a file), recursively.
///
/// # Errors
///
/// If a directory can't be read.
fn files(path: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    if path.is_dir() {
        for entry in std::fs::read_dir(path)? {
            files(&entry?.path(), out)?;
        }
    } else {
        out.push(path.to_owned());
    }
    Ok(())
}

/// What the repository's tree says at build time: the commit and whether tracked files differ,
/// `unknown` and dirty when git can't tell. Also returns git's directory, if found, for the
/// rerun triggers.
fn revision(repo: &Path) -> (String, bool, Option<PathBuf>) {
    let git = |args: &[&str]| {
        let output = std::process::Command::new("git")
            // No index refresh: a status that rewrote the index would make the next build rerun
            // build.rs, whose triggers include the index.
            .env("GIT_OPTIONAL_LOCKS", "0")
            .arg("-C")
            .arg(repo)
            .args(args)
            .output()
            .ok()?;
        output
            .status
            .success()
            .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
    };
    let dir = git(&["rev-parse", "--absolute-git-dir"]).map(PathBuf::from);
    match (
        git(&["rev-parse", "HEAD"]),
        git(&["status", "--porcelain", "--untracked-files=no"]),
    ) {
        (Some(commit), Some(status)) => (commit, !status.is_empty(), dir),
        _ => ("unknown".to_owned(), true, dir),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").ok_or("no manifest dir")?);
    let repo = manifest.join("..").join("..");
    // Build scripts talk to Cargo through stdout.
    let mut cargo = std::io::stdout().lock();
    let mut all = Vec::new();
    for input in INPUTS {
        let path = repo.join(input);
        writeln!(cargo, "cargo::rerun-if-changed={}", path.display())?;
        files(&path, &mut all)?;
    }
    let mut absent = Vec::new();
    for input in OPTIONAL_INPUTS {
        let path = repo.join(input);
        if path.exists() {
            writeln!(cargo, "cargo::rerun-if-changed={}", path.display())?;
            files(&path, &mut all)?;
        } else {
            absent.push(path);
        }
    }
    all.sort();
    let mut hash = FNV_OFFSET;
    for path in &all {
        let relative = path.strip_prefix(&repo)?;
        let bytes = std::fs::read(path)?;
        hash = fnv1a(hash, relative.to_string_lossy().as_bytes());
        hash = fnv1a(hash, &u64::try_from(bytes.len())?.to_le_bytes());
        hash = fnv1a(hash, &bytes);
    }
    for path in &absent {
        let relative = path.strip_prefix(&repo)?;
        hash = fnv1a(hash, relative.to_string_lossy().as_bytes());
        hash = fnv1a(hash, &ABSENT.to_le_bytes());
    }
    for name in FLAG_INPUTS {
        let value = std::env::var_os(name).unwrap_or_default();
        hash = fnv1a(hash, name.as_bytes());
        hash = fnv1a(hash, value.as_encoded_bytes());
    }
    for name in ["RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS"] {
        writeln!(cargo, "cargo::rerun-if-env-changed={name}")?;
    }
    writeln!(
        cargo,
        "cargo::rustc-env=BRUSHCRAFT_GENERATOR_HASH={hash:016x}"
    )?;
    let (commit, dirty, git_dir) = revision(&repo);
    let mut triggers = vec![
        repo.join("crates"),
        repo.join("tools"),
        repo.join("Cargo.toml"),
        repo.join("Cargo.lock"),
    ];
    if let Some(dir) = git_dir {
        triggers.extend([dir.join("logs").join("HEAD"), dir.join("index")]);
    }
    for path in triggers {
        writeln!(cargo, "cargo::rerun-if-changed={}", path.display())?;
    }
    writeln!(cargo, "cargo::rustc-env=BRUSHCRAFT_BUILD_COMMIT={commit}")?;
    writeln!(cargo, "cargo::rustc-env=BRUSHCRAFT_BUILD_DIRTY={dirty}")?;
    Ok(())
}
