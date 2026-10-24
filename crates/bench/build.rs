//! Fingerprints the code that decides what `generate` writes, for `perf`'s medium-project
//! cache key.
//!
//! The project `generate` makes depends on the generator (`generate.rs`, its noise, the
//! session's configuration), on how tiles are evaluated (`geom`), on how they are stored and
//! encoded (`store`, and the versions of its dependencies such as zstd in `Cargo.lock`), on the
//! build profiles (the root `Cargo.toml`), on `generate`'s options (`cli.rs`) and on the
//! toolchain (`mise.toml`). This hashes all of those files' contents, so any change to them
//! gives a new key and `perf` regenerates the project. Changes elsewhere (the brush, `perf`
//! itself, comments in other crates) keep the cached one. A changed comment in `geom` or `store`
//! regenerates needlessly; that costs seconds, and missing a real change would cost a wrong
//! comparison.
//!
//! The hash is FNV-1a (64-bit) over each file's path relative to the repository, its length and
//! its bytes, in sorted path order, and reaches the crate as `BRUSHCRAFT_GENERATOR_HASH`.

use std::io::Write as _;
use std::path::{Path, PathBuf};

/// The files and directories (whole trees) the generated project depends on, relative to the
/// repository.
const INPUTS: [&str; 11] = [
    "Cargo.toml",
    "Cargo.lock",
    "mise.toml",
    "crates/geom/Cargo.toml",
    "crates/geom/src",
    "crates/store/Cargo.toml",
    "crates/store/src",
    "crates/bench/src/cli.rs",
    "crates/bench/src/generate.rs",
    "crates/bench/src/noise.rs",
    "crates/bench/src/session.rs",
];

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
    all.sort();
    let mut hash = FNV_OFFSET;
    for path in &all {
        let relative = path.strip_prefix(&repo)?;
        let bytes = std::fs::read(path)?;
        hash = fnv1a(hash, relative.to_string_lossy().as_bytes());
        hash = fnv1a(hash, &u64::try_from(bytes.len())?.to_le_bytes());
        hash = fnv1a(hash, &bytes);
    }
    writeln!(
        cargo,
        "cargo::rustc-env=BRUSHCRAFT_GENERATOR_HASH={hash:016x}"
    )?;
    Ok(())
}
