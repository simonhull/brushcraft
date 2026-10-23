//! `brushcraft-bench`: the M0 benchmark harness.
//!
//! Five subcommands, headless, over the benchmark project (see [`cli::USAGE`]):
//!
//! - `generate` builds the benchmark sphere (`cube_sphere(20)`, levels 1 to 9: 1.26 billion
//!   triangles) with fBm-noise corrections at every level, and saves it ([`generate`]);
//! - `page-in` times evaluating and pinning a brush footprint and a 42M-quad region from an empty
//!   cache, cold and warm ([`page_in`]);
//! - `brush` times a stroke of 200 draw steps over a 50,000-vertex footprint ([`brush`]);
//! - `save` brushes, then times a save and the file's growth per edited vertex
//!   ([`brush::run_save`]);
//! - `budget` walks the model under a 1 GiB budget, sampling RSS every 100 ms ([`budget`]).
//!
//! Each result is a JSON line on stdout (`{"bench": …, "result": …, …}`); progress and a human
//! summary go to stderr. The project lives outside the repository, under
//! `$XDG_CACHE_HOME/brushcraft/bench/` by default ([`paths`]). Everything runs on one thread
//! except the recovery writer, prefetch and the RSS sampler: the single-threaded baseline that
//! later optimisation is measured against.

mod brush;
mod budget;
mod cli;
mod generate;
mod noise;
mod page_in;
mod paths;
mod report;
mod session;
mod surface;
mod sys;
#[cfg(test)]
mod tests;

use std::io::Write;
use std::process::ExitCode;

use anyhow::Result;

use crate::cli::{Command, Subcommand, USAGE};
use crate::paths::Paths;
use crate::report::Output;

/// Runs `command`, reporting to `out`.
///
/// # Errors
///
/// Whatever the subcommand fails with.
fn run<J: Write, H: Write>(command: &Command, out: &mut Output<J, H>) -> Result<()> {
    let options = &command.options;
    let paths = Paths::resolve(options.project.as_deref())?;
    match command.subcommand {
        Subcommand::Generate => generate::run(options, &paths, out),
        Subcommand::PageIn => page_in::run(options, &paths, out),
        Subcommand::Brush => brush::run(options, &paths, out),
        Subcommand::Save => brush::run_save(options, &paths, out),
        Subcommand::Budget => budget::run(options, &paths, out),
    }
}

fn main() -> ExitCode {
    let mut stderr = std::io::stderr();
    let command = match Command::parse(std::env::args_os().skip(1)) {
        Ok(command) => command,
        Err(e) => {
            // Nothing more to do if stderr is gone.
            drop(writeln!(stderr, "brushcraft-bench: {e:#}\n\n{USAGE}"));
            return ExitCode::from(2);
        }
    };
    let mut out = Output::new(std::io::stdout().lock(), std::io::stderr());
    match run(&command, &mut out) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            drop(writeln!(
                stderr,
                "brushcraft-bench {}: {e:#}",
                command.subcommand.name()
            ));
            ExitCode::FAILURE
        }
    }
}
