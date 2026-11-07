//! `soundcraft-cli`: SoundCraft without a window. See the library crate for the commands.

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

// publik's usage counts. The file is publik's reference client, kept as it is.
#[rustfmt::skip]
mod publik_telemetry;

/// The version in publik's usage counts: the release tag without its `v`, for example `0.18.0-publik.6`.
/// The build number comes from `packaging/PUBLIK_BUILD`, the file the bundle version comes from.
fn publik_version() -> String {
    format!("{}-publik.{}", env!("CARGO_PKG_VERSION"), include_str!("../../../packaging/PUBLIK_BUILD").trim())
}

fn main() {
    // publik's usage counts (README: Usage counts). A detached copy of this tool sends them, so the tool never waits.
    publik_telemetry::start_cli("soundcraft", &publik_version());
    soundcraft_cli::logger::install();
    // `args_os`, not `args`: a non-Unicode argument must reach clap, not panic before anything is printed.
    let code = soundcraft_cli::run(std::env::args_os().skip(1), &mut std::io::stdout(), &mut std::io::stderr());
    // The subcommand's engine and its plugin instances are gone by now. This is process teardown,
    // so it lives here and not in `run`, which tests call many times in one process.
    soundcraft_engine::shutdown_plugin_hosts();
    std::process::exit(code);
}
