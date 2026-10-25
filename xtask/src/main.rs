use std::ffi::OsStr;
use std::path::PathBuf;
use std::process::{Command, ExitCode};

const USAGE: &str = "usage: cargo xtask <check|scoreboard [args...]|perf|bytecode>";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let result = match args.first().map(String::as_str) {
        Some("check") => check(),
        Some("scoreboard") => harness("scoreboard", &args[1..]),
        Some("perf") => harness("perf", &args[1..]),
        Some("bytecode") => harness("bytecode", &args[1..]),
        _ => Err(USAGE.to_owned()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("xtask: {message}");
            ExitCode::FAILURE
        }
    }
}

fn check() -> Result<(), String> {
    cargo(&["clippy", "--workspace", "--all-targets", "--", "-D", "warnings"])?;
    cargo(&["clippy", "-p", "fazo", "--all-targets", "--no-default-features", "--", "-D", "warnings"])?;
    cargo(&["clippy", "-p", "fazo", "--all-targets", "--features", "opcode-counts", "--", "-D", "warnings"])?;
    cargo(&["clippy", "-p", "fazo", "--all-targets", "--features", "gc-stress", "--", "-D", "warnings"])?;
    cargo(&["clippy", "-p", "fazo", "--all-targets", "--features", "libm", "--", "-D", "warnings"])?;
    cargo(&["build", "-p", "fazo_cli"])?;
    let cli = target_dir()?.join("debug").join(format!("fazo{}", std::env::consts::EXE_SUFFIX));

    cargo_with(&["test", "--workspace", "--no-fail-fast"], &[("FAZO_CLI", cli.as_os_str())])?;
    cargo(&["test", "-p", "fazo", "--features", "libm", "--no-fail-fast"])
}

fn target_dir() -> Result<PathBuf, String> {
    let output = Command::new(cargo_program())
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .output()
        .map_err(|error| format!("starting cargo metadata: {error}"))?;
    let text = String::from_utf8_lossy(&output.stdout);
    let key = "\"target_directory\":\"";
    let start = text.find(key).ok_or("cargo metadata printed no target directory")? + key.len();
    let end = text[start..].find('"').ok_or("cargo metadata printed an unterminated target directory")?;

    Ok(PathBuf::from(text[start..start + end].replace("\\\\", "\\")))
}

fn harness(bin: &str, extra: &[String]) -> Result<(), String> {
    let mut args = vec!["run", "--quiet", "-p", "fazo_harness", "--bin", bin, "--"];
    args.extend(extra.iter().map(String::as_str));
    cargo(&args)
}

fn cargo(args: &[&str]) -> Result<(), String> {
    cargo_with(args, &[])
}

fn cargo_program() -> String {
    std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned())
}

fn cargo_with(args: &[&str], envs: &[(&str, &OsStr)]) -> Result<(), String> {
    let status = Command::new(cargo_program())
        .args(args)
        .envs(envs.iter().copied())
        .status()
        .map_err(|error| format!("starting cargo {}: {error}", args.join(" ")))?;
    if !status.success() {
        return Err(format!("cargo {} failed with {status}", args.join(" ")));
    }
    Ok(())
}
