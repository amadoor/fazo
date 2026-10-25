use std::{path::Path, process::Command};

use fazo_harness::cli;

#[test]
fn command_line_cases_match_the_reference() {
    let outcomes = cli::run_all(Path::new(env!("CARGO_BIN_EXE_fazo"))).expect("the reference is cached");
    let failures: Vec<String> = outcomes
        .iter()
        .filter(|outcome| outcome.gap.is_none() && !outcome.passed())
        .map(|outcome| format!("{}: {}", outcome.name, outcome.mismatch.as_deref().unwrap_or_default()))
        .collect();
    assert!(
        failures.is_empty(),
        "command-line cases differ from the reference:\n{}",
        failures.join("\n")
    );
}

#[test]
fn version_banner_names_fazo_and_its_lua_version() {
    let output = Command::new(env!("CARGO_BIN_EXE_fazo"))
        .arg("-v")
        .output()
        .expect("fazo starts");
    let text = String::from_utf8_lossy(&output.stdout).replace("\r\n", "\n");
    assert!(output.status.success());
    let expected = format!("fazo {} (Lua 5.5.1)  Copyright (C) 2026 Amador https://github.com/amadoor/fazo\n", env!("CARGO_PKG_VERSION"));
    assert_eq!(text, expected);
}
