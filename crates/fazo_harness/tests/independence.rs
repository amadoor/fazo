use std::{env, path::Path, process::Command};

const FAZO_MANIFEST: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../fazo/Cargo.toml");

fn fazo_dependency_names() -> Vec<String> {
    let cargo = env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let output = Command::new(cargo)
        .args(["tree", "--all-features", "--edges", "normal,build", "--prefix", "none", "--format", "{p}", "--manifest-path"])
        .arg(Path::new(FAZO_MANIFEST))
        .output()
        .expect("cargo tree runs");
    assert!(output.status.success(), "cargo tree failed: {}", String::from_utf8_lossy(&output.stderr));
    String::from_utf8(output.stdout)
        .expect("cargo tree prints UTF-8")
        .lines()
        .filter_map(|line| line.split_whitespace().next().map(str::to_owned))
        .collect()
}

#[test]
fn fazo_depends_on_no_kune_crate_directly_or_transitively() {
    let names = fazo_dependency_names();
    assert!(names.iter().any(|name| name == "hashbrown"), "cargo tree lists fazo's dependencies: {names:?}");
    let kune: Vec<String> = names.into_iter().filter(|name| name.starts_with("kune")).collect();
    assert!(kune.is_empty(), "fazo must stay usable by any embedder, but depends on {kune:?}");
}
