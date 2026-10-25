use std::{fs, io, path::Path};

pub const DRIVER: &str = "all.lua";

pub const PORTABLE_MODE: &str = "_port=true;_soft=true;";

pub const OUT_OF_SCOPE: &[(&str, &str)] = &[
    (
        "api.lua",
        "needs the C test library T and the C API; without T every test is skipped",
    ),
    (
        "heavy.lua",
        "stress test that all.lua does not run; too long for the scoreboard deadline",
    ),
];

pub fn out_of_scope_reason(name: &str) -> Option<&'static str> {
    OUT_OF_SCOPE
        .iter()
        .find(|(file, _)| *file == name)
        .map(|(_, reason)| *reason)
}

/// Files the reference column does not run because they exceed the scoreboard deadline.
pub const REFERENCE_SKIPS: &[&str] = &["heavy.lua"];

pub fn reference_skips(name: &str) -> bool {
    REFERENCE_SKIPS.contains(&name)
}

/// Puts the flags on the first line so line numbers are unchanged; a shebang is blanked, since the flags would precede it.
pub fn in_portable_mode(source: &[u8]) -> Vec<u8> {
    let mut prefixed = PORTABLE_MODE.as_bytes().to_vec();
    if source.first() == Some(&b'#') {
        let newline = source.iter().position(|&byte| byte == b'\n').unwrap_or(source.len());
        prefixed.extend_from_slice(&source[newline..]);
    } else {
        prefixed.extend_from_slice(source);
    }
    prefixed
}

pub fn files(dir: &Path) -> io::Result<Vec<String>> {
    let mut names = Vec::new();
    for entry in fs::read_dir(dir)? {
        let name = entry?.file_name().to_string_lossy().into_owned();
        if name.ends_with(".lua") && name != DRIVER {
            names.push(name);
        }
    }
    names.sort();
    Ok(names)
}

#[cfg(test)]
mod tests {
    use super::{DRIVER, files};
    use crate::reference;

    #[test]
    fn suite_lists_every_file_but_the_driver() {
        let suite = reference::shared().expect("the reference suite is cached").suite.clone();
        let names = files(&suite).expect("the suite folder exists");
        assert!(names.len() > 25);
        assert!(!names.iter().any(|name| name == DRIVER));
    }
}
