pub mod bytecode;
pub mod cli;
pub mod differential;
pub mod generator;
#[cfg(test)]
mod interchange;
pub mod minimize;
pub mod number;
pub mod official;
pub mod perf;
pub mod reference;
pub mod report;

use std::{
    fs, io,
    path::{Path, PathBuf},
};

pub const CASES_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../fazo/tests/cases");

pub fn lua_files(dir: &Path) -> io::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().is_some_and(|extension| extension == "lua") {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use super::{CASES_DIR, differential::run_reference, lua_files};

    #[test]
    fn cases_run_deterministically_on_the_reference_and_only_err_cases_fail() {
        let files = lua_files(Path::new(CASES_DIR)).expect("cases directory is vendored");
        assert!(!files.is_empty());
        for path in files {
            let source = fs::read(&path).expect("case is readable");
            let first = run_reference(&source, "t").expect("the reference is cached");
            let second = run_reference(&source, "t").expect("the reference is cached");
            let expects_error = path
                .file_name()
                .is_some_and(|name| name.to_string_lossy().starts_with("err_"));
            assert_eq!(
                first.error.is_some(),
                expects_error,
                "{} errors on the reference: {:?}",
                path.display(),
                first.error
            );
            assert_eq!(first, second, "{} is not deterministic", path.display());
        }
    }
}
