use std::{path::PathBuf, process::ExitCode};

const DEFAULT_SEEDS: u64 = 100;
/// The official files and seeds run in this thread; Windows' 1 MiB main-thread stack is too small.
const STACK_BYTES: usize = 8 << 20;

fn main() -> ExitCode {
    let worker = std::thread::Builder::new().stack_size(STACK_BYTES).spawn(run);
    match worker {
        Ok(handle) => handle.join().unwrap_or(ExitCode::FAILURE),
        Err(error) => {
            eprintln!("quick: cannot start the worker thread: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(fazo) = args.first().map(PathBuf::from) else {
        eprintln!("usage: quick FAZO_EXE [SEEDS_PER_PROFILE]");
        return ExitCode::FAILURE;
    };
    let seeds = args.get(1).and_then(|text| text.parse().ok()).unwrap_or(DEFAULT_SEEDS);
    let report = fazo_harness::quick::run(&fazo, seeds);
    print!("{}", fazo_harness::quick::render(&report));
    if report.passed() { ExitCode::SUCCESS } else { ExitCode::FAILURE }
}
