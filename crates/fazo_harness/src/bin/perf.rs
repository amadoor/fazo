use std::process::ExitCode;

fn main() -> ExitCode {
    match fazo_harness::perf::run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("perf: {message}");
            ExitCode::FAILURE
        }
    }
}
