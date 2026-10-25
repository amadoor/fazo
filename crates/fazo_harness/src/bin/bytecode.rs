use std::process::ExitCode;

fn main() -> ExitCode {
    match fazo_harness::bytecode::run() {
        Ok(summary) => {
            print!("{summary}");
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("bytecode: {message}");
            ExitCode::FAILURE
        }
    }
}
