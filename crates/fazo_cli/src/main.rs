mod app;
mod args;
mod platform;
mod readline;
mod repl;
mod session;

use std::process::ExitCode;

const INTERPRETER_STACK_BYTES: usize = 256 * 1024 * 1024;

fn main() -> ExitCode {
    let argv: Vec<Vec<u8>> = std::env::args_os()
        .map(|argument| platform::argument_bytes(&argument))
        .collect();
    let interpreter = std::thread::Builder::new()
        .stack_size(INTERPRETER_STACK_BYTES)
        .spawn(move || app::run(&argv));
    match interpreter.map(std::thread::JoinHandle::join) {
        Ok(Ok(true)) => ExitCode::SUCCESS,
        _ => ExitCode::FAILURE,
    }
}
