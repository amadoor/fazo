use std::io::{self, Write};

use crate::platform;

use fazo::{Options, Value, Values, Vm};

/// Stack slots lua.c holds below a script chunk; the depth at which stack overflow happens depends on them.
pub const SCRIPT_HOST_SLOTS: usize = 5;
/// Slots below a REPL chunk, which runs after `pmain` has emptied its arguments.
pub const REPL_HOST_SLOTS: usize = 3;

pub struct Session {
    vm: Vm,
    progname: String,
}

impl Session {
    pub fn new(progname: String, environment_variables: bool, extensions: &[Vec<u8>]) -> Self {
        let mut options = Options::default();
        options.os = true;
        options.io = true;
        options.package = true;
        options.environment_variables = environment_variables;
        for extension in extensions {
            options.set_extension(extension);
        }
        Self {
            vm: Vm::with_host(options, platform::host()),
            progname,
        }
    }

    pub fn progname(&self) -> &str {
        &self.progname
    }

    pub fn vm(&mut self) -> &mut Vm {
        &mut self.vm
    }

    pub fn load(&mut self, chunkname: &[u8], source: &[u8]) -> Result<Value, Vec<u8>> {
        self.vm
            .load(chunkname, source)
            .map_err(|error| self.vm.error_message(&error))
    }

    pub fn call(&mut self, function: Value, arguments: &[Value]) -> Result<Values, Vec<u8>> {
        let outcome = self.vm.call_with_traceback(function, arguments);
        outcome.map_err(|error| self.vm.error_message(&error))
    }
}

pub fn write_stdout(bytes: &[u8]) {
    let mut stdout = io::stdout().lock();
    let _ = stdout.write_all(&platform::to_text_output(bytes));
    let _ = stdout.flush();
}

pub fn write_stderr(bytes: &[u8]) {
    let _ = io::stderr().lock().write_all(&platform::to_text_output(bytes));
}

pub fn report(progname: Option<&str>, message: &[u8]) {
    let mut line = Vec::with_capacity(message.len() + 16);
    if let Some(name) = progname {
        line.extend_from_slice(name.as_bytes());
        line.extend_from_slice(b": ");
    }
    line.extend_from_slice(message);
    line.push(b'\n');
    write_stderr(&line);
}

pub fn print_version() {
    write_stdout(concat!("fazo ", env!("CARGO_PKG_VERSION"), " (Lua 5.5.1)  Copyright (C) 2026 Amador https://github.com/amadoor/fazo\n").as_bytes());
}
