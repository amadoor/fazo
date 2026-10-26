use std::{
    env,
    io::{self, IsTerminal},
};


use fazo::{Value, Values};

use crate::{
    args::{self, Command, Flags, Listing},
    platform, repl,
    session::{self, SCRIPT_HOST_SLOTS, Session},
};

const COMMAND_LINE_CHUNKNAME: &[u8] = b"=(command line)";

pub fn run(argv: &[Vec<u8>]) -> bool {
    let progname = progname(argv);
    match args::parse(argv) {
        Command::BadOption(index) => {
            print_usage(&progname, &argv[index]);
            false
        }
        Command::Run { flags, script } => execute(argv, progname, flags, script),
    }
}

fn progname(argv: &[Vec<u8>]) -> String {
    match argv.first() {
        Some(name) if !name.is_empty() => String::from_utf8_lossy(name).into_owned(),
        _ => "lua".to_string(),
    }
}

/// Reports an error before the state closes, so finalizer output follows it, as `lua.c` does.
fn execute(argv: &[Vec<u8>], progname: String, flags: Flags, script: usize) -> bool {
    if flags.version {
        session::print_version();
    }
    let mut session = Session::new(progname, !flags.ignore_env, &flags.extensions);
    let outcome = run_session(&mut session, argv, &flags, script);
    report_opcode_counts(&mut session, &flags);
    if session.vm().options().jit_stats {
        session::write_stderr(session.vm().jit_stats_report().as_bytes());
    }
    let ok = match outcome {
        Ok(()) => true,
        Err(message) => {
            session::report(Some(session.progname()), &message);
            false
        }
    };
    session.vm().close();
    ok
}

#[cfg(feature = "opcode-counts")]
fn report_opcode_counts(session: &mut Session, flags: &Flags) {
    if flags.opcode_counts {
        session::write_stderr(session.vm().opcode_counts_report().as_bytes());
    }
}

#[cfg(not(feature = "opcode-counts"))]
fn report_opcode_counts(_session: &mut Session, _flags: &Flags) {}

fn run_session(session: &mut Session, argv: &[Vec<u8>], flags: &Flags, script: usize) -> Result<(), Vec<u8>> {
    if flags.listing != Listing::None {
        return list_script(session, argv, script, flags.listing);
    }
    session.vm().use_generational_collector();
    if flags.warnings {
        session.vm().warning(b"@on", false);
    }
    session.vm().set_stack_top(SCRIPT_HOST_SLOTS);
    set_argument_table(session, argv, script)?;
    if !flags.ignore_env {
        run_environment_init(session)?;
    }
    let option_limit = if script > 0 { script } else { argv.len() };
    run_options(session, argv, option_limit)?;
    if script > 0 {
        run_script(session, argv, script)?;
    }
    if flags.interactive {
        return repl::run(session, !flags.ignore_env);
    }
    if script == 0 && !flags.execute && !flags.version {
        return run_without_script(session, flags);
    }
    Ok(())
}

fn list_script(session: &mut Session, argv: &[Vec<u8>], script: usize, listing: Listing) -> Result<(), Vec<u8>> {
    if script == 0 {
        return Err(b"no input files given".to_vec());
    }
    let function = session.vm().load_file(Some(argv[script].as_slice()))?;
    let text = session.vm().listing(function, listing == Listing::Full).unwrap_or_default();
    session::write_stdout(&text);
    Ok(())
}

fn run_without_script(session: &mut Session, flags: &Flags) -> Result<(), Vec<u8>> {
    if io::stdin().is_terminal() {
        session::print_version();
        return repl::run(session, !flags.ignore_env);
    }
    if let Err(message) = run_file(session, None) {
        session::report(Some(session.progname()), &message);
    }
    Ok(())
}

fn set_argument_table(session: &mut Session, argv: &[Vec<u8>], script: usize) -> Result<(), Vec<u8>> {
    let vm = session.vm();
    let table = vm.create_table();
    for (index, argument) in argv.iter().enumerate() {
        let key = Value::Integer(index as i64 - script as i64);
        let value = vm.string(argument);
        vm.set_raw(table, key, value)
            .map_err(|error| vm.error_message(&error))?;
    }
    vm.set_global("arg", table);
    Ok(())
}

fn run_environment_init(session: &mut Session) -> Result<(), Vec<u8>> {
    let Some((name, init)) = init_variable() else {
        return Ok(());
    };
    match init.strip_prefix(b"@") {
        Some(file) => run_file(session, Some(file)),
        None => {
            let chunkname = format!("={name}");
            run_text(session, &init, chunkname.as_bytes())
        }
    }
}

fn init_variable() -> Option<(&'static str, Vec<u8>)> {
    ["LUA_INIT_5_5", "LUA_INIT"].into_iter().find_map(|name| {
        env::var_os(name).map(|value| (name, platform::argument_bytes(&value)))
    })
}

fn run_text(session: &mut Session, source: &[u8], chunkname: &[u8]) -> Result<(), Vec<u8>> {
    let function = session.load(chunkname, source)?;
    session.call(function, &[])?;
    Ok(())
}

fn run_file(session: &mut Session, name: Option<&[u8]>) -> Result<(), Vec<u8>> {
    let function = session.vm().load_file(name)?;
    session.call(function, &[])?;
    Ok(())
}

fn run_options(session: &mut Session, argv: &[Vec<u8>], limit: usize) -> Result<(), Vec<u8>> {
    let mut index = 1;
    while index < limit {
        index = match argv[index].get(1).copied() {
            Some(b'e') => {
                let (statement, next) = option_argument(argv, index);
                run_text(session, statement, COMMAND_LINE_CHUNKNAME)?;
                next
            }
            Some(b'l') => {
                let (spec, next) = option_argument(argv, index);
                run_library(session, spec)?;
                next
            }
            Some(b'X') => index + 2,
            _ => index + 1,
        };
    }
    Ok(())
}

fn option_argument(argv: &[Vec<u8>], index: usize) -> (&[u8], usize) {
    let attached = &argv[index][2..];
    if attached.is_empty() {
        (argv[index + 1].as_slice(), index + 2)
    } else {
        (attached, index + 1)
    }
}

fn run_library(session: &mut Session, spec: &[u8]) -> Result<(), Vec<u8>> {
    let (global, module) = split_library(spec);
    let require = session.vm().global("require");
    let module_name = session.vm().string(module);
    let results = session.call(require, &[module_name])?;
    let value = results.first().copied().unwrap_or(Value::Nil);
    session.vm().set_global(&String::from_utf8_lossy(global), value);
    Ok(())
}

fn split_library(spec: &[u8]) -> (&[u8], &[u8]) {
    if let Some(at) = spec.iter().position(|&byte| byte == b'=') {
        return (&spec[..at], &spec[at + 1..]);
    }
    let global = match spec.iter().position(|&byte| byte == b'-') {
        Some(at) => &spec[..at],
        None => spec,
    };
    (global, spec)
}

fn run_script(session: &mut Session, argv: &[Vec<u8>], script: usize) -> Result<(), Vec<u8>> {
    let name = argv[script].as_slice();
    let from_stdin = name == b"-" && argv[script - 1] != b"--";
    let function = session.vm().load_file((!from_stdin).then_some(name))?;
    let arguments = script_arguments(session)?;
    session.call(function, &arguments)?;
    Ok(())
}

fn script_arguments(session: &mut Session) -> Result<Values, Vec<u8>> {
    let vm = session.vm();
    let table = vm.global("arg");
    if !matches!(table, Value::Table(_)) {
        return Err(b"'arg' is not a table".to_vec());
    }
    let count = vm.raw_len(table);
    Ok((1..=count)
        .map(|index| vm.raw_get(table, Value::Integer(index)))
        .collect())
}

fn print_usage(progname: &str, bad: &[u8]) {
    let mut text = format!("{progname}: ").into_bytes();
    if matches!(bad.get(1), Some(b'e' | b'l')) {
        text.extend_from_slice(b"'");
        text.extend_from_slice(bad);
        text.extend_from_slice(b"' needs argument\n");
    } else {
        text.extend_from_slice(b"unrecognized option '");
        text.extend_from_slice(bad);
        text.extend_from_slice(b"'\n");
    }
    text.extend_from_slice(usage(progname).as_bytes());
    session::write_stderr(&text);
}

fn usage(progname: &str) -> String {
    format!(
        "usage: {progname} [options] [script [args]]\n\
         Available options are:\n\
         \x20 -e stat   execute string 'stat'\n\
         \x20 -i        enter interactive mode after executing 'script'\n\
         \x20 -l mod    require library 'mod' into global 'mod'\n\
         \x20 -l g=mod  require library 'mod' into global 'g'\n\
         \x20 -v        show version information\n\
         \x20 -E        ignore environment variables\n\
         \x20 -W        turn warnings on\n\
         \x20 --        stop handling options\n\
         \x20 -         stop handling options and execute stdin\n"
    )
}

#[cfg(test)]
mod tests {
    use super::split_library;

    #[test]
    fn a_library_without_a_name_is_loaded_into_its_module_name() {
        assert_eq!(split_library(b"mod"), (&b"mod"[..], &b"mod"[..]));
    }

    #[test]
    fn a_version_suffix_is_dropped_from_the_global_name_only() {
        assert_eq!(split_library(b"mod-1.0"), (&b"mod"[..], &b"mod-1.0"[..]));
    }

    #[test]
    fn an_explicit_global_name_takes_the_module_after_the_equals_sign() {
        assert_eq!(split_library(b"g=mod-1.0"), (&b"g"[..], &b"mod-1.0"[..]));
    }
}
