use std::io::{self, BufRead, StdinLock};

use fazo::Value;

use crate::{
    readline::{self, Readline},
    session::{self, REPL_HOST_SLOTS, Session},
};

const MAX_INPUT: usize = 512;
const EOF_MARK: &[u8] = b"<eof>";
const CHUNKNAME: &[u8] = b"=stdin";

struct Input<'a> {
    readline: Option<Readline>,
    stdin: StdinLock<'a>,
}

impl Input<'_> {
    fn line(&mut self, prompt: &[u8]) -> Option<Vec<u8>> {
        if let Some(readline) = &self.readline {
            return readline.read(prompt);
        }
        session::write_stdout(prompt);
        read_bounded_line(&mut self.stdin)
    }

    fn save(&self, line: &[u8]) {
        if let Some(readline) = &self.readline
            && line.first().is_some_and(|&byte| byte != 0)
        {
            readline.save(line);
        }
    }
}

pub fn run(session: &mut Session, environment: bool) -> Result<(), Vec<u8>> {
    session.vm().set_stack_top(REPL_HOST_SLOTS);
    let mut input = Input {
        readline: readline::init(session, environment),
        stdin: io::stdin().lock(),
    };
    while let Some(line) = read_line(session, &mut input, true)? {
        let (text, statement) = read_statement(session, &mut input, line)?;
        input.save(&text);
        match statement {
            Ok(function) => evaluate(session, function),
            Err(message) => session::report(None, &message),
        }
    }
    session::write_stdout(b"\n");
    Ok(())
}

type Statement = Result<Value, Vec<u8>>;

fn read_statement(session: &mut Session, input: &mut Input, line: Vec<u8>) -> Result<(Vec<u8>, Statement), Vec<u8>> {
    let mut expression = b"return ".to_vec();
    expression.extend_from_slice(&line);
    expression.push(b';');
    if let Ok(function) = session.vm().load(CHUNKNAME, &expression) {
        return Ok((line, Ok(function)));
    }
    multiline(session, input, line)
}

/// Counts a `local` line too, as PUC Lua 5.5 does, since the check reads the byte after the keyword.
fn warn_about_local(line: &[u8]) {
    let trimmed = line.trim_ascii_start();
    let Some(rest) = trimmed.strip_prefix(b"local") else {
        return;
    };
    if matches!(rest.first(), None | Some(b' ' | b'\t')) {
        session::write_stderr(b"warning: locals do not survive across lines in interactive mode\n");
    }
}

fn multiline(session: &mut Session, input: &mut Input, mut line: Vec<u8>) -> Result<(Vec<u8>, Statement), Vec<u8>> {
    warn_about_local(&line);
    loop {
        let message = match session.vm().load(CHUNKNAME, &line) {
            Ok(function) => return Ok((line, Ok(function))),
            Err(error) => session.vm().error_message(&error),
        };
        if !message.ends_with(EOF_MARK) {
            return Ok((line, Err(message)));
        }
        let Some(more) = read_line(session, input, false)? else {
            return Ok((line, Err(message)));
        };
        line.push(b'\n');
        line.extend_from_slice(&more);
    }
}

fn evaluate(session: &mut Session, function: Value) {
    match session.call(function, &[]) {
        Ok(results) => print_results(session, &results),
        Err(message) => session::report(None, &message),
    }
}

fn print_results(session: &mut Session, results: &[Value]) {
    if results.is_empty() {
        return;
    }
    let vm = session.vm();
    let print = vm.global("print");
    if let Err(error) = vm.call(print, results) {
        let mut message = b"error calling 'print' (".to_vec();
        message.extend_from_slice(&vm.error_message(&error));
        message.push(b')');
        session::report(None, &message);
    }
}

fn read_line(session: &mut Session, input: &mut Input, first: bool) -> Result<Option<Vec<u8>>, Vec<u8>> {
    let prompt = prompt(session, first)?;
    Ok(input.line(&prompt).map(strip_newline))
}

fn prompt(session: &mut Session, first: bool) -> Result<Vec<u8>, Vec<u8>> {
    let vm = session.vm();
    let value = vm.global(if first { "_PROMPT" } else { "_PROMPT2" });
    if value.is_nil() {
        return Ok(if first { b"> ".to_vec() } else { b">> ".to_vec() });
    }
    vm.tostring(value).map_err(|error| vm.error_message(&error))
}

fn read_bounded_line(input: &mut impl BufRead) -> Option<Vec<u8>> {
    let mut line = Vec::new();
    while line.len() < MAX_INPUT - 1 {
        let mut byte = [0u8];
        if !matches!(input.read(&mut byte), Ok(1)) {
            break;
        }
        line.push(byte[0]);
        if byte[0] == b'\n' {
            break;
        }
    }
    (!line.is_empty()).then_some(line)
}

fn strip_newline(mut line: Vec<u8>) -> Vec<u8> {
    let length = line.iter().position(|&byte| byte == 0).unwrap_or(line.len());
    line.truncate(length);
    if line.last() == Some(&b'\n') {
        line.pop();
    }
    line
}

#[cfg(test)]
mod tests {
    use super::{read_bounded_line, strip_newline};

    #[test]
    fn a_line_is_read_up_to_its_newline_and_at_most_the_buffer_allows() {
        let mut input: &[u8] = b"ab\ncd";
        assert_eq!(read_bounded_line(&mut input), Some(b"ab\n".to_vec()));
        assert_eq!(read_bounded_line(&mut input), Some(b"cd".to_vec()));
        assert_eq!(read_bounded_line(&mut input), None);
        let long = vec![b'x'; 600];
        let mut input: &[u8] = &long;
        assert_eq!(read_bounded_line(&mut input).map(|line| line.len()), Some(511));
    }

    #[test]
    fn the_newline_goes_and_so_does_everything_after_a_nul() {
        assert_eq!(strip_newline(b"x = 1\n".to_vec()), b"x = 1");
        assert_eq!(strip_newline(b"a\0b\n".to_vec()), b"a");
    }
}
