
use super::{arg_error, argument, check_integer, exec_result, failure_values, got};
use crate::{
    LuaError, Values, values, Vm,
    arena::Ref,
    host::{ExitStatus, HostError, HostFile, Seek},
    number::{self, Number},
    object::{NativeFn, Userdata},
    table::Table,
    value::Value,
};

pub struct Stream {
    kind: Kind,
    closed: bool,
    peeked: Option<u8>,
    error: Option<HostError>,
    pending: Vec<u8>,
    buffering: Buffering,
    writable: bool,
}

enum Kind {
    Stdin,
    Stdout,
    Stderr,
    File(Box<dyn HostFile>),
}

/// An open file cannot be stored; a closed one is stored as a closed standard stream.
#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct StreamRecord {
    kind: StoredKind,
    closed: bool,
    peeked: Option<u8>,
    error: Option<HostError>,
    #[serde(with = "crate::bytes")]
    pending: Vec<u8>,
    buffering: Buffering,
    writable: bool,
}

#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
enum StoredKind {
    Stdin,
    Stdout,
    Stderr,
}

impl Stream {
    pub(crate) fn to_record(&self) -> Result<StreamRecord, &'static str> {
        let kind = match &self.kind {
            Kind::Stdin => StoredKind::Stdin,
            Kind::Stdout => StoredKind::Stdout,
            Kind::Stderr => StoredKind::Stderr,
            Kind::File(_) if self.closed => StoredKind::Stdin,
            Kind::File(_) => return Err("a file is open"),
        };
        Ok(StreamRecord {
            kind,
            closed: self.closed,
            peeked: self.peeked,
            error: self.error.clone(),
            pending: self.pending.clone(),
            buffering: self.buffering,
            writable: self.writable,
        })
    }

    pub(crate) fn from_record(record: StreamRecord) -> Self {
        let kind = match record.kind {
            StoredKind::Stdin => Kind::Stdin,
            StoredKind::Stdout => Kind::Stdout,
            StoredKind::Stderr => Kind::Stderr,
        };
        Self {
            kind,
            closed: record.closed,
            peeked: record.peeked,
            error: record.error,
            pending: record.pending,
            buffering: record.buffering,
            writable: record.writable,
        }
    }
}

#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
enum Buffering {
    No,
    Line,
    Full(usize),
}

const DEFAULT_BUFFER_SIZE: usize = 8192;
const EBADF: i32 = 9;
const ESPIPE: i64 = 29;
const LUAL_BUFFER_SIZE: usize = 1024;

const STDIN: usize = 0;
const STDOUT: usize = 1;
const STDERR: usize = 2;

pub(super) fn open(vm: &mut Vm) {
    vm.register_builtin("io.lines", crate::object::NativeBody::Plain(lines_step));
    let functions: &[(&'static str, NativeFn)] = &[
        ("write", io_write),
        ("read", io_read),
        ("input", io_input),
        ("output", io_output),
        ("lines", io_lines),
        ("open", io_open),
        ("close", io_close),
        ("flush", io_flush),
        ("tmpfile", io_tmpfile),
        ("type", io_type),
        ("popen", io_popen),
    ];
    let library = vm.new_table();
    for (name, function) in functions {
        let value = vm.new_native(name, *function);
        vm.set_field(library, name, value);
    }
    let methods = vm.new_table();
    let method_functions: &[(&'static str, NativeFn)] = &[
        ("read", file_read),
        ("write", file_write),
        ("lines", file_lines),
        ("close", file_close),
        ("seek", file_seek),
        ("setvbuf", file_setvbuf),
        ("flush", file_flush),
    ];
    for (name, function) in method_functions {
        let value = vm.new_native(name, *function);
        vm.set_field(methods, name, value);
    }
    let metatable = vm.new_table();
    let name = vm.string_value(b"FILE*");
    vm.set_field(metatable, "__name", name);
    vm.set_field(metatable, "__index", Value::Table(methods));
    let tostring = vm.new_native("tostring", file_tostring);
    vm.set_field(metatable, "__tostring", tostring);
    let gc = vm.new_native("__gc", file_gc);
    vm.set_field(metatable, "__gc", gc);
    let close = vm.new_native("__close", file_gc);
    vm.set_field(metatable, "__close", close);
    vm.set_hidden("FILE*", Value::Table(metatable));
    vm.resources.push(Stream {
        kind: Kind::Stdin,
        closed: false,
        peeked: None,
        error: None,
        pending: Vec::new(),
        buffering: Buffering::Full(DEFAULT_BUFFER_SIZE),
        writable: true,
    });
    vm.resources.push(Stream {
        kind: Kind::Stdout,
        closed: false,
        peeked: None,
        error: None,
        pending: Vec::new(),
        buffering: Buffering::Full(DEFAULT_BUFFER_SIZE),
        writable: true,
    });
    vm.resources.push(Stream {
        kind: Kind::Stderr,
        closed: false,
        peeked: None,
        error: None,
        pending: Vec::new(),
        buffering: Buffering::Full(DEFAULT_BUFFER_SIZE),
        writable: true,
    });
    for (name, index) in [("stdin", STDIN), ("stdout", STDOUT), ("stderr", STDERR)] {
        let file = new_file(vm, index);
        vm.set_field(library, name, file);
        vm.set_hidden(name, file);
    }
    let registry = vm.registry();
    let stdin = vm.hidden("stdin");
    vm.set_field(registry, "_IO_input", stdin);
    let stdout = vm.hidden("stdout");
    vm.set_field(registry, "_IO_output", stdout);
    vm.set_global("io", Value::Table(library));
}

fn new_file(vm: &mut Vm, resource: usize) -> Value {
    let metatable = match vm.hidden("FILE*") {
        Value::Table(table) => Some(table),
        _ => None,
    };
    let userdata = vm.heap.alloc(Userdata {
        metatable: None,
        resource,
        user_values: Vec::new(),
    });
    vm.set_userdata_metatable_checked(userdata, metatable);
    Value::Userdata(userdata)
}

fn file_metatable(vm: &mut Vm) -> Option<Ref<Table>> {
    match vm.hidden("FILE*") {
        Value::Table(table) => Some(table),
        _ => None,
    }
}

fn file_handle(vm: &mut Vm, args: &[Value], index: usize, name: &str) -> Result<usize, LuaError> {
    let expected = file_metatable(vm);
    if let Value::Userdata(userdata) = argument(args, index) {
        let file = vm.heap.userdata.get(userdata).clone();
        if file.metatable == expected && expected.is_some() {
            return Ok(file.resource);
        }
    }
    let message = format!("FILE* expected, got {}", got(vm, args, index));
    Err(arg_error(vm, index + 1, name, &message))
}

fn file_argument(vm: &mut Vm, args: &[Value], index: usize, name: &str) -> Result<usize, LuaError> {
    let resource = file_handle(vm, args, index, name)?;
    if vm.resources[resource].closed {
        return Err(vm.runtime_error_text("attempt to use a closed file"));
    }
    Ok(resource)
}

fn default_file(vm: &mut Vm, key: &str) -> Value {
    let registry = vm.registry();
    let key = vm.string_value(key.as_bytes());
    vm.table_get(registry, key)
}

fn open_default_file(vm: &mut Vm, key: &str, name: &str) -> Result<Value, LuaError> {
    let file = default_file(vm, key);
    if let Value::Userdata(userdata) = file {
        let resource = vm.heap.userdata.get(userdata).resource;
        if vm.resources[resource].closed {
            let message = format!("default {name} file is closed");
            return Err(vm.runtime_error_text(&message));
        }
    }
    Ok(file)
}

fn io_input(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    g_iofile(vm, args, "_IO_input", b"r")
}

fn io_output(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    g_iofile(vm, args, "_IO_output", b"w")
}

fn g_iofile(vm: &mut Vm, args: &[Value], key: &'static str, mode: &[u8]) -> Result<Values, LuaError> {
    let value = argument(args, 0);
    match value {
        Value::Nil => {}
        Value::String(name) => {
            let file_name = vm.heap_bytes(name).to_vec();
            let file = match vm.host.files.open(&file_name, mode) {
                Ok(handle) => open_stream(vm, handle, mode),
                Err(error) => {
                    let message = format!("cannot open file '{}' ({})", String::from_utf8_lossy(&file_name), error.reason);
                    return Err(vm.runtime_error_text(&message));
                }
            };
            let registry = vm.registry();
            vm.set_field(registry, key, file);
        }
        _ => {
            file_argument(vm, args, 0, key)?;
            let registry = vm.registry();
            vm.set_field(registry, key, value);
        }
    }
    Ok(values![default_file(vm, key)])
}

fn write_values(vm: &mut Vm, resource: usize, args: &[Value], first: usize) -> Result<Option<HostError>, LuaError> {
    let mut failure = None;
    for (index, value) in args.iter().enumerate().skip(first) {
        let bytes = match value {
            Value::String(string) => vm.heap_bytes(*string).to_vec(),
            Value::Integer(integer) => integer.to_string().into_bytes(),
            Value::Float(float) => number::format_float(*float).into_bytes(),
            _ => {
                let got = value.type_name();
                return Err(arg_error(vm, index + 1 - first, "write", &format!("string expected, got {got}")));
            }
        };
        if failure.is_none() {
            failure = write_bytes(vm, resource, &bytes).err();
        }
    }
    Ok(failure)
}

fn bad_descriptor() -> HostError {
    HostError::new(EBADF, "Bad file descriptor")
}

fn write_bytes(vm: &mut Vm, resource: usize, bytes: &[u8]) -> Result<(), HostError> {
    let stream = &mut vm.resources[resource];
    match &stream.kind {
        Kind::Stdout => vm.host.stdio.write_stdout(bytes),
        Kind::Stderr => vm.host.stdio.write_stderr(bytes),
        Kind::Stdin => return Err(bad_descriptor()),
        Kind::File(_) if !stream.writable => return Err(bad_descriptor()),
        Kind::File(_) => return buffered_write(stream, bytes),
    }
    Ok(())
}

fn buffered_write(stream: &mut Stream, bytes: &[u8]) -> Result<(), HostError> {
    match stream.buffering {
        Buffering::No => {
            flush_pending(stream)?;
            write_file(stream, bytes)
        }
        Buffering::Line => {
            stream.pending.extend_from_slice(bytes);
            if bytes.contains(&b'\n') {
                flush_pending(stream)
            } else {
                Ok(())
            }
        }
        Buffering::Full(size) => {
            stream.pending.extend_from_slice(bytes);
            if stream.pending.len() >= size {
                flush_pending(stream)
            } else {
                Ok(())
            }
        }
    }
}

fn flush_pending(stream: &mut Stream) -> Result<(), HostError> {
    if stream.pending.is_empty() {
        return Ok(());
    }
    let pending = std::mem::take(&mut stream.pending);
    write_file(stream, &pending)
}

fn write_file(stream: &mut Stream, bytes: &[u8]) -> Result<(), HostError> {
    match &mut stream.kind {
        Kind::File(handle) => handle.write(bytes),
        _ => Ok(()),
    }
}

fn flush_stream(stream: &mut Stream) -> Result<(), HostError> {
    flush_pending(stream)?;
    match &mut stream.kind {
        Kind::File(handle) => handle.flush(),
        _ => Ok(()),
    }
}

fn write_result(vm: &mut Vm, failure: Option<HostError>, file: Value) -> Values {
    match failure {
        None => values![file],
        Some(error) => failure_values(vm, &error, None),
    }
}

fn io_write(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let output = open_default_file(vm, "_IO_output", "output")?;
    let Value::Userdata(file) = output else {
        return Ok(Values::new());
    };
    let resource = vm.heap.userdata.get(file).resource;
    let failure = write_values(vm, resource, args, 0)?;
    Ok(write_result(vm, failure, output))
}

fn file_write(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let resource = file_argument(vm, args, 0, "write")?;
    let failure = write_values(vm, resource, args, 1)?;
    Ok(write_result(vm, failure, args[0]))
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
enum Format {
    Number,
    Line { keep_newline: bool },
    All,
    Count(usize),
}

fn parse_format(vm: &mut Vm, value: Value, position: usize) -> Result<Format, LuaError> {
    match value {
        Value::Integer(count) => Ok(Format::Count(count.max(0) as usize)),
        Value::Float(_) => Ok(Format::Count(check_format_integer(vm, value, position)?)),
        Value::String(string) => {
            let text = vm.heap_bytes(string).to_vec();
            let letters = text.strip_prefix(b"*").unwrap_or(&text);
            match letters.first() {
                Some(b'n') => Ok(Format::Number),
                Some(b'l') => Ok(Format::Line { keep_newline: false }),
                Some(b'L') => Ok(Format::Line { keep_newline: true }),
                Some(b'a') => Ok(Format::All),
                _ => Err(arg_error(vm, position, "read", "invalid format")),
            }
        }
        _ => Err(arg_error(vm, position, "read", "invalid format")),
    }
}

fn check_format_integer(vm: &mut Vm, value: Value, position: usize) -> Result<usize, LuaError> {
    let args = [value];
    Ok(check_integer(vm, &args, 0, "read").map(|count| count.max(0) as usize).unwrap_or_else(|_| {
        let _ = position;
        0
    }))
}

fn read_byte(vm: &mut Vm, resource: usize) -> Option<u8> {
    let stream = &mut vm.resources[resource];
    if let Some(byte) = stream.peeked.take() {
        return Some(byte);
    }
    if let Err(reason) = flush_pending(stream) {
        stream.error = Some(reason);
        return None;
    }
    let mut buffer = [0u8; 1];
    let outcome = match &mut stream.kind {
        Kind::Stdin => Ok(vm.host.stdio.read_stdin(&mut buffer)),
        Kind::File(handle) => handle.read(&mut buffer),
        Kind::Stdout | Kind::Stderr => Ok(0),
    };
    match outcome {
        Ok(count) => (count == 1).then_some(buffer[0]),
        Err(reason) => {
            vm.resources[resource].error = Some(reason);
            None
        }
    }
}

fn peek_byte(vm: &mut Vm, resource: usize) -> Option<u8> {
    let byte = read_byte(vm, resource)?;
    vm.resources[resource].peeked = Some(byte);
    Some(byte)
}

fn read_format(vm: &mut Vm, resource: usize, format: Format) -> Value {
    match format {
        Format::All => {
            let mut bytes = Vec::new();
            while let Some(byte) = read_byte(vm, resource) {
                bytes.push(byte);
            }
            vm.string_value(&bytes)
        }
        Format::Line { keep_newline } => {
            let mut bytes = Vec::new();
            let mut any = false;
            while let Some(byte) = read_byte(vm, resource) {
                any = true;
                if byte == b'\n' {
                    if keep_newline {
                        bytes.push(byte);
                    }
                    break;
                }
                bytes.push(byte);
            }
            if !any {
                return Value::Nil;
            }
            vm.string_value(&bytes)
        }
        Format::Count(count) => {
            if count == 0 {
                return match peek_byte(vm, resource) {
                    Some(_) => vm.string_value(b""),
                    None => Value::Nil,
                };
            }
            let mut bytes = Vec::new();
            while bytes.len() < count {
                match read_byte(vm, resource) {
                    Some(byte) => bytes.push(byte),
                    None => break,
                }
            }
            if bytes.is_empty() {
                return Value::Nil;
            }
            vm.string_value(&bytes)
        }
        Format::Number => read_number(vm, resource),
    }
}

const MAX_NUMERAL_LENGTH: usize = 200;

#[derive(Default, serde::Serialize, serde::Deserialize)]
struct Numeral {
    #[serde(with = "crate::bytes")]
    text: Vec<u8>,
    overflowed: bool,
}

impl Numeral {
    fn accept(&mut self, vm: &mut Vm, resource: usize, matches: impl Fn(u8) -> bool) -> bool {
        let Some(byte) = peek_byte(vm, resource) else {
            return false;
        };
        if !matches(byte) {
            return false;
        }
        if self.text.len() >= MAX_NUMERAL_LENGTH {
            self.overflowed = true;
            return false;
        }
        self.text.push(byte);
        read_byte(vm, resource);
        true
    }

    fn read_digits(&mut self, vm: &mut Vm, resource: usize, hex: bool) -> usize {
        let mut count = 0;
        while self.accept(vm, resource, |byte| if hex { byte.is_ascii_hexdigit() } else { byte.is_ascii_digit() }) {
            count += 1;
        }
        count
    }
}

fn read_number(vm: &mut Vm, resource: usize) -> Value {
    while peek_byte(vm, resource).is_some_and(|byte| matches!(byte, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')) {
        read_byte(vm, resource);
    }
    let mut numeral = Numeral::default();
    numeral.accept(vm, resource, |byte| matches!(byte, b'-' | b'+'));
    let mut count = 0;
    let mut hex = false;
    if numeral.accept(vm, resource, |byte| byte == b'0') {
        if numeral.accept(vm, resource, |byte| matches!(byte, b'x' | b'X')) {
            hex = true;
        } else {
            count = 1;
        }
    }
    count += numeral.read_digits(vm, resource, hex);
    if numeral.accept(vm, resource, |byte| byte == b'.') {
        count += numeral.read_digits(vm, resource, hex);
    }
    let exponent_mark = |byte: u8| if hex { matches!(byte, b'p' | b'P') } else { matches!(byte, b'e' | b'E') };
    if count > 0 && numeral.accept(vm, resource, exponent_mark) {
        numeral.accept(vm, resource, |byte| matches!(byte, b'-' | b'+'));
        numeral.read_digits(vm, resource, false);
    }
    if numeral.overflowed {
        return Value::Nil;
    }
    match number::parse_number(&numeral.text) {
        Some(Number::Integer(integer)) => Value::Integer(integer),
        Some(Number::Float(float)) => Value::Float(float),
        None => Value::Nil,
    }
}

/// Follows `g_read` in liolib.c; a format is checked only when its turn comes.
fn read_values(
    vm: &mut Vm,
    resource: usize,
    formats: &[Value],
    first_position: usize,
) -> Result<Result<Values, HostError>, LuaError> {
    vm.resources[resource].error = None;
    let mut results = Values::with_capacity(formats.len().max(1));
    if formats.is_empty() {
        results.push(read_format(vm, resource, Format::Line { keep_newline: false }));
    }
    for (index, value) in formats.iter().enumerate() {
        let format = parse_format(vm, *value, first_position + index)?;
        let value = read_format(vm, resource, format);
        let stop = value.is_nil();
        results.push(value);
        if stop {
            break;
        }
    }
    Ok(match vm.resources[resource].error.take() {
        Some(error) => Err(error),
        None => Ok(results),
    })
}

fn read_result(vm: &mut Vm, outcome: Result<Values, HostError>) -> Values {
    match outcome {
        Ok(values) => values,
        Err(error) => failure_values(vm, &error, None),
    }
}

fn io_read(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let input = open_default_file(vm, "_IO_input", "input")?;
    let Value::Userdata(file) = input else {
        return Ok(values![Value::Nil]);
    };
    let resource = vm.heap.userdata.get(file).resource;
    let outcome = read_values(vm, resource, args, 1)?;
    Ok(read_result(vm, outcome))
}

fn file_read(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let resource = file_argument(vm, args, 0, "read")?;
    let outcome = read_values(vm, resource, args.get(1..).unwrap_or_default(), 2)?;
    Ok(read_result(vm, outcome))
}

const MAX_LINE_FORMATS: usize = 250;

fn check_line_formats(vm: &mut Vm, args: &[Value]) -> Result<(), LuaError> {
    if args.len() > MAX_LINE_FORMATS + 1 {
        return Err(arg_error(vm, MAX_LINE_FORMATS + 2, "lines", "too many arguments"));
    }
    Ok(())
}

fn io_lines(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    if argument(args, 0).is_nil() {
        let input = open_default_file(vm, "_IO_input", "input")?;
        check_line_formats(vm, args)?;
        return Ok(values![lines_iterator(vm, input, args.get(1..).unwrap_or_default(), false)]);
    }
    let name = match argument(args, 0) {
        Value::String(string) => vm.heap_bytes(string).to_vec(),
        _ => return Err(arg_error(vm, 1, "lines", "string expected")),
    };
    let handle = match vm.host.files.open(&name, b"r") {
        Ok(handle) => handle,
        Err(error) => {
            let message = format!("cannot open file '{}' ({})", String::from_utf8_lossy(&name), error.reason);
            return Err(vm.runtime_error_text(&message));
        }
    };
    let file = open_stream(vm, handle, b"r");
    check_line_formats(vm, args)?;
    Ok(values![lines_iterator(vm, file, args.get(1..).unwrap_or_default(), true), Value::Nil, Value::Nil, file])
}

fn file_lines(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    file_argument(vm, args, 0, "lines")?;
    check_line_formats(vm, args)?;
    Ok(values![lines_iterator(vm, args[0], args.get(1..).unwrap_or_default(), false)])
}

fn lines_iterator(vm: &mut Vm, file: Value, formats: &[Value], close_at_end: bool) -> Value {
    let mut upvalues = vec![file, Value::Boolean(close_at_end)];
    upvalues.extend_from_slice(formats);
    vm.new_native_with("io.lines", crate::object::NativeBody::Plain(lines_step), upvalues)
}

fn lines_step(vm: &mut Vm, _args: &[Value]) -> Result<Values, LuaError> {
    let upvalues = vm.native_upvalues();
    let file = upvalues[0];
    let close_at_end = matches!(upvalues[1], Value::Boolean(true));
    let Value::Userdata(userdata) = file else {
        return Ok(values![Value::Nil]);
    };
    let resource = vm.heap.userdata.get(userdata).resource;
    if vm.resources[resource].closed {
        return Err(vm.runtime_error_text("file is already closed"));
    }
    let results = match read_values(vm, resource, &upvalues[2..], 2)? {
        Ok(results) => results,
        Err(error) => return Err(vm.runtime_error_text(&error.reason)),
    };
    if results.first().is_some_and(|value| value.is_nil()) && close_at_end {
        let _ = close_resource(vm, resource);
    }
    Ok(results)
}

fn open_stream(vm: &mut Vm, handle: Box<dyn HostFile>, mode: &[u8]) -> Value {
    let writable = mode.first() != Some(&b'r') || mode.contains(&b'+');
    vm.resources.push(Stream {
        kind: Kind::File(handle),
        closed: false,
        peeked: None,
        error: None,
        pending: Vec::new(),
        buffering: Buffering::Full(DEFAULT_BUFFER_SIZE),
        writable,
    });
    let resource = vm.resources.len() - 1;
    new_file(vm, resource)
}

fn io_open(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let name = match argument(args, 0) {
        Value::String(string) => vm.heap_bytes(string).to_vec(),
        _ => return Err(arg_error(vm, 1, "open", "string expected")),
    };
    let mode = match argument(args, 1) {
        Value::Nil => b"r".to_vec(),
        Value::String(string) => vm.heap_bytes(string).to_vec(),
        _ => return Err(arg_error(vm, 2, "open", "string expected")),
    };
    if !valid_mode(&mode) {
        return Err(arg_error(vm, 2, "open", "invalid mode"));
    }
    match vm.host.files.open(&name, &mode) {
        Ok(handle) => Ok(values![open_stream(vm, handle, &mode)]),
        Err(error) => Ok(failure_values(vm, &error, Some(&name))),
    }
}

fn io_popen(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let command = match argument(args, 0) {
        Value::String(string) => vm.heap_bytes(string).to_vec(),
        Value::Integer(_) | Value::Float(_) => vm.to_display_bytes(argument(args, 0)),
        _ => {
            let message = format!("string expected, got {}", got(vm, args, 0));
            return Err(arg_error(vm, 1, "popen", &message));
        }
    };
    let mode = match argument(args, 1) {
        Value::Nil => b"r".to_vec(),
        Value::String(string) => vm.heap_bytes(string).to_vec(),
        _ => {
            let message = format!("string expected, got {}", got(vm, args, 1));
            return Err(arg_error(vm, 2, "popen", &message));
        }
    };
    if mode != b"r" && mode != b"w" {
        return Err(arg_error(vm, 2, "popen", "invalid mode"));
    }
    match vm.host.files.popen(&command, &mode) {
        Ok(handle) => Ok(values![open_stream(vm, handle, &mode)]),
        Err(error) => Ok(failure_values(vm, &error, Some(&command))),
    }
}

fn io_tmpfile(vm: &mut Vm, _args: &[Value]) -> Result<Values, LuaError> {
    let name = match vm.host.files.tmpname() {
        Ok(name) => name,
        Err(error) => return Ok(failure_values(vm, &error, None)),
    };
    match vm.host.files.open(&name, b"w+") {
        Ok(handle) => {
            let _ = vm.host.files.remove(&name);
            Ok(values![open_stream(vm, handle, b"w+")])
        }
        Err(error) => Ok(failure_values(vm, &error, None)),
    }
}

fn valid_mode(mode: &[u8]) -> bool {
    let Some((&first, rest)) = mode.split_first() else {
        return false;
    };
    if !matches!(first, b'r' | b'w' | b'a') {
        return false;
    }
    let rest = rest.strip_prefix(b"+").unwrap_or(rest);
    let rest = rest.strip_prefix(b"b").unwrap_or(rest);
    rest.is_empty()
}

fn io_close(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    if argument(args, 0).is_nil() {
        let output = default_file(vm, "_IO_output");
        file_argument(vm, &[output], 0, "close")?;
        return close_value(vm, output);
    }
    file_argument(vm, args, 0, "close")?;
    close_value(vm, args[0])
}

fn file_close(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    file_argument(vm, args, 0, "close")?;
    close_value(vm, args[0])
}

fn file_gc(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let resource = file_handle(vm, args, 0, "__gc")?;
    if resource > STDERR && !vm.resources[resource].closed {
        close_value(vm, args[0])?;
    }
    Ok(Values::new())
}

fn close_value(vm: &mut Vm, file: Value) -> Result<Values, LuaError> {
    let Value::Userdata(userdata) = file else {
        return Ok(values![Value::Nil]);
    };
    let resource = vm.heap.userdata.get(userdata).resource;
    if resource <= STDERR {
        let message = vm.string_value(b"cannot close standard file");
        return Ok(values![Value::Nil, message]);
    }
    match close_resource(vm, resource) {
        Ok(None) => Ok(values![Value::Boolean(true)]),
        Ok(Some(status)) => Ok(exec_result(vm, status)),
        Err(error) => Ok(failure_values(vm, &error, None)),
    }
}

fn close_resource(vm: &mut Vm, resource: usize) -> Result<Option<ExitStatus>, HostError> {
    let stream = &mut vm.resources[resource];
    let flushed = flush_pending(stream);
    let kind = std::mem::replace(&mut stream.kind, Kind::Stdout);
    stream.closed = true;
    let status = match kind {
        Kind::File(mut handle) => handle.finish(),
        _ => Ok(None),
    };
    flushed?;
    status
}

fn io_type(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let expected = file_metatable(vm);
    let Value::Userdata(userdata) = argument(args, 0) else {
        return Ok(values![Value::Nil]);
    };
    let file = vm.heap.userdata.get(userdata).clone();
    if file.metatable != expected || expected.is_none() {
        return Ok(values![Value::Nil]);
    }
    let text: &[u8] = if vm.resources[file.resource].closed {
        b"closed file"
    } else {
        b"file"
    };
    Ok(values![vm.string_value(text)])
}

fn file_seek(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let resource = file_argument(vm, args, 0, "seek")?;
    let whence = match argument(args, 1) {
        Value::Nil => b"cur".to_vec(),
        Value::String(string) => vm.heap_bytes(string).to_vec(),
        _ => return Err(arg_error(vm, 2, "seek", "string expected")),
    };
    let offset = if argument(args, 2).is_nil() {
        0
    } else {
        check_integer(vm, args, 2, "seek")?
    };
    let position = match whence.as_slice() {
        b"set" => Seek::Start(offset.max(0) as u64),
        b"cur" => Seek::Current(offset),
        b"end" => Seek::End(offset),
        _ => {
            let name = String::from_utf8_lossy(&whence).into_owned();
            return Err(arg_error(vm, 2, "seek", &format!("invalid option '{name}'")));
        }
    };
    let stream = &mut vm.resources[resource];
    stream.peeked = None;
    if let Err(error) = flush_pending(stream) {
        return Ok(failure_values(vm, &error, None));
    }
    match &mut stream.kind {
        Kind::File(handle) => match handle.seek(position) {
            Ok(place) => Ok(values![Value::Integer(place as i64)]),
            Err(error) => Ok(failure_values(vm, &error, None)),
        },
        _ => Ok(values![Value::Nil, vm.string_value(b"Illegal seek"), Value::Integer(ESPIPE)]),
    }
}

fn file_setvbuf(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let resource = file_argument(vm, args, 0, "setvbuf")?;
    let mode = match argument(args, 1) {
        Value::String(string) => vm.heap_bytes(string).to_vec(),
        _ => {
            let got = got(vm, args, 1);
            return Err(arg_error(vm, 2, "setvbuf", &format!("string expected, got {got}")));
        }
    };
    let size = if argument(args, 2).is_nil() {
        LUAL_BUFFER_SIZE
    } else {
        check_integer(vm, args, 2, "setvbuf")?.max(0) as usize
    };
    let buffering = match mode.as_slice() {
        b"no" => Buffering::No,
        b"full" => Buffering::Full(size),
        b"line" => Buffering::Line,
        _ => {
            let name = String::from_utf8_lossy(&mode).into_owned();
            return Err(arg_error(vm, 2, "setvbuf", &format!("invalid option '{name}'")));
        }
    };
    let stream = &mut vm.resources[resource];
    let flushed = flush_pending(stream);
    stream.buffering = buffering;
    match flushed {
        Ok(()) => Ok(values![Value::Boolean(true)]),
        Err(error) => Ok(failure_values(vm, &error, None)),
    }
}

fn io_flush(vm: &mut Vm, _args: &[Value]) -> Result<Values, LuaError> {
    let output = open_default_file(vm, "_IO_output", "output")?;
    let Value::Userdata(file) = output else {
        return Ok(values![Value::Nil]);
    };
    let resource = vm.heap.userdata.get(file).resource;
    Ok(flush_resource(vm, resource))
}

fn file_flush(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let resource = file_argument(vm, args, 0, "flush")?;
    Ok(flush_resource(vm, resource))
}

fn flush_resource(vm: &mut Vm, resource: usize) -> Values {
    let outcome = flush_stream(&mut vm.resources[resource]);
    match outcome {
        Ok(()) => values![Value::Boolean(true)],
        Err(error) => failure_values(vm, &error, None),
    }
}

fn file_tostring(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let Value::Userdata(userdata) = argument(args, 0) else {
        return Ok(values![Value::Nil]);
    };
    let resource = vm.heap.userdata.get(userdata).resource;
    let text = if vm.resources[resource].closed {
        "file (closed)".to_string()
    } else {
        format!("file ({:#010x})", resource * 16 + 0x1000)
    };
    Ok(values![vm.string_value(text.as_bytes())])
}

#[cfg(test)]
mod tests {
    use super::valid_mode;

    #[test]
    fn modes_follow_the_reference_check() {
        assert!(valid_mode(b"r"));
        assert!(valid_mode(b"w+b"));
        assert!(!valid_mode(b"x"));
        assert!(!valid_mode(b"rw"));
    }
}
