use crate::vm::{Builtin, Event};
use crate::gc::{Kind, Param};
use super::{arg_error, argument, check_any, check_integer, check_table, got};
use crate::{
    LuaError, Values, values, Vm,
    object::{Protection, Resume, Step},
    number::{self, Number},
    value::Value,
};

pub(super) fn open(vm: &mut Vm) {
    let plain: &[(&'static str, crate::object::NativeFn)] = &[
        ("tonumber", tonumber),
        ("type", lua_type),
        ("select", select),
        ("rawequal", rawequal),
        ("rawlen", rawlen),
        ("rawget", rawget),
        ("rawset", rawset),
        ("ipairs", ipairs),
        ("next", next),
        ("error", error),
        ("assert", assert),
        ("loadfile", loadfile),
        ("getmetatable", getmetatable),
        ("setmetatable", setmetatable),
        ("warn", warn),
        ("collectgarbage", collectgarbage),
    ];
    for (name, function) in plain {
        let value = vm.new_native(name, *function);
        vm.set_global(name, value);
    }
    let stepped: &[(&'static str, crate::object::SteppedFn)] = &[
        ("print", print),
        ("tostring", tostring),
        ("pairs", pairs),
        ("load", load),
        ("dofile", dofile),
        ("pcall", pcall),
        ("xpcall", xpcall),
    ];
    for (name, function) in stepped {
        let value = vm.new_stepped(name, *function);
        vm.set_global(name, value);
    }
    let globals = vm.global_table();
    vm.set_global("_G", globals);
    let version = vm.string_value(b"Lua 5.5");
    vm.set_global("_VERSION", version);
    let step = vm.new_stepped("ipairs_step", ipairs_step);
    vm.mark_builtin(step, Builtin::IpairsStep);
    vm.set_hidden("ipairs_step", step);
    for (name, builtin) in BUILTINS {
        let value = vm.global(name);
        vm.mark_builtin(value, builtin);
    }
    let next_function = vm.global("next");
    vm.set_hidden("next", next_function);
}

const BUILTINS: [(&str, Builtin); 10] = [
    ("type", Builtin::Type),
    ("select", Builtin::Select),
    ("rawequal", Builtin::RawEqual),
    ("rawlen", Builtin::RawLen),
    ("rawget", Builtin::RawGet),
    ("rawset", Builtin::RawSet),
    ("next", Builtin::Next),
    ("getmetatable", Builtin::GetMetatable),
    ("setmetatable", Builtin::SetMetatable),
    ("tostring", Builtin::ToString),
];

pub enum Piece {
    Text(Vec<u8>),
    Call(Value),
}

pub(crate) fn tostring_piece(vm: &mut Vm, value: Value) -> Piece {
    let handler = vm.metamethod(value, Event::ToString);
    if !handler.is_nil() {
        return Piece::Call(handler);
    }
    Piece::Text(default_text(vm, value))
}

fn default_text(vm: &mut Vm, value: Value) -> Vec<u8> {
    let text = vm.to_display_bytes(value);
    if matches!(value, Value::Nil | Value::Boolean(_) | Value::Integer(_) | Value::Float(_) | Value::String(_)) {
        return text;
    }
    let Some(metatable) = vm.metatable_of(value) else {
        return text;
    };
    let name = vm.raw_special_field(metatable, crate::vm::Field::Name);
    let Value::String(name) = name else {
        return text;
    };
    let mut named = vm.heap_bytes(name).to_vec();
    let address = text.iter().position(|&byte| byte == b':').map_or(&text[..0], |at| &text[at..]);
    named.extend_from_slice(address);
    named
}

pub(crate) fn metamethod_text(vm: &mut Vm, result: Value) -> Result<Vec<u8>, LuaError> {
    match result {
        Value::String(_) | Value::Integer(_) | Value::Float(_) => Ok(vm.to_display_bytes(result)),
        _ => Err(vm.runtime_error_text("'__tostring' must return a string")),
    }
}

fn tostring(vm: &mut Vm, args: &[Value]) -> Result<Step, LuaError> {
    check_any(vm, args, 0, "tostring")?;
    match tostring_piece(vm, args[0]) {
        Piece::Text(text) => Ok(Step::Done(values![vm.string_value(&text)])),
        Piece::Call(handler) => Ok(Step::Call {
            function: handler,
            arguments: values![args[0]],
            resume: Resume::ToString,
        }),
    }
}

pub(crate) fn resume_tostring(vm: &mut Vm, values: Values) -> Result<Step, LuaError> {
    let text = metamethod_text(vm, values.first().copied().unwrap_or(Value::Nil))?;
    Ok(Step::Done(values![vm.string_value(&text)]))
}

impl PrintState {
    pub(crate) fn values(&self, out: &mut Values) {
        out.extend_from_slice(&self.arguments);
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PrintState {
    pub arguments: Values,
    pub next: usize,
    #[serde(with = "crate::bytes")]
    pub line: Vec<u8>,
}

fn print(vm: &mut Vm, args: &[Value]) -> Result<Step, LuaError> {
    let state = PrintState {
        arguments: Values::from_slice(args),
        next: 0,
        line: Vec::new(),
    };
    print_from(vm, state)
}

fn print_from(vm: &mut Vm, mut state: PrintState) -> Result<Step, LuaError> {
    while state.next < state.arguments.len() {
        let value = state.arguments[state.next];
        match tostring_piece(vm, value) {
            Piece::Text(text) => append_argument(&mut state, &text),
            Piece::Call(handler) => {
                return Ok(Step::Call {
                    function: handler,
                    arguments: values![value],
                    resume: Resume::Print(Box::new(state)),
                });
            }
        }
    }
    state.line.push(b'\n');
    vm.write_output(&state.line);
    Ok(Step::Done(Values::new()))
}

fn append_argument(state: &mut PrintState, text: &[u8]) {
    if state.next > 0 {
        state.line.push(b'\t');
    }
    state.line.extend_from_slice(text);
    state.next += 1;
}

pub(crate) fn resume_print(vm: &mut Vm, mut state: PrintState, values: Values) -> Result<Step, LuaError> {
    let text = metamethod_text(vm, values.first().copied().unwrap_or(Value::Nil))?;
    append_argument(&mut state, &text);
    print_from(vm, state)
}

fn lua_type(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    check_any(vm, args, 0, "type")?;
    Ok(values![vm.string_value(args[0].type_name().as_bytes())])
}

fn tonumber(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    if argument(args, 1).is_nil() {
        check_any(vm, args, 0, "tonumber")?;
        let converted = match args[0] {
            Value::Integer(_) | Value::Float(_) => Some(args[0]),
            Value::String(string) => number::parse_number(vm.heap_bytes(string)).map(Number::to_value),
            _ => None,
        };
        return Ok(values![converted.unwrap_or(Value::Nil)]);
    }
    let base = check_integer(vm, args, 1, "tonumber")?;
    let Value::String(string) = args[0] else {
        let got = if args.is_empty() { "no value" } else { args[0].type_name() };
        return Err(arg_error(vm, 1, "tonumber", &format!("string expected, got {got}")));
    };
    if !(2..=36).contains(&base) {
        return Err(arg_error(vm, 2, "tonumber", "base out of range"));
    }
    let text = vm.heap_bytes(string).to_vec();
    Ok(values![parse_in_base(&text, base as u32).map_or(Value::Nil, Value::Integer)])
}

fn parse_in_base(text: &[u8], base: u32) -> Option<i64> {
    let trimmed = trim_spaces(text);
    let (negative, digits) = match trimmed.first() {
        Some(b'-') => (true, &trimmed[1..]),
        Some(b'+') => (false, &trimmed[1..]),
        _ => (false, trimmed),
    };
    if digits.is_empty() {
        return None;
    }
    let mut accumulated: u64 = 0;
    for &byte in digits {
        let digit = (byte as char).to_digit(36)?;
        if digit >= base {
            return None;
        }
        accumulated = accumulated.wrapping_mul(u64::from(base)).wrapping_add(u64::from(digit));
    }
    let signed = accumulated as i64;
    Some(if negative { signed.wrapping_neg() } else { signed })
}

fn trim_spaces(text: &[u8]) -> &[u8] {
    let start = text.iter().position(|byte| !byte.is_ascii_whitespace()).unwrap_or(text.len());
    let end = text.iter().rposition(|byte| !byte.is_ascii_whitespace()).map_or(start, |last| last + 1);
    &text[start..end.max(start)]
}

fn select(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    if let Some(Value::String(selector)) = args.first()
        && vm.heap_bytes(*selector).first() == Some(&b'#')
    {
        return Ok(values![Value::Integer(args.len() as i64 - 1)]);
    }
    let count = args.len() as i64;
    let requested = check_integer(vm, args, 0, "select")?;
    let mut start = if requested < 0 {
        count + requested
    } else {
        requested.min(count)
    };
    if start < 1 {
        return Err(arg_error(vm, 1, "select", "index out of range"));
    }
    start = start.max(1);
    Ok(Values::from_slice(&args[start as usize..]))
}

fn rawequal(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    check_any(vm, args, 0, "rawequal")?;
    check_any(vm, args, 1, "rawequal")?;
    Ok(values![Value::Boolean(vm.raw_equal(args[0], args[1]))])
}

fn rawlen(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let length = match argument(args, 0) {
        Value::Table(table) => vm.table_border(table),
        Value::String(string) => vm.string_length(string) as i64,
        _ => {
            let message = format!("table or string expected, got {}", got(vm, args, 0));
            return Err(arg_error(vm, 1, "rawlen", &message));
        }
    };
    Ok(values![Value::Integer(length)])
}

fn rawget(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let table = check_table(vm, args, 0, "rawget")?;
    check_any(vm, args, 1, "rawget")?;
    Ok(values![vm.table_get(table, args[1])])
}

fn rawset(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let table = check_table(vm, args, 0, "rawset")?;
    check_any(vm, args, 1, "rawset")?;
    check_any(vm, args, 2, "rawset")?;
    vm.table_set(table, args[1], args[2])?;
    Ok(values![Value::Table(table)])
}

fn setmetatable(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let table = check_table(vm, args, 0, "setmetatable")?;
    let metatable = match argument(args, 1) {
        Value::Nil if args.len() >= 2 => None,
        Value::Table(metatable) if args.len() >= 2 => Some(metatable),
        _ => {
            let message = format!("nil or table expected, got {}", got(vm, args, 1));
            return Err(arg_error(vm, 2, "setmetatable", &message));
        }
    };
    if let Some(existing) = vm.metatable_of(Value::Table(table))
        && !vm.raw_special_field(existing, crate::vm::Field::Metatable).is_nil()
    {
        return Err(vm.runtime_error_text("cannot change a protected metatable"));
    }
    vm.attach_metatable(table, metatable);
    Ok(values![Value::Table(table)])
}

fn getmetatable(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    check_any(vm, args, 0, "getmetatable")?;
    let Some(metatable) = vm.metatable_of(args[0]) else {
        return Ok(values![Value::Nil]);
    };
    let protected = vm.raw_special_field(metatable, crate::vm::Field::Metatable);
    if !protected.is_nil() {
        return Ok(values![protected]);
    }
    Ok(values![Value::Table(metatable)])
}

fn ipairs(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    check_any(vm, args, 0, "ipairs")?;
    let step = vm.hidden("ipairs_step");
    Ok(values![step, args[0], Value::Integer(0)])
}

fn ipairs_step(vm: &mut Vm, args: &[Value]) -> Result<Step, LuaError> {
    let index = check_integer(vm, args, 1, "ipairs")?.wrapping_add(1);
    let object = argument(args, 0);
    match vm.index_chain(object, Value::Integer(index))? {
        crate::vm::IndexStep::Value(value) => Ok(ipairs_result(index, value)),
        crate::vm::IndexStep::Call { function, arguments } => Ok(Step::Call {
            function,
            arguments: arguments.into(),
            resume: Resume::Ipairs(index),
        }),
    }
}

fn ipairs_result(index: i64, value: Value) -> Step {
    if value.is_nil() {
        return Step::Done(values![Value::Nil]);
    }
    Step::Done(values![Value::Integer(index), value])
}

pub(crate) fn resume_ipairs(_vm: &mut Vm, index: i64, values: Values) -> Result<Step, LuaError> {
    let value = values.first().copied().unwrap_or(Value::Nil);
    Ok(ipairs_result(index, value))
}

fn pairs(vm: &mut Vm, args: &[Value]) -> Result<Step, LuaError> {
    check_any(vm, args, 0, "pairs")?;
    let handler = vm.metamethod(args[0], Event::Pairs);
    if handler.is_nil() {
        let next_function = vm.hidden("next");
        return Ok(Step::Done(values![next_function, args[0], Value::Nil, Value::Nil]));
    }
    Ok(Step::Call {
        function: handler,
        arguments: values![args[0]],
        resume: Resume::Pairs,
    })
}

pub(crate) fn resume_pairs(_vm: &mut Vm, values: Values) -> Result<Step, LuaError> {
    let mut results: Values = values.into_iter().take(4).collect();
    results.resize(4, Value::Nil);
    Ok(Step::Done(results))
}

fn next(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let table = check_table(vm, args, 0, "next")?;
    match vm.table_next(table, argument(args, 1))? {
        Some((key, value)) => Ok(values![key, value]),
        None => Ok(values![Value::Nil]),
    }
}

fn pcall(vm: &mut Vm, args: &[Value]) -> Result<Step, LuaError> {
    check_any(vm, args, 0, "pcall")?;
    Ok(Step::Call {
        function: args[0],
        arguments: Values::from_slice(&args[1..]),
        resume: Resume::Protected(Protection { handler: None, finalizer: false }),
    })
}

fn xpcall(vm: &mut Vm, args: &[Value]) -> Result<Step, LuaError> {
    let handler = argument(args, 1);
    if !handler.is_function() {
        let message = format!("function expected, got {}", got(vm, args, 1));
        return Err(arg_error(vm, 2, "xpcall", &message));
    }
    Ok(Step::Call {
        function: argument(args, 0),
        arguments: Values::from_slice(args.get(2..).unwrap_or_default()),
        resume: Resume::Protected(Protection {
            finalizer: false,
            handler: Some(handler),
        }),
    })
}

pub(crate) fn resume_pcall(values: Values) -> Result<Step, LuaError> {
    let mut results = Values::with_capacity(values.len() + 1);
    results.push(Value::Boolean(true));
    results.extend(values);
    Ok(Step::Done(results))
}

fn error(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let value = argument(args, 0);
    let level = if argument(args, 1).is_nil() {
        1
    } else {
        check_integer(vm, args, 1, "error")?
    };
    let value = if level > 0 {
        with_position(vm, value, level as usize)
    } else {
        value
    };
    Err(LuaError::from_value(value))
}

fn assert(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    check_any(vm, args, 0, "assert")?;
    if !args[0].is_falsy() {
        return Ok(Values::from_slice(args));
    }
    let message = args.get(1).copied().unwrap_or_else(|| vm.string_value(b"assertion failed!"));
    let positioned = with_position(vm, message, 1);
    Err(LuaError::from_value(positioned))
}

fn with_position(vm: &mut Vm, value: Value, level: usize) -> Value {
    let Value::String(_) = value else {
        return value;
    };
    match vm.position_at_level(level) {
        Some(mut prefix) => {
            prefix.extend(vm.to_display_bytes(value));
            vm.string_value(&prefix)
        }
        None => value,
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LoadSettings {
    #[serde(with = "crate::bytes")]
    chunk_name: Vec<u8>,
    #[serde(with = "crate::bytes")]
    mode: Vec<u8>,
    environment: Option<Value>,
}

fn read_pieces(vm: &mut Vm, reader: Value) -> Result<Vec<u8>, Value> {
    let mut source = Vec::new();
    loop {
        let piece = match vm.call_protected(reader, &[], None) {
            Ok(values) => values.first().copied().unwrap_or(Value::Nil),
            Err(error) => return Err(error.value()),
        };
        let bytes = match piece {
            Value::Nil => break,
            Value::String(string) => vm.heap_bytes(string).to_vec(),
            value @ (Value::Integer(_) | Value::Float(_)) => vm.to_display_bytes(value),
            _ => {
                let mut text = vm.position_at_level(1).unwrap_or_default();
                text.extend_from_slice(b"reader function must return a string");
                return Err(vm.string_value(&text));
            }
        };
        if bytes.is_empty() {
            break;
        }
        source.extend_from_slice(&bytes);
    }
    Ok(source)
}

fn load(vm: &mut Vm, args: &[Value]) -> Result<Step, LuaError> {
    match argument(args, 0) {
        Value::String(string) => {
            let source = vm.heap_bytes(string).to_vec();
            let settings = load_settings(vm, args, source.clone())?;
            Ok(Step::Done(load_source(vm, &source, &settings)?))
        }
        Value::Integer(_) | Value::Float(_) => {
            let source = vm.to_display_bytes(args[0]);
            let settings = load_settings(vm, args, source.clone())?;
            Ok(Step::Done(load_source(vm, &source, &settings)?))
        }
        reader @ (Value::Closure(_) | Value::Native(_)) => {
            let settings = load_settings(vm, args, b"=(load)".to_vec())?;
            match read_pieces(vm, reader) {
                Ok(source) => Ok(Step::Done(load_source(vm, &source, &settings)?)),
                Err(message) => Ok(Step::Done(values![Value::Nil, message])),
            }
        }
        _ => {
            let got = args.first().map_or("no value", |value| value.type_name());
            Err(arg_error(vm, 1, "load", &format!("function expected, got {got}")))
        }
    }
}

fn load_settings(vm: &mut Vm, args: &[Value], default_name: Vec<u8>) -> Result<LoadSettings, LuaError> {
    let chunk_name = match argument(args, 1) {
        Value::String(name) => vm.heap_bytes(name).to_vec(),
        _ => default_name,
    };
    let mode = mode_argument(vm, args, 2, "load")?;
    let environment = if args.len() >= 4 { Some(args[3]) } else { None };
    Ok(LoadSettings {
        chunk_name,
        mode,
        environment,
    })
}

fn mode_argument(vm: &mut Vm, args: &[Value], index: usize, name: &str) -> Result<Vec<u8>, LuaError> {
    let mode = match argument(args, index) {
        Value::String(mode) => vm.heap_bytes(mode).to_vec(),
        _ => return Ok(b"bt".to_vec()),
    };
    if mode.contains(&b'B') {
        return Err(arg_error(vm, index + 1, name, "invalid mode"));
    }
    Ok(mode)
}

fn load_source(vm: &mut Vm, source: &[u8], settings: &LoadSettings) -> Result<Values, LuaError> {
    if source.first() == Some(&0x1b) {
        if !settings.mode.contains(&b'b') {
            return Ok(load_failure(vm, &mode_message("binary", &settings.mode)));
        }
        return Ok(match vm.load_binary(&settings.chunk_name, source, settings.environment) {
            Ok(function) => values![function],
            Err(message) => load_failure(vm, &message),
        });
    }
    if !settings.mode.contains(&b't') {
        return Ok(load_failure(vm, &mode_message("text", &settings.mode)));
    }
    let flag = vm.interrupt_handle();
    match vm.compile_chunk_with(&settings.chunk_name, source, settings.environment, Some(&flag)) {
        Ok(function) => Ok(values![function]),
        Err(message) => {
            vm.check_raised()?;
            Ok(values![Value::Nil, vm.syntax_failure_value(message)])
        }
    }
}

fn mode_message(kind: &str, mode: &[u8]) -> Vec<u8> {
    let prefix = format!("attempt to load a {kind} chunk (mode is '");
    [prefix.as_bytes(), mode, &b"')"[..]].concat()
}

fn load_failure(vm: &mut Vm, message: &[u8]) -> Values {
    values![Value::Nil, vm.string_value(message)]
}

fn file_name_argument(vm: &mut Vm, args: &[Value], name: &str) -> Result<Option<Vec<u8>>, LuaError> {
    match argument(args, 0) {
        Value::Nil => Ok(None),
        Value::String(file) => Ok(Some(vm.heap_bytes(file).to_vec())),
        value @ (Value::Integer(_) | Value::Float(_)) => Ok(Some(vm.to_display_bytes(value))),
        _ => {
            let got = args.first().map_or("no value", |value| value.type_name());
            Err(arg_error(vm, 1, name, &format!("string expected, got {got}")))
        }
    }
}

fn chunk_name_of(file: Option<&[u8]>) -> Vec<u8> {
    match file {
        Some(name) => [b"@".as_slice(), name].concat(),
        None => b"=stdin".to_vec(),
    }
}

fn loadfile(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let file = file_name_argument(vm, args, "loadfile")?;
    let mode = mode_argument(vm, args, 1, "loadfile")?;
    let settings = LoadSettings {
        chunk_name: chunk_name_of(file.as_deref()),
        mode,
        environment: if args.len() >= 3 { Some(args[2]) } else { None },
    };
    load_file(vm, file.as_deref(), &settings)
}

fn dofile(vm: &mut Vm, args: &[Value]) -> Result<Step, LuaError> {
    let file = file_name_argument(vm, args, "dofile")?;
    let settings = LoadSettings {
        chunk_name: chunk_name_of(file.as_deref()),
        mode: b"bt".to_vec(),
        environment: None,
    };
    let results = load_file(vm, file.as_deref(), &settings)?;
    match results.first() {
        Some(Value::Nil) => Err(LuaError::from_value(results.get(1).copied().unwrap_or(Value::Nil))),
        Some(function) => Ok(Step::Call {
            function: *function,
            arguments: Values::new(),
            resume: Resume::All,
        }),
        None => Ok(Step::Done(Values::new())),
    }
}

fn load_file(vm: &mut Vm, file: Option<&[u8]>, settings: &LoadSettings) -> Result<Values, LuaError> {
    match read_source(vm, file) {
        Ok(source) => load_source(vm, &skip_prefix(&source), settings),
        Err(message) => Ok(load_failure(vm, &message)),
    }
}

fn read_source(vm: &mut Vm, file: Option<&[u8]>) -> Result<Vec<u8>, Vec<u8>> {
    let mut source = Vec::new();
    let mut buffer = [0u8; 4096];
    let Some(name) = file else {
        loop {
            let count = vm.host.stdio.read_stdin(&mut buffer);
            if count == 0 {
                return Ok(source);
            }
            source.extend_from_slice(&buffer[..count]);
        }
    };
    let text = read_file(vm, name, b"r", &mut source, &mut buffer)?;
    if text.first() != Some(&0x1b) {
        return Ok(text);
    }
    source.clear();
    read_file(vm, name, b"rb", &mut source, &mut buffer)
}

/// A binary chunk is reopened in binary mode, as PUC does, so CRLF bytes survive text-mode translation.
fn read_file(vm: &mut Vm, name: &[u8], mode: &[u8], source: &mut Vec<u8>, buffer: &mut [u8]) -> Result<Vec<u8>, Vec<u8>> {
    let mut handle = match vm.host.files.open(name, mode) {
        Ok(handle) => handle,
        Err(error) => return Err(format!("cannot open {}: {}", String::from_utf8_lossy(name), error.reason).into_bytes()),
    };
    loop {
        let count = handle
            .read(buffer)
            .map_err(|_| format!("cannot read {}", String::from_utf8_lossy(name)).into_bytes())?;
        if count == 0 {
            return Ok(source.clone());
        }
        source.extend_from_slice(&buffer[..count]);
    }
}

fn skip_prefix(source: &[u8]) -> Vec<u8> {
    let mut body = source.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(source);
    if body.first() == Some(&b'#') {
        let end = body.iter().position(|&byte| byte == b'\n').map_or(body.len(), |at| at + 1);
        body = &body[end..];
        if body.first() == Some(&0x1b) {
            return body.to_vec();
        }
        return [b"\n".as_slice(), body].concat();
    }
    body.to_vec()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum WarningState {
    Off,
    On,
    Continuing,
}

impl Vm {
    pub fn warning(&mut self, message: &[u8], tocont: bool) {
        if self.warnings != WarningState::Continuing {
            if !tocont && message.first() == Some(&b'@') {
                match &message[1..] {
                    b"on" => self.warnings = WarningState::On,
                    b"off" => self.warnings = WarningState::Off,
                    _ => {}
                }
                return;
            }
            if self.warnings == WarningState::Off {
                return;
            }
            self.write_error(b"Lua warning: ");
        }
        self.write_error(message);
        if tocont {
            self.warnings = WarningState::Continuing;
        } else {
            self.write_error(b"\n");
            self.warnings = WarningState::On;
        }
    }
}

fn warn(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let mut pieces = Vec::with_capacity(args.len());
    for (index, value) in args.iter().enumerate() {
        match value {
            Value::String(string) => pieces.push(vm.heap_bytes(*string).to_vec()),
            Value::Integer(_) | Value::Float(_) => pieces.push(vm.to_display_bytes(*value)),
            _ => {
                let got = value.type_name();
                return Err(arg_error(vm, index + 1, "warn", &format!("string expected, got {got}")));
            }
        }
    }
    if pieces.is_empty() {
        return Err(arg_error(vm, 1, "warn", "string expected, got no value"));
    }
    let last = pieces.len() - 1;
    for (index, piece) in pieces.iter().enumerate() {
        vm.warning(piece, index < last);
    }
    Ok(Values::new())
}

const GC_OPTIONS: [&str; 9] = [
    "stop",
    "restart",
    "collect",
    "count",
    "step",
    "isrunning",
    "generational",
    "incremental",
    "param",
];

fn collectgarbage(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let option = match argument(args, 0) {
        Value::Nil => b"collect".to_vec(),
        Value::String(name) => vm.heap_bytes(name).to_vec(),
        _ => {
            let got = args.first().map_or("no value", |value| value.type_name());
            return Err(arg_error(vm, 1, "collectgarbage", &format!("string expected, got {got}")));
        }
    };
    if !GC_OPTIONS.iter().any(|known| known.as_bytes() == option.as_slice()) {
        let name = String::from_utf8_lossy(&option).into_owned();
        return Err(arg_error(vm, 1, "collectgarbage", &format!("invalid option '{name}'")));
    }
    if vm.gc_stopped_by_collector() {
        return Ok(values![Value::Nil]);
    }
    let result = match option.as_slice() {
        b"collect" => {
            vm.gc_collect();
            Value::Integer(0)
        }
        b"stop" => {
            vm.gc_stop();
            Value::Integer(0)
        }
        b"restart" => {
            vm.gc_restart();
            Value::Integer(0)
        }
        b"count" => {
            let total = vm.gc_total_bytes();
            Value::Float((total >> 10) as f64 + (total & 0x3ff) as f64 / 1024.0)
        }
        b"step" => {
            let data = optional_integer(vm, args, 1)?;
            Value::Boolean(vm.gc_step_by(data))
        }
        b"param" => {
            let param = gc_param_name(vm, args)?;
            let value = match argument(args, 2) {
                Value::Nil => -1,
                _ => check_integer(vm, args, 2, "collectgarbage")?,
            };
            Value::Integer(vm.gc_param(param, value))
        }
        b"isrunning" => Value::Boolean(vm.gc_is_running()),
        b"generational" => {
            let previous = vm.gc_change_mode(Kind::GenMinor);
            mode_value(vm, previous)
        }
        b"incremental" => {
            let previous = vm.gc_change_mode(Kind::Inc);
            mode_value(vm, previous)
        }
        _ => Value::Integer(0),
    };
    Ok(values![result])
}

fn gc_param_name(vm: &mut Vm, args: &[Value]) -> Result<Param, LuaError> {
    let name = match argument(args, 1) {
        Value::String(name) => vm.heap_bytes(name).to_vec(),
        _ => {
            let got = args.get(1).map_or("no value", |value| value.type_name());
            return Err(arg_error(vm, 2, "collectgarbage", &format!("string expected, got {got}")));
        }
    };
    let param = match name.as_slice() {
        b"minormul" => Param::MinorMul,
        b"majorminor" => Param::MajorMinor,
        b"minormajor" => Param::MinorMajor,
        b"pause" => Param::Pause,
        b"stepmul" => Param::StepMul,
        b"stepsize" => Param::StepSize,
        _ => {
            let name = String::from_utf8_lossy(&name).into_owned();
            return Err(arg_error(vm, 2, "collectgarbage", &format!("invalid option '{name}'")));
        }
    };
    Ok(param)
}

fn mode_value(vm: &mut Vm, generational: bool) -> Value {
    let name: &[u8] = if generational { b"generational" } else { b"incremental" };
    vm.string_value(name)
}

fn optional_integer(vm: &mut Vm, args: &[Value], index: usize) -> Result<i64, LuaError> {
    if argument(args, index).is_nil() {
        return Ok(0);
    }
    check_integer(vm, args, index, "collectgarbage")
}

impl Vm {
    /// Stdin is compiled under `=stdin`.
    pub fn load_file(&mut self, file: Option<&[u8]>) -> Result<Value, Vec<u8>> {
        let settings = LoadSettings {
            chunk_name: chunk_name_of(file),
            mode: b"bt".to_vec(),
            environment: None,
        };
        let results = match load_file(self, file, &settings) {
            Ok(results) => results,
            Err(error) => return Err(self.error_message(&error).to_vec()),
        };
        match results.first().copied() {
            Some(Value::Nil) | None => {
                let message = match results.get(1) {
                    Some(Value::String(string)) => self.heap_bytes(*string).to_vec(),
                    _ => Vec::new(),
                };
                Err(message)
            }
            Some(function) => Ok(function),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{Options, Vm};

    fn output_of(source: &str) -> String {
        let mut vm = Vm::new(Options::default());
        if let Err(error) = vm.run_source(b"=t", source.as_bytes()) {
            return String::from_utf8_lossy(&vm.error_message(&error)).into_owned();
        }
        String::from_utf8_lossy(&vm.take_output()).into_owned()
    }

    #[test]
    fn ipairs_step_gets_its_state_and_control_and_wraps() {
        assert_eq!(output_of("local f = ipairs{}\nprint(f({1}, 0))"), "1\t1\n");
        assert_eq!(output_of("local f = ipairs{}\nprint(f({[1]=5}, 0))"), "1\t5\n");
        assert_eq!(output_of("local f = ipairs{}\nprint(f({1}, math.maxinteger))"), "nil\n");
    }

    #[test]
    fn a_computed_table_key_does_not_leave_a_register_in_the_argument_list() {
        assert_eq!(output_of("print(select('#', {[math.mininteger]=10}, 0))"), "2\n");
        assert_eq!(output_of("print(select('#', {[1]=10}, 0))"), "2\n");
    }

    #[test]
    fn a_lua_call_reuses_its_argument_locals_in_a_multiple_assignment() {
        assert_eq!(output_of("local function g(a, b) return b end\nlocal k, v = g(1, 2)\nk, v = g({}, k)\nprint(k, v)"), "2\tnil\n");
    }
}
