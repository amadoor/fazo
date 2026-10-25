use crate::vm::Event;
use super::{
    arg_error, argument, check_any, check_integer, pattern, printf, string_ref, strpack,
};
use crate::{
    LuaError, Values, values, Vm,
    arena::Ref,
    instruction::ArithOp,
    number::{self, Number},
    object::{LuaString, NativeBody, Resume, Step},
    value::Value,
    vm::{Builtin, IndexStep},
};

use super::base::{Piece, tostring_piece};
use super::pattern::{Capture, Match, PatternError};
use crate::vm::interrupt::Interrupts;
use printf::Spec;

const MAX_SIZE: u128 = i64::MAX as u128;
const CHECK_STRIDE: usize = 1024;

pub(super) fn open(vm: &mut Vm) {
    vm.register_builtin("string.gmatch", NativeBody::Plain(gmatch_iterator));
    let table = vm.new_table();
    let functions: &[(&'static str, crate::object::NativeFn)] = &[
        ("byte", byte),
        ("char", char),
        ("dump", dump),
        ("find", find),
        ("gmatch", gmatch),
        ("len", len),
        ("lower", lower),
        ("match", string_match),
        ("rep", rep),
        ("reverse", reverse),
        ("sub", sub),
        ("upper", upper),
        ("pack", strpack::pack),
        ("packsize", strpack::packsize),
        ("unpack", strpack::unpack),
    ];
    for (name, function) in functions {
        let value = vm.new_native(name, *function);
        let builtin = match *name {
            "byte" => Some(Builtin::StringByte),
            "sub" => Some(Builtin::StringSub),
            "len" => Some(Builtin::StringLen),
            _ => None,
        };
        if let Some(builtin) = builtin {
            vm.mark_builtin(value, builtin);
        }
        vm.set_field(table, name, value);
    }
    let format_value = vm.new_stepped("format", format);
    vm.set_field(table, "format", format_value);
    let gsub_value = vm.new_stepped("gsub", gsub);
    vm.set_field(table, "gsub", gsub_value);
    vm.set_global("string", Value::Table(table));

    let metatable = vm.new_table();
    vm.set_field(metatable, "__index", Value::Table(table));
    let arithmetic: &[(&'static str, crate::object::SteppedFn)] = &[
        ("__add", add),
        ("__sub", sub_metamethod),
        ("__mul", mul),
        ("__mod", modulo),
        ("__pow", pow),
        ("__div", div),
        ("__idiv", idiv),
        ("__unm", unm),
    ];
    for (name, function) in arithmetic {
        let value = vm.new_stepped(name, *function);
        vm.set_field(metatable, name, value);
    }
    vm.string_metatable = Some(metatable);
}

fn owned_text(vm: &mut Vm, args: &[Value], index: usize, name: &str) -> Result<Vec<u8>, LuaError> {
    let text = string_ref(vm, args, index, name)?;
    Ok(vm.heap_bytes(text).to_vec())
}

fn substring(vm: &mut Vm, text: Ref<LuaString>, start: usize, end: usize) -> Value {
    vm.substring_value(text, start, end)
}

fn start_position(position: i64, length: usize) -> usize {
    let length = length as i64;
    if position > 0 {
        position as usize
    } else if position == 0 || position < -length {
        1
    } else {
        (length + position + 1) as usize
    }
}

fn end_position(position: i64, length: usize) -> usize {
    let length = length as i64;
    if position > length {
        length as usize
    } else if position >= 0 {
        position as usize
    } else if position < -length {
        0
    } else {
        (length + position + 1) as usize
    }
}

fn optional_integer(vm: &mut Vm, args: &[Value], index: usize, name: &str, default: i64) -> Result<i64, LuaError> {
    if argument(args, index).is_nil() {
        return Ok(default);
    }
    check_integer(vm, args, index, name)
}

fn len(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let text = string_ref(vm, args, 0, "len")?;
    Ok(values![Value::Integer(vm.string_length(text) as i64)])
}

fn sub(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let text = string_ref(vm, args, 0, "sub")?;
    let length = vm.string_length(text);
    let start = start_position(optional_integer(vm, args, 1, "sub", 1)?, length);
    let end = end_position(optional_integer(vm, args, 2, "sub", -1)?, length);
    if start > end {
        return Ok(values![vm.string_value(b"")]);
    }
    Ok(values![substring(vm, text, start - 1, end)])
}

fn lower(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let text = string_ref(vm, args, 0, "lower")?;
    let lowered = vm.heap_bytes(text).to_ascii_lowercase();
    Ok(values![vm.string_value(&lowered)])
}

fn upper(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let text = string_ref(vm, args, 0, "upper")?;
    let uppered = vm.heap_bytes(text).to_ascii_uppercase();
    Ok(values![vm.string_value(&uppered)])
}

fn reverse(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let text = string_ref(vm, args, 0, "reverse")?;
    let reversed: Vec<u8> = vm.heap_bytes(text).iter().rev().copied().collect();
    Ok(values![vm.string_value(&reversed)])
}

fn rep(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let text = string_ref(vm, args, 0, "rep")?;
    let count = check_integer(vm, args, 1, "rep")?;
    let separator = if argument(args, 2).is_nil() {
        None
    } else {
        Some(string_ref(vm, args, 2, "rep")?)
    };
    let separator_length = separator.map_or(0, |separator| vm.string_length(separator));
    let unit = (vm.string_length(text) + separator_length) as u128;
    if count <= 0 || unit == 0 {
        return Ok(values![vm.string_value(b"")]);
    }
    if count as u128 > MAX_SIZE / unit {
        return Err(vm.runtime_error_text("resulting string too large"));
    }
    let size = unit as usize * count as usize;
    vm.reserve(size)?;
    let pooled = size <= crate::heap::SHORT_STRING_LIMIT;
    let mut out = if pooled { vm.take_text_buffer() } else { Vec::new() };
    if out.try_reserve_exact(size).is_err() {
        return Err(vm.memory_error());
    }
    let mut interrupts = Interrupts::new(vm);
    for index in 0..count {
        vm.check_interrupt(&mut interrupts)?;
        if index > 0 && let Some(separator) = separator {
            out.extend_from_slice(vm.heap_bytes(separator));
        }
        out.extend_from_slice(vm.heap_bytes(text));
    }
    if pooled {
        return Ok(values![vm.text_buffer_value(out)]);
    }
    Ok(values![Value::String(vm.heap.string_from_buffer(out))])
}

fn byte(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let text = string_ref(vm, args, 0, "byte")?;
    let length = vm.string_length(text);
    let first = optional_integer(vm, args, 1, "byte", 1)?;
    let start = start_position(first, length);
    let end = end_position(optional_integer(vm, args, 2, "byte", first)?, length);
    if start > end {
        return Ok(Values::new());
    }
    let mut values = Values::new();
    let mut position = start - 1;
    while position < end {
        vm.check_raised()?;
        let stop = (position + CHECK_STRIDE).min(end);
        values.extend(vm.heap_bytes(text)[position..stop].iter().map(|&byte| Value::Integer(i64::from(byte))));
        position = stop;
    }
    Ok(values)
}

fn char(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let mut out = Vec::with_capacity(args.len());
    for index in 0..args.len() {
        let code = check_integer(vm, args, index, "char")?;
        let Ok(byte) = u8::try_from(code) else {
            return Err(arg_error(vm, index + 1, "char", "value out of range"));
        };
        out.push(byte);
    }
    Ok(values![vm.string_value(&out)])
}

fn dump(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    check_any(vm, args, 0, "dump")?;
    let Value::Closure(closure) = args[0] else {
        return Err(arg_error(vm, 1, "dump", "Lua function expected"));
    };
    let strip = !argument(args, 1).is_falsy();
    let bytes = vm.dump_closure(closure, strip);
    Ok(values![vm.string_value(&bytes)])
}

fn capture_value(vm: &mut Vm, piece: impl Fn(&mut Vm, usize, usize) -> Value, capture: &Capture) -> Value {
    match *capture {
        Capture::Span { start, end } => piece(vm, start, end),
        Capture::Position(position) => Value::Integer(position as i64),
    }
}

fn match_values(vm: &mut Vm, found: &Match, piece: impl Fn(&mut Vm, usize, usize) -> Value) -> Values {
    if found.captures.is_empty() {
        return values![piece(vm, found.start, found.end)];
    }
    found.captures.iter().map(|capture| capture_value(vm, &piece, capture)).collect()
}

fn pattern_error(vm: &mut Vm, error: PatternError) -> LuaError {
    match error {
        PatternError::Message(message) => vm.runtime_error_text(&message),
        PatternError::Interrupted => vm.interrupted(),
    }
}

fn find(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    find_aux(vm, args, true)
}

fn string_match(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    find_aux(vm, args, false)
}

fn find_aux(vm: &mut Vm, args: &[Value], find: bool) -> Result<Values, LuaError> {
    let name = if find { "find" } else { "match" };
    let subject = string_ref(vm, args, 0, name)?;
    let pattern_ref = string_ref(vm, args, 1, name)?;
    let length = vm.heap_bytes(subject).len();
    let init = start_position(optional_integer(vm, args, 2, name, 1)?, length) - 1;
    if init > length {
        return Ok(values![Value::Nil]);
    }
    let plain = !argument(args, 3).is_falsy();
    if find && (plain || pattern::is_plain(vm.heap_bytes(pattern_ref))) {
        return Ok(plain_find(vm, subject, pattern_ref, init));
    }
    let mut interrupts = Interrupts::new(vm);
    let found = pattern::search(vm.heap_bytes(subject), vm.heap_bytes(pattern_ref), init, &mut interrupts)
        .map_err(|error| pattern_error(vm, error))?;
    let Some(found) = found else {
        return Ok(values![Value::Nil]);
    };
    let piece = |vm: &mut Vm, start: usize, end: usize| substring(vm, subject, start, end);
    if !find {
        return Ok(match_values(vm, &found, piece));
    }
    let mut results = values![
        Value::Integer(found.start as i64 + 1),
        Value::Integer(found.end as i64),
    ];
    results.extend(found.captures.iter().map(|capture| capture_value(vm, piece, capture)));
    Ok(results)
}

fn plain_find(vm: &Vm, subject: Ref<LuaString>, pattern_ref: Ref<LuaString>, init: usize) -> Values {
    let text = vm.heap_bytes(subject);
    let needle = vm.heap_bytes(pattern_ref);
    let position = if needle.is_empty() {
        Some(0)
    } else {
        memchr::memmem::find(&text[init..], needle)
    };
    match position {
        Some(offset) => {
            let start = init + offset;
            values![
                Value::Integer(start as i64 + 1),
                Value::Integer((start + needle.len()) as i64),
            ]
        }
        None => values![Value::Nil],
    }
}

fn gmatch(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let subject = string_ref(vm, args, 0, "gmatch")?;
    let pattern_ref = string_ref(vm, args, 1, "gmatch")?;
    let length = vm.heap_bytes(subject).len();
    let init = start_position(optional_integer(vm, args, 2, "gmatch", 1)?, length) - 1;
    let start = init.min(length + 1);
    let upvalues = vec![
        Value::String(subject),
        Value::String(pattern_ref),
        Value::Integer(start as i64),
        Value::Nil,
    ];
    let iterator = vm.new_native_with("string.gmatch", NativeBody::Plain(gmatch_iterator), upvalues);
    Ok(values![iterator])
}

fn gmatch_iterator(vm: &mut Vm, _args: &[Value]) -> Result<Values, LuaError> {
    vm.check_raised()?;
    let upvalues = vm.native_upvalues();
    let (Value::String(subject), Value::String(pattern_ref)) = (upvalues[0], upvalues[1]) else {
        unreachable!("gmatch stores its subject and pattern as strings");
    };
    let Value::Integer(start) = upvalues[2] else {
        unreachable!("gmatch keeps its position as an integer");
    };
    let last_match = match upvalues[3] {
        Value::Integer(end) => Some(end as usize),
        _ => None,
    };
    let Some(found) = next_gmatch_match(vm, subject, pattern_ref, start as usize, last_match)? else {
        return Ok(values![]);
    };
    vm.set_native_upvalue(2, Value::Integer(found.end as i64));
    vm.set_native_upvalue(3, Value::Integer(found.end as i64));
    Ok(match_values(vm, &found, |vm, start, end| substring(vm, subject, start, end)))
}

fn next_gmatch_match(
    vm: &mut Vm,
    subject: Ref<LuaString>,
    pattern_ref: Ref<LuaString>,
    start: usize,
    last_match: Option<usize>,
) -> Result<Option<Match>, LuaError> {
    let length = vm.heap_bytes(subject).len();
    let lead = pattern::required_lead(vm.heap_bytes(pattern_ref));
    let mut interrupts = Interrupts::new(vm);
    let mut position = start;
    loop {
        position = pattern::next_start(vm.heap_bytes(subject), vm.heap_bytes(pattern_ref), lead, position);
        if position > length {
            return Ok(None);
        }
        let found = pattern::match_at(vm.heap_bytes(subject), vm.heap_bytes(pattern_ref), position, &mut interrupts)
            .map_err(|error| pattern_error(vm, error))?;
        position += 1;
        let Some(found) = found else {
            continue;
        };
        if Some(found.end) != last_match {
            return Ok(Some(found));
        }
    }
}

impl Replacement {
    pub(crate) fn values(&self, out: &mut Values) {
        if let Replacement::Table(value) | Replacement::Function(value) = self {
            out.push(*value);
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum Replacement {
    Text(#[serde(with = "crate::bytes")] Vec<u8>),
    Table(Value),
    Function(Value),
}

impl GsubState {
    pub(crate) fn values(&self, out: &mut Values) {
        out.push(self.original);
        self.replacement.values(out);
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GsubState {
    #[serde(with = "crate::bytes")]
    subject: Vec<u8>,
    original: Value,
    changed: bool,
    #[serde(with = "crate::bytes")]
    pattern: Vec<u8>,
    anchored: bool,
    replacement: Replacement,
    position: usize,
    last_match: Option<usize>,
    #[serde(with = "crate::bytes")]
    output: Vec<u8>,
    count: i64,
    limit: i64,
    started: bool,
    pending: Option<(usize, usize)>,
}

fn gsub(vm: &mut Vm, args: &[Value]) -> Result<Step, LuaError> {
    let subject = owned_text(vm, args, 0, "gsub")?;
    let pattern_bytes = owned_text(vm, args, 1, "gsub")?;
    let replacement = match argument(args, 2) {
        Value::String(_) | Value::Integer(_) | Value::Float(_) => {
            Replacement::Text(vm.to_display_bytes(argument(args, 2)))
        }
        Value::Table(_) => Replacement::Table(args[2]),
        Value::Closure(_) | Value::Native(_) => Replacement::Function(args[2]),
        _ => {
            let got = if args.len() > 2 { args[2].type_name() } else { "no value" };
            let message = format!("string/function/table expected, got {got}");
            return Err(arg_error(vm, 3, "gsub", &message));
        }
    };
    let limit = optional_integer(vm, args, 3, "gsub", subject.len() as i64 + 1)?;
    let anchored = pattern_bytes.first() == Some(&b'^');
    let pattern = if anchored { pattern_bytes[1..].to_vec() } else { pattern_bytes };
    let state = GsubState {
        subject,
        original: argument(args, 0),
        changed: false,
        pattern,
        anchored,
        replacement,
        position: 0,
        last_match: None,
        output: Vec::new(),
        count: 0,
        limit,
        started: false,
        pending: None,
    };
    gsub_from(vm, state)
}

pub(crate) fn resume_gsub(vm: &mut Vm, mut state: GsubState, values: Values) -> Result<Step, LuaError> {
    let result = values.first().copied().unwrap_or(Value::Nil);
    apply_result(vm, &mut state, result)?;
    gsub_from(vm, state)
}

fn apply_result(vm: &mut Vm, state: &mut GsubState, result: Value) -> Result<(), LuaError> {
    let Some((start, end)) = state.pending.take() else {
        return Ok(());
    };
    match result {
        Value::Nil | Value::Boolean(false) => state.output.extend_from_slice(&state.subject[start..end]),
        Value::String(_) | Value::Integer(_) | Value::Float(_) => {
            let text = vm.to_display_bytes(result);
            state.output.extend_from_slice(&text);
            state.changed = true;
        }
        other => {
            let message = format!("invalid replacement value (a {})", other.type_name());
            return Err(vm.runtime_error_text(&message));
        }
    }
    Ok(())
}

fn gsub_from(vm: &mut Vm, mut state: GsubState) -> Result<Step, LuaError> {
    let mut interrupts = Interrupts::new(vm);
    let lead = if state.anchored { None } else { pattern::required_lead(&state.pattern) };
    while state.count < state.limit {
        if state.anchored && state.started {
            break;
        }
        state.started = true;
        let next = pattern::next_start(&state.subject, &state.pattern, lead, state.position);
        let copied = next.min(state.subject.len());
        state.output.extend_from_slice(&state.subject[state.position..copied]);
        state.position = copied;
        if next > state.subject.len() {
            break;
        }
        let found = pattern::match_at(&state.subject, &state.pattern, state.position, &mut interrupts)
            .map_err(|error| pattern_error(vm, error))?;
        let found = found.filter(|found| Some(found.end) != state.last_match);
        let Some(found) = found else {
            if state.position >= state.subject.len() {
                break;
            }
            state.output.push(state.subject[state.position]);
            state.position += 1;
            continue;
        };
        state.count += 1;
        state.position = found.end;
        state.last_match = Some(found.end);
        if let Some((function, arguments)) = add_replacement(vm, &mut state, &found)? {
            return Ok(Step::Call {
                function,
                arguments,
                resume: Resume::Gsub(Box::new(state)),
            });
        }
    }
    if !state.changed {
        return Ok(Step::Done(values![state.original, Value::Integer(state.count)]));
    }
    let rest = state.subject[state.position..].to_vec();
    state.output.extend_from_slice(&rest);
    let output = vm.string_value(&state.output);
    Ok(Step::Done(values![output, Value::Integer(state.count)]))
}

fn add_replacement(vm: &mut Vm, state: &mut GsubState, found: &Match) -> Result<Option<(Value, Values)>, LuaError> {
    if let Replacement::Text(text) = &mut state.replacement {
        let text = std::mem::take(text);
        let added = add_string_replacement(vm, state, found, &text);
        state.replacement = Replacement::Text(text);
        added?;
        return Ok(None);
    }
    match state.replacement.clone() {
        Replacement::Text(text) => {
            add_string_replacement(vm, state, found, &text)?;
            Ok(None)
        }
        Replacement::Function(function) => {
            let arguments = match_values(vm, found, |vm, start, end| vm.string_value(&state.subject[start..end]));
            state.pending = Some((found.start, found.end));
            Ok(Some((function, arguments)))
        }
        Replacement::Table(table) => {
            let key = match_values(vm, found, |vm, start, end| vm.string_value(&state.subject[start..end])).swap_remove(0);
            state.pending = Some((found.start, found.end));
            match vm.index_chain(table, key)? {
                IndexStep::Value(result) => {
                    apply_result(vm, state, result)?;
                    Ok(None)
                }
                IndexStep::Call { function, arguments } => Ok(Some((function, arguments.into()))),
            }
        }
    }
}

fn add_string_replacement(vm: &mut Vm, state: &mut GsubState, found: &Match, text: &[u8]) -> Result<(), LuaError> {
    state.changed = true;
    let mut index = 0;
    while index < text.len() {
        let byte = text[index];
        index += 1;
        if byte != b'%' {
            state.output.push(byte);
            continue;
        }
        let Some(&escape) = text.get(index) else {
            return Err(vm.runtime_error_text("invalid use of '%' in replacement string"));
        };
        index += 1;
        if escape == b'%' {
            state.output.push(b'%');
            continue;
        }
        if !escape.is_ascii_digit() {
            return Err(vm.runtime_error_text("invalid use of '%' in replacement string"));
        }
        let value = capture_for_digit(vm, state, found, escape)?;
        let piece = vm.to_display_bytes(value);
        state.output.extend_from_slice(&piece);
    }
    Ok(())
}

fn capture_for_digit(vm: &mut Vm, state: &GsubState, found: &Match, digit: u8) -> Result<Value, LuaError> {
    let piece = |vm: &mut Vm, start: usize, end: usize| vm.string_value(&state.subject[start..end]);
    if digit == b'0' {
        return Ok(piece(vm, found.start, found.end));
    }
    let index = (digit - b'1') as usize;
    if found.captures.is_empty() && index == 0 {
        return Ok(piece(vm, found.start, found.end));
    }
    match found.captures.get(index) {
        Some(capture) => Ok(capture_value(vm, piece, capture)),
        None => {
            let message = format!("invalid capture index %{}", index + 1);
            Err(vm.runtime_error_text(&message))
        }
    }
}

impl FormatState {
    pub(crate) fn values(&self, out: &mut Values) {
        out.extend_from_slice(&self.arguments);
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FormatState {
    #[serde(with = "crate::bytes")]
    format: Vec<u8>,
    position: usize,
    arguments: Values,
    next: usize,
    #[serde(with = "crate::bytes")]
    output: Vec<u8>,
    pending: Option<Spec>,
}

fn format(vm: &mut Vm, args: &[Value]) -> Result<Step, LuaError> {
    let text = string_ref(vm, args, 0, "format")?;
    let mut format_bytes = vm.take_text_buffer();
    format_bytes.extend_from_slice(vm.heap_bytes(text));
    let state = FormatState {
        format: format_bytes,
        position: 0,
        arguments: Values::from_slice(args),
        next: 1,
        output: vm.take_text_buffer(),
        pending: None,
    };
    format_from(vm, state)
}

pub(crate) fn resume_format(vm: &mut Vm, mut state: FormatState, values: Values) -> Result<Step, LuaError> {
    let Some(spec) = state.pending.take() else {
        return format_from(vm, state);
    };
    let result = values.first().copied().unwrap_or(Value::Nil);
    let text = match result {
        Value::String(_) | Value::Integer(_) | Value::Float(_) => vm.to_display_bytes(result),
        _ => return Err(vm.runtime_error_text("'__tostring' must return a string")),
    };
    string_conversion(vm, &mut state.output, &spec, state.next - 1, &text)?;
    format_from(vm, state)
}

fn has_modifiers(spec: &Spec) -> bool {
    spec.minus || spec.plus || spec.space || spec.hash || spec.zero || spec.width > 0 || spec.precision.is_some()
}

fn string_conversion(vm: &mut Vm, out: &mut Vec<u8>, spec: &Spec, argument_index: usize, text: &[u8]) -> Result<(), LuaError> {
    if has_modifiers(spec) && text.contains(&0) {
        return Err(arg_error(vm, argument_index + 1, "format", "string contains zeros"));
    }
    printf::format_string(out, spec, text);
    Ok(())
}

fn plain_conversion(vm: &mut Vm, out: &mut Vec<u8>, spec: &Spec, argument_index: usize, value: Value) -> Result<bool, LuaError> {
    if !matches!(value, Value::String(_) | Value::Integer(_)) || !vm.metamethod(value, crate::vm::Event::ToString).is_nil() {
        return Ok(false);
    }
    if let Value::Integer(integer) = value {
        printf::format_string(out, spec, &printf::decimal(integer));
        return Ok(true);
    }
    let Value::String(string) = value else {
        return Ok(false);
    };
    if has_modifiers(spec) && vm.heap_bytes(string).contains(&0) {
        return Err(arg_error(vm, argument_index + 1, "format", "string contains zeros"));
    }
    printf::format_string(out, spec, vm.heap_bytes(string));
    Ok(true)
}

fn format_from(vm: &mut Vm, mut state: FormatState) -> Result<Step, LuaError> {
    let mut interrupts = Interrupts::new(vm);
    while state.position < state.format.len() {
        vm.check_interrupt(&mut interrupts)?;
        let byte = state.format[state.position];
        if byte != b'%' {
            let rest = &state.format[state.position..];
            let window = &rest[..rest.len().min(CHECK_STRIDE)];
            let literal = window.iter().position(|&byte| byte == b'%').unwrap_or(window.len());
            state.output.extend_from_slice(&rest[..literal]);
            state.position += literal;
            continue;
        }
        if state.format.get(state.position + 1) == Some(&b'%') {
            state.output.push(b'%');
            state.position += 2;
            continue;
        }
        let rest = &state.format[state.position + 1..];
        let (spec, consumed) = match printf::parse_spec(rest) {
            Ok(parsed) => parsed,
            Err(message) => return Err(vm.runtime_error(&message)),
        };
        state.position += 1 + consumed;
        let argument_index = state.next;
        state.next += 1;
        if argument_index >= state.arguments.len() {
            return Err(arg_error(vm, argument_index + 1, "format", "no value"));
        }
        let value = state.arguments[argument_index];
        if spec.conversion == b's' {
            if plain_conversion(vm, &mut state.output, &spec, argument_index, value)? {
                continue;
            }
            match tostring_piece(vm, value) {
                Piece::Text(text) => string_conversion(vm, &mut state.output, &spec, argument_index, &text)?,
                Piece::Call(handler) => {
                    state.pending = Some(spec);
                    return Ok(Step::Call {
                        function: handler,
                        arguments: values![value],
                        resume: Resume::Format(Box::new(state)),
                    });
                }
            }
            continue;
        }
        convert_argument(vm, &mut state.output, &spec, argument_index, value)?;
    }
    vm.give_text_buffer(state.format);
    let output = vm.text_buffer_value(state.output);
    Ok(Step::Done(values![output]))
}

fn convert_argument(vm: &mut Vm, out: &mut Vec<u8>, spec: &Spec, argument_index: usize, value: Value) -> Result<(), LuaError> {
    let number_position = argument_index + 1;
    match spec.conversion {
        b'c' | b'd' | b'i' | b'u' | b'o' | b'x' | b'X' => {
            let integer = integer_for_format(vm, value, number_position)?;
            printf::format_integer(out, spec, integer);
        }
        b'a' | b'A' => {
            if cfg!(windows) && has_modifiers(spec) {
                return Err(vm.runtime_error_text("modifiers for format '%a'/'%A' not implemented"));
            }
            let float = float_for_format(vm, value, number_position)?;
            printf::format_float(out, spec, float);
        }
        b'e' | b'E' | b'f' | b'g' | b'G' => {
            let float = float_for_format(vm, value, number_position)?;
            printf::format_float(out, spec, float);
        }
        b'q' => out.extend_from_slice(&quote_argument(vm, value, number_position)?),
        b'p' => {
            let text = pointer_text(vm, value);
            printf::format_string(out, spec, &text);
        }
        _ => unreachable!("parse_spec only returns known conversions"),
    }
    Ok(())
}

fn integer_for_format(vm: &mut Vm, value: Value, position: usize) -> Result<i64, LuaError> {
    match vm.coerce_number(value) {
        Some(number) => match number::to_integer(number, number::Rounding::Exact) {
            Some(integer) => Ok(integer),
            None => Err(arg_error(vm, position, "format", "number has no integer representation")),
        },
        None => {
            let got = value.type_name();
            Err(arg_error(vm, position, "format", &format!("number expected, got {got}")))
        }
    }
}

fn float_for_format(vm: &mut Vm, value: Value, position: usize) -> Result<f64, LuaError> {
    match vm.coerce_number(value) {
        Some(number) => Ok(Number::to_float(number)),
        None => {
            let got = value.type_name();
            Err(arg_error(vm, position, "format", &format!("number expected, got {got}")))
        }
    }
}

fn quote_argument(vm: &mut Vm, value: Value, position: usize) -> Result<Vec<u8>, LuaError> {
    match value {
        Value::String(string) => Ok(printf::quote_string(vm.heap_bytes(string))),
        Value::Integer(integer) => Ok(printf::quote_integer(integer)),
        Value::Float(float) => Ok(printf::quote_float(float)),
        Value::Nil | Value::Boolean(_) => Ok(vm.to_display_bytes(value)),
        _ => Err(arg_error(vm, position, "format", "value has no literal form")),
    }
}

fn pointer_text(vm: &mut Vm, value: Value) -> Vec<u8> {
    match value {
        Value::String(string) => format!("{:#010x}", crate::vm::table_address(string)).into_bytes(),
        Value::Table(_) | Value::Closure(_) | Value::Native(_) | Value::Thread(_) | Value::Userdata(_) => {
            let text = vm.to_display_bytes(value);
            match text.iter().position(|&byte| byte == b' ') {
                Some(at) => text[at + 1..].to_vec(),
                None => text,
            }
        }
        _ => b"(null)".to_vec(),
    }
}

fn string_arithmetic(vm: &mut Vm, args: &[Value], op: ArithOp, event: Event) -> Result<Step, LuaError> {
    let left = numeric_operand(vm, argument(args, 0));
    let right = numeric_operand(vm, argument(args, 1));
    if let (Some(left), Some(right)) = (left, right) {
        let value = vm.arith_on_numbers(op, left, right)?;
        return Ok(Step::Done(values![value]));
    }
    let second = argument(args, 1);
    let handler = if matches!(second, Value::String(_)) {
        Value::Nil
    } else {
        vm.metamethod(second, event)
    };
    if handler.is_nil() {
        let message = format!(
            "attempt to {} a '{}' with a '{}'",
            &event.name()[2..],
            argument(args, 0).type_name(),
            second.type_name()
        );
        return Err(vm.runtime_error_text(&message));
    }
    Ok(Step::Call {
        function: handler,
        arguments: values![argument(args, 0), second],
        resume: Resume::First,
    })
}

fn numeric_operand(vm: &Vm, value: Value) -> Option<Number> {
    match value {
        Value::Integer(_) | Value::Float(_) | Value::String(_) => vm.coerce_number(value),
        _ => None,
    }
}

fn add(vm: &mut Vm, args: &[Value]) -> Result<Step, LuaError> {
    string_arithmetic(vm, args, ArithOp::Add, Event::Add)
}

fn sub_metamethod(vm: &mut Vm, args: &[Value]) -> Result<Step, LuaError> {
    string_arithmetic(vm, args, ArithOp::Sub, Event::Sub)
}

fn mul(vm: &mut Vm, args: &[Value]) -> Result<Step, LuaError> {
    string_arithmetic(vm, args, ArithOp::Mul, Event::Mul)
}

fn modulo(vm: &mut Vm, args: &[Value]) -> Result<Step, LuaError> {
    string_arithmetic(vm, args, ArithOp::Mod, Event::Mod)
}

fn pow(vm: &mut Vm, args: &[Value]) -> Result<Step, LuaError> {
    string_arithmetic(vm, args, ArithOp::Pow, Event::Pow)
}

fn div(vm: &mut Vm, args: &[Value]) -> Result<Step, LuaError> {
    string_arithmetic(vm, args, ArithOp::Div, Event::Div)
}

fn idiv(vm: &mut Vm, args: &[Value]) -> Result<Step, LuaError> {
    string_arithmetic(vm, args, ArithOp::IDiv, Event::IDiv)
}

fn unm(vm: &mut Vm, args: &[Value]) -> Result<Step, LuaError> {
    let operand = numeric_operand(vm, argument(args, 0));
    match operand {
        Some(number) => Ok(Step::Done(values![number::negate(number).to_value()])),
        None => string_arithmetic(vm, args, ArithOp::Sub, Event::Unm),
    }
}
