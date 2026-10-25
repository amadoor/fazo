
use super::{arg_error, argument, check_any, check_integer, string_ref};
use crate::{
    LuaError, Values, values, Vm,
    arena::Ref,
    object::{LuaString, NativeFn},
    value::Value,
    vm::interrupt::Interrupts,
};

const MAX_UNICODE: u32 = 0x10_ffff;
const MAX_UTF: u32 = 0x7fff_ffff;
const INVALID: &str = "invalid UTF-8 code";

pub(super) fn open(vm: &mut Vm) {
    let functions: &[(&'static str, NativeFn)] = &[
        ("offset", byte_offset),
        ("codepoint", codepoint),
        ("char", utf_char),
        ("len", utf_len),
    ];
    let table = vm.new_table();
    for (name, function) in functions {
        let value = vm.new_native(name, *function);
        vm.set_field(table, name, value);
    }
    let codes = vm.new_native("codes", codes);
    vm.set_field(table, "codes", codes);
    let pattern = vm.string_value(b"[\0-\x7F\xC2-\xFD][\x80-\xBF]*");
    vm.set_field(table, "charpattern", pattern);
    let strict = vm.new_native("codes_strict", iterate_strict);
    vm.set_hidden("utf8_strict", strict);
    let lax = vm.new_native("codes_lax", iterate_lax);
    vm.set_hidden("utf8_lax", lax);
    vm.set_global("utf8", Value::Table(table));
}

fn is_continuation(byte: u8) -> bool {
    byte & 0xc0 == 0x80
}

fn byte_at(bytes: &[u8], index: usize) -> u8 {
    bytes.get(index).copied().unwrap_or(0)
}

fn text_byte(vm: &Vm, text: Ref<LuaString>, index: i64) -> u8 {
    byte_at(vm.heap_bytes(text), index as usize)
}

fn relative_position(position: i64, length: usize) -> i64 {
    if position >= 0 {
        position
    } else if position.unsigned_abs() > length as u64 {
        0
    } else {
        length as i64 + position + 1
    }
}

fn decode(bytes: &[u8], start: usize, strict: bool) -> Option<(u32, usize)> {
    const LIMITS: [u32; 6] = [u32::MAX, 0x80, 0x800, 0x10000, 0x20_0000, 0x400_0000];
    let mut lead = u32::from(byte_at(bytes, start));
    let mut result: u32;
    let mut next = start + 1;
    if lead < 0x80 {
        result = lead;
    } else {
        let mut count = 0usize;
        result = 0;
        while lead & 0x40 != 0 {
            count += 1;
            let continuation = byte_at(bytes, start + count);
            if !is_continuation(continuation) {
                return None;
            }
            result = (result << 6) | u32::from(continuation & 0x3f);
            lead <<= 1;
        }
        if count > 5 {
            return None;
        }
        result |= (lead & 0x7f) << (count * 5);
        if result > MAX_UTF || result < LIMITS[count] {
            return None;
        }
        next = start + count + 1;
    }
    if strict && (result > MAX_UNICODE || (0xd800..=0xdfff).contains(&result)) {
        return None;
    }
    Some((result, next))
}

pub(crate) fn encode(code: u32) -> Vec<u8> {
    if code < 0x80 {
        return vec![code as u8];
    }
    let mut tail = Vec::new();
    let mut remaining = code;
    let mut limit = 0x3fu32;
    while remaining > limit {
        tail.push(0x80 | (remaining & 0x3f) as u8);
        remaining >>= 6;
        limit >>= 1;
    }
    let lead = ((!limit << 1) | remaining) as u8;
    let mut bytes = vec![lead];
    bytes.extend(tail.iter().rev());
    bytes
}


fn utf_len(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let text = string_ref(vm, args, 0, "len")?;
    let length = vm.string_length(text);
    let mut start = relative_position(optional(vm, args, 1, "len", 1)?, length);
    let end = relative_position(optional(vm, args, 2, "len", -1)?, length);
    let lax = !argument(args, 3).is_falsy();
    if !(1 <= start && start - 1 <= length as i64) {
        return Err(arg_error(vm, 2, "len", "initial position out of bounds"));
    }
    start -= 1;
    if end > length as i64 {
        return Err(arg_error(vm, 3, "len", "final position out of bounds"));
    }
    let last = end - 1;
    let mut count = 0;
    let mut position = start;
    let mut interrupts = Interrupts::new(vm);
    while position <= last {
        vm.check_interrupt(&mut interrupts)?;
        match decode(vm.heap_bytes(text), position as usize, !lax) {
            Some((_, next)) => {
                position = next as i64;
                count += 1;
            }
            None => return Ok(values![Value::Nil, Value::Integer(position + 1)]),
        }
    }
    Ok(values![Value::Integer(count)])
}

fn optional(vm: &mut Vm, args: &[Value], index: usize, name: &str, default: i64) -> Result<i64, LuaError> {
    if argument(args, index).is_nil() {
        return Ok(default);
    }
    check_integer(vm, args, index, name)
}

fn codepoint(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let text = string_ref(vm, args, 0, "codepoint")?;
    let length = vm.string_length(text);
    let start = relative_position(optional(vm, args, 1, "codepoint", 1)?, length);
    let end = relative_position(optional(vm, args, 2, "codepoint", start)?, length);
    let lax = !argument(args, 3).is_falsy();
    if start < 1 {
        return Err(arg_error(vm, 2, "codepoint", "out of bounds"));
    }
    if end > length as i64 {
        return Err(arg_error(vm, 3, "codepoint", "out of bounds"));
    }
    if start > end {
        return Ok(Values::new());
    }
    let mut results = Values::new();
    let mut position = (start - 1) as usize;
    let stop = end as usize;
    let mut interrupts = Interrupts::new(vm);
    while position < stop {
        vm.check_interrupt(&mut interrupts)?;
        match decode(vm.heap_bytes(text), position, !lax) {
            Some((code, next)) => {
                results.push(Value::Integer(i64::from(code)));
                position = next;
            }
            None => return Err(vm.runtime_error_text(INVALID)),
        }
    }
    Ok(results)
}

fn utf_char(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let mut bytes = Vec::new();
    for index in 0..args.len() {
        let code = check_integer(vm, args, index, "char")?;
        if !(0..=i64::from(MAX_UTF)).contains(&code) {
            return Err(arg_error(vm, index + 1, "char", "value out of range"));
        }
        bytes.extend(encode(code as u32));
    }
    Ok(values![vm.string_value(&bytes)])
}

fn byte_offset(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let text = string_ref(vm, args, 0, "offset")?;
    let length = vm.string_length(text);
    let mut count = check_integer(vm, args, 1, "offset")?;
    let default = if count >= 0 { 1 } else { length as i64 + 1 };
    let mut position = relative_position(optional(vm, args, 2, "offset", default)?, length);
    if !(1 <= position && position - 1 <= length as i64) {
        return Err(arg_error(vm, 3, "offset", "position out of bounds"));
    }
    position -= 1;
    if count == 0 {
        while position > 0 && is_continuation(text_byte(vm, text, position)) {
            position -= 1;
        }
    } else {
        if is_continuation(text_byte(vm, text, position)) {
            return Err(vm.runtime_error_text("initial position is a continuation byte"));
        }
        let mut interrupts = Interrupts::new(vm);
        if count < 0 {
            while count < 0 && position > 0 {
                vm.check_interrupt(&mut interrupts)?;
                position -= 1;
                while position > 0 && is_continuation(text_byte(vm, text, position)) {
                    position -= 1;
                }
                count += 1;
            }
        } else {
            count -= 1;
            while count > 0 && position < length as i64 {
                vm.check_interrupt(&mut interrupts)?;
                position += 1;
                while is_continuation(text_byte(vm, text, position)) {
                    position += 1;
                }
                count -= 1;
            }
        }
    }
    if count != 0 {
        return Ok(values![Value::Nil]);
    }
    let mut last = position;
    if text_byte(vm, text, position) & 0x80 != 0 {
        if is_continuation(text_byte(vm, text, position)) {
            return Err(vm.runtime_error_text("initial position is a continuation byte"));
        }
        while is_continuation(text_byte(vm, text, last + 1)) {
            last += 1;
        }
    }
    Ok(values![Value::Integer(position + 1), Value::Integer(last + 1)])
}

fn codes(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let lax = !argument(args, 1).is_falsy();
    let text = string_ref(vm, args, 0, "codes")?;
    if vm.heap_bytes(text).first().is_some_and(|&byte| is_continuation(byte)) {
        return Err(arg_error(vm, 1, "codes", INVALID));
    }
    let iterator = if lax {
        vm.hidden("utf8_lax")
    } else {
        vm.hidden("utf8_strict")
    };
    Ok(values![iterator, Value::String(text), Value::Integer(0)])
}

fn iterate_strict(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    iterate(vm, args, true)
}

fn iterate_lax(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    iterate(vm, args, false)
}

fn iterate(vm: &mut Vm, args: &[Value], strict: bool) -> Result<Values, LuaError> {
    vm.check_raised()?;
    check_any(vm, args, 0, "codes")?;
    let text = string_ref(vm, args, 0, "codes")?;
    let length = vm.string_length(text);
    let start = match argument(args, 1) {
        Value::Integer(index) => index as u64 as usize,
        _ => 0,
    };
    let mut position = start;
    if position < length {
        while is_continuation(byte_at(vm.heap_bytes(text), position)) {
            position += 1;
        }
    }
    if position >= length {
        return Ok(Values::new());
    }
    match decode(vm.heap_bytes(text), position, strict) {
        Some((code, next)) if !is_continuation(byte_at(vm.heap_bytes(text), next)) => {
            Ok(values![Value::Integer(position as i64 + 1), Value::Integer(i64::from(code))])
        }
        _ => Err(vm.runtime_error_text(INVALID)),
    }
}

