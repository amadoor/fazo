pub(crate) mod base;
pub(crate) mod debug;
pub(crate) mod coroutine;
pub(crate) mod machine;
pub(crate) mod io;
mod jitlib;
mod math;
mod os;
mod package;
mod pattern;
pub(crate) mod printf;
mod strpack;
pub(crate) mod string;
mod tablelib;
mod utf8;

use crate::{
    LuaError, Values, values, Vm,
    arena::Ref,
    host::{ExitStatus, HostError},
    number::{self, Number, Rounding},
    object::{LuaString, Resume, Step},
    table::Table,
    value::Value,
};

pub(super) fn failure_values(vm: &mut Vm, error: &HostError, name: Option<&[u8]>) -> Values {
    let mut message = Vec::new();
    if let Some(name) = name {
        message.extend_from_slice(name);
        message.extend_from_slice(b": ");
    }
    message.extend_from_slice(error.reason.as_bytes());
    values![Value::Nil, vm.string_value(&message), Value::Integer(i64::from(error.errno))]
}

pub(super) fn exec_result(vm: &mut Vm, status: ExitStatus) -> Values {
    let (ok, kind, code) = match status {
        ExitStatus::Exited(code) => (code == 0, "exit", code),
        ExitStatus::Signaled(signal) => (false, "signal", signal),
    };
    let first = if ok { Value::Boolean(true) } else { Value::Nil };
    values![first, vm.string_value(kind.as_bytes()), Value::Integer(i64::from(code))]
}

pub(crate) fn open(vm: &mut Vm) {
    base::open(vm);
    math::open(vm);
    string::open(vm);
    tablelib::open(vm);
    utf8::open(vm);
    coroutine::open(vm);
    debug::open(vm);
    if vm.options().jit_library {
        jitlib::open(vm);
    }
    if vm.options().os {
        os::open(vm);
    }
    if vm.options().io {
        io::open(vm);
    }
    if vm.options().package {
        package::open(vm);
    }
}

#[inline(never)]
pub(crate) fn resume(vm: &mut Vm, state: Resume, values: Values) -> Result<Step, LuaError> {
    match state {
        Resume::ToString => base::resume_tostring(vm, values),
        Resume::Pairs => base::resume_pairs(vm, values),
        Resume::Ipairs(index) => base::resume_ipairs(vm, index, values),
        Resume::Print(print) => base::resume_print(vm, *print, values),
        Resume::Format(format) => string::resume_format(vm, *format, values),
        Resume::Gsub(gsub) => string::resume_gsub(vm, *gsub, values),
        Resume::Machine(machine) => machine::resume(vm, *machine, values),
        Resume::Protected(_) => base::resume_pcall(values),
        Resume::Unwinding(state) => Ok(vm.resume_unwinding(state)),
        Resume::Raised | Resume::Hook(_) => Ok(Step::Done(values)),
        Resume::All => Ok(Step::Done(values)),
        Resume::First => Ok(Step::Done(values![values.first().copied().unwrap_or(Value::Nil)])),
    }
}

pub(super) fn argument(args: &[Value], index: usize) -> Value {
    args.get(index).copied().unwrap_or(Value::Nil)
}

pub(super) fn arg_error(vm: &mut Vm, index: usize, name: &str, message: &str) -> LuaError {
    vm.argument_error(index, name, message)
}

pub(super) fn string_ref(vm: &mut Vm, args: &[Value], index: usize, name: &str) -> Result<Ref<LuaString>, LuaError> {
    match argument(args, index) {
        Value::String(string) if index < args.len() => Ok(string),
        value @ (Value::Integer(_) | Value::Float(_)) => {
            let bytes = vm.to_display_bytes(value);
            match vm.string_value(&bytes) {
                Value::String(string) => Ok(string),
                _ => unreachable!("a string value holds a string"),
            }
        }
        _ => Err(string_expected(vm, args, index, name)),
    }
}

fn string_expected(vm: &mut Vm, args: &[Value], index: usize, name: &str) -> LuaError {
    let got = if index < args.len() { args[index].type_name() } else { "no value" };
    arg_error(vm, index + 1, name, &format!("string expected, got {got}"))
}

pub(super) fn got(vm: &mut Vm, args: &[Value], index: usize) -> String {
    let Some(value) = args.get(index).copied() else {
        return "no value".to_string();
    };
    if let Value::LightUserdata(_) = value {
        return "light userdata".to_string();
    }
    if let Some(metatable) = vm.metatable_of(value)
        && let Value::String(name) = vm.raw_special_field(metatable, crate::vm::Field::Name)
    {
        return String::from_utf8_lossy(vm.heap_bytes(name)).into_owned();
    }
    value.type_name().to_string()
}

pub(super) fn check_any(vm: &mut Vm, args: &[Value], index: usize, name: &str) -> Result<(), LuaError> {
    if args.len() <= index {
        return Err(arg_error(vm, index + 1, name, "value expected"));
    }
    Ok(())
}

pub(super) fn check_table(vm: &mut Vm, args: &[Value], index: usize, name: &str) -> Result<Ref<Table>, LuaError> {
    match argument(args, index) {
        Value::Table(table) if index < args.len() => Ok(table),
        _ => {
            let message = format!("table expected, got {}", got(vm, args, index));
            Err(arg_error(vm, index + 1, name, &message))
        }
    }
}

#[inline]
pub(super) fn check_number(vm: &mut Vm, args: &[Value], index: usize, name: &str) -> Result<Number, LuaError> {
    match argument(args, index) {
        Value::Integer(integer) => Ok(Number::Integer(integer)),
        Value::Float(float) => Ok(Number::Float(float)),
        _ => convert_number(vm, args, index, name),
    }
}

#[inline(never)]
fn convert_number(vm: &mut Vm, args: &[Value], index: usize, name: &str) -> Result<Number, LuaError> {
    let number = match argument(args, index) {
        Value::String(string) if index < args.len() => number::parse_number(vm.heap_bytes(string)),
        _ => None,
    };
    match number {
        Some(number) => Ok(number),
        None => {
            let message = format!("number expected, got {}", got(vm, args, index));
            Err(arg_error(vm, index + 1, name, &message))
        }
    }
}

#[inline]
pub(super) fn check_integer(vm: &mut Vm, args: &[Value], index: usize, name: &str) -> Result<i64, LuaError> {
    if let Some(&Value::Integer(integer)) = args.get(index) {
        return Ok(integer);
    }
    convert_integer(vm, args, index, name)
}

#[inline(never)]
fn convert_integer(vm: &mut Vm, args: &[Value], index: usize, name: &str) -> Result<i64, LuaError> {
    let number = check_number(vm, args, index, name)?;
    match number::to_integer(number, Rounding::Exact) {
        Some(integer) => Ok(integer),
        None => Err(arg_error(vm, index + 1, name, "number has no integer representation")),
    }
}
