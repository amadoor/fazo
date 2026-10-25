use super::{arg_error, argument, got};
use crate::{
    LuaError, Values, values, Vm,
    arena::Ref,
    object::{CoStatus, NativeBody, Thread},
    value::Value,
    vm::Builtin,
};

pub(super) fn open(vm: &mut Vm) {
    let table = vm.new_table();
    let plain: &[(&'static str, crate::object::NativeFn)] = &[
        ("create", create),
        ("status", status),
        ("wrap", wrap),
        ("isyieldable", isyieldable),
        ("running", running),
        ("close", close),
    ];
    for (name, function) in plain {
        let value = vm.new_native(name, *function);
        vm.set_field(table, name, value);
    }
    let resume_function = vm.new_native_with("resume", NativeBody::CoroutineResume, Vec::new());
    let yield_function = vm.new_native_with("coroutine.yield", NativeBody::Yield, Vec::new());
    vm.register_builtin("coroutine.wrap", NativeBody::CoroutineWrap);
    vm.mark_registered_builtin("coroutine.wrap", Builtin::Wrap);
    vm.mark_builtin(resume_function, Builtin::Resume);
    vm.mark_builtin(yield_function, Builtin::Yield);
    vm.set_field(table, "resume", resume_function);
    vm.set_field(table, "yield", yield_function);
    vm.set_global("coroutine", Value::Table(table));
}

pub(crate) fn check_thread(vm: &mut Vm, args: &[Value], name: &str) -> Result<Ref<Thread>, LuaError> {
    match argument(args, 0) {
        Value::Thread(thread) if !args.is_empty() => Ok(thread),
        _ => {
            let message = format!("thread expected, got {}", got(vm, args, 0));
            Err(arg_error(vm, 1, name, &message))
        }
    }
}

fn check_function(vm: &mut Vm, args: &[Value], name: &str) -> Result<(), LuaError> {
    if argument(args, 0).is_function() && !args.is_empty() {
        return Ok(());
    }
    let message = format!("function expected, got {}", got(vm, args, 0));
    Err(arg_error(vm, 1, name, &message))
}

fn create(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    check_function(vm, args, "create")?;
    let thread = vm.create_thread(args[0]);
    Ok(values![Value::Thread(thread)])
}

fn status(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let thread = check_thread(vm, args, "status")?;
    let text: &[u8] = if thread == vm.current_thread() {
        b"running"
    } else {
        match vm.thread_status(thread) {
            CoStatus::Suspended => b"suspended",
            CoStatus::Running => b"running",
            CoStatus::Normal => b"normal",
            CoStatus::Dead => b"dead",
        }
    };
    Ok(values![vm.string_value(text)])
}

fn running(vm: &mut Vm, _args: &[Value]) -> Result<Values, LuaError> {
    let thread = vm.current_thread();
    let main = vm.is_main_thread(thread);
    Ok(values![Value::Thread(thread), Value::Boolean(main)])
}

fn isyieldable(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let thread = if args.is_empty() {
        vm.current_thread()
    } else {
        check_thread(vm, args, "isyieldable")?
    };
    Ok(values![Value::Boolean(vm.is_yieldable(thread))])
}

fn wrap(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    check_function(vm, args, "wrap")?;
    let thread = vm.create_thread(args[0]);
    let function = vm.new_native_with("coroutine.wrap", NativeBody::CoroutineWrap, vec![Value::Thread(thread)]);
    Ok(values![function])
}

fn close(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let thread = if args.is_empty() {
        vm.current_thread()
    } else {
        check_thread(vm, args, "close")?
    };
    let state = match vm.thread_status(thread) {
        CoStatus::Running if vm.is_main_thread(thread) => {
            return Err(vm.runtime_error_text("cannot close main thread"));
        }
        CoStatus::Running => return Err(vm.close_running_thread()),
        CoStatus::Normal => Some("normal"),
        CoStatus::Suspended | CoStatus::Dead => None,
    };
    if let Some(state) = state {
        let message = format!("cannot close a {state} coroutine");
        return Err(vm.runtime_error_text(&message));
    }
    Ok(match vm.close_thread(thread) {
        Ok(()) => values![Value::Boolean(true)],
        Err(error) => values![Value::Boolean(false), error],
    })
}
