use crate::vm::Event;
use super::{arg_error, argument, check_any, check_integer, got};
use crate::{
    LuaError, Values, values, Vm,
    arena::Ref,
    object::{Hook, MASK_COUNT, Thread},
    table::Table,
    value::Value,
    vm::{Activation, ClosureFacts, mask_from_text, mask_text},
};

pub(super) fn open(vm: &mut Vm) {
    let table = vm.new_table();
    let plain: &[(&'static str, crate::object::NativeFn)] = &[
        ("debug", debug_debug),
        ("sethook", sethook),
        ("gethook", gethook),
        ("traceback", traceback),
        ("getinfo", getinfo),
        ("getlocal", getlocal),
        ("setlocal", setlocal),
        ("getupvalue", getupvalue),
        ("setupvalue", setupvalue),
        ("upvalueid", upvalueid),
        ("upvaluejoin", upvaluejoin),
        ("getmetatable", getmetatable),
        ("setmetatable", setmetatable),
        ("getregistry", getregistry),
        ("getuservalue", getuservalue),
        ("setuservalue", setuservalue),
    ];
    for (name, function) in plain {
        let value = vm.new_native(name, *function);
        vm.set_field(table, name, value);
    }
    vm.set_global("debug", Value::Table(table));
}

fn read_debug_line(vm: &mut Vm) -> Option<Vec<u8>> {
    const LINE_LIMIT: usize = 249;
    let mut line = Vec::new();
    let mut byte = [0u8; 1];
    while line.len() < LINE_LIMIT && vm.host.stdio.read_stdin(&mut byte) == 1 {
        line.push(byte[0]);
        if byte[0] == b'\n' {
            break;
        }
    }
    (!line.is_empty()).then_some(line)
}

fn debug_debug(vm: &mut Vm, _args: &[Value]) -> Result<Values, LuaError> {
    loop {
        vm.host.stdio.write_stderr(b"lua_debug> ");
        let Some(line) = read_debug_line(vm) else {
            return Ok(Values::new());
        };
        if line == b"cont\n" {
            return Ok(Values::new());
        }
        let outcome = match vm.compile_chunk(b"=(debug command)", &line, None) {
            Ok(function) => vm.call_protected_as(function, &[], None, false).err().map(|error| vm.error_message(&error)),
            Err(message) => Some(message),
        };
        if let Some(mut message) = outcome {
            message.push(b'\n');
            vm.host.stdio.write_stderr(&message);
        }
    }
}

fn getregistry(vm: &mut Vm, _args: &[Value]) -> Result<Values, LuaError> {
    Ok(values![Value::Table(vm.registry())])
}

fn user_value_index(vm: &mut Vm, args: &[Value], index: usize, name: &str) -> Result<i64, LuaError> {
    match argument(args, index) {
        Value::Nil => Ok(1),
        _ => check_integer(vm, args, index, name),
    }
}

fn getuservalue(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let number = user_value_index(vm, args, 1, "getuservalue")?;
    let Value::Userdata(userdata) = argument(args, 0) else {
        return Ok(values![Value::Nil]);
    };
    let values = &vm.heap.userdata.get(userdata).user_values;
    match usize::try_from(number).ok().filter(|number| (1..=values.len()).contains(number)) {
        Some(number) => Ok(values![values[number - 1], Value::Boolean(true)]),
        None => Ok(values![Value::Nil]),
    }
}

fn setuservalue(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let number = user_value_index(vm, args, 2, "setuservalue")?;
    let Value::Userdata(userdata) = argument(args, 0) else {
        let message = format!("userdata expected, got {}", got(vm, args, 0));
        return Err(arg_error(vm, 1, "setuservalue", &message));
    };
    check_any(vm, args, 1, "setuservalue")?;
    let value = argument(args, 1);
    let slot = usize::try_from(number).ok().filter(|number| (1..=vm.heap.userdata.get(userdata).user_values.len()).contains(number));
    match slot {
        Some(number) => {
            vm.heap.set_userdata_value(userdata, number - 1, value);
            Ok(values![Value::Userdata(userdata)])
        }
        None => Ok(values![Value::Nil]),
    }
}

fn store_hook(vm: &mut Vm, thread: Ref<Thread>, function: Value) -> Result<(), LuaError> {
    let registry = vm.registry();
    let key = vm.string_value(b"_HOOKKEY");
    let table = match vm.table_get(registry, key) {
        Value::Table(table) => table,
        _ => {
            let table = vm.new_table();
            let mode = vm.new_table();
            let weak = vm.string_value(b"k");
            vm.set_field(mode, "__mode", weak);
            vm.attach_metatable(table, Some(mode));
            vm.set_field(registry, "_HOOKKEY", Value::Table(table));
            table
        }
    };
    vm.table_set(table, Value::Thread(thread), function)
}

fn sethook(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let (thread, rest) = split_thread(vm, args);
    let function = argument(rest, 0);
    if function.is_nil() {
        store_hook(vm, thread, Value::Nil)?;
        vm.set_hook(thread, None);
        return Ok(Values::new());
    }
    let text = match argument(rest, 1) {
        Value::String(string) => vm.heap_bytes(string).to_vec(),
        _ => {
            let message = format!("string expected, got {}", got(vm, rest, 1));
            return Err(arg_error(vm, 2, "sethook", &message));
        }
    };
    if !function.is_function() {
        let offset = args.len() - rest.len();
        let message = format!("function expected, got {}", got(vm, rest, 0));
        return Err(arg_error(vm, offset + 1, "sethook", &message));
    }
    let count = match argument(rest, 2) {
        Value::Nil => 0,
        _ => check_integer(vm, rest, 2, "sethook")?,
    };
    let mut mask = mask_from_text(&text);
    if count > 0 {
        mask |= MASK_COUNT;
    }
    let hook = (mask != 0).then_some(Hook {
        function,
        mask,
        base_count: count,
    });
    store_hook(vm, thread, function)?;
    vm.set_hook(thread, hook);
    Ok(Values::new())
}

fn gethook(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let (thread, _) = split_thread(vm, args);
    let Some(hook) = vm.hook_of(thread) else {
        return Ok(values![Value::Nil]);
    };
    let mask = vm.string_value(&mask_text(hook.mask));
    Ok(values![hook.function, mask, Value::Integer(hook.base_count)])
}

fn split_thread<'a>(vm: &Vm, args: &'a [Value]) -> (Ref<Thread>, &'a [Value]) {
    match argument(args, 0) {
        Value::Thread(thread) => (thread, &args[1..]),
        _ => (vm.current_thread(), args),
    }
}

fn traceback(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let (thread, rest) = split_thread(vm, args);
    let default_level = if thread == vm.current_thread() { 1 } else { 0 };
    let message = match argument(rest, 0) {
        Value::Nil => None,
        Value::String(_) | Value::Integer(_) | Value::Float(_) => Some(vm.to_display_bytes(argument(rest, 0))),
        other => return Ok(values![other]),
    };
    let level = match argument(rest, 1) {
        Value::Integer(level) => level,
        Value::Float(level) => level as i64,
        _ => default_level,
    };
    let text = vm.traceback_of(thread, message.as_deref(), level);
    Ok(values![vm.string_value(&text)])
}

pub(crate) fn message_handler(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let error = argument(args, 0);
    let message = match error {
        Value::String(_) | Value::Integer(_) | Value::Float(_) => vm.to_display_bytes(error),
        _ => {
            let handler = vm.metamethod(error, Event::ToString);
            if !handler.is_nil()
                && let Some(Value::String(text)) = vm.call_value(handler, &[error])?.first().copied()
            {
                return Ok(values![Value::String(text)]);
            }
            format!("(error object is a {} value)", error.type_name()).into_bytes()
        }
    };
    let current = vm.current_thread();
    let text = vm.traceback_of(current, Some(&message), 1);
    Ok(values![vm.string_value(&text)])
}

const INFO_OPTIONS: &[u8] = b"SlnrutfL";

fn getinfo(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let (thread, rest) = split_thread(vm, args);
    let options = match argument(rest, 1) {
        Value::Nil => b"flnSrtu".to_vec(),
        Value::String(string) => vm.heap_bytes(string).to_vec(),
        _ => {
            let message = format!("string expected, got {}", got(vm, rest, 1));
            return Err(arg_error(vm, 2, "getinfo", &message));
        }
    };
    if options.first() == Some(&b'>') {
        return Err(arg_error(vm, 2, "getinfo", "invalid option '>'"));
    }
    if options.iter().any(|option| !INFO_OPTIONS.contains(option)) {
        return Err(arg_error(vm, 2, "getinfo", "invalid option"));
    }
    let table = vm.new_table();
    if argument(rest, 0).is_function() {
        describe_function(vm, table, argument(rest, 0));
    } else {
        let level = check_integer(vm, rest, 0, "getinfo")?;
        let Some(activation) = usize::try_from(level).ok().and_then(|level| vm.activation(thread, level)) else {
            return Ok(values![Value::Nil]);
        };
        let level = level as usize;
        match activation {
            Activation::Lua(frame) => {
                let facts = vm.closure_facts(frame.closure);
                let line = vm.activation_line(frame);
                describe_lua(vm, table, Value::Closure(frame.closure), &facts, line, frame.tail);
                vm.set_field(table, "extraargs", Value::Integer(i64::from(frame.ccmt)));
            }
            Activation::Native(function) => {
                describe_native(vm, table, function);
                vm.set_field(table, "extraargs", Value::Integer(0));
            }
        }
        let transfer = vm.transfer_of(thread, level).filter(|transfer| transfer.count > 0);
        let (first, count) = transfer.map_or((0, 0), |transfer| (transfer.start, transfer.count));
        vm.set_field(table, "ftransfer", Value::Integer(first as i64));
        vm.set_field(table, "ntransfer", Value::Integer(count as i64));
        match vm.activation_name(thread, level) {
            Some((kind, name)) => set_name(vm, table, kind, &name),
            None => {
                let namewhat = vm.string_value(b"");
                vm.set_field(table, "namewhat", namewhat);
            }
        }
    }
    prune_info(vm, table, &options);
    Ok(values![Value::Table(table)])
}

fn prune_info(vm: &mut Vm, table: Ref<Table>, options: &[u8]) {
    let groups: &[(u8, &[&str])] = &[
        (b'S', &["source", "short_src", "what", "linedefined", "lastlinedefined"]),
        (b'l', &["currentline"]),
        (b'u', &["nups", "nparams", "isvararg"]),
        (b'n', &["name", "namewhat"]),
        (b't', &["istailcall", "extraargs"]),
        (b'f', &["func"]),
        (b'L', &["activelines"]),
        (b'r', &["ftransfer", "ntransfer"]),
    ];
    for (option, fields) in groups {
        if options.contains(option) {
            continue;
        }
        for field in *fields {
            vm.set_field(table, field, Value::Nil);
        }
    }
}

fn describe_function(vm: &mut Vm, table: Ref<Table>, function: Value) {
    match function {
        Value::Closure(closure) => {
            let facts = vm.closure_facts(closure);
            describe_lua(vm, table, function, &facts, -1, false);
        }
        _ => describe_native(vm, table, function),
    }
    let namewhat = vm.string_value(b"");
    vm.set_field(table, "namewhat", namewhat);
}

fn describe_lua(vm: &mut Vm, table: Ref<Table>, function: Value, facts: &ClosureFacts, line: i64, tail: bool) {
    let source = vm.string_value(&facts.chunkname);
    let short_src = vm.string_value(&facts.source);
    let what = if facts.line_defined == 0 { "main" } else { "Lua" };
    let what = vm.string_value(what.as_bytes());
    vm.set_field(table, "source", source);
    vm.set_field(table, "short_src", short_src);
    vm.set_field(table, "what", what);
    vm.set_field(table, "linedefined", Value::Integer(facts.line_defined));
    vm.set_field(table, "lastlinedefined", Value::Integer(facts.last_line_defined));
    vm.set_field(table, "currentline", Value::Integer(line));
    vm.set_field(table, "nups", Value::Integer(facts.upvalue_count as i64));
    vm.set_field(table, "nparams", Value::Integer(facts.num_params as i64));
    vm.set_field(table, "isvararg", Value::Boolean(facts.is_vararg));
    vm.set_field(table, "istailcall", Value::Boolean(tail));
    vm.set_field(table, "func", function);
    if let Value::Closure(closure) = function {
        let lines = vm.new_table();
        for line in vm.closure_active_lines(closure) {
            vm.table_set(lines, Value::Integer(line), Value::Boolean(true))
                .expect("integer keys are never nil or NaN");
        }
        vm.set_field(table, "activelines", Value::Table(lines));
    }
}

fn describe_native(vm: &mut Vm, table: Ref<Table>, function: Value) {
    let source = vm.string_value(b"=[C]");
    let short = vm.string_value(b"[C]");
    let what = vm.string_value(b"C");
    vm.set_field(table, "source", source);
    vm.set_field(table, "short_src", short);
    vm.set_field(table, "what", what);
    vm.set_field(table, "linedefined", Value::Integer(-1));
    vm.set_field(table, "lastlinedefined", Value::Integer(-1));
    vm.set_field(table, "currentline", Value::Integer(-1));
    let upvalues = match function {
        Value::Native(native) => vm.native_upvalue_count(native),
        _ => 0,
    };
    vm.set_field(table, "nups", Value::Integer(upvalues as i64));
    vm.set_field(table, "nparams", Value::Integer(0));
    vm.set_field(table, "isvararg", Value::Boolean(true));
    vm.set_field(table, "istailcall", Value::Boolean(false));
    vm.set_field(table, "extraargs", Value::Integer(0));
    vm.set_field(table, "func", function);
}

fn set_name(vm: &mut Vm, table: Ref<Table>, kind: &str, name: &[u8]) {
    let name = vm.string_value(name);
    let kind = vm.string_value(kind.as_bytes());
    vm.set_field(table, "name", name);
    vm.set_field(table, "namewhat", kind);
}

fn getlocal(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let (thread, rest) = split_thread(vm, args);
    if argument(rest, 0).is_function() {
        return parameter_name(vm, rest);
    }
    let number = check_integer(vm, rest, 1, "getlocal")?;
    let Some((name, slot)) = local_at(vm, thread, rest, number)? else {
        return Ok(values![Value::Nil]);
    };
    let value = vm.stack_value(thread, slot);
    Ok(values![vm.string_value(&name), value])
}

fn parameter_name(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let number = check_integer(vm, args, 1, "getlocal")?;
    let name = match argument(args, 0) {
        Value::Closure(closure) => usize::try_from(number).ok().and_then(|number| vm.parameter_name(closure, number)),
        _ => None,
    };
    Ok(values![name.map_or(Value::Nil, |name| vm.string_value(&name))])
}

fn setlocal(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let (thread, rest) = split_thread(vm, args);
    let number = check_integer(vm, rest, 1, "setlocal")?;
    let Some((name, slot)) = local_at(vm, thread, rest, number)? else {
        return Ok(values![Value::Nil]);
    };
    vm.set_stack_value(thread, slot, argument(rest, 2));
    Ok(values![vm.string_value(&name)])
}

fn local_at(vm: &mut Vm, thread: Ref<Thread>, rest: &[Value], number: i64) -> Result<Option<(Vec<u8>, usize)>, LuaError> {
    let level = check_integer(vm, rest, 0, "getlocal")?;
    let Some(level) = usize::try_from(level).ok().filter(|level| vm.activation(thread, *level).is_some()) else {
        return Err(arg_error(vm, 1, "getlocal", "level out of range"));
    };
    Ok(vm.activation_local(thread, level, number))
}

fn function_argument(vm: &mut Vm, args: &[Value], name: &str) -> Result<Value, LuaError> {
    let function = argument(args, 0);
    if function.is_function() {
        return Ok(function);
    }
    let message = format!("function expected, got {}", got(vm, args, 0));
    Err(arg_error(vm, 1, name, &message))
}

fn getupvalue(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let function = function_argument(vm, args, "getupvalue")?;
    let number = check_integer(vm, args, 1, "getupvalue")?;
    let index = usize::try_from(number).unwrap_or(0);
    match function {
        Value::Closure(closure) => match vm.closure_upvalue(closure, index) {
            Some((name, upvalue)) => {
                let value = vm.upvalue_value(upvalue);
                Ok(values![vm.string_value(&name), value])
            }
            None => Ok(Values::new()),
        },
        Value::Native(native) => match vm.native_upvalue(native, index) {
            Some(value) => Ok(values![vm.string_value(b""), value]),
            None => Ok(Values::new()),
        },
        _ => Ok(Values::new()),
    }
}

fn setupvalue(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let function = function_argument(vm, args, "setupvalue")?;
    let number = check_integer(vm, args, 1, "setupvalue")?;
    let index = usize::try_from(number).unwrap_or(0);
    let value = argument(args, 2);
    match function {
        Value::Closure(closure) => match vm.closure_upvalue(closure, index) {
            Some((name, upvalue)) => {
                vm.set_upvalue_value(upvalue, value);
                Ok(values![vm.string_value(&name)])
            }
            None => Ok(Values::new()),
        },
        _ => Ok(Values::new()),
    }
}

fn upvalueid(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let function = function_argument(vm, args, "upvalueid")?;
    let number = check_integer(vm, args, 1, "upvalueid")?;
    let index = usize::try_from(number).unwrap_or(0);
    match function {
        Value::Closure(closure) => match vm.closure_upvalue(closure, index) {
            Some((_, upvalue)) => Ok(values![vm.upvalue_identity(upvalue)]),
            None => Ok(values![Value::Nil]),
        },
        Value::Native(native) if index >= 1 && index <= vm.native_upvalue_count(native) => {
            Ok(values![Value::LightUserdata((native.index() << 8) | index)])
        }
        _ => Ok(values![Value::Nil]),
    }
}

fn upvalue_reference(vm: &mut Vm, args: &[Value], function_index: usize) -> Result<(Value, i64), LuaError> {
    let index = check_integer(vm, args, function_index + 1, "upvaluejoin")?;
    let function = argument(args, function_index);
    if !function.is_function() {
        let message = format!("function expected, got {}", got(vm, args, function_index));
        return Err(arg_error(vm, function_index + 1, "upvaluejoin", &message));
    }
    let has_upvalue = match function {
        Value::Closure(closure) => usize::try_from(index)
            .ok()
            .and_then(|position| vm.closure_upvalue(closure, position))
            .is_some(),
        Value::Native(native) => usize::try_from(index)
            .ok()
            .is_some_and(|position| position >= 1 && position <= vm.heap.natives.get(native).upvalues.len()),
        _ => false,
    };
    if !has_upvalue {
        return Err(arg_error(vm, function_index + 2, "upvaluejoin", "invalid upvalue index"));
    }
    Ok((function, index))
}

fn upvaluejoin(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let (target, first) = upvalue_reference(vm, args, 0)?;
    let (source, second) = upvalue_reference(vm, args, 2)?;
    let Value::Closure(target) = target else {
        let message = format!("Lua function expected, got {}", got(vm, args, 0));
        return Err(arg_error(vm, 1, "upvaluejoin", &message));
    };
    let Value::Closure(source) = source else {
        let message = format!("Lua function expected, got {}", got(vm, args, 2));
        return Err(arg_error(vm, 3, "upvaluejoin", &message));
    };
    let Some((_, upvalue)) = usize::try_from(second).ok().and_then(|index| vm.closure_upvalue(source, index)) else {
        return Err(arg_error(vm, 4, "upvaluejoin", "invalid upvalue index"));
    };
    let Some(index) = usize::try_from(first)
        .ok()
        .filter(|index| vm.closure_upvalue(target, *index).is_some())
    else {
        return Err(arg_error(vm, 2, "upvaluejoin", "invalid upvalue index"));
    };
    vm.join_upvalue(target, index, upvalue);
    Ok(Values::new())
}

fn getmetatable(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let value = argument(args, 0);
    Ok(values![match vm.metatable_of(value) {
        Some(metatable) => Value::Table(metatable),
        None => Value::Nil,
    }])
}

fn setmetatable(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let value = argument(args, 0);
    let metatable = match argument(args, 1) {
        Value::Nil => None,
        Value::Table(table) => Some(table),
        _ => {
            let message = format!("nil or table expected, got {}", got(vm, args, 1));
            return Err(arg_error(vm, 2, "setmetatable", &message));
        }
    };
    vm.set_metatable_of(value, metatable);
    Ok(values![value])
}
