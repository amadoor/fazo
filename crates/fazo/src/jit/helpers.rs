use super::JitCtx;

type Entry = extern "C" fn(*mut JitCtx, u32) -> i64;

pub(crate) fn enter(entry: *const u8, ctx: &mut JitCtx, pc: usize) -> i64 {
    debug_assert!(!entry.is_null());
    debug_assert!(pc <= u32::MAX as usize);
    // SAFETY: `entry` is the finalized function of a live unit with signature (context, entry instruction) -> exit code,
    // and the prototype keeps the unit mapped.
    let function = unsafe { std::mem::transmute::<*const u8, Entry>(entry) };
    function(ctx, pc as u32)
}

pub(super) extern "C" fn step(ctx: *mut JitCtx, pc: u32) -> i64 {
    if pending(ctx) {
        return settle_then_entry(ctx, pc, step);
    }
    // SAFETY: the context lives on the driver's stack for the whole call, and its VM pointer names the VM that entered the code.
    let ctx = unsafe { &mut *ctx };
    let vm = unsafe { &mut *ctx.vm };
    vm.jit_step(ctx, pc as usize)
}

pub(super) extern "C" fn safe_point(ctx: *mut JitCtx, target: u32) -> i64 {
    if pending(ctx) {
        return settle_then_entry(ctx, target, safe_point);
    }
    // SAFETY: the context and VM pointers are valid for the duration of the call.
    let ctx = unsafe { &mut *ctx };
    let vm = unsafe { &mut *ctx.vm };
    vm.jit_safe_point(ctx, target as usize)
}

pub(super) extern "C" fn misc(ctx: *mut JitCtx, pc: u32) -> i64 {
    if pending(ctx) {
        return settle_then_entry(ctx, pc, misc);
    }
    // SAFETY: the context and VM pointers are valid for the duration of the call.
    let ctx = unsafe { &mut *ctx };
    let vm = unsafe { &mut *ctx.vm };
    vm.jit_misc(ctx, pc as usize)
}

/// Set on operands that name a constant rather than a register.
pub(crate) const CONSTANT_OPERAND: u32 = 1 << 31;

/// Register indexes are below the prototype's `max_stack`, and `registers` addresses the frame window the driver keeps inside the stack.
fn register(ctx: &JitCtx, index: u32) -> crate::value::Value {
    debug_assert!(index < 256);
    unsafe { *ctx.registers.add(index as usize) }
}

fn constant(ctx: &JitCtx, index: u32) -> crate::value::Value {
    unsafe { *ctx.constants.add(index as usize) }
}

fn write_register(ctx: &JitCtx, index: u32, value: crate::value::Value) {
    unsafe { ctx.registers.add(index as usize).write(value) }
}

fn operand(ctx: &JitCtx, index: u32) -> crate::value::Value {
    if index & CONSTANT_OPERAND != 0 {
        constant(ctx, index & !CONSTANT_OPERAND)
    } else {
        register(ctx, index)
    }
}

#[inline(always)]
fn pending(ctx: *mut JitCtx) -> bool {
    // SAFETY: compiled code passes the live context of the frame calling the helper.
    unsafe { (*ctx).pending != 0 }
}

fn push_pending(ctx: *mut JitCtx) {
    // SAFETY: the context and VM pointers are valid for the duration of the call.
    let ctx = unsafe { &mut *ctx };
    let vm = unsafe { &mut *ctx.vm };
    vm.jit_push_pending(ctx);
}

/// Out of line so the common helper path saves no arguments.
#[cold]
#[inline(never)]
fn settle_then(ctx: *mut JitCtx, pc: u32, a: u32, b: u32, c: u32, helper: super::sites::FastFn) -> i64 {
    push_pending(ctx);
    helper(ctx, pc, a, b, c)
}

#[cold]
#[inline(never)]
fn settle_then_entry(ctx: *mut JitCtx, pc: u32, helper: Entry) -> i64 {
    push_pending(ctx);
    helper(ctx, pc)
}

/// The helpers are `extern "C"`, so a panic aborts at the boundary instead of unwinding into compiled code.
///
/// A `None` from `body` runs the instruction at `pc` through the interpreter.
fn with_vm(ctx: *mut JitCtx, pc: u32, body: impl FnOnce(&mut JitCtx, &mut crate::Vm) -> Option<i64>) -> i64 {
    // SAFETY: the context and VM pointers are valid for the duration of the call.
    let ctx = unsafe { &mut *ctx };
    let vm = unsafe { &mut *ctx.vm };
    match body(ctx, vm) {
        Some(result) => {
            debug_assert!(result < 0 || vm.jit_context_current(ctx), "a helper that continues the frame leaves the context current");
            result
        }
        None => vm.jit_step(ctx, pc as usize),
    }
}

fn next(pc: u32) -> Option<i64> {
    Some(i64::from(pc) + 1)
}

fn read_into(ctx: *mut JitCtx, pc: u32, target: u32, object: u32, key: impl FnOnce(&JitCtx) -> crate::value::Value) -> i64 {
    with_vm(ctx, pc, |ctx, vm| {
        let key = key(ctx);
        let object = register(ctx, object);
        let value = vm.index_resolved(object, key)?;
        write_register(ctx, target, value);
        next(pc)
    })
}

fn write_into(ctx: *mut JitCtx, pc: u32, object: u32, key: impl FnOnce(&JitCtx) -> crate::value::Value, value: u32) -> i64 {
    with_vm(ctx, pc, |ctx, vm| {
        let key = key(ctx);
        let value = operand(ctx, value);
        let object = register(ctx, object);
        vm.jit_raw_set(object, key, value).then(|| i64::from(pc) + 1)
    })
}

pub(super) extern "C" fn get_field(ctx: *mut JitCtx, pc: u32, target: u32, object: u32, key: u32) -> i64 {
    if pending(ctx) {
        return settle_then(ctx, pc, target, object, key, get_field);
    }
    with_vm(ctx, pc, |ctx, vm| {
        let key = constant(ctx, key);
        let object = register(ctx, object);
        let value = vm.jit_get_field(ctx, pc as usize, object, key)?;
        write_register(ctx, target, value);
        next(pc)
    })
}

pub(super) extern "C" fn get_table(ctx: *mut JitCtx, pc: u32, target: u32, object: u32, key: u32) -> i64 {
    if pending(ctx) {
        return settle_then(ctx, pc, target, object, key, get_table);
    }
    read_into(ctx, pc, target, object, |ctx| register(ctx, key))
}

pub(super) extern "C" fn get_int(ctx: *mut JitCtx, pc: u32, target: u32, object: u32, key: u32) -> i64 {
    if pending(ctx) {
        return settle_then(ctx, pc, target, object, key, get_int);
    }
    read_into(ctx, pc, target, object, |_| crate::value::Value::Integer(i64::from(key as i32)))
}

pub(super) extern "C" fn set_field(ctx: *mut JitCtx, pc: u32, object: u32, key: u32, value: u32) -> i64 {
    if pending(ctx) {
        return settle_then(ctx, pc, object, key, value, set_field);
    }
    with_vm(ctx, pc, |ctx, vm| {
        let key = constant(ctx, key);
        let value = operand(ctx, value);
        let object = register(ctx, object);
        vm.jit_set_field(ctx, pc as usize, object, key, value).then(|| i64::from(pc) + 1)
    })
}

pub(super) extern "C" fn set_table(ctx: *mut JitCtx, pc: u32, object: u32, key: u32, value: u32) -> i64 {
    if pending(ctx) {
        return settle_then(ctx, pc, object, key, value, set_table);
    }
    write_into(ctx, pc, object, |ctx| register(ctx, key), value)
}

pub(super) extern "C" fn set_int(ctx: *mut JitCtx, pc: u32, object: u32, key: u32, value: u32) -> i64 {
    if pending(ctx) {
        return settle_then(ctx, pc, object, key, value, set_int);
    }
    write_into(ctx, pc, object, |_| crate::value::Value::Integer(i64::from(key as i32)), value)
}

/// The lookup runs before either register is written, so an object aliasing the target reads its old value.
pub(super) extern "C" fn get_self(ctx: *mut JitCtx, pc: u32, target: u32, object: u32, key: u32) -> i64 {
    if pending(ctx) {
        return settle_then(ctx, pc, target, object, key, get_self);
    }
    with_vm(ctx, pc, |ctx, vm| {
        let key = constant(ctx, key);
        let object = register(ctx, object);
        let method = vm.jit_get_method(ctx, pc as usize, object, key)?;
        write_register(ctx, target + 1, object);
        write_register(ctx, target, method);
        next(pc)
    })
}

pub(super) extern "C" fn table_barrier(ctx: *mut JitCtx, pc: u32, object: u32, value: u32, _unused: u32) -> i64 {
    if pending(ctx) {
        return settle_then(ctx, pc, object, value, _unused, table_barrier);
    }
    with_vm(ctx, pc, |ctx, vm| {
        let value = operand(ctx, value);
        let crate::value::Value::Table(table) = register(ctx, object) else {
            return None;
        };
        vm.jit_table_barrier(table, value);
        next(pc)
    })
}

/// What a call or return helper tells compiled code to do next: `Interpret` runs the instruction through the interpreter.
pub(crate) enum Control {
    Continue,
    Interpret,
    Error,
    Reload,
    Returned,
}

fn control(pc: u32, outcome: Control) -> Option<i64> {
    match outcome {
        Control::Continue => next(pc),
        Control::Interpret => None,
        Control::Error => Some(-super::EXIT_ERROR),
        Control::Reload => Some(-super::EXIT_RELOAD),
        Control::Returned => Some(-super::EXIT_RETURNED),
    }
}

pub(super) extern "C" fn call_direct(ctx: *mut JitCtx, pc: u32, a: u32, b: u32, c: u32) -> i64 {
    if pending(ctx) {
        return settle_then(ctx, pc, a, b, c, call_direct);
    }
    with_vm(ctx, pc, |ctx, vm| control(pc, vm.jit_call_direct(ctx, pc as usize, a as u8, b as u8, c as u8)))
}

/// Ends a direct call whose callee did not return in place; `frames` is the frame count before the call.
pub(super) extern "C" fn after_call(ctx: *mut JitCtx, pc: u32, code: u32, frames: u32, _unused: u32) -> i64 {
    if pending(ctx) {
        return settle_then(ctx, pc, code, frames, _unused, after_call);
    }
    with_vm(ctx, pc, |ctx, vm| Some(vm.jit_after_call(ctx, pc as usize, i64::from(code as i32), frames as usize)))
}

pub(super) extern "C" fn new_table(ctx: *mut JitCtx, pc: u32, a: u32, array: u32, hash_log: u32) -> i64 {
    if pending(ctx) {
        return settle_then(ctx, pc, a, array, hash_log, new_table);
    }
    with_vm(ctx, pc, |ctx, vm| Some(vm.jit_new_table(ctx, pc as usize, a as u8, array, hash_log as u8)))
}

pub(super) extern "C" fn closure(ctx: *mut JitCtx, pc: u32, a: u32, index: u32, _unused: u32) -> i64 {
    if pending(ctx) {
        return settle_then(ctx, pc, a, index, _unused, closure);
    }
    with_vm(ctx, pc, |ctx, vm| Some(vm.jit_closure(ctx, pc as usize, a as u8, index as usize)))
}

pub(super) extern "C" fn set_list(ctx: *mut JitCtx, pc: u32, a: u32, b: u32, c: u32) -> i64 {
    if pending(ctx) {
        return settle_then(ctx, pc, a, b, c, set_list);
    }
    with_vm(ctx, pc, |ctx, vm| Some(vm.jit_set_list(ctx, pc as usize, a as u8, b as u8, c as u8)))
}

/// Closes the frame's upvalues before an inline return; a pending to-be-closed variable leaves the return to the step.
pub(super) extern "C" fn close(ctx: *mut JitCtx, pc: u32, _a: u32, _b: u32, _c: u32) -> i64 {
    if pending(ctx) {
        return settle_then(ctx, pc, _a, _b, _c, close);
    }
    with_vm(ctx, pc, |ctx, vm| vm.jit_close_for_return(ctx).then_some(i64::from(pc)))
}

pub(super) extern "C" fn tail_call(ctx: *mut JitCtx, pc: u32, _a: u32, _b: u32, _c: u32) -> i64 {
    if pending(ctx) {
        return settle_then(ctx, pc, _a, _b, _c, tail_call);
    }
    with_vm(ctx, pc, |ctx, vm| control(pc, vm.jit_tail_call(ctx, pc as usize)))
}

/// A computed result skips the metamethod instruction that follows.
pub(super) extern "C" fn arith(ctx: *mut JitCtx, pc: u32, packed: u32, left: u32, right: u32) -> i64 {
    if pending(ctx) {
        return settle_then(ctx, pc, packed, left, right, arith);
    }
    with_vm(ctx, pc, |ctx, vm| vm.jit_arith(ctx, packed, left, right).then(|| i64::from(pc) + 2))
}

pub(super) extern "C" fn for_prep(ctx: *mut JitCtx, pc: u32, a: u32, bx: u32, _unused: u32) -> i64 {
    if pending(ctx) {
        return settle_then(ctx, pc, a, bx, _unused, for_prep);
    }
    with_vm(ctx, pc, |ctx, vm| {
        let runs = vm.jit_for_prep(ctx, a as u8)?;
        Some(if runs { i64::from(pc) + 1 } else { i64::from(pc) + i64::from(bx) + 2 })
    })
}

/// `count` is `u32::MAX` when the results run up to the stack top.
pub(super) extern "C" fn ret(ctx: *mut JitCtx, pc: u32, first: u32, count: u32, _unused: u32) -> i64 {
    if pending(ctx) {
        return settle_then(ctx, pc, first, count, _unused, ret);
    }
    with_vm(ctx, pc, |ctx, vm| {
        let count = (count != u32::MAX).then_some(count as usize);
        control(pc, vm.jit_return(ctx, pc as usize, first as u8, count))
    })
}

pub(super) extern "C" fn vararg(ctx: *mut JitCtx, pc: u32, a: u32, c: u32, _unused: u32) -> i64 {
    if pending(ctx) {
        return settle_then(ctx, pc, a, c, _unused, vararg);
    }
    with_vm(ctx, pc, |ctx, vm| vm.jit_vararg(ctx, a as u8, c as u8).then(|| i64::from(pc) + 1))
}

/// `next` or `ipairs`'s iterator runs without a call frame, leaving the registers and top as the interpreter's
/// `generic_call` and `generic_builtin` do.
pub(super) extern "C" fn tfor_call(ctx: *mut JitCtx, pc: u32, a: u32, c: u32, _unused: u32) -> i64 {
    if pending(ctx) {
        return settle_then(ctx, pc, a, c, _unused, tfor_call);
    }
    with_vm(ctx, pc, |ctx, vm| {
        let state = register(ctx, a + 1);
        let control = register(ctx, a + 3);
        let found = vm.jit_iterate(register(ctx, a), state, control)?;
        write_register(ctx, a + 5, control);
        write_register(ctx, a + 4, state);
        let placed = match found {
            Some((key, value)) => {
                write_register(ctx, a + 3, key);
                write_register(ctx, a + 4, value);
                2
            }
            None => {
                write_register(ctx, a + 3, crate::value::Value::Nil);
                1
            }
        };
        for index in placed..c {
            write_register(ctx, a + 3 + index, crate::value::Value::Nil);
        }
        vm.jit_set_top(ctx.base + a as usize + 6);
        next(pc)
    })
}

pub(super) extern "C" fn get_tabup(ctx: *mut JitCtx, pc: u32, target: u32, upvalue: u32, key: u32) -> i64 {
    if pending(ctx) {
        return settle_then(ctx, pc, target, upvalue, key, get_tabup);
    }
    with_vm(ctx, pc, |ctx, vm| vm.jit_get_tabup(ctx, pc as usize, target as u8, upvalue as u8, key).then(|| i64::from(pc) + 1))
}

pub(super) extern "C" fn get_upval(ctx: *mut JitCtx, pc: u32, a: u32, b: u32, _unused: u32) -> i64 {
    if pending(ctx) {
        return settle_then(ctx, pc, a, b, _unused, get_upval);
    }
    with_vm(ctx, pc, |ctx, vm| {
        let value = vm.jit_upvalue_value(ctx, b as u8);
        write_register(ctx, a, value);
        next(pc)
    })
}

pub(super) extern "C" fn set_upval(ctx: *mut JitCtx, pc: u32, a: u32, b: u32, _unused: u32) -> i64 {
    if pending(ctx) {
        return settle_then(ctx, pc, a, b, _unused, set_upval);
    }
    with_vm(ctx, pc, |ctx, vm| {
        let value = register(ctx, a);
        vm.jit_set_upvalue(ctx, b as u8, value);
        next(pc)
    })
}

pub(super) extern "C" fn tier_check(ctx: *mut JitCtx, pc: u32) -> i64 {
    if pending(ctx) {
        return settle_then_entry(ctx, pc, tier_check);
    }
    // SAFETY: the context and VM pointers are valid for the duration of the call.
    let ctx = unsafe { &mut *ctx };
    let vm = unsafe { &mut *ctx.vm };
    vm.jit_tier_check(ctx, pc as usize, crate::Tier::Baseline)
}
