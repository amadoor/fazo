use super::{
    Event,
    window::{Ip, Passed, Window},
};
use crate::{Vm, shape::KeyOrigin, value::Value};

pub(super) fn constant(window: Passed<'_>, index: u32) -> Value {
    window.constant(index)
}

pub(super) fn rk(window: Passed<'_>, index: u8, k: bool) -> Value {
    if k {
        constant(window.pass(), u32::from(index))
    } else {
        window.register(index)
    }
}

#[cfg_attr(not(debug_assertions), inline(always))]
pub(super) fn rk_ref(window: &Window, index: u8, k: bool) -> &Value {
    if k { window.constant_ref(u32::from(index)) } else { window.register_ref(index) }
}

pub(super) fn for_prep_integers(window: Passed<'_>, a: usize) -> Option<bool> {
    let (Value::Integer(start), Value::Integer(limit), Value::Integer(step)) =
        (window.register(a), window.register(a + 1), window.register(a + 2))
    else {
        return None;
    };
    if step == 0 {
        return None;
    }
    if (step > 0 && start > limit) || (step < 0 && start < limit) {
        return Some(true);
    }
    window.set_register(a, Value::Integer(int_loop_count(start, limit, step) as i64));
    window.set_register(a + 1, Value::Integer(step));
    window.set_register(a + 2, Value::Integer(start));
    Some(false)
}

/// Charged once at FORPREP, so the back-edges themselves charge nothing.
pub(super) fn loop_charge(window: Passed<'_>, a: usize, length: u32) -> u32 {
    let iterations = match (window.register(a), window.register(a + 1), window.register(a + 2)) {
        (Value::Integer(count), Value::Integer(_), _) => (count as u64).saturating_add(1),
        (Value::Float(limit), Value::Float(step), Value::Float(start)) => float_iterations(start, limit, step),
        _ => 1,
    };
    u32::try_from(iterations.saturating_mul(u64::from(length))).unwrap_or(u32::MAX)
}

fn float_iterations(start: f64, limit: f64, step: f64) -> u64 {
    let iterations = ((limit - start) / step).floor() + 1.0;
    if iterations >= 1.0 { iterations as u64 } else { 0 }
}

pub(super) fn int_loop_count(start: i64, last: i64, increment: i64) -> u64 {
    if increment > 0 {
        let mut count = (last as u64).wrapping_sub(start as u64);
        if increment != 1 {
            count /= increment as u64;
        }
        count
    } else {
        let divisor = (-(increment + 1)) as u64 + 1;
        (start as u64).wrapping_sub(last as u64) / divisor
    }
}

/// Follows `OP_FORLOOP` in lvm.c; `None` only for malformed bytecode.
#[cfg_attr(not(debug_assertions), inline(always))]
pub(super) fn for_loop(window: Passed<'_>, a: usize) -> Option<bool> {
    if let Value::Integer(step) = *window.register_ref(a + 1) {
        let Value::Integer(remaining) = window.register_mut(a) else {
            return None;
        };
        if *remaining == 0 {
            return Some(false);
        }
        *remaining = remaining.wrapping_sub(1);
        let Value::Integer(index) = window.register_mut(a + 2) else {
            return None;
        };
        *index = index.wrapping_add(step);
        return Some(true);
    }
    for_loop_float(window, a)
}

#[cfg_attr(not(debug_assertions), inline(always))]
fn for_loop_float(window: Passed<'_>, a: usize) -> Option<bool> {
    let (Value::Float(step), Value::Float(limit), Value::Float(index)) = (window.register(a + 1), window.register(a), window.register(a + 2)) else {
        return None;
    };
    let next = index + step;
    let continues = if 0.0 < step { next <= limit } else { limit <= next };
    if continues {
        window.set_register(a + 2, Value::Float(next));
    }
    Some(continues)
}

/// Reads before any metamethod, as the integer forms of `GETTABLE` and `GETI` do.
#[cfg_attr(not(debug_assertions), inline(always))]
pub(super) fn int_get(vm: &Vm, object: Value, key: Value) -> Value {
    let (Value::Table(table), Value::Integer(index)) = (object, key) else {
        return Value::Nil;
    };
    vm.heap.tables.live(table).get_int(index)
}

#[cfg_attr(not(debug_assertions), inline(always))]
pub(super) fn int_get_ref<'v>(vm: &'v Vm, object: &Value, key: &Value) -> &'v Value {
    static NIL: Value = Value::Nil;
    let (Value::Table(table), Value::Integer(index)) = (*object, *key) else {
        return &NIL;
    };
    Window::table_object(vm, table).int_ref(index)
}

/// Stores only into an existing key, where `__newindex` cannot intercept.
#[cfg_attr(not(debug_assertions), inline(always))]
pub(super) fn int_set(vm: &mut Vm, object: &Value, key: &Value, value: &Value) -> bool {
    let (Value::Table(table), Value::Integer(index)) = (*object, *key) else {
        return false;
    };
    let target = Window::table_object_mut(vm, table);
    let stored = if target.metatable.is_some() {
        target.replace_array_item(index, value)
    } else {
        target.replace_int(index, value)
    };
    if !stored {
        return false;
    }
    vm.heap.table_barrier_back(table, *value);
    true
}

#[cfg_attr(not(debug_assertions), inline(always))]
pub(super) fn raw_set(vm: &mut Vm, object: Value, key: Value, value: Value) -> bool {
    raw_store(vm, object, key, value, KeyOrigin::Dynamic)
}

pub(super) fn raw_set_field(vm: &mut Vm, object: Value, key: Value, value: Value) -> bool {
    raw_store(vm, object, key, value, KeyOrigin::Constant)
}

#[cfg_attr(not(debug_assertions), inline(always))]
fn raw_store(vm: &mut Vm, object: Value, key: Value, value: Value, origin: KeyOrigin) -> bool {
    let Value::Table(table) = object else {
        return false;
    };
    if let Value::Integer(index) = key
        && Window::table_object_mut(vm, table).replace_array_item(index, &value)
    {
        vm.heap.table_barrier_back(table, value);
        return true;
    }
    if let Some(metatable) = vm.heap.tables.live(table).metatable {
        let present = !vm.heap.tables.live(table).get(vm.heap.key_view(), key).is_nil();
        if !present && !vm.table_event(metatable, Event::NewIndex).is_nil() {
            return false;
        }
    }
    vm.heap.table_store(table, key, value, origin).is_ok()
}

#[cfg_attr(not(debug_assertions), inline(always))]
pub(super) fn conditional_jump(window: Passed<'_>, ip: &mut Ip, condition: bool, expected: bool) {
    if condition != expected {
        *ip = ip.add(2);
    } else {
        take_jump(window.pass(), ip);
    }
}

#[cfg_attr(not(debug_assertions), inline(always))]
pub(super) fn take_jump(window: Passed<'_>, ip: &mut Ip) {
    let word = window.fetch(ip.add(1));
    debug_assert!(matches!(crate::encoding::decode(word), Some(crate::instruction::Instruction::Jmp { .. })));
    let sj = crate::encoding::jump_offset(word);
    *ip = ip.offset(2 + sj as isize);
}
