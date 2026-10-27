//! Follows LuaJIT's fast functions (`vm_x64.dasc`, `ffunc`; MIT, Copyright (C) 2005-2025 Mike Pall).
//! Fast paths run only without hooks, and a call that would raise an error takes the native, so its error is PUC's.

use super::{Event, Field, MAX_STACK_SLOTS, MIN_C_STACK, fast};
use crate::{
    Vm,
    arena::Ref,
    number::{self, Rounding},
    object::Native,
    value::Value,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Builtin {
    Type,
    Select,
    RawGet,
    RawSet,
    RawEqual,
    RawLen,
    GetMetatable,
    SetMetatable,
    Next,
    IpairsStep,
    ToString,
    Floor,
    Ceil,
    Abs,
    Sqrt,
    Min,
    Max,
    StringByte,
    StringSub,
    StringLen,
    Resume,
    Wrap,
    Yield,
}

/// The type names `type` returns, interned with the metamethod names so the collector keeps them.
pub(crate) const TYPE_NAMES: [&str; 8] = ["nil", "boolean", "number", "string", "table", "function", "thread", "userdata"];

enum Results {
    Empty,
    One(Value),
    Two(Value, Value),
    Arguments { first: usize },
}

fn type_slot(value: Value) -> usize {
    match value {
        Value::Nil => 0,
        Value::Boolean(_) => 1,
        Value::Integer(_) | Value::Float(_) => 2,
        Value::String(_) => 3,
        Value::Table(_) => 4,
        Value::Closure(_) | Value::Native(_) => 5,
        Value::Thread(_) => 6,
        Value::Userdata(_) | Value::LightUserdata(_) => 7,
    }
}

impl Vm {
    pub(crate) fn mark_builtin(&mut self, native: Value, builtin: Builtin) {
        let Value::Native(native) = native else {
            return;
        };
        let id = self.heap.natives.get(native).function;
        self.natives[id.0 as usize].builtin = Some(builtin);
        self.heap.natives.get_mut(native).builtin = Some(builtin);
    }

    pub(crate) fn mark_registered_builtin(&mut self, name: &str, builtin: Builtin) {
        let ids = self.native_index.get(name).cloned().unwrap_or_default();
        for id in ids {
            self.natives[id.0 as usize].builtin = Some(builtin);
        }
    }

    #[inline(always)]
    pub(crate) fn builtin_of(&self, native: Ref<Native>) -> Option<Builtin> {
        self.heap.natives.live(native).builtin
    }

    /// False when the call is left to the native, which also raises any error it would.
    #[inline(always)]
    pub(crate) fn call_builtin(&mut self, builtin: Builtin, function_slot: usize, count: usize, wanted: i32) -> bool {
        if function_slot + 1 + count + MIN_C_STACK > MAX_STACK_SLOTS {
            return false;
        }
        let Some(results) = self.builtin_results(builtin, function_slot + 1, count) else {
            return false;
        };
        self.place_builtin_results(function_slot, function_slot + 1 + count, wanted, results);
        true
    }

    pub(crate) fn generic_builtin(&mut self, builtin: Builtin, base: usize, wanted: u8) -> bool {
        let state = self.stack_get(base + 4);
        let control = self.stack_get(base + 5);
        let results = match builtin {
            Builtin::Next => self.next_results(state, control),
            Builtin::IpairsStep => self.ipairs_results(state, control),
            _ => None,
        };
        let Some(results) = results else {
            return false;
        };
        self.place_builtin_results(base + 3, base + 6, i32::from(wanted), results);
        true
    }

    #[inline(always)]
    pub(crate) fn builtin_value(&mut self, builtin: Builtin, function_slot: usize, count: usize) -> Option<Value> {
        if matches!(builtin, Builtin::Resume | Builtin::Wrap | Builtin::Yield) || function_slot + 1 + count + MIN_C_STACK > MAX_STACK_SLOTS {
            return None;
        }
        Some(match self.builtin_results(builtin, function_slot + 1, count)? {
            Results::Empty => Value::Nil,
            Results::One(value) | Results::Two(value, _) => value,
            Results::Arguments { first } if first < count => self.stack_get(function_slot + 1 + first),
            Results::Arguments { .. } => Value::Nil,
        })
    }

    fn builtin_results(&mut self, builtin: Builtin, first: usize, count: usize) -> Option<Results> {
        let (one, two, three) = {
            let arguments = &self.thread().stack[first..first + count];
            let argument = |index: usize| arguments.get(index).copied().unwrap_or(Value::Nil);
            (argument(0), argument(1), argument(2))
        };
        match builtin {
            Builtin::Type if count >= 1 => Some(Results::One(self.type_name_value(one))),
            Builtin::Select => self.select_results(one, count),
            Builtin::RawGet if count >= 2 => match one {
                Value::Table(table) => Some(Results::One(self.table_get(table, two))),
                _ => None,
            },
            Builtin::RawSet if count >= 3 => self.raw_set_existing(one, two, three),
            Builtin::RawEqual if count >= 2 => Some(Results::One(Value::Boolean(self.raw_equal(one, two)))),
            Builtin::RawLen => match one {
                Value::Table(table) => Some(Results::One(Value::Integer(self.table_border(table)))),
                Value::String(string) => Some(Results::One(Value::Integer(self.string_length(string) as i64))),
                _ => None,
            },
            Builtin::GetMetatable if count >= 1 => Some(Results::One(self.visible_metatable(one))),
            Builtin::SetMetatable if count >= 2 => self.set_metatable_fast(one, two),
            Builtin::Next => self.next_results(one, two),
            Builtin::IpairsStep if count >= 2 => self.ipairs_results(one, two),
            Builtin::ToString if count >= 1 => self.tostring_fast(one).map(Results::One),
            Builtin::Floor => rounded(one, f64::floor).map(Results::One),
            Builtin::Ceil => rounded(one, f64::ceil).map(Results::One),
            Builtin::Abs => match one {
                Value::Integer(integer) => Some(Results::One(Value::Integer(integer.wrapping_abs()))),
                Value::Float(float) => Some(Results::One(Value::Float(float.abs()))),
                _ => None,
            },
            Builtin::Sqrt => match one {
                Value::Integer(integer) => Some(Results::One(Value::Float((integer as f64).sqrt()))),
                Value::Float(float) => Some(Results::One(Value::Float(float.sqrt()))),
                _ => None,
            },
            Builtin::Min => self.extreme(first, count, false).map(Results::One),
            Builtin::Max => self.extreme(first, count, true).map(Results::One),
            Builtin::StringLen => match one {
                Value::String(string) if count >= 1 => Some(Results::One(Value::Integer(self.string_length(string) as i64))),
                _ => None,
            },
            Builtin::StringSub => self.string_sub_fast(one, two, three, count),
            Builtin::StringByte => self.string_byte_fast(one, two, three, count),
            _ => None,
        }
    }

    #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub(crate) fn jit_builtin(&mut self, builtin: Builtin, one: Value, two: Value) -> Option<Value> {
        match builtin {
            Builtin::Type => Some(self.type_name_value(one)),
            Builtin::RawEqual => Some(Value::Boolean(self.raw_equal(one, two))),
            Builtin::RawGet => match one {
                Value::Table(table) => Some(self.table_get(table, two)),
                _ => None,
            },
            Builtin::GetMetatable => Some(self.visible_metatable(one)),
            Builtin::SetMetatable => match self.set_metatable_fast(one, two)? {
                Results::One(value) => Some(value),
                _ => None,
            },
            _ => None,
        }
    }

    #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub(crate) fn type_name_refs(&self) -> [u64; 8] {
        std::array::from_fn(|slot| self.heap.events[Event::ALL.len() + Field::ALL.len() + slot].packed())
    }

    fn type_name_value(&self, value: Value) -> Value {
        Value::String(self.heap.events[Event::ALL.len() + Field::ALL.len() + type_slot(value)])
    }

    /// Follows `luaB_select` in lbaselib.c.
    fn select_results(&self, selector: Value, count: usize) -> Option<Results> {
        let arguments = i64::try_from(count).ok()?;
        match selector {
            Value::String(string) if count >= 1 && self.heap_bytes(string).first() == Some(&b'#') => {
                Some(Results::One(Value::Integer(arguments - 1)))
            }
            Value::Integer(index) if count >= 1 => {
                let index = if index < 0 { arguments.wrapping_add(index) } else { index.min(arguments) };
                if index < 1 {
                    return None;
                }
                Some(Results::Arguments { first: index as usize })
            }
            _ => None,
        }
    }

    fn raw_set_existing(&mut self, object: Value, key: Value, value: Value) -> Option<Results> {
        let Value::Table(table) = object else {
            return None;
        };
        if self.table_get(table, key).is_nil() {
            return None;
        }
        self.heap.table_set(table, key, value).ok()?;
        Some(Results::One(object))
    }

    fn visible_metatable(&self, value: Value) -> Value {
        let Some(metatable) = self.metatable_of(value) else {
            return Value::Nil;
        };
        let protected = self.raw_special_field(metatable, Field::Metatable);
        if !protected.is_nil() {
            return protected;
        }
        Value::Table(metatable)
    }

    fn set_metatable_fast(&mut self, object: Value, metatable: Value) -> Option<Results> {
        let Value::Table(table) = object else {
            return None;
        };
        let metatable = match metatable {
            Value::Nil => None,
            Value::Table(metatable) => Some(metatable),
            _ => return None,
        };
        if let Some(existing) = self.heap.tables.live(table).metatable
            && !self.raw_special_field(existing, Field::Metatable).is_nil()
        {
            return None;
        }
        self.attach_metatable(table, metatable);
        Some(Results::One(object))
    }

    /// One step of `ipairs`'s iterator; `None` leaves the call to the interpreter.
    #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub(crate) fn ipairs_step(&mut self, state: Value, control: Value) -> Option<Option<(Value, Value)>> {
        match self.ipairs_results(state, control)? {
            Results::Two(index, value) => Some(Some((index, value))),
            _ => Some(None),
        }
    }

    fn next_results(&self, object: Value, key: Value) -> Option<Results> {
        let Value::Table(table) = object else {
            return None;
        };
        match self.heap.tables.get(table).next(self.heap.key_view(), key).ok()? {
            Some((key, value)) => Some(Results::Two(key, value)),
            None => Some(Results::One(Value::Nil)),
        }
    }

    fn ipairs_results(&mut self, object: Value, control: Value) -> Option<Results> {
        let Value::Integer(index) = control else {
            return None;
        };
        let index = index.wrapping_add(1);
        let value = self.index_resolved(object, Value::Integer(index))?;
        if value.is_nil() {
            return Some(Results::One(Value::Nil));
        }
        Some(Results::Two(Value::Integer(index), value))
    }

    /// Follows `luaL_tolstring` in lauxlib.c for values without `__tostring`.
    fn tostring_fast(&mut self, value: Value) -> Option<Value> {
        match value {
            Value::String(_) => {
                let plain = self.metatable_of(value).is_none_or(|metatable| self.table_event(metatable, Event::ToString).is_nil());
                plain.then_some(value)
            }
            Value::Integer(_) | Value::Float(_) if self.metatable_of(value).is_none() => {
                let text = self.to_display_bytes(value);
                Some(self.string_value(&text))
            }
            _ => None,
        }
    }

    /// Compares as `lua_compare` does.
    fn extreme(&self, first: usize, count: usize, maximum: bool) -> Option<Value> {
        let mut best = self.stack_get(first);
        if count == 0 || !best.is_number() {
            return None;
        }
        for index in 1..count {
            let value = self.stack_get(first + index);
            if !value.is_number() {
                return None;
            }
            let replaces = if maximum { fast::less_than(best, value)? } else { fast::less_than(value, best)? };
            if replaces {
                best = value;
            }
        }
        Some(best)
    }

    fn string_sub_fast(&mut self, text: Value, start: Value, end: Value, count: usize) -> Option<Results> {
        let Value::String(string) = text else {
            return None;
        };
        if count == 0 {
            return None;
        }
        let length = self.string_length(string);
        let start = start_position(optional_integer(start, 1)?, length);
        let end = end_position(optional_integer(end, -1)?, length);
        if start > end {
            return Some(Results::One(self.string_value(b"")));
        }
        let bytes = self.heap_bytes(string)[start - 1..end].to_vec();
        Some(Results::One(self.string_value(&bytes)))
    }

    /// Covers at most two positions; longer ranges take the native.
    fn string_byte_fast(&self, text: Value, first: Value, last: Value, count: usize) -> Option<Results> {
        let Value::String(string) = text else {
            return None;
        };
        if count == 0 {
            return None;
        }
        let length = self.string_length(string);
        let first = optional_integer(first, 1)?;
        let start = start_position(first, length);
        let end = end_position(optional_integer(last, first)?, length);
        let bytes = self.heap_bytes(string);
        let byte = |position: usize| Value::Integer(i64::from(bytes[position - 1]));
        match end.checked_sub(start) {
            None => Some(Results::Empty),
            Some(0) => Some(Results::One(byte(start))),
            Some(1) => Some(Results::Two(byte(start), byte(end))),
            Some(_) => None,
        }
    }

    fn place_builtin_results(&mut self, destination: usize, arguments: usize, wanted: i32, results: Results) {
        let count = match results {
            Results::Empty => 0,
            Results::One(_) => 1,
            Results::Two(..) => 2,
            Results::Arguments { first } => arguments - (destination + 1 + first),
        };
        let placed = if wanted < 0 { count } else { wanted as usize };
        self.ensure_stack(destination + placed.max(2));
        let stack = &mut self.thread_mut().stack;
        match results {
            Results::Empty => {}
            Results::One(value) => stack[destination] = value,
            Results::Two(first, second) => {
                stack[destination] = first;
                stack[destination + 1] = second;
            }
            Results::Arguments { first } => {
                let source = destination + 1 + first;
                for index in 0..count.min(placed) {
                    stack[destination + index] = stack[source + index];
                }
            }
        }
        for slot in &mut stack[destination + count.min(placed)..destination + placed] {
            *slot = Value::Nil;
        }
        if wanted < 0 {
            self.thread_mut().top = destination + count;
        }
    }
}

fn rounded(value: Value, round: fn(f64) -> f64) -> Option<Value> {
    match value {
        Value::Integer(_) => Some(value),
        Value::Float(float) => {
            let rounded = round(float);
            Some(number::float_to_integer(rounded, Rounding::Exact).map_or(Value::Float(rounded), Value::Integer))
        }
        _ => None,
    }
}

/// Anything but an integer or nil is left to the native, which converts or rejects it.
fn optional_integer(value: Value, default: i64) -> Option<i64> {
    match value {
        Value::Nil => Some(default),
        Value::Integer(integer) => Some(integer),
        _ => None,
    }
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
