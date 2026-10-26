use super::{
    Event,
    naming::Operand,
    ops::{Operands, arith_event, arithmetic_target, is_bitwise, unary_event},
};
use crate::{
    LuaError, Vm,
    arena::Ref,
    instruction::{ArithOp, Instruction, UnaryOp},
    number::{self, Rounding},
    object::{LuaFrame, Pending},
    table::{SetError, Table},
    value::Value,
};

const MAX_TAG_LOOP: usize = 2000;

pub(crate) enum IndexStep {
    Value(Value),
    Call { function: Value, arguments: Vec<Value> },
}

pub(crate) enum SetStep {
    Done,
    Call { function: Value, arguments: Vec<Value> },
}

enum Piece {
    String(Ref<crate::object::LuaString>),
    Digits(smallvec::SmallVec<[u8; 24]>),
    Text(Vec<u8>),
}

fn integer_digits(value: Value) -> smallvec::SmallVec<[u8; 24]> {
    match value {
        Value::Integer(integer) => crate::stdlib::printf::decimal(integer),
        _ => smallvec::SmallVec::new(),
    }
}

fn is_concatenable(value: Value) -> bool {
    matches!(value, Value::String(_) | Value::Integer(_) | Value::Float(_))
}

pub(crate) fn basic_type_slot(value: Value) -> Option<usize> {
    match value {
        Value::Nil => Some(0),
        Value::Boolean(_) => Some(1),
        Value::Integer(_) | Value::Float(_) => Some(2),
        Value::Closure(_) | Value::Native(_) => Some(3),
        Value::Thread(_) => Some(4),
        Value::LightUserdata(_) => Some(5),
        Value::Table(_) | Value::Userdata(_) | Value::String(_) => None,
    }
}

impl Vm {
    pub(crate) fn metatable_of(&self, value: Value) -> Option<Ref<Table>> {
        match value {
            Value::Table(table) => self.heap.tables.live(table).metatable,
            Value::Userdata(userdata) => self.heap.userdata.live(userdata).metatable,
            Value::String(_) => self.string_metatable,
            _ => self.type_metatables[basic_type_slot(value)?],
        }
    }

    pub(crate) fn set_metatable_of(&mut self, value: Value, metatable: Option<Ref<Table>>) {
        match value {
            Value::Table(table) => self.set_table_metatable_checked(table, metatable),
            Value::Userdata(userdata) => self.set_userdata_metatable_checked(userdata, metatable),
            Value::String(_) => self.string_metatable = metatable,
            _ => {
                if let Some(slot) = basic_type_slot(value) {
                    self.type_metatables[slot] = metatable;
                }
            }
        }
    }

    pub(crate) fn metamethod(&mut self, value: Value, event: Event) -> Value {
        match self.metatable_of(value) {
            Some(metatable) => self.table_event(metatable, event),
            None => Value::Nil,
        }
    }

    pub(crate) fn lacks_event(&mut self, table: Ref<Table>, event: Event) -> bool {
        match self.heap.tables.live(table).metatable {
            Some(metatable) => self.table_event(metatable, event).is_nil(),
            None => true,
        }
    }

    pub(crate) fn table_event(&mut self, metatable: Ref<Table>, event: Event) -> Value {
        let bit = event.absence_bit();
        if bit.is_some_and(|bit| self.heap.tables.live(metatable).is_absent(bit)) {
            return Value::Nil;
        }
        let name = self.heap.events[event as usize];
        let value = self.heap.tables.live(metatable).get(self.heap.key_view(), Value::String(name));
        if let Some(bit) = bit
            && value.is_nil()
        {
            self.heap.tables.live_mut(metatable).mark_absent(bit);
        }
        value
    }

    #[inline(always)]
    pub(crate) fn index_resolved(&mut self, object: Value, key: Value) -> Option<Value> {
        let Value::Table(table) = object else {
            return None;
        };
        let raw = match key {
            Value::String(string) if self.heap.string_keys[string.index()].interned => {
                let hash = self.heap.key_view().hash(string);
                self.heap.tables.live(table).get_short_str(self.heap.key_view(), string, hash)
            }
            _ => self.heap.tables.live(table).get(self.heap.key_view(), key),
        };
        if !raw.is_nil() {
            return Some(raw);
        }
        self.index_chain_resolved(table, key)
    }

    pub(crate) fn index_chain_resolved(&mut self, table: Ref<Table>, key: Value) -> Option<Value> {
        let mut table = table;
        for _ in 0..MAX_TAG_LOOP {
            let Some(metatable) = self.heap.tables.live(table).metatable else {
                return Some(Value::Nil);
            };
            match self.table_event(metatable, Event::Index) {
                Value::Nil => return Some(Value::Nil),
                Value::Table(next) => {
                    let raw = self.heap.tables.live(next).get(self.heap.key_view(), key);
                    if !raw.is_nil() {
                        return Some(raw);
                    }
                    table = next;
                }
                _ => return None,
            }
        }
        None
    }

    pub(crate) fn binary_metamethod(&mut self, left: Value, right: Value, event: Event) -> Value {
        let handler = self.metamethod(left, event);
        if handler.is_nil() {
            return self.metamethod(right, event);
        }
        handler
    }

    pub(crate) fn index(&mut self, object: Value, key: Value, operand: Operand) -> Result<Option<Value>, LuaError> {
        match self.index_walk(object, key, operand)? {
            IndexStep::Value(value) => Ok(Some(value)),
            IndexStep::Call { function, arguments } => self.call_meta(function, &arguments, Pending::Operation),
        }
    }

    pub(crate) fn index_chain(&mut self, object: Value, key: Value) -> Result<IndexStep, LuaError> {
        self.index_walk(object, key, Operand::Unknown)
    }

    fn index_walk(&mut self, object: Value, key: Value, operand: Operand) -> Result<IndexStep, LuaError> {
        let mut current = object;
        for step in 0..MAX_TAG_LOOP {
            if let Value::Table(table) = current {
                let raw = self.heap.tables.live(table).get(self.heap.key_view(), key);
                if !raw.is_nil() {
                    return Ok(IndexStep::Value(raw));
                }
            }
            let handler = self.metamethod(current, Event::Index);
            if handler.is_nil() {
                if let Value::Table(_) = current {
                    return Ok(IndexStep::Value(Value::Nil));
                }
                let info = if step == 0 { self.variable_info(operand) } else { Vec::new() };
                return Err(self.type_error(current, "index", info));
            }
            if handler.is_function() {
                return Ok(IndexStep::Call {
                    function: handler,
                    arguments: vec![current, key],
                });
            }
            current = handler;
        }
        Err(self.runtime_error(b"'__index' chain too long; possible loop"))
    }

    pub(crate) fn attach_metatable(&mut self, table: Ref<Table>, metatable: Option<Ref<Table>>) {
        self.set_table_metatable_checked(table, metatable);
    }

    pub(crate) fn set_index(
        &mut self,
        object: Value,
        key: Value,
        value: Value,
        operand: Operand,
    ) -> Result<bool, LuaError> {
        match self.set_index_step(object, key, value, operand)? {
            SetStep::Done => Ok(true),
            SetStep::Call { function, arguments } => {
                Ok(self.call_meta(function, &arguments, Pending::Operation)?.is_some())
            }
        }
    }

    pub(crate) fn set_index_step(
        &mut self,
        object: Value,
        key: Value,
        value: Value,
        operand: Operand,
    ) -> Result<SetStep, LuaError> {
        let mut current = object;
        for step in 0..MAX_TAG_LOOP {
            if let Value::Table(table) = current {
                let present = !self.heap.tables.live(table).get(self.heap.key_view(), key).is_nil();
                let handler = if present {
                    Value::Nil
                } else {
                    self.metamethod(current, Event::NewIndex)
                };
                if present || handler.is_nil() {
                    self.raw_set(table, key, value)?;
                    return Ok(SetStep::Done);
                }
                if handler.is_function() {
                    return Ok(SetStep::Call {
                        function: handler,
                        arguments: vec![current, key, value],
                    });
                }
                current = handler;
                continue;
            }
            let handler = self.metamethod(current, Event::NewIndex);
            if handler.is_nil() {
                let info = if step == 0 { self.variable_info(operand) } else { Vec::new() };
                return Err(self.type_error(current, "index", info));
            }
            if handler.is_function() {
                return Ok(SetStep::Call {
                    function: handler,
                    arguments: vec![current, key, value],
                });
            }
            current = handler;
        }
        Err(self.runtime_error(b"'__newindex' chain too long; possible loop"))
    }

    pub(crate) fn raw_set(&mut self, table: Ref<Table>, key: Value, value: Value) -> Result<(), LuaError> {
        match self.store_in_table(table, key, value) {
            Ok(()) => Ok(()),
            Err(SetError::NilKey) => Err(self.runtime_error(b"table index is nil")),
            Err(SetError::NanKey) => Err(self.runtime_error(b"table index is NaN")),
            Err(SetError::Memory) => Err(self.memory_error()),
        }
    }

    pub(crate) fn arith(
        &mut self,
        op: ArithOp,
        left: Value,
        right: Value,
        operands: Operands,
    ) -> Result<Option<Value>, LuaError> {
        if let (Some(a), Some(b)) = (number::number_of(left), number::number_of(right)) {
            return self.arith_numbers(op, a, b, operands).map(Some);
        }
        let handler = self.binary_metamethod(left, right, arith_event(op));
        if handler.is_nil() {
            return Err(self.arith_failure(op, left, right, operands));
        }
        let _ = is_bitwise(op);
        self.call_meta(handler, &[left, right], Pending::Operation)
    }

    pub(crate) fn unary(&mut self, op: UnaryOp, value: Value, operand: Operand) -> Result<Option<Value>, LuaError> {
        let operands = Operands {
            left: operand,
            right: operand,
        };
        match op {
            UnaryOp::Not => Ok(Some(Value::Boolean(value.is_falsy()))),
            UnaryOp::Len => self.length(value, operand),
            UnaryOp::Minus => {
                if let Some(number) = number::number_of(value) {
                    return Ok(Some(number::negate(number).to_value()));
                }
                self.unary_metamethod(op, value, operands)
            }
            UnaryOp::BNot => {
                if let Some(number) = number::number_of(value) {
                    return match number::to_integer(number, Rounding::Exact) {
                        Some(integer) => Ok(Some(Value::Integer(!integer))),
                        None => Err(self.no_integer_error(operand)),
                    };
                }
                self.unary_metamethod(op, value, operands)
            }
        }
    }

    fn unary_metamethod(&mut self, op: UnaryOp, value: Value, operands: Operands) -> Result<Option<Value>, LuaError> {
        let handler = unary_event(op).map_or(Value::Nil, |event| self.metamethod(value, event));
        if handler.is_nil() {
            if op == UnaryOp::BNot {
                return Err(self.arith_failure(ArithOp::BXor, value, value, operands));
            }
            return Err(self.operand_error(value, value, operands, "perform arithmetic on"));
        }
        self.call_meta(handler, &[value, value], Pending::Operation)
    }

    /// Follows `luaV_objlen` in lvm.c; `None` when the length needs a call.
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(crate) fn raw_length(&self, value: Value) -> Option<Value> {
        match value {
            Value::String(string) => Some(Value::Integer(self.heap.bytes(string).len() as i64)),
            Value::Table(table) => {
                let object = self.heap.tables.live(table);
                let plain = object.metatable.is_none_or(|metatable| Event::Len.absence_bit().is_some_and(|bit| self.heap.tables.live(metatable).is_absent(bit)));
                plain.then(|| Value::Integer(object.border()))
            }
            _ => None,
        }
    }

    pub(crate) fn length(&mut self, value: Value, operand: Operand) -> Result<Option<Value>, LuaError> {
        if let Value::String(string) = value {
            return Ok(Some(Value::Integer(self.heap.bytes(string).len() as i64)));
        }
        let handler = self.metamethod(value, Event::Len);
        if !handler.is_nil() {
            return self.call_meta(handler, &[value, value], Pending::Operation);
        }
        match value {
            Value::Table(table) => Ok(Some(Value::Integer(self.heap.tables.live(table).border()))),
            _ => {
                let info = self.variable_info(operand);
                Err(self.type_error(value, "get length of", info))
            }
        }
    }

    pub(crate) fn equals(&mut self, left: Value, right: Value) -> Result<Option<bool>, LuaError> {
        if self.heap.raw_equal(left, right) {
            return Ok(Some(true));
        }
        let both_tables = matches!((left, right), (Value::Table(_), Value::Table(_)));
        if !both_tables {
            return Ok(Some(false));
        }
        let handler = self.binary_metamethod(left, right, Event::Eq);
        if handler.is_nil() {
            return Ok(Some(false));
        }
        Ok(self
            .call_meta(handler, &[left, right], Pending::Operation)?
            .map(|result| !result.is_falsy()))
    }

    pub(crate) fn less_than(&mut self, left: Value, right: Value) -> Result<Option<bool>, LuaError> {
        match (left, right) {
            (Value::Integer(a), Value::Integer(b)) => Ok(Some(a < b)),
            (Value::Float(a), Value::Float(b)) => Ok(Some(a < b)),
            (Value::Integer(a), Value::Float(b)) => Ok(Some(number::less_integer_float(a, b))),
            (Value::Float(a), Value::Integer(b)) => Ok(Some(number::less_float_integer(a, b))),
            (Value::String(a), Value::String(b)) => Ok(Some(self.heap.bytes(a) < self.heap.bytes(b))),
            _ => self.order_metamethod(left, right, Event::Lt),
        }
    }

    pub(crate) fn less_equal(&mut self, left: Value, right: Value) -> Result<Option<bool>, LuaError> {
        match (left, right) {
            (Value::Integer(a), Value::Integer(b)) => Ok(Some(a <= b)),
            (Value::Float(a), Value::Float(b)) => Ok(Some(a <= b)),
            (Value::Integer(a), Value::Float(b)) => Ok(Some(number::less_equal_integer_float(a, b))),
            (Value::Float(a), Value::Integer(b)) => Ok(Some(number::less_equal_float_integer(a, b))),
            (Value::String(a), Value::String(b)) => Ok(Some(self.heap.bytes(a) <= self.heap.bytes(b))),
            _ => self.order_metamethod(left, right, Event::Le),
        }
    }

    fn order_metamethod(&mut self, left: Value, right: Value, event: Event) -> Result<Option<bool>, LuaError> {
        let handler = self.binary_metamethod(left, right, event);
        if handler.is_nil() {
            return Err(self.order_error(left, right));
        }
        Ok(self
            .call_meta(handler, &[left, right], Pending::Operation)?
            .map(|result| !result.is_falsy()))
    }

    /// Follows `luaV_concat` in lvm.c. A long result is allocated fallibly, so failure is a memory error.
    #[inline(never)]
    fn concat_string(&mut self, slots: std::ops::Range<usize>) -> Result<Ref<crate::object::LuaString>, LuaError> {
        let pieces: smallvec::SmallVec<[Piece; 4]> = slots
            .map(|slot| match self.stack_get(slot) {
                Value::String(string) => Piece::String(string),
                Value::Integer(integer) => Piece::Digits(crate::stdlib::printf::decimal(integer)),
                value => Piece::Text(self.to_display_bytes(value)),
            })
            .collect();
        let size = pieces.iter().fold(0usize, |size, piece| size.saturating_add(self.piece_bytes(piece).len()));
        self.reserve(size)?;

        if size <= crate::heap::SHORT_STRING_LIMIT {
            let mut buffer = [0u8; crate::heap::SHORT_STRING_LIMIT];
            let mut at = 0;
            for piece in &pieces {
                let bytes = self.piece_bytes(piece);
                buffer[at..at + bytes.len()].copy_from_slice(bytes);
                at += bytes.len();
            }
            return Ok(self.heap.string(&buffer[..size]));
        }
        let mut bytes = Vec::new();
        if bytes.try_reserve_exact(size).is_err() {
            return Err(self.memory_error());
        }
        for piece in &pieces {
            bytes.extend_from_slice(self.piece_bytes(piece));
        }
        Ok(self.heap.string_from_buffer(bytes))
    }

    fn piece_bytes<'a>(&'a self, piece: &'a Piece) -> &'a [u8] {
        match piece {
            Piece::String(string) => self.heap.bytes(*string),
            Piece::Digits(digits) => digits,
            Piece::Text(text) => text,
        }
    }

    /// Follows `luaV_concat` in lvm.c for two strings with a short result; `None` leaves the rest to `concat_loop`.
    #[inline(never)]
    pub(crate) fn concat_short_strings(&mut self, first: Value, second: Value) -> Option<Value> {
        let (Value::String(left), Value::String(right)) = (first, second) else {
            return None;
        };
        if self.options.memory_limit.is_some() {
            return None;
        }
        let (left, right) = (&self.heap.strings.live(left).bytes, &self.heap.strings.live(right).bytes);
        if right.is_empty() {
            return Some(first);
        }
        if left.is_empty() {
            return Some(second);
        }
        let size = left.len() + right.len();
        if size > crate::heap::SHORT_STRING_LIMIT {
            return None;
        }
        let mut buffer = [0u8; crate::heap::SHORT_STRING_LIMIT];
        buffer[..left.len()].copy_from_slice(left);
        buffer[left.len()..size].copy_from_slice(right);
        Some(Value::String(self.heap.string(&buffer[..size])))
    }

    /// Follows `luaV_concat` in lvm.c. `None` leaves a float operand or a long result to `concat_string`.
    fn concat_pair(&mut self, first: Value, second: Value) -> Result<Option<Value>, LuaError> {
        if let (Value::String(left), Value::String(right)) = (first, second) {
            let (left, right) = (&self.heap.strings.live(left).bytes, &self.heap.strings.live(right).bytes);
            if right.is_empty() {
                return Ok(Some(first));
            }
            if left.is_empty() {
                return Ok(Some(second));
            }
            return self.join_short(left.len() + right.len(), first, second, &[], &[]);
        }
        self.concat_number_pair(first, second)
    }

    #[inline(never)]
    fn concat_number_pair(&mut self, first: Value, second: Value) -> Result<Option<Value>, LuaError> {
        let (first_digits, second_digits) = (integer_digits(first), integer_digits(second));
        let (Some(first_bytes), Some(second_bytes)) = (self.pair_bytes(first, &first_digits), self.pair_bytes(second, &second_digits)) else {
            return Ok(None);
        };
        let size = first_bytes.len() + second_bytes.len();
        self.join_short(size, first, second, &first_digits, &second_digits)
    }

    #[inline(always)]
    fn join_short(&mut self, size: usize, first: Value, second: Value, first_digits: &[u8], second_digits: &[u8]) -> Result<Option<Value>, LuaError> {
        if size > crate::heap::SHORT_STRING_LIMIT {
            return Ok(None);
        }
        let mut buffer = [0u8; crate::heap::SHORT_STRING_LIMIT];
        let first_bytes = self.pair_bytes(first, first_digits).unwrap_or_default();
        let at = first_bytes.len();
        buffer[..at].copy_from_slice(first_bytes);
        buffer[at..size].copy_from_slice(self.pair_bytes(second, second_digits).unwrap_or_default());
        self.reserve(size)?;
        Ok(Some(Value::String(self.heap.string(&buffer[..size]))))
    }

    fn pair_bytes<'a>(&'a self, value: Value, digits: &'a [u8]) -> Option<&'a [u8]> {
        match value {
            Value::String(string) => Some(&self.heap.strings.live(string).bytes),
            Value::Integer(_) => Some(digits),
            _ => None,
        }
    }

    /// Follows `luaV_concat` in lvm.c; a metamethod joins two operands when either is not a string or number.
    pub(crate) fn concat_loop(&mut self, base: usize, a: u8, total: usize) -> Result<bool, LuaError> {
        let mut total = total;
        while total > 1 {
            let left_register = a + (total - 2) as u8;
            let left_slot = base + usize::from(left_register);
            let left = self.stack_get(left_slot);
            let right = self.stack_get(left_slot + 1);
            if !is_concatenable(left) || !is_concatenable(right) {
                let operands = Operands {
                    left: Operand::Register(left_register),
                    right: Operand::Register(left_register + 1),
                };
                let handler = self.binary_metamethod(left, right, Event::Concat);
                if handler.is_nil() {
                    return Err(self.concat_error(left, right, operands));
                }
                match self.call_meta(handler, &[left, right], Pending::Concat(total))? {
                    Some(value) => self.stack_set(left_slot, value),
                    None => return Ok(false),
                }
                total -= 1;
                continue;
            }
            let top = left_slot + 2;
            let mut joined = 2;
            while joined < total && is_concatenable(self.stack_get(top - joined - 1)) {
                joined += 1;
            }
            if joined == 2
                && let Some(value) = self.concat_pair(left, right)?
            {
                self.stack_set(left_slot, value);
                total -= 1;
                continue;
            }
            let value = Value::String(self.concat_string(top - joined..top)?);
            self.stack_set(top - joined, value);
            total -= joined - 1;
        }
        Ok(true)
    }

    pub(crate) fn finish_operation(&mut self, pending: Pending) -> Result<(), LuaError> {
        let frame = *self.top_frame().expect("a pending operation resumes in a Lua frame");
        let result = self.stack_get(frame.base + usize::from(self.proto_of(frame.closure).max_stack));
        let instruction = self.proto_of(frame.closure).code[frame.pc() - 1];
        match pending {
            Pending::Concat(total) => self.finish_concat(frame, instruction, total, result),
            _ => {
                self.finish_result(frame, instruction, result);
                Ok(())
            }
        }
    }

    fn finish_concat(
        &mut self,
        frame: LuaFrame,
        instruction: Instruction,
        total: usize,
        result: Value,
    ) -> Result<(), LuaError> {
        let Instruction::Concat { a, .. } = instruction else {
            unreachable!("a concatenation metamethod resumes a Concat instruction");
        };
        self.stack_set(frame.base + usize::from(a) + total - 2, result);
        self.concat_loop(frame.base, a, total - 1)?;
        Ok(())
    }

    fn finish_result(&mut self, frame: LuaFrame, instruction: Instruction, result: Value) {
        let base = frame.base;
        match instruction {
            Instruction::Eq { k, .. }
            | Instruction::Lt { k, .. }
            | Instruction::Le { k, .. }
            | Instruction::LtI { k, .. }
            | Instruction::LeI { k, .. }
            | Instruction::GtI { k, .. }
            | Instruction::GeI { k, .. } => {
                self.finish_condition(frame, !result.is_falsy(), k);
            }
            Instruction::MmBin { .. } | Instruction::MmBinI { .. } | Instruction::MmBinK { .. } => {
                let target = self.proto_of(frame.closure).code[frame.pc() - 2];
                let Some(a) = arithmetic_target(target) else {
                    unreachable!("a metamethod instruction follows an arithmetic instruction");
                };
                self.stack_set(base + usize::from(a), result);
            }
            Instruction::Unm { a, .. }
            | Instruction::BNot { a, .. }
            | Instruction::Len { a, .. }
            | Instruction::GetTabUp { a, .. }
            | Instruction::GetTable { a, .. }
            | Instruction::GetVarg { a, .. }
            | Instruction::GetI { a, .. }
            | Instruction::GetField { a, .. }
            | Instruction::SelfOp { a, .. } => self.stack_set(base + usize::from(a), result),
            _ => {}
        }
    }

    fn finish_condition(&mut self, frame: LuaFrame, condition: bool, expected: bool) {
        if condition != expected {
            self.save_pc(frame.pc() + 1);
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
    fn a_newindex_function_receives_the_table_it_was_assigned_on() {
        assert_eq!(
            output_of("local t = setmetatable({}, {__newindex = function(tb, k, v) rawset(tb, k, v * 2) end})\nt.k = 21\nprint(t.k)"),
            "42\n"
        );
    }

    #[test]
    fn a_concatenation_suspends_on_each_metamethod_and_resumes_with_the_rest() {
        assert_eq!(
            output_of("local M = {__concat = function(a, b) return '<' .. tostring(type(a)) .. '|' .. tostring(type(b)) .. '>' end}\nlocal o = setmetatable({}, M)\nprint('a' .. o .. 'b' .. 'c')"),
            "a<table|string>\n"
        );
    }

    #[test]
    fn an_index_chain_that_loops_is_an_error() {
        assert_eq!(
            output_of("local a = {}\nlocal b = setmetatable({}, {__index = a})\nsetmetatable(a, {__index = b})\nprint(b.x)"),
            "t:4: '__index' chain too long; possible loop"
        );
    }
}
