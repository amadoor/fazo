use super::{
    event::{Event, Field},
    naming::Operand,
};
use crate::{
    LuaError, Vm,
    arena::Ref,
    instruction::{ArithOp, Instruction, UnaryOp},
    number::{self, ArithFault, Number, Rounding},
    object::LuaString,
    table::Table,
    value::Value,
};

/// `LUAL_BUFFERSIZE` on a 64-bit target, from luaconf.h.
const TEXT_BUFFER_BYTES: usize = 16 * 8 * 8;

/// Enough for a native building one string at a time, plus `format`'s pooled copy and output.
const TEXT_BUFFERS: usize = 4;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Operands {
    pub left: Operand,
    pub right: Operand,
}

pub(crate) fn is_bitwise(op: ArithOp) -> bool {
    matches!(
        op,
        ArithOp::BAnd | ArithOp::BOr | ArithOp::BXor | ArithOp::Shl | ArithOp::Shr
    )
}

pub(crate) fn arith_event(op: ArithOp) -> Event {
    match op {
        ArithOp::Add => Event::Add,
        ArithOp::Sub => Event::Sub,
        ArithOp::Mul => Event::Mul,
        ArithOp::Mod => Event::Mod,
        ArithOp::Pow => Event::Pow,
        ArithOp::Div => Event::Div,
        ArithOp::IDiv => Event::IDiv,
        ArithOp::BAnd => Event::BAnd,
        ArithOp::BOr => Event::BOr,
        ArithOp::BXor => Event::BXor,
        ArithOp::Shl => Event::Shl,
        ArithOp::Shr => Event::Shr,
    }
}

pub(crate) fn unary_event(op: UnaryOp) -> Option<Event> {
    match op {
        UnaryOp::Minus => Some(Event::Unm),
        UnaryOp::BNot => Some(Event::BNot),
        UnaryOp::Len => Some(Event::Len),
        UnaryOp::Not => None,
    }
}

impl Vm {
    pub(crate) fn runtime_error(&mut self, message: &[u8]) -> LuaError {
        let mut text = self.position_prefix().unwrap_or_default();
        text.extend_from_slice(message);
        self.error_string(text)
    }

    pub(crate) fn type_error(&mut self, value: Value, operation: &str, info: Vec<u8>) -> LuaError {
        let type_name = self.object_type_name(value);
        let mut message = format!("attempt to {operation} a {type_name} value").into_bytes();
        message.extend_from_slice(&info);
        self.runtime_error(&message)
    }

    pub(crate) fn object_type_name(&mut self, value: Value) -> String {
        let metatable = match value {
            Value::Table(_) | Value::Userdata(_) => self.metatable_of(value),
            _ => None,
        };
        if let Some(metatable) = metatable
            && let Value::String(name) = self.raw_special_field(metatable, Field::Name)
        {
            return String::from_utf8_lossy(self.heap_bytes(name)).into_owned();
        }
        value.type_name().to_string()
    }

    pub(crate) fn to_display_bytes(&self, value: Value) -> Vec<u8> {
        match value {
            Value::Nil => b"nil".to_vec(),
            Value::Boolean(true) => b"true".to_vec(),
            Value::Boolean(false) => b"false".to_vec(),
            Value::Integer(integer) => integer.to_string().into_bytes(),
            Value::Float(float) => number::format_float(float).into_bytes(),
            Value::String(string) => self.heap.bytes(string).to_vec(),
            Value::Table(table) => format!("table: {:#010x}", table_address(table)).into_bytes(),
            Value::Closure(closure) => format!("function: {:#010x}", table_address(closure)).into_bytes(),
            Value::Native(native) => format!("function: {:#010x}", table_address(native)).into_bytes(),
            Value::Thread(thread) => format!("thread: {:#010x}", table_address(thread)).into_bytes(),
            Value::Userdata(userdata) => format!("userdata: {:#010x}", table_address(userdata)).into_bytes(),
            Value::LightUserdata(pointer) => format!("userdata: {pointer:#010x}").into_bytes(),
        }
    }

    pub(crate) fn coerce_number(&self, value: Value) -> Option<Number> {
        match value {
            Value::Integer(integer) => Some(Number::Integer(integer)),
            Value::Float(float) => Some(Number::Float(float)),
            Value::String(string) => number::parse_number(self.heap.bytes(string)),
            _ => None,
        }
    }

    pub(crate) fn arith_numbers(
        &mut self,
        op: ArithOp,
        left: Number,
        right: Number,
        operands: Operands,
    ) -> Result<Value, LuaError> {
        match number::arith(op, left, right) {
            Ok(result) => Ok(result.to_value()),
            Err(ArithFault::DivideByZero) => {
                let message: &[u8] = if op == ArithOp::IDiv {
                    b"attempt to divide by zero"
                } else {
                    b"attempt to perform 'n%0'"
                };
                Err(self.runtime_error(message))
            }
            Err(ArithFault::NoIntegerRepresentation) => {
                let culprit = if number::to_integer(left, Rounding::Exact).is_none() {
                    operands.left
                } else {
                    operands.right
                };
                Err(self.no_integer_error(culprit))
            }
        }
    }

    pub(crate) fn arith_on_numbers(&mut self, op: ArithOp, left: Number, right: Number) -> Result<Value, LuaError> {
        let operands = Operands {
            left: Operand::Unknown,
            right: Operand::Unknown,
        };
        self.arith_numbers(op, left, right, operands)
    }

    pub(crate) fn no_integer_error(&mut self, culprit: Operand) -> LuaError {
        let mut message = b"number".to_vec();
        message.extend(self.variable_info(culprit));
        message.extend_from_slice(b" has no integer representation");
        self.runtime_error(&message)
    }

    pub(crate) fn operand_error(&mut self, left: Value, right: Value, operands: Operands, operation: &str) -> LuaError {
        let (culprit, operand) = if left.is_number() {
            (right, operands.right)
        } else {
            (left, operands.left)
        };
        let info = self.variable_info(operand);
        self.type_error(culprit, operation, info)
    }

    pub(crate) fn arith_failure(&mut self, op: ArithOp, left: Value, right: Value, operands: Operands) -> LuaError {
        if !is_bitwise(op) {
            return self.operand_error(left, right, operands, "perform arithmetic on");
        }
        if left.is_number() && right.is_number() {
            let left_fails = number::number_of(left)
                .is_some_and(|n| number::to_integer(n, Rounding::Exact).is_none());
            let culprit = if left_fails { operands.left } else { operands.right };
            return self.no_integer_error(culprit);
        }
        self.operand_error(left, right, operands, "perform bitwise operation on")
    }

    pub(crate) fn concat_error(&mut self, left: Value, right: Value, operands: Operands) -> LuaError {
        let concatenable = |value: Value| matches!(value, Value::String(_) | Value::Integer(_) | Value::Float(_));
        let (culprit, operand) = if concatenable(left) {
            (right, operands.right)
        } else {
            (left, operands.left)
        };
        let info = self.variable_info(operand);
        self.type_error(culprit, "concatenate", info)
    }

    pub(crate) fn string_value(&mut self, bytes: &[u8]) -> Value {
        Value::String(self.heap.string(bytes))
    }

    /// Pooled so a string-building native does not allocate per call, as `luaL_Buffer` does on the C stack.
    pub(crate) fn take_text_buffer(&mut self) -> Vec<u8> {
        self.text_buffers.pop().unwrap_or_default()
    }

    /// Keeps at most `TEXT_BUFFERS` buffers of at most `TEXT_BUFFER_BYTES`; larger ones are freed.
    pub(crate) fn give_text_buffer(&mut self, mut buffer: Vec<u8>) {
        if buffer.capacity() > TEXT_BUFFER_BYTES || self.text_buffers.len() >= TEXT_BUFFERS {
            return;
        }
        buffer.clear();
        self.text_buffers.push(buffer);
    }

    pub(crate) fn text_buffer_value(&mut self, buffer: Vec<u8>) -> Value {
        let value = self.string_value(&buffer);
        self.give_text_buffer(buffer);
        value
    }

    pub(crate) fn substring_value(&mut self, text: Ref<LuaString>, start: usize, end: usize) -> Value {
        let mut buffer = self.take_text_buffer();
        buffer.extend_from_slice(&self.heap.bytes(text)[start..end]);
        self.text_buffer_value(buffer)
    }

    pub(crate) fn order_error(&mut self, left: Value, right: Value) -> LuaError {
        let message = self.order_message(left, right);
        self.runtime_error(message.as_bytes())
    }

    /// Follows `luaG_ordererror` in ldebug.c; no position, since a native is running.
    pub(crate) fn native_order_error(&mut self, left: Value, right: Value) -> LuaError {
        let message = self.order_message(left, right);
        self.error_string(message.into_bytes())
    }

    fn order_message(&mut self, left: Value, right: Value) -> String {
        let first = self.object_type_name(left);
        let second = self.object_type_name(right);
        if first == second {
            format!("attempt to compare two {first} values")
        } else {
            format!("attempt to compare {first} with {second}")
        }
    }

    pub(crate) fn raw_field(&mut self, table: Ref<Table>, name: &str) -> Value {
        let key = self.heap.string_value(name.as_bytes());
        self.heap.tables.live(table).get(self.heap.key_view(), key)
    }

    pub(crate) fn raw_special_field(&self, table: Ref<Table>, field: Field) -> Value {
        let key = Value::String(self.heap.events[field.index()]);
        self.heap.tables.live(table).get(self.heap.key_view(), key)
    }
}

pub(crate) fn table_address<T>(reference: crate::arena::Ref<T>) -> usize {
    reference.index() * 0x40 + 0x1000
}

pub(crate) fn arithmetic_target(instruction: Instruction) -> Option<u8> {
    match instruction {
        Instruction::Add { a, .. }
        | Instruction::Sub { a, .. }
        | Instruction::Mul { a, .. }
        | Instruction::Mod { a, .. }
        | Instruction::Pow { a, .. }
        | Instruction::Div { a, .. }
        | Instruction::IDiv { a, .. }
        | Instruction::BAnd { a, .. }
        | Instruction::BOr { a, .. }
        | Instruction::BXor { a, .. }
        | Instruction::Shl { a, .. }
        | Instruction::Shr { a, .. }
        | Instruction::AddK { a, .. }
        | Instruction::SubK { a, .. }
        | Instruction::MulK { a, .. }
        | Instruction::ModK { a, .. }
        | Instruction::PowK { a, .. }
        | Instruction::DivK { a, .. }
        | Instruction::IDivK { a, .. }
        | Instruction::BAndK { a, .. }
        | Instruction::BOrK { a, .. }
        | Instruction::BXorK { a, .. }
        | Instruction::AddI { a, .. }
        | Instruction::ShrI { a, .. }
        | Instruction::ShlI { a, .. } => Some(a),
        _ => None,
    }
}
