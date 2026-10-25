use super::Event;
use super::{IndexStep, Operand, SetStep};
use crate::{LuaError, Vm, number, value::Value};

impl Vm {
    pub(crate) fn store_step(&mut self, object: Value, key: Value, value: Value) -> Result<SetStep, LuaError> {
        self.set_index_step(object, key, value, Operand::Unknown)
    }

    pub(crate) fn length_step(&mut self, value: Value) -> Result<IndexStep, LuaError> {
        let operand = Operand::Unknown;
        if let Value::String(string) = value {
            return Ok(IndexStep::Value(Value::Integer(self.heap.bytes(string).len() as i64)));
        }
        let handler = self.metamethod(value, Event::Len);
        if !handler.is_nil() {
            return Ok(IndexStep::Call {
                function: handler,
                arguments: vec![value, value],
            });
        }
        match value {
            Value::Table(table) => Ok(IndexStep::Value(Value::Integer(self.table_border(table)))),
            _ => {
                let info = self.variable_info(operand);
                Err(self.type_error(value, "get length of", info))
            }
        }
    }

    pub(crate) fn order_step(&mut self, left: Value, right: Value) -> Result<IndexStep, LuaError> {
        let direct = match (left, right) {
            (Value::Integer(a), Value::Integer(b)) => Some(a < b),
            (Value::Float(a), Value::Float(b)) => Some(a < b),
            (Value::Integer(a), Value::Float(b)) => Some(number::less_integer_float(a, b)),
            (Value::Float(a), Value::Integer(b)) => Some(number::less_float_integer(a, b)),
            (Value::String(a), Value::String(b)) => Some(self.heap.bytes(a) < self.heap.bytes(b)),
            _ => None,
        };
        if let Some(result) = direct {
            return Ok(IndexStep::Value(Value::Boolean(result)));
        }
        let handler = self.binary_metamethod(left, right, Event::Lt);
        if handler.is_nil() {
            return Err(self.native_order_error(left, right));
        }
        Ok(IndexStep::Call {
            function: handler,
            arguments: vec![left, right],
        })
    }

    pub(crate) fn equal_step(&mut self, left: Value, right: Value) -> Result<IndexStep, LuaError> {
        if self.raw_equal(left, right) {
            return Ok(IndexStep::Value(Value::Boolean(true)));
        }
        if !matches!((left, right), (Value::Table(_), Value::Table(_))) {
            return Ok(IndexStep::Value(Value::Boolean(false)));
        }
        let handler = self.binary_metamethod(left, right, Event::Eq);
        if handler.is_nil() {
            return Ok(IndexStep::Value(Value::Boolean(false)));
        }
        Ok(IndexStep::Call {
            function: handler,
            arguments: vec![left, right],
        })
    }
}
