use super::ops::is_bitwise;
use crate::{
    instruction::{ArithOp, UnaryOp},
    number::{self, Rounding},
    value::Value,
};

#[cfg_attr(not(debug_assertions), inline(always))]
pub(crate) fn arith(op: ArithOp, left: Value, right: Value) -> Option<Value> {
    match (left, right) {
        (Value::Integer(a), Value::Integer(b)) => integer(op, a, b),
        (Value::Float(a), Value::Float(b)) if !is_bitwise(op) => Some(Value::Float(number::float_arith(op, a, b))),
        _ => mixed(op, left, right),
    }
}

#[cfg_attr(not(debug_assertions), inline(always))]
fn mixed(op: ArithOp, left: Value, right: Value) -> Option<Value> {
    if is_bitwise(op) {
        return bitwise(op, exact_integer(left)?, exact_integer(right)?);
    }
    Some(Value::Float(number::float_arith(op, to_float(left)?, to_float(right)?)))
}

#[cfg_attr(not(debug_assertions), inline(always))]
pub(crate) fn unary(op: UnaryOp, value: Value) -> Option<Value> {
    match (op, value) {
        (UnaryOp::Minus, Value::Integer(integer)) => Some(Value::Integer(integer.wrapping_neg())),
        (UnaryOp::Minus, Value::Float(float)) => Some(Value::Float(-float)),
        (UnaryOp::BNot, _) => exact_integer(value).map(|integer| Value::Integer(!integer)),
        _ => None,
    }
}

#[cfg_attr(not(debug_assertions), inline(always))]
pub(crate) fn equal(left: Value, right: Value) -> Option<bool> {
    match (left, right) {
        (Value::Integer(a), Value::Integer(b)) => Some(a == b),
        (Value::Float(a), Value::Float(b)) => Some(a == b),
        (Value::Integer(integer), Value::Float(float)) | (Value::Float(float), Value::Integer(integer)) => {
            Some(number::float_equals_integer(float, integer))
        }
        _ => None,
    }
}

#[cfg_attr(not(debug_assertions), inline(always))]
pub(crate) fn less_than(left: Value, right: Value) -> Option<bool> {
    match (left, right) {
        (Value::Integer(a), Value::Integer(b)) => Some(a < b),
        (Value::Float(a), Value::Float(b)) => Some(a < b),
        (Value::Integer(a), Value::Float(b)) => Some(number::less_integer_float(a, b)),
        (Value::Float(a), Value::Integer(b)) => Some(number::less_float_integer(a, b)),
        _ => None,
    }
}

#[cfg_attr(not(debug_assertions), inline(always))]
pub(crate) fn less_equal(left: Value, right: Value) -> Option<bool> {
    match (left, right) {
        (Value::Integer(a), Value::Integer(b)) => Some(a <= b),
        (Value::Float(a), Value::Float(b)) => Some(a <= b),
        (Value::Integer(a), Value::Float(b)) => Some(number::less_equal_integer_float(a, b)),
        (Value::Float(a), Value::Integer(b)) => Some(number::less_equal_float_integer(a, b)),
        _ => None,
    }
}

/// `None` sends float results and division by zero to the metamethod instruction's path.
#[cfg_attr(not(debug_assertions), inline(always))]
pub(super) fn integer_result(op: ArithOp, a: i64, b: i64) -> Option<i64> {
    match op {
        ArithOp::Add => Some(a.wrapping_add(b)),
        ArithOp::Sub => Some(a.wrapping_sub(b)),
        ArithOp::Mul => Some(a.wrapping_mul(b)),
        ArithOp::Mod => number::integer_mod(a, b).ok(),
        ArithOp::IDiv => number::integer_idiv(a, b).ok(),
        ArithOp::Div | ArithOp::Pow => None,
        ArithOp::BAnd => Some(a & b),
        ArithOp::BOr => Some(a | b),
        ArithOp::BXor => Some(a ^ b),
        ArithOp::Shl => Some(number::shift_left(a, b)),
        ArithOp::Shr => Some(number::shift_left(a, b.wrapping_neg())),
    }
}

/// Follows `op_orderI` in lvm.c; `None` for other types, whose metamethod reads `C`.
macro_rules! order_immediate {
    ($name:ident, $op:tt) => {
        #[cfg_attr(not(debug_assertions), inline(always))]
        pub(super) fn $name(value: &Value, immediate: i32) -> Option<bool> {
            match *value {
                Value::Integer(integer) => Some(integer $op i64::from(immediate)),
                Value::Float(float) => Some(float $op f64::from(immediate)),
                _ => None,
            }
        }
    };
}

order_immediate!(less_than_immediate, <);
order_immediate!(less_equal_immediate, <=);
order_immediate!(greater_than_immediate, >);
order_immediate!(greater_equal_immediate, >=);

pub(super) fn immediate(value: i32, isfloat: bool) -> Value {
    if isfloat {
        Value::Float(f64::from(value))
    } else {
        Value::Integer(i64::from(value))
    }
}

#[cfg_attr(not(debug_assertions), inline(always))]
fn integer(op: ArithOp, a: i64, b: i64) -> Option<Value> {
    let value = match op {
        ArithOp::Add => a.wrapping_add(b),
        ArithOp::Sub => a.wrapping_sub(b),
        ArithOp::Mul => a.wrapping_mul(b),
        ArithOp::Mod => number::integer_mod(a, b).ok()?,
        ArithOp::IDiv => number::integer_idiv(a, b).ok()?,
        ArithOp::Div | ArithOp::Pow => return Some(Value::Float(number::float_arith(op, a as f64, b as f64))),
        ArithOp::BAnd | ArithOp::BOr | ArithOp::BXor | ArithOp::Shl | ArithOp::Shr => return bitwise(op, a, b),
    };
    Some(Value::Integer(value))
}

fn bitwise(op: ArithOp, a: i64, b: i64) -> Option<Value> {
    let value = match op {
        ArithOp::BAnd => a & b,
        ArithOp::BOr => a | b,
        ArithOp::BXor => a ^ b,
        ArithOp::Shl => number::shift_left(a, b),
        ArithOp::Shr => number::shift_left(a, b.wrapping_neg()),
        _ => return None,
    };
    Some(Value::Integer(value))
}

fn exact_integer(value: Value) -> Option<i64> {
    match value {
        Value::Integer(integer) => Some(integer),
        Value::Float(float) => number::float_to_integer(float, Rounding::Exact),
        _ => None,
    }
}

fn to_float(value: Value) -> Option<f64> {
    match value {
        Value::Integer(integer) => Some(integer as f64),
        Value::Float(float) => Some(float),
        _ => None,
    }
}
