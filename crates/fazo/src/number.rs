use crate::{
    instruction::ArithOp,
    syntax::numeral::{self, Numeral},
    value::Value,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Number {
    Integer(i64),
    Float(f64),
}

impl Number {
    pub fn to_float(self) -> f64 {
        match self {
            Number::Integer(value) => value as f64,
            Number::Float(value) => value,
        }
    }

    pub fn to_value(self) -> Value {
        match self {
            Number::Integer(value) => Value::Integer(value),
            Number::Float(value) => Value::Float(value),
        }
    }

    pub fn from_numeral(numeral: Numeral) -> Self {
        match numeral {
            Numeral::Int(value) => Number::Integer(value),
            Numeral::Float(value) => Number::Float(value),
        }
    }
}

pub fn number_of(value: Value) -> Option<Number> {
    match value {
        Value::Integer(value) => Some(Number::Integer(value)),
        Value::Float(value) => Some(Number::Float(value)),
        _ => None,
    }
}

/// Follows `luaV_flttointns` in lvm.c.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rounding {
    Exact,
    Floor,
    Ceil,
}

pub fn float_to_integer(value: f64, rounding: Rounding) -> Option<i64> {
    let mut rounded = value.floor();
    if value != rounded {
        match rounding {
            Rounding::Exact => return None,
            Rounding::Ceil => rounded += 1.0,
            Rounding::Floor => {}
        }
    }
    (-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0)
        .contains(&rounded)
        .then_some(rounded as i64)
}

pub fn to_integer(number: Number, rounding: Rounding) -> Option<i64> {
    match number {
        Number::Integer(value) => Some(value),
        Number::Float(value) => float_to_integer(value, rounding),
    }
}

pub fn normalize_key(key: Value) -> Value {
    match key {
        Value::Float(value) => match float_to_integer(value, Rounding::Exact) {
            Some(integer) => Value::Integer(integer),
            None => key,
        },
        _ => key,
    }
}

pub fn float_equals_integer(float: f64, integer: i64) -> bool {
    float_to_integer(float, Rounding::Exact) == Some(integer)
}

const FLOAT_EXACT_INTEGER_LIMIT: i64 = 1 << 53;

fn fits_float(integer: i64) -> bool {
    (-FLOAT_EXACT_INTEGER_LIMIT..=FLOAT_EXACT_INTEGER_LIMIT).contains(&integer)
}

pub fn less_integer_float(integer: i64, float: f64) -> bool {
    if fits_float(integer) {
        return (integer as f64) < float;
    }
    match float_to_integer(float, Rounding::Ceil) {
        Some(ceiling) => integer < ceiling,
        None => float > 0.0,
    }
}

pub fn less_equal_integer_float(integer: i64, float: f64) -> bool {
    if fits_float(integer) {
        return (integer as f64) <= float;
    }
    match float_to_integer(float, Rounding::Floor) {
        Some(floor) => integer <= floor,
        None => float > 0.0,
    }
}

pub fn less_float_integer(float: f64, integer: i64) -> bool {
    if fits_float(integer) {
        return float < (integer as f64);
    }
    match float_to_integer(float, Rounding::Floor) {
        Some(floor) => floor < integer,
        None => float < 0.0,
    }
}

pub fn less_equal_float_integer(float: f64, integer: i64) -> bool {
    if fits_float(integer) {
        return float <= (integer as f64);
    }
    match float_to_integer(float, Rounding::Ceil) {
        Some(ceiling) => ceiling <= integer,
        None => float < 0.0,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArithFault {
    DivideByZero,
    NoIntegerRepresentation,
}

pub fn arith(op: ArithOp, left: Number, right: Number) -> Result<Number, ArithFault> {
    match op {
        ArithOp::Div => Ok(Number::Float(left.to_float() / right.to_float())),
        ArithOp::Pow => Ok(Number::Float(float_pow(left.to_float(), right.to_float()))),
        ArithOp::BAnd | ArithOp::BOr | ArithOp::BXor | ArithOp::Shl | ArithOp::Shr => {
            bitwise(op, left, right)
        }
        _ => match (left, right) {
            (Number::Integer(a), Number::Integer(b)) => integer_arith(op, a, b),
            _ => Ok(Number::Float(float_arith(op, left.to_float(), right.to_float()))),
        },
    }
}

fn integer_arith(op: ArithOp, a: i64, b: i64) -> Result<Number, ArithFault> {
    let value = match op {
        ArithOp::Add => a.wrapping_add(b),
        ArithOp::Sub => a.wrapping_sub(b),
        ArithOp::Mul => a.wrapping_mul(b),
        ArithOp::Mod => integer_mod(a, b)?,
        ArithOp::IDiv => integer_idiv(a, b)?,
        _ => unreachable!("only integer-preserving operators reach integer_arith"),
    };
    Ok(Number::Integer(value))
}

pub fn integer_idiv(a: i64, b: i64) -> Result<i64, ArithFault> {
    if b == 0 {
        return Err(ArithFault::DivideByZero);
    }
    if b == -1 {
        return Ok(0i64.wrapping_sub(a));
    }
    let quotient = a / b;
    if (a ^ b) < 0 && a % b != 0 {
        Ok(quotient - 1)
    } else {
        Ok(quotient)
    }
}

pub fn integer_mod(a: i64, b: i64) -> Result<i64, ArithFault> {
    if b == 0 {
        return Err(ArithFault::DivideByZero);
    }
    if b == -1 {
        return Ok(0);
    }
    let remainder = a % b;
    if remainder != 0 && (remainder ^ b) < 0 {
        Ok(remainder + b)
    } else {
        Ok(remainder)
    }
}

pub fn float_pow(a: f64, b: f64) -> f64 {
    if b == 2.0 { a * a } else { crate::math::pow(a, b) }
}

pub fn float_arith(op: ArithOp, a: f64, b: f64) -> f64 {
    match op {
        ArithOp::Add => a + b,
        ArithOp::Sub => a - b,
        ArithOp::Mul => a * b,
        ArithOp::Div => a / b,
        ArithOp::Pow => float_pow(a, b),
        ArithOp::IDiv => (a / b).floor(),
        ArithOp::Mod => float_mod(a, b),
        _ => unreachable!("bitwise operators never reach float_arith"),
    }
}

fn float_mod(a: f64, b: f64) -> f64 {
    let remainder = a % b;
    let takes_divisor_sign = if remainder > 0.0 {
        b < 0.0
    } else {
        remainder < 0.0 && b > 0.0
    };
    if takes_divisor_sign { remainder + b } else { remainder }
}

fn bitwise(op: ArithOp, left: Number, right: Number) -> Result<Number, ArithFault> {
    let a = to_integer(left, Rounding::Exact).ok_or(ArithFault::NoIntegerRepresentation)?;
    let b = to_integer(right, Rounding::Exact).ok_or(ArithFault::NoIntegerRepresentation)?;
    let value = match op {
        ArithOp::BAnd => a & b,
        ArithOp::BOr => a | b,
        ArithOp::BXor => a ^ b,
        ArithOp::Shl => shift_left(a, b),
        ArithOp::Shr => shift_left(a, b.wrapping_neg()),
        _ => unreachable!("only bitwise operators reach bitwise"),
    };
    Ok(Number::Integer(value))
}

pub fn shift_left(value: i64, amount: i64) -> i64 {
    if amount <= -64 || amount >= 64 {
        0
    } else if amount >= 0 {
        ((value as u64) << amount) as i64
    } else {
        ((value as u64) >> -amount) as i64
    }
}

pub fn negate(number: Number) -> Number {
    match number {
        Number::Integer(value) => Number::Integer(value.wrapping_neg()),
        Number::Float(value) => Number::Float(-value),
    }
}

pub fn parse_number(text: &[u8]) -> Option<Number> {
    numeral::str2number(text).map(Number::from_numeral)
}

/// Follows `tostringbuff` in lobject.c; a float that looks like an integer gets a `.0` suffix.
pub fn format_float(value: f64) -> String {
    let mut text = format_g(value, 15);
    if text.parse::<f64>().ok() != Some(value) {
        text = format_g(value, 17);
    }
    if text.bytes().all(|byte| byte == b'-' || byte.is_ascii_digit()) {
        return format!("{text}.0");
    }
    text
}

pub(crate) const CANONICAL_NAN_BITS: u64 = 0x7FF8_0000_0000_0000;
pub(crate) const CANONICAL_SINGLE_NAN_BITS: u32 = 0x7FC0_0000;

pub(crate) fn canonical_double_bits(value: f64) -> u64 {
    if value.is_nan() { CANONICAL_NAN_BITS } else { value.to_bits() }
}

pub(crate) fn canonical_single_bits(value: f64) -> u32 {
    if value.is_nan() { CANONICAL_SINGLE_NAN_BITS } else { (value as f32).to_bits() }
}

fn format_g(value: f64, precision: usize) -> String {
    if value.is_nan() {
        return "nan".to_string();
    }
    if value.is_infinite() {
        return if value > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    if value == 0.0 {
        return if value.is_sign_negative() { "-0" } else { "0" }.to_string();
    }
    let scientific = format!("{:.*e}", precision - 1, value);
    let (mantissa, exponent) = scientific.split_once('e').expect("scientific form has an exponent");
    let exponent: i32 = exponent.parse().expect("exponent is an integer");
    if exponent < -4 || exponent >= precision as i32 {
        let mantissa = trim_zeros(mantissa);
        let sign = if exponent < 0 { '-' } else { '+' };
        return format!("{mantissa}e{sign}{:02}", exponent.unsigned_abs());
    }
    let decimals = (precision as i32 - 1 - exponent) as usize;
    trim_zeros(&format!("{value:.decimals$}")).to_string()
}

fn trim_zeros(text: &str) -> &str {
    if !text.contains('.') {
        return text;
    }
    text.trim_end_matches('0').trim_end_matches('.')
}

#[cfg(test)]
mod tests {
    use super::*;

    fn float(value: f64) -> String {
        format_float(value)
    }

    #[test]
    fn floats_print_with_fifteen_digits_or_seventeen_to_round_trip_and_a_point() {
        assert_eq!(float(1.0), "1.0");
        assert_eq!(float(-7.0), "-7.0");
        assert_eq!(float(100.0), "100.0");
        assert_eq!(float(0.1 + 0.2), "0.30000000000000004");
        assert_eq!(float(100.0 / 3.0), "33.333333333333336");
        assert_eq!(float(2.0 / 3.0), "0.66666666666666663");
        assert_eq!(float(-0.0), "-0.0");
    }

    #[test]
    fn large_and_small_floats_switch_to_exponent_form() {
        assert_eq!(float(1e15), "1e+15");
        assert_eq!(float(1e100), "1e+100");
        assert_eq!(float(2f64.powi(63)), "9.2233720368547758e+18");
        assert_eq!(float(1e-5), "1e-05");
        assert_eq!(float(1.5e-7), "1.5e-07");
        assert_eq!(float(5e-324), "4.94065645841247e-324");
        assert_eq!(float(1e14), "100000000000000.0");
        assert_eq!(float(123456789012.0), "123456789012.0");
    }

    #[test]
    fn infinities_and_nan_have_no_point_suffix() {
        assert_eq!(float(f64::INFINITY), "inf");
        assert_eq!(float(f64::NEG_INFINITY), "-inf");
        assert_eq!(float(f64::NAN), "nan");
    }

    #[test]
    fn nan_prints_without_its_sign_or_payload() {
        assert_eq!(float(-f64::NAN), "nan");
        assert_eq!(float(f64::from_bits(0xfff8_0000_0000_0001)), "nan");
        assert_eq!(float(f64::from_bits(0x7ff0_0000_0000_0001)), "nan");
    }

    #[test]
    fn canonical_nan_bits_replace_every_nan_and_keep_other_values() {
        assert_eq!(canonical_double_bits(-f64::NAN), 0x7FF8_0000_0000_0000);
        assert_eq!(canonical_double_bits(f64::from_bits(0xfff0_0000_0000_0001)), 0x7FF8_0000_0000_0000);
        assert_eq!(canonical_double_bits(-1.5), (-1.5f64).to_bits());
        assert_eq!(canonical_single_bits(-f64::NAN), 0x7FC0_0000);
        assert_eq!(canonical_single_bits(-1.5), (-1.5f32).to_bits());
    }

    #[test]
    fn integer_float_comparisons_are_exact_beyond_2_to_the_53() {
        let big = (1i64 << 53) + 1;
        assert!(less_integer_float(big, 9007199254740994.0));
        assert!(!less_integer_float(big, 9007199254740992.0));
        assert!(less_equal_integer_float(i64::MAX, 9223372036854775807.0 * 2.0));
        assert!(less_integer_float(i64::MAX, 9223372036854775808.0));
        assert!(!less_float_integer(-9223372036854775808.0, i64::MIN));
        assert!(less_float_integer(-9223372036854777856.0, i64::MIN));
        assert!(!float_equals_integer(9223372036854775808.0, i64::MAX));
        assert!(float_equals_integer(3.0, 3));
    }

    #[test]
    fn integer_division_and_modulo_follow_floor_rules() {
        assert_eq!(integer_idiv(7, 2), Ok(3));
        assert_eq!(integer_idiv(-7, 2), Ok(-4));
        assert_eq!(integer_idiv(7, -2), Ok(-4));
        assert_eq!(integer_idiv(i64::MIN, -1), Ok(i64::MIN));
        assert_eq!(integer_mod(-7, 3), Ok(2));
        assert_eq!(integer_mod(7, -3), Ok(-2));
        assert_eq!(integer_mod(i64::MIN, -1), Ok(0));
        assert_eq!(integer_mod(5, 0), Err(ArithFault::DivideByZero));
    }

    #[test]
    fn float_modulo_takes_the_sign_of_the_divisor() {
        assert_eq!(float_arith(ArithOp::Mod, -7.5, 2.0), 0.5);
        assert_eq!(float_arith(ArithOp::Mod, 7.5, -2.0), -0.5);
        assert_eq!(float_arith(ArithOp::Mod, -2.25, -2.0), -0.25);
        assert_eq!(float_arith(ArithOp::Mod, 98.875, -7.0), -6.125);
        assert_eq!(float_arith(ArithOp::Mod, 5.0, -2.0), -1.0);
        assert_eq!(float_arith(ArithOp::IDiv, -7.5, 2.0), -4.0);
    }

    #[test]
    fn integer_arithmetic_wraps_and_division_is_float() {
        assert_eq!(arith(ArithOp::Add, Number::Integer(i64::MAX), Number::Integer(1)), Ok(Number::Integer(i64::MIN)));
        assert_eq!(arith(ArithOp::Div, Number::Integer(7), Number::Integer(2)), Ok(Number::Float(3.5)));
        assert_eq!(arith(ArithOp::Pow, Number::Integer(2), Number::Integer(10)), Ok(Number::Float(1024.0)));
    }

    #[test]
    fn bitwise_operators_need_integral_operands_and_shift_out_of_range_is_zero() {
        assert_eq!(arith(ArithOp::BAnd, Number::Float(2.5), Number::Integer(1)), Err(ArithFault::NoIntegerRepresentation));
        assert_eq!(arith(ArithOp::BOr, Number::Float(2.0), Number::Integer(1)), Ok(Number::Integer(3)));
        assert_eq!(shift_left(1, 64), 0);
        assert_eq!(shift_left(-1, -1), i64::MAX);
        assert_eq!(shift_left(1, -64), 0);
    }

    #[test]
    fn keys_with_integral_float_values_normalize_to_integers() {
        assert!(matches!(normalize_key(Value::Float(2.0)), Value::Integer(2)));
        assert!(matches!(normalize_key(Value::Float(2.5)), Value::Float(_)));
    }
}
