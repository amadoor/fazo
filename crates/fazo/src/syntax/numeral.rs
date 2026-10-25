use super::token::hex_value;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Numeral {
    Int(i64),
    Float(f64),
}

const MAX_BY_TEN: u64 = i64::MAX as u64 / 10;
const MAX_LAST_DIGIT: u64 = i64::MAX as u64 % 10;
const MANTISSA_LIMIT: u64 = 1 << 60;
const EXPONENT_CAP: i64 = 1_000_000_000;

/// Follows `luaO_str2num` in lobject.c; surrounding whitespace is allowed.
pub fn str2number(text: &[u8]) -> Option<Numeral> {
    str2int(text)
        .map(Numeral::Int)
        .or_else(|| str2float(text).map(Numeral::Float))
}

fn str2int(text: &[u8]) -> Option<i64> {
    let (negative, body) = split_sign(trim(text));
    if let Some(hex) = hex_digits(body) {
        return hex_integer(hex, negative);
    }
    decimal_integer(body, negative)
}

fn hex_integer(hex: &[u8], negative: bool) -> Option<i64> {
    if hex.is_empty() || !hex.iter().all(u8::is_ascii_hexdigit) {
        return None;
    }
    let magnitude = hex
        .iter()
        .fold(0u64, |value, &digit| value.wrapping_mul(16).wrapping_add(u64::from(hex_value(digit))));
    Some(apply_sign(magnitude, negative))
}

fn decimal_integer(body: &[u8], negative: bool) -> Option<i64> {
    if body.is_empty() || !body.iter().all(u8::is_ascii_digit) {
        return None;
    }
    let mut magnitude: u64 = 0;
    for &byte in body {
        let digit = u64::from(byte - b'0');
        let overflows = magnitude >= MAX_BY_TEN
            && (magnitude > MAX_BY_TEN || digit > MAX_LAST_DIGIT + u64::from(negative));
        if overflows {
            return None;
        }
        magnitude = magnitude * 10 + digit;
    }
    Some(apply_sign(magnitude, negative))
}

fn apply_sign(magnitude: u64, negative: bool) -> i64 {
    let value = if negative { 0u64.wrapping_sub(magnitude) } else { magnitude };
    value as i64
}

fn str2float(text: &[u8]) -> Option<f64> {
    let (negative, body) = split_sign(trim(text));
    let magnitude = match hex_digits(body) {
        Some(hex) => hex_float(hex)?,
        None => decimal_float(body)?,
    };
    Some(if negative { -magnitude } else { magnitude })
}

fn decimal_float(body: &[u8]) -> Option<f64> {
    let mut index = 0;
    let mut digits = 0;
    index = skip_while(body, index, |byte| byte.is_ascii_digit(), &mut digits);
    if body.get(index) == Some(&b'.') {
        index = skip_while(body, index + 1, |byte| byte.is_ascii_digit(), &mut digits);
    }
    if digits == 0 {
        return None;
    }
    if matches!(body.get(index), Some(b'e' | b'E')) {
        index += 1;
        if matches!(body.get(index), Some(b'+' | b'-')) {
            index += 1;
        }
        let mut exponent_digits = 0;
        index = skip_while(body, index, |byte| byte.is_ascii_digit(), &mut exponent_digits);
        if exponent_digits == 0 {
            return None;
        }
    }
    if index != body.len() {
        return None;
    }
    std::str::from_utf8(body).ok()?.parse().ok()
}

fn skip_while(body: &[u8], start: usize, accept: impl Fn(&u8) -> bool, count: &mut usize) -> usize {
    let mut index = start;
    while body.get(index).is_some_and(&accept) {
        index += 1;
        *count += 1;
    }
    index
}

fn hex_float(body: &[u8]) -> Option<f64> {
    let mut mantissa: u64 = 0;
    let mut exponent: i64 = 0;
    let mut sticky = false;
    let mut digits = 0;
    let mut fraction = false;
    let mut index = 0;
    while let Some(&byte) = body.get(index) {
        if byte == b'.' {
            if fraction {
                return None;
            }
            fraction = true;
        } else if byte.is_ascii_hexdigit() {
            digits += 1;
            let value = u64::from(hex_value(byte));
            if mantissa < MANTISSA_LIMIT {
                mantissa = mantissa * 16 + value;
                if fraction {
                    exponent -= 4;
                }
            } else {
                if !fraction {
                    exponent += 4;
                }
                sticky |= value != 0;
            }
        } else {
            break;
        }
        index += 1;
    }
    if digits == 0 {
        return None;
    }
    if matches!(body.get(index), Some(b'p' | b'P')) {
        let (power, end) = binary_exponent(body, index + 1)?;
        exponent += power;
        index = end;
    }
    if index != body.len() {
        return None;
    }
    if sticky {
        mantissa |= 1;
    }
    Some(scale(mantissa as f64, exponent))
}

fn binary_exponent(body: &[u8], start: usize) -> Option<(i64, usize)> {
    let (negative, mut index) = match body.get(start) {
        Some(b'-') => (true, start + 1),
        Some(b'+') => (false, start + 1),
        _ => (false, start),
    };
    let mut value: i64 = 0;
    let mut count = 0;
    while let Some(&byte) = body.get(index) {
        if !byte.is_ascii_digit() {
            break;
        }
        value = (value * 10 + i64::from(byte - b'0')).min(EXPONENT_CAP);
        index += 1;
        count += 1;
    }
    if count == 0 {
        return None;
    }
    Some((if negative { -value } else { value }, index))
}

fn scale(mut value: f64, mut exponent: i64) -> f64 {
    while exponent > 1000 {
        value *= 2f64.powi(1000);
        exponent -= 1000;
    }
    while exponent < -1000 {
        value *= 2f64.powi(-1000);
        exponent += 1000;
    }
    value * 2f64.powi(exponent as i32)
}

fn hex_digits(body: &[u8]) -> Option<&[u8]> {
    body.strip_prefix(b"0x").or_else(|| body.strip_prefix(b"0X"))
}

fn split_sign(text: &[u8]) -> (bool, &[u8]) {
    match text.first() {
        Some(b'-') => (true, &text[1..]),
        Some(b'+') => (false, &text[1..]),
        _ => (false, text),
    }
}

fn trim(text: &[u8]) -> &[u8] {
    let start = text.iter().position(|&byte| !is_space(byte)).unwrap_or(text.len());
    let end = text.iter().rposition(|&byte| !is_space(byte)).map_or(start, |last| last + 1);
    &text[start..end.max(start)]
}

fn is_space(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
}

#[cfg(test)]
mod tests {
    use super::{Numeral, str2number};

    fn int(text: &str) -> Option<i64> {
        match str2number(text.as_bytes()) {
            Some(Numeral::Int(value)) => Some(value),
            _ => None,
        }
    }

    fn float(text: &str) -> Option<f64> {
        match str2number(text.as_bytes()) {
            Some(Numeral::Float(value)) => Some(value),
            _ => None,
        }
    }

    #[test]
    fn integers_wrap_in_hex_and_fall_back_to_float_in_decimal() {
        assert_eq!(int("0x7fffffffffffffff"), Some(i64::MAX));
        assert_eq!(int("0xffffffffffffffff"), Some(-1));
        assert_eq!(int("0x10000000000000000"), Some(0));
        assert_eq!(int("9223372036854775807"), Some(i64::MAX));
        assert_eq!(int("9223372036854775808"), None);
        assert_eq!(int("-9223372036854775808"), Some(i64::MIN));
        assert_eq!(float("9223372036854775808"), Some(9_223_372_036_854_775_808.0));
    }

    #[test]
    fn decimal_floats_accept_a_bare_dot_and_exponents() {
        assert_eq!(float("3."), Some(3.0));
        assert_eq!(float(".5"), Some(0.5));
        assert_eq!(float("1e309"), Some(f64::INFINITY));
        assert_eq!(float("1E+2"), Some(100.0));
        assert_eq!(float("1e"), None);
        assert_eq!(float("."), None);
        assert_eq!(float("3..2"), None);
        assert_eq!(float("3f"), None);
    }

    #[test]
    fn hex_floats_take_binary_exponents() {
        assert_eq!(float("0x1p4"), Some(16.0));
        assert_eq!(float("0x.8"), Some(0.5));
        assert_eq!(float("0xA.8p1"), Some(21.0));
        assert_eq!(float("0x1p"), None);
        assert_eq!(float("0x"), None);
        assert_eq!(float("0x1.8.8"), None);
    }

    #[test]
    fn surrounding_whitespace_is_allowed_and_inner_space_is_not() {
        assert_eq!(int("  42\n"), Some(42));
        assert_eq!(int("4 2"), None);
        assert_eq!(str2number(b"inf"), None);
        assert_eq!(str2number(b"nan"), None);
    }
}
