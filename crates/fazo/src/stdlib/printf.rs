
const MAX_FORMAT: usize = 32;
const FLAGS_FLOAT: &[u8] = b"-+#0 ";
const FLAGS_HEX: &[u8] = b"-#0";
const FLAGS_INT: &[u8] = b"-+0 ";
const FLAGS_UNSIGNED: &[u8] = b"-0";
const FLAGS_CHAR: &[u8] = b"-";
const FRACTION_BITS: u32 = 52;
const FRACTION_DIGITS: usize = 13;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct Spec {
    pub minus: bool,
    pub plus: bool,
    pub space: bool,
    pub hash: bool,
    pub zero: bool,
    pub width: usize,
    pub precision: Option<usize>,
    pub conversion: u8,
}

pub(crate) fn parse_spec(text: &[u8]) -> Result<(Spec, usize), Vec<u8>> {
    let span = text.iter().take_while(|&&byte| is_span_byte(byte)).count();
    let consumed = span + 1;
    if consumed >= MAX_FORMAT - 10 {
        return Err(b"invalid format (too long)".to_vec());
    }

    let conversion = text.get(span).copied().unwrap_or(0);
    let form = &text[..consumed.min(text.len())];
    if conversion == b'q' {
        return match span {
            0 => Ok((plain(b'q'), consumed)),
            _ => Err(b"specifier '%q' cannot have modifiers".to_vec()),
        };
    }

    let Some((flags, precision_allowed)) = conversion_rules(conversion) else {
        return Err(error_text(b"invalid conversion '%", form, b"' to 'format'"));
    };
    let Some(spec) = parse_fields(&text[..span], flags, precision_allowed, conversion) else {
        return Err(error_text(b"invalid conversion specification: '%", form, b"'"));
    };
    Ok((spec, consumed))
}

pub(crate) fn format_integer(out: &mut Vec<u8>, spec: &Spec, value: i64) {
    if spec.conversion == b'c' {
        return pad(out, spec, b"", &[value as u8], false);
    }

    let signed = matches!(spec.conversion, b'd' | b'i');
    let negative = signed && value < 0;
    let magnitude = if signed { value.unsigned_abs() } else { value as u64 };

    let prefix = integer_prefix(spec, negative, magnitude != 0);
    let digits = integer_digits(spec, magnitude);
    pad(out, spec, prefix, &digits, spec.precision.is_none())
}

pub(crate) fn format_float(out: &mut Vec<u8>, spec: &Spec, value: f64) {
    let upper = spec.conversion.is_ascii_uppercase();
    let sign: &[u8] = if value.is_nan() { b"" } else { sign_prefix(spec, value.is_sign_negative()) };
    if !value.is_finite() {
        return pad(out, spec, sign, non_finite_word(value, upper).as_bytes(), false);
    }

    let magnitude = value.abs();
    if matches!(spec.conversion, b'a' | b'A') {
        return hex_float(out, spec, sign, magnitude);
    }
    let body = decimal_body(spec, magnitude);
    pad(out, spec, sign, body.as_bytes(), true)
}

pub(crate) fn format_string(out: &mut Vec<u8>, spec: &Spec, text: &[u8]) {
    let kept = spec.precision.map_or(text, |precision| &text[..precision.min(text.len())]);
    pad(out, spec, b"", kept, false)
}

/// The digits are counted first, so they fill the inline buffer in place and nothing is copied or reversed.
pub(crate) fn decimal(value: i64) -> smallvec::SmallVec<[u8; 24]> {
    let mut magnitude = value.unsigned_abs();
    let mut buffer = [0u8; 24];
    let mut length = usize::from(value < 0) + decimal_width(magnitude);
    let size = length;

    buffer[0] = b'-';
    loop {
        length -= 1;
        buffer[length] = b'0' + (magnitude % 10) as u8;
        magnitude /= 10;
        if magnitude == 0 {
            break;
        }
    }
    smallvec::SmallVec::from_buf_and_len(buffer, size)
}

fn decimal_width(mut magnitude: u64) -> usize {
    let mut width = 1;
    while magnitude >= 10 {
        magnitude /= 10;
        width += 1;
    }
    width
}

pub(crate) fn quote_string(text: &[u8]) -> Vec<u8> {
    let mut out = vec![b'"'];
    for (index, &byte) in text.iter().enumerate() {
        push_quoted_byte(&mut out, byte, text.get(index + 1).copied());
    }
    out.push(b'"');
    out
}

pub(crate) fn quote_integer(value: i64) -> Vec<u8> {
    if value == i64::MIN {
        return b"0x8000000000000000".to_vec();
    }
    value.to_string().into_bytes()
}

pub(crate) fn quote_float(value: f64) -> Vec<u8> {
    if value.is_nan() {
        return b"(0/0)".to_vec();
    }
    if value == f64::INFINITY {
        return b"1e9999".to_vec();
    }
    if value == f64::NEG_INFINITY {
        return b"-1e9999".to_vec();
    }
    let mut out = Vec::new();
    format_float(&mut out, &plain(b'a'), value);
    out
}

fn is_span_byte(byte: u8) -> bool {
    matches!(byte, b'-' | b'+' | b' ' | b'#' | b'0'..=b'9' | b'.')
}

fn plain(conversion: u8) -> Spec {
    Spec {
        minus: false,
        plus: false,
        space: false,
        hash: false,
        zero: false,
        width: 0,
        precision: None,
        conversion,
    }
}

fn error_text(prefix: &[u8], form: &[u8], suffix: &[u8]) -> Vec<u8> {
    [prefix, form, suffix].concat()
}

fn conversion_rules(conversion: u8) -> Option<(&'static [u8], bool)> {
    match conversion {
        b'c' | b'p' => Some((FLAGS_CHAR, false)),
        b'd' | b'i' => Some((FLAGS_INT, true)),
        b'u' => Some((FLAGS_UNSIGNED, true)),
        b'o' | b'x' | b'X' => Some((FLAGS_HEX, true)),
        b'a' | b'A' | b'e' | b'E' | b'f' | b'g' | b'G' => Some((FLAGS_FLOAT, true)),
        b's' => Some((FLAGS_CHAR, true)),
        _ => None,
    }
}

fn parse_fields(body: &[u8], flags: &[u8], precision_allowed: bool, conversion: u8) -> Option<Spec> {
    let flag_end = body.iter().take_while(|&&byte| flags.contains(&byte)).count();
    let mut spec = plain(conversion);
    apply_flags(&mut spec, &body[..flag_end]);

    let mut position = flag_end;
    if body.get(position) != Some(&b'0') {
        let (width, next) = read_digits(body, position);
        spec.width = width;
        position = next;
        if precision_allowed && body.get(position) == Some(&b'.') {
            let (precision, next) = read_digits(body, position + 1);
            spec.precision = Some(precision);
            position = next;
        }
    }
    (position == body.len()).then_some(spec)
}

fn apply_flags(spec: &mut Spec, flags: &[u8]) {
    for &flag in flags {
        match flag {
            b'-' => spec.minus = true,
            b'+' => spec.plus = true,
            b' ' => spec.space = true,
            b'#' => spec.hash = true,
            b'0' => spec.zero = true,
            _ => {}
        }
    }
}

fn read_digits(body: &[u8], start: usize) -> (usize, usize) {
    let digits = &body[start..];
    let count = digits.iter().take(2).take_while(|byte| byte.is_ascii_digit()).count();
    let value = digits[..count]
        .iter()
        .fold(0, |value, &digit| value * 10 + usize::from(digit - b'0'));
    (value, start + count)
}

fn push_quoted_byte(out: &mut Vec<u8>, byte: u8, next: Option<u8>) {
    if matches!(byte, b'"' | b'\\' | b'\n') {
        out.extend_from_slice(&[b'\\', byte]);
        return;
    }
    if !byte.is_ascii_control() {
        out.push(byte);
        return;
    }
    let code = match next.is_some_and(|next| next.is_ascii_digit()) {
        true => format!("\\{byte:03}"),
        false => format!("\\{byte}"),
    };
    out.extend_from_slice(code.as_bytes());
}

fn pad(out: &mut Vec<u8>, spec: &Spec, prefix: &[u8], body: &[u8], zero_allowed: bool) {
    let fill = spec.width.saturating_sub(prefix.len() + body.len());
    out.reserve(spec.width.max(prefix.len() + body.len()));
    if spec.minus {
        out.extend_from_slice(prefix);
        out.extend_from_slice(body);
        out.resize(out.len() + fill, b' ');
        return;
    }
    if zero_allowed && spec.zero {
        out.extend_from_slice(prefix);
        out.resize(out.len() + fill, b'0');
        out.extend_from_slice(body);
        return;
    }
    out.resize(out.len() + fill, b' ');
    out.extend_from_slice(prefix);
    out.extend_from_slice(body);
}

fn sign_prefix(spec: &Spec, negative: bool) -> &'static [u8] {
    if negative {
        return b"-";
    }
    if spec.plus {
        return b"+";
    }
    if spec.space {
        return b" ";
    }
    b""
}

fn integer_prefix(spec: &Spec, negative: bool, nonzero: bool) -> &'static [u8] {
    match spec.conversion {
        b'd' | b'i' => sign_prefix(spec, negative),
        b'x' if spec.hash && nonzero => b"0x",
        b'X' if spec.hash && nonzero => b"0X",
        _ => b"",
    }
}

fn integer_digits(spec: &Spec, magnitude: u64) -> smallvec::SmallVec<[u8; 128]> {
    let mut digits = match spec.conversion {
        b'o' => radix_digits(magnitude, 8, false),
        b'x' => radix_digits(magnitude, 16, false),
        b'X' => radix_digits(magnitude, 16, true),
        _ => radix_digits(magnitude, 10, false),
    };
    if spec.precision == Some(0) && magnitude == 0 {
        digits.clear();
    }

    let minimum = spec.precision.unwrap_or(1);
    let mut out = smallvec::SmallVec::new();
    out.resize(minimum.saturating_sub(digits.len()), b'0');
    out.extend_from_slice(&digits);
    if spec.conversion == b'o' && spec.hash && out.first() != Some(&b'0') {
        out.insert(0, b'0');
    }
    out
}

fn radix_digits(mut magnitude: u64, radix: u64, upper: bool) -> smallvec::SmallVec<[u8; 24]> {
    let symbols: &[u8; 16] = if upper { b"0123456789ABCDEF" } else { b"0123456789abcdef" };
    let mut digits = smallvec::SmallVec::<[u8; 24]>::new();
    loop {
        digits.push(symbols[(magnitude % radix) as usize]);
        magnitude /= radix;
        if magnitude == 0 {
            break;
        }
    }
    digits.reverse();
    digits
}

fn non_finite_word(value: f64, upper: bool) -> String {
    let word = if value.is_nan() { "nan" } else { "inf" };
    if upper { word.to_ascii_uppercase() } else { word.to_string() }
}

fn decimal_body(spec: &Spec, magnitude: f64) -> String {
    let upper = spec.conversion.is_ascii_uppercase();
    let precision = spec.precision.unwrap_or(6);
    match spec.conversion.to_ascii_lowercase() {
        b'e' => exponent_body(magnitude, precision, spec.hash, upper),
        b'f' => fixed_body(magnitude, precision, spec.hash),
        _ => general_body(magnitude, precision.max(1), spec.hash, upper),
    }
}

fn exponent_body(magnitude: f64, precision: usize, hash: bool, upper: bool) -> String {
    let rendered = format!("{magnitude:.precision$e}");
    let (mantissa, exponent) = rendered.split_once('e').unwrap_or((rendered.as_str(), "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);

    let mut body = String::from(mantissa);
    if hash && precision == 0 {
        body.push('.');
    }
    body.push(if upper { 'E' } else { 'e' });
    body.push(if exponent < 0 { '-' } else { '+' });
    body.push_str(&format!("{:02}", exponent.unsigned_abs()));
    body
}

fn fixed_body(magnitude: f64, precision: usize, hash: bool) -> String {
    let mut body = format!("{magnitude:.precision$}");
    if hash && precision == 0 {
        body.push('.');
    }
    body
}

fn general_body(magnitude: f64, precision: usize, hash: bool, upper: bool) -> String {
    let exponent = decimal_exponent(magnitude, precision);
    let body = match (-4..precision as i32).contains(&exponent) {
        true => fixed_body(magnitude, (precision as i32 - 1 - exponent) as usize, hash),
        false => exponent_body(magnitude, precision - 1, hash, upper),
    };
    match hash {
        true => body,
        false => strip_fraction_zeros(&body),
    }
}

fn decimal_exponent(magnitude: f64, precision: usize) -> i32 {
    let rendered = format!("{:.*e}", precision - 1, magnitude);
    let (_, exponent) = rendered.split_once('e').unwrap_or(("", "0"));
    exponent.parse().unwrap_or(0)
}

fn strip_fraction_zeros(body: &str) -> String {
    let split = body.find(['e', 'E']).unwrap_or(body.len());
    let (mantissa, tail) = body.split_at(split);
    if !mantissa.contains('.') {
        return body.to_string();
    }
    let trimmed = mantissa.trim_end_matches('0').trim_end_matches('.');
    format!("{trimmed}{tail}")
}

fn hex_float(out: &mut Vec<u8>, spec: &Spec, sign: &[u8], magnitude: f64) {
    let upper = spec.conversion.is_ascii_uppercase();
    let marker: &[u8] = if upper { b"0X" } else { b"0x" };
    let prefix = [sign, marker].concat();
    let body = hex_body(magnitude, spec.precision, spec.hash, upper);
    pad(out, spec, &prefix, body.as_bytes(), true)
}

fn hex_parts(magnitude: f64) -> (u64, u64, i64) {
    let bits = magnitude.to_bits();
    let fraction = bits & ((1 << FRACTION_BITS) - 1);
    let biased = (bits >> FRACTION_BITS) as i64;
    match (biased, fraction) {
        (0, 0) => (0, 0, 0),
        (0, _) => subnormal_parts(fraction),
        _ => (1, fraction, biased - 1023),
    }
}

#[cfg(windows)]
fn subnormal_parts(fraction: u64) -> (u64, u64, i64) {
    let shift = fraction.leading_zeros() - (u64::BITS - FRACTION_BITS - 1);
    (1, (fraction << shift) & ((1 << FRACTION_BITS) - 1), -1022 - i64::from(shift))
}

#[cfg(target_os = "macos")]
fn subnormal_parts(fraction: u64) -> (u64, u64, i64) {
    (0, fraction, -1022)
}

#[cfg(not(any(windows, target_os = "macos")))]
fn subnormal_parts(fraction: u64) -> (u64, u64, i64) {
    (0, fraction, -1022)
}

fn hex_body(magnitude: f64, precision: Option<usize>, hash: bool, upper: bool) -> String {
    let (lead, fraction, exponent) = hex_parts(magnitude);
    let (lead, digits) = hex_digits(lead, fraction, precision);

    let mut body = lead.to_string();
    if hash || !digits.is_empty() {
        body.push('.');
    }
    body.push_str(&digits);
    body.push('p');
    body.push(if exponent < 0 { '-' } else { '+' });
    body.push_str(&exponent.unsigned_abs().to_string());
    if upper {
        body.make_ascii_uppercase();
    }
    body
}

fn hex_digits(lead: u64, fraction: u64, precision: Option<usize>) -> (u64, String) {
    match precision {
        None => (lead, format!("{fraction:013x}").trim_end_matches('0').to_string()),
        Some(digits) if digits < FRACTION_DIGITS => {
            let (lead, rounded) = round_hex(lead, fraction, digits);
            let text = match digits {
                0 => String::new(),
                _ => format!("{rounded:0digits$x}"),
            };
            (lead, text)
        }
        Some(digits) => {
            let padding = "0".repeat(digits - FRACTION_DIGITS);
            (lead, format!("{fraction:013x}") + &padding)
        }
    }
}

fn round_hex(lead: u64, fraction: u64, digits: usize) -> (u64, u64) {
    let dropped = 4 * (FRACTION_DIGITS - digits);
    let full = (lead << FRACTION_BITS) | fraction;
    let kept = full >> dropped;
    let remainder = full & ((1 << dropped) - 1);
    let half = 1 << (dropped - 1);
    let round_up = remainder > half || (remainder == half && kept % 2 == 1);
    let rounded = kept + u64::from(round_up);
    let kept_bits = 4 * digits;
    (rounded >> kept_bits, rounded & ((1 << kept_bits) - 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(text: &str) -> Spec {
        parse_spec(text.as_bytes()).expect("valid conversion").0
    }

    fn integer(text: &str, value: i64) -> String {
        let mut out = b"<".to_vec();
        format_integer(&mut out, &spec(text), value);
        String::from_utf8(out[1..].to_vec()).expect("ascii output")
    }

    fn float(text: &str, value: f64) -> String {
        let mut out = b"<".to_vec();
        format_float(&mut out, &spec(text), value);
        String::from_utf8(out[1..].to_vec()).expect("ascii output")
    }

    fn string(text: &str, value: &str) -> String {
        let mut out = b"<".to_vec();
        format_string(&mut out, &spec(text), value.as_bytes());
        String::from_utf8(out[1..].to_vec()).expect("utf8 output")
    }

    #[test]
    fn decimal_digits_match_the_integer_conversion() {
        for value in [0, 7, -7, 1234567890, i64::MAX, i64::MIN] {
            assert_eq!(decimal(value).as_slice(), integer("d", value).as_bytes());
        }
    }

    fn error(text: &str) -> String {
        let bytes = parse_spec(text.as_bytes()).expect_err("invalid conversion");
        String::from_utf8(bytes).expect("ascii error")
    }

    #[test]
    fn parses_width_precision_and_flags() {
        let (parsed, consumed) = parse_spec(b"-+ #05.2f rest").expect("valid");
        assert_eq!(consumed, 9);
        assert_eq!(parsed.width, 5);
        assert_eq!(parsed.precision, Some(2));
        assert!(parsed.minus && parsed.plus && parsed.space && parsed.hash && parsed.zero);
        assert_eq!(parsed.conversion, b'f');
    }

    #[test]
    fn rejects_malformed_specs_with_lstrlib_messages() {
        assert_eq!(error("123d"), "invalid conversion specification: '%123d'");
        assert_eq!(error("#d"), "invalid conversion specification: '%#d'");
        assert_eq!(error(".3c"), "invalid conversion specification: '%.3c'");
        assert_eq!(error("y"), "invalid conversion '%y' to 'format'");
        assert_eq!(error("5"), "invalid conversion '%5' to 'format'");
        assert_eq!(error("5q"), "specifier '%q' cannot have modifiers");
        assert_eq!(error("1234567890123456789012d"), "invalid format (too long)");
    }

    #[test]
    fn integer_padding_and_precision() {
        assert_eq!(integer("-6d", 42), "42    ");
        assert_eq!(integer("+d", 5), "+5");
        assert_eq!(integer("05d", -42), "-0042");
        assert_eq!(integer(".3d", 7), "007");
        assert_eq!(integer(".0d", 0), "");
        assert_eq!(integer("3.0d", 0), "   ");
        assert_eq!(integer("-3c", 65), "A  ");
        assert_eq!(integer("5c", 65), "    A");
    }

    #[test]
    fn integer_radix_conversions() {
        assert_eq!(integer("#x", 255), "0xff");
        assert_eq!(integer("X", 255), "FF");
        assert_eq!(integer("#o", 8), "010");
        assert_eq!(integer("x", -1), "ffffffffffffffff");
        assert_eq!(integer("u", -1), "18446744073709551615");
    }

    #[test]
    fn fixed_and_exponent_floats() {
        assert_eq!(float("5.2f", 1.23456), " 1.23");
        assert_eq!(float("08.3f", -1.23456), "-001.235");
        assert_eq!(float("f", -0.0), "-0.000000");
        assert_eq!(float("e", 12345.678), "1.234568e+04");
        assert_eq!(float("E", 0.0), "0.000000E+00");
        assert_eq!(float("+.2e", 0.000123), "+1.23e-04");
    }

    #[test]
    fn general_floats() {
        assert_eq!(float("g", 0.0001), "0.0001");
        assert_eq!(float("g", 1e20), "1e+20");
        assert_eq!(float("g", 1000000.0), "1e+06");
        assert_eq!(float("g", 100000.0), "100000");
        assert_eq!(float("g", 0.0), "0");
        assert_eq!(float("#g", 1.0), "1.00000");
    }

    #[test]
    fn hex_floats() {
        assert_eq!(float("a", 1.0), "0x1p+0");
        assert_eq!(float(".3a", 1.0), "0x1.000p+0");
        assert_eq!(float("A", 255.5), "0X1.FFP+7");
        assert_eq!(float("a", 0.5), "0x1p-1");
        assert_eq!(float("a", 0.0), "0x0p+0");
        assert_eq!(float("a", -0.0), "-0x0p+0");
        assert_eq!(float(".0a", 1.5), "0x2p+0");
    }

    #[cfg(windows)]
    #[test]
    fn subnormal_hex_floats_are_normalized_as_the_windows_c_runtime() {
        assert_eq!(float("a", f64::from_bits(1)), "0x1p-1074");
        assert_eq!(float("a", f64::from_bits(0x2e05_5c9a_3f6c)), "0x1.702ae4d1fb6p-1029");
    }

    #[cfg(not(windows))]
    #[test]
    fn subnormal_hex_floats_keep_a_zero_lead_digit() {
        assert_eq!(float("a", f64::from_bits(1)), "0x0.0000000000001p-1022");
    }

    #[test]
    fn non_finite_floats() {
        assert_eq!(float("f", f64::INFINITY), "inf");
        assert_eq!(float("E", f64::NEG_INFINITY), "-INF");
        assert_eq!(float("5f", f64::INFINITY), "  inf");
        assert_eq!(float("05f", f64::INFINITY), "  inf");
        assert_eq!(float("f", f64::from_bits(0x7ff8_0000_0000_0000)), "nan");
        assert_eq!(float("f", f64::from_bits(0xfff8_0000_0000_0000)), "nan");
        assert_eq!(float("E", f64::from_bits(0xfff8_0000_0000_0000)), "NAN");
        assert_eq!(float("5f", f64::from_bits(0xfff8_0000_0000_0000)), "  nan");
        assert_eq!(float("+f", f64::from_bits(0xfff8_0000_0000_0000)), "nan");
        assert_eq!(float("-8f", f64::NEG_INFINITY), "-inf    ");
    }

    #[test]
    fn string_precision_and_width() {
        assert_eq!(string("-5s", "ab"), "ab   ");
        assert_eq!(string("5s", "ab"), "   ab");
        assert_eq!(string(".1s", "abc"), "a");
    }

    #[test]
    fn quoting_matches_addquoted() {
        assert_eq!(quote_string(b"a\n\0\""), b"\"a\\\n\\0\\\"\"");
        assert_eq!(quote_string(b"\x012"), b"\"\\0012\"");
    }

    #[test]
    fn quoting_numbers() {
        assert_eq!(quote_integer(i64::MIN), b"0x8000000000000000");
        assert_eq!(quote_integer(-7), b"-7");
        assert_eq!(quote_float(1.0), b"0x1p+0");
        assert_eq!(quote_float(f64::INFINITY), b"1e9999");
        assert_eq!(quote_float(f64::NEG_INFINITY), b"-1e9999");
        assert_eq!(quote_float(f64::from_bits(0x7ff8_0000_0000_0000)), b"(0/0)");
    }
}
