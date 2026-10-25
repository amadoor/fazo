/// Follows Lua 5.5's `tostring` for floats: `%.15g`, falling back to `%.17g` when that does not read back equal.
pub fn format_float(value: f64) -> String {
    let mut text = format_g(value, 15);
    if text.parse::<f64>().ok() != Some(value) {
        text = format_g(value, 17);
    }
    if text.bytes().all(|byte| byte.is_ascii_digit() || byte == b'-') {
        format!("{text}.0")
    } else {
        text
    }
}

fn format_g(value: f64, precision: usize) -> String {
    if value.is_nan() {
        return "nan".to_string();
    }
    if value.is_infinite() {
        return signed(value, "inf");
    }
    if value == 0.0 {
        return signed(value, "0");
    }
    let scientific = format!("{value:.*e}", precision - 1);
    let (mantissa, exponent) = scientific
        .split_once('e')
        .expect("`{:e}` always has an exponent");
    let exponent: i32 = exponent.parse().expect("`{:e}` exponents are integers");
    if exponent < -4 || exponent >= precision as i32 {
        return scientific_text(mantissa, exponent);
    }
    let decimals = (precision as i32 - 1 - exponent) as usize;
    trim_fraction_zeros(&format!("{value:.decimals$}"))
}

fn signed(value: f64, body: &str) -> String {
    if value.is_sign_negative() {
        format!("-{body}")
    } else {
        body.to_string()
    }
}

fn scientific_text(mantissa: &str, exponent: i32) -> String {
    let sign = if exponent < 0 { '-' } else { '+' };
    format!(
        "{}e{sign}{:02}",
        trim_fraction_zeros(mantissa),
        exponent.unsigned_abs()
    )
}

fn trim_fraction_zeros(text: &str) -> String {
    if !text.contains('.') {
        return text.to_string();
    }
    text.trim_end_matches('0').trim_end_matches('.').to_string()
}

#[cfg(test)]
mod tests {
    use super::format_float;
    use crate::differential::{normalize_nan_text, run_reference};

    const SAMPLES: &[f64] = &[
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        0.1,
        1.0 / 3.0,
        2.0 / 3.0,
        100.0,
        -2.5,
        1e14,
        1e15,
        1e16,
        123_456_789_012_345.0,
        1e100,
        -1e100,
        1e-4,
        1e-5,
        1.5e-7,
        f64::from_bits(1),
        f64::MIN_POSITIVE,
        f64::MAX,
        f64::MIN,
        9_007_199_254_740_992.0,
        9_223_372_036_854_775_808.0,
        1.234_567_890_123_45,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        -f64::NAN,
        f64::from_bits(0x7ff8_0000_0000_0001),
        f64::from_bits(0xfff8_0000_0000_0001),
        f64::from_bits(0x7ff0_0000_0000_0001),
        f64::from_bits(0xfff0_0000_0000_0001),
    ];

    #[test]
    fn float_formatting_matches_reference_tostring() {
        let script: String = SAMPLES
            .iter()
            .map(|value| {
                format!(
                    "print(tostring(string.unpack('<d', string.pack('<i8', 0x{:016x}))))\n",
                    value.to_bits()
                )
            })
            .collect();
        let outcome = run_reference(script.as_bytes(), "t")
            .expect("the reference is cached; see reference::shared for how to populate it");
        assert_eq!(outcome.error, None);
        let output = String::from_utf8_lossy(&outcome.output).into_owned();
        let normalized = normalize_nan_text(output.as_bytes());
        let expected: Vec<&str> = std::str::from_utf8(&normalized).expect("ascii output").lines().collect();
        assert_eq!(expected.len(), SAMPLES.len());
        for (value, text) in SAMPLES.iter().zip(expected) {
            assert_eq!(format_float(*value), text, "bits {:#018x}", value.to_bits());
        }
    }
}
