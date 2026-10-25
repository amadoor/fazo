use super::{arg_error, argument, check_integer, check_number};
use crate::{LuaError, Values, values, Vm, arena::Ref, object::LuaString, value::Value, vm::interrupt::Interrupts};

const LUAI_MAXSTACK: usize = 1_000_000;

const MAX_INT_SIZE: usize = 16;
const NATIVE_WORD: usize = 8;
const INT_SIZE: i64 = 4;
const MAX_SIZE: usize = i64::MAX as usize;
const PAD_BYTE: u8 = 0;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Int,
    Uint,
    Float,
    Number,
    Double,
    Char,
    String,
    Zstr,
    Padding,
    PadAlign,
    Nop,
}

#[derive(Clone, Copy)]
struct Detail {
    kind: Kind,
    size: usize,
    padding: usize,
}

struct Header {
    name: &'static str,
    little: bool,
    max_align: usize,
}

struct Parser<'a> {
    format: &'a [u8],
    position: usize,
    header: Header,
}

impl<'a> Parser<'a> {
    fn new(format: &'a [u8], name: &'static str) -> Self {
        let end = format.iter().position(|&byte| byte == 0).unwrap_or(format.len());
        Self {
            format: &format[..end],
            position: 0,
            header: Header {
                name,
                little: true,
                max_align: 1,
            },
        }
    }

    fn at_end(&self) -> bool {
        self.position >= self.format.len()
    }

    fn peek_digit(&self) -> bool {
        self.format.get(self.position).is_some_and(u8::is_ascii_digit)
    }

    fn advance(&mut self) -> u8 {
        let byte = self.format.get(self.position).copied().unwrap_or(0);
        self.position += 1;
        byte
    }

    fn read_digits(&mut self, default: i64) -> i64 {
        if !self.peek_digit() {
            return default;
        }
        let mut value: i64 = 0;
        loop {
            value = value * 10 + i64::from(self.advance() - b'0');
            if !self.peek_digit() || value > (MAX_SIZE as i64 - 9) / 10 {
                return value;
            }
        }
    }

    fn read_size_limit(&mut self, vm: &mut Vm, default: i64) -> Result<usize, LuaError> {
        let size = self.read_digits(default);
        if size <= 0 || size > MAX_INT_SIZE as i64 {
            let message = format!("integral size ({size}) out of limits [1,{MAX_INT_SIZE}]");
            return Err(vm.runtime_error_text(&message));
        }
        Ok(size as usize)
    }

    fn read_option(&mut self, vm: &mut Vm) -> Result<(Kind, usize), LuaError> {
        match self.advance() {
            b'b' => Ok((Kind::Int, 1)),
            b'B' => Ok((Kind::Uint, 1)),
            b'h' => Ok((Kind::Int, 2)),
            b'H' => Ok((Kind::Uint, 2)),
            b'l' | b'j' => Ok((Kind::Int, NATIVE_WORD)),
            b'L' | b'J' | b'T' => Ok((Kind::Uint, NATIVE_WORD)),
            b'f' => Ok((Kind::Float, 4)),
            b'n' => Ok((Kind::Number, NATIVE_WORD)),
            b'd' => Ok((Kind::Double, NATIVE_WORD)),
            b'i' => Ok((Kind::Int, self.read_size_limit(vm, INT_SIZE)?)),
            b'I' => Ok((Kind::Uint, self.read_size_limit(vm, INT_SIZE)?)),
            b's' => Ok((Kind::String, self.read_size_limit(vm, NATIVE_WORD as i64)?)),
            b'c' => {
                let size = self.read_digits(-1);
                if size < 0 {
                    return Err(vm.runtime_error_text("missing size for format option 'c'"));
                }
                Ok((Kind::Char, size as usize))
            }
            b'z' => Ok((Kind::Zstr, 0)),
            b'x' => Ok((Kind::Padding, 1)),
            b'X' => Ok((Kind::PadAlign, 0)),
            b' ' => Ok((Kind::Nop, 0)),
            b'<' | b'=' => {
                self.header.little = true;
                Ok((Kind::Nop, 0))
            }
            b'>' => {
                self.header.little = false;
                Ok((Kind::Nop, 0))
            }
            b'!' => {
                self.header.max_align = self.read_size_limit(vm, NATIVE_WORD as i64)?;
                Ok((Kind::Nop, 0))
            }
            other => {
                let message = format!("invalid format option '{}'", char::from(other));
                Err(vm.runtime_error_text(&message))
            }
        }
    }
}

fn option_details(vm: &mut Vm, parser: &mut Parser<'_>, total: usize) -> Result<Detail, LuaError> {
    let (kind, size) = parser.read_option(vm)?;
    let align = if kind == Kind::PadAlign {
        aligned_option_size(vm, parser)?
    } else {
        size
    };
    let padding = alignment_padding(vm, &parser.header, kind, align, total)?;
    Ok(Detail { kind, size, padding })
}

fn aligned_option_size(vm: &mut Vm, parser: &mut Parser<'_>) -> Result<usize, LuaError> {
    if !parser.at_end() {
        let (kind, size) = parser.read_option(vm)?;
        if kind != Kind::Char && size != 0 {
            return Ok(size);
        }
    }
    Err(arg_error(
        vm,
        1,
        parser.header.name,
        "invalid next option for option 'X'",
    ))
}

fn alignment_padding(
    vm: &mut Vm,
    header: &Header,
    kind: Kind,
    align: usize,
    total: usize,
) -> Result<usize, LuaError> {
    if align <= 1 || kind == Kind::Char {
        return Ok(0);
    }
    let align = align.min(header.max_align);
    if !align.is_power_of_two() {
        return Err(arg_error(
            vm,
            1,
            header.name,
            "format asks for alignment not power of 2",
        ));
    }
    Ok((align - (total & (align - 1))) & (align - 1))
}

fn consumes_value(kind: Kind) -> bool {
    matches!(
        kind,
        Kind::Int
            | Kind::Uint
            | Kind::Float
            | Kind::Number
            | Kind::Double
            | Kind::Char
            | Kind::String
            | Kind::Zstr
    )
}

fn check_string(vm: &mut Vm, args: &[Value], index: usize, name: &str) -> Result<Ref<LuaString>, LuaError> {
    let value = argument(args, index);
    if let Value::String(string) = value {
        return Ok(string);
    }
    if matches!(value, Value::Integer(_) | Value::Float(_)) {
        let text = vm.to_display_bytes(value);
        if let Value::String(string) = vm.string_value(&text) {
            return Ok(string);
        }
    }
    let got = args.get(index).map_or("no value", |value| value.type_name());
    let message = format!("string expected, got {got}");
    Err(arg_error(vm, index + 1, name, &message))
}

fn format_bytes(vm: &mut Vm, args: &[Value], name: &str) -> Result<Vec<u8>, LuaError> {
    let format = check_string(vm, args, 0, name)?;
    Ok(vm.heap_bytes(format).to_vec())
}

fn push_in_byte_order(output: &mut Vec<u8>, bytes: &[u8], little: bool) {
    if little {
        output.extend_from_slice(bytes);
    } else {
        output.extend(bytes.iter().rev());
    }
}

fn append_integer(output: &mut Vec<u8>, value: u64, size: usize, little: bool, negative: bool) {
    let mut bytes = [0u8; MAX_INT_SIZE];
    for (index, byte) in bytes.iter_mut().take(size).enumerate() {
        *byte = if index < NATIVE_WORD {
            (value >> (index * 8)) as u8
        } else if negative {
            0xff
        } else {
            0
        };
    }
    push_in_byte_order(output, &bytes[..size], little);
}

fn in_byte_order<const N: usize>(bytes: &[u8], little: bool) -> [u8; N] {
    let mut raw = [0u8; N];
    raw.copy_from_slice(bytes);
    if !little {
        raw.reverse();
    }
    raw
}

fn unpack_integer(bytes: &[u8], little: bool, signed: bool) -> Option<i64> {
    let size = bytes.len();
    let byte_at = |index: usize| if little { bytes[index] } else { bytes[size - 1 - index] };
    let mut result: u64 = 0;
    for index in (0..size.min(NATIVE_WORD)).rev() {
        result = (result << 8) | u64::from(byte_at(index));
    }
    if size < NATIVE_WORD && signed {
        let mask = 1u64 << (size * 8 - 1);
        result = (result ^ mask).wrapping_sub(mask);
    }
    if size > NATIVE_WORD {
        let fill: u8 = if signed && (result as i64) < 0 { 0xff } else { 0 };
        if (NATIVE_WORD..size).any(|index| byte_at(index) != fill) {
            return None;
        }
    }
    Some(result as i64)
}

fn integer_or_error(vm: &mut Vm, decoded: Option<i64>, size: usize) -> Result<i64, LuaError> {
    decoded.ok_or_else(|| vm.runtime_error_text(&format!("{size}-byte integer does not fit into Lua Integer")))
}

fn absolute_position(position: i64, length: usize) -> usize {
    if position > 0 {
        return position as usize;
    }
    if position == 0 || position < -(length as i64) {
        return 1;
    }
    (length as i64 + position + 1) as usize
}

pub(super) fn pack(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let format = format_bytes(vm, args, "pack")?;
    let mut parser = Parser::new(&format, "pack");
    let mut output = Vec::new();
    let mut total = 0;
    let mut index = 1;
    let mut interrupts = Interrupts::new(vm);

    while !parser.at_end() {
        vm.check_interrupt(&mut interrupts)?;
        let option = option_details(vm, &mut parser, total)?;
        let size = option.padding + option.size;
        if size > MAX_SIZE - total {
            return Err(arg_error(vm, index, parser.header.name, "result too long"));
        }
        total += size;
        output.resize(output.len() + option.padding, PAD_BYTE);

        let value_index = index;
        index += usize::from(consumes_value(option.kind));
        total += pack_option(vm, args, value_index, option, &parser.header, &mut output)?;
    }
    Ok(values![vm.string_value(&output)])
}

fn pack_option(
    vm: &mut Vm,
    args: &[Value],
    index: usize,
    option: Detail,
    header: &Header,
    output: &mut Vec<u8>,
) -> Result<usize, LuaError> {
    match option.kind {
        Kind::Int => pack_signed(vm, args, index, option.size, header, output).map(|()| 0),
        Kind::Uint => pack_unsigned(vm, args, index, option.size, header, output).map(|()| 0),
        Kind::Float => pack_float(vm, args, index, header, output).map(|()| 0),
        Kind::Number | Kind::Double => pack_double(vm, args, index, header, output).map(|()| 0),
        Kind::Char => pack_fixed(vm, args, index, option.size, header, output).map(|()| 0),
        Kind::String => pack_prefixed(vm, args, index, option.size, header, output),
        Kind::Zstr => pack_zero_terminated(vm, args, index, header, output),
        Kind::Padding => {
            output.push(PAD_BYTE);
            Ok(0)
        }
        Kind::PadAlign | Kind::Nop => Ok(0),
    }
}

fn pack_signed(
    vm: &mut Vm,
    args: &[Value],
    index: usize,
    size: usize,
    header: &Header,
    output: &mut Vec<u8>,
) -> Result<(), LuaError> {
    let value = check_integer(vm, args, index, header.name)?;
    if size < NATIVE_WORD {
        let limit = 1i64 << (size * 8 - 1);
        if !(-limit <= value && value < limit) {
            return Err(arg_error(vm, index + 1, header.name, "integer overflow"));
        }
    }
    append_integer(output, value as u64, size, header.little, value < 0);
    Ok(())
}

fn pack_unsigned(
    vm: &mut Vm,
    args: &[Value],
    index: usize,
    size: usize,
    header: &Header,
    output: &mut Vec<u8>,
) -> Result<(), LuaError> {
    let value = check_integer(vm, args, index, header.name)? as u64;
    if size < NATIVE_WORD && value >= (1u64 << (size * 8)) {
        return Err(arg_error(vm, index + 1, header.name, "unsigned overflow"));
    }
    append_integer(output, value, size, header.little, false);
    Ok(())
}

fn pack_float(vm: &mut Vm, args: &[Value], index: usize, header: &Header, output: &mut Vec<u8>) -> Result<(), LuaError> {
    let value = check_number(vm, args, index, header.name)?.to_float();
    let bits = crate::number::canonical_single_bits(value);
    push_in_byte_order(output, &bits.to_le_bytes(), header.little);
    Ok(())
}

fn pack_double(vm: &mut Vm, args: &[Value], index: usize, header: &Header, output: &mut Vec<u8>) -> Result<(), LuaError> {
    let value = check_number(vm, args, index, header.name)?.to_float();
    let bits = crate::number::canonical_double_bits(value);
    push_in_byte_order(output, &bits.to_le_bytes(), header.little);
    Ok(())
}

fn pack_fixed(
    vm: &mut Vm,
    args: &[Value],
    index: usize,
    size: usize,
    header: &Header,
    output: &mut Vec<u8>,
) -> Result<(), LuaError> {
    let string = check_string(vm, args, index, header.name)?;
    let text = vm.heap_bytes(string).to_vec();
    if text.len() > size {
        return Err(arg_error(vm, index + 1, header.name, "string longer than given size"));
    }
    output.extend_from_slice(&text);
    output.resize(output.len() + size - text.len(), PAD_BYTE);
    Ok(())
}

fn pack_prefixed(
    vm: &mut Vm,
    args: &[Value],
    index: usize,
    size: usize,
    header: &Header,
    output: &mut Vec<u8>,
) -> Result<usize, LuaError> {
    let string = check_string(vm, args, index, header.name)?;
    let text = vm.heap_bytes(string).to_vec();
    let length = text.len();
    if size < NATIVE_WORD && (length as u64) >= (1u64 << (size * 8)) {
        return Err(arg_error(
            vm,
            index + 1,
            header.name,
            "string length does not fit in given size",
        ));
    }
    append_integer(output, length as u64, size, header.little, false);
    output.extend_from_slice(&text);
    Ok(length)
}

fn pack_zero_terminated(
    vm: &mut Vm,
    args: &[Value],
    index: usize,
    header: &Header,
    output: &mut Vec<u8>,
) -> Result<usize, LuaError> {
    let string = check_string(vm, args, index, header.name)?;
    let text = vm.heap_bytes(string).to_vec();
    if text.contains(&0) {
        return Err(arg_error(vm, index + 1, header.name, "string contains zeros"));
    }
    output.extend_from_slice(&text);
    output.push(0);
    Ok(text.len() + 1)
}

pub(super) fn packsize(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let format = format_bytes(vm, args, "packsize")?;
    let mut parser = Parser::new(&format, "packsize");
    let mut total = 0;
    let mut interrupts = Interrupts::new(vm);

    while !parser.at_end() {
        vm.check_interrupt(&mut interrupts)?;
        let option = option_details(vm, &mut parser, total)?;
        let name = parser.header.name;
        if matches!(option.kind, Kind::String | Kind::Zstr) {
            return Err(arg_error(vm, 1, name, "variable-length format"));
        }
        let size = option.padding + option.size;
        if size > MAX_SIZE || total > MAX_SIZE - size {
            return Err(arg_error(vm, 1, name, "format result too large"));
        }
        total += size;
    }
    Ok(values![Value::Integer(total as i64)])
}

fn optional_start(vm: &mut Vm, args: &[Value]) -> Result<i64, LuaError> {
    if matches!(argument(args, 2), Value::Nil) {
        return Ok(1);
    }
    check_integer(vm, args, 2, "unpack")
}

fn bytes_at(vm: &Vm, data: Ref<LuaString>, start: usize, length: usize) -> &[u8] {
    &vm.heap_bytes(data)[start..start + length]
}

pub(super) fn unpack(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let format = format_bytes(vm, args, "unpack")?;
    let data = check_string(vm, args, 1, "unpack")?;
    let data_len = vm.heap_bytes(data).len();
    let start = optional_start(vm, args)?;

    let mut position = absolute_position(start, data_len) - 1;
    if position > data_len {
        return Err(arg_error(vm, 3, "unpack", "initial position out of string"));
    }
    let mut parser = Parser::new(&format, "unpack");
    let mut results = Values::new();
    let mut interrupts = Interrupts::new(vm);

    while !parser.at_end() {
        vm.check_interrupt(&mut interrupts)?;
        if results.len() + args.len() + 2 > LUAI_MAXSTACK {
            return Err(vm.runtime_error_text("stack overflow (too many results)"));
        }
        let option = option_details(vm, &mut parser, position)?;
        if option.padding + option.size > data_len - position {
            return Err(arg_error(vm, 2, parser.header.name, "data string too short"));
        }
        position += option.padding;
        let (value, extra) = unpack_option(vm, data, position, option, &parser.header)?;
        results.extend(value);
        position += extra + option.size;
    }
    results.push(Value::Integer(position as i64 + 1));
    Ok(results)
}

fn unpack_option(
    vm: &mut Vm,
    data: Ref<LuaString>,
    position: usize,
    option: Detail,
    header: &Header,
) -> Result<(Option<Value>, usize), LuaError> {
    let size = option.size;
    match option.kind {
        Kind::Int | Kind::Uint => {
            let decoded = unpack_integer(bytes_at(vm, data, position, size), header.little, option.kind == Kind::Int);
            let value = integer_or_error(vm, decoded, size)?;
            Ok((Some(Value::Integer(value)), 0))
        }
        Kind::Float => {
            let raw = in_byte_order::<4>(bytes_at(vm, data, position, size), header.little);
            Ok((Some(Value::Float(f64::from(f32::from_le_bytes(raw)))), 0))
        }
        Kind::Number | Kind::Double => {
            let raw = in_byte_order::<8>(bytes_at(vm, data, position, size), header.little);
            Ok((Some(Value::Float(f64::from_le_bytes(raw))), 0))
        }
        Kind::Char => {
            let text = bytes_at(vm, data, position, size).to_vec();
            Ok((Some(vm.string_value(&text)), 0))
        }
        Kind::String => unpack_prefixed(vm, data, position, size, header),
        Kind::Zstr => unpack_zero_terminated(vm, data, position, header),
        Kind::Padding | Kind::PadAlign | Kind::Nop => Ok((None, 0)),
    }
}

fn unpack_prefixed(
    vm: &mut Vm,
    data: Ref<LuaString>,
    position: usize,
    size: usize,
    header: &Header,
) -> Result<(Option<Value>, usize), LuaError> {
    let decoded = unpack_integer(bytes_at(vm, data, position, size), header.little, false);
    let length = integer_or_error(vm, decoded, size)? as usize;
    let remaining = vm.heap_bytes(data).len() - position - size;
    if length > remaining {
        return Err(arg_error(vm, 2, header.name, "data string too short"));
    }
    let text = bytes_at(vm, data, position + size, length).to_vec();
    Ok((Some(vm.string_value(&text)), length))
}

fn unpack_zero_terminated(
    vm: &mut Vm,
    data: Ref<LuaString>,
    position: usize,
    header: &Header,
) -> Result<(Option<Value>, usize), LuaError> {
    let data_len = vm.heap_bytes(data).len();
    let rest = &vm.heap_bytes(data)[position..];
    let length = rest.iter().position(|&byte| byte == 0).unwrap_or(rest.len());
    if position + length >= data_len {
        return Err(arg_error(vm, 2, header.name, "unfinished string for format 'z'"));
    }
    let text = bytes_at(vm, data, position, length).to_vec();
    Ok((Some(vm.string_value(&text)), length + 1))
}

#[cfg(test)]
mod tests {
    fn run(source: &str) -> String {
        let mut vm = crate::Vm::new(crate::Options::default());
        match vm.run_source(b"=t", source.as_bytes()) {
            Ok(()) => String::from_utf8_lossy(&vm.take_output()).into_owned(),
            Err(error) => String::from_utf8_lossy(&vm.error_message(&error)).into_owned(),
        }
    }

    #[test]
    fn packs_integers_in_both_byte_orders() {
        let source = r#"
print(string.pack("<i4", 258) == "\2\1\0\0")
print(string.pack(">i4", 258) == "\0\0\1\2")
print(string.pack("<i2", -2) == "\254\255")
print(string.pack("=I2", 1) == "\1\0")
"#;
        assert_eq!(run(source), "true\ntrue\ntrue\ntrue\n");
    }

    #[test]
    fn unpacks_integers() {
        let source = r#"
print(string.unpack("<i2", "\254\255"))
print(string.unpack(">I2", "\1\2"))
print(string.unpack("<I1", "\255"))
print(string.unpack("<i2", "\1\2\3", 2))
"#;
        assert_eq!(run(source), "-2\t3\n258\t3\n255\t2\n770\t4\n");
    }

    #[test]
    fn wide_integers_keep_sign() {
        let source = r#"
print(string.unpack("<i16", string.pack("<i16", -3)))
print(string.pack(">i10", -1) == "\255\255\255\255\255\255\255\255\255\255")
print(string.unpack("<I9", "\1\0\0\0\0\0\0\0\0"))
"#;
        assert_eq!(run(source), "-3\t17\ntrue\n1\t10\n");
    }

    #[test]
    fn packsize_sums_options() {
        let source = r#"
print(string.packsize("i4i8"))
print(string.packsize("!8 b d"))
print(string.packsize("c10 n f"))
print(string.packsize("!4 i1 Xi4 i2"))
print(string.packsize(""))
"#;
        assert_eq!(run(source), "12\n16\n22\n6\n0\n");
    }

    #[test]
    fn alignment_pads_to_option_size() {
        let source = r#"
print(string.pack("!4 b i4", 1, 2) == "\1\0\0\0\2\0\0\0")
print(string.pack("!2 b h", 1, 2) == "\1\0\2\0")
print(string.packsize("!4 i1 i2 i4"))
"#;
        assert_eq!(run(source), "true\ntrue\n8\n");
    }

    #[test]
    fn packs_floats() {
        let source = r#"
print(string.pack("<d", 1.0) == "\0\0\0\0\0\0\240\63")
print(string.pack(">f", 1.0) == "\63\128\0\0")
print(string.pack(">n", 1.0) == "\63\240\0\0\0\0\0\0")
"#;
        assert_eq!(run(source), "true\ntrue\ntrue\n");
    }

    #[test]
    fn nan_packs_as_the_canonical_quiet_nan() {
        let source = r#"
print(string.pack("<d", -(0/0)) == "\0\0\0\0\0\0\248\127")
print(string.pack(">d", 0/0) == "\127\248\0\0\0\0\0\0")
print(string.pack("<f", -(0/0)) == "\0\0\192\127")
print(string.pack(">n", 0/0) == "\127\248\0\0\0\0\0\0")
"#;
        assert_eq!(run(source), "true\ntrue\ntrue\ntrue\n");
    }

    #[test]
    fn unpacks_floats() {
        let source = r#"
print(string.unpack(">f", string.pack(">f", 0.5)))
print(string.unpack("<n", string.pack("<n", -2.5)))
print(string.unpack("<d", string.pack("<d", 3)))
"#;
        assert_eq!(run(source), "0.5\t5\n-2.5\t9\n3.0\t9\n");
    }

    #[test]
    fn packs_and_unpacks_strings() {
        let source = r#"
print(string.pack("s1", "hi") == "\2hi")
print(string.pack("z", "hi") == "hi\0")
print(string.pack("c4", "ab") == "ab\0\0")
print(string.unpack("c2", "ab"))
print(string.unpack("s1", "\2hi"))
print(string.unpack("z z", "ab\0cd\0"))
"#;
        assert_eq!(run(source), "true\ntrue\ntrue\nab\t3\nhi\t4\nab\tcd\t7\n");
    }

    #[test]
    fn unpack_honors_initial_position() {
        assert_eq!(run(r#"print(string.unpack("i1", "\1\2\3", -1))"#), "3\t4\n");
    }

    #[test]
    fn integer_range_errors() {
        assert_eq!(
            run(r#"string.pack("i17", 1)"#),
            "t:1: integral size (17) out of limits [1,16]"
        );
        assert_eq!(
            run(r#"string.pack("i0", 1)"#),
            "t:1: integral size (0) out of limits [1,16]"
        );
        assert_eq!(
            run(r#"string.pack("i1", 128)"#),
            "t:1: bad argument #2 to 'pack' (integer overflow)"
        );
        assert_eq!(
            run(r#"string.pack("I1", 256)"#),
            "t:1: bad argument #2 to 'pack' (unsigned overflow)"
        );
        assert_eq!(
            run(r#"string.pack("I1", -1)"#),
            "t:1: bad argument #2 to 'pack' (unsigned overflow)"
        );
    }

    #[test]
    fn option_errors() {
        assert_eq!(
            run(r#"string.pack("c", "x")"#),
            "t:1: missing size for format option 'c'"
        );
        assert_eq!(run(r#"string.pack("y")"#), "t:1: invalid format option 'y'");
        assert_eq!(
            run(r#"string.pack("!3 i4", 1)"#),
            "t:1: bad argument #1 to 'pack' (format asks for alignment not power of 2)"
        );
        assert_eq!(
            run(r#"string.pack("X", 1)"#),
            "t:1: bad argument #1 to 'pack' (invalid next option for option 'X')"
        );
        assert_eq!(
            run(r#"string.packsize("s")"#),
            "t:1: bad argument #1 to 'packsize' (variable-length format)"
        );
    }

    #[test]
    fn argument_errors() {
        assert_eq!(
            run(r#"string.pack("c2", "abc")"#),
            "t:1: bad argument #2 to 'pack' (string longer than given size)"
        );
        assert_eq!(
            run(r#"string.pack("z", "a\0b")"#),
            "t:1: bad argument #2 to 'pack' (string contains zeros)"
        );
        assert_eq!(
            run(r#"string.pack("i4", "x")"#),
            "t:1: bad argument #2 to 'pack' (number expected, got string)"
        );
        assert_eq!(
            run(r#"string.pack("i4")"#),
            "t:1: bad argument #2 to 'pack' (number expected, got no value)"
        );
    }

    #[test]
    fn unpack_errors() {
        assert_eq!(
            run(r#"string.unpack("i4", "ab")"#),
            "t:1: bad argument #2 to 'unpack' (data string too short)"
        );
        assert_eq!(
            run(r#"string.unpack("z", "abc")"#),
            "t:1: bad argument #2 to 'unpack' (unfinished string for format 'z')"
        );
        assert_eq!(
            run(r#"string.unpack("i1", "a", 5)"#),
            "t:1: bad argument #3 to 'unpack' (initial position out of string)"
        );
        assert_eq!(
            run(r#"string.unpack("i9", "\0\0\0\0\0\0\0\0\1")"#),
            "t:1: 9-byte integer does not fit into Lua Integer"
        );
        assert_eq!(
            run(r#"string.unpack("s1", "\5ab")"#),
            "t:1: bad argument #2 to 'unpack' (data string too short)"
        );
    }
}
