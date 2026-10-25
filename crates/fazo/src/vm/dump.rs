//! Binary chunks in the format of `ldump.c` and `lundump.c` of PUC Lua 5.5.1 (MIT, Copyright (C) 1994-2025 Lua.org, PUC-Rio).

use std::collections::HashMap;

use super::verify::{self, Shape};
use crate::{
    Vm,
    arena::Ref,
    encoding,
    instruction::Instruction,
    object::Closure,
    proto::{LocalVar, Proto, UpvalueDesc},
    syntax::chunk_id::chunk_id,
    value::Value,
};

const SIGNATURE: &[u8] = b"\x1bLua";
const VERSION: u8 = 0x55;
const FORMAT: u8 = 0;
const DATA: &[u8] = b"\x19\x93\r\n\x1a\n";
const CHECK_INTEGER: i64 = -0x5678;
const CHECK_INSTRUCTION: u32 = 0x1234_5678;
const CHECK_NUMBER: f64 = -370.5;
const ABSOLUTE_LINE: i8 = -0x80;
const LINE_DIFFERENCE_LIMIT: i64 = 0x80;
const ABSOLUTE_LINE_INTERVAL: u8 = 128;
const PF_VAHID: u8 = 1;
const PF_VATAB: u8 = 2;
const PF_FIXED: u8 = 4;
const SHORT_STRING_LIMIT: usize = 40;
const TAG_NIL: u8 = 0x00;
const TAG_FALSE: u8 = 0x01;
const TAG_TRUE: u8 = 0x11;
const TAG_FLOAT: u8 = 0x13;
const TAG_INTEGER: u8 = 0x03;
const TAG_SHORT_STRING: u8 = 0x04;
const TAG_LONG_STRING: u8 = 0x14;
const MISSING_SOURCE: &[u8] = b"=?";

struct Encoder {
    out: Vec<u8>,
    saved: HashMap<Vec<u8>, u64>,
    strip: bool,
}

impl Encoder {
    fn byte(&mut self, value: u8) {
        self.out.push(value);
    }

    fn varint(&mut self, mut value: u64) {
        let mut buffer = [0_u8; 10];
        let mut index = buffer.len() - 1;
        buffer[index] = (value & 0x7f) as u8;
        value >>= 7;
        while value != 0 {
            index -= 1;
            buffer[index] = ((value & 0x7f) as u8) | 0x80;
            value >>= 7;
        }
        self.out.extend_from_slice(&buffer[index..]);
    }

    fn integer(&mut self, value: i64) {
        let encoded = if value >= 0 {
            2 * value as u64
        } else {
            2 * !(value as u64) + 1
        };
        self.varint(encoded);
    }

    fn align(&mut self, alignment: usize) {
        let padding = (alignment - self.out.len() % alignment) % alignment;
        self.out.resize(self.out.len() + padding, 0);
    }

    /// Follows `dumpString` in ldump.c.
    fn string(&mut self, bytes: Option<&[u8]>) {
        let Some(bytes) = bytes else {
            self.varint(0);
            self.varint(0);
            return;
        };
        if let Some(index) = self.saved.get(bytes).copied() {
            self.varint(0);
            self.varint(index);
            return;
        }
        self.varint(bytes.len() as u64 + 1);
        self.out.extend_from_slice(bytes);
        self.out.push(0);
        let index = self.saved.len() as u64 + 1;
        self.saved.insert(bytes.to_vec(), index);
    }

    fn header(&mut self) {
        self.out.extend_from_slice(SIGNATURE);
        self.byte(VERSION);
        self.byte(FORMAT);
        self.out.extend_from_slice(DATA);
        self.byte(4);
        self.out.extend_from_slice(&(CHECK_INTEGER as i32).to_le_bytes());
        self.byte(4);
        self.out.extend_from_slice(&CHECK_INSTRUCTION.to_le_bytes());
        self.byte(8);
        self.out.extend_from_slice(&CHECK_INTEGER.to_le_bytes());
        self.byte(8);
        self.out.extend_from_slice(&CHECK_NUMBER.to_le_bytes());
    }
}

/// Follows `savelineinfo` in ldump.c.
fn line_info(line_defined: u32, lines: &[u32]) -> (Vec<i8>, Vec<(u32, u32)>) {
    let mut relative = Vec::with_capacity(lines.len());
    let mut absolute = Vec::new();
    let mut previous = i64::from(line_defined);
    let mut since_absolute: u8 = 0;
    for (pc, &line) in lines.iter().enumerate() {
        let line = i64::from(line);
        let difference = line - previous;
        let needs_absolute = difference.abs() >= LINE_DIFFERENCE_LIMIT || {
            let issued = since_absolute;
            since_absolute = since_absolute.wrapping_add(1);
            issued >= ABSOLUTE_LINE_INTERVAL
        };
        if needs_absolute {
            absolute.push((pc as u32, line as u32));
            relative.push(ABSOLUTE_LINE);
            since_absolute = 1;
        } else {
            relative.push(difference as i8);
        }
        previous = line;
    }
    (relative, absolute)
}

fn flag_byte(proto: &Proto) -> u8 {
    let mut flag = 0;
    if proto.is_vararg && !proto.vararg_table {
        flag |= PF_VAHID;
    }
    if proto.vararg_table {
        flag |= PF_VATAB;
    }
    flag
}

impl Vm {
    pub(crate) fn dump_closure(&self, closure: Ref<Closure>, strip: bool) -> Vec<u8> {
        let proto = self.heap.closures.get(closure).proto;
        let mut encoder = Encoder {
            out: Vec::new(),
            saved: HashMap::new(),
            strip,
        };
        encoder.header();
        let upvalues = self.heap.protos.get(proto).upvalues.len();
        encoder.byte(upvalues as u8);
        self.dump_function(&mut encoder, proto);
        encoder.out
    }

    fn dump_function(&self, encoder: &mut Encoder, reference: Ref<Proto>) {
        let proto = self.heap.protos.get(reference);
        encoder.varint(u64::from(proto.line_defined));
        encoder.varint(u64::from(proto.last_line_defined));
        encoder.byte(proto.num_params);
        encoder.byte(flag_byte(proto));
        encoder.byte(proto.max_stack);
        self.dump_code(encoder, proto);
        self.dump_constants(encoder, proto);
        self.dump_upvalues(encoder, proto);
        encoder.varint(proto.protos.len() as u64);
        for child in &proto.protos {
            self.dump_function(encoder, *child);
        }
        let source = (!encoder.strip).then_some(&proto.chunkname[..]);
        encoder.string(source);
        self.dump_debug(encoder, proto);
    }

    fn dump_code(&self, encoder: &mut Encoder, proto: &Proto) {
        encoder.varint(proto.code.len() as u64);
        encoder.align(4);
        for &instruction in &proto.code {
            encoder.out.extend_from_slice(&encoding::encode(instruction).to_le_bytes());
        }
    }

    fn dump_constants(&self, encoder: &mut Encoder, proto: &Proto) {
        encoder.varint(proto.constants.len() as u64);
        for &value in &proto.constants {
            match value {
                Value::Boolean(false) => encoder.byte(TAG_FALSE),
                Value::Boolean(true) => encoder.byte(TAG_TRUE),
                Value::Float(number) => {
                    encoder.byte(TAG_FLOAT);
                    let bits = crate::number::canonical_double_bits(number);
                    encoder.out.extend_from_slice(&bits.to_le_bytes());
                }
                Value::Integer(integer) => {
                    encoder.byte(TAG_INTEGER);
                    encoder.integer(integer);
                }
                Value::String(string) => {
                    let bytes = self.heap.bytes(string);
                    encoder.byte(if bytes.len() <= SHORT_STRING_LIMIT {
                        TAG_SHORT_STRING
                    } else {
                        TAG_LONG_STRING
                    });
                    encoder.string(Some(bytes));
                }
                _ => encoder.byte(TAG_NIL),
            }
        }
    }

    fn dump_upvalues(&self, encoder: &mut Encoder, proto: &Proto) {
        encoder.varint(proto.upvalues.len() as u64);
        for upvalue in &proto.upvalues {
            encoder.byte(u8::from(upvalue.in_stack));
            encoder.byte(upvalue.index);
            encoder.byte(0);
        }
    }

    fn dump_debug(&self, encoder: &mut Encoder, proto: &Proto) {
        if encoder.strip {
            for _ in 0..4 {
                encoder.varint(0);
            }
            return;
        }
        let (relative, absolute) = line_info(proto.line_defined, &proto.lines);
        encoder.varint(relative.len() as u64);
        encoder.out.extend(relative.iter().map(|&delta| delta as u8));
        encoder.varint(absolute.len() as u64);
        if !absolute.is_empty() {
            encoder.align(4);
            for (pc, line) in absolute {
                encoder.out.extend_from_slice(&pc.to_le_bytes());
                encoder.out.extend_from_slice(&line.to_le_bytes());
            }
        }
        encoder.varint(proto.local_vars.len() as u64);
        for var in &proto.local_vars {
            encoder.string(Some(super::naming::name_bytes(&self.heap, var.name)));
            encoder.varint(var.start_pc as u64);
            encoder.varint(var.end_pc as u64);
        }
        encoder.varint(proto.upvalues.len() as u64);
        for upvalue in &proto.upvalues {
            encoder.string(Some(super::naming::name_bytes(&self.heap, upvalue.name)));
        }
    }

    pub(crate) fn load_binary(&mut self, chunkname: &[u8], bytes: &[u8], environment: Option<Value>) -> Result<Value, Vec<u8>> {
        let mut decoder = Decoder {
            bytes,
            position: 0,
            name: binary_name(chunkname),
            saved: Vec::new(),
        };
        decoder.header()?;
        let _main_upvalues = decoder.byte()?;
        let function = decoder.function()?;
        if let Err(reason) = verify::check(&function.shape()) {
            return Err(decoder.fail(reason));
        }
        let proto = self.build_protos(function);
        let environment = environment.unwrap_or(Value::Table(self.globals));
        Ok(self.closure_with_environment(proto, environment))
    }

    fn build_protos(&mut self, function: Loaded) -> Ref<Proto> {
        let protos = function.protos.into_iter().map(|child| self.build_protos(child)).collect();
        let upvalues = function
            .upvalues
            .into_iter()
            .zip(function.upvalue_names)
            .map(|(upvalue, name)| UpvalueDesc { name: name.map(|name| self.heap.string(&name)), ..upvalue })
            .collect();
        let local_vars = function
            .local_vars
            .into_iter()
            .map(|(name, start_pc, end_pc)| LocalVar { name: name.map(|name| self.heap.string(&name)), start_pc, end_pc })
            .collect();
        let constants = function
            .constants
            .into_iter()
            .map(|constant| match constant {
                Constant::Nil => Value::Nil,
                Constant::Boolean(flag) => Value::Boolean(flag),
                Constant::Integer(integer) => Value::Integer(integer),
                Constant::Float(float) => Value::Float(float),
                Constant::String(bytes) => self.heap.string_value(&bytes),
            })
            .collect();
        let exec = Box::new(crate::vm::Exec::new(&function.code));
        self.heap.alloc_proto(Proto {
            source: chunk_id(&function.chunkname).into(),
            chunkname: function.chunkname.into(),
            line_defined: function.line_defined,
            last_line_defined: function.last_line_defined,
            num_params: function.num_params,
            is_vararg: function.is_vararg,
            vararg_table: function.vararg_table,
            max_stack: function.max_stack,
            code: function.code,
            lines: function.lines,
            constants,
            protos,
            upvalues,
            local_vars,
            exec,
            calls: 0,
            #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
            jit: Default::default(),
        })
    }

    fn closure_with_environment(&mut self, proto: Ref<Proto>, environment: Value) -> Value {
        let count = self.heap.protos.get(proto).upvalues.len();
        let upvalues = (0..count)
            .map(|index| {
                let value = if index == 0 { environment } else { Value::Nil };
                let upvalue = self.heap.alloc_closed_upvalue(value);
                self.heap.captured(upvalue)
            })
            .collect();
        Value::Closure(self.heap.alloc_closure(proto, upvalues))
    }
}

/// Follows `luaU_undump` in lundump.c.
fn binary_name(chunkname: &[u8]) -> Vec<u8> {
    match chunkname.first() {
        Some(b'@' | b'=') => chunkname[1..].to_vec(),
        Some(0x1b) => b"binary string".to_vec(),
        _ => chunkname.to_vec(),
    }
}

enum Constant {
    Nil,
    Boolean(bool),
    Integer(i64),
    Float(f64),
    String(Vec<u8>),
}

struct Loaded {
    chunkname: Vec<u8>,
    line_defined: u32,
    last_line_defined: u32,
    num_params: u8,
    is_vararg: bool,
    vararg_table: bool,
    max_stack: u8,
    code: Vec<Instruction>,
    lines: Vec<u32>,
    constants: Vec<Constant>,
    protos: Vec<Loaded>,
    upvalues: Vec<UpvalueDesc>,
    upvalue_names: Vec<Option<Vec<u8>>>,
    local_vars: Vec<(Option<Vec<u8>>, usize, usize)>,
}

impl Loaded {
    fn shape(&self) -> Shape<'_> {
        Shape {
            code: &self.code,
            num_params: self.num_params,
            max_stack: self.max_stack,
            constant_count: self.constants.len(),
            upvalues: &self.upvalues,
            children: self.protos.iter().map(Loaded::shape).collect(),
        }
    }
}

struct Decoder<'a> {
    bytes: &'a [u8],
    position: usize,
    name: Vec<u8>,
    saved: Vec<Vec<u8>>,
}

impl<'a> Decoder<'a> {
    fn fail(&self, reason: &str) -> Vec<u8> {
        let mut message = self.name.clone();
        message.extend_from_slice(b": bad binary format (");
        message.extend_from_slice(reason.as_bytes());
        message.push(b')');
        message
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], Vec<u8>> {
        let end = self.position.checked_add(count).filter(|end| *end <= self.bytes.len());
        let Some(end) = end else {
            return Err(self.fail("truncated chunk"));
        };
        let slice = &self.bytes[self.position..end];
        self.position = end;
        Ok(slice)
    }

    fn byte(&mut self) -> Result<u8, Vec<u8>> {
        Ok(self.take(1)?[0])
    }

    fn varint(&mut self) -> Result<u64, Vec<u8>> {
        let mut value: u64 = 0;
        loop {
            let byte = self.byte()?;
            if value >> 57 != 0 {
                return Err(self.fail("integer overflow"));
            }
            value = (value << 7) | u64::from(byte & 0x7f);
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
    }

    /// Values must fit a C `int`, as in lundump.c.
    fn int(&mut self) -> Result<u32, Vec<u8>> {
        let value = self.varint()?;
        u32::try_from(value)
            .ok()
            .filter(|value| *value <= i32::MAX as u32)
            .ok_or_else(|| self.fail("integer overflow"))
    }

    fn raw_int(&mut self) -> Result<u32, Vec<u8>> {
        let value = u32::from_le_bytes(self.take(4)?.try_into().expect("four bytes"));
        if value > i32::MAX as u32 {
            return Err(self.fail("integer overflow"));
        }
        Ok(value)
    }

    fn integer(&mut self) -> Result<i64, Vec<u8>> {
        let encoded = self.varint()?;
        let magnitude = (encoded >> 1) as i64;
        Ok(if encoded & 1 == 1 { !magnitude } else { magnitude })
    }

    fn align(&mut self, alignment: usize) -> Result<(), Vec<u8>> {
        let padding = (alignment - self.position % alignment) % alignment;
        self.take(padding)?;
        Ok(())
    }

    /// Follows `loadString` in lundump.c.
    fn string(&mut self) -> Result<Option<Vec<u8>>, Vec<u8>> {
        let size = self.varint()?;
        if size == 0 {
            let index = self.varint()?;
            if index == 0 {
                return Ok(None);
            }
            let saved = usize::try_from(index - 1).ok().and_then(|index| self.saved.get(index).cloned());
            return saved.map(Some).ok_or_else(|| self.fail("invalid string index"));
        }
        let length = usize::try_from(size - 1).map_err(|_| self.fail("integer overflow"))?;
        let stored = self.take(length + 1)?;
        if stored[length] != 0 {
            return Err(self.fail("invalid string"));
        }
        let bytes = stored[..length].to_vec();
        self.saved.push(bytes.clone());
        Ok(Some(bytes))
    }

    fn header(&mut self) -> Result<(), Vec<u8>> {
        if self.take(SIGNATURE.len())? != SIGNATURE {
            return Err(self.fail("not a binary chunk"));
        }
        if self.byte()? != VERSION {
            return Err(self.fail("version mismatch"));
        }
        if self.byte()? != FORMAT {
            return Err(self.fail("format mismatch"));
        }
        if self.take(DATA.len())? != DATA {
            return Err(self.fail("corrupted chunk"));
        }
        self.check_size(4, "int")?;
        let int_check = i32::from_le_bytes(self.take(4)?.try_into().expect("four bytes"));
        if i64::from(int_check) != CHECK_INTEGER {
            return Err(self.fail("int format mismatch"));
        }
        self.check_size(4, "instruction")?;
        let instruction_check = u32::from_le_bytes(self.take(4)?.try_into().expect("four bytes"));
        if instruction_check != CHECK_INSTRUCTION {
            return Err(self.fail("instruction format mismatch"));
        }
        self.check_size(8, "Lua integer")?;
        let integer_check = i64::from_le_bytes(self.take(8)?.try_into().expect("eight bytes"));
        if integer_check != CHECK_INTEGER {
            return Err(self.fail("Lua integer format mismatch"));
        }
        self.check_size(8, "Lua number")?;
        let number_check = f64::from_le_bytes(self.take(8)?.try_into().expect("eight bytes"));
        if number_check != CHECK_NUMBER {
            return Err(self.fail("Lua number format mismatch"));
        }
        Ok(())
    }

    fn check_size(&mut self, size: u8, name: &str) -> Result<(), Vec<u8>> {
        if self.byte()? != size {
            return Err(self.fail(&format!("{name} size mismatch")));
        }
        Ok(())
    }

    /// Follows `loadFunction` in lundump.c; protos come before the source, as ldump.c writes them.
    fn function(&mut self) -> Result<Loaded, Vec<u8>> {
        let line_defined = self.int()?;
        let last_line_defined = self.int()?;
        let num_params = self.byte()?;
        let flag = self.byte()? & !PF_FIXED;
        let max_stack = self.byte()?;
        let code = self.code()?;
        let constants = self.constants()?;
        let upvalues = self.upvalues()?;
        let count = self.int()?;
        let mut protos = Vec::new();
        for _ in 0..count {
            protos.push(self.function()?);
        }
        let chunkname = self.string()?.unwrap_or_else(|| MISSING_SOURCE.to_vec());
        let mut function = Loaded {
            chunkname,
            line_defined,
            last_line_defined,
            num_params,
            is_vararg: flag & (PF_VAHID | PF_VATAB) != 0,
            vararg_table: flag & PF_VATAB != 0,
            max_stack,
            code,
            lines: Vec::new(),
            constants,
            protos,
            upvalue_names: vec![None; upvalues.len()],
            upvalues,
            local_vars: Vec::new(),
        };
        self.debug(&mut function)?;
        Ok(function)
    }

    fn code(&mut self) -> Result<Vec<Instruction>, Vec<u8>> {
        let count = self.int()?;
        self.align(4)?;
        let mut code = Vec::with_capacity(count as usize);
        for _ in 0..count {
            let word = u32::from_le_bytes(self.take(4)?.try_into().expect("four bytes"));
            let Some(instruction) = encoding::decode(word) else {
                return Err(self.fail("invalid opcode"));
            };
            code.push(instruction);
        }
        Ok(code)
    }

    fn constants(&mut self) -> Result<Vec<Constant>, Vec<u8>> {
        let count = self.int()?;
        let mut constants = Vec::with_capacity(count as usize);
        for _ in 0..count {
            let constant = match self.byte()? {
                TAG_NIL => Constant::Nil,
                TAG_FALSE => Constant::Boolean(false),
                TAG_TRUE => Constant::Boolean(true),
                TAG_FLOAT => Constant::Float(f64::from_le_bytes(self.take(8)?.try_into().expect("eight bytes"))),
                TAG_INTEGER => Constant::Integer(self.integer()?),
                TAG_SHORT_STRING | TAG_LONG_STRING => match self.string()? {
                    Some(bytes) => Constant::String(bytes),
                    None => return Err(self.fail("bad format for constant string")),
                },
                _ => return Err(self.fail("invalid constant")),
            };
            constants.push(constant);
        }
        Ok(constants)
    }

    fn upvalues(&mut self) -> Result<Vec<UpvalueDesc>, Vec<u8>> {
        let count = self.int()?;
        let mut upvalues = Vec::with_capacity(count as usize);
        for _ in 0..count {
            let in_stack = self.byte()? != 0;
            let index = self.byte()?;
            let _kind = self.byte()?;
            upvalues.push(UpvalueDesc {
                name: None,
                in_stack,
                index,
            });
        }
        Ok(upvalues)
    }

    /// Follows `loadDebug` in lundump.c.
    fn debug(&mut self, function: &mut Loaded) -> Result<(), Vec<u8>> {
        let relative_count = self.int()?;
        let relative: Vec<i8> = self.take(relative_count as usize)?.iter().map(|&byte| byte as i8).collect();
        let absolute_count = self.int()?;
        let mut absolute = Vec::with_capacity(absolute_count as usize);
        if absolute_count > 0 {
            self.align(4)?;
        }
        for _ in 0..absolute_count {
            let pc = self.raw_int()?;
            let line = self.raw_int()?;
            absolute.push((pc, line));
        }
        function.lines = self.lines(function.line_defined, &relative, &absolute)?;
        let local_count = self.int()?;
        for _ in 0..local_count {
            let name = Some(self.string()?.unwrap_or_default());
            let start_pc = self.int()? as usize;
            let end_pc = self.int()? as usize;
            function.local_vars.push((name, start_pc, end_pc));
        }
        let names = self.int()? as usize;
        if names != 0 && names != function.upvalues.len() {
            return Err(self.fail("bad upvalue names"));
        }
        for index in 0..names {
            function.upvalue_names[index] = Some(self.string()?.unwrap_or_default());
        }
        Ok(())
    }

    fn lines(&self, line_defined: u32, relative: &[i8], absolute: &[(u32, u32)]) -> Result<Vec<u32>, Vec<u8>> {
        if relative.is_empty() {
            return Ok(Vec::new());
        }
        let mut lines = Vec::with_capacity(relative.len());
        let mut line = i64::from(line_defined);
        let mut next_absolute = 0;
        for (pc, &delta) in relative.iter().enumerate() {
            if delta == ABSOLUTE_LINE {
                let Some(&(absolute_pc, absolute_line)) = absolute.get(next_absolute) else {
                    return Err(self.fail("bad line information"));
                };
                if absolute_pc as usize != pc {
                    return Err(self.fail("bad line information"));
                }
                line = i64::from(absolute_line);
                next_absolute += 1;
            } else {
                line += i64::from(delta);
            }
            lines.push(line as u32);
        }
        if next_absolute != absolute.len() {
            return Err(self.fail("bad line information"));
        }
        Ok(lines)
    }
}

#[cfg(test)]
mod tests {
    use super::{CHECK_NUMBER, FORMAT, SIGNATURE, VERSION};
    use crate::{Options, Vm, value::Value};

    fn dumped(source: &str, strip: bool) -> Vec<u8> {
        let mut vm = Vm::new(Options::default());
        let function = vm.load(b"=t", source.as_bytes()).expect("the source compiles");
        let Value::Closure(closure) = function else {
            panic!("load returns a closure");
        };
        vm.dump_closure(closure, strip)
    }

    fn load_failure(bytes: &[u8]) -> String {
        let mut vm = Vm::new(Options::default());
        match vm.load_binary(b"=t", bytes, None) {
            Ok(_) => panic!("the chunk loads"),
            Err(message) => String::from_utf8_lossy(&message).into_owned(),
        }
    }

    #[test]
    fn a_compiled_function_round_trips_through_a_binary_chunk() {
        let bytes = dumped("local t = {1, 2}; return t[1] + #t, function() return t end", false);
        assert!(bytes.starts_with(SIGNATURE));
        let mut vm = Vm::new(Options::default());
        assert!(vm.load_binary(b"=t", &bytes, None).is_ok());
    }

    #[test]
    fn the_header_carries_the_reference_version_and_check_values() {
        let bytes = dumped("return 1", false);
        assert_eq!(bytes[4], VERSION);
        assert_eq!(bytes[5], FORMAT);
        let number_at = SIGNATURE.len() + 2 + 6 + 5 + 5 + 9 + 1;
        assert_eq!(f64::from_le_bytes(bytes[number_at..number_at + 8].try_into().unwrap()), CHECK_NUMBER);
    }

    #[test]
    fn a_chunk_with_the_wrong_signature_is_not_binary() {
        let mut bytes = dumped("return 1", false);
        bytes[0] = b'X';
        assert_eq!(load_failure(&bytes), "t: bad binary format (not a binary chunk)");
    }

    #[test]
    fn a_truncated_chunk_is_reported_as_truncated() {
        let bytes = dumped("return 1 + 2", false);
        assert_eq!(load_failure(&bytes[..bytes.len() - 3]), "t: bad binary format (truncated chunk)");
    }

    #[test]
    fn a_version_or_format_mismatch_is_reported() {
        let mut bytes = dumped("return 1", false);
        bytes[4] = 0x53;
        assert_eq!(load_failure(&bytes), "t: bad binary format (version mismatch)");
        let mut bytes = dumped("return 1", false);
        bytes[5] = 1;
        assert_eq!(load_failure(&bytes), "t: bad binary format (format mismatch)");
    }

    #[test]
    fn a_stripped_chunk_loads() {
        let bytes = dumped("local x = nil; return x.y", true);
        let mut loader = Vm::new(Options::default());
        assert!(loader.load_binary(b"=t", &bytes, None).is_ok());
    }
}

#[cfg(test)]
mod fuzz_tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use crate::{Options, Outcome, Vm};

    const SEEDS: u64 = 2_000;
    const PAUSES_BEFORE_INTERRUPT: usize = 200;

    const SAMPLE: &[u8] = b"local function f(a, ...)\n\
          local t = {a, ...}\n\
          local s = 0\n\
          for i = 1, #t do s = s + t[i] end\n\
          for k, v in pairs({x = 1}) do s = s + v end\n\
          local o = setmetatable({}, {__add = function(x, y) return 1 end, __call = function(self, k) return k end})\n\
          s = s + (o + 1) + o(2)\n\
          for j = 10, 1, -2 do s = s + j end\n\
          if s > 3 then goto skip end\n\
          s = s * 2\n\
          ::skip::\n\
          local g = function() return s, t end\n\
          return g(), select('#', ...)\n\
        end\n\
        BASE = string.dump(f)";

    fn base_chunk() -> Vec<u8> {
        let mut vm = Vm::new(Options::default());
        vm.run_source(b"=sample", SAMPLE).expect("the sample compiles and dumps");
        let base = vm.global("BASE");
        vm.string_bytes(base).expect("the dump is a string").to_vec()
    }

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, bound: usize) -> usize {
            (self.next() % bound as u64) as usize
        }
    }

    fn mutant(base: &[u8], seed: u64) -> Vec<u8> {
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let mut bytes = base.to_vec();
        for _ in 0..1 + rng.below(4) {
            let position = 12 + rng.below(bytes.len() - 12);
            bytes[position] = rng.next() as u8;
        }
        bytes
    }

    fn run_to_outcome(vm: &mut Vm, function: crate::Value) {
        let pause = vm.pause_handle();
        pause.request();
        let mut outcome = vm.start(function, &[]);
        for _ in 0..PAUSES_BEFORE_INTERRUPT {
            match outcome {
                Ok(Outcome::Paused) => {
                    pause.request();
                    outcome = vm.resume();
                }
                _ => return,
            }
        }
        vm.request_interrupt();
        let _ = vm.resume();
    }

    fn run_case(base: &[u8], seed: u64) {
        let bytes = mutant(base, seed);
        let mut vm = Vm::new(Options {
            memory_limit: Some(1 << 26),
            ..Options::default()
        });
        if let Ok(function) = vm.load_binary(b"=fuzz", &bytes, None) {
            run_to_outcome(&mut vm, function);
        }
    }

    #[test]
    fn mutated_binary_chunks_never_panic_or_abort() {
        let base = base_chunk();
        let failures: Vec<u64> = (0..SEEDS)
            .filter(|&seed| catch_unwind(AssertUnwindSafe(|| run_case(&base, seed))).is_err())
            .collect();
        assert!(failures.is_empty(), "mutated chunks panicked for seeds {failures:?}");
    }
}
