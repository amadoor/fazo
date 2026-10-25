use crate::{
    Vm,
    arena::Ref,
    heap::Heap,
    instruction::Instruction,
    number,
    proto::Proto,
    value::Value,
};

impl Vm {
    /// Returns `None` when `function` is not a Lua function; `full` adds constants, locals and upvalues, as `luac -l -l` does.
    pub fn listing(&self, function: Value, full: bool) -> Option<Vec<u8>> {
        let Value::Closure(closure) = function else {
            return None;
        };
        let proto = self.heap.closures.live(closure).proto;
        Some(render(&self.heap, proto, full))
    }
}

pub(crate) fn render(heap: &Heap, proto: Ref<Proto>, full: bool) -> Vec<u8> {
    let mut out = Vec::new();
    print_function(heap, proto, full, &mut out);
    out
}

fn print_function(heap: &Heap, reference: Ref<Proto>, full: bool, out: &mut Vec<u8>) {
    let proto = heap.protos.live(reference);
    print_header(proto, out);
    print_code(heap, proto, out);
    if full {
        print_debug(heap, proto, out);
    }
    for child in &proto.protos {
        print_function(heap, *child, full, out);
    }
}

fn print_header(proto: &Proto, out: &mut Vec<u8>) {
    let kind = if proto.line_defined == 0 { "main" } else { "function" };
    let instructions = count_of(proto.code.len(), "instruction");
    out.extend_from_slice(format!("\n{kind} <").as_bytes());
    out.extend_from_slice(&source_name(&proto.chunkname));
    let range = format!(
        ":{},{}> ({instructions})\n",
        proto.line_defined, proto.last_line_defined
    );
    out.extend_from_slice(range.as_bytes());
    let vararg = if proto.is_vararg { "+" } else { "" };
    let params = format!("{}{vararg} {}", proto.num_params, plural("param", usize::from(proto.num_params)));
    let slots = count_of(usize::from(proto.max_stack), "slot");
    let upvalues = count_of(proto.upvalues.len(), "upvalue");
    let locals = count_of(proto.local_vars.len(), "local");
    let constants = count_of(proto.constants.len(), "constant");
    let functions = count_of(proto.protos.len(), "function");
    let line = format!("{params}, {slots}, {upvalues}, {locals}, {constants}, {functions}\n");
    out.extend_from_slice(line.as_bytes());
}

fn source_name(chunkname: &[u8]) -> Vec<u8> {
    match chunkname.first() {
        Some(b'@' | b'=') => chunkname[1..].to_vec(),
        Some(0x1b) => b"(bstring)".to_vec(),
        _ => b"(string)".to_vec(),
    }
}

fn count_of(count: usize, word: &str) -> String {
    format!("{count} {}", plural(word, count))
}

fn plural(word: &str, count: usize) -> String {
    if count == 1 {
        word.to_owned()
    } else {
        format!("{word}s")
    }
}

fn print_code(heap: &Heap, proto: &Proto, out: &mut Vec<u8>) {
    for (pc, &instruction) in proto.code.iter().enumerate() {
        let line = proto.lines.get(pc).copied().unwrap_or(0);
        let opcode = instruction.opcode();
        let line_text = if line > 0 { format!("[{line}]") } else { "[-]".to_owned() };
        let text = format!(
            "\t{}\t{line_text}\t{:<9}\t{}\n",
            pc + 1,
            opcode.name(),
            describe(heap, proto, pc, instruction)
        );
        out.extend_from_slice(text.as_bytes());
    }
}

fn describe(heap: &Heap, proto: &Proto, pc: usize, instruction: Instruction) -> String {
    use Instruction as I;
    match instruction {
        I::Move { a, b } => format!("{a} {b}"),
        I::LoadI { a, sbx } | I::LoadF { a, sbx } => format!("{a} {sbx}"),
        I::LoadK { a, bx } => with_comment(format!("{a} {bx}"), constant(heap, proto, bx)),
        I::LoadKx { a } => with_comment(format!("{a}"), constant(heap, proto, extra_argument(proto, pc))),
        I::LoadFalse { a } | I::LFalseSkip { a } | I::LoadTrue { a } => format!("{a}"),
        I::LoadNil { a, b } => with_comment(format!("{a} {b}"), format!("{} out", u32::from(b) + 1)),
        I::GetUpval { a, b } | I::SetUpval { a, b } => with_comment(format!("{a} {b}"), upvalue_name(heap, proto, b)),
        I::GetTabUp { a, b, c } => with_comment(
            format!("{a} {b} {c}"),
            format!("{} {}", upvalue_name(heap, proto, b), constant(heap, proto, u32::from(c))),
        ),
        I::GetTable { a, b, c } | I::GetI { a, b, c } | I::GetVarg { a, b, c } => format!("{a} {b} {c}"),
        I::GetField { a, b, c } => with_comment(format!("{a} {b} {c}"), constant(heap, proto, u32::from(c))),
        I::SetTabUp { a, b, c, k } => {
            let mut comment = format!("{} {}", upvalue_name(heap, proto, a), constant(heap, proto, u32::from(b)));
            if k {
                comment.push_str(&format!(" {}", constant(heap, proto, u32::from(c))));
            }
            with_comment(format!("{a} {b} {c}{}", flag(k)), comment)
        }
        I::SetTable { a, b, c, k } | I::SetI { a, b, c, k } => {
            let operands = format!("{a} {b} {c}{}", flag(k));
            if k {
                return with_comment(operands, constant(heap, proto, u32::from(c)));
            }
            operands
        }
        I::SetField { a, b, c, k } => {
            let mut comment = constant(heap, proto, u32::from(b));
            if k {
                comment.push_str(&format!(" {}", constant(heap, proto, u32::from(c))));
            }
            with_comment(format!("{a} {b} {c}{}", flag(k)), comment)
        }
        I::NewTable { a, b, c, k } => with_comment(
            format!("{a} {b} {c}{}", flag(k)),
            format!("{}", c + extra_argument(proto, pc) * LUAC_EXTRA_ARG_STEP),
        ),
        I::SelfOp { a, b, c } => format!("{a} {b} {c}"),
        I::AddI { a, b, sc } | I::ShrI { a, b, sc } | I::ShlI { a, b, sc } => format!("{a} {b} {sc}"),
        I::AddK { a, b, c }
        | I::SubK { a, b, c }
        | I::MulK { a, b, c }
        | I::ModK { a, b, c }
        | I::PowK { a, b, c }
        | I::DivK { a, b, c }
        | I::IDivK { a, b, c }
        | I::BAndK { a, b, c }
        | I::BOrK { a, b, c }
        | I::BXorK { a, b, c } => with_comment(format!("{a} {b} {c}"), constant(heap, proto, u32::from(c))),
        I::Add { a, b, c }
        | I::Sub { a, b, c }
        | I::Mul { a, b, c }
        | I::Mod { a, b, c }
        | I::Pow { a, b, c }
        | I::Div { a, b, c }
        | I::IDiv { a, b, c }
        | I::BAnd { a, b, c }
        | I::BOr { a, b, c }
        | I::BXor { a, b, c }
        | I::Shl { a, b, c }
        | I::Shr { a, b, c } => format!("{a} {b} {c}"),
        I::MmBin { a, b, c } => with_comment(format!("{a} {b} {c}"), event_name(c).to_owned()),
        I::MmBinI { a, sb, c, k } => {
            let mut comment = event_name(c).to_owned();
            if k {
                comment.push_str(" flip");
            }
            with_comment(format!("{a} {sb} {c} {}", u8::from(k)), comment)
        }
        I::MmBinK { a, b, c, k } => {
            let mut comment = format!("{} {}", event_name(c), constant(heap, proto, u32::from(b)));
            if k {
                comment.push_str(" flip");
            }
            with_comment(format!("{a} {b} {c} {}", u8::from(k)), comment)
        }
        I::Unm { a, b } | I::BNot { a, b } | I::Not { a, b } | I::Len { a, b } | I::Concat { a, b } => format!("{a} {b}"),
        I::Close { a } | I::Tbc { a } => format!("{a}"),
        I::Jmp { sj } => with_comment(format!("{sj}"), format!("to {}", i64::from(sj) + pc as i64 + 2)),
        I::Eq { a, b, k } | I::Lt { a, b, k } | I::Le { a, b, k } => format!("{a} {b} {}", u8::from(k)),
        I::EqK { a, b, k } => with_comment(format!("{a} {b} {}", u8::from(k)), constant(heap, proto, u32::from(b))),
        I::EqI { a, sb, k, .. }
        | I::LtI { a, sb, k, .. }
        | I::LeI { a, sb, k, .. }
        | I::GtI { a, sb, k, .. }
        | I::GeI { a, sb, k, .. } => format!("{a} {sb} {}", u8::from(k)),
        I::Test { a, k } => format!("{a} {}", u8::from(k)),
        I::TestSet { a, b, k } => format!("{a} {b} {}", u8::from(k)),
        I::Call { a, b, c } => with_comment(format!("{a} {b} {c}"), call_comment(b, c)),
        I::TailCall { a, b, c, k } => with_comment(format!("{a} {b} {c}{}", flag(k)), format!("{} in", i32::from(b) - 1)),
        I::Return { a, b, c, k } => with_comment(format!("{a} {b} {c}{}", flag(k)), call_out(b)),
        I::Return0 { a } | I::Return1 { a } => format!("{a}"),
        I::ForLoop { a, bx } => with_comment(format!("{a} {bx}"), format!("to {}", pc as i64 - i64::from(bx) + 2)),
        I::ForPrep { a, bx } => with_comment(format!("{a} {bx}"), format!("exit to {}", pc as i64 + i64::from(bx) + 3)),
        I::TForPrep { a, bx } => with_comment(format!("{a} {bx}"), format!("to {}", pc as i64 + i64::from(bx) + 2)),
        I::TForCall { a, c } => format!("{a} {c}"),
        I::TForLoop { a, bx } => with_comment(format!("{a} {bx}"), format!("to {}", pc as i64 - i64::from(bx) + 2)),
        I::SetList { a, b, c, k } => {
            let operands = format!("{a} {b} {c}{}", flag(k));
            if k {
                return with_comment(operands, format!("{}", c + extra_argument(proto, pc) * LUAC_EXTRA_ARG_STEP));
            }
            operands
        }
        I::Closure { a, bx } => format!("{a} {bx}"),
        I::Vararg { a, b, c, k } => with_comment(format!("{a} {b} {c}{}", flag(k)), call_out(c)),
        I::ErrNil { a, bx } => with_comment(
            format!("{a} {bx}"),
            match bx.checked_sub(1) {
                Some(index) => constant(heap, proto, index),
                None => "?".to_owned(),
            },
        ),
        I::VarargPrep { a } => format!("{a}"),
        I::ExtraArg { ax } => format!("{ax}"),
    }
}

/// The step `luac -l` uses for `EXTRAARG` comments.
const LUAC_EXTRA_ARG_STEP: u32 = 256;

fn extra_argument(proto: &Proto, pc: usize) -> u32 {
    match proto.code.get(pc + 1) {
        Some(Instruction::ExtraArg { ax }) => *ax,
        _ => 0,
    }
}

/// Drops the underscores of the metamethod name, as `luac -l` prints it.
fn event_name(event: u8) -> &'static str {
    const NAMES: [&str; 24] = [
        "index", "newindex", "gc", "mode", "len", "eq", "add", "sub", "mul", "mod", "pow", "div", "idiv", "band", "bor",
        "bxor", "shl", "shr", "unm", "bnot", "lt", "le", "concat", "call",
    ];
    NAMES.get(usize::from(event)).copied().unwrap_or("close")
}

fn with_comment(operands: String, comment: String) -> String {
    if comment.is_empty() {
        return operands;
    }
    format!("{operands}\t; {comment}")
}

fn flag(set: bool) -> &'static str {
    if set { "k" } else { "" }
}

fn call_comment(b: u8, c: u8) -> String {
    let inputs = if b == 0 { "all in ".to_owned() } else { format!("{} in ", b - 1) };
    format!("{inputs}{}", call_out(c))
}

fn call_out(c: u8) -> String {
    if c == 0 { "all out".to_owned() } else { format!("{} out", c - 1) }
}

fn upvalue_name(heap: &Heap, proto: &Proto, index: u8) -> String {
    let name = proto.upvalues.get(usize::from(index)).and_then(|upvalue| upvalue.name).map(|name| heap.bytes(name));
    match name {
        Some(name) if !name.is_empty() => String::from_utf8_lossy(name).into_owned(),
        _ => "-".to_owned(),
    }
}

fn constant(heap: &Heap, proto: &Proto, k: u32) -> String {
    match proto.constants.get(k as usize) {
        Some(value) => constant_text(heap, *value),
        None => "?".to_owned(),
    }
}

fn constant_text(heap: &Heap, value: Value) -> String {
    match value {
        Value::Nil => "nil".to_owned(),
        Value::Boolean(value) => value.to_string(),
        Value::Integer(value) => value.to_string(),
        Value::Float(value) => number::format_float(value),
        Value::String(string) => quoted(heap.bytes(string)),
        _ => "?".to_owned(),
    }
}

fn constant_type(value: Value) -> &'static str {
    match value {
        Value::Nil => "N",
        Value::Boolean(_) => "B",
        Value::Integer(_) => "I",
        Value::Float(_) => "F",
        Value::String(_) => "S",
        _ => "?",
    }
}

fn quoted(bytes: &[u8]) -> String {
    let mut text = String::from("\"");
    for &byte in bytes {
        match byte {
            b'"' => text.push_str("\\\""),
            b'\\' => text.push_str("\\\\"),
            0x07 => text.push_str("\\a"),
            0x08 => text.push_str("\\b"),
            0x0c => text.push_str("\\f"),
            b'\n' => text.push_str("\\n"),
            b'\r' => text.push_str("\\r"),
            b'\t' => text.push_str("\\t"),
            0x0b => text.push_str("\\v"),
            0x20..=0x7e => text.push(char::from(byte)),
            _ => text.push_str(&format!("\\{byte:03}")),
        }
    }
    text.push('"');
    text
}

fn print_debug(heap: &Heap, proto: &Proto, out: &mut Vec<u8>) {
    let mut text = format!("constants ({}):\n", proto.constants.len());
    for (index, value) in proto.constants.iter().enumerate() {
        text.push_str(&format!("\t{index}\t{}\t{}\n", constant_type(*value), constant_text(heap, *value)));
    }
    text.push_str(&format!("locals ({}):\n", proto.local_vars.len()));
    for (index, local) in proto.local_vars.iter().enumerate() {
        let name = String::from_utf8_lossy(local.name.map_or(&[][..], |name| heap.bytes(name)));
        text.push_str(&format!("\t{index}\t{name}\t{}\t{}\n", local.start_pc + 1, local.end_pc + 1));
    }
    text.push_str(&format!("upvalues ({}):\n", proto.upvalues.len()));
    for (index, upvalue) in proto.upvalues.iter().enumerate() {
        let name = upvalue_name(heap, proto, index as u8);
        text.push_str(&format!(
            "\t{index}\t{name}\t{}\t{}\n",
            u8::from(upvalue.in_stack),
            upvalue.index
        ));
    }
    out.extend_from_slice(text.as_bytes());
}

#[cfg(test)]
mod tests {
    use crate::{Options, Vm};

    fn listing_of(source: &str, full: bool) -> String {
        let mut vm = Vm::new(Options::default());
        let function = vm.load(b"@t.lua", source.as_bytes()).expect("the chunk compiles");
        let text = vm.listing(function, full).expect("a Lua function lists");
        String::from_utf8(text).expect("the listing is ASCII for this chunk")
    }

    const CODE: &str = concat!(
        "\nmain <t.lua:0,0> (5 instructions)\n",
        "0+ params, 2 slots, 1 upvalue, 0 locals, 1 constant, 0 functions\n",
        "\t1\t[1]\tVARARGPREP\t0\n",
        "\t2\t[1]\tGETTABUP \t0 0 0\t; _ENV \"print\"\n",
        "\t3\t[1]\tLOADI    \t1 2\n",
        "\t4\t[1]\tCALL     \t0 2 1\t; 1 in 0 out\n",
        "\t5\t[1]\tRETURN   \t0 1 1\t; 0 out\n",
    );

    #[test]
    fn a_small_chunk_lists_each_instruction_with_its_line_and_operands() {
        assert_eq!(listing_of("print(1 + 1)", false), CODE);
    }

    #[test]
    fn the_full_listing_adds_constants_locals_and_upvalues() {
        let expected = format!(
            "{CODE}constants (1):\n\t0\tS\t\"print\"\nlocals (0):\nupvalues (1):\n\t0\t_ENV\t1\t0\n"
        );
        assert_eq!(listing_of("print(1 + 1)", true), expected);
    }
}
