//! Port of the symbolic execution in `ldebug.c` of PUC Lua 5.5.1 (MIT, Copyright (C) 1994-2025 Lua.org, PUC-Rio).

use crate::{Vm, arena::Ref, heap::Heap, instruction::Instruction, object::{Frame, LuaString}, proto::Proto, value::Value};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Operand {
    Register(u8),
    Upvalue(u8),
    Constant,
    Unknown,
}

/// Event names of `ltm.h`, without their underscores.
const EVENT_NAMES: [&str; 24] = [
    "index", "newindex", "gc", "mode", "len", "eq", "add", "sub", "mul", "mod", "pow", "div", "idiv", "band", "bor",
    "bxor", "shl", "shr", "unm", "bnot", "lt", "le", "concat", "call",
];

pub(super) fn event_text(event: u8) -> &'static str {
    EVENT_NAMES.get(usize::from(event)).copied().unwrap_or("close")
}

/// Follows `testMMMode` in ldebug.c.
fn is_mm_instruction(instruction: Instruction) -> bool {
    matches!(
        instruction,
        Instruction::MmBin { .. } | Instruction::MmBinI { .. } | Instruction::MmBinK { .. }
    )
}

/// Follows `testAMode` in ldebug.c, with the instructions `findsetreg` treats specially.
fn sets_register(instruction: Instruction, reg: u8) -> bool {
    use Instruction::*;
    match instruction {
        LoadNil { a, b } => a <= reg && reg <= a + b,
        TForCall { a, .. } => reg >= a + 2,
        Call { a, .. } | TailCall { a, .. } => reg >= a,
        Move { a, .. }
        | LoadI { a, .. }
        | LoadF { a, .. }
        | LoadK { a, .. }
        | LoadKx { a }
        | LoadFalse { a }
        | LFalseSkip { a }
        | LoadTrue { a }
        | GetUpval { a, .. }
        | GetTabUp { a, .. }
        | GetTable { a, .. }
        | GetI { a, .. }
        | GetField { a, .. }
        | NewTable { a, .. }
        | SelfOp { a, .. }
        | AddI { a, .. }
        | AddK { a, .. }
        | SubK { a, .. }
        | MulK { a, .. }
        | ModK { a, .. }
        | PowK { a, .. }
        | DivK { a, .. }
        | IDivK { a, .. }
        | BAndK { a, .. }
        | BOrK { a, .. }
        | BXorK { a, .. }
        | ShrI { a, .. }
        | ShlI { a, .. }
        | Add { a, .. }
        | Sub { a, .. }
        | Mul { a, .. }
        | Mod { a, .. }
        | Pow { a, .. }
        | Div { a, .. }
        | IDiv { a, .. }
        | BAnd { a, .. }
        | BOr { a, .. }
        | BXor { a, .. }
        | Shl { a, .. }
        | Shr { a, .. }
        | Unm { a, .. }
        | BNot { a, .. }
        | Not { a, .. }
        | Len { a, .. }
        | Concat { a, .. }
        | TestSet { a, .. }
        | ForLoop { a, .. }
        | ForPrep { a, .. }
        | TForLoop { a, .. }
        | Closure { a, .. }
        | Vararg { a, .. }
        | GetVarg { a, .. } => a == reg,
        _ => false,
    }
}

/// Follows `findsetreg` in ldebug.c.
fn find_set_register(proto: &Proto, last_pc: usize, reg: u8) -> Option<usize> {
    let last_pc = if last_pc < proto.code.len() && is_mm_instruction(proto.code[last_pc]) {
        last_pc.checked_sub(1)?
    } else {
        last_pc
    };
    let mut set = None;
    let mut jump_target = 0;
    for (pc, instruction) in proto.code.iter().enumerate().take(last_pc) {
        if let Instruction::Jmp { sj } = *instruction {
            let destination = (pc as isize + 1 + sj as isize) as usize;
            if destination <= last_pc && destination > jump_target {
                jump_target = destination;
            }
            continue;
        }
        if sets_register(*instruction, reg) {
            set = (pc >= jump_target).then_some(pc);
        }
    }
    set
}

pub(super) fn local_name<'a>(heap: &'a Heap, proto: &Proto, mut number: usize, pc: usize) -> Option<&'a [u8]> {
    for var in &proto.local_vars {
        if var.start_pc > pc {
            break;
        }
        if pc < var.end_pc {
            number -= 1;
            if number == 0 {
                return Some(name_bytes(heap, var.name));
            }
        }
    }
    None
}

fn upvalue_name<'a>(heap: &'a Heap, proto: &Proto, index: u8) -> &'a [u8] {
    name_bytes(heap, proto.upvalues[usize::from(index)].name)
}

/// A name the chunk did not carry reads as empty.
pub(super) fn name_bytes(heap: &Heap, name: Option<Ref<LuaString>>) -> &[u8] {
    name.map_or(&[], |name| heap.bytes(name))
}

pub(super) fn constant_text(heap: &Heap, proto: &Proto, k: u32) -> Option<Vec<u8>> {
    match proto.constants.get(k as usize) {
        Some(Value::String(string)) => Some(heap.bytes(*string).to_vec()),
        _ => None,
    }
}

/// Follows `basicgetobjname` in ldebug.c.
fn basic_object_name(heap: &Heap, proto: &Proto, pc: &mut Option<usize>, reg: u8) -> Option<(&'static str, Vec<u8>)> {
    let current = (*pc)?;
    if let Some(name) = local_name(heap, proto, usize::from(reg) + 1, current) {
        return Some(("local", name.to_vec()));
    }
    *pc = find_set_register(proto, current, reg);
    let set_pc = (*pc)?;
    match proto.code[set_pc] {
        Instruction::Move { a, b } if b < a => basic_object_name(heap, proto, pc, b),
        Instruction::GetUpval { b, .. } => Some(("upvalue", upvalue_name(heap, proto, b).to_vec())),
        Instruction::LoadK { bx, .. } => constant_text(heap, proto, bx).map(|name| ("constant", name)),
        Instruction::LoadKx { .. } => match proto.code.get(set_pc + 1) {
            Some(Instruction::ExtraArg { ax }) => constant_text(heap, proto, *ax).map(|name| ("constant", name)),
            _ => None,
        },
        _ => None,
    }
}

/// Follows `isEnv` in ldebug.c.
fn is_environment(heap: &Heap, proto: &Proto, pc: usize, table: u8, upvalue: bool) -> &'static str {
    let name = if upvalue {
        Some(("upvalue", upvalue_name(heap, proto, table).to_vec()))
    } else {
        let mut position = Some(pc);
        basic_object_name(heap, proto, &mut position, table)
    };
    match name {
        Some((kind, name)) if matches!(kind, "local" | "upvalue") && name == b"_ENV" => "global",
        _ => "field",
    }
}

fn register_key_name(heap: &Heap, proto: &Proto, pc: usize, key: u8) -> Vec<u8> {
    let mut position = Some(pc);
    match basic_object_name(heap, proto, &mut position, key) {
        Some(("constant", name)) => name,
        _ => b"?".to_vec(),
    }
}

pub(super) fn calling_name(heap: &Heap, proto: &Proto, last_pc: usize) -> Option<(&'static str, Vec<u8>)> {
    match *proto.code.get(last_pc)? {
        Instruction::Call { a, .. } | Instruction::TailCall { a, .. } => object_name(heap, proto, last_pc, a),
        instruction => event_name(instruction).map(|(kind, name)| (kind, name.as_bytes().to_vec())),
    }
}

/// Follows `funcnamefromcode` in ldebug.c.
pub(super) fn event_name(instruction: Instruction) -> Option<(&'static str, &'static str)> {
    let name = match instruction {
        Instruction::GetTabUp { .. }
        | Instruction::GetTable { .. }
        | Instruction::GetI { .. }
        | Instruction::GetField { .. }
        | Instruction::SelfOp { .. } => "index",
        Instruction::SetTabUp { .. }
        | Instruction::SetTable { .. }
        | Instruction::SetI { .. }
        | Instruction::SetField { .. } => "newindex",
        Instruction::MmBin { c, .. } | Instruction::MmBinI { c, .. } | Instruction::MmBinK { c, .. } => event_text(c),
        Instruction::Unm { .. } => "unm",
        Instruction::BNot { .. } => "bnot",
        Instruction::Len { .. } => "len",
        Instruction::Concat { .. } => "concat",
        Instruction::Eq { .. } => "eq",
        Instruction::Lt { .. } | Instruction::LtI { .. } | Instruction::GtI { .. } => "lt",
        Instruction::Le { .. } | Instruction::LeI { .. } | Instruction::GeI { .. } => "le",
        Instruction::Close { .. } | Instruction::Return { .. } => "close",
        Instruction::TForCall { .. } => return Some(("for iterator", "for iterator")),
        _ => return None,
    };
    Some(("metamethod", name))
}

/// Follows `getobjname` in ldebug.c.
pub(super) fn object_name(heap: &Heap, proto: &Proto, last_pc: usize, reg: u8) -> Option<(&'static str, Vec<u8>)> {
    let mut position = Some(last_pc);
    if let Some(named) = basic_object_name(heap, proto, &mut position, reg) {
        return Some(named);
    }
    let set_pc = position?;
    let unknown = || b"?".to_vec();
    match proto.code[set_pc] {
        Instruction::GetTabUp { b, c, .. } => Some((
            is_environment(heap, proto, set_pc, b, true),
            constant_text(heap, proto, u32::from(c)).unwrap_or_else(unknown),
        )),
        Instruction::GetTable { b, c, .. } | Instruction::GetVarg { b, c, .. } => Some((
            is_environment(heap, proto, set_pc, b, false),
            register_key_name(heap, proto, set_pc, c),
        )),
        Instruction::GetI { .. } => Some(("field", b"integer index".to_vec())),
        Instruction::GetField { b, c, .. } => Some((
            is_environment(heap, proto, set_pc, b, false),
            constant_text(heap, proto, u32::from(c)).unwrap_or_else(unknown),
        )),
        Instruction::SelfOp { c, .. } => Some((
            "method",
            constant_text(heap, proto, u32::from(c)).unwrap_or_else(unknown),
        )),
        _ => None,
    }
}

impl Vm {
    pub(crate) fn event_info(&self) -> Vec<u8> {
        let Some(frame) = self.top_frame() else {
            return Vec::new();
        };
        let proto = self.proto_of(frame.closure);
        let instruction = frame.pc().checked_sub(1).and_then(|pc| proto.code.get(pc));
        match instruction.and_then(|instruction| event_name(*instruction)) {
            Some((kind, name)) => format!(" ({kind} '{name}')").into_bytes(),
            None => Vec::new(),
        }
    }

    pub(crate) fn position_prefix(&self) -> Option<Vec<u8>> {
        let Some(Frame::Lua(frame)) = self.thread().frames.iter().rev().find(|frame| !matches!(frame, Frame::Yielded(_) | Frame::Coroutine(_)))
        else {
            return None;
        };
        let frame = *frame;
        let proto = self.proto_of(frame.closure);
        if proto.lines.is_empty() {
            return Some(b"?:?: ".to_vec());
        }
        let pc = frame.pc().saturating_sub(1);
        let line = proto.lines.get(pc).map_or(-1, |line| i64::from(*line));
        let mut prefix = proto.source.to_vec();
        prefix.extend_from_slice(format!(":{line}: ").as_bytes());
        Some(prefix)
    }

    pub(crate) fn variable_info(&self, operand: Operand) -> Vec<u8> {
        let Some(frame) = self.nearest_lua_frame() else {
            return Vec::new();
        };
        let proto = self.proto_of(frame.closure);
        let described = match operand {
            Operand::Upvalue(index) => Some(("upvalue", upvalue_name(&self.heap, proto, index).to_vec())),
            Operand::Register(reg) => object_name(&self.heap, proto, frame.pc().saturating_sub(1), reg),
            Operand::Constant | Operand::Unknown => None,
        };
        match described {
            Some((kind, name)) => {
                let mut info = format!(" ({kind} '").into_bytes();
                info.extend_from_slice(&name);
                info.extend_from_slice(b"')");
                info
            }
            None => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{Options, Vm};

    fn error_text(source: &str) -> String {
        let mut vm = Vm::new(Options::default());
        let error = vm
            .run_source(b"=t", source.as_bytes())
            .expect_err("the chunk fails at runtime");
        String::from_utf8_lossy(&vm.error_message(&error)).into_owned()
    }

    #[test]
    fn a_local_operand_is_named_by_its_variable() {
        assert_eq!(
            error_text("local a\nprint(a + 1)"),
            "t:2: attempt to perform arithmetic on a nil value (local 'a')"
        );
    }

    #[test]
    fn a_global_table_is_named_as_a_global_and_a_field_as_a_field() {
        assert_eq!(error_text("print(g.x)"), "t:1: attempt to index a nil value (global 'g')");
        assert_eq!(
            error_text("local t = {}\nprint(t.a.b)"),
            "t:2: attempt to index a nil value (field 'a')"
        );
    }

    #[test]
    fn a_method_call_on_a_missing_method_is_named_as_a_method() {
        assert_eq!(
            error_text("local o = {}\no:run()"),
            "t:2: attempt to call a nil value (method 'run')"
        );
    }

    #[test]
    fn an_upvalue_operand_is_named_as_an_upvalue() {
        assert_eq!(
            error_text("local c\nlocal function f() return c + 1 end\nf()"),
            "t:2: attempt to perform arithmetic on a nil value (upvalue 'c')"
        );
    }

    #[test]
    fn a_constant_string_in_a_bitwise_operation_is_named_as_a_constant() {
        assert_eq!(
            error_text("print('3' & 1)"),
            "t:1: attempt to perform bitwise operation on a string value (constant '3')"
        );
    }

    #[test]
    fn a_comparison_names_both_types_and_never_a_variable() {
        assert_eq!(error_text("print(1 < nil)"), "t:1: attempt to compare number with nil");
        assert_eq!(error_text("print({} < {})"), "t:1: attempt to compare two table values");
    }
}
