//! One pass over a prototype's bytecode. Each instruction's templates are copied in order, value holes are written at once,
//! and branches once every block's position is known.

use super::{
    Library,
    masm::{Field, Hole, SplitStencil, ext, slot},
    templates::{ArithKind, CmpKind, HelperId, Key, Rhs, TagConstant},
};
use crate::{
    instruction::{ArithOp, Instruction},
    jit::{
        sites::{
            FAST_ARITH, FAST_CALL, FAST_FOR_PREP, FAST_GET_FIELD, FAST_GET_INT, FAST_GET_SELF, FAST_GET_TABLE, FAST_GET_TABUP,
            FAST_CLOSE, FAST_CLOSURE, FAST_GET_UPVAL, FAST_SET_LIST, FAST_NEW_TABLE, FAST_RETURN, FAST_SET_FIELD, FAST_SET_INT, FAST_SET_TABLE, FAST_SET_UPVAL,
            CallTarget, FAST_TABLE_BARRIER, FAST_TAIL_CALL, add_entries, all_method_entries, field_entries,
        },
        helpers::CONSTANT_OPERAND,
        jit_operand::IMMEDIATE,
        unwind::{TRAILER_ALIGNMENT, Trailer},
    },
    proto::Proto,
    value::Value,
    vm::ic::{ABSENT, InlineCache},
};

pub(crate) struct BaselineFacts<'a> {
    pub(crate) interned: &'a [bool],
    pub(crate) tier_budget: Option<*const std::cell::Cell<i64>>,
    pub(crate) inline_caches: bool,
    pub(crate) calls: &'a [Vec<CallTarget>],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Label {
    Block(usize),
    Mark(usize),
    Dispatch,
    Negate,
    Exit,
    Table,
    Trap,
    SharedStep,
}

struct Fixup {
    cold: bool,
    at: u32,
    field: Field,
    label: Label,
}

type Values = [i64; slot::COUNT];
type Labels = [Label; ext::COUNT];

/// Largest register index whose byte offset fits every template's displacement on both architectures.
const MAX_REGISTER: u8 = 250;
/// Largest constant index whose byte offset fits a scaled 12-bit displacement.
const MAX_CONSTANT: usize = 2046;
const MAX_SLOT: u32 = 2046;
const VALUE: i64 = 16;

struct Emitter<'a> {
    library: &'a mut Library,
    proto: &'a Proto,
    facts: &'a BaselineFacts<'a>,
    code: Vec<u8>,
    cold: Vec<u8>,
    fixups: Vec<Fixup>,
    blocks: Vec<u32>,
    marks: Vec<(bool, u32)>,
    dispatch: u32,
    negate: u32,
    exit: u32,
    trap: u32,
    shared_step: u32,
}

fn values(pairs: &[(u8, i64)]) -> Values {
    let mut values = [0; slot::COUNT];
    for &(slot, value) in pairs {
        values[usize::from(slot)] = value;
    }
    values
}

fn offset(register: u8) -> i64 {
    i64::from(register) * VALUE
}

fn offset_pc(from: usize, offset: i32) -> usize {
    (from as isize + offset as isize) as usize
}

pub(super) fn emit(library: &mut Library, proto: &Proto, facts: &BaselineFacts<'_>) -> Option<(Vec<u8>, Option<Trailer>)> {
    let mut emitter = Emitter {
        library,
        proto,
        facts,
        code: Vec::with_capacity(proto.code.len() * 128 + 1024),
        cold: Vec::with_capacity(proto.code.len() * 64),
        fixups: Vec::with_capacity(proto.code.len() * 8),
        marks: Vec::with_capacity(proto.code.len()),
        blocks: Vec::with_capacity(proto.code.len()),
        dispatch: 0,
        negate: 0,
        exit: 0,
        trap: 0,
        shared_step: 0,
    };

    emitter.frame();
    for pc in 0..proto.code.len() {
        emitter.blocks.push(emitter.code.len() as u32);
        emitter.lower(pc, proto.code[pc]);
        emitter.elide_jump_to(pc + 1);
    }
    emitter.finish()
}

impl<'a> Emitter<'a> {
    fn frame(&mut self) {
        let frame = &self.library.frame;
        self.code.extend_from_slice(&frame.prologue);
        self.dispatch = self.code.len() as u32;
        let mut labels = [Label::Trap; ext::COUNT];
        labels[usize::from(ext::EXIT)] = Label::Negate;
        labels[usize::from(ext::TARGET)] = Label::Table;
        let count = values(&[(slot::I, self.proto.code.len() as i64)]);
        place(&mut self.code, &mut self.fixups, false, &frame.dispatch, &count, &labels);
        self.trap = self.code.len() as u32 - trap_length();
        self.negate = self.code.len() as u32;
        self.code.extend_from_slice(&frame.negate);
        self.exit = self.code.len() as u32;
        self.code.extend_from_slice(&frame.epilogue);
        self.shared_step = self.code.len() as u32;
        let labels = self.labels(0);
        self.put(false, Key::SharedStep, &values(&[]), &labels);
    }

    fn finish(mut self) -> Option<(Vec<u8>, Option<Trailer>)> {
        let cold_start = self.code.len() as u32;
        let cold = std::mem::take(&mut self.cold);
        self.code.extend_from_slice(&cold);
        while !self.code.len().is_multiple_of(4) {
            self.code.push(0);
        }
        let table = self.code.len() as u32;
        for index in 0..self.blocks.len() {
            let offset = self.blocks[index] as i64 - i64::from(table);
            self.code.extend_from_slice(&(offset as i32).to_le_bytes());
        }
        let fixups = std::mem::take(&mut self.fixups);
        for fixup in fixups {
            let at = if fixup.cold { cold_start + fixup.at } else { fixup.at };
            let target = self.resolve(fixup.label, cold_start, table)?;
            patch_branch(&mut self.code, at as usize, fixup.field, i64::from(target))?;
        }
        let code_len = self.code.len();
        let trailer_offset = code_len.next_multiple_of(TRAILER_ALIGNMENT);
        let trailer = super::unwind::trailer(&self.library.frame.unwind, code_len, trailer_offset);
        if let Some(trailer) = &trailer {
            self.code.resize(trailer_offset, 0);
            self.code.extend_from_slice(&trailer.bytes);
        }
        Some((self.code, trailer))
    }

    fn resolve(&self, label: Label, cold_start: u32, table: u32) -> Option<u32> {
        Some(match label {
            Label::Block(pc) => *self.blocks.get(pc)?,
            Label::Mark(mark) => {
                let (cold, offset) = self.marks[mark];
                if offset == u32::MAX {
                    return None;
                }
                if cold { cold_start + offset } else { offset }
            }
            Label::Dispatch => self.dispatch,
            Label::Negate => self.negate,
            Label::Exit => self.exit,
            Label::Table => table,
            Label::Trap => self.trap,
            Label::SharedStep => self.shared_step,
        })
    }

    fn put(&mut self, cold: bool, key: Key, values: &Values, labels: &Labels) {
        let stencil = self.library.stencil(key);
        let buffer = if cold { &mut self.cold } else { &mut self.code };
        place(buffer, &mut self.fixups, cold, stencil, values, labels);
    }

    fn labels(&self, pc: usize) -> Labels {
        let block = |target: usize| if target < self.proto.code.len() { Label::Block(target) } else { Label::Trap };
        let mut labels = [Label::Trap; ext::COUNT];
        labels[usize::from(ext::NEXT)] = block(pc + 1);
        labels[usize::from(ext::SKIP)] = block(pc + 2);
        labels[usize::from(ext::DISPATCH)] = Label::Dispatch;
        labels[usize::from(ext::EXIT)] = Label::Exit;
        labels
    }

    fn mark(&mut self) -> usize {
        self.marks.push((false, u32::MAX));
        self.marks.len() - 1
    }

    fn bind_cold(&mut self, mark: usize) {
        self.marks[mark] = (true, self.cold.len() as u32);
    }

    fn bind_here(&mut self, mark: usize) {
        self.marks[mark] = (false, self.code.len() as u32);
    }

    fn elide_jump_to(&mut self, next: usize) {
        let Some(last) = self.fixups.last() else {
            return;
        };
        let (length, field) = if cfg!(target_arch = "x86_64") { (5, Field::Rel32) } else { (4, Field::Branch26) };
        let end = self.code.len() as u32;
        let is_jump = !last.cold && last.label == Label::Block(next) && last.field == field && last.at + 4 == end;
        if !is_jump {
            return;
        }
        let start = end as usize - length;
        if cfg!(target_arch = "x86_64") && self.code[start] != 0xe9 {
            return;
        }
        if self.marks.iter().any(|&(cold, offset)| !cold && offset == end) {
            return;
        }
        self.fixups.pop();
        self.code.truncate(start);
    }

    fn step(&mut self, pc: usize) {
        self.helper(false, pc, HelperId::Step, [0; 3], Some(pc + 1));
    }

    fn helper(&mut self, cold: bool, pc: usize, id: HelperId, args: [i64; 3], next: Option<usize>) {
        let mut labels = self.labels(pc);
        let expected = match next {
            Some(next) if next < self.proto.code.len() => {
                labels[usize::from(ext::NEXT)] = Label::Block(next);
                next as i64
            }
            _ => {
                labels[usize::from(ext::NEXT)] = Label::Dispatch;
                -1
            }
        };
        let values = values(&[(slot::PC, pc as i64), (slot::A, args[0]), (slot::B, args[1]), (slot::C, args[2]), (slot::I, expected)]);
        self.put(cold, Key::Helper(id), &values, &labels);
    }

    fn fast(&mut self, pc: usize, which: usize, args: [i64; 3], next: Option<usize>) {
        self.helper(false, pc, HelperId::Fast(which as u8), args, next);
    }

    fn cold_helper(&mut self, pc: usize, id: HelperId, args: [i64; 3], next: Option<usize>) -> Label {
        let mark = self.mark();
        self.bind_cold(mark);
        self.helper(true, pc, id, args, next);
        Label::Mark(mark)
    }

    fn cold_step(&mut self, pc: usize) -> Label {
        let mark = self.mark();
        self.bind_cold(mark);
        let labels = self.with(pc, &[(ext::TARGET, Label::SharedStep)]);
        self.put(true, Key::StepStub, &values(&[(slot::PC, pc as i64)]), &labels);
        Label::Mark(mark)
    }

    fn cold_template(&mut self, key: Key, values: &Values, labels: &Labels) -> Label {
        let mark = self.mark();
        self.bind_cold(mark);
        self.put(true, key, values, labels);
        Label::Mark(mark)
    }

    fn cold_fast(&mut self, pc: usize, which: usize, args: [i64; 3], next: Option<usize>) -> Label {
        self.cold_helper(pc, HelperId::Fast(which as u8), args, next)
    }

    fn with(&self, pc: usize, pairs: &[(u8, Label)]) -> Labels {
        let mut labels = self.labels(pc);
        for &(id, label) in pairs {
            labels[usize::from(id)] = label;
        }
        labels
    }

    fn constant_ok(&self, index: usize) -> bool {
        index < self.proto.constants.len() && index <= MAX_CONSTANT
    }

    fn lower(&mut self, pc: usize, instruction: Instruction) {
        if matches!(instruction, Instruction::Return { .. } | Instruction::Return0 { .. } | Instruction::Return1 { .. } | Instruction::TailCall { .. }) {
            self.charge_at_return(pc);
        }
        if registers_too_high(instruction) {
            return self.step(pc);
        }
        match instruction {
            Instruction::Move { a, b } => self.put(false, Key::Move, &values(&[(slot::A, offset(a)), (slot::B, offset(b))]), &self.labels(pc)),
            Instruction::LoadI { a, sbx } => self.put(false, Key::LoadInt, &values(&[(slot::A, offset(a)), (slot::I, i64::from(sbx))]), &self.labels(pc)),
            Instruction::LoadF { a, sbx } => {
                let bits = f64::from(sbx).to_bits() as i64;
                self.put(false, Key::LoadFloat, &values(&[(slot::A, offset(a)), (slot::I, bits)]), &self.labels(pc));
            }
            Instruction::LoadK { a, bx } => self.load_constant(pc, a, bx as usize, pc + 1),
            Instruction::LoadKx { a } => match self.proto.code.get(pc + 1) {
                Some(&Instruction::ExtraArg { ax }) => self.load_constant(pc, a, ax as usize, pc + 2),
                _ => self.step(pc),
            },
            Instruction::LoadFalse { a } => self.put(false, Key::LoadBool(false), &values(&[(slot::A, offset(a))]), &self.labels(pc)),
            Instruction::LoadTrue { a } => self.put(false, Key::LoadBool(true), &values(&[(slot::A, offset(a))]), &self.labels(pc)),
            Instruction::LFalseSkip { a } => {
                self.put(false, Key::LoadBool(false), &values(&[(slot::A, offset(a))]), &self.labels(pc));
                self.jump(pc, pc + 2);
            }
            Instruction::LoadNil { a, b } => {
                for register in a..=a.saturating_add(b) {
                    self.put(false, Key::LoadNil, &values(&[(slot::A, offset(register))]), &self.labels(pc));
                }
            }
            Instruction::Add { a, b, c } => self.arith(pc, ArithKind::Add, a, b, Some(c)),
            Instruction::Sub { a, b, c } => self.arith(pc, ArithKind::Sub, a, b, Some(c)),
            Instruction::Mul { a, b, c } => self.arith(pc, ArithKind::Mul, a, b, Some(c)),
            Instruction::BAnd { a, b, c } => self.arith(pc, ArithKind::BAnd, a, b, Some(c)),
            Instruction::BOr { a, b, c } => self.arith(pc, ArithKind::BOr, a, b, Some(c)),
            Instruction::BXor { a, b, c } => self.arith(pc, ArithKind::BXor, a, b, Some(c)),
            Instruction::Mod { a, b, c } => self.arith(pc, ArithKind::Mod, a, b, Some(c)),
            Instruction::IDiv { a, b, c } => self.arith(pc, ArithKind::IDiv, a, b, Some(c)),
            Instruction::Div { a, b, c } => self.arith(pc, ArithKind::Div, a, b, Some(c)),
            Instruction::AddK { a, b, c } => self.arith_constant(pc, ArithKind::Add, a, b, c),
            Instruction::SubK { a, b, c } => self.arith_constant(pc, ArithKind::Sub, a, b, c),
            Instruction::MulK { a, b, c } => self.arith_constant(pc, ArithKind::Mul, a, b, c),
            Instruction::BAndK { a, b, c } => self.arith_constant(pc, ArithKind::BAnd, a, b, c),
            Instruction::BOrK { a, b, c } => self.arith_constant(pc, ArithKind::BOr, a, b, c),
            Instruction::BXorK { a, b, c } => self.arith_constant(pc, ArithKind::BXor, a, b, c),
            Instruction::ModK { a, b, c } => self.arith_constant(pc, ArithKind::Mod, a, b, c),
            Instruction::IDivK { a, b, c } => self.arith_constant(pc, ArithKind::IDiv, a, b, c),
            Instruction::DivK { a, b, c } => self.arith_constant(pc, ArithKind::Div, a, b, c),
            Instruction::AddI { a, b, sc } => {
                let slow = self.cold_step(pc);
                let vals = values(&[(slot::A, offset(a)), (slot::B, offset(b)), (slot::I, i64::from(sc))]);
                self.put(false, Key::Arith { op: ArithKind::Add, rhs: Rhs::Imm }, &vals, &self.with(pc, &[(ext::SLOW, slow)]));
            }
            Instruction::Unm { a, b } => self.unary(pc, Key::Unm, a, b),
            Instruction::BNot { a, b } => self.unary(pc, Key::BNot, a, b),
            Instruction::Not { a, b } => self.put(false, Key::Not, &values(&[(slot::A, offset(a)), (slot::B, offset(b))]), &self.labels(pc)),
            Instruction::Jmp { sj } => self.jump(pc, offset_pc(pc + 1, sj)),
            Instruction::Test { a, k } => self.test(pc, a, k),
            Instruction::TestSet { a, b, k } => self.test_set(pc, a, b, k),
            Instruction::Eq { a, b, k } => self.compare(pc, Key::Compare { cmp: CmpKind::Eq, rhs: Rhs::Reg, int: true, float: true, default_false: false }, a, offset(b), k),
            Instruction::Lt { a, b, k } => self.compare(pc, Key::Compare { cmp: CmpKind::Lt, rhs: Rhs::Reg, int: true, float: true, default_false: false }, a, offset(b), k),
            Instruction::Le { a, b, k } => self.compare(pc, Key::Compare { cmp: CmpKind::Le, rhs: Rhs::Reg, int: true, float: true, default_false: false }, a, offset(b), k),
            Instruction::EqK { a, b, k } => self.equal_constant(pc, a, b, k),
            Instruction::EqI { a, sb, k, .. } => {
                let key = Key::Compare { cmp: CmpKind::Eq, rhs: Rhs::Imm, int: true, float: true, default_false: true };
                self.compare_immediate(pc, key, a, sb, k);
            }
            Instruction::LtI { a, sb, c, k } => self.compare_immediate(pc, immediate(CmpKind::Lt, c), a, sb, k),
            Instruction::LeI { a, sb, c, k } => self.compare_immediate(pc, immediate(CmpKind::Le, c), a, sb, k),
            Instruction::GtI { a, sb, c, k } => self.compare_immediate(pc, immediate(CmpKind::Gt, c), a, sb, k),
            Instruction::GeI { a, sb, c, k } => self.compare_immediate(pc, immediate(CmpKind::Ge, c), a, sb, k),
            Instruction::ForLoop { a, bx } => {
                let slow = self.cold_step(pc);
                self.put(false, Key::ForLoop, &values(&[(slot::A, offset(a))]), &self.with(pc, &[(ext::SLOW, slow)]));
                self.back_edge(pc, offset_pc(pc + 1, -(bx as i32)));
            }
            Instruction::TForLoop { a, bx } => {
                self.put(false, Key::TForLoop, &values(&[(slot::A, offset(a))]), &self.labels(pc));
                self.back_edge(pc, offset_pc(pc + 1, -(bx as i32)));
            }
            Instruction::ForPrep { a, bx } => self.fast(pc, FAST_FOR_PREP, [i64::from(a), i64::from(bx), 0], Some(pc + 1)),
            Instruction::GetTable { a, b, c } => self.array_get(pc, a, b, c, false),
            Instruction::GetI { a, b, c } => self.array_get(pc, a, b, c, true),
            Instruction::SetTable { a, b, c, k } => self.array_set(pc, a, b, c, k, false),
            Instruction::SetI { a, b, c, k } => self.array_set(pc, a, b, c, k, true),
            Instruction::GetField { a, b, c } => self.get_field(pc, a, b, c, false),
            Instruction::SelfOp { a, b, c } => self.get_method(pc, a, b, c),
            Instruction::GetTabUp { a, b, c } => self.get_field(pc, a, b, c, true),
            Instruction::SetField { a, b, c, k } => self.set_field(pc, a, b, c, k),
            Instruction::GetUpval { a, b } => {
                let slow = self.cold_fast(pc, FAST_GET_UPVAL, [i64::from(a), i64::from(b), 0], Some(pc + 1));
                self.put(false, Key::GetUpval, &values(&[(slot::A, offset(a)), (slot::B, crate::jit::layout::upvalue::cell(b))]), &self.with(pc, &[(ext::SLOW, slow)]));
            }
            Instruction::SetUpval { a, b } => {
                let slow = self.cold_fast(pc, FAST_SET_UPVAL, [i64::from(a), i64::from(b), 0], Some(pc + 1));
                self.put(false, Key::SetUpval, &values(&[(slot::A, offset(a)), (slot::B, crate::jit::layout::upvalue::cell(b))]), &self.with(pc, &[(ext::SLOW, slow)]));
            }
            Instruction::Call { a, b, c } => self.call(pc, a, b, c),
            Instruction::Return0 { .. } => self.lower_return(pc, 0, 0),
            Instruction::Return1 { a } => self.lower_return(pc, a, 1),
            Instruction::Return { a, b, k: false, .. } if b != 0 => self.lower_return(pc, a, u32::from(b) - 1),
            Instruction::Return { a, k: false, .. } => self.fast(pc, FAST_RETURN, [i64::from(a), i64::from(u32::MAX as i32), 0], None),
            Instruction::Return { a, b, k: true, .. } if b != 0 => {
                self.close_before_return(pc);
                self.lower_return(pc, a, u32::from(b) - 1);
            }
            Instruction::TailCall { a, b, k, .. } => self.tail_call(pc, a, b, k),
            Instruction::NewTable { a, b, c, k } => self.new_table(pc, a, b, c, k),
            Instruction::Closure { a, bx } => self.fast(pc, FAST_CLOSURE, [i64::from(a), i64::from(bx), 0], Some(pc + 1)),
            Instruction::SetList { a, b, c, k: false } => self.fast(pc, FAST_SET_LIST, [i64::from(a), i64::from(b), i64::from(c)], Some(pc + 1)),
            Instruction::Pow { a, b, c } => self.arith_helper(pc, a, ArithOp::Pow, i64::from(b), i64::from(c)),
            Instruction::PowK { a, b, c } => self.arith_helper(pc, a, ArithOp::Pow, i64::from(b), i64::from(CONSTANT_OPERAND | u32::from(c))),
            Instruction::Shl { a, b, c } => self.arith_helper(pc, a, ArithOp::Shl, i64::from(b), i64::from(c)),
            Instruction::Shr { a, b, c } => self.arith_helper(pc, a, ArithOp::Shr, i64::from(b), i64::from(c)),
            Instruction::ShlI { a, b, sc } => self.arith_helper(pc, a, ArithOp::Shl, i64::from(IMMEDIATE | (sc as u32 & 0x3fff_ffff)), i64::from(b)),
            Instruction::ShrI { a, b, sc } => self.arith_helper(pc, a, ArithOp::Shr, i64::from(b), i64::from(IMMEDIATE | (sc as u32 & 0x3fff_ffff))),
            Instruction::Len { .. } => self.helper(false, pc, HelperId::Misc, [0; 3], Some(pc + 1)),
            _ => self.step(pc),
        }
    }

    fn load_constant(&mut self, pc: usize, a: u8, index: usize, next: usize) {
        if !self.constant_ok(index) {
            return self.step(pc);
        }
        self.put(false, Key::LoadConstant, &values(&[(slot::A, offset(a)), (slot::K, index as i64 * VALUE)]), &self.labels(pc));
        if next != pc + 1 {
            self.jump(pc, next);
        }
    }

    fn jump(&mut self, pc: usize, target: usize) {
        if target <= pc {
            return self.back_edge(pc, target);
        }
        self.put(false, Key::Jump, &values(&[]), &self.with(pc, &[(ext::TARGET, Label::Block(target))]));
    }

    fn back_edge(&mut self, pc: usize, target: usize) {
        let labels = self.back_edge_labels(pc, target);
        let values = self.back_edge_values(pc, target);
        let charge = self.facts.tier_budget.is_some();
        self.put(false, Key::BackEdge { charge }, &values, &labels);
    }

    /// Returns charge the budget too, so a function without loops still reaches the next tier, as V8's Sparkplug does.
    fn charge_at_return(&mut self, pc: usize) {
        let Some(cell) = self.facts.tier_budget else {
            return;
        };
        let tier = self.cold_helper(pc, HelperId::TierCheck, [0; 3], Some(pc));
        let values = values(&[(slot::X, cell as i64), (slot::Y, pc as i64 + 1)]);
        self.put(false, Key::Charge, &values, &self.with(pc, &[(ext::SLOW, tier)]));
    }

    fn back_edge_values(&self, pc: usize, target: usize) -> Values {
        let cell = self.facts.tier_budget.map_or(0, |cell| cell as i64);
        values(&[(slot::X, cell), (slot::Y, (pc + 1 - target) as i64)])
    }

    fn back_edge_labels(&mut self, pc: usize, target: usize) -> Labels {
        let safe_point = self.cold_helper(target, HelperId::SafePoint, [0; 3], Some(target));
        let tier = if self.facts.tier_budget.is_some() { self.cold_helper(target, HelperId::TierCheck, [0; 3], Some(target)) } else { Label::Trap };
        self.with(pc, &[(ext::TARGET, Label::Block(target)), (ext::COLD, safe_point), (ext::SLOW, tier)])
    }

    fn jump_label(&mut self, pc: usize, target: usize) -> Label {
        if target > pc {
            return Label::Block(target);
        }
        let mark = self.mark();
        let labels = self.back_edge_labels(pc, target);
        let values = self.back_edge_values(pc, target);
        let charge = self.facts.tier_budget.is_some();
        self.bind_cold(mark);
        self.put(true, Key::BackEdge { charge }, &values, &labels);
        Label::Mark(mark)
    }

    fn test_target(&self, pc: usize) -> Option<usize> {
        match self.proto.code.get(pc + 1)? {
            Instruction::Jmp { sj } => Some(offset_pc(pc + 2, *sj)),
            _ => None,
        }
    }

    fn outcomes(&mut self, pc: usize, k: bool) -> Option<[(u8, Label); 2]> {
        let target = self.test_target(pc)?;
        let jump = self.jump_label(pc, target);
        let fall = Label::Block(pc + 2);
        let (taken, other) = if k { (jump, fall) } else { (fall, jump) };
        Some([(ext::TRUE, taken), (ext::FALSE, other)])
    }

    fn test(&mut self, pc: usize, a: u8, k: bool) {
        let Some(outcomes) = self.outcomes(pc, k) else {
            return self.step(pc);
        };
        self.put(false, Key::Truthy, &values(&[(slot::A, offset(a))]), &self.with(pc, &outcomes));
    }

    fn test_set(&mut self, pc: usize, a: u8, b: u8, k: bool) {
        let Some(target) = self.test_target(pc) else {
            return self.step(pc);
        };
        let jump = self.jump_label(pc, target);
        let copy = self.mark();
        self.bind_cold(copy);
        let move_values = values(&[(slot::A, offset(a)), (slot::B, offset(b))]);
        self.put(true, Key::MoveJump, &move_values, &self.with(pc, &[(ext::TARGET, jump)]));
        let fall = Label::Block(pc + 2);
        let (taken, other) = if k { (Label::Mark(copy), fall) } else { (fall, Label::Mark(copy)) };
        self.put(false, Key::Truthy, &values(&[(slot::A, offset(b))]), &self.with(pc, &[(ext::TRUE, taken), (ext::FALSE, other)]));
    }

    fn compare(&mut self, pc: usize, key: Key, a: u8, b: i64, k: bool) {
        let Some(outcomes) = self.outcomes(pc, k) else {
            return self.step(pc);
        };
        let slow = self.cold_step(pc);
        let labels = self.with(pc, &[outcomes[0], outcomes[1], (ext::SLOW, slow)]);
        self.put(false, key, &values(&[(slot::A, offset(a)), (slot::B, b), (slot::K, b)]), &labels);
    }

    fn compare_immediate(&mut self, pc: usize, key: Key, a: u8, sb: i32, k: bool) {
        let Some(outcomes) = self.outcomes(pc, k) else {
            return self.step(pc);
        };
        let slow = self.cold_step(pc);
        let labels = self.with(pc, &[outcomes[0], outcomes[1], (ext::SLOW, slow)]);
        self.put(false, key, &values(&[(slot::A, offset(a)), (slot::I, i64::from(sb))]), &labels);
    }

    fn equal_constant(&mut self, pc: usize, a: u8, b: u8, k: bool) {
        let index = usize::from(b);
        let interned = self.facts.interned.get(index).copied().unwrap_or(false);
        let constant = self.proto.constants.get(index).copied();
        let (key, immediate) = match constant {
            Some(Value::Integer(_)) if self.constant_ok(index) => {
                let key = Key::Compare { cmp: CmpKind::Eq, rhs: Rhs::KInt, int: true, float: false, default_false: false };
                return self.compare(pc, key, a, index as i64 * VALUE, k);
            }
            Some(Value::Nil) => (Key::EqTag(TagConstant::Nil), 0),
            Some(Value::Boolean(flag)) => (Key::EqTag(TagConstant::Bool), i64::from(flag)),
            Some(Value::String(string)) if interned => (Key::EqTag(TagConstant::String), string.packed() as i64),
            _ => return self.step(pc),
        };
        let Some(outcomes) = self.outcomes(pc, k) else {
            return self.step(pc);
        };
        self.put(false, key, &values(&[(slot::A, offset(a)), (slot::I, immediate)]), &self.with(pc, &outcomes));
    }

    fn arith(&mut self, pc: usize, op: ArithKind, a: u8, b: u8, c: Option<u8>) {
        let Some(c) = c else {
            return self.step(pc);
        };
        let slow = self.cold_step(pc);
        let vals = values(&[(slot::A, offset(a)), (slot::B, offset(b)), (slot::C, offset(c))]);
        self.put(false, Key::Arith { op, rhs: Rhs::Reg }, &vals, &self.with(pc, &[(ext::SLOW, slow)]));
    }

    fn arith_constant(&mut self, pc: usize, op: ArithKind, a: u8, b: u8, c: u8) {
        let index = usize::from(c);
        if !self.constant_ok(index) {
            return self.step(pc);
        }
        let float_path = matches!(op, ArithKind::Add | ArithKind::Sub | ArithKind::Mul | ArithKind::Div);
        let rhs = match self.proto.constants[index] {
            Value::Integer(value) if matches!(op, ArithKind::Mod | ArithKind::IDiv) && (value == 0 || value == -1) => return self.step(pc),
            Value::Integer(_) => Rhs::KInt,
            Value::Float(_) if float_path => Rhs::KFloat,
            _ => return self.step(pc),
        };
        let slow = self.cold_step(pc);
        let vals = values(&[(slot::A, offset(a)), (slot::B, offset(b)), (slot::K, index as i64 * VALUE)]);
        self.put(false, Key::Arith { op, rhs }, &vals, &self.with(pc, &[(ext::SLOW, slow)]));
    }

    fn unary(&mut self, pc: usize, key: Key, a: u8, b: u8) {
        let slow = self.cold_step(pc);
        let mut labels = self.with(pc, &[(ext::SLOW, slow)]);
        labels[usize::from(ext::SKIP)] = labels[usize::from(ext::NEXT)];
        self.put(false, key, &values(&[(slot::A, offset(a)), (slot::B, offset(b))]), &labels);
    }

    fn arith_helper(&mut self, pc: usize, a: u8, op: ArithOp, left: i64, right: i64) {
        let packed = i64::from(a) | ((op as i64) << 16);
        self.fast(pc, FAST_ARITH, [packed, left, right], Some(pc + 2));
    }

    fn new_table(&mut self, pc: usize, a: u8, b: u8, c: u32, k: bool) {
        let extra = match (k, self.proto.code.get(pc + 1)) {
            (false, Some(Instruction::ExtraArg { .. })) => 0,
            (true, Some(&Instruction::ExtraArg { ax })) => u64::from(ax),
            _ => return self.step(pc),
        };
        let array = u64::from(c) + extra * 1024;
        if array > i32::MAX as u64 {
            return self.step(pc);
        }
        self.fast(pc, FAST_NEW_TABLE, [i64::from(a), array as i64, i64::from(b)], Some(pc + 2));
    }

    fn array_get(&mut self, pc: usize, a: u8, b: u8, c: u8, constant_key: bool) {
        let which = if constant_key { FAST_GET_INT } else { FAST_GET_TABLE };
        let args = [i64::from(a), i64::from(b), i64::from(c)];
        if !self.library.has_tables() {
            return self.fast(pc, which, args, Some(pc + 1));
        }
        let mut slow = self.cold_fast(pc, which, args, Some(pc + 1));
        let vals = values(&[(slot::A, offset(a)), (slot::B, offset(b)), (slot::C, offset(c)), (slot::I, i64::from(c))]);
        if !constant_key {
            slow = self.cold_template(Key::HashGet, &vals, &self.with(pc, &[(ext::SLOW, slow)]));
        }
        self.put(false, Key::ArrayGet { constant_key }, &vals, &self.with(pc, &[(ext::SLOW, slow)]));
    }

    fn array_set(&mut self, pc: usize, a: u8, b: u8, c: u8, k: bool, constant_key: bool) {
        let which = if constant_key { FAST_SET_INT } else { FAST_SET_TABLE };
        let operand = operand_index(c, k);
        let args = [i64::from(a), i64::from(b), operand];
        if !self.library.has_tables() || (k && !self.constant_ok(usize::from(c))) {
            return self.fast(pc, which, args, Some(pc + 1));
        }
        let mut slow = self.cold_fast(pc, which, args, Some(pc + 1));
        let barrier = self.cold_fast(pc, FAST_TABLE_BARRIER, [i64::from(a), operand, 0], Some(pc + 1));
        let vals = values(&[
            (slot::A, offset(a)),
            (slot::B, offset(b)),
            (slot::C, offset(c)),
            (slot::K, i64::from(c) * VALUE),
            (slot::I, i64::from(b)),
        ]);
        if !constant_key {
            let labels = self.with(pc, &[(ext::SLOW, slow), (ext::COLD, barrier)]);
            slow = self.cold_template(Key::HashSet { constant_value: k }, &vals, &labels);
        }
        let key = Key::ArraySet { constant_key, constant_value: k };
        self.put(false, key, &vals, &self.with(pc, &[(ext::SLOW, slow), (ext::COLD, barrier)]));
    }

    fn get_field(&mut self, pc: usize, a: u8, b: u8, c: u8, upvalue: bool) {
        let which = if upvalue { FAST_GET_TABUP } else { FAST_GET_FIELD };
        let args = [i64::from(a), i64::from(b), i64::from(c)];
        let entries: Vec<_> = field_entries(self.proto, pc).into_iter().filter(|entry| entry.slot == ABSENT || entry.slot <= MAX_SLOT).collect();
        if !self.library.has_tables() || !self.facts.inline_caches || entries.is_empty() {
            return self.fast(pc, which, args, Some(pc + 1));
        }
        let slow = self.cold_fast(pc, which, args, Some(pc + 1));
        let labels = self.with(pc, &[(ext::SLOW, slow), (ext::TARGET, slow)]);
        let table = if upvalue { Key::UpvalueTable } else { Key::RegisterTable };
        self.put(false, table, &values(&[(slot::X, offset(b)), (slot::B, crate::jit::layout::upvalue::cell(b))]), &labels);
        for entry in entries {
            let absent = entry.slot == ABSENT;
            let vals = values(&[(slot::A, offset(a)), (slot::Y, entry.shape.packed() as i64), (slot::X, i64::from(entry.slot) * VALUE)]);
            self.put(false, Key::FieldGet { absent, method: false }, &vals, &labels);
        }
        self.put(false, Key::Jump, &values(&[]), &labels);
    }

    fn get_method(&mut self, pc: usize, a: u8, b: u8, c: u8) {
        let args = [i64::from(a), i64::from(b), i64::from(c)];
        let entries: Vec<_> = field_entries(self.proto, pc).into_iter().filter(|entry| entry.slot == ABSENT || entry.slot <= MAX_SLOT).collect();
        let methods = all_method_entries(self.proto, pc);
        if !self.library.has_tables() || !self.facts.inline_caches || (entries.is_empty() && methods.is_empty()) {
            return self.fast(pc, FAST_GET_SELF, args, Some(pc + 1));
        }
        let slow = self.cold_fast(pc, FAST_GET_SELF, args, Some(pc + 1));
        let labels = self.with(pc, &[(ext::SLOW, slow), (ext::TARGET, slow)]);
        self.put(false, Key::RegisterTable, &values(&[(slot::X, offset(b))]), &labels);
        if methods.is_empty() {
            for entry in entries {
                let absent = entry.slot == ABSENT;
                let vals = values(&[(slot::A, offset(a)), (slot::B, offset(b)), (slot::Y, entry.shape.packed() as i64), (slot::X, i64::from(entry.slot) * VALUE)]);
                self.put(false, Key::FieldGet { absent, method: true }, &vals, &labels);
            }
        } else {
            for method in methods {
                let vals = values(&[
                    (slot::A, offset(a)),
                    (slot::B, offset(b)),
                    (slot::Y, method.receiver.packed() as i64),
                    (slot::Z, method.metatable_shape.packed() as i64),
                    (slot::W, i64::from(method.index_slot) * VALUE),
                    (slot::V, method.class.packed() as i64),
                    (slot::C, method.class.index() as i64),
                    (slot::U, method.class_shape.packed() as i64),
                    (slot::X, i64::from(method.class_slot) * VALUE),
                ]);
                if method.depth == 0 {
                    self.put(false, Key::MethodGet, &vals, &labels);
                    continue;
                }
                self.deep_method(pc, &method, &vals, slow);
            }
        }
        self.put(false, Key::Jump, &values(&[]), &labels);
    }

    fn deep_method(&mut self, pc: usize, method: &crate::vm::ic::MethodEntry, first: &Values, slow: Label) {
        let next = self.mark();
        let labels = self.with(pc, &[(ext::SLOW, slow), (ext::TARGET, Label::Mark(next))]);
        self.put(false, Key::MethodStart, first, &labels);
        for hop in method.hops() {
            let vals = values(&[
                (slot::Z, hop.metatable_shape.packed() as i64),
                (slot::W, i64::from(hop.index_slot) * VALUE),
                (slot::V, hop.class.packed() as i64),
                (slot::C, hop.class.index() as i64),
                (slot::U, hop.class_shape.packed() as i64),
            ]);
            self.put(false, Key::MethodHop, &vals, &labels);
        }
        self.put(false, Key::MethodFinish, first, &labels);
        self.bind_here(next);
        self.put(false, Key::RegisterTable, &values(&[(slot::X, first[usize::from(slot::B)])]), &labels);
    }

    fn set_field(&mut self, pc: usize, a: u8, b: u8, c: u8, k: bool) {
        let operand = operand_index(c, k);
        let args = [i64::from(a), i64::from(b), operand];
        let entries: Vec<_> = field_entries(self.proto, pc).into_iter().filter(|entry| entry.slot != ABSENT).collect();
        let additions: Vec<_> = add_entries(self.proto, pc).into_iter().filter(|entry| entry.slot < MAX_SLOT).collect();
        if !self.library.has_tables() || !self.facts.inline_caches || (entries.is_empty() && additions.is_empty()) || (k && !self.constant_ok(usize::from(c))) {
            return self.fast(pc, FAST_SET_FIELD, args, Some(pc + 1));
        }
        let slow = self.cold_fast(pc, FAST_SET_FIELD, args, Some(pc + 1));
        let barrier = self.cold_fast(pc, FAST_TABLE_BARRIER, [i64::from(a), operand, 0], Some(pc + 1));
        let labels = self.with(pc, &[(ext::SLOW, slow), (ext::TARGET, slow), (ext::COLD, barrier)]);
        self.put(false, Key::RegisterTable, &values(&[(slot::X, offset(a))]), &labels);
        for entry in entries {
            let vals = values(&[
                (slot::A, offset(a)),
                (slot::C, offset(c)),
                (slot::K, i64::from(c) * VALUE),
                (slot::Y, entry.shape.packed() as i64),
                (slot::I, i64::from(entry.slot) * VALUE),
            ]);
            self.put(false, Key::FieldSet { constant: k }, &vals, &labels);
        }
        for entry in additions {
            let vals = values(&[
                (slot::A, offset(a)),
                (slot::C, offset(c)),
                (slot::K, i64::from(c) * VALUE),
                (slot::Y, entry.shape.packed() as i64),
                (slot::Z, entry.next.packed() as i64),
                (slot::W, i64::from(entry.slot) + 1),
                (slot::I, i64::from(entry.slot) * VALUE),
            ]);
            self.put(false, Key::FieldAdd { constant: k }, &vals, &labels);
        }
        self.put(false, Key::Jump, &values(&[]), &labels);
    }

    fn call_targets(&self, pc: usize, a: u8, b: u8) -> Vec<&'a CallTarget> {
        let mut targets: Vec<&CallTarget> = Vec::new();
        for target in self.facts.calls.get(pc).map_or(&[][..], Vec::as_slice) {
            let fits = u32::from(a) + 1 + u32::from(target.num_params.max(b.saturating_sub(1))) <= u32::from(MAX_REGISTER);
            if fits && !targets.iter().any(|known| known.proto == target.proto) {
                targets.push(target);
            }
        }
        targets
    }

    fn tail_call(&mut self, pc: usize, a: u8, b: u8, k: bool) {
        let targets = self.call_targets(pc, a, b);
        if targets.is_empty() || b == 0 || k || self.proto.is_vararg || !self.library.has_frames() {
            return self.fast(pc, FAST_TAIL_CALL, [0; 3], None);
        }
        let slow = self.cold_fast(pc, FAST_TAIL_CALL, [0; 3], None);
        let arguments = i64::from(b) - 1;
        let count = targets.len();
        for (index, target) in targets.into_iter().enumerate() {
            let fresh = (index + 1 < count).then(|| self.mark());
            let next = fresh.map_or(slow, Label::Mark);
            let register_count = arguments + i64::from(target.max_stack);
            let span = register_count.max(crate::vm::REGISTER_SPAN as i64);
            let check = values(&[
                (slot::A, offset(a)),
                (slot::Y, target.proto as i64),
                (slot::X, target.entry as i64),
                (slot::I, 1 + span),
                (slot::W, 1 + register_count),
            ]);
            self.put(false, Key::TailCheck, &check, &self.with(pc, &[(ext::SLOW, slow), (ext::TARGET, next)]));
            for index in 0..=arguments {
                let vals = values(&[(slot::A, (i64::from(a) + index) * VALUE), (slot::B, index * VALUE)]);
                self.put(false, Key::ReturnCopy, &vals, &self.labels(pc));
            }
            for parameter in arguments..i64::from(target.num_params) {
                self.put(false, Key::ClearAt, &values(&[(slot::A, (1 + parameter) * VALUE)]), &self.labels(pc));
            }
            let enter = values(&[(slot::K, target.constants as i64), (slot::Y, target.proto as i64)]);
            self.put(false, Key::TailEnter { upvalues: target.has_upvalues }, &enter, &self.labels(pc));
            if let Some(mark) = fresh {
                self.bind_here(mark);
            }
        }
    }

    fn call(&mut self, pc: usize, a: u8, b: u8, c: u8) {
        let args = [i64::from(a), i64::from(b), i64::from(c)];
        let targets = self.call_targets(pc, a, b);
        let megamorphic = self.facts.inline_caches && matches!(self.proto.exec.feedback.get(pc), Some(InlineCache::Megamorphic));
        if b == 0 || !self.library.has_frames() || u32::from(a) + u32::from(b) > u32::from(MAX_REGISTER) || (targets.is_empty() && !megamorphic) {
            return self.fast(pc, FAST_CALL, args, Some(pc + 1));
        }
        let slow = self.cold_fast(pc, FAST_CALL, args, Some(pc + 1));
        let generic = megamorphic.then(|| self.mark());
        let arguments = i64::from(b) - 1;
        let first = i64::from(a) + 1;
        let enter = |constants: i64, proto: i64| {
            values(&[
                (slot::A, offset(a)),
                (slot::B, first),
                (slot::C, i64::from(c) - 1),
                (slot::K, constants),
                (slot::Y, proto),
                (slot::V, first * VALUE),
                (slot::I, pc as i64 + 1),
                (slot::PC, pc as i64),
            ])
        };
        for (index, target) in targets.iter().enumerate() {
            let fresh = (index + 1 < targets.len()).then(|| self.mark());
            let next = fresh.or(generic).map_or(slow, Label::Mark);
            let register_count = arguments + i64::from(target.max_stack);
            let span = register_count.max(crate::vm::REGISTER_SPAN as i64);
            let check = values(&[
                (slot::A, offset(a)),
                (slot::Y, target.proto as i64),
                (slot::X, target.entry as i64),
                (slot::I, first + span),
                (slot::W, first + register_count),
            ]);
            self.put(false, Key::CallCheck, &check, &self.with(pc, &[(ext::SLOW, slow), (ext::TARGET, next)]));
            for parameter in arguments..i64::from(target.num_params) {
                self.put(false, Key::LoadNil, &values(&[(slot::A, (first + parameter) * VALUE)]), &self.labels(pc));
            }
            let values = enter(target.constants as i64, target.proto as i64);
            self.put(false, Key::CallEnter { upvalues: target.has_upvalues }, &values, &self.with(pc, &[(ext::SLOW, slow)]));
            if let Some(mark) = fresh {
                self.bind_here(mark);
            }
        }
        let Some(generic) = generic else {
            return;
        };
        self.bind_here(generic);
        let check = values(&[(slot::A, offset(a)), (slot::B, first), (slot::W, first + arguments), (slot::U, arguments), (slot::V, first * VALUE)]);
        self.put(false, Key::CallAnyCheck, &check, &self.with(pc, &[(ext::SLOW, slow)]));
        self.put(false, Key::CallAnyEnter, &enter(0, 0), &self.with(pc, &[(ext::SLOW, slow)]));
    }

    /// The helper answers this instruction's pc to continue into the inline return.
    fn close_before_return(&mut self, pc: usize) {
        let after = self.mark();
        let mut labels = self.labels(pc);
        labels[usize::from(ext::NEXT)] = Label::Mark(after);
        let values = values(&[(slot::PC, pc as i64), (slot::A, 0), (slot::B, 0), (slot::C, 0), (slot::I, pc as i64)]);
        self.put(false, Key::Helper(HelperId::Fast(FAST_CLOSE as u8)), &values, &labels);
        self.bind_here(after);
    }

    fn lower_return(&mut self, pc: usize, first: u8, count: u32) {
        if u32::from(first) + count > u32::from(MAX_REGISTER) {
            return self.fast(pc, FAST_RETURN, [i64::from(first), i64::from(count), 0], None);
        }
        let slow = self.cold_fast(pc, FAST_RETURN, [i64::from(first), i64::from(count), 0], None);
        let done = self.mark();
        let wide = self.mark();
        let labels = self.with(pc, &[(ext::SLOW, slow), (ext::TRUE, Label::Mark(done)), (ext::TARGET, Label::Mark(wide))]);
        let counted = values(&[(slot::I, i64::from(count))]);
        self.put(false, Key::ReturnCheck, &counted, &labels);
        self.return_copies(false, first, count, &labels);
        self.bind_here(done);
        self.put(false, Key::ReturnDone, &values(&[]), &labels);
        self.bind_cold(wide);
        self.put(true, Key::ReturnWide, &values(&[]), &labels);
        self.return_copies(true, first, count, &labels);
        self.put(true, Key::ReturnFill, &counted, &labels);
    }

    fn return_copies(&mut self, cold: bool, first: u8, count: u32, labels: &Labels) {
        for index in 0..count {
            let vals = values(&[(slot::A, offset(first) + i64::from(index) * VALUE), (slot::B, i64::from(index) * VALUE)]);
            self.put(cold, Key::ReturnCopy, &vals, labels);
        }
    }
}

fn place(buffer: &mut Vec<u8>, fixups: &mut Vec<Fixup>, cold: bool, stencil: &SplitStencil, values: &Values, labels: &Labels) {
    let base = buffer.len() as u32;
    buffer.extend_from_slice(&stencil.bytes);
    for &Hole { offset, field, slot, add } in &stencil.values {
        let value = values[usize::from(slot)].wrapping_add(i64::from(add));
        patch_value(buffer, (base + offset) as usize, field, value);
    }
    fixups.extend(stencil.branches.iter().map(|&Hole { offset, field, slot, .. }| Fixup {
        cold,
        at: base + offset,
        field,
        label: labels[usize::from(slot)],
    }));
}

fn immediate(cmp: CmpKind, c: u8) -> Key {
    let float = c != 0;
    Key::Compare { cmp, rhs: Rhs::Imm, int: !float, float, default_false: false }
}

fn operand_index(register: u8, k: bool) -> i64 {
    if k { i64::from(u32::from(register) | CONSTANT_OPERAND) } else { i64::from(register) }
}

/// Templates' displacements cannot reach registers past `MAX_REGISTER`, so those take the step.
fn registers_too_high(instruction: Instruction) -> bool {
    let high = |register: u8| register > MAX_REGISTER;
    match instruction {
        Instruction::Move { a, b }
        | Instruction::Not { a, b }
        | Instruction::Unm { a, b }
        | Instruction::BNot { a, b }
        | Instruction::GetUpval { a, b }
        | Instruction::SetUpval { a, b } => high(a) || high(b),
        Instruction::Add { a, b, c }
        | Instruction::Sub { a, b, c }
        | Instruction::Mul { a, b, c }
        | Instruction::Div { a, b, c }
        | Instruction::Mod { a, b, c }
        | Instruction::IDiv { a, b, c }
        | Instruction::BAnd { a, b, c }
        | Instruction::BOr { a, b, c }
        | Instruction::BXor { a, b, c }
        | Instruction::GetTable { a, b, c }
        | Instruction::SetTable { a, b, c, .. } => high(a) || high(b) || high(c),
        Instruction::ForLoop { a, .. } | Instruction::TForLoop { a, .. } | Instruction::LoadNil { a, .. } => a > MAX_REGISTER - 4,
        Instruction::SelfOp { a, b, .. } => a >= MAX_REGISTER || high(b),
        _ => false,
    }
}

fn trap_length() -> u32 {
    if cfg!(target_arch = "x86_64") { 2 } else { 4 }
}

fn read_word(code: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(code[at..at + 4].try_into().expect("a four-byte field"))
}

fn write_word(code: &mut [u8], at: usize, word: u32) {
    code[at..at + 4].copy_from_slice(&word.to_le_bytes());
}

/// The compile only gives a scaled field values it can hold.
pub(super) fn patch_value(code: &mut [u8], at: usize, field: Field, value: i64) {
    match field {
        Field::Le32 => code[at..at + 4].copy_from_slice(&(value as i32).to_le_bytes()),
        Field::Le64 => code[at..at + 8].copy_from_slice(&value.to_le_bytes()),
        Field::Imm12 { scale } => {
            debug_assert!(value >= 0 && value % (1 << scale) == 0 && (value >> scale) < 4096, "displacement {value} does not fit");
            let word = read_word(code, at) & !(0xfff << 10);
            write_word(code, at, word | (((value >> scale) as u32 & 0xfff) << 10));
        }
        Field::Imm16 { shift } => {
            let word = read_word(code, at) & !(0xffff << 5);
            write_word(code, at, word | ((((value as u64) >> shift) as u32 & 0xffff) << 5));
        }
        Field::Rel32 | Field::Branch26 | Field::Branch19 | Field::Branch14 | Field::Adr => unreachable!("branch fields are patched once laid out"),
    }
}

/// `None` when the target is out of range for the field.
pub(super) fn patch_branch(code: &mut [u8], at: usize, field: Field, target: i64) -> Option<()> {
    let from = at as i64;
    match field {
        Field::Rel32 => {
            let rel = i32::try_from(target - (from + 4)).ok()?;
            code[at..at + 4].copy_from_slice(&rel.to_le_bytes());
        }
        Field::Branch26 | Field::Branch19 | Field::Branch14 => {
            let rel = (target - from) >> 2;
            let (bits, shift) = match field {
                Field::Branch26 => (26, 0),
                Field::Branch19 => (19, 5),
                _ => (14, 5),
            };
            if rel < -(1 << (bits - 1)) || rel >= 1 << (bits - 1) {
                return None;
            }
            let mask = ((1u32 << bits) - 1) << shift;
            let word = read_word(code, at) & !mask;
            write_word(code, at, word | (((rel as u32) << shift) & mask));
        }
        Field::Adr => {
            let rel = target - from;
            if !(-(1 << 20)..1 << 20).contains(&rel) {
                return None;
            }
            let rel = rel as u32;
            let word = read_word(code, at) & !((3 << 29) | (0x7ffff << 5));
            write_word(code, at, word | ((rel & 3) << 29) | (((rel >> 2) & 0x7ffff) << 5));
        }
        Field::Le32 | Field::Le64 | Field::Imm12 { .. } | Field::Imm16 { .. } => unreachable!("value fields are patched when placed"),
    }
    Some(())
}
