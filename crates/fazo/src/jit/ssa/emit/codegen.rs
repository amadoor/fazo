//! One pass that allocates registers and emits each node's code, after Maglev's code generator (src/maglev/maglev-code-generator.cc).

use std::mem::offset_of;

use super::{
    Addresses,
    alloc::{self, Alloc, Entry, Have, Loc, Moves, Part, Want, part, value_of},
    asm::{Asm, Cond, DUMP, Label},
    exits::{ExitTable, Place, Word},
    live::{self, Liveness},
    regs::{Class, PReg},
};
use crate::jit::{
    JitCtx,
    ssa::{
        ir::{BlockId, Const, Edge, Func, Op, Rep, StateId, Term, Type, Val},
        lower::Env,
    },
};

pub(super) const NIL_TAG: u8 = 0;
pub(super) const BOOL_TAG: u8 = 1;
pub(super) const INT_TAG: u8 = 2;
pub(super) const FLOAT_TAG: u8 = 3;
pub(super) const STRING_TAG: u8 = 4;
pub(super) const TABLE_TAG: u8 = 5;
pub(super) const CLOSURE_TAG: u8 = 6;
pub(super) const LIGHT_USERDATA_TAG: u8 = 10;
pub(super) const VALUE_SIZE: i32 = 16;
pub(super) const PAYLOAD: i32 = 8;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Src {
    R(u8),
    F(u8),
    I(i64),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Def {
    None,
    Gp(u8),
    Float(u8),
    Bool(u8),
    Two(u8, u8),
}

pub(super) struct Mover<'x, A: Asm> {
    pub(super) asm: &'x mut A,
    pub(super) spill_at: i32,
}

impl<A: Asm> Mover<'_, A> {
    fn at(&self, slot: u32) -> i32 {
        self.spill_at + 8 * slot as i32
    }
}

impl<A: Asm> Moves for Mover<'_, A> {
    fn copy(&mut self, dst: PReg, src: PReg) {
        match dst.class {
            Class::Gp => self.asm.mov(dst.hw, src.hw),
            Class::Fp => self.asm.fmov(dst.hw, src.hw),
        }
    }

    fn spill(&mut self, slot: u32, src: PReg) {
        let at = self.at(slot);
        match src.class {
            Class::Gp => self.asm.store(super::asm::Width::Q, src.hw, A::SP, at),
            Class::Fp => self.asm.store_f(src.hw, A::SP, at),
        }
    }

    fn reload(&mut self, dst: PReg, slot: u32) {
        let at = self.at(slot);
        match dst.class {
            Class::Gp => self.asm.load(super::asm::Width::Q, dst.hw, A::SP, at),
            Class::Fp => self.asm.load_f(dst.hw, A::SP, at),
        }
    }

    fn constant(&mut self, dst: PReg, bits: u64) {
        match dst.class {
            Class::Gp => self.asm.mov_imm(dst.hw, bits as i64),
            Class::Fp => self.asm.fconst(dst.hw, bits),
        }
    }

    fn constant_slot(&mut self, slot: u32, bits: u64) {
        let at = self.at(slot);
        self.asm.store_imm(super::asm::Width::Q, A::SP, at, bits as i64);
    }

    fn slot_copy(&mut self, _class: Class, dst: u32, src: u32) {
        let (to, from) = (self.at(dst), self.at(src));
        self.asm.load(super::asm::Width::Q, A::SCRATCH, A::SP, from);
        self.asm.store(super::asm::Width::Q, A::SCRATCH, A::SP, to);
    }

    fn save_scratch(&mut self, src: PReg) {
        match src.class {
            Class::Gp => self.asm.mov(A::SCRATCH, src.hw),
            Class::Fp => self.asm.fmov(A::FSCRATCH, src.hw),
        }
    }

    fn restore_scratch(&mut self, dst: PReg) {
        match dst.class {
            Class::Gp => self.asm.mov(dst.hw, A::SCRATCH),
            Class::Fp => self.asm.fmov(dst.hw, A::FSCRATCH),
        }
    }
}

/// A loop poll's slow path, placed after the code; it re-enters the loop through its OSR block, so it needs no registers.
pub(super) struct SafepointStub {
    pub(super) label: Label,
    pub(super) state: Option<u32>,
    pub(super) pc: u32,
    pub(super) charge: bool,
    pub(super) target: Label,
}

pub(super) struct Gen<'a, A: Asm> {
    pub(super) func: &'a Func,
    pub(super) env: &'a Env<'a>,
    pub(super) addresses: &'a Addresses,
    pub(super) live: Liveness,
    pub(super) asm: A,
    pub(super) alloc: Alloc,
    labels: Vec<Option<Label>>,
    snapshots: Vec<Option<Vec<Entry>>>,
    stubs: Vec<(Label, u32)>,
    pub(super) safepoints: Vec<SafepointStub>,
    pub(super) table: Box<ExitTable>,
    pub(super) epilogue: Label,
    pub(super) exit_with: Label,
    pub(super) tail_exit: Label,
    exit_tail: Label,
    pub(super) dying: Vec<Vec<Val>>,
    pub(super) position: usize,
    pub(super) current: Val,
    pub(super) spill_at: i32,
    pub(super) scratch_at: i32,
    pub(super) ctx_at: i32,
    next_block: Option<BlockId>,
    pub(super) switched: Option<Val>,
    /// A load that already matched the next node's type guard jumps past the guard's check.
    pub(super) guarded: Option<(Val, Label)>,
    pub(super) block_now: BlockId,
    idom: Option<Vec<Option<BlockId>>>,
    /// Lets a call's settled path jump straight to this pc's case.
    pub(super) settled: Option<(u32, Label)>,
    /// A call into a vararg prototype moves the results' place, so every call writes `ret_offset`.
    pub(super) calls_vararg: bool,
    pub(super) failed: bool,
}

pub(super) struct Generated<A> {
    pub(super) asm: A,
    pub(super) table: Box<ExitTable>,
    pub(super) slots: u32,
    pub(super) used: super::regs::RegSet,
    pub(super) epilogue: Label,
    pub(super) exit_with: Label,
    pub(super) tail_exit: Label,
    pub(super) roots: Vec<(u32, Label)>,
}

/// One extra value past the instructions holds the VM register base between nodes that cannot move the stack.
fn classes(func: &Func) -> Vec<Class> {
    let mut classes = Vec::with_capacity(func.insts.len() * 2 + 2);
    for inst in &func.insts {
        classes.push(if inst.rep == Rep::Float { Class::Fp } else { Class::Gp });
        classes.push(Class::Gp);
    }
    classes.extend([Class::Gp, Class::Gp]);
    classes
}

pub(super) fn parts(func: &Func, value: Val) -> &'static [u32] {
    match func.rep(value) {
        Rep::Boxed => &[0, 1],
        Rep::None => &[],
        _ => &[0],
    }
}

impl<'a, A: Asm> Gen<'a, A> {
    pub(super) fn new(func: &'a Func, env: &'a Env<'a>, addresses: &'a Addresses, table: Box<ExitTable>, (scratch_at, ctx_at, spill_at): (i32, i32, i32)) -> Self {
        let mut asm = A::new();
        let epilogue = asm.new_label();
        let exit_with = asm.new_label();
        let exit_tail = asm.new_label();
        let tail_exit = asm.new_label();
        Self {
            func,
            env,
            addresses,
            live: Liveness::new(func),
            asm,
            alloc: Alloc::new(if has_loop(func) { A::REGISTERS } else { A::CALLER_SAVED_FIRST }, classes(func)),
            labels: vec![None; func.blocks.len()],
            snapshots: vec![None; func.blocks.len()],
            stubs: Vec::new(),
            safepoints: Vec::new(),
            table,
            epilogue,
            exit_with,
            tail_exit,
            exit_tail,
            dying: Vec::new(),
            position: 0,
            current: Val(u32::MAX),
            spill_at,
            scratch_at,
            ctx_at,
            next_block: None,
            switched: None,
            guarded: None,
            block_now: BlockId(0),
            idom: None,
            settled: None,
            calls_vararg: env.calls.iter().flatten().any(|target| target.vararg),
            failed: false,
        }
    }

    pub(super) fn run(mut self) -> Option<Generated<A>> {
        let roots = self.roots();
        let order = self.live.order.clone();
        for (index, &block) in order.iter().enumerate() {
            self.next_block = order.get(index + 1).copied();
            self.block(block);
            if self.failed {
                return None;
            }
        }
        self.tails();
        Some(Generated {
            asm: self.asm,
            table: self.table,
            slots: self.alloc.slots,
            used: self.alloc.used,
            epilogue: self.epilogue,
            exit_with: self.exit_with,
            tail_exit: self.tail_exit,
            roots,
        })
    }

    fn roots(&mut self) -> Vec<(u32, Label)> {
        let start = super::super::entry_pc(self.env.vararg);
        let mut roots: Vec<(u32, BlockId)> = self.func.entry.map(|block| (start, block)).into_iter().collect();
        roots.extend(self.func.osr.iter().copied().filter(|&(at, _)| at != 0 && at != start));
        roots.into_iter().map(|(pc, block)| {
            self.snapshots[block.0 as usize] = Some(Vec::new());
            (pc, self.label(block))
        }).collect()
    }

    pub(super) fn label(&mut self, block: BlockId) -> Label {
        if let Some(label) = self.labels[block.0 as usize] {
            return label;
        }
        let label = self.asm.new_label();
        self.labels[block.0 as usize] = Some(label);
        label
    }

    pub(super) fn mover(&mut self) -> (&mut Alloc, Mover<'_, A>) {
        (&mut self.alloc, Mover { asm: &mut self.asm, spill_at: self.spill_at })
    }

    fn block(&mut self, block: BlockId) {
        let label = self.label(block);
        self.asm.bind(label);
        let entries = match self.snapshots[block.0 as usize].clone() {
            Some(entries) => entries,
            None => {
                let parts = self.block_parts(block);
                self.alloc.in_slots(&parts)
            }
        };
        self.alloc.enter(&entries);
        self.block_now = block;
        self.dying = self.live.deaths(self.func, block, &mut self.alloc.last);
        let data = self.func.block(block);
        self.switched = match data.term {
            Term::Switch { code, .. } if data.insts.last() == Some(&code) => Some(code),
            _ => None,
        };
        for (position, &value) in data.insts.iter().enumerate() {
            self.position = position;
            self.current = value;
            self.node(value);
            let dying = std::mem::take(&mut self.dying[position]);
            self.alloc.end_node(&dying);
            self.dying[position] = dying;
            if self.failed || self.alloc.broken {
                self.failed = true;
                return;
            }
        }
        self.position = data.insts.len();
        self.current = Val(u32::MAX);
        self.term(&data.term);
    }

    fn block_parts(&self, block: BlockId) -> Vec<Part> {
        let mut parts = Vec::new();
        for value in self.live.live_in(block) {
            parts.extend(parts_of(self.func, value));
        }
        for &param in &self.func.block(block).params {
            if live::has_location(self.func, param) {
                parts.extend(parts_of(self.func, param));
            }
        }
        parts
    }

    pub(super) fn is_const(&self, value: Val) -> Option<Const> {
        match self.func.op(value) {
            Op::Const(constant) => Some(*constant),
            _ => None,
        }
    }

    pub(super) fn use_part(&mut self, part: Part) -> PReg {
        let (alloc, mut moves) = self.mover();
        alloc.use_reg(part, &mut moves)
    }

    pub(super) fn temp(&mut self) -> u8 {
        let (alloc, mut moves) = self.mover();
        alloc.temp(Class::Gp, &mut moves).hw
    }

    pub(super) fn ftemp(&mut self) -> u8 {
        let (alloc, mut moves) = self.mover();
        alloc.temp(Class::Fp, &mut moves).hw
    }

    /// A boxed value's word is its payload.
    pub(super) fn word(&mut self, value: Val) -> Src {
        if let Some(constant) = self.is_const(value) {
            return Src::I(constant.payload() as i64);
        }
        let reg = self.use_part(part(value, 0));
        match reg.class {
            Class::Gp => Src::R(reg.hw),
            Class::Fp => Src::F(reg.hw),
        }
    }

    /// A float's bits move into a temporary.
    pub(super) fn reg(&mut self, value: Val) -> u8 {
        match self.word(value) {
            Src::R(reg) => reg,
            Src::I(bits) => {
                let reg = self.temp();
                self.asm.mov_imm(reg, bits);
                reg
            }
            Src::F(reg) => {
                let bits = self.temp();
                self.asm.float_to_bits(bits, reg);
                bits
            }
        }
    }

    pub(super) fn freg(&mut self, value: Val) -> u8 {
        match self.word(value) {
            Src::F(reg) => reg,
            Src::I(bits) => {
                let reg = self.ftemp();
                self.asm.fconst(reg, bits as u64);
                reg
            }
            Src::R(bits) => {
                let reg = self.ftemp();
                self.asm.bits_to_float(reg, bits);
                reg
            }
        }
    }

    pub(super) fn tag(&mut self, value: Val) -> Src {
        if let Some(constant) = self.is_const(value) {
            return Src::I(i64::from(constant.tag()));
        }
        match self.func.rep(value) {
            Rep::Boxed => Src::R(self.use_part(part(value, 1)).hw),
            Rep::Int => Src::I(i64::from(INT_TAG)),
            Rep::Float => Src::I(i64::from(FLOAT_TAG)),
            Rep::Bool => Src::I(i64::from(BOOL_TAG)),
            _ => Src::I(i64::from(self.func.ty(value).tag().unwrap_or(NIL_TAG))),
        }
    }

    pub(super) fn pair(&mut self, value: Val) -> (Src, Src) {
        let tag = self.tag(value);
        let payload = self.word(value);
        (tag, payload)
    }

    pub(super) fn pair_regs(&mut self, value: Val) -> (u8, u8) {
        let (tag, payload) = self.pair(value);
        (self.src_reg(tag), self.src_reg(payload))
    }

    pub(super) fn src_reg(&mut self, src: Src) -> u8 {
        match src {
            Src::R(reg) => reg,
            Src::I(bits) => {
                let reg = self.temp();
                self.asm.mov_imm(reg, bits);
                reg
            }
            Src::F(reg) => {
                let bits = self.temp();
                self.asm.float_to_bits(bits, reg);
                bits
            }
        }
    }

    pub(super) fn define(&mut self, value: Val) -> Def {
        let rep = self.func.rep(value);
        if !live::has_location(self.func, value) {
            return Def::None;
        }
        let (alloc, mut moves) = self.mover();
        match rep {
            Rep::None => Def::None,
            Rep::Boxed => {
                let tag = alloc.define(part(value, 1), &mut moves).hw;
                let payload = alloc.define(part(value, 0), &mut moves).hw;
                Def::Two(tag, payload)
            }
            Rep::Float => Def::Float(alloc.define(part(value, 0), &mut moves).hw),
            Rep::Bool => Def::Bool(alloc.define(part(value, 0), &mut moves).hw),
            _ => Def::Gp(alloc.define(part(value, 0), &mut moves).hw),
        }
    }

    /// Reuses `input`'s registers when this node is its last use and both share a representation, so the guard moves nothing; `false` leaves the result to define and convert.
    pub(super) fn same_as_input(&mut self, value: Val, input: Val) -> bool {
        let rep = self.func.rep(value);
        let dies = self.dying.get(self.position).is_some_and(|dying| dying.contains(&input));
        let parts: &[u32] = match rep {
            Rep::Boxed => &[0, 1],
            Rep::Int | Rep::Float | Rep::Bool => &[0],
            _ => return false,
        };
        let unboxed = rep == Rep::Int && self.func.rep(input) == Rep::Boxed;
        let placed = parts.iter().all(|&half| self.alloc.reg_of(part(input, half)).is_some());
        if !dies || (rep != self.func.rep(input) && !unboxed) || self.is_const(input).is_some() || !live::has_location(self.func, value) || !placed {
            return false;
        }
        for &half in parts {
            self.alloc.define_over(part(value, half), part(input, half));
        }
        true
    }

    pub(super) fn out_over(&mut self, value: Val, input: Val) -> u8 {
        let single = matches!(self.func.rep(value), Rep::Int | Rep::Float) && self.func.rep(input) == self.func.rep(value);
        let dies = self.dying.get(self.position).is_some_and(|dying| dying.contains(&input));
        if single
            && dies
            && self.is_const(input).is_none()
            && live::has_location(self.func, value)
            && let Some(reg) = self.alloc.define_over(part(value, 0), part(input, 0))
        {
            return reg.hw;
        }
        self.out(value)
    }

    pub(super) fn out_in(&mut self, value: Val, preferred: u8) -> u8 {
        if self.func.rep(value) == Rep::Int
            && live::has_location(self.func, value)
            && let Some(reg) = self.alloc.define_in(part(value, 0), PReg::gp(preferred))
        {
            return reg.hw;
        }
        self.out(value)
    }

    pub(super) fn out(&mut self, value: Val) -> u8 {
        match self.define(value) {
            Def::Gp(reg) | Def::Bool(reg) | Def::Float(reg) => reg,
            Def::Two(_, payload) => payload,
            Def::None => A::SCRATCH,
        }
    }

    pub(super) fn load_def(&mut self, def: Def, base: u8, disp: i32) {
        use super::asm::Width;
        match def {
            Def::None => {}
            Def::Two(tag, payload) => {
                self.asm.load(Width::Q, tag, base, disp);
                self.asm.load(Width::Q, payload, base, disp + PAYLOAD);
            }
            Def::Float(reg) => self.asm.load_f(reg, base, disp + PAYLOAD),
            Def::Bool(reg) => self.asm.load(Width::B, reg, base, disp + PAYLOAD),
            Def::Gp(reg) => self.asm.load(Width::Q, reg, base, disp + PAYLOAD),
        }
    }

    pub(super) fn convert(&mut self, def: Def, value: Val) {
        let (tag, payload) = self.pair(value);
        match def {
            Def::None => {}
            Def::Two(out_tag, out_payload) => {
                self.src_to(out_tag, tag);
                self.src_to(out_payload, payload);
            }
            Def::Float(reg) => match payload {
                Src::F(src) => self.asm.fmov(reg, src),
                Src::R(src) => self.asm.bits_to_float(reg, src),
                Src::I(bits) => self.asm.fconst(reg, bits as u64),
            },
            Def::Bool(reg) => match payload {
                Src::R(src) if self.func.rep(value) == Rep::Boxed => self.asm.alu_imm(super::asm::Alu::And, reg, src, 0xff),
                other => self.src_to(reg, other),
            },
            Def::Gp(reg) => self.src_to(reg, payload),
        }
    }

    pub(super) fn src_to(&mut self, dst: u8, src: Src) {
        match src {
            Src::R(reg) => self.asm.mov(dst, reg),
            Src::I(bits) => self.asm.mov_imm(dst, bits),
            Src::F(reg) => self.asm.float_to_bits(dst, reg),
        }
    }

    pub(super) fn store_pair(&mut self, (tag, payload): (Src, Src), base: u8, disp: i32) {
        use super::asm::Width;
        match tag {
            Src::R(reg) => self.asm.store(Width::Q, reg, base, disp),
            Src::I(bits) => self.asm.store_imm(Width::Q, base, disp, bits),
            Src::F(reg) => self.asm.store_f(reg, base, disp),
        }
        match payload {
            Src::R(reg) => self.asm.store(Width::Q, reg, base, disp + PAYLOAD),
            Src::I(bits) => self.asm.store_imm(Width::Q, base, disp + PAYLOAD, bits),
            Src::F(reg) => self.asm.store_f(reg, base, disp + PAYLOAD),
        }
    }

    pub(super) fn ctx_load(&mut self, dst: u8, offset: usize) {
        self.asm.load(super::asm::Width::Q, dst, A::CTX, offset as i32);
    }

    pub(super) fn registers(&mut self, dst: u8) {
        self.ctx_load(dst, offset_of!(JitCtx, registers));
    }

    fn base_value(&self) -> Val {
        Val(self.func.insts.len() as u32)
    }

    /// The register base, kept from an earlier node until a call can move the stack.
    pub(super) fn registers_base(&mut self) -> u8 {
        let part = part(self.base_value(), 0);
        if let Some(reg) = self.alloc.reg_of(part) {
            self.alloc.pin(reg);
            return reg.hw;
        }
        let (alloc, mut moves) = self.mover();
        let reg = alloc.define(part, &mut moves);
        alloc.mark_rebuilt(part);
        self.registers(reg.hw);
        reg.hw
    }

    fn place(&mut self, value: Val) -> Place {
        let ty = self.func.ty(value);
        if let Some(constant) = self.is_const(value) {
            return Place::Tagged { tag: constant.tag(), payload: Word::Imm(constant.payload()) };
        }
        let mut broken = false;
        let mut word = |half: u32| match self.alloc.location(part(value, half)) {
            Some(Loc::Reg(reg)) => Word::Reg(reg.index() as u8),
            Some(Loc::Slot(slot)) => Word::Slot(self.spill_at + 8 * slot as i32),
            None => {
                broken = true;
                Word::Imm(0)
            }
        };
        let place = match self.func.rep(value) {
            Rep::Boxed => Place::Boxed { tag: word(1), payload: word(0) },
            Rep::Int => Place::Tagged { tag: INT_TAG, payload: word(0) },
            Rep::Float => Place::Tagged { tag: FLOAT_TAG, payload: word(0) },
            Rep::Bool => Place::Tagged { tag: BOOL_TAG, payload: word(0) },
            _ => Place::Tagged { tag: ty.tag().unwrap_or(NIL_TAG), payload: word(0) },
        };
        self.failed |= broken;
        place
    }

    pub(super) fn exit(&mut self, state: StateId) -> Label {
        let index = self.exit_index(state);
        if let Some(&(label, last)) = self.stubs.last()
            && last == index
        {
            return label;
        }
        let label = self.asm.new_label();
        self.stubs.push((label, index));
        label
    }

    pub(super) fn exit_index(&mut self, state: StateId) -> u32 {
        let mut places = Vec::new();
        self.func.state(state).for_each_operand(|value| places.push(value));
        let places = places.into_iter().map(|value| (value, self.place(value))).collect();
        self.table.add(state.0, places)
    }

    pub(super) fn store_state(&mut self, state: StateId) {
        if self.func.state(state).slots.is_empty() {
            return;
        }
        let index = self.exit_index(state);
        self.store_state_at(index);
    }

    pub(super) fn store_state_at(&mut self, index: u32) {
        self.asm.dump_registers();
        self.runtime_exit_call(self.addresses.store, index);
        self.asm.alu_imm(super::asm::Alu::Add, A::SP, A::SP, DUMP + A::OUTGOING);
    }

    fn runtime_exit_call(&mut self, address: u64, index: u32) {
        self.asm.mov_imm(A::SCRATCH, i64::from(index));
        self.exit_call_from_scratch(address);
    }

    fn exit_call_from_scratch(&mut self, address: u64) {
        let table = std::ptr::from_ref::<ExitTable>(&self.table) as i64;
        let arguments = [Src::R(A::CTX), Src::I(table), Src::R(A::SCRATCH)];
        for (index, &argument) in arguments.iter().enumerate().rev() {
            self.place_arg(index, argument);
        }
        self.arg_address(3, A::OUTGOING);
        self.arg_address(4, A::OUTGOING + DUMP);
        self.asm.call(address);
    }

    pub(super) fn arg_address(&mut self, index: usize, offset: i32) {
        match A::arg(index) {
            super::asm::ArgLoc::Reg(reg) => self.asm.lea(reg, A::SP, offset),
            super::asm::ArgLoc::Stack(at) => {
                self.asm.lea(A::SCRATCH, A::SP, offset);
                self.asm.store(super::asm::Width::Q, A::SCRATCH, A::SP, at);
            }
        }
    }

    /// Only for arguments whose sources no earlier argument overwrites.
    pub(super) fn place_arg(&mut self, index: usize, src: Src) {
        match (A::arg(index), src) {
            (super::asm::ArgLoc::Reg(reg), Src::R(from)) => self.asm.mov(reg, from),
            (super::asm::ArgLoc::Reg(reg), Src::I(bits)) => self.asm.mov_imm(reg, bits),
            (super::asm::ArgLoc::Reg(reg), Src::F(from)) => self.asm.float_to_bits(reg, from),
            (super::asm::ArgLoc::Stack(at), Src::R(from)) => self.asm.store(super::asm::Width::Q, from, A::SP, at),
            (super::asm::ArgLoc::Stack(at), Src::I(bits)) => self.asm.store_imm(super::asm::Width::Q, A::SP, at, bits),
            (super::asm::ArgLoc::Stack(at), Src::F(from)) => self.asm.store_f(from, A::SP, at),
        }
    }

    fn tails(&mut self) {
        for stub in std::mem::take(&mut self.safepoints) {
            self.safepoint_tail(stub);
        }
        for (label, index) in std::mem::take(&mut self.stubs) {
            self.asm.bind(label);
            self.asm.mov_imm(A::SCRATCH, i64::from(index));
            self.asm.jump(self.exit_tail);
        }
        self.asm.bind(self.exit_tail);
        self.asm.dump_registers();
        self.exit_call_from_scratch(self.addresses.exit);
        self.asm.jump(self.epilogue);
        self.asm.bind(self.exit_with);
        self.exit_with_tail();
    }

    /// Follows `exit_with` in `lower.rs`.
    fn exit_with_tail(&mut self) {
        let leaving = self.asm.new_label();
        self.asm.cmp_imm(A::SCRATCH, 0);
        self.asm.branch(Cond::Lt, leaving);
        self.place_arg(1, Src::R(A::SCRATCH));
        self.place_arg(0, Src::R(A::CTX));
        self.asm.call(self.addresses.resume);
        self.asm.mov(A::SCRATCH, A::result());
        self.asm.bind(leaving);
        self.asm.neg(A::SCRATCH, A::SCRATCH);
        self.asm.jump(self.epilogue);
    }

    /// The values this node's frame states name, which an exit after a call reads from their registers.
    fn state_values(&self, value: Val) -> Vec<Val> {
        let mut values = Vec::new();
        if value.0 == u32::MAX {
            return values;
        }
        live::for_each_use(self.func, value, |used| values.push(used));
        values
    }

    pub(super) fn saves(&mut self, after: &[Val]) -> Vec<(PReg, u32, bool)> {
        self.alloc.release(self.base_value());
        let value = self.current;
        let needed = self.state_values(value);
        let dying = self.dying.get(self.position).cloned().unwrap_or_default();
        self.alloc.across_call(|part| {
            let held = value_of(part);
            held != value && (!dying.contains(&held) || needed.contains(&held) || after.contains(&held))
        })
    }

    pub(super) fn save(&mut self, saves: &[(PReg, u32, bool)]) {
        let (_, mut moves) = self.mover();
        for &(reg, slot, stored) in saves {
            if !stored {
                moves.spill(slot, reg);
            }
        }
    }

    pub(super) fn restore(&mut self, saves: &[(PReg, u32, bool)]) {
        let (_, mut moves) = self.mover();
        for &(reg, slot, _) in saves {
            moves.reload(reg, slot);
        }
    }

    pub(super) fn call_c(&mut self, address: u64, args: &[Src]) {
        let mut wants = Vec::new();
        let mut floats = 0;
        for (index, &arg) in args.iter().enumerate() {
            match (arg, A::arg(index)) {
                (Src::F(_), _) => {
                    let reg = A::float_arg(index, floats);
                    floats += 1;
                    wants.push((PReg::fp(reg), arg));
                }
                (_, super::asm::ArgLoc::Reg(reg)) => wants.push((PReg::gp(reg), arg)),
                (_, super::asm::ArgLoc::Stack(_)) => self.place_arg(index, arg),
            }
        }
        let pairs: Vec<(Want, Have)> = wants
            .into_iter()
            .map(|(reg, arg)| {
                let have = match arg {
                    Src::R(from) => Have { reg: Some(PReg::gp(from)), ..Have::default() },
                    Src::F(from) => Have { reg: Some(PReg::fp(from)), ..Have::default() },
                    Src::I(bits) => Have { constant: Some(bits as u64), ..Have::default() },
                };
                (Want { class: reg.class, reg: Some(reg), slot: None }, have)
            })
            .collect();
        let (_, mut moves) = self.mover();
        alloc::resolve(&pairs, &mut moves);
        self.asm.call(address);
    }

    /// Calls the `value` helper of `ssa/runtime.rs`; the answer lands in `SCRATCH`.
    pub(super) fn call_value(&mut self, kind: u32, operands: &[(Src, Src)], extra: u32, saves: &[(PReg, u32, bool)]) {
        for (index, &pair) in operands.iter().enumerate() {
            self.store_pair(pair, A::SP, self.scratch_at + index as i32 * VALUE_SIZE);
        }
        self.save(saves);
        self.place_arg(0, Src::R(A::CTX));
        self.place_arg(1, Src::I(i64::from(kind)));
        self.arg_address(2, self.scratch_at);
        self.place_arg(3, Src::I(i64::from(extra as i32)));
        self.asm.call(self.addresses.value);
        self.asm.mov(A::SCRATCH, A::result());
        self.restore(saves);
    }

    pub(super) fn scratch_load(&mut self, index: i32, tag: u8, payload: u8) {
        let at = self.scratch_at + index * VALUE_SIZE;
        self.asm.load(super::asm::Width::Q, tag, A::SP, at);
        self.asm.load(super::asm::Width::Q, payload, A::SP, at + PAYLOAD);
    }

    pub(super) fn may_collect(&self, value: Val) -> bool {
        let ty = self.func.ty(value);
        (STRING_TAG..LIGHT_USERDATA_TAG).any(|tag| !ty.intersect(Type::of_tag(tag)).is_none())
    }

    fn term(&mut self, term: &Term) {
        match term {
            Term::Open => self.asm.trap(),
            Term::Jump(edge) => self.jump(edge),
            Term::Branch { cond, then, other } => self.branch(*cond, then, other),
            Term::Switch { code, cases } => self.switch(*code, cases),
            Term::Return { pc, first, count, close } => self.ret(*pc, *first, *count, *close),
            Term::TailCall { pc, a, b, k } => self.tail_call(*pc, (*a, *b, *k)),
            Term::Deopt { state } => {
                let exit = self.exit(*state);
                self.asm.jump(exit);
            }
        }
    }

    fn jump(&mut self, edge: &Edge) {
        self.edge_moves(edge);
        if self.next_block != Some(edge.block) {
            let label = self.label(edge.block);
            self.asm.jump(label);
        }
    }

    /// The edge without moves is the one taken by the branch, so the other falls through into its moves.
    fn branch(&mut self, cond: Val, then: &Edge, other: &Edge) {
        let then_moves = self.needs_moves(then);
        let other_moves = self.needs_moves(other);
        if !then_moves {
            let label = self.label(then.block);
            self.branch_on(cond, false, label);
            self.jump(other);
            return;
        }
        if !other_moves {
            let label = self.label(other.block);
            self.branch_on(cond, true, label);
            self.jump(then);
            return;
        }
        let stub = self.asm.new_label();
        self.branch_on(cond, false, stub);
        self.edge_moves(other);
        let label = self.label(other.block);
        self.asm.jump(label);
        self.asm.bind(stub);
        self.jump(then);
    }

    fn branch_on(&mut self, cond: Val, negate: bool, target: Label) {
        if self.live.fused[cond.0 as usize] {
            self.compare_branch(cond, negate, target);
            return;
        }
        let flag = self.reg(cond);
        self.asm.cmp_imm(flag, 0);
        self.asm.branch(if negate { Cond::Eq } else { Cond::Ne }, target);
    }

    /// Every case is compared before any edge moves a register, since the moves may overwrite the code.
    fn switch(&mut self, code: Val, cases: &[(u32, Edge)]) {
        let result = self.reg(code);
        let stubs: Vec<Label> = cases.iter().map(|_| self.asm.new_label()).collect();
        for ((pc, _), &stub) in cases.iter().zip(&stubs) {
            self.asm.cmp_imm(result, *pc as i32);
            self.asm.branch(Cond::Eq, stub);
        }
        self.asm.mov(A::SCRATCH, result);
        self.asm.jump(self.exit_with);
        let settled = self.settled.take();
        for ((pc, edge), stub) in cases.iter().zip(stubs) {
            self.asm.bind(stub);
            if let Some((_, label)) = settled.filter(|&(at, _)| at == *pc) {
                self.asm.bind(label);
            }
            self.edge_moves(edge);
            let label = self.label(edge.block);
            self.asm.jump(label);
        }
    }

    /// Fixes the target's entry state the first time an edge reaches it.
    fn target_entries(&mut self, edge: &Edge) -> Vec<Entry> {
        let target = edge.block;
        if let Some(entries) = &self.snapshots[target.0 as usize] {
            return entries.clone();
        }
        let mut live = Vec::new();
        for value in self.live.live_in(target) {
            live.extend(parts_of(self.func, value));
        }
        let mut params = Vec::new();
        for (&param, &arg) in self.func.block(target).params.iter().zip(&edge.args) {
            if !live::has_location(self.func, param) {
                continue;
            }
            for &half in parts(self.func, param) {
                let preferred = if live::has_location(self.func, arg) && !self.live.is_live_in(target, arg) {
                    self.alloc.reg_of(part(arg, half))
                } else {
                    None
                };
                params.push((part(param, half), preferred));
            }
        }
        let used = self.used_in_loop(target);
        let spilled = |part: Part| used.as_ref().is_some_and(|(used, _)| !used[value_of(part).0 as usize]);
        let hot = |part: Part| used.as_ref().is_some_and(|(_, hot)| hot[value_of(part).0 as usize]);
        let entries = self.alloc.fix(&live, &params, spilled, hot);
        self.snapshots[target.0 as usize] = Some(entries.clone());
        entries
    }

    /// For a loop header, the values its loop reads as operands, and those read on every iteration; the rest are only
    /// carried for frame states.
    fn used_in_loop(&mut self, header: BlockId) -> Option<(Vec<bool>, Vec<bool>)> {
        let looped = self.func.loops.iter().find(|candidate| candidate.header == header)?;
        let mut inside = vec![false; self.func.blocks.len()];
        for &block in &looped.blocks {
            inside[block.0 as usize] = true;
        }
        let idom = self.idom.get_or_insert_with(|| self.func.dominators());
        let every = crate::jit::ssa::passes::hoist::blocks_on_every_iteration(self.func, looped, &inside, idom);
        let mut used = vec![false; self.func.insts.len()];
        let mut hot = vec![false; self.func.insts.len()];
        for &block in &looped.blocks {
            let data = self.func.block(block);
            let mut mark = |operand: Val| {
                used[operand.0 as usize] = true;
                hot[operand.0 as usize] |= every[block.0 as usize];
            };
            for &value in &data.insts {
                let op = self.func.op(value);
                op.for_each_operand(&mut mark);
                if let Some(payload) = live::payload_use(self.func, op) {
                    mark(payload);
                }
            }
            match &data.term {
                Term::Branch { cond, .. } => mark(*cond),
                Term::Switch { code, .. } => mark(*code),
                _ => {}
            }
        }
        Some((used, hot))
    }

    fn edge_pairs(&mut self, edge: &Edge) -> Vec<(Want, Have)> {
        let entries = self.target_entries(edge);
        let params = &self.func.block(edge.block).params;
        let mut pairs = Vec::with_capacity(entries.len());
        for entry in &entries {
            let value = value_of(entry.part);
            let half = entry.part % 2;
            let have = match params.iter().position(|&param| param == value) {
                Some(index) => self.arg_have(edge.args[index], half, value),
                None => self.alloc.have(entry.part),
            };
            if have.reg.is_none() && have.slot.is_none() && have.constant.is_none() {
                debug_assert!(false, "an edge's value is somewhere");
                self.failed = true;
                continue;
            }
            let want = self.alloc.want(entry);
            pairs.push((want, have));
        }
        pairs
    }

    fn arg_have(&mut self, arg: Val, half: u32, param: Val) -> Have {
        if let Some(constant) = self.is_const(arg) {
            let bits = if half == 1 { u64::from(constant.tag()) } else { constant.payload() };
            return Have { constant: Some(bits), ..Have::default() };
        }
        if self.func.rep(arg) != self.func.rep(param) {
            self.failed = true;
        }
        self.alloc.have(part(arg, half))
    }

    fn needs_moves(&mut self, edge: &Edge) -> bool {
        let pairs = self.edge_pairs(edge);
        pairs.iter().any(|(want, have)| (want.reg.is_some() && want.reg != have.reg) || (want.slot.is_some() && want.slot != have.slot))
    }

    fn edge_moves(&mut self, edge: &Edge) {
        let pairs = self.edge_pairs(edge);
        let (_, mut moves) = self.mover();
        alloc::resolve(&pairs, &mut moves);
    }
}

pub(super) fn parts_of(func: &Func, value: Val) -> impl Iterator<Item = Part> + '_ {
    parts(func, value).iter().map(move |&half| part(value, half))
}

fn has_loop(func: &Func) -> bool {
    func.blocks.iter().enumerate().any(|(index, block)| block.term.successors().iter().any(|target| target.0 as usize <= index))
}
