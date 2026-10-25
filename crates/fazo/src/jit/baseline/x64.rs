//! x86-64 encoder for the Baseline templates, over asmkit. Holes are assembled with placeholders that force their width, and are found again by their bytes.

use std::vec::Vec;

use asmkit::{Arch, CodeBuffer, Environment, Label, x86::*};

use super::masm::{Alu, Cond, Division, FAlu, FCond, FReg, Field, Hole, Imm, Masm, Reg, Stencil, Target, Width};

const CTX: u32 = 3;
const BASE: u32 = 12;
const KONST: u32 = 13;
const SCRATCH: u32 = 11;
const RSP: u32 = 4;

#[cfg(windows)]
const ARGS: [u32; 4] = [1, 2, 8, 9];
#[cfg(not(windows))]
const ARGS: [u32; 5] = [7, 6, 2, 1, 8];

/// Win64 passes the fifth integer argument on the stack, above the 32-byte shadow space.
const STACK_ARG: i32 = 32;

pub(crate) fn register(reg: Reg) -> u32 {
    match reg {
        Reg::Ctx => CTX,
        Reg::Base => BASE,
        Reg::Konst => KONST,
        Reg::T0 | Reg::Ret => 0,
        Reg::T1 => 1,
        Reg::T2 => 2,
        Reg::T3 => 8,
        Reg::T4 => 9,
        Reg::T5 => 10,
    }
}

fn gp(reg: Reg) -> Gpq {
    gpq(register(reg))
}

fn fp(reg: FReg) -> Xmm {
    match reg {
        FReg::F0 => xmm(0),
        FReg::F1 => xmm(1),
        FReg::F2 => xmm(2),
        FReg::F3 => xmm(3),
    }
}

pub(crate) struct X64 {
    buffer: CodeBuffer,
    labels: Vec<Label>,
    zero: Label,
    holes: Vec<Hole>,
    next_sentinel: u32,
}

impl X64 {
    pub(crate) fn new() -> Self {
        let mut buffer = CodeBuffer::new(Environment::new(Arch::X64));
        let zero = buffer.get_label();
        buffer.bind_label(zero);
        Self {
            buffer,
            labels: Vec::new(),
            zero,
            holes: Vec::new(),
            next_sentinel: 0,
        }
    }

    fn asm(&mut self) -> Assembler<'_> {
        Assembler::new(&mut self.buffer)
    }

    fn offset(&self) -> u32 {
        self.buffer.cur_offset()
    }

    /// A placeholder that forces the 32-bit form; no other operand of the instruction holds it.
    fn sentinel32(&mut self) -> i32 {
        self.next_sentinel += 1;
        0x2d00_005a | ((self.next_sentinel as i32 & 0xffff) << 8)
    }

    fn sentinel64(&mut self) -> i64 {
        self.next_sentinel += 1;
        0x2c3b_4a59_0000_0000 | i64::from(self.next_sentinel & 0xffff) << 8 | 0x5a
    }

    fn locate(&mut self, start: u32, bytes: &[u8], field: Field, slot: u8, add: i32) {
        let end = self.offset() as usize;
        let code = &self.buffer.data()[start as usize..end];
        let found = code.windows(bytes.len()).position(|window| window == bytes).expect("the placeholder is in the instruction");
        debug_assert!(code.windows(bytes.len()).filter(|window| *window == bytes).count() == 1);
        self.holes.push(Hole {
            offset: start + found as u32,
            field,
            slot,
            add,
        });
    }

    fn memory(&mut self, size: u32, base: Reg, disp: Imm) -> (Mem, Option<(i32, u8, i32)>) {
        let (value, hole) = match disp {
            Imm::Fixed(value) => (i32::try_from(value).expect("a fixed displacement fits 32 bits"), None),
            Imm::Hole { slot, add } => {
                let sentinel = self.sentinel32();
                (sentinel, Some((sentinel, slot, add)))
            }
        };
        let base = gp(base);
        let mem = match size {
            1 => byte_ptr(base, value),
            4 => dword_ptr(base, value),
            _ => qword_ptr(base, value),
        };
        (mem, hole)
    }

    fn finish_memory(&mut self, start: u32, hole: Option<(i32, u8, i32)>) {
        if let Some((sentinel, slot, add)) = hole {
            self.locate(start, &sentinel.to_le_bytes(), Field::Le32, slot, add);
        }
    }

    fn label(&self, target: Target) -> Label {
        match target {
            Target::Local(index) => self.labels[index as usize],
            Target::Ext(_) => self.zero,
        }
    }

    fn record_branch(&mut self, target: Target) {
        if let Target::Ext(id) = target {
            let offset = self.offset() - 4;
            self.holes.push(Hole {
                offset,
                field: Field::Rel32,
                slot: id,
                add: 0,
            });
        }
    }

    fn jcc(&mut self, cond: Cond, target: Target) {
        let label = self.label(target);
        let mut asm = self.asm();
        let asm = asm.long();
        match cond {
            Cond::Eq => asm.jz(label),
            Cond::Ne => asm.jnz(label),
            Cond::Lt => asm.jl(label),
            Cond::Le => asm.jle(label),
            Cond::Gt => asm.jnle(label),
            Cond::Ge => asm.jnl(label),
            Cond::BelowEq => asm.jbe(label),
            Cond::Above => asm.jnbe(label),
            Cond::AboveEq => asm.jnb(label),
        }
        self.record_branch(target);
    }

    fn set_arg_value(&mut self, index: u8, value: Imm) {
        let start = self.offset();
        let sentinel = self.sentinel32();
        let placed = match value {
            Imm::Fixed(value) => value as i32,
            Imm::Hole { .. } => sentinel,
        };
        if cfg!(windows) && usize::from(index) >= ARGS.len() {
            self.asm().mov(dword_ptr(gpq(RSP), STACK_ARG), imm(placed));
        } else {
            self.asm().long().mov(gpq(ARGS[usize::from(index)]).r32(), imm(placed));
        }
        if let Imm::Hole { slot, add } = value {
            self.locate(start, &sentinel.to_le_bytes(), Field::Le32, slot, add);
        }
    }
}

impl Masm for X64 {
    fn new_label(&mut self) -> Target {
        let label = self.buffer.get_label();
        self.labels.push(label);
        Target::Local(self.labels.len() as u32 - 1)
    }

    fn bind(&mut self, label: Target) {
        let label = self.label(label);
        self.buffer.bind_label(label);
    }

    fn load(&mut self, width: Width, dst: Reg, base: Reg, disp: Imm) {
        let start = self.offset();
        let (mem, hole) = self.memory(width.bytes(), base, disp);
        let dst = gp(dst);
        match width {
            Width::U8 => self.asm().movzx(dst.r32(), mem),
            Width::U32 => self.asm().mov(dst.r32(), mem),
            Width::U64 => self.asm().mov(dst, mem),
        }
        self.finish_memory(start, hole);
    }

    fn store(&mut self, width: Width, src: Reg, base: Reg, disp: Imm) {
        let start = self.offset();
        let (mem, hole) = self.memory(width.bytes(), base, disp);
        let src = gp(src);
        match width {
            Width::U8 => self.asm().mov(mem, src.r8()),
            Width::U32 => self.asm().mov(mem, src.r32()),
            Width::U64 => self.asm().mov(mem, src),
        }
        self.finish_memory(start, hole);
    }

    fn store_imm(&mut self, width: Width, base: Reg, disp: Imm, value: i32) {
        let start = self.offset();
        let (mem, hole) = self.memory(width.bytes(), base, disp);
        match width {
            Width::U8 => self.asm().mov(mem, imm(value as i8)),
            Width::U32 | Width::U64 => self.asm().mov(mem, imm(value)),
        }
        self.finish_memory(start, hole);
    }

    fn load_f(&mut self, dst: FReg, base: Reg, disp: Imm) {
        let start = self.offset();
        let (mem, hole) = self.memory(8, base, disp);
        self.asm().movsd(fp(dst), mem);
        self.finish_memory(start, hole);
    }

    fn store_f(&mut self, src: FReg, base: Reg, disp: Imm) {
        let start = self.offset();
        let (mem, hole) = self.memory(8, base, disp);
        self.asm().movsd(mem, fp(src));
        self.finish_memory(start, hole);
    }

    fn mov(&mut self, dst: Reg, src: Reg) {
        self.asm().mov(gp(dst), gp(src));
    }

    fn mov_imm(&mut self, dst: Reg, value: Imm) {
        let start = self.offset();
        match value {
            Imm::Fixed(value) => self.asm().mov(gp(dst), imm(value)),
            Imm::Hole { slot, add } => {
                let sentinel = self.sentinel64();
                self.asm().movabs(gp(dst), imm(sentinel));
                self.locate(start, &sentinel.to_le_bytes(), Field::Le64, slot, add);
            }
        }
    }

    fn alu(&mut self, op: Alu, dst: Reg, src: Reg) {
        let (dst, src) = (gp(dst), gp(src));
        let mut asm = self.asm();
        match op {
            Alu::Add => asm.add(dst, src),
            Alu::Sub => asm.sub(dst, src),
            Alu::Mul => asm.imul(dst, src),
            Alu::And => asm.and_(dst, src),
            Alu::Or => asm.or_(dst, src),
            Alu::Xor => asm.xor_(dst, src),
        }
    }

    fn alu_imm(&mut self, op: Alu, dst: Reg, value: i32) {
        let dst = gp(dst);
        let mut asm = self.asm();
        match op {
            Alu::Add => asm.add(dst, imm(value)),
            Alu::Sub => asm.sub(dst, imm(value)),
            Alu::And => asm.and_(dst, imm(value)),
            Alu::Or => asm.or_(dst, imm(value)),
            Alu::Xor => asm.xor_(dst, imm(value)),
            Alu::Mul => asm.imul_3(dst, dst, imm(value)),
        }
    }

    fn neg(&mut self, dst: Reg) {
        self.asm().neg(gp(dst));
    }

    fn not(&mut self, dst: Reg) {
        self.asm().not_(gp(dst));
    }

    fn shl_imm(&mut self, dst: Reg, amount: u8) {
        self.asm().shl(gp(dst), imm(amount as i32));
    }

    fn floor_div(&mut self, op: Division, dst: Reg, x: Reg, y: Reg) {
        debug_assert!(![Reg::T0, Reg::T1, Reg::T2].contains(&x) && ![Reg::T0, Reg::T1, Reg::T2].contains(&y));
        let done = self.new_label();
        let rax = gpq(0);
        let rcx = gpq(1);
        let rdx = gpq(2);
        self.asm().mov(rax, gp(x));
        self.asm().cqo(rdx, rax);
        self.asm().idiv_3(*rdx, *rax, *gp(y));
        self.asm().test(rdx, rdx);
        self.jcc(Cond::Eq, done);
        match op {
            Division::Mod => self.asm().mov(rcx, rdx),
            Division::IDiv => self.asm().mov(rcx, gp(x)),
        }
        self.asm().xor_(rcx, gp(y));
        let label = self.label(done);
        self.asm().long().jns(label);
        match op {
            Division::Mod => self.asm().add(rdx, gp(y)),
            Division::IDiv => self.asm().sub(rax, imm(1)),
        }
        self.bind(done);
        match op {
            Division::Mod => self.asm().mov(gp(dst), rdx),
            Division::IDiv => self.asm().mov(gp(dst), rax),
        }
    }

    fn cmp(&mut self, left: Reg, right: Reg) {
        self.asm().cmp(gp(left), gp(right));
    }

    fn cmp_imm(&mut self, left: Reg, value: i32) {
        self.asm().cmp(gp(left), imm(value));
    }

    fn branch(&mut self, cond: Cond, target: Target) {
        self.jcc(cond, target);
    }

    fn jump(&mut self, target: Target) {
        let label = self.label(target);
        self.asm().long().jmp(label);
        self.record_branch(target);
    }


    fn fop(&mut self, op: FAlu, dst: FReg, src: FReg) {
        let (dst, src) = (fp(dst), fp(src));
        let mut asm = self.asm();
        match op {
            FAlu::Add => asm.addsd(dst, src),
            FAlu::Sub => asm.subsd(dst, src),
            FAlu::Mul => asm.mulsd(dst, src),
            FAlu::Div => asm.divsd(dst, src),
        }
    }

    /// `UCOMISD` sets PF on unordered pairs, so `a < b` is tested as `b > a` and equality also checks PF clear.
    fn fbranch(&mut self, cond: FCond, left: FReg, right: FReg, target: Target) {
        match cond {
            FCond::Lt => {
                self.asm().ucomisd(fp(right), fp(left));
                self.jcc(Cond::Above, target);
            }
            FCond::Le => {
                self.asm().ucomisd(fp(right), fp(left));
                self.jcc(Cond::AboveEq, target);
            }
            FCond::Eq => {
                let unordered = self.new_label();
                self.asm().ucomisd(fp(left), fp(right));
                let label = self.label(unordered);
                self.asm().long().jp(label);
                self.jcc(Cond::Eq, target);
                self.bind(unordered);
            }
        }
    }

    fn int_to_float(&mut self, dst: FReg, src: Reg) {
        self.asm().cvtsi2sd(fp(dst), gp(src));
    }

    fn call(&mut self, address: u64) {
        let scratch = gpq(SCRATCH);
        self.asm().mov(scratch, imm(address as i64));
        self.asm().call(scratch);
    }

    fn arg_imm(&mut self, index: u8, value: Imm) {
        self.set_arg_value(index, value);
    }

    fn arg_reg(&mut self, index: u8, src: Reg) {
        if cfg!(windows) && usize::from(index) >= ARGS.len() {
            self.asm().mov(qword_ptr(gpq(RSP), STACK_ARG), gp(src));
            return;
        }
        self.asm().mov(gpq(ARGS[usize::from(index)]), gp(src));
    }

    fn call_reg(&mut self, target: Reg) {
        self.asm().call(gp(target));
    }

    fn stack_address(&mut self, dst: Reg, offset: i32) {
        self.asm().lea(gp(dst), qword_ptr(gpq(RSP), offset));
    }

    fn callee_context(&self) -> i32 {
        frame::CONTEXT
    }

    fn finish(mut self: Box<Self>) -> Stencil {
        let holes = std::mem::take(&mut self.holes);
        let code = self.buffer.finish().expect("a template assembles");
        Stencil {
            bytes: code.data().to_vec(),
            holes,
        }
    }
}

/// The prologue, epilogue and dispatch of a Baseline function, which the unwind tables describe.
pub(crate) mod frame {
    use super::*;

    pub(crate) const SAVED: [u32; 4] = [CTX, BASE, KONST, 14];
    /// Keeps `rsp` a multiple of 16 at every call.
    pub(crate) fn reserved(context: usize) -> u32 {
        (CONTEXT as u32 + context as u32).next_multiple_of(16)
    }

    pub(crate) const CONTEXT: i32 = 48;

    pub(crate) struct Prologue {
        pub(crate) bytes: Vec<u8>,
        pub(crate) push_rbp: u8,
        pub(crate) set_rbp: u8,
        pub(crate) pushes: [u8; 4],
        pub(crate) alloc: u8,
    }

    pub(crate) fn prologue(reserved: u32, registers: i32, constants: i32) -> Prologue {
        let mut x = X64::new();
        let rbp = gpq(5);
        let rsp = gpq(RSP);
        x.asm().push(rbp);
        let push_rbp = x.offset() as u8;
        x.asm().mov(rbp, rsp);
        let set_rbp = x.offset() as u8;
        let mut pushes = [0u8; 4];
        for (index, &register) in SAVED.iter().enumerate() {
            x.asm().push(gpq(register));
            pushes[index] = x.offset() as u8;
        }
        x.asm().sub(rsp, imm(reserved as i32));
        let alloc = x.offset() as u8;
        x.asm().mov(gpq(CTX), gpq(ARGS[0]));
        x.asm().mov(gpq(BASE), qword_ptr(gpq(CTX), registers));
        x.asm().mov(gpq(KONST), qword_ptr(gpq(CTX), constants));
        x.asm().mov(gpq(0).r32(), gpq(ARGS[1]).r32());
        let bytes = Box::new(x).finish().bytes;
        Prologue {
            bytes,
            push_rbp,
            set_rbp,
            pushes,
            alloc,
        }
    }

    pub(crate) fn epilogue(reserved: u32) -> Vec<u8> {
        let mut x = X64::new();
        x.asm().add(gpq(RSP), imm(reserved as i32));
        for &register in SAVED.iter().rev() {
            x.asm().pop(gpq(register));
        }
        x.asm().pop(gpq(5));
        x.asm().ret();
        Box::new(x).finish().bytes
    }

    pub(crate) fn dispatch(exit: u8, table: u8) -> Stencil {
        let mut x = Box::new(X64::new());
        let rax = gpq(0);
        let rcx = gpq(1);
        let rdx = gpq(2);
        let trap = x.new_label();
        x.asm().test(rax, rax);
        x.jcc(Cond::Lt, Target::Ext(exit));
        let start = x.offset();
        let sentinel = x.sentinel32();
        x.asm().cmp(rax, imm(sentinel));
        x.locate(start, &sentinel.to_le_bytes(), Field::Le32, super::super::masm::slot::I, 0);
        x.jcc(Cond::AboveEq, trap);
        let zero = x.zero;
        x.asm().lea(rcx, asmkit::x86::label_ptr(zero, 0, 8));
        x.holes.push(Hole {
            offset: x.offset() - 4,
            field: Field::Rel32,
            slot: table,
            add: 0,
        });
        x.asm().movsxd(rdx, asmkit::x86::ptr_index(rcx, rax, 2, 0, 4));
        x.asm().add(rcx, rdx);
        x.asm().jmp(rcx);
        x.bind(trap);
        x.asm().ud2();
        x.finish()
    }

    pub(crate) fn negate_exit() -> Vec<u8> {
        let mut x = X64::new();
        x.asm().neg(gpq(0));
        Box::new(x).finish().bytes
    }
}
