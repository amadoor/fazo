//! AArch64 encoder for the Baseline templates, over asmkit. Holes are found again by their instruction word's offset.

use std::vec::Vec;

use asmkit::{Arch, CodeBuffer, Environment, Label, aarch64::*, imm};

type VReg = asmkit::aarch64::Vec;

use super::masm::{Alu, Cond, Division, FAlu, FCond, FReg, Field, Hole, Imm, Masm, Reg, Stencil, Target, Width};

const CTX: u32 = 19;
const BASE: u32 = 20;
const KONST: u32 = 21;
const SCRATCH: u32 = 16;
const SCRATCH2: u32 = 17;

pub(crate) fn register(reg: Reg) -> u32 {
    match reg {
        Reg::Ctx => CTX,
        Reg::Base => BASE,
        Reg::Konst => KONST,
        Reg::Ret => 0,
        Reg::T0 => 9,
        Reg::T1 => 10,
        Reg::T2 => 11,
        Reg::T3 => 12,
        Reg::T4 => 13,
        Reg::T5 => 14,
    }
}

fn gp(reg: Reg) -> Gp {
    x(register(reg))
}

fn fp(reg: FReg) -> VReg {
    match reg {
        FReg::F0 => d(0),
        FReg::F1 => d(1),
        FReg::F2 => d(2),
        FReg::F3 => d(3),
    }
}

pub(crate) struct A64 {
    buffer: CodeBuffer,
    labels: Vec<Label>,
    zero: Label,
    holes: Vec<Hole>,
}

impl A64 {
    pub(crate) fn new() -> Self {
        let mut buffer = CodeBuffer::new(Environment::new(Arch::AArch64));
        let zero = buffer.get_label();
        buffer.bind_label(zero);
        Self {
            buffer,
            labels: Vec::new(),
            zero,
            holes: Vec::new(),
        }
    }

    fn asm(&mut self) -> Assembler<'_> {
        Assembler::new(&mut self.buffer)
    }

    fn offset(&self) -> u32 {
        self.buffer.cur_offset()
    }

    fn hole_here(&mut self, field: Field, slot: u8, add: i32) {
        let offset = self.offset() - 4;
        self.holes.push(Hole { offset, field, slot, add });
    }

    fn label(&self, target: Target) -> Label {
        match target {
            Target::Local(index) => self.labels[index as usize],
            Target::Ext(_) => self.zero,
        }
    }

    fn record_branch(&mut self, target: Target, field: Field) {
        if let Target::Ext(id) = target {
            self.hole_here(field, id, 0);
        }
    }

    /// `x16` holds displacements that neither the scaled nor the unscaled form can encode.
    fn fixed_memory(&mut self, size: u32, base: Reg, disp: i64) -> Mem {
        let scaled = disp >= 0 && disp % i64::from(size) == 0 && disp / i64::from(size) < 4096;
        if scaled || (-256..256).contains(&disp) {
            return ptr(gp(base), disp as i32);
        }
        self.asm().mov(x(SCRATCH), imm(disp));
        asmkit::aarch64::ptr_reg(gp(base), x(SCRATCH))
    }

    fn access(&mut self, size: u32, base: Reg, disp: Imm, access: impl FnOnce(&mut Self, Mem)) {
        match disp {
            Imm::Fixed(value) => {
                let mem = self.fixed_memory(size, base, value);
                access(self, mem);
            }
            Imm::Hole { slot, add } => {
                let placeholder = (0x5a5 * size) as i32;
                access(self, ptr(gp(base), placeholder));
                self.hole_here(Field::Imm12 { scale: size.trailing_zeros() as u8 }, slot, add);
            }
        }
    }

    fn mov_hole(&mut self, dst: Gp, slot: u8, add: i32) {
        let start = self.buffer.cur_offset();
        self.asm().patchable_mov(dst, 0x5a5a_5a5a_5a5a_5a5a_u64 & if dst.is_gp64() { u64::MAX } else { u64::from(u32::MAX) });
        let end = self.buffer.cur_offset();
        for (piece, at) in (start..end).step_by(4).enumerate() {
            self.holes.push(Hole { offset: at, field: Field::Imm16 { shift: piece as u8 * 16 }, slot, add });
        }
    }

    fn b_cond(&mut self, cond: Cond, label: Label) {
        let mut asm = self.asm();
        match cond {
            Cond::Eq => asm.b_eq(label),
            Cond::Ne => asm.b_ne(label),
            Cond::Lt => asm.b_lt(label),
            Cond::Le => asm.b_le(label),
            Cond::Gt => asm.b_gt(label),
            Cond::Ge => asm.b_ge(label),
            Cond::BelowEq => asm.b_ls(label),
            Cond::Above => asm.b_hi(label),
            Cond::AboveEq => asm.b_hs(label),
        }
    }
}

impl Masm for A64 {
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
        let register = register(dst);
        self.access(width.bytes(), base, disp, |a, mem| match width {
            Width::U8 => a.asm().ldrb(w(register), mem),
            Width::U32 => a.asm().ldr(w(register), mem),
            Width::U64 => a.asm().ldr(x(register), mem),
        });
    }

    fn store(&mut self, width: Width, src: Reg, base: Reg, disp: Imm) {
        let register = register(src);
        self.access(width.bytes(), base, disp, |a, mem| match width {
            Width::U8 => a.asm().strb(w(register), mem),
            Width::U32 => a.asm().str(w(register), mem),
            Width::U64 => a.asm().str(x(register), mem),
        });
    }

    fn store_imm(&mut self, width: Width, base: Reg, disp: Imm, value: i32) {
        self.asm().mov(x(SCRATCH2), imm(i64::from(value)));
        self.access(width.bytes(), base, disp, |a, mem| match width {
            Width::U8 => a.asm().strb(w(SCRATCH2), mem),
            Width::U32 => a.asm().str(w(SCRATCH2), mem),
            Width::U64 => a.asm().str(x(SCRATCH2), mem),
        });
    }

    fn load_f(&mut self, dst: FReg, base: Reg, disp: Imm) {
        self.access(8, base, disp, |a, mem| a.asm().ldr(fp(dst), mem));
    }

    fn store_f(&mut self, src: FReg, base: Reg, disp: Imm) {
        self.access(8, base, disp, |a, mem| a.asm().str(fp(src), mem));
    }

    fn mov(&mut self, dst: Reg, src: Reg) {
        self.asm().mov(gp(dst), gp(src));
    }

    fn mov_imm(&mut self, dst: Reg, value: Imm) {
        match value {
            Imm::Fixed(value) => self.asm().mov(gp(dst), imm(value)),
            Imm::Hole { slot, add } => self.mov_hole(gp(dst), slot, add),
        }
    }

    fn alu(&mut self, op: Alu, dst: Reg, src: Reg) {
        let (dst, src) = (gp(dst), gp(src));
        let mut asm = self.asm();
        match op {
            Alu::Add => asm.add(dst, dst, src),
            Alu::Sub => asm.sub(dst, dst, src),
            Alu::Mul => asm.mul(dst, dst, src),
            Alu::And => asm.and_(dst, dst, src),
            Alu::Or => asm.orr(dst, dst, src),
            Alu::Xor => asm.eor(dst, dst, src),
        }
    }

    fn alu_imm(&mut self, op: Alu, dst: Reg, value: i32) {
        let dst = gp(dst);
        match op {
            Alu::Add => self.asm().add(dst, dst, imm(value)),
            Alu::Sub => self.asm().sub(dst, dst, imm(value)),
            Alu::And => self.asm().and_(dst, dst, imm(value)),
            Alu::Or => self.asm().orr(dst, dst, imm(value)),
            Alu::Xor => self.asm().eor(dst, dst, imm(value)),
            Alu::Mul => {
                self.asm().mov(x(SCRATCH), imm(value));
                self.asm().mul(dst, dst, x(SCRATCH));
            }
        }
    }

    fn neg(&mut self, dst: Reg) {
        self.asm().neg(gp(dst), gp(dst));
    }

    fn not(&mut self, dst: Reg) {
        self.asm().mvn(gp(dst), gp(dst));
    }

    fn shl_imm(&mut self, dst: Reg, amount: u8) {
        self.asm().lsl(gp(dst), gp(dst), imm(amount));
    }

    /// Follows `luaV_mod` and `luaV_idiv` in lvm.c: `SDIV` truncates, so the remainder moves to the divisor's sign.
    fn floor_div(&mut self, op: Division, dst: Reg, x_reg: Reg, y_reg: Reg) {
        let done = self.new_label();
        let (q, r, t) = (x(SCRATCH), x(SCRATCH2), gp(Reg::T0));
        let (xv, yv) = (gp(x_reg), gp(y_reg));
        self.asm().sdiv(q, xv, yv);
        self.asm().msub(r, q, yv, xv);
        let label = self.label(done);
        self.asm().cbz(r, label);
        match op {
            Division::Mod => self.asm().eor(t, r, yv),
            Division::IDiv => self.asm().eor(t, xv, yv),
        }
        self.asm().tbz(t, imm(63), label);
        match op {
            Division::Mod => self.asm().add(r, r, yv),
            Division::IDiv => self.asm().sub(q, q, imm(1)),
        }
        self.bind(done);
        match op {
            Division::Mod => self.asm().mov(gp(dst), r),
            Division::IDiv => self.asm().mov(gp(dst), q),
        }
    }

    fn cmp(&mut self, left: Reg, right: Reg) {
        self.asm().cmp(gp(left), gp(right));
    }

    fn cmp_imm(&mut self, left: Reg, value: i32) {
        self.asm().cmp(gp(left), imm(value));
    }

    fn branch(&mut self, cond: Cond, target: Target) {
        let label = self.label(target);
        self.b_cond(cond, label);
        self.record_branch(target, Field::Branch19);
    }

    fn jump(&mut self, target: Target) {
        let label = self.label(target);
        self.asm().b(label);
        self.record_branch(target, Field::Branch26);
    }


    fn fop(&mut self, op: FAlu, dst: FReg, src: FReg) {
        let (dst, src) = (fp(dst), fp(src));
        let mut asm = self.asm();
        match op {
            FAlu::Add => asm.fadd(dst, dst, src),
            FAlu::Sub => asm.fsub(dst, dst, src),
            FAlu::Mul => asm.fmul(dst, dst, src),
            FAlu::Div => asm.fdiv(dst, dst, src),
        }
    }

    /// `FCMP` sets flags so that `MI`, `LS` and `EQ` test ordered less, less-or-equal and equal.
    fn fbranch(&mut self, cond: FCond, left: FReg, right: FReg, target: Target) {
        self.asm().fcmp(fp(left), fp(right));
        let label = self.label(target);
        match cond {
            FCond::Lt => self.asm().b_mi(label),
            FCond::Le => self.asm().b_ls(label),
            FCond::Eq => self.asm().b_eq(label),
        }
        self.record_branch(target, Field::Branch19);
    }

    fn int_to_float(&mut self, dst: FReg, src: Reg) {
        self.asm().scvtf(fp(dst), gp(src));
    }

    fn call(&mut self, address: u64) {
        self.asm().mov(x(SCRATCH), imm(address as i64));
        self.asm().blr(x(SCRATCH));
    }

    fn arg_imm(&mut self, index: u8, value: Imm) {
        let register = w(u32::from(index));
        match value {
            Imm::Fixed(value) => self.asm().mov(register, imm(value as i32 as u32)),
            Imm::Hole { slot, add } => self.mov_hole(register, slot, add),
        }
    }

    fn arg_reg(&mut self, index: u8, src: Reg) {
        self.asm().mov(x(u32::from(index)), gp(src));
    }

    fn call_reg(&mut self, target: Reg) {
        self.asm().blr(gp(target));
    }

    fn stack_address(&mut self, dst: Reg, offset: i32) {
        self.asm().add(gp(dst), sp, imm(offset));
    }

    fn callee_context(&self) -> i32 {
        0
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

/// The prologue has Cranelift's AArch64 shape, so its unwind codes match Cranelift's.
pub(crate) mod frame {
    use super::*;

    pub(crate) fn reserved(context: usize) -> u32 {
        (context as u32).next_multiple_of(16)
    }

    pub(crate) struct Prologue {
        pub(crate) bytes: Vec<u8>,
        pub(crate) save_frame: u32,
        pub(crate) set_frame: u32,
        pub(crate) pairs: [u32; 2],
    }

    pub(crate) fn prologue(reserved: u32, registers: i32, constants: i32) -> Prologue {
        let mut a = A64::new();
        a.asm().stp(x(29), x(30), ptr_pre(sp, -16));
        let save_frame = a.offset();
        a.asm().mov(x(29), sp);
        let set_frame = a.offset();
        a.asm().stp(x(19), x(20), ptr_pre(sp, -16));
        let first = a.offset();
        a.asm().stp(x(21), x(22), ptr_pre(sp, -16));
        let second = a.offset();
        if reserved > 0 {
            a.asm().sub(sp, sp, imm(reserved));
        }
        a.asm().mov(x(CTX), x(0));
        a.asm().ldr(x(BASE), ptr(x(CTX), registers));
        a.asm().ldr(x(KONST), ptr(x(CTX), constants));
        a.asm().mov(w(0), w(1));
        Prologue {
            bytes: Box::new(a).finish().bytes,
            save_frame,
            set_frame,
            pairs: [first, second],
        }
    }

    pub(crate) fn epilogue(reserved: u32) -> Vec<u8> {
        let mut a = A64::new();
        if reserved > 0 {
            a.asm().add(sp, sp, imm(reserved));
        }
        a.asm().ldp(x(21), x(22), ptr_post(sp, 16));
        a.asm().ldp(x(19), x(20), ptr_post(sp, 16));
        a.asm().ldp(x(29), x(30), ptr_post(sp, 16));
        a.asm().ret(x(30));
        Box::new(a).finish().bytes
    }

    /// `x0` holds the pc or negated exit code.
    pub(crate) fn dispatch(exit: u8, table: u8) -> Stencil {
        let mut a = Box::new(A64::new());
        let trap = a.new_label();
        let zero = a.zero;
        a.asm().tbnz(x(0), imm(63), zero);
        a.hole_here(Field::Branch14, exit, 0);
        a.mov_hole(x(SCRATCH), super::super::masm::slot::I, 0);
        a.asm().cmp(x(0), x(SCRATCH));
        let label = a.label(trap);
        a.asm().b_hs(label);
        a.asm().adr(x(SCRATCH), zero);
        a.hole_here(Field::Adr, table, 0);
        a.asm().ldrsw(x(SCRATCH2), ptr_shift(x(SCRATCH), x(0), lsl(2)));
        a.asm().add(x(SCRATCH), x(SCRATCH), x(SCRATCH2));
        a.asm().br(x(SCRATCH));
        a.bind(trap);
        a.asm().brk(imm(0));
        a.finish()
    }

    pub(crate) fn negate_exit() -> Vec<u8> {
        let mut a = A64::new();
        a.asm().neg(x(0), x(0));
        Box::new(a).finish().bytes
    }
}
