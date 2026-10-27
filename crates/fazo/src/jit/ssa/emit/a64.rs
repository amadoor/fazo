//! Methods may clobber `x17` and `x30` (saved by the prologue) but never `SCRATCH` or `FSCRATCH`.

use asmkit::{Arch, CodeBuffer, Environment, Label as AsmLabel, aarch64::*};

use super::{
    asm::{Alu, ArgLoc, Asm, Cond, DUMP, Division, FAlu, FCond, Label, Width},
    regs::{RegSet, Registers},
    unwind,
};

type VReg = asmkit::aarch64::Vec;

const CTX: u8 = 19;
const SCRATCH: u8 = 16;
const SCRATCH2: u8 = 17;
const LINK: u8 = 30;
const FRAME: u8 = 29;
const STACK: u8 = 31;
const FLOAT_SCRATCH: u8 = 31;
const PAIRS: [u8; 5] = [19, 21, 23, 25, 27];

const GP_ORDER: [u8; 25] = [20, 21, 22, 23, 24, 25, 26, 27, 28, 9, 10, 11, 12, 13, 14, 15, 8, 7, 6, 5, 4, 3, 2, 1, 0];
/// d8 to d15 are callee-saved only in their low halves (AAPCS64 6.1.2), so they are left out.
const FP_ORDER: [u8; 23] = [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30];

/// asmkit's condition codes (`core::globals::CondCode`), which it lowers by subtracting 2.
const EQ: u32 = 2;
const NE: u32 = 3;
const HS: u32 = 4;
const LO: u32 = 5;
const MI: u32 = 6;
const PL: u32 = 7;
const HI: u32 = 10;
const LS: u32 = 11;
const GE: u32 = 12;
const LT: u32 = 13;
const GT: u32 = 14;
const LE: u32 = 15;

const fn callee_saved() -> RegSet {
    let mut bits = 0u64;
    let mut reg = 19;
    while reg <= 28 {
        bits |= 1 << reg;
        reg += 1;
    }
    RegSet(bits)
}

fn r(reg: u8) -> Gp {
    x(u32::from(reg))
}

fn rw(reg: u8) -> Gp {
    w(u32::from(reg))
}

fn f(reg: u8) -> VReg {
    d(u32::from(reg))
}

fn code(cond: Cond) -> u32 {
    match cond {
        Cond::Eq => EQ,
        Cond::Ne => NE,
        Cond::Lt => LT,
        Cond::Le => LE,
        Cond::Gt => GT,
        Cond::Ge => GE,
        Cond::Below => LO,
        Cond::BelowEq => LS,
        Cond::Above => HI,
        Cond::AboveEq => HS,
    }
}

/// After `FCMP`, `MI`, `LS` and `EQ` hold only for ordered less, less-or-equal and equal (Arm ARM C1.2.4).
fn float_code(cond: FCond) -> u32 {
    match cond {
        FCond::Lt => MI,
        FCond::Le => LS,
        FCond::Eq => EQ,
    }
}

pub(crate) struct A64 {
    buffer: CodeBuffer,
    labels: std::vec::Vec<AsmLabel>,
}

impl A64 {
    fn asm(&mut self) -> Assembler<'_> {
        Assembler::new(&mut self.buffer)
    }

    fn label(&self, label: Label) -> AsmLabel {
        self.labels[label.0 as usize]
    }

    /// `x17` holds a displacement neither the scaled nor the unscaled form encodes.
    fn mem(&mut self, size: i32, base: u8, disp: i32) -> Mem {
        let scaled = disp >= 0 && disp % size == 0 && disp / size < 4096;
        if scaled || (-256..256).contains(&disp) {
            return ptr(r(base), disp);
        }
        self.asm().mov(r(SCRATCH2), imm(i64::from(disp)));
        ptr_reg(r(base), r(SCRATCH2))
    }

    fn b_cond(&mut self, cond: u32, target: AsmLabel) {
        let mut asm = self.asm();
        match cond {
            EQ => asm.b_eq(target),
            NE => asm.b_ne(target),
            HS => asm.b_hs(target),
            LO => asm.b_lo(target),
            MI => asm.b_mi(target),
            HI => asm.b_hi(target),
            LS => asm.b_ls(target),
            GE => asm.b_ge(target),
            LT => asm.b_lt(target),
            GT => asm.b_gt(target),
            LE => asm.b_le(target),
            PL => asm.b_pl(target),
            _ => unreachable!("a condition code"),
        }
    }

    /// `dst` and `base` may be the stack pointer only when the immediate form encodes `value`, as every frame offset is (`MAX_FRAME`).
    fn add_imm(&mut self, dst: u8, base: u8, value: i64) {
        if (0..4096).contains(&value) {
            self.asm().add(r(dst), r(base), imm(value));
        } else if (-4095..0).contains(&value) {
            self.asm().sub(r(dst), r(base), imm(-value));
        } else {
            debug_assert!(dst != STACK && base != STACK, "the register form reads register 31 as zero");
            self.asm().mov(r(SCRATCH2), imm(value));
            self.asm().add(r(dst), r(base), r(SCRATCH2));
        }
    }

    fn compare_imm(&mut self, left: u8, value: i64) {
        if (0..4096).contains(&value) {
            self.asm().cmp(r(left), imm(value));
        } else if (-4095..0).contains(&value) {
            self.asm().cmn(r(left), imm(-value));
        } else {
            self.asm().mov(r(SCRATCH2), imm(value));
            self.asm().cmp(r(left), r(SCRATCH2));
        }
    }
}

impl Asm for A64 {
    const CTX: u8 = CTX;
    const SP: u8 = STACK;
    const SCRATCH: u8 = SCRATCH;
    const FSCRATCH: u8 = FLOAT_SCRATCH;
    const OUTGOING: i32 = 0;
    const REGISTERS: Registers = Registers {
        gp: &GP_ORDER,
        fp: &FP_ORDER,
        callee_saved: callee_saved(),
    };
    /// The prologue saves every pair anyway, so caller-saved registers would only add saves around calls.
    const CALLER_SAVED_FIRST: Registers = Self::REGISTERS;
    /// One `sub` sets up the frame, so Windows unwind codes describe it as one `alloc`.
    const MAX_FRAME: u32 = 4080;

    fn new() -> Self {
        Self {
            buffer: CodeBuffer::new(Environment::new(Arch::AArch64)),
            labels: std::vec::Vec::new(),
        }
    }

    fn new_label(&mut self) -> Label {
        let label = self.buffer.get_label();
        self.labels.push(label);
        Label(self.labels.len() as u32 - 1)
    }

    fn bind(&mut self, label: Label) {
        let label = self.label(label);
        self.buffer.bind_label(label);
    }

    fn offset(&self) -> u32 {
        self.buffer.cur_offset()
    }

    fn mov(&mut self, dst: u8, src: u8) {
        if dst != src {
            self.asm().mov(r(dst), r(src));
        }
    }

    fn mov_imm(&mut self, dst: u8, value: i64) {
        self.asm().mov(r(dst), imm(value));
    }

    fn load(&mut self, width: Width, dst: u8, base: u8, disp: i32) {
        match width {
            Width::B => {
                let mem = self.mem(1, base, disp);
                self.asm().ldrb(rw(dst), mem);
            }
            Width::W => {
                let mem = self.mem(4, base, disp);
                self.asm().ldr(rw(dst), mem);
            }
            Width::Q => {
                let mem = self.mem(8, base, disp);
                self.asm().ldr(r(dst), mem);
            }
        }
    }

    fn store(&mut self, width: Width, src: u8, base: u8, disp: i32) {
        match width {
            Width::B => {
                let mem = self.mem(1, base, disp);
                self.asm().strb(rw(src), mem);
            }
            Width::W => {
                let mem = self.mem(4, base, disp);
                self.asm().str(rw(src), mem);
            }
            Width::Q => {
                let mem = self.mem(8, base, disp);
                self.asm().str(r(src), mem);
            }
        }
    }

    fn store_imm(&mut self, width: Width, base: u8, disp: i32, value: i64) {
        self.asm().mov(r(LINK), imm(value));
        self.store(width, LINK, base, disp);
    }

    fn load_f(&mut self, dst: u8, base: u8, disp: i32) {
        let mem = self.mem(8, base, disp);
        self.asm().ldr(f(dst), mem);
    }

    fn store_f(&mut self, src: u8, base: u8, disp: i32) {
        let mem = self.mem(8, base, disp);
        self.asm().str(f(src), mem);
    }

    fn fmov(&mut self, dst: u8, src: u8) {
        if dst != src {
            self.asm().fmov(f(dst), f(src));
        }
    }

    fn bits_to_float(&mut self, dst: u8, src: u8) {
        self.asm().fmov(f(dst), r(src));
    }

    fn float_to_bits(&mut self, dst: u8, src: u8) {
        self.asm().fmov(r(dst), f(src));
    }

    fn lea(&mut self, dst: u8, base: u8, disp: i32) {
        self.add_imm(dst, base, i64::from(disp));
    }

    fn load_i32(&mut self, dst: u8, base: u8, disp: i32) {
        let mem = self.mem(4, base, disp);
        self.asm().ldrsw(r(dst), mem);
    }

    fn zext32(&mut self, dst: u8, src: u8) {
        self.asm().mov(rw(dst), rw(src));
    }

    fn shl_imm(&mut self, dst: u8, src: u8, amount: u8) {
        self.asm().lsl(r(dst), r(src), imm(amount));
    }

    fn alu(&mut self, op: Alu, dst: u8, left: u8, right: u8) {
        let (dst, left, right) = (r(dst), r(left), r(right));
        let mut asm = self.asm();
        match op {
            Alu::Add => asm.add(dst, left, right),
            Alu::Sub => asm.sub(dst, left, right),
            Alu::Mul => asm.mul(dst, left, right),
            Alu::And => asm.and_(dst, left, right),
            Alu::Or => asm.orr(dst, left, right),
            Alu::Xor => asm.eor(dst, left, right),
        }
    }

    fn alu_imm(&mut self, op: Alu, dst: u8, left: u8, value: i32) {
        match op {
            Alu::Add => self.add_imm(dst, left, i64::from(value)),
            Alu::Sub => self.add_imm(dst, left, -i64::from(value)),
            _ => {
                self.asm().mov(r(SCRATCH2), imm(i64::from(value)));
                self.alu(op, dst, left, SCRATCH2);
            }
        }
    }

    fn add_mem(&mut self, dst: u8, base: u8, disp: i32) {
        self.load(Width::Q, SCRATCH2, base, disp);
        self.asm().add(r(dst), r(dst), r(SCRATCH2));
    }

    fn sub_mem(&mut self, base: u8, disp: i32, value: i32) {
        self.load(Width::Q, SCRATCH2, base, disp);
        if (0..4096).contains(&value) {
            self.asm().subs(r(SCRATCH2), r(SCRATCH2), imm(i64::from(value)));
        } else {
            self.asm().mov(r(LINK), imm(i64::from(value)));
            self.asm().subs(r(SCRATCH2), r(SCRATCH2), r(LINK));
        }
        self.store(Width::Q, SCRATCH2, base, disp);
    }

    fn neg(&mut self, dst: u8, src: u8) {
        self.asm().neg(r(dst), r(src));
    }

    fn not(&mut self, dst: u8, src: u8) {
        self.asm().mvn(r(dst), r(src));
    }

    fn unsigned_rem(&mut self, dst: u8, x_reg: u8, y_reg: u8) {
        let q = r(SCRATCH2);
        self.asm().udiv(q, r(x_reg), r(y_reg));
        self.asm().msub(r(dst), q, r(y_reg), r(x_reg));
    }

    /// Follows Baseline's `floor_div`: `SDIV` truncates, so the remainder is corrected when its sign differs.
    fn floor_div(&mut self, op: Division, dst: u8, x_reg: u8, y_reg: u8) {
        let done = self.buffer.get_label();
        let (q, rem, t) = (r(SCRATCH2), r(LINK), r(SCRATCH));
        let (xv, yv) = (r(x_reg), r(y_reg));
        self.asm().sdiv(q, xv, yv);
        self.asm().msub(rem, q, yv, xv);
        self.asm().cbz(rem, done);
        match op {
            Division::Mod => self.asm().eor(t, rem, yv),
            Division::IDiv => self.asm().eor(t, xv, yv),
        }
        self.asm().tbz(t, imm(63), done);
        match op {
            Division::Mod => self.asm().add(rem, rem, yv),
            Division::IDiv => self.asm().sub(q, q, imm(1)),
        }
        self.buffer.bind_label(done);
        match op {
            Division::Mod => self.asm().mov(r(dst), rem),
            Division::IDiv => self.asm().mov(r(dst), q),
        }
    }

    fn cmp(&mut self, left: u8, right: u8) {
        self.asm().cmp(r(left), r(right));
    }

    fn cmp_imm(&mut self, left: u8, value: i32) {
        self.compare_imm(left, i64::from(value));
    }

    fn cmp_imm64(&mut self, left: u8, value: i64) {
        self.compare_imm(left, value);
    }

    fn cmp_mem_byte(&mut self, base: u8, disp: i32, value: u8) {
        self.load(Width::B, SCRATCH2, base, disp);
        self.cmp_imm(SCRATCH2, i32::from(value));
    }

    fn cmp_byte(&mut self, left: u8, value: u8) {
        self.asm().uxtb(rw(SCRATCH2), rw(left));
        self.asm().cmp(rw(SCRATCH2), imm(value));
    }

    fn test_imm(&mut self, left: u8, mask: i32) {
        self.asm().mov(r(SCRATCH2), imm(i64::from(mask)));
        self.asm().tst(r(left), r(SCRATCH2));
    }

    fn set(&mut self, cond: Cond, dst: u8) {
        self.asm().cset(r(dst), imm(code(cond)));
    }

    fn select(&mut self, cond: Cond, dst: u8, yes: u8, no: u8) {
        self.asm().csel(r(dst), r(yes), r(no), imm(code(cond)));
    }

    fn branch(&mut self, cond: Cond, target: Label) {
        let target = self.label(target);
        self.b_cond(code(cond), target);
    }

    fn jump(&mut self, target: Label) {
        let target = self.label(target);
        self.asm().b(target);
    }

    fn falu(&mut self, op: FAlu, dst: u8, left: u8, right: u8) {
        let (dst, left, right) = (f(dst), f(left), f(right));
        let mut asm = self.asm();
        match op {
            FAlu::Add => asm.fadd(dst, left, right),
            FAlu::Sub => asm.fsub(dst, left, right),
            FAlu::Mul => asm.fmul(dst, left, right),
            FAlu::Div => asm.fdiv(dst, left, right),
        }
    }

    fn fneg(&mut self, dst: u8, src: u8) {
        self.asm().fneg(f(dst), f(src));
    }

    fn fsqrt(&mut self, dst: u8, src: u8) {
        self.asm().fsqrt(f(dst), f(src));
    }

    fn ffloor(&mut self, dst: u8, src: u8) {
        self.asm().frintm(f(dst), f(src));
    }

    fn fbranch(&mut self, cond: FCond, left: u8, right: u8, target: Label) {
        let target = self.label(target);
        self.asm().fcmp(f(left), f(right));
        self.b_cond(float_code(cond), target);
    }

    /// The negations of `MI`, `LS` and `EQ` (`PL`, `HI`, `NE`) all hold for unordered operands.
    fn fbranch_not(&mut self, cond: FCond, left: u8, right: u8, target: Label) {
        let target = self.label(target);
        self.asm().fcmp(f(left), f(right));
        let negated = match cond {
            FCond::Lt => PL,
            FCond::Le => HI,
            FCond::Eq => NE,
        };
        self.b_cond(negated, target);
    }

    fn fset(&mut self, cond: FCond, dst: u8, left: u8, right: u8) {
        self.asm().fcmp(f(left), f(right));
        self.asm().cset(r(dst), imm(float_code(cond)));
    }

    fn int_to_float(&mut self, dst: u8, src: u8) {
        self.asm().scvtf(f(dst), r(src));
    }

    fn call(&mut self, address: u64) {
        self.asm().mov(r(SCRATCH2), imm(address as i64));
        self.asm().blr(r(SCRATCH2));
    }

    fn call_reg(&mut self, target: u8) {
        self.asm().blr(r(target));
    }

    fn arg(index: usize) -> ArgLoc {
        if index < 8 { ArgLoc::Reg(index as u8) } else { ArgLoc::Stack(8 * (index as i32 - 8)) }
    }

    fn float_arg(_index: usize, float: usize) -> u8 {
        float as u8
    }

    fn result() -> u8 {
        0
    }

    fn float_result() -> u8 {
        0
    }

    fn dump_registers(&mut self) {
        self.asm().sub(r(STACK), r(STACK), imm(DUMP + Self::OUTGOING));
        for reg in Self::REGISTERS.all().iter() {
            let at = Self::OUTGOING + reg.index() as i32 * 8;
            match reg.class {
                super::regs::Class::Gp => self.store(Width::Q, reg.hw, STACK, at),
                super::regs::Class::Fp => self.store_f(reg.hw, STACK, at),
            }
        }
    }

    fn trap(&mut self) {
        self.asm().brk(imm(0));
    }

    /// Follows Baseline's AArch64 prologue (`baseline/a64.rs`, `frame::prologue`) with two more pairs.
    fn prologue(&mut self, frame: u32, _floats: &[(u8, u32)], pc_dst: u8, _saved: &[u8]) -> unwind::Prologue {
        let mut prologue = unwind::Prologue { arch: unwind::Arch::A64, frame, ..unwind::Prologue::default() };
        self.asm().stp(r(FRAME), r(LINK), ptr_pre(r(STACK), -16));
        prologue.save_frame = self.offset();
        self.asm().mov(r(FRAME), r(STACK));
        prologue.set_frame = self.offset();
        for first in PAIRS {
            self.asm().stp(r(first), r(first + 1), ptr_pre(r(STACK), -16));
            prologue.saves.push((self.offset(), first));
        }
        if frame > 0 {
            self.asm().sub(r(STACK), r(STACK), imm(frame));
        }
        prologue.alloc = self.offset();
        self.asm().mov(r(CTX), r(0));
        self.asm().mov(rw(pc_dst), rw(1));
        prologue
    }

    fn epilogue(&mut self, _frame: u32, _floats: &[(u8, u32)], code: u8, _saved: &[u8]) {
        self.mov(0, code);
        self.asm().sub(r(STACK), r(FRAME), imm(16 * PAIRS.len() as u32));
        for first in PAIRS.iter().rev() {
            self.asm().ldp(r(*first), r(first + 1), ptr_post(r(STACK), 16));
        }
        self.asm().ldp(r(FRAME), r(LINK), ptr_post(r(STACK), 16));
        self.asm().ret(r(LINK));
    }

    fn tail_epilogue(&mut self, _frame: u32, _floats: &[(u8, u32)], _saved: &[u8]) {
        self.asm().sub(r(STACK), r(FRAME), imm(16 * PAIRS.len() as u32));
        for first in PAIRS.iter().rev() {
            self.asm().ldp(r(*first), r(first + 1), ptr_post(r(STACK), 16));
        }
        self.asm().ldp(r(FRAME), r(LINK), ptr_post(r(STACK), 16));
        self.asm().br(r(SCRATCH));
    }

    fn finish(mut self) -> Option<std::vec::Vec<u8>> {
        let code = self.buffer.finish().ok()?;
        Some(code.data().to_vec())
    }

    /// Every pair is saved, so the prologue keeps one unwind shape.
    fn saved_registers(_used: RegSet) -> std::vec::Vec<u8> {
        std::vec::Vec::new()
    }

    /// The pushes are whole pairs, so the frame alone keeps the stack 16-byte aligned.
    fn frame_size(needed: u32, _saved: usize) -> u32 {
        needed.next_multiple_of(16)
    }

    fn raw_jump(distance: u32) -> std::vec::Vec<u8> {
        let word = 0x1400_0000 | ((distance / 4 + 1) & 0x03ff_ffff);
        word.to_le_bytes().to_vec()
    }
}
