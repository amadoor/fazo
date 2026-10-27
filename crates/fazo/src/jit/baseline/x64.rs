//! x86-64 encoder for the Baseline templates. Encodes directly (Intel SDM volume 2, chapters 2 and 3), like the Optimizing
//! encoder: asmkit's operand and CPU checks cost most of a first compile. Holes always take the 32- or 64-bit form.

use std::vec::Vec;

use super::masm::{Alu, Cond, Division, FAlu, FCond, FReg, Field, Hole, Imm, Masm, Reg, Stencil, Target, Width};

const CTX: u8 = 3;
const BASE: u8 = 12;
const KONST: u8 = 13;
const SCRATCH: u8 = 11;
const RAX: u8 = 0;
const RCX: u8 = 1;
const RDX: u8 = 2;
const RSP: u8 = 4;
const RBP: u8 = 5;

#[cfg(windows)]
const ARGS: [u8; 4] = [1, 2, 8, 9];
#[cfg(not(windows))]
const ARGS: [u8; 5] = [7, 6, 2, 1, 8];

/// Win64 passes the fifth integer argument on the stack, above the 32-byte shadow space.
const STACK_ARG: i32 = 32;

fn gp(reg: Reg) -> u8 {
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

fn fp(reg: FReg) -> u8 {
    match reg {
        FReg::F0 => 0,
        FReg::F1 => 1,
        FReg::F2 => 2,
        FReg::F3 => 3,
    }
}

/// The `tttn` condition field of `Jcc` (SDM volume 1, appendix B).
fn cc(cond: Cond) -> u8 {
    match cond {
        Cond::Eq => 0x4,
        Cond::Ne => 0x5,
        Cond::Lt => 0xc,
        Cond::Le => 0xe,
        Cond::Gt => 0xf,
        Cond::Ge => 0xd,
        Cond::BelowEq => 0x6,
        Cond::Above => 0x7,
        Cond::AboveEq => 0x3,
    }
}

const PARITY: u8 = 0xa;
const NOT_SIGN: u8 = 0x9;

/// The `/digit` of the immediate group-1 forms and the opcode of the register forms.
fn group(op: Alu) -> (u8, u8) {
    match op {
        Alu::Add => (0, 0x01),
        Alu::Or => (1, 0x09),
        Alu::And => (4, 0x21),
        Alu::Sub => (5, 0x29),
        Alu::Xor => (6, 0x31),
        Alu::Mul => unreachable!("multiplication has its own encodings"),
    }
}

const CMP: (u8, u8) = (7, 0x39);

#[derive(Clone, Copy)]
enum Size {
    Byte,
    Dword,
    Qword,
}

/// `wide` forces a 32-bit displacement, which a hole needs.
#[derive(Clone, Copy)]
enum Rm {
    Reg(u8),
    Mem { base: u8, disp: i32, wide: bool },
}

fn mem(base: u8, disp: i32) -> Rm {
    Rm::Mem { base, disp, wide: false }
}

pub(crate) struct X64 {
    code: Vec<u8>,
    labels: Vec<Option<u32>>,
    fixups: Vec<(u32, u32)>,
    holes: Vec<Hole>,
}

impl X64 {
    pub(crate) fn new() -> Self {
        Self {
            code: Vec::new(),
            labels: Vec::new(),
            fixups: Vec::new(),
            holes: Vec::new(),
        }
    }

    fn offset(&self) -> u32 {
        self.code.len() as u32
    }

    fn byte(&mut self, byte: u8) {
        self.code.push(byte);
    }

    fn bytes(&mut self, bytes: &[u8]) {
        self.code.extend_from_slice(bytes);
    }

    fn imm32(&mut self, value: i32) {
        self.bytes(&value.to_le_bytes());
    }

    /// A REX prefix when the operand size, an extended register or a byte register past `bl` needs one.
    fn rex(&mut self, size: Size, reg: u8, rm: Rm) {
        let base = match rm {
            Rm::Reg(reg) | Rm::Mem { base: reg, .. } => reg,
        };
        let byte_reg = |reg: u8| matches!(size, Size::Byte) && (4..8).contains(&reg);
        let low_byte = byte_reg(reg) || matches!(rm, Rm::Reg(rm) if byte_reg(rm));
        let w = matches!(size, Size::Qword);
        let value = 0x40 | u8::from(w) << 3 | (reg >> 3 & 1) << 2 | (base >> 3 & 1);
        if value != 0x40 || low_byte {
            self.byte(value);
        }
    }

    /// ModRM, SIB and displacement; `rsp` and `r12` bases need a SIB byte, `rbp` and `r13` a displacement.
    fn modrm(&mut self, reg: u8, rm: Rm) {
        let reg = (reg & 7) << 3;
        let (base, disp, wide) = match rm {
            Rm::Reg(rm) => {
                self.byte(0xc0 | reg | (rm & 7));
                return;
            }
            Rm::Mem { base, disp, wide } => (base & 7, disp, wide),
        };
        let (mode, short) = match disp {
            _ if wide => (0x80, Some(false)),
            0 if base != 5 => (0x00, None),
            -128..=127 => (0x40, Some(true)),
            _ => (0x80, Some(false)),
        };
        self.byte(mode | reg | base);
        if base == 4 {
            self.byte(0x24);
        }
        match short {
            Some(true) => self.byte(disp as i8 as u8),
            Some(false) => self.imm32(disp),
            None => {}
        }
    }

    fn op(&mut self, prefix: Option<u8>, size: Size, opcode: &[u8], reg: u8, rm: Rm) {
        if let Some(prefix) = prefix {
            self.byte(prefix);
        }
        self.rex(size, reg, rm);
        self.bytes(opcode);
        self.modrm(reg, rm);
    }

    fn hole(&mut self, offset: u32, field: Field, slot: u8, add: i32) {
        self.holes.push(Hole { offset, field, slot, add });
    }

    /// A hole's displacement is the last four bytes of the ModRM encoding, before any immediate.
    fn mem_op(&mut self, prefix: Option<u8>, size: Size, opcode: &[u8], reg: u8, (base, disp): (Reg, Imm)) {
        let (value, hole) = match disp {
            Imm::Fixed(value) => (i32::try_from(value).expect("a fixed displacement fits 32 bits"), None),
            Imm::Hole { slot, add } => (0, Some((slot, add))),
        };
        self.op(prefix, size, opcode, reg, Rm::Mem { base: gp(base), disp: value, wide: hole.is_some() });
        if let Some((slot, add)) = hole {
            self.hole(self.offset() - 4, Field::Le32, slot, add);
        }
    }

    fn group_imm(&mut self, (digit, _): (u8, u8), size: Size, rm: Rm, value: i32) {
        match i8::try_from(value) {
            Ok(short) => {
                self.op(None, size, &[0x83], digit, rm);
                self.byte(short as u8);
            }
            Err(_) => {
                self.op(None, size, &[0x81], digit, rm);
                self.imm32(value);
            }
        }
    }

    fn mov_rr(&mut self, dst: u8, src: u8) {
        self.op(None, Size::Qword, &[0x89], src, Rm::Reg(dst));
    }

    fn target_rel32(&mut self, target: Target) {
        let at = self.offset();
        self.imm32(0);
        match target {
            Target::Local(index) => self.fixups.push((at, index)),
            Target::Ext(id) => self.hole(at, Field::Rel32, id, 0),
        }
    }

    fn jcc_code(&mut self, code: u8, target: Target) {
        self.bytes(&[0x0f, 0x80 | code]);
        self.target_rel32(target);
    }

    fn jcc(&mut self, cond: Cond, target: Target) {
        self.jcc_code(cc(cond), target);
    }

    fn sse(&mut self, prefix: u8, opcode: u8, reg: u8, rm: Rm) {
        self.op(Some(prefix), Size::Dword, &[0x0f, opcode], reg, rm);
    }

    fn push(&mut self, reg: u8) {
        if reg >= 8 {
            self.byte(0x41);
        }
        self.byte(0x50 | (reg & 7));
    }

    fn pop(&mut self, reg: u8) {
        if reg >= 8 {
            self.byte(0x41);
        }
        self.byte(0x58 | (reg & 7));
    }

    fn set_arg_value(&mut self, index: u8, value: Imm) {
        let placed = match value {
            Imm::Fixed(value) => value as i32,
            Imm::Hole { .. } => 0,
        };
        if cfg!(windows) && usize::from(index) >= ARGS.len() {
            self.op(None, Size::Dword, &[0xc7], 0, mem(RSP, STACK_ARG));
        } else {
            let dst = ARGS[usize::from(index)];
            if dst >= 8 {
                self.byte(0x41);
            }
            self.byte(0xb8 | (dst & 7));
        }
        self.imm32(placed);
        if let Imm::Hole { slot, add } = value {
            self.hole(self.offset() - 4, Field::Le32, slot, add);
        }
    }
}

impl Masm for X64 {
    fn new_label(&mut self) -> Target {
        self.labels.push(None);
        Target::Local(self.labels.len() as u32 - 1)
    }

    fn bind(&mut self, label: Target) {
        let Target::Local(index) = label else {
            unreachable!("only local labels are bound");
        };
        self.labels[index as usize] = Some(self.offset());
    }

    fn load(&mut self, width: Width, dst: Reg, base: Reg, disp: Imm) {
        let dst = gp(dst);
        match width {
            Width::U8 => self.mem_op(None, Size::Dword, &[0x0f, 0xb6], dst, (base, disp)),
            Width::U32 => self.mem_op(None, Size::Dword, &[0x8b], dst, (base, disp)),
            Width::U64 => self.mem_op(None, Size::Qword, &[0x8b], dst, (base, disp)),
        }
    }

    fn store(&mut self, width: Width, src: Reg, base: Reg, disp: Imm) {
        let src = gp(src);
        match width {
            Width::U8 => self.mem_op(None, Size::Byte, &[0x88], src, (base, disp)),
            Width::U32 => self.mem_op(None, Size::Dword, &[0x89], src, (base, disp)),
            Width::U64 => self.mem_op(None, Size::Qword, &[0x89], src, (base, disp)),
        }
    }

    fn store_imm(&mut self, width: Width, base: Reg, disp: Imm, value: i32) {
        match width {
            Width::U8 => {
                self.mem_op(None, Size::Byte, &[0xc6], 0, (base, disp));
                self.byte(value as i8 as u8);
            }
            Width::U32 => {
                self.mem_op(None, Size::Dword, &[0xc7], 0, (base, disp));
                self.imm32(value);
            }
            Width::U64 => {
                self.mem_op(None, Size::Qword, &[0xc7], 0, (base, disp));
                self.imm32(value);
            }
        }
    }

    fn load_f(&mut self, dst: FReg, base: Reg, disp: Imm) {
        self.mem_op(Some(0xf2), Size::Dword, &[0x0f, 0x10], fp(dst), (base, disp));
    }

    fn store_f(&mut self, src: FReg, base: Reg, disp: Imm) {
        self.mem_op(Some(0xf2), Size::Dword, &[0x0f, 0x11], fp(src), (base, disp));
    }

    fn mov(&mut self, dst: Reg, src: Reg) {
        self.mov_rr(gp(dst), gp(src));
    }

    fn mov_imm(&mut self, dst: Reg, value: Imm) {
        let dst = gp(dst);
        let (bits, hole) = match value {
            Imm::Fixed(value) => (value, None),
            Imm::Hole { slot, add } => (0, Some((slot, add))),
        };
        if hole.is_none()
            && let Ok(value) = i32::try_from(bits)
        {
            self.op(None, Size::Qword, &[0xc7], 0, Rm::Reg(dst));
            self.imm32(value);
            return;
        }
        self.byte(0x48 | (dst >> 3 & 1));
        self.byte(0xb8 | (dst & 7));
        self.bytes(&bits.to_le_bytes());
        if let Some((slot, add)) = hole {
            self.hole(self.offset() - 8, Field::Le64, slot, add);
        }
    }

    fn alu(&mut self, op: Alu, dst: Reg, src: Reg) {
        let (dst, src) = (gp(dst), gp(src));
        if op == Alu::Mul {
            self.op(None, Size::Qword, &[0x0f, 0xaf], dst, Rm::Reg(src));
            return;
        }
        self.op(None, Size::Qword, &[group(op).1], src, Rm::Reg(dst));
    }

    fn alu_imm(&mut self, op: Alu, dst: Reg, value: i32) {
        let dst = gp(dst);
        if op != Alu::Mul {
            return self.group_imm(group(op), Size::Qword, Rm::Reg(dst), value);
        }
        match i8::try_from(value) {
            Ok(short) => {
                self.op(None, Size::Qword, &[0x6b], dst, Rm::Reg(dst));
                self.byte(short as u8);
            }
            Err(_) => {
                self.op(None, Size::Qword, &[0x69], dst, Rm::Reg(dst));
                self.imm32(value);
            }
        }
    }

    fn neg(&mut self, dst: Reg) {
        self.op(None, Size::Qword, &[0xf7], 3, Rm::Reg(gp(dst)));
    }

    fn not(&mut self, dst: Reg) {
        self.op(None, Size::Qword, &[0xf7], 2, Rm::Reg(gp(dst)));
    }

    fn shl_imm(&mut self, dst: Reg, amount: u8) {
        self.op(None, Size::Qword, &[0xc1], 4, Rm::Reg(gp(dst)));
        self.byte(amount);
    }

    fn floor_div(&mut self, op: Division, dst: Reg, x: Reg, y: Reg) {
        debug_assert!(![Reg::T0, Reg::T1, Reg::T2].contains(&x) && ![Reg::T0, Reg::T1, Reg::T2].contains(&y));
        let done = self.new_label();
        let (x, y) = (gp(x), gp(y));
        self.mov_rr(RAX, x);
        self.bytes(&[0x48, 0x99]);
        self.op(None, Size::Qword, &[0xf7], 7, Rm::Reg(y));
        self.op(None, Size::Qword, &[0x85], RDX, Rm::Reg(RDX));
        self.jcc(Cond::Eq, done);
        match op {
            Division::Mod => self.mov_rr(RCX, RDX),
            Division::IDiv => self.mov_rr(RCX, x),
        }
        self.op(None, Size::Qword, &[group(Alu::Xor).1], y, Rm::Reg(RCX));
        self.jcc_code(NOT_SIGN, done);
        match op {
            Division::Mod => self.op(None, Size::Qword, &[group(Alu::Add).1], y, Rm::Reg(RDX)),
            Division::IDiv => self.group_imm(group(Alu::Sub), Size::Qword, Rm::Reg(RAX), 1),
        }
        self.bind(done);
        match op {
            Division::Mod => self.mov_rr(gp(dst), RDX),
            Division::IDiv => self.mov_rr(gp(dst), RAX),
        }
    }

    fn cmp(&mut self, left: Reg, right: Reg) {
        self.op(None, Size::Qword, &[CMP.1], gp(right), Rm::Reg(gp(left)));
    }

    fn cmp_imm(&mut self, left: Reg, value: i32) {
        self.group_imm(CMP, Size::Qword, Rm::Reg(gp(left)), value);
    }

    fn branch(&mut self, cond: Cond, target: Target) {
        self.jcc(cond, target);
    }

    fn jump(&mut self, target: Target) {
        self.byte(0xe9);
        self.target_rel32(target);
    }

    fn fop(&mut self, op: FAlu, dst: FReg, src: FReg) {
        let opcode = match op {
            FAlu::Add => 0x58,
            FAlu::Mul => 0x59,
            FAlu::Sub => 0x5c,
            FAlu::Div => 0x5e,
        };
        self.sse(0xf2, opcode, fp(dst), Rm::Reg(fp(src)));
    }

    /// `UCOMISD` sets PF on unordered pairs, so `a < b` is tested as `b > a` and equality also checks PF clear.
    fn fbranch(&mut self, cond: FCond, left: FReg, right: FReg, target: Target) {
        let ucomisd = |x: &mut Self, left: FReg, right: FReg| x.sse(0x66, 0x2e, fp(left), Rm::Reg(fp(right)));
        match cond {
            FCond::Lt => {
                ucomisd(self, right, left);
                self.jcc(Cond::Above, target);
            }
            FCond::Le => {
                ucomisd(self, right, left);
                self.jcc(Cond::AboveEq, target);
            }
            FCond::Eq => {
                let unordered = self.new_label();
                ucomisd(self, left, right);
                self.jcc_code(PARITY, unordered);
                self.jcc(Cond::Eq, target);
                self.bind(unordered);
            }
        }
    }

    fn int_to_float(&mut self, dst: FReg, src: Reg) {
        self.op(Some(0xf2), Size::Qword, &[0x0f, 0x2a], fp(dst), Rm::Reg(gp(src)));
    }

    fn call(&mut self, address: u64) {
        self.byte(0x49);
        self.byte(0xb8 | (SCRATCH & 7));
        self.bytes(&address.to_le_bytes());
        self.op(None, Size::Dword, &[0xff], 2, Rm::Reg(SCRATCH));
    }

    fn arg_imm(&mut self, index: u8, value: Imm) {
        self.set_arg_value(index, value);
    }

    fn arg_reg(&mut self, index: u8, src: Reg) {
        if cfg!(windows) && usize::from(index) >= ARGS.len() {
            self.op(None, Size::Qword, &[0x89], gp(src), mem(RSP, STACK_ARG));
            return;
        }
        self.mov_rr(ARGS[usize::from(index)], gp(src));
    }

    fn call_reg(&mut self, target: Reg) {
        self.op(None, Size::Dword, &[0xff], 2, Rm::Reg(gp(target)));
    }

    fn stack_address(&mut self, dst: Reg, offset: i32) {
        self.op(None, Size::Qword, &[0x8d], gp(dst), mem(RSP, offset));
    }

    fn callee_context(&self) -> i32 {
        frame::CONTEXT
    }

    fn finish(mut self: Box<Self>) -> Stencil {
        for &(at, label) in &self.fixups {
            let target = self.labels[label as usize].expect("every branch target is bound");
            let rel = target as i32 - (at as i32 + 4);
            self.code[at as usize..at as usize + 4].copy_from_slice(&rel.to_le_bytes());
        }
        Stencil {
            bytes: std::mem::take(&mut self.code),
            holes: std::mem::take(&mut self.holes),
        }
    }
}

/// The prologue, epilogue and dispatch of a Baseline function, which the unwind tables describe.
pub(crate) mod frame {
    use super::*;

    pub(crate) const SAVED: [u32; 4] = [CTX as u32, BASE as u32, KONST as u32, 14];
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
        x.push(RBP);
        let push_rbp = x.offset() as u8;
        x.mov_rr(RBP, RSP);
        let set_rbp = x.offset() as u8;
        let mut pushes = [0u8; 4];
        for (index, &register) in SAVED.iter().enumerate() {
            x.push(register as u8);
            pushes[index] = x.offset() as u8;
        }
        x.group_imm(group(Alu::Sub), Size::Qword, Rm::Reg(RSP), reserved as i32);
        let alloc = x.offset() as u8;
        x.mov_rr(CTX, ARGS[0]);
        x.op(None, Size::Qword, &[0x8b], BASE, mem(CTX, registers));
        x.op(None, Size::Qword, &[0x8b], KONST, mem(CTX, constants));
        x.op(None, Size::Dword, &[0x89], ARGS[1], Rm::Reg(RAX));
        Prologue {
            bytes: x.code,
            push_rbp,
            set_rbp,
            pushes,
            alloc,
        }
    }

    pub(crate) fn epilogue(reserved: u32) -> Vec<u8> {
        let mut x = X64::new();
        x.group_imm(group(Alu::Add), Size::Qword, Rm::Reg(RSP), reserved as i32);
        for &register in SAVED.iter().rev() {
            x.pop(register as u8);
        }
        x.pop(RBP);
        x.byte(0xc3);
        x.code
    }

    /// `rax` holds the pc to resume at; the table of `rel32` entries is at the `table` hole.
    pub(crate) fn dispatch(exit: u8, table: u8) -> Stencil {
        let mut x = Box::new(X64::new());
        let trap = x.new_label();
        x.op(None, Size::Qword, &[0x85], RAX, Rm::Reg(RAX));
        x.jcc(Cond::Lt, Target::Ext(exit));
        x.op(None, Size::Qword, &[0x81], CMP.0, Rm::Reg(RAX));
        x.imm32(0);
        x.hole(x.offset() - 4, Field::Le32, super::super::masm::slot::I, 0);
        x.jcc(Cond::AboveEq, trap);
        x.bytes(&[0x48, 0x8d, 0x0d]);
        x.target_rel32(Target::Ext(table));
        x.bytes(&[0x48, 0x63, 0x14, 0x81]);
        x.op(None, Size::Qword, &[0x01], RDX, Rm::Reg(RCX));
        x.op(None, Size::Dword, &[0xff], 4, Rm::Reg(RCX));
        x.bind(trap);
        x.bytes(&[0x0f, 0x0b]);
        x.finish()
    }

    pub(crate) fn negate_exit() -> Vec<u8> {
        let mut x = X64::new();
        x.neg(Reg::T0);
        x.code
    }
}
