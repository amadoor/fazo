//! Methods may clobber `r11` but never `SCRATCH` or `FSCRATCH`, which the code generator owns between calls. Encodes directly (Intel SDM volume 2, chapters 2 and 3): asmkit's operand and CPU checks cost more than the rest of the compile.

use super::{
    asm::{Alu, ArgLoc, Asm, Cond, DUMP, Division, FAlu, FCond, Label, Width},
    regs::{PReg, RegSet, Registers},
    unwind,
};

const RAX: u8 = 0;
const RDX: u8 = 2;
const RBX: u8 = 3;
const RSP: u8 = 4;
const RBP: u8 = 5;
const R11: u8 = 11;

#[cfg(windows)]
const SAVED: [u8; 7] = [RBX, 12, 13, 14, 15, 6, 7];
#[cfg(not(windows))]
const SAVED: [u8; 5] = [RBX, 12, 13, 14, 15];

#[cfg(windows)]
const ARGS: [u8; 4] = [1, 2, 8, 9];
#[cfg(not(windows))]
const ARGS: [u8; 6] = [7, 6, 2, 1, 8, 9];

const GP_ORDER: [u8; 11] = [12, 13, 14, 15, 6, 7, 8, 9, 10, 1, 2];
#[cfg(windows)]
const CALLER_GP_ORDER: [u8; 11] = [8, 9, 10, 1, 2, 12, 13, 14, 15, 6, 7];
#[cfg(not(windows))]
const CALLER_GP_ORDER: [u8; 11] = [6, 7, 8, 9, 10, 1, 2, 12, 13, 14, 15];

#[cfg(windows)]
const FP_ORDER: [u8; 15] = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
#[cfg(not(windows))]
const FP_ORDER: [u8; 15] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];

#[cfg(windows)]
const FLOAT_SCRATCH: u8 = 5;
#[cfg(not(windows))]
const FLOAT_SCRATCH: u8 = 15;

const fn callee_saved() -> RegSet {
    let gp = (1 << 12) | (1 << 13) | (1 << 14) | (1 << 15);
    if cfg!(windows) {
        let fp: u64 = 0xffc0;
        RegSet(gp | (1 << 6) | (1 << 7) | (fp << 32))
    } else {
        RegSet(gp)
    }
}

/// The `tttn` condition field of `Jcc`, `SETcc` and `CMOVcc` (SDM volume 1, appendix B).
fn cc(cond: Cond) -> u8 {
    match cond {
        Cond::Eq => 0x4,
        Cond::Ne => 0x5,
        Cond::Lt => 0xc,
        Cond::Le => 0xe,
        Cond::Gt => 0xf,
        Cond::Ge => 0xd,
        Cond::Below => 0x2,
        Cond::BelowEq => 0x6,
        Cond::Above => 0x7,
        Cond::AboveEq => 0x3,
    }
}

const PARITY: u8 = 0xa;
const NO_PARITY: u8 = 0xb;

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

#[derive(Clone, Copy)]
enum Rm {
    Reg(u8),
    Mem(u8, i32),
}

pub(crate) struct X64 {
    code: Vec<u8>,
    labels: Vec<Option<u32>>,
    /// Each `rel32` field's offset and the label it reaches.
    fixups: Vec<(u32, u32)>,
    /// Float constants placed after the code, with the `rel32` fields that load them.
    pool: Vec<u64>,
    pool_fixups: Vec<(u32, u32)>,
}

impl X64 {
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
    fn rex(&mut self, size: Size, reg: u8, rm: Rm, force: bool) {
        let base = match rm {
            Rm::Reg(reg) | Rm::Mem(reg, _) => reg,
        };
        let byte_reg = |reg: u8| matches!(size, Size::Byte) && (4..8).contains(&reg);
        let low_byte = byte_reg(reg) || matches!(rm, Rm::Reg(rm) if byte_reg(rm));
        let w = matches!(size, Size::Qword);
        let value = 0x40 | u8::from(w) << 3 | (reg >> 3 & 1) << 2 | (base >> 3 & 1);
        if value != 0x40 || low_byte || force {
            self.byte(value);
        }
    }

    /// ModRM, SIB and displacement; `rsp` and `r12` bases need a SIB byte, `rbp` and `r13` a displacement.
    fn modrm(&mut self, reg: u8, rm: Rm) {
        let reg = (reg & 7) << 3;
        let (base, disp) = match rm {
            Rm::Reg(rm) => {
                self.byte(0xc0 | reg | (rm & 7));
                return;
            }
            Rm::Mem(base, disp) => (base & 7, disp),
        };
        let (mode, short) = match disp {
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
        self.rex(size, reg, rm, false);
        self.bytes(opcode);
        self.modrm(reg, rm);
    }

    fn sse(&mut self, prefix: Option<u8>, opcode: u8, reg: u8, rm: Rm) {
        self.op(prefix, Size::Dword, &[0x0f, opcode], reg, rm);
    }

    fn mov_rr(&mut self, dst: u8, src: u8) {
        self.op(None, Size::Qword, &[0x89], src, Rm::Reg(dst));
    }

    fn alu_rr(&mut self, op: Alu, dst: u8, src: u8) {
        if op == Alu::Mul {
            self.op(None, Size::Qword, &[0x0f, 0xaf], dst, Rm::Reg(src));
            return;
        }
        self.op(None, Size::Qword, &[group(op).1], src, Rm::Reg(dst));
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

    fn rel32(&mut self, label: Label) {
        self.fixups.push((self.code.len() as u32, label.0));
        self.imm32(0);
    }

    fn jcc_code(&mut self, code: u8, target: Label) {
        self.bytes(&[0x0f, 0x80 | code]);
        self.rel32(target);
    }

    fn setcc_code(&mut self, code: u8, dst: u8) {
        self.op(None, Size::Byte, &[0x0f, 0x90 | code], 0, Rm::Reg(dst));
    }

    fn movzx_byte(&mut self, dst: u8, src: u8) {
        self.rex(Size::Byte, dst, Rm::Reg(src), false);
        self.bytes(&[0x0f, 0xb6]);
        self.modrm(dst, Rm::Reg(src));
    }

    fn cmov(&mut self, cond: Cond, dst: u8, src: u8) {
        self.op(None, Size::Qword, &[0x0f, 0x40 | cc(cond)], dst, Rm::Reg(src));
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

    fn fop(&mut self, op: FAlu, dst: u8, src: Rm) {
        let opcode = match op {
            FAlu::Add => 0x58,
            FAlu::Mul => 0x59,
            FAlu::Sub => 0x5c,
            FAlu::Div => 0x5e,
        };
        self.sse(Some(0xf2), opcode, dst, src);
    }

    fn pooled(&mut self, opcode: u8, dst: u8, bits: u64) {
        let index = match self.pool.iter().position(|&known| known == bits) {
            Some(index) => index,
            None => {
                self.pool.push(bits);
                self.pool.len() - 1
            }
        };
        self.byte(0xf2);
        if dst >= 8 {
            self.byte(0x44);
        }
        self.bytes(&[0x0f, opcode, 0x05 | (dst & 7) << 3]);
        self.pool_fixups.push((self.code.len() as u32, index as u32));
        self.imm32(0);
    }

    fn movaps(&mut self, dst: u8, src: u8) {
        self.sse(None, 0x28, dst, Rm::Reg(src));
    }

    fn ucomisd(&mut self, left: u8, right: u8) {
        self.sse(Some(0x66), 0x2e, left, Rm::Reg(right));
    }

    fn movq_to_float(&mut self, dst: u8, src: u8) {
        self.op(Some(0x66), Size::Qword, &[0x0f, 0x6e], dst, Rm::Reg(src));
    }

    fn movq_to_int(&mut self, dst: u8, src: u8) {
        self.op(Some(0x66), Size::Qword, &[0x0f, 0x7e], src, Rm::Reg(dst));
    }
}

fn commutes(op: Alu) -> bool {
    !matches!(op, Alu::Sub)
}

impl Asm for X64 {
    const CTX: u8 = RBX;
    const SP: u8 = RSP;
    const SCRATCH: u8 = RAX;
    const FSCRATCH: u8 = FLOAT_SCRATCH;
    const DIVIDE_CLOBBERS: Option<u8> = Some(RDX);
    const OUTGOING: i32 = if cfg!(windows) { 48 } else { 0 };
    const REGISTERS: Registers = Registers {
        gp: &GP_ORDER,
        fp: &FP_ORDER,
        callee_saved: callee_saved(),
    };
    const CALLER_SAVED_FIRST: Registers = Registers {
        gp: &CALLER_GP_ORDER,
        fp: &FP_ORDER,
        callee_saved: callee_saved(),
    };
    const MAX_FRAME: u32 = i32::MAX as u32;

    fn new() -> Self {
        Self { code: Vec::with_capacity(4096), labels: Vec::new(), fixups: Vec::new(), pool: Vec::new(), pool_fixups: Vec::new() }
    }

    fn new_label(&mut self) -> Label {
        self.labels.push(None);
        Label(self.labels.len() as u32 - 1)
    }

    fn bind(&mut self, label: Label) {
        self.labels[label.0 as usize] = Some(self.code.len() as u32);
    }

    fn offset(&self) -> u32 {
        self.code.len() as u32
    }

    fn mov(&mut self, dst: u8, src: u8) {
        if dst != src {
            self.mov_rr(dst, src);
        }
    }

    /// `MOV r32, imm32` zero-extends; `MOV r/m64, imm32` sign-extends; otherwise `MOV r64, imm64`.
    fn mov_imm(&mut self, dst: u8, value: i64) {
        if let Ok(small) = u32::try_from(value) {
            self.rex(Size::Dword, 0, Rm::Reg(dst), false);
            self.byte(0xb8 | (dst & 7));
            self.bytes(&small.to_le_bytes());
            return;
        }
        if let Ok(signed) = i32::try_from(value) {
            self.op(None, Size::Qword, &[0xc7], 0, Rm::Reg(dst));
            self.imm32(signed);
            return;
        }
        self.rex(Size::Qword, 0, Rm::Reg(dst), false);
        self.byte(0xb8 | (dst & 7));
        self.bytes(&value.to_le_bytes());
    }

    fn load(&mut self, width: Width, dst: u8, base: u8, disp: i32) {
        let mem = Rm::Mem(base, disp);
        match width {
            Width::B => self.op(None, Size::Dword, &[0x0f, 0xb6], dst, mem),
            Width::W => self.op(None, Size::Dword, &[0x8b], dst, mem),
            Width::Q => self.op(None, Size::Qword, &[0x8b], dst, mem),
        }
    }

    fn store(&mut self, width: Width, src: u8, base: u8, disp: i32) {
        let mem = Rm::Mem(base, disp);
        match width {
            Width::B => {
                self.rex(Size::Byte, src, mem, (4..8).contains(&src));
                self.byte(0x88);
                self.modrm(src, mem);
            }
            Width::W => self.op(None, Size::Dword, &[0x89], src, mem),
            Width::Q => self.op(None, Size::Qword, &[0x89], src, mem),
        }
    }

    fn store_imm(&mut self, width: Width, base: u8, disp: i32, value: i64) {
        let mem = Rm::Mem(base, disp);
        match width {
            Width::B => {
                self.op(None, Size::Byte, &[0xc6], 0, mem);
                self.byte(value as u8);
            }
            Width::W => {
                self.op(None, Size::Dword, &[0xc7], 0, mem);
                self.imm32(value as i32);
            }
            Width::Q => match i32::try_from(value) {
                Ok(small) => {
                    self.op(None, Size::Qword, &[0xc7], 0, mem);
                    self.imm32(small);
                }
                Err(_) => {
                    self.mov_imm(R11, value);
                    self.store(Width::Q, R11, base, disp);
                }
            },
        }
    }

    fn load_f(&mut self, dst: u8, base: u8, disp: i32) {
        self.sse(Some(0xf2), 0x10, dst, Rm::Mem(base, disp));
    }

    fn store_f(&mut self, src: u8, base: u8, disp: i32) {
        self.sse(Some(0xf2), 0x11, src, Rm::Mem(base, disp));
    }

    fn fmov(&mut self, dst: u8, src: u8) {
        if dst != src {
            self.movaps(dst, src);
        }
    }

    fn bits_to_float(&mut self, dst: u8, src: u8) {
        self.movq_to_float(dst, src);
    }

    /// `MOVSD xmm, [rip+rel32]` from a pool after the code; zero is `XORPS`.
    fn fconst(&mut self, dst: u8, bits: u64) {
        if bits == 0 {
            self.sse(None, 0x57, dst, Rm::Reg(dst));
            return;
        }
        self.pooled(0x10, dst, bits);
    }

    /// The constant is the instruction's RIP-relative memory operand, so it needs no register.
    fn falu_const(&mut self, op: FAlu, dst: u8, left: u8, (bits, _scratch): (u64, u8)) {
        if dst != left {
            self.movaps(dst, left);
        }
        let opcode = match op {
            FAlu::Add => 0x58,
            FAlu::Mul => 0x59,
            FAlu::Sub => 0x5c,
            FAlu::Div => 0x5e,
        };
        self.pooled(opcode, dst, bits);
    }

    fn float_to_bits(&mut self, dst: u8, src: u8) {
        self.movq_to_int(dst, src);
    }

    fn lea(&mut self, dst: u8, base: u8, disp: i32) {
        self.op(None, Size::Qword, &[0x8d], dst, Rm::Mem(base, disp));
    }

    fn load_i32(&mut self, dst: u8, base: u8, disp: i32) {
        self.op(None, Size::Qword, &[0x63], dst, Rm::Mem(base, disp));
    }

    fn zext32(&mut self, dst: u8, src: u8) {
        self.op(None, Size::Dword, &[0x89], src, Rm::Reg(dst));
    }

    fn shl_imm(&mut self, dst: u8, src: u8, amount: u8) {
        self.mov(dst, src);
        self.op(None, Size::Qword, &[0xc1], 4, Rm::Reg(dst));
        self.byte(amount);
    }

    fn alu(&mut self, op: Alu, dst: u8, left: u8, right: u8) {
        if dst == left {
            self.alu_rr(op, dst, right);
        } else if dst == right && commutes(op) {
            self.alu_rr(op, dst, left);
        } else if dst == right {
            self.mov_rr(R11, right);
            self.mov_rr(dst, left);
            self.alu_rr(op, dst, R11);
        } else {
            self.mov_rr(dst, left);
            self.alu_rr(op, dst, right);
        }
    }

    fn alu_imm(&mut self, op: Alu, dst: u8, left: u8, value: i32) {
        if op == Alu::Mul {
            match i8::try_from(value) {
                Ok(short) => {
                    self.op(None, Size::Qword, &[0x6b], dst, Rm::Reg(left));
                    self.byte(short as u8);
                }
                Err(_) => {
                    self.op(None, Size::Qword, &[0x69], dst, Rm::Reg(left));
                    self.imm32(value);
                }
            }
            return;
        }
        self.mov(dst, left);
        self.group_imm(group(op), Size::Qword, Rm::Reg(dst), value);
    }

    fn neg(&mut self, dst: u8, src: u8) {
        self.mov(dst, src);
        self.op(None, Size::Qword, &[0xf7], 3, Rm::Reg(dst));
    }

    fn not(&mut self, dst: u8, src: u8) {
        self.mov(dst, src);
        self.op(None, Size::Qword, &[0xf7], 2, Rm::Reg(dst));
    }

    /// `IDIV` takes its dividend in `rdx:rax`, so `rdx` is kept on the stack around it.
    fn floor_div(&mut self, op: Division, dst: u8, x_reg: u8, y_reg: u8) {
        let done = self.new_label();
        let divisor = if y_reg == RDX { R11 } else { y_reg };
        self.push(RDX);
        self.mov_rr(R11, y_reg);
        self.mov_rr(RAX, x_reg);
        self.bytes(&[0x48, 0x99]);
        self.op(None, Size::Qword, &[0xf7], 7, Rm::Reg(divisor));
        self.op(None, Size::Qword, &[0x85], RDX, Rm::Reg(RDX));
        self.jcc_code(cc(Cond::Eq), done);
        match op {
            Division::Mod => {
                self.mov_rr(RAX, RDX);
                self.alu_rr(Alu::Xor, RAX, R11);
                self.jcc_code(0x9, done);
                self.alu_rr(Alu::Add, RDX, R11);
            }
            Division::IDiv => {
                if x_reg == RDX {
                    self.load(Width::Q, RDX, RSP, 0);
                } else {
                    self.mov_rr(RDX, x_reg);
                }
                self.alu_rr(Alu::Xor, RDX, R11);
                self.jcc_code(0x9, done);
                self.group_imm(group(Alu::Sub), Size::Qword, Rm::Reg(RAX), 1);
            }
        }
        self.bind(done);
        let result = if op == Division::Mod { RDX } else { RAX };
        if dst == RDX {
            self.mov_rr(R11, result);
            self.group_imm(group(Alu::Add), Size::Qword, Rm::Reg(RSP), 8);
            self.mov_rr(RDX, R11);
            return;
        }
        self.mov_rr(dst, result);
        self.pop(RDX);
    }

    /// With a positive divisor the remainder's sign alone says whether to round toward minus infinity.
    fn floor_div_by(&mut self, op: Division, dst: u8, x_reg: u8, divisor: i64, keep: bool) {
        let done = self.new_label();
        if keep {
            self.push(RDX);
        }
        self.mov_imm(R11, divisor);
        self.mov_rr(RAX, x_reg);
        self.bytes(&[0x48, 0x99]);
        self.op(None, Size::Qword, &[0xf7], 7, Rm::Reg(R11));
        self.op(None, Size::Qword, &[0x85], RDX, Rm::Reg(RDX));
        self.jcc_code(0x9, done);
        match op {
            Division::Mod => self.alu_rr(Alu::Add, RDX, R11),
            Division::IDiv => self.group_imm(group(Alu::Sub), Size::Qword, Rm::Reg(RAX), 1),
        }
        self.bind(done);
        let result = if op == Division::Mod { RDX } else { RAX };
        if dst != result {
            self.mov_rr(dst, result);
        }
        if keep {
            self.pop(RDX);
        }
    }

    /// `DIV` takes its dividend in `rdx:rax`, so `rdx` is kept on the stack around it.
    fn unsigned_rem(&mut self, dst: u8, x_reg: u8, y_reg: u8) {
        self.push(RDX);
        self.mov_rr(R11, y_reg);
        self.mov_rr(RAX, x_reg);
        self.alu_rr(Alu::Xor, RDX, RDX);
        self.op(None, Size::Qword, &[0xf7], 6, Rm::Reg(R11));
        if dst == RDX {
            self.mov_rr(R11, RDX);
            self.group_imm(group(Alu::Add), Size::Qword, Rm::Reg(RSP), 8);
            self.mov_rr(RDX, R11);
            return;
        }
        self.mov_rr(dst, RDX);
        self.pop(RDX);
    }

    fn add_mem(&mut self, dst: u8, base: u8, disp: i32) {
        self.op(None, Size::Qword, &[0x03], dst, Rm::Mem(base, disp));
    }

    fn sub_mem(&mut self, base: u8, disp: i32, value: i32) {
        self.group_imm(group(Alu::Sub), Size::Qword, Rm::Mem(base, disp), value);
    }

    fn cmp(&mut self, left: u8, right: u8) {
        self.op(None, Size::Qword, &[CMP.1], right, Rm::Reg(left));
    }

    fn cmp_imm(&mut self, left: u8, value: i32) {
        self.group_imm(CMP, Size::Qword, Rm::Reg(left), value);
    }

    fn cmp_imm64(&mut self, left: u8, value: i64) {
        match i32::try_from(value) {
            Ok(small) => self.cmp_imm(left, small),
            Err(_) => {
                self.mov_imm(R11, value);
                self.cmp(left, R11);
            }
        }
    }

    fn cmp_byte(&mut self, left: u8, value: u8) {
        self.op(None, Size::Byte, &[0x80], 7, Rm::Reg(left));
        self.byte(value);
    }

    fn cmp_mem_byte(&mut self, base: u8, disp: i32, value: u8) {
        self.op(None, Size::Byte, &[0x80], 7, Rm::Mem(base, disp));
        self.byte(value);
    }

    fn test_imm(&mut self, left: u8, mask: i32) {
        self.op(None, Size::Qword, &[0xf7], 0, Rm::Reg(left));
        self.imm32(mask);
    }

    fn set(&mut self, cond: Cond, dst: u8) {
        self.setcc_code(cc(cond), dst);
        self.movzx_byte(dst, dst);
    }

    fn select(&mut self, cond: Cond, dst: u8, yes: u8, no: u8) {
        if dst == yes {
            self.cmov(cond.negate(), dst, no);
        } else if dst == no {
            self.cmov(cond, dst, yes);
        } else {
            self.mov_rr(dst, no);
            self.cmov(cond, dst, yes);
        }
    }

    fn branch(&mut self, cond: Cond, target: Label) {
        self.jcc_code(cc(cond), target);
    }

    fn jump(&mut self, target: Label) {
        self.byte(0xe9);
        self.rel32(target);
    }

    fn falu(&mut self, op: FAlu, dst: u8, left: u8, right: u8) {
        if dst == left {
            self.fop(op, dst, Rm::Reg(right));
        } else if dst == right && matches!(op, FAlu::Add | FAlu::Mul) {
            self.fop(op, dst, Rm::Reg(left));
        } else if dst == right {
            self.group_imm(group(Alu::Sub), Size::Qword, Rm::Reg(RSP), 16);
            self.store_f(right, RSP, 0);
            self.movaps(dst, left);
            self.fop(op, dst, Rm::Mem(RSP, 0));
            self.group_imm(group(Alu::Add), Size::Qword, Rm::Reg(RSP), 16);
        } else {
            self.movaps(dst, left);
            self.fop(op, dst, Rm::Reg(right));
        }
    }

    fn fneg(&mut self, dst: u8, src: u8) {
        self.movq_to_int(R11, src);
        self.op(None, Size::Qword, &[0x0f, 0xba], 7, Rm::Reg(R11));
        self.byte(63);
        self.movq_to_float(dst, R11);
    }

    fn fsqrt(&mut self, dst: u8, src: u8) {
        self.sse(Some(0xf2), 0x51, dst, Rm::Reg(src));
    }

    /// `ROUNDSD` is SSE4.1, which `compile` requires; mode 9 rounds toward negative infinity without the precision
    /// exception.
    fn ffloor(&mut self, dst: u8, src: u8) {
        self.byte(0x66);
        self.rex(Size::Dword, dst, Rm::Reg(src), false);
        self.bytes(&[0x0f, 0x3a, 0x0b]);
        self.modrm(dst, Rm::Reg(src));
        self.byte(9);
    }

    /// `UCOMISD` sets PF on unordered pairs, so `a < b` is tested as `b > a` and equality also checks PF clear.
    fn fbranch(&mut self, cond: FCond, left: u8, right: u8, target: Label) {
        match cond {
            FCond::Lt => {
                self.ucomisd(right, left);
                self.jcc_code(cc(Cond::Above), target);
            }
            FCond::Le => {
                self.ucomisd(right, left);
                self.jcc_code(cc(Cond::AboveEq), target);
            }
            FCond::Eq => {
                let unordered = self.new_label();
                self.ucomisd(left, right);
                self.jcc_code(PARITY, unordered);
                self.jcc_code(cc(Cond::Eq), target);
                self.bind(unordered);
            }
        }
    }

    fn fbranch_not(&mut self, cond: FCond, left: u8, right: u8, target: Label) {
        match cond {
            FCond::Lt => {
                self.ucomisd(right, left);
                self.jcc_code(cc(Cond::BelowEq), target);
            }
            FCond::Le => {
                self.ucomisd(right, left);
                self.jcc_code(cc(Cond::Below), target);
            }
            FCond::Eq => {
                self.ucomisd(left, right);
                self.jcc_code(PARITY, target);
                self.jcc_code(cc(Cond::Ne), target);
            }
        }
    }

    fn fset(&mut self, cond: FCond, dst: u8, left: u8, right: u8) {
        match cond {
            FCond::Lt => {
                self.ucomisd(right, left);
                self.set(Cond::Above, dst);
            }
            FCond::Le => {
                self.ucomisd(right, left);
                self.set(Cond::AboveEq, dst);
            }
            FCond::Eq => {
                self.ucomisd(left, right);
                self.setcc_code(cc(Cond::Eq), dst);
                self.setcc_code(NO_PARITY, R11);
                self.op(None, Size::Byte, &[0x20], R11, Rm::Reg(dst));
                self.movzx_byte(dst, dst);
            }
        }
    }

    fn int_to_float(&mut self, dst: u8, src: u8) {
        self.sse(None, 0x57, dst, Rm::Reg(dst));
        self.op(Some(0xf2), Size::Qword, &[0x0f, 0x2a], dst, Rm::Reg(src));
    }

    fn call(&mut self, address: u64) {
        self.mov_imm(R11, address as i64);
        self.call_reg(R11);
    }

    fn call_reg(&mut self, target: u8) {
        self.op(None, Size::Dword, &[0xff], 2, Rm::Reg(target));
    }

    fn arg(index: usize) -> ArgLoc {
        match ARGS.get(index) {
            Some(&reg) => ArgLoc::Reg(reg),
            None => ArgLoc::Stack(32 + 8 * (index - ARGS.len()) as i32),
        }
    }

    fn float_arg(index: usize, float: usize) -> u8 {
        if cfg!(windows) { index as u8 } else { float as u8 }
    }

    fn result() -> u8 {
        RAX
    }

    fn float_result() -> u8 {
        0
    }

    fn dump_registers(&mut self) {
        self.group_imm(group(Alu::Sub), Size::Qword, Rm::Reg(RSP), DUMP + Self::OUTGOING);
        for reg in Self::REGISTERS.all().iter() {
            let at = Self::OUTGOING + reg.index() as i32 * 8;
            match reg.class {
                super::regs::Class::Gp => self.store(Width::Q, reg.hw, RSP, at),
                super::regs::Class::Fp => self.store_f(reg.hw, RSP, at),
            }
        }
    }

    fn trap(&mut self) {
        self.bytes(&[0x0f, 0x0b]);
    }

    fn prologue(&mut self, frame: u32, floats: &[(u8, u32)], pc_dst: u8, saved: &[u8]) -> unwind::Prologue {
        let mut prologue = unwind::Prologue { arch: unwind::Arch::X64, frame, ..unwind::Prologue::default() };
        self.push(RBP);
        prologue.save_frame = self.offset();
        self.mov_rr(RBP, RSP);
        prologue.set_frame = self.offset();
        for &reg in saved {
            self.push(reg);
            prologue.saves.push((self.offset(), reg));
        }
        self.op(None, Size::Qword, &[0x81], 5, Rm::Reg(RSP));
        self.imm32(frame as i32);
        prologue.alloc = self.offset();
        for &(reg, at) in floats {
            self.op(None, Size::Dword, &[0x0f, 0x29], reg, Rm::Mem(RSP, at as i32));
            prologue.float_saves.push((self.offset(), reg, at));
        }
        self.mov_rr(RBX, ARGS[0]);
        self.op(None, Size::Dword, &[0x89], ARGS[1], Rm::Reg(pc_dst));
        prologue
    }

    fn epilogue(&mut self, frame: u32, floats: &[(u8, u32)], code: u8, saved: &[u8]) {
        let pushed = 8 * saved.len() as i32;
        self.mov(RAX, code);
        if !floats.is_empty() {
            self.lea(RSP, RBP, -pushed - frame as i32);
            for &(reg, at) in floats {
                self.op(None, Size::Dword, &[0x0f, 0x28], reg, Rm::Mem(RSP, at as i32));
            }
        }
        self.lea(RSP, RBP, -pushed);
        for &reg in saved.iter().rev() {
            self.pop(reg);
        }
        self.pop(RBP);
        self.byte(0xc3);
    }

    fn tail_epilogue(&mut self, frame: u32, floats: &[(u8, u32)], saved: &[u8]) {
        let pushed = 8 * saved.len() as i32;
        if !floats.is_empty() {
            self.lea(RSP, RBP, -pushed - frame as i32);
            for &(reg, at) in floats {
                self.op(None, Size::Dword, &[0x0f, 0x28], reg, Rm::Mem(RSP, at as i32));
            }
        }
        self.lea(RSP, RBP, -pushed);
        for &reg in saved.iter().rev() {
            self.pop(reg);
        }
        self.pop(RBP);
        self.op(None, Size::Dword, &[0xff], 4, Rm::Reg(RAX));
    }

    fn finish(mut self) -> Option<Vec<u8>> {
        for &(at, label) in &self.fixups {
            let target = self.labels[label as usize]?;
            let relative = target as i32 - (at as i32 + 4);
            self.code[at as usize..at as usize + 4].copy_from_slice(&relative.to_le_bytes());
        }
        if self.pool.is_empty() {
            return Some(self.code);
        }
        while !self.code.len().is_multiple_of(8) {
            self.byte(0xcc);
        }
        let start = self.code.len() as i32;
        for &bits in &self.pool {
            self.code.extend_from_slice(&bits.to_le_bytes());
        }
        for &(at, index) in &self.pool_fixups {
            let relative = start + 8 * index as i32 - (at as i32 + 4);
            self.code[at as usize..at as usize + 4].copy_from_slice(&relative.to_le_bytes());
        }
        Some(self.code)
    }

    /// A multiple of 16 once the return address and the pushes are counted.
    fn saved_registers(used: RegSet) -> Vec<u8> {
        SAVED.iter().copied().filter(|&reg| reg == RBX || used.contains(PReg::gp(reg))).collect()
    }

    fn frame_size(needed: u32, saved: usize) -> u32 {
        let pushes = 2 + saved as u32;
        let total = (needed + 8 * pushes).next_multiple_of(16);
        total - 8 * pushes
    }

    /// `JMP rel32` (E9), relative to the end of the jump.
    fn raw_jump(distance: u32) -> Vec<u8> {
        let mut bytes = vec![0xe9];
        bytes.extend_from_slice(&distance.to_le_bytes());
        bytes
    }
}
