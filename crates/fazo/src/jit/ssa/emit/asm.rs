//! Per-architecture macro assembler, shaped after V8's `MacroAssembler` for Maglev's per-node code (src/maglev/maglev-ir.h, `GenerateCode`).

use super::regs::Registers;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Label(pub(crate) u32);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Width {
    B,
    W,
    Q,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Alu {
    Add,
    Sub,
    Mul,
    And,
    Or,
    Xor,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum FAlu {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Cond {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Below,
    BelowEq,
    Above,
    AboveEq,
}

impl Cond {
    pub(crate) fn negate(self) -> Self {
        match self {
            Self::Eq => Self::Ne,
            Self::Ne => Self::Eq,
            Self::Lt => Self::Ge,
            Self::Le => Self::Gt,
            Self::Gt => Self::Le,
            Self::Ge => Self::Lt,
            Self::Below => Self::AboveEq,
            Self::BelowEq => Self::Above,
            Self::Above => Self::BelowEq,
            Self::AboveEq => Self::Below,
        }
    }
}

/// Ordered float conditions: false when either operand is NaN.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum FCond {
    Eq,
    Lt,
    Le,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Division {
    Mod,
    IDiv,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ArgLoc {
    Reg(u8),
    Stack(i32),
}

pub(crate) trait Asm {
    /// The context register, preserved by calls.
    const CTX: u8;
    const SP: u8;
    const SCRATCH: u8;
    const FSCRATCH: u8;
    /// Bytes below the frame's own areas that outgoing calls use.
    const OUTGOING: i32;
    const REGISTERS: Registers;
    /// Without loops, values seldom outlive a call, so a register the prologue need not save is cheapest.
    const CALLER_SAVED_FIRST: Registers;
    const MAX_FRAME: u32;
    /// The register a hardware divide overwrites besides the scratch register, when it is allocatable.
    const DIVIDE_CLOBBERS: Option<u8>;

    fn new() -> Self;
    fn new_label(&mut self) -> Label;
    fn bind(&mut self, label: Label);
    fn offset(&self) -> u32;

    fn mov(&mut self, dst: u8, src: u8);
    fn mov_imm(&mut self, dst: u8, value: i64);
    fn load(&mut self, width: Width, dst: u8, base: u8, disp: i32);
    fn store(&mut self, width: Width, src: u8, base: u8, disp: i32);
    fn store_imm(&mut self, width: Width, base: u8, disp: i32, value: i64);
    fn load_f(&mut self, dst: u8, base: u8, disp: i32);
    fn store_f(&mut self, src: u8, base: u8, disp: i32);
    /// Copies 16 bytes through the vector register `via`.
    fn copy16(&mut self, via: u8, to: (u8, i32), from: (u8, i32));
    fn fmov(&mut self, dst: u8, src: u8);
    fn bits_to_float(&mut self, dst: u8, src: u8);
    fn fconst(&mut self, dst: u8, bits: u64) {
        self.mov_imm(Self::SCRATCH, bits as i64);
        self.bits_to_float(dst, Self::SCRATCH);
    }
    fn float_to_bits(&mut self, dst: u8, src: u8);
    fn lea(&mut self, dst: u8, base: u8, disp: i32);
    fn load_i32(&mut self, dst: u8, base: u8, disp: i32);
    fn zext32(&mut self, dst: u8, src: u8);
    fn shl_imm(&mut self, dst: u8, src: u8, amount: u8);

    fn alu(&mut self, op: Alu, dst: u8, left: u8, right: u8);
    fn alu_imm(&mut self, op: Alu, dst: u8, left: u8, value: i32);
    fn neg(&mut self, dst: u8, src: u8);
    fn not(&mut self, dst: u8, src: u8);
    /// Follows lvm.c's `luaV_mod` and `luaV_idiv`; `y` is neither 0 nor -1.
    fn floor_div(&mut self, op: Division, dst: u8, x: u8, y: u8);
    /// `floor_div` by a positive constant; `keep` preserves `DIVIDE_CLOBBERS` around the divide.
    fn floor_div_by(&mut self, op: Division, dst: u8, x: u8, divisor: i64, keep: bool);
    /// `y` is not zero.
    fn unsigned_rem(&mut self, dst: u8, x: u8, y: u8);

    /// Leaves the flags `cmp old, value` would.
    fn sub_mem(&mut self, base: u8, disp: i32, value: i32);
    fn add_mem(&mut self, dst: u8, base: u8, disp: i32);
    fn cmp(&mut self, left: u8, right: u8);
    fn cmp_imm(&mut self, left: u8, value: i32);
    fn cmp_imm64(&mut self, left: u8, value: i64);
    fn cmp_byte(&mut self, left: u8, value: u8);
    fn cmp_mem_byte(&mut self, base: u8, disp: i32, value: u8);
    fn cmp_mem(&mut self, left: u8, base: u8, disp: i32);
    /// `Width::W` compares the sign-extended word.
    fn cmp_mem_imm(&mut self, width: Width, base: u8, disp: i32, value: i32);
    /// Sets the flags as `cmp` against zero would for `left & mask`.
    fn test_imm(&mut self, left: u8, mask: i32);
    fn set(&mut self, cond: Cond, dst: u8);
    fn select(&mut self, cond: Cond, dst: u8, yes: u8, no: u8);
    fn branch(&mut self, cond: Cond, target: Label);
    fn jump(&mut self, target: Label);

    fn falu(&mut self, op: FAlu, dst: u8, left: u8, right: u8);
    fn falu_const(&mut self, op: FAlu, dst: u8, left: u8, (bits, scratch): (u64, u8)) {
        self.fconst(scratch, bits);
        self.falu(op, dst, left, scratch);
    }
    fn fneg(&mut self, dst: u8, src: u8);
    fn fsqrt(&mut self, dst: u8, src: u8);
    fn ffloor(&mut self, dst: u8, src: u8);
    fn fbranch(&mut self, cond: FCond, left: u8, right: u8, target: Label);
    /// Taken when the ordered condition is false, NaN operands included.
    fn fbranch_not(&mut self, cond: FCond, left: u8, right: u8, target: Label);
    fn fset(&mut self, cond: FCond, dst: u8, left: u8, right: u8);
    fn int_to_float(&mut self, dst: u8, src: u8);

    fn call(&mut self, address: u64);
    fn call_reg(&mut self, target: u8);
    fn arg(index: usize) -> ArgLoc;
    /// `index` counts all arguments; `float` counts only the float ones.
    fn float_arg(index: usize, float: usize) -> u8;
    fn result() -> u8;
    fn float_result() -> u8;

    /// Stores each allocatable register at `OUTGOING + index * 8` in the dump the runtime reads.
    fn dump_registers(&mut self);
    fn trap(&mut self);

    /// `floats` are callee-saved float registers, stored at their offsets from the stack pointer.
    fn prologue(&mut self, frame: u32, floats: &[(u8, u32)], pc_dst: Option<u8>, saved: &[u8]) -> super::unwind::Prologue;
    /// Returns `code` to the caller from any stack depth inside the frame.
    fn epilogue(&mut self, frame: u32, floats: &[(u8, u32)], code: u8, saved: &[u8]);
    fn tail_epilogue(&mut self, frame: u32, floats: &[(u8, u32)], saved: &[u8]);
    fn saved_registers(used: super::regs::RegSet) -> Vec<u8>;
    fn finish(self) -> Option<Vec<u8>>;
    fn frame_size(needed: u32, saved: usize) -> u32;
    /// An unconditional jump placed right before the code it jumps into, `distance` bytes past the jump's end.
    fn raw_jump(distance: u32) -> Vec<u8>;
}

pub(crate) const DUMP: i32 = 64 * 8;
