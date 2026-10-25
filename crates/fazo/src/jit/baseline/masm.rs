//! Macro assembler the Baseline templates are written against, modeled on V8's `BaselineAssembler`. Operations are
//! implemented once per architecture over asmkit; per-instruction operands are holes that each compile fills.

/// `Ctx` holds the `JitCtx` pointer; temporaries do not survive a call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Reg {
    Ctx,
    Base,
    Konst,
    T0,
    T1,
    T2,
    T3,
    T4,
    T5,
    Ret,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FReg {
    F0,
    F1,
    F2,
    F3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Imm {
    Fixed(i64),
    Hole { slot: u8, add: i32 },
}

impl Imm {
    pub(crate) const fn hole(slot: u8) -> Self {
        Self::Hole { slot, add: 0 }
    }

    pub(crate) const fn hole_plus(slot: u8, add: i32) -> Self {
        Self::Hole { slot, add }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Width {
    U8,
    U32,
    U64,
}

impl Width {
    pub(crate) fn bytes(self) -> u32 {
        match self {
            Self::U8 => 1,
            Self::U32 => 4,
            Self::U64 => 8,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Alu {
    Add,
    Sub,
    Mul,
    And,
    Or,
    Xor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FAlu {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Cond {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    BelowEq,
    Above,
    AboveEq,
}

/// Ordered float conditions: false when either operand is NaN.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FCond {
    Eq,
    Lt,
    Le,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Division {
    Mod,
    IDiv,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Target {
    Local(u32),
    Ext(u8),
}

/// Where a hole's value goes. The x86 fields are little-endian words; the AArch64 ones are bit
/// fields of the 32-bit instruction word at the hole's offset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Field {
    #[cfg_attr(all(target_arch = "aarch64", not(test)), expect(dead_code, reason = "only x86-64 templates have it"))]
    Le32,
    #[cfg_attr(all(target_arch = "aarch64", not(test)), expect(dead_code, reason = "only x86-64 templates have it"))]
    Le64,
    Rel32,
    #[cfg_attr(all(target_arch = "x86_64", not(test)), expect(dead_code, reason = "only AArch64 templates have it"))]
    Imm12 { scale: u8 },
    #[cfg_attr(all(target_arch = "x86_64", not(test)), expect(dead_code, reason = "only AArch64 templates have it"))]
    Imm16 { shift: u8 },
    Branch26,
    #[cfg_attr(all(target_arch = "x86_64", not(test)), expect(dead_code, reason = "only AArch64 templates have it"))]
    Branch19,
    #[cfg_attr(all(target_arch = "x86_64", not(test)), expect(dead_code, reason = "only AArch64 templates have it"))]
    Branch14,
    #[cfg_attr(all(target_arch = "x86_64", not(test)), expect(dead_code, reason = "only AArch64 templates have it"))]
    Adr,
}

impl Field {
    pub(crate) fn is_branch(self) -> bool {
        matches!(self, Self::Rel32 | Self::Branch26 | Self::Branch19 | Self::Branch14 | Self::Adr)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Hole {
    pub(crate) offset: u32,
    pub(crate) field: Field,
    pub(crate) slot: u8,
    pub(crate) add: i32,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Stencil {
    pub(crate) bytes: Vec<u8>,
    pub(crate) holes: Vec<Hole>,
}

impl Stencil {
    pub(crate) fn split(self) -> SplitStencil {
        let (branches, values) = self.holes.into_iter().partition(|hole| hole.field.is_branch());
        SplitStencil { bytes: self.bytes, values, branches }
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct SplitStencil {
    pub(crate) bytes: Vec<u8>,
    pub(crate) values: Vec<Hole>,
    pub(crate) branches: Vec<Hole>,
}

/// Templates clobber only `T0`..`T5`, `Ret` and the call argument registers.
pub(crate) trait Masm {
    fn new_label(&mut self) -> Target;
    fn bind(&mut self, label: Target);

    fn load(&mut self, width: Width, dst: Reg, base: Reg, disp: Imm);
    fn store(&mut self, width: Width, src: Reg, base: Reg, disp: Imm);
    fn store_imm(&mut self, width: Width, base: Reg, disp: Imm, value: i32);
    fn load_f(&mut self, dst: FReg, base: Reg, disp: Imm);
    fn store_f(&mut self, src: FReg, base: Reg, disp: Imm);

    fn mov(&mut self, dst: Reg, src: Reg);
    fn mov_imm(&mut self, dst: Reg, value: Imm);
    fn alu(&mut self, op: Alu, dst: Reg, src: Reg);
    fn alu_imm(&mut self, op: Alu, dst: Reg, value: i32);
    fn neg(&mut self, dst: Reg);
    fn not(&mut self, dst: Reg);
    fn shl_imm(&mut self, dst: Reg, amount: u8);
    /// Clobbers `T0`..`T2` besides `dst`; `y` must not be 0 or -1.
    fn floor_div(&mut self, op: Division, dst: Reg, x: Reg, y: Reg);

    fn cmp(&mut self, left: Reg, right: Reg);
    fn cmp_imm(&mut self, left: Reg, value: i32);
    fn branch(&mut self, cond: Cond, target: Target);
    fn jump(&mut self, target: Target);

    fn fop(&mut self, op: FAlu, dst: FReg, src: FReg);
    fn fbranch(&mut self, cond: FCond, left: FReg, right: FReg, target: Target);
    fn int_to_float(&mut self, dst: FReg, src: Reg);

    fn call(&mut self, address: u64);
    fn arg_imm(&mut self, index: u8, value: Imm);
    fn arg_reg(&mut self, index: u8, src: Reg);
    fn call_reg(&mut self, target: Reg);
    fn stack_address(&mut self, dst: Reg, offset: i32);
    fn callee_context(&self) -> i32;

    fn finish(self: Box<Self>) -> Stencil;
}

pub(crate) mod slot {
    pub(crate) const A: u8 = 0;
    pub(crate) const B: u8 = 1;
    pub(crate) const C: u8 = 2;
    pub(crate) const K: u8 = 3;
    pub(crate) const I: u8 = 4;
    pub(crate) const PC: u8 = 5;
    pub(crate) const X: u8 = 6;
    pub(crate) const Y: u8 = 7;
    pub(crate) const Z: u8 = 8;
    pub(crate) const W: u8 = 9;
    pub(crate) const V: u8 = 10;
    pub(crate) const U: u8 = 11;
    pub(crate) const COUNT: usize = 12;
}

pub(crate) mod ext {
    pub(crate) const NEXT: u8 = 0;
    pub(crate) const SKIP: u8 = 1;
    pub(crate) const TRUE: u8 = 2;
    pub(crate) const FALSE: u8 = 3;
    pub(crate) const SLOW: u8 = 4;
    pub(crate) const DISPATCH: u8 = 5;
    pub(crate) const TARGET: u8 = 6;
    pub(crate) const COLD: u8 = 7;
    pub(crate) const EXIT: u8 = 8;
    pub(crate) const COUNT: usize = 9;
}
