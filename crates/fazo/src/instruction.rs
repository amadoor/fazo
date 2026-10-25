/// Variant order matches Lua's `LUA_OP*` codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Mod,
    Pow,
    Div,
    IDiv,
    BAnd,
    BOr,
    BXor,
    Shl,
    Shr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum UnaryOp {
    Minus,
    BNot,
    Not,
    Len,
}

/// Follows the operand layout of `lopcodes.h`; signed operands are stored without their excess-K encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Instruction {
    Move { a: u8, b: u8 },
    LoadI { a: u8, sbx: i32 },
    LoadF { a: u8, sbx: i32 },
    LoadK { a: u8, bx: u32 },
    LoadKx { a: u8 },
    LoadFalse { a: u8 },
    LFalseSkip { a: u8 },
    LoadTrue { a: u8 },
    LoadNil { a: u8, b: u8 },
    GetUpval { a: u8, b: u8 },
    SetUpval { a: u8, b: u8 },
    GetTabUp { a: u8, b: u8, c: u8 },
    GetTable { a: u8, b: u8, c: u8 },
    GetI { a: u8, b: u8, c: u8 },
    GetField { a: u8, b: u8, c: u8 },
    SetTabUp { a: u8, b: u8, c: u8, k: bool },
    SetTable { a: u8, b: u8, c: u8, k: bool },
    SetI { a: u8, b: u8, c: u8, k: bool },
    SetField { a: u8, b: u8, c: u8, k: bool },
    NewTable { a: u8, b: u8, c: u32, k: bool },
    SelfOp { a: u8, b: u8, c: u8 },
    AddI { a: u8, b: u8, sc: i32 },
    AddK { a: u8, b: u8, c: u8 },
    SubK { a: u8, b: u8, c: u8 },
    MulK { a: u8, b: u8, c: u8 },
    ModK { a: u8, b: u8, c: u8 },
    PowK { a: u8, b: u8, c: u8 },
    DivK { a: u8, b: u8, c: u8 },
    IDivK { a: u8, b: u8, c: u8 },
    BAndK { a: u8, b: u8, c: u8 },
    BOrK { a: u8, b: u8, c: u8 },
    BXorK { a: u8, b: u8, c: u8 },
    ShrI { a: u8, b: u8, sc: i32 },
    ShlI { a: u8, b: u8, sc: i32 },
    Add { a: u8, b: u8, c: u8 },
    Sub { a: u8, b: u8, c: u8 },
    Mul { a: u8, b: u8, c: u8 },
    Mod { a: u8, b: u8, c: u8 },
    Pow { a: u8, b: u8, c: u8 },
    Div { a: u8, b: u8, c: u8 },
    IDiv { a: u8, b: u8, c: u8 },
    BAnd { a: u8, b: u8, c: u8 },
    BOr { a: u8, b: u8, c: u8 },
    BXor { a: u8, b: u8, c: u8 },
    Shl { a: u8, b: u8, c: u8 },
    Shr { a: u8, b: u8, c: u8 },
    MmBin { a: u8, b: u8, c: u8 },
    MmBinI { a: u8, sb: i32, c: u8, k: bool },
    MmBinK { a: u8, b: u8, c: u8, k: bool },
    Unm { a: u8, b: u8 },
    BNot { a: u8, b: u8 },
    Not { a: u8, b: u8 },
    Len { a: u8, b: u8 },
    Concat { a: u8, b: u8 },
    Close { a: u8 },
    Tbc { a: u8 },
    Jmp { sj: i32 },
    Eq { a: u8, b: u8, k: bool },
    Lt { a: u8, b: u8, k: bool },
    Le { a: u8, b: u8, k: bool },
    EqK { a: u8, b: u8, k: bool },
    EqI { a: u8, sb: i32, c: u8, k: bool },
    LtI { a: u8, sb: i32, c: u8, k: bool },
    LeI { a: u8, sb: i32, c: u8, k: bool },
    GtI { a: u8, sb: i32, c: u8, k: bool },
    GeI { a: u8, sb: i32, c: u8, k: bool },
    Test { a: u8, k: bool },
    TestSet { a: u8, b: u8, k: bool },
    Call { a: u8, b: u8, c: u8 },
    TailCall { a: u8, b: u8, c: u8, k: bool },
    Return { a: u8, b: u8, c: u8, k: bool },
    Return0 { a: u8 },
    Return1 { a: u8 },
    ForLoop { a: u8, bx: u32 },
    ForPrep { a: u8, bx: u32 },
    TForPrep { a: u8, bx: u32 },
    TForCall { a: u8, c: u8 },
    TForLoop { a: u8, bx: u32 },
    SetList { a: u8, b: u8, c: u32, k: bool },
    Closure { a: u8, bx: u32 },
    Vararg { a: u8, b: u8, c: u8, k: bool },
    GetVarg { a: u8, b: u8, c: u8 },
    ErrNil { a: u8, bx: u32 },
    VarargPrep { a: u8 },
    ExtraArg { ax: u32 },
}

impl Instruction {
    /// Follows `testTMode` in lopcodes.c.
    pub fn is_test(self) -> bool {
        matches!(
            self,
            Instruction::Eq { .. }
                | Instruction::Lt { .. }
                | Instruction::Le { .. }
                | Instruction::EqK { .. }
                | Instruction::EqI { .. }
                | Instruction::LtI { .. }
                | Instruction::LeI { .. }
                | Instruction::GtI { .. }
                | Instruction::GeI { .. }
                | Instruction::Test { .. }
                | Instruction::TestSet { .. }
        )
    }
}
