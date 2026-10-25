//! Follows the instruction layout of PUC Lua 5.5.1's lopcodes.h (MIT, Copyright (C) 1994-2025 Lua.org, PUC-Rio).
//! Binary chunks depend on this layout, so every field is encoded the way PUC encodes it.

use crate::{
    instruction::Instruction,
    opcode::Opcode,
};

const SIZE_OP: u32 = 7;
const POS_A: u32 = SIZE_OP;
const POS_K: u32 = POS_A + 8;
const POS_B: u32 = POS_K + 1;
const POS_C: u32 = POS_B + 8;
const POS_VB: u32 = POS_K + 1;
const POS_VC: u32 = POS_VB + 6;
const POS_BX: u32 = POS_K;
const POS_AX: u32 = POS_A;
const POS_SJ: u32 = POS_A;
const MAXARG_BX: u32 = (1 << 17) - 1;
const MAXARG_SJ: u32 = (1 << 25) - 1;
const OFFSET_SBX: i32 = (MAXARG_BX >> 1) as i32;
const OFFSET_SJ: i32 = (MAXARG_SJ >> 1) as i32;
const OFFSET_SC: i32 = 127;

fn abc(opcode: Opcode, a: u8, b: u8, c: u8, k: bool) -> u32 {
    (opcode as u32)
        | (u32::from(a) << POS_A)
        | (u32::from(k) << POS_K)
        | (u32::from(b) << POS_B)
        | (u32::from(c) << POS_C)
}

fn vabc(opcode: Opcode, a: u8, vb: u8, vc: u32, k: bool) -> u32 {
    (opcode as u32)
        | (u32::from(a) << POS_A)
        | (u32::from(k) << POS_K)
        | (u32::from(vb) << POS_VB)
        | (vc << POS_VC)
}

fn abx(opcode: Opcode, a: u8, bx: u32) -> u32 {
    (opcode as u32) | (u32::from(a) << POS_A) | (bx << POS_BX)
}

fn asbx(opcode: Opcode, a: u8, sbx: i32) -> u32 {
    abx(opcode, a, (sbx + OFFSET_SBX) as u32)
}

fn ax(opcode: Opcode, ax: u32) -> u32 {
    (opcode as u32) | (ax << POS_AX)
}

fn sj(opcode: Opcode, sj: i32, k: bool) -> u32 {
    (opcode as u32) | (((sj + OFFSET_SJ) as u32) << POS_SJ) | (u32::from(k) << POS_K)
}

fn signed_c(value: i32) -> u8 {
    (value + OFFSET_SC) as u8
}

pub(crate) fn encode(instruction: Instruction) -> u32 {
    use Instruction as I;
    use Opcode as O;
    match instruction {
        I::Move { a, b } => abc(O::Move, a, b, 0, false),
        I::LoadI { a, sbx } => asbx(O::LoadI, a, sbx),
        I::LoadF { a, sbx } => asbx(O::LoadF, a, sbx),
        I::LoadK { a, bx } => abx(O::LoadK, a, bx),
        I::LoadKx { a } => abx(O::LoadKx, a, 0),
        I::LoadFalse { a } => abc(O::LoadFalse, a, 0, 0, false),
        I::LFalseSkip { a } => abc(O::LFalseSkip, a, 0, 0, false),
        I::LoadTrue { a } => abc(O::LoadTrue, a, 0, 0, false),
        I::LoadNil { a, b } => abc(O::LoadNil, a, b, 0, false),
        I::GetUpval { a, b } => abc(O::GetUpval, a, b, 0, false),
        I::SetUpval { a, b } => abc(O::SetUpval, a, b, 0, false),
        I::GetTabUp { a, b, c } => abc(O::GetTabUp, a, b, c, false),
        I::GetTable { a, b, c } => abc(O::GetTable, a, b, c, false),
        I::GetI { a, b, c } => abc(O::GetI, a, b, c, false),
        I::GetField { a, b, c } => abc(O::GetField, a, b, c, false),
        I::SetTabUp { a, b, c, k } => abc(O::SetTabUp, a, b, c, k),
        I::SetTable { a, b, c, k } => abc(O::SetTable, a, b, c, k),
        I::SetI { a, b, c, k } => abc(O::SetI, a, b, c, k),
        I::SetField { a, b, c, k } => abc(O::SetField, a, b, c, k),
        I::NewTable { a, b, c, k } => vabc(O::NewTable, a, b, c, k),
        I::SelfOp { a, b, c } => abc(O::SelfOp, a, b, c, false),
        I::AddI { a, b, sc } => abc(O::AddI, a, b, signed_c(sc), false),
        I::AddK { a, b, c } => abc(O::AddK, a, b, c, false),
        I::SubK { a, b, c } => abc(O::SubK, a, b, c, false),
        I::MulK { a, b, c } => abc(O::MulK, a, b, c, false),
        I::ModK { a, b, c } => abc(O::ModK, a, b, c, false),
        I::PowK { a, b, c } => abc(O::PowK, a, b, c, false),
        I::DivK { a, b, c } => abc(O::DivK, a, b, c, false),
        I::IDivK { a, b, c } => abc(O::IDivK, a, b, c, false),
        I::BAndK { a, b, c } => abc(O::BAndK, a, b, c, false),
        I::BOrK { a, b, c } => abc(O::BOrK, a, b, c, false),
        I::BXorK { a, b, c } => abc(O::BXorK, a, b, c, false),
        I::ShrI { a, b, sc } => abc(O::ShrI, a, b, signed_c(sc), false),
        I::ShlI { a, b, sc } => abc(O::ShlI, a, b, signed_c(sc), false),
        I::Add { a, b, c } => abc(O::Add, a, b, c, false),
        I::Sub { a, b, c } => abc(O::Sub, a, b, c, false),
        I::Mul { a, b, c } => abc(O::Mul, a, b, c, false),
        I::Mod { a, b, c } => abc(O::Mod, a, b, c, false),
        I::Pow { a, b, c } => abc(O::Pow, a, b, c, false),
        I::Div { a, b, c } => abc(O::Div, a, b, c, false),
        I::IDiv { a, b, c } => abc(O::IDiv, a, b, c, false),
        I::BAnd { a, b, c } => abc(O::BAnd, a, b, c, false),
        I::BOr { a, b, c } => abc(O::BOr, a, b, c, false),
        I::BXor { a, b, c } => abc(O::BXor, a, b, c, false),
        I::Shl { a, b, c } => abc(O::Shl, a, b, c, false),
        I::Shr { a, b, c } => abc(O::Shr, a, b, c, false),
        I::MmBin { a, b, c } => abc(O::MmBin, a, b, c, false),
        I::MmBinI { a, sb, c, k } => abc(O::MmBinI, a, signed_c(sb), c, k),
        I::MmBinK { a, b, c, k } => abc(O::MmBinK, a, b, c, k),
        I::Unm { a, b } => abc(O::Unm, a, b, 0, false),
        I::BNot { a, b } => abc(O::BNot, a, b, 0, false),
        I::Not { a, b } => abc(O::Not, a, b, 0, false),
        I::Len { a, b } => abc(O::Len, a, b, 0, false),
        I::Concat { a, b } => abc(O::Concat, a, b, 0, false),
        I::Close { a } => abc(O::Close, a, 0, 0, false),
        I::Tbc { a } => abc(O::Tbc, a, 0, 0, false),
        I::Jmp { sj: offset } => sj(O::Jmp, offset, false),
        I::Eq { a, b, k } => abc(O::Eq, a, b, 0, k),
        I::Lt { a, b, k } => abc(O::Lt, a, b, 0, k),
        I::Le { a, b, k } => abc(O::Le, a, b, 0, k),
        I::EqK { a, b, k } => abc(O::EqK, a, b, 0, k),
        I::EqI { a, sb, c, k } => abc(O::EqI, a, signed_c(sb), c, k),
        I::LtI { a, sb, c, k } => abc(O::LtI, a, signed_c(sb), c, k),
        I::LeI { a, sb, c, k } => abc(O::LeI, a, signed_c(sb), c, k),
        I::GtI { a, sb, c, k } => abc(O::GtI, a, signed_c(sb), c, k),
        I::GeI { a, sb, c, k } => abc(O::GeI, a, signed_c(sb), c, k),
        I::Test { a, k } => abc(O::Test, a, 0, 0, k),
        I::TestSet { a, b, k } => abc(O::TestSet, a, b, 0, k),
        I::Call { a, b, c } => abc(O::Call, a, b, c, false),
        I::TailCall { a, b, c, k } => abc(O::TailCall, a, b, c, k),
        I::Return { a, b, c, k } => abc(O::Return, a, b, c, k),
        I::Return0 { a } => abc(O::Return0, a, 1, 0, false),
        I::Return1 { a } => abc(O::Return1, a, 2, 0, false),
        I::ForLoop { a, bx } => abx(O::ForLoop, a, bx),
        I::ForPrep { a, bx } => abx(O::ForPrep, a, bx),
        I::TForPrep { a, bx } => abx(O::TForPrep, a, bx),
        I::TForCall { a, c } => abc(O::TForCall, a, 0, c, false),
        I::TForLoop { a, bx } => abx(O::TForLoop, a, bx),
        I::SetList { a, b, c, k } => vabc(O::SetList, a, b, c, k),
        I::Closure { a, bx } => abx(O::Closure, a, bx),
        I::Vararg { a, b, c, k } => abc(O::Vararg, a, b, c, k),
        I::GetVarg { a, b, c } => abc(O::GetVarg, a, b, c, false),
        I::ErrNil { a, bx } => abx(O::ErrNil, a, bx),
        I::VarargPrep { a } => abc(O::VarargPrep, a, 0, 0, false),
        I::ExtraArg { ax: value } => ax(O::ExtraArg, value),
    }
}

/// Returns `None` for an opcode Lua 5.5.1 does not define.
pub(crate) fn decode(word: u32) -> Option<Instruction> {
    let opcode = *Opcode::ALL.get((word & 0x7f) as usize)?;
    Some(operands(opcode, word))
}

pub(crate) fn dispatch_words(code: &[Instruction]) -> crate::vm::quicken::Words {
    crate::vm::quicken::Words::of(code)
}

#[inline(always)]
pub(crate) fn jump_offset(word: u32) -> i32 {
    (word >> POS_SJ) as i32 - OFFSET_SJ
}

pub(crate) mod field {
    use super::{OFFSET_SBX, OFFSET_SC, OFFSET_SJ, POS_A, POS_B, POS_BX, POS_C, POS_K, POS_SJ};

    #[inline(always)]
    pub(crate) fn a(word: u32) -> u8 {
        ((word >> POS_A) & 0xff) as u8
    }

    #[inline(always)]
    pub(crate) fn k(word: u32) -> bool {
        (word >> POS_K) & 1 == 1
    }

    #[inline(always)]
    pub(crate) fn b(word: u32) -> u8 {
        ((word >> POS_B) & 0xff) as u8
    }

    #[inline(always)]
    pub(crate) fn c(word: u32) -> u8 {
        ((word >> POS_C) & 0xff) as u8
    }

    #[inline(always)]
    pub(crate) fn bx(word: u32) -> u32 {
        word >> POS_BX
    }

    #[inline(always)]
    pub(crate) fn sbx(word: u32) -> i32 {
        bx(word) as i32 - OFFSET_SBX
    }

    #[inline(always)]
    pub(crate) fn sb(word: u32) -> i32 {
        i32::from(b(word)) - OFFSET_SC
    }

    #[inline(always)]
    pub(crate) fn sc(word: u32) -> i32 {
        i32::from(c(word)) - OFFSET_SC
    }

    #[inline(always)]
    pub(crate) fn sj(word: u32) -> i32 {
        (word >> POS_SJ) as i32 - OFFSET_SJ
    }
}

#[cfg_attr(not(debug_assertions), inline(always))]
pub(crate) fn operands(opcode: Opcode, word: u32) -> Instruction {
    let a = ((word >> POS_A) & 0xff) as u8;
    let k = (word >> POS_K) & 1 == 1;
    let b = ((word >> POS_B) & 0xff) as u8;
    let c = ((word >> POS_C) & 0xff) as u8;
    let bx = word >> POS_BX;
    let sbx = bx as i32 - OFFSET_SBX;
    let sb = i32::from(b) - OFFSET_SC;
    let sc = i32::from(c) - OFFSET_SC;
    let vb = ((word >> POS_VB) & 0x3f) as u8;
    let vc = (word >> POS_VC) & 0x3ff;
    let jump = (word >> POS_SJ) as i32 - OFFSET_SJ;
    let extra = word >> POS_AX;
    use Instruction as I;
    use Opcode as O;
    match opcode {
        O::Move => I::Move { a, b },
        O::LoadI => I::LoadI { a, sbx },
        O::LoadF => I::LoadF { a, sbx },
        O::LoadK => I::LoadK { a, bx },
        O::LoadKx => I::LoadKx { a },
        O::LoadFalse => I::LoadFalse { a },
        O::LFalseSkip => I::LFalseSkip { a },
        O::LoadTrue => I::LoadTrue { a },
        O::LoadNil => I::LoadNil { a, b },
        O::GetUpval => I::GetUpval { a, b },
        O::SetUpval => I::SetUpval { a, b },
        O::GetTabUp => I::GetTabUp { a, b, c },
        O::GetTable => I::GetTable { a, b, c },
        O::GetI => I::GetI { a, b, c },
        O::GetField => I::GetField { a, b, c },
        O::SetTabUp => I::SetTabUp { a, b, c, k },
        O::SetTable => I::SetTable { a, b, c, k },
        O::SetI => I::SetI { a, b, c, k },
        O::SetField => I::SetField { a, b, c, k },
        O::NewTable => I::NewTable { a, b: vb, c: vc, k },
        O::SelfOp => I::SelfOp { a, b, c },
        O::AddI => I::AddI { a, b, sc },
        O::AddK => I::AddK { a, b, c },
        O::SubK => I::SubK { a, b, c },
        O::MulK => I::MulK { a, b, c },
        O::ModK => I::ModK { a, b, c },
        O::PowK => I::PowK { a, b, c },
        O::DivK => I::DivK { a, b, c },
        O::IDivK => I::IDivK { a, b, c },
        O::BAndK => I::BAndK { a, b, c },
        O::BOrK => I::BOrK { a, b, c },
        O::BXorK => I::BXorK { a, b, c },
        O::ShrI => I::ShrI { a, b, sc },
        O::ShlI => I::ShlI { a, b, sc },
        O::Add => I::Add { a, b, c },
        O::Sub => I::Sub { a, b, c },
        O::Mul => I::Mul { a, b, c },
        O::Mod => I::Mod { a, b, c },
        O::Pow => I::Pow { a, b, c },
        O::Div => I::Div { a, b, c },
        O::IDiv => I::IDiv { a, b, c },
        O::BAnd => I::BAnd { a, b, c },
        O::BOr => I::BOr { a, b, c },
        O::BXor => I::BXor { a, b, c },
        O::Shl => I::Shl { a, b, c },
        O::Shr => I::Shr { a, b, c },
        O::MmBin => I::MmBin { a, b, c },
        O::MmBinI => I::MmBinI { a, sb, c, k },
        O::MmBinK => I::MmBinK { a, b, c, k },
        O::Unm => I::Unm { a, b },
        O::BNot => I::BNot { a, b },
        O::Not => I::Not { a, b },
        O::Len => I::Len { a, b },
        O::Concat => I::Concat { a, b },
        O::Close => I::Close { a },
        O::Tbc => I::Tbc { a },
        O::Jmp => I::Jmp { sj: jump },
        O::Eq => I::Eq { a, b, k },
        O::Lt => I::Lt { a, b, k },
        O::Le => I::Le { a, b, k },
        O::EqK => I::EqK { a, b, k },
        O::EqI => I::EqI { a, sb, c, k },
        O::LtI => I::LtI { a, sb, c, k },
        O::LeI => I::LeI { a, sb, c, k },
        O::GtI => I::GtI { a, sb, c, k },
        O::GeI => I::GeI { a, sb, c, k },
        O::Test => I::Test { a, k },
        O::TestSet => I::TestSet { a, b, k },
        O::Call => I::Call { a, b, c },
        O::TailCall => I::TailCall { a, b, c, k },
        O::Return => I::Return { a, b, c, k },
        O::Return0 => I::Return0 { a },
        O::Return1 => I::Return1 { a },
        O::ForLoop => I::ForLoop { a, bx },
        O::ForPrep => I::ForPrep { a, bx },
        O::TForPrep => I::TForPrep { a, bx },
        O::TForCall => I::TForCall { a, c },
        O::TForLoop => I::TForLoop { a, bx },
        O::SetList => I::SetList { a, b: vb, c: vc, k },
        O::Closure => I::Closure { a, bx },
        O::Vararg => I::Vararg { a, b, c, k },
        O::GetVarg => I::GetVarg { a, b, c },
        O::ErrNil => I::ErrNil { a, bx },
        O::VarargPrep => I::VarargPrep { a },
        O::ExtraArg => I::ExtraArg { ax: extra },
    }
}

#[cfg(test)]
mod tests {
    use super::{decode, encode};
    use crate::instruction::Instruction;

    #[test]
    fn every_encoding_decodes_to_the_same_instruction() {
        let samples = [
            Instruction::Move { a: 3, b: 7 },
            Instruction::LoadI { a: 1, sbx: -65535 },
            Instruction::LoadK { a: 2, bx: 131071 },
            Instruction::AddI { a: 0, b: 1, sc: -127 },
            Instruction::ShlI { a: 0, b: 1, sc: 128 },
            Instruction::MmBinI { a: 4, sb: -3, c: 6, k: true },
            Instruction::EqI { a: 9, sb: 5, c: 1, k: false },
            Instruction::Jmp { sj: -16_777_215 },
            Instruction::NewTable { a: 2, b: 5, c: 1023, k: true },
            Instruction::SetList { a: 0, b: 0, c: 512, k: true },
            Instruction::Return0 { a: 4 },
            Instruction::ExtraArg { ax: 33_554_431 },
        ];
        for sample in samples {
            assert_eq!(decode(encode(sample)), Some(sample));
        }
    }

    #[test]
    fn an_opcode_past_the_last_one_is_rejected() {
        assert_eq!(decode(85), None);
    }
}
