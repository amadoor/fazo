//! Follows PUC Lua 5.5.1's lcode.c (MIT, Copyright (C) 1994-2025 Lua.org, PUC-Rio).
//! Emitted instructions, constant indexes and registers must match PUC's, so each step keeps PUC's order.

use super::func::{ExpDesc, ExpKind, FuncState, Literal, NO_JUMP, NO_REG, Res};
use crate::{
    instruction::{ArithOp, Instruction, UnaryOp},
    number::{self, Number, Rounding},
};

const MAX_INDEX_RK: u32 = 255;
const MAX_ARG_B: usize = 255;
const MAX_ARG_C: i64 = 255;
const OFFSET_SC: i64 = 127;
pub(super) const MAX_ARG_BX: i64 = (1 << 17) - 1;
const OFFSET_SBX: i64 = MAX_ARG_BX >> 1;
const SETLIST_BLOCK: u32 = 1 << 10;

/// Variant order matches `BinOpr` in lcode.h.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BinOpr {
    Add,
    Sub,
    Mul,
    Mod,
    Pow,
    Div,
    IntDiv,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    Concat,
    Eq,
    Lt,
    Le,
    NotEq,
    Gt,
    Ge,
    And,
    Or,
}

pub(super) fn arith_of(op: BinOpr) -> Option<ArithOp> {
    use BinOpr as B;
    Some(match op {
        B::Add => ArithOp::Add,
        B::Sub => ArithOp::Sub,
        B::Mul => ArithOp::Mul,
        B::Mod => ArithOp::Mod,
        B::Pow => ArithOp::Pow,
        B::Div => ArithOp::Div,
        B::IntDiv => ArithOp::IDiv,
        B::BitAnd => ArithOp::BAnd,
        B::BitOr => ArithOp::BOr,
        B::BitXor => ArithOp::BXor,
        B::Shl => ArithOp::Shl,
        B::Shr => ArithOp::Shr,
        _ => return None,
    })
}

/// Event numbers follow `ltm.h`.
fn event_of(op: ArithOp) -> u8 {
    6 + op as u8
}

fn fits_sc(value: i64) -> bool {
    (-OFFSET_SC..=MAX_ARG_C - OFFSET_SC).contains(&value)
}

fn fits_bx(value: i64) -> bool {
    (-OFFSET_SBX..=MAX_ARG_BX - OFFSET_SBX).contains(&value)
}

pub(super) fn has_jumps(expression: &ExpDesc) -> bool {
    expression.t != expression.f
}

/// Follows `tonumeral` in lcode.c.
fn numeral(expression: &ExpDesc) -> Option<Number> {
    if has_jumps(expression) {
        return None;
    }
    match expression.kind {
        ExpKind::Integer(value) => Some(Number::Integer(value)),
        ExpKind::Float(value) => Some(Number::Float(value)),
        _ => None,
    }
}

/// Follows `isSCnumber` in lcode.c, which sets `isfloat` for floats that do not fit.
fn sc_number(expression: &ExpDesc, isfloat: &mut bool) -> Option<i32> {
    let value = match expression.kind {
        ExpKind::Integer(value) => value,
        ExpKind::Float(value) => {
            let integer = number::float_to_integer(value, Rounding::Exact)?;
            *isfloat = true;
            integer
        }
        _ => return None,
    };
    if !has_jumps(expression) && fits_sc(value) {
        Some(value as i32)
    } else {
        None
    }
}

fn is_kint(expression: &ExpDesc) -> bool {
    matches!(expression.kind, ExpKind::Integer(_)) && !has_jumps(expression)
}

fn is_sc_int(expression: &ExpDesc) -> bool {
    match expression.kind {
        ExpKind::Integer(value) => !has_jumps(expression) && fits_sc(value),
        _ => false,
    }
}

fn negate_flag(instruction: &mut Instruction) {
    match instruction {
        Instruction::Eq { k, .. }
        | Instruction::Lt { k, .. }
        | Instruction::Le { k, .. }
        | Instruction::EqK { k, .. }
        | Instruction::EqI { k, .. }
        | Instruction::LtI { k, .. }
        | Instruction::LeI { k, .. }
        | Instruction::GtI { k, .. }
        | Instruction::GeI { k, .. }
        | Instruction::Test { k, .. }
        | Instruction::TestSet { k, .. } => *k = !*k,
        _ => {}
    }
}

/// Only instructions that `luaK_dischargevars` made relocatable carry an A field to set.
fn set_a(instruction: &mut Instruction, reg: u8) {
    match instruction {
        Instruction::GetUpval { a, .. }
        | Instruction::GetTabUp { a, .. }
        | Instruction::GetTable { a, .. }
        | Instruction::GetI { a, .. }
        | Instruction::GetField { a, .. }
        | Instruction::GetVarg { a, .. }
        | Instruction::AddI { a, .. }
        | Instruction::AddK { a, .. }
        | Instruction::SubK { a, .. }
        | Instruction::MulK { a, .. }
        | Instruction::ModK { a, .. }
        | Instruction::PowK { a, .. }
        | Instruction::DivK { a, .. }
        | Instruction::IDivK { a, .. }
        | Instruction::BAndK { a, .. }
        | Instruction::BOrK { a, .. }
        | Instruction::BXorK { a, .. }
        | Instruction::ShrI { a, .. }
        | Instruction::ShlI { a, .. }
        | Instruction::Add { a, .. }
        | Instruction::Sub { a, .. }
        | Instruction::Mul { a, .. }
        | Instruction::Mod { a, .. }
        | Instruction::Pow { a, .. }
        | Instruction::Div { a, .. }
        | Instruction::IDiv { a, .. }
        | Instruction::BAnd { a, .. }
        | Instruction::BOr { a, .. }
        | Instruction::BXor { a, .. }
        | Instruction::Shl { a, .. }
        | Instruction::Shr { a, .. }
        | Instruction::Unm { a, .. }
        | Instruction::BNot { a, .. }
        | Instruction::Not { a, .. }
        | Instruction::Len { a, .. }
        | Instruction::Closure { a, .. }
        | Instruction::Vararg { a, .. } => *a = reg,
        _ => {}
    }
}

fn arith_register(op: ArithOp, b: u8, c: u8) -> Instruction {
    match op {
        ArithOp::Add => Instruction::Add { a: 0, b, c },
        ArithOp::Sub => Instruction::Sub { a: 0, b, c },
        ArithOp::Mul => Instruction::Mul { a: 0, b, c },
        ArithOp::Mod => Instruction::Mod { a: 0, b, c },
        ArithOp::Pow => Instruction::Pow { a: 0, b, c },
        ArithOp::Div => Instruction::Div { a: 0, b, c },
        ArithOp::IDiv => Instruction::IDiv { a: 0, b, c },
        ArithOp::BAnd => Instruction::BAnd { a: 0, b, c },
        ArithOp::BOr => Instruction::BOr { a: 0, b, c },
        ArithOp::BXor => Instruction::BXor { a: 0, b, c },
        ArithOp::Shl => Instruction::Shl { a: 0, b, c },
        ArithOp::Shr => Instruction::Shr { a: 0, b, c },
    }
}

fn arith_constant(op: ArithOp, b: u8, k: u8) -> Instruction {
    match op {
        ArithOp::Add => Instruction::AddK { a: 0, b, c: k },
        ArithOp::Sub => Instruction::SubK { a: 0, b, c: k },
        ArithOp::Mul => Instruction::MulK { a: 0, b, c: k },
        ArithOp::Mod => Instruction::ModK { a: 0, b, c: k },
        ArithOp::Pow => Instruction::PowK { a: 0, b, c: k },
        ArithOp::Div => Instruction::DivK { a: 0, b, c: k },
        ArithOp::IDiv => Instruction::IDivK { a: 0, b, c: k },
        ArithOp::BAnd => Instruction::BAndK { a: 0, b, c: k },
        ArithOp::BOr => Instruction::BOrK { a: 0, b, c: k },
        ArithOp::BXor => Instruction::BXorK { a: 0, b, c: k },
        ArithOp::Shl | ArithOp::Shr => unreachable!("shifts by a constant use SHLI and SHRI"),
    }
}

fn immediate_arith(op: ArithOp, b: u8, sc: i32) -> Instruction {
    match op {
        ArithOp::Add => Instruction::AddI { a: 0, b, sc },
        ArithOp::Shr => Instruction::ShrI { a: 0, b, sc },
        ArithOp::Shl => Instruction::ShlI { a: 0, b, sc },
        _ => unreachable!("only addition and shifts take an immediate"),
    }
}

#[derive(Clone, Copy)]
enum Right {
    Register(u8),
    Constant(u8),
    /// `sb` differs from `sc` when a subtraction became an addition.
    Immediate { op: ArithOp, sc: i32, sb: i32 },
}

impl FuncState {
    pub(super) fn discharge_vars(&mut self, expression: &mut ExpDesc) {
        match expression.kind {
            ExpKind::Const { value, .. } => expression.kind = value.kind(),
            ExpKind::VarargVar { reg, .. } => {
                self.vararg_table = true;
                expression.kind = ExpKind::NonReloc(reg);
            }
            ExpKind::Local { reg, .. } => expression.kind = ExpKind::NonReloc(reg),
            ExpKind::Upval(upval) => {
                let pc = self.emit(Instruction::GetUpval { a: 0, b: upval });
                expression.kind = ExpKind::Reloc(pc);
            }
            ExpKind::IndexUp { table, key } => {
                let pc = self.emit(Instruction::GetTabUp { a: 0, b: table, c: key });
                expression.kind = ExpKind::Reloc(pc);
            }
            ExpKind::IndexInt { table, index } => {
                self.free_register(table);
                let pc = self.emit(Instruction::GetI { a: 0, b: table, c: index });
                expression.kind = ExpKind::Reloc(pc);
            }
            ExpKind::IndexStr { table, key } => {
                self.free_register(table);
                let pc = self.emit(Instruction::GetField { a: 0, b: table, c: key });
                expression.kind = ExpKind::Reloc(pc);
            }
            ExpKind::Indexed { table, key } => {
                self.free_registers(table, key);
                let pc = self.emit(Instruction::GetTable { a: 0, b: table, c: key });
                expression.kind = ExpKind::Reloc(pc);
            }
            ExpKind::VarargIndexed { table, key } => {
                self.free_registers(table, key);
                let pc = self.emit(Instruction::GetVarg { a: 0, b: table, c: key });
                expression.kind = ExpKind::Reloc(pc);
            }
            ExpKind::Call(_) | ExpKind::Vararg(_) => self.set_one_return(expression),
            _ => {}
        }
    }

    pub(super) fn set_one_return(&mut self, expression: &mut ExpDesc) {
        match expression.kind {
            ExpKind::Call(pc) => {
                let Instruction::Call { a, .. } = self.code[pc] else {
                    unreachable!("a call expression names a call instruction");
                };
                expression.kind = ExpKind::NonReloc(a);
            }
            ExpKind::Vararg(pc) => {
                if let Instruction::Vararg { c, .. } = &mut self.code[pc] {
                    *c = 2;
                }
                expression.kind = ExpKind::Reloc(pc);
            }
            _ => {}
        }
    }

    pub(super) fn set_multiple_returns(&mut self, expression: &mut ExpDesc) -> Res {
        self.set_returns(expression, -1)
    }

    pub(super) fn set_returns(&mut self, expression: &mut ExpDesc, count: i32) -> Res {
        if i64::from(count) + 1 > MAX_ARG_C {
            return Err(self.limit("multiple results", MAX_ARG_C as usize));
        }
        let encoded = (count + 1) as u8;
        match expression.kind {
            ExpKind::Call(pc) => {
                if let Instruction::Call { c, .. } = &mut self.code[pc] {
                    *c = encoded;
                }
            }
            ExpKind::Vararg(pc) => {
                let reg = self.freereg;
                if let Instruction::Vararg { a, c, .. } = &mut self.code[pc] {
                    *a = reg;
                    *c = encoded;
                }
                self.reserve_registers(1)?;
            }
            _ => {}
        }
        Ok(())
    }

    /// Follows `luaK_codek` in lcode.c.
    fn code_constant(&mut self, reg: u8, k: u32) {
        if i64::from(k) <= MAX_ARG_BX {
            self.emit(Instruction::LoadK { a: reg, bx: k });
        } else {
            self.emit(Instruction::LoadKx { a: reg });
            self.emit(Instruction::ExtraArg { ax: k });
        }
    }

    /// Follows `luaK_int` in lcode.c.
    pub(super) fn load_integer(&mut self, reg: u8, value: i64) {
        if fits_bx(value) {
            self.emit(Instruction::LoadI { a: reg, sbx: value as i32 });
        } else {
            let k = self.integer_constant(value);
            self.code_constant(reg, k);
        }
    }

    fn load_float(&mut self, reg: u8, value: f64) {
        match number::float_to_integer(value, Rounding::Exact) {
            Some(integer) if fits_bx(integer) => {
                self.emit(Instruction::LoadF { a: reg, sbx: integer as i32 });
            }
            _ => {
                let k = self.float_constant(value);
                self.code_constant(reg, k);
            }
        }
    }

    fn discharge_to_reg(&mut self, expression: &mut ExpDesc, reg: u8) -> Res {
        self.discharge_vars(expression);
        match expression.kind {
            ExpKind::Nil => self.nil(reg, 1),
            ExpKind::False => {
                self.emit(Instruction::LoadFalse { a: reg });
            }
            ExpKind::True => {
                self.emit(Instruction::LoadTrue { a: reg });
            }
            ExpKind::Str(string) => {
                let k = self.string_constant(string);
                self.code_constant(reg, k);
            }
            ExpKind::Constant(k) => self.code_constant(reg, k),
            ExpKind::Float(value) => self.load_float(reg, value),
            ExpKind::Integer(value) => self.load_integer(reg, value),
            ExpKind::Reloc(pc) => set_a(&mut self.code[pc], reg),
            ExpKind::NonReloc(source) => {
                if source != reg {
                    self.emit(Instruction::Move { a: reg, b: source });
                }
            }
            ExpKind::Jump(_) | ExpKind::Void => return Ok(()),
            _ => unreachable!("expression kind is discharged before loading"),
        }
        expression.kind = ExpKind::NonReloc(reg);
        Ok(())
    }

    /// Follows `luaK_nil` in lcode.c.
    pub(super) fn nil(&mut self, from: u8, count: u8) {
        let mut from = from;
        let mut last = from + count - 1;
        if let Some(Instruction::LoadNil { a: previous_from, b: previous_count }) = self.previous_instruction() {
            let previous_last = previous_from + previous_count;
            let connected = (previous_from <= from && from <= previous_last.saturating_add(1))
                || (from <= previous_from && previous_from <= last.saturating_add(1));
            if connected {
                from = from.min(previous_from);
                last = last.max(previous_last);
                let pc = self.pc() - 1;
                self.code[pc] = Instruction::LoadNil { a: from, b: last - from };
                return;
            }
        }
        self.emit(Instruction::LoadNil { a: from, b: count - 1 });
    }

    fn discharge_to_any_reg(&mut self, expression: &mut ExpDesc) -> Res {
        if !matches!(expression.kind, ExpKind::NonReloc(_)) {
            self.reserve_registers(1)?;
            let reg = self.freereg - 1;
            self.discharge_to_reg(expression, reg)?;
        }
        Ok(())
    }

    pub(super) fn exp_to_reg(&mut self, expression: &mut ExpDesc, reg: u8) -> Res {
        self.discharge_to_reg(expression, reg)?;
        if let ExpKind::Jump(pc) = expression.kind {
            let mut true_list = expression.t;
            self.concat(&mut true_list, pc as isize);
            expression.t = true_list;
        }
        if has_jumps(expression) {
            let mut p_false = NO_JUMP;
            let mut p_true = NO_JUMP;
            if self.need_value(expression.t) || self.need_value(expression.f) {
                let skip = if matches!(expression.kind, ExpKind::Jump(_)) {
                    NO_JUMP
                } else {
                    self.jump()
                };
                p_false = self.load_boolean(reg, false) as isize;
                p_true = self.load_boolean(reg, true) as isize;
                self.patch_to_here(skip);
            }
            let end = self.label();
            self.patch_list_aux(expression.f, end, reg, p_false as usize);
            self.patch_list_aux(expression.t, end, reg, p_true as usize);
        }
        expression.t = NO_JUMP;
        expression.f = NO_JUMP;
        expression.kind = ExpKind::NonReloc(reg);
        Ok(())
    }

    fn load_boolean(&mut self, reg: u8, value: bool) -> usize {
        self.label();
        let instruction = if value {
            Instruction::LoadTrue { a: reg }
        } else {
            Instruction::LFalseSkip { a: reg }
        };
        self.emit(instruction)
    }

    fn need_value(&self, mut list: isize) -> bool {
        while list != NO_JUMP {
            let control = self.jump_control(list as usize);
            if !matches!(self.code[control], Instruction::TestSet { .. }) {
                return true;
            }
            list = self.get_jump(list as usize);
        }
        false
    }

    pub(super) fn exp_to_next_reg(&mut self, expression: &mut ExpDesc) -> Res {
        self.discharge_vars(expression);
        self.free_expression(expression);
        self.reserve_registers(1)?;
        let reg = self.freereg - 1;
        self.exp_to_reg(expression, reg)
    }

    pub(super) fn exp_to_any_reg(&mut self, expression: &mut ExpDesc) -> Res<u8> {
        self.discharge_vars(expression);
        if let ExpKind::NonReloc(reg) = expression.kind {
            if !has_jumps(expression) {
                return Ok(reg);
            }
            if reg >= self.nvarstack() {
                self.exp_to_reg(expression, reg)?;
                return Ok(reg);
            }
        }
        self.exp_to_next_reg(expression)?;
        match expression.kind {
            ExpKind::NonReloc(reg) => Ok(reg),
            _ => unreachable!("exp_to_next_reg leaves the value in a register"),
        }
    }

    /// Follows `luaK_exp2anyregup` in lcode.c.
    pub(super) fn exp_to_any_reg_up(&mut self, expression: &mut ExpDesc) -> Res {
        if !matches!(expression.kind, ExpKind::Upval(_) | ExpKind::VarargVar { .. }) || has_jumps(expression) {
            self.exp_to_any_reg(expression)?;
        }
        Ok(())
    }

    pub(super) fn exp_to_value(&mut self, expression: &mut ExpDesc) -> Res {
        if has_jumps(expression) || matches!(expression.kind, ExpKind::Jump(_)) {
            self.exp_to_any_reg(expression)?;
        } else {
            self.discharge_vars(expression);
        }
        Ok(())
    }

    /// Follows `luaK_exp2K` in lcode.c.
    pub(super) fn exp_to_constant(&mut self, expression: &mut ExpDesc) -> bool {
        if has_jumps(expression) {
            return false;
        }
        let index = match expression.kind {
            ExpKind::True => self.boolean_constant(true),
            ExpKind::False => self.boolean_constant(false),
            ExpKind::Nil => self.nil_constant(),
            ExpKind::Integer(value) => self.integer_constant(value),
            ExpKind::Float(value) => self.float_constant(value),
            ExpKind::Str(string) => self.string_constant(string),
            ExpKind::Constant(index) => index,
            _ => return false,
        };
        if index > MAX_INDEX_RK {
            return false;
        }
        expression.kind = ExpKind::Constant(index);
        true
    }

    /// Follows `exp2RK` in lcode.c.
    pub(super) fn exp_to_rk(&mut self, expression: &mut ExpDesc) -> Res<bool> {
        if self.exp_to_constant(expression) {
            return Ok(true);
        }
        self.exp_to_any_reg(expression)?;
        Ok(false)
    }

    fn rk_operand(expression: &ExpDesc) -> u8 {
        match expression.kind {
            ExpKind::Constant(index) => index as u8,
            ExpKind::NonReloc(reg) => reg,
            _ => unreachable!("a stored value is a register or a constant after exp_to_rk"),
        }
    }

    pub(super) fn store_var(&mut self, var: &ExpDesc, expression: &mut ExpDesc) -> Res {
        match var.kind {
            ExpKind::Local { reg, .. } => {
                self.free_expression(expression);
                return self.exp_to_reg(expression, reg);
            }
            ExpKind::Upval(upval) => {
                let source = self.exp_to_any_reg(expression)?;
                self.emit(Instruction::SetUpval { a: source, b: upval });
            }
            ExpKind::IndexUp { table, key } => {
                let k = self.exp_to_rk(expression)?;
                let c = Self::rk_operand(expression);
                self.emit(Instruction::SetTabUp { a: table, b: key, c, k });
            }
            ExpKind::IndexInt { table, index } => {
                let k = self.exp_to_rk(expression)?;
                let c = Self::rk_operand(expression);
                self.emit(Instruction::SetI { a: table, b: index, c, k });
            }
            ExpKind::IndexStr { table, key } => {
                let k = self.exp_to_rk(expression)?;
                let c = Self::rk_operand(expression);
                self.emit(Instruction::SetField { a: table, b: key, c, k });
            }
            ExpKind::Indexed { table, key } | ExpKind::VarargIndexed { table, key } => {
                if matches!(var.kind, ExpKind::VarargIndexed { .. }) {
                    self.vararg_table = true;
                }
                let k = self.exp_to_rk(expression)?;
                let c = Self::rk_operand(expression);
                self.emit(Instruction::SetTable { a: table, b: key, c, k });
            }
            _ => unreachable!("only variables are stored into"),
        }
        self.free_expression(expression);
        Ok(())
    }

    /// Follows `luaK_self` in lcode.c.
    pub(super) fn self_op(&mut self, expression: &mut ExpDesc, key: &mut ExpDesc, key_is_short: bool) -> Res {
        let object = self.exp_to_any_reg(expression)?;
        self.free_expression(expression);
        let base = self.freereg;
        expression.kind = ExpKind::NonReloc(base);
        self.reserve_registers(2)?;
        if key_is_short && self.exp_to_constant(key) {
            let ExpKind::Constant(index) = key.kind else {
                unreachable!("a constant key has a constant index");
            };
            self.emit(Instruction::SelfOp {
                a: base,
                b: object,
                c: index as u8,
            });
        } else {
            let key_reg = self.exp_to_any_reg(key)?;
            self.emit(Instruction::Move { a: base + 1, b: object });
            self.emit(Instruction::GetTable {
                a: base,
                b: object,
                c: key_reg,
            });
        }
        self.free_expression(key);
        Ok(())
    }

    fn negate_condition(&mut self, expression: &ExpDesc) {
        let ExpKind::Jump(pc) = expression.kind else {
            unreachable!("only jump expressions have conditions to negate");
        };
        let control = self.jump_control(pc);
        negate_flag(&mut self.code[control]);
    }

    fn jump_on_condition(&mut self, expression: &mut ExpDesc, condition: bool) -> Res<isize> {
        if let ExpKind::Reloc(pc) = expression.kind
            && let Instruction::Not { b, .. } = self.code[pc]
        {
            self.remove_last_instruction();
            self.emit(Instruction::Test { a: b, k: !condition });
            return Ok(self.jump());
        }
        self.discharge_to_any_reg(expression)?;
        self.free_expression(expression);
        let register = match expression.kind {
            ExpKind::NonReloc(register) => register,
            _ => unreachable!("discharged to a register"),
        };
        self.emit(Instruction::TestSet {
            a: NO_REG,
            b: register,
            k: condition,
        });
        Ok(self.jump())
    }

    pub(super) fn go_if_true(&mut self, expression: &mut ExpDesc) -> Res {
        self.discharge_vars(expression);
        let pc = match expression.kind {
            ExpKind::Jump(pc) => {
                self.negate_condition(expression);
                pc as isize
            }
            ExpKind::Constant(_)
            | ExpKind::Float(_)
            | ExpKind::Integer(_)
            | ExpKind::Str(_)
            | ExpKind::True => NO_JUMP,
            _ => self.jump_on_condition(expression, false)?,
        };
        let mut false_list = expression.f;
        self.concat(&mut false_list, pc);
        expression.f = false_list;
        self.patch_to_here(expression.t);
        expression.t = NO_JUMP;
        Ok(())
    }

    pub(super) fn go_if_false(&mut self, expression: &mut ExpDesc) -> Res {
        self.discharge_vars(expression);
        let pc = match expression.kind {
            ExpKind::Jump(pc) => pc as isize,
            ExpKind::Nil | ExpKind::False => NO_JUMP,
            _ => self.jump_on_condition(expression, true)?,
        };
        let mut true_list = expression.t;
        self.concat(&mut true_list, pc);
        expression.t = true_list;
        self.patch_to_here(expression.f);
        expression.f = NO_JUMP;
        Ok(())
    }

    fn code_not(&mut self, expression: &mut ExpDesc) -> Res {
        match expression.kind {
            ExpKind::Nil | ExpKind::False => expression.kind = ExpKind::True,
            ExpKind::Constant(_)
            | ExpKind::Float(_)
            | ExpKind::Integer(_)
            | ExpKind::Str(_)
            | ExpKind::True => expression.kind = ExpKind::False,
            ExpKind::Jump(_) => self.negate_condition(expression),
            ExpKind::Reloc(_) | ExpKind::NonReloc(_) => {
                self.discharge_to_any_reg(expression)?;
                self.free_expression(expression);
                let b = match expression.kind {
                    ExpKind::NonReloc(register) => register,
                    _ => unreachable!("discharged to a register"),
                };
                let pc = self.emit(Instruction::Not { a: 0, b });
                expression.kind = ExpKind::Reloc(pc);
            }
            _ => {}
        }
        std::mem::swap(&mut expression.f, &mut expression.t);
        self.remove_values_of(expression.f);
        self.remove_values_of(expression.t);
        Ok(())
    }

    /// Follows `luaK_indexed` in lcode.c.
    pub(super) fn indexed(&mut self, table: &mut ExpDesc, key: &mut ExpDesc, key_is_short: bool) -> Res {
        let mut keystr = -1;
        if let ExpKind::Str(string) = key.kind {
            let index = self.string_constant(string);
            keystr = index as i32;
            key.kind = ExpKind::Constant(index);
        }
        let key_index = match key.kind {
            ExpKind::Constant(index) if key_is_short && !has_jumps(key) && index <= MAX_INDEX_RK => Some(index),
            _ => None,
        };
        if matches!(table.kind, ExpKind::Upval(_)) && key_index.is_none() {
            self.exp_to_any_reg(table)?;
        }
        table.keystr = keystr;
        table.ro = false;
        if let ExpKind::VarargVar { reg, .. } = table.kind {
            let key_register = self.exp_to_any_reg(key)?;
            table.kind = ExpKind::VarargIndexed {
                table: reg,
                key: key_register,
            };
            return Ok(());
        }
        table.kind = match table.kind {
            ExpKind::Upval(upval) => {
                let Some(index) = key_index else {
                    unreachable!("upvalue indexing takes a short string constant key");
                };
                ExpKind::IndexUp {
                    table: upval,
                    key: index as u8,
                }
            }
            _ => {
                let register = match table.kind {
                    ExpKind::Local { reg, .. } | ExpKind::NonReloc(reg) => reg,
                    _ => unreachable!("indexed tables sit in registers"),
                };
                match (key_index, key.kind) {
                    (Some(index), _) => ExpKind::IndexStr {
                        table: register,
                        key: index as u8,
                    },
                    (None, ExpKind::Integer(value)) if !has_jumps(key) && (0..=MAX_ARG_C).contains(&value) => {
                        ExpKind::IndexInt {
                            table: register,
                            index: value as u8,
                        }
                    }
                    _ => {
                        let key_register = self.exp_to_any_reg(key)?;
                        ExpKind::Indexed {
                            table: register,
                            key: key_register,
                        }
                    }
                }
            }
        };
        Ok(())
    }

    fn code_unary_value(&mut self, op: UnaryOp, expression: &mut ExpDesc, line: u32) -> Res {
        let reg = self.exp_to_any_reg(expression)?;
        self.free_expression(expression);
        let instruction = match op {
            UnaryOp::Minus => Instruction::Unm { a: 0, b: reg },
            UnaryOp::BNot => Instruction::BNot { a: 0, b: reg },
            UnaryOp::Len => Instruction::Len { a: 0, b: reg },
            UnaryOp::Not => unreachable!("not is coded by code_not"),
        };
        let pc = self.emit(instruction);
        expression.kind = ExpKind::Reloc(pc);
        self.fix_line(line);
        Ok(())
    }

    /// Follows `luaK_prefix` in lcode.c.
    pub(super) fn prefix(&mut self, op: UnaryOp, expression: &mut ExpDesc, line: u32) -> Res {
        self.discharge_vars(expression);
        match op {
            UnaryOp::Minus | UnaryOp::BNot => {
                if !self.fold_unary(op, expression) {
                    self.code_unary_value(op, expression, line)?;
                }
            }
            UnaryOp::Len => self.code_unary_value(op, expression, line)?,
            UnaryOp::Not => self.code_not(expression)?,
        }
        Ok(())
    }

    fn fold_unary(&mut self, op: UnaryOp, expression: &mut ExpDesc) -> bool {
        let Some(value) = numeral(expression) else {
            return false;
        };
        let folded = match (op, value) {
            (UnaryOp::Minus, value) => number::negate(value),
            (UnaryOp::BNot, value) => match number::to_integer(value, Rounding::Exact) {
                Some(integer) => Number::Integer(!integer),
                None => return false,
            },
            _ => return false,
        };
        if let Number::Float(result) = folded
            && (result.is_nan() || result == 0.0)
        {
            return false;
        }
        expression.kind = fold_kind(folded);
        true
    }

    /// Follows `luaK_infix` in lcode.c.
    pub(super) fn infix(&mut self, op: BinOpr, value: &mut ExpDesc) -> Res {
        use BinOpr as B;
        self.discharge_vars(value);
        match op {
            B::And => self.go_if_true(value)?,
            B::Or => self.go_if_false(value)?,
            B::Concat => self.exp_to_next_reg(value)?,
            B::Eq | B::NotEq => {
                if numeral(value).is_none() {
                    self.exp_to_rk(value)?;
                }
            }
            B::Lt | B::Le | B::Gt | B::Ge => {
                let mut isfloat = false;
                if sc_number(value, &mut isfloat).is_none() {
                    self.exp_to_any_reg(value)?;
                }
            }
            _ => {
                if numeral(value).is_none() {
                    self.exp_to_any_reg(value)?;
                }
            }
        }
        Ok(())
    }

    /// Follows `luaK_posfix` in lcode.c.
    pub(super) fn posfix(
        &mut self,
        op: BinOpr,
        first: &mut ExpDesc,
        second: &mut ExpDesc,
        line: u32,
    ) -> Res {
        use BinOpr as B;
        self.discharge_vars(second);
        if let Some(arith) = arith_of(op)
            && self.fold_binary(arith, first, second)
        {
            return Ok(());
        }
        match op {
            B::And => {
                let mut false_list = second.f;
                self.concat(&mut false_list, first.f);
                second.f = false_list;
                *first = *second;
            }
            B::Or => {
                let mut true_list = second.t;
                self.concat(&mut true_list, first.t);
                second.t = true_list;
                *first = *second;
            }
            B::Concat => {
                self.exp_to_next_reg(second)?;
                self.code_concat(first, second, line)?;
            }
            B::Add | B::Mul => self.code_commutative(arith_of(op).expect("arithmetic"), first, second, line)?,
            B::Sub => {
                if !self.finish_negated_immediate(first, second, ArithOp::Add, line, ArithOp::Sub)? {
                    self.code_arith(ArithOp::Sub, first, second, false, line)?;
                }
            }
            B::Div | B::IntDiv | B::Mod | B::Pow => {
                self.code_arith(arith_of(op).expect("arithmetic"), first, second, false, line)?;
            }
            B::BitAnd | B::BitOr | B::BitXor => {
                self.code_bitwise(arith_of(op).expect("bitwise"), first, second, line)?;
            }
            B::Shl => {
                if is_sc_int(first) {
                    std::mem::swap(first, second);
                    self.code_immediate(ArithOp::Shl, first, second, true, line)?;
                } else if !self.finish_negated_immediate(first, second, ArithOp::Shr, line, ArithOp::Shl)? {
                    self.code_binary_value(ArithOp::Shl, first, second, line)?;
                }
            }
            B::Shr => {
                if is_sc_int(second) {
                    self.code_immediate(ArithOp::Shr, first, second, false, line)?;
                } else {
                    self.code_binary_value(ArithOp::Shr, first, second, line)?;
                }
            }
            B::Eq | B::NotEq => self.code_equality(op == B::Eq, first, second)?,
            B::Gt | B::Ge => {
                std::mem::swap(first, second);
                let order = if op == B::Gt { B::Lt } else { B::Le };
                self.code_order(order, first, second)?;
            }
            B::Lt | B::Le => self.code_order(op, first, second)?,
        }
        Ok(())
    }

    /// Follows `constfolding` in lcode.c, which folds neither NaN nor 0.0 to avoid -0.0 problems.
    fn fold_binary(&mut self, op: ArithOp, first: &mut ExpDesc, second: &ExpDesc) -> bool {
        let (Some(left), Some(right)) = (numeral(first), numeral(second)) else {
            return false;
        };
        let divides = matches!(op, ArithOp::Div | ArithOp::IDiv | ArithOp::Mod);
        if divides && right.to_float() == 0.0 {
            return false;
        }
        let Ok(result) = number::arith(op, left, right) else {
            return false;
        };
        if let Number::Float(value) = result
            && (value.is_nan() || value == 0.0)
        {
            return false;
        }
        first.kind = fold_kind(result);
        true
    }

    /// Follows `finishbinexpval` in lcode.c; the arithmetic instruction skips its metamethod instruction on success.
    fn finish_binary_value(
        &mut self,
        first: &mut ExpDesc,
        second: &ExpDesc,
        line: u32,
        op: ArithOp,
        right: Right,
        flip: bool,
    ) -> Res {
        let left = self.exp_to_any_reg(first)?;
        let (instruction, metamethod) = match right {
            Right::Register(c) => (
                arith_register(op, left, c),
                Instruction::MmBin {
                    a: left,
                    b: c,
                    c: event_of(op),
                },
            ),
            Right::Constant(k) => (
                arith_constant(op, left, k),
                Instruction::MmBinK {
                    a: left,
                    b: k,
                    c: event_of(op),
                    k: flip,
                },
            ),
            Right::Immediate { op: immediate, sc, sb } => (
                immediate_arith(immediate, left, sc),
                Instruction::MmBinI {
                    a: left,
                    sb,
                    c: event_of(op),
                    k: flip,
                },
            ),
        };
        let pc = self.emit(instruction);
        self.free_expressions(first, second);
        first.kind = ExpKind::Reloc(pc);
        self.fix_line(line);
        self.emit(metamethod);
        self.fix_line(line);
        Ok(())
    }

    /// Follows `codebinexpval` in lcode.c.
    fn code_binary_value(&mut self, op: ArithOp, first: &mut ExpDesc, second: &mut ExpDesc, line: u32) -> Res {
        let right = self.exp_to_any_reg(second)?;
        self.finish_binary_value(first, second, line, op, Right::Register(right), false)
    }

    /// Follows `codebinK` in lcode.c.
    fn code_constant_operand(
        &mut self,
        op: ArithOp,
        first: &mut ExpDesc,
        second: &ExpDesc,
        line: u32,
        flip: bool,
    ) -> Res {
        let ExpKind::Constant(k) = second.kind else {
            unreachable!("the caller converted the operand to a constant");
        };
        self.finish_binary_value(first, second, line, op, Right::Constant(k as u8), flip)
    }

    /// Follows `codebini` in lcode.c.
    fn code_immediate(&mut self, op: ArithOp, first: &mut ExpDesc, second: &ExpDesc, flip: bool, line: u32) -> Res {
        let ExpKind::Integer(value) = second.kind else {
            unreachable!("an immediate operand is an integer");
        };
        let sc = value as i32;
        self.finish_binary_value(first, second, line, op, Right::Immediate { op, sc, sb: sc }, flip)
    }

    /// Follows `finishbinexpneg` in lcode.c; its metamethod instruction keeps `n` and the original event.
    fn finish_negated_immediate(
        &mut self,
        first: &mut ExpDesc,
        second: &mut ExpDesc,
        op: ArithOp,
        line: u32,
        event: ArithOp,
    ) -> Res<bool> {
        let ExpKind::Integer(value) = second.kind else {
            return Ok(false);
        };
        if !is_kint(second) || !(fits_sc(value) && fits_sc(value.wrapping_neg())) {
            return Ok(false);
        }
        let original = value as i32;
        self.finish_binary_value(
            first,
            second,
            line,
            event,
            Right::Immediate {
                op,
                sc: -original,
                sb: original,
            },
            false,
        )?;
        Ok(true)
    }

    /// Follows `codearith` in lcode.c.
    fn code_arith(&mut self, op: ArithOp, first: &mut ExpDesc, second: &mut ExpDesc, flip: bool, line: u32) -> Res {
        if numeral(second).is_some() && self.exp_to_constant(second) {
            return self.code_constant_operand(op, first, second, line, flip);
        }
        if flip {
            std::mem::swap(first, second);
        }
        self.code_binary_value(op, first, second, line)
    }

    /// Follows `codecommutative` in lcode.c.
    fn code_commutative(&mut self, op: ArithOp, first: &mut ExpDesc, second: &mut ExpDesc, line: u32) -> Res {
        let mut flip = false;
        if numeral(first).is_some() {
            std::mem::swap(first, second);
            flip = true;
        }
        if op == ArithOp::Add && is_sc_int(second) {
            return self.code_immediate(op, first, second, flip, line);
        }
        self.code_arith(op, first, second, flip, line)
    }

    /// Follows `codebitwise` in lcode.c.
    fn code_bitwise(&mut self, op: ArithOp, first: &mut ExpDesc, second: &mut ExpDesc, line: u32) -> Res {
        let mut flip = false;
        if matches!(first.kind, ExpKind::Integer(_)) {
            std::mem::swap(first, second);
            flip = true;
        }
        if matches!(second.kind, ExpKind::Integer(_)) && self.exp_to_constant(second) {
            return self.code_constant_operand(op, first, second, line, flip);
        }
        if flip {
            std::mem::swap(first, second);
        }
        self.code_binary_value(op, first, second, line)
    }

    /// Follows `codeorder` in lcode.c.
    fn code_order(&mut self, op: BinOpr, first: &mut ExpDesc, second: &mut ExpDesc) -> Res {
        use BinOpr as B;
        let less = op == B::Lt;
        let mut isfloat = false;
        let instruction = if let Some(immediate) = sc_number(second, &mut isfloat) {
            let register = self.exp_to_any_reg(first)?;
            self.free_expressions(first, second);
            order_immediate(less, register, immediate, isfloat)
        } else if let Some(immediate) = sc_number(first, &mut isfloat) {
            let register = self.exp_to_any_reg(second)?;
            self.free_expressions(first, second);
            order_flipped_immediate(less, register, immediate, isfloat)
        } else {
            let left = self.exp_to_any_reg(first)?;
            let right = self.exp_to_any_reg(second)?;
            self.free_expressions(first, second);
            if less {
                Instruction::Lt { a: left, b: right, k: true }
            } else {
                Instruction::Le { a: left, b: right, k: true }
            }
        };
        self.emit(instruction);
        let jump = self.jump();
        first.kind = ExpKind::Jump(jump as usize);
        Ok(())
    }

    /// Follows `codeeq` in lcode.c.
    fn code_equality(&mut self, equal: bool, first: &mut ExpDesc, second: &mut ExpDesc) -> Res {
        if !matches!(first.kind, ExpKind::NonReloc(_)) {
            std::mem::swap(first, second);
        }
        let left = self.exp_to_any_reg(first)?;
        let mut isfloat = false;
        let instruction = if let Some(immediate) = sc_number(second, &mut isfloat) {
            Instruction::EqI {
                a: left,
                sb: immediate,
                c: u8::from(isfloat),
                k: equal,
            }
        } else if self.exp_to_rk(second)? {
            let ExpKind::Constant(k) = second.kind else {
                unreachable!("a constant operand has a constant index");
            };
            Instruction::EqK {
                a: left,
                b: k as u8,
                k: equal,
            }
        } else {
            let right = self.exp_to_any_reg(second)?;
            Instruction::Eq {
                a: left,
                b: right,
                k: equal,
            }
        };
        self.free_expressions(first, second);
        self.emit(instruction);
        let jump = self.jump();
        first.kind = ExpKind::Jump(jump as usize);
        Ok(())
    }

    /// Follows `codeconcat` in lcode.c.
    fn code_concat(&mut self, first: &mut ExpDesc, second: &mut ExpDesc, line: u32) -> Res {
        let ExpKind::NonReloc(first_reg) = first.kind else {
            unreachable!("the left operand of '..' is in a register");
        };
        if let Some(Instruction::Concat { b: count, .. }) = self.previous_instruction() {
            self.free_expression(second);
            let pc = self.pc() - 1;
            self.code[pc] = Instruction::Concat {
                a: first_reg,
                b: count + 1,
            };
            return Ok(());
        }
        self.emit(Instruction::Concat { a: first_reg, b: 2 });
        self.free_expression(second);
        self.fix_line(line);
        Ok(())
    }

    /// Follows `luaK_setlist` in lcode.c; a `to_store` of `None` is encoded as 0, meaning up to the top.
    pub(super) fn set_list(&mut self, base: u8, stored: u32, to_store: Option<u8>) {
        let count = to_store.unwrap_or(0);
        if stored < SETLIST_BLOCK {
            self.emit(Instruction::SetList {
                a: base,
                b: count,
                c: stored,
                k: false,
            });
        } else {
            self.emit(Instruction::SetList {
                a: base,
                b: count,
                c: stored % SETLIST_BLOCK,
                k: true,
            });
            self.emit(Instruction::ExtraArg {
                ax: stored / SETLIST_BLOCK,
            });
        }
        self.freereg = base + 1;
    }

    /// Follows `luaK_settablesize` in lcode.c.
    pub(super) fn set_table_size(&mut self, pc: usize, a: u8, array: u32, hash: u32) {
        let extra = array / SETLIST_BLOCK;
        let log_hash = if hash == 0 { 0 } else { ceil_log2(hash) + 1 };
        self.code[pc] = Instruction::NewTable {
            a,
            b: log_hash,
            c: array % SETLIST_BLOCK,
            k: extra > 0,
        };
        self.code[pc + 1] = Instruction::ExtraArg { ax: extra };
    }

    /// Follows `luaK_ret` in lcode.c.
    pub(super) fn ret(&mut self, first: u8, count: i32) -> Res {
        if count + 1 > MAX_ARG_B as i32 {
            return Err(self.limit("returns", MAX_ARG_B));
        }
        let instruction = match count {
            0 => Instruction::Return0 { a: first },
            1 => Instruction::Return1 { a: first },
            _ => Instruction::Return {
                a: first,
                b: (count + 1) as u8,
                c: 0,
                k: false,
            },
        };
        self.emit(instruction);
        Ok(())
    }

    /// Follows `luaK_finish` in lcode.c.
    pub(super) fn finish(&mut self) {
        let needclose = self.needclose;
        let vararg_table = self.vararg_table;
        let hidden_varargs = self.is_vararg && !vararg_table;
        let numparams = self.num_params;
        for index in 0..self.code.len() {
            match self.code[index] {
                Instruction::Return0 { a } if needclose || hidden_varargs => {
                    self.code[index] = Instruction::Return { a, b: 1, c: 0, k: false };
                    self.finish_return(index, needclose, hidden_varargs, numparams);
                }
                Instruction::Return1 { a } if needclose || hidden_varargs => {
                    self.code[index] = Instruction::Return { a, b: 2, c: 0, k: false };
                    self.finish_return(index, needclose, hidden_varargs, numparams);
                }
                Instruction::Return { .. } | Instruction::TailCall { .. } => {
                    self.finish_return(index, needclose, hidden_varargs, numparams);
                }
                Instruction::GetVarg { a, b, c } if vararg_table => {
                    self.code[index] = Instruction::GetTable { a, b, c };
                }
                Instruction::Vararg { a, b, c, .. } => {
                    self.code[index] = Instruction::Vararg {
                        a,
                        b,
                        c,
                        k: vararg_table,
                    };
                }
                Instruction::Jmp { .. } => {
                    let target = self.final_target(index);
                    self.fix_jump(index, target);
                }
                _ => {}
            }
        }
    }

    fn finish_return(&mut self, index: usize, needclose: bool, hidden_varargs: bool, numparams: u8) {
        if let Instruction::Return { c, k, .. } | Instruction::TailCall { c, k, .. } = &mut self.code[index] {
            if needclose {
                *k = true;
            }
            if hidden_varargs {
                *c = numparams + 1;
            }
        }
    }

    fn final_target(&self, mut index: usize) -> usize {
        for _ in 0..100 {
            let Instruction::Jmp { sj } = self.code[index] else {
                break;
            };
            index = (index as isize + 1 + sj as isize) as usize;
        }
        index
    }

    /// Follows `luaK_exp2const` in lcode.c.
    pub(super) fn exp_to_literal(expression: &ExpDesc) -> Option<Literal> {
        if has_jumps(expression) {
            return None;
        }
        Some(match expression.kind {
            ExpKind::False => Literal::False,
            ExpKind::True => Literal::True,
            ExpKind::Nil => Literal::Nil,
            ExpKind::Str(string) => Literal::Str(string),
            ExpKind::Const { value, .. } => value,
            ExpKind::Integer(value) => Literal::Integer(value),
            ExpKind::Float(value) => Literal::Float(value),
            _ => return None,
        })
    }
}

/// Follows `luaO_ceillog2` in lobject.c.
fn ceil_log2(value: u32) -> u8 {
    value.next_power_of_two().trailing_zeros() as u8
}

fn fold_kind(value: Number) -> ExpKind {
    match value {
        Number::Integer(value) => ExpKind::Integer(value),
        Number::Float(value) => ExpKind::Float(value),
    }
}

fn order_immediate(less: bool, register: u8, immediate: i32, isfloat: bool) -> Instruction {
    let c = u8::from(isfloat);
    if less {
        Instruction::LtI {
            a: register,
            sb: immediate,
            c,
            k: true,
        }
    } else {
        Instruction::LeI {
            a: register,
            sb: immediate,
            c,
            k: true,
        }
    }
}

fn order_flipped_immediate(less: bool, register: u8, immediate: i32, isfloat: bool) -> Instruction {
    let c = u8::from(isfloat);
    if less {
        Instruction::GtI {
            a: register,
            sb: immediate,
            c,
            k: true,
        }
    } else {
        Instruction::GeI {
            a: register,
            sb: immediate,
            c,
            k: true,
        }
    }
}
