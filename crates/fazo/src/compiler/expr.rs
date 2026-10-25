//! Follows PUC Lua 5.5.1's lparser.c (MIT, Copyright (C) 1994-2025 Lua.org, PUC-Rio).

use super::{
    Compiler,
    code::BinOpr,
    func::{ExpDesc, ExpKind, MAX_REGISTERS, Res, VarKind},
};
use crate::{
    instruction::{Instruction, UnaryOp},
    syntax::token::{Kind, Value},
};

const UNARY_PRIORITY: u8 = 12;
const MAX_CONSTRUCTOR_ITEMS: usize = (i32::MAX / 2) as usize;

/// Follows `ConsControl` in lparser.c.
struct ListControl {
    table: ExpDesc,
    pending: ExpDesc,
    hashed: usize,
    stored: usize,
    to_store: usize,
    max_to_store: usize,
}

impl Compiler<'_> {
    pub(super) fn expr(&mut self) -> Res<ExpDesc> {
        let mut value = ExpDesc::new(ExpKind::Void);
        self.subexpr(&mut value, 0)?;
        Ok(value)
    }

    /// Follows `explist` in lparser.c.
    pub(super) fn explist(&mut self) -> Res<(ExpDesc, usize)> {
        let mut count = 1;
        let mut value = self.expr()?;
        while self.testnext(Kind::Char(b','))? {
            self.fs().exp_to_next_reg(&mut value)?;
            value = self.expr()?;
            count += 1;
        }
        Ok((value, count))
    }

    /// Follows `subexpr` in lparser.c.
    fn subexpr(&mut self, value: &mut ExpDesc, limit: u8) -> Res<Option<BinOpr>> {
        self.enter_level()?;
        if let Some(op) = unary_operator(self.token.kind) {
            let line = self.lexer.line;
            self.next()?;
            self.subexpr(value, UNARY_PRIORITY)?;
            self.fs().prefix(op, value, line)?;
        } else {
            *value = self.simple_exp()?;
        }
        let mut op = binary_operator(self.token.kind);
        while let Some(current) = op {
            let (left, right) = priority(current);
            if left <= limit {
                break;
            }
            let line = self.lexer.line;
            self.next()?;
            self.fs().infix(current, value)?;
            let mut second = ExpDesc::new(ExpKind::Void);
            let next = self.subexpr(&mut second, right)?;
            self.fs().posfix(current, value, &mut second, line)?;
            op = next;
        }
        self.leave_level();
        Ok(op)
    }

    /// Follows `simpleexp` in lparser.c.
    fn simple_exp(&mut self) -> Res<ExpDesc> {
        let value = match (self.token.kind, self.token.value) {
            (Kind::Float, Value::Float(number)) => ExpDesc::new(ExpKind::Float(number)),
            (Kind::Int, Value::Int(number)) => ExpDesc::new(ExpKind::Integer(number)),
            (Kind::Str, Value::Str(string)) => ExpDesc::new(ExpKind::Str(string)),
            (Kind::Nil, _) => ExpDesc::new(ExpKind::Nil),
            (Kind::True, _) => ExpDesc::new(ExpKind::True),
            (Kind::False, _) => ExpDesc::new(ExpKind::False),
            (Kind::Dots, _) => self.vararg()?,
            (Kind::Char(b'{'), _) => return self.constructor(),
            (Kind::Function, _) => {
                self.next()?;
                let line = self.lexer.line;
                return self.body(false, line);
            }
            _ => return self.suffixed_exp(),
        };
        self.next()?;
        Ok(value)
    }

    fn vararg(&mut self) -> Res<ExpDesc> {
        if !self.fs_ref().is_vararg {
            return self.syntax_error("cannot use '...' outside a vararg function");
        }
        let fs = self.fs();
        let num_params = fs.num_params;
        let pc = fs.emit(Instruction::Vararg {
            a: 0,
            b: num_params,
            c: 1,
            k: false,
        });
        Ok(ExpDesc::new(ExpKind::Vararg(pc)))
    }

    /// Follows `primaryexp` in lparser.c.
    fn primary_exp(&mut self) -> Res<ExpDesc> {
        match self.token.kind {
            Kind::Char(b'(') => {
                let line = self.lexer.line;
                self.next()?;
                let mut value = self.expr()?;
                self.check_match(Kind::Char(b')'), Kind::Char(b'('), line)?;
                self.fs().discharge_vars(&mut value);
                Ok(value)
            }
            Kind::Name => self.single_var(),
            _ => self.syntax_error("unexpected symbol"),
        }
    }

    /// Follows `suffixedexp` in lparser.c.
    pub(super) fn suffixed_exp(&mut self) -> Res<ExpDesc> {
        let mut value = self.primary_exp()?;
        loop {
            match self.token.kind {
                Kind::Char(b'.') => self.field_select(&mut value)?,
                Kind::Char(b'[') => {
                    self.fs().exp_to_any_reg_up(&mut value)?;
                    let mut key = self.index()?;
                    let short = self.is_short(&key);
                    self.fs().indexed(&mut value, &mut key, short)?;
                }
                Kind::Char(b':') => {
                    self.next()?;
                    let mut key = ExpDesc::new(ExpKind::Str(self.str_checkname()?));
                    let short = self.is_short(&key);
                    self.fs().self_op(&mut value, &mut key, short)?;
                    self.function_arguments(&mut value)?;
                }
                Kind::Char(b'(' | b'{') | Kind::Str => {
                    self.fs().exp_to_next_reg(&mut value)?;
                    self.function_arguments(&mut value)?;
                }
                _ => return Ok(value),
            }
        }
    }

    /// Follows `fieldsel` in lparser.c.
    pub(super) fn field_select(&mut self, value: &mut ExpDesc) -> Res {
        self.fs().exp_to_any_reg_up(value)?;
        self.next()?;
        let mut key = ExpDesc::new(ExpKind::Str(self.str_checkname()?));
        let short = self.is_short(&key);
        self.fs().indexed(value, &mut key, short)
    }

    /// Follows `yindex` in lparser.c.
    fn index(&mut self) -> Res<ExpDesc> {
        self.next()?;
        let mut key = self.expr()?;
        self.fs().exp_to_value(&mut key)?;
        self.checknext(Kind::Char(b']'))?;
        Ok(key)
    }

    /// Follows `funcargs` in lparser.c.
    fn function_arguments(&mut self, function: &mut ExpDesc) -> Res {
        let line = self.lexer.line;
        let mut args = match (self.token.kind, self.token.value) {
            (Kind::Char(b'('), _) => {
                self.next()?;
                let args = if self.token.kind == Kind::Char(b')') {
                    ExpDesc::new(ExpKind::Void)
                } else {
                    let (mut args, _) = self.explist()?;
                    if args.has_multiple_results() {
                        self.fs().set_multiple_returns(&mut args)?;
                    }
                    args
                };
                self.check_match(Kind::Char(b')'), Kind::Char(b'('), line)?;
                args
            }
            (Kind::Char(b'{'), _) => self.constructor()?,
            (Kind::Str, Value::Str(string)) => {
                self.next()?;
                ExpDesc::new(ExpKind::Str(string))
            }
            _ => return self.syntax_error("function arguments expected"),
        };
        let ExpKind::NonReloc(base) = function.kind else {
            unreachable!("a callee sits in a register before its arguments");
        };
        let fs = self.fs();
        let b = if args.has_multiple_results() {
            0
        } else {
            if !matches!(args.kind, ExpKind::Void) {
                fs.exp_to_next_reg(&mut args)?;
            }
            fs.freereg - base
        };
        let pc = fs.emit(Instruction::Call { a: base, b, c: 2 });
        fs.fix_line(line);
        fs.freereg = base + 1;
        *function = ExpDesc::new(ExpKind::Call(pc));
        Ok(())
    }

    /// Follows `constructor` in lparser.c.
    fn constructor(&mut self) -> Res<ExpDesc> {
        let line = self.lexer.line;
        let fs = self.fs();
        let pc = fs.emit(Instruction::NewTable {
            a: 0,
            b: 0,
            c: 0,
            k: false,
        });
        fs.emit(Instruction::ExtraArg { ax: 0 });
        let table_reg = fs.freereg;
        fs.reserve_registers(1)?;
        let mut control = ListControl {
            table: ExpDesc::new(ExpKind::NonReloc(table_reg)),
            pending: ExpDesc::new(ExpKind::Void),
            hashed: 0,
            stored: 0,
            to_store: 0,
            max_to_store: 0,
        };
        self.checknext(Kind::Char(b'{'))?;
        control.max_to_store = max_to_store(self.fs_ref().freereg);
        loop {
            if self.token.kind == Kind::Char(b'}') {
                break;
            }
            if !matches!(control.pending.kind, ExpKind::Void) {
                self.close_list_field(&mut control)?;
            }
            self.field(&mut control)?;
            if control.to_store + control.stored + control.hashed > MAX_CONSTRUCTOR_ITEMS {
                return Err(self.fs_ref().limit("items in a constructor", MAX_CONSTRUCTOR_ITEMS));
            }
            if !(self.testnext(Kind::Char(b','))? || self.testnext(Kind::Char(b';'))?) {
                break;
            }
        }
        self.check_match(Kind::Char(b'}'), Kind::Char(b'{'), line)?;
        self.last_list_field(&mut control)?;
        self.fs()
            .set_table_size(pc, table_reg, control.stored as u32, control.hashed as u32);
        Ok(control.table)
    }

    /// Follows `field` in lparser.c.
    fn field(&mut self, control: &mut ListControl) -> Res {
        let kind = self.token.kind;
        let record = match kind {
            Kind::Name => self.lookahead_kind()? == Kind::Char(b'='),
            Kind::Char(b'[') => true,
            _ => false,
        };
        if record {
            return self.record_field(control);
        }
        control.pending = self.expr()?;
        control.to_store += 1;
        Ok(())
    }

    /// Follows `recfield` in lparser.c.
    fn record_field(&mut self, control: &mut ListControl) -> Res {
        let saved = self.fs_ref().freereg;
        let mut key = if self.token.kind == Kind::Name {
            ExpDesc::new(ExpKind::Str(self.str_checkname()?))
        } else {
            self.index()?
        };
        control.hashed += 1;
        self.checknext(Kind::Char(b'='))?;
        let mut target = control.table;
        let short = self.is_short(&key);
        self.fs().indexed(&mut target, &mut key, short)?;
        let mut value = self.expr()?;
        let fs = self.fs();
        fs.store_var(&target, &mut value)?;
        fs.freereg = saved;
        Ok(())
    }

    /// Follows `closelistfield` in lparser.c.
    fn close_list_field(&mut self, control: &mut ListControl) -> Res {
        let fs = self.fs();
        fs.exp_to_next_reg(&mut control.pending)?;
        control.pending = ExpDesc::new(ExpKind::Void);
        if control.to_store >= control.max_to_store {
            let ExpKind::NonReloc(table) = control.table.kind else {
                unreachable!("the table of a constructor sits in a register");
            };
            fs.set_list(table, control.stored as u32, Some(control.to_store as u8));
            control.stored += control.to_store;
            control.to_store = 0;
        }
        Ok(())
    }

    /// Follows `lastlistfield` in lparser.c; a trailing call or vararg stores all its values, which are not counted.
    fn last_list_field(&mut self, control: &mut ListControl) -> Res {
        if control.to_store == 0 {
            return Ok(());
        }
        let ExpKind::NonReloc(table) = control.table.kind else {
            unreachable!("the table of a constructor sits in a register");
        };
        let fs = self.fs();
        if control.pending.has_multiple_results() {
            fs.set_multiple_returns(&mut control.pending)?;
            fs.set_list(table, control.stored as u32, None);
            control.stored += control.to_store - 1;
            return Ok(());
        }
        if !matches!(control.pending.kind, ExpKind::Void) {
            fs.exp_to_next_reg(&mut control.pending)?;
        }
        fs.set_list(table, control.stored as u32, Some(control.to_store as u8));
        control.stored += control.to_store;
        Ok(())
    }

    /// Follows `body` in lparser.c.
    pub(super) fn body(&mut self, is_method: bool, line: u32) -> Res<ExpDesc> {
        self.open_function(line);
        self.checknext(Kind::Char(b'('))?;
        if is_method {
            self.new_local_literal(b"self");
            self.adjust_local_vars(1)?;
        }
        self.parameter_list()?;
        self.checknext(Kind::Char(b')'))?;
        self.statlist()?;
        self.fs().last_line_defined = self.lexer.line;
        self.check_match(Kind::End, Kind::Function, line)?;
        let closure = self.code_closure()?;
        self.close_function()?;
        Ok(closure)
    }

    /// Follows `parlist` in lparser.c.
    fn parameter_list(&mut self) -> Res {
        let mut params = 0;
        let mut is_vararg = false;
        if self.token.kind != Kind::Char(b')') {
            loop {
                match self.token.kind {
                    Kind::Name => {
                        let name = self.str_checkname()?;
                        self.new_local(name);
                        params += 1;
                    }
                    Kind::Dots => {
                        is_vararg = true;
                        self.next()?;
                        if self.token.kind == Kind::Name {
                            let name = self.str_checkname()?;
                            self.new_var(Some(name), VarKind::Vararg);
                        } else {
                            self.new_local_literal(b"(vararg table)");
                        }
                    }
                    _ => return self.syntax_error("<name> or '...' expected"),
                }
                if is_vararg || !self.testnext(Kind::Char(b','))? {
                    break;
                }
            }
        }
        self.adjust_local_vars(params)?;
        let fs = self.fs();
        fs.num_params = fs.nactvar as u8;
        if is_vararg {
            self.set_vararg();
            self.adjust_local_vars(1)?;
        }
        let fs = self.fs();
        let count = fs.nactvar as u8;
        fs.reserve_registers(count)
    }
}

fn max_to_store(freereg: u8) -> usize {
    let free = MAX_REGISTERS - freereg as usize;
    if free >= 160 {
        free / 5
    } else if free >= 80 {
        10
    } else {
        1
    }
}

fn unary_operator(kind: Kind) -> Option<UnaryOp> {
    Some(match kind {
        Kind::Not => UnaryOp::Not,
        Kind::Char(b'-') => UnaryOp::Minus,
        Kind::Char(b'~') => UnaryOp::BNot,
        Kind::Char(b'#') => UnaryOp::Len,
        _ => return None,
    })
}

fn binary_operator(kind: Kind) -> Option<BinOpr> {
    Some(match kind {
        Kind::Char(b'+') => BinOpr::Add,
        Kind::Char(b'-') => BinOpr::Sub,
        Kind::Char(b'*') => BinOpr::Mul,
        Kind::Char(b'%') => BinOpr::Mod,
        Kind::Char(b'^') => BinOpr::Pow,
        Kind::Char(b'/') => BinOpr::Div,
        Kind::IntDiv => BinOpr::IntDiv,
        Kind::Char(b'&') => BinOpr::BitAnd,
        Kind::Char(b'|') => BinOpr::BitOr,
        Kind::Char(b'~') => BinOpr::BitXor,
        Kind::Shl => BinOpr::Shl,
        Kind::Shr => BinOpr::Shr,
        Kind::Concat => BinOpr::Concat,
        Kind::NotEq => BinOpr::NotEq,
        Kind::EqEq => BinOpr::Eq,
        Kind::Char(b'<') => BinOpr::Lt,
        Kind::LessEq => BinOpr::Le,
        Kind::Char(b'>') => BinOpr::Gt,
        Kind::GreaterEq => BinOpr::Ge,
        Kind::And => BinOpr::And,
        Kind::Or => BinOpr::Or,
        _ => return None,
    })
}

/// Follows the `priority` table in lparser.c.
fn priority(op: BinOpr) -> (u8, u8) {
    match op {
        BinOpr::Add | BinOpr::Sub => (10, 10),
        BinOpr::Mul | BinOpr::Mod | BinOpr::Div | BinOpr::IntDiv => (11, 11),
        BinOpr::Pow => (14, 13),
        BinOpr::BitAnd => (6, 6),
        BinOpr::BitOr => (4, 4),
        BinOpr::BitXor => (5, 5),
        BinOpr::Shl | BinOpr::Shr => (7, 7),
        BinOpr::Concat => (9, 8),
        BinOpr::Eq | BinOpr::Lt | BinOpr::Le | BinOpr::NotEq | BinOpr::Gt | BinOpr::Ge => (3, 3),
        BinOpr::And => (2, 2),
        BinOpr::Or => (1, 1),
    }
}
