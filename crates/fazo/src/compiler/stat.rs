//! Follows PUC Lua 5.5.1's lparser.c (MIT, Copyright (C) 1994-2025 Lua.org, PUC-Rio).

use smallvec::SmallVec;

use super::{
    Compiler,
    code::MAX_ARG_BX,
    func::{ExpDesc, ExpKind, Fault, FuncState, NO_JUMP, Res, VarKind},
};
use crate::{
    arena::Ref,
    instruction::Instruction,
    object::LuaString,
    syntax::token::{Kind, Value},
};

impl Compiler<'_> {
    /// Follows `statlist` in lparser.c.
    pub(super) fn statlist(&mut self) -> Res {
        while !self.block_follow(true) {
            if self.token.kind == Kind::Return {
                return self.statement();
            }
            self.statement()?;
        }
        Ok(())
    }

    fn block(&mut self) -> Res {
        self.enter_block(0);
        self.statlist()?;
        self.leave_block()
    }

    /// Follows `statement` in lparser.c.
    fn statement(&mut self) -> Res {
        let line = self.lexer.line;
        self.check_interrupt()?;
        self.enter_level()?;
        if self.is_global_statement()? {
            self.global_statement(line)?;
            return self.finish_statement();
        }
        match self.token.kind {
            Kind::Char(b';') => self.next()?,
            Kind::If => self.if_statement(line)?,
            Kind::While => self.while_statement(line)?,
            Kind::Do => {
                self.next()?;
                self.block()?;
                self.check_match(Kind::End, Kind::Do, line)?;
            }
            Kind::For => self.for_statement(line)?,
            Kind::Repeat => self.repeat_statement(line)?,
            Kind::Function => self.function_statement(line)?,
            Kind::Local => {
                self.next()?;
                if self.testnext(Kind::Function)? {
                    self.local_function()?;
                } else {
                    self.local_statement()?;
                }
            }
            Kind::DbColon => {
                self.next()?;
                let name = self.str_checkname()?;
                self.label_statement(name, line)?;
            }
            Kind::Return => {
                self.next()?;
                self.return_statement()?;
            }
            Kind::Break => self.break_statement(line)?,
            Kind::Goto => {
                self.next()?;
                let name = self.str_checkname()?;
                self.goto_entry(name, line);
            }
            _ => self.expression_statement()?,
        }
        self.finish_statement()
    }

    fn finish_statement(&mut self) -> Res {
        let fs = self.fs();
        fs.freereg = fs.nvarstack();
        self.leave_level();
        Ok(())
    }

    /// Follows PUC Lua 5.5's default build (`LUA_COMPAT_GLOBAL`), where `global` stays a plain name unless a name, `*`, `<` or `function` follows.
    fn is_global_statement(&mut self) -> Res<bool> {
        if self.token.kind != Kind::Name || self.token.value != Value::Str(self.global) {
            return Ok(false);
        }
        Ok(matches!(
            self.lookahead_kind()?,
            Kind::Char(b'<' | b'*') | Kind::Name | Kind::Function
        ))
    }

    /// Follows `cond` in lparser.c.
    fn condition(&mut self) -> Res<isize> {
        let mut value = self.expr()?;
        if matches!(value.kind, ExpKind::Nil) {
            value.kind = ExpKind::False;
        }
        self.fs().go_if_true(&mut value)?;
        Ok(value.f)
    }

    /// Follows `ifstat` in lparser.c.
    fn if_statement(&mut self, line: u32) -> Res {
        let mut escape = NO_JUMP;
        self.test_then_block(&mut escape)?;
        while self.token.kind == Kind::Elseif {
            self.test_then_block(&mut escape)?;
        }
        if self.testnext(Kind::Else)? {
            self.block()?;
        }
        self.check_match(Kind::End, Kind::If, line)?;
        self.fs().patch_to_here(escape);
        Ok(())
    }

    /// Follows `test_then_block` in lparser.c.
    fn test_then_block(&mut self, escape: &mut isize) -> Res {
        self.next()?;
        let exit = self.condition()?;
        self.checknext(Kind::Then)?;
        self.block()?;
        let escapes = matches!(self.token.kind, Kind::Else | Kind::Elseif);
        let fs = self.fs();
        if escapes {
            let jump = fs.jump();
            fs.concat(escape, jump);
        }
        fs.patch_to_here(exit);
        Ok(())
    }

    /// Follows `whilestat` in lparser.c.
    fn while_statement(&mut self, line: u32) -> Res {
        self.next()?;
        let start = self.fs().label();
        let exit = self.condition()?;
        self.enter_block(1);
        self.checknext(Kind::Do)?;
        self.block()?;
        self.fs().code_jump_back(start);
        self.check_match(Kind::End, Kind::While, line)?;
        self.leave_block()?;
        self.fs().patch_to_here(exit);
        Ok(())
    }

    /// Follows `repeatstat` in lparser.c; captured body locals are closed before the jump back.
    fn repeat_statement(&mut self, line: u32) -> Res {
        let start = self.fs().label();
        self.enter_block(1);
        self.enter_block(0);
        self.next()?;
        self.statlist()?;
        self.check_match(Kind::Until, Kind::Repeat, line)?;
        let mut exit = self.condition()?;
        let scope = *self.fs_ref().blocks.last().expect("the scope block is open");
        if scope.upval {
            let level = self.reglevel(scope.nactvar);
            let fs = self.fs();
            let out = fs.jump();
            fs.patch_to_here(exit);
            fs.emit(Instruction::Close { a: level });
            exit = fs.jump();
            fs.patch_to_here(out);
        }
        self.fs().patch_list(exit, start);
        self.leave_block()?;
        self.leave_block()
    }

    /// Follows `exp1` in lparser.c.
    fn next_register_expression(&mut self) -> Res {
        let mut value = self.expr()?;
        self.fs().exp_to_next_reg(&mut value)
    }

    /// Follows `forstat` in lparser.c.
    fn for_statement(&mut self, line: u32) -> Res {
        self.enter_block(1);
        self.next()?;
        let name = self.str_checkname()?;
        match self.token.kind {
            Kind::Char(b'=') => self.numeric_for(name, line)?,
            Kind::Char(b',') | Kind::In => self.generic_for(name)?,
            _ => return self.syntax_error("'=' or 'in' expected"),
        }
        self.check_match(Kind::End, Kind::For, line)?;
        self.leave_block()
    }

    /// Follows `fornum` in lparser.c.
    fn numeric_for(&mut self, name: Ref<LuaString>, line: u32) -> Res {
        let base = self.fs_ref().freereg;
        self.new_local_literal(b"(for state)");
        self.new_local_literal(b"(for state)");
        self.new_var(Some(name), VarKind::Const);
        self.checknext(Kind::Char(b'='))?;
        self.next_register_expression()?;
        self.checknext(Kind::Char(b','))?;
        self.next_register_expression()?;
        if self.testnext(Kind::Char(b','))? {
            self.next_register_expression()?;
        } else {
            let fs = self.fs();
            let reg = fs.freereg;
            fs.load_integer(reg, 1);
            fs.reserve_registers(1)?;
        }
        self.adjust_local_vars(2)?;
        self.for_body(base, line, 1, false)
    }

    /// Follows `forlist` in lparser.c.
    fn generic_for(&mut self, name: Ref<LuaString>) -> Res {
        let base = self.fs_ref().freereg;
        let mut nvars = 4;
        self.new_local_literal(b"(for state)");
        self.new_local_literal(b"(for state)");
        self.new_local_literal(b"(for state)");
        self.new_var(Some(name), VarKind::Const);
        while self.testnext(Kind::Char(b','))? {
            let name = self.str_checkname()?;
            self.new_local(name);
            nvars += 1;
        }
        self.checknext(Kind::In)?;
        let line = self.lexer.line;
        let (mut values, count) = self.explist()?;
        self.adjust_assign(4, count, &mut values)?;
        self.adjust_local_vars(3)?;
        self.mark_to_be_closed();
        self.fs().check_stack(2)?;
        self.for_body(base, line, nvars - 3, true)
    }

    /// Follows `forbody` in lparser.c.
    fn for_body(&mut self, base: u8, line: u32, nvars: usize, generic: bool) -> Res {
        self.checknext(Kind::Do)?;
        let fs = self.fs();
        let prep = fs.emit(loop_prepare(generic, base, 0));
        fs.freereg -= 1;
        self.enter_block(0);
        self.adjust_local_vars(nvars)?;
        self.fs().reserve_registers(nvars as u8)?;
        self.block()?;
        self.leave_block()?;
        let fs = self.fs();
        let here = fs.label();
        let offset = for_jump(prep, here, false)?;
        fs.code[prep] = loop_prepare(generic, base, offset);
        if generic {
            fs.emit(Instruction::TForCall { a: base, c: nvars as u8 });
            fs.fix_line(line);
        }
        let endfor = fs.emit(loop_back(generic, base, 0));
        let offset = for_jump(endfor, prep + 1, true)?;
        fs.code[endfor] = loop_back(generic, base, offset);
        fs.fix_line(line);
        Ok(())
    }

    /// Follows `localfunc` in lparser.c; the debug information sees the variable only after the closure is made.
    fn local_function(&mut self) -> Res {
        let fvar = self.fs_ref().nactvar;
        let name = self.str_checkname()?;
        self.new_local(name);
        self.adjust_local_vars(1)?;
        let line = self.lexer.line;
        self.body(false, line)?;
        let pidx = self.var(fvar).pidx as usize;
        let fs = self.fs();
        fs.local_vars[pidx].start_pc = fs.pc();
        Ok(())
    }

    /// Follows `getvarattribute` in lparser.c.
    fn variable_attribute(&mut self, default: VarKind) -> Res<VarKind> {
        if !self.testnext(Kind::Char(b'<'))? {
            return Ok(default);
        }
        let name = self.str_checkname()?;
        self.checknext(Kind::Char(b'>'))?;
        match self.heap.bytes(name) {
            b"const" => Ok(VarKind::Const),
            b"close" => Ok(VarKind::ToClose),
            _ => {
                let text = self.name_text(name);
                self.semantic_error(format!("unknown attribute '{text}'"))
            }
        }
    }

    /// Follows `localstat` in lparser.c.
    fn local_statement(&mut self) -> Res {
        let default = self.variable_attribute(VarKind::Regular)?;
        let mut to_close = None;
        let mut nvars = 0;
        let mut vidx;
        loop {
            let name = self.str_checkname()?;
            let kind = self.variable_attribute(default)?;
            vidx = self.new_var(Some(name), kind);
            if kind == VarKind::ToClose {
                if to_close.is_some() {
                    return self.semantic_error("multiple to-be-closed variables in local list".to_string());
                }
                to_close = Some(self.fs_ref().nactvar + nvars);
            }
            nvars += 1;
            if !self.testnext(Kind::Char(b','))? {
                break;
            }
        }
        let (mut values, nexps) = if self.testnext(Kind::Char(b'='))? {
            self.explist()?
        } else {
            (ExpDesc::new(ExpKind::Void), 0)
        };
        let index = self.fs_ref().first_local + vidx;
        let constant = (nvars == nexps && self.vars[index].kind == VarKind::Const)
            .then(|| FuncState::exp_to_literal(&values))
            .flatten();
        if let Some(value) = constant {
            self.vars[index].kind = VarKind::CompileTimeConstant;
            self.vars[index].value = value;
            self.adjust_local_vars(nvars - 1)?;
            self.fs().nactvar += 1;
        } else {
            self.adjust_assign(nvars, nexps, &mut values)?;
            self.adjust_local_vars(nvars)?;
        }
        self.check_to_close(to_close);
        Ok(())
    }

    /// Follows `checktoclose` in lparser.c.
    fn check_to_close(&mut self, level: Option<usize>) {
        let Some(level) = level else {
            return;
        };
        self.mark_to_be_closed();
        let register = self.reglevel(level);
        self.fs().emit(Instruction::Tbc { a: register });
    }

    /// Follows `getglobalattribute` in lparser.c.
    fn global_attribute(&mut self, default: VarKind) -> Res<VarKind> {
        match self.variable_attribute(default)? {
            VarKind::ToClose => self.semantic_error("global variables cannot be to-be-closed".to_string()),
            VarKind::Const => Ok(VarKind::GlobalConst),
            kind => Ok(kind),
        }
    }

    /// Follows `checkglobal` in lparser.c.
    fn check_global(&mut self, name: Ref<LuaString>, line: u32) -> Res {
        let mut var = self.build_global(name)?;
        let k = var.keystr;
        let fs = self.fs();
        let register = fs.exp_to_any_reg(&mut var)?;
        fs.fix_line(line);
        let bx = if i64::from(k) >= MAX_ARG_BX { 0 } else { k as u32 + 1 };
        fs.emit(Instruction::ErrNil { a: register, bx });
        fs.fix_line(line);
        fs.free_expression(&var);
        Ok(())
    }

    /// Follows `initglobal` in lparser.c; the stores run last name first.
    fn init_global(&mut self, nvars: usize, first: usize, n: usize, line: u32) -> Res {
        if n == nvars {
            let (mut values, count) = self.explist()?;
            return self.adjust_assign(nvars, count, &mut values);
        }
        let name = self.var(first + n).name.expect("a declared global has a name");
        let var = self.build_global(name)?;
        self.enter_level()?;
        self.init_global(nvars, first, n + 1, line)?;
        self.leave_level();
        self.check_global(name, line)?;
        self.store_var_top(&var)
    }

    /// Follows `globalnames` in lparser.c.
    fn global_names(&mut self, default: VarKind) -> Res {
        let mut nvars = 0;
        let mut last;
        loop {
            let name = self.str_checkname()?;
            let kind = self.global_attribute(default)?;
            last = self.new_var(Some(name), kind);
            nvars += 1;
            if !self.testnext(Kind::Char(b','))? {
                break;
            }
        }
        if self.testnext(Kind::Char(b'='))? {
            let line = self.lexer.line;
            self.init_global(nvars, last + 1 - nvars, 0, line)?;
        }
        self.fs().nactvar += nvars;
        Ok(())
    }

    /// Follows `globalstatfunc` in lparser.c.
    fn global_statement(&mut self, line: u32) -> Res {
        self.next()?;
        if self.testnext(Kind::Function)? {
            return self.global_function(line);
        }
        let default = self.global_attribute(VarKind::GlobalRegular)?;
        if !self.testnext(Kind::Char(b'*'))? {
            return self.global_names(default);
        }
        self.new_var(None, default);
        self.fs().nactvar += 1;
        Ok(())
    }

    /// Follows `globalfunc` in lparser.c.
    fn global_function(&mut self, line: u32) -> Res {
        let name = self.str_checkname()?;
        self.new_var(Some(name), VarKind::GlobalRegular);
        self.fs().nactvar += 1;
        let var = self.build_global(name)?;
        let body_line = self.lexer.line;
        let mut body = self.body(false, body_line)?;
        self.check_global(name, line)?;
        let fs = self.fs();
        fs.store_var(&var, &mut body)?;
        fs.fix_line(line);
        Ok(())
    }

    /// Follows `funcstat` in lparser.c.
    fn function_statement(&mut self, line: u32) -> Res {
        self.next()?;
        let mut var = self.single_var()?;
        let mut is_method = false;
        while self.token.kind == Kind::Char(b'.') {
            self.field_select(&mut var)?;
        }
        if self.token.kind == Kind::Char(b':') {
            is_method = true;
            self.field_select(&mut var)?;
        }
        self.check_readonly(&mut var)?;
        let mut body = self.body(is_method, line)?;
        let fs = self.fs();
        fs.store_var(&var, &mut body)?;
        fs.fix_line(line);
        Ok(())
    }

    /// Follows `exprstat` in lparser.c.
    fn expression_statement(&mut self) -> Res {
        let first = self.suffixed_exp()?;
        if matches!(self.token.kind, Kind::Char(b'=' | b',')) {
            return self.assignment(first);
        }
        let ExpKind::Call(pc) = first.kind else {
            return self.syntax_error("syntax error");
        };
        if let Instruction::Call { c, .. } = &mut self.fs().code[pc] {
            *c = 1;
        }
        Ok(())
    }

    /// Follows `restassign` in lparser.c, unrolled; the stores run last target first.
    fn assignment(&mut self, first: ExpDesc) -> Res {
        let mut targets: SmallVec<[ExpDesc; 4]> = SmallVec::new();
        targets.push(first);
        loop {
            let last = targets.len() - 1;
            if !targets[last].kind.is_var() {
                return self.syntax_error("syntax error");
            }
            let mut target = targets[last];
            self.check_readonly(&mut target)?;
            targets[last] = target;
            if !self.testnext(Kind::Char(b','))? {
                break;
            }
            let next = self.suffixed_exp()?;
            if !next.kind.is_indexed() {
                self.check_conflict(&mut targets, &next)?;
            }
            self.enter_level()?;
            targets.push(next);
        }
        let nvars = targets.len();
        self.checknext(Kind::Char(b'='))?;
        let (mut values, nexps) = self.explist()?;
        let mut defaults = nvars;
        if nexps == nvars {
            let fs = self.fs();
            fs.set_one_return(&mut values);
            fs.store_var(&targets[nvars - 1], &mut values)?;
            defaults -= 1;
        } else {
            self.adjust_assign(nvars, nexps, &mut values)?;
        }
        for target in targets[..defaults].iter().rev() {
            self.store_var_top(target)?;
        }
        for _ in 1..nvars {
            self.leave_level();
        }
        Ok(())
    }

    /// Follows `check_conflict` in lparser.c; a local or upvalue that an earlier target indexes is copied first.
    fn check_conflict(&mut self, previous: &mut [ExpDesc], variable: &ExpDesc) -> Res {
        let fs = self.fs();
        let extra = fs.freereg;
        let mut conflict = false;
        for target in previous.iter_mut() {
            if !target.kind.is_indexed() {
                continue;
            }
            match (target.kind, variable.kind) {
                (ExpKind::IndexUp { table, key }, ExpKind::Upval(upval)) => {
                    if table == upval {
                        conflict = true;
                        target.kind = ExpKind::IndexStr { table: extra, key };
                    }
                }
                (ExpKind::IndexUp { .. }, _) => {}
                (kind, ExpKind::Local { reg, .. }) => {
                    let (table, key) = indexed_parts(kind);
                    let mut table_reg = table;
                    let mut key_reg = key;
                    if table == reg {
                        conflict = true;
                        table_reg = extra;
                    }
                    if matches!(kind, ExpKind::Indexed { .. }) && key == reg {
                        conflict = true;
                        key_reg = extra;
                    }
                    target.kind = with_indexed_parts(kind, table_reg, key_reg);
                }
                _ => {}
            }
        }
        if !conflict {
            return Ok(());
        }
        match variable.kind {
            ExpKind::Local { reg, .. } => fs.emit(Instruction::Move { a: extra, b: reg }),
            ExpKind::Upval(upval) => fs.emit(Instruction::GetUpval { a: extra, b: upval }),
            _ => unreachable!("only locals and upvalues conflict"),
        };
        fs.reserve_registers(1)
    }

    /// Follows `storevartop` in lparser.c.
    fn store_var_top(&mut self, var: &ExpDesc) -> Res {
        let fs = self.fs();
        let mut value = ExpDesc::new(ExpKind::NonReloc(fs.freereg - 1));
        fs.store_var(var, &mut value)
    }

    /// Follows `retstat` in lparser.c.
    fn return_statement(&mut self) -> Res {
        let mut first = self.fs_ref().nvarstack();
        let mut count = 0;
        if !self.block_follow(true) && self.token.kind != Kind::Char(b';') {
            let (mut values, listed) = self.explist()?;
            let fs = self.fs();
            if values.has_multiple_results() {
                fs.set_multiple_returns(&mut values)?;
                let inside_tbc = fs.blocks.last().is_some_and(|block| block.inside_tbc);
                if let ExpKind::Call(pc) = values.kind
                    && listed == 1
                    && !inside_tbc
                    && let Instruction::Call { a, b, c } = fs.code[pc]
                {
                    fs.code[pc] = Instruction::TailCall { a, b, c, k: false };
                }
                count = -1;
            } else if listed == 1 {
                first = fs.exp_to_any_reg(&mut values)?;
                count = 1;
            } else {
                fs.exp_to_next_reg(&mut values)?;
                count = listed as i32;
            }
        }
        self.fs().ret(first, count)?;
        self.testnext(Kind::Char(b';'))?;
        Ok(())
    }

    /// Follows `breakstat` in lparser.c.
    fn break_statement(&mut self, line: u32) -> Res {
        let Some(block) = self.fs().blocks.iter_mut().rev().find(|block| block.is_loop != 0) else {
            return self.syntax_error("break outside loop");
        };
        block.is_loop = 2;
        self.next()?;
        let brk = self.brk;
        self.goto_entry(brk, line);
        Ok(())
    }

    /// Follows `labelstat` in lparser.c.
    fn label_statement(&mut self, name: Ref<LuaString>, line: u32) -> Res {
        self.checknext(Kind::DbColon)?;
        while matches!(self.token.kind, Kind::Char(b';') | Kind::DbColon) {
            self.statement()?;
        }
        let first = self.fs_ref().first_label;
        if let Some(label) = self.find_label(name, first) {
            let text = self.name_text(name);
            return self.semantic_error(format!("label '{text}' already defined on line {}", label.line));
        }
        let last = self.block_follow(false);
        self.create_label(name, line, last);
        Ok(())
    }
}

/// Follows `fixforjump` in lparser.c; the offset must fit `Bx`.
fn for_jump(pc: usize, destination: usize, back: bool) -> Res<u32> {
    let mut offset = destination as i64 - (pc as i64 + 1);
    if back {
        offset = -offset;
    }
    if offset > MAX_ARG_BX {
        return Err(Fault::Syntax("control structure too long".to_string()));
    }
    Ok(offset as u32)
}

fn indexed_parts(kind: ExpKind) -> (u8, u8) {
    match kind {
        ExpKind::Indexed { table, key }
        | ExpKind::VarargIndexed { table, key }
        | ExpKind::IndexStr { table, key }
        | ExpKind::IndexUp { table, key } => (table, key),
        ExpKind::IndexInt { table, index } => (table, index),
        _ => unreachable!("only indexed targets have parts"),
    }
}

fn with_indexed_parts(kind: ExpKind, table: u8, key: u8) -> ExpKind {
    match kind {
        ExpKind::Indexed { .. } => ExpKind::Indexed { table, key },
        ExpKind::VarargIndexed { .. } => ExpKind::VarargIndexed { table, key },
        ExpKind::IndexStr { .. } => ExpKind::IndexStr { table, key },
        ExpKind::IndexUp { .. } => ExpKind::IndexUp { table, key },
        ExpKind::IndexInt { .. } => ExpKind::IndexInt { table, index: key },
        _ => unreachable!("only indexed targets have parts"),
    }
}

fn loop_prepare(generic: bool, a: u8, bx: u32) -> Instruction {
    if generic {
        Instruction::TForPrep { a, bx }
    } else {
        Instruction::ForPrep { a, bx }
    }
}

fn loop_back(generic: bool, a: u8, bx: u32) -> Instruction {
    if generic {
        Instruction::TForLoop { a, bx }
    } else {
        Instruction::ForLoop { a, bx }
    }
}
