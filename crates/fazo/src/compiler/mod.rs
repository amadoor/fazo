//! The compiler: one pass that parses and emits code together. Follows PUC Lua 5.5.1's lparser.c (MIT, Copyright (C) 1994-2025 Lua.org, PUC-Rio).
//! Instructions, constants, registers, lines and error messages must match PUC's, so each rule keeps PUC's order.

mod code;
mod expr;
mod func;
mod stat;
#[cfg(test)]
mod tests;

use crate::{
    arena::Ref,
    heap::{Heap, SHORT_STRING_LIMIT},
    instruction::Instruction,
    object::LuaString,
    proto::{LocalVar, Proto, UpvalueDesc},
    session::RequestHandle,
    syntax::{
        C_STACK_OVERFLOW,
        chunk_id::chunk_id,
        lexer::{Lexer, Sink, Strings},
        token::{self, Kind, Token, Value},
    },
};

use func::{BlockScope, ExpDesc, ExpKind, Fault, FuncState, Label, Literal, Res, VarDesc, VarKind};

const MAX_C_LEVELS: u32 = 200;
const MAX_VARS: usize = 200;
const MAX_UPVALUES: usize = 255;

/// Chunk names follow the convention of `load`.
pub(crate) fn compile(
    heap: &mut Heap,
    chunkname: &[u8],
    source: &[u8],
    interrupt: Option<&RequestHandle>,
) -> Result<Ref<Proto>, Vec<u8>> {
    let display = chunk_id(chunkname);
    let scratch = std::mem::take(&mut heap.compiler);
    let mut compiler = Compiler::new(heap, &display, chunkname, source, interrupt, scratch);

    let result = match compiler.main_function() {
        Ok(proto) => Ok(proto),
        Err(fault) => Err(compiler.render(fault)),
    };
    let scratch = compiler.into_scratch();
    heap.compiler = scratch;
    result
}

#[derive(Clone, Default)]
pub(crate) struct Scratch {
    functions: Vec<FuncState>,
    vars: Vec<VarDesc>,
    gotos: Vec<Label>,
    labels: Vec<Label>,
    strings: Strings,
    buffer: Vec<u8>,
}

struct Compiler<'a> {
    heap: &'a mut Heap,
    strings: Strings,
    lexer: Lexer<'a>,
    interrupt: Option<&'a RequestHandle>,
    token: Token,
    lookahead: Option<Token>,
    last_line: u32,
    depth: u32,
    source: std::sync::Arc<[u8]>,
    chunkname: std::sync::Arc<[u8]>,
    functions: Vec<FuncState>,
    open: usize,
    vars: Vec<VarDesc>,
    var_count: usize,
    gotos: Vec<Label>,
    labels: Vec<Label>,
    env: Ref<LuaString>,
    brk: Ref<LuaString>,
    global: Ref<LuaString>,
}

impl<'a> Compiler<'a> {
    fn new(
        heap: &'a mut Heap,
        display: &'a [u8],
        chunkname: &'a [u8],
        source: &'a [u8],
        interrupt: Option<&'a RequestHandle>,
        scratch: Scratch,
    ) -> Self {
        let env = heap.string(b"_ENV");
        let brk = heap.string(b"break");
        let global = heap.string(b"global");
        Self {
            heap,
            strings: scratch.strings,
            lexer: Lexer::new(display, source, scratch.buffer),
            interrupt,
            token: Token::new(Kind::Eos),
            lookahead: None,
            last_line: 1,
            depth: 0,
            source: display.into(),
            chunkname: chunkname.into(),
            functions: scratch.functions,
            open: 0,
            vars: scratch.vars,
            var_count: 0,
            gotos: scratch.gotos,
            labels: scratch.labels,
            env,
            brk,
            global,
        }
    }

    fn into_scratch(self) -> Scratch {
        let mut scratch = Scratch {
            functions: self.functions,
            vars: self.vars,
            gotos: self.gotos,
            labels: self.labels,
            strings: self.strings,
            buffer: self.lexer.into_buffer(),
        };
        scratch.vars.clear();
        scratch.gotos.clear();
        scratch.labels.clear();
        scratch.strings.clear();
        scratch
    }

    /// Follows `lexerror` in llex.c.
    fn render(&self, fault: Fault) -> Vec<u8> {
        match fault {
            Fault::Syntax(message) => self.lexer.message(&message, Some(self.lexer.near(self.token.kind))),
            Fault::Semantic(message) => self.lexer.message_at(self.last_line, &message, None),
            Fault::Limit {
                what,
                limit,
                line_defined,
            } => {
                let place = if line_defined == 0 {
                    "main function".to_string()
                } else {
                    format!("function at line {line_defined}")
                };
                let message = format!("too many {what} (limit is {limit}) in {place}");
                self.lexer.message(&message, Some(self.lexer.near(self.token.kind)))
            }
            Fault::Message(message) => message,
        }
    }

    fn fs(&mut self) -> &mut FuncState {
        &mut self.functions[self.open - 1]
    }

    fn fs_ref(&self) -> &FuncState {
        &self.functions[self.open - 1]
    }

    fn level(&self) -> usize {
        self.open - 1
    }

    fn lex(&mut self) -> Res<Token> {
        let mut sink = Sink {
            heap: self.heap,
            strings: &mut self.strings,
        };
        Ok(self.lexer.lex(&mut sink)?)
    }

    /// Follows `luaX_next` in llex.c.
    fn next(&mut self) -> Res {
        self.last_line = self.lexer.line;
        self.token = match self.lookahead.take() {
            Some(token) => token,
            None => self.lex()?,
        };
        if self.open > 0 {
            let line = self.last_line;
            self.fs().line = line;
        }
        Ok(())
    }

    fn lookahead_kind(&mut self) -> Res<Kind> {
        if self.lookahead.is_none() {
            let token = self.lex()?;
            self.lookahead = Some(token);
        }
        Ok(self.lookahead.map_or(Kind::Eos, |token| token.kind))
    }

    fn syntax_error<T>(&self, message: impl Into<String>) -> Res<T> {
        Err(Fault::Syntax(message.into()))
    }

    fn semantic_error<T>(&self, message: String) -> Res<T> {
        Err(Fault::Semantic(message))
    }

    fn error_expected<T>(&self, kind: Kind) -> Res<T> {
        self.syntax_error(format!("{} expected", token::describe(kind)))
    }

    fn check(&self, kind: Kind) -> Res {
        if self.token.kind != kind {
            return self.error_expected(kind);
        }
        Ok(())
    }

    fn checknext(&mut self, kind: Kind) -> Res {
        self.check(kind)?;
        self.next()
    }

    fn testnext(&mut self, kind: Kind) -> Res<bool> {
        if self.token.kind != kind {
            return Ok(false);
        }
        self.next()?;
        Ok(true)
    }

    fn check_match(&mut self, what: Kind, who: Kind, line: u32) -> Res {
        if self.testnext(what)? {
            return Ok(());
        }
        if line == self.lexer.line {
            return self.error_expected(what);
        }
        self.syntax_error(format!(
            "{} expected (to close {} at line {line})",
            token::describe(what),
            token::describe(who)
        ))
    }

    fn str_checkname(&mut self) -> Res<Ref<LuaString>> {
        self.check(Kind::Name)?;
        let Value::Str(name) = self.token.value else {
            unreachable!("a name token carries its string");
        };
        self.next()?;
        Ok(name)
    }

    fn name_text(&self, name: Ref<LuaString>) -> String {
        String::from_utf8_lossy(self.heap.bytes(name)).into_owned()
    }

    fn is_short(&self, expression: &ExpDesc) -> bool {
        match expression.kind {
            ExpKind::Str(string) => self.heap.bytes(string).len() <= SHORT_STRING_LIMIT,
            _ => false,
        }
    }

    fn enter_level(&mut self) -> Res {
        self.depth += 1;
        if self.depth >= MAX_C_LEVELS {
            return Err(Fault::Message(C_STACK_OVERFLOW.to_vec()));
        }
        Ok(())
    }

    fn leave_level(&mut self) {
        self.depth -= 1;
    }

    fn block_follow(&self, with_until: bool) -> bool {
        match self.token.kind {
            Kind::Else | Kind::Elseif | Kind::End | Kind::Eos => true,
            Kind::Until => with_until,
            _ => false,
        }
    }

    fn check_interrupt(&self) -> Res {
        if self.interrupt.is_some_and(RequestHandle::is_requested) {
            return Err(Fault::Message(b"interrupted".to_vec()));
        }
        Ok(())
    }
}

impl Compiler<'_> {
    /// Follows `new_varkind` in lparser.c; entries above `var_count` stay readable for the goto checks, as PUC reads them.
    fn new_var(&mut self, name: Option<Ref<LuaString>>, kind: VarKind) -> usize {
        self.vars.truncate(self.var_count);
        self.vars.push(VarDesc {
            kind,
            ridx: 0,
            pidx: 0,
            name,
            value: Literal::Nil,
        });
        self.var_count += 1;
        self.var_count - 1 - self.fs_ref().first_local
    }

    fn new_local(&mut self, name: Ref<LuaString>) -> usize {
        self.new_var(Some(name), VarKind::Regular)
    }

    fn new_local_literal(&mut self, name: &[u8]) -> usize {
        let name = self.heap.string(name);
        self.new_local(name)
    }

    fn var(&self, vidx: usize) -> &VarDesc {
        &self.vars[self.fs_ref().first_local + vidx]
    }

    /// Follows `reglevel` in lparser.c.
    fn reglevel(&self, nvar: usize) -> u8 {
        let first = self.fs_ref().first_local;
        self.vars[first..first + nvar]
            .iter()
            .rev()
            .find(|var| var.kind.in_register())
            .map_or(0, |var| var.ridx + 1)
    }

    /// Follows `registerlocalvar` in lparser.c.
    fn register_local_var(&mut self, name: Ref<LuaString>) -> u16 {
        let fs = &mut self.functions[self.open - 1];
        let pc = fs.pc();
        fs.local_vars.push(LocalVar {
            name: Some(name),
            start_pc: pc,
            end_pc: pc,
        });
        (fs.local_vars.len() - 1) as u16
    }

    /// Follows `adjustlocalvars` in lparser.c.
    fn adjust_local_vars(&mut self, count: usize) -> Res {
        let mut level = self.fs_ref().reg_level;
        for _ in 0..count {
            let fs = &mut self.functions[self.open - 1];
            let index = fs.first_local + fs.nactvar;
            fs.nactvar += 1;
            let name = self.vars[index].name.expect("a local has a name");
            self.vars[index].ridx = level;
            level += 1;
            let pidx = self.register_local_var(name);
            self.vars[index].pidx = pidx;
            let fs = self.fs();
            fs.reg_level = level;
            if level as usize > MAX_VARS {
                return Err(fs.limit("local variables", MAX_VARS));
            }
        }
        Ok(())
    }

    /// Follows `removevars` in lparser.c.
    fn remove_vars(&mut self, to_level: usize) {
        let fs = &mut self.functions[self.open - 1];
        let pc = fs.pc();
        self.var_count -= fs.nactvar - to_level;
        while fs.nactvar > to_level {
            fs.nactvar -= 1;
            let var = &self.vars[fs.first_local + fs.nactvar];
            if var.kind.in_register() {
                fs.local_vars[var.pidx as usize].end_pc = pc;
            }
        }
        let level = self.reglevel(to_level);
        self.fs().reg_level = level;
    }

    fn search_upvalue(&self, level: usize, name: Ref<LuaString>) -> Option<u8> {
        self.functions[level]
            .upvalue_names
            .iter()
            .position(|&upvalue| upvalue == name)
            .map(|index| index as u8)
    }

    /// Follows `allocupvalue` in lparser.c.
    fn alloc_upvalue(&mut self, level: usize, name: Ref<LuaString>, desc: (bool, u8, VarKind)) -> Res<u8> {
        let (in_stack, index, kind) = desc;
        let fs = &mut self.functions[level];
        if fs.upvalues.len() + 1 > MAX_UPVALUES {
            return Err(fs.limit("upvalues", MAX_UPVALUES));
        }
        fs.upvalues.push(UpvalueDesc {
            name: Some(name),
            in_stack,
            index,
        });
        fs.upvalue_names.push(name);
        fs.upvalue_kinds.push(kind);
        Ok((fs.upvalues.len() - 1) as u8)
    }

    /// Follows `newupvalue` in lparser.c.
    fn new_upvalue(&mut self, level: usize, name: Ref<LuaString>, var: &ExpDesc) -> Res<u8> {
        let previous = &self.functions[level - 1];
        let desc = match var.kind {
            ExpKind::Local { reg, var } => (true, reg, self.vars[previous.first_local + var as usize].kind),
            ExpKind::Upval(index) => (false, index, previous.upvalue_kinds[index as usize]),
            _ => unreachable!("only locals and upvalues become upvalues"),
        };
        self.alloc_upvalue(level, name, desc)
    }

    /// Follows `searchvar` in lparser.c; `info` is -1 for the default global declaration and -2 once a named one hides it.
    fn search_var(&self, level: usize, name: Ref<LuaString>, var: &mut ExpDesc) -> bool {
        let fs = &self.functions[level];
        let ExpKind::Global(mut info) = var.kind else {
            unreachable!("a search starts from a global");
        };
        for index in (0..fs.nactvar).rev() {
            let absolute = fs.first_local + index;
            let vd = &self.vars[absolute];
            if vd.kind.is_global() {
                match vd.name {
                    None => {
                        if info < 0 {
                            info = absolute as i32;
                        }
                    }
                    Some(declared) if declared == name => {
                        *var = ExpDesc::new(ExpKind::Global(absolute as i32));
                        return true;
                    }
                    Some(_) => {
                        if info == -1 {
                            info = -2;
                        }
                    }
                }
                continue;
            }
            if vd.name != Some(name) {
                continue;
            }
            *var = ExpDesc::new(match vd.kind {
                VarKind::CompileTimeConstant => ExpKind::Const {
                    var: absolute as u32,
                    value: vd.value,
                },
                VarKind::Vararg => ExpKind::VarargVar {
                    reg: vd.ridx,
                    var: index as u16,
                },
                _ => ExpKind::Local {
                    reg: vd.ridx,
                    var: index as u16,
                },
            });
            return true;
        }
        var.kind = ExpKind::Global(info);
        false
    }

    /// Follows `markupval` in lparser.c.
    fn mark_upval(&mut self, level: usize, var: usize) {
        let fs = &mut self.functions[level];
        let block = fs
            .blocks
            .iter_mut()
            .rev()
            .find(|block| block.nactvar <= var)
            .expect("the function's outer block covers every local");
        block.upval = true;
        fs.needclose = true;
    }

    /// Follows `marktobeclosed` in lparser.c.
    fn mark_to_be_closed(&mut self) {
        let fs = self.fs();
        let block = fs.blocks.last_mut().expect("a block is open");
        block.upval = true;
        block.inside_tbc = true;
        fs.needclose = true;
    }

    /// Follows `singlevaraux` in lparser.c.
    fn single_var_aux(&mut self, level: usize, name: Ref<LuaString>, var: &mut ExpDesc, base: bool) -> Res {
        if self.search_var(level, name, var) {
            if base {
                return Ok(());
            }
            if let ExpKind::VarargVar { reg, var: index } = var.kind {
                self.functions[level].vararg_table = true;
                var.kind = ExpKind::Local { reg, var: index };
            }
            if let ExpKind::Local { var: index, .. } = var.kind {
                self.mark_upval(level, index as usize);
            }
            return Ok(());
        }
        let index = match self.search_upvalue(level, name) {
            Some(index) => index,
            None => {
                if level > 0 {
                    self.single_var_aux(level - 1, name, var, false)?;
                }
                if !matches!(var.kind, ExpKind::Local { .. } | ExpKind::Upval(_)) {
                    return Ok(());
                }
                self.new_upvalue(level, name, var)?
            }
        };
        *var = ExpDesc::new(ExpKind::Upval(index));
        Ok(())
    }

    /// Follows `buildglobal` in lparser.c.
    fn build_global(&mut self, name: Ref<LuaString>) -> Res<ExpDesc> {
        let level = self.level();
        let mut var = ExpDesc::new(ExpKind::Global(-1));
        let env = self.env;
        self.single_var_aux(level, env, &mut var, true)?;
        if matches!(var.kind, ExpKind::Global(_)) {
            let text = self.name_text(name);
            return self.semantic_error(format!("_ENV is global when accessing variable '{text}'"));
        }
        self.fs().exp_to_any_reg_up(&mut var)?;
        let mut key = ExpDesc::new(ExpKind::Str(name));
        let short = self.is_short(&key);
        self.fs().indexed(&mut var, &mut key, short)?;
        Ok(var)
    }

    /// Follows `buildvar` in lparser.c.
    fn build_var(&mut self, name: Ref<LuaString>) -> Res<ExpDesc> {
        let level = self.level();
        let mut var = ExpDesc::new(ExpKind::Global(-1));
        self.single_var_aux(level, name, &mut var, true)?;
        let ExpKind::Global(info) = var.kind else {
            return Ok(var);
        };
        if info == -2 {
            let text = self.name_text(name);
            return self.semantic_error(format!("variable '{text}' not declared"));
        }
        let mut var = self.build_global(name)?;
        if info != -1 && self.vars[info as usize].kind == VarKind::GlobalConst {
            var.ro = true;
        }
        Ok(var)
    }

    fn single_var(&mut self) -> Res<ExpDesc> {
        let name = self.str_checkname()?;
        self.build_var(name)
    }

    /// Follows `check_readonly` in lparser.c.
    fn check_readonly(&mut self, expression: &mut ExpDesc) -> Res {
        let name = match expression.kind {
            ExpKind::Const { var, .. } => self.vars[var as usize].name,
            ExpKind::Local { var, .. } | ExpKind::VarargVar { var, .. } => {
                let vd = self.var(var as usize);
                (vd.kind != VarKind::Regular).then_some(vd.name).flatten()
            }
            ExpKind::Upval(index) => {
                let fs = self.fs_ref();
                (fs.upvalue_kinds[index as usize] != VarKind::Regular).then_some(fs.upvalue_names[index as usize])
            }
            ExpKind::VarargIndexed { table, key } => {
                self.fs().vararg_table = true;
                expression.kind = ExpKind::Indexed { table, key };
                self.readonly_key(expression)
            }
            ExpKind::IndexUp { .. } | ExpKind::IndexStr { .. } | ExpKind::Indexed { .. } => self.readonly_key(expression),
            _ => return Ok(()),
        };
        let Some(name) = name else {
            return Ok(());
        };
        let text = self.name_text(name);
        self.semantic_error(format!("attempt to assign to const variable '{text}'"))
    }

    fn readonly_key(&self, expression: &ExpDesc) -> Option<Ref<LuaString>> {
        if !expression.ro {
            return None;
        }
        match self.fs_ref().constants[expression.keystr as usize] {
            crate::value::Value::String(name) => Some(name),
            _ => None,
        }
    }

    /// Follows `adjust_assign` in lparser.c.
    fn adjust_assign(&mut self, nvars: usize, nexps: usize, expression: &mut ExpDesc) -> Res {
        let needed = nvars as i64 - nexps as i64;
        let fs = self.fs();
        if needed > 0 {
            fs.check_stack(needed as usize)?;
        }
        if expression.has_multiple_results() {
            let extra = (needed + 1).max(0);
            fs.set_returns(expression, extra as i32)?;
        } else {
            if !matches!(expression.kind, ExpKind::Void) {
                fs.exp_to_next_reg(expression)?;
            }
            if needed > 0 {
                let from = fs.freereg;
                fs.nil(from, needed as u8);
            }
        }
        if needed > 0 {
            fs.reserve_registers(needed as u8)?;
        } else {
            fs.freereg = (i64::from(fs.freereg) + needed) as u8;
        }
        Ok(())
    }
}

impl Compiler<'_> {
    fn label_name(&self, label: &Label) -> String {
        self.name_text(label.name)
    }

    /// Follows `jumpscopeerror` in lparser.c.
    fn jump_scope_error<T>(&self, goto: &Label) -> Res<T> {
        let var = self.vars[self.fs_ref().first_local + goto.nactvar].name;
        let var = var.map_or_else(|| "*".to_string(), |name| self.name_text(name));
        self.semantic_error(format!(
            "<goto {}> at line {} jumps into the scope of '{var}'",
            self.label_name(goto),
            goto.line
        ))
    }

    /// Follows `closegoto` in lparser.c.
    fn close_goto(&mut self, index: usize, label: Label, bup: bool) -> Res {
        let goto = self.gotos[index];
        if goto.nactvar < label.nactvar {
            return self.jump_scope_error(&goto);
        }
        let mut pc = goto.pc;
        if goto.close || (label.nactvar < goto.nactvar && bup) {
            let level = self.reglevel(label.nactvar);
            let fs = self.fs();
            fs.code[pc + 1] = fs.code[pc];
            fs.code[pc] = Instruction::Close { a: level };
            pc += 1;
        }
        self.fs().patch_list(pc as isize, label.pc);
        self.gotos.remove(index);
        Ok(())
    }

    fn find_label(&self, name: Ref<LuaString>, first: usize) -> Option<Label> {
        self.labels[first..].iter().find(|label| label.name == name).copied()
    }

    fn label_entry(&self, name: Ref<LuaString>, line: u32, pc: usize) -> Label {
        Label {
            name,
            pc,
            line,
            nactvar: self.fs_ref().nactvar,
            close: false,
        }
    }

    /// Follows `newgotoentry` in lparser.c; the placeholder `CLOSE` after the jump is dead unless `close_goto` moves it in front.
    fn goto_entry(&mut self, name: Ref<LuaString>, line: u32) {
        let fs = self.fs();
        let pc = fs.jump() as usize;
        fs.emit(Instruction::Close { a: 0 });
        let goto = self.label_entry(name, line, pc);
        self.gotos.push(goto);
    }

    /// Follows `createlabel` in lparser.c.
    fn create_label(&mut self, name: Ref<LuaString>, line: u32, last: bool) {
        let pc = self.fs().label();
        let mut label = self.label_entry(name, line, pc);
        if last {
            label.nactvar = self.fs_ref().blocks.last().expect("a block is open").nactvar;
        }
        self.labels.push(label);
    }

    /// Follows `solvegotos` in lparser.c.
    fn solve_gotos(&mut self, block: &BlockScope) -> Res {
        let out_level = self.reglevel(block.nactvar);
        let mut index = block.first_goto;
        while index < self.gotos.len() {
            let goto = self.gotos[index];
            if let Some(label) = self.find_label(goto.name, block.first_label) {
                self.close_goto(index, label, block.upval)?;
                continue;
            }
            if block.upval && self.reglevel(goto.nactvar) > out_level {
                self.gotos[index].close = true;
            }
            self.gotos[index].nactvar = block.nactvar;
            index += 1;
        }
        self.labels.truncate(block.first_label);
        Ok(())
    }

    /// Follows `enterblock` in lparser.c.
    fn enter_block(&mut self, is_loop: u8) {
        let first_label = self.labels.len();
        let first_goto = self.gotos.len();
        let fs = self.fs();
        let inside_tbc = fs.blocks.last().is_some_and(|block| block.inside_tbc);
        let block = BlockScope {
            first_label,
            first_goto,
            nactvar: fs.nactvar,
            upval: false,
            is_loop,
            inside_tbc,
        };
        fs.blocks.push(block);
    }

    /// Follows `leaveblock` in lparser.c.
    fn leave_block(&mut self) -> Res {
        let block = *self.fs_ref().blocks.last().expect("a block is open");
        let nested = self.fs_ref().blocks.len() > 1;
        let level = self.reglevel(block.nactvar);
        if nested && block.upval {
            self.fs().emit(Instruction::Close { a: level });
        }
        self.fs().freereg = level;
        self.remove_vars(block.nactvar);
        if block.is_loop == 2 {
            let brk = self.brk;
            self.create_label(brk, 0, false);
        }
        self.solve_gotos(&block)?;
        if !nested && let Some(goto) = self.gotos.get(block.first_goto).copied() {
            return self.semantic_error(format!(
                "no visible label '{}' for <goto> at line {}",
                self.label_name(&goto),
                goto.line
            ));
        }
        self.fs().blocks.pop();
        Ok(())
    }
}

impl Compiler<'_> {
    /// Follows `open_func` in lparser.c.
    fn open_function(&mut self, line_defined: u32) {
        let first_local = self.var_count;
        let first_label = self.labels.len();
        let line = self.last_line;
        if self.functions.len() == self.open {
            self.functions.push(FuncState::default());
        }
        self.functions[self.open].reset(first_local, first_label, line_defined, line);
        self.open += 1;
        self.enter_block(0);
    }

    /// Follows `close_func` in lparser.c; the prototype's buffers are shrunk to exact size as `luaM_shrinkvector` does.
    fn close_function(&mut self) -> Res<Ref<Proto>> {
        let level = self.fs_ref().nvarstack();
        self.fs().ret(level, 0)?;
        self.leave_block()?;
        let fs = &mut self.functions[self.open - 1];
        fs.finish();
        let proto = Proto {
            source: std::sync::Arc::clone(&self.source),
            chunkname: std::sync::Arc::clone(&self.chunkname),
            line_defined: fs.line_defined,
            last_line_defined: fs.last_line_defined,
            num_params: fs.num_params,
            is_vararg: fs.is_vararg,
            vararg_table: fs.vararg_table,
            max_stack: u8::try_from(fs.max_stack).expect("register count is below the limit"),
            exec: Box::new(crate::vm::Exec::new(&fs.code)),
            code: fs.code.as_slice().into(),
            lines: fs.lines.as_slice().into(),
            constants: fs.constants.as_slice().into(),
            protos: fs.protos.as_slice().into(),
            upvalues: std::mem::take(&mut fs.upvalues),
            local_vars: std::mem::take(&mut fs.local_vars),
            calls: 0,
            #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
            jit: Default::default(),
        };
        let proto = self.heap.alloc_proto(proto);
        self.open -= 1;
        if self.open > 0 {
            let line = self.last_line;
            let parent = self.fs();
            parent.protos.push(proto);
            parent.line = line;
        }
        Ok(proto)
    }

    /// Follows `codeclosure` in lparser.c; the `CLOSURE` is emitted before the nested function closes.
    fn code_closure(&mut self) -> Res<ExpDesc> {
        let line = self.last_line;
        let parent = &mut self.functions[self.open - 2];
        parent.line = line;
        let index = parent.protos.len() as u32;
        let pc = parent.emit(Instruction::Closure { a: 0, bx: index });
        let mut closure = ExpDesc::new(ExpKind::Reloc(pc));
        parent.exp_to_next_reg(&mut closure)?;
        Ok(closure)
    }

    /// Follows `setvararg` in lparser.c.
    fn set_vararg(&mut self) {
        let fs = self.fs();
        fs.is_vararg = true;
        fs.emit(Instruction::VarargPrep { a: 0 });
    }

    /// Follows `mainfunc` in lparser.c.
    fn main_function(&mut self) -> Res<Ref<Proto>> {
        self.open_function(0);
        self.set_vararg();
        let env = self.env;
        self.alloc_upvalue(0, env, (true, 0, VarKind::Regular))?;
        self.next()?;
        self.statlist()?;
        self.check(Kind::Eos)?;
        self.close_function()
    }
}
