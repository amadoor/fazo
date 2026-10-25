//! Follows the state structs of PUC Lua 5.5.1's lparser.h (MIT, Copyright (C) 1994-2025 Lua.org, PUC-Rio).

use hashbrown::HashMap;

use crate::{
    arena::Ref,
    instruction::Instruction,
    object::LuaString,
    proto::{LocalVar, UpvalueDesc},
    value::Value,
};

pub(super) const NO_JUMP: isize = -1;
pub(super) const NO_REG: u8 = u8::MAX;
pub(super) const MAX_REGISTERS: usize = 255;

#[derive(Debug)]
pub(super) enum Fault {
    Syntax(String),
    Semantic(String),
    /// Follows `errorlimit` in lparser.c.
    Limit {
        what: &'static str,
        limit: usize,
        line_defined: u32,
    },
    /// Final as is: a scanner error, or a nesting or interrupt abort.
    Message(Vec<u8>),
}

impl From<Vec<u8>> for Fault {
    fn from(message: Vec<u8>) -> Self {
        Self::Message(message)
    }
}

pub(super) type Res<T = ()> = Result<T, Fault>;

/// The value of a compile-time constant (`RDKCTC` in PUC).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Literal {
    Nil,
    True,
    False,
    Integer(i64),
    Float(f64),
    Str(Ref<LuaString>),
}

impl Literal {
    pub fn kind(self) -> ExpKind {
        match self {
            Literal::Nil => ExpKind::Nil,
            Literal::True => ExpKind::True,
            Literal::False => ExpKind::False,
            Literal::Integer(value) => ExpKind::Integer(value),
            Literal::Float(value) => ExpKind::Float(value),
            Literal::Str(string) => ExpKind::Str(string),
        }
    }
}

/// Follows `expkind` in lparser.h.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum ExpKind {
    Void,
    Nil,
    True,
    False,
    Constant(u32),
    Float(f64),
    Integer(i64),
    Str(Ref<LuaString>),
    NonReloc(u8),
    Local { reg: u8, var: u16 },
    VarargVar { reg: u8, var: u16 },
    Global(i32),
    Upval(u8),
    Const { var: u32, value: Literal },
    Indexed { table: u8, key: u8 },
    VarargIndexed { table: u8, key: u8 },
    IndexUp { table: u8, key: u8 },
    IndexInt { table: u8, index: u8 },
    IndexStr { table: u8, key: u8 },
    Jump(usize),
    Reloc(usize),
    Call(usize),
    Vararg(usize),
}

impl ExpKind {
    /// Follows `vkisvar` in lparser.h.
    pub fn is_var(self) -> bool {
        matches!(
            self,
            ExpKind::Local { .. }
                | ExpKind::VarargVar { .. }
                | ExpKind::Global(_)
                | ExpKind::Upval(_)
                | ExpKind::Const { .. }
                | ExpKind::Indexed { .. }
                | ExpKind::VarargIndexed { .. }
                | ExpKind::IndexUp { .. }
                | ExpKind::IndexInt { .. }
                | ExpKind::IndexStr { .. }
        )
    }

    /// Follows `vkisindexed` in lparser.h.
    pub fn is_indexed(self) -> bool {
        matches!(
            self,
            ExpKind::Indexed { .. }
                | ExpKind::VarargIndexed { .. }
                | ExpKind::IndexUp { .. }
                | ExpKind::IndexInt { .. }
                | ExpKind::IndexStr { .. }
        )
    }
}

/// Follows `expdesc` in lparser.h; `keystr` is -1 when the key is not a string.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct ExpDesc {
    pub kind: ExpKind,
    pub t: isize,
    pub f: isize,
    pub ro: bool,
    pub keystr: i32,
}

impl ExpDesc {
    pub fn new(kind: ExpKind) -> Self {
        Self {
            kind,
            t: NO_JUMP,
            f: NO_JUMP,
            ro: false,
            keystr: -1,
        }
    }

    pub fn has_multiple_results(&self) -> bool {
        matches!(self.kind, ExpKind::Call(_) | ExpKind::Vararg(_))
    }
}

/// Variant order matches PUC's lparser.h.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum VarKind {
    Regular,
    Const,
    Vararg,
    ToClose,
    CompileTimeConstant,
    GlobalRegular,
    GlobalConst,
}

impl VarKind {
    pub fn in_register(self) -> bool {
        self <= VarKind::ToClose
    }

    pub fn is_global(self) -> bool {
        self >= VarKind::GlobalRegular
    }
}

/// `name` is `None` for a collective `global *` declaration.
#[derive(Clone, Copy, Debug)]
pub(super) struct VarDesc {
    pub kind: VarKind,
    pub ridx: u8,
    pub pidx: u16,
    pub name: Option<Ref<LuaString>>,
    pub value: Literal,
}

/// Follows `BlockCnt` in lparser.c; `is_loop` is 2 for a loop with pending breaks.
#[derive(Clone, Copy, Debug)]
pub(super) struct BlockScope {
    pub first_label: usize,
    pub first_goto: usize,
    pub nactvar: usize,
    pub upval: bool,
    pub is_loop: u8,
    pub inside_tbc: bool,
}

/// Follows `Labeldesc` in lparser.h.
#[derive(Clone, Copy, Debug)]
pub(super) struct Label {
    pub name: Ref<LuaString>,
    pub pc: usize,
    pub line: u32,
    pub nactvar: usize,
    pub close: bool,
}

/// Floats are keyed by `value * (1 + 2^-52)`, which keeps integral floats apart from integers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum ConstKey {
    Nil,
    Boolean(bool),
    Integer(i64),
    Float(u64),
    Str(Ref<LuaString>),
}

/// Follows `FuncState` in lparser.h.
#[derive(Clone, Debug, Default)]
pub(super) struct FuncState {
    pub code: Vec<Instruction>,
    pub lines: Vec<u32>,
    pub constants: Vec<Value>,
    pub constant_index: HashMap<ConstKey, u32>,
    pub protos: Vec<Ref<crate::proto::Proto>>,
    pub upvalues: Vec<UpvalueDesc>,
    pub upvalue_names: Vec<Ref<LuaString>>,
    pub upvalue_kinds: Vec<VarKind>,
    pub local_vars: Vec<LocalVar>,
    pub blocks: Vec<BlockScope>,
    pub first_local: usize,
    pub first_label: usize,
    pub nactvar: usize,
    pub reg_level: u8,
    pub freereg: u8,
    pub max_stack: usize,
    pub last_target: usize,
    pub line: u32,
    pub is_vararg: bool,
    pub vararg_table: bool,
    pub num_params: u8,
    pub line_defined: u32,
    pub last_line_defined: u32,
    pub needclose: bool,
}

impl FuncState {
    /// Follows `open_func` in lparser.c.
    pub fn reset(&mut self, first_local: usize, first_label: usize, line_defined: u32, line: u32) {
        self.code.clear();
        self.lines.clear();
        self.constants.clear();
        self.constant_index.clear();
        self.protos.clear();
        self.upvalues.clear();
        self.upvalue_names.clear();
        self.upvalue_kinds.clear();
        self.local_vars.clear();
        self.blocks.clear();
        self.first_local = first_local;
        self.first_label = first_label;
        self.nactvar = 0;
        self.reg_level = 0;
        self.freereg = 0;
        self.max_stack = 2;
        self.last_target = 0;
        self.line = line;
        self.is_vararg = false;
        self.vararg_table = false;
        self.num_params = 0;
        self.line_defined = line_defined;
        self.last_line_defined = 0;
        self.needclose = false;
    }

    pub fn pc(&self) -> usize {
        self.code.len()
    }

    pub fn nvarstack(&self) -> u8 {
        self.reg_level
    }

    pub fn emit(&mut self, instruction: Instruction) -> usize {
        self.code.push(instruction);
        self.lines.push(self.line);
        self.code.len() - 1
    }

    pub fn remove_last_instruction(&mut self) {
        self.code.pop();
        self.lines.pop();
    }

    pub fn fix_line(&mut self, line: u32) {
        if let Some(last) = self.lines.last_mut() {
            *last = line;
        }
    }

    /// Follows `previousinstruction` in lcode.c; yields nothing when a jump target lies after the last instruction.
    pub fn previous_instruction(&self) -> Option<Instruction> {
        if self.pc() > self.last_target {
            self.code.last().copied()
        } else {
            None
        }
    }

    pub fn jump(&mut self) -> isize {
        self.emit(Instruction::Jmp { sj: NO_JUMP as i32 }) as isize
    }

    pub fn label(&mut self) -> usize {
        self.last_target = self.pc();
        self.last_target
    }

    pub fn get_jump(&self, pc: usize) -> isize {
        match self.code[pc] {
            Instruction::Jmp { sj } if sj as isize != NO_JUMP => pc as isize + 1 + sj as isize,
            _ => NO_JUMP,
        }
    }

    pub fn fix_jump(&mut self, pc: usize, destination: usize) {
        let offset = destination as isize - (pc as isize + 1);
        self.code[pc] = Instruction::Jmp { sj: offset as i32 };
    }

    pub fn concat(&mut self, list: &mut isize, other: isize) {
        if other == NO_JUMP {
            return;
        }
        if *list == NO_JUMP {
            *list = other;
            return;
        }
        let mut last = *list as usize;
        loop {
            let next = self.get_jump(last);
            if next == NO_JUMP {
                break;
            }
            last = next as usize;
        }
        self.fix_jump(last, other as usize);
    }

    pub fn jump_control(&self, pc: usize) -> usize {
        if pc >= 1 && self.code[pc - 1].is_test() {
            pc - 1
        } else {
            pc
        }
    }

    /// Follows `patchtestreg` in lcode.c.
    pub fn patch_test_register(&mut self, node: usize, register: u8) -> bool {
        let control = self.jump_control(node);
        let Instruction::TestSet { a: _, b, k } = self.code[control] else {
            return false;
        };
        self.code[control] = if register != NO_REG && register != b {
            Instruction::TestSet { a: register, b, k }
        } else {
            Instruction::Test { a: b, k }
        };
        true
    }

    fn remove_values(&mut self, mut list: isize) {
        while list != NO_JUMP {
            self.patch_test_register(list as usize, NO_REG);
            list = self.get_jump(list as usize);
        }
    }

    pub fn patch_list_aux(&mut self, mut list: isize, value_target: usize, register: u8, default_target: usize) {
        while list != NO_JUMP {
            let next = self.get_jump(list as usize);
            if self.patch_test_register(list as usize, register) {
                self.fix_jump(list as usize, value_target);
            } else {
                self.fix_jump(list as usize, default_target);
            }
            list = next;
        }
    }

    pub fn patch_list(&mut self, list: isize, target: usize) {
        self.patch_list_aux(list, target, NO_REG, target);
    }

    pub fn patch_to_here(&mut self, list: isize) {
        let here = self.label();
        self.patch_list(list, here);
    }

    pub fn code_jump_back(&mut self, target: usize) {
        let jump = self.jump();
        self.patch_list(jump, target);
    }

    pub fn remove_values_of(&mut self, list: isize) {
        self.remove_values(list);
    }

    pub fn reserve_registers(&mut self, count: u8) -> Res {
        self.check_stack(count as usize)?;
        self.freereg += count;
        Ok(())
    }

    /// Follows `luaK_checkstack` in lcode.c.
    pub fn check_stack(&mut self, count: usize) -> Res {
        let needed = self.freereg as usize + count;
        if needed > self.max_stack {
            if needed > MAX_REGISTERS {
                return Err(self.limit("registers", MAX_REGISTERS));
            }
            self.max_stack = needed;
        }
        Ok(())
    }

    /// Follows `errorlimit` in lparser.c.
    pub fn limit(&self, what: &'static str, limit: usize) -> Fault {
        Fault::Limit {
            what,
            limit,
            line_defined: self.line_defined,
        }
    }

    pub fn free_register(&mut self, register: u8) {
        if register >= self.nvarstack() {
            self.freereg -= 1;
        }
    }

    pub fn free_registers(&mut self, first: u8, second: u8) {
        if first > second {
            self.free_register(first);
            self.free_register(second);
        } else {
            self.free_register(second);
            self.free_register(first);
        }
    }

    pub fn free_expression(&mut self, expression: &ExpDesc) {
        if let ExpKind::NonReloc(register) = expression.kind {
            self.free_register(register);
        }
    }

    pub fn free_expressions(&mut self, first: &ExpDesc, second: &ExpDesc) {
        let first_reg = match first.kind {
            ExpKind::NonReloc(register) => Some(register),
            _ => None,
        };
        let second_reg = match second.kind {
            ExpKind::NonReloc(register) => Some(register),
            _ => None,
        };
        match (first_reg, second_reg) {
            (Some(a), Some(b)) if a > b => {
                self.free_register(a);
                self.free_register(b);
            }
            (Some(a), Some(b)) => {
                self.free_register(b);
                self.free_register(a);
            }
            (Some(a), None) | (None, Some(a)) => self.free_register(a),
            (None, None) => {}
        }
    }

    fn push_constant(&mut self, value: Value) -> u32 {
        let index = self.constants.len() as u32;
        self.constants.push(value);
        index
    }

    pub fn add_constant(&mut self, key: ConstKey, value: Value) -> u32 {
        if let Some(index) = self.constant_index.get(&key) {
            return *index;
        }
        let index = self.push_constant(value);
        self.constant_index.insert(key, index);
        index
    }

    pub fn string_constant(&mut self, string: Ref<LuaString>) -> u32 {
        self.add_constant(ConstKey::Str(string), Value::String(string))
    }

    pub fn integer_constant(&mut self, value: i64) -> u32 {
        self.add_constant(ConstKey::Integer(value), Value::Integer(value))
    }

    /// Follows `luaK_numberK` in lcode.c.
    pub fn float_constant(&mut self, value: f64) -> u32 {
        if value == 0.0 {
            return self.add_constant(ConstKey::Float(0.0_f64.to_bits()), Value::Float(value));
        }
        let key = value * (1.0 + f64::EPSILON);
        if crate::number::float_to_integer(key, crate::number::Rounding::Exact).is_some() {
            return self.push_constant(Value::Float(value));
        }
        let cache_key = ConstKey::Float(key.to_bits());
        let Some(index) = self.constant_index.get(&cache_key).copied() else {
            let index = self.push_constant(Value::Float(value));
            self.constant_index.insert(cache_key, index);
            return index;
        };
        if matches!(self.constants[index as usize], Value::Float(known) if known == value) {
            return index;
        }
        self.push_constant(Value::Float(value))
    }

    pub fn boolean_constant(&mut self, value: bool) -> u32 {
        self.add_constant(ConstKey::Boolean(value), Value::Boolean(value))
    }

    pub fn nil_constant(&mut self) -> u32 {
        self.add_constant(ConstKey::Nil, Value::Nil)
    }
}
