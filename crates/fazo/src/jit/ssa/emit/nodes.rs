//! Ported from the Cranelift lowering in `lower.rs` so both tiers compute the same values. Operands and outputs are placed before any branch, so a spill runs on every path.

use std::mem::offset_of;

use super::{
    asm::{Alu, Asm, Cond, Division, FAlu, FCond, Label, Width},
    codegen::{BOOL_TAG, CLOSURE_TAG, Def, FLOAT_TAG, Gen, INT_TAG, NIL_TAG, PAYLOAD, STRING_TAG, SafepointStub, Src, TABLE_TAG, VALUE_SIZE},
    regs::PReg,
};
use crate::jit::{
    JitCtx,
    layout::upvalue,
    ssa::{
        ir::{BinOp, BuiltinFn, Cc, Closure, Const, NumOp, Op, PureFn, Rep, StateId, Type, UnOp, Val, Virtual},
        runtime,
    },
};

const NODE_SIZE: i32 = crate::table::offsets::NODE_SIZE as i32;
const NODE_KEY: i32 = crate::table::offsets::NODE_KEY as i32;
const NODE_VALUE: i32 = crate::table::offsets::NODE_VALUE as i32;

type Saves = Vec<(PReg, u32, bool)>;

pub(super) fn int_cond(cc: Cc) -> Cond {
    match cc {
        Cc::Eq => Cond::Eq,
        Cc::Lt => Cond::Lt,
        Cc::Le => Cond::Le,
    }
}

pub(super) fn float_cond(cc: Cc) -> FCond {
    match cc {
        Cc::Eq => FCond::Eq,
        Cc::Lt => FCond::Lt,
        Cc::Le => FCond::Le,
    }
}

fn slot_offset(slot: u32) -> i32 {
    slot as i32 * VALUE_SIZE
}

/// Follows `luaV_flttointns` with `F2Ieq`.
fn exact_integer(float: f64) -> Option<i64> {
    crate::number::float_to_integer(float, crate::number::Rounding::Exact)
}

fn small(value: Option<Const>) -> Option<i32> {
    match value? {
        Const::Int(integer) => i32::try_from(integer).ok(),
        _ => None,
    }
}

impl<A: Asm> Gen<'_, A> {
    pub(super) fn node(&mut self, value: Val) {
        if self.live.fused[value.0 as usize] {
            return;
        }
        let func = self.func;
        match &func.inst(value).op {
            Op::Param | Op::Const(_) => {}
            Op::LoadReg { slot } => {
                let base = self.temp();
                let def = self.define(value);
                self.registers(base);
                self.load_def(def, base, slot_offset(*slot));
            }
            Op::StoreReg { slot, value: stored } => {
                let pair = self.pair(*stored);
                let base = self.temp();
                self.registers(base);
                self.store_pair(pair, base, slot_offset(*slot));
            }
            Op::LoadUpval { closure, index } => {
                let owner = self.closure_reg(*closure);
                let address = self.temp();
                let def = self.define(value);
                self.upvalue_address(address, owner, *index);
                self.load_def(def, address, 0);
            }
            Op::StoreUpval { closure, index, value: stored } => self.store_upvalue(*closure, *index, *stored),
            Op::Unary(op, operand) => self.unary(value, *op, *operand),
            Op::Binary(op, left, right) => self.binary(value, *op, *left, *right),
            Op::Pure(function, args) => self.pure(value, *function, args),
            Op::EqConst(operand, constant) => self.equal_constant(value, *operand, *constant),
            Op::Select { cond, yes, no } => self.select(value, *cond, *yes, *no),
            Op::Arith { op, left, right, state } => self.arith(value, *op, (*left, *right), *state),
            Op::Compare { cc, left, right, state } => self.compare(value, *cc, (*left, *right), *state),
            Op::Len { value: operand, state } => self.len(value, *operand, *state),
            Op::ShapeIs { table, shape } => self.shape_is(value, *table, *shape),
            Op::LoadSlot { table, slot } => {
                let payload = self.reg(*table);
                let address = self.temp();
                let def = self.define(value);
                self.table_address(address, payload);
                self.asm.load(Width::Q, address, address, self.env.layout.slots);
                self.load_def(def, address, slot_offset(*slot));
            }
            Op::LoadProto { closure } => {
                let payload = self.reg(*closure);
                let out = self.out(value);
                self.closure_address(out, payload);
                self.asm.load(Width::Q, out, out, offset_of!(crate::object::Closure, proto) as i32);
            }
            Op::ArrayBase { table } => {
                let payload = self.reg(*table);
                let Def::Gp(out) = self.define(value) else {
                    self.failed = true;
                    return;
                };
                self.table_address(out, payload);
                self.asm.load(Width::Q, out, out, self.env.layout.array);
            }
            Op::LoadArray { table, base, index, state } => self.load_array(value, (*table, *base), *index, *state),
            Op::Requests => {
                let Def::Gp(out) = self.define(value) else {
                    self.failed = true;
                    return;
                };
                self.ctx_load(out, offset_of!(JitCtx, requests));
            }
            Op::GuardType { value: input, ty, state } => self.guard_type(value, *input, *ty, *state),
            Op::GuardShape { table, shape, state } => {
                let payload = self.reg(*table);
                let address = self.temp();
                let exit = self.exit(*state);
                self.table_address(address, payload);
                self.check_shape(address, *shape, exit);
            }
            Op::GuardHasMeta { table, state } => {
                let payload = self.reg(*table);
                let address = self.temp();
                let out = self.out(value);
                let exit = self.exit(*state);
                let layout = self.env.layout;
                self.table_address(address, payload);
                self.asm.load(Width::W, A::SCRATCH, address, layout.metatable_tag);
                self.asm.cmp_imm64(A::SCRATCH, i64::from(layout.metatable_none));
                self.asm.branch(Cond::Eq, exit);
                self.asm.load(Width::Q, out, address, layout.metatable_index);
            }
            Op::GuardNoMeta { table, bit, state } => self.guard_no_meta(*table, *bit, *state),
            Op::GuardEq { value: input, expected, state } => {
                let word = self.reg(*input);
                let exit = self.exit(*state);
                self.asm.cmp_imm64(word, *expected as i64);
                self.asm.branch(Cond::Ne, exit);
            }
            Op::GuardFalse { cond, state } => {
                let flag = self.reg(*cond);
                let exit = self.exit(*state);
                self.asm.cmp_imm(flag, 0);
                self.asm.branch(Cond::Ne, exit);
            }
            Op::GuardNotNil { value: input, state } => {
                let tag = self.tag(*input);
                let exit = self.exit(*state);
                match tag {
                    Src::R(tag) => {
                        self.asm.cmp_byte(tag, NIL_TAG);
                        self.asm.branch(Cond::Eq, exit);
                    }
                    Src::I(0) => self.asm.jump(exit),
                    _ => {}
                }
                if !self.same_as_input(value, *input) {
                    let def = self.define(value);
                    self.convert(def, *input);
                }
            }
            Op::GuardRange { table, low, high, state } => self.guard_range(*table, (*low, *high), *state),
            Op::Index { object, key, state } => self.index(value, *object, *key, *state),
            Op::NewIndex { object, key, value: stored, state } => self.new_index(*object, *key, *stored, *state),
            Op::StoreSlot { table, slot, value: stored, state } => {
                let disp = slot_offset(*slot);
                let layout = self.env.layout;
                self.store_checked(*table, *stored, *state, |g, address| {
                    g.asm.load(Width::Q, address, address, layout.slots);
                    disp
                });
            }
            Op::AddSlot { table, key, shape, next, slot, value: stored, state } => self.add_slot(*table, *key, (*shape, *next, *slot), *stored, *state),
            Op::GuardNode { table, node, nodes, key, state } => self.guard_node(*table, (*node, *nodes), *key, *state),
            Op::LoadNode { table, node } => {
                let payload = self.reg(*table);
                let address = self.temp();
                let def = self.define(value);
                self.table_address(address, payload);
                self.asm.load(Width::Q, address, address, self.env.layout.nodes);
                self.load_def(def, address, *node as i32 * NODE_SIZE + NODE_VALUE);
            }
            Op::StoreNode { table, node, value: stored, state } => {
                let disp = *node as i32 * NODE_SIZE + NODE_VALUE;
                let layout = self.env.layout;
                self.store_checked(*table, *stored, *state, |g, address| {
                    g.asm.load(Width::Q, address, address, layout.nodes);
                    disp
                });
            }
            Op::Helper(helper) => self.helper(value, helper.pc, helper.kind, helper.args),
            Op::Builtin { function, args, state } => self.builtin(value, *function, args, *state),
            Op::NewTable { hash_log, top, shape, fields, state, after } => self.new_table(value, (*hash_log, *top), (*shape, fields), (*state, *after)),
            Op::Materialize { object } => self.materialize(value, object),
            Op::GuardAbsent { table, bit, state } => {
                let payload = self.reg(*table);
                let address = self.temp();
                let exit = self.exit(*state);
                self.table_address(address, payload);
                self.asm.load(Width::B, A::SCRATCH, address, self.env.layout.absent);
                self.asm.test_imm(A::SCRATCH, i32::from(*bit));
                self.asm.branch(Cond::Eq, exit);
            }
            Op::CallLua { pc, a, b, c } => self.call_lua(value, *pc as usize, (*a, *b, *c)),
            Op::Safepoint { pc, weight, requests, state } => self.safepoint(*pc, *weight, *requests, *state),
        }
    }

    pub(super) fn table_address(&mut self, dst: u8, payload: u8) {
        self.asm.zext32(dst, payload);
        self.asm.alu_imm(Alu::Mul, dst, dst, self.env.layout.size as i32);
        self.asm.add_mem(dst, A::CTX, offset_of!(JitCtx, tables) as i32);
    }

    pub(super) fn closure_address(&mut self, dst: u8, payload: u8) {
        self.asm.zext32(dst, payload);
        self.asm.alu_imm(Alu::Mul, dst, dst, std::mem::size_of::<crate::object::Closure>() as i32);
        self.asm.add_mem(dst, A::CTX, offset_of!(JitCtx, closures) as i32);
    }

    fn check_shape(&mut self, address: u8, shape: u64, fail: Label) {
        let layout = self.env.layout;
        self.asm.load(Width::B, A::SCRATCH, address, layout.shaped);
        self.asm.cmp_imm(A::SCRATCH, 0);
        self.asm.branch(Cond::Eq, fail);
        self.asm.load(Width::Q, A::SCRATCH, address, layout.shape);
        self.asm.cmp_imm64(A::SCRATCH, shape as i64);
        self.asm.branch(Cond::Ne, fail);
    }

    fn closure_reg(&mut self, closure: Closure) -> Option<u8> {
        match closure {
            Closure::Root => None,
            Closure::Value(value) => Some(self.reg(value)),
        }
    }

    fn upvalue_cell(&mut self, dst: u8, owner: Option<u8>, index: u8) {
        match owner {
            None => self.ctx_load(dst, offset_of!(JitCtx, upvalues)),
            Some(closure) => {
                self.closure_address(dst, closure);
                self.asm.load(Width::Q, dst, dst, self.env.frames.closure_upvalues);
            }
        }
        self.asm.load(Width::Q, dst, dst, upvalue::cell(index) as i32);
    }

    fn upvalue_address(&mut self, dst: u8, owner: Option<u8>, index: u8) {
        self.upvalue_cell(dst, owner, index);
        self.asm.load(Width::Q, dst, dst, upvalue::VALUE);
    }

    /// A closed upvalue needs the collector's barrier for a collectable value, so that store runs the helper.
    fn store_upvalue(&mut self, closure: Closure, index: u8, stored: Val) {
        let pair = self.pair(stored);
        let owner = self.closure_reg(closure);
        let address = self.temp();
        if !self.may_collect(stored) {
            self.upvalue_address(address, owner, index);
            self.store_pair(pair, address, 0);
            return;
        }
        let saves = self.saves(&[]);
        let slow = self.asm.new_label();
        let done = self.asm.new_label();
        self.upvalue_cell(address, owner, index);
        self.asm.load(Width::B, A::SCRATCH, address, upvalue::TAG);
        self.asm.cmp_imm(A::SCRATCH, upvalue::OPEN as i32);
        self.asm.branch(Cond::Ne, slow);
        self.asm.load(Width::Q, address, address, upvalue::VALUE);
        self.store_pair(pair, address, 0);
        self.asm.jump(done);
        self.asm.bind(slow);
        let owner = match owner {
            Some(closure) => closure,
            None => {
                self.ctx_load(address, offset_of!(JitCtx, closure));
                address
            }
        };
        self.call_value(runtime::KIND_SET_UPVALUE, &[pair, (Src::I(i64::from(CLOSURE_TAG)), Src::R(owner))], u32::from(index), &saves);
        self.asm.bind(done);
    }

    fn unary(&mut self, value: Val, op: UnOp, operand: Val) {
        match op {
            UnOp::Box | UnOp::Unbox => {
                let def = self.define(value);
                self.convert(def, operand);
            }
            UnOp::Truthy => self.truthy(value, operand),
            UnOp::Not => {
                let flag = self.reg(operand);
                let out = self.out(value);
                self.asm.alu_imm(Alu::Xor, out, flag, 1);
            }
            UnOp::INeg | UnOp::IBnot => {
                let x = self.reg(operand);
                let out = self.out(value);
                if op == UnOp::INeg {
                    self.asm.neg(out, x);
                } else {
                    self.asm.not(out, x);
                }
            }
            UnOp::FNeg => {
                let x = self.freg(operand);
                let out = self.out(value);
                self.asm.fneg(out, x);
            }
            UnOp::FSqrt => {
                let x = self.freg(operand);
                let out = self.out(value);
                self.asm.fsqrt(out, x);
            }
            UnOp::IToF => {
                let x = self.reg(operand);
                let out = self.out(value);
                self.asm.int_to_float(out, x);
            }
            UnOp::NumToF => self.number_to_float(value, operand),
        }
    }

    fn number_to_float(&mut self, value: Val, operand: Val) {
        let (tag, word) = self.pair_regs(operand);
        let out = self.out(value);
        let integer = self.asm.new_label();
        let done = self.asm.new_label();
        self.asm.cmp_byte(tag, INT_TAG);
        self.asm.branch(Cond::Eq, integer);
        self.asm.bits_to_float(out, word);
        self.asm.jump(done);
        self.asm.bind(integer);
        self.asm.int_to_float(out, word);
        self.asm.bind(done);
    }

    fn truthy(&mut self, value: Val, operand: Val) {
        match self.func.rep(operand) {
            Rep::Bool => {
                let flag = self.reg(operand);
                let out = self.out(value);
                self.asm.mov(out, flag);
            }
            Rep::Boxed if self.is_const(operand).is_none() => {
                let (tag, payload) = self.pair_regs(operand);
                let out = self.out(value);
                let zero = self.asm.new_label();
                let done = self.asm.new_label();
                self.asm.mov_imm(out, 1);
                self.asm.cmp_byte(tag, NIL_TAG);
                self.asm.branch(Cond::Eq, zero);
                self.asm.cmp_byte(tag, BOOL_TAG);
                self.asm.branch(Cond::Ne, done);
                self.asm.test_imm(payload, 0xff);
                self.asm.branch(Cond::Ne, done);
                self.asm.bind(zero);
                self.asm.mov_imm(out, 0);
                self.asm.bind(done);
            }
            _ => {
                let truth = match self.is_const(operand) {
                    Some(Const::Nil | Const::Bool(false)) => 0,
                    _ => 1,
                };
                let out = self.out(value);
                self.asm.mov_imm(out, truth);
            }
        }
    }

    fn binary(&mut self, value: Val, op: BinOp, left: Val, right: Val) {
        match op {
            BinOp::BoolEq => {
                let (x, y) = (self.reg(left), self.reg(right));
                let out = self.out(value);
                self.asm.cmp(x, y);
                self.asm.set(Cond::Eq, out);
            }
            BinOp::ICmp(cc) => {
                let x = self.reg(left);
                let right_imm = small(self.is_const(right));
                let y = if right_imm.is_none() { Some(self.reg(right)) } else { None };
                let out = self.out(value);
                match (y, right_imm) {
                    (Some(y), _) => self.asm.cmp(x, y),
                    (None, Some(immediate)) => self.asm.cmp_imm(x, immediate),
                    (None, None) => unreachable!("a register or an immediate"),
                }
                self.asm.set(int_cond(cc), out);
            }
            BinOp::FCmp(cc) => {
                let (x, y) = (self.freg(left), self.freg(right));
                let out = self.out(value);
                self.asm.fset(float_cond(cc), out, x, y);
            }
            BinOp::IAdd | BinOp::ISub | BinOp::IMul | BinOp::IBand | BinOp::IBor | BinOp::IBxor => self.int_binary(value, op, left, right),
            BinOp::IMod | BinOp::IDiv => self.floor_division(value, op, left, right),
            BinOp::FAdd | BinOp::FSub | BinOp::FMul | BinOp::FDiv | BinOp::FIDiv => {
                let x = self.freg(left);
                let y = match self.word(right) {
                    Src::I(bits) if bits != 0 => Err((bits as u64, self.ftemp())),
                    _ => Ok(self.freg(right)),
                };
                let out = self.out_over(value, left);
                let alu = match op {
                    BinOp::FAdd => FAlu::Add,
                    BinOp::FSub => FAlu::Sub,
                    BinOp::FMul => FAlu::Mul,
                    _ => FAlu::Div,
                };
                match y {
                    Ok(y) => self.asm.falu(alu, out, x, y),
                    Err(constant) => self.asm.falu_const(alu, out, x, constant),
                }
                if op == BinOp::FIDiv {
                    self.asm.ffloor(out, out);
                }
            }
        }
    }

    fn int_binary(&mut self, value: Val, op: BinOp, left: Val, right: Val) {
        let alu = match op {
            BinOp::IAdd => Alu::Add,
            BinOp::ISub => Alu::Sub,
            BinOp::IMul => Alu::Mul,
            BinOp::IBand => Alu::And,
            BinOp::IBor => Alu::Or,
            _ => Alu::Xor,
        };
        let commutes = alu != Alu::Sub;
        let (left, right) = if commutes && self.is_const(left).is_some() && self.is_const(right).is_none() { (right, left) } else { (left, right) };
        let x = self.reg(left);
        if let Some(immediate) = small(self.is_const(right)) {
            let out = self.out_over(value, left);
            self.asm.alu_imm(alu, out, x, immediate);
            return;
        }
        let y = self.reg(right);
        let out = self.out_over(value, left);
        self.asm.alu(alu, out, x, y);
    }

    /// Follows lvm.c's `luaV_mod` and `luaV_idiv`; a hoisted copy may run before the zero-divisor guard, so a zero divisor
    /// answers as one would instead of trapping, as `lower.rs` does.
    fn floor_division(&mut self, value: Val, op: BinOp, left: Val, right: Val) {
        let division = if op == BinOp::IMod { Division::Mod } else { Division::IDiv };
        let ordinary = self.is_const(right).is_some_and(|divisor| (divisor.payload() as i64) > 0);
        let x = self.reg(left);
        let y = self.reg(right);
        if ordinary {
            let out = self.out(value);
            self.asm.floor_div(division, out, x, y);
            return;
        }
        let probe = self.temp();
        let out = self.out(value);
        let special = self.asm.new_label();
        let done = self.asm.new_label();
        self.asm.alu_imm(Alu::Add, probe, y, 1);
        self.asm.cmp_imm(probe, 1);
        self.asm.branch(Cond::BelowEq, special);
        self.asm.floor_div(division, out, x, y);
        self.asm.jump(done);
        self.asm.bind(special);
        if op == BinOp::IMod {
            self.asm.mov_imm(out, 0);
        } else {
            self.asm.mov(out, x);
            self.asm.cmp_imm(y, -1);
            self.asm.branch(Cond::Ne, done);
            self.asm.neg(out, x);
        }
        self.asm.bind(done);
    }

    fn pure(&mut self, value: Val, function: PureFn, args: &[Val]) {
        let call_args: Vec<Src> = match function {
            PureFn::FMod | PureFn::FPow => {
                let kind = if function == PureFn::FMod { runtime::PURE_MOD } else { runtime::PURE_POW };
                vec![Src::I(i64::from(kind)), Src::F(self.freg(args[0])), Src::F(self.freg(args[1]))]
            }
            PureFn::Shl | PureFn::Shr => {
                let kind = if function == PureFn::Shl { runtime::PURE_SHL } else { runtime::PURE_SHR };
                vec![Src::I(i64::from(kind)), self.word(args[0]), self.word(args[1])]
            }
            PureFn::LoopCount => args.iter().map(|&arg| self.word(arg)).collect(),
            PureFn::LtIF | PureFn::LeIF | PureFn::LtFI | PureFn::LeFI | PureFn::EqFI => {
                let code = match function {
                    PureFn::LtIF => runtime::MIXED_LT_IF,
                    PureFn::LeIF => runtime::MIXED_LE_IF,
                    PureFn::LtFI => runtime::MIXED_LT_FI,
                    PureFn::LeFI => runtime::MIXED_LE_FI,
                    _ => runtime::MIXED_EQ_FI,
                };
                let (integer, float) = match function {
                    PureFn::LtIF | PureFn::LeIF => (args[0], args[1]),
                    _ => (args[1], args[0]),
                };
                vec![Src::I(i64::from(code)), self.word(integer), Src::F(self.freg(float))]
            }
        };
        let address = match function {
            PureFn::FMod | PureFn::FPow => self.addresses.pure_float,
            PureFn::Shl | PureFn::Shr => self.addresses.pure_int,
            PureFn::LoopCount => self.addresses.loop_count,
            _ => self.addresses.mixed,
        };
        let out = self.out(value);
        let saves = self.saves(&[]);
        self.save(&saves);
        self.call_c(address, &call_args);
        if self.func.rep(value) == Rep::Float {
            self.asm.fmov(out, A::float_result());
        } else {
            self.asm.mov(out, A::result());
        }
        self.restore(&saves);
    }

    /// Follows `luaV_equalobj`: a float equals an integer only when it is exactly that integer.
    fn equal_constant(&mut self, value: Val, operand: Val, constant: Const) {
        let rep = self.func.rep(operand);
        if self.is_const(operand).is_some() {
            let same = self.is_const(operand).is_some_and(|known| known.same(constant));
            let out = self.out(value);
            self.asm.mov_imm(out, i64::from(same));
            return;
        }
        match (rep, constant) {
            (Rep::Int, Const::Int(k)) => {
                let x = self.reg(operand);
                let out = self.out(value);
                self.asm.cmp_imm64(x, k);
                self.asm.set(Cond::Eq, out);
            }
            (Rep::Int, Const::Float(f)) => {
                let x = self.reg(operand);
                let out = self.out(value);
                match exact_integer(f) {
                    Some(k) => {
                        self.asm.cmp_imm64(x, k);
                        self.asm.set(Cond::Eq, out);
                    }
                    None => self.asm.mov_imm(out, 0),
                }
            }
            (Rep::Float, Const::Float(f)) => {
                let x = self.freg(operand);
                let k = self.ftemp();
                let out = self.out(value);
                self.float_constant(k, f);
                self.asm.fset(FCond::Eq, out, x, k);
            }
            (Rep::Float, Const::Int(k)) => {
                let x = self.freg(operand);
                let temp = self.ftemp();
                let out = self.out(value);
                let saves = self.saves(&[]);
                self.float_equals_integer(out, (x, temp), k, &saves);
            }
            (Rep::Bool, Const::Bool(flag)) => {
                let x = self.reg(operand);
                let out = self.out(value);
                self.asm.cmp_imm(x, i32::from(flag));
                self.asm.set(Cond::Eq, out);
            }
            (Rep::Payload, Const::Gc { tag, payload }) if self.func.ty(operand).tag() == Some(tag) => {
                let x = self.reg(operand);
                let out = self.out(value);
                self.asm.cmp_imm64(x, payload as i64);
                self.asm.set(Cond::Eq, out);
            }
            (Rep::Boxed, _) => self.boxed_equal_constant(value, operand, constant),
            _ => {
                let out = self.out(value);
                self.asm.mov_imm(out, 0);
            }
        }
    }

    fn float_constant(&mut self, dst: u8, float: f64) {
        self.asm.fconst(dst, float.to_bits());
    }

    fn float_equals_integer(&mut self, out: u8, (float, temp): (u8, u8), integer: i64, saves: &Saves) {
        if integer.unsigned_abs() <= 1 << 53 {
            self.float_constant(temp, integer as f64);
            self.asm.fset(FCond::Eq, out, float, temp);
            return;
        }
        self.save(saves);
        self.call_c(self.addresses.mixed, &[Src::I(i64::from(runtime::MIXED_EQ_FI)), Src::I(integer), Src::F(float)]);
        self.asm.mov(out, A::result());
        self.restore(saves);
    }

    fn boxed_equal_constant(&mut self, value: Val, operand: Val, constant: Const) {
        let (tag, payload) = self.pair_regs(operand);
        let temp = self.ftemp();
        let float = self.ftemp();
        let out = self.out(value);
        let saves = self.saves(&[]);
        let done = self.asm.new_label();
        let other = self.asm.new_label();
        self.asm.mov_imm(out, 0);
        self.asm.cmp_byte(tag, constant.tag());
        self.asm.branch(Cond::Ne, other);
        match constant {
            Const::Nil => self.asm.mov_imm(out, 1),
            Const::Bool(flag) => {
                self.asm.alu_imm(Alu::And, A::SCRATCH, payload, 0xff);
                self.asm.cmp_imm(A::SCRATCH, i32::from(flag));
                self.asm.set(Cond::Eq, out);
            }
            Const::Int(_) | Const::Gc { .. } => {
                self.asm.cmp_imm64(payload, constant.payload() as i64);
                self.asm.set(Cond::Eq, out);
            }
            Const::Float(f) => {
                self.asm.bits_to_float(float, payload);
                self.float_constant(temp, f);
                self.asm.fset(FCond::Eq, out, float, temp);
            }
        }
        self.asm.jump(done);
        self.asm.bind(other);
        match constant {
            Const::Int(k) => {
                self.asm.cmp_byte(tag, FLOAT_TAG);
                self.asm.branch(Cond::Ne, done);
                self.asm.bits_to_float(float, payload);
                self.float_equals_integer(out, (float, temp), k, &saves);
            }
            Const::Float(f) => {
                if let Some(k) = exact_integer(f) {
                    self.asm.cmp_byte(tag, INT_TAG);
                    self.asm.branch(Cond::Ne, done);
                    self.asm.cmp_imm64(payload, k);
                    self.asm.set(Cond::Eq, out);
                }
            }
            _ => {}
        }
        self.asm.bind(done);
    }

    fn select(&mut self, value: Val, cond: Val, yes: Val, no: Val) {
        let flag = self.reg(cond);
        match self.func.rep(value) {
            Rep::Boxed => {
                let (yes_tag, yes_payload) = self.pair_regs(yes);
                let (no_tag, no_payload) = self.pair_regs(no);
                let Def::Two(tag, payload) = self.define(value) else {
                    return;
                };
                self.asm.cmp_imm(flag, 0);
                self.asm.select(Cond::Ne, tag, yes_tag, no_tag);
                self.asm.select(Cond::Ne, payload, yes_payload, no_payload);
            }
            Rep::Float => {
                let (x, y) = (self.freg(yes), self.freg(no));
                let out = self.out(value);
                let skip = self.asm.new_label();
                self.asm.fmov(out, y);
                self.asm.cmp_imm(flag, 0);
                self.asm.branch(Cond::Eq, skip);
                self.asm.fmov(out, x);
                self.asm.bind(skip);
            }
            Rep::None => {}
            _ => {
                let (x, y) = (self.reg(yes), self.reg(no));
                let out = self.out(value);
                self.asm.cmp_imm(flag, 0);
                self.asm.select(Cond::Ne, out, x, y);
            }
        }
    }

    fn check_tag(&mut self, tag: Src, wanted: u8, otherwise: Label) {
        match tag {
            Src::R(reg) => {
                self.asm.cmp_byte(reg, wanted);
                self.asm.branch(Cond::Ne, otherwise);
            }
            Src::I(known) if known == i64::from(wanted) => {}
            _ => self.asm.jump(otherwise),
        }
    }

    fn arith(&mut self, value: Val, op: NumOp, (left, right): (Val, Val), state: StateId) {
        let unary = matches!(op, NumOp::Neg | NumOp::BNot);
        let int_op = matches!(op, NumOp::Add | NumOp::Sub | NumOp::Mul | NumOp::BAnd | NumOp::BOr | NumOp::BXor | NumOp::Neg | NumOp::BNot);
        let float_op = matches!(op, NumOp::Add | NumOp::Sub | NumOp::Mul | NumOp::Div | NumOp::Neg);
        let x = self.pair(left);
        let y = if unary { x } else { self.pair(right) };
        let x_payload = self.src_reg(x.1);
        let y_payload = self.src_reg(y.1);
        let floats = if float_op { Some((self.ftemp(), self.ftemp())) } else { None };
        let Def::Two(tag, payload) = self.define(value) else {
            self.failed = true;
            return;
        };
        let saves = self.saves(&[]);
        let join = self.asm.new_label();
        let slow = self.asm.new_label();
        if int_op {
            let next = self.asm.new_label();
            self.check_tag(x.0, INT_TAG, next);
            self.check_tag(y.0, INT_TAG, next);
            match op {
                NumOp::Neg => self.asm.neg(payload, x_payload),
                NumOp::BNot => self.asm.not(payload, x_payload),
                _ => {
                    let alu = match op {
                        NumOp::Add => Alu::Add,
                        NumOp::Sub => Alu::Sub,
                        NumOp::Mul => Alu::Mul,
                        NumOp::BAnd => Alu::And,
                        NumOp::BOr => Alu::Or,
                        _ => Alu::Xor,
                    };
                    self.asm.alu(alu, payload, x_payload, y_payload);
                }
            }
            self.asm.mov_imm(tag, i64::from(INT_TAG));
            self.asm.jump(join);
            self.asm.bind(next);
        }
        if let Some((a, b)) = floats {
            self.check_tag(x.0, FLOAT_TAG, slow);
            self.check_tag(y.0, FLOAT_TAG, slow);
            self.asm.bits_to_float(a, x_payload);
            self.asm.bits_to_float(b, y_payload);
            match op {
                NumOp::Add => self.asm.falu(FAlu::Add, a, a, b),
                NumOp::Sub => self.asm.falu(FAlu::Sub, a, a, b),
                NumOp::Mul => self.asm.falu(FAlu::Mul, a, a, b),
                NumOp::Div => self.asm.falu(FAlu::Div, a, a, b),
                _ => self.asm.fneg(a, a),
            }
            self.asm.float_to_bits(payload, a);
            self.asm.mov_imm(tag, i64::from(FLOAT_TAG));
            self.asm.jump(join);
        }
        self.asm.bind(slow);
        self.call_value(runtime::KIND_ARITH, &[x, y], runtime::num_op_code(op), &saves);
        let exit = self.exit(state);
        self.asm.cmp_imm(A::SCRATCH, 0);
        self.asm.branch(Cond::Eq, exit);
        self.scratch_load(0, tag, payload);
        self.asm.bind(join);
    }

    fn compare(&mut self, value: Val, cc: Cc, (left, right): (Val, Val), state: StateId) {
        let x = self.pair(left);
        let y = self.pair(right);
        let x_payload = self.src_reg(x.1);
        let y_payload = self.src_reg(y.1);
        let (a, b) = (self.ftemp(), self.ftemp());
        let out = self.out(value);
        let saves = self.saves(&[]);
        let join = self.asm.new_label();
        let float_test = self.asm.new_label();
        let slow = self.asm.new_label();
        self.check_tag(x.0, INT_TAG, float_test);
        self.check_tag(y.0, INT_TAG, float_test);
        self.asm.cmp(x_payload, y_payload);
        self.asm.set(int_cond(cc), out);
        self.asm.jump(join);
        self.asm.bind(float_test);
        self.check_tag(x.0, FLOAT_TAG, slow);
        self.check_tag(y.0, FLOAT_TAG, slow);
        self.asm.bits_to_float(a, x_payload);
        self.asm.bits_to_float(b, y_payload);
        self.asm.fset(float_cond(cc), out, a, b);
        self.asm.jump(join);
        self.asm.bind(slow);
        self.call_value(runtime::KIND_COMPARE, &[x, y], cc as u32, &saves);
        let exit = self.exit(state);
        self.asm.cmp_imm(A::SCRATCH, 0);
        self.asm.branch(Cond::Eq, exit);
        self.asm.cmp_imm(A::SCRATCH, 2);
        self.asm.set(Cond::Eq, out);
        self.asm.bind(join);
    }

    fn len(&mut self, value: Val, operand: Val, state: StateId) {
        let pair = self.pair(operand);
        let out = self.out(value);
        let saves = self.saves(&[]);
        self.call_value(runtime::KIND_LEN, &[pair], 0, &saves);
        let exit = self.exit(state);
        self.asm.cmp_imm(A::SCRATCH, 0);
        self.asm.branch(Cond::Eq, exit);
        self.asm.load(Width::Q, out, A::SP, self.scratch_at + PAYLOAD);
    }

    fn shape_is(&mut self, value: Val, table: Val, shape: u64) {
        let payload = self.reg(table);
        let address = self.temp();
        let out = self.out(value);
        let done = self.asm.new_label();
        self.table_address(address, payload);
        self.asm.mov_imm(out, 0);
        self.check_shape(address, shape, done);
        self.asm.mov_imm(out, 1);
        self.asm.bind(done);
    }

    fn load_array(&mut self, value: Val, (table, base): (Val, Val), index: Val, state: StateId) {
        let payload = self.reg(table);
        let elements = self.reg(base);
        let position = self.reg(index);
        let key = self.pair(index);
        let address = self.temp();
        let Def::Two(tag, word) = self.define(value) else {
            self.failed = true;
            return;
        };
        let saves = self.saves(&[]);
        let join = self.asm.new_label();
        self.asm.shl_imm(address, position, 4);
        self.asm.alu(Alu::Add, address, address, elements);
        self.asm.load(Width::Q, tag, address, -VALUE_SIZE);
        self.asm.load(Width::Q, word, address, -VALUE_SIZE + PAYLOAD);
        self.asm.cmp_byte(tag, NIL_TAG);
        self.asm.branch(Cond::Ne, join);
        self.call_value(runtime::KIND_INDEX, &[(Src::I(i64::from(TABLE_TAG)), Src::R(payload)), key], 0, &saves);
        let exit = self.exit(state);
        self.asm.cmp_imm(A::SCRATCH, 0);
        self.asm.branch(Cond::Eq, exit);
        self.scratch_load(0, tag, word);
        self.asm.bind(join);
    }

    fn guard_type(&mut self, value: Val, input: Val, ty: Type, state: StateId) {
        let tag = self.tag(input);
        let exit = self.exit(state);
        match tag {
            Src::R(reg) => match ty.tag() {
                Some(single) => {
                    self.asm.cmp_byte(reg, single);
                    self.asm.branch(Cond::Ne, exit);
                }
                None => {
                    let ok = self.asm.new_label();
                    let tags = (0..11u8).filter(|&tag| !ty.intersect(Type::of_tag(tag)).is_none());
                    let count = tags.clone().count();
                    if count == 0 {
                        self.asm.jump(exit);
                    }
                    for (index, wanted) in tags.enumerate() {
                        self.asm.cmp_byte(reg, wanted);
                        if index + 1 == count {
                            self.asm.branch(Cond::Ne, exit);
                        } else {
                            self.asm.branch(Cond::Eq, ok);
                        }
                    }
                    self.asm.bind(ok);
                }
            },
            Src::I(known) if ty.intersect(Type::of_tag(known as u8)).is_none() => self.asm.jump(exit),
            _ => {}
        }
        if !self.same_as_input(value, input) {
            let def = self.define(value);
            self.convert(def, input);
        }
    }

    fn guard_no_meta(&mut self, table: Val, bit: u8, state: StateId) {
        let payload = self.reg(table);
        let address = self.temp();
        let exit = self.exit(state);
        let layout = self.env.layout;
        let absent = self.asm.new_label();
        self.table_address(address, payload);
        self.asm.load(Width::W, A::SCRATCH, address, layout.metatable_tag);
        self.asm.cmp_imm64(A::SCRATCH, i64::from(layout.metatable_none));
        self.asm.branch(Cond::Eq, absent);
        self.asm.load(Width::W, A::SCRATCH, address, layout.metatable_index);
        self.table_address(address, A::SCRATCH);
        self.asm.load(Width::B, A::SCRATCH, address, layout.absent);
        self.asm.test_imm(A::SCRATCH, i32::from(bit));
        self.asm.branch(Cond::Eq, exit);
        self.asm.bind(absent);
    }

    fn guard_range(&mut self, table: Val, (low, high): (Val, Val), state: StateId) {
        let payload = self.reg(table);
        let low = self.reg(low);
        let high = self.reg(high);
        let length = self.temp();
        let exit = self.exit(state);
        self.table_address(length, payload);
        self.asm.load(Width::Q, length, length, self.env.layout.array_len);
        for index in [low, high] {
            self.asm.alu_imm(Alu::Sub, A::SCRATCH, index, 1);
            self.asm.cmp(A::SCRATCH, length);
            self.asm.branch(Cond::AboveEq, exit);
        }
    }

    /// An integer key's payload, with its tag when it may be something else.
    fn integer_key(&mut self, key: Val) -> Option<(u8, Option<u8>)> {
        match self.func.rep(key) {
            Rep::Int => Some((self.reg(key), None)),
            Rep::Boxed if !self.func.ty(key).intersect(Type::INT).is_none() && self.is_const(key).is_none() => {
                let (tag, payload) = self.pair_regs(key);
                Some((payload, Some(tag)))
            }
            _ => None,
        }
    }

    fn index(&mut self, value: Val, object: Val, key: Val, state: StateId) {
        let payload = self.reg(object);
        let pair = self.pair(key);
        let integer = self.integer_key(key);
        let address = self.temp();
        let Def::Two(tag, word) = self.define(value) else {
            self.failed = true;
            return;
        };
        let string = self.string_key(key);
        let scratch = (string.is_some() || integer.is_some()).then(|| [self.temp(), self.temp(), self.temp()]);
        let probe = string.and(scratch).map(|[entry, key, node]| (entry, key, node));
        let hashed = integer.and(scratch);
        let saves = self.saves(&[]);
        let join = self.asm.new_label();
        let slow = self.asm.new_label();
        let strings = if string.is_some() { self.asm.new_label() } else { slow };
        if let (Some((index, is_int)), Some(temps)) = (integer, hashed) {
            let ints = self.asm.new_label();
            if let Some(key_tag) = is_int {
                self.asm.cmp_byte(key_tag, INT_TAG);
                self.asm.branch(Cond::Ne, strings);
            }
            self.inside(address, payload, index, tag, ints);
            self.asm.load(Width::Q, address, address, self.env.layout.array);
            self.asm.shl_imm(tag, tag, 4);
            self.asm.alu(Alu::Add, address, address, tag);
            self.asm.load(Width::Q, tag, address, 0);
            self.asm.load(Width::Q, word, address, PAYLOAD);
            self.asm.cmp_byte(tag, NIL_TAG);
            self.asm.branch(Cond::Ne, join);
            self.asm.jump(slow);
            self.asm.bind(ints);
            self.int_node((address, index), temps, slow);
            self.asm.load(Width::Q, tag, temps[2], NODE_VALUE);
            self.asm.load(Width::Q, word, temps[2], NODE_VALUE + PAYLOAD);
            self.asm.cmp_byte(tag, NIL_TAG);
            self.asm.branch(Cond::Ne, join);
            if string.is_some() {
                self.asm.jump(slow);
            }
        }
        if let (Some((key_tag, key_word)), Some((entry, key, node))) = (string, probe) {
            self.asm.bind(strings);
            self.string_node((payload, key_tag, key_word), [address, entry, tag, key, node], slow);
            self.asm.load(Width::Q, tag, node, NODE_VALUE);
            self.asm.load(Width::Q, word, node, NODE_VALUE + PAYLOAD);
            self.asm.cmp_byte(tag, NIL_TAG);
            self.asm.branch(Cond::Ne, join);
        }
        self.asm.bind(slow);
        self.call_value(runtime::KIND_INDEX, &[(Src::I(i64::from(TABLE_TAG)), Src::R(payload)), pair], 0, &saves);
        let exit = self.exit(state);
        self.asm.cmp_imm(A::SCRATCH, 0);
        self.asm.branch(Cond::Eq, exit);
        self.scratch_load(0, tag, word);
        self.asm.bind(join);
    }

    fn string_key(&mut self, key: Val) -> Option<(u8, u8)> {
        let may_be_string = !self.func.ty(key).intersect(Type::of_tag(STRING_TAG)).is_none();
        if self.func.rep(key) != Rep::Boxed || !may_be_string || self.is_const(key).is_some() {
            return None;
        }
        Some(self.pair_regs(key))
    }

    /// Follows LuaJIT's `asm_href` (lj_asm.c).
    /// The key is copied first, since the result registers may share the key's.
    fn string_node(&mut self, (table, key_tag, key_word): (u8, u8, u8), [address, entry, index, key, node]: [u8; 5], slow: Label) {
        let layout = self.env.layout;
        let walk = self.asm.new_label();
        let follow = self.asm.new_label();
        let found = self.asm.new_label();
        self.asm.cmp_byte(key_tag, STRING_TAG);
        self.asm.branch(Cond::Ne, slow);
        self.asm.mov(key, key_word);
        self.table_address(address, table);
        self.asm.load(Width::B, A::SCRATCH, address, layout.shaped);
        self.asm.cmp_imm(A::SCRATCH, 0);
        self.asm.branch(Cond::Ne, slow);
        self.asm.zext32(entry, key);
        self.asm.alu_imm(Alu::Mul, entry, entry, std::mem::size_of::<crate::heap::StringKey>() as i32);
        self.ctx_load(A::SCRATCH, offset_of!(JitCtx, vm));
        self.asm.add_mem(entry, A::SCRATCH, layout.string_keys);
        self.asm.load(Width::B, A::SCRATCH, entry, offset_of!(crate::heap::StringKey, interned) as i32);
        self.asm.cmp_imm(A::SCRATCH, 0);
        self.asm.branch(Cond::Eq, slow);
        self.asm.load(Width::Q, index, address, layout.nodes_len);
        self.asm.alu_imm(Alu::Sub, index, index, 1);
        self.asm.load(Width::Q, A::SCRATCH, entry, offset_of!(crate::heap::StringKey, hash) as i32);
        self.asm.alu(Alu::And, index, index, A::SCRATCH);
        self.asm.load(Width::Q, entry, address, layout.nodes);
        self.asm.bind(walk);
        self.asm.alu_imm(Alu::Mul, node, index, crate::table::offsets::NODE_SIZE as i32);
        self.asm.alu(Alu::Add, node, node, entry);
        self.asm.load(Width::B, A::SCRATCH, node, NODE_KEY);
        self.asm.cmp_imm(A::SCRATCH, i32::from(STRING_TAG));
        self.asm.branch(Cond::Ne, follow);
        self.asm.load(Width::Q, A::SCRATCH, node, NODE_KEY + PAYLOAD);
        self.asm.cmp(A::SCRATCH, key);
        self.asm.branch(Cond::Ne, follow);
        self.asm.load(Width::B, A::SCRATCH, node, crate::table::offsets::NODE_DEAD as i32);
        self.asm.cmp_imm(A::SCRATCH, 0);
        self.asm.branch(Cond::Eq, found);
        self.asm.bind(follow);
        self.asm.load(Width::W, index, node, crate::table::offsets::NODE_NEXT as i32);
        self.asm.cmp_imm(index, 0);
        self.asm.branch(Cond::Eq, slow);
        self.asm.alu_imm(Alu::Sub, index, index, 1);
        self.asm.jump(walk);
        self.asm.bind(found);
    }

    /// Follows `Table::find_int`: PUC's `hashint` main position (ltable.c), then the collision chain. Leaves the node in
    /// the last temporary; `address` holds the table.
    fn int_node(&mut self, (address, key): (u8, u8), [entry, index, node]: [u8; 3], slow: Label) {
        let layout = self.env.layout;
        let walk = self.asm.new_label();
        let follow = self.asm.new_label();
        let found = self.asm.new_label();
        self.asm.load(Width::Q, index, address, layout.nodes_len);
        self.asm.cmp_imm(index, 0);
        self.asm.branch(Cond::Eq, slow);
        self.asm.alu_imm(Alu::Sub, index, index, 1);
        self.asm.alu_imm(Alu::Or, index, index, 1);
        self.asm.unsigned_rem(index, key, index);
        self.asm.load(Width::Q, entry, address, layout.nodes);
        self.asm.bind(walk);
        self.asm.alu_imm(Alu::Mul, node, index, NODE_SIZE);
        self.asm.alu(Alu::Add, node, node, entry);
        self.asm.load(Width::B, A::SCRATCH, node, NODE_KEY);
        self.asm.cmp_imm(A::SCRATCH, i32::from(INT_TAG));
        self.asm.branch(Cond::Ne, follow);
        self.asm.load(Width::Q, A::SCRATCH, node, NODE_KEY + PAYLOAD);
        self.asm.cmp(A::SCRATCH, key);
        self.asm.branch(Cond::Ne, follow);
        self.asm.load(Width::B, A::SCRATCH, node, crate::table::offsets::NODE_DEAD as i32);
        self.asm.cmp_imm(A::SCRATCH, 0);
        self.asm.branch(Cond::Eq, found);
        self.asm.bind(follow);
        self.asm.load(Width::W, index, node, crate::table::offsets::NODE_NEXT as i32);
        self.asm.cmp_imm(index, 0);
        self.asm.branch(Cond::Eq, slow);
        self.asm.alu_imm(Alu::Sub, index, index, 1);
        self.asm.jump(walk);
        self.asm.bind(found);
    }

    /// Jumps to `slow` past the array part.
    fn inside(&mut self, address: u8, table: u8, index: u8, position: u8, slow: Label) {
        self.table_address(address, table);
        self.asm.load(Width::Q, A::SCRATCH, address, self.env.layout.array_len);
        self.asm.alu_imm(Alu::Sub, position, index, 1);
        self.asm.cmp(position, A::SCRATCH);
        self.asm.branch(Cond::AboveEq, slow);
    }

    /// The raw store runs only when no `__newindex` can fire (`quick::raw_set` decides).
    fn new_index(&mut self, object: Val, key: Val, stored: Val, state: StateId) {
        let value = self.pair(stored);
        let payload = self.reg(object);
        let key_pair = self.pair(key);
        let integer = self.integer_key(key);
        let string = self.string_key(key);
        let address = self.temp();
        let element = self.temp();
        let scratch = (string.is_some() || integer.is_some()).then(|| [self.temp(), self.temp()]);
        let probe = string.and(scratch).map(|[entry, key]| (entry, key, self.temp()));
        let hashed = integer.and(scratch).map(|[entry, index]| [entry, index, element]);
        let saves = self.saves(&[]);
        let done = self.asm.new_label();
        let slow = self.asm.new_label();
        let strings = if string.is_some() { self.asm.new_label() } else { slow };
        if let (Some((index, is_int)), Some(temps)) = (integer, hashed) {
            let layout = self.env.layout;
            let store = self.asm.new_label();
            let ints = self.asm.new_label();
            if let Some(key_tag) = is_int {
                self.asm.cmp_byte(key_tag, INT_TAG);
                self.asm.branch(Cond::Ne, strings);
            }
            self.inside(address, payload, index, element, ints);
            self.asm.shl_imm(element, element, 4);
            self.asm.load(Width::Q, A::SCRATCH, address, layout.array);
            self.asm.alu(Alu::Add, element, element, A::SCRATCH);
            self.asm.load(Width::B, A::SCRATCH, element, 0);
            self.asm.cmp_imm(A::SCRATCH, 0);
            self.asm.branch(Cond::Ne, store);
            self.asm.load(Width::W, A::SCRATCH, address, layout.metatable_tag);
            self.asm.cmp_imm64(A::SCRATCH, i64::from(layout.metatable_none));
            self.asm.branch(Cond::Ne, slow);
            self.asm.bind(store);
            self.store_pair(value, element, 0);
            if self.may_collect(stored) {
                self.barrier(payload, value, address, &saves);
            }
            self.asm.jump(done);
            self.asm.bind(ints);
            self.int_node((address, index), temps, slow);
            self.asm.load(Width::B, A::SCRATCH, element, NODE_VALUE);
            self.asm.cmp_imm(A::SCRATCH, i32::from(NIL_TAG));
            self.asm.branch(Cond::Eq, slow);
            self.store_pair(value, element, NODE_VALUE);
            if self.may_collect(stored) {
                self.barrier(payload, value, address, &saves);
            }
            self.asm.jump(done);
        }
        if let (Some((key_tag, key_word)), Some((entry, key, node))) = (string, probe) {
            self.asm.bind(strings);
            self.string_node((payload, key_tag, key_word), [address, entry, element, key, node], slow);
            self.asm.load(Width::B, A::SCRATCH, node, NODE_VALUE);
            self.asm.cmp_imm(A::SCRATCH, i32::from(NIL_TAG));
            self.asm.branch(Cond::Eq, slow);
            self.store_pair(value, node, NODE_VALUE);
            if self.may_collect(stored) {
                self.barrier(payload, value, address, &saves);
            }
            self.asm.jump(done);
        }
        self.asm.bind(slow);
        self.call_value(runtime::KIND_NEW_INDEX, &[(Src::I(i64::from(TABLE_TAG)), Src::R(payload)), key_pair, value], 0, &saves);
        let exit = self.exit(state);
        self.asm.cmp_imm(A::SCRATCH, 0);
        self.asm.branch(Cond::Eq, exit);
        self.asm.bind(done);
    }

    /// Only a collectable value stored into a black table needs the collector's barrier.
    fn barrier(&mut self, table: u8, (tag, payload): (Src, Src), temp: u8, saves: &Saves) {
        let done = self.asm.new_label();
        match tag {
            Src::R(reg) => {
                self.asm.alu_imm(Alu::And, A::SCRATCH, reg, 0xff);
                self.asm.alu_imm(Alu::Sub, A::SCRATCH, A::SCRATCH, i32::from(STRING_TAG));
                self.asm.cmp_imm(A::SCRATCH, i32::from(super::codegen::LIGHT_USERDATA_TAG - STRING_TAG));
                self.asm.branch(Cond::AboveEq, done);
            }
            Src::I(known) if !(i64::from(STRING_TAG)..i64::from(super::codegen::LIGHT_USERDATA_TAG)).contains(&known) => return,
            _ => {}
        }
        self.asm.zext32(temp, table);
        self.ctx_load(A::SCRATCH, offset_of!(JitCtx, table_marks));
        self.asm.alu(Alu::Add, temp, temp, A::SCRATCH);
        self.asm.load(Width::B, A::SCRATCH, temp, 0);
        self.asm.test_imm(A::SCRATCH, i32::from(crate::gc::BLACK));
        self.asm.branch(Cond::Eq, done);
        self.call_value(runtime::KIND_BARRIER, &[(Src::I(i64::from(TABLE_TAG)), Src::R(table)), (tag, payload)], 0, saves);
        self.asm.bind(done);
    }

    /// Follows `Vm::transition_store_hit`; anything else takes the helper's generic store.
    fn add_slot(&mut self, table: Val, key: Val, (shape, next, slot): (u64, u64, u32), stored: Val, state: StateId) {
        let layout = self.env.layout;
        let value = self.pair(stored);
        let payload = self.reg(table);
        let key_pair = self.pair(key);
        let address = self.temp();
        let length = self.temp();
        let mark = self.temp();
        let saves = self.saves(&[]);
        let slow = self.asm.new_label();
        let done = self.asm.new_label();
        self.table_address(address, payload);
        self.check_shape(address, shape, slow);
        self.asm.load(Width::W, A::SCRATCH, address, layout.metatable_tag);
        self.asm.cmp_imm64(A::SCRATCH, i64::from(layout.metatable_none));
        self.asm.branch(Cond::Ne, slow);
        self.asm.load(Width::Q, length, address, layout.slots_len);
        self.asm.load(Width::Q, A::SCRATCH, address, layout.slots_capacity);
        self.asm.cmp(length, A::SCRATCH);
        self.asm.branch(Cond::AboveEq, slow);
        self.asm.zext32(mark, payload);
        self.ctx_load(A::SCRATCH, offset_of!(JitCtx, table_marks));
        self.asm.alu(Alu::Add, mark, mark, A::SCRATCH);
        self.asm.load(Width::B, A::SCRATCH, mark, 0);
        self.asm.test_imm(A::SCRATCH, i32::from(crate::gc::BLACK));
        self.asm.branch(Cond::Ne, slow);
        if !self.func.ty(stored).intersect(Type::NIL).is_none() {
            self.check_not_nil(value.0, slow);
        }
        self.asm.load(Width::Q, mark, address, layout.slots);
        self.store_pair(value, mark, slot as i32 * VALUE_SIZE);
        self.asm.alu_imm(Alu::Add, length, length, 1);
        self.asm.store(Width::Q, length, address, layout.slots_len);
        self.asm.mov_imm(A::SCRATCH, next as i64);
        self.asm.store(Width::Q, A::SCRATCH, address, layout.shape);
        self.asm.store_imm(Width::B, address, layout.absent, 0);
        self.asm.jump(done);
        self.asm.bind(slow);
        self.call_value(runtime::KIND_NEW_INDEX, &[(Src::I(i64::from(TABLE_TAG)), Src::R(payload)), key_pair, value], 0, &saves);
        let exit = self.exit(state);
        self.asm.cmp_imm(A::SCRATCH, 0);
        self.asm.branch(Cond::Eq, exit);
        self.asm.bind(done);
    }

    fn check_not_nil(&mut self, tag: Src, nil: Label) {
        match tag {
            Src::R(reg) => {
                self.asm.cmp_byte(reg, NIL_TAG);
                self.asm.branch(Cond::Eq, nil);
            }
            Src::I(0) => self.asm.jump(nil),
            _ => {}
        }
    }

    /// `address` returns the displacement from the table's base.
    fn store_checked(&mut self, table: Val, stored: Val, state: StateId, address: impl FnOnce(&mut Self, u8) -> i32) {
        let value = self.pair(stored);
        let payload = self.reg(table);
        let base = self.temp();
        let temp = self.temp();
        let collects = self.may_collect(stored);
        let saves = if collects { self.saves(&[]) } else { Vec::new() };
        let exit = self.exit(state);
        self.table_address(base, payload);
        let disp = address(self, base);
        self.asm.cmp_mem_byte(base, disp, 0);
        self.asm.branch(Cond::Eq, exit);
        self.store_pair(value, base, disp);
        if collects {
            self.barrier(payload, value, temp, &saves);
        }
    }

    fn guard_node(&mut self, table: Val, (node, nodes): (u32, u32), key: u64, state: StateId) {
        let layout = self.env.layout;
        let payload = self.reg(table);
        let address = self.temp();
        let exit = self.exit(state);
        let key_at = node as i32 * NODE_SIZE + NODE_KEY;
        self.table_address(address, payload);
        self.asm.load(Width::B, A::SCRATCH, address, layout.shaped);
        self.asm.cmp_imm(A::SCRATCH, 0);
        self.asm.branch(Cond::Ne, exit);
        self.asm.load(Width::Q, A::SCRATCH, address, layout.nodes_len);
        self.asm.cmp_imm64(A::SCRATCH, i64::from(nodes));
        self.asm.branch(Cond::Ne, exit);
        self.asm.load(Width::Q, address, address, layout.nodes);
        self.asm.load(Width::B, A::SCRATCH, address, key_at);
        self.asm.cmp_imm(A::SCRATCH, i32::from(STRING_TAG));
        self.asm.branch(Cond::Ne, exit);
        self.asm.load(Width::Q, A::SCRATCH, address, key_at + PAYLOAD);
        self.asm.cmp_imm64(A::SCRATCH, key as i64);
        self.asm.branch(Cond::Ne, exit);
    }

    fn helper(&mut self, value: Val, pc: u32, kind: crate::jit::ssa::ir::HelperKind, args: [u32; 3]) {
        let out = self.out(value);
        let saves = self.saves(&[]);
        self.save(&saves);
        let address = match kind {
            crate::jit::ssa::ir::HelperKind::Step => {
                self.place_arg(0, Src::R(A::CTX));
                self.place_arg(1, Src::I(i64::from(pc)));
                self.addresses.step
            }
            crate::jit::ssa::ir::HelperKind::Fast(which) => {
                self.fast_args(pc, args);
                self.addresses.fast[which]
            }
        };
        self.asm.call(address);
        self.asm.mov(out, A::result());
        self.restore(&saves);
    }

    pub(super) fn fast_args(&mut self, pc: u32, args: [u32; 3]) {
        for (index, &arg) in args.iter().enumerate().rev() {
            self.place_arg(index + 2, Src::I(i64::from(arg as i32)));
        }
        self.place_arg(1, Src::I(i64::from(pc)));
        self.place_arg(0, Src::R(A::CTX));
    }

    /// `rawget` of a non-table and a refused `setmetatable` answer zero and resume the call in the interpreter.
    fn builtin(&mut self, value: Val, function: BuiltinFn, args: &[Val], state: StateId) {
        let pairs: Vec<(Src, Src)> = args.iter().map(|&arg| self.pair(arg)).collect();
        let Def::Two(tag, payload) = self.define(value) else {
            self.failed = true;
            return;
        };
        let saves = self.saves(&[]);
        self.call_value(runtime::KIND_BUILTIN, &pairs, runtime::builtin_code(function), &saves);
        if matches!(function, BuiltinFn::RawGet | BuiltinFn::SetMetatable) {
            let exit = self.exit(state);
            self.asm.cmp_imm(A::SCRATCH, 0);
            self.asm.branch(Cond::Eq, exit);
        }
        self.scratch_load(0, tag, payload);
    }

    /// Every register the frame state holds is written back first, so the collector step sees it.
    fn new_table(&mut self, value: Val, (hash_log, top): (u8, u32), (shape, fields): (Option<u64>, &[Val]), (state, after): (StateId, StateId)) {
        let pairs: Vec<(Src, Src)> = fields.iter().map(|&field| self.pair(field)).collect();
        let ran = self.temp();
        let base = self.temp();
        let out = self.out(value);
        let saves = self.saves(fields);
        let failed = self.exit(state);
        let data = self.func.state(state);
        let pc = data.frames.first().map_or(data.pc, |frame| frame.caller_pc);
        self.save(&saves);
        self.store_state_for_step(state);
        self.call_c(self.addresses.new_table, &[Src::R(A::CTX), Src::I(i64::from(pc)), Src::I(i64::from(hash_log)), Src::I(i64::from(top))]);
        self.asm.mov(out, A::result());
        self.restore(&saves);
        self.asm.cmp_imm(out, 0);
        self.asm.branch(Cond::Lt, failed);
        self.asm.mov_imm(ran, runtime::RAN_CODE);
        self.asm.alu(Alu::And, ran, ran, out);
        self.asm.alu(Alu::Xor, out, out, ran);
        if let Some(shape) = shape {
            let layout = self.env.layout;
            self.table_address(base, out);
            self.asm.load(Width::Q, A::SCRATCH, base, layout.slots);
            for (index, &pair) in pairs.iter().enumerate() {
                self.store_pair(pair, A::SCRATCH, index as i32 * VALUE_SIZE);
            }
            self.asm.store_imm(Width::Q, base, layout.slots_len, fields.len() as i64);
            self.asm.mov_imm(A::SCRATCH, shape as i64);
            self.asm.store(Width::Q, A::SCRATCH, base, layout.shape);
        }
        let quiet = self.exit(after);
        self.asm.cmp_imm(ran, 0);
        self.asm.branch(Cond::Ne, quiet);
    }

    /// Stored only when the allocation debt is paid and a collection step runs.
    fn store_state_for_step(&mut self, state: StateId) {
        let quiet = self.asm.new_label();
        self.ctx_load(A::SCRATCH, offset_of!(JitCtx, vm));
        self.asm.load(Width::Q, A::SCRATCH, A::SCRATCH, offset_of!(crate::Vm, heap.gc.debt) as i32);
        self.asm.cmp_imm(A::SCRATCH, 0);
        self.asm.branch(Cond::Gt, quiet);
        self.store_state(state);
        self.asm.bind(quiet);
    }

    fn materialize(&mut self, value: Val, object: &Virtual) {
        let meta = match object.meta {
            Some(meta) => self.pair(meta),
            None => (Src::I(0), Src::I(0)),
        };
        let mut pairs = vec![meta];
        pairs.extend(object.fields.iter().map(|&field| self.pair(field)));
        let original = object.original.map(|(flag, original)| (self.reg(flag), self.reg(original)));
        let out = self.out(value);
        let saves = self.saves(&[]);
        for (index, &pair) in pairs.iter().enumerate() {
            self.store_pair(pair, A::SP, self.scratch_at + index as i32 * VALUE_SIZE);
        }
        self.save(&saves);
        let count = pairs.len() as i64;
        for (index, arg) in [(4, Src::I(count)), (2, Src::I(i64::from(object.hash_log))), (1, Src::I(object.shape as i64))] {
            self.place_arg(index, arg);
        }
        self.arg_address(3, self.scratch_at);
        self.place_arg(0, Src::R(A::CTX));
        self.asm.call(self.addresses.materialize);
        self.asm.mov(out, A::result());
        self.restore(&saves);
        if let Some((flag, original)) = original {
            self.asm.cmp_imm(flag, 0);
            self.asm.select(Cond::Ne, out, out, original);
        }
    }

    /// The branch of a comparison only its branch reads: compared here, so no flag lives in a register.
    pub(super) fn compare_branch(&mut self, cond: Val, negate: bool, target: Label) {
        let Op::Binary(op, left, right) = self.func.op(cond).clone() else {
            unreachable!("a fused value is a comparison");
        };
        match op {
            BinOp::ICmp(cc) => {
                let x = self.reg(left);
                match small(self.is_const(right)) {
                    Some(immediate) => self.asm.cmp_imm(x, immediate),
                    None => {
                        let y = self.reg(right);
                        self.asm.cmp(x, y);
                    }
                }
                let cond = int_cond(cc);
                self.asm.branch(if negate { cond.negate() } else { cond }, target);
            }
            BinOp::FCmp(cc) => {
                let (x, y) = (self.freg(left), self.freg(right));
                if negate {
                    self.asm.fbranch_not(float_cond(cc), x, y, target);
                } else {
                    self.asm.fbranch(float_cond(cc), x, y, target);
                }
            }
            _ => unreachable!("only comparisons fuse"),
        }
    }

    /// Charges the loop's length to the budget as Baseline does at a back edge; running out enters the tier check with state written back.
    fn safepoint(&mut self, pc: u32, weight: u32, requests: Val, state: StateId) {
        let Some(&(_, osr)) = self.func.osr.iter().find(|&&(at, _)| at == pc) else {
            return;
        };
        let charge = self.env.tier_budget;
        let cell = charge.map(|_| self.temp());
        let target = self.label(osr);
        let slow = self.asm.new_label();
        let stored = (!self.func.state(state).slots.is_empty()).then(|| self.exit_index(state));
        self.safepoints.push(SafepointStub { label: slow, state: stored, pc, charge: charge.is_some(), target });
        if self.env.stress_exits {
            self.asm.jump(slow);
            return;
        }
        let flags = self.reg(requests);
        self.asm.cmp_mem_byte(flags, 0, 0);
        self.asm.branch(Cond::Ne, slow);
        if let (Some(address), Some(cell)) = (charge, cell) {
            self.spend(cell, address as i64, weight);
            self.asm.branch(Cond::Le, slow);
        }
    }

    pub(super) fn safepoint_tail(&mut self, stub: SafepointStub) {
        self.asm.bind(stub.label);
        if let Some(index) = stub.state {
            self.store_state_at(index);
        }
        self.poll(self.addresses.safe_point, stub.pc);
        if stub.charge {
            self.poll(self.addresses.tier_check, stub.pc);
        }
        self.asm.jump(stub.target);
    }

    /// Leaves the flags of the old budget against `weight`.
    pub(super) fn spend(&mut self, cell: u8, address: i64, weight: u32) {
        self.asm.mov_imm(cell, address);
        self.asm.sub_mem(cell, 0, weight as i32);
    }

    /// Leaves the code unless the call answers `pc`.
    fn poll(&mut self, address: u64, pc: u32) {
        self.place_arg(1, Src::I(i64::from(pc)));
        self.place_arg(0, Src::R(A::CTX));
        self.asm.call(address);
        self.asm.mov(A::SCRATCH, A::result());
        self.asm.cmp_imm(A::SCRATCH, pc as i32);
        self.asm.branch(Cond::Ne, self.exit_with);
    }

}
