//! The Baseline templates: one machine-code sequence per bytecode case, written once against `Masm`.
//! Hole slots are named per template; `A`, `B` and `C` are register byte offsets or raw helper operands.

use std::mem::offset_of;

use super::masm::{Alu, Cond, Division, FAlu, FCond, FReg, Imm, Masm, Reg, Target, Width, ext, slot};
use crate::jit::{FrameLayout, JitCtx, TableLayout, layout::{lua_frame, upvalue}};
use crate::vm::exec_layout;

pub(crate) const NIL: i32 = 0;
pub(crate) const BOOL: i32 = 1;
pub(crate) const INT: i32 = 2;
pub(crate) const FLOAT: i32 = 3;
pub(crate) const STRING: i32 = 4;
pub(crate) const TABLE: i32 = 5;
pub(crate) const CLOSURE: i32 = 6;
pub(crate) const LIGHT_USERDATA: i32 = 10;
const PAYLOAD: i32 = 8;
const STRING_KEY_HASH: i64 = offset_of!(crate::heap::StringKey, hash) as i64;
const STRING_KEY_INTERNED: i64 = offset_of!(crate::heap::StringKey, interned) as i64;
const NODE_KEY: i64 = crate::table::offsets::NODE_KEY as i64;
const NODE_VALUE: i64 = crate::table::offsets::NODE_VALUE as i64;
const NODE_NEXT: i64 = crate::table::offsets::NODE_NEXT as i64;
const NODE_DEAD: i64 = crate::table::offsets::NODE_DEAD as i64;
const _: () = assert!(crate::table::offsets::NODE_SIZE == 48 && std::mem::size_of::<crate::heap::StringKey>() == 16);

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct HelperAddresses {
    pub(crate) step: u64,
    pub(crate) safe_point: u64,
    pub(crate) misc: u64,
    pub(crate) tier_check: u64,
    pub(crate) fast: [u64; 21],
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Env {
    pub(crate) table: Option<TableLayout>,
    pub(crate) frames: Option<FrameLayout>,
    pub(crate) helpers: HelperAddresses,
    pub(crate) depth_limit: u32,
}

pub(crate) const CALL_AREA: usize = std::mem::size_of::<JitCtx>() + 32;
const SPILL_COUNT: usize = std::mem::size_of::<JitCtx>();
const SPILL_ENTRY: usize = SPILL_COUNT + 8;
const SPILL_EXEC: usize = SPILL_COUNT + 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum HelperId {
    Step,
    SafePoint,
    Misc,
    TierCheck,
    Fast(u8),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum ArithKind {
    Add,
    Sub,
    Mul,
    BAnd,
    BOr,
    BXor,
    Div,
    Mod,
    IDiv,
}

impl ArithKind {
    fn has_float(self) -> bool {
        matches!(self, Self::Add | Self::Sub | Self::Mul | Self::Div)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Rhs {
    Reg,
    KInt,
    KFloat,
    Imm,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum CmpKind {
    Eq,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum TagConstant {
    Nil,
    Bool,
    String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Key {
    Move,
    LoadConstant,
    LoadInt,
    LoadFloat,
    LoadBool(bool),
    LoadNil,
    Jump,
    BackEdge { charge: bool },
    Charge,
    Helper(HelperId),
    Arith { op: ArithKind, rhs: Rhs },
    Unm,
    BNot,
    Not,
    Truthy,
    MoveJump,
    Compare { cmp: CmpKind, rhs: Rhs, int: bool, float: bool, default_false: bool },
    EqTag(TagConstant),
    ForLoop,
    TForLoop,
    ArrayGet { constant_key: bool },
    ArraySet { constant_key: bool, constant_value: bool },
    RegisterTable,
    UpvalueTable,
    FieldGet { absent: bool, method: bool },
    MethodGet,
    FieldSet { constant: bool },
    FieldAdd { constant: bool },
    HashGet,
    HashSet { constant_value: bool },
    MethodStart,
    MethodHop,
    MethodFinish,
    GetUpval,
    SetUpval,
    ReturnCheck,
    ReturnCopy,
    ReturnFill,
    ReturnWide,
    ReturnDone,
    StepStub,
    SharedStep,
    CallCheck,
    CallEnter { upvalues: bool },
    CallAnyCheck,
    CallAnyEnter,
    TailCheck,
    ClearAt,
    TailEnter { upvalues: bool },
}

pub(crate) const KEY_COUNT: usize = 283;
const HELPER_SLOTS: u8 = 19;

impl Key {
    pub(crate) fn index(self) -> usize {
        match self {
            Key::Move => 0,
            Key::LoadConstant => 1,
            Key::LoadInt => 2,
            Key::LoadFloat => 3,
            Key::LoadBool(flag) => 4 + usize::from(flag),
            Key::LoadNil => 6,
            Key::Jump => 7,
            Key::BackEdge { charge } => 8 + usize::from(charge),
            Key::Charge => 10,
            Key::Unm => 11,
            Key::BNot => 12,
            Key::Not => 13,
            Key::Truthy => 14,
            Key::MoveJump => 15,
            Key::ForLoop => 16,
            Key::TForLoop => 17,
            Key::RegisterTable => 18,
            Key::UpvalueTable => 19,
            Key::MethodGet => 20,
            Key::GetUpval => 21,
            Key::SetUpval => 22,
            Key::ReturnCheck => 23,
            Key::ReturnCopy => 24,
            Key::ReturnDone => 25,
            Key::StepStub => 26,
            Key::SharedStep => 27,
            Key::CallCheck => 262,
            Key::CallEnter { upvalues } => 263 + usize::from(upvalues),
            Key::TailCheck => 265,
            Key::ClearAt => 266,
            Key::TailEnter { upvalues } => 267 + usize::from(upvalues),
            Key::FieldAdd { constant } => 269 + usize::from(constant),
            Key::HashGet => 271,
            Key::HashSet { constant_value } => 272 + usize::from(constant_value),
            Key::MethodStart => 274,
            Key::MethodHop => 275,
            Key::MethodFinish => 276,
            Key::ReturnFill => 279,
            Key::CallAnyCheck => 280,
            Key::CallAnyEnter => 281,
            Key::ReturnWide => 282,
            Key::EqTag(tag) => 28 + tag as usize,
            Key::ArrayGet { constant_key } => 31 + usize::from(constant_key),
            Key::ArraySet { constant_key, constant_value } => 33 + 2 * usize::from(constant_key) + usize::from(constant_value),
            Key::FieldGet { absent, method } => 37 + 2 * usize::from(absent) + usize::from(method),
            Key::FieldSet { constant } => 41 + usize::from(constant),
            Key::Helper(HelperId::Fast(which)) if which >= HELPER_SLOTS => 277 + usize::from(which - HELPER_SLOTS),
            Key::Helper(id) => {
                43 + match id {
                    HelperId::Step => 0,
                    HelperId::SafePoint => 1,
                    HelperId::Misc => 2,
                    HelperId::TierCheck => 3,
                    HelperId::Fast(which) => 4 + usize::from(which),
                }
            }
            Key::Arith { op, rhs } => 66 + op as usize * 4 + rhs as usize,
            Key::Compare { cmp, rhs, int, float, default_false } => {
                let shape = ((cmp as usize * 4 + rhs as usize) * 2 + usize::from(int)) * 2 + usize::from(float);
                102 + shape * 2 + usize::from(default_false)
            }
        }
    }
}

fn reg(slot: u8) -> Imm {
    Imm::hole(slot)
}

fn payload(slot: u8) -> Imm {
    Imm::hole_plus(slot, PAYLOAD)
}

fn fixed(value: i64) -> Imm {
    Imm::Fixed(value)
}

fn context(offset: usize) -> Imm {
    Imm::Fixed(offset as i64)
}

pub(crate) fn build(key: Key, m: &mut dyn Masm, env: &Env) {
    match key {
        Key::Move => copy(m, Reg::Base, reg(slot::B), Reg::Base, reg(slot::A)),
        Key::LoadConstant => copy(m, Reg::Konst, reg(slot::K), Reg::Base, reg(slot::A)),
        Key::LoadInt => load_number(m, INT),
        Key::LoadFloat => load_number(m, FLOAT),
        Key::LoadBool(value) => {
            m.store_imm(Width::U8, Reg::Base, reg(slot::A), BOOL);
            m.store_imm(Width::U8, Reg::Base, payload(slot::A), i32::from(value));
        }
        Key::LoadNil => m.store_imm(Width::U8, Reg::Base, reg(slot::A), NIL),
        Key::Jump => m.jump(Target::Ext(ext::TARGET)),
        Key::BackEdge { charge } => back_edge(m, charge),
        Key::Charge => charge_budget(m),
        Key::Helper(id) => helper(m, env, id),
        Key::Arith { op, rhs } => arith(m, op, rhs),
        Key::Unm => unm(m),
        Key::BNot => bnot(m),
        Key::Not => not(m),
        Key::Truthy => truthy(m, slot::A),
        Key::MoveJump => {
            copy(m, Reg::Base, reg(slot::B), Reg::Base, reg(slot::A));
            m.jump(Target::Ext(ext::TARGET));
        }
        Key::Compare { cmp, rhs, int, float, default_false } => compare(m, cmp, rhs, (int, float), default_false),
        Key::EqTag(tag) => eq_tag(m, tag),
        Key::ForLoop => for_loop(m),
        Key::TForLoop => {
            m.load(Width::U8, Reg::T0, Reg::Base, Imm::hole_plus(slot::A, 48));
            m.cmp_imm(Reg::T0, NIL);
            m.branch(Cond::Eq, Target::Ext(ext::NEXT));
        }
        Key::ArrayGet { constant_key } => array_get(m, env, constant_key),
        Key::ArraySet { constant_key, constant_value } => array_set(m, env, constant_key, constant_value),
        Key::RegisterTable => register_table(m, env),
        Key::UpvalueTable => upvalue_table(m, env),
        Key::FieldGet { absent, method } => field_get(m, env, absent, method),
        Key::MethodGet => method_get(m, env),
        Key::MethodStart => method_start(m, env),
        Key::MethodHop => method_hop(m, env),
        Key::MethodFinish => method_finish(m, env),
        Key::FieldSet { constant } => field_set(m, env, constant),
        Key::FieldAdd { constant } => field_add(m, env, constant),
        Key::HashGet => hash_get(m, env),
        Key::HashSet { constant_value } => hash_set(m, env, constant_value),
        Key::GetUpval => get_upval(m),
        Key::SetUpval => set_upval(m),
        Key::ReturnCheck => return_check(m),
        Key::ReturnWide => return_wide(m, env),
        Key::ReturnCopy => copy(m, Reg::Base, reg(slot::A), Reg::T2, reg(slot::B)),
        Key::ReturnFill => return_fill(m, env),
        Key::ReturnDone => {
            m.mov_imm(Reg::Ret, fixed(crate::jit::EXIT_RETURNED));
            m.jump(Target::Ext(ext::EXIT));
        }
        Key::StepStub => {
            m.arg_imm(1, reg(slot::PC));
            m.jump(Target::Ext(ext::TARGET));
        }
        Key::SharedStep => {
            m.arg_reg(0, Reg::Ctx);
            m.call(env.helpers.step);
            m.load(Width::U64, Reg::Base, Reg::Ctx, context(offset_of!(JitCtx, registers)));
            m.jump(Target::Ext(ext::DISPATCH));
        }
        Key::CallCheck => call_check(m, env),
        Key::CallEnter { upvalues } => call_enter(m, env, Callee::Known { upvalues }),
        Key::CallAnyCheck => call_any_check(m, env),
        Key::CallAnyEnter => call_enter(m, env, Callee::Any),
        Key::TailCheck => tail_check(m, env),
        Key::ClearAt => m.store_imm(Width::U8, Reg::T2, reg(slot::A), NIL),
        Key::TailEnter { upvalues } => tail_enter(m, env, upvalues),
    }
}

fn frames(env: &Env) -> FrameLayout {
    env.frames.expect("call templates are built only with a frame layout")
}

fn closure_at(m: &mut dyn Masm, dst: Reg, scratch: Reg, base: Reg, disp: Imm) {
    m.load(Width::U32, dst, base, plus(disp, PAYLOAD));
    m.mov_imm(scratch, fixed(std::mem::size_of::<crate::object::Closure>() as i64));
    m.alu(Alu::Mul, dst, scratch);
    m.load(Width::U64, scratch, Reg::Ctx, context(offset_of!(JitCtx, closures)));
    m.alu(Alu::Add, dst, scratch);
}

fn closure_address(m: &mut dyn Masm, dst: Reg, scratch: Reg) {
    m.load(Width::U32, dst, Reg::Base, payload(slot::A));
    m.mov_imm(scratch, fixed(std::mem::size_of::<crate::object::Closure>() as i64));
    m.alu(Alu::Mul, dst, scratch);
    m.load(Width::U64, scratch, Reg::Ctx, context(offset_of!(JitCtx, closures)));
    m.alu(Alu::Add, dst, scratch);
}

fn call_check(m: &mut dyn Masm, env: &Env) {
    let frames = frames(env);
    callee_check(m, env);
    m.load(Width::U64, Reg::T1, Reg::Ctx, context(offset_of!(JitCtx, thread)));
    m.load(Width::U64, Reg::T2, Reg::T1, fixed(i64::from(frames.stack_len)));
    m.load(Width::U64, Reg::T3, Reg::Ctx, context(offset_of!(JitCtx, base)));
    m.mov_imm(Reg::T0, reg(slot::I));
    m.alu(Alu::Add, Reg::T0, Reg::T3);
    m.cmp(Reg::T0, Reg::T2);
    m.branch(Cond::Above, slow());
    m.mov_imm(Reg::T0, reg(slot::W));
    m.alu(Alu::Add, Reg::T0, Reg::T3);
    m.mov_imm(Reg::T5, fixed(crate::vm::MAX_STACK_SLOTS as i64));
    m.cmp(Reg::T0, Reg::T5);
    m.branch(Cond::Above, slow());
    m.load(Width::U64, Reg::T2, Reg::T1, fixed(i64::from(frames.frames_len)));
    m.load(Width::U64, Reg::T0, Reg::T1, fixed(i64::from(frames.frames_capacity)));
    m.cmp(Reg::T2, Reg::T0);
    m.branch(Cond::AboveEq, slow());
}

fn frame_field(frames: FrameLayout, offset: usize) -> Imm {
    fixed(i64::from(frames.field(offset)))
}

#[derive(Clone, Copy)]
enum Callee {
    Known { upvalues: bool },
    Any,
}

fn call_enter(m: &mut dyn Masm, env: &Env, callee: Callee) {
    let frames = frames(env);
    let area = m.callee_context();
    let at = |offset: usize| fixed(i64::from(area) + offset as i64);
    let other = m.new_label();

    m.stack_address(Reg::T5, 0);
    m.store(Width::U64, Reg::T2, Reg::T5, at(SPILL_COUNT));
    m.store(Width::U64, Reg::T4, Reg::T5, at(SPILL_ENTRY));
    push_frame(m, frames, at(SPILL_COUNT));
    fill_context(m, frames, callee, area);
    m.load(Width::U64, Reg::T2, Reg::Ctx, context(offset_of!(JitCtx, call_depth)));
    m.load(Width::U32, Reg::T0, Reg::T2, fixed(0));
    m.alu_imm(Alu::Add, Reg::T0, 1);
    m.store(Width::U32, Reg::T0, Reg::T2, fixed(0));
    m.load(Width::U64, Reg::T0, Reg::T5, at(SPILL_ENTRY));
    m.stack_address(Reg::T5, area);
    m.arg_reg(0, Reg::T5);
    m.arg_imm(1, fixed(0));
    m.call_reg(Reg::T0);
    m.load(Width::U64, Reg::T2, Reg::Ctx, context(offset_of!(JitCtx, call_depth)));
    m.load(Width::U32, Reg::T1, Reg::T2, fixed(0));
    m.alu_imm(Alu::Sub, Reg::T1, 1);
    m.store(Width::U32, Reg::T1, Reg::T2, fixed(0));
    m.cmp_imm(Reg::Ret, crate::jit::EXIT_RETURNED as i32);
    m.branch(Cond::Ne, other);

    m.stack_address(Reg::T5, 0);
    m.load(Width::U64, Reg::T1, Reg::T5, at(offset_of!(JitCtx, thread)));
    m.load(Width::U64, Reg::T0, Reg::T1, fixed(i64::from(frames.frames_len)));
    m.alu_imm(Alu::Sub, Reg::T0, 1);
    m.store(Width::U64, Reg::T0, Reg::T1, fixed(i64::from(frames.frames_len)));
    m.load(Width::U64, Reg::T0, Reg::T5, at(offset_of!(JitCtx, registers)));
    m.mov_imm(Reg::T2, reg(slot::V));
    m.alu(Alu::Sub, Reg::T0, Reg::T2);
    m.store(Width::U64, Reg::T0, Reg::Ctx, context(offset_of!(JitCtx, registers)));
    m.mov(Reg::Base, Reg::T0);
    for offset in REFRESHED {
        m.load(Width::U64, Reg::T2, Reg::T5, at(offset));
        m.store(Width::U64, Reg::T2, Reg::Ctx, context(offset));
    }
    m.store(Width::U64, Reg::T1, Reg::Ctx, context(offset_of!(JitCtx, thread)));
    m.jump(Target::Ext(ext::NEXT));

    m.bind(other);
    m.stack_address(Reg::T5, 0);
    m.arg_reg(2, Reg::Ret);
    m.load(Width::U64, Reg::T0, Reg::T5, at(SPILL_COUNT));
    m.arg_reg(3, Reg::T0);
    m.arg_reg(0, Reg::Ctx);
    m.arg_imm(1, reg(slot::PC));
    m.arg_imm(4, fixed(0));
    m.call(env.helpers.fast[crate::jit::compile::FAST_AFTER_CALL]);
    m.load(Width::U64, Reg::Base, Reg::Ctx, context(offset_of!(JitCtx, registers)));
    m.mov_imm(Reg::T1, reg(slot::I));
    m.cmp(Reg::Ret, Reg::T1);
    m.branch(Cond::Eq, Target::Ext(ext::NEXT));
    m.jump(Target::Ext(ext::DISPATCH));
}

fn push_frame(m: &mut dyn Masm, frames: FrameLayout, count: Imm) {
    m.load(Width::U64, Reg::T0, Reg::T1, fixed(i64::from(frames.frames_data)));
    m.mov_imm(Reg::T4, fixed(frames.frame_size));
    m.alu(Alu::Mul, Reg::T4, Reg::T2);
    m.alu(Alu::Add, Reg::T0, Reg::T4);
    m.mov(Reg::T4, Reg::T0);
    m.alu_imm(Alu::Sub, Reg::T4, frames.frame_size as i32);
    m.mov_imm(Reg::T2, reg(slot::I));
    m.store(Width::U32, Reg::T2, Reg::T4, frame_field(frames, lua_frame::PC));
    m.mov_imm(Reg::T4, reg(slot::B));
    m.alu(Alu::Add, Reg::T4, Reg::T3);
    m.store_imm(Width::U8, Reg::T0, fixed(0), 0);
    m.load(Width::U64, Reg::T2, Reg::Base, payload(slot::A));
    m.store(Width::U64, Reg::T2, Reg::T0, frame_field(frames, lua_frame::CLOSURE));
    m.store(Width::U64, Reg::T4, Reg::T0, frame_field(frames, lua_frame::BASE));
    for word in 0..lua_frame::CACHE_WORDS {
        m.store_imm(Width::U64, Reg::T0, frame_field(frames, lua_frame::CACHE + word * std::mem::size_of::<usize>()), 0);
    }
    m.mov_imm(Reg::T2, reg(slot::C));
    m.alu_imm(Alu::Add, Reg::T2, 1);
    m.shl_imm(Reg::T2, lua_frame::STATUS_SHIFT);
    m.store(Width::U64, Reg::T2, Reg::T0, frame_field(frames, lua_frame::PC));
    m.store_imm(Width::U64, Reg::T0, frame_field(frames, lua_frame::NVARARGS), 0);
    m.load(Width::U64, Reg::T2, Reg::T5, count);
    m.alu_imm(Alu::Add, Reg::T2, 1);
    m.store(Width::U64, Reg::T2, Reg::T1, fixed(i64::from(frames.frames_len)));
}

fn fill_context(m: &mut dyn Masm, frames: FrameLayout, callee: Callee, area: i32) {
    let at = |offset: usize| fixed(i64::from(area) + offset as i64);
    for offset in COPIED {
        m.load(Width::U64, Reg::T2, Reg::Ctx, context(offset));
        m.store(Width::U64, Reg::T2, Reg::T5, at(offset));
    }
    m.mov_imm(Reg::T2, reg(slot::V));
    m.alu(Alu::Add, Reg::T2, Reg::Base);
    m.store(Width::U64, Reg::T2, Reg::T5, at(offset_of!(JitCtx, registers)));
    if matches!(callee, Callee::Known { upvalues: false }) {
        m.store_imm(Width::U64, Reg::T5, at(offset_of!(JitCtx, upvalues)), 0);
    } else {
        closure_at(m, Reg::T2, Reg::T0, Reg::Base, reg(slot::A));
        m.load(Width::U64, Reg::T2, Reg::T2, fixed(i64::from(frames.closure_upvalues)));
        m.store(Width::U64, Reg::T2, Reg::T5, at(offset_of!(JitCtx, upvalues)));
    }
    match callee {
        Callee::Known { .. } => {
            m.mov_imm(Reg::T2, reg(slot::K));
            m.store(Width::U64, Reg::T2, Reg::T5, at(offset_of!(JitCtx, constants)));
            m.mov_imm(Reg::T2, reg(slot::Y));
            m.store(Width::U64, Reg::T2, Reg::T5, at(offset_of!(JitCtx, proto)));
        }
        Callee::Any => {
            m.load(Width::U64, Reg::T0, Reg::T5, at(SPILL_EXEC));
            m.load(Width::U64, Reg::T2, Reg::T0, fixed(exec_layout::CONSTANTS as i64));
            m.store(Width::U64, Reg::T2, Reg::T5, at(offset_of!(JitCtx, constants)));
            m.load(Width::U64, Reg::T2, Reg::T0, fixed(exec_layout::PROTO as i64));
            m.store(Width::U64, Reg::T2, Reg::T5, at(offset_of!(JitCtx, proto)));
        }
    }
    m.load(Width::U64, Reg::T2, Reg::Base, payload(slot::A));
    m.store(Width::U64, Reg::T2, Reg::T5, at(offset_of!(JitCtx, closure)));
    m.store(Width::U64, Reg::T4, Reg::T5, at(offset_of!(JitCtx, base)));
    m.mov_imm(Reg::T2, reg(slot::C));
    m.store(Width::U32, Reg::T2, Reg::T5, at(offset_of!(JitCtx, direct_results)));
    m.store_imm(Width::U64, Reg::T5, at(offset_of!(JitCtx, ret_offset)), -(std::mem::size_of::<crate::value::Value>() as i32));
}

/// Follows LuaJIT's `BC_CALL`, which reads the callee's frame facts from its prototype.
/// Leaves `T1` the thread, `T2` the frame count, `T3` the caller's base and `T4` the entry, as `CallCheck` does.
fn call_any_check(m: &mut dyn Masm, env: &Env) {
    let frames = frames(env);
    let area = m.callee_context();
    let at = |offset: usize| fixed(i64::from(area) + offset as i64);
    let fill = m.new_label();
    let filled = m.new_label();

    m.load(Width::U8, Reg::T0, Reg::Base, reg(slot::A));
    m.cmp_imm(Reg::T0, CLOSURE);
    m.branch(Cond::Ne, slow());
    closure_address(m, Reg::T0, Reg::T5);
    m.load(Width::U64, Reg::T0, Reg::T0, fixed(exec_layout::CLOSURE_CODE as i64));
    m.load(Width::U64, Reg::T4, Reg::T0, fixed(exec_layout::ENTRY as i64));
    m.load(Width::U64, Reg::T4, Reg::T4, fixed(0));
    m.cmp_imm(Reg::T4, 0);
    m.branch(Cond::Eq, slow());
    m.load(Width::U64, Reg::T1, Reg::Ctx, context(offset_of!(JitCtx, call_depth)));
    m.load(Width::U32, Reg::T1, Reg::T1, fixed(0));
    m.cmp_imm(Reg::T1, env.depth_limit as i32);
    m.branch(Cond::AboveEq, slow());
    m.stack_address(Reg::T5, 0);
    m.store(Width::U64, Reg::T0, Reg::T5, at(SPILL_EXEC));
    m.store(Width::U64, Reg::T4, Reg::T5, at(SPILL_ENTRY));

    m.load(Width::U64, Reg::T1, Reg::Ctx, context(offset_of!(JitCtx, thread)));
    m.load(Width::U64, Reg::T2, Reg::T1, fixed(i64::from(frames.stack_len)));
    m.load(Width::U64, Reg::T3, Reg::Ctx, context(offset_of!(JitCtx, base)));
    m.load(Width::U8, Reg::T0, Reg::T0, fixed(exec_layout::MAX_STACK as i64));
    m.mov_imm(Reg::T5, reg(slot::W));
    m.alu(Alu::Add, Reg::T0, Reg::T5);
    m.alu(Alu::Add, Reg::T0, Reg::T3);
    m.mov_imm(Reg::T5, fixed(crate::vm::MAX_STACK_SLOTS as i64));
    m.cmp(Reg::T0, Reg::T5);
    m.branch(Cond::Above, slow());
    m.cmp(Reg::T0, Reg::T2);
    m.branch(Cond::Above, slow());
    m.mov_imm(Reg::T5, Imm::hole_plus(slot::B, crate::vm::REGISTER_SPAN as i32));
    m.alu(Alu::Add, Reg::T5, Reg::T3);
    m.cmp(Reg::T5, Reg::T2);
    m.branch(Cond::Above, slow());
    m.load(Width::U64, Reg::T2, Reg::T1, fixed(i64::from(frames.frames_len)));
    m.load(Width::U64, Reg::T0, Reg::T1, fixed(i64::from(frames.frames_capacity)));
    m.cmp(Reg::T2, Reg::T0);
    m.branch(Cond::AboveEq, slow());

    m.stack_address(Reg::T5, 0);
    m.load(Width::U64, Reg::T0, Reg::T5, at(SPILL_EXEC));
    m.load(Width::U8, Reg::T0, Reg::T0, fixed(exec_layout::NUM_PARAMS as i64));
    m.mov_imm(Reg::T5, reg(slot::U));
    m.bind(fill);
    m.cmp(Reg::T5, Reg::T0);
    m.branch(Cond::AboveEq, filled);
    m.mov(Reg::T4, Reg::T5);
    m.shl_imm(Reg::T4, 4);
    m.alu(Alu::Add, Reg::T4, Reg::Base);
    m.store_imm(Width::U8, Reg::T4, reg(slot::V), NIL);
    m.alu_imm(Alu::Add, Reg::T5, 1);
    m.jump(fill);
    m.bind(filled);
    m.stack_address(Reg::T5, 0);
    m.load(Width::U64, Reg::T4, Reg::T5, at(SPILL_ENTRY));
}

fn callee_check(m: &mut dyn Masm, env: &Env) {
    m.load(Width::U8, Reg::T0, Reg::Base, reg(slot::A));
    m.cmp_imm(Reg::T0, CLOSURE);
    m.branch(Cond::Ne, slow());
    closure_address(m, Reg::T0, Reg::T5);
    m.load(Width::U64, Reg::T1, Reg::T0, context(offset_of!(crate::object::Closure, proto)));
    m.mov_imm(Reg::T2, reg(slot::Y));
    m.cmp(Reg::T1, Reg::T2);
    m.branch(Cond::Ne, Target::Ext(ext::TARGET));
    m.mov_imm(Reg::T4, reg(slot::X));
    m.load(Width::U64, Reg::T4, Reg::T4, fixed(0));
    m.cmp_imm(Reg::T4, 0);
    m.branch(Cond::Eq, slow());
    m.load(Width::U64, Reg::T1, Reg::Ctx, context(offset_of!(JitCtx, call_depth)));
    m.load(Width::U32, Reg::T1, Reg::T1, fixed(0));
    m.cmp_imm(Reg::T1, env.depth_limit as i32);
    m.branch(Cond::AboveEq, slow());
}

fn tail_check(m: &mut dyn Masm, env: &Env) {
    let frames = frames(env);
    let area = m.callee_context();
    let at = |offset: usize| fixed(i64::from(area) + offset as i64);
    callee_check(m, env);
    m.load(Width::U64, Reg::T1, Reg::Ctx, context(offset_of!(JitCtx, thread)));
    m.load(Width::U64, Reg::T2, Reg::T1, fixed(i64::from(frames.frames_len)));
    m.load(Width::U64, Reg::T0, Reg::T1, fixed(i64::from(frames.frames_data)));
    m.alu_imm(Alu::Sub, Reg::T2, 1);
    m.mov_imm(Reg::T5, fixed(frames.frame_size));
    m.alu(Alu::Mul, Reg::T2, Reg::T5);
    m.alu(Alu::Add, Reg::T0, Reg::T2);
    m.load(Width::U32, Reg::T2, Reg::T0, frame_field(frames, lua_frame::STATUS));
    m.mov_imm(Reg::T5, fixed(i64::from(lua_frame::OWN_CALLER)));
    m.alu(Alu::And, Reg::T2, Reg::T5);
    m.cmp_imm(Reg::T2, 0);
    m.branch(Cond::Ne, slow());
    m.load(Width::U64, Reg::T3, Reg::T0, frame_field(frames, lua_frame::BASE));
    m.alu_imm(Alu::Sub, Reg::T3, 1);
    m.load(Width::U64, Reg::T2, Reg::T1, fixed(i64::from(frames.stack_len)));
    m.mov_imm(Reg::T5, reg(slot::I));
    m.alu(Alu::Add, Reg::T5, Reg::T3);
    m.cmp(Reg::T5, Reg::T2);
    m.branch(Cond::Above, slow());
    m.mov_imm(Reg::T5, reg(slot::W));
    m.alu(Alu::Add, Reg::T5, Reg::T3);
    m.mov_imm(Reg::T2, fixed(crate::vm::MAX_STACK_SLOTS as i64));
    m.cmp(Reg::T5, Reg::T2);
    m.branch(Cond::Above, slow());
    m.stack_address(Reg::T5, 0);
    m.store(Width::U64, Reg::T0, Reg::T5, at(SPILL_COUNT));
    m.store(Width::U64, Reg::T4, Reg::T5, at(SPILL_ENTRY));
    m.mov(Reg::T2, Reg::T3);
    m.load(Width::U64, Reg::T5, Reg::Ctx, context(offset_of!(JitCtx, base)));
    m.alu(Alu::Sub, Reg::T2, Reg::T5);
    m.shl_imm(Reg::T2, 4);
    m.alu(Alu::Add, Reg::T2, Reg::Base);
}

fn tail_enter(m: &mut dyn Masm, env: &Env, upvalues: bool) {
    let frames = frames(env);
    let area = m.callee_context();
    let at = |offset: usize| fixed(i64::from(area) + offset as i64);

    m.stack_address(Reg::T5, 0);
    m.load(Width::U64, Reg::T0, Reg::T5, at(SPILL_COUNT));
    m.mov(Reg::T4, Reg::T3);
    m.alu_imm(Alu::Add, Reg::T4, 1);
    m.load(Width::U64, Reg::T1, Reg::T2, fixed(i64::from(PAYLOAD)));
    m.store(Width::U64, Reg::T1, Reg::T0, frame_field(frames, lua_frame::CLOSURE));
    m.store(Width::U64, Reg::T4, Reg::T0, frame_field(frames, lua_frame::BASE));
    for word in 0..lua_frame::CACHE_WORDS {
        m.store_imm(Width::U64, Reg::T0, frame_field(frames, lua_frame::CACHE + word * std::mem::size_of::<usize>()), 0);
    }
    m.load(Width::U32, Reg::T1, Reg::T0, frame_field(frames, lua_frame::STATUS));
    m.mov_imm(Reg::T3, fixed(i64::from(lua_frame::TAIL_KEEPS)));
    m.alu(Alu::And, Reg::T1, Reg::T3);
    m.mov_imm(Reg::T3, fixed(i64::from(lua_frame::TAIL)));
    m.alu(Alu::Or, Reg::T1, Reg::T3);
    m.shl_imm(Reg::T1, lua_frame::STATUS_SHIFT);
    m.store(Width::U64, Reg::T1, Reg::T0, frame_field(frames, lua_frame::PC));
    m.store_imm(Width::U64, Reg::T0, frame_field(frames, lua_frame::NVARARGS), 0);

    for offset in COPIED {
        m.load(Width::U64, Reg::T1, Reg::Ctx, context(offset));
        m.store(Width::U64, Reg::T1, Reg::T5, at(offset));
    }
    m.mov(Reg::T1, Reg::T2);
    m.alu_imm(Alu::Add, Reg::T1, std::mem::size_of::<crate::value::Value>() as i32);
    m.store(Width::U64, Reg::T1, Reg::T5, at(offset_of!(JitCtx, registers)));
    if upvalues {
        closure_at(m, Reg::T1, Reg::T0, Reg::T2, fixed(0));
        m.load(Width::U64, Reg::T1, Reg::T1, fixed(i64::from(frames.closure_upvalues)));
        m.store(Width::U64, Reg::T1, Reg::T5, at(offset_of!(JitCtx, upvalues)));
    } else {
        m.store_imm(Width::U64, Reg::T5, at(offset_of!(JitCtx, upvalues)), 0);
    }
    m.mov_imm(Reg::T1, reg(slot::K));
    m.store(Width::U64, Reg::T1, Reg::T5, at(offset_of!(JitCtx, constants)));
    m.load(Width::U64, Reg::T1, Reg::T2, fixed(i64::from(PAYLOAD)));
    m.store(Width::U64, Reg::T1, Reg::T5, at(offset_of!(JitCtx, closure)));
    m.store(Width::U64, Reg::T4, Reg::T5, at(offset_of!(JitCtx, base)));
    m.mov_imm(Reg::T1, reg(slot::Y));
    m.store(Width::U64, Reg::T1, Reg::T5, at(offset_of!(JitCtx, proto)));
    m.load(Width::U32, Reg::T1, Reg::Ctx, context(offset_of!(JitCtx, direct_results)));
    m.store(Width::U32, Reg::T1, Reg::T5, at(offset_of!(JitCtx, direct_results)));
    m.store_imm(Width::U64, Reg::T5, at(offset_of!(JitCtx, ret_offset)), -(std::mem::size_of::<crate::value::Value>() as i32));
    m.load(Width::U64, Reg::T2, Reg::Ctx, context(offset_of!(JitCtx, call_depth)));
    m.load(Width::U32, Reg::T0, Reg::T2, fixed(0));
    m.alu_imm(Alu::Add, Reg::T0, 1);
    m.store(Width::U32, Reg::T0, Reg::T2, fixed(0));

    m.load(Width::U64, Reg::T0, Reg::T5, at(SPILL_ENTRY));
    m.stack_address(Reg::T5, area);
    m.arg_reg(0, Reg::T5);
    m.arg_imm(1, fixed(0));
    m.call_reg(Reg::T0);
    m.load(Width::U64, Reg::T2, Reg::Ctx, context(offset_of!(JitCtx, call_depth)));
    m.load(Width::U32, Reg::T1, Reg::T2, fixed(0));
    m.alu_imm(Alu::Sub, Reg::T1, 1);
    m.store(Width::U32, Reg::T1, Reg::T2, fixed(0));
    m.cmp_imm(Reg::Ret, crate::jit::EXIT_RETURNED as i32);
    m.branch(Cond::Ne, Target::Ext(ext::EXIT));

    m.stack_address(Reg::T5, 0);
    for offset in REFRESHED {
        m.load(Width::U64, Reg::T2, Reg::T5, at(offset));
        m.store(Width::U64, Reg::T2, Reg::Ctx, context(offset));
    }
    m.load(Width::U64, Reg::T2, Reg::T5, at(offset_of!(JitCtx, thread)));
    m.store(Width::U64, Reg::T2, Reg::Ctx, context(offset_of!(JitCtx, thread)));
    m.load(Width::U64, Reg::T0, Reg::T5, at(offset_of!(JitCtx, registers)));
    m.load(Width::U64, Reg::T1, Reg::Ctx, context(offset_of!(JitCtx, base)));
    m.load(Width::U64, Reg::T2, Reg::T5, at(offset_of!(JitCtx, base)));
    m.alu(Alu::Sub, Reg::T1, Reg::T2);
    m.shl_imm(Reg::T1, 4);
    m.alu(Alu::Add, Reg::T0, Reg::T1);
    m.store(Width::U64, Reg::T0, Reg::Ctx, context(offset_of!(JitCtx, registers)));
    m.mov_imm(Reg::Ret, fixed(crate::jit::EXIT_RETURNED));
    m.jump(Target::Ext(ext::EXIT));
}

const COPIED: [usize; 9] = [
    offset_of!(JitCtx, requests),
    offset_of!(JitCtx, tables),
    offset_of!(JitCtx, table_marks),
    offset_of!(JitCtx, closures),
    offset_of!(JitCtx, vm),
    offset_of!(JitCtx, thread),
    offset_of!(JitCtx, thread_ref),
    offset_of!(JitCtx, call_depth),
    offset_of!(JitCtx, error),
];

const REFRESHED: [usize; 3] = [
    offset_of!(JitCtx, tables),
    offset_of!(JitCtx, table_marks),
    offset_of!(JitCtx, closures),
];

fn copy(m: &mut dyn Masm, from: Reg, source: Imm, to: Reg, destination: Imm) {
    assert!(from != Reg::T1 && to != Reg::T0 && to != Reg::T1, "a copy keeps its addresses");
    m.load(Width::U64, Reg::T1, from, plus(source, PAYLOAD));
    m.load(Width::U64, Reg::T0, from, source);
    m.store(Width::U64, Reg::T0, to, destination);
    m.store(Width::U64, Reg::T1, to, plus(destination, PAYLOAD));
}

fn plus(value: Imm, add: i32) -> Imm {
    match value {
        Imm::Fixed(value) => Imm::Fixed(value + i64::from(add)),
        Imm::Hole { slot, add: base } => Imm::Hole { slot, add: base + add },
    }
}

fn load_number(m: &mut dyn Masm, tag: i32) {
    m.store_imm(Width::U8, Reg::Base, reg(slot::A), tag);
    m.mov_imm(Reg::T0, reg(slot::I));
    m.store(Width::U64, Reg::T0, Reg::Base, payload(slot::A));
}

fn back_edge(m: &mut dyn Masm, charge: bool) {
    m.load(Width::U64, Reg::T0, Reg::Ctx, context(offset_of!(JitCtx, requests)));
    m.load(Width::U8, Reg::T0, Reg::T0, fixed(0));
    m.cmp_imm(Reg::T0, 0);
    m.branch(Cond::Ne, Target::Ext(ext::COLD));
    if charge {
        charge_budget(m);
    }
    m.jump(Target::Ext(ext::TARGET));
}

pub(crate) fn charge_budget(m: &mut dyn Masm) {
    m.mov_imm(Reg::T0, reg(slot::X));
    m.load(Width::U64, Reg::T1, Reg::T0, fixed(0));
    m.mov_imm(Reg::T2, reg(slot::Y));
    m.alu(Alu::Sub, Reg::T1, Reg::T2);
    m.store(Width::U64, Reg::T1, Reg::T0, fixed(0));
    m.cmp_imm(Reg::T1, 0);
    m.branch(Cond::Le, Target::Ext(ext::SLOW));
}

fn helper(m: &mut dyn Masm, env: &Env, id: HelperId) {
    let address = match id {
        HelperId::Step => env.helpers.step,
        HelperId::SafePoint => env.helpers.safe_point,
        HelperId::Misc => env.helpers.misc,
        HelperId::TierCheck => env.helpers.tier_check,
        HelperId::Fast(index) => env.helpers.fast[usize::from(index)],
    };
    m.arg_reg(0, Reg::Ctx);
    m.arg_imm(1, reg(slot::PC));
    if let HelperId::Fast(_) = id {
        m.arg_imm(2, reg(slot::A));
        m.arg_imm(3, reg(slot::B));
        m.arg_imm(4, reg(slot::C));
    }
    m.call(address);
    m.load(Width::U64, Reg::Base, Reg::Ctx, context(offset_of!(JitCtx, registers)));
    m.mov_imm(Reg::T1, reg(slot::I));
    m.cmp(Reg::Ret, Reg::T1);
    m.branch(Cond::Eq, Target::Ext(ext::NEXT));
    m.jump(Target::Ext(ext::DISPATCH));
}

fn slow() -> Target {
    Target::Ext(ext::SLOW)
}

fn rhs_payload_int(m: &mut dyn Masm, rhs: Rhs, dst: Reg) {
    match rhs {
        Rhs::Reg => m.load(Width::U64, dst, Reg::Base, payload(slot::C)),
        Rhs::KInt => m.load(Width::U64, dst, Reg::Konst, payload(slot::K)),
        Rhs::Imm => m.mov_imm(dst, reg(slot::I)),
        Rhs::KFloat => unreachable!("a float constant has no integer path"),
    }
}

fn rhs_float(m: &mut dyn Masm, rhs: Rhs, dst: FReg) {
    match rhs {
        Rhs::Reg => m.load_f(dst, Reg::Base, payload(slot::C)),
        Rhs::KFloat => m.load_f(dst, Reg::Konst, payload(slot::K)),
        Rhs::KInt => {
            m.load(Width::U64, Reg::T2, Reg::Konst, payload(slot::K));
            m.int_to_float(dst, Reg::T2);
        }
        Rhs::Imm => {
            m.mov_imm(Reg::T2, reg(slot::I));
            m.int_to_float(dst, Reg::T2);
        }
    }
}

/// The success path skips the `MMBIN` that follows.
fn arith(m: &mut dyn Masm, op: ArithKind, rhs: Rhs) {
    let left_float = m.new_label();
    let float_op = m.new_label();
    let int_left_float_right = m.new_label();

    m.load(Width::U8, Reg::T0, Reg::Base, reg(slot::B));
    if rhs == Rhs::Reg {
        m.load(Width::U8, Reg::T1, Reg::Base, reg(slot::C));
    }
    m.cmp_imm(Reg::T0, INT);
    m.branch(Cond::Ne, if op.has_float() { left_float } else { slow() });
    match rhs {
        Rhs::Reg => {
            m.cmp_imm(Reg::T1, INT);
            m.branch(Cond::Ne, if op.has_float() { int_left_float_right } else { slow() });
            int_arith(m, op, rhs);
        }
        Rhs::KInt | Rhs::Imm => int_arith(m, op, rhs),
        Rhs::KFloat => m.jump(int_left_float_right),
    }
    if !op.has_float() {
        m.bind(left_float);
        m.bind(float_op);
        m.bind(int_left_float_right);
        m.jump(slow());
        return;
    }

    m.bind(int_left_float_right);
    if rhs == Rhs::Reg {
        m.cmp_imm(Reg::T1, FLOAT);
        m.branch(Cond::Ne, slow());
    }
    if matches!(rhs, Rhs::Reg | Rhs::KFloat) {
        m.load(Width::U64, Reg::T2, Reg::Base, payload(slot::B));
        m.int_to_float(FReg::F0, Reg::T2);
        rhs_float(m, rhs, FReg::F1);
        m.jump(float_op);
    } else {
        m.jump(slow());
    }

    m.bind(left_float);
    m.cmp_imm(Reg::T0, FLOAT);
    m.branch(Cond::Ne, slow());
    m.load_f(FReg::F0, Reg::Base, payload(slot::B));
    if rhs == Rhs::Reg {
        let right_float = m.new_label();
        m.cmp_imm(Reg::T1, FLOAT);
        m.branch(Cond::Eq, right_float);
        m.cmp_imm(Reg::T1, INT);
        m.branch(Cond::Ne, slow());
        m.load(Width::U64, Reg::T2, Reg::Base, payload(slot::C));
        m.int_to_float(FReg::F1, Reg::T2);
        m.jump(float_op);
        m.bind(right_float);
    }
    rhs_float(m, rhs, FReg::F1);

    m.bind(float_op);
    let fop = match op {
        ArithKind::Add => FAlu::Add,
        ArithKind::Sub => FAlu::Sub,
        ArithKind::Mul => FAlu::Mul,
        ArithKind::Div => FAlu::Div,
        _ => unreachable!("only these operators have a float path"),
    };
    m.fop(fop, FReg::F0, FReg::F1);
    m.store_imm(Width::U8, Reg::Base, reg(slot::A), FLOAT);
    m.store_f(FReg::F0, Reg::Base, payload(slot::A));
    m.jump(Target::Ext(ext::SKIP));
}

fn int_arith(m: &mut dyn Masm, op: ArithKind, rhs: Rhs) {
    match op {
        ArithKind::Div => {
            m.load(Width::U64, Reg::T2, Reg::Base, payload(slot::B));
            m.int_to_float(FReg::F0, Reg::T2);
            rhs_payload_int(m, rhs, Reg::T2);
            m.int_to_float(FReg::F1, Reg::T2);
            m.fop(FAlu::Div, FReg::F0, FReg::F1);
            m.store_imm(Width::U8, Reg::Base, reg(slot::A), FLOAT);
            m.store_f(FReg::F0, Reg::Base, payload(slot::A));
            m.jump(Target::Ext(ext::SKIP));
            return;
        }
        ArithKind::Mod | ArithKind::IDiv => {
            m.load(Width::U64, Reg::T3, Reg::Base, payload(slot::B));
            rhs_payload_int(m, rhs, Reg::T4);
            if rhs == Rhs::Reg {
                m.mov(Reg::T5, Reg::T4);
                m.alu_imm(Alu::Add, Reg::T5, 1);
                m.cmp_imm(Reg::T5, 1);
                m.branch(Cond::BelowEq, slow());
            }
            let division = if op == ArithKind::Mod { Division::Mod } else { Division::IDiv };
            m.floor_div(division, Reg::T3, Reg::T3, Reg::T4);
            store_int(m, Reg::T3);
            return;
        }
        _ => {}
    }
    m.load(Width::U64, Reg::T2, Reg::Base, payload(slot::B));
    rhs_payload_int(m, rhs, Reg::T3);
    let alu = match op {
        ArithKind::Add => Alu::Add,
        ArithKind::Sub => Alu::Sub,
        ArithKind::Mul => Alu::Mul,
        ArithKind::BAnd => Alu::And,
        ArithKind::BOr => Alu::Or,
        ArithKind::BXor => Alu::Xor,
        _ => unreachable!("handled above"),
    };
    m.alu(alu, Reg::T2, Reg::T3);
    store_int(m, Reg::T2);
}

fn store_int(m: &mut dyn Masm, value: Reg) {
    m.store_imm(Width::U8, Reg::Base, reg(slot::A), INT);
    m.store(Width::U64, value, Reg::Base, payload(slot::A));
    m.jump(Target::Ext(ext::SKIP));
}

fn unm(m: &mut dyn Masm) {
    let float = m.new_label();
    m.load(Width::U8, Reg::T0, Reg::Base, reg(slot::B));
    m.load(Width::U64, Reg::T1, Reg::Base, payload(slot::B));
    m.cmp_imm(Reg::T0, INT);
    m.branch(Cond::Ne, float);
    m.neg(Reg::T1);
    m.store_imm(Width::U8, Reg::Base, reg(slot::A), INT);
    m.store(Width::U64, Reg::T1, Reg::Base, payload(slot::A));
    m.jump(Target::Ext(ext::SKIP));
    m.bind(float);
    m.cmp_imm(Reg::T0, FLOAT);
    m.branch(Cond::Ne, slow());
    m.mov_imm(Reg::T2, fixed(i64::MIN));
    m.alu(Alu::Xor, Reg::T1, Reg::T2);
    m.store_imm(Width::U8, Reg::Base, reg(slot::A), FLOAT);
    m.store(Width::U64, Reg::T1, Reg::Base, payload(slot::A));
    m.jump(Target::Ext(ext::SKIP));
}

fn bnot(m: &mut dyn Masm) {
    m.load(Width::U8, Reg::T0, Reg::Base, reg(slot::B));
    m.cmp_imm(Reg::T0, INT);
    m.branch(Cond::Ne, slow());
    m.load(Width::U64, Reg::T1, Reg::Base, payload(slot::B));
    m.not(Reg::T1);
    m.store_imm(Width::U8, Reg::Base, reg(slot::A), INT);
    m.store(Width::U64, Reg::T1, Reg::Base, payload(slot::A));
    m.jump(Target::Ext(ext::SKIP));
}

fn not(m: &mut dyn Masm) {
    let store = m.new_label();
    let truthy = m.new_label();
    m.mov_imm(Reg::T1, fixed(1));
    m.load(Width::U8, Reg::T0, Reg::Base, reg(slot::B));
    m.cmp_imm(Reg::T0, NIL);
    m.branch(Cond::Eq, store);
    m.cmp_imm(Reg::T0, BOOL);
    m.branch(Cond::Ne, truthy);
    m.load(Width::U8, Reg::T0, Reg::Base, payload(slot::B));
    m.cmp_imm(Reg::T0, 0);
    m.branch(Cond::Eq, store);
    m.bind(truthy);
    m.mov_imm(Reg::T1, fixed(0));
    m.bind(store);
    m.store_imm(Width::U8, Reg::Base, reg(slot::A), BOOL);
    m.store(Width::U8, Reg::T1, Reg::Base, payload(slot::A));
}

fn truthy(m: &mut dyn Masm, register: u8) {
    m.load(Width::U8, Reg::T0, Reg::Base, reg(register));
    m.cmp_imm(Reg::T0, NIL);
    m.branch(Cond::Eq, Target::Ext(ext::FALSE));
    m.cmp_imm(Reg::T0, BOOL);
    m.branch(Cond::Ne, Target::Ext(ext::TRUE));
    m.load(Width::U8, Reg::T0, Reg::Base, payload(register));
    m.cmp_imm(Reg::T0, 0);
    m.branch(Cond::Eq, Target::Ext(ext::FALSE));
    m.jump(Target::Ext(ext::TRUE));
}

fn compare(m: &mut dyn Masm, cmp: CmpKind, rhs: Rhs, (int, float): (bool, bool), default_false: bool) {
    let other = m.new_label();
    let fallback = if default_false { Target::Ext(ext::FALSE) } else { slow() };
    m.load(Width::U8, Reg::T0, Reg::Base, reg(slot::A));
    if rhs == Rhs::Reg {
        m.load(Width::U8, Reg::T1, Reg::Base, reg(slot::B));
    }
    if int {
        m.cmp_imm(Reg::T0, INT);
        m.branch(Cond::Ne, other);
        if rhs == Rhs::Reg {
            m.cmp_imm(Reg::T1, INT);
            m.branch(Cond::Ne, slow());
        }
        m.load(Width::U64, Reg::T2, Reg::Base, payload(slot::A));
        match rhs {
            Rhs::Reg => m.load(Width::U64, Reg::T3, Reg::Base, payload(slot::B)),
            Rhs::KInt => m.load(Width::U64, Reg::T3, Reg::Konst, payload(slot::K)),
            Rhs::Imm => m.mov_imm(Reg::T3, reg(slot::I)),
            Rhs::KFloat => unreachable!("a float constant compares on the float path"),
        }
        m.cmp(Reg::T2, Reg::T3);
        let cond = match cmp {
            CmpKind::Eq => Cond::Eq,
            CmpKind::Lt => Cond::Lt,
            CmpKind::Le => Cond::Le,
            CmpKind::Gt => Cond::Gt,
            CmpKind::Ge => Cond::Ge,
        };
        m.branch(cond, Target::Ext(ext::TRUE));
        m.jump(Target::Ext(ext::FALSE));
    }
    m.bind(other);
    if !float {
        m.jump(fallback);
        return;
    }
    m.cmp_imm(Reg::T0, FLOAT);
    m.branch(Cond::Ne, fallback);
    if rhs == Rhs::Reg {
        m.cmp_imm(Reg::T1, FLOAT);
        m.branch(Cond::Ne, slow());
    }
    m.load_f(FReg::F0, Reg::Base, payload(slot::A));
    match rhs {
        Rhs::Reg => m.load_f(FReg::F1, Reg::Base, payload(slot::B)),
        _ => rhs_float(m, rhs, FReg::F1),
    }
    let (cond, left, right) = match cmp {
        CmpKind::Eq => (FCond::Eq, FReg::F0, FReg::F1),
        CmpKind::Lt => (FCond::Lt, FReg::F0, FReg::F1),
        CmpKind::Le => (FCond::Le, FReg::F0, FReg::F1),
        CmpKind::Gt => (FCond::Lt, FReg::F1, FReg::F0),
        CmpKind::Ge => (FCond::Le, FReg::F1, FReg::F0),
    };
    m.fbranch(cond, left, right, Target::Ext(ext::TRUE));
    m.jump(Target::Ext(ext::FALSE));
}

fn eq_tag(m: &mut dyn Masm, tag: TagConstant) {
    let expected = match tag {
        TagConstant::Nil => NIL,
        TagConstant::Bool => BOOL,
        TagConstant::String => STRING,
    };
    m.load(Width::U8, Reg::T0, Reg::Base, reg(slot::A));
    m.cmp_imm(Reg::T0, expected);
    m.branch(Cond::Ne, Target::Ext(ext::FALSE));
    match tag {
        TagConstant::Nil => {}
        TagConstant::Bool => {
            m.load(Width::U8, Reg::T0, Reg::Base, payload(slot::A));
            m.mov_imm(Reg::T1, reg(slot::I));
            m.cmp(Reg::T0, Reg::T1);
            m.branch(Cond::Ne, Target::Ext(ext::FALSE));
        }
        TagConstant::String => {
            m.load(Width::U64, Reg::T0, Reg::Base, payload(slot::A));
            m.mov_imm(Reg::T1, reg(slot::I));
            m.cmp(Reg::T0, Reg::T1);
            m.branch(Cond::Ne, Target::Ext(ext::FALSE));
        }
    }
    m.jump(Target::Ext(ext::TRUE));
}

fn for_loop(m: &mut dyn Masm) {
    let float = m.new_label();
    let goes_on = m.new_label();
    let up = m.new_label();
    let store = m.new_label();
    m.load(Width::U8, Reg::T0, Reg::Base, reg(slot::A));
    m.load(Width::U8, Reg::T1, Reg::Base, Imm::hole_plus(slot::A, 16));
    m.load(Width::U8, Reg::T2, Reg::Base, Imm::hole_plus(slot::A, 32));
    m.cmp_imm(Reg::T0, INT);
    m.branch(Cond::Ne, float);
    m.cmp_imm(Reg::T1, INT);
    m.branch(Cond::Ne, slow());
    m.cmp_imm(Reg::T2, INT);
    m.branch(Cond::Ne, slow());
    m.load(Width::U64, Reg::T0, Reg::Base, payload(slot::A));
    m.cmp_imm(Reg::T0, 0);
    m.branch(Cond::Eq, Target::Ext(ext::NEXT));
    m.alu_imm(Alu::Sub, Reg::T0, 1);
    m.store(Width::U64, Reg::T0, Reg::Base, payload(slot::A));
    m.load(Width::U64, Reg::T1, Reg::Base, Imm::hole_plus(slot::A, 24));
    m.load(Width::U64, Reg::T2, Reg::Base, Imm::hole_plus(slot::A, 40));
    m.alu(Alu::Add, Reg::T2, Reg::T1);
    m.store(Width::U64, Reg::T2, Reg::Base, Imm::hole_plus(slot::A, 40));
    m.jump(goes_on);

    m.bind(float);
    m.cmp_imm(Reg::T0, FLOAT);
    m.branch(Cond::Ne, slow());
    m.cmp_imm(Reg::T1, FLOAT);
    m.branch(Cond::Ne, slow());
    m.cmp_imm(Reg::T2, FLOAT);
    m.branch(Cond::Ne, slow());
    m.load_f(FReg::F2, Reg::Base, payload(slot::A));
    m.load_f(FReg::F1, Reg::Base, Imm::hole_plus(slot::A, 24));
    m.load_f(FReg::F0, Reg::Base, Imm::hole_plus(slot::A, 40));
    m.fop(FAlu::Add, FReg::F0, FReg::F1);
    m.mov_imm(Reg::T0, fixed(0));
    m.int_to_float(FReg::F3, Reg::T0);
    m.fbranch(FCond::Lt, FReg::F3, FReg::F1, up);
    m.fbranch(FCond::Le, FReg::F2, FReg::F0, store);
    m.jump(Target::Ext(ext::NEXT));
    m.bind(up);
    m.fbranch(FCond::Le, FReg::F0, FReg::F2, store);
    m.jump(Target::Ext(ext::NEXT));
    m.bind(store);
    m.store_f(FReg::F0, Reg::Base, Imm::hole_plus(slot::A, 40));
    m.bind(goes_on);
}

fn layout(env: &Env) -> TableLayout {
    env.table.expect("table templates are built only with a table layout")
}

fn table_address(m: &mut dyn Masm, env: &Env, index: Reg) {
    let layout = layout(env);
    m.mov_imm(Reg::T5, fixed(layout.size));
    m.alu(Alu::Mul, index, Reg::T5);
    m.load(Width::U64, Reg::T5, Reg::Ctx, context(offset_of!(JitCtx, tables)));
    m.alu(Alu::Add, index, Reg::T5);
    if index != Reg::T0 {
        m.mov(Reg::T0, index);
    }
}

fn table_in_register(m: &mut dyn Masm, env: &Env, register: u8) {
    m.load(Width::U8, Reg::T0, Reg::Base, reg(register));
    m.cmp_imm(Reg::T0, TABLE);
    m.branch(Cond::Ne, slow());
    m.load(Width::U32, Reg::T0, Reg::Base, payload(register));
    table_address(m, env, Reg::T0);
}

fn array_element(m: &mut dyn Masm, env: &Env, constant_key: bool, key: u8) {
    let layout = layout(env);
    if constant_key {
        m.mov_imm(Reg::T1, reg(slot::I));
    } else {
        m.load(Width::U8, Reg::T1, Reg::Base, reg(key));
        m.cmp_imm(Reg::T1, INT);
        m.branch(Cond::Ne, slow());
        m.load(Width::U64, Reg::T1, Reg::Base, payload(key));
    }
    m.alu_imm(Alu::Sub, Reg::T1, 1);
    m.load(Width::U64, Reg::T2, Reg::T0, fixed(i64::from(layout.array_len)));
    m.cmp(Reg::T1, Reg::T2);
    m.branch(Cond::AboveEq, slow());
    m.load(Width::U64, Reg::T2, Reg::T0, fixed(i64::from(layout.array)));
    m.shl_imm(Reg::T1, 4);
    m.alu(Alu::Add, Reg::T2, Reg::T1);
}

fn array_get(m: &mut dyn Masm, env: &Env, constant_key: bool) {
    table_in_register(m, env, slot::B);
    array_element(m, env, constant_key, slot::C);
    present_into(m, Reg::T2, fixed(0));
    m.store(Width::U64, Reg::T1, Reg::Base, reg(slot::A));
    m.store(Width::U64, Reg::T3, Reg::Base, payload(slot::A));
}

fn present_into(m: &mut dyn Masm, base: Reg, disp: Imm) {
    m.load(Width::U64, Reg::T1, base, disp);
    m.load(Width::U64, Reg::T3, base, plus(disp, PAYLOAD));
    m.mov(Reg::T4, Reg::T1);
    m.alu_imm(Alu::And, Reg::T4, 0xff);
    m.cmp_imm(Reg::T4, NIL);
    m.branch(Cond::Eq, slow());
}

fn store_value(m: &mut dyn Masm, constant: bool) {
    if constant {
        m.load(Width::U64, Reg::T1, Reg::Konst, reg(slot::K));
        m.load(Width::U64, Reg::T3, Reg::Konst, payload(slot::K));
    } else {
        m.load(Width::U64, Reg::T1, Reg::Base, reg(slot::C));
        m.load(Width::U64, Reg::T3, Reg::Base, payload(slot::C));
    }
}

fn array_set(m: &mut dyn Masm, env: &Env, constant_key: bool, constant_value: bool) {
    let layout = layout(env);
    let store = m.new_label();
    table_in_register(m, env, slot::A);
    array_element(m, env, constant_key, slot::B);
    m.load(Width::U8, Reg::T1, Reg::T2, fixed(0));
    m.cmp_imm(Reg::T1, NIL);
    m.branch(Cond::Ne, store);
    m.load(Width::U32, Reg::T1, Reg::T0, fixed(i64::from(layout.metatable_tag)));
    m.mov_imm(Reg::T5, fixed(i64::from(layout.metatable_none)));
    m.cmp(Reg::T1, Reg::T5);
    m.branch(Cond::Ne, slow());
    m.bind(store);
    store_value(m, constant_value);
    m.store(Width::U64, Reg::T1, Reg::T2, fixed(0));
    m.store(Width::U64, Reg::T3, Reg::T2, fixed(i64::from(PAYLOAD)));
    barrier(m, slot::A);
}

fn barrier(m: &mut dyn Masm, register: u8) {
    m.mov(Reg::T4, Reg::T1);
    m.alu_imm(Alu::And, Reg::T4, 0xff);
    m.alu_imm(Alu::Sub, Reg::T4, STRING);
    m.cmp_imm(Reg::T4, LIGHT_USERDATA - STRING);
    m.branch(Cond::AboveEq, Target::Ext(ext::NEXT));
    m.load(Width::U32, Reg::T4, Reg::Base, payload(register));
    m.load(Width::U64, Reg::T5, Reg::Ctx, context(offset_of!(JitCtx, table_marks)));
    m.alu(Alu::Add, Reg::T5, Reg::T4);
    m.load(Width::U8, Reg::T5, Reg::T5, fixed(0));
    m.alu_imm(Alu::And, Reg::T5, i32::from(crate::gc::BLACK));
    m.cmp_imm(Reg::T5, 0);
    m.branch(Cond::Eq, Target::Ext(ext::NEXT));
    m.jump(Target::Ext(ext::COLD));
}

fn register_table(m: &mut dyn Masm, env: &Env) {
    table_in_register(m, env, slot::X);
    table_shape(m, env);
}

fn table_shape(m: &mut dyn Masm, env: &Env) {
    let layout = layout(env);
    m.load(Width::U8, Reg::T1, Reg::T0, fixed(i64::from(layout.shaped)));
    m.cmp_imm(Reg::T1, 0);
    m.branch(Cond::Eq, slow());
    m.load(Width::U64, Reg::T1, Reg::T0, fixed(i64::from(layout.shape)));
}

fn upvalue_cell(m: &mut dyn Masm) {
    m.load(Width::U64, Reg::T0, Reg::Ctx, context(offset_of!(JitCtx, upvalues)));
    m.load(Width::U64, Reg::T0, Reg::T0, reg(slot::B));
}

fn upvalue_address(m: &mut dyn Masm) {
    upvalue_cell(m);
    m.load(Width::U64, Reg::T0, Reg::T0, fixed(i64::from(upvalue::VALUE)));
}

fn get_upval(m: &mut dyn Masm) {
    upvalue_address(m);
    copy(m, Reg::T0, fixed(0), Reg::Base, reg(slot::A));
}

/// Leaves the collector's marks as `Heap::set_closed_upvalue` would.
fn set_upval(m: &mut dyn Masm) {
    let store = m.new_label();
    upvalue_cell(m);
    m.load(Width::U8, Reg::T1, Reg::T0, fixed(i64::from(upvalue::TAG)));
    m.cmp_imm(Reg::T1, upvalue::OPEN as i32);
    m.branch(Cond::Eq, store);
    m.load(Width::U8, Reg::T1, Reg::Base, reg(slot::A));
    m.alu_imm(Alu::Sub, Reg::T1, STRING);
    m.cmp_imm(Reg::T1, LIGHT_USERDATA - STRING - 1);
    m.branch(Cond::BelowEq, slow());
    m.bind(store);
    m.load(Width::U64, Reg::T2, Reg::T0, fixed(i64::from(upvalue::VALUE)));
    copy(m, Reg::Base, reg(slot::A), Reg::T2, fixed(0));
}

fn upvalue_table(m: &mut dyn Masm, env: &Env) {
    upvalue_address(m);
    m.load(Width::U8, Reg::T1, Reg::T0, fixed(0));
    m.cmp_imm(Reg::T1, TABLE);
    m.branch(Cond::Ne, slow());
    m.load(Width::U32, Reg::T0, Reg::T0, fixed(i64::from(PAYLOAD)));
    table_address(m, env, Reg::T0);
    table_shape(m, env);
}

fn field_get(m: &mut dyn Masm, env: &Env, absent: bool, method: bool) {
    let layout = layout(env);
    let other = m.new_label();
    m.mov_imm(Reg::T2, reg(slot::Y));
    m.cmp(Reg::T1, Reg::T2);
    m.branch(Cond::Ne, other);
    if absent {
        let nil = m.new_label();
        m.load(Width::U32, Reg::T2, Reg::T0, fixed(i64::from(layout.metatable_tag)));
        m.mov_imm(Reg::T3, fixed(i64::from(layout.metatable_none)));
        m.cmp(Reg::T2, Reg::T3);
        m.branch(Cond::Eq, nil);
        m.load(Width::U32, Reg::T2, Reg::T0, fixed(i64::from(layout.metatable_index)));
        table_address(m, env, Reg::T2);
        m.load(Width::U8, Reg::T3, Reg::T0, fixed(i64::from(layout.absent)));
        let bit = crate::vm::Event::Index.absence_bit().map_or(0, i32::from);
        m.alu_imm(Alu::And, Reg::T3, bit);
        m.cmp_imm(Reg::T3, 0);
        m.branch(Cond::Eq, slow());
        m.bind(nil);
        m.mov_imm(Reg::T1, fixed(i64::from(NIL)));
        m.mov_imm(Reg::T3, fixed(0));
    } else {
        m.load(Width::U64, Reg::T2, Reg::T0, fixed(i64::from(layout.slots)));
        present_into(m, Reg::T2, reg(slot::X));
    }
    finish_read(m, method);
    m.bind(other);
}

fn finish_read(m: &mut dyn Masm, method: bool) {
    if method {
        m.load(Width::U64, Reg::T4, Reg::Base, reg(slot::B));
        m.load(Width::U64, Reg::T5, Reg::Base, payload(slot::B));
        m.store(Width::U64, Reg::T4, Reg::Base, Imm::hole_plus(slot::A, 16));
        m.store(Width::U64, Reg::T5, Reg::Base, Imm::hole_plus(slot::A, 24));
    }
    m.store(Width::U64, Reg::T1, Reg::Base, reg(slot::A));
    m.store(Width::U64, Reg::T3, Reg::Base, payload(slot::A));
    m.jump(Target::Ext(ext::NEXT));
}

fn method_get(m: &mut dyn Masm, env: &Env) {
    let layout = layout(env);
    let other = m.new_label();
    m.mov_imm(Reg::T2, reg(slot::Y));
    m.cmp(Reg::T1, Reg::T2);
    m.branch(Cond::Ne, other);
    index_class(m, env, other);
    m.load(Width::U64, Reg::T2, Reg::T0, fixed(i64::from(layout.slots)));
    m.mov_imm(Reg::T3, reg(slot::X));
    m.alu(Alu::Add, Reg::T2, Reg::T3);
    present_into(m, Reg::T2, fixed(0));
    finish_read(m, true);
    m.bind(other);
    m.load(Width::U32, Reg::T0, Reg::Base, payload(slot::B));
    table_address(m, env, Reg::T0);
    m.load(Width::U64, Reg::T1, Reg::T0, fixed(i64::from(layout.shape)));
}

fn method_start(m: &mut dyn Masm, env: &Env) {
    m.mov_imm(Reg::T2, reg(slot::Y));
    m.cmp(Reg::T1, Reg::T2);
    m.branch(Cond::Ne, Target::Ext(ext::TARGET));
    index_class(m, env, Target::Ext(ext::TARGET));
}

fn method_hop(m: &mut dyn Masm, env: &Env) {
    index_class(m, env, Target::Ext(ext::TARGET));
}

fn method_finish(m: &mut dyn Masm, env: &Env) {
    let layout = layout(env);
    m.load(Width::U64, Reg::T2, Reg::T0, fixed(i64::from(layout.slots)));
    m.mov_imm(Reg::T3, reg(slot::X));
    m.alu(Alu::Add, Reg::T2, Reg::T3);
    present_into(m, Reg::T2, fixed(0));
    finish_read(m, true);
}

fn index_class(m: &mut dyn Masm, env: &Env, other: Target) {
    let layout = layout(env);
    m.load(Width::U32, Reg::T2, Reg::T0, fixed(i64::from(layout.metatable_tag)));
    m.mov_imm(Reg::T3, fixed(i64::from(layout.metatable_none)));
    m.cmp(Reg::T2, Reg::T3);
    m.branch(Cond::Eq, other);
    m.load(Width::U32, Reg::T2, Reg::T0, fixed(i64::from(layout.metatable_index)));
    table_address(m, env, Reg::T2);
    m.load(Width::U8, Reg::T1, Reg::T0, fixed(i64::from(layout.shaped)));
    m.cmp_imm(Reg::T1, 0);
    m.branch(Cond::Eq, other);
    m.load(Width::U64, Reg::T1, Reg::T0, fixed(i64::from(layout.shape)));
    m.mov_imm(Reg::T2, reg(slot::Z));
    m.cmp(Reg::T1, Reg::T2);
    m.branch(Cond::Ne, other);
    m.load(Width::U64, Reg::T2, Reg::T0, fixed(i64::from(layout.slots)));
    m.mov_imm(Reg::T3, reg(slot::W));
    m.alu(Alu::Add, Reg::T2, Reg::T3);
    m.load(Width::U8, Reg::T1, Reg::T2, fixed(0));
    m.cmp_imm(Reg::T1, TABLE);
    m.branch(Cond::Ne, other);
    m.load(Width::U64, Reg::T1, Reg::T2, fixed(i64::from(PAYLOAD)));
    m.mov_imm(Reg::T3, reg(slot::V));
    m.cmp(Reg::T1, Reg::T3);
    m.branch(Cond::Ne, other);
    m.mov_imm(Reg::T0, reg(slot::C));
    table_address(m, env, Reg::T0);
    m.load(Width::U8, Reg::T1, Reg::T0, fixed(i64::from(layout.shaped)));
    m.cmp_imm(Reg::T1, 0);
    m.branch(Cond::Eq, other);
    m.load(Width::U64, Reg::T1, Reg::T0, fixed(i64::from(layout.shape)));
    m.mov_imm(Reg::T2, reg(slot::U));
    m.cmp(Reg::T1, Reg::T2);
    m.branch(Cond::Ne, other);
}

fn field_set(m: &mut dyn Masm, env: &Env, constant: bool) {
    let layout = layout(env);
    let other = m.new_label();
    m.mov_imm(Reg::T2, reg(slot::Y));
    m.cmp(Reg::T1, Reg::T2);
    m.branch(Cond::Ne, other);
    m.load(Width::U64, Reg::T2, Reg::T0, fixed(i64::from(layout.slots)));
    m.mov_imm(Reg::T3, reg(slot::I));
    m.alu(Alu::Add, Reg::T2, Reg::T3);
    m.load(Width::U8, Reg::T3, Reg::T2, fixed(0));
    m.cmp_imm(Reg::T3, NIL);
    m.branch(Cond::Eq, slow());
    store_value(m, constant);
    m.store(Width::U64, Reg::T1, Reg::T2, fixed(0));
    m.store(Width::U64, Reg::T3, Reg::T2, fixed(i64::from(PAYLOAD)));
    barrier(m, slot::A);
    m.bind(other);
}

/// The slot vector must already have room, so the store allocates nothing.
fn field_add(m: &mut dyn Masm, env: &Env, constant: bool) {
    let layout = layout(env);
    let other = m.new_label();
    m.mov_imm(Reg::T2, reg(slot::Y));
    m.cmp(Reg::T1, Reg::T2);
    m.branch(Cond::Ne, other);
    m.load(Width::U32, Reg::T2, Reg::T0, fixed(i64::from(layout.metatable_tag)));
    m.mov_imm(Reg::T3, fixed(i64::from(layout.metatable_none)));
    m.cmp(Reg::T2, Reg::T3);
    m.branch(Cond::Ne, slow());
    m.load(Width::U32, Reg::T4, Reg::Base, payload(slot::A));
    m.load(Width::U64, Reg::T5, Reg::Ctx, context(offset_of!(JitCtx, table_marks)));
    m.alu(Alu::Add, Reg::T5, Reg::T4);
    m.load(Width::U8, Reg::T5, Reg::T5, fixed(0));
    m.alu_imm(Alu::And, Reg::T5, i32::from(crate::gc::BLACK));
    m.cmp_imm(Reg::T5, 0);
    m.branch(Cond::Ne, slow());
    m.load(Width::U64, Reg::T2, Reg::T0, fixed(i64::from(layout.slots_capacity)));
    m.mov_imm(Reg::T3, reg(slot::W));
    m.cmp(Reg::T3, Reg::T2);
    m.branch(Cond::Above, slow());
    m.store(Width::U64, Reg::T3, Reg::T0, fixed(i64::from(layout.slots_len)));
    m.mov_imm(Reg::T2, reg(slot::Z));
    m.store(Width::U64, Reg::T2, Reg::T0, fixed(i64::from(layout.shape)));
    m.store_imm(Width::U8, Reg::T0, fixed(i64::from(layout.absent)), 0);
    m.load(Width::U64, Reg::T2, Reg::T0, fixed(i64::from(layout.slots)));
    m.mov_imm(Reg::T3, reg(slot::I));
    m.alu(Alu::Add, Reg::T2, Reg::T3);
    store_value(m, constant);
    m.store(Width::U64, Reg::T1, Reg::T2, fixed(0));
    m.store(Width::U64, Reg::T3, Reg::T2, fixed(i64::from(PAYLOAD)));
    m.jump(Target::Ext(ext::NEXT));
    m.bind(other);
}

fn find_string_node(m: &mut dyn Masm, env: &Env, (table, key): (u8, u8), missing: Target) {
    let layout = layout(env);
    let walk = m.new_label();
    let follow = m.new_label();
    let found = m.new_label();
    m.load(Width::U8, Reg::T1, Reg::Base, reg(key));
    m.cmp_imm(Reg::T1, STRING);
    m.branch(Cond::Ne, slow());
    table_in_register(m, env, table);
    m.load(Width::U8, Reg::T1, Reg::T0, fixed(i64::from(layout.shaped)));
    m.cmp_imm(Reg::T1, 0);
    m.branch(Cond::Ne, slow());
    m.load(Width::U32, Reg::T4, Reg::Base, payload(key));
    m.shl_imm(Reg::T4, 4);
    m.load(Width::U64, Reg::T5, Reg::Ctx, context(offset_of!(JitCtx, vm)));
    m.load(Width::U64, Reg::T5, Reg::T5, fixed(i64::from(layout.string_keys)));
    m.alu(Alu::Add, Reg::T5, Reg::T4);
    m.load(Width::U8, Reg::T1, Reg::T5, fixed(STRING_KEY_INTERNED));
    m.cmp_imm(Reg::T1, 0);
    m.branch(Cond::Eq, slow());
    m.load(Width::U64, Reg::T2, Reg::T0, fixed(i64::from(layout.nodes_len)));
    m.alu_imm(Alu::Sub, Reg::T2, 1);
    m.load(Width::U64, Reg::T1, Reg::T5, fixed(STRING_KEY_HASH));
    m.alu(Alu::And, Reg::T1, Reg::T2);
    m.load(Width::U64, Reg::T3, Reg::Base, payload(key));
    m.load(Width::U64, Reg::T2, Reg::T0, fixed(i64::from(layout.nodes)));
    m.bind(walk);
    node_address(m);
    m.load(Width::U8, Reg::T4, Reg::T5, fixed(NODE_KEY));
    m.cmp_imm(Reg::T4, STRING);
    m.branch(Cond::Ne, follow);
    m.load(Width::U64, Reg::T4, Reg::T5, fixed(NODE_KEY + i64::from(PAYLOAD)));
    m.cmp(Reg::T4, Reg::T3);
    m.branch(Cond::Ne, follow);
    m.load(Width::U8, Reg::T4, Reg::T5, fixed(NODE_DEAD));
    m.cmp_imm(Reg::T4, 0);
    m.branch(Cond::Eq, found);
    m.bind(follow);
    m.load(Width::U32, Reg::T1, Reg::T5, fixed(NODE_NEXT));
    m.cmp_imm(Reg::T1, 0);
    m.branch(Cond::Eq, missing);
    m.alu_imm(Alu::Sub, Reg::T1, 1);
    m.jump(walk);
    m.bind(found);
}

fn node_address(m: &mut dyn Masm) {
    m.mov(Reg::T5, Reg::T1);
    m.shl_imm(Reg::T5, 1);
    m.alu(Alu::Add, Reg::T5, Reg::T1);
    m.shl_imm(Reg::T5, 4);
    m.alu(Alu::Add, Reg::T5, Reg::T2);
}

fn hash_get(m: &mut dyn Masm, env: &Env) {
    let layout = layout(env);
    let missing = m.new_label();
    find_string_node(m, env, (slot::B, slot::C), missing);
    m.load(Width::U64, Reg::T1, Reg::T5, fixed(NODE_VALUE));
    m.load(Width::U64, Reg::T3, Reg::T5, fixed(NODE_VALUE + i64::from(PAYLOAD)));
    m.mov(Reg::T4, Reg::T1);
    m.alu_imm(Alu::And, Reg::T4, 0xff);
    m.cmp_imm(Reg::T4, NIL);
    m.branch(Cond::Eq, missing);
    m.store(Width::U64, Reg::T1, Reg::Base, reg(slot::A));
    m.store(Width::U64, Reg::T3, Reg::Base, payload(slot::A));
    m.jump(Target::Ext(ext::NEXT));
    m.bind(missing);
    m.load(Width::U32, Reg::T2, Reg::T0, fixed(i64::from(layout.metatable_tag)));
    m.mov_imm(Reg::T3, fixed(i64::from(layout.metatable_none)));
    m.cmp(Reg::T2, Reg::T3);
    m.branch(Cond::Ne, slow());
    m.store_imm(Width::U8, Reg::Base, reg(slot::A), NIL);
    m.jump(Target::Ext(ext::NEXT));
}

fn hash_set(m: &mut dyn Masm, env: &Env, constant_value: bool) {
    find_string_node(m, env, (slot::A, slot::B), slow());
    m.load(Width::U8, Reg::T1, Reg::T5, fixed(NODE_VALUE));
    m.cmp_imm(Reg::T1, NIL);
    m.branch(Cond::Eq, slow());
    m.mov(Reg::T2, Reg::T5);
    store_value(m, constant_value);
    m.store(Width::U64, Reg::T1, Reg::T2, fixed(NODE_VALUE));
    m.store(Width::U64, Reg::T3, Reg::T2, fixed(NODE_VALUE + i64::from(PAYLOAD)));
    barrier(m, slot::A);
}

/// The common exact count falls through; other counts take `TARGET`, the cold `ReturnWide`.
fn return_check(m: &mut dyn Masm) {
    let exact = m.new_label();
    m.load(Width::U32, Reg::T0, Reg::Ctx, context(offset_of!(JitCtx, direct_results)));
    m.mov_imm(Reg::T1, reg(slot::I));
    m.cmp(Reg::T0, Reg::T1);
    m.branch(Cond::Eq, exact);
    m.cmp_imm(Reg::T0, 0);
    m.branch(Cond::Eq, Target::Ext(ext::TRUE));
    m.jump(Target::Ext(ext::TARGET));
    m.bind(exact);
    m.load(Width::U64, Reg::T2, Reg::Ctx, context(offset_of!(JitCtx, ret_offset)));
    m.alu(Alu::Add, Reg::T2, Reg::Base);
}

fn return_wide(m: &mut dyn Masm, env: &Env) {
    let place = m.new_label();
    m.cmp_imm(Reg::T0, i32::from(u8::MAX));
    m.branch(Cond::BelowEq, place);
    if env.frames.is_none() {
        m.jump(slow());
    } else {
        m.mov_imm(Reg::T1, fixed(i64::from(u32::MAX)));
        m.cmp(Reg::T0, Reg::T1);
        m.branch(Cond::Ne, slow());
    }
    m.bind(place);
    m.load(Width::U64, Reg::T2, Reg::Ctx, context(offset_of!(JitCtx, ret_offset)));
    m.alu(Alu::Add, Reg::T2, Reg::Base);
}

/// Follows `moveresults` in ldo.c: missing results become nil, and a caller taking all of them gets `top` after the last.
fn return_fill(m: &mut dyn Masm, env: &Env) {
    let fill = m.new_label();
    m.load(Width::U32, Reg::T0, Reg::Ctx, context(offset_of!(JitCtx, direct_results)));
    m.mov_imm(Reg::T1, reg(slot::I));
    m.cmp(Reg::T0, Reg::T1);
    m.branch(Cond::Eq, Target::Ext(ext::TRUE));
    if let Some(frames) = env.frames {
        m.mov_imm(Reg::T3, fixed(i64::from(u32::MAX)));
        m.cmp(Reg::T0, Reg::T3);
        m.branch(Cond::Ne, fill);
        m.load(Width::U64, Reg::T3, Reg::Ctx, context(offset_of!(JitCtx, base)));
        m.alu(Alu::Add, Reg::T3, Reg::T1);
        m.alu_imm(Alu::Sub, Reg::T3, 1);
        m.load(Width::U64, Reg::T4, Reg::Ctx, context(offset_of!(JitCtx, thread)));
        m.store(Width::U64, Reg::T3, Reg::T4, fixed(i64::from(frames.top)));
        m.jump(Target::Ext(ext::TRUE));
    }
    m.bind(fill);
    m.cmp(Reg::T1, Reg::T0);
    m.branch(Cond::AboveEq, Target::Ext(ext::TRUE));
    m.mov(Reg::T3, Reg::T1);
    m.shl_imm(Reg::T3, 4);
    m.alu(Alu::Add, Reg::T3, Reg::T2);
    m.store_imm(Width::U8, Reg::T3, fixed(0), NIL);
    m.alu_imm(Alu::Add, Reg::T1, 1);
    m.jump(fill);
}
