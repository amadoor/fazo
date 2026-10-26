use super::{JitCtx, helpers};
use crate::{
    instruction::Instruction,
    proto::Proto,
    value::Value as LuaValue,
    vm::ic::{Entry, InlineCache, MethodEntry},
};

pub(crate) const STEP: &str = "fazo_jit_step";
const GET_FIELD: &str = "fazo_jit_get_field";
const GET_TABLE: &str = "fazo_jit_get_table";
const GET_INT: &str = "fazo_jit_get_int";
const GET_SELF: &str = "fazo_jit_get_self";
const SET_FIELD: &str = "fazo_jit_set_field";
const SET_TABLE: &str = "fazo_jit_set_table";
const SET_INT: &str = "fazo_jit_set_int";
const CALL_DIRECT: &str = "fazo_jit_call_direct";
const RETURN: &str = "fazo_jit_return";
const FOR_PREP: &str = "fazo_jit_for_prep";
const ARITH: &str = "fazo_jit_arith";
const GET_TABUP: &str = "fazo_jit_get_tabup";
const GET_UPVAL: &str = "fazo_jit_get_upval";
const SET_UPVAL: &str = "fazo_jit_set_upval";
const TAIL_CALL: &str = "fazo_jit_tail_call";
const TABLE_BARRIER: &str = "fazo_jit_table_barrier";
const AFTER_CALL: &str = "fazo_jit_after_call";
const NEW_TABLE: &str = "fazo_jit_new_table";
const CLOSURE: &str = "fazo_jit_closure";
const CLOSE: &str = "fazo_jit_close";
const SET_LIST: &str = "fazo_jit_set_list";

/// Indexes match the function references in `Helpers::fast`.
pub(crate) const FAST_NAMES: [&str; 21] = [
    GET_FIELD, GET_TABLE, GET_INT, GET_SELF, SET_FIELD, SET_TABLE, SET_INT, CALL_DIRECT, RETURN, FOR_PREP, GET_TABUP, ARITH, GET_UPVAL,
    SET_UPVAL, TAIL_CALL, TABLE_BARRIER, AFTER_CALL, NEW_TABLE, CLOSURE, CLOSE, SET_LIST,
];
pub(crate) const FAST_GET_FIELD: usize = 0;
pub(crate) const FAST_GET_TABLE: usize = 1;
pub(crate) const FAST_GET_INT: usize = 2;
pub(crate) const FAST_GET_SELF: usize = 3;
pub(crate) const FAST_SET_FIELD: usize = 4;
pub(crate) const FAST_SET_TABLE: usize = 5;
pub(crate) const FAST_SET_INT: usize = 6;
pub(crate) const FAST_CALL: usize = 7;
pub(crate) const FAST_RETURN: usize = 8;
pub(crate) const FAST_FOR_PREP: usize = 9;
pub(crate) const FAST_GET_TABUP: usize = 10;
pub(crate) const FAST_ARITH: usize = 11;
pub(crate) const FAST_GET_UPVAL: usize = 12;
pub(crate) const FAST_SET_UPVAL: usize = 13;
pub(crate) const FAST_TAIL_CALL: usize = 14;
pub(crate) const FAST_TABLE_BARRIER: usize = 15;
pub(crate) const FAST_AFTER_CALL: usize = 16;
pub(crate) const FAST_NEW_TABLE: usize = 17;
pub(crate) const FAST_CLOSURE: usize = 18;
pub(crate) const FAST_CLOSE: usize = 19;
pub(crate) const FAST_SET_LIST: usize = 20;

pub(crate) type FastFn = extern "C" fn(*mut JitCtx, u32, u32, u32, u32) -> i64;

pub(crate) fn fast_symbols() -> [(&'static str, *const u8); 21] {
    [
        (GET_FIELD, helpers::get_field as FastFn as *const u8),
        (GET_TABLE, helpers::get_table as FastFn as *const u8),
        (GET_INT, helpers::get_int as FastFn as *const u8),
        (GET_SELF, helpers::get_self as FastFn as *const u8),
        (SET_FIELD, helpers::set_field as FastFn as *const u8),
        (SET_TABLE, helpers::set_table as FastFn as *const u8),
        (SET_INT, helpers::set_int as FastFn as *const u8),
        (CALL_DIRECT, helpers::call_direct as FastFn as *const u8),
        (RETURN, helpers::ret as FastFn as *const u8),
        (FOR_PREP, helpers::for_prep as FastFn as *const u8),
        (GET_TABUP, helpers::get_tabup as FastFn as *const u8),
        (ARITH, helpers::arith as FastFn as *const u8),
        (GET_UPVAL, helpers::get_upval as FastFn as *const u8),
        (SET_UPVAL, helpers::set_upval as FastFn as *const u8),
        (TAIL_CALL, helpers::tail_call as FastFn as *const u8),
        (TABLE_BARRIER, helpers::table_barrier as FastFn as *const u8),
        (AFTER_CALL, helpers::after_call as FastFn as *const u8),
        (NEW_TABLE, helpers::new_table as FastFn as *const u8),
        (CLOSURE, helpers::closure as FastFn as *const u8),
        (CLOSE, helpers::close as FastFn as *const u8),
        (SET_LIST, helpers::set_list as FastFn as *const u8),
    ]
}
pub(crate) const ENTRY: &str = "fazo_jit_entry";


#[derive(Clone, Copy, PartialEq, Eq)]
enum Lowered {
    Inline,
    Helper,
    Call,
    Direct,
    Skipped,
}

/// Costs follow the classes `Lowering::lower` emits.
pub(crate) fn payoff(proto: &Proto, policy: crate::TierPolicy) -> i64 {
    let depths = loop_depths(proto);
    let weight = i64::from(policy.loop_weight.max(1));
    let mut total: i64 = 0;
    for (pc, &instruction) in proto.code.iter().enumerate() {
        let per_run = match lowered_at(proto, pc, instruction) {
            Lowered::Call if matches!(instruction, Instruction::Call { .. }) && calls_lua(proto, pc) => i64::from(policy.call_gain),
            Lowered::Inline | Lowered::Call => i64::from(policy.inline_gain),
            Lowered::Direct => -i64::from(policy.direct_cost),
            Lowered::Helper | Lowered::Skipped => 0,
        };
        let scaled = per_run.saturating_mul(weight.saturating_pow(depths[pc].min(3)));
        total = total.saturating_add(scaled);
    }
    total
}

pub(crate) fn calls_lua(proto: &Proto, pc: usize) -> bool {
    match proto.exec.feedback.get(pc) {
        Some(InlineCache::Callee(callee)) => matches!(callee.value, LuaValue::Closure(_)),
        Some(InlineCache::Callees(callees)) => callees.iter().any(|callee| matches!(callee.value, LuaValue::Closure(_))),
        _ => false,
    }
}

/// Forced compiles meet sites the interpreter never reached; those and Lua sites with no known callee read the callee's facts at run time.
pub(crate) fn calls_unknown_lua(proto: &Proto, pc: usize, forced: bool) -> bool {
    match proto.exec.feedback.get(pc) {
        None | Some(InlineCache::Uninitialized) => forced,
        _ => calls_lua(proto, pc),
    }
}

pub(crate) fn calls_out(proto: &Proto) -> bool {
    let depths = loop_depths(proto);
    proto.code.iter().zip(depths).any(|(instruction, depth)| {
        depth > 0 && matches!(instruction, Instruction::Call { .. } | Instruction::TailCall { .. })
    })
}

pub(crate) fn loop_depths(proto: &Proto) -> Vec<u32> {
    let mut depths = vec![0u32; proto.code.len()];
    for (pc, &instruction) in proto.code.iter().enumerate() {
        let target = match instruction {
            Instruction::ForLoop { bx, .. } | Instruction::TForLoop { bx, .. } => pc + 1 - bx as usize,
            Instruction::Jmp { sj } if sj < 0 => offset_pc(pc + 1, sj),
            _ => continue,
        };
        for depth in &mut depths[target..=pc] {
            *depth += 1;
        }
    }
    depths
}

fn lowered_at(proto: &Proto, pc: usize, instruction: Instruction) -> Lowered {
    match instruction {
        Instruction::GetField { .. } | Instruction::SelfOp { .. } | Instruction::SetField { .. } if site_is_cached(proto, pc) => {
            Lowered::Inline
        }
        _ => lowered(proto, instruction),
    }
}

pub(crate) fn site_is_cached(proto: &Proto, pc: usize) -> bool {
    match proto.exec.feedback.get(pc) {
        Some(InlineCache::Monomorphic(entry)) => !entry.adds_key(),
        Some(InlineCache::Polymorphic(entries)) => entries.iter().any(|entry| !entry.adds_key()),
        Some(InlineCache::Method(_)) => true,
        _ => false,
    }
}

pub(crate) fn field_entries(proto: &Proto, pc: usize) -> Vec<Entry> {
    let entries: &[Entry] = match proto.exec.feedback.get(pc) {
        Some(InlineCache::Monomorphic(entry)) => std::slice::from_ref(entry),
        Some(InlineCache::Polymorphic(entries)) => entries,
        _ => &[],
    };
    entries.iter().filter(|entry| !entry.adds_key()).copied().collect()
}

pub(crate) fn add_entries(proto: &Proto, pc: usize) -> Vec<Entry> {
    let entries: &[Entry] = match proto.exec.feedback.get(pc) {
        Some(InlineCache::Monomorphic(entry)) => std::slice::from_ref(entry),
        Some(InlineCache::Polymorphic(entries)) => entries,
        _ => &[],
    };
    entries.iter().filter(|entry| entry.adds_key()).copied().collect()
}

pub(crate) fn method_entries(proto: &Proto, pc: usize) -> Vec<MethodEntry> {
    all_method_entries(proto, pc).into_iter().filter(|method| method.depth == 0).collect()
}

pub(crate) fn all_method_entries(proto: &Proto, pc: usize) -> Vec<MethodEntry> {
    match proto.exec.feedback.get(pc) {
        Some(InlineCache::Method(methods)) => methods.to_vec(),
        _ => Vec::new(),
    }
}

fn lowered(proto: &Proto, instruction: Instruction) -> Lowered {
    let numeric = |index: u32| matches!(proto.constants.get(index as usize), Some(LuaValue::Integer(_) | LuaValue::Float(_)));
    match instruction {
        Instruction::Move { .. }
        | Instruction::LoadI { .. }
        | Instruction::LoadF { .. }
        | Instruction::LoadK { .. }
        | Instruction::LoadKx { .. }
        | Instruction::LoadFalse { .. }
        | Instruction::LFalseSkip { .. }
        | Instruction::LoadTrue { .. }
        | Instruction::LoadNil { .. }
        | Instruction::Add { .. }
        | Instruction::Sub { .. }
        | Instruction::Mul { .. }
        | Instruction::BAnd { .. }
        | Instruction::BOr { .. }
        | Instruction::BXor { .. }
        | Instruction::Mod { .. }
        | Instruction::IDiv { .. }
        | Instruction::Div { .. }
        | Instruction::AddI { .. }
        | Instruction::Unm { .. }
        | Instruction::BNot { .. }
        | Instruction::Not { .. }
        | Instruction::Jmp { .. }
        | Instruction::Test { .. }
        | Instruction::TestSet { .. }
        | Instruction::Eq { .. }
        | Instruction::Lt { .. }
        | Instruction::Le { .. }
        | Instruction::EqI { .. }
        | Instruction::LtI { .. }
        | Instruction::LeI { .. }
        | Instruction::GtI { .. }
        | Instruction::GeI { .. }
        | Instruction::ForLoop { .. }
        | Instruction::TForLoop { .. }
        | Instruction::GetTable { .. }
        | Instruction::GetI { .. }
        | Instruction::SetTable { .. }
        | Instruction::SetI { .. } => Lowered::Inline,
        Instruction::AddK { c, .. }
        | Instruction::SubK { c, .. }
        | Instruction::MulK { c, .. }
        | Instruction::BAndK { c, .. }
        | Instruction::BOrK { c, .. }
        | Instruction::BXorK { c, .. }
        | Instruction::ModK { c, .. }
        | Instruction::IDivK { c, .. }
        | Instruction::DivK { c, .. } => {
            if numeric(u32::from(c)) { Lowered::Inline } else { Lowered::Direct }
        }
        Instruction::EqK { b, .. } => match proto.constants.get(usize::from(b)) {
            Some(LuaValue::Integer(_) | LuaValue::Nil | LuaValue::Boolean(_) | LuaValue::String(_)) => Lowered::Inline,
            _ => Lowered::Direct,
        },
        Instruction::GetTabUp { .. }
        | Instruction::GetField { .. }
        | Instruction::SelfOp { .. }
        | Instruction::SetField { .. }
        | Instruction::GetUpval { .. }
        | Instruction::SetUpval { .. }
        | Instruction::Len { .. }
        | Instruction::Pow { .. }
        | Instruction::PowK { .. }
        | Instruction::Shl { .. }
        | Instruction::Shr { .. }
        | Instruction::ShlI { .. }
        | Instruction::ShrI { .. }
        | Instruction::ForPrep { .. } => Lowered::Helper,
        Instruction::TailCall { .. } => Lowered::Helper,
        Instruction::Call { .. }
        | Instruction::Return0 { .. }
        | Instruction::Return1 { .. }
        | Instruction::Return { k: false, .. } => Lowered::Call,
        Instruction::NewTable { .. } | Instruction::Closure { .. } => Lowered::Helper,
        Instruction::Concat { .. }
        | Instruction::Close { .. }
        | Instruction::SetList { .. }
        | Instruction::Vararg { .. }
        | Instruction::TForPrep { .. }
        | Instruction::TForCall { .. }
        | Instruction::VarargPrep { .. }
        | Instruction::Return { k: true, .. }
        | Instruction::SetTabUp { .. }
        | Instruction::GetVarg { .. }
        | Instruction::Tbc { .. }
        | Instruction::ErrNil { .. } => Lowered::Direct,
        Instruction::MmBin { .. } | Instruction::MmBinI { .. } | Instruction::MmBinK { .. } | Instruction::ExtraArg { .. } => {
            Lowered::Skipped
        }
    }
}

/// Read after the prototype check; a live closure keeps its prototype, so the fields stay valid.
#[derive(Clone, Copy)]
pub(crate) struct CallTarget {
    pub(crate) closure: u64,
    pub(crate) proto: u64,
    pub(crate) entry: *const std::cell::Cell<usize>,
    pub(crate) constants: *const LuaValue,
    pub(crate) num_params: u8,
    pub(crate) max_stack: u8,
    pub(crate) has_upvalues: bool,
}

pub(crate) fn offset_pc(from: usize, offset: i32) -> usize {
    (from as isize + offset as isize) as usize
}
