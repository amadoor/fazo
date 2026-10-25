use std::{mem::offset_of, sync::Arc};

use cranelift_codegen::ir::{
    AbiParam, AliasRegionData, Block, BlockArg, FuncRef, Function, InstBuilder, JumpTableData, MemFlagsData, SigRef,
    StackSlot, StackSlotData, StackSlotKind, TrapCode, Type, Value,
    condcodes::{FloatCC, IntCC},
    types,
};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::JITModule;
use cranelift_module::{FuncId, Linkage, Module};

use super::{
    FrameLayout, JitCtx, JitResult, TableLayout, helpers,
    layout::{lua_frame, upvalue},
    memory::CodeBudget,
    unit::{Compiler, JitUnit},
};
use crate::{
    instruction::{ArithOp, Instruction},
    proto::Proto,
    value::Value as LuaValue,
    vm::{
        Event,
        ic::{ABSENT, Entry, InlineCache, MethodEntry},
    },
};

const CONSTANT_BIT: u32 = 1 << 31;
const IMMEDIATE_BIT: u32 = 1 << 30;

const STEP: &str = "fazo_jit_step";
const SAFE_POINT: &str = "fazo_jit_safe_point";
const MISC: &str = "fazo_jit_misc";
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

/// Indexes match the function references in `Helpers::fast`.
const FAST_NAMES: [&str; 19] = [
    GET_FIELD, GET_TABLE, GET_INT, GET_SELF, SET_FIELD, SET_TABLE, SET_INT, CALL_DIRECT, RETURN, FOR_PREP, GET_TABUP, ARITH, GET_UPVAL,
    SET_UPVAL, TAIL_CALL, TABLE_BARRIER, AFTER_CALL, NEW_TABLE, CLOSURE,
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

type FastFn = extern "C" fn(*mut JitCtx, u32, u32, u32, u32) -> i64;

pub(crate) fn fast_symbols() -> [(&'static str, *const u8); 19] {
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
    ]
}
const ENTRY: &str = "fazo_jit_entry";

const NIL_TAG: i64 = 0;
const BOOL_TAG: i64 = 1;
const INT_TAG: i64 = 2;
const FLOAT_TAG: i64 = 3;
const VALUE_SIZE: i32 = 16;
const PAYLOAD_OFFSET: i32 = 8;
const BOOL_OFFSET: i32 = PAYLOAD_OFFSET;
const STRING_TAG: i64 = 4;
const TABLE_TAG: i64 = 5;
const CLOSURE_TAG: i64 = 6;
const LIGHT_USERDATA_TAG: i64 = 10;

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

fn calls_lua(proto: &Proto, pc: usize) -> bool {
    match proto.exec.feedback.get(pc) {
        Some(InlineCache::Callee(callee)) => matches!(callee.value, LuaValue::Closure(_)),
        Some(InlineCache::Callees(callees)) => callees.iter().any(|callee| matches!(callee.value, LuaValue::Closure(_))),
        _ => false,
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

fn site_is_cached(proto: &Proto, pc: usize) -> bool {
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

/// Compiled code reads the fields after the closure's prototype compares equal; a live closure keeps its prototype, so they stay valid.
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

pub(crate) struct CompileFacts {
    pub(crate) layout: Option<TableLayout>,
    pub(crate) frames: Option<FrameLayout>,
    pub(crate) calls: Vec<Vec<CallTarget>>,
    pub(crate) call_depth_limit: u32,
    /// Interned strings compare by identity, so equal constants need no content check.
    pub(crate) interned: Vec<bool>,
    pub(crate) inline_caches: bool,
}

/// Each instruction starts a block, so the entry can resume at any instruction.
pub(crate) fn compile(compiler: &mut Compiler, budget: Arc<CodeBudget>, proto: &Proto, facts: &CompileFacts) -> JitResult<JitUnit> {
    let symbols = [
        (STEP, helpers::step as extern "C" fn(*mut JitCtx, u32) -> i64 as *const u8),
        (SAFE_POINT, helpers::safe_point as extern "C" fn(*mut JitCtx, u32) -> i64 as *const u8),
        (MISC, helpers::misc as extern "C" fn(*mut JitCtx, u32) -> i64 as *const u8),
    ];
    let symbols: Vec<(&str, *const u8)> = symbols.into_iter().chain(fast_symbols()).collect();
    JitUnit::compile(compiler, budget, &symbols, |module, func, context| define(module, func, context, proto, facts))
}

fn define(
    module: &mut JITModule,
    func: &mut Function,
    context: &mut FunctionBuilderContext,
    proto: &Proto,
    facts: &CompileFacts,
) -> JitResult<FuncId> {
    let config = module.target_config();
    let pointer = config.pointer_type();
    let mut signature = module.make_signature();
    signature.params.push(AbiParam::new(pointer));
    signature.params.push(AbiParam::new(types::I32));
    signature.returns.push(AbiParam::new(types::I64));
    func.signature = signature.clone();
    let id = module.declare_function(ENTRY, Linkage::Export, &signature)?;
    let step_id = module.declare_function(STEP, Linkage::Import, &signature)?;
    let safe_point_id = module.declare_function(SAFE_POINT, Linkage::Import, &signature)?;
    let misc_id = module.declare_function(MISC, Linkage::Import, &signature)?;
    let mut fast_signature = module.make_signature();
    fast_signature.params.push(AbiParam::new(pointer));
    fast_signature.params.extend([AbiParam::new(types::I32); 4]);
    fast_signature.returns.push(AbiParam::new(types::I64));
    let mut fast = Vec::with_capacity(FAST_NAMES.len());
    for name in FAST_NAMES {
        let id = module.declare_function(name, Linkage::Import, &fast_signature)?;
        fast.push(module.declare_func_in_func(id, func));
    }
    let helpers = Helpers {
        step: module.declare_func_in_func(step_id, func),
        safe_point: module.declare_func_in_func(safe_point_id, func),
        misc: module.declare_func_in_func(misc_id, func),
        fast,
    };

    let mut builder = FunctionBuilder::new(func, context);
    let entry_signature = builder.import_signature(signature);
    let mut lowering = Lowering::new(builder, proto, pointer, helpers);
    lowering.layout = facts.layout;
    lowering.interned = &facts.interned;
    lowering.caches = facts.inline_caches;
    lowering.direct = facts.frames.map(|frames| DirectCalls {
        frames,
        calls: &facts.calls,
        signature: entry_signature,
        depth_limit: facts.call_depth_limit,
        context: None,
    });
    lowering.lower_all();
    lowering.builder.finalize(config);
    Ok(id)
}

#[derive(Clone, Copy)]
enum Operand {
    Register(u8),
    Int(i64),
    Float(f64),
}

#[derive(Clone, Copy)]
enum ArrayKey {
    Register(u8),
    Constant(u8),
}

impl ArrayKey {
    fn operand(self) -> u32 {
        match self {
            Self::Register(index) | Self::Constant(index) => index.into(),
        }
    }

    fn get_helper(self) -> usize {
        match self {
            Self::Register(_) => FAST_GET_TABLE,
            Self::Constant(_) => FAST_GET_INT,
        }
    }

    fn set_helper(self) -> usize {
        match self {
            Self::Register(_) => FAST_SET_TABLE,
            Self::Constant(_) => FAST_SET_INT,
        }
    }
}

#[derive(Clone, Copy)]
enum Test {
    Always,
    Never,
    When(Value),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Arith {
    Add,
    Sub,
    Mul,
    BAnd,
    BOr,
    BXor,
}

impl Arith {
    fn has_float(self) -> bool {
        matches!(self, Arith::Add | Arith::Sub | Arith::Mul)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Path {
    Int,
    Float,
    Mixed,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Division {
    Mod,
    IDiv,
}

#[derive(Clone, Copy)]
struct Compare {
    int: Option<(Operand, Operand)>,
    float: Option<(Operand, Operand)>,
    int_cc: IntCC,
    float_cc: FloatCC,
}

#[derive(Clone)]
struct Helpers {
    step: FuncRef,
    safe_point: FuncRef,
    misc: FuncRef,
    fast: Vec<FuncRef>,
}

struct Lowering<'a> {
    builder: FunctionBuilder<'a>,
    proto: &'a Proto,
    blocks: Vec<Block>,
    trap: Block,
    dispatch: Block,
    context_flags: MemFlagsData,
    value_flags: MemFlagsData,
    heap_flags: MemFlagsData,
    frame_flags: MemFlagsData,
    layout: Option<TableLayout>,
    direct: Option<DirectCalls<'a>>,
    interned: &'a [bool],
    caches: bool,
    ctx: Value,
    pointer: Type,
    helpers: Helpers,
}

struct DirectCalls<'a> {
    frames: FrameLayout,
    calls: &'a [Vec<CallTarget>],
    signature: SigRef,
    depth_limit: u32,
    context: Option<StackSlot>,
}

/// Lua values are never trapped on, and context and Lua value accesses are separate alias regions, so the optimizer reuses context loads across register stores.
fn alias_flags(builder: &mut FunctionBuilder<'_>, user_id: u32, description: &'static str) -> MemFlagsData {
    let region = builder.func.dfg.alias_regions.insert(AliasRegionData {
        user_id,
        description: description.into(),
    });
    MemFlagsData::new().with_notrap().with_alias_region(Some(region))
}

fn slot(register: u8) -> i32 {
    i32::from(register) * VALUE_SIZE
}

fn constant_slot(index: u32) -> i32 {
    index as i32 * VALUE_SIZE
}

fn offset_pc(from: usize, offset: i32) -> usize {
    (from as isize + offset as isize) as usize
}

impl Lowering<'_> {
    fn arith_helper(&mut self, pc: usize, target: u8, op: crate::instruction::ArithOp, left: u32, right: u32) {
        let packed = u32::from(target) | ((op as u32) << 16);
        self.fast_helper(pc, FAST_ARITH, [packed, left, right], pc + 2);
    }
}

fn operand_index(register: u8, k: bool) -> u32 {
    if k { u32::from(register) | helpers::CONSTANT_OPERAND } else { u32::from(register) }
}

fn ordered(register: Operand, other: Operand, reversed: bool) -> (Operand, Operand) {
    if reversed { (other, register) } else { (register, other) }
}

impl<'a> Lowering<'a> {
    fn new(mut builder: FunctionBuilder<'a>, proto: &'a Proto, pointer: Type, helpers: Helpers) -> Self {
        let entry = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);
        let ctx = builder.block_params(entry)[0];
        let blocks: Vec<Block> = (0..proto.code.len()).map(|_| builder.create_block()).collect();
        let trap = builder.create_block();
        let dispatch = builder.create_block();
        builder.append_block_param(dispatch, types::I64);
        let context_flags = alias_flags(&mut builder, 0, "context");
        let value_flags = alias_flags(&mut builder, 1, "lua values");
        let heap_flags = alias_flags(&mut builder, 2, "tables");
        let frame_flags = alias_flags(&mut builder, 3, "threads and frames");
        let entry_pc = builder.block_params(entry)[1];
        let entry_pc = builder.ins().uextend(types::I64, entry_pc);
        let at_start = builder.ins().icmp_imm_s(IntCC::Equal, entry_pc, 0);
        builder.ins().brif(at_start, blocks[0], &[], dispatch, &[BlockArg::Value(entry_pc)]);

        Self {
            builder,
            proto,
            blocks,
            trap,
            dispatch,
            context_flags,
            value_flags,
            heap_flags,
            frame_flags,
            layout: None,
            direct: None,
            interned: &[],
            caches: true,
            ctx,
            pointer,
            helpers,
        }
    }

    fn field_entries(&self, pc: usize) -> Vec<Entry> {
        if !self.caches {
            return Vec::new();
        }
        field_entries(self.proto, pc)
    }

    fn method_entries(&self, pc: usize) -> Vec<MethodEntry> {
        if !self.caches {
            return Vec::new();
        }
        method_entries(self.proto, pc)
    }

    fn lower_all(&mut self) {
        for pc in 0..self.proto.code.len() {
            self.builder.switch_to_block(self.blocks[pc]);
            self.lower(pc, self.proto.code[pc]);
        }
        self.lower_dispatch();
        self.builder.switch_to_block(self.trap);
        self.builder.ins().trap(TrapCode::unwrap_user(1));
        self.builder.seal_all_blocks();
    }

    fn lower_dispatch(&mut self) {
        self.builder.switch_to_block(self.dispatch);
        let result = self.builder.block_params(self.dispatch)[0];
        let exit = self.builder.create_block();
        let table = self.builder.create_block();
        let leaving = self.builder.ins().icmp_imm_s(IntCC::SignedLessThan, result, 0);
        self.builder.ins().brif(leaving, exit, &[], table, &[]);
        self.builder.switch_to_block(exit);
        let code = self.builder.ins().ineg(result);
        self.builder.ins().return_(&[code]);
        self.builder.switch_to_block(table);
        let in_range = self.builder.ins().icmp_imm_u(IntCC::UnsignedLessThan, result, self.blocks.len() as i64);
        let index = self.builder.ins().ireduce(types::I32, result);
        let jump = self.builder.create_block();
        self.builder.ins().brif(in_range, jump, &[], self.trap, &[]);
        self.builder.switch_to_block(jump);
        let default = self.builder.func.dfg.block_call(self.trap, &[]);
        let targets: Vec<_> = self.blocks.iter().map(|&block| self.builder.func.dfg.block_call(block, &[])).collect();
        let table = self.builder.create_jump_table(JumpTableData::new(default, &targets));
        self.builder.ins().br_table(index, table);
    }

    /// The stack may move, so the register window is read from the context at each use.
    fn registers(&mut self) -> Value {
        self.context_pointer(offset_of!(JitCtx, registers))
    }

    fn constants(&mut self) -> Value {
        self.context_pointer(offset_of!(JitCtx, constants))
    }

    fn context_pointer(&mut self, offset: usize) -> Value {
        self.builder.ins().load(self.pointer, self.context_flags, self.ctx, offset as i32)
    }

    fn jump_to(&mut self, pc: usize) {
        let block = self.blocks[pc];
        self.builder.ins().jump(block, &[]);
    }

    fn lower(&mut self, pc: usize, instruction: Instruction) {
        match instruction {
            Instruction::Move { a, b } => {
                let base = self.registers();
                self.copy_value(base, slot(b), base, slot(a));
                self.jump_to(pc + 1);
            }
            Instruction::LoadI { a, sbx } => {
                let base = self.registers();
                let value = self.builder.ins().iconst(types::I64, i64::from(sbx));
                self.store_int(base, slot(a), value);
                self.jump_to(pc + 1);
            }
            Instruction::LoadF { a, sbx } => {
                let base = self.registers();
                let value = self.builder.ins().f64const(f64::from(sbx));
                self.store_float(base, slot(a), value);
                self.jump_to(pc + 1);
            }
            Instruction::LoadK { a, bx } => self.load_constant(pc, a, bx, pc + 1),
            Instruction::LoadKx { a } => match self.proto.code.get(pc + 1) {
                Some(Instruction::ExtraArg { ax }) => {
                    let index = *ax;
                    self.load_constant(pc, a, index, pc + 2);
                }
                _ => self.call_step(pc),
            },
            Instruction::LoadFalse { a } => {
                self.store_bool_constant(a, false);
                self.jump_to(pc + 1);
            }
            Instruction::LFalseSkip { a } => {
                self.store_bool_constant(a, false);
                self.jump_to(pc + 2);
            }
            Instruction::LoadTrue { a } => {
                self.store_bool_constant(a, true);
                self.jump_to(pc + 1);
            }
            Instruction::LoadNil { a, b } => {
                let base = self.registers();
                let nil = self.builder.ins().iconst(types::I8, NIL_TAG);
                for register in a..=a + b {
                    self.builder.ins().store(self.value_flags, nil, base, slot(register));
                }
                self.jump_to(pc + 1);
            }
            Instruction::Add { a, b, c } => self.arithmetic(pc, a, Operand::Register(b), Operand::Register(c), Arith::Add),
            Instruction::Sub { a, b, c } => self.arithmetic(pc, a, Operand::Register(b), Operand::Register(c), Arith::Sub),
            Instruction::Mul { a, b, c } => self.arithmetic(pc, a, Operand::Register(b), Operand::Register(c), Arith::Mul),
            Instruction::BAnd { a, b, c } => self.arithmetic(pc, a, Operand::Register(b), Operand::Register(c), Arith::BAnd),
            Instruction::BOr { a, b, c } => self.arithmetic(pc, a, Operand::Register(b), Operand::Register(c), Arith::BOr),
            Instruction::BXor { a, b, c } => self.arithmetic(pc, a, Operand::Register(b), Operand::Register(c), Arith::BXor),
            Instruction::AddK { a, b, c } => self.arithmetic_constant(pc, a, b, c, Arith::Add),
            Instruction::SubK { a, b, c } => self.arithmetic_constant(pc, a, b, c, Arith::Sub),
            Instruction::MulK { a, b, c } => self.arithmetic_constant(pc, a, b, c, Arith::Mul),
            Instruction::BAndK { a, b, c } => self.arithmetic_constant(pc, a, b, c, Arith::BAnd),
            Instruction::BOrK { a, b, c } => self.arithmetic_constant(pc, a, b, c, Arith::BOr),
            Instruction::BXorK { a, b, c } => self.arithmetic_constant(pc, a, b, c, Arith::BXor),
            Instruction::Mod { a, b, c } => self.integer_division(pc, a, Operand::Register(b), Operand::Register(c), Division::Mod),
            Instruction::ModK { a, b, c } => self.integer_division_constant(pc, a, b, c, Division::Mod),
            Instruction::IDiv { a, b, c } => self.integer_division(pc, a, Operand::Register(b), Operand::Register(c), Division::IDiv),
            Instruction::IDivK { a, b, c } => self.integer_division_constant(pc, a, b, c, Division::IDiv),
            Instruction::Div { a, b, c } => self.divide(pc, a, Operand::Register(b), Operand::Register(c)),
            Instruction::DivK { a, b, c } => match self.constant_operand(u32::from(c)) {
                Some(constant) => self.divide(pc, a, Operand::Register(b), constant),
                None => self.call_step(pc),
            },
            Instruction::AddI { a, b, sc } => self.arithmetic(pc, a, Operand::Register(b), Operand::Int(i64::from(sc)), Arith::Add),
            Instruction::Unm { a, b } => self.negate(pc, a, b),
            Instruction::BNot { a, b } => self.bitwise_not(pc, a, b),
            Instruction::Not { a, b } => {
                let base = self.registers();
                let falsy = self.falsy(base, slot(b));
                self.store_bool(base, slot(a), falsy);
                self.jump_to(pc + 1);
            }
            Instruction::Jmp { sj } => self.edge(offset_pc(pc + 1, sj), sj < 0),
            Instruction::Test { a, k } => {
                let base = self.registers();
                let truthy = self.truthy(base, slot(a));
                self.branch_on(pc, truthy, k, None);
            }
            Instruction::TestSet { a, b, k } => {
                let base = self.registers();
                let truthy = self.truthy(base, slot(b));
                let copy = self.builder.create_block();
                self.branch_on(pc, truthy, k, Some(copy));
                self.builder.switch_to_block(copy);
                let base = self.registers();
                self.copy_value(base, slot(b), base, slot(a));
                match self.test_target(pc) {
                    Some(target) => self.jump_to(target),
                    None => self.call_step(pc),
                }
            }
            Instruction::Eq { a, b, k } => {
                let compare = Compare {
                    int: Some((Operand::Register(a), Operand::Register(b))),
                    float: Some((Operand::Register(a), Operand::Register(b))),
                    int_cc: IntCC::Equal,
                    float_cc: FloatCC::Equal,
                };
                self.compare(pc, k, compare, None);
            }
            Instruction::Lt { a, b, k } => {
                let compare = Compare {
                    int: Some((Operand::Register(a), Operand::Register(b))),
                    float: Some((Operand::Register(a), Operand::Register(b))),
                    int_cc: IntCC::SignedLessThan,
                    float_cc: FloatCC::LessThan,
                };
                self.compare(pc, k, compare, None);
            }
            Instruction::Le { a, b, k } => {
                let compare = Compare {
                    int: Some((Operand::Register(a), Operand::Register(b))),
                    float: Some((Operand::Register(a), Operand::Register(b))),
                    int_cc: IntCC::SignedLessThanOrEqual,
                    float_cc: FloatCC::LessThanOrEqual,
                };
                self.compare(pc, k, compare, None);
            }
            Instruction::EqK { a, b, k } => match self.constant_operand(u32::from(b)) {
                Some(constant @ Operand::Int(_)) => {
                    let compare = Compare {
                        int: Some((Operand::Register(a), constant)),
                        float: None,
                        int_cc: IntCC::Equal,
                        float_cc: FloatCC::Equal,
                    };
                    self.compare(pc, k, compare, None);
                }
                _ => self.equal_constant(pc, a, b, k),
            },
            Instruction::EqI { a, sb, k, .. } => {
                let compare = Compare {
                    int: Some((Operand::Register(a), Operand::Int(i64::from(sb)))),
                    float: Some((Operand::Register(a), Operand::Float(f64::from(sb)))),
                    int_cc: IntCC::Equal,
                    float_cc: FloatCC::Equal,
                };
                self.compare(pc, k, compare, Some(false));
            }
            Instruction::LtI { a, sb, c, k } => {
                let compare = immediate_compare(a, sb, c, false, IntCC::SignedLessThan, FloatCC::LessThan);
                self.compare(pc, k, compare, None);
            }
            Instruction::LeI { a, sb, c, k } => {
                let compare = immediate_compare(a, sb, c, false, IntCC::SignedLessThanOrEqual, FloatCC::LessThanOrEqual);
                self.compare(pc, k, compare, None);
            }
            Instruction::GtI { a, sb, c, k } => {
                let compare = immediate_compare(a, sb, c, true, IntCC::SignedLessThan, FloatCC::LessThan);
                self.compare(pc, k, compare, None);
            }
            Instruction::GeI { a, sb, c, k } => {
                let compare = immediate_compare(a, sb, c, true, IntCC::SignedLessThanOrEqual, FloatCC::LessThanOrEqual);
                self.compare(pc, k, compare, None);
            }
            Instruction::GetTabUp { a, b, c } => self.get_tabup(pc, a, b, c),
            Instruction::GetField { a, b, c } => self.get_field(pc, a, b, c),
            Instruction::GetTable { a, b, c } => self.get_array(pc, a, b, ArrayKey::Register(c)),
            Instruction::GetI { a, b, c } => self.get_array(pc, a, b, ArrayKey::Constant(c)),
            Instruction::SelfOp { a, b, c } => self.get_method(pc, a, b, c),
            Instruction::SetField { a, b, c, k } => self.set_field(pc, a, b, c, k),
            Instruction::SetTable { a, b, c, k } => self.set_array(pc, a, ArrayKey::Register(b), c, k),
            Instruction::SetI { a, b, c, k } => self.set_array(pc, a, ArrayKey::Constant(b), c, k),
            Instruction::Call { a, b, c } => self.lower_call(pc, a, b, c),
            Instruction::NewTable { a, b, c, k } => self.lower_new_table(pc, a, b, c, k),
            Instruction::Closure { a, bx } => self.fast_helper(pc, FAST_CLOSURE, [u32::from(a), bx, 0], pc + 1),
            Instruction::Return0 { .. } => self.lower_return(pc, 0, 0),
            Instruction::TailCall { .. } => self.control_helper(pc, FAST_TAIL_CALL, [0, 0, 0], None),
            Instruction::Pow { a, b, c } => self.arith_helper(pc, a, ArithOp::Pow, u32::from(b), u32::from(c)),
            Instruction::PowK { a, b, c } => self.arith_helper(pc, a, ArithOp::Pow, u32::from(b), CONSTANT_BIT | u32::from(c)),
            Instruction::Shl { a, b, c } => self.arith_helper(pc, a, ArithOp::Shl, u32::from(b), u32::from(c)),
            Instruction::Shr { a, b, c } => self.arith_helper(pc, a, ArithOp::Shr, u32::from(b), u32::from(c)),
            Instruction::ShlI { a, b, sc } => self.arith_helper(pc, a, ArithOp::Shl, IMMEDIATE_BIT | (sc as u32 & 0x3fff_ffff), u32::from(b)),
            Instruction::ShrI { a, b, sc } => self.arith_helper(pc, a, ArithOp::Shr, u32::from(b), IMMEDIATE_BIT | (sc as u32 & 0x3fff_ffff)),
            Instruction::Return { a, b, k: false, .. } if b != 0 => self.lower_return(pc, a, u32::from(b) - 1),
            Instruction::Return { a, k: false, .. } => self.control_helper(pc, FAST_RETURN, [u32::from(a), u32::MAX, 0], None),
            Instruction::Return { .. } => self.call_step(pc),
            Instruction::ForPrep { a, bx } => self.for_prep_helper(pc, a, bx),
            Instruction::Return1 { a } => self.lower_return(pc, a, 1),
            Instruction::GetUpval { a, b } => self.get_upvalue(pc, a, b),
            Instruction::SetUpval { a, b } => self.set_upvalue(pc, a, b),
            Instruction::Len { .. } => self.call_helper(pc, self.helpers.misc),
            Instruction::ForLoop { a, bx } => self.for_loop(pc, a, bx),
            Instruction::TForLoop { a, bx } => {
                let base = self.registers();
                let control = self.builder.ins().load(types::I8, self.value_flags, base, slot(a + 3));
                let present = self.builder.ins().icmp_imm_s(IntCC::NotEqual, control, NIL_TAG);
                let back = self.builder.create_block();
                let fall = self.blocks[pc + 1];
                self.builder.ins().brif(present, back, &[], fall, &[]);
                self.builder.switch_to_block(back);
                self.edge(offset_pc(pc + 1, -(bx as i32)), true);
            }
            _ => self.call_step(pc),
        }
    }

    fn lower_call(&mut self, pc: usize, a: u8, b: u8, c: u8) {
        let arguments = [u32::from(a), u32::from(b), u32::from(c)];
        let mut targets = self.direct.as_ref().and_then(|direct| direct.calls.get(pc)).cloned().unwrap_or_default();
        let mut seen = Vec::with_capacity(targets.len());
        targets.retain(|target| {
            let fresh = !seen.contains(&target.proto);
            seen.push(target.proto);
            fresh
        });
        if targets.is_empty() || b == 0 || c == 0 {
            return self.control_helper(pc, FAST_CALL, arguments, Some(pc + 1));
        }
        let helper = self.builder.create_block();
        let check = self.builder.create_block();

        let base = self.registers();
        let tag = self.builder.ins().load(types::I8, self.value_flags, base, slot(a));
        let is_closure = self.builder.ins().icmp_imm_s(IntCC::Equal, tag, CLOSURE_TAG);
        self.builder.ins().brif(is_closure, check, &[], helper, &[]);
        self.builder.switch_to_block(check);
        let closure = self.builder.ins().load(types::I64, self.value_flags, base, slot(a) + PAYLOAD_OFFSET);
        let matched: Vec<Block> = targets.iter().map(|_| self.builder.create_block()).collect();
        for (target, &block) in targets.iter().zip(&matched) {
            self.continue_unless(closure, IntCC::Equal, target.closure as i64, block);
        }
        let address = self.closure_address(closure);
        let proto = self.builder.ins().load(types::I64, self.heap_flags, address, offset_of!(crate::object::Closure, proto) as i32);
        for (target, &block) in targets.iter().zip(&matched) {
            self.continue_unless(proto, IntCC::Equal, target.proto as i64, block);
        }
        self.builder.ins().jump(helper, &[]);
        for (target, &block) in targets.iter().zip(&matched) {
            self.builder.switch_to_block(block);
            self.direct_call(pc, (a, b, c), target, closure, helper);
        }
        self.builder.switch_to_block(helper);
        self.control_helper(pc, FAST_CALL, arguments, Some(pc + 1));
    }

    /// Mirrors `Vm::jit_call_compiled`; any exit other than a return in place goes to `Vm::jit_after_call`.
    fn direct_call(&mut self, pc: usize, (a, b, c): (u8, u8, u8), target: &CallTarget, closure: Value, helper: Block) {
        let Some(direct) = self.direct.as_ref() else {
            self.builder.ins().jump(helper, &[]);
            return;
        };
        let frames = direct.frames;
        let depth_limit = direct.depth_limit;
        let signature = direct.signature;
        let context = self.callee_context();
        let flags = self.frame_flags;
        let arguments = u32::from(b) - 1;
        let wanted = i64::from(c) - 1;
        let register_count = u64::from(arguments) + u64::from(target.max_stack);
        let span = register_count.max(crate::vm::REGISTER_SPAN as u64) as i64;
        let callee_offset = (i32::from(a) + 1) * VALUE_SIZE;

        let cell = self.builder.ins().iconst(types::I64, target.entry as i64);
        let entry = self.builder.ins().load(types::I64, flags, cell, 0);
        self.continue_unless(entry, IntCC::Equal, 0, helper);
        let depth_pointer = self.context_pointer(offset_of!(JitCtx, call_depth));
        let depth = self.builder.ins().load(types::I32, flags, depth_pointer, 0);
        self.continue_unless(depth, IntCC::UnsignedGreaterThanOrEqual, i64::from(depth_limit), helper);
        let thread = self.context_pointer(offset_of!(JitCtx, thread));
        let stack_len = self.builder.ins().load(types::I64, flags, thread, frames.stack_len);
        let caller_base = self.builder.ins().load(types::I64, self.context_flags, self.ctx, offset_of!(JitCtx, base) as i32);
        let callee_base = self.builder.ins().iadd_imm_s(caller_base, i64::from(a) + 1);
        let needed = self.builder.ins().iadd_imm_s(callee_base, span);
        let short = self.builder.ins().icmp(IntCC::UnsignedGreaterThan, needed, stack_len);
        self.branch_unless(short, helper);
        let top = self.builder.ins().iadd_imm_s(callee_base, register_count as i64);
        self.continue_unless(top, IntCC::UnsignedGreaterThan, crate::vm::MAX_STACK_SLOTS as i64, helper);
        let count = self.builder.ins().load(types::I64, flags, thread, frames.frames_len);
        let capacity = self.builder.ins().load(types::I64, flags, thread, frames.frames_capacity);
        let full = self.builder.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, count, capacity);
        self.branch_unless(full, helper);

        let base = self.registers();
        let nil = self.builder.ins().iconst(types::I8, NIL_TAG);
        for parameter in arguments..u32::from(target.num_params) {
            let offset = callee_offset + parameter as i32 * VALUE_SIZE;
            self.builder.ins().store(self.value_flags, nil, base, offset);
        }
        let data = self.builder.ins().load(types::I64, flags, thread, frames.frames_data);
        let frame_offset = self.builder.ins().imul_imm_s(count, frames.frame_size);
        let frame = self.builder.ins().iadd(data, frame_offset);
        let resume = self.builder.ins().iconst(types::I32, pc as i64 + 1);
        let caller_pc = frames.field(lua_frame::PC) - frames.frame_size as i32;
        self.builder.ins().store(flags, resume, frame, caller_pc);
        self.push_lua_frame(frame, closure, callee_base, wanted);
        let pushed = self.builder.ins().iadd_imm_s(count, 1);
        self.builder.ins().store(flags, pushed, thread, frames.frames_len);
        self.fill_callee_context(context, target, base, (callee_offset, callee_base, closure), wanted);
        let deeper = self.builder.ins().iadd_imm_s(depth, 1);
        self.builder.ins().store(flags, deeper, depth_pointer, 0);
        let callee_context = self.builder.ins().stack_addr(self.pointer, context, 0);
        let start = self.builder.ins().iconst(types::I32, 0);
        let call = self.builder.ins().call_indirect(signature, entry, &[callee_context, start]);
        let code = self.builder.inst_results(call)[0];
        let depth_pointer = self.context_pointer(offset_of!(JitCtx, call_depth));
        self.builder.ins().store(flags, depth, depth_pointer, 0);

        let returned = self.builder.create_block();
        let other = self.builder.create_block();
        let is_returned = self.builder.ins().icmp_imm_s(IntCC::Equal, code, super::EXIT_RETURNED);
        self.builder.ins().brif(is_returned, returned, &[], other, &[]);
        self.builder.switch_to_block(returned);
        let thread = self.builder.ins().stack_load(self.pointer, types::I64, context, offset_of!(JitCtx, thread) as i32);
        let current = self.builder.ins().load(types::I64, flags, thread, frames.frames_len);
        let popped = self.builder.ins().iadd_imm_s(current, -1);
        self.builder.ins().store(flags, popped, thread, frames.frames_len);
        self.refresh_from_callee(context, thread, callee_offset);
        self.jump_to(pc + 1);
        self.builder.switch_to_block(other);
        let code = self.builder.ins().ireduce(types::I32, code);
        let before = self.builder.ins().ireduce(types::I32, count);
        let pc_value = self.builder.ins().iconst(types::I32, pc as i64);
        let zero = self.builder.ins().iconst(types::I32, 0);
        let after = self.helpers.fast[FAST_AFTER_CALL];
        let call = self.builder.ins().call(after, &[self.ctx, pc_value, code, before, zero]);
        let result = self.builder.inst_results(call)[0];
        self.dispatch_result(result, Some(pc + 1));
    }

    /// Calls do not overlap, so one stack slot serves every call site.
    fn callee_context(&mut self) -> StackSlot {
        if let Some(slot) = self.direct.as_ref().and_then(|direct| direct.context) {
            return slot;
        }
        let size = std::mem::size_of::<JitCtx>() as u32;
        let align = std::mem::align_of::<JitCtx>().trailing_zeros() as u8;
        let slot = self.builder.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, size, align));
        if let Some(direct) = self.direct.as_mut() {
            direct.context = Some(slot);
        }
        slot
    }

    fn push_lua_frame(&mut self, frame: Value, closure: Value, callee_base: Value, wanted: i64) {
        let Some(frames) = self.direct.as_ref().map(|direct| direct.frames) else {
            return;
        };
        let flags = self.frame_flags;
        let zero_byte = self.builder.ins().iconst(types::I8, 0);
        let zero = self.builder.ins().iconst(types::I64, 0);
        let status = (crate::object::LuaFrame::call_status(wanted as i32, 0) as i64) << lua_frame::STATUS_SHIFT;
        let pc_status = self.builder.ins().iconst(types::I64, status);
        self.builder.ins().store(flags, zero_byte, frame, 0);
        self.builder.ins().store(flags, closure, frame, frames.field(lua_frame::CLOSURE));
        self.builder.ins().store(flags, callee_base, frame, frames.field(lua_frame::BASE));
        for word in 0..lua_frame::CACHE_WORDS {
            let offset = frames.field(lua_frame::CACHE + word * std::mem::size_of::<usize>());
            self.builder.ins().store(flags, zero, frame, offset);
        }
        self.builder.ins().store(flags, pc_status, frame, frames.field(lua_frame::PC));
        self.builder.ins().store(flags, zero, frame, frames.field(lua_frame::NVARARGS));
    }

    fn fill_callee_context(&mut self, context: StackSlot, target: &CallTarget, base: Value, (callee_offset, callee_base, closure): (i32, Value, Value), wanted: i64) {
        let copied = [
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
        for offset in copied {
            let value = self.builder.ins().load(types::I64, self.context_flags, self.ctx, offset as i32);
            self.builder.ins().stack_store(self.pointer, value, context, offset as i32);
        }
        let registers = self.builder.ins().iadd_imm_s(base, i64::from(callee_offset));
        let upvalues = if target.has_upvalues { self.closure_upvalues(closure) } else { self.builder.ins().iconst(types::I64, 0) };
        let constants = self.builder.ins().iconst(types::I64, target.constants as i64);
        let proto = self.builder.ins().iconst(types::I64, target.proto as i64);
        let results = self.builder.ins().iconst(types::I32, wanted);
        let ret_offset = self.builder.ins().iconst(types::I64, -i64::from(VALUE_SIZE));
        let set = [
            (offset_of!(JitCtx, registers), registers),
            (offset_of!(JitCtx, upvalues), upvalues),
            (offset_of!(JitCtx, constants), constants),
            (offset_of!(JitCtx, closure), closure),
            (offset_of!(JitCtx, base), callee_base),
            (offset_of!(JitCtx, proto), proto),
            (offset_of!(JitCtx, direct_results), results),
            (offset_of!(JitCtx, ret_offset), ret_offset),
        ];
        for (offset, value) in set {
            self.builder.ins().stack_store(self.pointer, value, context, offset as i32);
        }
    }

    fn closure_address(&mut self, closure: Value) -> Value {
        let index = self.builder.ins().ireduce(types::I32, closure);
        let index = self.builder.ins().uextend(types::I64, index);
        let closures = self.context_pointer(offset_of!(JitCtx, closures));
        let offset = self.builder.ins().imul_imm_s(index, std::mem::size_of::<crate::object::Closure>() as i64);
        self.builder.ins().iadd(closures, offset)
    }

    fn closure_upvalues(&mut self, closure: Value) -> Value {
        let Some(frames) = self.direct.as_ref().map(|direct| direct.frames) else {
            return self.builder.ins().iconst(types::I64, 0);
        };
        let address = self.closure_address(closure);
        self.builder.ins().load(types::I64, self.heap_flags, address, frames.closure_upvalues)
    }

    /// The callee's helpers refresh its context whenever they move its stack, tables or thread.
    fn refresh_from_callee(&mut self, context: StackSlot, thread: Value, callee_offset: i32) {
        let registers = self.builder.ins().stack_load(self.pointer, types::I64, context, offset_of!(JitCtx, registers) as i32);
        let registers = self.builder.ins().iadd_imm_s(registers, -i64::from(callee_offset));
        let tables = self.builder.ins().stack_load(self.pointer, types::I64, context, offset_of!(JitCtx, tables) as i32);
        let marks = self.builder.ins().stack_load(self.pointer, types::I64, context, offset_of!(JitCtx, table_marks) as i32);
        let closures = self.builder.ins().stack_load(self.pointer, types::I64, context, offset_of!(JitCtx, closures) as i32);
        for (offset, value) in [
            (offset_of!(JitCtx, registers), registers),
            (offset_of!(JitCtx, tables), tables),
            (offset_of!(JitCtx, table_marks), marks),
            (offset_of!(JitCtx, closures), closures),
            (offset_of!(JitCtx, thread), thread),
        ] {
            self.builder.ins().store(self.context_flags, value, self.ctx, offset as i32);
        }
    }

    fn continue_unless(&mut self, value: Value, condition: IntCC, immediate: i64, target: Block) {
        let taken = self.builder.ins().icmp_imm_s(condition, value, immediate);
        self.branch_unless(taken, target);
    }

    fn branch_unless(&mut self, taken: Value, target: Block) {
        let next = self.builder.create_block();
        self.builder.ins().brif(taken, target, &[], next, &[]);
        self.builder.switch_to_block(next);
    }

    /// A size that does not fit one operand, or a missing `EXTRAARG`, takes the interpreter's step.
    fn lower_new_table(&mut self, pc: usize, a: u8, b: u8, c: u32, k: bool) {
        let extra = match (k, self.proto.code.get(pc + 1)) {
            (false, Some(Instruction::ExtraArg { .. })) => 0,
            (true, Some(Instruction::ExtraArg { ax })) => u64::from(*ax),
            _ => return self.call_step(pc),
        };
        let array = u64::from(c) + extra * 1024;
        if array > i32::MAX as u64 {
            return self.call_step(pc);
        }
        self.fast_helper(pc, FAST_NEW_TABLE, [u32::from(a), array as u32, u32::from(b)], pc + 2);
    }

    /// Only nil, booleans and interned strings compare by tag and payload; other constants take the step.
    fn equal_constant(&mut self, pc: usize, a: u8, b: u8, k: bool) {
        let constant = self.proto.constants.get(usize::from(b)).copied();
        let interned = self.interned.get(usize::from(b)).copied().unwrap_or(false);
        let (tag, payload) = match constant {
            Some(LuaValue::Nil) => (NIL_TAG, None),
            Some(LuaValue::Boolean(flag)) => (BOOL_TAG, Some((types::I8, i64::from(flag)))),
            Some(LuaValue::String(string)) if interned => (STRING_TAG, Some((types::I64, string.packed() as i64))),
            _ => return self.call_step(pc),
        };
        let base = self.registers();
        let actual = self.builder.ins().load(types::I8, self.value_flags, base, slot(a));
        let same_tag = self.builder.ins().icmp_imm_s(IntCC::Equal, actual, tag);
        let condition = match payload {
            None => same_tag,
            Some((kind, expected)) => {
                let value = self.builder.ins().load(kind, self.value_flags, base, slot(a) + PAYLOAD_OFFSET);
                let same = self.builder.ins().icmp_imm_s(IntCC::Equal, value, expected);
                self.builder.ins().band(same_tag, same)
            }
        };
        self.branch_on(pc, condition, k, None);
    }

    /// A frame a compiled call entered that wants exactly these results is finished in place; other returns use the helper.
    fn lower_return(&mut self, pc: usize, first: u8, count: u32) {
        let helper = self.builder.create_block();
        let copy = self.builder.create_block();
        let check_none = self.builder.create_block();
        let done = self.builder.create_block();
        let results = self.builder.ins().load(types::I32, self.context_flags, self.ctx, offset_of!(JitCtx, direct_results) as i32);

        let exact = self.builder.ins().icmp_imm_s(IntCC::Equal, results, i64::from(count));
        self.builder.ins().brif(exact, copy, &[], check_none, &[]);
        self.builder.switch_to_block(check_none);
        let none = self.builder.ins().icmp_imm_s(IntCC::Equal, results, 0);
        self.builder.ins().brif(none, done, &[], helper, &[]);
        self.builder.switch_to_block(copy);
        let base = self.registers();
        let offset = self.builder.ins().load(types::I64, self.context_flags, self.ctx, offset_of!(JitCtx, ret_offset) as i32);
        let destination = self.builder.ins().iadd(base, offset);
        for index in 0..count {
            let register = first as i32 + index as i32;
            self.copy_value(base, register * VALUE_SIZE, destination, index as i32 * VALUE_SIZE);
        }
        self.builder.ins().jump(done, &[]);
        self.builder.switch_to_block(done);
        let code = self.builder.ins().iconst(types::I64, super::EXIT_RETURNED);
        self.builder.ins().return_(&[code]);
        self.builder.switch_to_block(helper);
        self.control_helper(pc, FAST_RETURN, [u32::from(first), count, 0], None);
    }

    fn load_constant(&mut self, pc: usize, a: u8, index: u32, next: usize) {
        if index as usize >= self.proto.constants.len() {
            return self.call_step(pc);
        }
        let source = self.constants();
        let base = self.registers();
        self.copy_value(source, constant_slot(index), base, slot(a));
        self.jump_to(next);
    }

    fn store_bool_constant(&mut self, a: u8, value: bool) {
        let base = self.registers();
        let flag = self.builder.ins().iconst(types::I8, i64::from(value));
        self.store_bool(base, slot(a), flag);
    }

    fn constant_operand(&self, index: u32) -> Option<Operand> {
        match self.proto.constants.get(index as usize)? {
            LuaValue::Integer(integer) => Some(Operand::Int(*integer)),
            LuaValue::Float(float) => Some(Operand::Float(*float)),
            _ => None,
        }
    }

    fn arithmetic_constant(&mut self, pc: usize, a: u8, b: u8, c: u8, op: Arith) {
        match self.constant_operand(u32::from(c)) {
            Some(constant) => self.arithmetic(pc, a, Operand::Register(b), constant, op),
            None => self.call_step(pc),
        }
    }

    /// The interpreter's fast path skips the metamethod instruction that follows, so success continues two instructions on.
    fn arithmetic(&mut self, pc: usize, target: u8, left: Operand, right: Operand, op: Arith) {
        let int_test = self.operands_test(left, right, INT_TAG);
        let float_test = if op.has_float() { self.operands_test(left, right, FLOAT_TAG) } else { Test::Never };
        let mixed_test = if op.has_float() { self.numeric_pair_test(left, right) } else { Test::Never };
        let arms = self.arms(&[(int_test, Path::Int), (float_test, Path::Float), (mixed_test, Path::Mixed)]);
        let slow = self.builder.create_block();
        self.dispatch(&arms, slow);
        for (block, path) in arms.iter().map(|&(_, block, path)| (block, path)) {
            self.builder.switch_to_block(block);
            let base = self.registers();
            let (x, y) = match path {
                Path::Int => (self.read_int(base, left), self.read_int(base, right)),
                Path::Float => (self.read_float(base, left), self.read_float(base, right)),
                Path::Mixed => (self.read_as_float(base, left), self.read_as_float(base, right)),
            };
            let value = match path {
                Path::Int => self.int_op(op, x, y),
                Path::Float | Path::Mixed => self.float_op(op, x, y),
            };
            match path {
                Path::Int => self.store_int(base, slot(target), value),
                Path::Float | Path::Mixed => self.store_float(base, slot(target), value),
            }
            self.jump_to(pc + 2);
        }
        self.builder.switch_to_block(slow);
        self.call_step(pc);
    }

    fn numeric_pair_test(&mut self, left: Operand, right: Operand) -> Test {
        let first = self.numeric_test(left);
        let second = self.numeric_test(right);
        self.combine(first, second)
    }

    fn numeric_test(&mut self, operand: Operand) -> Test {
        match operand {
            Operand::Register(register) => {
                let base = self.registers();
                let tag = self.builder.ins().load(types::I8, self.value_flags, base, slot(register));
                let integer = self.builder.ins().icmp_imm_s(IntCC::Equal, tag, INT_TAG);
                let float = self.builder.ins().icmp_imm_s(IntCC::Equal, tag, FLOAT_TAG);
                Test::When(self.builder.ins().bor(integer, float))
            }
            Operand::Int(_) | Operand::Float(_) => Test::Always,
        }
    }

    fn read_as_float(&mut self, base: Value, operand: Operand) -> Value {
        match operand {
            Operand::Register(register) => {
                let integer = self.read_int(base, operand);
                let as_float = self.builder.ins().fcvt_from_sint(types::F64, integer);
                let stored = self.read_float(base, operand);
                let tag = self.builder.ins().load(types::I8, self.value_flags, base, slot(register));
                let is_integer = self.builder.ins().icmp_imm_s(IntCC::Equal, tag, INT_TAG);
                self.builder.ins().select(is_integer, as_float, stored)
            }
            Operand::Int(integer) => self.builder.ins().f64const(integer as f64),
            Operand::Float(float) => self.builder.ins().f64const(float),
        }
    }

    fn integer_division_constant(&mut self, pc: usize, a: u8, b: u8, c: u8, op: Division) {
        match self.constant_operand(u32::from(c)) {
            Some(constant @ Operand::Int(_)) => self.integer_division(pc, a, Operand::Register(b), constant, op),
            _ => self.call_step(pc),
        }
    }

    /// Divisors 0 and -1 take the interpreter's step.
    fn integer_division(&mut self, pc: usize, target: u8, left: Operand, right: Operand, op: Division) {
        let int_test = self.operands_test(left, right, INT_TAG);
        let divisor = self.divisor_test(right);
        let test = self.combine(int_test, divisor);
        let arms = self.arms(&[(test, false)]);
        let slow = self.builder.create_block();
        self.dispatch(&arms, slow);
        if let Some(&(_, block, _)) = arms.first() {
            self.builder.switch_to_block(block);
            let base = self.registers();
            let x = self.read_int(base, left);
            let y = self.read_int(base, right);
            let value = self.floor_division(op, x, y);
            self.store_int(base, slot(target), value);
            self.jump_to(pc + 2);
        }
        self.builder.switch_to_block(slow);
        self.call_step(pc);
    }

    fn floor_division(&mut self, op: Division, x: Value, y: Value) -> Value {
        let remainder = self.builder.ins().srem(x, y);
        let nonzero = self.builder.ins().icmp_imm_s(IntCC::NotEqual, remainder, 0);
        match op {
            Division::Mod => {
                let signs = self.builder.ins().bxor(remainder, y);
                let negative = self.builder.ins().icmp_imm_s(IntCC::SignedLessThan, signs, 0);
                let adjust = self.builder.ins().band(nonzero, negative);
                let shifted = self.builder.ins().iadd(remainder, y);
                self.builder.ins().select(adjust, shifted, remainder)
            }
            Division::IDiv => {
                let quotient = self.builder.ins().sdiv(x, y);
                let signs = self.builder.ins().bxor(x, y);
                let negative = self.builder.ins().icmp_imm_s(IntCC::SignedLessThan, signs, 0);
                let adjust = self.builder.ins().band(nonzero, negative);
                let lowered = self.builder.ins().iadd_imm_s(quotient, -1);
                self.builder.ins().select(adjust, lowered, quotient)
            }
        }
    }

    /// The payload is read whatever the tag: the frame always holds that memory.
    fn divisor_test(&mut self, right: Operand) -> Test {
        match right {
            Operand::Int(0 | -1) => Test::Never,
            Operand::Int(_) => Test::Always,
            Operand::Float(_) => Test::Never,
            Operand::Register(register) => {
                let base = self.registers();
                let value = self.read_int(base, Operand::Register(register));
                let nonzero = self.builder.ins().icmp_imm_s(IntCC::NotEqual, value, 0);
                let not_minus_one = self.builder.ins().icmp_imm_s(IntCC::NotEqual, value, -1);
                Test::When(self.builder.ins().band(nonzero, not_minus_one))
            }
        }
    }

    fn divide(&mut self, pc: usize, target: u8, left: Operand, right: Operand) {
        let int_test = self.operands_test(left, right, INT_TAG);
        let float_test = self.operands_test(left, right, FLOAT_TAG);
        let mixed_test = self.numeric_pair_test(left, right);
        let arms = self.arms(&[(int_test, Path::Int), (float_test, Path::Float), (mixed_test, Path::Mixed)]);
        let slow = self.builder.create_block();
        self.dispatch(&arms, slow);
        for (block, path) in arms.iter().map(|&(_, block, path)| (block, path)) {
            self.builder.switch_to_block(block);
            let base = self.registers();
            let (x, y) = match path {
                Path::Int => {
                    let x = self.read_int(base, left);
                    let y = self.read_int(base, right);
                    (self.builder.ins().fcvt_from_sint(types::F64, x), self.builder.ins().fcvt_from_sint(types::F64, y))
                }
                Path::Float => (self.read_float(base, left), self.read_float(base, right)),
                Path::Mixed => (self.read_as_float(base, left), self.read_as_float(base, right)),
            };
            let quotient = self.builder.ins().fdiv(x, y);
            self.store_float(base, slot(target), quotient);
            self.jump_to(pc + 2);
        }
        self.builder.switch_to_block(slow);
        self.call_step(pc);
    }

    fn int_op(&mut self, op: Arith, x: Value, y: Value) -> Value {
        let ins = self.builder.ins();
        match op {
            Arith::Add => ins.iadd(x, y),
            Arith::Sub => ins.isub(x, y),
            Arith::Mul => ins.imul(x, y),
            Arith::BAnd => ins.band(x, y),
            Arith::BOr => ins.bor(x, y),
            Arith::BXor => ins.bxor(x, y),
        }
    }

    fn float_op(&mut self, op: Arith, x: Value, y: Value) -> Value {
        let ins = self.builder.ins();
        match op {
            Arith::Add => ins.fadd(x, y),
            Arith::Sub => ins.fsub(x, y),
            Arith::Mul => ins.fmul(x, y),
            Arith::BAnd | Arith::BOr | Arith::BXor => unreachable!("bitwise operators have no float path"),
        }
    }

    fn negate(&mut self, pc: usize, a: u8, b: u8) {
        let operand = Operand::Register(b);
        let int_test = self.tag_test(operand, INT_TAG);
        let float_test = self.tag_test(operand, FLOAT_TAG);
        let arms = self.arms(&[(int_test, false), (float_test, true)]);
        let slow = self.builder.create_block();
        self.dispatch(&arms, slow);
        for (block, is_float) in arms.iter().map(|&(_, block, is_float)| (block, is_float)) {
            self.builder.switch_to_block(block);
            let base = self.registers();
            if is_float {
                let value = self.read_float(base, operand);
                let negated = self.builder.ins().fneg(value);
                self.store_float(base, slot(a), negated);
            } else {
                let value = self.read_int(base, operand);
                let negated = self.builder.ins().ineg(value);
                self.store_int(base, slot(a), negated);
            }
            self.jump_to(pc + 1);
        }
        self.builder.switch_to_block(slow);
        self.call_step(pc);
    }

    fn bitwise_not(&mut self, pc: usize, a: u8, b: u8) {
        let operand = Operand::Register(b);
        let int_test = self.tag_test(operand, INT_TAG);
        let arms = self.arms(&[(int_test, false)]);
        let slow = self.builder.create_block();
        self.dispatch(&arms, slow);
        let (_, block, _) = arms[0];
        self.builder.switch_to_block(block);
        let base = self.registers();
        let value = self.read_int(base, operand);
        let inverted = self.builder.ins().bnot(value);
        self.store_int(base, slot(a), inverted);
        self.jump_to(pc + 1);
        self.builder.switch_to_block(slow);
        self.call_step(pc);
    }

    /// Blocks exist only for tests that can hold, so every block created is lowered and terminated.
    fn arms<K: Copy>(&mut self, tests: &[(Test, K)]) -> Vec<(Test, Block, K)> {
        let mut arms = Vec::new();
        for &(test, kind) in tests {
            if !matches!(test, Test::Never) {
                arms.push((test, self.builder.create_block(), kind));
            }
        }
        arms
    }

    fn dispatch<K>(&mut self, arms: &[(Test, Block, K)], slow: Block) {
        for &(test, body, _) in arms {
            let next = self.builder.create_block();
            self.branch(test, body, next);
            self.builder.switch_to_block(next);
        }
        self.builder.ins().jump(slow, &[]);
    }

    fn branch(&mut self, test: Test, then: Block, otherwise: Block) {
        match test {
            Test::Always => {
                self.builder.ins().jump(then, &[]);
            }
            Test::Never => {
                self.builder.ins().jump(otherwise, &[]);
            }
            Test::When(condition) => {
                self.builder.ins().brif(condition, then, &[], otherwise, &[]);
            }
        }
    }

    fn compare(&mut self, pc: usize, k: bool, compare: Compare, default: Option<bool>) {
        let Some(taken_target) = self.test_target(pc) else {
            return self.call_step(pc);
        };
        let join = self.builder.create_block();
        self.builder.append_block_param(join, types::I8);
        let int_test = match compare.int {
            Some((left, right)) => self.operands_test(left, right, INT_TAG),
            None => Test::Never,
        };
        let float_test = match compare.float {
            Some((left, right)) => self.operands_test(left, right, FLOAT_TAG),
            None => Test::Never,
        };
        let arms = self.arms(&[(int_test, false), (float_test, true)]);
        let slow = self.builder.create_block();
        self.dispatch(&arms, slow);
        for (block, is_float) in arms.iter().map(|&(_, block, is_float)| (block, is_float)) {
            self.builder.switch_to_block(block);
            let base = self.registers();
            let condition = if is_float {
                let (left, right) = compare.float.expect("a float arm exists only with float operands");
                let x = self.read_float(base, left);
                let y = self.read_float(base, right);
                self.builder.ins().fcmp(compare.float_cc, x, y)
            } else {
                let (left, right) = compare.int.expect("an integer arm exists only with integer operands");
                let x = self.read_int(base, left);
                let y = self.read_int(base, right);
                self.builder.ins().icmp(compare.int_cc, x, y)
            };
            self.builder.ins().jump(join, &[BlockArg::Value(condition)]);
        }
        self.builder.switch_to_block(slow);
        match default {
            None => self.call_step(pc),
            Some(value) => {
                let condition = self.builder.ins().iconst(types::I8, i64::from(value));
                self.builder.ins().jump(join, &[BlockArg::Value(condition)]);
            }
        }
        self.builder.switch_to_block(join);
        let condition = self.builder.block_params(join)[0];
        let fall = self.blocks[pc + 2];
        let taken = self.blocks[taken_target];
        self.branch_on_flag(condition, k, taken, fall);
    }

    fn branch_on(&mut self, pc: usize, condition: Value, k: bool, copy: Option<Block>) {
        let Some(target) = self.test_target(pc) else {
            return self.call_step(pc);
        };
        let taken = copy.unwrap_or(self.blocks[target]);
        let fall = self.blocks[pc + 2];
        self.branch_on_flag(condition, k, taken, fall);
    }

    fn branch_on_flag(&mut self, condition: Value, k: bool, taken: Block, fall: Block) {
        if k {
            self.builder.ins().brif(condition, taken, &[], fall, &[]);
        } else {
            self.builder.ins().brif(condition, fall, &[], taken, &[]);
        }
    }

    /// Never `None` for verified code.
    fn test_target(&self, pc: usize) -> Option<usize> {
        match self.proto.code.get(pc + 1)? {
            Instruction::Jmp { sj } => Some(offset_pc(pc + 2, *sj)),
            _ => None,
        }
    }

    fn for_loop(&mut self, pc: usize, a: u8, bx: u32) {
        let count = Operand::Register(a);
        let step = Operand::Register(a + 1);
        let index = Operand::Register(a + 2);
        let count_test = self.tag_test(count, INT_TAG);
        let step_test = self.tag_test(step, INT_TAG);
        let index_test = self.tag_test(index, INT_TAG);
        let integer_test = self.combine(count_test, step_test);
        let integer_test = self.combine(integer_test, index_test);
        let limit_test = self.tag_test(count, FLOAT_TAG);
        let float_step_test = self.tag_test(step, FLOAT_TAG);
        let float_index_test = self.tag_test(index, FLOAT_TAG);
        let float_test = self.combine(limit_test, float_step_test);
        let float_test = self.combine(float_test, float_index_test);
        let arms = self.arms(&[(integer_test, false), (float_test, true)]);
        let slow = self.builder.create_block();
        self.dispatch(&arms, slow);
        for (block, is_float) in arms.iter().map(|&(_, block, is_float)| (block, is_float)) {
            self.builder.switch_to_block(block);
            if is_float {
                self.float_for_loop(pc, a, bx);
            } else {
                self.integer_for_loop(pc, a, bx);
            }
        }
        self.builder.switch_to_block(slow);
        self.call_step(pc);
    }

    fn float_for_loop(&mut self, pc: usize, a: u8, bx: u32) {
        let base = self.registers();
        let limit = self.read_float(base, Operand::Register(a));
        let step = self.read_float(base, Operand::Register(a + 1));
        let index = self.read_float(base, Operand::Register(a + 2));
        let next = self.builder.ins().fadd(index, step);
        let zero = self.builder.ins().f64const(0.0);
        let stepping_up = self.builder.ins().fcmp(FloatCC::LessThan, zero, step);
        let up = self.builder.ins().fcmp(FloatCC::LessThanOrEqual, next, limit);
        let down = self.builder.ins().fcmp(FloatCC::LessThanOrEqual, limit, next);
        let continues = self.builder.ins().select(stepping_up, up, down);
        let looping = self.builder.create_block();
        let fall = self.blocks[pc + 1];
        self.builder.ins().brif(continues, looping, &[], fall, &[]);
        self.builder.switch_to_block(looping);
        let base = self.registers();
        self.store_float(base, slot(a + 2), next);
        self.edge(offset_pc(pc + 1, -(bx as i32)), true);
    }

    fn integer_for_loop(&mut self, pc: usize, a: u8, bx: u32) {
        let count = Operand::Register(a);
        let step = Operand::Register(a + 1);
        let index = Operand::Register(a + 2);
        let base = self.registers();
        let remaining = self.read_int(base, count);
        let finished = self.builder.ins().icmp_imm_s(IntCC::Equal, remaining, 0);
        let fall = self.blocks[pc + 1];
        let looping = self.builder.create_block();
        self.builder.ins().brif(finished, fall, &[], looping, &[]);
        self.builder.switch_to_block(looping);
        let base = self.registers();
        let remaining = self.read_int(base, count);
        let decremented = self.builder.ins().iadd_imm_s(remaining, -1);
        self.builder.ins().store(self.value_flags, decremented, base, slot(a) + PAYLOAD_OFFSET);
        let increment = self.read_int(base, step);
        let control = self.read_int(base, index);
        let advanced = self.builder.ins().iadd(control, increment);
        self.builder.ins().store(self.value_flags, advanced, base, slot(a + 2) + PAYLOAD_OFFSET);
        self.edge(offset_pc(pc + 1, -(bx as i32)), true);
    }

    fn edge(&mut self, target: usize, backward: bool) {
        if !backward {
            self.jump_to(target);
            return;
        }
        let requests = self.builder.ins().load(self.pointer, self.context_flags, self.ctx, offset_of!(JitCtx, requests) as i32);
        let requested = self.builder.ins().atomic_load(types::I8, MemFlagsData::new().with_notrap(), requests);
        let check = self.builder.create_block();
        let destination = self.blocks[target];
        self.builder.ins().brif(requested, check, &[], destination, &[]);
        self.builder.switch_to_block(check);
        let target_value = self.builder.ins().iconst(types::I32, target as i64);
        let call = self.builder.ins().call(self.helpers.safe_point, &[self.ctx, target_value]);
        let result = self.builder.inst_results(call)[0];
        self.dispatch_result(result, Some(target));
    }

    /// A cache miss or an uncovered site runs the helper.
    fn get_field(&mut self, pc: usize, a: u8, b: u8, c: u8) {
        let entries = self.field_entries(pc);
        let Some(layout) = self.layout.filter(|_| !entries.is_empty()) else {
            return self.fast_helper(pc, FAST_GET_FIELD, [a.into(), b.into(), c.into()], pc + 1);
        };
        let miss = self.builder.create_block();

        let table = self.table_in_register(layout, b, miss);
        self.shape_switch(layout, table, &entries, miss, |lowering, entry| {
            let found = lowering.read_entry(layout, table, entry, miss);
            let base = lowering.registers();
            lowering.store_pair(base, slot(a), found);
            lowering.jump_to(pc + 1);
        });
        self.builder.switch_to_block(miss);
        self.fast_helper(pc, FAST_GET_FIELD, [a.into(), b.into(), c.into()], pc + 1);
    }

    fn get_tabup(&mut self, pc: usize, a: u8, b: u8, c: u8) {
        let entries = self.field_entries(pc);
        let Some(layout) = self.layout.filter(|_| !entries.is_empty()) else {
            return self.fast_helper(pc, FAST_GET_TABUP, [a.into(), b.into(), c.into()], pc + 1);
        };
        let miss = self.builder.create_block();

        let address = self.upvalue_address(b);
        let anywhere = MemFlagsData::new().with_notrap();
        let tag = self.builder.ins().uload8(types::I64, anywhere, address, 0);
        let next = self.builder.create_block();
        let is_table = self.builder.ins().icmp_imm_s(IntCC::Equal, tag, TABLE_TAG);
        self.builder.ins().brif(is_table, next, &[], miss, &[]);
        self.builder.switch_to_block(next);
        let index = self.builder.ins().uload32(anywhere, address, PAYLOAD_OFFSET);
        let table = self.table_address(layout, index);
        self.shape_switch(layout, table, &entries, miss, |lowering, entry| {
            let found = lowering.read_entry(layout, table, entry, miss);
            let base = lowering.registers();
            lowering.store_pair(base, slot(a), found);
            lowering.jump_to(pc + 1);
        });
        self.builder.switch_to_block(miss);
        self.fast_helper(pc, FAST_GET_TABUP, [a.into(), b.into(), c.into()], pc + 1);
    }

    fn get_upvalue(&mut self, pc: usize, a: u8, b: u8) {
        let address = self.upvalue_address(b);
        let anywhere = MemFlagsData::new().with_notrap();
        let low = self.builder.ins().load(types::I64, anywhere, address, 0);
        let high = self.builder.ins().load(types::I64, anywhere, address, PAYLOAD_OFFSET);
        let base = self.registers();
        self.store_pair(base, slot(a), (low, high));
        self.jump_to(pc + 1);
    }

    /// A closed upvalue needs the collector's barrier, so its store runs the helper.
    fn set_upvalue(&mut self, pc: usize, a: u8, b: u8) {
        let miss = self.builder.create_block();

        let cell = self.upvalue_cell(b);
        let tag = self.builder.ins().uload8(types::I64, self.heap_flags, cell, upvalue::TAG);
        let is_open = self.builder.ins().icmp_imm_s(IntCC::Equal, tag, upvalue::OPEN);
        let next = self.builder.create_block();
        self.builder.ins().brif(is_open, next, &[], miss, &[]);
        self.builder.switch_to_block(next);
        let address = self.builder.ins().load(types::I64, self.heap_flags, cell, upvalue::VALUE);
        let base = self.registers();
        let value = self.load_pair(base, slot(a));
        let anywhere = MemFlagsData::new().with_notrap();
        self.builder.ins().store(anywhere, value.0, address, 0);
        self.builder.ins().store(anywhere, value.1, address, PAYLOAD_OFFSET);
        self.jump_to(pc + 1);
        self.builder.switch_to_block(miss);
        self.fast_helper(pc, FAST_SET_UPVAL, [a.into(), b.into(), 0], pc + 1);
    }

    fn upvalue_cell(&mut self, b: u8) -> Value {
        let upvalues = self.context_pointer(offset_of!(JitCtx, upvalues));
        let offset = i32::try_from(upvalue::cell(b)).expect("an upvalue index is a byte");
        self.builder.ins().load(types::I64, self.heap_flags, upvalues, offset)
    }

    /// The address may be a register's, so accesses through it alias everything.
    fn upvalue_address(&mut self, b: u8) -> Value {
        let cell = self.upvalue_cell(b);
        self.builder.ins().load(types::I64, self.heap_flags, cell, upvalue::VALUE)
    }

    fn get_method(&mut self, pc: usize, a: u8, b: u8, c: u8) {
        let entries = self.field_entries(pc);
        let methods = self.method_entries(pc);
        let Some(layout) = self.layout.filter(|_| !entries.is_empty() || !methods.is_empty()) else {
            return self.fast_helper(pc, FAST_GET_SELF, [a.into(), b.into(), c.into()], pc + 1);
        };
        let miss = self.builder.create_block();

        let table = self.table_in_register(layout, b, miss);
        if methods.is_empty() {
            self.shape_switch(layout, table, &entries, miss, |lowering, entry| {
                let found = lowering.read_entry(layout, table, entry, miss);
                lowering.finish_method(pc, a, b, found);
            });
        } else {
            for method in &methods {
                let next = self.builder.create_block();
                let found = self.read_method(layout, table, method, next);
                self.finish_method(pc, a, b, found);
                self.builder.switch_to_block(next);
            }
            self.builder.ins().jump(miss, &[]);
        }
        self.builder.switch_to_block(miss);
        self.fast_helper(pc, FAST_GET_SELF, [a.into(), b.into(), c.into()], pc + 1);
    }

    /// The object is read before either store, so a method register that is also the object reads its old value.
    fn finish_method(&mut self, pc: usize, a: u8, b: u8, method: (Value, Value)) {
        let base = self.registers();
        let object = self.load_pair(base, slot(b));
        self.store_pair(base, slot(a + 1), object);
        self.store_pair(base, slot(a), method);
        self.jump_to(pc + 1);
    }

    fn set_field(&mut self, pc: usize, a: u8, b: u8, c: u8, k: bool) {
        let entries: Vec<Entry> = self.field_entries(pc).into_iter().filter(|entry| entry.slot != ABSENT).collect();
        let Some(layout) = self.layout.filter(|_| !entries.is_empty()) else {
            return self.fast_helper(pc, FAST_SET_FIELD, [a.into(), b.into(), operand_index(c, k)], pc + 1);
        };
        let miss = self.builder.create_block();

        let table = self.table_in_register(layout, a, miss);
        self.shape_switch(layout, table, &entries, miss, |lowering, entry| {
            let address = lowering.slot_address(layout, table, entry.slot);
            let tag = lowering.builder.ins().load(types::I8, lowering.heap_flags, address, 0);
            let present = lowering.builder.ins().icmp_imm_s(IntCC::NotEqual, tag, NIL_TAG);
            let store = lowering.builder.create_block();
            lowering.builder.ins().brif(present, store, &[], miss, &[]);
            lowering.builder.switch_to_block(store);
            let value = lowering.store_operand(c, k);
            let flags = lowering.heap_flags;
            lowering.builder.ins().store(flags, value.0, address, 0);
            lowering.builder.ins().store(flags, value.1, address, PAYLOAD_OFFSET);
            lowering.table_barrier(pc, a, c, k, value.0);
        });
        self.builder.switch_to_block(miss);
        self.fast_helper(pc, FAST_SET_FIELD, [a.into(), b.into(), operand_index(c, k)], pc + 1);
    }

    /// Only a collectable value stored into a black table needs the barrier helper.
    fn table_barrier(&mut self, pc: usize, a: u8, c: u8, k: bool, low: Value) {
        let tag = self.builder.ins().band_imm_s(low, 0xff);
        let shifted = self.builder.ins().iadd_imm_s(tag, -STRING_TAG);
        let collectable = self.builder.ins().icmp_imm_u(IntCC::UnsignedLessThan, shifted, LIGHT_USERDATA_TAG - STRING_TAG);
        let check = self.builder.create_block();
        self.builder.ins().brif(collectable, check, &[], self.blocks[pc + 1], &[]);
        self.builder.switch_to_block(check);
        let base = self.registers();
        let index = self.builder.ins().uload32(self.value_flags, base, slot(a) + PAYLOAD_OFFSET);
        let marks = self.context_pointer(offset_of!(JitCtx, table_marks));
        let mark_address = self.builder.ins().iadd(marks, index);
        let mark = self.builder.ins().uload8(types::I64, self.heap_flags, mark_address, 0);
        let black = self.builder.ins().band_imm_s(mark, i64::from(crate::gc::BLACK));
        let barrier = self.builder.create_block();
        self.builder.ins().brif(black, barrier, &[], self.blocks[pc + 1], &[]);
        self.builder.switch_to_block(barrier);
        self.fast_helper(pc, FAST_TABLE_BARRIER, [a.into(), operand_index(c, k), 0], pc + 1);
    }

    fn store_operand(&mut self, c: u8, k: bool) -> (Value, Value) {
        if k {
            let constants = self.constants();
            return self.load_pair(constants, constant_slot(u32::from(c)));
        }
        let base = self.registers();
        self.load_pair(base, slot(c))
    }

    fn shape_switch(&mut self, layout: TableLayout, table: Value, entries: &[Entry], miss: Block, mut hit: impl FnMut(&mut Self, Entry)) {
        let shape = self.table_shape(layout, table, miss);
        for entry in entries {
            let matched = self.builder.create_block();
            let next = self.builder.create_block();
            let same = self.builder.ins().icmp_imm_s(IntCC::Equal, shape, entry.shape.packed() as i64);
            self.builder.ins().brif(same, matched, &[], next, &[]);
            self.builder.switch_to_block(matched);
            hit(self, *entry);
            self.builder.switch_to_block(next);
        }
        self.builder.ins().jump(miss, &[]);
    }

    fn read_entry(&mut self, layout: TableLayout, table: Value, entry: Entry, miss: Block) -> (Value, Value) {
        if entry.slot == ABSENT {
            return self.absent_value(layout, table, miss);
        }
        let address = self.slot_address(layout, table, entry.slot);
        self.present_value(address, miss)
    }

    fn present_value(&mut self, address: Value, miss: Block) -> (Value, Value) {
        let low = self.builder.ins().load(types::I64, self.heap_flags, address, 0);
        let high = self.builder.ins().load(types::I64, self.heap_flags, address, PAYLOAD_OFFSET);
        let tag = self.builder.ins().band_imm_s(low, 0xff);
        let present = self.builder.ins().icmp_imm_s(IntCC::NotEqual, tag, NIL_TAG);
        let next = self.builder.create_block();
        self.builder.ins().brif(present, next, &[], miss, &[]);
        self.builder.switch_to_block(next);
        (low, high)
    }

    fn absent_value(&mut self, layout: TableLayout, table: Value, miss: Block) -> (Value, Value) {
        let nil = self.builder.create_block();
        let has_metatable = self.builder.create_block();
        let tag = self.builder.ins().uload32(self.heap_flags, table, layout.metatable_tag);
        let none = self.builder.ins().icmp_imm_s(IntCC::Equal, tag, i64::from(layout.metatable_none));
        self.builder.ins().brif(none, nil, &[], has_metatable, &[]);
        self.builder.switch_to_block(has_metatable);
        let index = self.builder.ins().uload32(self.heap_flags, table, layout.metatable_index);
        let metatable = self.table_address(layout, index);
        let absent = self.builder.ins().uload8(types::I64, self.heap_flags, metatable, layout.absent);
        let bit = Event::Index.absence_bit().map_or(0, i64::from);
        let flagged = self.builder.ins().band_imm_s(absent, bit);
        self.builder.ins().brif(flagged, nil, &[], miss, &[]);
        self.builder.switch_to_block(nil);
        let low = self.builder.ins().iconst(types::I64, NIL_TAG);
        let high = self.builder.ins().iconst(types::I64, 0);
        (low, high)
    }

    fn read_method(&mut self, layout: TableLayout, table: Value, method: &MethodEntry, miss: Block) -> (Value, Value) {
        let receiver_shape = self.table_shape(layout, table, miss);
        self.expect_word(receiver_shape, method.receiver.packed(), miss);
        let tag = self.builder.ins().uload32(self.heap_flags, table, layout.metatable_tag);
        let none = self.builder.ins().icmp_imm_s(IntCC::Equal, tag, i64::from(layout.metatable_none));
        let some = self.builder.create_block();
        self.builder.ins().brif(none, miss, &[], some, &[]);
        self.builder.switch_to_block(some);
        let metatable_index = self.builder.ins().uload32(self.heap_flags, table, layout.metatable_index);
        let metatable = self.table_address(layout, metatable_index);
        let metatable_shape = self.table_shape(layout, metatable, miss);
        self.expect_word(metatable_shape, method.metatable_shape.packed(), miss);
        let index_slot = self.slot_address(layout, metatable, method.index_slot);
        let index_tag = self.builder.ins().uload8(types::I64, self.heap_flags, index_slot, 0);
        self.expect_word(index_tag, TABLE_TAG as u64, miss);
        let class_ref = self.builder.ins().load(types::I64, self.heap_flags, index_slot, PAYLOAD_OFFSET);
        self.expect_word(class_ref, method.class.packed(), miss);
        let class = self.constant_table(layout, method.class.index());
        let class_shape = self.table_shape(layout, class, miss);
        self.expect_word(class_shape, method.class_shape.packed(), miss);
        let address = self.slot_address(layout, class, method.class_slot);
        self.present_value(address, miss)
    }

    fn expect_word(&mut self, actual: Value, expected: u64, miss: Block) {
        let same = self.builder.ins().icmp_imm_s(IntCC::Equal, actual, expected as i64);
        let next = self.builder.create_block();
        self.builder.ins().brif(same, next, &[], miss, &[]);
        self.builder.switch_to_block(next);
    }

    fn table_in_register(&mut self, layout: TableLayout, register: u8, miss: Block) -> Value {
        let base = self.registers();
        let tag = self.builder.ins().load(types::I8, self.value_flags, base, slot(register));
        let is_table = self.builder.ins().icmp_imm_s(IntCC::Equal, tag, TABLE_TAG);
        let next = self.builder.create_block();
        self.builder.ins().brif(is_table, next, &[], miss, &[]);
        self.builder.switch_to_block(next);
        let index = self.builder.ins().uload32(self.value_flags, base, slot(register) + PAYLOAD_OFFSET);
        self.table_address(layout, index)
    }

    /// The arena moves when it grows, so its start is read from the context at each use.
    fn table_address(&mut self, layout: TableLayout, index: Value) -> Value {
        let tables = self.context_pointer(offset_of!(JitCtx, tables));
        let offset = self.builder.ins().imul_imm_s(index, layout.size);
        self.builder.ins().iadd(tables, offset)
    }

    fn constant_table(&mut self, layout: TableLayout, index: usize) -> Value {
        let index = self.builder.ins().iconst(types::I64, index as i64);
        self.table_address(layout, index)
    }

    fn table_shape(&mut self, layout: TableLayout, table: Value, miss: Block) -> Value {
        let shaped = self.builder.ins().uload8(types::I64, self.heap_flags, table, layout.shaped);
        let next = self.builder.create_block();
        self.builder.ins().brif(shaped, next, &[], miss, &[]);
        self.builder.switch_to_block(next);
        self.builder.ins().load(types::I64, self.heap_flags, table, layout.shape)
    }

    /// A cached shape holds the slot, so the slot lies inside the table's buffer.
    fn slot_address(&mut self, layout: TableLayout, table: Value, slot: u32) -> Value {
        let slots = self.builder.ins().load(self.pointer, self.heap_flags, table, layout.slots);
        self.builder.ins().iadd_imm_s(slots, i64::from(slot) * i64::from(VALUE_SIZE))
    }

    fn load_pair(&mut self, base: Value, offset: i32) -> (Value, Value) {
        let low = self.builder.ins().load(types::I64, self.value_flags, base, offset);
        let high = self.builder.ins().load(types::I64, self.value_flags, base, offset + PAYLOAD_OFFSET);
        (low, high)
    }

    fn store_pair(&mut self, base: Value, offset: i32, value: (Value, Value)) {
        self.builder.ins().store(self.value_flags, value.0, base, offset);
        self.builder.ins().store(self.value_flags, value.1, base, offset + PAYLOAD_OFFSET);
    }

    fn get_array(&mut self, pc: usize, a: u8, b: u8, key: ArrayKey) {
        let arguments = [a.into(), b.into(), key.operand()];
        let Some(layout) = self.layout else {
            return self.fast_helper(pc, key.get_helper(), arguments, pc + 1);
        };
        let miss = self.builder.create_block();

        let table = self.table_in_register(layout, b, miss);
        let element = self.array_element(layout, table, key, miss);
        let found = self.present_value(element, miss);
        let base = self.registers();
        self.store_pair(base, slot(a), found);
        self.jump_to(pc + 1);
        self.builder.switch_to_block(miss);
        self.fast_helper(pc, key.get_helper(), arguments, pc + 1);
    }

    /// The raw store runs only when no `__newindex` can fire, as `Heap::table_store` decides.
    fn set_array(&mut self, pc: usize, a: u8, key: ArrayKey, c: u8, k: bool) {
        let arguments = [a.into(), key.operand(), operand_index(c, k)];
        let Some(layout) = self.layout else {
            return self.fast_helper(pc, key.set_helper(), arguments, pc + 1);
        };
        let miss = self.builder.create_block();

        let table = self.table_in_register(layout, a, miss);
        let element = self.array_element(layout, table, key, miss);
        let store = self.builder.create_block();
        let check_metatable = self.builder.create_block();
        let tag = self.builder.ins().load(types::I8, self.heap_flags, element, 0);
        let present = self.builder.ins().icmp_imm_s(IntCC::NotEqual, tag, NIL_TAG);
        self.builder.ins().brif(present, store, &[], check_metatable, &[]);
        self.builder.switch_to_block(check_metatable);
        let metatable_tag = self.builder.ins().uload32(self.heap_flags, table, layout.metatable_tag);
        let none = self.builder.ins().icmp_imm_s(IntCC::Equal, metatable_tag, i64::from(layout.metatable_none));
        self.builder.ins().brif(none, store, &[], miss, &[]);
        self.builder.switch_to_block(store);
        let value = self.store_operand(c, k);
        self.builder.ins().store(self.heap_flags, value.0, element, 0);
        self.builder.ins().store(self.heap_flags, value.1, element, PAYLOAD_OFFSET);
        self.table_barrier(pc, a, c, k, value.0);
        self.builder.switch_to_block(miss);
        self.fast_helper(pc, key.set_helper(), arguments, pc + 1);
    }

    fn array_element(&mut self, layout: TableLayout, table: Value, key: ArrayKey, miss: Block) -> Value {
        let index = match key {
            ArrayKey::Constant(index) => self.builder.ins().iconst(types::I64, i64::from(index)),
            ArrayKey::Register(register) => {
                let base = self.registers();
                let tag = self.builder.ins().load(types::I8, self.value_flags, base, slot(register));
                let is_integer = self.builder.ins().icmp_imm_s(IntCC::Equal, tag, INT_TAG);
                let next = self.builder.create_block();
                self.builder.ins().brif(is_integer, next, &[], miss, &[]);
                self.builder.switch_to_block(next);
                self.builder.ins().load(types::I64, self.value_flags, base, slot(register) + PAYLOAD_OFFSET)
            }
        };
        let position = self.builder.ins().iadd_imm_s(index, -1);
        let length = self.builder.ins().load(types::I64, self.heap_flags, table, layout.array_len);
        let inside = self.builder.ins().icmp(IntCC::UnsignedLessThan, position, length);
        let next = self.builder.create_block();
        self.builder.ins().brif(inside, next, &[], miss, &[]);
        self.builder.switch_to_block(next);
        let elements = self.builder.ins().load(self.pointer, self.heap_flags, table, layout.array);
        let offset = self.builder.ins().imul_imm_s(position, i64::from(VALUE_SIZE));
        self.builder.ins().iadd(elements, offset)
    }

    fn call_step(&mut self, pc: usize) {
        self.call_helper(pc, self.helpers.step);
    }

    fn fast_helper(&mut self, pc: usize, which: usize, args: [u32; 3], next: usize) {
        let result = self.call_fast(pc, which, args);
        self.dispatch_result(result, Some(next));
    }

    fn for_prep_helper(&mut self, pc: usize, a: u8, bx: u32) {
        self.fast_helper(pc, FAST_FOR_PREP, [u32::from(a), bx, 0], pc + 1);
    }

    fn control_helper(&mut self, pc: usize, which: usize, args: [u32; 3], next: Option<usize>) {
        let result = self.call_fast(pc, which, args);
        self.dispatch_result(result, next);
    }

    fn call_fast(&mut self, pc: usize, which: usize, args: [u32; 3]) -> Value {
        let pc_value = self.builder.ins().iconst(types::I32, pc as i64);
        let values: Vec<Value> = args.iter().map(|&arg| self.builder.ins().iconst(types::I32, i64::from(arg as i32))).collect();
        let helper = self.helpers.fast[which];
        let ctx = self.ctx;
        let call = self.builder.ins().call(helper, &[ctx, pc_value, values[0], values[1], values[2]]);
        self.builder.inst_results(call)[0]
    }

    fn call_helper(&mut self, pc: usize, helper: FuncRef) {
        let pc_value = self.builder.ins().iconst(types::I32, pc as i64);
        let call = self.builder.ins().call(helper, &[self.ctx, pc_value]);
        let result = self.builder.inst_results(call)[0];
        let next = (pc + 1 < self.blocks.len()).then_some(pc + 1);
        self.dispatch_result(result, next);
    }

    fn dispatch_result(&mut self, result: Value, expected: Option<usize>) {
        let Some(expected) = expected else {
            self.builder.ins().jump(self.dispatch, &[BlockArg::Value(result)]);
            return;
        };
        let matched = self.builder.ins().icmp_imm_s(IntCC::Equal, result, expected as i64);
        let destination = self.blocks[expected];
        self.builder.ins().brif(matched, destination, &[], self.dispatch, &[BlockArg::Value(result)]);
    }

    fn operands_test(&mut self, left: Operand, right: Operand, tag: i64) -> Test {
        let first = self.tag_test(left, tag);
        let second = self.tag_test(right, tag);
        self.combine(first, second)
    }

    fn tag_test(&mut self, operand: Operand, tag: i64) -> Test {
        match operand {
            Operand::Register(register) => {
                let base = self.registers();
                let actual = self.builder.ins().load(types::I8, self.value_flags, base, slot(register));
                Test::When(self.builder.ins().icmp_imm_s(IntCC::Equal, actual, tag))
            }
            Operand::Int(_) if tag == INT_TAG => Test::Always,
            Operand::Float(_) if tag == FLOAT_TAG => Test::Always,
            Operand::Int(_) | Operand::Float(_) => Test::Never,
        }
    }

    fn combine(&mut self, first: Test, second: Test) -> Test {
        match (first, second) {
            (Test::Never, _) | (_, Test::Never) => Test::Never,
            (Test::Always, test) | (test, Test::Always) => test,
            (Test::When(x), Test::When(y)) => Test::When(self.builder.ins().band(x, y)),
        }
    }

    fn read_int(&mut self, base: Value, operand: Operand) -> Value {
        match operand {
            Operand::Register(register) => self.builder.ins().load(types::I64, self.value_flags, base, slot(register) + PAYLOAD_OFFSET),
            Operand::Int(integer) => self.builder.ins().iconst(types::I64, integer),
            Operand::Float(_) => unreachable!("an integer path reads only integer operands"),
        }
    }

    fn read_float(&mut self, base: Value, operand: Operand) -> Value {
        match operand {
            Operand::Register(register) => self.builder.ins().load(types::F64, self.value_flags, base, slot(register) + PAYLOAD_OFFSET),
            Operand::Float(float) => self.builder.ins().f64const(float),
            Operand::Int(_) => unreachable!("a float path reads only float operands"),
        }
    }

    fn store_int(&mut self, base: Value, offset: i32, value: Value) {
        let tag = self.builder.ins().iconst(types::I8, INT_TAG);
        self.builder.ins().store(self.value_flags, tag, base, offset);
        self.builder.ins().store(self.value_flags, value, base, offset + PAYLOAD_OFFSET);
    }

    fn store_float(&mut self, base: Value, offset: i32, value: Value) {
        let tag = self.builder.ins().iconst(types::I8, FLOAT_TAG);
        self.builder.ins().store(self.value_flags, tag, base, offset);
        self.builder.ins().store(self.value_flags, value, base, offset + PAYLOAD_OFFSET);
    }

    fn store_bool(&mut self, base: Value, offset: i32, flag: Value) {
        let tag = self.builder.ins().iconst(types::I8, BOOL_TAG);
        self.builder.ins().store(self.value_flags, tag, base, offset);
        self.builder.ins().store(self.value_flags, flag, base, offset + BOOL_OFFSET);
    }

    fn falsy(&mut self, base: Value, offset: i32) -> Value {
        let tag = self.builder.ins().load(types::I8, self.value_flags, base, offset);
        let flag = self.builder.ins().load(types::I8, self.value_flags, base, offset + BOOL_OFFSET);
        let nil = self.builder.ins().icmp_imm_s(IntCC::Equal, tag, NIL_TAG);
        let is_bool = self.builder.ins().icmp_imm_s(IntCC::Equal, tag, BOOL_TAG);
        let is_false = self.builder.ins().icmp_imm_s(IntCC::Equal, flag, 0);
        let false_value = self.builder.ins().band(is_bool, is_false);
        self.builder.ins().bor(nil, false_value)
    }

    fn truthy(&mut self, base: Value, offset: i32) -> Value {
        let falsy = self.falsy(base, offset);
        self.builder.ins().icmp_imm_s(IntCC::Equal, falsy, 0)
    }

    fn copy_value(&mut self, source: Value, from: i32, destination: Value, to: i32) {
        let flags = self.value_flags;
        let low = self.builder.ins().load(types::I64, flags, source, from);
        let high = self.builder.ins().load(types::I64, flags, source, from + PAYLOAD_OFFSET);
        self.builder.ins().store(flags, low, destination, to);
        self.builder.ins().store(flags, high, destination, to + PAYLOAD_OFFSET);
    }
}

fn immediate_compare(a: u8, sb: i32, c: u8, reversed: bool, int_cc: IntCC, float_cc: FloatCC) -> Compare {
    let register = Operand::Register(a);
    if c == 0 {
        return Compare {
            int: Some(ordered(register, Operand::Int(i64::from(sb)), reversed)),
            float: None,
            int_cc,
            float_cc,
        };
    }
    Compare {
        int: None,
        float: Some(ordered(register, Operand::Float(f64::from(sb)), reversed)),
        int_cc,
        float_cc,
    }
}
