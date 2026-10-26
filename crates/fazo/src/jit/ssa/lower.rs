//! Helpers answer the next pc or a negative exit code.

use std::mem::offset_of;

use cranelift_codegen::ir::{
    AbiParam, AliasRegionData, Block, BlockArg, FuncRef, Function, InstBuilder, MemFlagsData, SigRef, StackSlot, StackSlotData,
    StackSlotKind, TrapCode, Type as ClifType, Value, condcodes::{FloatCC, IntCC}, types,
};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::JITModule;
use cranelift_module::{FuncId, Linkage, Module};

use super::{
    DeoptPoint,
    ir::{BinOp, BlockId, BuiltinFn, Cc, Closure, Const, Func, HelperKind, NumOp, Op, PureFn, Rep, StateId, Term, Type, UnOp, Val},
    runtime,
};
use crate::jit::{
    FrameLayout, JitCtx, JitResult, TableLayout,
    layout::{lua_frame, upvalue},
    sites::{self, CallTarget},
};

const NIL_TAG: i64 = 0;
const BOOL_TAG: i64 = 1;
const INT_TAG: i64 = 2;
const FLOAT_TAG: i64 = 3;
const STRING_TAG: i64 = 4;
const NODE_SIZE: i64 = crate::table::offsets::NODE_SIZE as i64;
const NODE_KEY: i64 = crate::table::offsets::NODE_KEY as i64;
const NODE_VALUE: i64 = crate::table::offsets::NODE_VALUE as i64;
const TABLE_TAG: i64 = 5;
const CLOSURE_TAG: i64 = 6;
const LIGHT_USERDATA_TAG: i64 = 10;
const VALUE_SIZE: i32 = 16;
const PAYLOAD: i32 = 8;

pub(crate) struct Env<'a> {
    pub(crate) layout: TableLayout,
    pub(crate) frames: FrameLayout,
    pub(crate) calls: &'a [Vec<CallTarget>],
    pub(crate) call_depth_limit: u32,
    pub(crate) points: &'a [DeoptPoint],
    pub(crate) stress_exits: bool,
    pub(crate) tier_budget: Option<*const std::cell::Cell<i64>>,
}

#[derive(Clone, Copy)]
enum Word {
    None,
    One(Value),
    Two(Value, Value),
}

struct Imports {
    step: FuncRef,
    fast: Vec<FuncRef>,
    deopt: FuncRef,
    safe_point: FuncRef,
    resume: FuncRef,
    value: FuncRef,
    pure_float: FuncRef,
    pure_int: FuncRef,
    loop_count: FuncRef,
    mixed: FuncRef,
    new_table: FuncRef,
    materialize: FuncRef,
    entry_signature: SigRef,
}

pub(crate) fn symbols() -> Vec<(&'static str, *const u8)> {
    let mut symbols = vec![(sites::STEP, crate::jit::helpers::step as extern "C" fn(*mut JitCtx, u32) -> i64 as *const u8)];
    symbols.extend(sites::fast_symbols());
    symbols.extend(runtime::symbols());
    symbols
}

pub(crate) fn define(module: &mut JITModule, func: &mut Function, context: &mut FunctionBuilderContext, ir: &Func, env: &Env<'_>) -> JitResult<FuncId> {
    let config = module.target_config();
    let pointer = config.pointer_type();
    let mut signature = module.make_signature();
    signature.params.push(AbiParam::new(pointer));
    signature.params.push(AbiParam::new(types::I32));
    signature.returns.push(AbiParam::new(types::I64));
    func.signature = signature.clone();
    let id = module.declare_function(sites::ENTRY, Linkage::Export, &signature)?;
    let step = declare(module, func, sites::STEP, &signature)?;
    let mut fast_signature = module.make_signature();
    fast_signature.params.push(AbiParam::new(pointer));
    fast_signature.params.extend([AbiParam::new(types::I32); 4]);
    fast_signature.returns.push(AbiParam::new(types::I64));
    let mut fast = Vec::with_capacity(sites::FAST_NAMES.len());
    for name in sites::FAST_NAMES {
        fast.push(declare(module, func, name, &fast_signature)?);
    }
    let make = |module: &mut JITModule, params: &[ClifType], returns: ClifType| {
        let mut signature = module.make_signature();
        signature.params.extend(params.iter().map(|&kind| AbiParam::new(kind)));
        signature.returns.push(AbiParam::new(returns));
        signature
    };
    let deopt = make(module, &[pointer, types::I64], types::I64);
    let safe_point = make(module, &[pointer, types::I32], types::I64);
    let value = make(module, &[pointer, types::I32, pointer, types::I32], types::I64);
    let pure_float = make(module, &[types::I32, types::F64, types::F64], types::F64);
    let pure_int = make(module, &[types::I32, types::I64, types::I64], types::I64);
    let loop_count = make(module, &[types::I64, types::I64, types::I64], types::I64);
    let mixed = make(module, &[types::I32, types::I64, types::F64], types::I64);
    let new_table = make(module, &[pointer, types::I32, types::I32, types::I32], types::I64);
    let materialize = make(module, &[pointer, types::I64, types::I32, pointer, types::I32], types::I64);
    let imports = Imports {
        step,
        fast,
        deopt: declare(module, func, runtime::DEOPT, &deopt)?,
        safe_point: declare(module, func, runtime::SAFE_POINT, &safe_point)?,
        resume: declare(module, func, runtime::RESUME, &deopt)?,
        value: declare(module, func, runtime::VALUE, &value)?,
        pure_float: declare(module, func, runtime::PURE_FLOAT, &pure_float)?,
        pure_int: declare(module, func, runtime::PURE_INT, &pure_int)?,
        loop_count: declare(module, func, runtime::LOOP_COUNT, &loop_count)?,
        mixed: declare(module, func, runtime::MIXED, &mixed)?,
        new_table: declare(module, func, runtime::NEW_TABLE, &new_table)?,
        materialize: declare(module, func, runtime::MATERIALIZE, &materialize)?,
        entry_signature: func.import_signature(signature),
    };
    let builder = FunctionBuilder::new(func, context);
    let mut lowering = Lowering::new(builder, ir, env, imports, pointer);
    lowering.lower();
    lowering.builder.finalize(config);
    Ok(id)
}

fn declare(module: &mut JITModule, func: &mut Function, name: &str, signature: &cranelift_codegen::ir::Signature) -> JitResult<FuncRef> {
    let id = module.declare_function(name, Linkage::Import, signature)?;
    Ok(module.declare_func_in_func(id, func))
}

struct Lowering<'a, 'f> {
    builder: FunctionBuilder<'f>,
    ir: &'a Func,
    env: &'a Env<'a>,
    imports: Imports,
    pointer: ClifType,
    ctx: Value,
    blocks: Vec<Block>,
    values: Vec<Word>,
    scratch: StackSlot,
    callee: Option<StackSlot>,
    stubs: Vec<Option<Block>>,
    context_flags: MemFlagsData,
    value_flags: MemFlagsData,
    heap_flags: MemFlagsData,
    frame_flags: MemFlagsData,
}

/// Lua accesses are `notrap`, and separate alias regions keep context loads across stores to registers and tables.
fn alias_flags(builder: &mut FunctionBuilder<'_>, user_id: u32, description: &'static str) -> MemFlagsData {
    let region = builder.func.dfg.alias_regions.insert(AliasRegionData {
        user_id,
        description: description.into(),
    });
    MemFlagsData::new().with_notrap().with_alias_region(Some(region))
}

fn slot_offset(slot: u32) -> i32 {
    slot as i32 * VALUE_SIZE
}

fn word_type(rep: Rep) -> ClifType {
    match rep {
        Rep::Float => types::F64,
        Rep::Bool => types::I8,
        _ => types::I64,
    }
}

impl<'a, 'f> Lowering<'a, 'f> {
    fn new(mut builder: FunctionBuilder<'f>, ir: &'a Func, env: &'a Env<'a>, imports: Imports, pointer: ClifType) -> Self {
        let entry = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);
        let ctx = builder.block_params(entry)[0];
        let context_flags = alias_flags(&mut builder, 0, "context");
        let value_flags = alias_flags(&mut builder, 1, "lua values");
        let heap_flags = alias_flags(&mut builder, 2, "tables");
        let frame_flags = alias_flags(&mut builder, 3, "threads and frames");
        let scratch = builder.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 3 * VALUE_SIZE as u32, 4));
        let blocks: Vec<Block> = (0..ir.blocks.len()).map(|_| builder.create_block()).collect();
        let mut lowering = Self {
            builder,
            ir,
            env,
            imports,
            pointer,
            ctx,
            blocks,
            values: vec![Word::None; ir.insts.len()],
            scratch,
            callee: None,
            stubs: vec![None; ir.states.len()],
            context_flags,
            value_flags,
            heap_flags,
            frame_flags,
        };
        lowering.declare_params();
        lowering.dispatch_entry(entry);
        lowering
    }

    fn declare_params(&mut self) {
        for (index, block) in self.ir.blocks.iter().enumerate() {
            let target = self.blocks[index];
            if block.cold {
                self.builder.set_cold_block(target);
            }
            for &param in &block.params {
                let rep = self.ir.rep(param);
                self.values[param.0 as usize] = match rep {
                    Rep::None => Word::None,
                    Rep::Boxed => {
                        let tag = self.builder.append_block_param(target, types::I64);
                        let payload = self.builder.append_block_param(target, types::I64);
                        Word::Two(tag, payload)
                    }
                    other => Word::One(self.builder.append_block_param(target, word_type(other))),
                };
            }
        }
    }

    /// Compiled code is entered at the first instruction or at a loop header; the driver enters nowhere else.
    fn dispatch_entry(&mut self, entry: Block) {
        let pc = self.builder.block_params(entry)[1];
        let pc = self.builder.ins().uextend(types::I64, pc);
        let mut targets: Vec<(u32, BlockId)> = self.ir.entry.map(|block| (0, block)).into_iter().collect();
        targets.extend(self.ir.osr.iter().copied().filter(|&(at, _)| at != 0));
        for (at, block) in targets {
            let next = self.builder.create_block();
            let here = self.builder.ins().icmp_imm_s(IntCC::Equal, pc, i64::from(at));
            self.builder.ins().brif(here, self.blocks[block.0 as usize], &[], next, &[]);
            self.builder.switch_to_block(next);
        }
        self.builder.ins().trap(TrapCode::unwrap_user(1));
    }

    fn lower(&mut self) {
        for block in self.ir.reverse_postorder() {
            self.builder.switch_to_block(self.blocks[block.0 as usize]);
            let data = self.ir.block(block);
            for &value in &data.insts {
                self.lower_inst(value);
            }
            self.lower_term(&data.term.clone());
        }
        self.fill_stubs();
        self.builder.seal_all_blocks();
    }

    fn word(&self, value: Val) -> Word {
        self.values[value.0 as usize]
    }

    fn one(&self, value: Val) -> Value {
        match self.word(value) {
            Word::One(word) => word,
            Word::Two(_, payload) => payload,
            Word::None => unreachable!("an operand with no machine value"),
        }
    }

    fn pair(&mut self, value: Val) -> (Value, Value) {
        match self.word(value) {
            Word::Two(tag, payload) => (tag, payload),
            Word::One(_) => self.boxed(value),
            Word::None => unreachable!("an operand with no machine value"),
        }
    }

    fn set(&mut self, value: Val, word: Word) {
        self.values[value.0 as usize] = word;
    }

    fn iconst(&mut self, kind: ClifType, value: i64) -> Value {
        self.builder.ins().iconst(kind, value)
    }

    fn ctx_load(&mut self, kind: ClifType, offset: usize) -> Value {
        self.builder.ins().load(kind, self.context_flags, self.ctx, offset as i32)
    }

    fn registers(&mut self) -> Value {
        self.ctx_load(self.pointer, offset_of!(JitCtx, registers))
    }

    fn tag_of(&mut self, tag_word: Value) -> Value {
        self.builder.ins().band_imm_s(tag_word, 0xff)
    }

    fn boxed(&mut self, value: Val) -> (Value, Value) {
        let word = self.one(value);
        let rep = self.ir.rep(value);
        let ty = self.ir.ty(value);
        match rep {
            Rep::Int => (self.iconst(types::I64, INT_TAG), word),
            Rep::Float => {
                let bits = self.builder.ins().bitcast(types::I64, MemFlagsData::new(), word);
                (self.iconst(types::I64, FLOAT_TAG), bits)
            }
            Rep::Bool => {
                let wide = self.builder.ins().uextend(types::I64, word);
                (self.iconst(types::I64, BOOL_TAG), wide)
            }
            _ => {
                let tag = ty.tag().map_or(NIL_TAG, i64::from);
                (self.iconst(types::I64, tag), word)
            }
        }
    }

    fn unbox(&mut self, (_, payload): (Value, Value), rep: Rep) -> Word {
        match rep {
            Rep::Float => Word::One(self.builder.ins().bitcast(types::F64, MemFlagsData::new(), payload)),
            Rep::Bool => Word::One(self.builder.ins().ireduce(types::I8, payload)),
            Rep::None => Word::None,
            _ => Word::One(payload),
        }
    }

    fn convert(&mut self, value: Val, rep: Rep) -> Word {
        let from = self.ir.rep(value);
        if from == rep {
            return self.word(value);
        }
        if rep == Rep::Boxed {
            let (tag, payload) = self.boxed(value);
            return Word::Two(tag, payload);
        }
        if from == Rep::Boxed {
            let pair = self.pair(value);
            return self.unbox(pair, rep);
        }
        self.word(value)
    }

    fn load_pair(&mut self, flags: MemFlagsData, base: Value, offset: i32) -> (Value, Value) {
        let tag = self.builder.ins().load(types::I64, flags, base, offset);
        let payload = self.builder.ins().load(types::I64, flags, base, offset + PAYLOAD);
        (tag, payload)
    }

    fn store_pair(&mut self, flags: MemFlagsData, (tag, payload): (Value, Value), base: Value, offset: i32) {
        self.builder.ins().store(flags, tag, base, offset);
        self.builder.ins().store(flags, payload, base, offset + PAYLOAD);
    }

    fn table_address(&mut self, payload: Value) -> Value {
        let index = self.builder.ins().band_imm_s(payload, 0xffff_ffff);
        let tables = self.ctx_load(self.pointer, offset_of!(JitCtx, tables));
        let offset = self.builder.ins().imul_imm_s(index, self.env.layout.size);
        self.builder.ins().iadd(tables, offset)
    }

    fn closure_address(&mut self, payload: Value) -> Value {
        let index = self.builder.ins().band_imm_s(payload, 0xffff_ffff);
        let closures = self.ctx_load(self.pointer, offset_of!(JitCtx, closures));
        let offset = self.builder.ins().imul_imm_s(index, std::mem::size_of::<crate::object::Closure>() as i64);
        self.builder.ins().iadd(closures, offset)
    }

    fn continue_if(&mut self, cond: Value, otherwise: Block) {
        let next = self.builder.create_block();
        self.builder.ins().brif(cond, next, &[], otherwise, &[]);
        self.builder.switch_to_block(next);
    }

    fn guard(&mut self, cond: Value, state: StateId) {
        let stub = self.stub(state);
        let next = self.builder.create_block();
        self.builder.ins().brif(cond, next, &[], stub, &[]);
        self.builder.switch_to_block(next);
    }

    /// Guards that resume at the same frame state share one exit.
    fn stub(&mut self, state: StateId) -> Block {
        if let Some(stub) = self.stubs[state.0 as usize] {
            return stub;
        }
        let stub = self.builder.create_block();
        self.builder.set_cold_block(stub);
        self.stubs[state.0 as usize] = Some(stub);
        stub
    }

    fn fill_stubs(&mut self) {
        for index in 0..self.stubs.len() {
            let Some(stub) = self.stubs[index] else {
                continue;
            };
            self.builder.switch_to_block(stub);
            self.deoptimize(StateId(index as u32));
        }
    }

    /// Sunk tables are built last, after every other register is in place.
    fn store_state(&mut self, state: StateId) {
        let data = self.ir.state(state).clone();
        if data.slots.is_empty() {
            return;
        }
        let registers = self.registers();
        for &(slot, value) in &data.slots {
            if data.object(value).is_some() {
                continue;
            }
            let pair = self.pair(value);
            self.store_pair(self.value_flags, pair, registers, slot_offset(slot));
        }
        for &(slot, value) in &data.slots {
            let Some(object) = data.object(value) else {
                continue;
            };
            let table = self.materialize(object);
            let tag = self.iconst(types::I64, TABLE_TAG);
            let registers = self.registers();
            self.store_pair(self.value_flags, (tag, table), registers, slot_offset(slot));
        }
    }

    fn materialize(&mut self, object: &super::ir::Virtual) -> Value {
        let meta = match object.meta {
            Some(meta) => self.pair(meta),
            None => (self.iconst(types::I64, NIL_TAG), self.iconst(types::I64, 0)),
        };
        let fields: Vec<(Value, Value)> = object.fields.iter().map(|&field| self.pair(field)).collect();
        let count = fields.len() + 1;
        let buffer = self.builder.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, count as u32 * VALUE_SIZE as u32, 4));
        let address = self.builder.ins().stack_addr(self.pointer, buffer, 0);
        for (index, pair) in std::iter::once(meta).chain(fields).enumerate() {
            self.store_pair(MemFlagsData::new().with_notrap(), pair, address, index as i32 * VALUE_SIZE);
        }
        let shape = self.iconst(types::I64, object.shape as i64);
        let hash_log = self.iconst(types::I32, i64::from(object.hash_log));
        let count = self.iconst(types::I32, count as i64);
        let call = self.builder.ins().call(self.imports.materialize, &[self.ctx, shape, hash_log, address, count]);
        let built = self.builder.inst_results(call)[0];
        let Some((flag, original)) = object.original else {
            return built;
        };
        let flag = self.one(flag);
        let original = self.one(original);
        self.builder.ins().select(flag, built, original)
    }

    fn deoptimize(&mut self, state: StateId) {
        self.store_state(state);
        let point = std::ptr::from_ref(&self.env.points[state.0 as usize]) as i64;
        let point = self.iconst(types::I64, point);
        let call = self.builder.ins().call(self.imports.deopt, &[self.ctx, point]);
        let code = self.builder.inst_results(call)[0];
        self.builder.ins().return_(&[code]);
    }

    fn exit_with(&mut self, result: Value) {
        let leaving = self.builder.create_block();
        let resume = self.builder.create_block();
        let negative = self.builder.ins().icmp_imm_s(IntCC::SignedLessThan, result, 0);
        self.builder.ins().brif(negative, leaving, &[], resume, &[]);
        self.builder.switch_to_block(leaving);
        let code = self.builder.ins().ineg(result);
        self.builder.ins().return_(&[code]);
        self.builder.switch_to_block(resume);
        self.builder.set_cold_block(resume);
        let call = self.builder.ins().call(self.imports.resume, &[self.ctx, result]);
        let answer = self.builder.inst_results(call)[0];
        let code = self.builder.ins().ineg(answer);
        self.builder.ins().return_(&[code]);
    }

    fn scratch_address(&mut self) -> Value {
        self.builder.ins().stack_addr(self.pointer, self.scratch, 0)
    }

    fn scratch_store(&mut self, address: Value, index: i32, pair: (Value, Value)) {
        self.store_pair(MemFlagsData::new().with_notrap(), pair, address, index * VALUE_SIZE);
    }

    fn scratch_load(&mut self, index: i32) -> (Value, Value) {
        let address = self.scratch_address();
        self.load_pair(MemFlagsData::new().with_notrap(), address, index * VALUE_SIZE)
    }

    fn call_value(&mut self, kind: u32, operands: &[(Value, Value)], extra: u32) -> Value {
        let address = self.scratch_address();
        for (index, &pair) in operands.iter().enumerate() {
            self.scratch_store(address, index as i32, pair);
        }
        let kind = self.iconst(types::I32, i64::from(kind));
        let extra = self.iconst(types::I32, i64::from(extra as i32));
        let call = self.builder.ins().call(self.imports.value, &[self.ctx, kind, address, extra]);
        self.builder.inst_results(call)[0]
    }

    /// `rawget` of a non-table and a refused `setmetatable` answer zero and resume the call in the interpreter.
    fn builtin(&mut self, function: BuiltinFn, args: &[Val], state: StateId) -> Word {
        let pairs: Vec<(Value, Value)> = args.iter().map(|&arg| self.pair(arg)).collect();
        let ok = self.call_value(runtime::KIND_BUILTIN, &pairs, runtime::builtin_code(function));
        if matches!(function, BuiltinFn::RawGet | BuiltinFn::SetMetatable) {
            self.guard(ok, state);
        }
        let (tag, payload) = self.scratch_load(0);
        Word::Two(tag, payload)
    }

    /// Every register the frame state holds is written back first, so the collector step sees it.
    fn new_table(&mut self, value: Val, (hash_log, top): (u8, u32), (shape, fields): (Option<u64>, &[Val]), (state, after): (StateId, StateId)) -> Word {
        self.store_state(state);
        let pc = self.iconst(types::I32, i64::from(self.ir.state(state).frames.first().map_or(self.ir.state(state).pc, |frame| frame.caller_pc)));
        let hash_log = self.iconst(types::I32, i64::from(hash_log));
        let top = self.iconst(types::I32, i64::from(top));
        let call = self.builder.ins().call(self.imports.new_table, &[self.ctx, pc, hash_log, top]);
        let answer = self.builder.inst_results(call)[0];
        let allocated = self.builder.ins().icmp_imm_s(IntCC::SignedGreaterThanOrEqual, answer, 0);
        self.guard(allocated, state);
        let table = self.builder.ins().band_imm_s(answer, !runtime::RAN_CODE);
        self.set(value, Word::One(table));
        if let Some(shape) = shape {
            self.fill_fresh_table(value, shape, fields);
        }
        let ran = self.builder.ins().band_imm_s(answer, runtime::RAN_CODE);
        let quiet = self.builder.ins().icmp_imm_s(IntCC::Equal, ran, 0);
        self.guard(quiet, after);
        Word::One(table)
    }

    /// The fresh table is white and its slots were reserved for the hash part, so the stores need no barrier or growth.
    fn fill_fresh_table(&mut self, table: Val, shape: u64, fields: &[Val]) {
        let layout = self.env.layout;
        let pairs: Vec<(Value, Value)> = fields.iter().map(|&field| self.pair(field)).collect();
        let address = self.table_address(self.one(table));
        let slots = self.builder.ins().load(self.pointer, self.heap_flags, address, layout.slots);
        for (index, pair) in pairs.into_iter().enumerate() {
            self.store_pair(self.heap_flags, pair, slots, index as i32 * VALUE_SIZE);
        }
        let length = self.iconst(types::I64, fields.len() as i64);
        self.builder.ins().store(self.heap_flags, length, address, layout.slots_len);
        let shape = self.iconst(types::I64, shape as i64);
        self.builder.ins().store(self.heap_flags, shape, address, layout.shape);
    }

    fn table_pair(&mut self, table: Val) -> (Value, Value) {
        let payload = self.one(table);
        (self.iconst(types::I64, TABLE_TAG), payload)
    }

    fn lower_inst(&mut self, value: Val) {
        let inst = self.ir.inst(value);
        let rep = inst.rep;
        let word = match inst.op.clone() {
            Op::Param => return,
            Op::Const(constant) => self.constant(constant, rep),
            Op::LoadReg { slot } => {
                let registers = self.registers();
                let (tag, payload) = self.load_pair(self.value_flags, registers, slot_offset(slot));
                Word::Two(tag, payload)
            }
            Op::StoreReg { slot, value } => {
                let pair = self.pair(value);
                let registers = self.registers();
                self.store_pair(self.value_flags, pair, registers, slot_offset(slot));
                Word::None
            }
            Op::LoadUpval { closure, index } => {
                let address = self.upvalue_address(closure, index);
                let (tag, payload) = self.load_pair(MemFlagsData::new().with_notrap(), address, 0);
                Word::Two(tag, payload)
            }
            Op::StoreUpval { closure, index, value } => {
                self.store_upvalue(closure, index, value);
                Word::None
            }
            Op::Unary(op, operand) => self.unary(op, operand, value),
            Op::Binary(op, left, right) => Word::One(self.binary(op, left, right)),
            Op::Pure(function, args) => Word::One(self.pure(function, &args)),
            Op::EqConst(operand, constant) => Word::One(self.equal_constant(operand, constant)),
            Op::Select { cond, yes, no } => self.select(cond, yes, no, rep),
            Op::Arith { op, left, right, state } => self.arith(op, (left, right), state),
            Op::Compare { cc, left, right, state } => Word::One(self.compare(cc, (left, right), state)),
            Op::Len { value: operand, state } => {
                let pair = self.pair(operand);
                let ok = self.call_value(runtime::KIND_LEN, &[pair], 0);
                self.guard(ok, state);
                let (_, payload) = self.scratch_load(0);
                Word::One(payload)
            }
            Op::ShapeIs { table, shape } => Word::One(self.shape_is(table, shape)),
            Op::LoadSlot { table, slot } => {
                let address = self.slot_address(table, slot);
                let (tag, payload) = self.load_pair(self.heap_flags, address, 0);
                Word::Two(tag, payload)
            }
            Op::LoadProto { closure } => {
                let address = self.closure_address(self.one(closure));
                Word::One(self.builder.ins().load(types::I64, self.heap_flags, address, offset_of!(crate::object::Closure, proto) as i32))
            }
            Op::LoadArray { table, index, state } => self.load_array(table, index, state),
            Op::GuardType { value: input, ty, state } => self.guard_type(input, ty, state, rep),
            Op::GuardShape { table, shape, state } => {
                let same = self.shape_is(table, shape);
                self.guard(same, state);
                Word::None
            }
            Op::GuardHasMeta { table, state } => {
                let address = self.table_address(self.one(table));
                let tag = self.builder.ins().uload32(self.heap_flags, address, self.env.layout.metatable_tag);
                let some = self.builder.ins().icmp_imm_s(IntCC::NotEqual, tag, i64::from(self.env.layout.metatable_none));
                self.guard(some, state);
                Word::One(self.builder.ins().load(types::I64, self.heap_flags, address, self.env.layout.metatable_index))
            }
            Op::GuardNoMeta { table, bit, state } => {
                let absent = self.absent(table, bit);
                self.guard(absent, state);
                Word::None
            }
            Op::GuardEq { value: input, expected, state } => {
                let word = self.one(input);
                let same = self.builder.ins().icmp_imm_s(IntCC::Equal, word, expected as i64);
                self.guard(same, state);
                Word::None
            }
            Op::GuardFalse { cond, state } => {
                let flag = self.one(cond);
                let inverted = self.builder.ins().bxor_imm_s(flag, 1);
                self.guard(inverted, state);
                Word::None
            }
            Op::GuardNotNil { value: input, state } => {
                if self.ir.rep(input) == Rep::Boxed {
                    let (tag, _) = self.pair(input);
                    let tag = self.tag_of(tag);
                    let present = self.builder.ins().icmp_imm_s(IntCC::NotEqual, tag, NIL_TAG);
                    self.guard(present, state);
                }
                self.convert(input, rep)
            }
            Op::GuardRange { table, low, high, state } => {
                let low_inside = self.inside(table, self.one(low));
                let high_inside = self.inside(table, self.one(high));
                let both = self.builder.ins().band(low_inside, high_inside);
                self.guard(both, state);
                Word::None
            }
            Op::Index { object, key, state } => self.index(object, key, state),
            Op::NewIndex { object, key, value: stored, state } => {
                self.new_index(object, key, stored, state);
                Word::None
            }
            Op::AddSlot { table, key, shape, next, slot, value: stored, state } => {
                self.add_slot(table, key, (shape, next, slot), stored, state);
                Word::None
            }
            Op::StoreSlot { table, slot, value: stored, state } => {
                let address = self.slot_address(table, slot);
                let tag = self.builder.ins().uload8(types::I64, self.heap_flags, address, 0);
                let present = self.builder.ins().icmp_imm_s(IntCC::NotEqual, tag, NIL_TAG);
                self.guard(present, state);
                let pair = self.pair(stored);
                self.store_pair(self.heap_flags, pair, address, 0);
                if self.may_collect(stored) {
                    self.barrier(table, pair);
                }
                Word::None
            }
            Op::GuardNode { table, node, nodes, key, state } => {
                let found = self.node_holds(table, (node, nodes), key);
                self.guard(found, state);
                Word::None
            }
            Op::LoadNode { table, node } => {
                let address = self.node_value_address(table, node);
                let (tag, payload) = self.load_pair(self.heap_flags, address, 0);
                Word::Two(tag, payload)
            }
            Op::StoreNode { table, node, value: stored, state } => {
                let address = self.node_value_address(table, node);
                let tag = self.builder.ins().uload8(types::I64, self.heap_flags, address, 0);
                let present = self.builder.ins().icmp_imm_s(IntCC::NotEqual, tag, NIL_TAG);
                self.guard(present, state);
                let pair = self.pair(stored);
                self.store_pair(self.heap_flags, pair, address, 0);
                if self.may_collect(stored) {
                    self.barrier(table, pair);
                }
                Word::None
            }
            Op::Helper(helper) => Word::One(self.helper(helper.pc, helper.kind, helper.args)),
            Op::Builtin { function, args, state } => self.builtin(function, &args, state),
            Op::NewTable { hash_log, top, shape, fields, state, after } => self.new_table(value, (hash_log, top), (shape, &fields), (state, after)),
            Op::Materialize { object } => Word::One(self.materialize(&object)),
            Op::GuardAbsent { table, bit, state } => {
                let layout = self.env.layout;
                let address = self.table_address(self.one(table));
                let absent = self.builder.ins().uload8(types::I64, self.heap_flags, address, layout.absent);
                let flagged = self.builder.ins().band_imm_s(absent, i64::from(bit));
                self.guard(flagged, state);
                Word::None
            }
            Op::CallLua { pc, a, b, c } => Word::One(self.call_lua(pc as usize, (a, b, c))),
            Op::Safepoint { pc, state, .. } => {
                self.safepoint(pc, state);
                Word::None
            }
        };
        self.set(value, word);
    }

    fn constant(&mut self, constant: Const, rep: Rep) -> Word {
        match rep {
            Rep::Int | Rep::Payload | Rep::Word => Word::One(self.iconst(types::I64, constant.payload() as i64)),
            Rep::Float => Word::One(self.builder.ins().f64const(f64::from_bits(constant.payload()))),
            Rep::Bool => Word::One(self.iconst(types::I8, constant.payload() as i64)),
            Rep::Boxed => {
                let tag = self.iconst(types::I64, i64::from(constant.tag()));
                let payload = self.iconst(types::I64, constant.payload() as i64);
                Word::Two(tag, payload)
            }
            Rep::None => Word::None,
        }
    }

    fn unary(&mut self, op: UnOp, operand: Val, value: Val) -> Word {
        let rep = self.ir.rep(value);
        match op {
            UnOp::Box => {
                let (tag, payload) = self.pair(operand);
                Word::Two(tag, payload)
            }
            UnOp::Unbox => self.convert(operand, rep),
            UnOp::Truthy => Word::One(self.truthy(operand)),
            UnOp::Not => {
                let flag = self.one(operand);
                Word::One(self.builder.ins().bxor_imm_s(flag, 1))
            }
            UnOp::INeg => {
                let x = self.one(operand);
                Word::One(self.builder.ins().ineg(x))
            }
            UnOp::IBnot => {
                let x = self.one(operand);
                Word::One(self.builder.ins().bnot(x))
            }
            UnOp::FNeg => {
                let x = self.one(operand);
                Word::One(self.builder.ins().fneg(x))
            }
            UnOp::IToF => {
                let x = self.one(operand);
                Word::One(self.builder.ins().fcvt_from_sint(types::F64, x))
            }
        }
    }

    fn truthy(&mut self, operand: Val) -> Value {
        match self.ir.rep(operand) {
            Rep::Bool => self.one(operand),
            Rep::Boxed => {
                let (tag, payload) = self.pair(operand);
                let tag = self.tag_of(tag);
                let nil = self.builder.ins().icmp_imm_s(IntCC::Equal, tag, NIL_TAG);
                let is_bool = self.builder.ins().icmp_imm_s(IntCC::Equal, tag, BOOL_TAG);
                let flag = self.builder.ins().band_imm_s(payload, 0xff);
                let is_false = self.builder.ins().icmp_imm_s(IntCC::Equal, flag, 0);
                let false_value = self.builder.ins().band(is_bool, is_false);
                let falsy = self.builder.ins().bor(nil, false_value);
                self.builder.ins().bxor_imm_s(falsy, 1)
            }
            _ => self.iconst(types::I8, 1),
        }
    }

    fn binary(&mut self, op: BinOp, left: Val, right: Val) -> Value {
        if op == BinOp::BoolEq {
            let (x, y) = (self.one(left), self.one(right));
            return self.builder.ins().icmp(IntCC::Equal, x, y);
        }
        let (x, y) = (self.one(left), self.one(right));
        let ins = self.builder.ins();
        match op {
            BinOp::IAdd => ins.iadd(x, y),
            BinOp::ISub => ins.isub(x, y),
            BinOp::IMul => ins.imul(x, y),
            BinOp::IBand => ins.band(x, y),
            BinOp::IBor => ins.bor(x, y),
            BinOp::IBxor => ins.bxor(x, y),
            BinOp::FAdd => ins.fadd(x, y),
            BinOp::FSub => ins.fsub(x, y),
            BinOp::FMul => ins.fmul(x, y),
            BinOp::FDiv => ins.fdiv(x, y),
            BinOp::FIDiv => {
                let quotient = ins.fdiv(x, y);
                self.builder.ins().floor(quotient)
            }
            BinOp::IMod | BinOp::IDiv => self.floor_division(op, x, y),
            BinOp::ICmp(cc) => ins.icmp(int_cc(cc), x, y),
            BinOp::FCmp(cc) => ins.fcmp(float_cc(cc), x, y),
            BinOp::BoolEq => unreachable!("handled above"),
        }
    }

    /// Follows lvm.c's `luaV_mod` and `luaV_idiv`. A hoisted copy may run before the zero-divisor guard, so the divisor is replaced by one instead of trapping.
    fn floor_division(&mut self, op: BinOp, x: Value, y: Value) -> Value {
        let zero = self.builder.ins().icmp_imm_s(IntCC::Equal, y, 0);
        let minus_one = self.builder.ins().icmp_imm_s(IntCC::Equal, y, -1);
        let special = self.builder.ins().bor(zero, minus_one);
        let one = self.iconst(types::I64, 1);
        let divisor = self.builder.ins().select(special, one, y);
        let remainder = self.builder.ins().srem(x, divisor);
        let nonzero = self.builder.ins().icmp_imm_s(IntCC::NotEqual, remainder, 0);
        let signs = if op == BinOp::IMod { self.builder.ins().bxor(remainder, divisor) } else { self.builder.ins().bxor(x, divisor) };
        let negative = self.builder.ins().icmp_imm_s(IntCC::SignedLessThan, signs, 0);
        let adjust = self.builder.ins().band(nonzero, negative);
        if op == BinOp::IMod {
            let shifted = self.builder.ins().iadd(remainder, divisor);
            let result = self.builder.ins().select(adjust, shifted, remainder);
            let zero_value = self.iconst(types::I64, 0);
            return self.builder.ins().select(minus_one, zero_value, result);
        }
        let quotient = self.builder.ins().sdiv(x, divisor);
        let lowered = self.builder.ins().iadd_imm_s(quotient, -1);
        let result = self.builder.ins().select(adjust, lowered, quotient);
        let negated = self.builder.ins().ineg(x);
        self.builder.ins().select(minus_one, negated, result)
    }

    fn pure(&mut self, function: PureFn, args: &[Val]) -> Value {
        let words: Vec<Value> = args.iter().map(|&arg| self.one(arg)).collect();
        let (callee, call_args) = match function {
            PureFn::FMod | PureFn::FPow => {
                let kind = self.iconst(types::I32, i64::from(if function == PureFn::FMod { runtime::PURE_MOD } else { runtime::PURE_POW }));
                (self.imports.pure_float, vec![kind, words[0], words[1]])
            }
            PureFn::Shl | PureFn::Shr => {
                let kind = self.iconst(types::I32, i64::from(if function == PureFn::Shl { runtime::PURE_SHL } else { runtime::PURE_SHR }));
                (self.imports.pure_int, vec![kind, words[0], words[1]])
            }
            PureFn::LoopCount => (self.imports.loop_count, words.clone()),
            PureFn::LtIF | PureFn::LeIF | PureFn::LtFI | PureFn::LeFI | PureFn::EqFI => {
                let code = match function {
                    PureFn::LtIF => runtime::MIXED_LT_IF,
                    PureFn::LeIF => runtime::MIXED_LE_IF,
                    PureFn::LtFI => runtime::MIXED_LT_FI,
                    PureFn::LeFI => runtime::MIXED_LE_FI,
                    _ => runtime::MIXED_EQ_FI,
                };
                let kind = self.iconst(types::I32, i64::from(code));
                let (integer, float) = match function {
                    PureFn::LtIF | PureFn::LeIF => (words[0], words[1]),
                    _ => (words[1], words[0]),
                };
                (self.imports.mixed, vec![kind, integer, float])
            }
        };
        let call = self.builder.ins().call(callee, &call_args);
        let result = self.builder.inst_results(call)[0];
        match function {
            PureFn::LtIF | PureFn::LeIF | PureFn::LtFI | PureFn::LeFI | PureFn::EqFI => self.builder.ins().ireduce(types::I8, result),
            _ => result,
        }
    }

    fn select(&mut self, cond: Val, yes: Val, no: Val, rep: Rep) -> Word {
        let flag = self.one(cond);
        match (self.convert(yes, rep), self.convert(no, rep)) {
            (Word::One(x), Word::One(y)) => Word::One(self.builder.ins().select(flag, x, y)),
            (Word::Two(xt, xp), Word::Two(yt, yp)) => {
                let tag = self.builder.ins().select(flag, xt, yt);
                let payload = self.builder.ins().select(flag, xp, yp);
                Word::Two(tag, payload)
            }
            _ => Word::None,
        }
    }

    /// Follows `luaV_equalobj`: a float equals an integer only when it is exactly that integer.
    fn equal_constant(&mut self, operand: Val, constant: Const) -> Value {
        let rep = self.ir.rep(operand);
        let word = if rep == Rep::Boxed || rep == Rep::None { None } else { Some(self.one(operand)) };
        match (rep, constant, word) {
            (Rep::Int, Const::Int(k), Some(x)) => self.builder.ins().icmp_imm_s(IntCC::Equal, x, k),
            (Rep::Int, Const::Float(f), Some(x)) => match exact_integer(f) {
                Some(k) => self.builder.ins().icmp_imm_s(IntCC::Equal, x, k),
                None => self.iconst(types::I8, 0),
            },
            (Rep::Float, Const::Float(f), Some(x)) => {
                let constant = self.builder.ins().f64const(f);
                self.builder.ins().fcmp(FloatCC::Equal, x, constant)
            }
            (Rep::Float, Const::Int(k), Some(x)) => self.float_equals_integer(x, k),
            (Rep::Bool, Const::Bool(flag), Some(x)) => self.builder.ins().icmp_imm_s(IntCC::Equal, x, i64::from(flag)),
            (Rep::Payload, Const::Gc { tag, payload }, Some(x)) if self.ir.ty(operand).tag() == Some(tag) => {
                self.builder.ins().icmp_imm_s(IntCC::Equal, x, payload as i64)
            }
            (Rep::Boxed, _, _) => self.boxed_equal_constant(operand, constant),
            _ => self.iconst(types::I8, 0),
        }
    }

    fn float_equals_integer(&mut self, float: Value, integer: i64) -> Value {
        if integer.unsigned_abs() <= 1 << 53 {
            let constant = self.builder.ins().f64const(integer as f64);
            return self.builder.ins().fcmp(FloatCC::Equal, float, constant);
        }
        let kind = self.iconst(types::I32, i64::from(runtime::MIXED_EQ_FI));
        let integer = self.iconst(types::I64, integer);
        let call = self.builder.ins().call(self.imports.mixed, &[kind, integer, float]);
        let result = self.builder.inst_results(call)[0];
        self.builder.ins().ireduce(types::I8, result)
    }

    fn boxed_equal_constant(&mut self, operand: Val, constant: Const) -> Value {
        let (tag, payload) = self.pair(operand);
        let tag = self.tag_of(tag);
        let same_tag = self.builder.ins().icmp_imm_s(IntCC::Equal, tag, i64::from(constant.tag()));
        let same = match constant {
            Const::Nil => return same_tag,
            Const::Bool(flag) => {
                let low = self.builder.ins().band_imm_s(payload, 0xff);
                self.builder.ins().icmp_imm_s(IntCC::Equal, low, i64::from(flag))
            }
            Const::Int(_) | Const::Gc { .. } => self.builder.ins().icmp_imm_s(IntCC::Equal, payload, constant.payload() as i64),
            Const::Float(f) => {
                let float = self.builder.ins().bitcast(types::F64, MemFlagsData::new(), payload);
                let constant = self.builder.ins().f64const(f);
                self.builder.ins().fcmp(FloatCC::Equal, float, constant)
            }
        };
        let matched = self.builder.ins().band(same_tag, same);
        let other = match constant {
            Const::Int(k) => {
                let is_float = self.builder.ins().icmp_imm_s(IntCC::Equal, tag, FLOAT_TAG);
                let float = self.builder.ins().bitcast(types::F64, MemFlagsData::new(), payload);
                let equal = self.float_equals_integer(float, k);
                Some(self.builder.ins().band(is_float, equal))
            }
            Const::Float(f) => exact_integer(f).map(|k| {
                let is_int = self.builder.ins().icmp_imm_s(IntCC::Equal, tag, INT_TAG);
                let equal = self.builder.ins().icmp_imm_s(IntCC::Equal, payload, k);
                self.builder.ins().band(is_int, equal)
            }),
            _ => None,
        };
        match other {
            Some(other) => self.builder.ins().bor(matched, other),
            None => matched,
        }
    }

    fn arith(&mut self, op: NumOp, (left, right): (Val, Val), state: StateId) -> Word {
        let x = self.pair(left);
        let y = self.pair(right);
        let join = self.builder.create_block();
        let tag = self.builder.append_block_param(join, types::I64);
        let payload = self.builder.append_block_param(join, types::I64);
        let slow = self.builder.create_block();
        self.builder.set_cold_block(slow);
        let unary = matches!(op, NumOp::Neg | NumOp::BNot);
        let int_op = matches!(op, NumOp::Add | NumOp::Sub | NumOp::Mul | NumOp::BAnd | NumOp::BOr | NumOp::BXor | NumOp::Neg | NumOp::BNot);
        let float_op = matches!(op, NumOp::Add | NumOp::Sub | NumOp::Mul | NumOp::Div | NumOp::Neg);
        let x_tag = self.tag_of(x.0);
        let y_tag = if unary { x_tag } else { self.tag_of(y.0) };
        if int_op {
            let both = self.both_tags(x_tag, y_tag, INT_TAG);
            let int_path = self.builder.create_block();
            let next = self.builder.create_block();
            self.builder.ins().brif(both, int_path, &[], next, &[]);
            self.builder.switch_to_block(int_path);
            let result = self.int_arith(op, x.1, y.1);
            let int_tag = self.iconst(types::I64, INT_TAG);
            self.builder.ins().jump(join, &[BlockArg::Value(int_tag), BlockArg::Value(result)]);
            self.builder.switch_to_block(next);
        }
        if float_op {
            let both = self.both_tags(x_tag, y_tag, FLOAT_TAG);
            let float_path = self.builder.create_block();
            let next = self.builder.create_block();
            self.builder.ins().brif(both, float_path, &[], next, &[]);
            self.builder.switch_to_block(float_path);
            let a = self.builder.ins().bitcast(types::F64, MemFlagsData::new(), x.1);
            let b = self.builder.ins().bitcast(types::F64, MemFlagsData::new(), y.1);
            let result = match op {
                NumOp::Add => self.builder.ins().fadd(a, b),
                NumOp::Sub => self.builder.ins().fsub(a, b),
                NumOp::Mul => self.builder.ins().fmul(a, b),
                NumOp::Div => self.builder.ins().fdiv(a, b),
                _ => self.builder.ins().fneg(a),
            };
            let bits = self.builder.ins().bitcast(types::I64, MemFlagsData::new(), result);
            let float_tag = self.iconst(types::I64, FLOAT_TAG);
            self.builder.ins().jump(join, &[BlockArg::Value(float_tag), BlockArg::Value(bits)]);
            self.builder.switch_to_block(next);
        }
        self.builder.ins().jump(slow, &[]);
        self.builder.switch_to_block(slow);
        let ok = self.call_value(runtime::KIND_ARITH, &[x, y], runtime::num_op_code(op));
        self.guard(ok, state);
        let (tag_result, payload_result) = self.scratch_load(0);
        self.builder.ins().jump(join, &[BlockArg::Value(tag_result), BlockArg::Value(payload_result)]);
        self.builder.switch_to_block(join);
        Word::Two(tag, payload)
    }

    fn both_tags(&mut self, x_tag: Value, y_tag: Value, tag: i64) -> Value {
        let x = self.builder.ins().icmp_imm_s(IntCC::Equal, x_tag, tag);
        let y = self.builder.ins().icmp_imm_s(IntCC::Equal, y_tag, tag);
        self.builder.ins().band(x, y)
    }

    fn int_arith(&mut self, op: NumOp, x: Value, y: Value) -> Value {
        let ins = self.builder.ins();
        match op {
            NumOp::Add => ins.iadd(x, y),
            NumOp::Sub => ins.isub(x, y),
            NumOp::Mul => ins.imul(x, y),
            NumOp::BAnd => ins.band(x, y),
            NumOp::BOr => ins.bor(x, y),
            NumOp::BXor => ins.bxor(x, y),
            NumOp::Neg => ins.ineg(x),
            _ => ins.bnot(x),
        }
    }

    fn compare(&mut self, cc: Cc, (left, right): (Val, Val), state: StateId) -> Value {
        let x = self.pair(left);
        let y = self.pair(right);
        let join = self.builder.create_block();
        let result = self.builder.append_block_param(join, types::I8);
        let x_tag = self.tag_of(x.0);
        let y_tag = self.tag_of(y.0);
        let int_path = self.builder.create_block();
        let float_test = self.builder.create_block();
        let float_path = self.builder.create_block();
        let slow = self.builder.create_block();
        self.builder.set_cold_block(slow);
        let ints = self.both_tags(x_tag, y_tag, INT_TAG);
        self.builder.ins().brif(ints, int_path, &[], float_test, &[]);
        self.builder.switch_to_block(int_path);
        let flag = self.builder.ins().icmp(int_cc(cc), x.1, y.1);
        self.builder.ins().jump(join, &[BlockArg::Value(flag)]);
        self.builder.switch_to_block(float_test);
        let floats = self.both_tags(x_tag, y_tag, FLOAT_TAG);
        self.builder.ins().brif(floats, float_path, &[], slow, &[]);
        self.builder.switch_to_block(float_path);
        let a = self.builder.ins().bitcast(types::F64, MemFlagsData::new(), x.1);
        let b = self.builder.ins().bitcast(types::F64, MemFlagsData::new(), y.1);
        let flag = self.builder.ins().fcmp(float_cc(cc), a, b);
        self.builder.ins().jump(join, &[BlockArg::Value(flag)]);
        self.builder.switch_to_block(slow);
        let answer = self.call_value(runtime::KIND_COMPARE, &[x, y], cc as u32);
        let known = self.builder.ins().icmp_imm_s(IntCC::NotEqual, answer, 0);
        self.guard(known, state);
        let truth = self.builder.ins().icmp_imm_s(IntCC::Equal, answer, 2);
        self.builder.ins().jump(join, &[BlockArg::Value(truth)]);
        self.builder.switch_to_block(join);
        result
    }

    fn guard_type(&mut self, input: Val, ty: Type, state: StateId, rep: Rep) -> Word {
        let (tag_word, _) = self.pair(input);
        let tag = self.tag_of(tag_word);
        let matches = match ty.tag() {
            Some(single) => self.builder.ins().icmp_imm_s(IntCC::Equal, tag, i64::from(single)),
            None => {
                let one = self.iconst(types::I64, 1);
                let bit = self.builder.ins().ishl(one, tag);
                let mask = self.builder.ins().band_imm_s(bit, i64::from(type_mask(ty)));
                self.builder.ins().icmp_imm_s(IntCC::NotEqual, mask, 0)
            }
        };
        self.guard(matches, state);
        if self.ir.rep(input) == Rep::Boxed {
            let pair = self.pair(input);
            return if rep == Rep::Boxed { Word::Two(pair.0, pair.1) } else { self.unbox(pair, rep) };
        }
        self.convert(input, rep)
    }

    fn shape_is(&mut self, table: Val, shape: u64) -> Value {
        let address = self.table_address(self.one(table));
        let shaped = self.builder.ins().uload8(types::I64, self.heap_flags, address, self.env.layout.shaped);
        let current = self.builder.ins().load(types::I64, self.heap_flags, address, self.env.layout.shape);
        let same = self.builder.ins().icmp_imm_s(IntCC::Equal, current, shape as i64);
        let is_shaped = self.builder.ins().icmp_imm_s(IntCC::NotEqual, shaped, 0);
        self.builder.ins().band(is_shaped, same)
    }

    fn absent(&mut self, table: Val, bit: u8) -> Value {
        let layout = self.env.layout;
        let address = self.table_address(self.one(table));
        let tag = self.builder.ins().uload32(self.heap_flags, address, layout.metatable_tag);
        let none = self.builder.ins().icmp_imm_s(IntCC::Equal, tag, i64::from(layout.metatable_none));
        let join = self.builder.create_block();
        let result = self.builder.append_block_param(join, types::I8);
        let check = self.builder.create_block();
        self.builder.ins().brif(none, join, &[BlockArg::Value(none)], check, &[]);
        self.builder.switch_to_block(check);
        let index = self.builder.ins().uload32(self.heap_flags, address, layout.metatable_index);
        let metatable = self.table_address(index);
        let absent = self.builder.ins().uload8(types::I64, self.heap_flags, metatable, layout.absent);
        let flagged = self.builder.ins().band_imm_s(absent, i64::from(bit));
        let known = self.builder.ins().icmp_imm_s(IntCC::NotEqual, flagged, 0);
        self.builder.ins().jump(join, &[BlockArg::Value(known)]);
        self.builder.switch_to_block(join);
        result
    }

    fn slot_address(&mut self, table: Val, slot: u32) -> Value {
        let address = self.table_address(self.one(table));
        let slots = self.builder.ins().load(self.pointer, self.heap_flags, address, self.env.layout.slots);
        self.builder.ins().iadd_imm_s(slots, i64::from(slot) * i64::from(VALUE_SIZE))
    }

    /// Unshaped, still `nodes` long, and the node's key is the string: its value is the key's, as LuaJIT's `asm_hrefk` checks.
    fn node_holds(&mut self, table: Val, (node, nodes): (u32, u32), key: u64) -> Value {
        let layout = self.env.layout;
        let address = self.table_address(self.one(table));
        let shaped = self.builder.ins().uload8(types::I64, self.heap_flags, address, layout.shaped);
        let length = self.builder.ins().load(types::I64, self.heap_flags, address, layout.nodes_len);
        let unshaped = self.builder.ins().icmp_imm_s(IntCC::Equal, shaped, 0);
        let same_length = self.builder.ins().icmp_imm_s(IntCC::Equal, length, i64::from(nodes));
        let nodes_base = self.builder.ins().load(self.pointer, self.heap_flags, address, layout.nodes);
        let key_address = self.builder.ins().iadd_imm_s(nodes_base, i64::from(node) * NODE_SIZE + NODE_KEY);
        let key_tag = self.builder.ins().uload8(types::I64, self.heap_flags, key_address, 0);
        let key_payload = self.builder.ins().load(types::I64, self.heap_flags, key_address, PAYLOAD);
        let is_string = self.builder.ins().icmp_imm_s(IntCC::Equal, key_tag, STRING_TAG);
        let same_key = self.builder.ins().icmp_imm_s(IntCC::Equal, key_payload, key as i64);
        let layout_holds = self.builder.ins().band(unshaped, same_length);
        let key_holds = self.builder.ins().band(is_string, same_key);
        self.builder.ins().band(layout_holds, key_holds)
    }

    fn node_value_address(&mut self, table: Val, node: u32) -> Value {
        let address = self.table_address(self.one(table));
        let nodes_base = self.builder.ins().load(self.pointer, self.heap_flags, address, self.env.layout.nodes);
        self.builder.ins().iadd_imm_s(nodes_base, i64::from(node) * NODE_SIZE + NODE_VALUE)
    }

    fn inside(&mut self, table: Val, index: Value) -> Value {
        let address = self.table_address(self.one(table));
        let length = self.builder.ins().load(types::I64, self.heap_flags, address, self.env.layout.array_len);
        let position = self.builder.ins().iadd_imm_s(index, -1);
        self.builder.ins().icmp(IntCC::UnsignedLessThan, position, length)
    }

    fn element_address(&mut self, table: Val, index: Value) -> Value {
        let address = self.table_address(self.one(table));
        let elements = self.builder.ins().load(self.pointer, self.heap_flags, address, self.env.layout.array);
        let position = self.builder.ins().iadd_imm_s(index, -1);
        let offset = self.builder.ins().imul_imm_s(position, i64::from(VALUE_SIZE));
        self.builder.ins().iadd(elements, offset)
    }

    fn integer_key(&mut self, key: Val) -> Option<(Value, Option<Value>)> {
        match self.ir.rep(key) {
            Rep::Int => Some((self.one(key), None)),
            Rep::Boxed if !self.ir.ty(key).intersect(Type::INT).is_none() => {
                let (tag, payload) = self.pair(key);
                let tag = self.tag_of(tag);
                let is_int = self.builder.ins().icmp_imm_s(IntCC::Equal, tag, INT_TAG);
                Some((payload, Some(is_int)))
            }
            _ => None,
        }
    }

    fn index(&mut self, object: Val, key: Val, state: StateId) -> Word {
        let Some((index, is_int)) = self.integer_key(key) else {
            let (tag, payload) = self.index_helper(object, key, state);
            return Word::Two(tag, payload);
        };
        let join = self.builder.create_block();
        let tag = self.builder.append_block_param(join, types::I64);
        let payload = self.builder.append_block_param(join, types::I64);
        let slow = self.builder.create_block();
        self.builder.set_cold_block(slow);
        if let Some(is_int) = is_int {
            self.continue_if(is_int, slow);
        }
        let inside = self.inside(object, index);
        self.continue_if(inside, slow);
        let element = self.element_address(object, index);
        let (found_tag, found) = self.load_pair(self.heap_flags, element, 0);
        let low = self.tag_of(found_tag);
        let present = self.builder.ins().icmp_imm_s(IntCC::NotEqual, low, NIL_TAG);
        self.builder.ins().brif(present, join, &[BlockArg::Value(found_tag), BlockArg::Value(found)], slow, &[]);
        self.builder.switch_to_block(slow);
        let (result_tag, result) = self.index_helper(object, key, state);
        self.builder.ins().jump(join, &[BlockArg::Value(result_tag), BlockArg::Value(result)]);
        self.builder.switch_to_block(join);
        Word::Two(tag, payload)
    }

    fn index_helper(&mut self, object: Val, key: Val, state: StateId) -> (Value, Value) {
        let table = self.table_pair(object);
        let key = self.pair(key);
        let ok = self.call_value(runtime::KIND_INDEX, &[table, key], 0);
        self.guard(ok, state);
        self.scratch_load(0)
    }

    fn load_array(&mut self, table: Val, index: Val, state: StateId) -> Word {
        let join = self.builder.create_block();
        let tag = self.builder.append_block_param(join, types::I64);
        let payload = self.builder.append_block_param(join, types::I64);
        let slow = self.builder.create_block();
        self.builder.set_cold_block(slow);
        let position = self.one(index);
        let element = self.element_address(table, position);
        let (found_tag, found) = self.load_pair(self.heap_flags, element, 0);
        let low = self.tag_of(found_tag);
        let present = self.builder.ins().icmp_imm_s(IntCC::NotEqual, low, NIL_TAG);
        self.builder.ins().brif(present, join, &[BlockArg::Value(found_tag), BlockArg::Value(found)], slow, &[]);
        self.builder.switch_to_block(slow);
        let table = self.table_pair(table);
        let key = self.pair(index);
        let ok = self.call_value(runtime::KIND_INDEX, &[table, key], 0);
        self.guard(ok, state);
        let (result_tag, result) = self.scratch_load(0);
        self.builder.ins().jump(join, &[BlockArg::Value(result_tag), BlockArg::Value(result)]);
        self.builder.switch_to_block(join);
        Word::Two(tag, payload)
    }

    /// The raw store runs only when no `__newindex` can fire (`quick::raw_set` decides).
    fn new_index(&mut self, object: Val, key: Val, stored: Val, state: StateId) {
        let value = self.pair(stored);
        let Some((index, is_int)) = self.integer_key(key) else {
            self.new_index_helper(object, key, value, state);
            return;
        };
        let done = self.builder.create_block();
        let slow = self.builder.create_block();
        self.builder.set_cold_block(slow);
        if let Some(is_int) = is_int {
            self.continue_if(is_int, slow);
        }
        let inside = self.inside(object, index);
        self.continue_if(inside, slow);
        self.array_store(object, index, value, slow);
        self.builder.ins().jump(done, &[]);
        self.builder.switch_to_block(slow);
        self.new_index_helper(object, key, value, state);
        self.builder.ins().jump(done, &[]);
        self.builder.switch_to_block(done);
    }

    /// Follows `Vm::transition_store_hit`; anything else takes the helper's generic store.
    fn add_slot(&mut self, table: Val, key: Val, (shape, next, slot): (u64, u64, u32), stored: Val, state: StateId) {
        let layout = self.env.layout;
        let value = self.pair(stored);
        let slow = self.builder.create_block();
        self.builder.set_cold_block(slow);
        let done = self.builder.create_block();
        let address = self.table_address(self.one(table));
        let same = self.shape_is(table, shape);
        self.continue_if(same, slow);
        let metatable = self.builder.ins().uload32(self.heap_flags, address, layout.metatable_tag);
        let has_meta = self.builder.ins().icmp_imm_s(IntCC::NotEqual, metatable, i64::from(layout.metatable_none));
        self.continue_unless_flag(has_meta, slow);
        let length = self.builder.ins().load(types::I64, self.heap_flags, address, layout.slots_len);
        let capacity = self.builder.ins().load(types::I64, self.heap_flags, address, layout.slots_capacity);
        let full = self.builder.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, length, capacity);
        self.continue_unless_flag(full, slow);
        let payload = self.one(table);
        let index = self.builder.ins().band_imm_s(payload, 0xffff_ffff);
        let marks = self.ctx_load(self.pointer, offset_of!(JitCtx, table_marks));
        let mark_address = self.builder.ins().iadd(marks, index);
        let mark = self.builder.ins().uload8(types::I64, self.heap_flags, mark_address, 0);
        let black = self.builder.ins().band_imm_s(mark, i64::from(crate::gc::BLACK));
        self.continue_unless_flag(black, slow);
        if !self.ir.ty(stored).intersect(Type::NIL).is_none() {
            let tag = self.tag_of(value.0);
            let nil = self.builder.ins().icmp_imm_s(IntCC::Equal, tag, NIL_TAG);
            self.continue_unless_flag(nil, slow);
        }
        let slots = self.builder.ins().load(self.pointer, self.heap_flags, address, layout.slots);
        self.store_pair(self.heap_flags, value, slots, slot as i32 * VALUE_SIZE);
        let grown = self.builder.ins().iadd_imm_s(length, 1);
        self.builder.ins().store(self.heap_flags, grown, address, layout.slots_len);
        let next = self.iconst(types::I64, next as i64);
        self.builder.ins().store(self.heap_flags, next, address, layout.shape);
        let cleared = self.iconst(types::I8, 0);
        self.builder.ins().store(self.heap_flags, cleared, address, layout.absent);
        self.builder.ins().jump(done, &[]);
        self.builder.switch_to_block(slow);
        let table_pair = self.table_pair(table);
        let key = self.pair(key);
        let ok = self.call_value(runtime::KIND_NEW_INDEX, &[table_pair, key, value], 0);
        self.guard(ok, state);
        self.builder.ins().jump(done, &[]);
        self.builder.switch_to_block(done);
    }

    fn continue_unless_flag(&mut self, flag: Value, otherwise: Block) {
        let next = self.builder.create_block();
        self.builder.ins().brif(flag, otherwise, &[], next, &[]);
        self.builder.switch_to_block(next);
    }

    fn new_index_helper(&mut self, object: Val, key: Val, value: (Value, Value), state: StateId) {
        let table = self.table_pair(object);
        let key = self.pair(key);
        let ok = self.call_value(runtime::KIND_NEW_INDEX, &[table, key, value], 0);
        self.guard(ok, state);
    }

    /// An empty item of a table with a metatable goes to `slow`, since `__newindex` may fire.
    fn array_store(&mut self, table: Val, index: Value, value: (Value, Value), slow: Block) {
        let layout = self.env.layout;
        let element = self.element_address(table, index);
        let tag = self.builder.ins().uload8(types::I64, self.heap_flags, element, 0);
        let present = self.builder.ins().icmp_imm_s(IntCC::NotEqual, tag, NIL_TAG);
        let store = self.builder.create_block();
        let check = self.builder.create_block();
        self.builder.ins().brif(present, store, &[], check, &[]);
        self.builder.switch_to_block(check);
        let address = self.table_address(self.one(table));
        let metatable = self.builder.ins().uload32(self.heap_flags, address, layout.metatable_tag);
        let none = self.builder.ins().icmp_imm_s(IntCC::Equal, metatable, i64::from(layout.metatable_none));
        self.builder.ins().brif(none, store, &[], slow, &[]);
        self.builder.switch_to_block(store);
        self.store_pair(self.heap_flags, value, element, 0);
        self.barrier(table, value);
    }

    fn may_collect(&self, value: Val) -> bool {
        let ty = self.ir.ty(value);
        (STRING_TAG as u8..LIGHT_USERDATA_TAG as u8).any(|tag| !ty.intersect(Type::of_tag(tag)).is_none())
    }

    /// Only a collectable value stored into a black table needs the collector's barrier.
    fn barrier(&mut self, table: Val, (tag, payload): (Value, Value)) {
        let done = self.builder.create_block();
        let check = self.builder.create_block();
        let call = self.builder.create_block();
        self.builder.set_cold_block(call);
        let low = self.tag_of(tag);
        let shifted = self.builder.ins().iadd_imm_s(low, -STRING_TAG);
        let collectable = self.builder.ins().icmp_imm_s(IntCC::UnsignedLessThan, shifted, LIGHT_USERDATA_TAG - STRING_TAG);
        self.builder.ins().brif(collectable, check, &[], done, &[]);
        self.builder.switch_to_block(check);
        let table_payload = self.one(table);
        let index = self.builder.ins().band_imm_s(table_payload, 0xffff_ffff);
        let marks = self.ctx_load(self.pointer, offset_of!(JitCtx, table_marks));
        let mark_address = self.builder.ins().iadd(marks, index);
        let mark = self.builder.ins().uload8(types::I64, self.heap_flags, mark_address, 0);
        let black = self.builder.ins().band_imm_s(mark, i64::from(crate::gc::BLACK));
        self.builder.ins().brif(black, call, &[], done, &[]);
        self.builder.switch_to_block(call);
        let table_pair = self.table_pair(table);
        self.call_value(runtime::KIND_BARRIER, &[table_pair, (tag, payload)], 0);
        self.builder.ins().jump(done, &[]);
        self.builder.switch_to_block(done);
    }

    fn upvalue_list(&mut self, closure: Closure) -> Value {
        match closure {
            Closure::Root => self.ctx_load(self.pointer, offset_of!(JitCtx, upvalues)),
            Closure::Value(value) => {
                let address = self.closure_address(self.one(value));
                self.builder.ins().load(types::I64, self.heap_flags, address, self.env.frames.closure_upvalues)
            }
        }
    }

    fn upvalue_cell(&mut self, closure: Closure, index: u8) -> Value {
        let list = self.upvalue_list(closure);
        let offset = i32::try_from(upvalue::cell(index)).expect("an upvalue index is a byte");
        self.builder.ins().load(types::I64, self.heap_flags, list, offset)
    }

    /// The address may be a register's, so accesses through it alias everything.
    fn upvalue_address(&mut self, closure: Closure, index: u8) -> Value {
        let cell = self.upvalue_cell(closure, index);
        self.builder.ins().load(types::I64, self.heap_flags, cell, upvalue::VALUE)
    }

    /// A closed upvalue needs the collector's barrier for a collectable value, so that store runs the helper.
    fn store_upvalue(&mut self, closure: Closure, index: u8, value: Val) {
        let pair = self.pair(value);
        if !self.may_collect(value) {
            let address = self.upvalue_address(closure, index);
            self.store_pair(MemFlagsData::new().with_notrap(), pair, address, 0);
            return;
        }
        let done = self.builder.create_block();
        let slow = self.builder.create_block();
        self.builder.set_cold_block(slow);
        let cell = self.upvalue_cell(closure, index);
        let tag = self.builder.ins().uload8(types::I64, self.heap_flags, cell, upvalue::TAG);
        let open = self.builder.ins().icmp_imm_s(IntCC::Equal, tag, upvalue::OPEN);
        let fast = self.builder.create_block();
        self.builder.ins().brif(open, fast, &[], slow, &[]);
        self.builder.switch_to_block(fast);
        let address = self.builder.ins().load(types::I64, self.heap_flags, cell, upvalue::VALUE);
        self.store_pair(MemFlagsData::new().with_notrap(), pair, address, 0);
        self.builder.ins().jump(done, &[]);
        self.builder.switch_to_block(slow);
        let owner = match closure {
            Closure::Root => self.ctx_load(types::I64, offset_of!(JitCtx, closure)),
            Closure::Value(value) => self.one(value),
        };
        let owner = (self.iconst(types::I64, CLOSURE_TAG), owner);
        self.call_value(runtime::KIND_SET_UPVALUE, &[pair, owner], u32::from(index));
        self.builder.ins().jump(done, &[]);
        self.builder.switch_to_block(done);
    }

    fn helper(&mut self, pc: u32, kind: HelperKind, args: [u32; 3]) -> Value {
        let pc_value = self.iconst(types::I32, i64::from(pc));
        let call = match kind {
            HelperKind::Step => self.builder.ins().call(self.imports.step, &[self.ctx, pc_value]),
            HelperKind::Fast(which) => {
                let values: Vec<Value> = args.iter().map(|&arg| self.builder.ins().iconst(types::I32, i64::from(arg as i32))).collect();
                let helper = self.imports.fast[which];
                self.builder.ins().call(helper, &[self.ctx, pc_value, values[0], values[1], values[2]])
            }
        };
        self.builder.inst_results(call)[0]
    }

    fn safepoint(&mut self, pc: u32, state: StateId) {
        let Some(&(_, osr)) = self.ir.osr.iter().find(|&&(at, _)| at == pc) else {
            return;
        };
        let poll = self.builder.create_block();
        self.builder.set_cold_block(poll);
        let next = self.builder.create_block();
        if self.env.stress_exits {
            self.builder.ins().jump(poll, &[]);
        } else {
            let requests = self.ctx_load(self.pointer, offset_of!(JitCtx, requests));
            let requested = self.builder.ins().atomic_load(types::I8, MemFlagsData::new().with_notrap(), requests);
            self.builder.ins().brif(requested, poll, &[], next, &[]);
        }
        self.builder.switch_to_block(poll);
        self.store_state(state);
        let target = self.iconst(types::I32, i64::from(pc));
        let call = self.builder.ins().call(self.imports.safe_point, &[self.ctx, target]);
        let result = self.builder.inst_results(call)[0];
        let rejoin = self.builder.ins().icmp_imm_s(IntCC::Equal, result, i64::from(pc));
        let leave = self.builder.create_block();
        self.builder.ins().brif(rejoin, self.blocks[osr.0 as usize], &[], leave, &[]);
        self.builder.switch_to_block(leave);
        self.exit_with(result);
        self.builder.switch_to_block(next);
    }

    fn edge_args(&mut self, edge: &super::ir::Edge) -> Vec<BlockArg> {
        let mut args = Vec::new();
        for &arg in &edge.args {
            match self.word(arg) {
                Word::One(word) => args.push(BlockArg::Value(word)),
                Word::Two(tag, payload) => {
                    args.push(BlockArg::Value(tag));
                    args.push(BlockArg::Value(payload));
                }
                Word::None => {}
            }
        }
        args
    }

    fn lower_term(&mut self, term: &Term) {
        match term {
            Term::Open => {
                self.builder.ins().trap(TrapCode::unwrap_user(1));
            }
            Term::Jump(edge) => {
                let args = self.edge_args(edge);
                self.builder.ins().jump(self.blocks[edge.block.0 as usize], &args);
            }
            Term::Branch { cond, then, other } => {
                let flag = self.one(*cond);
                let then_args = self.edge_args(then);
                let other_args = self.edge_args(other);
                self.builder.ins().brif(flag, self.blocks[then.block.0 as usize], &then_args, self.blocks[other.block.0 as usize], &other_args);
            }
            Term::Switch { code, cases } => {
                let result = self.one(*code);
                for (pc, edge) in cases {
                    let args = self.edge_args(edge);
                    let next = self.builder.create_block();
                    let here = self.builder.ins().icmp_imm_s(IntCC::Equal, result, i64::from(*pc));
                    self.builder.ins().brif(here, self.blocks[edge.block.0 as usize], &args, next, &[]);
                    self.builder.switch_to_block(next);
                }
                self.exit_with(result);
            }
            Term::Return { pc, first, count, close } => {
                if *close {
                    self.close_in_place(*pc as usize);
                }
                self.lower_return(*pc as usize, *first, *count);
            }
            Term::TailCall { pc, .. } => {
                let pc_value = self.iconst(types::I32, i64::from(*pc));
                let zero = self.iconst(types::I32, 0);
                let helper = self.imports.fast[sites::FAST_TAIL_CALL];
                let call = self.builder.ins().call(helper, &[self.ctx, pc_value, zero, zero, zero]);
                let result = self.builder.inst_results(call)[0];
                self.exit_with(result);
            }
            Term::Deopt { state } => {
                let stub = self.stub(*state);
                self.builder.ins().jump(stub, &[]);
            }
        }
    }

    fn call_fast(&mut self, pc: usize, which: usize, args: [u32; 3]) -> Value {
        self.helper(pc as u32, HelperKind::Fast(which), args)
    }

    fn close_in_place(&mut self, pc: usize) {
        let result = self.call_fast(pc, sites::FAST_CLOSE, [0, 0, 0]);
        let after = self.builder.create_block();
        let leave = self.builder.create_block();
        let closed = self.builder.ins().icmp_imm_s(IntCC::Equal, result, pc as i64);
        self.builder.ins().brif(closed, after, &[], leave, &[]);
        self.builder.switch_to_block(leave);
        self.exit_with(result);
        self.builder.switch_to_block(after);
    }

    /// Follows `moveresults` in ldo.c.
    fn lower_return(&mut self, pc: usize, first: u8, count: u32) {
        let helper = self.builder.create_block();
        let exact = self.builder.create_block();
        let other = self.builder.create_block();
        let wide = self.builder.create_block();
        let place = self.builder.create_block();
        let done = self.builder.create_block();
        let results = self.ctx_load(types::I32, offset_of!(JitCtx, direct_results));

        let is_exact = self.builder.ins().icmp_imm_s(IntCC::Equal, results, i64::from(count));
        self.builder.ins().brif(is_exact, exact, &[], other, &[]);
        self.builder.switch_to_block(exact);
        self.copy_results(first, count);
        self.builder.ins().jump(done, &[]);
        self.builder.switch_to_block(other);
        self.builder.set_cold_block(other);
        let none = self.builder.ins().icmp_imm_s(IntCC::Equal, results, 0);
        let not_none = self.builder.create_block();
        self.builder.ins().brif(none, done, &[], not_none, &[]);
        self.builder.switch_to_block(not_none);
        let small = self.builder.ins().icmp_imm_s(IntCC::UnsignedLessThanOrEqual, results, i64::from(u8::MAX));
        self.builder.ins().brif(small, place, &[], wide, &[]);
        self.builder.switch_to_block(wide);
        self.builder.set_cold_block(wide);
        let open = self.builder.ins().icmp_imm_s(IntCC::Equal, results, -1);
        self.builder.ins().brif(open, place, &[], helper, &[]);
        self.builder.switch_to_block(place);
        self.builder.set_cold_block(place);
        let destination = self.copy_results(first, count);
        self.fill_results(results, destination, count, done);
        self.builder.switch_to_block(done);
        let code = self.iconst(types::I64, crate::jit::EXIT_RETURNED);
        self.builder.ins().return_(&[code]);
        self.builder.switch_to_block(helper);
        let result = self.call_fast(pc, sites::FAST_RETURN, [u32::from(first), count, 0]);
        self.exit_with(result);
    }

    fn copy_results(&mut self, first: u8, count: u32) -> Value {
        let base = self.registers();
        let offset = self.ctx_load(types::I64, offset_of!(JitCtx, ret_offset));
        let destination = self.builder.ins().iadd(base, offset);
        for index in 0..count {
            let pair = self.load_pair(self.value_flags, base, slot_offset(u32::from(first) + index));
            self.store_pair(self.value_flags, pair, destination, index as i32 * VALUE_SIZE);
        }
        destination
    }

    fn fill_results(&mut self, results: Value, destination: Value, count: u32, done: Block) {
        let fill = self.builder.create_block();
        let store_nil = self.builder.create_block();
        self.builder.append_block_param(fill, types::I32);
        let start = self.iconst(types::I32, i64::from(count));
        let open_block = self.builder.create_block();
        let open = self.builder.ins().icmp_imm_s(IntCC::Equal, results, -1);
        self.builder.ins().brif(open, open_block, &[], fill, &[BlockArg::Value(start)]);
        self.builder.switch_to_block(open_block);
        let caller_base = self.ctx_load(types::I64, offset_of!(JitCtx, base));
        let top = self.builder.ins().iadd_imm_s(caller_base, i64::from(count) - 1);
        let thread = self.ctx_load(self.pointer, offset_of!(JitCtx, thread));
        self.builder.ins().store(self.frame_flags, top, thread, self.env.frames.top);
        self.builder.ins().jump(done, &[]);
        self.builder.switch_to_block(fill);
        let index = self.builder.block_params(fill)[0];
        let filled = self.builder.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, index, results);
        self.builder.ins().brif(filled, done, &[], store_nil, &[]);
        self.builder.switch_to_block(store_nil);
        let wide = self.builder.ins().uextend(types::I64, index);
        let offset = self.builder.ins().imul_imm_s(wide, i64::from(VALUE_SIZE));
        let slot = self.builder.ins().iadd(destination, offset);
        let nil = self.iconst(types::I8, NIL_TAG);
        self.builder.ins().store(self.value_flags, nil, slot, 0);
        let next = self.builder.ins().iadd_imm_s(index, 1);
        self.builder.ins().jump(fill, &[BlockArg::Value(next)]);
    }

    fn call_lua(&mut self, pc: usize, (a, b, c): (u8, u8, u8)) -> Value {
        let join = self.builder.create_block();
        let result = self.builder.append_block_param(join, types::I64);
        let targets = self.call_targets(pc);
        let helper = self.builder.create_block();
        if b != 0 && !targets.is_empty() {
            let (closure, matched) = self.match_callee(a, &targets, helper);
            for (target, &block) in targets.iter().zip(&matched) {
                self.builder.switch_to_block(block);
                self.direct_call(pc, (a, b, c), target, closure, (helper, join));
            }
        } else {
            self.builder.ins().jump(helper, &[]);
        }
        self.builder.switch_to_block(helper);
        let code = self.call_fast(pc, sites::FAST_CALL, [u32::from(a), u32::from(b), u32::from(c)]);
        self.builder.ins().jump(join, &[BlockArg::Value(code)]);
        self.builder.switch_to_block(join);
        result
    }

    fn call_targets(&self, pc: usize) -> Vec<CallTarget> {
        let mut targets = self.env.calls.get(pc).cloned().unwrap_or_default();
        let mut seen = Vec::with_capacity(targets.len());
        targets.retain(|target| {
            let fresh = !seen.contains(&target.proto);
            seen.push(target.proto);
            fresh
        });
        targets
    }

    fn continue_unless(&mut self, value: Value, condition: IntCC, immediate: i64, target: Block) {
        let taken = self.builder.ins().icmp_imm_s(condition, value, immediate);
        let next = self.builder.create_block();
        self.builder.ins().brif(taken, target, &[], next, &[]);
        self.builder.switch_to_block(next);
    }

    fn match_callee(&mut self, a: u8, targets: &[CallTarget], helper: Block) -> (Value, Vec<Block>) {
        let check = self.builder.create_block();
        let base = self.registers();
        let tag = self.builder.ins().uload8(types::I64, self.value_flags, base, slot_offset(u32::from(a)));
        let is_closure = self.builder.ins().icmp_imm_s(IntCC::Equal, tag, CLOSURE_TAG);
        self.builder.ins().brif(is_closure, check, &[], helper, &[]);
        self.builder.switch_to_block(check);
        let closure = self.builder.ins().load(types::I64, self.value_flags, base, slot_offset(u32::from(a)) + PAYLOAD);
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
        (closure, matched)
    }

    /// Calls do not overlap, so one stack slot serves every call site.
    fn callee_context(&mut self) -> StackSlot {
        if let Some(slot) = self.callee {
            return slot;
        }
        let size = std::mem::size_of::<JitCtx>() as u32;
        let align = std::mem::align_of::<JitCtx>().trailing_zeros() as u8;
        let slot = self.builder.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, size, align));
        self.callee = Some(slot);
        slot
    }

    /// Mirrors `Vm::jit_call_compiled`; other exits go to `Vm::jit_after_call`.
    fn direct_call(&mut self, pc: usize, (a, b, c): (u8, u8, u8), target: &CallTarget, closure: Value, (helper, join): (Block, Block)) {
        let frames = self.env.frames;
        let depth_limit = self.env.call_depth_limit;
        let signature = self.imports.entry_signature;
        let context = self.callee_context();
        let flags = self.frame_flags;
        let arguments = u32::from(b) - 1;
        let wanted = i64::from(c) - 1;
        let callee_offset = (i32::from(a) + 1) * VALUE_SIZE;

        let cell = self.iconst(types::I64, target.entry as i64);
        let entry = self.builder.ins().load(types::I64, flags, cell, 0);
        self.continue_unless(entry, IntCC::Equal, 0, helper);
        let depth_pointer = self.ctx_load(self.pointer, offset_of!(JitCtx, call_depth));
        let depth = self.builder.ins().load(types::I32, flags, depth_pointer, 0);
        self.continue_unless(depth, IntCC::UnsignedGreaterThanOrEqual, i64::from(depth_limit), helper);
        let thread = self.ctx_load(self.pointer, offset_of!(JitCtx, thread));
        let stack_len = self.builder.ins().load(types::I64, flags, thread, frames.stack_len);
        let caller_base = self.ctx_load(types::I64, offset_of!(JitCtx, base));
        let callee_base = self.builder.ins().iadd_imm_s(caller_base, i64::from(a) + 1);
        let register_count = i64::from(arguments) + i64::from(target.max_stack);
        let needed = self.builder.ins().iadd_imm_s(callee_base, register_count.max(crate::vm::REGISTER_SPAN as i64));
        let short = self.builder.ins().icmp(IntCC::UnsignedGreaterThan, needed, stack_len);
        let roomy = self.builder.create_block();
        self.builder.ins().brif(short, helper, &[], roomy, &[]);
        self.builder.switch_to_block(roomy);
        let top = self.builder.ins().iadd_imm_s(callee_base, register_count);
        self.continue_unless(top, IntCC::UnsignedGreaterThan, crate::vm::MAX_STACK_SLOTS as i64, helper);
        let count = self.builder.ins().load(types::I64, flags, thread, frames.frames_len);
        let capacity = self.builder.ins().load(types::I64, flags, thread, frames.frames_capacity);
        let full = self.builder.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, count, capacity);
        let fits = self.builder.create_block();
        self.builder.ins().brif(full, helper, &[], fits, &[]);
        self.builder.switch_to_block(fits);

        let base = self.registers();
        let nil = self.iconst(types::I8, NIL_TAG);
        for parameter in arguments..u32::from(target.num_params) {
            let offset = callee_offset + parameter as i32 * VALUE_SIZE;
            self.builder.ins().store(self.value_flags, nil, base, offset);
        }
        let data = self.builder.ins().load(types::I64, flags, thread, frames.frames_data);
        let frame_offset = self.builder.ins().imul_imm_s(count, frames.frame_size);
        let frame = self.builder.ins().iadd(data, frame_offset);
        let resume = self.iconst(types::I32, pc as i64 + 1);
        let caller_pc = frames.field(lua_frame::PC) - frames.frame_size as i32;
        self.builder.ins().store(flags, resume, frame, caller_pc);
        self.push_lua_frame(frame, closure, callee_base, wanted);
        let pushed = self.builder.ins().iadd_imm_s(count, 1);
        self.builder.ins().store(flags, pushed, thread, frames.frames_len);
        let upvalues = if target.has_upvalues {
            let address = self.closure_address(closure);
            self.builder.ins().load(types::I64, self.heap_flags, address, frames.closure_upvalues)
        } else {
            self.iconst(types::I64, 0)
        };
        let constants = self.iconst(types::I64, target.constants as i64);
        let proto = self.iconst(types::I64, target.proto as i64);
        self.fill_callee_context(context, (upvalues, constants, proto), base, (callee_offset, callee_base, closure), wanted);
        let deeper = self.builder.ins().iadd_imm_s(depth, 1);
        self.builder.ins().store(flags, deeper, depth_pointer, 0);
        let callee_context = self.builder.ins().stack_addr(self.pointer, context, 0);
        let start = self.iconst(types::I32, 0);
        let call = self.builder.ins().call_indirect(signature, entry, &[callee_context, start]);
        let code = self.builder.inst_results(call)[0];
        let depth_pointer = self.ctx_load(self.pointer, offset_of!(JitCtx, call_depth));
        self.builder.ins().store(flags, depth, depth_pointer, 0);

        let returned = self.builder.create_block();
        let other = self.builder.create_block();
        let is_returned = self.builder.ins().icmp_imm_s(IntCC::Equal, code, crate::jit::EXIT_RETURNED);
        self.builder.ins().brif(is_returned, returned, &[], other, &[]);
        self.builder.switch_to_block(returned);
        let thread = self.builder.ins().stack_load(self.pointer, types::I64, context, offset_of!(JitCtx, thread) as i32);
        let current = self.builder.ins().load(types::I64, flags, thread, frames.frames_len);
        let popped = self.builder.ins().iadd_imm_s(current, -1);
        self.builder.ins().store(flags, popped, thread, frames.frames_len);
        self.refresh_from_callee(context, thread, callee_offset);
        let next = self.iconst(types::I64, pc as i64 + 1);
        self.builder.ins().jump(join, &[BlockArg::Value(next)]);
        self.builder.switch_to_block(other);
        let code = self.builder.ins().ireduce(types::I32, code);
        let before = self.builder.ins().ireduce(types::I32, count);
        let pc_value = self.iconst(types::I32, pc as i64);
        let zero = self.iconst(types::I32, 0);
        let after = self.imports.fast[sites::FAST_AFTER_CALL];
        let call = self.builder.ins().call(after, &[self.ctx, pc_value, code, before, zero]);
        let result = self.builder.inst_results(call)[0];
        self.builder.ins().jump(join, &[BlockArg::Value(result)]);
    }

    fn push_lua_frame(&mut self, frame: Value, closure: Value, callee_base: Value, wanted: i64) {
        let frames = self.env.frames;
        let flags = self.frame_flags;
        let zero_byte = self.iconst(types::I8, 0);
        let zero = self.iconst(types::I64, 0);
        let status = (crate::object::LuaFrame::call_status(wanted as i32, 0) as i64) << lua_frame::STATUS_SHIFT;
        let pc_status = self.iconst(types::I64, status);
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

    fn fill_callee_context(&mut self, context: StackSlot, (upvalues, constants, proto): (Value, Value, Value), base: Value, (callee_offset, callee_base, closure): (i32, Value, Value), wanted: i64) {
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
            let value = self.ctx_load(types::I64, offset);
            self.builder.ins().stack_store(self.pointer, value, context, offset as i32);
        }
        let registers = self.builder.ins().iadd_imm_s(base, i64::from(callee_offset));
        let results = self.iconst(types::I32, wanted);
        let ret_offset = self.iconst(types::I64, -i64::from(VALUE_SIZE));
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
}

fn int_cc(cc: Cc) -> IntCC {
    match cc {
        Cc::Eq => IntCC::Equal,
        Cc::Lt => IntCC::SignedLessThan,
        Cc::Le => IntCC::SignedLessThanOrEqual,
    }
}

fn float_cc(cc: Cc) -> FloatCC {
    match cc {
        Cc::Eq => FloatCC::Equal,
        Cc::Lt => FloatCC::LessThan,
        Cc::Le => FloatCC::LessThanOrEqual,
    }
}

fn type_mask(ty: Type) -> u16 {
    (0..11u8).filter(|&tag| !ty.intersect(Type::of_tag(tag)).is_none()).fold(0, |bits, tag| bits | 1 << tag)
}

/// Follows `luaV_flttointns` with `F2Ieq`.
fn exact_integer(float: f64) -> Option<i64> {
    let integer = crate::number::float_to_integer(float, crate::number::Rounding::Exact)?;
    Some(integer)
}
