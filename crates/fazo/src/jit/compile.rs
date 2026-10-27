//! The Peak tier: builds the SSA graph from the prototype and its feedback, optimizes it and lowers it to Cranelift.

use std::sync::Arc;

use cranelift_codegen::CodegenError;
use cranelift_module::ModuleError;

use super::{
    FrameLayout, JitResult, TableLayout,
    memory::CodeBudget,
    sites::CallTarget,
    ssa::{self, PeakUnit, build, lower, passes},
    unit::{Compiler, JitUnit},
};
use crate::{arena::Ref, proto::Proto, value::Value as LuaValue};

pub(crate) struct CompileFacts<'a> {
    pub(crate) layout: Option<TableLayout>,
    pub(crate) frames: Option<FrameLayout>,
    pub(crate) calls: Vec<Vec<CallTarget>>,
    pub(crate) inline_caches: bool,
    pub(crate) avoid: &'a [u32],
    pub(crate) insufficient: &'a [u32],
    pub(crate) float_guess: bool,
    pub(crate) forced: bool,
    pub(crate) observed: &'a [(u32, u8)],
    pub(crate) interned: &'a dyn Fn(LuaValue) -> bool,
    pub(crate) proto_of: &'a dyn Fn(Ref<Proto>) -> Option<&'a Proto>,
    pub(crate) stress_exits: bool,
    pub(crate) osr_at: Option<u32>,
    pub(crate) node_hint: &'a dyn Fn(&Proto, usize, bool) -> Option<build::NodeHint>,
    pub(crate) builtin_of: &'a dyn Fn(Ref<crate::object::Native>) -> Option<crate::vm::Builtin>,
    pub(crate) type_names: [u64; 8],
    pub(crate) sink_allocations: bool,
    pub(crate) running_floats: &'a [bool],
    /// The cell Optimizing code charges its work to, when it may tier up to Peak.
    pub(crate) tier_budget: Option<*const std::cell::Cell<i64>>,
    pub(crate) call_targets: &'a dyn Fn(Ref<Proto>, usize) -> Vec<CallTarget>,
}

/// The compiled unit, and the optimized graph's text when `dump` asks for it.
pub(crate) fn compile<'a>(compiler: &mut Compiler, budget: Arc<CodeBudget>, proto: &'a Proto, facts: &CompileFacts<'a>, dump: bool) -> JitResult<(PeakUnit, Option<String>)> {
    let (Some(layout), Some(frames)) = (facts.layout, facts.frames) else {
        return Err(unsupported("the table and frame layouts are unknown"));
    };
    let func = optimized(proto, facts, (passes::PEAK, false))?;
    let unknown_calls = unknown_calls(proto, facts.forced);
    let text = dump.then(|| func.to_string());
    let points = ssa::points(&func);
    let env = lower::Env {
        layout,
        frames,
        calls: &facts.calls,
        unknown_calls: &unknown_calls,
        inlined_targets: &[],
        vararg: proto.is_vararg,
        points: &points,
        stress_exits: facts.stress_exits,
        tier_budget: None,
    };
    let symbols = lower::symbols();
    let unit = JitUnit::compile(compiler, budget, &symbols, |module, function, context| lower::define(module, function, context, &func, &env))?;
    let mut entries: Vec<u32> = func.osr.iter().map(|&(pc, _)| pc).collect();
    if func.entry.is_some() {
        entries.push(ssa::entry_pc(proto.is_vararg));
    }
    let shapes = func.shapes.clone();
    Ok((PeakUnit { code: ssa::Code::Cranelift { unit: Box::new(unit), _points: points }, entries, shapes }, text))
}

fn inlined_targets(func: &ssa::ir::Func, facts: &CompileFacts<'_>) -> Vec<(Vec<CallTarget>, bool)> {
    func.inlined_calls
        .iter()
        .map(|call| {
            let reference = Ref::new(call.proto as u32, (call.proto >> 32) as u32);
            let unknown = (facts.proto_of)(reference).is_some_and(|proto| super::sites::calls_unknown_lua(proto, call.pc as usize, facts.forced));
            ((facts.call_targets)(reference, call.pc as usize), unknown)
        })
        .collect()
}

fn unknown_calls(proto: &Proto, forced: bool) -> Vec<bool> {
    (0..proto.code.len()).map(|pc| super::sites::calls_unknown_lua(proto, pc, forced)).collect()
}

pub(crate) fn compile_optimizing<'a>(budget: Arc<CodeBudget>, proto: &'a Proto, facts: &CompileFacts<'a>, dump: bool) -> JitResult<(PeakUnit, Option<String>)> {
    let (Some(layout), Some(frames)) = (facts.layout, facts.frames) else {
        return Err(unsupported("the table and frame layouts are unknown"));
    };
    let func = optimized(proto, facts, (passes::OPTIMIZING, true))?;
    let unknown_calls = unknown_calls(proto, facts.forced);
    let inlined_targets = inlined_targets(&func, facts);
    let text = dump.then(|| func.to_string());
    let points = ssa::points(&func);
    let env = lower::Env {
        layout,
        frames,
        calls: &facts.calls,
        unknown_calls: &unknown_calls,
        inlined_targets: &inlined_targets,
        vararg: proto.is_vararg,
        points: &[],
        stress_exits: facts.stress_exits,
        tier_budget: facts.tier_budget,
    };
    let native = native(&func, &env, points, (budget, dump)).ok_or_else(|| unsupported("the Optimizing backend declined the graph"))?;
    let text = text.map(|text| format!("{text}code {}\n", native.listing.as_deref().unwrap_or_default()));
    let mut entries: Vec<u32> = func.osr.iter().map(|&(pc, _)| pc).collect();
    if func.entry.is_some() {
        entries.push(ssa::entry_pc(proto.is_vararg));
    }
    let shapes = func.shapes.clone();
    Ok((PeakUnit { code: ssa::Code::Native(native), entries, shapes }, text))
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn native(func: &ssa::ir::Func, env: &lower::Env<'_>, points: Box<[ssa::DeoptPoint]>, (budget, listing): (Arc<CodeBudget>, bool)) -> Option<ssa::emit::Native> {
    ssa::emit::compile(func, env, points, budget, listing)
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn native(_func: &ssa::ir::Func, _env: &lower::Env<'_>, _points: Box<[ssa::DeoptPoint]>, _budget: (Arc<CodeBudget>, bool)) -> Option<ssa::emit::Native> {
    None
}

/// `inlined_calls` lets inlined frames make Lua calls, which only the Optimizing backend emits.
pub(crate) fn optimized<'a>(proto: &'a Proto, facts: &CompileFacts<'a>, (pipeline, inlined_calls): (&[passes::Pass], bool)) -> JitResult<ssa::ir::Func> {
    let builder_facts = build::Facts {
        inline_caches: facts.inline_caches,
        avoid: facts.avoid,
        insufficient: facts.insufficient,
        interned: facts.interned,
        proto_of: facts.proto_of,
        policy: passes::inline::Policy::default(),
        osr_at: facts.osr_at,
        node_hint: facts.node_hint,
        builtin_of: facts.builtin_of,
        running_floats: facts.running_floats,
        inlined_calls,
    };
    let mut func = build::build(proto, &builder_facts);
    func.float_guess = facts.float_guess;
    func.avoid = facts.avoid.to_vec();
    func.observed = facts.observed.to_vec();
    func.type_names = Some(facts.type_names);
    func.sink_allocations = facts.sink_allocations;
    // V8 verifies Maglev graphs only in DEBUG builds (src/maglev/maglev-compiler.cc, `VerifyGraph`).
    let verify = cfg!(debug_assertions);
    if verify {
        ssa::check::check(&func, false).map_err(|message| unsupported(&format!("built graph: {message}")))?;
    }
    passes::optimize(&mut func, pipeline, verify).map_err(|message| unsupported(&format!("optimized graph {message}")))?;
    Ok(func)
}

fn unsupported(message: &str) -> Box<ModuleError> {
    Box::new(ModuleError::Compilation(CodegenError::Unsupported(message.to_owned())))
}
