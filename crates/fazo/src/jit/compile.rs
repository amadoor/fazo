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
    pub(crate) call_depth_limit: u32,
    pub(crate) inline_caches: bool,
    pub(crate) avoid: &'a [u32],
    pub(crate) float_guess: bool,
    pub(crate) observed: &'a [(u32, u8)],
    pub(crate) interned: &'a dyn Fn(LuaValue) -> bool,
    pub(crate) proto_of: &'a dyn Fn(Ref<Proto>) -> Option<&'a Proto>,
    pub(crate) stress_exits: bool,
    pub(crate) require_payoff: bool,
}

/// The compiled unit, and the optimized graph's text when `dump` asks for it.
pub(crate) fn compile<'a>(compiler: &mut Compiler, budget: Arc<CodeBudget>, proto: &'a Proto, facts: &CompileFacts<'a>, dump: bool) -> JitResult<(PeakUnit, Option<String>)> {
    let (Some(layout), Some(frames)) = (facts.layout, facts.frames) else {
        return Err(unsupported("the table and frame layouts are unknown"));
    };
    let func = optimized(proto, facts)?;
    if facts.require_payoff && payoff(&func) <= 0 {
        return Err(unsupported("the optimized graph specializes no more than Baseline does"));
    }
    let text = dump.then(|| func.to_string());
    let points = ssa::points(&func);
    let env = lower::Env {
        layout,
        frames,
        calls: &facts.calls,
        call_depth_limit: facts.call_depth_limit,
        points: &points,
        stress_exits: facts.stress_exits,
    };
    let symbols = lower::symbols();
    let unit = JitUnit::compile(compiler, budget, &symbols, |module, function, context| lower::define(module, function, context, &func, &env))?;
    let mut entries: Vec<u32> = func.osr.iter().map(|&(pc, _)| pc).collect();
    entries.push(0);
    Ok((PeakUnit { unit, _points: points, entries }, text))
}

/// Ops weighted by loop depth: unboxed and array-part ops count for Peak, generic table ops and helpers against it.
fn payoff(func: &ssa::ir::Func) -> i64 {
    use ssa::ir::{Op, UnOp};
    let mut total: i64 = 0;
    for block in func.reverse_postorder() {
        let depth = func.loops.iter().filter(|entry| entry.blocks.contains(&block)).count().min(3) as u32;
        let weight = 8i64.pow(depth);
        for &value in &func.block(block).insts {
            let gain = match func.op(value) {
                Op::Unary(UnOp::Box | UnOp::Unbox, _) => 0,
                Op::Binary(..) | Op::Unary(..) | Op::Pure(..) | Op::LoadArray { .. } | Op::LoadSlot { .. } | Op::StoreSlot { .. } => 1,
                Op::Index { .. } | Op::NewIndex { .. } | Op::Helper(_) | Op::Arith { .. } | Op::Compare { .. } => -1,
                _ => 0,
            };
            total = total.saturating_add(gain * weight);
        }
    }
    total
}

pub(crate) fn optimized<'a>(proto: &'a Proto, facts: &CompileFacts<'a>) -> JitResult<ssa::ir::Func> {
    let builder_facts = build::Facts {
        inline_caches: facts.inline_caches,
        avoid: facts.avoid,
        interned: facts.interned,
        proto_of: facts.proto_of,
        policy: passes::inline::Policy::default(),
    };
    let mut func = build::build(proto, &builder_facts);
    func.float_guess = facts.float_guess;
    func.avoid = facts.avoid.to_vec();
    func.observed = facts.observed.to_vec();
    ssa::check::check(&func, false).map_err(|message| unsupported(&format!("built graph: {message}")))?;
    passes::optimize(&mut func, cfg!(debug_assertions)).map_err(|message| unsupported(&format!("optimized graph {message}")))?;
    ssa::check::check(&func, true).map_err(|message| unsupported(&format!("optimized graph: {message}")))?;
    Ok(func)
}

fn unsupported(message: &str) -> Box<ModuleError> {
    Box::new(ModuleError::Compilation(CodegenError::Unsupported(message.to_owned())))
}
