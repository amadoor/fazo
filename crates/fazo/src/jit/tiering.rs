//! Compiles a prototype to the next tier once the work it has done, times the savings per bytecode its feedback predicts, covers the compile cost, after V8's `TieringManager` (src/execution/tiering-manager.cc).

use super::CallTarget;
use crate::{
    arena::Ref,
    instruction::Instruction,
    object::Native,
    proto::Proto,
    value::Value as LuaValue,
    vm::{Builtin, ic::InlineCache},
};

/// Bytecode predicts a graph's size poorly, so Peak compiles are also estimated from Optimizing code.
const COMPILE_IR_PER_UNIT: i64 = 1_823_874;
const COMPILE_IR_PER_CLASS: Sizes = [0, 876_703, 0, 495_796, 0, 0];
const OPTIMIZING_IR_PER_UNIT: i64 = 166_137;
const OPTIMIZING_IR_PER_CLASS: Sizes = [0, 19_195, 0, 21_132, 1_771, 63_939];
/// Each loop adds blocks, parameters and fixpoint rounds to every pass.
const OPTIMIZING_IR_PER_LOOP: i64 = 187_708;
/// Each forward jump adds blocks, edges and exit states.
const OPTIMIZING_IR_PER_JUMP: i64 = 88_707;
/// Peak compile Ir per byte of the prototype's Optimizing code.
const PEAK_IR_PER_OPTIMIZING_BYTE: i64 = 3_096;
const LOOP_WEIGHT: i64 = 8;
const MAX_DEPTH: u32 = 3;

/// Measured with `cargo xtask perf`.
const ARITHMETIC_GAIN: i64 = 9;
const MOVE_GAIN: i64 = 4;
const UPVALUE_GAIN: i64 = 7;
const COMPARE_GAIN: i64 = 8;
const STORE_GAIN: i64 = 16;
const GLOBAL_GAIN: i64 = 30;
const LEN_GAIN: i64 = 90;
const LOAD_GAIN: i64 = 6;
/// Only an empty constructor allocates inline.
const NEW_TABLE_GAIN: i64 = 200;
const HELPER_COST: i64 = 5;
const CONCAT_COST: i64 = 180;
const VARARG_COST: i64 = 1100;
/// A non-integer key goes through the value helper, where Baseline probes the hash part inline.
const GENERIC_TABLE_COST: i64 = 20;
/// Optimizing tests a polymorphic method's entries one after another.
const POLYMORPHIC_METHOD_COST: i64 = 150;
/// Optimizing stores and reloads every register below a call's base; Baseline keeps them in the frame.
const CALL_REGISTER_COST: i64 = 8;
/// An unseen shape deoptimizes and recompiles; Baseline's chain does not.
const POLYMORPHIC_FIELD_COST: i64 = 40;
/// A call through the call helper costs more than Baseline's.
const HELPER_CALL_COST: i64 = 620;
const DIRECT_CALL_GAIN: i64 = 40;
/// Unknown tail callees exit through the helper.
const TAIL_CALL_COST: i64 = 1000;
const DIRECT_TAIL_CALL_GAIN: i64 = 50;
/// Inlining saves Baseline's call and return.
const INLINED_CALL_GAIN: i64 = 170;
/// An inlined builtin saves Baseline's native call.
const BUILTIN_GAIN: i64 = 250;
/// Matches `passes::inline::Policy` so the estimate inlines what the builder will.
const INLINE_DEPTH: usize = 3;

pub(crate) const CLASSES: usize = 6;
pub(crate) type Sizes = [i64; CLASSES];

fn class(instruction: Instruction) -> usize {
    use Instruction as I;
    match instruction {
        I::Move { .. }
        | I::LoadI { .. }
        | I::LoadF { .. }
        | I::LoadK { .. }
        | I::LoadKx { .. }
        | I::LoadFalse { .. }
        | I::LFalseSkip { .. }
        | I::LoadTrue { .. }
        | I::LoadNil { .. }
        | I::Jmp { .. }
        | I::MmBin { .. }
        | I::MmBinI { .. }
        | I::MmBinK { .. }
        | I::ExtraArg { .. }
        | I::VarargPrep { .. } => 0,
        I::GetUpval { .. } | I::SetUpval { .. } | I::GetTabUp { .. } | I::SetTabUp { .. } => 2,
        I::GetTable { .. } | I::GetI { .. } | I::GetField { .. } | I::SelfOp { .. } | I::Len { .. } => 3,
        I::SetTable { .. } | I::SetI { .. } | I::SetField { .. } | I::SetList { .. } | I::NewTable { .. } => 4,
        I::Call { .. }
        | I::TailCall { .. }
        | I::Return { .. }
        | I::Return0 { .. }
        | I::Return1 { .. }
        | I::Closure { .. }
        | I::Vararg { .. }
        | I::GetVarg { .. }
        | I::Concat { .. }
        | I::TForPrep { .. }
        | I::TForCall { .. }
        | I::TForLoop { .. }
        | I::Tbc { .. }
        | I::Close { .. }
        | I::ErrNil { .. } => 5,
        _ => 1,
    }
}

fn add(sizes: &mut Sizes, other: &Sizes) {
    for (size, more) in sizes.iter_mut().zip(other) {
        *size = size.saturating_add(*more);
    }
}

pub(crate) struct Lookup<'a> {
    pub(crate) proto_of: &'a dyn Fn(Ref<Proto>) -> Option<&'a Proto>,
    pub(crate) builtin_of: &'a dyn Fn(Ref<Native>) -> Option<Builtin>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Estimate {
    pub(crate) cost: i64,
    pub(crate) gain: i64,
    pub(crate) weight: i64,
    /// Includes the inlined callees' bytecodes, since the compile pays for them.
    pub(crate) sizes: Sizes,
    /// Inlined callees have none, since the inlining policy takes no loops.
    pub(crate) loops: i64,
    pub(crate) jumps: i64,
}

impl Estimate {
    /// The Baseline work, in executed bytecodes, whose savings repeated would pay for the compile; `None` when Peak saves
    /// nothing. Compiling there is the break-even rule of the ski rental problem, the best deterministic choice when the
    /// remaining run is unknown (https://en.wikipedia.org/wiki/Ski_rental_problem).
    pub(crate) fn break_even(self) -> Option<i64> {
        if self.gain <= 0 {
            return None;
        }
        Some(self.cost.saturating_mul(self.weight) / self.gain)
    }

    pub(crate) fn bytecodes(self) -> i64 {
        self.sizes.iter().sum()
    }

    pub(crate) fn pays_after(self, executed: i64) -> bool {
        self.break_even().is_some_and(|needed| executed >= needed)
    }
}

/// Before the graph exists, so a decline costs one pass over the bytecode: compile cost from the prototype's length, savings from its bytecodes and their feedback.
/// Inlined callees' bytecodes add to compile cost but not to executed work.
pub(crate) fn of_proto(proto: &Proto, calls: &[Vec<CallTarget>], lookup: &Lookup<'_>, target: crate::Tier) -> Estimate {
    let depths = super::sites::loop_depths(proto);
    let peak = target == crate::Tier::Peak;
    let mut gain: i64 = 0;
    let mut weight: i64 = 0;
    let mut sizes: Sizes = [0; CLASSES];
    for (pc, &instruction) in proto.code.iter().enumerate() {
        let scale = depth_weight(depths[pc]);
        sizes[class(instruction)] += 1;
        let site_gain = match call_inlines(proto, pc, instruction, lookup, 1) {
            Some((site_gain, inlined, peak_gain)) => {
                add(&mut sizes, &inlined);
                if peak { peak_gain } else { site_gain }
            }
            None if peak => peak_gain(instruction),
            None => bytecode_gain(proto, calls, pc, instruction),
        };
        gain = gain.saturating_add(site_gain.saturating_mul(scale));
        weight = weight.saturating_add(scale);
    }
    let loops = super::sites::back_edges(proto) as i64;
    let jumps = super::sites::forward_jumps(proto) as i64;
    let (per_unit, per_class, per_loop, per_jump) = if peak {
        (COMPILE_IR_PER_UNIT, COMPILE_IR_PER_CLASS, 0, 0)
    } else {
        (OPTIMIZING_IR_PER_UNIT, OPTIMIZING_IR_PER_CLASS, OPTIMIZING_IR_PER_LOOP, OPTIMIZING_IR_PER_JUMP)
    };
    let fixed = per_unit.saturating_add(loops.saturating_mul(per_loop)).saturating_add(jumps.saturating_mul(per_jump));
    let cost = sizes.iter().zip(per_class).fold(fixed, |cost, (&size, per)| cost.saturating_add(size.saturating_mul(per)));
    Estimate { cost, gain, weight: weight.max(1), sizes, loops, jumps }
}

/// Cranelift is slower than Optimizing except on table construction and wide calls.
fn peak_gain(instruction: Instruction) -> i64 {
    use Instruction as I;
    match instruction {
        I::NewTable { .. } => 24,
        I::Concat { .. } | I::TailCall { .. } => 2,
        I::GetField { .. } | I::GetTable { .. } | I::GetI { .. } | I::SelfOp { .. } => -15,
        I::Vararg { .. } => -14,
        I::Call { .. } => -3,
        I::Move { .. } | I::LoadI { .. } | I::LoadK { .. } | I::LoadF { .. } | I::GetTabUp { .. } => -3,
        other if super::ssa::flow::is_arithmetic(other) => -3,
        I::SetTable { .. } | I::SetI { .. } | I::SetField { .. } | I::Lt { .. } | I::Le { .. } | I::LtI { .. } | I::LeI { .. } | I::GtI { .. } | I::GeI { .. } => -2,
        I::GetUpval { .. } | I::SetUpval { .. } | I::Len { .. } | I::Closure { .. } => -1,
        _ => 0,
    }
}

pub(crate) fn peak_cost_from_optimizing(code_bytes: i64) -> i64 {
    PEAK_IR_PER_OPTIMIZING_BYTE.saturating_mul(code_bytes)
}

fn call_inlines(proto: &Proto, pc: usize, instruction: Instruction, lookup: &Lookup<'_>, depth: usize) -> Option<(i64, Sizes, i64)> {
    let (b, c) = match instruction {
        Instruction::Call { b, c, .. } if depth == 1 || b != 0 => (b, c),
        Instruction::TailCall { b, .. } if depth > 1 => (b, 1),
        _ => return None,
    };
    if b == 0 || c == 0 || depth > INLINE_DEPTH {
        return None;
    }
    let Some(InlineCache::Callee(callee)) = proto.exec.feedback.get(pc) else {
        return None;
    };
    if let LuaValue::Native(native) = callee.value {
        return (lookup.builtin_of)(native).filter(|&builtin| inlined_builtin(builtin)).map(|_| (BUILTIN_GAIN, [0; CLASSES], 0));
    }
    let function = (lookup.proto_of)(callee.proto?)?;
    let policy = super::ssa::passes::inline::Policy::default();
    let nested = matches!(instruction, Instruction::Call { .. }) && depth > 1;
    if nested && depth > policy.depth {
        return None;
    }
    super::ssa::passes::inline::candidate(function, callee.proto?, &policy)?;
    let mut gain = INLINED_CALL_GAIN;
    let mut over_optimizing = 0;
    let mut sizes: Sizes = [0; CLASSES];
    for (inner, &instruction) in function.code.iter().enumerate() {
        let calls = matches!(instruction, Instruction::Call { .. } | Instruction::TailCall { .. });
        sizes[class(instruction)] += 1;
        let (site_gain, site_over) = match call_inlines(function, inner, instruction, lookup, depth + 1) {
            Some((site_gain, inlined, site_over)) => {
                add(&mut sizes, &inlined);
                (site_gain, site_over)
            }
            None if calls => return None,
            None => (bytecode_gain(function, &[], inner, instruction), peak_gain(instruction)),
        };
        gain = gain.saturating_add(site_gain);
        over_optimizing += site_over;
    }
    Some((gain, sizes, over_optimizing))
}

fn inlined_builtin(builtin: Builtin) -> bool {
    matches!(builtin, Builtin::Type | Builtin::RawEqual | Builtin::RawGet | Builtin::GetMetatable | Builtin::SetMetatable | Builtin::Sqrt)
}

fn depth_weight(depth: u32) -> i64 {
    LOOP_WEIGHT.pow(depth.min(MAX_DEPTH))
}

fn direct_lua_call(proto: &Proto, calls: &[Vec<CallTarget>], pc: usize) -> Option<bool> {
    super::sites::calls_lua(proto, pc).then(|| calls.get(pc).is_some_and(|targets| !targets.is_empty()))
}

/// A site the interpreter never reached records no feedback and counts for nothing.
fn bytecode_gain(proto: &Proto, calls: &[Vec<CallTarget>], pc: usize, instruction: Instruction) -> i64 {
    let reached = !matches!(proto.exec.feedback.get(pc), None | Some(InlineCache::Uninitialized));
    let cached = super::sites::site_is_cached(proto, pc);
    let methods = match proto.exec.feedback.get(pc) {
        Some(InlineCache::Method(methods)) => methods.len(),
        _ => 0,
    };
    let polymorphic = matches!(proto.exec.feedback.get(pc), Some(InlineCache::Polymorphic(_)));
    match instruction {
        Instruction::TailCall { .. } if direct_lua_call(proto, calls, pc) == Some(true) => DIRECT_TAIL_CALL_GAIN,
        Instruction::TailCall { .. } => -TAIL_CALL_COST,
        Instruction::Call { a, .. } => match direct_lua_call(proto, calls, pc) {
            Some(true) => DIRECT_CALL_GAIN - i64::from(a) * CALL_REGISTER_COST,
            Some(false) => -HELPER_CALL_COST,
            None => 0,
        },
        Instruction::GetField { .. } | Instruction::SelfOp { .. } | Instruction::SetField { .. } if reached && !cached => -GENERIC_TABLE_COST,
        Instruction::SelfOp { .. } if methods > 1 => -POLYMORPHIC_METHOD_COST,
        Instruction::GetField { .. } | Instruction::SetField { .. } if polymorphic => -POLYMORPHIC_FIELD_COST,
        Instruction::Concat { .. } => -CONCAT_COST,
        Instruction::Vararg { .. } => -VARARG_COST,
        Instruction::NewTable { b: 0, c: 0, k: false, .. } => NEW_TABLE_GAIN,
        Instruction::NewTable { .. } | Instruction::Closure { .. } | Instruction::SetList { .. } | Instruction::Close { .. } | Instruction::Tbc { .. } => {
            -HELPER_COST
        }
        Instruction::SetTable { .. } | Instruction::SetI { .. } | Instruction::SetField { .. } | Instruction::SetTabUp { .. } => STORE_GAIN,
        Instruction::GetField { .. } | Instruction::SelfOp { .. } | Instruction::GetTable { .. } | Instruction::GetI { .. } => LOAD_GAIN,
        Instruction::GetTabUp { .. } if reached => GLOBAL_GAIN,
        Instruction::Len { .. } => LEN_GAIN,
        Instruction::GetUpval { .. } => UPVALUE_GAIN,
        Instruction::Move { .. } | Instruction::LoadI { .. } | Instruction::LoadK { .. } | Instruction::LoadF { .. } => MOVE_GAIN,
        other if super::ssa::flow::is_arithmetic(other) => ARITHMETIC_GAIN,
        Instruction::Lt { .. }
        | Instruction::Le { .. }
        | Instruction::Eq { .. }
        | Instruction::LtI { .. }
        | Instruction::LeI { .. }
        | Instruction::GtI { .. }
        | Instruction::GeI { .. }
        | Instruction::EqI { .. }
        | Instruction::EqK { .. } => COMPARE_GAIN,
        _ => 0,
    }
}
