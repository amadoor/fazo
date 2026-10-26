//! When a prototype compiles to Peak: the Baseline-to-Peak savings its feedback predicts must pay for the compile.
//! Following V8's `TieringManager` (src/execution/tiering-manager.cc), the work a prototype has done in Baseline stands
//! for the work it will still do, so it compiles once that work times the savings per bytecode covers the compile cost.

use super::CallTarget;
use crate::{instruction::Instruction, proto::Proto, vm::ic::InlineCache};

/// Peak compile Ir outside Cranelift's per-instruction work (module, declarations, finalize, unwind registration), measured on
/// a fib compile, and per bytecode: the least-squares fit of the rest over the perf workloads under `force=peak`.
const COMPILE_IR_PER_UNIT: i64 = 300_000;
const COMPILE_IR_PER_BYTECODE: i64 = 350_000;
const LOOP_WEIGHT: i64 = 8;
const MAX_DEPTH: u32 = 3;

/// Ir saved per execution against Baseline, from the per-operation rows of the scoreboard (Baseline against Peak):
/// `addi` 9.1 against 1.0, `move` 4.1 against 0.5, `getupval` 7.1 against -0.4, `lt` 11.2 against 4.7, `setfield` 30.3
/// against 11.7, `gettabup` 139.1 against 107.4, `len` 192.2 against 99.8, `getfield` 25.4 against 36.3, `gettable`
/// 27.5 against 39.3, `newtable` 407.8 against 413.0, `concat` 349.0 against 531.4, `vararg` 493.2 against 1612.0.
const ARITHMETIC_GAIN: i64 = 9;
const MOVE_GAIN: i64 = 4;
const UPVALUE_GAIN: i64 = 7;
const COMPARE_GAIN: i64 = 6;
const STORE_GAIN: i64 = 12;
const GLOBAL_GAIN: i64 = 30;
const LEN_GAIN: i64 = 90;
const LOAD_COST: i64 = 11;
const HELPER_COST: i64 = 5;
const CONCAT_COST: i64 = 180;
const VARARG_COST: i64 = 1100;
/// A non-integer key goes through the value helper, where Baseline probes the hash part inline.
const GENERIC_TABLE_COST: i64 = 20;
/// A call with no compiled callee goes through the call helper: the `call0` operation costs about 650 more in Peak.
const HELPER_CALL_COST: i64 = 650;
/// A tail call leaves Peak code through its helper: the `tailcall` operation costs about 1000 more in Peak.
const TAIL_CALL_COST: i64 = 1000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Estimate {
    pub(crate) cost: i64,
    pub(crate) gain: i64,
    pub(crate) weight: i64,
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

    pub(crate) fn pays_after(self, executed: i64) -> bool {
        self.break_even().is_some_and(|needed| executed >= needed)
    }
}

/// Before the graph exists, so a decline costs one pass over the bytecode: compile cost from the prototype's length, savings from its bytecodes and their feedback.
pub(crate) fn of_proto(proto: &Proto, calls: &[Vec<CallTarget>]) -> Estimate {
    let depths = super::sites::loop_depths(proto);
    let mut gain: i64 = 0;
    let mut weight: i64 = 0;
    for (pc, &instruction) in proto.code.iter().enumerate() {
        let scale = depth_weight(depths[pc]);
        gain = gain.saturating_add(bytecode_gain(proto, calls, pc, instruction).saturating_mul(scale));
        weight = weight.saturating_add(scale);
    }
    Estimate {
        cost: COMPILE_IR_PER_BYTECODE.saturating_mul(proto.code.len() as i64).saturating_add(COMPILE_IR_PER_UNIT),
        gain,
        weight: weight.max(1),
    }
}

fn depth_weight(depth: u32) -> i64 {
    LOOP_WEIGHT.pow(depth.min(MAX_DEPTH))
}

fn calls_uncompiled_lua(proto: &Proto, calls: &[Vec<CallTarget>], pc: usize) -> bool {
    super::sites::calls_lua(proto, pc) && calls.get(pc).is_none_or(Vec::is_empty)
}

/// A site the interpreter never reached records no feedback and counts for nothing.
fn bytecode_gain(proto: &Proto, calls: &[Vec<CallTarget>], pc: usize, instruction: Instruction) -> i64 {
    let reached = !matches!(proto.exec.feedback.get(pc), None | Some(InlineCache::Uninitialized));
    let cached = super::sites::site_is_cached(proto, pc);
    match instruction {
        Instruction::TailCall { .. } => -TAIL_CALL_COST,
        Instruction::Call { .. } if calls_uncompiled_lua(proto, calls, pc) => -HELPER_CALL_COST,
        Instruction::GetField { .. } | Instruction::SelfOp { .. } | Instruction::SetField { .. } if reached && !cached => -GENERIC_TABLE_COST,
        Instruction::Concat { .. } => -CONCAT_COST,
        Instruction::Vararg { .. } => -VARARG_COST,
        Instruction::NewTable { .. } | Instruction::Closure { .. } | Instruction::SetList { .. } | Instruction::Close { .. } | Instruction::Tbc { .. } => {
            -HELPER_COST
        }
        Instruction::SetTable { .. } | Instruction::SetI { .. } | Instruction::SetField { .. } | Instruction::SetTabUp { .. } => STORE_GAIN,
        Instruction::GetField { .. } | Instruction::SelfOp { .. } | Instruction::GetTable { .. } | Instruction::GetI { .. } => -LOAD_COST,
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
