//! Compiled tiers modeled on V8's Sparkplug (Baseline) and TurboFan (Peak). Both keep Lua registers
//! in the thread's stack and run what they do not inline through the interpreter, so semantics match.

#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
use cranelift_module::ModuleError;

#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
type JitResult<T> = Result<T, Box<ModuleError>>;

#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
#[expect(unsafe_code, reason = "the Baseline tier maps executable memory for the code it emits")]
pub(crate) mod baseline;
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
pub(crate) mod compile;
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
pub(crate) mod sites;
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
#[expect(unsafe_code, reason = "Peak code reaches the VM and its deoptimization points through raw pointers")]
pub(crate) mod ssa;
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
#[expect(unsafe_code, reason = "a compiled frame records its error through the slot its context names")]
mod ctx;
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
#[expect(
    unsafe_code,
    reason = "the helpers the compiled code calls reach the VM through the context pointer"
)]
pub(crate) mod helpers;
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
#[expect(unsafe_code, reason = "the layout probe reads the bytes of values whose layout Rust does not declare")]
pub(crate) mod layout;
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
#[expect(
    unsafe_code,
    reason = "the JIT maps executable memory and calls compiled code"
)]
mod memory;
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
#[expect(unsafe_code, reason = "the JIT maps executable memory")]
mod sys;
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
#[expect(unsafe_code, reason = "the JIT calls compiled code")]
mod unit;
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
#[expect(unsafe_code, reason = "the JIT registers unwind tables")]
pub(crate) mod unwind;

#[cfg(all(test, feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
#[expect(unsafe_code, reason = "the tests call compiled code")]
mod tests;

#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
pub(crate) use ctx::{JitCtx, NOT_DIRECT};
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
pub(crate) use helpers::{Control, enter};
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
pub(crate) use layout::{FrameLayout, TableLayout};
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
pub(crate) use memory::CodeBudget;
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
pub(crate) use unit::{Compiler, JitUnit};
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
pub(crate) use compile::{CompileFacts, compile};
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
pub(crate) use sites::{CallTarget, calls_out, payoff};

#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
pub(crate) mod jit_operand {
    pub(crate) const CONSTANT: u32 = 1 << 31;
    pub(crate) const IMMEDIATE: u32 = 1 << 30;
}

/// Execution tiers, lowest first, named after V8's Ignition, Sparkplug, Maglev and TurboFan.
/// Only x86-64 and AArch64 build the compiled tiers; elsewhere everything runs in the interpreter.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
pub enum Tier {
    Interpreter,
    Baseline,
    Optimizing,
    #[default]
    Peak,
}

impl Tier {
    /// Accepts `interpreter`, `baseline`, `optimizing` or `peak`, the names `-X tier=` takes.
    pub fn from_name(name: &[u8]) -> Option<Self> {
        match name {
            b"interpreter" => Some(Self::Interpreter),
            b"baseline" => Some(Self::Baseline),
            b"optimizing" => Some(Self::Optimizing),
            b"peak" => Some(Self::Peak),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Interpreter => "interpreter",
            Self::Baseline => "baseline",
            Self::Optimizing => "optimizing",
            Self::Peak => "peak",
        }
    }

    /// Optimizing is not built, so it resolves to Baseline.
    #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub(crate) fn built(self) -> Self {
        match self {
            Self::Optimizing => Self::Baseline,
            other => other,
        }
    }
}

/// When a prototype moves up one tier. A zero budget tiers up on first use.
/// With `estimate_payoff` set, a prototype whose budget runs out compiles only if its estimated
/// payoff over the tier below is positive.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TierPolicy {
    pub budget_base: u32,
    pub budget_per_instruction: u32,
    pub estimate_payoff: bool,
    pub inline_gain: u32,
    pub call_gain: u32,
    pub direct_cost: u32,
    pub loop_weight: u32,
}

impl TierPolicy {
    pub const fn immediate() -> Self {
        Self {
            budget_base: 0,
            budget_per_instruction: 0,
            estimate_payoff: false,
            inline_gain: 0,
            call_gain: 0,
            direct_cost: 0,
            loop_weight: 1,
        }
    }

    #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub(crate) fn budget(self, instructions: usize, divisor: u32) -> i64 {
        let instructions = i64::try_from(instructions).unwrap_or(i64::MAX);
        i64::from(self.budget_per_instruction)
            .saturating_mul(instructions)
            .saturating_add(i64::from(self.budget_base))
            / i64::from(divisor.max(1))
    }
}

/// Zero `observe_budget` compiles at once; zero `resumes_per_instruction` never sets compiled code aside.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct JitPolicy {
    pub baseline: TierPolicy,
    pub peak: TierPolicy,
    pub observe_budget: u32,
    pub switch_cost: u32,
    pub resumes_per_instruction: u32,
    pub recompile_limit: u32,
}

impl JitPolicy {
    /// Tiers up on first use (Peak once the prototype's first calls have recorded feedback) and keeps compiled code; `-X force=` selects it.
    pub const fn forced() -> Self {
        Self {
            baseline: TierPolicy::immediate(),
            peak: TierPolicy::immediate(),
            observe_budget: 0,
            switch_cost: 0,
            resumes_per_instruction: 0,
            recompile_limit: 0,
        }
    }

    #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub(crate) fn backed_off(budget: i64, attempts: u32) -> i64 {
        budget.max(1).saturating_mul(1i64.checked_shl(attempts.min(40)).unwrap_or(i64::MAX))
    }

    #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub(crate) fn resume_limit(self, instructions: usize) -> Option<u64> {
        if self.resumes_per_instruction == 0 {
            return None;
        }
        Some(u64::from(self.resumes_per_instruction).saturating_mul(instructions as u64))
    }
}

impl Default for JitPolicy {
    /// Measured on the perf workloads (`fazo_harness` `perf`).
    fn default() -> Self {
        Self {
            baseline: TierPolicy {
                budget_base: 2_000,
                budget_per_instruction: 100,
                estimate_payoff: true,
                inline_gain: 35,
                call_gain: 35,
                direct_cost: 140,
                loop_weight: 8,
            },
            peak: TierPolicy {
                budget_base: 200_000,
                budget_per_instruction: 80_000,
                estimate_payoff: true,
                inline_gain: 0,
                call_gain: 35,
                direct_cost: 0,
                loop_weight: 8,
            },
            observe_budget: 50_000,
            switch_cost: 750,
            resumes_per_instruction: 64,
            recompile_limit: 8,
        }
    }
}

/// Stress modes for the compiled tiers, modeled on V8's `--stress-maglev`, `--stress-ic` and
/// `--deopt-every-n-times`. They change when and how code tiers, never what a program prints.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct JitStress {
    /// Divides every tier-up budget by this factor; zero and one leave them as they are.
    pub threshold_divisor: u32,
    /// Leaves compiled code at every `exit_every`-th safe point; zero never forces an exit.
    pub exit_every: u32,
    /// Compiles no inline cache: every field access of compiled code goes through its helper.
    pub inline_caches_off: bool,
}

pub(crate) const ENTRY_CHARGE: u32 = 8;

/// Measured on the closures workload: the nested interpreter run a call into an uncompiled prototype takes.
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
pub(crate) const NESTED_CALL_CHARGE: u32 = 64;

#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
pub(crate) const EXIT_RELOAD: i64 = 1;
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
pub(crate) const EXIT_ERROR: i64 = 2;
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
pub(crate) const EXIT_HOOK: i64 = 3;
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
pub(crate) const EXIT_RETURNED: i64 = 4;

#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
#[derive(Default)]
pub(crate) struct JitState {
    pub(crate) budget: Option<std::sync::Arc<CodeBudget>>,
    pub(crate) compiler: Option<Box<Compiler>>,
    pub(crate) layout: Option<Option<TableLayout>>,
    pub(crate) frames: Option<Option<FrameLayout>>,
    pub(crate) call_depth: u32,
    pub(crate) entries: u32,
    pub(crate) stack_floor: usize,
    pub(crate) switches: u64,
    pub(crate) library: Option<Box<baseline::Library>>,
    pub(crate) safe_points: u64,
}

/// Budgets and `entry` sit in their own allocation, so compiled code can hold a pointer that survives
/// the prototype's arena growing. A prototype that moved to Peak keeps its Baseline code mapped, since a frame of it may still be on the native stack.
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
#[derive(Default)]
struct JitCells {
    budget: std::cell::Cell<i64>,
    tier_budget: std::cell::Cell<i64>,
    entry: std::cell::Cell<usize>,
}

#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
#[derive(Default)]
pub(crate) struct ProtoJit {
    cells: Box<JitCells>,
    pub(crate) primed: bool,
    pub(crate) observed_from: Option<u64>,
    pub(crate) failed: bool,
    pub(crate) declined: bool,
    pub(crate) never: bool,
    pub(crate) prepared: bool,
    pub(crate) hidden: bool,
    pub(crate) resumes: u64,
    pub(crate) baseline: Option<Box<baseline::BaselineUnit>>,
    pub(crate) retired: Vec<baseline::BaselineUnit>,
    pub(crate) feedback_changes: u32,
    pub(crate) compiled_changes: u32,
    pub(crate) recompiles: u32,
    pub(crate) declines: u32,
    pub(crate) set_asides: u32,
    pub(crate) settled: bool,
    pub(crate) unit: Option<Box<ssa::PeakUnit>>,
    /// Peak units replaced after a deoptimization, kept mapped for the frames that may still run them.
    pub(crate) retired_peak: Vec<ssa::PeakUnit>,
    /// Sites whose speculation failed; a recompile runs them through the interpreter's helpers.
    pub(crate) avoid: Vec<u32>,
    pub(crate) deopts: u32,
    pub(crate) float_guess: bool,
    /// Number kinds a numeric site's registers held when its guess failed, as `ssa::ir::KIND_*` bits.
    pub(crate) observed: Vec<(u32, u8)>,
    /// Kept alive while the prototype lives, so compiled code never names a released shape.
    pub(crate) shapes: Vec<crate::arena::Ref<crate::shape::Shape>>,
}

#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
impl ProtoJit {
    pub(crate) fn budget(&self) -> &std::cell::Cell<i64> {
        &self.cells.budget
    }

    pub(crate) fn tier_budget(&self) -> &std::cell::Cell<i64> {
        &self.cells.tier_budget
    }

    pub(crate) fn entry(&self) -> &std::cell::Cell<usize> {
        &self.cells.entry
    }

    pub(crate) fn code(&self) -> Option<*const u8> {
        self.code_at(0)
    }

    /// Peak code is entered only at the first instruction and at loop headers; anywhere else Baseline runs the frame.
    pub(crate) fn code_at(&self, pc: usize) -> Option<*const u8> {
        if self.failed || self.hidden {
            return None;
        }
        match (&self.unit, &self.baseline) {
            (Some(unit), _) if unit.enters_at(pc) => Some(unit.entry()),
            (_, Some(baseline)) => Some(baseline.entry()),
            _ => None,
        }
    }

    pub(crate) fn tier(&self) -> Tier {
        match (self.failed || self.hidden, &self.unit, &self.baseline) {
            (true, _, _) => Tier::Interpreter,
            (false, Some(_), _) => Tier::Peak,
            (false, None, Some(_)) => Tier::Baseline,
            (false, None, None) => Tier::Interpreter,
        }
    }

    #[cfg(test)]
    pub(crate) fn is_compiled(&self) -> bool {
        self.code().is_some()
    }
}

#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
impl Clone for ProtoJit {
    fn clone(&self) -> Self {
        Self::default()
    }
}

#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
impl std::fmt::Debug for ProtoJit {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ProtoJit")
            .field("budget", &self.budget().get())
            .field("primed", &self.primed)
            .field("failed", &self.failed)
            .field("resumes", &self.resumes)
            .field("tier", &self.tier())
            .finish()
    }
}
