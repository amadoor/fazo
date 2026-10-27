use crate::{LuaError, Vm, arena::Ref, object::Closure, proto::Proto};

#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
use crate::object::status as frame_status;

#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
use crate::{instruction::Instruction, instruction::UnaryOp, value::Value};
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
use std::sync::Arc;

#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
use super::{Builtin, CallRequest, CallStart, Direct, Operand, RunScope};
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
use crate::jit::Tier;
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
use crate::jit::{Control, EXIT_ERROR, EXIT_HOOK, EXIT_RELOAD, EXIT_RETURNED, JitCtx, NOT_DIRECT};
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
use crate::jit::ssa::ir::{KIND_FLOAT, KIND_INT, KIND_OTHER};

#[cfg(not(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64"))))]
impl Vm {
    pub(crate) fn jit_budget_spent(
        &mut self,
        _closure: Ref<Closure>,
        _base: usize,
        _proto: Ref<Proto>,
        _pc: usize,
        _registers: *mut crate::value::Value,
    ) -> Result<bool, LuaError> {
        Ok(false)
    }

    pub(crate) fn jit_count_switch(&mut self) {}

    pub(crate) fn jit_prepare(&mut self, _proto: Ref<Proto>, _now: bool) {}

    pub(crate) fn jit_debug_compile(&mut self, _proto: Ref<Proto>, _tier: crate::Tier) -> crate::Tier {
        crate::Tier::Interpreter
    }

    pub(crate) fn jit_never(&mut self, _proto: Ref<Proto>) {}

    pub(crate) fn jit_feedback_changed(&mut self, _proto: Ref<Proto>) {}

    pub(crate) fn jit_deoptimize(&mut self, _proto: Ref<Proto>) {}

    pub(crate) fn jit_tier_of(&self, _proto: Ref<Proto>) -> crate::Tier {
        crate::Tier::Interpreter
    }

    pub fn jit_stats_report(&self) -> String {
        String::new()
    }
}

impl Vm {
    pub(crate) fn closure_proto(&self, closure: Ref<Closure>) -> Option<Ref<Proto>> {
        self.heap.closures.try_get(closure).map(|closure| closure.proto)
    }
}

#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
#[derive(Clone, Copy)]
struct CompiledEntry {
    code: *const u8,
    constants: *const Value,
}

#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
impl Vm {
    /// True when compiled code ran the frame from `pc`.
    #[inline(never)]
    pub(crate) fn jit_budget_spent(
        &mut self,
        closure: Ref<Closure>,
        base: usize,
        proto: Ref<Proto>,
        pc: usize,
        registers: *mut Value,
    ) -> Result<bool, LuaError> {
        let entry = match self.compiled_call_entry(proto, pc) {
            Some(entry) => entry,
            None if pc != 0 && !self.tracing && self.heap.protos.live(proto).jit.code_at(pc).is_some() => match self.jit_resume_compiled(proto, pc) {
                Some(entry) => entry,
                None => return Ok(false),
            },
            None if pc != 0 && !self.tracing && self.jit_resumes_in_baseline(proto, pc) => match self.jit_resume_compiled(proto, pc) {
                Some(entry) => entry,
                None => return Ok(false),
            },
            None => match self.jit_tier_up(proto, pc) {
                Some(entry) => entry,
                None => return Ok(false),
            },
        };
        self.jit_run_entry(closure, base, proto, pc, entry, registers)
    }

    /// Optimized code enters only at the first instruction and at loop headers, so a frame the interpreter resumes elsewhere,
    /// after a return or a yield, gets Baseline code to resume in, which goes back at its next back edge. After a deopt the
    /// interpreter runs the frame instead, since its feedback is what the recompile needs.
    fn jit_resumes_in_baseline(&mut self, proto: Ref<Proto>, pc: usize) -> bool {
        let function = self.heap.protos.live(proto);
        let optimized_only = function.jit.unit.is_some() && function.jit.baseline.is_none();
        if !optimized_only || function.jit.deopts > 0 || function.jit.failed || function.jit.hidden || function.jit.never {
            return false;
        }
        let Some(unit) = self.jit_emit_baseline(proto, true) else {
            return false;
        };
        self.jit_install_baseline(proto, unit);
        self.heap.protos.live(proto).jit.code_at(pc).is_some()
    }

    /// A call's first instruction is neither a resume nor a policy decision, so it skips `jit_tier_up`.
    #[inline(always)]
    fn compiled_call_entry(&self, proto: Ref<Proto>, pc: usize) -> Option<CompiledEntry> {
        if pc != 0 || self.tracing {
            return None;
        }
        let function = self.heap.protos.live(proto);
        Some(CompiledEntry {
            code: function.jit.code()?,
            constants: function.constants.as_ptr(),
        })
    }

    pub(crate) fn jit_count_switch(&mut self) {
        self.jit.switches += 1;
    }

    /// False when the interpreter should run it.
    #[inline(always)]
    fn jit_enter_callee(&mut self, closure: Ref<Closure>, base: usize, proto: Ref<Proto>) -> Result<bool, LuaError> {
        self.jit_compile_declined(proto);
        let function = self.heap.protos.live(proto);
        if function.jit.code().is_none() {
            function.jit.budget().set(function.jit.budget().get().saturating_sub(i64::from(crate::jit::NESTED_CALL_CHARGE)));
        }
        let compiled = function.jit.code().map(|code| CompiledEntry {
            code,
            constants: function.constants.as_ptr(),
        });
        let spent = function.jit.budget().get() <= 0;
        let entry = match compiled {
            Some(entry) => entry,
            None if spent => match self.jit_tier_up(proto, 0) {
                Some(entry) => entry,
                None => return Ok(false),
            },
            None => return Ok(false),
        };
        self.ensure_stack(base + super::REGISTER_SPAN);
        let registers = self.thread_mut().stack.as_mut_ptr().wrapping_add(base);
        self.jit_run_entry(closure, base, proto, 0, entry, registers)
    }

    /// Entering the interpreter from compiled code costs more than the payoff estimate counts, so the callee compiles on its next compiled call.
    fn jit_compile_declined(&mut self, proto: Ref<Proto>) {
        let function = self.heap.protos.live_mut(proto);
        if !function.jit.declined || function.jit.failed {
            return;
        }
        function.jit.declined = false;
        self.jit_compile(proto, Tier::Baseline.min(self.jit_ceiling()));
    }

    /// Behind LuaJIT's `jit.prepare` and `jit.osr`; `osr` brings the tier-up to the next back edge in every tier.
    pub(crate) fn jit_prepare(&mut self, proto: Ref<Proto>, now: bool) {
        let function = self.heap.protos.live_mut(proto);
        function.jit.prepared = true;
        function.jit.primed = true;
        function.jit.declined = false;
        function.jit.never = false;
        if now {
            function.jit.budget().set(0);
            function.jit.tier_budget().set(0);
            return;
        }
        function.jit.budget().set(function.jit.budget().get().min(i64::from(crate::jit::ENTRY_CHARGE)));
    }

    pub(crate) fn jit_debug_compile(&mut self, proto: Ref<Proto>, tier: Tier) -> Tier {
        let function = self.heap.protos.live_mut(proto);
        function.jit.never = false;
        function.jit.failed = false;
        if tier >= Tier::Baseline {
            self.jit_compile(proto, Tier::Baseline);
        }
        if tier >= Tier::Optimizing {
            self.jit_compile(proto, tier);
        }
        self.jit_tier_of(proto)
    }

    pub(crate) fn jit_never(&mut self, proto: Ref<Proto>) {
        self.jit_deoptimize(proto);
        let function = self.heap.protos.live_mut(proto);
        function.jit.never = true;
        function.jit.budget().set(i64::MAX);
    }

    /// The code stays mapped for frames still running in it, and a later tier-up reuses it.
    pub(crate) fn jit_deoptimize(&mut self, proto: Ref<Proto>) {
        self.jit_trace(proto, "deoptimize");
        let function = self.heap.protos.live_mut(proto);
        function.jit.hidden = true;
        function.jit.primed = false;
        function.jit.prepared = false;
        function.jit.entry().set(0);
        function.jit.vararg_entry().set(0);
        function.jit.budget().set(0);
        function.jit.tier_budget().set(i64::MAX);
    }

    pub(crate) fn jit_tier_of(&self, proto: Ref<Proto>) -> Tier {
        self.heap.protos.live(proto).jit.tier()
    }

    pub(crate) fn jit_ceiling(&self) -> Tier {
        self.options.tier
    }

    fn jit_forced(&self) -> Option<Tier> {
        self.options.force.map(|tier| tier.min(self.jit_ceiling())).filter(|&tier| tier != Tier::Interpreter)
    }

    /// A prototype that never runs compiled gets a budget that does not run out. Compiled code can be entered at any instruction, through its dispatch.
    fn jit_tier_up(&mut self, proto: Ref<Proto>, pc: usize) -> Option<CompiledEntry> {
        let ceiling = self.jit_ceiling();
        let tracing = self.tracing;
        let policy = self.options.jit_policy;
        let divisor = self.options.jit_stress.threshold_divisor;
        let switches = self.jit.switches;
        let forced = self.jit_forced();
        let function = self.heap.protos.live_mut(proto);
        if ceiling == Tier::Interpreter || function.jit.failed || function.jit.never {
            function.jit.budget().set(i64::MAX);
            return None;
        }
        if tracing {
            return None;
        }
        if function.jit.code_at(pc).is_some() {
            return self.jit_resume_compiled(proto, pc);
        }
        if let Some(tier) = forced {
            if tier >= Tier::Optimizing && !function.jit.primed {
                function.jit.primed = true;
                function.jit.budget().set(FORCED_PEAK_WAIT);
                self.jit_watch_loop(proto, pc);
                return None;
            }
            return self.jit_compile_entry(proto, tier, pc);
        }
        if function.jit.prepared {
            function.jit.prepared = false;
            return self.jit_compile_entry(proto, Tier::Baseline, pc);
        }
        if !function.jit.primed {
            function.jit.primed = true;
            let budget = policy.baseline.budget(function.code.len(), divisor);
            if budget > 0 {
                function.jit.budget().set(budget);
                self.jit_watch_loop(proto, pc);
                return None;
            }
        }
        if function.jit.observed_from.is_none() && policy.observe_budget > 0 && switches > 0 && crate::jit::calls_out(function) {
            function.jit.observed_from = Some(switches);
            function.jit.budget().set(i64::from(policy.observe_budget));
            self.jit_watch_loop(proto, pc);
            return None;
        }
        if let Some(start) = function.jit.observed_from
            && switches.saturating_sub(start).saturating_mul(u64::from(policy.switch_cost)) >= u64::from(policy.observe_budget)
        {
            return self.jit_set_aside(proto);
        }
        if policy.baseline.estimate_payoff && crate::jit::payoff(function, policy.baseline) <= 0 {
            function.jit.declined = true;
            function.jit.declines += 1;
            let wait = if function.jit.declines > policy.recompile_limit {
                i64::MAX
            } else {
                crate::JitPolicy::backed_off(policy.baseline.budget(function.code.len(), divisor), function.jit.declines)
            };
            function.jit.budget().set(wait);
            self.jit_trace(proto, "baseline declined");
            return None;
        }
        self.jit_compile_entry(proto, Tier::Baseline, pc)
    }

    /// A prototype waiting inside its own body is one long activation, which calls do not count toward feedback, so it gets feedback now.
    fn jit_watch_loop(&mut self, proto: Ref<Proto>, pc: usize) {
        let function = self.heap.protos.live_mut(proto);
        if pc != 0 && function.exec.feedback.is_empty() {
            function.exec.feedback = vec![super::InlineCache::Uninitialized; function.code.len()];
        }
        if self.deferred_loop_charge.is_some() {
            return;
        }
        let function = self.heap.protos.live(proto);
        let enclosing = function.code.iter().enumerate().filter_map(|(start, instruction)| match *instruction {
            Instruction::ForPrep { bx, .. } if start < pc && pc <= start + bx as usize + 1 => Some((start, bx)),
            _ => None,
        });
        let Some((_, bx)) = enclosing.max_by_key(|&(start, _)| start) else {
            return;
        };
        self.deferred_loop_charge = Some(super::DeferredLoopCharge {
            proto,
            charge: bx + 1,
            thread: self.current_thread,
            depth: self.frame_count(),
            watch: true,
        });
        self.requests.fetch_or(crate::session::LOOP_CHARGE, std::sync::atomic::Ordering::Relaxed);
    }

    fn jit_compile_entry(&mut self, proto: Ref<Proto>, tier: Tier, pc: usize) -> Option<CompiledEntry> {
        let compiled = match tier.min(self.jit_ceiling()) {
            tier @ (Tier::Peak | Tier::Optimizing) => self.jit_compile_optimized_at(proto, Some(pc), tier),
            _ => self.jit_compile(proto, tier),
        };
        if !compiled {
            return None;
        }
        let function = self.heap.protos.live(proto);
        Some(CompiledEntry {
            code: function.jit.code_at(pc)?,
            constants: function.constants.as_ptr(),
        })
    }

    /// Falls back to the highest built tier below `tier`.
    pub(crate) fn jit_compile(&mut self, proto: Ref<Proto>, tier: Tier) -> bool {
        match tier.min(self.jit_ceiling()) {
            Tier::Interpreter => false,
            Tier::Baseline => self.jit_compile_baseline(proto),
            Tier::Optimizing => self.jit_compile_optimized_at(proto, None, Tier::Optimizing),
            Tier::Peak => self.jit_compile_peak(proto),
        }
    }

    fn jit_reveal(&mut self, proto: Ref<Proto>) {
        let function = self.heap.protos.live_mut(proto);
        function.jit.hidden = false;
        let entry = function.jit.code().filter(|_| !function.is_vararg).map_or(0, |code| code as usize);
        function.jit.entry().set(entry);
        let hides_arguments = function.is_vararg && !function.vararg_table;
        let vararg_entry = function.jit.code_at(VARARG_BODY).filter(|_| hides_arguments).map_or(0, |code| code as usize);
        function.jit.vararg_entry().set(vararg_entry);
    }

    fn jit_code_budget(&mut self) -> Arc<crate::jit::CodeBudget> {
        Arc::clone(self.jit.budget.get_or_insert_with(|| crate::jit::CodeBudget::new(CODE_BUDGET)))
    }

    fn jit_interned_constants(&self, proto: Ref<Proto>) -> Vec<bool> {
        self.heap.protos.live(proto).constants.iter().map(|&constant| self.jit_interned(constant)).collect()
    }

    /// Unless the VM stops at Baseline, the code charges the Peak budget, which starts from the policy now.
    fn jit_compile_baseline(&mut self, proto: Ref<Proto>) -> bool {
        if self.heap.protos.live(proto).jit.baseline.is_some() {
            self.jit_reveal(proto);
            return true;
        }
        let charges = self.jit_forced().is_none();
        let Some(unit) = self.jit_emit_baseline(proto, charges) else {
            return self.jit_compile_failed(proto);
        };
        self.jit_install_baseline(proto, unit);
        true
    }

    fn jit_install_baseline(&mut self, proto: Ref<Proto>, unit: crate::jit::baseline::BaselineUnit) {
        let function = self.heap.protos.live_mut(proto);
        if function.exec.feedback.is_empty() {
            function.exec.feedback = vec![super::InlineCache::Uninitialized; function.code.len()];
        }
        function.jit.declined = false;
        function.jit.compiled_changes = function.jit.feedback_changes;
        let start = unit.entry() as usize;
        function.jit.baseline = Some(Box::new(unit));
        self.jit_reveal(proto);
        self.jit_trace(proto, &format!("baseline at {start:#x}"));
    }

    fn jit_emit_baseline(&mut self, proto: Ref<Proto>, charges: bool) -> Option<crate::jit::baseline::BaselineUnit> {
        let budget = self.jit_code_budget();
        let layout = *self.jit.layout.get_or_insert_with(crate::jit::TableLayout::probe);
        let interned = self.jit_interned_constants(proto);
        let peak_wait = if charges { self.jit_baseline_wait(proto) } else { i64::MAX };
        let inline_caches = !self.options.jit_stress.inline_caches_off;
        let frames = *self.jit.frames.get_or_insert_with(crate::jit::FrameLayout::probe);
        let calls = self.jit_call_targets(proto);
        if inline_caches {
            self.jit_keep_added_shapes(proto);
        }
        let function = self.heap.protos.live_mut(proto);
        if function.exec.arith_kinds.is_empty() {
            function.exec.arith_kinds = function.code.iter().flat_map(|_| [std::cell::Cell::new(0), std::cell::Cell::new(0)]).collect();
        }
        let function = self.heap.protos.live(proto);
        if charges {
            self.jit_peak_wait(proto, peak_wait);
        } else {
            function.jit.tier_budget().set(i64::MAX);
        }
        let function = self.heap.protos.live(proto);
        let facts = crate::jit::baseline::BaselineFacts {
            interned: &interned,
            tier_budget: charges.then_some(std::ptr::from_ref(function.jit.tier_budget())),
            inline_caches,
            calls: &calls,
        };
        let library = self.jit.library.get_or_insert_with(|| Box::new(crate::jit::baseline::Library::new(layout, frames)));
        crate::jit::baseline::compile(library, budget, function, &facts)
    }

    /// A recompile keeps the work already done toward the next tier: V8 keeps the interrupt budget across feedback changes and
    /// only raises it to `minimum_invocations_after_ic_update` (src/execution/tiering-manager.cc `NotifyICChanged`).
    fn jit_baseline_wait(&mut self, proto: Ref<Proto>) -> i64 {
        let policy = self.options.jit_policy;
        let divisor = self.options.jit_stress.threshold_divisor;
        let declines = self.heap.protos.live(proto).jit.declines;
        let needed = crate::JitPolicy::backed_off(self.jit_peak_budget(proto, Tier::Baseline), declines);
        let executed = self.jit_executed(proto) / i64::from(divisor.max(1));
        if executed <= 0 {
            return needed;
        }
        let after_change = policy.baseline.budget(self.heap.protos.live(proto).code.len(), divisor);
        needed.saturating_sub(executed).max(after_change)
    }

    /// Keeps the child shapes alive so compiled code can move tables into them without checking they exist; a dead one is revived first, as `Heap::take_transition` does.
    fn jit_keep_added_shapes(&mut self, proto: Ref<Proto>) {
        let function = self.heap.protos.live(proto);
        let mut added = Vec::new();
        for (pc, instruction) in function.code.iter().enumerate() {
            if !matches!(instruction, Instruction::SetField { .. }) {
                continue;
            }
            for entry in crate::jit::sites::add_entries(function, pc) {
                added.push(entry.next);
            }
        }
        self.jit_keep_shapes(proto, added);
    }

    fn jit_keep_shapes(&mut self, proto: Ref<Proto>, shapes: Vec<crate::arena::Ref<crate::shape::Shape>>) {
        let function = self.heap.protos.live(proto);
        let mut added = Vec::new();
        for shape in shapes {
            if !function.jit.shapes.contains(&shape) && !added.contains(&shape) {
                added.push(shape);
            }
        }
        for shape in added {
            debug_assert!(self.heap.shapes.marks(shape).is_some(), "the prototype's feedback keeps its shapes alive");
            self.heap.revive_dead_shape(shape);
            self.heap.barrier(crate::gc::Gc::Proto(proto), Some(crate::gc::Gc::Shape(shape)));
            self.heap.protos.live_mut(proto).jit.shapes.push(shape);
        }
    }

    /// The old code stays mapped for the frames still running in it.
    fn jit_recompile_baseline(&mut self, proto: Ref<Proto>, charges: bool) -> bool {
        let Some(unit) = self.jit_emit_baseline(proto, charges) else {
            return false;
        };
        let function = self.heap.protos.live_mut(proto);
        if let Some(old) = function.jit.baseline.replace(Box::new(unit)) {
            function.jit.retired.push(*old);
        }
        function.jit.recompiles += 1;
        function.jit.settled = !charges;
        function.jit.compiled_changes = function.jit.feedback_changes;
        self.jit_reveal(proto);
        self.jit_trace(proto, if charges { "baseline recompiled" } else { "baseline settled" });
        true
    }

    /// Follows V8's `TieringManager::NotifyICChanged` in src/execution/tiering-manager.cc. Baseline code compiled from older feedback reaches its tier check within a budget that doubles with each recompile.
    pub(crate) fn jit_feedback_changed(&mut self, proto: Ref<Proto>) {
        let policy = self.options.jit_policy;
        let divisor = self.options.jit_stress.threshold_divisor;
        let function = self.heap.protos.live_mut(proto);
        function.jit.feedback_changes = function.jit.feedback_changes.wrapping_add(1);
        let runs_baseline = function.jit.baseline.is_some() && function.jit.unit.is_none() && !function.jit.failed && !function.jit.settled;
        if !runs_baseline || function.jit.recompiles >= policy.recompile_limit {
            return;
        }
        let soon = crate::JitPolicy::backed_off(policy.baseline.budget(function.code.len(), divisor), function.jit.recompiles);
        let unspent = function.jit.tier_budget().get();
        if unspent <= soon {
            return;
        }
        if unspent != i64::MAX {
            let dropped = (unspent - soon).saturating_mul(i64::from(divisor.max(1)));
            function.jit.peak_executed = function.jit.peak_executed.saturating_sub(dropped);
        }
        function.jit.tier_budget().set(soon);
    }

    fn jit_compile_peak(&mut self, proto: Ref<Proto>) -> bool {
        self.jit_compile_optimized_at(proto, None, Tier::Peak)
    }

    /// A failed compile leaves the prototype on the code it runs. `at` is the pc the running frame enters the code at.
    fn jit_compile_optimized_at(&mut self, proto: Ref<Proto>, at: Option<usize>, tier: Tier) -> bool {
        let limit = self.options.jit_policy.recompile_limit;
        let function = self.heap.protos.live(proto);
        if function.jit.unit.as_ref().is_some_and(|unit| unit.tier() >= tier) {
            self.jit_reveal(proto);
            return true;
        }
        if function.jit.deopts > limit.max(1) {
            return false;
        }
        let budget = self.jit_code_budget();
        if tier == Tier::Peak && self.jit.compiler.is_none() {
            self.jit.compiler = crate::jit::Compiler::new().ok().map(Box::new);
        }
        let layout = *self.jit.layout.get_or_insert_with(crate::jit::TableLayout::probe);
        let frames = *self.jit.frames.get_or_insert_with(crate::jit::FrameLayout::probe);
        let calls = self.jit_call_targets(proto);
        let estimate = self.jit_estimate(proto, &calls, tier);
        let executed = self.heap.protos.live(proto).jit.peak_executed;
        let charges = tier == Tier::Optimizing && self.jit_ceiling() == Tier::Peak && self.jit_forced().is_none() && self.jit_estimate(proto, &calls, Tier::Peak).gain > 0;
        let judged = self.options.jit_policy.peak.estimate_payoff && self.jit_forced().is_none();
        if judged && !estimate.pays_after(executed) {
            self.jit_decline_peak(proto, estimate);
            return false;
        }
        let has_baseline = self.heap.protos.live(proto).jit.baseline.is_some();
        self.jit_trace_sites(proto);
        let avoid = self.heap.protos.live(proto).jit.avoid.clone();
        let insufficient = self.heap.protos.live(proto).jit.insufficient.clone();
        let observed = observed_kinds(self.heap.protos.live(proto));
        let nodes = self.jit_node_hints(proto);
        let running_floats = if at.is_some() { self.jit_running_floats(proto) } else { Vec::new() };
        let environment = self.jit_environment(proto);
        let dump = self.options.jit_trace;
        let facts_stress = self.options.jit_stress;
        let heap = &self.heap;
        let interned = |constant: Value| matches!(constant, Value::String(string) if heap.key_view().is_interned(string));
        let proto_of = |reference: Ref<Proto>| heap.protos.try_get(reference);
        let builtin_of = |native: Ref<crate::object::Native>| heap.natives.try_get(native).and_then(|native| native.builtin);
        let type_names = self.type_name_refs();
        let call_targets = |inlined: Ref<Proto>, pc: usize| inlined_call_targets(heap, proto, inlined, pc);
        let node_hint = |function: &Proto, pc: usize, root: bool| {
            if root {
                return nodes.get(pc).copied().flatten();
            }
            inlined_node_hint(heap, environment?, function, pc)
        };
        let facts = crate::jit::CompileFacts {
            layout,
            frames,
            calls,
            inline_caches: !facts_stress.inline_caches_off,
            avoid: &avoid,
            insufficient: &insufficient,
            float_guess: self.heap.protos.live(proto).jit.float_guess,
            forced: self.jit_forced().is_some(),
            observed: &observed,
            interned: &interned,
            proto_of: &proto_of,
            stress_exits: facts_stress.exit_every != 0,
            osr_at: at.map(|pc| pc as u32),
            node_hint: &node_hint,
            builtin_of: &builtin_of,
            type_names,
            sink_allocations: self.options.memory_limit.is_none(),
            running_floats: &running_floats,
            tier_budget: charges.then(|| std::ptr::from_ref(heap.protos.live(proto).jit.tier_budget())),
            call_targets: &call_targets,
        };
        let compiled = if tier == Tier::Optimizing {
            Some(crate::jit::compile_optimizing(budget, heap.protos.live(proto), &facts, dump))
        } else {
            self.jit.compiler.as_deref_mut().map(|compiler| crate::jit::compile(compiler, budget, heap.protos.live(proto), &facts, dump))
        };
        let (unit, text) = match compiled {
            Some(Ok((unit, text))) => {
                if self.options.jit_trace {
                    let length = self.heap.protos.live(proto).code.len();
                    let clif_insts = unit.size();
                    let line = format!(
                        "{} estimate (cost {}, gain {} per {} bytecodes, after {executed}; {length} bytecodes, sizes {:?}, loops {}, jumps {}, {clif_insts} {})",
                        tier.name(),
                        estimate.cost,
                        estimate.gain,
                        estimate.weight,
                        estimate.sizes,
                        estimate.loops,
                        estimate.jumps,
                        if tier == Tier::Peak { "Cranelift instructions" } else { "code bytes" }
                    );
                    self.jit_trace(proto, &line);
                }
                (unit, text)
            }
            failure => {
                if let Some(Err(error)) = failure
                    && self.options.jit_trace
                {
                    let line = format!("[jit] peak error: {error:?}\n");
                    self.write_error(line.as_bytes());
                }
                if has_baseline {
                    self.jit_trace(proto, "peak compile failed");
                    return false;
                }
                return self.jit_compile_failed(proto);
            }
        };
        if let Some(text) = text {
            self.jit_trace(proto, "ir");
            self.write_error(text.as_bytes());
        }
        let function = self.heap.protos.live_mut(proto);
        if function.exec.feedback.is_empty() {
            function.exec.feedback = vec![super::InlineCache::Uninitialized; function.code.len()];
        }
        function.jit.tier_budget().set(0);
        function.jit.peak_executed = 0;
        match tier {
            Tier::Peak => function.jit.peak_compiles += 1,
            _ => function.jit.optimizing_compiles += 1,
        }
        function.jit.compiled_bytecodes[usize::from(tier == Tier::Peak)] += estimate.bytecodes();
        let shapes = unit.shapes.clone();
        let start = unit.entry() as usize;
        if let Some(old) = function.jit.unit.replace(Box::new(unit)) {
            function.jit.retired_peak.push(*old);
        }
        self.jit_keep_shapes(proto, shapes);
        self.jit_reveal(proto);
        self.jit_trace(proto, &format!("{} at {start:#x}", tier.name()));
        true
    }

    /// Follows V8's `Deoptimizer::DoComputeOutputFrames` (src/deoptimizer/deoptimizer.cc): the compiled code has
    /// written the frame state's registers back, so this pushes one interpreter frame per inlined call, outer first,
    /// and resumes the innermost at the deoptimizing pc. The unit is retired and the site recompiles without the failed
    /// speculation, as V8 discards optimized code on an eager deopt.
    pub(crate) fn jit_deopt(&mut self, ctx: &mut JitCtx, point: &crate::jit::ssa::DeoptPoint) -> i64 {
        self.jit_push_inlined_frames(ctx, point);
        if point.lazy {
            return -EXIT_RELOAD;
        }
        let proto = ctx.proto;
        let forced = self.jit_forced().is_some();
        let policy = self.options.jit_policy;
        let divisor = self.options.jit_stress.threshold_divisor;
        if self.options.jit_trace {
            self.jit_trace(proto, &format!("deoptimize peak at pc {}", point.site));
        }
        if point.site & crate::jit::ssa::build::SOFT_SITE != 0 {
            return self.jit_soft_deopt(proto, point.site & !crate::jit::ssa::build::SOFT_SITE);
        }
        let kinds = point.frames.is_empty().then(|| self.jit_observed_kinds(ctx, proto, point.site)).flatten();
        let function = self.heap.protos.live_mut(proto);
        function.jit.float_guess |= kinds.is_some_and(|kinds| kinds & KIND_FLOAT != 0);
        let repeated = function.jit.avoid.contains(&point.site);
        let learned = kinds.is_some_and(|kinds| learn_kinds(&mut function.jit.observed, point.site, kinds));
        function.jit.deopts += 1;
        match function.jit.deopt_sites.iter_mut().find(|(site, _)| *site == point.site) {
            Some((_, count)) => *count += 1,
            None => function.jit.deopt_sites.push((point.site, 1)),
        }
        if !repeated && !learned {
            function.jit.avoid.push(point.site);
        }
        if let Some(unit) = function.jit.unit.take() {
            function.jit.retired_peak.push(*unit);
        }
        let deopts = function.jit.deopts;
        let wait = if repeated { crate::JitPolicy::backed_off(self.jit_peak_budget(proto, Tier::Baseline), deopts) } else { policy.peak.budget(0, divisor) };
        self.jit_peak_wait(proto, wait);
        self.jit_reveal(proto);
        if let Some(tier) = self.jit_forced().filter(|_| forced) {
            self.jit_compile(proto, tier);
        }
        -EXIT_RELOAD
    }

    /// Retiers on a fresh budget after an insufficient-feedback deopt, as V8 does.
    fn jit_soft_deopt(&mut self, proto: Ref<Proto>, site: u32) -> i64 {
        let divisor = self.options.jit_stress.threshold_divisor;
        let wait = self.options.jit_policy.peak.budget(0, divisor);
        let function = self.heap.protos.live_mut(proto);
        function.jit.insufficient.push(site);
        function.jit.primed = false;
        if let Some(unit) = function.jit.unit.take() {
            function.jit.retired_peak.push(*unit);
        }
        self.jit_peak_wait(proto, wait);
        self.jit_reveal(proto);
        -EXIT_RELOAD
    }

    /// The kinds the site's registers hold now that the deoptimized frame has them back, as V8 updates feedback from
    /// the values an eager deopt sees.
    fn jit_observed_kinds(&self, ctx: &JitCtx, proto: Ref<Proto>, site: u32) -> Option<u8> {
        let instruction = *self.heap.protos.live(proto).code.get(site as usize)?;
        let (first, second) = numeric_registers(instruction)?;
        let kind = |register: u8| match self.stack_get(ctx.base + usize::from(register)) {
            Value::Integer(_) => KIND_INT,
            Value::Float(_) => KIND_FLOAT,
            _ => KIND_OTHER,
        };
        Some(kind(first) | second.map_or(0, kind))
    }

    fn jit_push_inlined_frames(&mut self, ctx: &JitCtx, point: &crate::jit::ssa::DeoptPoint) {
        let Some(first) = point.frames.first() else {
            self.save_pc(point.pc as usize);
            return;
        };
        self.save_pc(first.caller_pc as usize);
        for (index, frame) in point.frames.iter().enumerate() {
            let base = ctx.base + frame.base as usize;
            let Value::Closure(closure) = self.stack_get(base - 1) else {
                unreachable!("an inlined call's function slot holds the closure its guard checked");
            };
            let pc = point.frames.get(index + 1).map_or(point.pc, |next| next.caller_pc);
            let mut lua = crate::object::LuaFrame::called(closure, base, frame.wanted, 0);
            if frame.tail {
                lua.set_status(lua.status() | crate::object::status::TAIL);
            }
            lua.set_pc(pc as usize);
            self.thread_mut().frames.push(crate::object::Frame::Lua(lua));
        }
    }

    pub(crate) fn jit_set_closure_upvalue(&mut self, closure: Ref<Closure>, index: u8, value: Value) {
        let captured = self.heap.closures.live(closure).upvalues[usize::from(index)];
        if super::window::upvalue_write(self, captured.cell, value) && value.is_collectable() {
            self.heap.closed_upvalue_barrier(captured.reference, value);
        }
    }

    /// Compiles now when the estimated gain pays the next tier's compile, as V8's on-stack replacement does at a loop; otherwise waits for a budget that doubles with each decline.
    pub(crate) fn jit_tier_check(&mut self, ctx: &mut JitCtx, pc: usize, running: Tier) -> i64 {
        let proto = ctx.proto;
        let policy = self.options.jit_policy;
        let function = self.heap.protos.live(proto);
        function.jit.tier_budget().set(i64::MAX);
        if self.tracing || function.jit.failed || function.jit.never {
            return pc as i64;
        }
        if let Some(unit) = function.jit.unit.as_ref().filter(|unit| unit.tier() > running) {
            if unit.enters_at(pc) {
                self.save_pc(pc);
                return -EXIT_RELOAD;
            }
            function.jit.tier_budget().set(i64::from(crate::jit::ENTRY_CHARGE) * 128);
            return pc as i64;
        }
        let stale = running == Tier::Baseline && function.jit.feedback_changes != function.jit.compiled_changes && function.jit.recompiles < policy.recompile_limit;
        if stale {
            return self.jit_recompile_at(proto, pc, true);
        }
        self.heap.protos.live_mut(proto).jit.prepared = false;
        let target = running.next();
        let declined = self.jit_ceiling() < target;
        if declined {
            self.jit_trace(proto, "peak declined");
            let function = self.heap.protos.live_mut(proto);
            function.jit.declines += 1;
            if function.jit.declines >= SETTLE_DECLINES {
                return self.jit_recompile_at(proto, pc, false);
            }
            let declines = function.jit.declines;
            let wait = crate::JitPolicy::backed_off(self.jit_peak_budget(proto, running), declines);
            self.jit_peak_wait(proto, wait);
            return pc as i64;
        }
        if !self.jit_compile_optimized_at(proto, Some(pc), target) {
            return pc as i64;
        }
        self.save_pc(pc);
        -EXIT_RELOAD
    }

    fn jit_estimate(&self, proto: Ref<Proto>, calls: &[Vec<crate::jit::CallTarget>], target: Tier) -> crate::jit::tiering::Estimate {
        let heap = &self.heap;
        let proto_of = |reference: Ref<Proto>| heap.protos.try_get(reference);
        let builtin_of = |native: Ref<crate::object::Native>| heap.natives.try_get(native).and_then(|native| native.builtin);
        let lookup = crate::jit::tiering::Lookup { proto_of: &proto_of, builtin_of: &builtin_of };
        let function = heap.protos.live(proto);
        let mut estimate = crate::jit::tiering::of_proto(function, calls, &lookup, target);
        let optimizing = function.jit.unit.as_ref().filter(|unit| unit.tier() == Tier::Optimizing);
        if let Some(unit) = optimizing.filter(|_| target == Tier::Peak) {
            estimate.cost = crate::jit::tiering::peak_cost_from_optimizing(unit.size() as i64);
        }
        estimate
    }

    /// Work after which the next compile breaks even by the bytecode estimate; with no predicted savings, each bytecode is taken to save one instruction.
    fn jit_peak_budget(&self, proto: Ref<Proto>, running: Tier) -> i64 {
        let divisor = i64::from(self.options.jit_stress.threshold_divisor.max(1));
        let calls = self.jit_call_targets(proto);
        let target = running.next().min(self.jit_ceiling());
        let estimate = self.jit_estimate(proto, &calls, target);
        estimate.break_even().unwrap_or(estimate.cost) / divisor
    }

    fn jit_peak_wait(&mut self, proto: Ref<Proto>, wait: i64) {
        let divisor = i64::from(self.options.jit_stress.threshold_divisor.max(1));
        let executed = self.jit_executed(proto);
        let function = self.heap.protos.live_mut(proto);
        if wait != i64::MAX {
            function.jit.peak_executed = executed.saturating_add(wait.saturating_mul(divisor));
        }
        function.jit.tier_budget().set(wait);
    }

    /// The work done toward the next tier: the budgets handed out less what is left of the current one.
    fn jit_executed(&mut self, proto: Ref<Proto>) -> i64 {
        let divisor = i64::from(self.options.jit_stress.threshold_divisor.max(1));
        let function = self.heap.protos.live_mut(proto);
        let unspent = function.jit.tier_budget().get();
        if unspent > 0 && unspent != i64::MAX {
            function.jit.peak_executed = function.jit.peak_executed.saturating_sub(unspent.saturating_mul(divisor));
            function.jit.tier_budget().set(0);
        }
        function.jit.peak_executed
    }

    /// Waits until the estimated savings would have paid for the compile. A prototype that saves nothing waits for new feedback, which recompiles Baseline and hands out a fresh budget.
    fn jit_decline_peak(&mut self, proto: Ref<Proto>, estimate: crate::jit::tiering::Estimate) {
        let policy = self.options.jit_policy;
        let divisor = i64::from(self.options.jit_stress.threshold_divisor.max(1));
        if self.options.jit_trace {
            self.jit_trace(proto, &format!("peak declined (cost {}, gain {} per {} bytecodes)", estimate.cost, estimate.gain, estimate.weight));
        }
        let function = self.heap.protos.live_mut(proto);
        function.jit.declines += 1;
        let declines = function.jit.declines;
        let executed = function.jit.peak_executed;
        let wait = match estimate.break_even() {
            _ if declines > policy.recompile_limit => i64::MAX,
            Some(needed) => ((needed - executed) / divisor).max(1),
            None => i64::MAX,
        };
        self.jit_peak_wait(proto, wait);
    }

    fn jit_recompile_at(&mut self, proto: Ref<Proto>, pc: usize, charges: bool) -> i64 {
        if !self.jit_recompile_baseline(proto, charges) {
            return pc as i64;
        }
        self.save_pc(pc);
        -EXIT_RELOAD
    }

    pub fn jit_stats_report(&self) -> String {
        let mut lines: Vec<String> = Vec::new();
        for (_, function) in self.heap.protos.iter() {
            let jit = &function.jit;
            if jit.peak_compiles == 0 && jit.optimizing_compiles == 0 && jit.deopts == 0 && jit.recompiles == 0 && jit.declines == 0 {
                continue;
            }
            let sites: Vec<String> = jit.deopt_sites.iter().map(|(site, count)| format!("{site}:{count}")).collect();
            lines.push(format!(
                "jitstats {}:{} tier={} optimizing={} peak={} deopts={} recompiles={} declines={} bytecodes={}/{} sites={}
",
                String::from_utf8_lossy(trace_source(&function.chunkname)).chars().take(60).collect::<String>(),
                function.line_defined,
                jit.tier().name(),
                jit.optimizing_compiles,
                jit.peak_compiles,
                jit.deopts,
                jit.recompiles,
                jit.declines,
                jit.compiled_bytecodes[0],
                jit.compiled_bytecodes[1],
                sites.join(",")
            ));
        }
        lines.sort();
        lines.concat()
    }

    pub(crate) fn jit_trace(&mut self, proto: Ref<Proto>, event: &str) {
        if !self.options.jit_trace {
            return;
        }
        let function = self.heap.protos.live(proto);
        let mut line = format!("[jit] {event} {}:{}", String::from_utf8_lossy(trace_source(&function.chunkname)), function.line_defined).into_bytes();
        line.push(b'\n');
        self.write_error(&line);
    }

    fn jit_trace_sites(&mut self, proto: Ref<Proto>) {
        if !self.options.jit_trace {
            return;
        }
        let function = self.heap.protos.live(proto);
        let mut lines = Vec::new();
        for (pc, cache) in function.exec.feedback.iter().enumerate() {
            let kind = match cache {
                super::InlineCache::Polymorphic(entries) => format!("polymorphic ({} shapes)", entries.len()),
                super::InlineCache::Callees(callees) => format!("polymorphic call ({} callees)", callees.len()),
                super::InlineCache::Megamorphic => "megamorphic".to_string(),
                _ => continue,
            };
            let line = function.lines.get(pc).copied().unwrap_or(0);
            lines.extend_from_slice(format!("[jit] site {}:{line} {kind}\n", String::from_utf8_lossy(trace_source(&function.chunkname))).as_bytes());
        }
        self.write_error(&lines);
    }

    /// Read from the running closure's environment, as LuaJIT's recorder reads the table it specializes `HREFK` on.
    fn jit_node_hints(&self, proto: Ref<Proto>) -> Vec<Option<crate::jit::NodeHint>> {
        let Some(super::Frame::Lua(frame)) = self.thread().frames.last() else {
            return Vec::new();
        };
        let closure = self.heap.closures.live(frame.closure);
        if closure.proto != proto {
            return Vec::new();
        }
        let function = self.heap.protos.live(proto);
        function.code.iter().map(|&instruction| self.jit_node_hint(closure, function, instruction)).collect()
    }

    /// Float-holding registers that OSR speculates on.
    fn jit_running_floats(&self, proto: Ref<Proto>) -> Vec<bool> {
        let Some(super::Frame::Lua(frame)) = self.thread().frames.last() else {
            return Vec::new();
        };
        if self.heap.closures.live(frame.closure).proto != proto {
            return Vec::new();
        }
        let width = usize::from(self.heap.protos.live(proto).max_stack);
        (0..width).map(|register| matches!(self.stack_get(frame.base + register), Value::Float(_))).collect()
    }

    /// The running closure's `_ENV`; an inlined prototype usually shares it, and its guard checks the node either way.
    fn jit_environment(&self, proto: Ref<Proto>) -> Option<Ref<crate::table::Table>> {
        let Some(super::Frame::Lua(frame)) = self.thread().frames.last() else {
            return None;
        };
        let closure = self.heap.closures.live(frame.closure);
        let function = self.heap.protos.live(proto);
        if closure.proto != proto {
            return None;
        }
        let index = function.upvalues.iter().position(|upvalue| upvalue.name.is_some_and(|name| self.heap.bytes(name) == b"_ENV"))?;
        match self.upvalue_value(closure.upvalues.get(index)?.reference) {
            Value::Table(table) => Some(table),
            _ => None,
        }
    }

    fn jit_node_hint(&self, closure: &crate::object::Closure, function: &Proto, instruction: Instruction) -> Option<crate::jit::NodeHint> {
        let (upvalue, constant) = match instruction {
            Instruction::GetTabUp { b, c, .. } => (b, c),
            Instruction::SetTabUp { a, b, .. } => (a, b),
            _ => return None,
        };
        let captured = closure.upvalues.get(usize::from(upvalue))?;
        let Value::Table(table) = self.upvalue_value(captured.reference) else {
            return None;
        };
        let Some(&Value::String(key)) = function.constants.get(usize::from(constant)) else {
            return None;
        };
        let hash = super::window::Window::string_hash(self, key);
        let (node, nodes) = self.heap.tables.live(table).short_str_node(key, hash)?;
        Some(crate::jit::NodeHint { node, nodes })
    }

    fn jit_call_targets(&self, proto: Ref<Proto>) -> Vec<Vec<crate::jit::CallTarget>> {
        let function = self.heap.protos.live(proto);
        function
            .exec
            .feedback
            .iter()
            .map(|cache| {
                let callees: &[super::ic::Callee] = match cache {
                    super::InlineCache::Callee(callee) => std::slice::from_ref(callee),
                    super::InlineCache::Callees(callees) => callees,
                    _ => &[],
                };
                callees.iter().filter_map(|&callee| call_target(&self.heap, proto, callee)).collect()
            })
            .collect()
    }

    fn jit_interned(&self, constant: Value) -> bool {
        matches!(constant, Value::String(string) if self.heap.key_view().is_interned(string))
    }

    pub(crate) fn jit_after_call(&mut self, ctx: &mut JitCtx, pc: usize, code: i64, frames: usize) -> i64 {
        if code == EXIT_ERROR {
            return -EXIT_ERROR;
        }
        if self.call_settled(ctx.thread_ref, frames) && self.jit_continues() {
            self.jit_refresh(ctx);
            return pc as i64 + 1;
        }
        -EXIT_RELOAD
    }

    /// Past the policy's limit the prototype is set aside.
    fn jit_resume_compiled(&mut self, proto: Ref<Proto>, pc: usize) -> Option<CompiledEntry> {
        let limit = self.options.jit_policy.resume_limit(self.heap.protos.live(proto).code.len());
        let function = self.heap.protos.live_mut(proto);
        if pc != 0 {
            function.jit.resumes += 1;
        }
        if limit.is_some_and(|limit| function.jit.resumes > limit) {
            return self.jit_set_aside(proto);
        }
        let function = self.heap.protos.live(proto);
        Some(CompiledEntry {
            code: function.jit.code_at(pc)?,
            constants: function.constants.as_ptr(),
        })
    }

    /// Its compiled code stays mapped, since a frame of it may still run further down the native stack.
    fn jit_set_aside(&mut self, proto: Ref<Proto>) -> Option<CompiledEntry> {
        self.jit_trace(proto, "set aside");
        let policy = self.options.jit_policy;
        let divisor = self.options.jit_stress.threshold_divisor;
        let function = self.heap.protos.live_mut(proto);
        function.jit.set_asides += 1;
        function.jit.entry().set(0);
        function.jit.vararg_entry().set(0);
        if function.jit.set_asides > policy.recompile_limit {
            function.jit.failed = true;
            function.jit.budget().set(i64::MAX);
            return None;
        }
        function.jit.hidden = true;
        function.jit.observed_from = None;
        function.jit.resumes = 0;
        let wait = crate::JitPolicy::backed_off(policy.baseline.budget(function.code.len(), divisor), function.jit.set_asides);
        function.jit.budget().set(wait);
        None
    }

    fn jit_stack_room(&self, ctx: &JitCtx, needed: usize) -> bool {
        std::ptr::from_ref(ctx) as usize > self.jit.stack_floor.saturating_add(needed)
    }

    /// A call that turns hooks on ends compiled code, since only the interpreter runs hooks.
    fn jit_continues(&self) -> bool {
        self.caller_is_idle() && !self.tracing
    }

    fn jit_compile_failed(&mut self, proto: Ref<Proto>) -> bool {
        self.jit_trace(proto, "compile failed");
        let function = self.heap.protos.live_mut(proto);
        function.jit.failed = true;
        function.jit.entry().set(0);
        function.jit.vararg_entry().set(0);
        function.jit.budget().set(i64::MAX);
        false
    }

    fn jit_run_entry(
        &mut self,
        closure: Ref<Closure>,
        base: usize,
        proto: Ref<Proto>,
        pc: usize,
        entry: CompiledEntry,
        registers: *mut Value,
    ) -> Result<bool, LuaError> {
        let address = stack_address();
        if self.jit.entries == 0 {
            self.jit.stack_floor = address.saturating_sub(JIT_NATIVE_STACK_BUDGET);
        } else if address < self.jit.stack_floor.saturating_add(JIT_INTERPRETER_STACK) {
            return Ok(false);
        }
        let mut error = None;
        let direct_results = self.jit_entry_results();
        let mut ctx = JitCtx {
            requests: Arc::as_ptr(&self.requests),
            registers,
            constants: entry.constants,
            tables: self.heap.tables.objects_ptr(),
            table_marks: self.heap.tables.marks_ptr(),
            closures: self.heap.closures.objects_ptr(),
            upvalues: self.heap.closures.live(closure).upvalues.as_ptr(),
            vm: self,
            closure,
            base,
            proto,
            direct_results,
            ret_offset: -(std::mem::size_of::<Value>() as i64),
            thread: self.jit_thread_object(),
            thread_ref: self.current_thread,
            stack_limit: self.jit_stack_limit(),
            error: &raw mut error,
            parent: std::ptr::null_mut(),
            child: std::ptr::null_mut(),
            pending: 0,
            saved_pc: 0,
            frame_index: self.jit_top_index(),
            moves: self.heap.moves,
            inline_point: std::ptr::null(),
        };
        self.jit.entries += 1;
        let code = crate::jit::enter(entry.code, &mut ctx, pc);
        self.jit.entries -= 1;
        debug_assert!(matches!(code, EXIT_RELOAD | EXIT_HOOK | EXIT_ERROR | EXIT_RETURNED));
        match code {
            EXIT_ERROR => Err(error.take().expect("compiled code exits with an error only after recording it")),
            EXIT_RETURNED => {
                self.thread_mut().frames.pop();
                Ok(true)
            }
            _ => Ok(true),
        }
    }

    fn jit_entry_results(&self) -> i32 {
        let thread = self.thread();
        let [.., crate::object::Frame::Lua(caller), crate::object::Frame::Lua(frame)] = thread.frames.as_slice() else {
            return NOT_DIRECT;
        };
        if !frame.nested() || !caller.is_idle() || frame.counted() || frame.hidden_params() != 0 || thread.hook.is_some() {
            return NOT_DIRECT;
        }
        frame.nresults()
    }

    #[inline(never)]
    pub(crate) fn jit_step(&mut self, ctx: &mut JitCtx, pc: usize) -> i64 {
        if self.tracing {
            return self.jit_run_one(ctx, pc);
        }
        match self.run_direct(ctx.closure, ctx.base, ctx.proto, pc) {
            Ok(Direct::Next(next)) if !self.tracing => {
                if self.jit_exit_stressed(next) {
                    return -EXIT_HOOK;
                }
                self.jit_refresh(ctx);
                next as i64
            }
            Ok(Direct::Next(next)) => {
                self.save_pc(next);
                -EXIT_HOOK
            }
            Ok(Direct::Reload) => -EXIT_RELOAD,
            Ok(Direct::Unsupported) => self.jit_run_one(ctx, pc),
            Err(error) => {
                ctx.fail(error);
                -EXIT_ERROR
            }
        }
    }

    fn jit_run_one(&mut self, ctx: &mut JitCtx, pc: usize) -> i64 {
        let frames = self.frame_count();
        let thread = self.current_thread;
        let activation = self.top_activation();
        self.save_pc(pc);
        if let Err(error) = self.run_one(RunScope { thread, frames: frames - 1, return_on_switch: true }) {
            ctx.fail(error);
            return -EXIT_ERROR;
        }
        if self.current_thread != thread || self.frame_count() != frames || self.top_activation() != activation {
            return -EXIT_RELOAD;
        }
        self.jit_refresh(ctx);
        if self.tracing {
            return -EXIT_HOOK;
        }
        self.top_pc() as i64
    }

    pub(crate) fn jit_exit_stressed(&mut self, pc: usize) -> bool {
        let every = u64::from(self.options.jit_stress.exit_every);
        if every == 0 {
            return false;
        }
        self.jit.safe_points += 1;
        if !self.jit.safe_points.is_multiple_of(every) {
            return false;
        }
        self.save_pc(pc);
        true
    }

    pub(crate) fn jit_safe_point(&mut self, ctx: &mut JitCtx, target: usize) -> i64 {
        match self.safe_point(target) {
            Ok(()) => target as i64,
            Err(error) => {
                ctx.fail(error);
                -EXIT_ERROR
            }
        }
    }

    /// A Lua callee's compiled code runs on the native stack above this call while `JIT_NATIVE_STACK_BUDGET` lasts.
    /// A call that turns hooks on leaves the frame to the interpreter, which runs them.
    pub(crate) fn jit_call_direct(&mut self, ctx: &mut JitCtx, pc: usize, a: u8, b: u8, c: u8) -> Control {
        let function_slot = ctx.base + usize::from(a);
        if self.tracing {
            return Control::Interpret;
        }
        let argument_count = if b != 0 {
            usize::from(b) - 1
        } else {
            let Some(count) = self.thread().top.checked_sub(function_slot + 1) else {
                return Control::Interpret;
            };
            count
        };
        let callee = self.stack_get(function_slot);
        self.record_call(ctx.proto, pc, callee);
        match callee {
            Value::Closure(closure) if self.jit_stack_room(ctx, JIT_CALL_STACK) => {
                self.jit_call_compiled(ctx, pc, function_slot, closure, argument_count, c)
            }
            Value::Native(native) if matches!(self.builtin_of(native), Some(Builtin::Resume | Builtin::Wrap)) => {
                self.jit_resume(ctx, pc, native, (function_slot, argument_count), c)
            }
            Value::Native(native) if self.builtin_of(native) == Some(Builtin::Yield) => {
                if self.yield_fast(native, function_slot, argument_count, i32::from(c) - 1, pc + 1) {
                    return Control::Reload;
                }
                self.jit_call_general(ctx, pc)
            }
            Value::Native(native)
                if c == 2
                    && let Some(builtin) = self.builtin_of(native)
                    && let Some(value) = self.builtin_value(builtin, function_slot, argument_count) =>
            {
                self.stack_set(function_slot, value);
                self.jit_refresh(ctx);
                Control::Continue
            }
            Value::Native(native)
                if let Some(builtin) = self.builtin_of(native)
                    && self.call_builtin(builtin, function_slot, argument_count, i32::from(c) - 1) =>
            {
                self.jit_refresh(ctx);
                Control::Continue
            }
            Value::Native(native) if matches!(self.native_body(native), crate::object::NativeBody::Plain(_) | crate::object::NativeBody::Stepped(_)) => {
                self.jit_call_native(ctx, pc, native, argument_count, a, c)
            }
            _ => self.jit_call_general(ctx, pc),
        }
    }

    /// A resume the fast switch does not cover takes the general call.
    fn jit_resume(&mut self, ctx: &mut JitCtx, pc: usize, native: Ref<crate::object::Native>, (function_slot, argument_count): (usize, usize), c: u8) -> Control {
        let thread = self.current_thread;
        let frames = self.frame_count();
        let wrapped = self.builtin_of(native) == Some(Builtin::Wrap);
        if !self.jit_stack_room(ctx, JIT_INTERPRETER_STACK) {
            return self.jit_call_general(ctx, pc);
        }

        if !self.resume_fast(native, wrapped, function_slot, argument_count, i32::from(c) - 1, pc + 1) {
            return self.jit_call_general(ctx, pc);
        }
        let ran = self.jit_run_resumed(thread, frames);
        match ran {
            Err(error) => {
                ctx.fail(error);
                Control::Error
            }
            Ok(()) if self.call_settled(thread, frames) && self.jit_continues() => {
                self.jit_refresh(ctx);
                Control::Continue
            }
            Ok(()) => Control::Reload,
        }
    }

    fn jit_run_resumed(&mut self, thread: Ref<crate::object::Thread>, frames: usize) -> Result<(), LuaError> {
        if !self.switch_landed_idle() {
            self.call_loop(None)?;
        } else if self.jit_enter_landed()? && self.call_settled(thread, frames) {
            return Ok(());
        }
        self.execute(RunScope { thread, frames, return_on_switch: false })
    }

    fn jit_enter_landed(&mut self) -> Result<bool, LuaError> {
        let Some(crate::object::Frame::Lua(frame)) = self.thread().frames.last() else {
            return Ok(false);
        };
        let (closure, base, pc) = (frame.closure, frame.base, frame.pc());
        let proto = self.heap.closures.live(closure).proto;
        let function = self.heap.protos.live(proto);
        let budget = function.jit.budget();
        if function.jit.code().is_none() || budget.get() > i64::from(crate::jit::ENTRY_CHARGE) {
            return Ok(false);
        }
        budget.set(budget.get().wrapping_sub(i64::from(crate::jit::ENTRY_CHARGE)));
        self.ensure_stack(base + super::REGISTER_SPAN);
        let registers = self.thread_mut().stack.as_mut_ptr().wrapping_add(base);
        self.jit_budget_spent(closure, base, proto, pc, registers)
    }

    /// Stepped natives, `__call` and Lua callees past the nesting limit take the interpreter's call instruction.
    fn jit_call_general(&mut self, ctx: &mut JitCtx, pc: usize) -> Control {
        let thread = self.current_thread;
        let frames = self.frame_count();
        match self.run_direct(ctx.closure, ctx.base, ctx.proto, pc) {
            Ok(Direct::Next(_)) if !self.tracing => {
                self.jit_refresh(ctx);
                Control::Continue
            }
            Ok(Direct::Next(_)) => Control::Reload,
            Ok(Direct::Reload) => self.jit_run_nested(ctx, thread, frames),
            Ok(Direct::Unsupported) => Control::Interpret,
            Err(error) => {
                ctx.fail(error);
                Control::Error
            }
        }
    }

    /// Past the nesting limit the caller leaves compiled code and the driver runs the call instead.
    fn jit_run_nested(&mut self, ctx: &mut JitCtx, thread: Ref<crate::object::Thread>, frames: usize) -> Control {
        if self.call_settled(thread, frames) && self.jit_continues() {
            self.jit_refresh(ctx);
            return Control::Continue;
        }
        if !self.jit_stack_room(ctx, JIT_INTERPRETER_STACK) {
            return Control::Reload;
        }
        let ran = self.execute(RunScope { thread, frames, return_on_switch: true });
        match ran {
            Err(error) => {
                ctx.fail(error);
                Control::Error
            }
            Ok(()) if self.call_settled(thread, frames) && self.jit_continues() => {
                self.jit_refresh(ctx);
                Control::Continue
            }
            Ok(()) => Control::Reload,
        }
    }

    /// Returns `EXIT_RETURNED` with the results in place when the call wants them; other callees take `jit_call_lua`'s path.
    fn jit_call_compiled(
        &mut self,
        ctx: &mut JitCtx,
        pc: usize,
        function_slot: usize,
        closure: Ref<Closure>,
        argument_count: usize,
        c: u8,
    ) -> Control {
        let proto = self.heap.closures.live(closure).proto;
        let function = self.heap.protos.live(proto);
        let vararg = function.is_vararg;
        let entry = match vararg {
            false => function.jit.code(),
            true if function.vararg_table => None,
            true => function.jit.code_at(VARARG_BODY),
        };
        let Some(entry) = entry else {
            return self.jit_call_lua(ctx, pc, function_slot, closure, argument_count, c);
        };
        let constants = function.constants.as_ptr();
        let num_params = usize::from(function.num_params);
        let max_stack = usize::from(function.max_stack);
        let function_base = function_slot + 1;
        let frames = self.frame_count();
        let thread = self.current_thread;
        let wanted = i32::from(c) - 1;

        if let Err(error) = self.check_stack(function_base + argument_count + max_stack + usize::from(vararg)) {
            ctx.fail(error);
            return Control::Error;
        }
        self.ensure_stack(function_base + super::REGISTER_SPAN);
        self.count_call_for_feedback(proto);
        let state = self.thread_mut();
        for index in argument_count..num_params {
            state.stack[function_base + index] = Value::Nil;
        }
        if let Some(crate::object::Frame::Lua(caller)) = state.frames.last_mut() {
            caller.set_pc(pc + 1);
        }
        let frame = crate::object::LuaFrame::called(closure, function_base, wanted, 0);
        let frame = if vararg { frame.with_arguments(argument_count) } else { frame };
        state.frames.push(crate::object::Frame::Lua(frame));
        let base = if vararg { self.jit_hide_arguments(function_base, num_params, max_stack) } else { function_base };
        let registers = self.thread_mut().stack.as_mut_ptr().wrapping_add(base);
        let mut callee = JitCtx {
            requests: ctx.requests,
            registers,
            constants,
            tables: self.heap.tables.objects_ptr(),
            table_marks: self.heap.tables.marks_ptr(),
            closures: self.heap.closures.objects_ptr(),
            upvalues: self.heap.closures.live(closure).upvalues.as_ptr(),
            vm: self,
            closure,
            base,
            proto,
            direct_results: if vararg { NOT_DIRECT } else { wanted },
            ret_offset: -(std::mem::size_of::<Value>() as i64),
            thread: ctx.thread,
            thread_ref: ctx.thread_ref,
            stack_limit: ctx.stack_limit,
            error: ctx.error,
            parent: std::ptr::null_mut(),
            child: std::ptr::null_mut(),
            pending: 0,
            saved_pc: 0,
            frame_index: self.jit_top_index(),
            moves: self.heap.moves,
            inline_point: std::ptr::null(),
        };
        let code = crate::jit::enter(entry, &mut callee, if vararg { VARARG_BODY } else { 0 });
        match code {
            EXIT_RETURNED => {
                self.thread_mut().frames.pop();
                self.jit_refresh(ctx);
                Control::Continue
            }
            EXIT_ERROR => Control::Error,
            _ if self.call_settled(thread, frames) && self.jit_continues() => {
                self.jit_refresh(ctx);
                Control::Continue
            }
            _ => Control::Reload,
        }
    }

    /// Runs `VARARGPREP` here as the interpreter does, so compiled code starts past it.
    fn jit_hide_arguments(&mut self, base: usize, params: usize, max_stack: usize) -> usize {
        let total = self.top_frame().expect("the callee's frame was just pushed").argument_count().max(params);
        let moved = base + total + 1;
        self.ensure_stack(moved + super::REGISTER_SPAN.max(max_stack));
        let thread = self.thread_mut();
        super::exec::hide_arguments(&mut thread.stack, base, moved, params);
        let frame = super::window::top_lua_frame(&mut thread.frames);
        frame.base = moved;
        frame.set_hidden(params as u8, total - params);
        moved
    }

    fn jit_call_lua(
        &mut self,
        ctx: &mut JitCtx,
        pc: usize,
        function_slot: usize,
        closure: Ref<Closure>,
        argument_count: usize,
        c: u8,
    ) -> Control {
        let frames = self.frame_count();
        let thread = self.current_thread;
        self.save_pc(pc + 1);
        if let Err(error) = self.push_lua_frame(closure, function_slot, argument_count, i32::from(c) - 1, 0) {
            ctx.fail(error);
            return Control::Error;
        }
        let Some(&crate::object::LuaFrame { base, .. }) = self.top_lua_frame() else {
            unreachable!("a pushed Lua call leaves a Lua frame on top");
        };
        let proto = self.heap.closures.live(closure).proto;
        let ran = self.jit_enter_callee(closure, base, proto);
        match ran {
            Err(error) => {
                ctx.fail(error);
                Control::Error
            }
            Ok(true) if self.call_settled(thread, frames) && self.jit_continues() => {
                self.jit_refresh(ctx);
                Control::Continue
            }
            Ok(true) => Control::Reload,
            Ok(false) => self.jit_run_nested(ctx, thread, frames),
        }
    }

    fn jit_call_native(
        &mut self,
        ctx: &mut JitCtx,
        pc: usize,
        native: Ref<crate::object::Native>,
        argument_count: usize,
        a: u8,
        c: u8,
    ) -> Control {
        let function_slot = ctx.base + usize::from(a);
        let frames = self.frame_count();
        let thread = self.current_thread;
        self.save_pc(pc + 1);
        let call = CallRequest {
            function_slot,
            argument_count,
            nresults: i32::from(c) - 1,
            ccmt: 0,
            operand: Operand::Register(a),
        };
        let quick = argument_count <= super::ARGUMENTS_INLINE;
        let started = match self.native_body(native) {
            crate::object::NativeBody::Plain(function) if quick => self.call_plain(native, function, (function_slot, argument_count, call.nresults)).map(|()| CallStart::Done),
            crate::object::NativeBody::Stepped(function) if quick => self.call_stepped(native, function, (function_slot, argument_count, call.nresults)),
            _ => self.start_native(native, call),
        };
        match started {
            Err(error) => {
                ctx.fail(error);
                Control::Error
            }
            Ok(CallStart::Done) if self.call_settled(thread, frames) && self.jit_continues() => {
                self.jit_refresh(ctx);
                Control::Continue
            }
            Ok(CallStart::Callee(next)) => match self.call_loop(Some(next)) {
                Ok(()) => self.jit_run_nested(ctx, thread, frames),
                Err(error) => {
                    ctx.fail(error);
                    Control::Error
                }
            },
            Ok(_) => Control::Reload,
        }
    }

    /// The outcome is `Reload` unless an error is recorded.
    pub(crate) fn jit_tail_call(&mut self, ctx: &mut JitCtx, pc: usize) -> Control {
        if self.tracing {
            return Control::Interpret;
        }
        if let Instruction::TailCall { a, .. } = self.heap.protos.live(ctx.proto).code[pc] {
            let callee = self.stack_get(ctx.base + usize::from(a));
            self.record_call(ctx.proto, pc, callee);
        }
        if let Some(outcome) = self.jit_tail_fast(ctx, pc) {
            return outcome;
        }
        let thread = self.current_thread;
        let frames = self.frame_count();
        if let Err(error) = self.run_direct(ctx.closure, ctx.base, ctx.proto, pc) {
            ctx.fail(error);
            return Control::Error;
        }
        if !self.call_settled(thread, frames) || !self.jit_stack_room(ctx, JIT_CALL_STACK) {
            return Control::Reload;
        }
        let Some(frame) = self.top_lua_frame() else {
            return Control::Reload;
        };
        let (closure, base, nested) = (frame.closure, frame.base, frame.nested());
        let proto = self.heap.closures.live(closure).proto;
        if let Some(outcome) = self.jit_enter_tail(ctx, closure, base, proto) {
            return outcome;
        }
        // A nested frame's caller belongs to the run that entered this code, so that run interprets the callee.
        if nested {
            return Control::Reload;
        }
        let ran = self.jit_enter_callee(closure, base, proto);
        match ran {
            Err(error) => {
                ctx.fail(error);
                Control::Error
            }
            Ok(true) => Control::Reload,
            Ok(false) => match self.jit_run_nested(ctx, thread, frames - 1) {
                Control::Continue => Control::Reload,
                outcome => outcome,
            },
        }
    }

    /// The frame closes no upvalue, as `TAILCALL` without `k` promises. `None` leaves the call to the interpreter's tail call.
    fn jit_tail_fast(&mut self, ctx: &mut JitCtx, pc: usize) -> Option<Control> {
        let Instruction::TailCall { a, b, .. } = self.heap.protos.live(ctx.proto).code[pc] else {
            return None;
        };
        let frame = self.top_lua_frame()?;
        let (ret, counted) = (frame.ret(), frame.counted());
        if counted || self.thread().hook.is_some() {
            return None;
        }
        let function_slot = ctx.base + usize::from(a);
        let argument_count = if b != 0 { usize::from(b) - 1 } else { self.thread().top.checked_sub(function_slot + 1)? };
        if self.thread().open_upvalues.last().is_some_and(|&upvalue| self.open_index(upvalue) >= ctx.base) {
            self.close_upvalues(ctx.base);
        }

        match self.stack_get(function_slot) {
            Value::Closure(closure) => self.jit_tail_closure(ctx, ret, function_slot, closure, argument_count),
            Value::Native(native) => self.jit_tail_builtin(ctx, ret, function_slot, native, argument_count),
            _ => None,
        }
    }

    fn jit_tail_closure(&mut self, ctx: &mut JitCtx, ret: usize, function_slot: usize, closure: Ref<Closure>, argument_count: usize) -> Option<Control> {
        let proto = self.heap.closures.live(closure).proto;
        self.jit_compile_declined(proto);
        let function = self.heap.protos.live(proto);
        function.jit.code().filter(|_| !function.is_vararg)?;
        let num_params = usize::from(function.num_params);
        let max_stack = usize::from(function.max_stack);
        let base = ret + 1;
        if base + argument_count + max_stack > super::MAX_STACK_SLOTS || !self.jit_stack_room(ctx, JIT_CALL_STACK) {
            return None;
        }

        self.count_call_for_feedback(proto);
        self.ensure_stack(base + super::REGISTER_SPAN.max(argument_count + max_stack));
        let thread = self.thread_mut();
        thread.stack.copy_within(function_slot..function_slot + 1 + argument_count, ret);
        for slot in &mut thread.stack[base + argument_count..base + num_params.max(argument_count)] {
            *slot = Value::Nil;
        }
        if let Some(crate::object::Frame::Lua(frame)) = thread.frames.last_mut() {
            let kept = frame.status() & (frame_status::NRESULTS | frame_status::COUNTED | frame_status::NESTED);
            *frame = crate::object::LuaFrame::with_status(closure, base, kept | frame_status::TAIL);
        }
        self.jit_enter_tail(ctx, closure, base, proto)
    }

    fn jit_tail_builtin(&mut self, ctx: &mut JitCtx, ret: usize, function_slot: usize, native: Ref<crate::object::Native>, argument_count: usize) -> Option<Control> {
        let builtin = self.builtin_of(native)?;
        let wanted = ctx.direct_results;
        if wanted == NOT_DIRECT || !self.call_builtin(builtin, function_slot, argument_count, wanted) {
            return None;
        }

        let thread = self.thread_mut();
        let count = if wanted < 0 { thread.top - function_slot } else { wanted as usize };
        thread.stack.copy_within(function_slot..function_slot + count, ret);
        if wanted < 0 {
            thread.top = ret + count;
        }
        self.jit_refresh(ctx);
        Some(Control::Returned)
    }

    /// `None` leaves a callee that is not compiled to `jit_enter_callee`.
    fn jit_enter_tail(&mut self, ctx: &mut JitCtx, closure: Ref<Closure>, base: usize, proto: Ref<Proto>) -> Option<Control> {
        self.jit_compile_declined(proto);
        let function = self.heap.protos.live(proto);
        let entry = function.jit.code().filter(|_| !function.is_vararg)?;
        let constants = function.constants.as_ptr();

        self.ensure_stack(base + super::REGISTER_SPAN);
        let registers = self.thread_mut().stack.as_mut_ptr().wrapping_add(base);
        let mut callee = JitCtx {
            requests: ctx.requests,
            registers,
            constants,
            tables: self.heap.tables.objects_ptr(),
            table_marks: self.heap.tables.marks_ptr(),
            closures: self.heap.closures.objects_ptr(),
            upvalues: self.heap.closures.live(closure).upvalues.as_ptr(),
            vm: self,
            closure,
            base,
            proto,
            direct_results: ctx.direct_results,
            ret_offset: -(std::mem::size_of::<Value>() as i64),
            thread: self.jit_thread_object(),
            thread_ref: ctx.thread_ref,
            stack_limit: ctx.stack_limit,
            error: ctx.error,
            parent: std::ptr::null_mut(),
            child: std::ptr::null_mut(),
            pending: 0,
            saved_pc: 0,
            frame_index: self.jit_top_index(),
            moves: self.heap.moves,
            inline_point: std::ptr::null(),
        };
        let code = crate::jit::enter(entry, &mut callee, 0);
        Some(match code {
            EXIT_RETURNED => {
                self.jit_refresh(ctx);
                Control::Returned
            }
            EXIT_ERROR => Control::Error,
            _ => Control::Reload,
        })
    }

    /// `count` is `None` when the results run to the stack top. Outcomes are as for `jit_call_direct`; a hook makes the interpreter perform the return.
    pub(crate) fn jit_close_for_return(&mut self, ctx: &JitCtx) -> bool {
        !self.tracing && self.closed_for_return(ctx.base)
    }

    pub(crate) fn jit_return(&mut self, ctx: &mut JitCtx, pc: usize, first: u8, count: Option<usize>) -> Control {
        if self.tracing {
            return Control::Interpret;
        }
        let first_slot = ctx.base + usize::from(first);
        let count = match count {
            Some(count) => count,
            None => match self.thread().top.checked_sub(first_slot) {
                Some(count) => count,
                None => return Control::Interpret,
            },
        };
        self.save_pc(pc + 1);
        if let Err(error) = self.finish_return(first_slot, count, false) {
            ctx.fail(error);
            return Control::Error;
        }
        if !self.caller_is_idle() && let Err(error) = self.call_loop(None) {
            ctx.fail(error);
            return Control::Error;
        }
        Control::Reload
    }

    /// `false` leaves VARARG without a vararg table to the interpreter (`copy_varargs`).
    pub(crate) fn jit_vararg(&mut self, ctx: &mut JitCtx, a: u8, c: u8) -> bool {
        if self.tracing {
            return false;
        }
        let Some(available) = self.top_lua_frame().map(crate::object::LuaFrame::nvarargs) else {
            return false;
        };
        let destination = ctx.base + usize::from(a);
        let wanted = if c == 0 { available } else { usize::from(c) - 1 };
        let copied = wanted.min(available);
        let source = ctx.base - 1 - available;
        self.ensure_stack(destination + wanted);
        let thread = self.thread_mut();
        thread.stack.copy_within(source..source + copied, destination);
        thread.stack[destination + copied..destination + wanted].fill(Value::Nil);
        if c == 0 {
            thread.top = destination + available;
        }
        self.jit_refresh(ctx);
        true
    }

    pub(crate) fn jit_new_table(&mut self, ctx: &mut JitCtx, pc: usize, a: u8, array: u32, hash_log: u8) -> i64 {
        match self.direct_new_table(ctx.closure, ctx.base, ctx.proto, pc, (a, u64::from(array), hash_log)) {
            Ok(next) => {
                self.jit_refresh(ctx);
                next as i64
            }
            Err(error) => {
                ctx.fail(error);
                -EXIT_ERROR
            }
        }
    }

    pub(crate) fn jit_set_list(&mut self, ctx: &mut JitCtx, pc: usize, a: u8, b: u8, c: u8) -> i64 {
        match self.direct_set_list(ctx.closure, ctx.base, ctx.proto, pc, (a, b, c)) {
            Ok(next) => {
                self.jit_refresh(ctx);
                next as i64
            }
            Err(error) => {
                ctx.fail(error);
                -EXIT_ERROR
            }
        }
    }

    pub(crate) fn jit_closure(&mut self, ctx: &mut JitCtx, pc: usize, a: u8, index: usize) -> i64 {
        match self.direct_closure(ctx.closure, ctx.base, ctx.proto, pc, (a, index)) {
            Ok(next) => {
                self.jit_refresh(ctx);
                next as i64
            }
            Err(error) => {
                ctx.fail(error);
                -EXIT_ERROR
            }
        }
    }

    pub(crate) fn jit_table_barrier(&mut self, table: Ref<crate::table::Table>, value: Value) {
        self.heap.barrier_back_for(crate::gc::Gc::Table(table), Value::Nil, value);
    }

    pub(crate) fn jit_raw_set(&mut self, object: Value, key: Value, value: Value) -> bool {
        super::quick::raw_set(self, object, key, value)
    }

    /// `None` leaves the instruction to the interpreter's step.
    pub(crate) fn jit_get_field(&mut self, ctx: &JitCtx, pc: usize, object: Value, key: Value) -> Option<Value> {
        let probe = self.field_probe(ctx.proto, pc, object, key);
        self.field_read_probed(ctx.proto, pc, object, key, probe)
    }

    pub(crate) fn jit_get_method(&mut self, ctx: &JitCtx, pc: usize, object: Value, key: Value) -> Option<Value> {
        let probe = self.field_probe(ctx.proto, pc, object, key);
        self.method_read_probed(ctx.proto, pc, object, key, probe)
    }

    pub(crate) fn jit_set_field(&mut self, ctx: &JitCtx, pc: usize, object: Value, key: Value, value: Value) -> bool {
        self.store_field(ctx.proto, pc, object, key, value)
    }

    /// False leaves the instruction to the interpreter's step.
    pub(crate) fn jit_get_tabup(&mut self, ctx: &mut JitCtx, pc: usize, target: u8, upvalue: u8, key: u32) -> bool {
        let object = self.jit_upvalue_value(ctx, upvalue);
        let key = self.jit_constant(ctx, key);
        let value = *self.field_value_in(|| self.site_feedback(ctx.proto, pc), object, key);
        if !value.is_nil() {
            self.stack_set(ctx.base + usize::from(target), value);
            return true;
        }
        match self.field_read_unprobed(ctx.proto, pc, object, key) {
            Some(value) => {
                self.stack_set(ctx.base + usize::from(target), value);
                true
            }
            None => false,
        }
    }

    /// Operands use the encodings of `crate::jit::jit_operand`. False leaves the instruction to the interpreter's step.
    pub(crate) fn jit_arith(&mut self, ctx: &JitCtx, packed: u32, left: u32, right: u32) -> bool {
        let op = ARITH_OPS[(packed >> 16) as usize];
        let target = (packed & 0xffff) as u8;
        let left = self.jit_encoded(ctx, left);
        let right = self.jit_encoded(ctx, right);
        match super::fast::arith(op, left, right) {
            Some(value) => {
                self.stack_set(ctx.base + usize::from(target), value);
                true
            }
            None => false,
        }
    }

    fn jit_encoded(&self, ctx: &JitCtx, operand: u32) -> Value {
        use crate::jit::jit_operand::{CONSTANT, IMMEDIATE};
        if operand & IMMEDIATE != 0 {
            let bits = operand & 0x3fff_ffff;
            let value = if bits & 0x2000_0000 != 0 { (bits | 0xc000_0000) as i32 } else { bits as i32 };
            Value::Integer(i64::from(value))
        } else if operand & CONSTANT != 0 {
            self.jit_constant(ctx, operand & !CONSTANT)
        } else {
            self.jit_register(ctx, operand as u8)
        }
    }

    /// `None` leaves float loops, zero steps and non-numbers to the interpreter's step.
    pub(crate) fn jit_for_prep(&mut self, ctx: &JitCtx, a: u8) -> Option<bool> {
        let slot = ctx.base + usize::from(a);
        let (Value::Integer(start), Value::Integer(limit), Value::Integer(step)) =
            (self.stack_get(slot), self.stack_get(slot + 1), self.stack_get(slot + 2))
        else {
            return None;
        };
        if step == 0 {
            return None;
        }
        if (step > 0 && start > limit) || (step < 0 && start < limit) {
            return Some(false);
        }
        let count = super::quick::int_loop_count(start, limit, step) as i64;
        self.stack_set(slot, Value::Integer(count));
        self.stack_set(slot + 1, Value::Integer(step));
        self.stack_set(slot + 2, Value::Integer(start));
        Some(true)
    }

    pub(crate) fn caller_is_idle(&self) -> bool {
        matches!(self.top_lua_frame(), Some(frame) if frame.is_idle())
    }

    /// Anything else runs through the interpreter's step.
    pub(crate) fn jit_misc(&mut self, ctx: &mut JitCtx, pc: usize) -> i64 {
        let instruction = self.heap.protos.live(ctx.proto).code[pc];
        match instruction {
            Instruction::Len { a, b } => {
                let value = self.jit_register(ctx, b);
                match value {
                    Value::Table(table) if self.heap.tables.live(table).metatable.is_none() => {
                        match self.unary(UnaryOp::Len, value, Operand::Register(b)) {
                            Ok(Some(length)) => {
                                self.stack_set(ctx.base + usize::from(a), length);
                                (pc + 1) as i64
                            }
                            Ok(None) => self.jit_step(ctx, pc),
                            Err(error) => {
                                ctx.fail(error);
                                -EXIT_ERROR
                            }
                        }
                    }
                    _ => self.jit_step(ctx, pc),
                }
            }
            _ => self.jit_step(ctx, pc),
        }
    }

    pub(crate) fn jit_upvalue_value(&self, ctx: &JitCtx, index: u8) -> Value {
        let cell = self.heap.closures.live(ctx.closure).upvalues[usize::from(index)].cell;
        super::window::upvalue_read(self, cell)
    }

    /// The collector's barrier applies once the upvalue is closed.
    pub(crate) fn jit_set_upvalue(&mut self, ctx: &JitCtx, index: u8, value: Value) {
        let captured = self.heap.closures.live(ctx.closure).upvalues[usize::from(index)];
        if super::window::upvalue_write(self, captured.cell, value) && value.is_collectable() {
            self.heap.closed_upvalue_barrier(captured.reference, value);
        }
    }

    fn jit_register(&self, ctx: &JitCtx, register: u8) -> Value {
        self.stack_get(ctx.base + usize::from(register))
    }

    fn jit_constant(&self, ctx: &JitCtx, index: u32) -> Value {
        self.heap.protos.live(ctx.proto).constants[index as usize]
    }

    /// The constants belong to the same prototype, so they do not change.
    #[inline]
    pub(crate) fn jit_refresh_context(&mut self, ctx: &mut JitCtx) {
        self.jit_refresh(ctx);
    }

    #[inline]
    fn jit_refresh(&mut self, ctx: &mut JitCtx) {
        if ctx.moves != self.heap.moves {
            self.jit_refresh_moved(ctx);
        }
    }

    #[cold]
    #[inline(never)]
    fn jit_refresh_moved(&mut self, ctx: &mut JitCtx) {
        let stack = self.thread_mut().stack.as_mut_ptr();
        ctx.registers = stack.wrapping_add(ctx.base);
        ctx.tables = self.heap.tables.objects_ptr();
        ctx.table_marks = self.heap.tables.marks_ptr();
        ctx.closures = self.heap.closures.objects_ptr();
        ctx.thread = self.jit_thread_object();
        ctx.moves = self.heap.moves;
        let (tables, table_marks, closures, thread, moves) = (ctx.tables, ctx.table_marks, ctx.closures, ctx.thread, ctx.moves);
        ctx.refresh_nest(|other| {
            other.registers = stack.wrapping_add(other.base);
            other.tables = tables;
            other.table_marks = table_marks;
            other.closures = closures;
            other.thread = thread;
            other.moves = moves;
        });
    }

    pub(crate) fn jit_stack_limit(&self) -> usize {
        self.jit.stack_floor.saturating_add(JIT_CALL_STACK)
    }

    fn jit_top_index(&self) -> u32 {
        self.thread().frames.len().saturating_sub(1) as u32
    }

    /// Pushes pending frames outermost first, before VM code reads `thread.frames`.
    #[inline]
    pub(crate) fn jit_push_pending(&mut self, ctx: &mut JitCtx) {
        if ctx.pending != 0 {
            self.jit_push_pending_frames(ctx);
        }
    }

    #[cold]
    #[inline(never)]
    fn jit_push_pending_frames(&mut self, ctx: &mut JitCtx) {
        self.jit_push_pending_frame(ctx, None);
    }

    /// Pushes the outer pending frames first; `pc` is `None` for the innermost, whose helper sets its pc.
    /// A caller inside a call from an inlined frame pushes the inlined frames as a deopt would.
    fn jit_push_pending_frame(&mut self, ctx: &mut JitCtx, pc: Option<u32>) {
        let caller = ctx.pending_caller();
        let resumed = caller.saved_pc;
        if caller.pending != 0 {
            self.jit_push_pending_frame(caller, Some(resumed));
        } else if let Some(crate::object::Frame::Lua(frame)) = self.thread_mut().frames.last_mut() {
            frame.set_pc(resumed as usize);
        }
        if let Some(point) = caller.inline_point() {
            self.jit_push_inlined_frames(caller, point);
        }
        let mut called = crate::object::LuaFrame::called(ctx.closure, ctx.base, ctx.direct_results, 0);
        if let Some((params, extra)) = self.jit_hidden_arguments(ctx) {
            called.set_hidden(params, extra);
        }
        let mut frame = if ctx.pending & crate::jit::PENDING_TAIL != 0 { crate::object::LuaFrame::with_status(ctx.closure, ctx.base, called.status() | frame_status::TAIL) } else { called };
        if let Some(pc) = pc {
            frame.set_pc(pc as usize);
        }
        let frames = &mut self.thread_mut().frames;
        frames.push(crate::object::Frame::Lua(frame));
        ctx.pending = 0;
        ctx.frame_index = frames.len() as u32 - 1;
    }

    /// A direct call into a vararg prototype hid the arguments below the frame, and its results go past them: `ret_offset`
    /// counts the hidden slots, the copied function and the fixed parameters (ltm.c `buildhiddenargs`).
    fn jit_hidden_arguments(&self, ctx: &JitCtx) -> Option<(u8, usize)> {
        let function = self.heap.protos.live(ctx.proto);
        if !function.is_vararg || ctx.ret_offset == -(std::mem::size_of::<Value>() as i64) {
            return None;
        }
        let total = (-ctx.ret_offset) as usize / std::mem::size_of::<Value>() - 2;
        Some((function.num_params, total - usize::from(function.num_params)))
    }

    fn jit_thread_object(&mut self) -> *mut crate::object::Thread {
        std::ptr::from_mut(self.thread_mut())
    }

    /// Checked after helpers that do not refresh the context.
    pub(crate) fn jit_context_current(&self, ctx: &JitCtx) -> bool {
        ctx.registers.cast_const() == self.thread().stack.as_ptr().wrapping_add(ctx.base)
            && ctx.tables == self.heap.tables.objects_ptr()
            && ctx.table_marks == self.heap.tables.marks_ptr()
            && ctx.closures == self.heap.closures.objects_ptr()
            && ctx.thread.cast_const() == std::ptr::from_ref(self.thread())
    }

    /// A tail call replaces the frame in place, so the frame count alone does not show a changed code.
    fn top_activation(&self) -> Option<(crate::arena::Ref<Closure>, usize, usize)> {
        self.top_lua_frame().map(|frame| (frame.closure, frame.base, frame.ret()))
    }

    fn top_pc(&self) -> usize {
        self.top_lua_frame().map_or(0, crate::object::LuaFrame::pc)
    }

    fn top_lua_frame(&self) -> Option<&crate::object::LuaFrame> {
        match self.thread().frames.last() {
            Some(crate::object::Frame::Lua(frame)) => Some(frame),
            _ => None,
        }
    }
}

#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
const ARITH_OPS: [crate::instruction::ArithOp; 12] = [
    crate::instruction::ArithOp::Add,
    crate::instruction::ArithOp::Sub,
    crate::instruction::ArithOp::Mul,
    crate::instruction::ArithOp::Mod,
    crate::instruction::ArithOp::Pow,
    crate::instruction::ArithOp::Div,
    crate::instruction::ArithOp::IDiv,
    crate::instruction::ArithOp::BAnd,
    crate::instruction::ArithOp::BOr,
    crate::instruction::ArithOp::BXor,
    crate::instruction::ArithOp::Shl,
    crate::instruction::ArithOp::Shr,
];

#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
const CODE_BUDGET: usize = 256 * 1024 * 1024;

#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
const VARARG_BODY: usize = 1;
/// A forced Peak compile waits for call feedback, so it builds the code normal tiering would.
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
const FORCED_PEAK_WAIT: i64 = crate::jit::ENTRY_CHARGE as i64 * super::ic::FEEDBACK_CALLS as i64;

/// Declines after which Baseline code whose feedback held still stops charging the tier-up budget, so hot loops cost no more than the jump.
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
const SETTLE_DECLINES: u32 = 2;

/// Native stack the compiled call nesting may use below the outermost entry; past it, calls return to the driver. A quarter of the 1 MiB main thread stack is left to the driver and the host.
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
const JIT_NATIVE_STACK_BUDGET: usize = 256 * 1024;

/// Native stack one nested compiled call level needs before the next check.
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
const JIT_CALL_STACK: usize = if cfg!(debug_assertions) { 2560 } else { 608 };

/// Native stack a nested interpreter run needs before it checks again.
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
const JIT_INTERPRETER_STACK: usize = 64 * 1024;

/// The native stack grows down on every supported target, so a smaller address is deeper.
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline(always)]
fn stack_address() -> usize {
    let marker = 0u8;
    std::hint::black_box(&marker) as *const u8 as usize
}

#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
fn inlined_node_hint(heap: &crate::heap::Heap, environment: Ref<crate::table::Table>, function: &Proto, pc: usize) -> Option<crate::jit::NodeHint> {
    let (upvalue, constant) = match *function.code.get(pc)? {
        Instruction::GetTabUp { b, c, .. } => (b, c),
        Instruction::SetTabUp { a, b, .. } => (a, b),
        _ => return None,
    };
    let name = function.upvalues.get(usize::from(upvalue))?.name?;
    if heap.bytes(name) != b"_ENV" {
        return None;
    }
    let Some(&Value::String(key)) = function.constants.get(usize::from(constant)) else {
        return None;
    };
    let hash = heap.string_keys.get(key.index())?.hash.get();
    let (node, nodes) = heap.tables.try_get(environment)?.short_str_node(key, hash)?;
    Some(crate::jit::NodeHint { node, nodes })
}

/// Drops the `@` or `=` that marks the chunk's kind.
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
fn trace_source(chunkname: &[u8]) -> &[u8] {
    match chunkname.first() {
        Some(b'@' | b'=') => &chunkname[1..],
        _ => chunkname,
    }
}

#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
fn observed_kinds(function: &Proto) -> Vec<(u32, u8)> {
    let mut observed = function.jit.observed.clone();
    for (site, [float, mixed]) in function.exec.arith_kinds.as_chunks::<2>().0.iter().enumerate() {
        let (site, kinds) = (site as u32, float.get() | mixed.get());
        if kinds != 0 && !observed.iter().any(|&(known, _)| known == site) {
            observed.push((site, kinds));
        }
    }
    observed
}

/// True when the site saw number kinds it had not; a recompile then guesses them rather than giving up the site, as V8 widens `kSignedSmall` to `kNumber` (src/objects/feedback-vector.h, `BinaryOperationHint`).
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
fn learn_kinds(observed: &mut Vec<(u32, u8)>, site: u32, kinds: u8) -> bool {
    if kinds & !(KIND_INT | KIND_FLOAT) != 0 {
        return false;
    }
    match observed.iter_mut().find(|(known, _)| *known == site) {
        Some((_, known)) if *known | kinds == *known => false,
        Some((_, known)) => {
            *known |= kinds;
            true
        }
        None => {
            observed.push((site, kinds));
            true
        }
    }
}

/// The registers a numeric instruction reads; constants and immediates have a known kind.
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
fn numeric_registers(instruction: crate::instruction::Instruction) -> Option<(u8, Option<u8>)> {
    use crate::instruction::Instruction as I;
    Some(match instruction {
        I::Add { b, c, .. } | I::Sub { b, c, .. } | I::Mul { b, c, .. } | I::Mod { b, c, .. } | I::Pow { b, c, .. } | I::Div { b, c, .. }
        | I::IDiv { b, c, .. } | I::BAnd { b, c, .. } | I::BOr { b, c, .. } | I::BXor { b, c, .. } | I::Shl { b, c, .. } | I::Shr { b, c, .. } => (b, Some(c)),
        I::AddI { b, .. } | I::AddK { b, .. } | I::SubK { b, .. } | I::MulK { b, .. } | I::ModK { b, .. } | I::PowK { b, .. } | I::DivK { b, .. }
        | I::IDivK { b, .. } | I::BAndK { b, .. } | I::BOrK { b, .. } | I::BXorK { b, .. } | I::ShrI { b, .. } | I::ShlI { b, .. } | I::Unm { b, .. }
        | I::BNot { b, .. } => (b, None),
        I::Eq { a, b, .. } | I::Lt { a, b, .. } | I::Le { a, b, .. } => (a, Some(b)),
        I::LtI { a, .. } | I::LeI { a, .. } | I::GtI { a, .. } | I::GeI { a, .. } => (a, None),
        _ => return None,
    })
}

#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
fn inlined_call_targets(heap: &crate::heap::Heap, caller: Ref<Proto>, proto: Ref<Proto>, pc: usize) -> Vec<crate::jit::CallTarget> {
    let callees: &[super::ic::Callee] = match heap.protos.try_get(proto).and_then(|function| function.exec.feedback.get(pc)) {
        Some(super::InlineCache::Callee(callee)) => std::slice::from_ref(callee),
        Some(super::InlineCache::Callees(callees)) => callees,
        _ => &[],
    };
    callees.iter().filter_map(|&callee| call_target(heap, caller, callee)).collect()
}

/// `caller` is still compiling, so its code is not installed yet.
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
fn call_target(heap: &crate::heap::Heap, caller: Ref<Proto>, callee: super::ic::Callee) -> Option<crate::jit::CallTarget> {
    let (Value::Closure(closure), Some(proto)) = (callee.value, callee.proto) else {
        return None;
    };
    let function = heap.protos.try_get(proto)?;
    if function.vararg_table {
        return None;
    }
    let entry = if function.is_vararg { function.jit.vararg_entry() } else { function.jit.entry() };
    Some(crate::jit::CallTarget {
        closure: closure.packed(),
        proto: proto.packed(),
        entry: std::ptr::from_ref(entry),
        vararg: function.is_vararg,
        constants: function.constants.as_ptr(),
        num_params: function.num_params,
        max_stack: function.max_stack,
        has_upvalues: !function.upvalues.is_empty(),
        pushes_frame: (function.jit.unit.is_none() && proto != caller) || function.is_vararg || crate::jit::sites::needs_frame(function),
    })
}
