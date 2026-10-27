//! Ported from `lower.rs` (`call_lua`, `direct_call`, `lower_return`); mirrors `Vm::jit_call_compiled` and `moveresults` in ldo.c.

use std::mem::offset_of;

use super::{
    asm::{Alu, Asm, Cond, Label, Width},
    codegen::{CLOSURE_TAG, Gen, PAYLOAD, Src, VALUE_SIZE},
};
use crate::vm::exec_layout;
use crate::jit::{
    JitCtx,
    layout::lua_frame,
    sites::{self, CallTarget},
    ssa::ir::{MULTIPLE_RESULTS, Val},
};

/// A callee known from feedback, or one whose facts the call reads from its closure's `Exec`, as Baseline's `CallAny` does.
#[derive(Clone, Copy)]
enum Callee<'t> {
    Known(&'t CallTarget),
    Any,
}

#[derive(Clone, Copy)]
struct Site {
    pc: usize,
    slot: u32,
    b: u8,
    c: u8,
    inlined: Option<u32>,
}

/// Registers a direct call works in, all clobbered by the call.
#[derive(Clone, Copy)]
struct Temps {
    exec: u8,
    closure: u8,
    entry: u8,
    depth_pointer: u8,
    depth: u8,
    thread: u8,
    callee_base: u8,
    count: u8,
    frame: u8,
}

impl<A: Asm> Gen<'_, A> {
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

    pub(super) fn call_lua(&mut self, value: Val, pc: usize, (a, b, c): (u8, u8, u8)) {
        let targets = self.call_targets(pc);
        let out = self.out(value);
        let temps = self.temps();
        let saves = self.saves(&[]);
        let helper = self.asm.new_label();
        let join = self.asm.new_label();
        let settled = (saves.is_empty() && self.switched == Some(value)).then(|| self.asm.new_label());
        if let Some(label) = settled {
            self.settled = Some((pc as u32 + 1, label));
        }
        self.save(&saves);
        let unknown = targets.is_empty() && self.env.unknown_calls.get(pc).copied().unwrap_or(false);
        if b != 0 && (unknown || !targets.is_empty()) {
            let site = Site { pc, slot: u32::from(a), b, c, inlined: None };
            self.direct_calls(site, (&targets, unknown), temps, (helper, join, out), settled);
        }
        self.asm.bind(helper);
        self.fast_args(pc as u32, [u32::from(a), u32::from(b), u32::from(c)]);
        self.asm.call(self.addresses.fast[sites::FAST_CALL]);
        self.asm.mov(out, A::result());
        self.asm.bind(join);
        self.restore(&saves);
    }

    /// A call from an inlined frame takes no helper: its slow path resumes the call in the interpreter.
    pub(super) fn call_inlined(&mut self, value: Val, index: usize) {
        let call = self.func.inlined_calls[index];
        let (targets, unknown) = self.env.inlined_targets.get(index).cloned().unwrap_or_default();
        let frames = self.func.state(call.resume).frames.len() as u32;
        let out = self.out(value);
        let temps = self.temps();
        let saves = self.saves(&[]);
        let interpret = self.asm.new_label();
        let join = self.asm.new_label();
        let settled = (saves.is_empty() && self.switched == Some(value)).then(|| self.asm.new_label());
        if let Some(label) = settled {
            self.settled = Some((call.pc + 1, label));
        }
        self.save(&saves);
        if unknown || !targets.is_empty() {
            let point = std::ptr::from_ref(&self.table.points[call.resume.0 as usize]) as i64;
            self.asm.mov_imm(A::SCRATCH, point);
            self.asm.store(Width::Q, A::SCRATCH, A::CTX, offset_of!(JitCtx, inline_point) as i32);
            let site = Site { pc: call.pc as usize, slot: call.slot, b: call.b, c: call.c, inlined: Some(frames) };
            self.direct_calls(site, (&targets, unknown), temps, (interpret, join, out), settled);
        }
        self.asm.bind(interpret);
        let exit = self.exit(call.eager);
        self.asm.jump(exit);
        self.asm.bind(join);
        self.restore(&saves);
    }

    fn direct_calls(&mut self, site: Site, (targets, unknown): (&[CallTarget], bool), temps: Temps, (helper, join, out): (Label, Label, u8), settled: Option<Label>) {
        let any = if unknown { self.asm.new_label() } else { helper };
        let matched = self.match_callee(site.slot, targets, temps, (helper, any));
        for (target, label) in targets.iter().zip(matched) {
            self.asm.bind(label);
            self.direct_call(site, Callee::Known(target), temps, (helper, join, out), settled);
        }
        if unknown {
            self.asm.bind(any);
            self.direct_call(site, Callee::Any, temps, (helper, join, out), settled);
        }
    }

    fn match_callee(&mut self, slot: u32, targets: &[CallTarget], temps: Temps, (helper, any): (Label, Label)) -> Vec<Label> {
        let at = slot as i32 * VALUE_SIZE;
        let matched: Vec<Label> = targets.iter().map(|_| self.asm.new_label()).collect();
        self.registers(temps.entry);
        self.asm.load(Width::B, A::SCRATCH, temps.entry, at);
        self.asm.cmp_imm(A::SCRATCH, i32::from(CLOSURE_TAG));
        self.asm.branch(Cond::Ne, helper);
        self.asm.load(Width::Q, temps.closure, temps.entry, at + PAYLOAD);
        if targets.is_empty() {
            self.asm.jump(any);
            return matched;
        }
        for (target, &label) in targets.iter().zip(&matched) {
            self.asm.cmp_imm64(temps.closure, target.closure as i64);
            self.asm.branch(Cond::Eq, label);
        }
        self.closure_address(temps.entry, temps.closure);
        self.asm.load(Width::Q, temps.entry, temps.entry, offset_of!(crate::object::Closure, proto) as i32);
        for (target, &label) in targets.iter().zip(&matched) {
            self.asm.cmp_imm64(temps.entry, target.proto as i64);
            self.asm.branch(Cond::Eq, label);
        }
        self.asm.jump(any);
        matched
    }

    /// Leaves the callee's heap frame pending, as V8's optimized frames stay on the native stack (maglev-ir.cc `CallKnownJSFunction`).
    fn direct_call(&mut self, site: Site, callee: Callee<'_>, t: Temps, (helper, join, out): (Label, Label, u8), settled: Option<Label>) {
        let Site { pc, slot: function, b, c, inlined } = site;
        let arguments = u32::from(b) - 1;
        let wanted = i64::from(c) - 1;
        let callee_offset = (function as i32 + 1) * VALUE_SIZE;
        let slot = self.ctx_at;
        let popped = self.asm.new_label();
        let other = self.asm.new_label();

        self.load_entry(callee, t, helper);
        self.stack_check(t, helper);
        self.ctx_load(t.callee_base, offset_of!(JitCtx, base));
        self.asm.alu_imm(Alu::Add, t.callee_base, t.callee_base, function as i32 + 1);
        self.check_stack_room(callee, arguments, t, helper);
        self.clear_missing_parameters(callee, arguments, callee_offset, t);
        let saved = if inlined.is_some() { i64::from(crate::jit::INLINED_CALL) } else { pc as i64 + 1 };
        self.asm.store_imm(Width::W, A::CTX, offset_of!(JitCtx, saved_pc) as i32, saved);
        self.fill_callee_context(callee, t, (callee_offset, wanted));
        if inlined.is_none() && matches!(callee, Callee::Known(target) if target.pushes_frame) {
            self.push_callee_frame(pc, t, wanted);
        }

        self.asm.mov(A::SCRATCH, t.entry);
        self.arg_address(0, slot);
        self.place_arg(1, Src::I(0));
        self.asm.call_reg(A::SCRATCH);
        self.asm.mov(A::SCRATCH, A::result());
        self.asm.cmp_imm(A::SCRATCH, crate::jit::EXIT_RETURNED as i32);
        self.asm.branch(Cond::Ne, other);
        self.asm.cmp_mem_byte(A::SP, slot + offset_of!(JitCtx, pending) as i32, 0);
        self.asm.branch(Cond::Eq, popped);
        self.continue_after_call(pc, (join, out), settled);

        self.asm.bind(popped);
        self.pop_callee_frame(t, 1 + inlined.unwrap_or(0));
        self.continue_after_call(pc, (join, out), settled);

        self.asm.bind(other);
        if inlined.is_some() {
            self.leave_after_inlined_call();
            return;
        }
        self.place_arg(2, Src::R(A::SCRATCH));
        match A::arg(3) {
            super::asm::ArgLoc::Reg(reg) => self.asm.load(Width::W, reg, A::SP, slot + offset_of!(JitCtx, frame_index) as i32),
            super::asm::ArgLoc::Stack(_) => unreachable!("four integer arguments go in registers"),
        }
        self.place_arg(4, Src::I(0));
        self.place_arg(1, Src::I(pc as i64));
        self.place_arg(0, Src::R(A::CTX));
        self.asm.call(self.addresses.fast[sites::FAST_AFTER_CALL]);
        self.asm.mov(out, A::result());
        self.asm.jump(join);
    }

    /// Pushes the heap frame now, as the callee's first helper would; a pending caller or full frame vector stays lazy.
    fn push_callee_frame(&mut self, pc: usize, t: Temps, wanted: i64) {
        let frames = self.env.frames;
        let slot = self.ctx_at;
        let lazy = self.asm.new_label();
        let status = i64::from(crate::object::LuaFrame::call_status(wanted as i32, 0)) << lua_frame::STATUS_SHIFT;

        self.asm.cmp_mem_byte(A::CTX, offset_of!(JitCtx, pending) as i32, 0);
        self.asm.branch(Cond::Ne, lazy);
        self.ctx_load(t.thread, offset_of!(JitCtx, thread));
        self.asm.load(Width::Q, t.count, t.thread, frames.frames_len);
        self.asm.load(Width::Q, A::SCRATCH, t.thread, frames.frames_capacity);
        self.asm.cmp(t.count, A::SCRATCH);
        self.asm.branch(Cond::AboveEq, lazy);
        self.asm.alu_imm(Alu::Mul, t.frame, t.count, frames.frame_size as i32);
        self.asm.add_mem(t.frame, t.thread, frames.frames_data);
        self.asm.store_imm(Width::W, t.frame, frames.field(lua_frame::PC) - frames.frame_size as i32, pc as i64 + 1);
        self.asm.store_imm(Width::B, t.frame, 0, 0);
        self.asm.store(Width::Q, t.closure, t.frame, frames.field(lua_frame::CLOSURE));
        self.asm.store(Width::Q, t.callee_base, t.frame, frames.field(lua_frame::BASE));
        for word in 0..lua_frame::CACHE_WORDS {
            self.asm.store_imm(Width::Q, t.frame, frames.field(lua_frame::CACHE + word * std::mem::size_of::<usize>()), 0);
        }
        self.asm.store_imm(Width::Q, t.frame, frames.field(lua_frame::PC), status);
        self.asm.store_imm(Width::Q, t.frame, frames.field(lua_frame::NVARARGS), 0);
        self.asm.store(Width::W, t.count, A::SP, slot + offset_of!(JitCtx, frame_index) as i32);
        self.asm.alu_imm(Alu::Add, t.count, t.count, 1);
        self.asm.store(Width::Q, t.count, t.thread, frames.frames_len);
        self.asm.store_imm(Width::W, A::SP, slot + offset_of!(JitCtx, pending) as i32, 0);
        self.asm.bind(lazy);
    }

    /// A callee that did not return in place was pushed with the inlined frames under it, so the interpreter resumes them.
    fn leave_after_inlined_call(&mut self) {
        let failed = self.asm.new_label();
        self.asm.cmp_imm(A::SCRATCH, crate::jit::EXIT_ERROR as i32);
        self.asm.branch(Cond::Eq, failed);
        self.asm.mov_imm(A::SCRATCH, -crate::jit::EXIT_RELOAD);
        self.asm.jump(self.exit_with);
        self.asm.bind(failed);
        self.asm.mov_imm(A::SCRATCH, -crate::jit::EXIT_ERROR);
        self.asm.jump(self.exit_with);
    }

    fn continue_after_call(&mut self, pc: usize, (join, out): (Label, u8), settled: Option<Label>) {
        if let Some(label) = settled {
            self.asm.jump(label);
            return;
        }
        self.asm.mov_imm(out, pc as i64 + 1);
        self.asm.jump(join);
    }

    /// The caller pops the callee's heap frame, left on top, as the driver does for `EXIT_RETURNED`.
    /// The callee's first helper pushed the inlined frames of a call from one under its own, so those go too.
    fn pop_callee_frame(&mut self, t: Temps, count: u32) {
        let frames = self.env.frames;
        self.ctx_load(t.thread, offset_of!(JitCtx, thread));
        self.asm.load(Width::Q, t.count, t.thread, frames.frames_len);
        self.asm.alu_imm(Alu::Sub, t.count, t.count, count as i32);
        self.asm.store(Width::Q, t.count, t.thread, frames.frames_len);
    }

    /// One compare against the context limit covers every native frame below (V8 `StackCheck`, maglev-assembler-x64.cc).
    fn stack_check(&mut self, t: Temps, helper: Label) {
        self.asm.lea(t.depth, A::SP, 0);
        self.ctx_load(t.depth_pointer, offset_of!(JitCtx, stack_limit));
        self.asm.cmp(t.depth, t.depth_pointer);
        self.asm.branch(Cond::BelowEq, helper);
    }

    /// A vararg prototype's entry cell stays zero, so its calls take the helper.
    fn load_entry(&mut self, callee: Callee<'_>, t: Temps, helper: Label) {
        match callee {
            Callee::Known(target) => self.asm.mov_imm(t.entry, target.entry as i64),
            Callee::Any => {
                self.closure_address(t.exec, t.closure);
                self.asm.load(Width::Q, t.exec, t.exec, exec_layout::CLOSURE_CODE as i32);
                self.asm.load(Width::Q, t.entry, t.exec, exec_layout::ENTRY as i32);
            }
        }
        self.asm.load(Width::Q, t.entry, t.entry, 0);
        self.asm.cmp_imm(t.entry, 0);
        self.asm.branch(Cond::Eq, helper);
    }

    /// Follows `luaD_precall` (ldo.c); the call limit covers `REGISTER_SPAN` slots past the base.
    fn check_stack_room(&mut self, callee: Callee<'_>, arguments: u32, t: Temps, helper: Label) {
        let frames = self.env.frames;
        self.ctx_load(t.thread, offset_of!(JitCtx, thread));
        self.asm.load(Width::Q, t.count, t.thread, frames.call_limit);
        self.asm.cmp(t.callee_base, t.count);
        self.asm.branch(Cond::Above, helper);
        let register_count = match callee {
            Callee::Known(target) => i64::from(arguments) + i64::from(target.max_stack),
            Callee::Any => i64::from(arguments) + i64::from(u8::MAX),
        };
        if register_count <= crate::vm::REGISTER_SPAN as i64 {
            return;
        }
        self.asm.load(Width::Q, t.count, t.thread, frames.stack_len);
        self.asm.alu_imm(Alu::Add, A::SCRATCH, t.callee_base, register_count as i32);
        self.asm.cmp(A::SCRATCH, t.count);
        self.asm.branch(Cond::Above, helper);
    }

    fn clear_missing_parameters(&mut self, callee: Callee<'_>, arguments: u32, callee_offset: i32, t: Temps) {
        if let Callee::Known(target) = callee {
            if arguments >= u32::from(target.num_params) {
                return;
            }
            self.registers(A::SCRATCH);
            for parameter in arguments..u32::from(target.num_params) {
                self.asm.store_imm(Width::B, A::SCRATCH, callee_offset + parameter as i32 * VALUE_SIZE, 0);
            }
            return;
        }
        let next = self.asm.new_label();
        let done = self.asm.new_label();
        self.asm.load(Width::B, t.count, t.exec, exec_layout::NUM_PARAMS as i32);
        self.asm.mov_imm(t.frame, i64::from(arguments));
        self.asm.bind(next);
        self.asm.cmp(t.frame, t.count);
        self.asm.branch(Cond::AboveEq, done);
        self.asm.shl_imm(A::SCRATCH, t.frame, 4);
        self.asm.add_mem(A::SCRATCH, A::CTX, offset_of!(JitCtx, registers) as i32);
        self.asm.store_imm(Width::B, A::SCRATCH, callee_offset, 0);
        self.asm.alu_imm(Alu::Add, t.frame, t.frame, 1);
        self.asm.jump(next);
        self.asm.bind(done);
    }

    /// Written at the activation's first call; the dispatch clears `requests` on every entry.
    fn fill_callee_context(&mut self, callee: Callee<'_>, t: Temps, (callee_offset, wanted): (i32, i64)) {
        let context = self.ctx_at;
        let filled = self.asm.new_label();
        self.asm.load(Width::Q, A::SCRATCH, A::SP, context + offset_of!(JitCtx, requests) as i32);
        self.asm.cmp_imm(A::SCRATCH, 0);
        self.asm.branch(Cond::Ne, filled);
        for offset in [
            offset_of!(JitCtx, requests),
            offset_of!(JitCtx, vm),
            offset_of!(JitCtx, thread_ref),
            offset_of!(JitCtx, stack_limit),
            offset_of!(JitCtx, error),
            offset_of!(JitCtx, tables),
            offset_of!(JitCtx, table_marks),
            offset_of!(JitCtx, closures),
            offset_of!(JitCtx, thread),
            offset_of!(JitCtx, moves),
        ] {
            self.ctx_load(A::SCRATCH, offset);
            self.asm.store(Width::Q, A::SCRATCH, A::SP, context + offset as i32);
        }
        self.asm.store_imm(Width::Q, A::SP, context + offset_of!(JitCtx, ret_offset) as i32, -i64::from(VALUE_SIZE));
        self.asm.store(Width::Q, A::CTX, A::SP, context + offset_of!(JitCtx, parent) as i32);
        self.asm.bind(filled);
        self.registers(A::SCRATCH);
        self.asm.alu_imm(Alu::Add, A::SCRATCH, A::SCRATCH, callee_offset);
        self.asm.store(Width::Q, A::SCRATCH, A::SP, context + offset_of!(JitCtx, registers) as i32);
        self.callee_code_fields(callee, t, (A::SP, context));
        self.asm.store(Width::Q, t.closure, A::SP, context + offset_of!(JitCtx, closure) as i32);
        self.asm.store(Width::Q, t.callee_base, A::SP, context + offset_of!(JitCtx, base) as i32);
        self.asm.store_imm(Width::W, A::SP, context + offset_of!(JitCtx, direct_results) as i32, wanted);
        self.asm.store_imm(Width::W, A::SP, context + offset_of!(JitCtx, pending) as i32, i64::from(crate::jit::PENDING));
        self.asm.store_imm(Width::Q, A::SP, context + offset_of!(JitCtx, child) as i32, 0);
    }

    fn callee_code_fields(&mut self, callee: Callee<'_>, t: Temps, (base, context): (u8, i32)) {
        if matches!(callee, Callee::Known(target) if !target.has_upvalues) {
            self.asm.store_imm(Width::Q, base, context + offset_of!(JitCtx, upvalues) as i32, 0);
        } else {
            self.closure_address(t.frame, t.closure);
            self.asm.load(Width::Q, t.frame, t.frame, self.env.frames.closure_upvalues);
            self.asm.store(Width::Q, t.frame, base, context + offset_of!(JitCtx, upvalues) as i32);
        }
        match callee {
            Callee::Known(target) => {
                self.asm.store_imm(Width::Q, base, context + offset_of!(JitCtx, constants) as i32, target.constants as i64);
                self.asm.store_imm(Width::Q, base, context + offset_of!(JitCtx, proto) as i32, target.proto as i64);
            }
            Callee::Any => {
                self.asm.load(Width::Q, A::SCRATCH, t.exec, exec_layout::CONSTANTS as i32);
                self.asm.store(Width::Q, A::SCRATCH, base, context + offset_of!(JitCtx, constants) as i32);
                self.asm.load(Width::Q, A::SCRATCH, t.exec, exec_layout::PROTO as i32);
                self.asm.store(Width::Q, A::SCRATCH, base, context + offset_of!(JitCtx, proto) as i32);
            }
        }
    }

    /// The stack check finishes before `count` and `frame` take values, so it borrows them.
    fn temps(&mut self) -> Temps {
        let exec = self.temp();
        let closure = self.temp();
        let entry = self.temp();
        let thread = self.temp();
        let callee_base = self.temp();
        let count = self.temp();
        let frame = self.temp();

        Temps { exec, closure, entry, depth_pointer: count, depth: frame, thread, callee_base, count, frame }
    }

    pub(super) fn tail_call(&mut self, pc: u32, (a, b, k): (u8, u8, bool)) {
        let targets = self.call_targets(pc as usize);
        let helper = self.asm.new_label();
        let unknown = targets.is_empty() && self.env.unknown_calls.get(pc as usize).copied().unwrap_or(false);
        if (unknown || !targets.is_empty()) && b != 0 && !k && !self.env.vararg {
            let temps = self.temps();
            let any = if unknown { self.asm.new_label() } else { helper };
            let matched = self.match_callee(u32::from(a), &targets, temps, (helper, any));
            for (target, label) in targets.iter().zip(matched) {
                self.asm.bind(label);
                self.direct_tail_call((a, b), Callee::Known(target), temps, helper);
            }
            if unknown {
                self.asm.bind(any);
                self.direct_tail_call((a, b), Callee::Any, temps, helper);
            }
        }
        self.asm.bind(helper);
        self.fast_args(pc, [0, 0, 0]);
        self.asm.call(self.addresses.fast[sites::FAST_TAIL_CALL]);
        self.asm.mov(A::SCRATCH, A::result());
        self.asm.jump(self.exit_with);
    }

    /// Follows `luaD_pretailcall` (ldo.c); the jump keeps the caller's return address, as LuaJIT's `BC_CALLT` (lj_frame.h).
    fn direct_tail_call(&mut self, (a, b): (u8, u8), callee: Callee<'_>, t: Temps, helper: Label) {
        let frames = self.env.frames;
        let arguments = u32::from(b) - 1;
        let lazy = self.asm.new_label();
        let moved = self.asm.new_label();

        self.load_entry(callee, t, helper);
        self.ctx_load(t.callee_base, offset_of!(JitCtx, base));
        self.check_stack_room(callee, arguments, t, helper);
        self.asm.load(Width::W, A::SCRATCH, A::CTX, offset_of!(JitCtx, pending) as i32);
        self.asm.cmp_imm(A::SCRATCH, 0);
        self.asm.branch(Cond::Ne, lazy);
        self.top_frame(t);
        self.asm.load(Width::W, A::SCRATCH, t.frame, frames.field(lua_frame::STATUS));
        self.asm.test_imm(A::SCRATCH, lua_frame::OWN_CALLER as i32);
        self.asm.branch(Cond::Ne, helper);
        self.replace_frame(t);
        self.asm.jump(moved);
        self.asm.bind(lazy);
        self.asm.store_imm(Width::W, A::CTX, offset_of!(JitCtx, pending) as i32, i64::from(crate::jit::PENDING | crate::jit::PENDING_TAIL));
        self.asm.bind(moved);

        self.move_down(a, arguments, t.count);
        self.clear_missing_parameters(callee, arguments, 0, t);
        self.callee_code_fields(callee, t, (A::CTX, 0));
        self.asm.store(Width::Q, t.closure, A::CTX, offset_of!(JitCtx, closure) as i32);
        self.asm.store_imm(Width::Q, A::CTX, offset_of!(JitCtx, child) as i32, 0);
        self.asm.mov(A::SCRATCH, t.entry);
        self.asm.jump(self.tail_exit);
    }

    fn top_frame(&mut self, t: Temps) {
        let frames = self.env.frames;
        self.asm.load(Width::Q, t.count, t.thread, frames.frames_len);
        self.asm.alu_imm(Alu::Sub, t.count, t.count, 1);
        self.asm.alu_imm(Alu::Mul, t.frame, t.count, frames.frame_size as i32);
        self.asm.load(Width::Q, A::SCRATCH, t.thread, frames.frames_data);
        self.asm.alu(Alu::Add, t.frame, t.frame, A::SCRATCH);
    }

    /// The source is never below the destination, so copying upward is safe.
    fn move_down(&mut self, a: u8, arguments: u32, registers: u8) {
        self.registers(registers);
        for index in 0..=arguments as i32 {
            let from = (i32::from(a) + index) * VALUE_SIZE;
            let to = (index - 1) * VALUE_SIZE;
            for half in [0, PAYLOAD] {
                self.asm.load(Width::Q, A::SCRATCH, registers, from + half);
                self.asm.store(Width::Q, A::SCRATCH, registers, to + half);
            }
        }
    }

    /// Follows `luaD_pretailcall` (ldo.c).
    fn replace_frame(&mut self, t: Temps) {
        let frames = self.env.frames;
        self.asm.store(Width::Q, t.closure, t.frame, frames.field(lua_frame::CLOSURE));
        for word in 0..lua_frame::CACHE_WORDS {
            self.asm.store_imm(Width::Q, t.frame, frames.field(lua_frame::CACHE + word * std::mem::size_of::<usize>()), 0);
        }
        self.asm.load(Width::W, A::SCRATCH, t.frame, frames.field(lua_frame::STATUS));
        self.asm.alu_imm(Alu::And, A::SCRATCH, A::SCRATCH, lua_frame::TAIL_KEEPS as i32);
        self.asm.alu_imm(Alu::Or, A::SCRATCH, A::SCRATCH, lua_frame::TAIL as i32);
        self.asm.shl_imm(A::SCRATCH, A::SCRATCH, lua_frame::STATUS_SHIFT);
        self.asm.store(Width::Q, A::SCRATCH, t.frame, frames.field(lua_frame::PC));
        self.asm.store_imm(Width::Q, t.frame, frames.field(lua_frame::NVARARGS), 0);
    }

    pub(super) fn ret(&mut self, pc: u32, first: u8, count: u32, close: bool) {
        if close {
            self.close_in_place(pc);
        }
        if count == MULTIPLE_RESULTS {
            self.lower_multiple_return(pc, first);
            return;
        }
        self.lower_return(pc, first, count);
    }

    /// Follows `moveresults` (ldo.c) for results that run to the stack top.
    fn lower_multiple_return(&mut self, pc: u32, first: u8) {
        let results = self.temp();
        let count = self.temp();
        let index = self.temp();
        let source = self.temp();
        let destination = self.temp();
        let word = self.temp();
        let helper = self.asm.new_label();
        let next = self.asm.new_label();
        let fill = self.asm.new_label();
        let done = self.asm.new_label();
        self.asm.load_i32(results, A::CTX, offset_of!(JitCtx, direct_results) as i32);
        self.asm.cmp_imm(results, 0);
        self.asm.branch(Cond::Lt, helper);
        self.ctx_load(word, offset_of!(JitCtx, thread));
        self.asm.load(Width::Q, count, word, self.env.frames.top);
        self.ctx_load(word, offset_of!(JitCtx, base));
        self.asm.alu(Alu::Sub, count, count, word);
        self.asm.alu_imm(Alu::Sub, count, count, i32::from(first));
        self.registers(source);
        self.asm.lea(destination, source, -VALUE_SIZE);
        self.asm.alu_imm(Alu::Add, source, source, i32::from(first) * VALUE_SIZE);
        self.asm.mov_imm(index, 0);
        self.asm.bind(next);
        self.asm.cmp(index, results);
        self.asm.branch(Cond::Ge, done);
        self.asm.cmp(index, count);
        self.asm.branch(Cond::Ge, fill);
        for half in [0, PAYLOAD] {
            self.asm.load(Width::Q, word, source, half);
            self.asm.store(Width::Q, word, destination, half);
        }
        self.asm.alu_imm(Alu::Add, source, source, VALUE_SIZE);
        self.asm.alu_imm(Alu::Add, destination, destination, VALUE_SIZE);
        self.asm.alu_imm(Alu::Add, index, index, 1);
        self.asm.jump(next);
        self.asm.bind(fill);
        self.asm.store_imm(Width::B, destination, 0, 0);
        self.asm.alu_imm(Alu::Add, destination, destination, VALUE_SIZE);
        self.asm.alu_imm(Alu::Add, index, index, 1);
        self.asm.jump(next);
        self.asm.bind(done);
        self.asm.mov_imm(A::SCRATCH, crate::jit::EXIT_RETURNED);
        self.asm.jump(self.epilogue);
        self.asm.bind(helper);
        self.fast_args(pc, [u32::from(first), MULTIPLE_RESULTS, 0]);
        self.asm.call(self.addresses.fast[sites::FAST_RETURN]);
        self.asm.mov(A::SCRATCH, A::result());
        self.asm.jump(self.exit_with);
    }

    fn close_in_place(&mut self, pc: u32) {
        let after = self.asm.new_label();
        self.fast_args(pc, [0, 0, 0]);
        self.asm.call(self.addresses.fast[sites::FAST_CLOSE]);
        self.asm.mov(A::SCRATCH, A::result());
        self.asm.cmp_imm(A::SCRATCH, pc as i32);
        self.asm.branch(Cond::Eq, after);
        self.asm.jump(self.exit_with);
        self.asm.bind(after);
    }

    /// Follows `moveresults` in ldo.c.
    fn lower_return(&mut self, pc: u32, first: u8, count: u32) {
        let results = self.temp();
        let base = self.temp();
        let destination = self.temp();
        let word = self.temp();
        let helper = self.asm.new_label();
        let other = self.asm.new_label();
        let wide = self.asm.new_label();
        let place = self.asm.new_label();
        let done = self.asm.new_label();
        self.asm.load_i32(results, A::CTX, offset_of!(JitCtx, direct_results) as i32);
        self.asm.cmp_imm(results, count as i32);
        self.asm.branch(Cond::Ne, other);
        if count > 0 {
            self.copy_results(first, count, (base, destination, word));
        }
        self.asm.bind(done);
        self.asm.mov_imm(A::SCRATCH, crate::jit::EXIT_RETURNED);
        self.asm.jump(self.epilogue);
        self.asm.bind(other);
        self.asm.cmp_imm(results, 0);
        self.asm.branch(Cond::Eq, done);
        self.asm.cmp_imm(results, i32::from(u8::MAX));
        self.asm.branch(Cond::Above, wide);
        self.asm.bind(place);
        self.copy_results(first, count, (base, destination, word));
        self.fill_results(results, (destination, word), count, done);
        self.asm.bind(wide);
        self.asm.cmp_imm(results, -1);
        self.asm.branch(Cond::Eq, place);
        self.asm.bind(helper);
        self.fast_args(pc, [u32::from(first), count, 0]);
        self.asm.call(self.addresses.fast[sites::FAST_RETURN]);
        self.asm.mov(A::SCRATCH, A::result());
        self.asm.jump(self.exit_with);
    }

    fn copy_results(&mut self, first: u8, count: u32, (base, destination, word): (u8, u8, u8)) {
        self.registers(base);
        self.asm.lea(destination, base, -VALUE_SIZE);
        for index in 0..count as i32 {
            let from = (i32::from(first) + index) * VALUE_SIZE;
            for half in [0, PAYLOAD] {
                self.asm.load(Width::Q, word, base, from + half);
                self.asm.store(Width::Q, word, destination, index * VALUE_SIZE + half);
            }
        }
    }

    fn fill_results(&mut self, results: u8, (destination, index): (u8, u8), count: u32, done: Label) {
        let fill = self.asm.new_label();
        let next = self.asm.new_label();
        self.asm.cmp_imm(results, -1);
        self.asm.branch(Cond::Ne, fill);
        self.ctx_load(index, offset_of!(JitCtx, base));
        self.asm.alu_imm(Alu::Add, index, index, count as i32 - 1);
        self.ctx_load(A::SCRATCH, offset_of!(JitCtx, thread));
        self.asm.store(Width::Q, index, A::SCRATCH, self.env.frames.top);
        self.asm.jump(done);
        self.asm.bind(fill);
        self.asm.mov_imm(index, i64::from(count));
        self.asm.bind(next);
        self.asm.cmp(index, results);
        self.asm.branch(Cond::AboveEq, done);
        self.asm.shl_imm(A::SCRATCH, index, 4);
        self.asm.alu(Alu::Add, A::SCRATCH, A::SCRATCH, destination);
        self.asm.store_imm(Width::B, A::SCRATCH, 0, 0);
        self.asm.alu_imm(Alu::Add, index, index, 1);
        self.asm.jump(next);
    }
}
