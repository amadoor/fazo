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
    ssa::ir::Val,
};

/// A callee known from feedback, or one whose facts the call reads from its closure's `Exec`, as Baseline's `CallAny` does.
#[derive(Clone, Copy)]
enum Callee<'t> {
    Known(&'t CallTarget),
    Any,
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
        let temps = Temps {
            exec: self.temp(),
            closure: self.temp(),
            entry: self.temp(),
            depth_pointer: self.temp(),
            depth: self.temp(),
            thread: self.temp(),
            callee_base: self.temp(),
            count: self.temp(),
            frame: self.temp(),
        };
        let saves = self.saves(&[]);
        let helper = self.asm.new_label();
        let join = self.asm.new_label();
        self.save(&saves);
        let unknown = targets.is_empty() && self.env.unknown_calls.get(pc).copied().unwrap_or(false);
        if b != 0 && (unknown || !targets.is_empty()) {
            let any = if unknown { self.asm.new_label() } else { helper };
            let matched = self.match_callee(a, &targets, temps, (helper, any));
            for (target, label) in targets.iter().zip(matched) {
                self.asm.bind(label);
                self.direct_call(pc, (a, b, c), Callee::Known(target), temps, (helper, join, out));
            }
            if unknown {
                self.asm.bind(any);
                self.direct_call(pc, (a, b, c), Callee::Any, temps, (helper, join, out));
            }
        }
        self.asm.bind(helper);
        self.fast_args(pc as u32, [u32::from(a), u32::from(b), u32::from(c)]);
        self.asm.call(self.addresses.fast[sites::FAST_CALL]);
        self.asm.mov(out, A::result());
        self.asm.bind(join);
        self.restore(&saves);
    }

    fn match_callee(&mut self, a: u8, targets: &[CallTarget], temps: Temps, (helper, any): (Label, Label)) -> Vec<Label> {
        let at = i32::from(a) * VALUE_SIZE;
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

    /// Mirrors `Vm::jit_call_compiled`; other exits go to `Vm::jit_after_call`. The depth and frame count are stashed in the scratch area across the call.
    fn direct_call(&mut self, pc: usize, (a, b, c): (u8, u8, u8), callee: Callee<'_>, t: Temps, (helper, join, out): (Label, Label, u8)) {
        let frames = self.env.frames;
        let arguments = u32::from(b) - 1;
        let wanted = i64::from(c) - 1;
        let callee_offset = (i32::from(a) + 1) * VALUE_SIZE;
        let stash = self.scratch_at;

        self.load_entry(callee, t, helper);
        self.ctx_load(t.depth_pointer, offset_of!(JitCtx, call_depth));
        self.asm.load(Width::W, t.depth, t.depth_pointer, 0);
        self.asm.cmp_imm64(t.depth, i64::from(self.env.call_depth_limit));
        self.asm.branch(Cond::AboveEq, helper);
        self.ctx_load(t.thread, offset_of!(JitCtx, thread));
        self.ctx_load(t.callee_base, offset_of!(JitCtx, base));
        self.asm.alu_imm(Alu::Add, t.callee_base, t.callee_base, i32::from(a) + 1);
        self.check_stack_room(callee, arguments, t, helper);
        self.clear_missing_parameters(callee, arguments, callee_offset, t);
        self.asm.load(Width::Q, t.count, t.thread, frames.frames_len);
        self.asm.load(Width::Q, A::SCRATCH, t.thread, frames.frames_capacity);
        self.asm.cmp(t.count, A::SCRATCH);
        self.asm.branch(Cond::AboveEq, helper);

        self.asm.alu_imm(Alu::Mul, t.frame, t.count, frames.frame_size as i32);
        self.asm.load(Width::Q, A::SCRATCH, t.thread, frames.frames_data);
        self.asm.alu(Alu::Add, t.frame, t.frame, A::SCRATCH);
        let caller_pc = frames.field(lua_frame::PC) - frames.frame_size as i32;
        self.asm.store_imm(Width::W, t.frame, caller_pc, pc as i64 + 1);
        self.push_lua_frame(t, wanted);
        self.asm.alu_imm(Alu::Add, A::SCRATCH, t.count, 1);
        self.asm.store(Width::Q, A::SCRATCH, t.thread, frames.frames_len);
        self.fill_callee_context(callee, t, (callee_offset, wanted));
        self.asm.alu_imm(Alu::Add, A::SCRATCH, t.depth, 1);
        self.asm.store(Width::W, A::SCRATCH, t.depth_pointer, 0);
        self.asm.store(Width::Q, t.depth, A::SP, stash);
        self.asm.store(Width::Q, t.count, A::SP, stash + 8);

        self.asm.mov(A::SCRATCH, t.entry);
        self.arg_address(0, self.ctx_at);
        self.place_arg(1, Src::I(0));
        self.asm.call_reg(A::SCRATCH);
        self.asm.mov(A::SCRATCH, A::result());
        self.ctx_load(t.depth_pointer, offset_of!(JitCtx, call_depth));
        self.asm.load(Width::Q, t.depth, A::SP, stash);
        self.asm.store(Width::W, t.depth, t.depth_pointer, 0);

        let other = self.asm.new_label();
        self.asm.cmp_imm(A::SCRATCH, crate::jit::EXIT_RETURNED as i32);
        self.asm.branch(Cond::Ne, other);
        self.asm.load(Width::Q, t.thread, A::SP, self.ctx_at + offset_of!(JitCtx, thread) as i32);
        self.asm.load(Width::Q, t.count, t.thread, frames.frames_len);
        self.asm.alu_imm(Alu::Sub, t.count, t.count, 1);
        self.asm.store(Width::Q, t.count, t.thread, frames.frames_len);
        self.refresh_from_callee(t.thread, callee_offset);
        self.asm.mov_imm(out, pc as i64 + 1);
        self.asm.jump(join);

        self.asm.bind(other);
        self.place_arg(2, Src::R(A::SCRATCH));
        match A::arg(3) {
            super::asm::ArgLoc::Reg(reg) => self.asm.load(Width::Q, reg, A::SP, stash + 8),
            super::asm::ArgLoc::Stack(_) => unreachable!("four integer arguments go in registers"),
        }
        self.place_arg(4, Src::I(0));
        self.place_arg(1, Src::I(pc as i64));
        self.place_arg(0, Src::R(A::CTX));
        self.asm.call(self.addresses.fast[sites::FAST_AFTER_CALL]);
        self.asm.mov(out, A::result());
        self.asm.jump(join);
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

    /// `luaD_precall` checks the callee's `maxstacksize` (ldo.c); the interpreter keeps `REGISTER_SPAN` slots past every base.
    fn check_stack_room(&mut self, callee: Callee<'_>, arguments: u32, t: Temps, helper: Label) {
        let frames = self.env.frames;
        self.asm.load(Width::Q, t.count, t.thread, frames.stack_len);
        if let Callee::Known(target) = callee {
            let register_count = i64::from(arguments) + i64::from(target.max_stack);
            self.asm.alu_imm(Alu::Add, A::SCRATCH, t.callee_base, register_count.max(crate::vm::REGISTER_SPAN as i64) as i32);
            self.asm.cmp(A::SCRATCH, t.count);
            self.asm.branch(Cond::Above, helper);
            self.asm.alu_imm(Alu::Add, A::SCRATCH, t.callee_base, register_count as i32);
        } else {
            self.asm.alu_imm(Alu::Add, A::SCRATCH, t.callee_base, crate::vm::REGISTER_SPAN as i32);
            self.asm.cmp(A::SCRATCH, t.count);
            self.asm.branch(Cond::Above, helper);
            self.asm.load(Width::B, A::SCRATCH, t.exec, exec_layout::MAX_STACK as i32);
            self.asm.alu(Alu::Add, A::SCRATCH, A::SCRATCH, t.callee_base);
            self.asm.alu_imm(Alu::Add, A::SCRATCH, A::SCRATCH, arguments as i32);
            self.asm.cmp(A::SCRATCH, t.count);
            self.asm.branch(Cond::Above, helper);
        }
        self.asm.cmp_imm64(A::SCRATCH, crate::vm::MAX_STACK_SLOTS as i64);
        self.asm.branch(Cond::Above, helper);
    }

    fn clear_missing_parameters(&mut self, callee: Callee<'_>, arguments: u32, callee_offset: i32, t: Temps) {
        if let Callee::Known(target) = callee {
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

    fn push_lua_frame(&mut self, t: Temps, wanted: i64) {
        let frames = self.env.frames;
        let status = (crate::object::LuaFrame::call_status(wanted as i32, 0) as i64) << lua_frame::STATUS_SHIFT;
        self.asm.store_imm(Width::B, t.frame, 0, 0);
        self.asm.store(Width::Q, t.closure, t.frame, frames.field(lua_frame::CLOSURE));
        self.asm.store(Width::Q, t.callee_base, t.frame, frames.field(lua_frame::BASE));
        for word in 0..lua_frame::CACHE_WORDS {
            self.asm.store_imm(Width::Q, t.frame, frames.field(lua_frame::CACHE + word * std::mem::size_of::<usize>()), 0);
        }
        self.asm.store_imm(Width::Q, t.frame, frames.field(lua_frame::PC), status);
        self.asm.store_imm(Width::Q, t.frame, frames.field(lua_frame::NVARARGS), 0);
    }

    fn fill_callee_context(&mut self, callee: Callee<'_>, t: Temps, (callee_offset, wanted): (i32, i64)) {
        let context = self.ctx_at;
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
            self.ctx_load(A::SCRATCH, offset);
            self.asm.store(Width::Q, A::SCRATCH, A::SP, context + offset as i32);
        }
        self.registers(A::SCRATCH);
        self.asm.alu_imm(Alu::Add, A::SCRATCH, A::SCRATCH, callee_offset);
        self.asm.store(Width::Q, A::SCRATCH, A::SP, context + offset_of!(JitCtx, registers) as i32);
        if matches!(callee, Callee::Known(target) if !target.has_upvalues) {
            self.asm.store_imm(Width::Q, A::SP, context + offset_of!(JitCtx, upvalues) as i32, 0);
        } else {
            self.closure_address(t.frame, t.closure);
            self.asm.load(Width::Q, t.frame, t.frame, self.env.frames.closure_upvalues);
            self.asm.store(Width::Q, t.frame, A::SP, context + offset_of!(JitCtx, upvalues) as i32);
        }
        match callee {
            Callee::Known(target) => {
                self.asm.store_imm(Width::Q, A::SP, context + offset_of!(JitCtx, constants) as i32, target.constants as i64);
                self.asm.store_imm(Width::Q, A::SP, context + offset_of!(JitCtx, proto) as i32, target.proto as i64);
            }
            Callee::Any => {
                self.asm.load(Width::Q, A::SCRATCH, t.exec, exec_layout::CONSTANTS as i32);
                self.asm.store(Width::Q, A::SCRATCH, A::SP, context + offset_of!(JitCtx, constants) as i32);
                self.asm.load(Width::Q, A::SCRATCH, t.exec, exec_layout::PROTO as i32);
                self.asm.store(Width::Q, A::SCRATCH, A::SP, context + offset_of!(JitCtx, proto) as i32);
            }
        }
        self.asm.store(Width::Q, t.closure, A::SP, context + offset_of!(JitCtx, closure) as i32);
        self.asm.store(Width::Q, t.callee_base, A::SP, context + offset_of!(JitCtx, base) as i32);
        self.asm.store_imm(Width::W, A::SP, context + offset_of!(JitCtx, direct_results) as i32, wanted);
        self.asm.store_imm(Width::Q, A::SP, context + offset_of!(JitCtx, ret_offset) as i32, -i64::from(VALUE_SIZE));
    }

    /// The callee's helpers refresh its context whenever they move its stack, tables or thread.
    fn refresh_from_callee(&mut self, thread: u8, callee_offset: i32) {
        let context = self.ctx_at;
        self.asm.load(Width::Q, A::SCRATCH, A::SP, context + offset_of!(JitCtx, registers) as i32);
        self.asm.alu_imm(Alu::Sub, A::SCRATCH, A::SCRATCH, callee_offset);
        self.asm.store(Width::Q, A::SCRATCH, A::CTX, offset_of!(JitCtx, registers) as i32);
        for offset in [offset_of!(JitCtx, tables), offset_of!(JitCtx, table_marks), offset_of!(JitCtx, closures)] {
            self.asm.load(Width::Q, A::SCRATCH, A::SP, context + offset as i32);
            self.asm.store(Width::Q, A::SCRATCH, A::CTX, offset as i32);
        }
        self.asm.store(Width::Q, thread, A::CTX, offset_of!(JitCtx, thread) as i32);
    }

    pub(super) fn tail_call(&mut self, pc: u32) {
        self.fast_args(pc, [0, 0, 0]);
        self.asm.call(self.addresses.fast[sites::FAST_TAIL_CALL]);
        self.asm.mov(A::SCRATCH, A::result());
        self.asm.jump(self.exit_with);
    }

    pub(super) fn ret(&mut self, pc: u32, first: u8, count: u32, close: bool) {
        if close {
            self.close_in_place(pc);
        }
        self.lower_return(pc, first, count);
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
        self.copy_results(first, count, (base, destination, word));
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
        self.ctx_load(destination, offset_of!(JitCtx, ret_offset));
        self.asm.alu(Alu::Add, destination, destination, base);
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
