use super::Frame;
use crate::{
    LuaError, Vm,
    arena::Ref,
    instruction::Instruction,
    object::{Hook, LuaFrame, MASK_CALL, MASK_COUNT, MASK_LINE, MASK_RET, NativeFrame, Resume, Thread, Transfer},
    value::Value,
};

impl Vm {
    pub(crate) fn sync_tracing(&mut self) {
        self.tracing = self.thread().hook.is_some();
    }

    pub(crate) fn set_hook(&mut self, thread: Ref<Thread>, hook: Option<Hook>) {
        let target = self.heap.threads.live_mut(thread);
        target.hook = hook;
        target.hook_counter = hook.map_or(0, |hook| hook.base_count);
        if thread == self.current_thread {
            self.sync_tracing();
        }
    }

    pub(crate) fn hook_of(&self, thread: Ref<Thread>) -> Option<Hook> {
        self.heap.threads.live(thread).hook
    }

    fn hook_mask(&self) -> u8 {
        self.thread().hook.map_or(0, |hook| hook.mask)
    }

    fn run_hook(&mut self, event: &'static str, line: Option<i64>, transfer: Transfer) -> Result<(), LuaError> {
        let Some(hook) = self.thread().hook else {
            return Ok(());
        };
        if self.thread().hook_running {
            return Ok(());
        }
        let top = self.thread().top;
        self.thread_mut().hook_running = true;
        let event = self.string_value(event.as_bytes());
        let line = line.map_or(Value::Nil, Value::Integer);
        self.mark_top_hooked(Some(transfer));
        let outcome = self.call_value(hook.function, &[event, line]);
        self.mark_top_hooked(None);
        self.thread_mut().hook_running = false;
        self.thread_mut().top = top;
        outcome.map(|_| ())
    }

    /// A vararg function's hook runs after its `VARARGPREP`, as `luaV_execute` does in lvm.c.
    pub(crate) fn hook_call(&mut self, tail: bool) -> Result<(), LuaError> {
        if !self.tracing {
            return Ok(());
        }
        if self.top_frame().is_some_and(|frame| self.proto_of(frame.closure).is_vararg) {
            return Ok(());
        }
        self.run_call_hook(tail)
    }

    pub(crate) fn hook_call_after_prologue(&mut self) -> Result<(), LuaError> {
        let tail = self.top_frame().is_some_and(LuaFrame::tail);
        self.run_call_hook(tail)?;
        self.thread_mut().old_pc = 1;
        Ok(())
    }

    fn run_call_hook(&mut self, tail: bool) -> Result<(), LuaError> {
        if self.hook_mask() & MASK_CALL == 0 {
            return Ok(());
        }
        self.thread_mut().old_pc = 0;
        let transfer = match self.top_frame() {
            Some(frame) => {
                let params = usize::from(self.proto_of(frame.closure).num_params);
                Transfer {
                    start: 1,
                    count: params,
                    base: frame.base,
                }
            }
            None => Transfer::default(),
        };
        self.run_hook(if tail { "tail call" } else { "call" }, None, transfer)
    }

    #[inline(always)]
    pub(crate) fn hook_native_call(&mut self) -> Result<(), LuaError> {
        if !self.tracing {
            return Ok(());
        }
        self.hook_traced_native_call()
    }

    #[inline(never)]
    fn hook_traced_native_call(&mut self) -> Result<(), LuaError> {
        if self.hook_mask() & MASK_CALL == 0 {
            return Ok(());
        }
        let Some(call) = self.current_native else {
            return self.run_hook("call", None, Transfer::default());
        };
        let transfer = Transfer {
            start: 1,
            count: call.count,
            base: call.slot + 1,
        };
        self.in_native_marker(Value::Native(call.native), transfer, |vm| vm.run_hook("call", None, transfer))
    }

    pub(crate) fn hook_native_return(&mut self, owner: Value, first: usize, count: usize) -> Result<(), LuaError> {
        if !self.tracing || self.hook_mask() & MASK_RET == 0 {
            return Ok(());
        }
        let base = self.current_native.map_or(first, |call| call.slot + 1);
        let transfer = Transfer {
            start: first + 1 - base,
            count,
            base,
        };
        self.in_native_marker(owner, transfer, |vm| vm.return_hook(first, count, transfer))
    }

    pub(crate) fn hook_return(&mut self, first: usize, count: usize) -> Result<(), LuaError> {
        if !self.tracing || self.hook_mask() & MASK_RET == 0 {
            return Ok(());
        }
        let transfer = match self.top_frame() {
            Some(frame) => Transfer {
                start: first + 1 - frame.base,
                count,
                base: frame.base,
            },
            None => Transfer {
                start: 1,
                count,
                base: first,
            },
        };
        self.return_hook(first, count, transfer)
    }

    fn return_hook(&mut self, first: usize, count: usize, transfer: Transfer) -> Result<(), LuaError> {
        let top = self.thread().top.max(first + count);
        self.thread_mut().top = top;
        self.run_hook("return", None, transfer)
    }

    fn in_native_marker<T>(
        &mut self,
        owner: Value,
        transfer: Transfer,
        hook: impl FnOnce(&mut Self) -> Result<T, LuaError>,
    ) -> Result<T, LuaError> {
        let slot = self.free_slot();
        self.thread_mut().frames.push(Frame::Native(NativeFrame {
            ret: slot,
            nresults: 0,
            scratch: slot,
            owner,
            resume: Resume::Hook(transfer),
        }));
        let outcome = hook(self);
        self.thread_mut().frames.pop();
        outcome
    }

    pub(crate) fn hook_installed(&self) -> bool {
        self.thread().hook.is_some()
    }

    pub(crate) fn note_return_to_caller(&mut self) {
        if let Some(Frame::Lua(caller)) = self.thread().frames.last() {
            let pc = caller.pc().saturating_sub(1);
            self.thread_mut().old_pc = pc;
        }
    }

    pub(crate) fn trace_instruction(&mut self, npci: usize) -> Result<(), LuaError> {
        let Some(hook) = self.thread().hook else {
            return Ok(());
        };
        self.save_pc(npci + 1);
        let frame = self.top_frame().expect("an instruction is traced in a Lua frame");
        let is_prologue = matches!(self.proto_of(frame.closure).code.get(npci), Some(Instruction::VarargPrep { .. }));
        if is_prologue {
            self.thread_mut().old_pc = 1;
            return Ok(());
        }
        let counter = self.thread().hook_counter - 1;
        let counthook = counter == 0 && hook.mask & MASK_COUNT != 0;
        self.thread_mut().hook_counter = if counthook { hook.base_count } else { counter };
        if !counthook && hook.mask & MASK_LINE == 0 {
            return Ok(());
        }
        if counthook {
            self.run_hook("count", None, Transfer::default())?;
        }
        if hook.mask & MASK_LINE == 0 {
            return Ok(());
        }
        let frame = self.top_frame().expect("an instruction is traced in a Lua frame");
        let proto = self.proto_of(frame.closure);
        let old = self.thread().old_pc;
        let old = if old < proto.lines.len() { old } else { 0 };
        let changed = npci <= old || proto.lines.get(npci) != proto.lines.get(old);
        let line = proto.lines.get(npci).copied();
        self.thread_mut().old_pc = npci;
        if changed {
            self.run_hook("line", line.map(i64::from), Transfer::default())?;
        }
        Ok(())
    }
}

pub(crate) fn mask_from_text(text: &[u8]) -> u8 {
    text.iter().fold(0, |mask, byte| match byte {
        b'c' => mask | MASK_CALL,
        b'r' => mask | MASK_RET,
        b'l' => mask | MASK_LINE,
        _ => mask,
    })
}

pub(crate) fn mask_text(mask: u8) -> Vec<u8> {
    let mut text = Vec::new();
    for (bit, letter) in [(MASK_CALL, b'c'), (MASK_RET, b'r'), (MASK_LINE, b'l')] {
        if mask & bit != 0 {
            text.push(letter);
        }
    }
    text
}
