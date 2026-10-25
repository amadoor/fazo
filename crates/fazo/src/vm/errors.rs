use super::{CallRequest, Leave};
use crate::{
    LuaError, Values, Vm,
    arena::Ref,
    object::{Frame, NativeFrame, Protection, Resume, Thread, Unwinding},
    value::Value,
};

pub(super) const MAX_C_CALLS: u32 = 200;

const MAX_C_CALLS_HANDLING: u32 = MAX_C_CALLS / 10 * 11;

#[derive(Clone, Copy)]
pub(crate) struct RunScope {
    pub(crate) thread: Ref<Thread>,
    pub(crate) frames: usize,
    pub(crate) return_on_switch: bool,
}

impl Vm {
    pub(crate) fn free_slot(&self) -> usize {
        let thread = self.thread();
        let mut slot = thread.top;
        for frame in thread.frames.iter().rev() {
            match frame {
                Frame::Lua(lua) => {
                    let extent = lua.base + usize::from(self.proto_of(lua.closure).max_stack);
                    return slot.max(extent);
                }
                Frame::Native(native) => slot = slot.max(native.scratch),
                Frame::Yielded(_) | Frame::Coroutine(_) => {}
            }
        }
        slot
    }

    pub(crate) fn count_c_call(&mut self) -> Result<(), LuaError> {
        self.c_calls += 1;
        self.check_c_stack()
    }

    /// Follows `luaE_checkcstack` in lstate.c.
    pub(crate) fn check_c_stack(&mut self) -> Result<(), LuaError> {
        if self.c_calls == MAX_C_CALLS {
            return Err(self.error_here("C stack overflow"));
        }
        if self.c_calls >= MAX_C_CALLS_HANDLING {
            return Err(self.error_in_error_handling());
        }
        Ok(())
    }

    pub(crate) fn enter_c_call(&mut self) -> Result<(), LuaError> {
        let outcome = self.count_c_call();
        if outcome.is_err() {
            self.c_calls -= 1;
        }
        outcome
    }

    pub(crate) fn leave_c_call(&mut self) {
        self.c_calls -= 1;
    }

    pub(crate) fn call_from_native(&mut self, function: Value, arguments: &[Value]) -> Result<Values, LuaError> {
        if self.c_calls + 1 == MAX_C_CALLS {
            return Err(self.error_string("C stack overflow"));
        }
        self.call_value(function, arguments)
    }

    pub(crate) fn error_here(&mut self, message: &str) -> LuaError {
        match self.top_frame() {
            Some(_) => self.runtime_error_text(message),
            None => self.error_string(message),
        }
    }

    pub(crate) fn error_in_error_handling(&mut self) -> LuaError {
        self.error_string("error in error handling")
    }

    pub(crate) fn run(&mut self, scope: RunScope, first: Option<CallRequest>) -> Result<(), LuaError> {
        self.run_depth += 1;
        let outcome = self.run_frames(scope, first);
        self.run_depth -= 1;
        outcome
    }

    pub(crate) fn run_here(&mut self, frames: usize, first: Option<CallRequest>) -> Result<(), LuaError> {
        let scope = RunScope {
            thread: self.current_thread,
            frames,
            return_on_switch: false,
        };
        self.run(scope, first)
    }

    fn run_frames(&mut self, scope: RunScope, first: Option<CallRequest>) -> Result<(), LuaError> {
        let mut outcome = self.call_loop(first).and_then(|()| self.execute(scope));
        while let Err(error) = outcome {
            if error.is_pause() {
                return Err(error);
            }
            if error.is_interrupted() {
                self.abandon_coroutines(scope.thread);
                return Err(error);
            }
            self.recover(scope, error)?;
            outcome = self.call_loop(None).and_then(|()| self.execute(scope));
        }
        Ok(())
    }

    fn leave_failed_coroutine(&mut self, scope: RunScope, error: LuaError) -> Result<(), LuaError> {
        if self.current_thread == scope.thread {
            return Err(error);
        }
        match self.leave_coroutine(Leave::Failed(error.value())) {
            Ok(()) => Ok(()),
            Err(raised) => self.recover(scope, raised),
        }
    }

    pub(crate) fn memory_error(&mut self) -> LuaError {
        let message = self.heap.string(b"not enough memory");
        LuaError::handled(Value::String(message))
    }

    pub(crate) fn handle_at_raise(&mut self, error: LuaError) -> LuaError {
        if error.is_handled() || self.handling_error > 0 {
            return error;
        }
        let Some(index) = self.innermost_protected(0) else {
            return error;
        };
        let Frame::Native(frame) = self.thread().frames[index].clone() else {
            unreachable!("a protected frame is a native frame");
        };
        let Resume::Protected(protection) = frame.resume else {
            return error;
        };
        let value = self.message_for(protection, error.value());
        LuaError::handled(value)
    }

    /// The nesting limit is a runtime error, so the innermost message handler sees it as in PUC; other syntax errors are plain strings.
    pub(crate) fn syntax_failure_value(&mut self, message: Vec<u8>) -> Value {
        let value = self.string_value(&message);
        if message != crate::syntax::C_STACK_OVERFLOW {
            return value;
        }
        let Some(index) = self.innermost_protected(0) else {
            return value;
        };
        let Frame::Native(frame) = self.thread().frames[index].clone() else {
            return value;
        };
        match frame.resume {
            Resume::Protected(protection) => self.message_for(protection, value),
            _ => value,
        }
    }

    fn innermost_protected(&self, lower: usize) -> Option<usize> {
        let frames = &self.thread().frames;
        (lower..frames.len()).rev().find(|&index| {
            matches!(
                frames[index],
                Frame::Native(NativeFrame {
                    resume: Resume::Protected(_) | Resume::Unwinding(_),
                    ..
                })
            )
        })
    }

    #[inline(never)]
    pub(crate) fn recover(&mut self, scope: RunScope, error: LuaError) -> Result<(), LuaError> {
        if error.is_unwinding() {
            return self.unwind_coroutine(scope, error);
        }
        let lower = if self.current_thread == scope.thread { scope.frames } else { 0 };
        let Some(index) = self.innermost_protected(lower) else {
            return self.leave_failed_coroutine(scope, error);
        };
        let Frame::Native(frame) = self.thread().frames[index].clone() else {
            unreachable!("a protected frame is a native frame");
        };
        let value = match frame.resume {
            Resume::Protected(protection) if !error.is_handled() => self.message_for(protection, error.value()),
            _ => error.value(),
        };
        let value = self.named_error_object(value);
        let removed = self.counted_frames_from(index + 1);
        self.c_calls -= removed;
        self.thread_mut().frames.truncate(index + 1);
        self.close_upvalues(frame.scratch);
        self.raise_top_above_pending(frame.scratch);
        self.shrink_stack();
        self.thread_mut().frames[index] = Frame::Native(NativeFrame {
            resume: Resume::Unwinding(Unwinding {
                level: frame.scratch,
                error: value,
            }),
            ..frame
        });
        Ok(())
    }

    /// Replaces a nil error object with this text after the message handler sees it, as PUC Lua 5.5 does.
    fn named_error_object(&mut self, value: Value) -> Value {
        if value.is_nil() {
            return self.string_value(b"<no error object>");
        }
        value
    }

    fn message_for(&mut self, protection: Protection, error: Value) -> Value {
        let Some(handler) = protection.handler else {
            return error;
        };
        self.handling_error += 1;
        let outcome = self.call_message_handler(handler, error);
        self.handling_error -= 1;
        match outcome {
            Ok(value) => value,
            Err(_) => self.string_value(b"error in error handling"),
        }
    }

    fn call_message_handler(&mut self, handler: Value, error: Value) -> Result<Value, LuaError> {
        let mut error = error;
        let mut nested = 0;
        let mark = self.pin([error]);
        let outcome = loop {
            self.pin([error]);
            match self.call_value(handler, &[error]) {
                Ok(results) => break Ok(results.first().copied().unwrap_or(Value::Nil)),
                Err(failure) => {
                    self.c_calls += 1;
                    nested += 1;
                    if self.c_calls >= MAX_C_CALLS_HANDLING {
                        break Err(failure);
                    }
                    error = failure.value();
                }
            }
        };
        for _ in 0..nested {
            self.leave_c_call();
        }
        self.unpin(mark);
        outcome
    }

    pub(crate) fn call_protected(
        &mut self,
        function: Value,
        arguments: &[Value],
        handler: Option<Value>,
    ) -> Result<Values, LuaError> {
        self.call_protected_as(function, arguments, handler, false)
    }

    pub(crate) fn call_protected_as(
        &mut self,
        function: Value,
        arguments: &[Value],
        handler: Option<Value>,
        finalizer: bool,
    ) -> Result<Values, LuaError> {
        let slot = self.free_slot();
        let stop = self.frame_count();
        self.thread_mut().frames.push(Frame::Native(NativeFrame {
            ret: slot,
            nresults: -1,
            scratch: slot,
            owner: Value::Nil,
            resume: Resume::Protected(Protection { handler, finalizer }),
        }));
        self.non_yieldable += 1;
        let floor = std::mem::replace(&mut self.floor, stop);
        self.place_call(slot, function, arguments);
        self.thread_mut().top = slot + 1 + arguments.len();
        let request = CallRequest {
            function_slot: slot,
            argument_count: arguments.len(),
            nresults: -1,
            ccmt: 0,
            operand: super::naming::Operand::Unknown,
        };
        let scope = RunScope {
            thread: self.current_thread,
            frames: stop,
            return_on_switch: false,
        };
        let outcome = match self.count_c_call() {
            Ok(()) => self.run(scope, Some(request)),
            Err(error) => self.recover(scope, error).and_then(|()| self.run(scope, None)),
        };
        if outcome.as_ref().is_err_and(LuaError::is_interrupted) {
            self.unwind(stop, slot);
        }
        self.floor = floor;
        self.non_yieldable -= 1;
        let top = self.thread().top;
        let results = self.thread().stack[slot..top].to_vec();
        self.thread_mut().top = slot;
        outcome?;
        match results.first() {
            Some(Value::Boolean(true)) => Ok(Values::from_slice(&results[1..])),
            _ => Err(LuaError::from_value(results.get(1).copied().unwrap_or(Value::Nil))),
        }
    }
}

#[cfg(test)]
mod syntax_failure_tests {
    use crate::{Options, Vm};

    #[test]
    fn a_parser_nesting_limit_runs_the_message_handler_and_a_syntax_error_does_not() {
        let mut vm = Vm::new(Options::default());
        vm.run_source(
            b"=t",
            br#"
            local deep = "return " .. string.rep("(", 1000) .. "1" .. string.rep(")", 1000)
            local ok, nothing, message = xpcall(load, function(m) return "HANDLED:" .. m end, deep)
            assert(ok and nothing == nil and message == "HANDLED:C stack overflow")
            local _, _, plain = xpcall(load, function(m) return "HANDLED:" .. m end, "x = = 1")
            assert(plain ~= nil and not string.find(plain, "HANDLED", 1, true))
            "#,
        )
        .expect("the parser nesting limit reaches the message handler");
    }
}
