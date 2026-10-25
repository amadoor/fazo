use super::Event;
use crate::values;
use crate::{
    LuaError, Vm,
    object::{Pending, Resume, Step, Unwinding},
    value::Value,
};

impl Vm {
    pub(crate) fn mark_to_close(&mut self, register: u8) -> Result<(), LuaError> {
        let frame = self.top_frame().expect("a to-be-closed variable is declared in a Lua frame");
        let slot = frame.base + usize::from(register);
        let value = self.stack_get(slot);
        if value.is_falsy() {
            return Ok(());
        }
        if self.metamethod(value, Event::Close).is_nil() {
            let name = self.local_variable_name(register);
            let message = format!("variable '{name}' got a non-closable value");
            return Err(self.runtime_error_text(&message));
        }
        self.thread_mut().to_close.push(slot);
        Ok(())
    }

    pub(crate) fn close_after_error(&mut self, level: usize, error: Value) -> Value {
        self.close_upvalues(level);
        let mark = self.pin([error]);
        let mut error = error;
        while let Some(slot) = self.pending_close(level) {
            if let Err(failure) = self.call_close(slot, Some(error)) {
                if let Some(outcome) = self.thread_mut().self_close.take() {
                    error = outcome;
                    break;
                }
                error = failure.value();
                self.pin([error]);
            }
        }
        self.unpin(mark);
        error
    }

    pub(crate) fn close_pending_at_exit(&mut self) {
        let mark = self.pins.len();
        let mut failure = None;

        while let Some(slot) = self.pending_close(0) {
            failure = self.close_at_exit_step(slot, failure);
        }
        self.unpin(mark);
    }

    fn close_at_exit_step(&mut self, slot: usize, failure: Option<Value>) -> Option<Value> {
        let Err(error) = self.call_close_at_exit(slot, failure) else {
            return failure;
        };
        let value = error.value();
        self.pin([value]);
        Some(value)
    }

    pub(crate) fn has_to_close(&self, level: usize) -> bool {
        self.pending_close(level).is_some()
    }

    pub(crate) fn close_next(&mut self, level: usize) -> Result<(), LuaError> {
        let slot = self.pending_close(level).expect("the caller checked for a pending variable");
        let value = self.stack_get(slot);
        self.thread_mut().to_close.pop();
        let handler = self.metamethod(value, Event::Close);
        if handler.is_nil() {
            return Err(self.type_error(Value::Nil, "call", b" (metamethod 'close')".to_vec()));
        }
        let top = self.thread().top;
        let scratch = self.scratch_top_above(top);
        self.call_meta_at(scratch, handler, &[value], Pending::Closing(top))?;
        Ok(())
    }

    pub(crate) fn resume_unwinding(&mut self, state: Unwinding) -> Step {
        let Some(slot) = self.pending_close(state.level) else {
            return Step::Done(values![Value::Boolean(false), state.error]);
        };
        let value = self.stack_get(slot);
        self.thread_mut().to_close.pop();
        Step::Call {
            function: self.metamethod(value, Event::Close),
            arguments: values![value, state.error],
            resume: Resume::Unwinding(state),
        }
    }

    pub(crate) fn raise_top_above_pending(&mut self, level: usize) {
        let highest = self.thread().to_close.iter().copied().filter(|&slot| slot >= level).max();
        if let Some(slot) = highest {
            let top = self.thread().top.max(slot + 1);
            self.thread_mut().top = top;
        }
    }

    fn pending_close(&self, level: usize) -> Option<usize> {
        self.thread().to_close.last().copied().filter(|&slot| slot >= level)
    }

    fn call_close(&mut self, slot: usize, error: Option<Value>) -> Result<(), LuaError> {
        let (handler, arguments) = self.prepare_close(slot, error)?;
        self.call_value(handler, &arguments).map(|_| ())
    }

    fn call_close_at_exit(&mut self, slot: usize, error: Option<Value>) -> Result<(), LuaError> {
        let (handler, arguments) = self.prepare_close(slot, error)?;
        self.call_protected_as(handler, &arguments, None, false).map(|_| ())
    }

    fn prepare_close(&mut self, slot: usize, error: Option<Value>) -> Result<(Value, Vec<Value>), LuaError> {
        let value = self.stack_get(slot);
        self.thread_mut().to_close.pop();
        let handler = self.metamethod(value, Event::Close);
        if handler.is_nil() {
            return Err(self.type_error(Value::Nil, "call", b" (metamethod 'close')".to_vec()));
        }
        let top = self.thread().top.max(slot + 1);
        self.thread_mut().top = top;
        let arguments = error.map_or_else(|| vec![value], |error| vec![value, error]);
        Ok((handler, arguments))
    }

    fn local_variable_name(&self, register: u8) -> String {
        let frame = self.top_frame().expect("a Lua frame names its variables");
        let proto = self.proto_of(frame.closure);
        let pc = frame.pc().saturating_sub(1);
        super::naming::local_name(&self.heap, proto, usize::from(register) + 1, pc)
            .map_or_else(|| "?".to_string(), |name| String::from_utf8_lossy(name).into_owned())
    }
}
