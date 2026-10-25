use super::naming;
use crate::{
    LuaError, Vm,
    arena::Ref,
    object::{Closure, Frame, Native, NativeFrame, Resume, Thread, Transfer, UpVal},
    value::Value,
};

pub(crate) struct ClosureFacts {
    pub source: Vec<u8>,
    pub chunkname: Vec<u8>,
    pub line_defined: i64,
    pub last_line_defined: i64,
    pub num_params: usize,
    pub is_vararg: bool,
    pub upvalue_count: usize,
}

#[derive(Clone, Copy)]
pub(crate) enum Activation {
    Native(Value),
    Lua(LuaActivation),
}

#[derive(Clone, Copy)]
pub(crate) struct LuaActivation {
    pub closure: Ref<Closure>,
    pub pc: usize,
    pub base: usize,
    pub tail: bool,
    pub nvarargs: usize,
    pub ccmt: u8,
}

impl Vm {
    pub(crate) fn activation(&self, thread: Ref<Thread>, level: usize) -> Option<Activation> {
        if level == 0 {
            if thread != self.current_thread {
                return match self.heap.threads.get(thread).frames.last() {
                    Some(Frame::Yielded(yielded)) => Some(Activation::Native(yielded.owner)),
                    _ => None,
                };
            }
            return self
                .current_native
                .map(|call| Activation::Native(Value::Native(call.native)));
        }
        Some(match self.frame_at(thread, level)? {
            Frame::Lua(lua) => Activation::Lua(LuaActivation {
                closure: lua.closure,
                pc: lua.pc(),
                base: lua.base,
                tail: lua.tail(),
                nvarargs: lua.nvarargs(),
                ccmt: lua.ccmt(),
            }),
            Frame::Native(native) => Activation::Native(native.owner),
            Frame::Yielded(yielded) => Activation::Native(yielded.owner),
            Frame::Coroutine(_) => return None,
        })
    }

    pub(crate) fn activation_line(&self, activation: LuaActivation) -> i64 {
        let proto = self.proto_of(activation.closure);
        proto
            .lines
            .get(activation.pc.saturating_sub(1))
            .map_or(-1, |line| i64::from(*line))
    }

    pub(crate) fn activation_name(&self, thread: Ref<Thread>, level: usize) -> Option<(&'static str, Vec<u8>)> {
        match self.activation(thread, level)? {
            Activation::Lua(callee) if callee.tail => None,
            _ => self.call_name_of_caller(thread, level),
        }
    }

    pub(crate) fn argument_error(&mut self, index: usize, fallback: &str, message: &str) -> LuaError {
        let thread = self.current_thread;
        let called_by_native = self.current_native.is_some_and(|call| call.from_native);
        let call_site = if called_by_native { None } else { self.activation_name(thread, 0) };
        let (method, name) = match call_site {
            Some((kind, name)) => (kind == "method", String::from_utf8_lossy(&name).into_owned()),
            None => {
                let global = match self.current_native {
                    Some(call) => self.global_function_name(Value::Native(call.native)),
                    None => None,
                };
                let caller_is_native = matches!(self.activation(thread, 1), Some(Activation::Native(_)));
                let unnamed = if called_by_native || caller_is_native { "?" } else { fallback };
                (false, global.map_or_else(|| unnamed.to_string(), |name| String::from_utf8_lossy(&name).into_owned()))
            }
        };
        let text = if method && index == 1 {
            format!("calling '{name}' on bad self ({message})")
        } else {
            let index = if method { index - 1 } else { index };
            format!("bad argument #{index} to '{name}' ({message})")
        };
        self.runtime_error_text(&text)
    }

    fn frame_at(&self, thread: Ref<Thread>, level: usize) -> Option<&Frame> {
        self.heap
            .threads
            .get(thread)
            .frames
            .iter()
            .rev()
            .filter(|frame| !matches!(frame, Frame::Yielded(_) | Frame::Coroutine(_)))
            .nth(level - 1)
    }

    pub(crate) fn activation_local(
        &self,
        thread: Ref<Thread>,
        level: usize,
        number: i64,
    ) -> Option<(Vec<u8>, usize)> {
        match self.activation(thread, level)? {
            Activation::Lua(frame) => self.lua_local(thread, level, frame, number),
            Activation::Native(_) if level == 0 => self.native_temporary(number),
            Activation::Native(_) => self.hook_marker_local(thread, level, number),
        }
    }

    fn lua_local(&self, thread: Ref<Thread>, level: usize, frame: LuaActivation, number: i64) -> Option<(Vec<u8>, usize)> {
        if number < 0 {
            let index = number.unsigned_abs() as usize;
            if index > frame.nvarargs {
                return None;
            }
            return Some((b"(vararg)".to_vec(), frame.base - 1 - frame.nvarargs + index - 1));
        }
        let index = usize::try_from(number).ok().filter(|index| *index > 0)?;
        let slot = frame.base + index - 1;
        let proto = self.proto_of(frame.closure);
        if let Some(name) = naming::local_name(&self.heap, proto, index, frame.pc.saturating_sub(1)) {
            return Some((name.to_vec(), slot));
        }
        (slot < self.callee_slot(thread, level)?).then(|| (b"(temporary)".to_vec(), slot))
    }

    fn callee_slot(&self, thread: Ref<Thread>, level: usize) -> Option<usize> {
        if level == 1 {
            return match self.current_native.filter(|_| thread == self.current_thread) {
                Some(call) => Some(call.slot),
                None => Some(self.heap.threads.get(thread).top),
            };
        }
        match self.frame_at(thread, level - 1)? {
            Frame::Lua(frame) => Some(frame.ret()),
            Frame::Native(frame) => Some(frame.ret),
            Frame::Yielded(yielded) => Some(yielded.ret),
            Frame::Coroutine(_) => None,
        }
    }

    fn hook_marker_local(&self, thread: Ref<Thread>, level: usize, number: i64) -> Option<(Vec<u8>, usize)> {
        let Some(Frame::Native(NativeFrame {
            resume: Resume::Hook(transfer),
            ..
        })) = self.frame_at(thread, level)
        else {
            return None;
        };
        let index = usize::try_from(number).ok().filter(|index| *index > 0)?;
        let slot = transfer.base + index - 1;
        (slot < self.callee_slot(thread, level)?).then(|| (b"(C temporary)".to_vec(), slot))
    }

    fn native_temporary(&self, number: i64) -> Option<(Vec<u8>, usize)> {
        let call = self.current_native?;
        let index = usize::try_from(number).ok().filter(|index| (1..=call.count).contains(index))?;
        Some((b"(C temporary)".to_vec(), call.slot + index))
    }

    pub(crate) fn parameter_name(&self, closure: Ref<Closure>, number: usize) -> Option<Vec<u8>> {
        let proto = self.proto_of(closure);
        if number == 0 || number > usize::from(proto.num_params) {
            return None;
        }
        naming::local_name(&self.heap, proto, number, 0).map(<[u8]>::to_vec)
    }

    pub(crate) fn closure_active_lines(&self, closure: Ref<Closure>) -> Vec<i64> {
        let proto = self.proto_of(closure);
        let skip = usize::from(proto.is_vararg);
        let mut lines: Vec<i64> = proto.lines.iter().skip(skip).map(|line| i64::from(*line)).collect();
        lines.sort_unstable();
        lines.dedup();
        lines
    }

    pub(crate) fn closure_facts(&self, closure: Ref<Closure>) -> ClosureFacts {
        let proto = self.proto_of(closure);
        ClosureFacts {
            source: proto.source.to_vec(),
            chunkname: proto.chunkname.to_vec(),
            line_defined: i64::from(proto.line_defined),
            last_line_defined: i64::from(proto.last_line_defined),
            num_params: usize::from(proto.num_params),
            is_vararg: proto.is_vararg,
            upvalue_count: self.heap.closures.get(closure).upvalues.len(),
        }
    }

    pub(crate) fn closure_upvalue(&self, closure: Ref<Closure>, index: usize) -> Option<(Vec<u8>, Ref<UpVal>)> {
        let upvalue = self.heap.closures.get(closure).upvalues.get(index.checked_sub(1)?)?.reference;
        let proto = self.proto_of(closure);
        let name = naming::name_bytes(&self.heap, proto.upvalues.get(index - 1)?.name);
        let name = if name.is_empty() { b"(no name)".to_vec() } else { name.to_vec() };
        Some((name, upvalue))
    }

    pub(crate) fn join_upvalue(&mut self, closure: Ref<Closure>, index: usize, upvalue: Ref<UpVal>) {
        self.heap.set_closure_upvalue(closure, index - 1, upvalue);
    }

    pub(crate) fn stack_value(&self, thread: Ref<Thread>, slot: usize) -> Value {
        self.heap.threads.get(thread).stack[slot]
    }

    pub(crate) fn set_stack_value(&mut self, thread: Ref<Thread>, slot: usize, value: Value) {
        self.heap.threads.get_mut(thread).stack[slot] = value;
    }

    pub(crate) fn native_upvalue(&self, native: Ref<Native>, index: usize) -> Option<Value> {
        self.heap.natives.get(native).upvalues.get(index.checked_sub(1)?).copied()
    }

    pub(crate) fn upvalue_identity(&self, upvalue: Ref<UpVal>) -> Value {
        Value::LightUserdata(upvalue.index() + 1)
    }

    pub(crate) fn upvalue_value(&self, upvalue: Ref<UpVal>) -> Value {
        self.upvalue_get(upvalue)
    }

    pub(crate) fn set_upvalue_value(&mut self, upvalue: Ref<UpVal>, value: Value) {
        self.upvalue_set(upvalue, value);
    }

    pub(crate) fn transfer_of(&self, thread: Ref<Thread>, level: usize) -> Option<Transfer> {
        match self.frame_at(thread, level)? {
            Frame::Lua(frame) => frame.hooked().then_some(self.heap.threads.live(thread).transfer),
            Frame::Native(NativeFrame {
                resume: Resume::Hook(transfer),
                ..
            }) => Some(*transfer),
            _ => None,
        }
    }

    fn hooked_at(&self, thread: Ref<Thread>, level: usize) -> bool {
        match self.frame_at(thread, level) {
            Some(Frame::Lua(frame)) => frame.hooked(),
            Some(Frame::Native(frame)) => matches!(frame.resume, Resume::Hook(_)),
            _ => false,
        }
    }

    fn finalizer_at(&self, thread: Ref<Thread>, level: usize) -> bool {
        matches!(
            self.frame_at(thread, level),
            Some(Frame::Native(NativeFrame { resume: Resume::Protected(protection), .. })) if protection.finalizer
        )
    }

    fn call_name_of_caller(&self, thread: Ref<Thread>, level: usize) -> Option<(&'static str, Vec<u8>)> {
        if self.hooked_at(thread, level + 1) {
            return Some(("hook", b"?".to_vec()));
        }
        if self.finalizer_at(thread, level + 1) {
            return Some(("metamethod", b"__gc".to_vec()));
        }
        let Some(Activation::Lua(caller)) = self.activation(thread, level + 1) else {
            return None;
        };
        let proto = self.proto_of(caller.closure);
        naming::calling_name(&self.heap, proto, caller.pc.checked_sub(1)?)
    }
}
