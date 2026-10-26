mod api;
mod builtin;
mod close;
mod coroutine;
mod dump;
mod errors;
mod event;
mod exec;
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
use exec::Direct;
mod fast;
mod hooks;
pub(crate) mod ic;

#[derive(Clone, Copy)]
pub(crate) struct DeferredLoopCharge {
    pub(crate) proto: crate::arena::Ref<crate::proto::Proto>,
    pub(crate) charge: u32,
    pub(crate) thread: crate::arena::Ref<crate::object::Thread>,
    pub(crate) depth: usize,
    /// Each back-edge charges again until the budget is spent, since FORPREP's charge waited for more work.
    pub(crate) watch: bool,
}
mod inspect;
pub(crate) mod interrupt;
mod jit;
mod meta;
mod naming;
mod ops;
mod quick;
pub(crate) mod quicken;
mod steps;
mod window;
mod traceback;
mod verify;

#[cfg(test)]
mod native_tests;

use ic::{FEEDBACK_CALLS, InlineCache};
pub(crate) use window::{CodeRef, Exec, Raw, UpvaluesRef};
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
pub(crate) use window::exec_layout;

pub(crate) use naming::Operand;

pub(crate) use coroutine::Leave;
pub(crate) use errors::RunScope;
pub(crate) use hooks::{mask_from_text, mask_text};
pub(crate) use builtin::{Builtin, TYPE_NAMES};
pub(crate) use event::{Event, Field, MODE_ABSENCE_BIT};
pub(crate) use inspect::{Activation, ClosureFacts};
pub(crate) use ops::table_address;
pub(crate) use meta::{IndexStep, SetStep};

use crate::{
    LuaError, Values, Vm,
    arena::Ref,
    object::{
        Closure, Frame, LuaFrame, NativeBody, NativeFrame, Pending, Resume, Step, Thread, UpState, UpVal,
        YieldFrame, status as frame_status,
    },
    proto::Proto,
    stdlib,
    value::Value,
};

pub(crate) const MAX_STACK_SLOTS: usize = 1_000_000;

/// Slots past the base that the window addresses, whatever `max_stack` says.
pub(crate) const REGISTER_SPAN: usize = u8::MAX as usize + 4;

const ERROR_STACK_RESERVE: usize = 200;

const MIN_C_STACK: usize = 20;

pub(crate) const ARGUMENTS_INLINE: usize = 4;

#[derive(Clone, Copy, Debug)]
pub(crate) struct NativeCall {
    pub(crate) native: Ref<crate::object::Native>,
    pub(crate) slot: usize,
    pub(crate) count: usize,
    pub(crate) from_native: bool,
}

pub(crate) struct CallRequest {
    pub function_slot: usize,
    pub argument_count: usize,
    pub nresults: i32,
    pub ccmt: u8,
    pub operand: Operand,
}

pub(crate) const MAX_CALL_METAMETHODS: u8 = 15;

enum CallStart {
    Lua,
    Done,
    Callee(CallRequest),
}

fn yields_through(resume: &Resume) -> bool {
    matches!(resume, Resume::Protected(_) | Resume::Unwinding(_) | Resume::Pairs | Resume::All)
}

fn counts_c_call(resume: &Resume) -> bool {
    matches!(resume, Resume::Protected(_) | Resume::Unwinding(_) | Resume::Pairs | Resume::All)
}

enum Waiting {
    Stop,
    Continue,
    Lua,
    Callee(CallRequest),
}

impl Vm {
    /// Follows `luaD_growstack` in ldo.c; the error reserve lets a message handler run after overflow.
    pub(crate) fn check_stack(&mut self, needed: usize) -> Result<(), LuaError> {
        if needed <= MAX_STACK_SLOTS {
            return Ok(());
        }
        if self.handling_error == 0 {
            return Err(self.runtime_error(b"stack overflow"));
        }
        if needed <= MAX_STACK_SLOTS + ERROR_STACK_RESERVE {
            return Ok(());
        }
        Err(self.error_in_error_handling())
    }

    pub(crate) fn shrink_stack(&mut self) {
        let extent = self.heap.thread_extent(self.current_thread);
        self.heap.shrink_thread_stack(self.current_thread, extent);
    }

    pub(crate) fn thread(&self) -> &Thread {
        self.heap.threads.live(self.current_thread)
    }

    pub(crate) fn thread_mut(&mut self) -> &mut Thread {
        self.heap.threads.live_mut(self.current_thread)
    }

    fn stack_get(&self, index: usize) -> Value {
        self.thread().stack[index]
    }

    fn stack_set(&mut self, index: usize, value: Value) {
        self.thread_mut().stack[index] = value;
    }

    #[inline(always)]
    pub(crate) fn ensure_stack(&mut self, size: usize) {
        if self.thread().stack.len() < size {
            self.grow_stack(size);
        }
    }

    #[cold]
    #[inline(never)]
    pub(crate) fn grow_stack(&mut self, size: usize) {
        let before = self.thread().stack.len();
        let grown = (before + before / 2).min(MAX_STACK_SLOTS).max(size);
        self.thread_mut().stack.resize(grown, Value::Nil);
        self.heap.stack_resized(self.current_thread);
        self.heap.gc.grow(((grown - before) * std::mem::size_of::<Value>()) as i64);
    }

    fn top_frame(&self) -> Option<&LuaFrame> {
        match self.thread().frames.last() {
            Some(Frame::Lua(frame)) => Some(frame),
            _ => None,
        }
    }

    fn mark_top_hooked(&mut self, transfer: Option<crate::object::Transfer>) {
        if let Some(Frame::Lua(frame)) = self.thread_mut().frames.last_mut() {
            frame.set_hooked(transfer.is_some());
        }
        self.thread_mut().transfer = transfer.unwrap_or_default();
    }

    fn nearest_lua_frame(&self) -> Option<LuaFrame> {
        self.thread().frames.iter().rev().find_map(|frame| match frame {
            Frame::Lua(lua) => Some(*lua),
            Frame::Native(_) | Frame::Yielded(_) | Frame::Coroutine(_) => None,
        })
    }

    pub(crate) fn frame_count(&self) -> usize {
        self.thread().frames.len()
    }

    #[inline(always)]
    pub(crate) fn save_pc(&mut self, pc: usize) {
        if let Some(Frame::Lua(frame)) = self.thread_mut().frames.last_mut() {
            frame.set_pc(pc);
        }
    }

    fn proto_of(&self, closure: Ref<Closure>) -> &Proto {
        let proto = self.heap.closures.live(closure).proto;
        self.heap.protos.live(proto)
    }

    fn take_arguments(&mut self, function_slot: usize, count: usize) -> Values {
        let mut arguments = self.argument_pool.pop().unwrap_or_default();
        arguments.extend_from_slice(&self.thread().stack[function_slot + 1..function_slot + 1 + count]);
        arguments
    }

    fn give_back_arguments(&mut self, mut arguments: Values) {
        arguments.clear();
        self.argument_pool.push(arguments);
    }

    pub(crate) fn place_call(&mut self, slot: usize, function: Value, arguments: &[Value]) {
        self.ensure_stack(slot + 1 + arguments.len());
        self.stack_set(slot, function);
        for (offset, argument) in arguments.iter().enumerate() {
            self.stack_set(slot + 1 + offset, *argument);
        }
    }

    fn scratch_top(&self) -> usize {
        let frame = self.top_frame().expect("a metamethod starts in a Lua frame");
        frame.base + usize::from(self.proto_of(frame.closure).max_stack)
    }

    pub(crate) fn scratch_top_above(&self, top: usize) -> usize {
        self.scratch_top().max(top)
    }

    pub(crate) fn call_value(&mut self, function: Value, arguments: &[Value]) -> Result<Values, LuaError> {
        let function_slot = self.free_slot();
        let stop = self.frame_count();
        self.enter_c_call()?;
        self.non_yieldable += 1;
        let floor = std::mem::replace(&mut self.floor, stop);
        self.place_call(function_slot, function, arguments);
        self.thread_mut().top = function_slot + 1 + arguments.len();
        let request = CallRequest {
            function_slot,
            argument_count: arguments.len(),
            nresults: -1,
            ccmt: 0,
            operand: Operand::Unknown,
        };
        let pausable = std::mem::replace(&mut self.pausable, false);
        let outcome = self.run_here(stop, Some(request));
        self.pausable = pausable;
        self.floor = floor;
        self.non_yieldable -= 1;
        self.leave_c_call();
        let results = match outcome {
            Ok(()) => {
                let top = self.thread().top;
                Ok(Values::from_slice(&self.thread().stack[function_slot..top]))
            }
            Err(error) => {
                let error = self.handle_at_raise(error);
                self.unwind(stop, function_slot);
                Err(error)
            }
        };
        self.thread_mut().top = function_slot;
        results
    }

    pub(crate) fn unwind(&mut self, frame_count: usize, level: usize) {
        let removed = self.counted_frames_from(frame_count);
        self.c_calls -= removed;
        self.thread_mut().frames.truncate(frame_count);
        self.close_upvalues(level);
    }

    pub(crate) fn counted_frames_from(&self, index: usize) -> u32 {
        let frames = self.thread().frames.get(index..).unwrap_or(&[]);
        frames
            .iter()
            .filter(|frame| match frame {
                Frame::Lua(lua) => lua.counted(),
                Frame::Native(native) => counts_c_call(&native.resume),
                Frame::Yielded(_) | Frame::Coroutine(_) => false,
            })
            .count() as u32
    }

    pub(crate) fn call_loop(&mut self, first: Option<CallRequest>) -> Result<(), LuaError> {
        let mut request = first;
        loop {
            if let Some(call) = request.take() {
                match self.start_call(call)? {
                    CallStart::Lua => return Ok(()),
                    CallStart::Callee(next) => {
                        request = Some(next);
                        continue;
                    }
                    CallStart::Done => {}
                }
            }
            if self.frame_count() <= self.floor {
                return Ok(());
            }
            match self.resume_waiting()? {
                Waiting::Stop | Waiting::Lua => return Ok(()),
                Waiting::Continue => {}
                Waiting::Callee(next) => request = Some(next),
            }
        }
    }

    fn start_call(&mut self, call: CallRequest) -> Result<CallStart, LuaError> {
        match self.stack_get(call.function_slot) {
            Value::Closure(closure) => {
                self.push_lua_frame(closure, call.function_slot, call.argument_count, call.nresults, call.ccmt)?;
                self.hook_call(false)?;
                Ok(CallStart::Lua)
            }
            Value::Native(native) => self.start_native(native, call),
            other => self.start_callable(other, call),
        }
    }

    fn start_native(&mut self, native: Ref<crate::object::Native>, call: CallRequest) -> Result<CallStart, LuaError> {
        let top = call.function_slot + 1 + call.argument_count;
        self.check_stack(top + MIN_C_STACK)?;
        let running = NativeCall {
            native,
            slot: call.function_slot,
            count: call.argument_count,
            from_native: self.current_native.is_some(),
        };
        let previous = self.current_native.replace(running);
        let outcome = self.call_native(native, call);
        self.current_native = previous;
        outcome
    }

    /// Follows `precallC` and `luaD_poscall` in ldo.c; a hook the native sets sees its return.
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(crate) fn call_plain(&mut self, native: Ref<crate::object::Native>, function: crate::object::NativeFn, (function_slot, count, wanted): (usize, usize, i32)) -> Result<(), LuaError> {
        debug_assert!(!self.tracing && count <= ARGUMENTS_INLINE);
        let first = function_slot + 1;
        let owner = Value::Native(native);
        self.check_stack(first + count + MIN_C_STACK)?;

        let previous = self.enter_native_quick(native, function_slot, count);
        let arguments = Values::from_slice(&self.thread().stack[first..first + count]);
        let outcome = function(self, &arguments);
        let settled = match outcome {
            Ok(results) if !self.tracing => {
                self.place_results(function_slot, wanted, &results);
                Ok(())
            }
            Ok(results) => self.settle_results(function_slot, wanted, owner, &results),
            Err(error) => Err(error),
        };
        self.current_native = previous;
        settled.map_err(|error| self.raise_native(function_slot, wanted, owner, error))
    }

    #[inline(never)]
    fn call_stepped(&mut self, native: Ref<crate::object::Native>, function: crate::object::SteppedFn, (function_slot, count, wanted): (usize, usize, i32)) -> Result<CallStart, LuaError> {
        debug_assert!(!self.tracing && count <= ARGUMENTS_INLINE);
        let first = function_slot + 1;
        let owner = Value::Native(native);
        self.check_stack(first + count + MIN_C_STACK)?;

        let previous = self.enter_native_quick(native, function_slot, count);
        let arguments = Values::from_slice(&self.thread().stack[first..first + count]);
        let settled = match function(self, &arguments) {
            Ok(Step::Done(results)) => self.settle_results(function_slot, wanted, owner, &results).map(|()| CallStart::Done),
            Ok(step) => self.settle_native(function_slot, wanted, first + count, owner, step),
            Err(error) => Err(error),
        };
        self.current_native = previous;
        settled.map_err(|error| self.raise_native(function_slot, wanted, owner, error))
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    fn enter_native_quick(&mut self, native: Ref<crate::object::Native>, slot: usize, count: usize) -> Option<NativeCall> {
        let running = NativeCall {
            native,
            slot,
            count,
            from_native: self.current_native.is_some(),
        };
        self.current_native.replace(running)
    }

    /// Copies arguments into a native-stack or pooled buffer, so the call does not allocate.
    fn call_native(&mut self, native: Ref<crate::object::Native>, call: CallRequest) -> Result<CallStart, LuaError> {
        let body = self.native_body(native);
        let (first, count) = (call.function_slot + 1, call.argument_count);
        if count <= ARGUMENTS_INLINE {
            let arguments = Values::from_slice(&self.thread().stack[first..first + count]);
            return self.run_native_body(native, body, call, &arguments);
        }
        let arguments = self.take_arguments(call.function_slot, count);
        let outcome = self.run_native_body(native, body, call, &arguments);
        self.give_back_arguments(arguments);
        outcome
    }

    fn run_native_body(
        &mut self,
        native: Ref<crate::object::Native>,
        body: NativeBody,
        call: CallRequest,
        arguments: &[Value],
    ) -> Result<CallStart, LuaError> {
        let owner = Value::Native(native);
        let outcome = match body {
            NativeBody::Yield => {
                self.hook_native_call()?;
                return self.yield_values(call, arguments);
            }
            NativeBody::CoroutineResume => return self.start_resume(call, arguments, false),
            NativeBody::CoroutineWrap => return self.start_resume(call, arguments, true),
            NativeBody::Plain(function) => return self.run_plain_native(owner, function, &call, arguments),
            NativeBody::Stepped(function) => self.hook_native_call().and_then(|()| function(self, arguments)),
        };
        let settled = match outcome {
            Ok(step) => {
                let scratch = call.function_slot + 1 + call.argument_count;
                self.settle_native(call.function_slot, call.nresults, scratch, owner, step)
            }
            Err(error) => Err(error),
        };
        settled.map_err(|error| self.raise_native(call.function_slot, call.nresults, owner, error))
    }

    #[inline(always)]
    fn run_plain_native(&mut self, owner: Value, function: crate::object::NativeFn, call: &CallRequest, arguments: &[Value]) -> Result<CallStart, LuaError> {
        let settled = self.hook_native_call().and_then(|()| function(self, arguments)).and_then(|results| self.settle_results(call.function_slot, call.nresults, owner, &results));
        match settled {
            Ok(()) => Ok(CallStart::Done),
            Err(error) => Err(self.raise_native(call.function_slot, call.nresults, owner, error)),
        }
    }

    fn settle_results(&mut self, ret: usize, nresults: i32, owner: Value, results: &[Value]) -> Result<(), LuaError> {
        if self.hook_installed() {
            self.hook_settled_native(owner, results)?;
        }
        self.place_results(ret, nresults, results);
        Ok(())
    }

    pub(crate) fn raise_native(&mut self, ret: usize, nresults: i32, owner: Value, error: LuaError) -> LuaError {
        self.thread_mut().frames.push(Frame::Native(NativeFrame {
            ret,
            nresults,
            scratch: ret,
            owner,
            resume: Resume::Raised,
        }));
        error
    }

    fn current_native_value(&self) -> Value {
        Value::Native(self.current_native.expect("a native call is running").native)
    }

    fn settle_native(
        &mut self,
        ret: usize,
        nresults: i32,
        scratch: usize,
        owner: Value,
        step: Step,
    ) -> Result<CallStart, LuaError> {
        let mut step = step;
        loop {
            match step {
                Step::Done(results) => {
                    self.settle_results(ret, nresults, owner, &results)?;
                    return Ok(CallStart::Done);
                }
                Step::Call {
                    function,
                    arguments,
                    resume,
                } if yields_through(&resume) => {
                    let frame = NativeFrame {
                        ret,
                        nresults,
                        scratch,
                        owner,
                        resume,
                    };
                    return self.push_native_frame(frame, function, &arguments);
                }
                Step::Call {
                    function,
                    arguments,
                    resume,
                } => {
                    let mark = self.pin_resume(&resume, owner);
                    let results = self.call_from_native(function, &arguments);
                    self.unpin(mark);
                    step = stdlib::resume(self, resume, results?)?;
                }
            }
        }
    }

    #[inline(never)]
    fn hook_settled_native(&mut self, owner: Value, results: &[Value]) -> Result<(), LuaError> {
        let saved_top = self.thread().top;
        let first = self.free_slot();
        self.place_results(first, -1, results);
        self.hook_native_return(owner, first, results.len())?;
        self.thread_mut().top = saved_top;
        self.note_return_to_caller();
        Ok(())
    }

    #[inline(never)]
    fn yield_values(&mut self, call: CallRequest, values: &[Value]) -> Result<CallStart, LuaError> {
        if self.current_thread == self.main_thread {
            return Err(self.error_string("attempt to yield from outside a coroutine"));
        }
        if self.non_yieldable > 0 {
            return Err(self.error_string("attempt to yield across a C-call boundary"));
        }
        self.yielded.clear();
        self.yielded.extend_from_slice(values);
        let owner = self.current_native.map_or(Value::Nil, |running| Value::Native(running.native));
        self.thread_mut().frames.push(Frame::Yielded(YieldFrame {
            ret: call.function_slot,
            nresults: call.nresults,
            owner,
        }));
        Ok(CallStart::Done)
    }

    #[inline(never)]
    fn push_native_frame(&mut self, frame: NativeFrame, function: Value, arguments: &[Value]) -> Result<CallStart, LuaError> {
        let unwinding = matches!(frame.resume, Resume::Unwinding(_));
        let counted = counts_c_call(&frame.resume);
        let scratch = if unwinding { self.free_slot() } else { frame.scratch };
        self.thread_mut().frames.push(Frame::Native(frame));
        if counted {
            self.count_c_call()?;
        }
        self.place_call(scratch, function, arguments);
        Ok(CallStart::Callee(CallRequest {
            function_slot: scratch,
            argument_count: arguments.len(),
            ccmt: 0,
            nresults: -1,
            operand: Operand::Unknown,
        }))
    }

    fn start_callable(&mut self, object: Value, call: CallRequest) -> Result<CallStart, LuaError> {
        let handler = self.metamethod(object, Event::Call);
        if handler.is_nil() {
            let info = match call.operand {
                Operand::Unknown => self.event_info(),
                operand => self.variable_info(operand),
            };
            return Err(self.type_error(object, "call", info));
        }
        if call.ccmt == MAX_CALL_METAMETHODS {
            return Err(self.runtime_error(b"'__call' chain too long"));
        }
        let slot = call.function_slot;
        let count = call.argument_count;
        self.ensure_stack(slot + count + 2);
        for offset in (0..=count).rev() {
            let value = self.stack_get(slot + offset);
            self.stack_set(slot + offset + 1, value);
        }
        self.stack_set(slot, handler);
        Ok(CallStart::Callee(CallRequest {
            argument_count: count + 1,
            ccmt: call.ccmt + 1,
            ..call
        }))
    }

    #[inline(never)]
    fn resume_waiting(&mut self) -> Result<Waiting, LuaError> {
        match self.thread().frames.last() {
            None | Some(Frame::Yielded(_) | Frame::Coroutine(_)) => Ok(Waiting::Stop),
            Some(Frame::Native(_)) => self.resume_native(),
            Some(Frame::Lua(frame)) => {
                let pending = frame.pending();
                self.resume_lua(pending)
            }
        }
    }

    fn resume_native(&mut self) -> Result<Waiting, LuaError> {
        let Some(Frame::Native(frame)) = self.thread_mut().frames.pop() else {
            unreachable!("a native frame is on top");
        };
        if counts_c_call(&frame.resume) {
            self.c_calls -= 1;
        }
        let top = self.thread().top;
        let values = match frame.resume {
            Resume::Unwinding(_) => Values::new(),
            _ => Values::from_slice(&self.thread().stack[frame.scratch..top]),
        };
        let previous = match frame.owner {
            Value::Native(native) => Some(self.current_native.replace(NativeCall {
                native,
                slot: frame.ret,
                count: 0,
                from_native: true,
            })),
            _ => None,
        };
        let step = stdlib::resume(self, frame.resume, values);
        let settled = step.and_then(|step| self.settle_native(frame.ret, frame.nresults, frame.scratch, frame.owner, step));
        if let Some(previous) = previous {
            self.current_native = previous;
        }
        match settled? {
            CallStart::Done => Ok(Waiting::Continue),
            CallStart::Callee(request) => Ok(Waiting::Callee(request)),
            CallStart::Lua => Ok(Waiting::Lua),
        }
    }

    fn resume_lua(&mut self, pending: Pending) -> Result<Waiting, LuaError> {
        match pending {
            Pending::Idle => Ok(Waiting::Stop),
            Pending::Closing(top) => {
                self.clear_pending();
                self.thread_mut().top = top;
                if let Some(Frame::Lua(caller)) = self.thread_mut().frames.last_mut() {
                    caller.set_pc(caller.pc() - 1);
                }
                Ok(Waiting::Lua)
            }
            Pending::Return(first) => {
                self.clear_pending();
                let top = self.thread().top;
                self.finish_return(first, top - first, true)?;
                Ok(Waiting::Continue)
            }
            Pending::Operation | Pending::Concat(_) => {
                self.clear_pending();
                self.finish_operation(pending)?;
                Ok(Waiting::Lua)
            }
        }
    }

    fn clear_pending(&mut self) {
        if let Some(index) = self.frame_count().checked_sub(1) {
            self.set_pending(index, Pending::Idle);
        }
    }

    pub(crate) fn set_pending(&mut self, index: usize, pending: Pending) {
        let thread = self.current_thread;
        self.set_pending_in(thread, index, pending);
    }

    pub(crate) fn set_pending_in(&mut self, thread: Ref<Thread>, index: usize, pending: Pending) {
        if let Some(Frame::Lua(frame)) = self.heap.threads.live_mut(thread).frames.get_mut(index) {
            frame.set_pending(pending);
        }
    }

    pub(crate) fn call_settled(&self, thread: Ref<Thread>, depth: usize) -> bool {
        self.current_thread == thread && self.frame_count() == depth
    }

    pub(crate) fn call_meta(
        &mut self,
        handler: Value,
        arguments: &[Value],
        pending: Pending,
    ) -> Result<Option<Value>, LuaError> {
        let scratch = self.scratch_top();
        self.call_meta_at(scratch, handler, arguments, pending)
    }

    pub(crate) fn call_meta_at(
        &mut self,
        scratch: usize,
        handler: Value,
        arguments: &[Value],
        pending: Pending,
    ) -> Result<Option<Value>, LuaError> {
        self.place_call(scratch, handler, arguments);
        let caller = self.current_thread;
        let depth = self.frame_count();
        self.enter_c_call()?;
        let started = self.call_loop(Some(CallRequest {
            function_slot: scratch,
            argument_count: arguments.len(),
            nresults: 1,
            ccmt: 0,
            operand: Operand::Unknown,
        }));
        if self.current_thread != caller {
            self.mark_entry_counted(caller);
            self.set_pending_in(caller, depth - 1, pending);
            return started.map(|()| None);
        }
        if let Err(error) = started {
            self.leave_c_call();
            return Err(error);
        }
        if self.frame_count() == depth {
            self.leave_c_call();
            return Ok(Some(self.stack_get(scratch)));
        }
        if let Some(Frame::Lua(handler_frame)) = self.thread_mut().frames.get_mut(depth) {
            handler_frame.set_counted(true);
        } else {
            self.leave_c_call();
        }
        self.set_pending(depth - 1, pending);
        Ok(None)
    }

    pub(crate) fn push_lua_frame(
        &mut self,
        closure: Ref<Closure>,
        function_slot: usize,
        argument_count: usize,
        nresults: i32,
        ccmt: u8,
    ) -> Result<usize, LuaError> {
        self.push_lua_frame_from(None, closure, function_slot, argument_count, nresults, ccmt)
    }

    #[inline(always)]
    pub(crate) fn push_lua_frame_from(
        &mut self,
        caller_pc: Option<usize>,
        closure: Ref<Closure>,
        function_slot: usize,
        argument_count: usize,
        nresults: i32,
        ccmt: u8,
    ) -> Result<usize, LuaError> {
        let proto = self.heap.closures.live(closure).proto;
        let function = self.heap.protos.live(proto);
        let (num_params, is_vararg, max_stack) = (usize::from(function.num_params), function.is_vararg, usize::from(function.max_stack));
        if function.exec.feedback.is_empty() {
            self.count_call_for_feedback(proto);
        }
        let base = function_slot + 1;
        let needed = base + argument_count + max_stack + usize::from(is_vararg);
        let status = LuaFrame::call_status(nresults, ccmt) | if caller_pc.is_some() { frame_status::NESTED } else { 0 };
        if needed > MAX_STACK_SLOTS || self.thread().stack.len() < base + max_stack {
            if let Some(pc) = caller_pc {
                self.save_pc(pc);
            }
            self.check_stack(needed)?;
            return Ok(self.push_lua_frame_grown(closure, function_slot, argument_count, status));
        }
        let thread = self.thread_mut();
        if let Some(pc) = caller_pc
            && let Some(Frame::Lua(caller)) = thread.frames.last_mut()
        {
            caller.set_pc(pc);
        }
        if argument_count < num_params {
            thread.stack[base + argument_count..base + num_params].fill(Value::Nil);
        }
        window::push_frame(&mut thread.frames, Frame::Lua(LuaFrame::with_status(closure, base, status).with_arguments(argument_count)));
        Ok(base)
    }

    /// Fills missing parameters with nil, as `luaD_precall` does.
    #[inline(never)]
    fn push_lua_frame_grown(&mut self, closure: Ref<Closure>, function_slot: usize, argument_count: usize, status: u32) -> usize {
        let function = self.proto_of(closure);
        let (num_params, max_stack) = (usize::from(function.num_params), usize::from(function.max_stack));
        let base = function_slot + 1;

        self.ensure_stack(base + max_stack.max(num_params));
        let thread = self.thread_mut();
        if argument_count < num_params {
            thread.stack[base + argument_count..base + num_params].fill(Value::Nil);
        }
        thread.frames.push(Frame::Lua(LuaFrame::with_status(closure, base, status).with_arguments(argument_count)));
        base
    }
    #[inline(always)]
    pub(crate) fn count_call_for_feedback(&mut self, proto: Ref<Proto>) {
        let function = self.heap.protos.live_mut(proto);
        if !function.exec.feedback.is_empty() {
            return;
        }
        function.calls = function.calls.saturating_add(1);
        if function.calls == FEEDBACK_CALLS {
            function.exec.feedback = vec![InlineCache::Uninitialized; function.code.len()];
        }
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(crate) fn place_results(&mut self, destination: usize, wanted: i32, results: &[Value]) {
        let count = if wanted < 0 {
            results.len()
        } else {
            wanted as usize
        };
        self.ensure_stack(destination + count);
        let thread = self.thread_mut();
        let slots = &mut thread.stack[destination..destination + count];
        if let ([slot], [value, ..]) = (&mut *slots, results) {
            *slot = *value;
        } else {
            for (index, slot) in slots.iter_mut().enumerate() {
                *slot = results.get(index).copied().unwrap_or(Value::Nil);
            }
        }
        if wanted < 0 {
            thread.top = destination + count;
        }
    }

    fn pop_top_frame(&mut self) {
        let thread = self.thread_mut();
        let top = thread.frames.len() - 1;
        thread.frames.truncate(top);
    }

    pub(crate) fn finish_return(&mut self, first: usize, count: usize, needclose: bool) -> Result<(), LuaError> {
        let frame = *self.top_frame().expect("a Lua frame returns");
        let (counted, base, wanted, destination) = (frame.counted(), frame.base, frame.nresults(), frame.ret());
        if counted {
            self.leave_c_call();
        }
        if needclose {
            self.close_upvalues(base);
        }
        if self.tracing {
            self.hook_return(first, count)?;
        }
        let thread = self.thread_mut();
        let hooked = thread.hook.is_some();
        move_results(&mut thread.stack, first, count, destination, wanted);
        if wanted < 0 {
            thread.top = destination + count;
        }
        let top = thread.frames.len() - 1;
        thread.frames.truncate(top);
        if hooked {
            self.note_return_to_caller();
        }
        Ok(())
    }

    pub(crate) fn open_index(&self, upvalue: Ref<UpVal>) -> usize {
        match self.heap.upvalues.live(upvalue).state {
            UpState::Open { index, .. } => index,
            UpState::Closed(_) => unreachable!("the open list holds open upvalues"),
        }
    }

    /// Walks down from the top, as `luaF_findupval` in lfunc.c does, since new upvalues are usually the highest.
    pub(crate) fn find_or_open_upvalue(&mut self, index: usize) -> Ref<UpVal> {
        let open = &self.thread().open_upvalues;
        let top = open.last().map(|&upvalue| (upvalue, self.open_index(upvalue)));
        match top {
            Some((upvalue, slot)) if slot == index => return upvalue,
            Some((_, slot)) if slot > index => {}
            _ => {
                let thread = self.current_thread;
                let upvalue = self.heap.alloc_open_upvalue(thread, index);
                self.thread_mut().open_upvalues.push(upvalue);
                return upvalue;
            }
        }
        let open = &self.thread().open_upvalues;
        let position = open.partition_point(|&upvalue| self.open_index(upvalue) < index);
        if let Some(&upvalue) = open.get(position)
            && self.open_index(upvalue) == index
        {
            return upvalue;
        }
        let thread = self.current_thread;
        let upvalue = self.heap.alloc_open_upvalue(thread, index);
        self.thread_mut().open_upvalues.insert(position, upvalue);
        upvalue
    }

    pub(crate) fn close_upvalues(&mut self, level: usize) {
        let thread = self.current_thread;
        self.heap.close_upvalues(thread, level);
    }

    pub(crate) fn upvalue_get(&self, upvalue: Ref<UpVal>) -> Value {
        match self.heap.upvalues.live(upvalue).state {
            UpState::Open { thread, index } => self.heap.threads.live(thread).stack[index],
            UpState::Closed(value) => value,
        }
    }

    pub(crate) fn upvalue_set(&mut self, upvalue: Ref<UpVal>, value: Value) {
        match self.heap.upvalues.live(upvalue).state {
            UpState::Open { thread, index } => self.heap.threads.live_mut(thread).stack[index] = value,
            UpState::Closed(_) => self.heap.set_closed_upvalue(upvalue, value),
        }
    }
}

/// Follows `moveresults` in ldo.c. The destination is below the results, so copying forward is safe.
#[inline(always)]
pub(crate) fn move_results(stack: &mut [Value], first: usize, count: usize, destination: usize, wanted: i32) {
    if count == 1 && wanted == 1 {
        stack[destination] = stack[first];
        return;
    }
    let wanted = if wanted < 0 { count } else { wanted as usize };
    let moved = wanted.min(count);
    for index in 0..moved {
        stack[destination + index] = stack[first + index];
    }
    for slot in &mut stack[destination + moved..destination + wanted] {
        *slot = Value::Nil;
    }
}
