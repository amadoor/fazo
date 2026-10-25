use super::{CallRequest, CallStart, Operand, errors::{MAX_C_CALLS, RunScope}};
use crate::{
    LuaError, Values, Vm, values,
    arena::{Arena, Ref},
    object::{CoStatus, CoroutineFrame, Frame, Step, Thread, YieldFrame},
    stdlib,
    value::Value,
};

#[derive(Clone, Copy)]
pub(crate) enum Leave {
    Yielded,
    Finished,
    Failed(Value),
    Closed,
}

enum Target {
    Refused(&'static [u8]),
    Run(Ref<Thread>),
}

pub(crate) struct Switch {
    pins: usize,
    floor: usize,
    non_yieldable: u32,
    c_calls: u32,
    caller: Ref<Thread>,
}

impl Vm {
    pub(crate) fn create_thread(&mut self, function: Value) -> Ref<Thread> {
        let hook = self.thread().hook;
        let counter = hook.map_or(0, |hook| hook.base_count);
        self.heap.alloc(Thread {
            stack: vec![function],
            top: 1,
            hook,
            hook_counter: counter,
            ..Thread::default()
        })
    }

    pub(crate) fn current_thread(&self) -> Ref<Thread> {
        self.current_thread
    }

    pub(crate) fn is_main_thread(&self, thread: Ref<Thread>) -> bool {
        thread == self.main_thread
    }

    pub(crate) fn thread_status(&self, thread: Ref<Thread>) -> CoStatus {
        self.heap.threads.live(thread).status
    }

    pub(crate) fn is_yieldable(&self, thread: Ref<Thread>) -> bool {
        if thread == self.main_thread {
            return false;
        }
        thread != self.current_thread || self.non_yieldable == 0
    }

    #[inline(never)]
    pub(super) fn start_resume(&mut self, call: CallRequest, arguments: &[Value], wrapped: bool) -> Result<CallStart, LuaError> {
        let owner = self.current_native_value();
        let target = match self.resume_target(arguments, wrapped) {
            Ok(target) => target,
            Err(error) => return Err(self.raise_native(call.function_slot, call.nresults, owner, error)),
        };
        match target {
            Target::Refused(message) => self.refuse_resume(&call, owner, wrapped, message),
            Target::Run(co) => {
                let passed = arguments.get(usize::from(!wrapped)..).unwrap_or_default();
                self.run_coroutine(co, &call, owner, passed, wrapped)
            }
        }
    }

    /// Follows LuaJIT's `coroutine_resume_wrap` in `vm_x64.dasc`. False leaves the resume to the native.
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(crate) fn resume_fast(&mut self, native: Ref<crate::object::Native>, wrapped: bool, function_slot: usize, count: usize, wanted: i32, pc: usize) -> bool {
        let skipped = usize::from(!wrapped);
        let caller = self.current_thread;
        let target = if wrapped {
            self.heap.natives.live(native).upvalues.first().copied()
        } else {
            (count >= 1).then(|| self.stack_get(function_slot + 1))
        };
        let Some(Value::Thread(co)) = target else {
            return false;
        };
        if self.c_calls >= MAX_C_CALLS || count < skipped {
            return false;
        }
        let (floor, non_yieldable, c_calls) = (self.floor, self.non_yieldable, self.c_calls);
        let threads = &mut self.heap.threads;
        let thread = threads.live_mut(co);
        let waiting = thread.status == CoStatus::Suspended && thread.started && thread.hook.is_none();
        let stack_len = thread.stack.len();
        let [Frame::Coroutine(parked), .., Frame::Yielded(YieldFrame { ret, nresults, .. })] = thread.frames.as_mut_slice() else {
            return false;
        };
        let (ret, nresults) = (*ret, *nresults);
        let arguments = count - skipped;
        let placed = if nresults < 0 { arguments } else { nresults as usize };
        if !waiting || stack_len < ret + placed {
            return false;
        }

        *parked = CoroutineFrame {
            caller,
            wrapped,
            ret: function_slot,
            nresults: wanted,
            owner: Value::Native(native),
            floor,
            non_yieldable,
            c_calls,
            counted: false,
        };
        thread.frames.pop();
        thread.status = CoStatus::Running;
        if nresults < 0 {
            thread.top = ret + arguments;
        }
        let inner = thread.c_calls;
        let resumer = threads.live_mut(caller);
        resumer.status = CoStatus::Normal;
        if let Some(Frame::Lua(frame)) = resumer.frames.last_mut() {
            frame.set_pc(pc);
        }
        move_values(threads, (caller, function_slot + 1 + skipped, arguments), (co, ret, placed));
        self.c_calls += 1 + inner;
        self.floor = 0;
        self.non_yieldable = 0;
        self.current_thread = co;
        self.tracing = false;
        self.jit_count_switch();
        true
    }

    /// Follows LuaJIT's `coroutine_yield` in `vm_x64.dasc`. False leaves the yield to the native.
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(crate) fn yield_fast(&mut self, native: Ref<crate::object::Native>, function_slot: usize, count: usize, wanted: i32, pc: usize) -> bool {
        let co = self.current_thread;
        if co == self.main_thread || self.non_yieldable > 0 {
            return false;
        }
        let threads = &self.heap.threads;
        let Some(Frame::Coroutine(entry)) = threads.live(co).frames.first() else {
            return false;
        };
        let CoroutineFrame { caller, wrapped, ret, nresults, floor, non_yieldable, c_calls, counted, .. } = *entry;
        let flagged = usize::from(!wrapped && nresults != 0);
        let placed = if nresults < 0 { count + flagged } else { nresults as usize };
        let resumer = threads.live(caller);
        if counted || resumer.hook.is_some() || resumer.stack.len() < ret + placed {
            return false;
        }

        let inner = self.c_calls - (c_calls + 1);
        let threads = &mut self.heap.threads;
        let thread = threads.live_mut(co);
        thread.status = CoStatus::Suspended;
        thread.c_calls = inner;
        if let [Frame::Coroutine(parked), .., Frame::Lua(frame)] = thread.frames.as_mut_slice() {
            parked.caller = co;
            parked.owner = Value::Nil;
            frame.set_pc(pc);
        }
        super::window::push_frame(&mut thread.frames, Frame::Yielded(YieldFrame {
            ret: function_slot,
            nresults: wanted,
            owner: Value::Native(native),
        }));
        let resumer = threads.live_mut(caller);
        resumer.status = CoStatus::Running;
        if flagged == 1 {
            resumer.stack[ret] = Value::Boolean(true);
        }
        if nresults < 0 {
            resumer.top = ret + flagged + count;
        }
        move_values(threads, (co, function_slot + 1, count), (caller, ret + flagged, placed - flagged));
        self.current_thread = caller;
        self.floor = floor;
        self.non_yieldable = non_yieldable;
        self.c_calls = c_calls;
        self.tracing = false;
        true
    }

    pub(crate) fn switch_landed_idle(&self) -> bool {
        matches!(self.thread().frames.last(), Some(Frame::Lua(frame)) if frame.is_idle())
    }

    fn resume_target(&mut self, arguments: &[Value], wrapped: bool) -> Result<Target, LuaError> {
        self.hook_native_call()?;
        let co = if wrapped {
            self.wrapped_thread()
        } else {
            stdlib::coroutine::check_thread(self, arguments, "resume")?
        };
        if let Some(message) = self.resume_refusal(co) {
            return Ok(Target::Refused(message));
        }
        if self.c_calls >= MAX_C_CALLS {
            return Ok(Target::Refused(b"C stack overflow"));
        }
        Ok(Target::Run(co))
    }

    fn wrapped_thread(&self) -> Ref<Thread> {
        let Some(Value::Thread(thread)) = self.native_upvalues().first().copied() else {
            unreachable!("a wrapped coroutine keeps its thread as its upvalue");
        };
        thread
    }

    fn resume_refusal(&self, co: Ref<Thread>) -> Option<&'static [u8]> {
        match self.heap.threads.live(co).status {
            CoStatus::Suspended => None,
            CoStatus::Running | CoStatus::Normal => Some(b"cannot resume non-suspended coroutine"),
            CoStatus::Dead => Some(b"cannot resume dead coroutine"),
        }
    }

    fn refuse_resume(&mut self, call: &CallRequest, owner: Value, wrapped: bool, message: &[u8]) -> Result<CallStart, LuaError> {
        let text = self.string_value(message);
        if !wrapped {
            let results = values![Value::Boolean(false), text];
            return self.settle_native(call.function_slot, call.nresults, call.function_slot, owner, Step::Done(results));
        }
        let error = self.positioned_error(text);
        Err(self.raise_native(call.function_slot, call.nresults, owner, LuaError::from_value(error)))
    }

    fn run_coroutine(
        &mut self,
        co: Ref<Thread>,
        call: &CallRequest,
        owner: Value,
        arguments: &[Value],
        wrapped: bool,
    ) -> Result<CallStart, LuaError> {
        self.enter_coroutine(co, call, owner, wrapped);
        match self.continue_coroutine(arguments) {
            Ok(start) => Ok(start),
            Err(error) => self.leave_coroutine(Leave::Failed(error.value())).map(|()| CallStart::Done),
        }
    }

    fn enter_coroutine(&mut self, co: Ref<Thread>, call: &CallRequest, owner: Value, wrapped: bool) {
        let caller = self.current_thread;
        let entry = CoroutineFrame {
            caller,
            wrapped,
            ret: call.function_slot,
            nresults: call.nresults,
            owner,
            floor: self.floor,
            non_yieldable: self.non_yieldable,
            c_calls: self.c_calls,
            counted: false,
        };
        self.heap.threads.live_mut(caller).status = CoStatus::Normal;
        let inner = self.heap.threads.live(co).c_calls;
        let coroutine = self.heap.threads.live_mut(co);
        coroutine.status = CoStatus::Running;
        match coroutine.frames.first_mut() {
            Some(Frame::Coroutine(parked)) => *parked = entry,
            _ => coroutine.frames.insert(0, Frame::Coroutine(entry)),
        }
        self.c_calls += 1 + inner;
        self.floor = 0;
        self.non_yieldable = 0;
        self.current_thread = co;
        self.jit_count_switch();
        self.sync_tracing();
    }

    fn continue_coroutine(&mut self, arguments: &[Value]) -> Result<CallStart, LuaError> {
        if !self.thread().started {
            self.thread_mut().started = true;
            self.check_c_stack()?;
            self.ensure_stack(1 + arguments.len());
            for (offset, argument) in arguments.iter().enumerate() {
                self.thread_mut().stack[1 + offset] = *argument;
            }
            self.thread_mut().top = 1 + arguments.len();
            return Ok(CallStart::Callee(CallRequest {
                function_slot: 0,
                argument_count: arguments.len(),
                nresults: -1,
                ccmt: 0,
                operand: Operand::Unknown,
            }));
        }
        let Some(Frame::Yielded(frame)) = self.thread_mut().frames.pop() else {
            unreachable!("a suspended coroutine waits on a yielded frame");
        };
        self.place_results(frame.ret, frame.nresults, arguments);
        if self.tracing {
            self.hook_return(frame.ret, arguments.len())?;
            self.note_return_to_caller();
        }
        Ok(CallStart::Done)
    }

    pub(crate) fn leave_coroutine(&mut self, leave: Leave) -> Result<(), LuaError> {
        let co = self.current_thread;
        let entry = match leave {
            Leave::Yielded => self.park_entry(co),
            Leave::Finished | Leave::Failed(_) | Leave::Closed => self.take_entry(),
        };
        let (results, failure) = self.settle_coroutine(co, leave, &entry);
        self.exit_coroutine(&entry);
        let Some(error) = failure else {
            let results = if entry.wrapped { results } else { with_flag(true, results) };
            return self.settle_entry(&entry, results);
        };
        if !entry.wrapped {
            return self.settle_entry(&entry, values![Value::Boolean(false), error]);
        }
        let thrown = self.close_thread(co).err().unwrap_or(error);
        let positioned = self.positioned_error(thrown);
        Err(self.raise_native(entry.ret, entry.nresults, entry.owner, LuaError::from_value(positioned)))
    }

    pub(crate) fn coroutine_exit(&self) -> Option<Leave> {
        match self.thread().frames.last() {
            Some(Frame::Yielded(_)) => Some(Leave::Yielded),
            Some(Frame::Coroutine(_)) => Some(Leave::Finished),
            _ => None,
        }
    }

    pub(crate) fn abandon_coroutines(&mut self, thread: Ref<Thread>) {
        while self.current_thread != thread {
            let co = self.current_thread;
            let entry = self.take_entry();
            self.exit_coroutine(&entry);
            let _ = self.close_thread(co);
        }
    }

    /// Kept at the bottom of the frames so the next resume overwrites it instead of shifting them.
    fn park_entry(&mut self, co: Ref<Thread>) -> CoroutineFrame {
        let Some(Frame::Coroutine(entry)) = self.thread_mut().frames.first_mut() else {
            unreachable!("a running coroutine keeps its entry frame at the bottom");
        };
        let parked = CoroutineFrame {
            caller: co,
            owner: Value::Nil,
            ..entry.clone()
        };
        std::mem::replace(entry, parked)
    }

    fn take_entry(&mut self) -> CoroutineFrame {
        let Frame::Coroutine(entry) = self.thread_mut().frames.remove(0) else {
            unreachable!("a running coroutine keeps its entry frame at the bottom");
        };
        entry
    }

    fn settle_coroutine(&mut self, co: Ref<Thread>, leave: Leave, entry: &CoroutineFrame) -> (Values, Option<Value>) {
        match leave {
            Leave::Yielded => {
                let inner = self.c_calls - (entry.c_calls + 1);
                let thread = self.heap.threads.live_mut(co);
                thread.status = CoStatus::Suspended;
                thread.c_calls = inner;
                (std::mem::take(&mut self.yielded), None)
            }
            Leave::Finished => {
                let top = self.thread().top;
                let results = Values::from_slice(&self.thread().stack[..top]);
                self.end_coroutine(co, None);
                (results, None)
            }
            Leave::Failed(error) => {
                self.end_coroutine(co, Some(error));
                (Values::new(), Some(error))
            }
            Leave::Closed => {
                self.end_coroutine(co, None);
                (Values::new(), None)
            }
        }
    }

    fn exit_coroutine(&mut self, entry: &CoroutineFrame) {
        self.current_thread = entry.caller;
        self.heap.threads.live_mut(entry.caller).status = CoStatus::Running;
        self.floor = entry.floor;
        self.non_yieldable = entry.non_yieldable;
        self.c_calls = entry.c_calls;
        if entry.counted {
            self.leave_c_call();
        }
        self.sync_tracing();
    }

    fn settle_entry(&mut self, entry: &CoroutineFrame, results: Values) -> Result<(), LuaError> {
        self.settle_native(entry.ret, entry.nresults, entry.ret, entry.owner, Step::Done(results))
            .map(|_| ())
    }

    fn end_coroutine(&mut self, co: Ref<Thread>, failure: Option<Value>) {
        let thread = self.heap.threads.live_mut(co);
        thread.status = CoStatus::Dead;
        thread.c_calls = 0;
        thread.failure = failure;
    }

    pub(crate) fn mark_entry_counted(&mut self, caller: Ref<Thread>) {
        let mut thread = self.current_thread;
        while let Some(Frame::Coroutine(entry)) = self.heap.threads.live_mut(thread).frames.first_mut() {
            if entry.caller == caller {
                entry.counted = true;
                return;
            }
            thread = entry.caller;
        }
    }

    fn positioned_error(&mut self, error: Value) -> Value {
        let Value::String(message) = error else {
            return error;
        };
        match self.position_at_level(1) {
            Some(mut prefix) => {
                prefix.extend_from_slice(self.heap_bytes(message));
                self.string_value(&prefix)
            }
            None => error,
        }
    }

    pub(crate) fn close_running_thread(&mut self) -> LuaError {
        let outcome = self.close_after_error(0, Value::Nil);
        self.thread_mut().self_close = Some(outcome);
        LuaError::unwinding(outcome)
    }

    pub(crate) fn unwind_coroutine(&mut self, scope: RunScope, error: LuaError) -> Result<(), LuaError> {
        if self.current_thread == scope.thread {
            return Err(error);
        }
        self.thread_mut().frames.truncate(1);
        let leave = match error.value() {
            Value::Nil => Leave::Closed,
            value => Leave::Failed(value),
        };
        match self.leave_coroutine(leave) {
            Ok(()) => Ok(()),
            Err(raised) => self.recover(scope, raised),
        }
    }

    pub(crate) fn close_thread(&mut self, co: Ref<Thread>) -> Result<(), Value> {
        let failure = self.heap.threads.live_mut(co).failure.take();
        let switch = self.switch_into(co);
        self.thread_mut().frames.clear();
        let error = self.close_after_error(0, failure.unwrap_or(Value::Nil));
        self.thread_mut().top = 0;
        self.switch_back(switch);
        self.heap.threads.live_mut(co).status = CoStatus::Dead;
        if failure.is_some() || !error.is_nil() {
            return Err(error);
        }
        Ok(())
    }

    pub(crate) fn switch_into(&mut self, co: Ref<Thread>) -> Switch {
        let caller = self.current_thread;
        self.heap.threads.live_mut(caller).status = CoStatus::Normal;
        self.heap.threads.live_mut(co).status = CoStatus::Running;
        let switch = Switch {
            pins: self.pin([Value::Thread(caller)]),
            floor: std::mem::replace(&mut self.floor, 0),
            non_yieldable: std::mem::replace(&mut self.non_yieldable, 0),
            c_calls: self.c_calls,
            caller,
        };
        self.current_thread = co;
        self.sync_tracing();
        switch
    }

    pub(crate) fn switch_back(&mut self, switch: Switch) {
        self.unpin(switch.pins);
        self.current_thread = switch.caller;
        self.sync_tracing();
        self.heap.threads.live_mut(switch.caller).status = CoStatus::Running;
        self.floor = switch.floor;
        self.non_yieldable = switch.non_yieldable;
        self.c_calls = switch.c_calls;
    }
}

fn with_flag(flag: bool, values: Values) -> Values {
    let mut results = Values::with_capacity(values.len() + 1);
    results.push(Value::Boolean(flag));
    results.extend(values);
    results
}

/// Pads missing values with nil, as `place_results` does.
#[inline(always)]
fn move_values(threads: &mut Arena<Thread>, (from, first, count): (Ref<Thread>, usize, usize), (to, destination, placed): (Ref<Thread>, usize, usize)) {
    for index in 0..placed {
        let value = if index < count { threads.live(from).stack[first + index] } else { Value::Nil };
        threads.live_mut(to).stack[destination + index] = value;
    }
}
