#![expect(
    unsafe_code,
    reason = "the register window reads and writes a frame's registers without per-access checks"
)]

use std::sync::atomic::{AtomicU8, Ordering};

use super::REGISTER_SPAN;
use crate::{
    Vm,
    arena::Ref,
    instruction::Instruction,
    object::{Captured, Closure, Frame, LuaFrame, Thread, UpVal},
    proto::Proto,
    value::Value,
    vm::{ic::InlineCache, quicken::Dispatch},
};

#[cfg_attr(not(debug_assertions), inline(always))]
pub(super) fn opcode_of(word: u32) -> Dispatch {
    let opcode = (word & 0x7f) as u8;
    debug_assert!(usize::from(opcode) < Dispatch::COUNT);
    // SAFETY: every dispatch word holds an `Opcode` or `Dispatch` discriminant, and `Dispatch` is `repr(u8)` with the same numbers.
    unsafe { std::mem::transmute::<u8, Dispatch>(opcode) }
}

/// Avoids the out-of-line call `Vec::push` makes in the run loop.
#[cfg_attr(not(debug_assertions), inline(always))]
pub(super) fn push_frame(frames: &mut Vec<Frame>, frame: Frame) {
    if frames.len() == frames.capacity() {
        frames.reserve(1);
    }
    let len = frames.len();
    debug_assert!(len < frames.capacity());
    // SAFETY: `reserve` made room for one more frame past `len`, and the slot is written before the length covers it.
    unsafe {
        frames.as_mut_ptr().add(len).write(frame);
        frames.set_len(len + 1);
    }
}

#[cfg_attr(not(debug_assertions), inline(always))]
pub(super) fn top_lua_frame(frames: &mut [Frame]) -> &mut LuaFrame {
    debug_assert!(matches!(frames.last(), Some(Frame::Lua(_))));
    match frames.last_mut() {
        Some(Frame::Lua(frame)) => frame,
        // SAFETY: the run loop runs only Lua frames, so the top frame is one whenever this is called.
        _ => unsafe { std::hint::unreachable_unchecked() },
    }
}

/// A Lua frame owns nothing to drop.
#[cfg_attr(not(debug_assertions), inline(always))]
pub(super) fn pop_lua_frame(frames: &mut Vec<Frame>) {
    let len = frames.len();
    debug_assert!(matches!(frames.last(), Some(Frame::Lua(_))));
    // SAFETY: a `Frame::Lua` holds only `Copy` fields, so shortening the length drops nothing.
    unsafe { frames.set_len(len - 1) }
}

/// One allocation the prototype owns and never moves, so frames and closures can point at it.
#[derive(Clone, Debug, Default)]
pub(crate) struct Exec {
    pub(crate) words: super::quicken::Words,
    pub(crate) feedback: Vec<InlineCache>,
    pub(crate) proto: Ref<Proto>,
    pub(crate) num_params: u8,
    pub(crate) max_stack: u8,
    pub(crate) is_vararg: bool,
    pub(crate) vararg_table: bool,
    constants: Pointer<Value>,
    constants_len: usize,
    upvalue_count: usize,
    #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
    budget: Pointer<std::cell::Cell<i64>>,
    #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
    entry: Pointer<std::cell::Cell<usize>>,
    /// Two cells per instruction (float, mixed) so each path stores a constant instead of merging; written by Baseline code
    /// at a fixed address (V8 feedback-vector.h, `BinaryOperationHint`).
    #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub(crate) arith_kinds: Box<[std::cell::Cell<u8>]>,
}

#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
pub(crate) mod exec_layout {
    use super::{CodeRef, Exec};
    use crate::object::Closure;
    use std::mem::offset_of;

    pub(crate) const CLOSURE_CODE: usize = offset_of!(Closure, code) + offset_of!(CodeRef, 0);
    pub(crate) const PROTO: usize = offset_of!(Exec, proto);
    pub(crate) const NUM_PARAMS: usize = offset_of!(Exec, num_params);
    pub(crate) const MAX_STACK: usize = offset_of!(Exec, max_stack);
    pub(crate) const CONSTANTS: usize = offset_of!(Exec, constants);
    pub(crate) const ENTRY: usize = offset_of!(Exec, entry);
}

#[derive(Debug)]
#[repr(transparent)]
struct Pointer<T>(*const T);

impl<T> Clone for Pointer<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Pointer<T> {}

impl<T> Default for Pointer<T> {
    fn default() -> Self {
        Self(std::ptr::null())
    }
}

// SAFETY: each pointee is owned by the VM's heap, which moves to another thread only with the whole VM.
unsafe impl<T> Send for Pointer<T> {}
// SAFETY: the VM runs on one thread at a time.
unsafe impl<T> Sync for Pointer<T> {}

/// Null until filled.
#[derive(Clone, Copy, Debug, Default)]
#[repr(transparent)]
pub(crate) struct CodeRef(Pointer<Exec>);

/// Null until filled.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct UpvaluesRef(Pointer<Captured>);

/// Null until set; only this module reads through it.
#[derive(Debug)]
pub(crate) struct Raw<T>(*mut T);

impl<T> Clone for Raw<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Raw<T> {}

impl<T> Default for Raw<T> {
    fn default() -> Self {
        Self(std::ptr::null_mut())
    }
}

// SAFETY: the pointee is owned by the VM's heap, which moves to another thread only with the whole VM.
unsafe impl<T> Send for Raw<T> {}
// SAFETY: the VM runs on one thread at a time.
unsafe impl<T> Sync for Raw<T> {}

impl<T> Raw<T> {
    pub(crate) fn new(pointer: *mut T) -> Self {
        Self(pointer)
    }
}

#[cfg(debug_assertions)]
fn check_upvalue(vm: &Vm, cell: Raw<UpVal>) {
    assert!(!cell.0.is_null(), "a closure's upvalue address is filled");
    // SAFETY: the upvalue arena never moves its objects, and the closure keeps the upvalue alive.
    let upvalue = unsafe { &*cell.0 };
    let expected = match &upvalue.state {
        crate::object::UpState::Open { thread, index } => vm.heap.threads.live(*thread).stack.as_ptr().wrapping_add(*index),
        crate::object::UpState::Closed(own) => std::ptr::from_ref(own),
    };
    assert!(std::ptr::eq(upvalue.value.0.cast_const(), expected), "an upvalue points at its variable");
}

/// Follows `cl->upvals[b]->v.p` in `luaV_execute` (lvm.c).
#[cfg_attr(not(debug_assertions), inline(always))]
pub(crate) fn upvalue_read(vm: &Vm, cell: Raw<UpVal>) -> Value {
    #[cfg(debug_assertions)]
    check_upvalue(vm, cell);
    #[cfg(not(debug_assertions))]
    let _ = vm;
    // SAFETY: the value pointer names a closed value or an open stack slot, which `Heap::stack_resized` repoints on every stack resize.
    unsafe { *(*cell.0).value.0 }
}

/// Follows `OP_SETUPVAL` in lvm.c; returns false for a closed upvalue, which needs the collector's barrier.
#[cfg_attr(not(debug_assertions), inline(always))]
pub(crate) fn upvalue_write(vm: &Vm, cell: Raw<UpVal>, value: Value) -> bool {
    #[cfg(debug_assertions)]
    check_upvalue(vm, cell);
    #[cfg(not(debug_assertions))]
    let _ = vm;
    // SAFETY: the state and pointer are read before the write, and the run loop holds no reference into the upvalue during it.
    unsafe {
        let open = (*cell.0).is_open();
        let variable = (*cell.0).value.0;
        variable.write(value);
        !open
    }
}

impl CodeRef {
    pub(crate) fn of(exec: &Exec) -> Self {
        Self(Pointer(std::ptr::from_ref(exec)))
    }

    pub(crate) fn is_null(self) -> bool {
        self.0.0.is_null()
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(crate) fn get<'a>(self) -> &'a Exec {
        debug_assert!(!self.is_null());
        // SAFETY: a filled reference names an `Exec` kept alive by its closure, and every in-place closure change clears it.
        unsafe { &*self.0.0 }
    }
}

impl UpvaluesRef {
    pub(crate) fn of(closure: &Closure) -> Self {
        Self(Pointer(closure.upvalues.as_ptr()))
    }
}

impl Exec {
    pub(crate) fn new(code: &[Instruction]) -> Self {
        Self {
            words: crate::encoding::dispatch_words(code),
            feedback: super::ic::initial_feedback(code),
            ..Self::default()
        }
    }
}

impl Proto {
    /// Fills the `Exec` once the prototype has its heap reference.
    pub(crate) fn seal(&mut self, reference: Ref<Proto>) {
        let exec = &mut *self.exec;
        exec.proto = reference;
        exec.num_params = self.num_params;
        exec.max_stack = self.max_stack;
        exec.is_vararg = self.is_vararg;
        exec.vararg_table = self.vararg_table;
        exec.constants = Pointer(self.constants.as_ptr());
        exec.constants_len = self.constants.len();
        exec.upvalue_count = self.upvalues.len();
        #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            exec.budget = Pointer(std::ptr::from_ref(self.jit.budget()));
            exec.entry = Pointer(std::ptr::from_ref(self.jit.entry()));
        }
    }
}

impl crate::heap::Heap {
    pub(crate) fn alloc_proto(&mut self, proto: Proto) -> Ref<Proto> {
        let reference = self.alloc(proto);
        self.protos.live_mut(reference).seal(reference);
        reference
    }

    pub(crate) fn alloc_closure(&mut self, proto: Ref<Proto>, upvalues: Vec<Captured>) -> Ref<Closure> {
        let code = CodeRef::of(&self.protos.live(proto).exec);
        self.alloc(Closure { proto, upvalues, code })
    }

    /// Snapshots do not store the code references, so every restored prototype and closure is rebuilt.
    pub(crate) fn rebuild_code(&mut self) {
        let protos: Vec<Ref<Proto>> = self.protos.iter().map(|(reference, _)| reference).collect();
        for proto in protos {
            self.protos.live_mut(proto).seal(proto);
        }
        let closures: Vec<Ref<Closure>> = self.closures.iter().map(|(reference, _)| reference).collect();
        for closure in closures {
            let proto = self.closures.live(closure).proto;
            let code = CodeRef::of(&self.protos.live(proto).exec);
            self.closures.live_mut(closure).code = code;
        }
    }
}

/// Passed by copy in optimized builds, so the window stays in registers instead of memory.
#[cfg(debug_assertions)]
pub(super) type Passed<'w> = &'w Window;
#[cfg(not(debug_assertions))]
pub(super) type Passed<'w> = Window;

/// The dispatch word of the next instruction; every arm advances it itself.
#[derive(Clone, Copy)]
pub(super) struct Ip(*const super::quicken::Site);

impl Ip {
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn add(self, count: usize) -> Self {
        Self(self.0.wrapping_add(count))
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn sub(self, count: usize) -> Self {
        Self(self.0.wrapping_sub(count))
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn offset(self, count: isize) -> Self {
        Self(self.0.wrapping_offset(count))
    }
}

/// Registers and constants stay in machine registers; the rest of the frame is read through the record pointer, as `luaV_execute` reads `ci`.
#[derive(Clone, Copy)]
pub(super) struct Window {
    registers: *mut Value,
    record: *mut Record,
}

/// One record serves every window of one run of the dispatch loop; building a window rewrites it.
pub(super) struct Record {
    words: *const super::quicken::Site,
    constants: *const Value,
    frame: *mut LuaFrame,
    thread: *mut Thread,
    requests: *const AtomicU8,
    #[cfg(debug_assertions)]
    register_count: usize,
    #[cfg(debug_assertions)]
    stack_len: usize,
    #[cfg(debug_assertions)]
    code_len: usize,
    #[cfg(debug_assertions)]
    constants_len: usize,
    #[cfg(debug_assertions)]
    depth: usize,
}

impl Record {
    pub(super) fn polling(requests: &AtomicU8) -> Self {
        Self {
            requests: std::ptr::from_ref(requests),
            ..Self::default()
        }
    }
}

impl Default for Record {
    fn default() -> Self {
        Self {
            words: std::ptr::null(),
            constants: std::ptr::null(),
            frame: std::ptr::null_mut(),
            thread: std::ptr::null_mut(),
            requests: std::ptr::null(),
            #[cfg(debug_assertions)]
            register_count: 0,
            #[cfg(debug_assertions)]
            stack_len: 0,
            #[cfg(debug_assertions)]
            code_len: 0,
            #[cfg(debug_assertions)]
            constants_len: 0,
            #[cfg(debug_assertions)]
            depth: 0,
        }
    }
}

impl Window {
    #[cfg(debug_assertions)]
    #[inline(always)]
    pub(super) fn pass(&self) -> Passed<'_> {
        self
    }

    #[cfg(not(debug_assertions))]
    #[inline(always)]
    pub(super) fn pass(&self) -> Passed<'_> {
        *self
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn enter(vm: &mut Vm, record: *mut Record) -> Self {
        let heap = &mut vm.heap;
        let thread = heap.threads.live_mut(vm.current_thread);
        let Some(Frame::Lua(frame)) = thread.frames.last_mut() else {
            unreachable!("the run loop enters a Lua frame");
        };
        if frame.code.is_null() {
            let closure = heap.closures.live(frame.closure);
            frame.code = closure.code;
            frame.upvalues = UpvaluesRef::of(closure);
        }
        let needed = frame.base + REGISTER_SPAN;
        if thread.stack.len() < needed {
            Self::grown(vm, needed);
        }
        Self::top(vm.heap.threads.live_mut(vm.current_thread), record)
    }

    #[cold]
    #[inline(never)]
    fn grown(vm: &mut Vm, needed: usize) {
        vm.grow_stack(needed);
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    fn top(thread: &mut Thread, record: *mut Record) -> Self {
        let Some(Frame::Lua(_)) = thread.frames.last() else {
            unreachable!("the run loop enters a Lua frame");
        };
        Self::top_lua(thread, record)
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn top_lua(thread: &mut Thread, record: *mut Record) -> Self {
        let depth = thread.frames.len();
        let stack = (thread.stack.as_mut_ptr(), thread.stack.len());
        let owner = std::ptr::from_mut(thread);
        Self::of(record, owner, top_lua_frame(&mut thread.frames), stack, depth)
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn called(&self, thread: &mut Thread) -> Self {
        self.same_thread(thread)
    }

    /// The stack still holds the caller's registers, since a collection shrinks it only down to what the frames above use.
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn returned(&self, thread: &mut Thread) -> Self {
        self.same_thread(thread)
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    fn same_thread(&self, thread: &mut Thread) -> Self {
        debug_assert!(std::ptr::eq(self.shared().thread, thread));
        let stack = thread.stack.as_mut_ptr();
        #[cfg(debug_assertions)]
        let (stack_len, depth) = (thread.stack.len(), thread.frames.len());
        let frame = top_lua_frame(&mut thread.frames);
        let exec = frame.code.get();
        #[cfg(debug_assertions)]
        assert!(stack_len >= frame.base + REGISTER_SPAN, "the stack holds the register window");
        // SAFETY: the record is the dispatch loop's, and frame and thread are the running thread's top frame just read.
        unsafe {
            let shared = &mut *self.record;
            shared.words = exec.words.as_ptr();
            shared.constants = exec.constants.0;
            shared.frame = std::ptr::from_mut(frame);
            #[cfg(debug_assertions)]
            {
                shared.register_count = usize::from(exec.max_stack);
                shared.stack_len = stack_len;
                shared.code_len = exec.words.len();
                shared.constants_len = exec.constants_len;
                shared.depth = depth;
            }
        }
        Self {
            registers: stack.wrapping_add(frame.base),
            record: self.record,
        }
    }

    pub(super) fn record(&self) -> *mut Record {
        self.record
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    fn shared(&self) -> &Record {
        // SAFETY: the record outlives every window of its dispatch loop and changes only when a window is built or refreshed.
        unsafe { &*self.record }
    }

    /// A window lives while the thread arena has not moved; `refresh` runs after every operation that can move it.
    #[expect(clippy::mut_from_ref, reason = "the thread lives in the heap's thread arena, not in the window")]
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn thread(&self) -> &mut Thread {
        // SAFETY: `refresh` re-reads the thread's arena slot after every operation that can create a thread, and no other reference is held.
        unsafe { &mut *self.shared().thread }
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn returns_to(&self) -> (bool, usize, i32) {
        let frame = self.frame();
        #[cfg(debug_assertions)]
        if frame.nested() {
            assert!(!frame.counted(), "a nested frame counts no C call");
        }
        (frame.nested(), frame.ret(), frame.nresults())
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    fn of(record: *mut Record, thread: *mut Thread, frame: &mut LuaFrame, (stack_ptr, stack_len): (*mut Value, usize), depth: usize) -> Self {
        let exec = frame.code.get();
        debug_assert!(stack_len >= frame.base + REGISTER_SPAN);
        #[cfg(not(debug_assertions))]
        let _ = (stack_len, depth);
        // SAFETY: `record` is the dispatch loop's record, initialized before any window is built from it.
        let requests = unsafe { (*record).requests };
        let shared = Record {
            words: exec.words.as_ptr(),
            constants: exec.constants.0,
            frame: std::ptr::from_mut(frame),
            thread,
            requests,
            #[cfg(debug_assertions)]
            register_count: usize::from(exec.max_stack),
            #[cfg(debug_assertions)]
            stack_len,
            #[cfg(debug_assertions)]
            code_len: exec.words.len(),
            #[cfg(debug_assertions)]
            constants_len: exec.constants_len,
            #[cfg(debug_assertions)]
            depth,
        };
        // SAFETY: the record is the dispatch loop's, and no reference to it is held while a window is built.
        unsafe { record.write(shared) };
        Self {
            registers: stack_ptr.wrapping_add(frame.base),
            record,
        }
    }

    pub(super) fn registers(&self) -> *mut Value {
        self.registers
    }

    /// A window lives while the frame vector has not moved; `refresh` runs after every operation that can move it.
    #[cfg_attr(not(debug_assertions), inline(always))]
    fn frame(&self) -> &LuaFrame {
        // SAFETY: `refresh` re-reads the frame vector's slot after every push, and no `&mut` of the frame is held while the window reads it.
        unsafe { &*self.shared().frame }
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn exec(&self) -> &Exec {
        self.frame().code.get()
    }

    /// Follows PUC's `savepc`.
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn save_pc(&self, pc: usize) {
        // SAFETY: the write is the only access to the record while it runs.
        unsafe { (*self.shared().frame).set_pc(pc) }
    }

    /// Follows `OP_FORLOOP` reloading `ci->u.l.trap` in lvm.c.
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn requested(&self) -> bool {
        let requests = self.shared().requests;
        debug_assert!(!requests.is_null());
        // SAFETY: the record points at the running VM's request flag, which the VM's `Arc` keeps alive and in place.
        unsafe { (*requests).load(Ordering::Relaxed) != 0 }
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    fn stack(&self) -> *mut Value {
        self.registers.wrapping_sub(self.base())
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    fn budget_spent(&self) -> bool {
        #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
        return self.budget().get() <= 0;
        #[cfg(not(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64"))))]
        false
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn would_spend(&self, amount: u32) -> bool {
        #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
        return self.budget().get() <= i64::from(amount);
        #[cfg(not(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64"))))]
        {
            let _ = amount;
            false
        }
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn charge(&self, amount: u32) {
        #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            let budget = self.budget();
            budget.set(budget.get().wrapping_sub(i64::from(amount)));
        }
        #[cfg(not(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64"))))]
        let _ = amount;
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn charge_spent(&self, amount: u32) -> bool {
        self.charge(amount);
        self.budget_spent()
    }

    #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn budget(&self) -> &std::cell::Cell<i64> {
        let budget = self.exec().budget.0;
        debug_assert!(!budget.is_null());
        // SAFETY: the budget is owned by the frame's prototype, which the closure keeps alive; `Cell` allows the write through a shared reference.
        unsafe { &*budget }
    }

    pub(super) fn closure(&self) -> Ref<Closure> {
        self.frame().closure
    }

    pub(super) fn proto(&self) -> Ref<Proto> {
        self.exec().proto
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn base(&self) -> usize {
        self.frame().base
    }

    /// Re-reads the stack buffer and the frame record after an operation that may have moved either.
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn refresh(&mut self, vm: &mut Vm) {
        let thread = vm.thread_mut();
        #[cfg(debug_assertions)]
        assert_eq!(thread.frames.len(), self.shared().depth, "the window's frame is the top frame");
        let owner = std::ptr::from_mut(thread);
        let Some(Frame::Lua(frame)) = thread.frames.last_mut() else {
            unreachable!("the window's frame is a Lua frame");
        };
        let base = frame.base;
        let words = frame.code.get().words.as_ptr();
        let frame = std::ptr::from_mut(frame);
        assert!(thread.stack.len() >= base + REGISTER_SPAN, "the stack holds the register window");
        // SAFETY: the record is the dispatch loop's, and frame and thread are the running thread's top frame just read.
        unsafe {
            let shared = &mut *self.record;
            shared.thread = owner;
            shared.frame = frame;
            shared.words = words;
            #[cfg(debug_assertions)]
            {
                shared.stack_len = thread.stack.len();
            }
        }
        self.registers = thread.stack.as_mut_ptr().wrapping_add(base);
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn ip_at(&self, pc: usize) -> Ip {
        Ip(self.shared().words.wrapping_add(pc))
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn pc(&self, ip: Ip) -> usize {
        (ip.0.addr() - self.shared().words.addr()) / std::mem::size_of::<super::quicken::Site>()
    }

    /// PUC's saved `pc` points past the running instruction.
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn next_pc(&self, ip: Ip) -> usize {
        self.pc(ip) + 1
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn fetch(&self, ip: Ip) -> u32 {
        #[cfg(debug_assertions)]
        assert!(ip.0.addr() >= self.shared().words.addr() && self.pc(ip) < self.shared().code_len);
        // SAFETY: the verifier keeps every jump and fall-through of the frame below code_len.
        unsafe { (*ip.0).word.get() }
    }

    pub(super) fn instruction(&self, pc: usize) -> Instruction {
        let word = self.word(pc);
        crate::encoding::operands(opcode_of(word).generic(), word)
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn word(&self, pc: usize) -> u32 {
        #[cfg(debug_assertions)]
        assert!(pc < self.shared().code_len);
        // SAFETY: the verifier bounds every jump and resumed pc below code_len; the words are never resized, and change only through `Cell`.
        unsafe { (*self.shared().words.add(pc)).word.get() }
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn rewrite(&self, pc: usize, form: Dispatch) {
        #[cfg(debug_assertions)]
        assert!(pc < self.shared().code_len);
        // SAFETY: the verifier bounds `pc` below code_len, and each site's word changes only through `Cell`.
        let site = unsafe { &*self.shared().words.add(pc) };
        site.word.set((site.word.get() & !0x7f) | u32::from(form as u8));
        site.misses.set(super::quicken::MISS_BUDGET);
    }

    /// Rewrites the site to `generic` once its miss budget is spent.
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn miss(&self, pc: usize, generic: Dispatch) {
        #[cfg(debug_assertions)]
        assert!(pc < self.shared().code_len);
        // SAFETY: the verifier bounds `pc` below code_len.
        let misses = unsafe { &(*self.shared().words.add(pc)).misses };
        let left = misses.get().saturating_sub(1);
        misses.set(left);
        if left == 0 {
            self.rewrite(pc, generic);
        }
    }

    pub(super) fn constant(&self, index: u32) -> Value {
        let index = index as usize;
        #[cfg(debug_assertions)]
        assert!(index < self.shared().constants_len);
        // SAFETY: the verifier bounds every constant index, and the constant buffer is never resized after the prototype is built.
        unsafe { *self.shared().constants.add(index) }
    }

    /// `None` while the prototype has no feedback.
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn feedback(&self, site: usize) -> Option<&InlineCache> {
        self.exec().feedback.get(site)
    }

    /// The feedback stride is a whole multiple of the words' stride, so the address is one scaled subtraction.
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn feedback_at(&self, ip: Ip) -> Option<&InlineCache> {
        const STRIDE: usize = std::mem::size_of::<InlineCache>() / std::mem::size_of::<super::quicken::Site>();
        const _: () = assert!(std::mem::size_of::<InlineCache>().is_multiple_of(std::mem::size_of::<super::quicken::Site>()));
        let feedback = &self.exec().feedback;
        let offset = ip.0.addr() - self.shared().words.addr();

        if feedback.is_empty() {
            return None;
        }
        #[cfg(debug_assertions)]
        assert!(feedback.len() == self.shared().code_len && self.pc(ip) < feedback.len());
        // SAFETY: feedback is empty or sized to the code, and `ip` is an instruction of this frame.
        Some(unsafe { &*feedback.as_ptr().byte_add(offset * STRIDE) })
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    fn captured(&self, index: u8) -> Captured {
        let index = usize::from(index);
        let upvalues = self.frame().upvalues.0.0;
        debug_assert!(index < self.exec().upvalue_count && !upvalues.is_null());
        // SAFETY: the verifier bounds every upvalue index; the closure keeps its upvalue list alive, and the list never changes length.
        unsafe { *upvalues.add(index) }
    }

    /// Follows `cl->upvals[b]->v.p` in `luaV_execute` (lvm.c).
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn upvalue_value(&self, vm: &Vm, index: u8) -> Value {
        upvalue_read(vm, self.captured(index).cell)
    }

    /// A closed upvalue's write needs the collector's barrier.
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn set_upvalue(&self, vm: &mut Vm, index: u8, value: Value) {
        let captured = self.captured(index);
        if upvalue_write(vm, captured.cell, value) && value.is_collectable() {
            vm.heap.closed_upvalue_barrier(captured.reference, value);
        }
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn register_ref(&self, index: impl Into<usize>) -> &Value {
        let index = index.into();
        #[cfg(debug_assertions)]
        assert!(index < self.shared().register_count);
        debug_assert!(index < REGISTER_SPAN);
        // SAFETY: in bounds as for `register`; the reference is dropped before any write through the window.
        unsafe { &*self.registers.add(index) }
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn constant_ref(&self, index: u32) -> &Value {
        let index = index as usize;
        #[cfg(debug_assertions)]
        assert!(index < self.shared().constants_len);
        // SAFETY: the constant buffer is never written while the frame runs.
        unsafe { &*self.shared().constants.add(index) }
    }

    pub(super) fn register(&self, index: impl Into<usize>) -> Value {
        let index = index.into();
        #[cfg(debug_assertions)]
        assert!(index < self.shared().register_count);
        debug_assert!(index < REGISTER_SPAN);
        // SAFETY: `enter` sized the stack for `REGISTER_SPAN` slots above base, and `refresh` re-reads it after every resize.
        unsafe { *self.registers.add(index) }
    }

    /// Follows `chgivalue` in lvm.h; the caller holds at most one reference per register at a time.
    #[expect(clippy::mut_from_ref, reason = "the registers live in the thread's stack, not in the window")]
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn register_mut(&self, index: usize) -> &mut Value {
        #[cfg(debug_assertions)]
        assert!(index < self.shared().register_count);
        debug_assert!(index < REGISTER_SPAN);
        // SAFETY: `index` is below `REGISTER_SPAN`, and callers take distinct registers for references held together.
        unsafe { &mut *self.registers.add(index) }
    }

    pub(super) fn set_register(&self, index: impl Into<usize>, value: Value) {
        let index = index.into();
        #[cfg(debug_assertions)]
        assert!(index < self.shared().register_count);
        debug_assert!(index < REGISTER_SPAN);
        // SAFETY: `index` is below `REGISTER_SPAN`, and nothing else references the stack buffer while the window runs.
        unsafe { self.registers.add(index).write(value) }
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn move_register(&self, from: impl Into<usize>, to: impl Into<usize>) {
        let (from, to) = (from.into(), to.into());
        #[cfg(debug_assertions)]
        assert!(from < self.shared().register_count && to < self.shared().register_count);
        debug_assert!(from < REGISTER_SPAN && to < REGISTER_SPAN);
        // SAFETY: both indexes are below `REGISTER_SPAN`; `copy` allows `from == to`.
        unsafe { std::ptr::copy(self.registers.add(from), self.registers.add(to), 1) }
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn copy_register(&self, index: impl Into<usize>, value: &Value) {
        let index = index.into();
        #[cfg(debug_assertions)]
        assert!(index < self.shared().register_count);
        debug_assert!(index < REGISTER_SPAN);
        // SAFETY: `value` is a table's slot or node, never a register, so the ranges do not overlap.
        unsafe { std::ptr::copy_nonoverlapping(value, self.registers.add(index), 1) }
    }

    /// The buffer moves when the stack resizes and the frame vector when a frame is pushed, so `refresh` must run after either.
    #[cfg(debug_assertions)]
    pub(super) fn assert_current(&self, vm: &Vm) {
        let thread = vm.thread();
        let top = match thread.frames.last() {
            Some(Frame::Lua(frame)) => std::ptr::from_ref(frame),
            _ => std::ptr::null(),
        };
        assert!(
            self.stack().cast_const() == thread.stack.as_ptr()
                && self.shared().stack_len == thread.stack.len()
                && top == self.shared().frame.cast_const()
                && std::ptr::from_ref(thread) == self.shared().thread.cast_const(),
            "the register window is stale: it was not refreshed after the stack or the frames changed"
        );
    }

    /// Follows `hvalue(rb)` in `luaV_execute` (lvm.c); a register or upvalue holds the table, so it is live.
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn table_object(vm: &Vm, table: Ref<crate::table::Table>) -> &crate::table::Table {
        debug_assert!(vm.heap.tables.try_get(table).is_some());
        // SAFETY: a live table's arena slot is below the arena length and stays valid while a register or upvalue refers to it.
        unsafe { &*vm.heap.tables.objects_ptr().add(table.index()) }
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn table_object_mut(vm: &mut Vm, table: Ref<crate::table::Table>) -> &mut crate::table::Table {
        debug_assert!(vm.heap.tables.try_get(table).is_some());
        // SAFETY: as for `table_object`; the `&mut Vm` makes this the only reference into the arena.
        unsafe { &mut *vm.heap.tables.objects_mut_ptr().add(table.index()) }
    }

    /// Read without a bounds check; the cache entry was recorded for this shape.
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn cached_slot(table: &crate::table::Table, slot: usize) -> &Value {
        let slots = table.slot_values();
        debug_assert!(slot < slots.len());
        // SAFETY: a table holds one slot per key of its shape, and a cache entry's slot is a key's position in that shape.
        unsafe { slots.get_unchecked(slot) }
    }

    /// Follows `key->hash` in `luaH_getshortstr` (ltable.c).
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn string_hash(vm: &Vm, string: Ref<crate::object::LuaString>) -> u64 {
        debug_assert!(vm.heap.strings.try_get(string).is_some() && string.index() < vm.heap.string_keys.len());
        // SAFETY: every live string has its key entry, below the arena length.
        unsafe { (*vm.heap.string_keys.as_ptr().add(string.index())).hash.get() }
    }

    /// Follows `clLvalue(s2v(ra))` in `luaV_execute` (lvm.c).
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn closure_object(vm: &Vm, closure: Ref<Closure>) -> &Closure {
        debug_assert!(vm.heap.closures.try_get(closure).is_some());
        // SAFETY: a register holds the closure, so its arena slot is live and below the arena length.
        unsafe { &*vm.heap.closures.objects_ptr().add(closure.index()) }
    }

    /// Fills missing parameters with nil, as `luaD_precall` does.
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn fill_missing(&self, from: usize, to: usize) {
        if from < to {
            Self::fill_nil(*self, from, to);
        }
    }

    #[inline(never)]
    fn fill_nil(self, from: usize, to: usize) {
        for register in from..to {
            self.set_register(register, Value::Nil);
        }
    }

    /// One value at a time, as `luaD_pretailcall` moves a tail call's function and arguments.
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn move_down(&self, from: usize, to: usize, count: usize) {
        debug_assert!(to <= from);
        #[cfg(debug_assertions)]
        assert!(from + count <= self.shared().stack_len);
        let stack = self.stack();
        for offset in 0..count {
            // SAFETY: the slots lie below `from + count`, which the stack length covers, and `to` is below them.
            unsafe { stack.add(to + offset).write(*stack.add(from + offset)) }
        }
    }

    /// Follows `moveresults` in ldo.c.
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn move_results(&self, first: usize, count: usize, destination: usize, wanted: i32) {
        debug_assert!(destination < first || count == 0);
        if wanted == 0 {
            return;
        }
        if wanted == 1 {
            if count == 0 {
                self.set_stack_value(destination, Value::Nil);
            } else {
                self.copy_stack_value(first, destination);
            }
            return;
        }
        Self::move_many(*self, first, count, destination, wanted);
    }

    #[inline(never)]
    fn move_many(self, first: usize, count: usize, destination: usize, wanted: i32) {
        let wanted = if wanted < 0 { count } else { wanted as usize };
        let moved = wanted.min(count);
        self.move_down(first, destination, moved);
        for slot in destination + moved..destination + wanted {
            self.set_stack_value(slot, Value::Nil);
        }
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn copy_stack_value(&self, from: usize, to: usize) {
        #[cfg(debug_assertions)]
        assert!(from < self.shared().stack_len && to < self.shared().stack_len);
        // SAFETY: both slots are below stack_len while the window is current; `copy` allows `from == to`.
        unsafe { std::ptr::copy(self.stack().add(from), self.stack().add(to), 1) }
    }

    pub(super) fn set_stack_value(&self, index: usize, value: Value) {
        #[cfg(debug_assertions)]
        assert!(index < self.shared().stack_len);
        // SAFETY: the index is below stack_len while the window is current, and the write goes through the buffer the window owns.
        unsafe { self.stack().add(index).write(value) }
    }
}

#[cfg(test)]
mod tests {
    use crate::{Options, Vm};

    #[test]
    fn a_write_through_an_open_upvalue_is_seen_by_the_frame_that_owns_the_slot() {
        let mut vm = Vm::new(Options::default());
        vm.run_source(
            b"=t",
            b"local function counter()\n\
                local n = 0\n\
                local function bump() n = n + 1 return n end\n\
                for _ = 1, 100 do bump() end\n\
                assert(n == 100)\n\
                n = n + 1\n\
                assert(bump() == 102)\n\
                assert(n == 102)\n\
              end\n\
              counter()",
        )
        .expect("the open upvalue writes reach the owning frame");
    }
}
