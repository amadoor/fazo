use std::borrow::Cow;

use crate::{
    LuaError, Values, Vm,
    arena::Ref,
    proto::Proto,
    stdlib::{
        base::PrintState,
        machine::Machine,
        string::{FormatState, GsubState},
    },
    table::Table,
    value::Value,
};

pub type NativeFn = fn(&mut Vm, &[Value]) -> Result<Values, LuaError>;

/// Returns a [`Step`] instead of results, so the VM runs the callee in its own frames.
pub type SteppedFn = fn(&mut Vm, &[Value]) -> Result<Step, LuaError>;

#[derive(Clone, Copy, Debug)]
pub enum NativeBody {
    Plain(NativeFn),
    Stepped(SteppedFn),
    Yield,
    CoroutineResume,
    CoroutineWrap,
}

#[derive(Clone, Debug)]
pub(crate) struct NativeEntry {
    pub(crate) name: Cow<'static, str>,
    pub(crate) body: NativeBody,
    pub(crate) data: u32,
    pub(crate) builtin: Option<crate::vm::Builtin>,
}

/// Restore maps the index to the same registered name on the restoring VM.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct NativeId(pub u32);

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Native {
    pub function: NativeId,
    pub upvalues: Vec<Value>,
    #[serde(skip)]
    pub(crate) builtin: Option<crate::vm::Builtin>,
}

/// Either finishes with results, or calls a value and resumes with its results.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum Step {
    Done(Values),
    Call {
        function: Value,
        arguments: Values,
        resume: Resume,
    },
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum Resume {
    ToString,
    Pairs,
    Ipairs(i64),
    Print(Box<PrintState>),
    Format(Box<FormatState>),
    Gsub(Box<GsubState>),
    Machine(Box<Machine>),
    Protected(Protection),
    Unwinding(Unwinding),
    Raised,
    Hook(Transfer),
    First,
    All,
}

impl Resume {
    pub(crate) fn values(&self, out: &mut Values) {
        match self {
            Resume::Print(state) => state.values(out),
            Resume::Format(state) => state.values(out),
            Resume::Gsub(state) => state.values(out),
            Resume::Machine(state) => state.values(out),
            Resume::Protected(protection) => out.extend(protection.handler),
            Resume::Unwinding(state) => out.push(state.error),
            Resume::ToString
            | Resume::Pairs
            | Resume::Ipairs(_)
            | Resume::Raised
            | Resume::Hook(_)
            | Resume::First
            | Resume::All => {}
        }
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct Protection {
    pub handler: Option<Value>,
    pub finalizer: bool,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Userdata {
    pub metatable: Option<Ref<Table>>,
    pub resource: usize,
    pub user_values: Vec<Value>,
}

/// Follows `createstrobj` in lstring.c; up to 16 bytes live in the arena slot itself.
pub type StringBytes = smallvec::SmallVec<[u8; 16]>;

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct LuaString {
    #[serde(with = "crate::bytes")]
    pub bytes: StringBytes,
}

#[cfg(target_pointer_width = "64")]
const _: () = assert!(std::mem::size_of::<LuaString>() == 24);

/// Compiled code compares the prototype at offset 0 to recognize a callee of any of its closures.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[repr(C)]
pub struct Closure {
    pub proto: Ref<Proto>,
    pub upvalues: Vec<Captured>,
    #[serde(skip)]
    pub(crate) code: crate::vm::CodeRef,
}

/// Follows `LClosure::upvals` in lobject.h; compiled code reads the address at offset 8.
#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
#[repr(C)]
pub struct Captured {
    pub reference: Ref<UpVal>,
    #[serde(skip)]
    pub(crate) cell: crate::vm::Raw<UpVal>,
}

/// Follows `UpVal` in lobject.h; compiled code reads `value` at offset 0 and the state tag at offset 8.
/// A stack resize repoints the thread's open upvalues, as `correctstack` in ldo.c does.
#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
#[repr(C)]
pub struct UpVal {
    #[serde(skip)]
    pub(crate) value: crate::vm::Raw<Value>,
    pub state: UpState,
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
#[repr(C, u8)]
pub enum UpState {
    Open { thread: Ref<Thread>, index: usize },
    Closed(Value),
}

impl Default for UpState {
    fn default() -> Self {
        UpState::Closed(Value::Nil)
    }
}

impl UpVal {
    pub(crate) fn open(thread: Ref<Thread>, index: usize, slot: *mut Value) -> Self {
        Self {
            value: crate::vm::Raw::new(slot),
            state: UpState::Open { thread, index },
        }
    }

    pub(crate) fn point_at(&mut self, slot: *mut Value) {
        self.value = crate::vm::Raw::new(slot);
    }

    /// Follows `luaF_closeupval` in lfunc.c.
    pub(crate) fn close(&mut self, value: Value) {
        self.state = UpState::Closed(value);
        if let UpState::Closed(own) = &mut self.state {
            self.value = crate::vm::Raw::new(std::ptr::from_mut(own));
        }
    }

    pub(crate) fn is_open(&self) -> bool {
        matches!(self.state, UpState::Open { .. })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Pending {
    Idle,
    Operation,
    Concat(usize),
    Return(usize),
    Closing(usize),
}

/// Follows `finishpcallk` in ldo.c.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct Unwinding {
    pub level: usize,
    pub error: Value,
}

/// Follows `callstatus` in lstate.h.
pub(crate) mod status {
    pub(crate) const NRESULTS: u32 = 0xff;
    pub(crate) const CCMT_SHIFT: u32 = 8;
    pub(crate) const CCMT: u32 = 0xf << CCMT_SHIFT;
    pub(crate) const TAIL: u32 = 1 << 12;
    pub(crate) const HOOKED: u32 = 1 << 13;
    pub(crate) const COUNTED: u32 = 1 << 14;
    pub(crate) const NESTED: u32 = 1 << 15;
    pub(crate) const PENDING_SHIFT: u32 = 16;
    pub(crate) const PENDING: u32 = 0x7 << PENDING_SHIFT;
    pub(crate) const HIDDEN_SHIFT: u32 = 24;
    pub(crate) const HIDDEN: u32 = 0xff << HIDDEN_SHIFT;
}

/// Follows `CallInfo` in lstate.h.
/// `nvarargs` locates the moved function slot that `OP_RETURN` restores, as in lvm.c.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
#[serde(from = "LuaFrameRecord", into = "LuaFrameRecord")]
#[repr(C)]
pub struct LuaFrame {
    pub closure: Ref<Closure>,
    pub base: usize,
    pub(crate) code: crate::vm::CodeRef,
    pub(crate) upvalues: crate::vm::UpvaluesRef,
    pc: u32,
    status: u32,
    nvarargs: u32,
    operand: u32,
}

#[cfg(target_pointer_width = "64")]
const _: () = assert!(std::mem::size_of::<LuaFrame>() == 48);

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
struct LuaFrameRecord {
    closure: Ref<Closure>,
    base: usize,
    pc: u32,
    status: u32,
    nvarargs: u32,
    operand: u32,
}

impl From<LuaFrameRecord> for LuaFrame {
    fn from(record: LuaFrameRecord) -> Self {
        Self {
            closure: record.closure,
            base: record.base,
            code: crate::vm::CodeRef::default(),
            upvalues: crate::vm::UpvaluesRef::default(),
            pc: record.pc,
            status: record.status & !status::NESTED,
            nvarargs: record.nvarargs,
            operand: record.operand,
        }
    }
}

impl From<LuaFrame> for LuaFrameRecord {
    fn from(frame: LuaFrame) -> Self {
        Self {
            closure: frame.closure,
            base: frame.base,
            pc: frame.pc,
            status: frame.status & !status::NESTED,
            nvarargs: frame.nvarargs,
            operand: frame.operand,
        }
    }
}

impl LuaFrame {
    #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub(crate) const PC_OFFSET: usize = std::mem::offset_of!(LuaFrame, pc);
    #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub(crate) const STATUS_OFFSET: usize = std::mem::offset_of!(LuaFrame, status);
    #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub(crate) const NVARARGS_OFFSET: usize = std::mem::offset_of!(LuaFrame, nvarargs);

    #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
    #[inline(always)]
    pub(crate) fn called(closure: Ref<Closure>, base: usize, nresults: i32, ccmt: u8) -> Self {
        Self::with_status(closure, base, Self::call_status(nresults, ccmt))
    }

    #[inline(always)]
    pub(crate) fn with_status(closure: Ref<Closure>, base: usize, status: u32) -> Self {
        Self::entered(closure, base, status, (crate::vm::CodeRef::default(), crate::vm::UpvaluesRef::default()))
    }

    #[inline(always)]
    pub(crate) fn entered(closure: Ref<Closure>, base: usize, status: u32, (code, upvalues): (crate::vm::CodeRef, crate::vm::UpvaluesRef)) -> Self {
        Self {
            closure,
            base,
            code,
            upvalues,
            pc: 0,
            status,
            nvarargs: 0,
            operand: 0,
        }
    }

    #[inline(always)]
    pub(crate) fn call_status(nresults: i32, ccmt: u8) -> u32 {
        debug_assert!((-1..=254).contains(&nresults) && ccmt <= 15);
        ((nresults + 1) as u32 & status::NRESULTS) | (u32::from(ccmt) << status::CCMT_SHIFT)
    }

    #[inline(always)]
    pub fn pc(&self) -> usize {
        self.pc as usize
    }

    #[inline(always)]
    pub(crate) fn set_pc(&mut self, pc: usize) {
        debug_assert!(u32::try_from(pc).is_ok());
        self.pc = pc as u32;
    }

    #[inline(always)]
    pub(crate) fn status(&self) -> u32 {
        self.status
    }

    #[inline(always)]
    pub(crate) fn set_status(&mut self, status: u32) {
        self.status = status;
    }

    #[inline(always)]
    pub fn nresults(&self) -> i32 {
        (self.status & status::NRESULTS) as i32 - 1
    }

    #[inline(always)]
    pub fn ccmt(&self) -> u8 {
        ((self.status & status::CCMT) >> status::CCMT_SHIFT) as u8
    }

    #[inline(always)]
    pub fn tail(&self) -> bool {
        self.status & status::TAIL != 0
    }

    #[inline(always)]
    pub fn hooked(&self) -> bool {
        self.status & status::HOOKED != 0
    }

    #[inline(always)]
    pub fn counted(&self) -> bool {
        self.status & status::COUNTED != 0
    }

    #[inline(always)]
    pub(crate) fn nested(&self) -> bool {
        self.status & status::NESTED != 0
    }

    #[inline(always)]
    fn set_flag(&mut self, flag: u32, on: bool) {
        if on {
            self.status |= flag;
        } else {
            self.status &= !flag;
        }
    }

    pub(crate) fn set_hooked(&mut self, on: bool) {
        self.set_flag(status::HOOKED, on);
    }

    pub(crate) fn set_counted(&mut self, on: bool) {
        self.set_flag(status::COUNTED, on);
    }

    #[inline(always)]
    pub(crate) fn hidden_params(&self) -> usize {
        (self.status >> status::HIDDEN_SHIFT) as usize
    }

    #[inline(always)]
    pub fn nvarargs(&self) -> usize {
        if self.hidden_params() == 0 {
            return 0;
        }
        self.nvarargs as usize
    }

    #[inline(always)]
    pub(crate) fn argument_count(&self) -> usize {
        self.nvarargs as usize
    }

    #[inline(always)]
    pub(crate) fn with_arguments(mut self, count: usize) -> Self {
        debug_assert!(u32::try_from(count).is_ok());
        self.nvarargs = count as u32;
        self
    }

    #[inline(always)]
    pub(crate) fn set_hidden(&mut self, params: u8, nvarargs: usize) {
        debug_assert!(u32::try_from(nvarargs).is_ok());
        self.nvarargs = nvarargs as u32;
        self.status = (self.status & !status::HIDDEN) | ((u32::from(params) + 1) << status::HIDDEN_SHIFT);
    }

    #[inline(always)]
    pub fn ret(&self) -> usize {
        let hidden = self.hidden_params();
        if hidden == 0 {
            return self.base - 1;
        }
        self.base - 1 - self.nvarargs as usize - hidden
    }

    #[inline(always)]
    pub fn pending(&self) -> Pending {
        let operand = self.operand as usize;
        match (self.status & status::PENDING) >> status::PENDING_SHIFT {
            0 => Pending::Idle,
            1 => Pending::Operation,
            2 => Pending::Concat(operand),
            3 => Pending::Return(operand),
            _ => Pending::Closing(operand),
        }
    }

    #[inline(always)]
    pub(crate) fn is_idle(&self) -> bool {
        self.status & status::PENDING == 0
    }

    pub(crate) fn set_pending(&mut self, pending: Pending) {
        let (kind, operand) = match pending {
            Pending::Idle => (0, 0),
            Pending::Operation => (1, 0),
            Pending::Concat(total) => (2, total),
            Pending::Return(first) => (3, first),
            Pending::Closing(top) => (4, top),
        };
        debug_assert!(u32::try_from(operand).is_ok());
        self.status = (self.status & !status::PENDING) | (kind << status::PENDING_SHIFT);
        self.operand = operand as u32;
    }
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
#[repr(C)]
pub struct Transfer {
    pub start: usize,
    pub count: usize,
    pub base: usize,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct NativeFrame {
    pub ret: usize,
    pub nresults: i32,
    pub scratch: usize,
    pub owner: Value,
    pub resume: Resume,
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct YieldFrame {
    pub ret: usize,
    pub nresults: i32,
    pub owner: Value,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CoroutineFrame {
    pub caller: Ref<Thread>,
    pub wrapped: bool,
    pub ret: usize,
    pub nresults: i32,
    pub owner: Value,
    pub floor: usize,
    pub non_yieldable: u32,
    pub c_calls: u32,
    pub counted: bool,
}

/// Compiled code pushes a frame itself, so the layout is `#[repr(C, u8)]`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[repr(C, u8)]
pub enum Frame {
    Lua(LuaFrame),
    Native(NativeFrame),
    Yielded(YieldFrame),
    Coroutine(CoroutineFrame),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum CoStatus {
    #[default]
    Suspended,
    Running,
    Normal,
    Dead,
}

impl Thread {
    pub(crate) fn size(&self) -> usize {
        200 + std::mem::size_of::<Value>() * self.stack.len()
    }

    pub(crate) fn stack_resized(&mut self) {
        self.call_limit = self.stack.len().saturating_sub(crate::vm::REGISTER_SPAN).min(crate::vm::MAX_STACK_SLOTS - CALL_FRAME_SLOTS);
    }
}

/// Covers up to 254 arguments, the moved function of a vararg frame, and up to 255 registers.
const CALL_FRAME_SLOTS: usize = 2 * u8::MAX as usize;

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Thread {
    pub stack: Vec<Value>,
    pub frames: Vec<Frame>,
    pub open_upvalues: Vec<Ref<UpVal>>,
    pub to_close: Vec<usize>,
    pub top: usize,
    pub status: CoStatus,
    pub started: bool,
    pub c_calls: u32,
    pub failure: Option<Value>,
    pub self_close: Option<Value>,
    pub hook: Option<Hook>,
    pub hook_running: bool,
    pub hook_counter: i64,
    pub old_pc: usize,
    /// Follows `L->transferinfo` in lstate.h, for `debug.getinfo`'s `r` option.
    pub transfer: Transfer,
    /// Follows `checkstackp` in ldo.c.
    #[serde(skip)]
    pub(crate) call_limit: usize,
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct Hook {
    pub function: Value,
    pub mask: u8,
    pub base_count: i64,
}

pub const MASK_CALL: u8 = 1;
pub const MASK_RET: u8 = 2;
pub const MASK_LINE: u8 = 4;
pub const MASK_COUNT: u8 = 8;

pub(crate) fn same_body(left: NativeBody, right: NativeBody) -> bool {
    match (left, right) {
        (NativeBody::Plain(a), NativeBody::Plain(b)) => std::ptr::fn_addr_eq(a, b),
        (NativeBody::Stepped(a), NativeBody::Stepped(b)) => std::ptr::fn_addr_eq(a, b),
        (NativeBody::Yield, NativeBody::Yield)
        | (NativeBody::CoroutineResume, NativeBody::CoroutineResume)
        | (NativeBody::CoroutineWrap, NativeBody::CoroutineWrap) => true,
        _ => false,
    }
}
