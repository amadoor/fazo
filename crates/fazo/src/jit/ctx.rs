use std::sync::atomic::AtomicU8;

use crate::{LuaError, Vm, arena::Ref, object::{Captured, Closure, Thread}, proto::Proto, table::Table, value::Value};

/// `#[repr(C)]`: compiled code reads the fields at fixed offsets. The register, table, mark and thread
/// pointers are refreshed after every helper that can move their storage.
#[repr(C)]
pub(crate) struct JitCtx {
    pub(crate) requests: *const AtomicU8,
    pub(crate) registers: *mut Value,
    pub(crate) constants: *const Value,
    pub(crate) tables: *const Table,
    pub(crate) table_marks: *const u8,
    pub(crate) closures: *const Closure,
    /// Stable for the closure's lifetime.
    pub(crate) upvalues: *const Captured,
    pub(crate) vm: *mut Vm,
    pub(crate) closure: Ref<Closure>,
    pub(crate) base: usize,
    pub(crate) proto: Ref<Proto>,
    /// Results the compiled caller takes from this frame, or `NOT_DIRECT` for a frame the driver entered.
    pub(crate) direct_results: i32,
    pub(crate) ret_offset: i64,
    pub(crate) thread: *mut Thread,
    pub(crate) thread_ref: Ref<Thread>,
    pub(crate) call_depth: *mut u32,
    /// Error slot shared by every frame of one nest; `EXIT_ERROR` writes into it.
    pub(crate) error: *mut Option<LuaError>,
}

pub(crate) const NOT_DIRECT: i32 = -2;

impl JitCtx {
    pub(crate) fn fail(&mut self, error: LuaError) {
        debug_assert!(!self.error.is_null());
        // SAFETY: `error` is the slot of the driver call that entered the outermost compiled frame, which
        // outlives the nest. Compiled code runs on one thread and nothing else touches the slot meanwhile.
        unsafe { *self.error = Some(error) }
    }
}
