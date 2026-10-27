use std::sync::atomic::AtomicU8;

use crate::{LuaError, Vm, arena::Ref, object::{Captured, Closure, Thread}, proto::Proto, table::Table, value::Value};

/// `#[repr(C)]`: compiled code reads the fields at fixed offsets. The register, table, mark and thread
/// pointers are refreshed after every helper that can move their storage.
#[repr(C)]
pub(crate) struct JitCtx {
    /// The fields up to `moves` are the same in every frame of one nest, so a callee's context copies them in pairs.
    pub(crate) requests: *const AtomicU8,
    pub(crate) vm: *mut Vm,
    pub(crate) tables: *const Table,
    pub(crate) table_marks: *const u8,
    pub(crate) closures: *const Closure,
    pub(crate) thread: *mut Thread,
    pub(crate) thread_ref: Ref<Thread>,
    /// Native stack floor that compiled calls check before recursing (V8 `StackCheck`).
    pub(crate) stack_limit: usize,
    /// Error slot shared by every frame of one nest; `EXIT_ERROR` writes into it.
    pub(crate) error: *mut Option<LuaError>,
    pub(crate) moves: u64,
    pub(crate) registers: *mut Value,
    pub(crate) constants: *const Value,
    /// Stable for the closure's lifetime.
    pub(crate) upvalues: *const Captured,
    pub(crate) closure: Ref<Closure>,
    pub(crate) base: usize,
    pub(crate) proto: Ref<Proto>,
    /// Results the compiled caller takes from this frame, or `NOT_DIRECT` for a frame the driver entered.
    pub(crate) direct_results: i32,
    pub(crate) ret_offset: i64,
    /// Caller context an optimized direct call filled; such callers skip the refresh after the call.
    pub(crate) parent: *mut JitCtx,
    pub(crate) child: *mut JitCtx,
    /// `PENDING` while the frame lives only on the native stack until a helper pushes it (LuaJIT lj_frame.h).
    pub(crate) pending: u32,
    pub(crate) saved_pc: u32,
    pub(crate) frame_index: u32,
    /// Valid only while `saved_pc` is `INLINED_CALL`.
    pub(crate) inline_point: *const crate::jit::ssa::DeoptPoint,
}

pub(crate) const PENDING: u32 = 1;
pub(crate) const PENDING_TAIL: u32 = 2;

pub(crate) const INLINED_CALL: u32 = u32::MAX;

pub(crate) const NOT_DIRECT: i32 = -2;

impl JitCtx {
    pub(crate) fn fail(&mut self, error: LuaError) {
        debug_assert!(!self.error.is_null());
        // SAFETY: `error` is the slot of the driver call that entered the outermost compiled frame, which
        // outlives the nest. Compiled code runs on one thread and nothing else touches the slot meanwhile.
        unsafe { *self.error = Some(error) }
    }
}

impl JitCtx {
    pub(crate) fn pending_caller(&mut self) -> &mut JitCtx {
        // SAFETY: a pending frame's `parent` is its compiled caller's context, live deeper in the native stack while it
        // runs and distinct from `self`.
        unsafe { self.parent.as_mut() }.expect("a pending frame has a compiled caller")
    }

    pub(crate) fn inline_point(&self) -> Option<&crate::jit::ssa::DeoptPoint> {
        if self.saved_pc != INLINED_CALL {
            return None;
        }
        // SAFETY: the call wrote the point, which belongs to the unit running this frame and stays mapped with its points.
        unsafe { self.inline_point.as_ref() }
    }

    pub(crate) fn refresh_nest(&mut self, refresh: impl Fn(&mut JitCtx)) {
        // SAFETY: `child` is null or the call area in this activation's native frame, and every `parent` is a caller's
        // context deeper in the native stack, all live while this frame runs and none the same as `self`.
        unsafe {
            if let Some(child) = self.child.as_mut() {
                refresh(child);
            }
            let mut parent = self.parent;
            while let Some(caller) = parent.as_mut() {
                refresh(caller);
                parent = caller.parent;
            }
        }
    }
}
