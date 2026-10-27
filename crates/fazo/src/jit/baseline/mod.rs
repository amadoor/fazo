//! The Baseline tier, modeled on V8's Sparkplug. Each instruction's template is assembled once per VM and copied per compile,
//! so frames and calls match the interpreter's and the Cranelift tier's.

#[cfg(any(target_arch = "aarch64", test))]
pub(crate) mod a64;
mod lower;
pub(crate) mod masm;
pub(crate) mod templates;
mod unwind;
#[cfg(any(target_arch = "x86_64", test))]
pub(crate) mod x64;

#[cfg(test)]
mod tests;

use std::sync::Arc;

use self::{
    masm::{Masm, SplitStencil},
    templates::{Env, HelperAddresses, Key},
};
use super::{CodeBudget, FrameLayout, JitCtx, TableLayout, helpers, sys};

pub(crate) use lower::BaselineFacts;

struct Frame {
    prologue: Vec<u8>,
    unwind: unwind::Prologue,
    dispatch: SplitStencil,
    negate: Vec<u8>,
    epilogue: Vec<u8>,
}

pub(crate) struct Library {
    env: Env,
    frame: Frame,
    stencils: Vec<Option<SplitStencil>>,
}

pub(crate) fn host_masm() -> Box<dyn Masm> {
    #[cfg(target_arch = "x86_64")]
    return Box::new(x64::X64::new());
    #[cfg(target_arch = "aarch64")]
    return Box::new(a64::A64::new());
}

fn host_frame() -> Frame {
    let registers = std::mem::offset_of!(JitCtx, registers) as i32;
    let constants = std::mem::offset_of!(JitCtx, constants) as i32;
    let exit = masm::ext::EXIT;
    let table = masm::ext::TARGET;
    #[cfg(target_arch = "x86_64")]
    {
        let reserved = x64::frame::reserved(templates::CALL_AREA);
        let prologue = x64::frame::prologue(reserved, registers, constants);
        let pushes = std::array::from_fn(|index| (u32::from(prologue.pushes[index]), x64::frame::SAVED[index] as u8));
        Frame {
            unwind: unwind::Prologue::X64 {
                push_rbp: u32::from(prologue.push_rbp),
                set_rbp: u32::from(prologue.set_rbp),
                pushes,
                alloc: u32::from(prologue.alloc),
                reserved,
            },
            prologue: prologue.bytes,
            dispatch: x64::frame::dispatch(exit, table).split(),
            negate: x64::frame::negate_exit(),
            epilogue: x64::frame::epilogue(reserved),
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        let reserved = a64::frame::reserved(templates::CALL_AREA);
        let prologue = a64::frame::prologue(reserved, registers, constants);
        Frame {
            unwind: unwind::Prologue::A64 {
                save_frame: prologue.save_frame,
                set_frame: prologue.set_frame,
                pairs: prologue.pairs,
                reserved,
            },
            prologue: prologue.bytes,
            dispatch: a64::frame::dispatch(exit, table).split(),
            negate: a64::frame::negate_exit(),
            epilogue: a64::frame::epilogue(reserved),
        }
    }
}

impl Library {
    pub(crate) fn new(table: Option<TableLayout>, frames: Option<FrameLayout>) -> Self {
        let fast = super::sites::fast_symbols().map(|(_, address)| address as u64);
        let helpers = HelperAddresses {
            step: helpers::step as extern "C" fn(*mut JitCtx, u32) -> i64 as usize as u64,
            safe_point: helpers::safe_point as extern "C" fn(*mut JitCtx, u32) -> i64 as usize as u64,
            misc: helpers::misc as extern "C" fn(*mut JitCtx, u32) -> i64 as usize as u64,
            tier_check: helpers::tier_check as extern "C" fn(*mut JitCtx, u32) -> i64 as usize as u64,
            fast,
        };
        Self {
            env: Env { table, frames, helpers },
            frame: host_frame(),
            stencils: vec![None; templates::KEY_COUNT],
        }
    }

    pub(crate) fn has_tables(&self) -> bool {
        self.env.table.is_some()
    }

    pub(crate) fn has_frames(&self) -> bool {
        self.env.frames.is_some()
    }

    fn stencil(&mut self, key: Key) -> &SplitStencil {
        let Self { env, stencils, .. } = self;
        stencils[key.index()].get_or_insert_with(|| {
            let mut masm = host_masm();
            templates::build(key, masm.as_mut(), env);
            masm.finish().split()
        })
    }
}

/// Dropping the unit unmaps the code, so the function must not be called afterwards.
pub(crate) struct BaselineUnit {
    _unwind: Option<sys::UnwindRegistration>,
    start: *mut u8,
    len: usize,
    budget: Arc<CodeBudget>,
}

// SAFETY: the unit owns its mapping, and the pointer names only memory this unit maps.
unsafe impl Send for BaselineUnit {}

impl BaselineUnit {
    pub(crate) fn entry(&self) -> *const u8 {
        self.start
    }

    pub(crate) fn install(code: &[u8], trailer: Option<super::unwind::Trailer>, budget: Arc<CodeBudget>) -> Option<Self> {
        let len = code.len().max(1).next_multiple_of(PAGE);
        budget.reserve(len).ok()?;
        let Ok(start) = sys::reserve(len) else {
            budget.release(len);
            return None;
        };
        debug_assert!(!start.is_null() && code.len() <= len);
        // SAFETY: `start` is a fresh writable mapping of `len` bytes, and `code` fits in it.
        unsafe { std::ptr::copy_nonoverlapping(code.as_ptr(), start, code.len()) };
        let mut unit = Self {
            _unwind: None,
            start,
            len,
            budget,
        };
        debug_assert!(!unit.start.is_null());
        // SAFETY: the mapping's bytes are complete and are not written again.
        unsafe { sys::make_executable(unit.start, unit.len) }.ok()?;
        debug_assert!(!unit.start.is_null());
        // SAFETY: the range is the mapping just made executable.
        unsafe { wasmtime_jit_icache_coherence::clear_cache(unit.start.cast(), unit.len) }.ok()?;
        wasmtime_jit_icache_coherence::pipeline_flush_mt().ok()?;
        if let Some(table) = trailer.and_then(|trailer| trailer.table) {
            // SAFETY: the trailer's offsets lie inside `code`, which the unit keeps mapped until the registration is dropped.
            unit._unwind = Some(unsafe { super::unwind::register(&table, unit.start) }.ok()?);
        }
        Some(unit)
    }
}

const PAGE: usize = 16 * 1024;

impl Drop for BaselineUnit {
    fn drop(&mut self) {
        self._unwind = None;
        sys::end_write();
        debug_assert!(!self.start.is_null() && self.len > 0);
        // SAFETY: the mapping came from `sys::reserve` and only this unit reaches it; its unwind registration is gone.
        unsafe { sys::release(self.start, self.len) };
        self.budget.release(self.len);
    }
}

pub(crate) fn compile(library: &mut Library, budget: Arc<CodeBudget>, proto: &crate::proto::Proto, facts: &BaselineFacts<'_>) -> Option<BaselineUnit> {
    let (code, trailer) = lower::emit(library, proto, facts)?;
    BaselineUnit::install(&code, trailer, budget)
}
