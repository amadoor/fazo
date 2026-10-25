use std::{ffi::c_void, io, ptr};

use windows_sys::Win32::System::{
    Diagnostics::Debug::{RtlAddFunctionTable, RtlDeleteFunctionTable},
    Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_EXECUTE_READ, PAGE_READWRITE, VirtualAlloc,
        VirtualFree, VirtualProtect,
    },
};

pub(in crate::jit) fn reserve(len: usize) -> io::Result<*mut u8> {
    debug_assert!(len > 0);
    // SAFETY: a null base asks for fresh committed pages that no allocation aliases.
    let mapping = unsafe { VirtualAlloc(ptr::null(), len, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) };
    if mapping.is_null() {
        return Err(io::Error::last_os_error());
    }
    Ok(mapping.cast())
}

/// # Safety
///
/// `ptr` must be the base of a region from [`reserve`] that no compiled code or unwind registration still refers to.
pub(in crate::jit) unsafe fn release(ptr: *mut u8, len: usize) {
    debug_assert!(!ptr.is_null() && len > 0);
    // SAFETY: the caller owns the region and gives it up here.
    let status = unsafe { VirtualFree(ptr.cast::<c_void>(), 0, MEM_RELEASE) };
    debug_assert!(status != 0);
}

/// # Safety
///
/// `ptr` and `len` must describe a region from [`reserve`] whose bytes are fully written and never written again.
pub(in crate::jit) unsafe fn make_executable(ptr: *mut u8, len: usize) -> io::Result<()> {
    debug_assert!(!ptr.is_null() && len > 0);
    let mut previous = 0;
    // SAFETY: the caller owns the region; the protection goes from read-write to read-execute, so pages are never both.
    let status = unsafe {
        VirtualProtect(ptr.cast::<c_void>(), len, PAGE_EXECUTE_READ, &mut previous)
    };
    if status == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

pub(in crate::jit) fn end_write() {}

pub(in crate::jit) struct UnwindRegistration {
    table: *const u8,
}

/// # Safety
///
/// `table` must point to one RUNTIME_FUNCTION entry (IMAGE_RUNTIME_FUNCTION_ENTRY on x86-64, IMAGE_ARM64_RUNTIME_FUNCTION_ENTRY on Arm64)
/// with offsets from `base`. The table and the unwind data it names must stay readable until the registration is dropped.
pub(in crate::jit) unsafe fn register_function_table(
    table: *const u8,
    base: *const u8,
) -> io::Result<UnwindRegistration> {
    debug_assert!(!table.is_null() && !base.is_null());
    // SAFETY: the table and base stay valid for the registration's lifetime, which removes the table in `drop`.
    let registered = unsafe { RtlAddFunctionTable(table.cast(), 1, base as _) };
    if !registered {
        return Err(io::Error::last_os_error());
    }
    Ok(UnwindRegistration { table })
}

/// # Safety
///
/// Never called on Windows: Windows unwind data is always a function table.
pub(in crate::jit) unsafe fn register_eh_frame(
    _eh_frame: *const u8,
    _fdes: &[*const u8],
) -> io::Result<UnwindRegistration> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "eh_frame registration is not available on Windows",
    ))
}

impl Drop for UnwindRegistration {
    fn drop(&mut self) {
        // SAFETY: the table was added by `register_function_table`, and the same unit owns the memory it names.
        let removed = unsafe { RtlDeleteFunctionTable(self.table.cast()) };
        debug_assert!(removed);
    }
}
