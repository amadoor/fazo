use std::{io, ptr};

use libc::c_int;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
const PROTECTION: c_int = libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
const FLAGS: c_int = libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_JIT;

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
const PROTECTION: c_int = libc::PROT_READ | libc::PROT_WRITE;
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
const FLAGS: c_int = libc::MAP_PRIVATE | libc::MAP_ANON;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe extern "C" {
    fn pthread_jit_write_protect_np(enabled: c_int);
}

unsafe extern "C" {
    fn __register_frame(frame: *const u8);
    fn __deregister_frame(frame: *const u8);
}

pub(in crate::jit) fn reserve(len: usize) -> io::Result<*mut u8> {
    debug_assert!(len > 0);
    // SAFETY: a null hint with an anonymous mapping gets fresh pages that no other mapping aliases.
    let mapping = unsafe { libc::mmap(ptr::null_mut(), len, PROTECTION, FLAGS, -1, 0) };
    if mapping == libc::MAP_FAILED {
        return Err(io::Error::last_os_error());
    }
    begin_write();
    Ok(mapping.cast())
}

/// # Safety
///
/// `ptr` and `len` must be a mapping from [`reserve`] that no compiled code or registration still refers to.
pub(in crate::jit) unsafe fn release(ptr: *mut u8, len: usize) {
    debug_assert!(!ptr.is_null() && len > 0);
    // SAFETY: the caller owns the mapping and gives it up here.
    let status = unsafe { libc::munmap(ptr.cast(), len) };
    debug_assert_eq!(status, 0);
}

/// # Safety
///
/// `ptr` and `len` must be a mapping from [`reserve`] whose bytes are fully written and never written again.
pub(in crate::jit) unsafe fn make_executable(ptr: *mut u8, len: usize) -> io::Result<()> {
    debug_assert!(!ptr.is_null() && len > 0);
    end_write();
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        return Ok(());
    }
    // SAFETY: the caller owns the range; the call drops write permission as it adds execute permission.
    let status = unsafe { libc::mprotect(ptr.cast(), len, libc::PROT_READ | libc::PROT_EXEC) };
    if status != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn begin_write() {
    // SAFETY: the call only changes the calling thread's write-protect state for MAP_JIT mappings.
    unsafe { pthread_jit_write_protect_np(0) };
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(in crate::jit) fn end_write() {
    // SAFETY: the call only changes the calling thread's write-protect state for MAP_JIT mappings.
    unsafe { pthread_jit_write_protect_np(1) };
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn begin_write() {}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
pub(in crate::jit) fn end_write() {}

pub(in crate::jit) struct UnwindRegistration {
    frames: Vec<*const u8>,
}

/// # Safety
///
/// `eh_frame` must be a CIE-then-FDE sequence ending in a zero word, and every pointer in `fdes` must address an FDE of it.
/// Both must stay readable until the registration is dropped.
pub(in crate::jit) unsafe fn register_eh_frame(
    eh_frame: *const u8,
    fdes: &[*const u8],
) -> io::Result<UnwindRegistration> {
    let frames = if cfg!(target_os = "macos") {
        fdes.to_vec()
    } else {
        vec![eh_frame]
    };
    for &frame in &frames {
        debug_assert!(!frame.is_null());
        // SAFETY: the caller's contract makes every frame an FDE or a valid sequence start, and the registration keeps the bytes alive.
        unsafe { __register_frame(frame) };
    }
    Ok(UnwindRegistration { frames })
}

/// # Safety
///
/// Never called on Unix: Unix unwind data is always an eh_frame sequence.
pub(in crate::jit) unsafe fn register_function_table(
    _table: *const u8,
    _base: *const u8,
) -> io::Result<UnwindRegistration> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "function-table registration is not available on this platform",
    ))
}

impl Drop for UnwindRegistration {
    fn drop(&mut self) {
        for &frame in self.frames.iter().rev() {
            debug_assert!(!frame.is_null());
            // SAFETY: each frame was registered by `register_eh_frame`, and the registration owns the bytes it names.
            unsafe { __deregister_frame(frame) };
        }
    }
}
