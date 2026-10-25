#![expect(
    unsafe_code,
    reason = "lua.c loads the readline library at run time and calls it through C function pointers"
)]

use std::ffi::{CStr, CString, c_char, c_void};

type ReadlineFunction = unsafe extern "C" fn(*const c_char) -> *mut c_char;
type AddHistoryFunction = unsafe extern "C" fn(*const c_char);

const READLINE: &CStr = c"readline";
const ADD_HISTORY: &CStr = c"add_history";
const READLINE_NAME: &CStr = c"rl_readline_name";
const PROGRAM_NAME: &CStr = c"lua";

/// Stays loaded for the process's lifetime, as in lua.c, so both function pointers stay valid.
pub struct Readline {
    readline: ReadlineFunction,
    add_history: Option<AddHistoryFunction>,
}

pub fn load(name: &[u8]) -> Option<Readline> {
    let name = CString::new(name).ok()?;
    // SAFETY: name is a NUL-terminated string that outlives the call, and the flags are valid.
    let library = unsafe { libc::dlopen(name.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
    if library.is_null() {
        return None;
    }

    set_program_name(library);
    let readline = symbol(library, READLINE)?;
    let add_history = symbol(library, ADD_HISTORY);
    // SAFETY: `readline` has the signature `char *readline(const char *prompt)`, and the library stays loaded.
    let readline = unsafe { std::mem::transmute::<*mut c_void, ReadlineFunction>(readline) };
    // SAFETY: readline's `add_history` has the C signature `void add_history(const char *line)`.
    let add_history = add_history.map(|address| unsafe { std::mem::transmute::<*mut c_void, AddHistoryFunction>(address) });
    Some(Readline { readline, add_history })
}

fn symbol(library: *mut c_void, name: &CStr) -> Option<*mut c_void> {
    debug_assert!(!library.is_null());
    // SAFETY: library is a handle dlopen returned and never closes; name is NUL-terminated.
    let address = unsafe { libc::dlsym(library, name.as_ptr()) };
    (!address.is_null()).then_some(address)
}

fn set_program_name(library: *mut c_void) {
    let Some(address) = symbol(library, READLINE_NAME) else {
        return;
    };
    debug_assert!(address.cast::<*const c_char>().is_aligned());
    // SAFETY: `rl_readline_name` is a `const char *` variable, and PROGRAM_NAME lives for the whole process.
    unsafe { address.cast::<*const c_char>().write(PROGRAM_NAME.as_ptr()) };
}

impl Readline {
    /// Returns `None` at the end of input.
    pub fn read(&self, prompt: &[u8]) -> Option<Vec<u8>> {
        let prompt = c_text(prompt);
        // SAFETY: prompt is a NUL-terminated string that outlives the call.
        let line = unsafe { (self.readline)(prompt.as_ptr()) };
        flush_c_streams();
        if line.is_null() {
            return None;
        }

        // SAFETY: readline returns a NUL-terminated string from malloc that the caller owns.
        let bytes = unsafe { CStr::from_ptr(line) }.to_bytes().to_vec();
        // SAFETY: line came from readline's malloc and is freed once, after its bytes are copied.
        unsafe { libc::free(line.cast()) };
        Some(bytes)
    }

    pub fn save(&self, line: &[u8]) {
        let Some(add_history) = self.add_history else {
            return;
        };

        let line = c_text(line);
        // SAFETY: line is a NUL-terminated string; add_history copies it.
        unsafe { add_history(line.as_ptr()) };
    }
}

fn c_text(bytes: &[u8]) -> CString {
    let end = bytes.iter().position(|&byte| byte == 0).unwrap_or(bytes.len());
    CString::new(&bytes[..end]).unwrap_or_default()
}

fn flush_c_streams() {
    // SAFETY: fflush with a null stream flushes every C output stream and touches no Rust memory.
    unsafe { libc::fflush(std::ptr::null_mut()) };
}
