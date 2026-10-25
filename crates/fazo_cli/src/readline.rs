#[cfg(any(target_os = "linux", target_os = "macos"))]
mod library;

#[cfg(any(target_os = "linux", target_os = "macos"))]
pub use library::Readline;

use crate::session::Session;

#[cfg(target_os = "linux")]
const DEFAULT_LIBRARY: &[u8] = b"libreadline.so";
#[cfg(target_os = "linux")]
const LIBRARY_VARIABLE: &str = "LUA_READLINELIB";

#[cfg(target_os = "linux")]
pub fn init(session: &mut Session, environment: bool) -> Option<Readline> {
    let name = environment
        .then(|| std::env::var_os(LIBRARY_VARIABLE))
        .flatten()
        .map_or_else(|| DEFAULT_LIBRARY.to_vec(), |value| crate::platform::argument_bytes(&value));
    let readline = library::load(&name);
    if readline.is_some() {
        return readline;
    }

    let vm = session.vm();
    vm.warning(b"unable to load readline library '", true);
    vm.warning(&name, true);
    vm.warning(b"'", false);
    None
}

/// A `LUA_USE_MACOSX` build links the system readline, so it reads no variable and never warns.
#[cfg(target_os = "macos")]
pub fn init(_session: &mut Session, _environment: bool) -> Option<Readline> {
    library::load(b"libreadline.dylib")
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub enum Readline {}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
impl Readline {
    pub fn read(&self, _prompt: &[u8]) -> Option<Vec<u8>> {
        match *self {}
    }

    pub fn save(&self, _line: &[u8]) {
        match *self {}
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub fn init(_session: &mut Session, _environment: bool) -> Option<Readline> {
    None
}
