use std::{
    fmt,
    sync::{
        Arc,
        atomic::{AtomicU8, Ordering},
    },
};

use serde::{Deserialize, Serialize};

use crate::{
    Host, LuaError, Options, Value, Values, Vm,
    arena::Ref,
    heap::Heap,
    object::Thread,
    stdlib::{base::WarningState, io::{Stream, StreamRecord}},
    table::Table,
    vm::{CallRequest, Operand, RunScope},
};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub(crate) struct Suspended {
    pub(crate) thread: Ref<Thread>,
    pub(crate) stop: usize,
    pub(crate) function_slot: usize,
    pub(crate) floor: usize,
}

/// A call's results, or a pause at an instruction boundary that [`Vm::resume`] continues.
#[derive(Debug)]
pub enum Outcome {
    Finished(Values),
    Paused,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SnapshotError {
    /// A host file is still open; a snapshot cannot hold a host file handle.
    OpenFile,
    /// Not at an instruction boundary, because a native is running.
    NativeRunning,
    /// The snapshot uses a native function the restoring host does not provide.
    MissingNative(String),
    /// The bytes do not start with `FAZO` and this build's format version.
    Header { magic: Vec<u8>, version: Option<u32> },
    Format(String),
}

const MAGIC: &[u8; 4] = b"FAZO";
pub const SNAPSHOT_VERSION: u32 = 1;
const HEADER_LEN: usize = 16;

impl fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SnapshotError::OpenFile => write!(f, "cannot snapshot the VM while a file it opened is open"),
            SnapshotError::NativeRunning => write!(f, "cannot snapshot the VM while a native function is running"),
            SnapshotError::MissingNative(name) => {
                write!(f, "the snapshot uses the native function `{name}`, which the restoring host does not provide")
            }
            SnapshotError::Header { magic, version } => {
                write!(f, "the snapshot starts with \"{}\"", magic.escape_ascii())?;
                match version {
                    Some(version) => write!(f, " and format version {version}")?,
                    None => write!(f, " and has no format version")?,
                }
                write!(f, "; this build reads \"FAZO\" version {SNAPSHOT_VERSION}")
            }
            SnapshotError::Format(message) => write!(f, "the snapshot does not decode: {message}"),
        }
    }
}

impl std::error::Error for SnapshotError {}

/// The options that change what Lua code sees; the rest belong to the restoring host.
#[derive(Serialize, Deserialize)]
struct StateOptions {
    os: bool,
    io: bool,
    environment_variables: bool,
    package: bool,
    random_seed: Option<i64>,
    memory_limit: Option<usize>,
    jit_library: bool,
}

impl StateOptions {
    fn of(options: &Options) -> Self {
        Self {
            os: options.os,
            io: options.io,
            environment_variables: options.environment_variables,
            package: options.package,
            random_seed: options.random_seed,
            memory_limit: options.memory_limit,
            jit_library: options.jit_library,
        }
    }

    fn applied_to(&self, host: Options) -> Options {
        Options {
            os: self.os,
            io: self.io,
            environment_variables: self.environment_variables,
            package: self.package,
            random_seed: self.random_seed,
            memory_limit: self.memory_limit,
            jit_library: self.jit_library,
            ..host
        }
    }
}

#[derive(Serialize, Deserialize)]
struct Image {
    options: StateOptions,
    natives: Vec<(String, usize)>,
    heap: Heap,
    globals: Ref<Table>,
    registry: Ref<Table>,
    main_thread: Ref<Thread>,
    current_thread: Ref<Thread>,
    tracing: bool,
    hidden: Vec<(String, Value)>,
    string_metatable: Option<Ref<Table>>,
    floor: usize,
    c_calls: u32,
    non_yieldable: u32,
    handling_error: u32,
    yielded: Values,
    type_metatables: [Option<Ref<Table>>; 6],
    random: [u64; 4],
    warnings: WarningState,
    resources: Vec<StreamRecord>,
    pins: Vec<Value>,
    suspended: Option<Suspended>,
}

pub(crate) const PAUSE: u8 = 1;
pub(crate) const INTERRUPT: u8 = 2;
/// Charged at the first back-edge, since a charge at `FORPREP` would compile before the body's sites have feedback.
pub(crate) const LOOP_CHARGE: u8 = 4;

/// A request on a [`Vm`] that any thread can set; clones share it.
#[derive(Clone)]
pub struct RequestHandle {
    flags: Arc<AtomicU8>,
    bit: u8,
}

impl RequestHandle {
    pub fn request(&self) {
        self.flags.fetch_or(self.bit, Ordering::Relaxed);
    }

    pub fn clear(&self) {
        self.flags.fetch_and(!self.bit, Ordering::Relaxed);
    }

    pub fn is_requested(&self) -> bool {
        self.flags.load(Ordering::Relaxed) & self.bit != 0
    }
}

impl Vm {
    /// Pauses at the outermost call's next instruction boundary; a native in progress finishes first.
    pub fn pause_handle(&self) -> RequestHandle {
        RequestHandle {
            flags: Arc::clone(&self.requests),
            bit: PAUSE,
        }
    }

    pub fn request_pause(&self) {
        self.pause_handle().request();
    }

    /// Stops the running call with an error `pcall` cannot catch; the call is not resumable.
    pub fn interrupt_handle(&self) -> RequestHandle {
        RequestHandle {
            flags: Arc::clone(&self.requests),
            bit: INTERRUPT,
        }
    }

    pub fn request_interrupt(&self) {
        self.interrupt_handle().request();
    }

    pub fn start(&mut self, function: Value, arguments: &[Value]) -> Result<Outcome, LuaError> {
        if self.suspended.is_some() {
            return Err(self.error_string("the VM is paused; resume it before starting another call"));
        }
        let function_slot = self.free_slot();
        let stop = self.frame_count();
        self.enter_c_call()?;
        self.non_yieldable += 1;
        let floor = std::mem::replace(&mut self.floor, stop);
        self.place_call(function_slot, function, arguments);
        self.thread_mut().top = function_slot + 1 + arguments.len();
        let request = CallRequest {
            function_slot,
            argument_count: arguments.len(),
            nresults: -1,
            ccmt: 0,
            operand: Operand::Unknown,
        };
        let thread = self.current_thread;
        let suspended = Suspended {
            thread,
            stop,
            function_slot,
            floor,
        };
        let outcome = self.run_pausable(RunScope { thread, frames: stop, return_on_switch: false }, Some(request));
        self.finish(suspended, outcome)
    }

    pub fn resume(&mut self) -> Result<Outcome, LuaError> {
        let Some(suspended) = self.suspended.take() else {
            return Err(self.error_string("the VM is not paused"));
        };
        let scope = RunScope {
            thread: suspended.thread,
            frames: suspended.stop,
            return_on_switch: false,
        };
        let outcome = self.run_pausable(scope, None);
        self.finish(suspended, outcome)
    }

    /// Returns `None` when the VM is not paused.
    pub fn cancel(&mut self, error: Value) -> Option<LuaError> {
        let suspended = self.suspended.take()?;
        self.finish(suspended, Err(LuaError::from_value(error))).err()
    }

    /// Fails while a host file is open, since host files cannot be stored.
    pub fn snapshot(&self) -> Result<Vec<u8>, SnapshotError> {
        if self.current_native.is_some() {
            return Err(SnapshotError::NativeRunning);
        }
        let resources = self
            .resources
            .iter()
            .map(|stream: &Stream| stream.to_record().map_err(|_| SnapshotError::OpenFile))
            .collect::<Result<Vec<_>, _>>()?;
        let image = Image {
            options: StateOptions::of(&self.options),
            natives: self.native_names(),
            heap: self.heap.clone(),
            globals: self.globals,
            registry: self.registry,
            main_thread: self.main_thread,
            current_thread: self.current_thread,
            tracing: self.tracing,
            hidden: self.hidden.clone(),
            string_metatable: self.string_metatable,
            floor: self.floor,
            c_calls: self.c_calls,
            non_yieldable: self.non_yieldable,
            handling_error: self.handling_error,
            yielded: self.yielded.clone(),
            type_metatables: self.type_metatables,
            random: self.random,
            warnings: self.warnings,
            resources,
            pins: self.pins.clone(),
            suspended: self.suspended,
        };
        let mut bytes = Vec::with_capacity(HEADER_LEN);
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&SNAPSHOT_VERSION.to_le_bytes());
        bytes.resize(HEADER_LEN, 0);
        rmp_serde::encode::write(&mut bytes, &image).map_err(|error| SnapshotError::Format(error.to_string()))?;
        Ok(bytes)
    }

    /// The host must register every native the snapshot uses under the same name; a paused VM restores paused.
    /// Options that change what Lua code sees come from the snapshot; the tier and JIT settings come from `options`.
    pub fn restore(bytes: &[u8], options: Options, host: Host) -> Result<Vm, SnapshotError> {
        let body = snapshot_body(bytes)?;
        let image: Image = rmp_serde::from_slice(body).map_err(|error| SnapshotError::Format(error.to_string()))?;
        let mut vm = Vm::with_host(image.options.applied_to(options), host);
        let remap = image
            .natives
            .iter()
            .map(|(name, occurrence)| {
                vm.native_index
                    .get(name.as_str())
                    .and_then(|ids| ids.get(*occurrence))
                    .copied()
                    .ok_or_else(|| SnapshotError::MissingNative(name.clone()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut heap = image.heap;
        heap.rebuild_short_strings();
        heap.rebuild_transitions();
        heap.rebuild_feedback();
        heap.rebuild_code();
        heap.rebuild_upvalues();
        for native in heap.natives.iter_mut() {
            native.function = *remap
                .get(native.function.0 as usize)
                .ok_or_else(|| SnapshotError::Format("a native function index is out of range".to_owned()))?;
            native.builtin = vm.natives[native.function.0 as usize].builtin;
        }
        vm.heap = heap;
        vm.globals = image.globals;
        vm.registry = image.registry;
        vm.main_thread = image.main_thread;
        vm.current_thread = image.current_thread;
        vm.tracing = image.tracing;
        vm.hidden = image.hidden;
        vm.string_metatable = image.string_metatable;
        vm.floor = image.floor;
        vm.c_calls = image.c_calls;
        vm.non_yieldable = image.non_yieldable;
        vm.handling_error = image.handling_error;
        vm.yielded = image.yielded;
        vm.type_metatables = image.type_metatables;
        vm.random = image.random;
        vm.warnings = image.warnings;
        vm.resources = image.resources.into_iter().map(Stream::from_record).collect();
        vm.pins = image.pins;
        vm.suspended = image.suspended;
        vm.verify_image().map_err(|reason| SnapshotError::Format(reason.to_owned()))?;
        Ok(vm)
    }

    fn run_pausable(&mut self, scope: RunScope, first: Option<CallRequest>) -> Result<(), LuaError> {
        let previous = std::mem::replace(&mut self.pausable, true);
        self.pause_holdoff = true;
        let outcome = self.run(scope, first);
        self.pausable = previous;
        outcome
    }

    fn finish(&mut self, suspended: Suspended, outcome: Result<(), LuaError>) -> Result<Outcome, LuaError> {
        match outcome {
            Ok(()) => {
                let top = self.thread().top;
                let results = Values::from_slice(&self.thread().stack[suspended.function_slot..top]);
                self.complete(suspended);
                Ok(Outcome::Finished(results))
            }
            Err(error) if error.is_pause() => {
                self.suspended = Some(suspended);
                Ok(Outcome::Paused)
            }
            Err(error) => {
                let error = self.handle_at_raise(error);
                self.unwind(suspended.stop, suspended.function_slot);
                self.complete(suspended);
                Err(error)
            }
        }
    }

    fn complete(&mut self, suspended: Suspended) {
        self.floor = suspended.floor;
        self.non_yieldable -= 1;
        self.leave_c_call();
        self.thread_mut().top = suspended.function_slot;
    }

    /// Taking the request clears it, so the resumed call runs on.
    pub(crate) fn take_pause_request(&self) -> bool {
        self.pausable && self.run_depth == 1 && take_request(&self.requests, PAUSE)
    }
}

fn snapshot_body(bytes: &[u8]) -> Result<&[u8], SnapshotError> {
    let magic = &bytes[..bytes.len().min(MAGIC.len())];
    let version = bytes
        .get(MAGIC.len()..MAGIC.len() + 4)
        .map(|field| u32::from_le_bytes([field[0], field[1], field[2], field[3]]));
    if magic != MAGIC || version != Some(SNAPSHOT_VERSION) {
        return Err(SnapshotError::Header { magic: magic.to_vec(), version });
    }
    bytes
        .get(HEADER_LEN..)
        .ok_or_else(|| SnapshotError::Format("the snapshot ends inside its header".to_owned()))
}

pub(crate) fn take_request(flags: &AtomicU8, bit: u8) -> bool {

    flags.load(Ordering::Relaxed) & bit != 0 && {
        flags.fetch_and(!bit, Ordering::Relaxed);
        true
    }
}

impl Vm {
    /// Libraries reuse names such as `write`, so the name alone does not identify a native.
    fn native_names(&self) -> Vec<(String, usize)> {
        self.natives
            .iter()
            .enumerate()
            .map(|(index, entry)| {
                let occurrence = self.native_index[entry.name.as_ref()]
                    .iter()
                    .position(|id| id.0 as usize == index)
                    .unwrap_or(0);
                (entry.name.to_string(), occurrence)
            })
            .collect()
    }
}

#[cfg(test)]
mod pause_tests {
    use crate::{Options, Outcome, Value, Vm};

    #[test]
    fn a_pause_requested_before_every_resume_still_lets_the_call_finish() {
        let mut vm = Vm::new(Options::default());
        let function = vm
            .load(b"=t", b"local n = 0 for i = 1, 1000 do n = n + i end return n")
            .expect("the chunk compiles");
        let pause = vm.pause_handle();
        pause.request();
        let mut outcome = vm.start(function, &[]).expect("the call starts");
        let mut resumes = 0;
        while let Outcome::Paused = outcome {
            resumes += 1;
            assert!(resumes < 10_000, "the call never progresses");
            pause.request();
            outcome = vm.resume().expect("the call resumes");
        }
        let Outcome::Finished(values) = outcome else { unreachable!("the loop ends on a finished call") };
        assert!(matches!(values.first(), Some(Value::Integer(500_500))));
    }
}

#[cfg(test)]
mod header_tests {
    use crate::{Host, Options, SnapshotError, Value, Vm};

    fn snapshot() -> Vec<u8> {
        let mut vm = Vm::new(Options::default());
        vm.run_source(b"=t", b"kept = 42").expect("the chunk runs");
        vm.snapshot().expect("an idle VM snapshots")
    }

    #[test]
    fn a_snapshot_starts_with_the_magic_the_version_and_zeroed_reserved_bytes() {
        let bytes = snapshot();
        assert_eq!(&bytes[..16], b"FAZO\x01\0\0\0\0\0\0\0\0\0\0\0");
    }

    #[test]
    fn restore_ignores_the_reserved_bytes() {
        let mut bytes = snapshot();
        bytes[8..16].fill(0xff);
        let mut restored = Vm::restore(&bytes, Options::default(), Host::captured()).expect("the reserved bytes are ignored");
        assert!(matches!(restored.global("kept"), Value::Integer(42)));
    }

    #[test]
    fn restore_names_the_magic_and_version_it_found() {
        let mut bytes = snapshot();
        bytes[4] = 2;
        let Err(error) = Vm::restore(&bytes, Options::default(), Host::captured()) else {
            panic!("version 2 must not restore");
        };
        assert_eq!(error, SnapshotError::Header { magic: b"FAZO".to_vec(), version: Some(2) });
        assert_eq!(error.to_string(), "the snapshot starts with \"FAZO\" and format version 2; this build reads \"FAZO\" version 1");

        let Err(error) = Vm::restore(b"LUA", Options::default(), Host::captured()) else {
            panic!("three bytes must not restore");
        };
        assert_eq!(error.to_string(), "the snapshot starts with \"LUA\" and has no format version; this build reads \"FAZO\" version 1");
    }
}
