#![deny(unsafe_code)]

mod arena;
mod bytes;
mod compiler;
mod embed;
mod encoding;
mod gc;
mod heap;
mod host;
mod instruction;
mod jit;
mod listing;
mod math;
mod number;
mod object;
mod opcode;
#[cfg(feature = "opcode-counts")]
mod opcode_counts;
mod proto;
mod session;
mod shape;
mod stdlib;
pub mod syntax;
mod table;
#[cfg(test)]
mod table_tests;
mod value;
mod vm;

use std::{
    any::Any,
    borrow::Cow,
    collections::BTreeMap,
    sync::{Arc, atomic::AtomicU8},
};

use arena::Ref;
use heap::Heap;
use object::{Native, NativeEntry, Thread};
use stdlib::base::WarningState;
use table::Table;
pub use host::{
    Captured, ExitStatus, FileSystem, FixedSystem, Host, HostError, HostFile, MemoryFiles, NoFiles,
    RegistrationError, Seek, Stdio, System, TimeFields,
};
pub use jit::{JitPolicy, JitStress, Tier, TierPolicy};
pub use object::{NativeFn, Step, SteppedFn};
pub use session::{Outcome, RequestHandle, SnapshotError};
pub use value::{Value, Values};

/// `os`, `io` and `package` are off by default, so the embedder has no clock, environment, file or process access unless it enables them.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Options {
    pub os: bool,
    pub io: bool,
    pub environment_variables: bool,
    pub package: bool,
    /// `None` uses a fixed seed, so runs repeat.
    pub random_seed: Option<i64>,
    /// A run that goes over it raises `not enough memory`.
    pub memory_limit: Option<usize>,
    pub tier: Tier,
    pub force: Option<Tier>,
    pub jit_policy: JitPolicy,
    pub jit_stress: JitStress,
    pub jit_trace: bool,
    pub jit_library: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            os: false,
            io: false,
            environment_variables: true,
            package: false,
            random_seed: None,
            memory_limit: None,
            tier: Tier::default(),
            force: None,
            jit_policy: JitPolicy::default(),
            jit_stress: JitStress::default(),
            jit_trace: false,
            jit_library: false,
        }
    }
}

impl Options {
    /// Returns false for an option fazo does not know; accepts the standalone interpreter's `-X` options.
    pub fn set_extension(&mut self, option: &[u8]) -> bool {
        let (name, value) = match option.iter().position(|&byte| byte == b'=') {
            Some(at) => (&option[..at], Some(&option[at + 1..])),
            None => (option, None),
        };
        let number = || value.and_then(|value| std::str::from_utf8(value).ok()?.parse::<u32>().ok());
        match (name, value) {
            (b"jit", Some(b"off")) => self.tier = Tier::Interpreter,
            (b"tier", Some(value)) => match Tier::from_name(value) {
                Some(tier) => self.tier = tier,
                None => return false,
            },
            (b"force", Some(value)) => match Tier::from_name(value) {
                Some(Tier::Interpreter) => self.tier = Tier::Interpreter,
                Some(tier) => {
                    self.force = Some(tier);
                    self.jit_policy = JitPolicy::forced();
                }
                None => return false,
            },
            (b"stress-threshold", Some(_)) => match number() {
                Some(divisor) if divisor > 0 => self.jit_stress.threshold_divisor = divisor,
                _ => return false,
            },
            (b"stress-exit", Some(_)) => match number() {
                Some(every) => self.jit_stress.exit_every = every,
                None => return false,
            },
            (b"baseline-budget", Some(_)) => match number() {
                Some(budget) => self.jit_policy.baseline.budget_per_instruction = budget,
                None => return false,
            },
            (b"stress-ic", None) => self.jit_stress.inline_caches_off = true,
            (b"trace", None) => self.jit_trace = true,
            (b"jitlib", None) => self.jit_library = true,
            _ => return false,
        }
        true
    }
}

/// Carries any error value; [`Vm::error_message`] renders it as the standalone interpreter prints it.
#[derive(Debug, Clone, Copy)]
pub struct LuaError {
    value: Value,
    kind: ErrorKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ErrorKind {
    Raised,
    Handled,
    Paused,
    Interrupted,
    Unwinding,
}

impl LuaError {
    pub fn from_value(value: Value) -> Self {
        Self {
            value,
            kind: ErrorKind::Raised,
        }
    }

    pub(crate) fn is_handled(&self) -> bool {
        self.kind != ErrorKind::Raised
    }

    pub(crate) fn handled(value: Value) -> Self {
        Self {
            value,
            kind: ErrorKind::Handled,
        }
    }

    pub(crate) fn unwinding(value: Value) -> Self {
        Self {
            value,
            kind: ErrorKind::Unwinding,
        }
    }

    pub(crate) fn pause() -> Self {
        Self {
            value: Value::Nil,
            kind: ErrorKind::Paused,
        }
    }

    pub(crate) fn is_pause(&self) -> bool {
        self.kind == ErrorKind::Paused
    }

    pub(crate) fn interrupted(value: Value) -> Self {
        Self {
            value,
            kind: ErrorKind::Interrupted,
        }
    }

    /// An interrupt cannot be caught by `pcall` and is not resumable.
    pub fn is_interrupted(&self) -> bool {
        self.kind == ErrorKind::Interrupted
    }

    pub(crate) fn is_unwinding(&self) -> bool {
        self.kind == ErrorKind::Unwinding
    }

    pub fn value(&self) -> Value {
        self.value
    }
}

pub struct Vm {
    options: Options,
    host: Host,
    heap: Heap,
    globals: Ref<Table>,
    registry: Ref<Table>,
    main_thread: Ref<Thread>,
    current_thread: Ref<Thread>,
    tracing: bool,
    hidden: Vec<(String, Value)>,
    string_metatable: Option<Ref<Table>>,
    current_native: Option<vm::NativeCall>,
    floor: usize,
    c_calls: u32,
    non_yieldable: u32,
    handling_error: u32,
    yielded: Values,
    argument_pool: Vec<Values>,
    text_buffers: Vec<Vec<u8>>,
    type_metatables: [Option<Ref<Table>>; 6],
    random: [u64; 4],
    warnings: WarningState,
    resources: Vec<stdlib::io::Stream>,
    pins: Vec<Value>,
    natives: Vec<NativeEntry>,
    megamorphic: vm::ic::MegamorphicCache,
    native_index: BTreeMap<Cow<'static, str>, Vec<object::NativeId>>,
    requests: Arc<AtomicU8>,
    #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
    jit: jit::JitState,
    pausable: bool,
    pause_holdoff: bool,
    deferred_loop_charge: Option<vm::DeferredLoopCharge>,
    run_depth: u32,
    suspended: Option<session::Suspended>,
    host_data: Option<Box<dyn Any + Send>>,
    #[cfg(feature = "opcode-counts")]
    opcode_counts: opcode_counts::OpcodeCounts,
}

impl Vm {
    /// Captures the standard streams in memory, with no files and a fixed clock.
    pub fn new(options: Options) -> Self {
        Self::with_host(options, Host::captured())
    }

    pub fn with_host(options: Options, mut host: Host) -> Self {
        let host_natives = host.take_natives();
        let mut heap = Heap::default();
        heap.gc.limit = options.memory_limit;
        heap.init_shapes();
        let names: Vec<&'static str> = vm::Event::ALL
            .iter()
            .map(|event| event.name())
            .chain(vm::Field::ALL.iter().map(|field| field.name()))
            .chain(vm::TYPE_NAMES)
            .collect();
        heap.events = names.into_iter().map(|name| heap.string(name.as_bytes())).collect();
        let globals = heap.alloc(Table::default());
        let registry = heap.alloc(Table::default());
        let main_thread = heap.alloc(Thread {
            status: object::CoStatus::Running,
            started: true,
            ..Thread::default()
        });
        let mut vm = Self {
            options,
            host,
            heap,
            globals,
            registry,
            main_thread,
            current_thread: main_thread,
            tracing: false,
            hidden: Vec::new(),
            string_metatable: None,
            current_native: None,
            floor: 0,
            c_calls: 0,
            non_yieldable: 0,
            handling_error: 0,
            yielded: Values::new(),
            argument_pool: Vec::new(),
            text_buffers: Vec::new(),
            type_metatables: [None; 6],
            random: [0; 4],
            warnings: WarningState::Off,
            resources: Vec::new(),
            pins: Vec::new(),
            natives: Vec::new(),
            megamorphic: Default::default(),
            native_index: BTreeMap::new(),
            requests: Arc::new(AtomicU8::new(0)),
            #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
            jit: jit::JitState::default(),
            pausable: false,
            pause_holdoff: false,
            deferred_loop_charge: None,
            run_depth: 0,
            suspended: None,
            host_data: None,
            #[cfg(feature = "opcode-counts")]
            opcode_counts: opcode_counts::OpcodeCounts::default(),
        };
        vm.set_registry_slots(main_thread, globals);
        stdlib::open(&mut vm);
        for entry in host_natives {
            if vm.native_index.contains_key(entry.name.as_ref()) {
                panic!("the native function `{}` is provided by both the host and the libraries", entry.name);
            }
            vm.register_entry(entry);
        }
        vm
    }

    pub fn options(&self) -> &Options {
        &self.options
    }

    /// Returns and clears the captured output; empty when the host writes through.
    pub fn take_output(&mut self) -> Vec<u8> {
        self.host.stdio.take_stdout()
    }

    pub fn take_error_output(&mut self) -> Vec<u8> {
        self.host.stdio.take_stderr()
    }

    /// Renders strings and numbers as text and any other value as `(error object is a <type> value)`.
    pub fn error_message(&self, error: &LuaError) -> Vec<u8> {
        match error.value {
            Value::String(string) => self.heap.bytes(string).to_vec(),
            Value::Integer(_) | Value::Float(_) => self.to_display_bytes(error.value),
            other => format!("(error object is a {} value)", other.type_name()).into_bytes(),
        }
    }

    /// Syntax and runtime errors both come back as the error object.
    pub fn run_source(&mut self, chunkname: &[u8], source: &[u8]) -> Result<(), LuaError> {
        let function = self
            .compile_chunk(chunkname, source, None)
            .map_err(|message| self.error_string(message))?;
        self.call_value(function, &[])?;
        Ok(())
    }

    /// Runs from a C function, so tracebacks end with `[C]: in ?`, and errors carry a traceback.
    pub fn run_script(&mut self, chunkname: &[u8], source: &[u8]) -> Result<(), LuaError> {
        let function = self
            .compile_chunk(chunkname, source, None)
            .map_err(|message| self.error_string(message))?;
        self.call_with_traceback(function, &[])?;
        Ok(())
    }

    /// Names are unique per VM, since a snapshot refers to a native by name.
    pub fn register_native(&mut self, name: impl Into<Cow<'static, str>>, function: NativeFn) -> Result<Value, RegistrationError> {
        self.register_native_with_data(name, function, 0)
    }

    /// A restored VM has the data only once its host registers the native again.
    pub fn register_native_with_data(
        &mut self,
        name: impl Into<Cow<'static, str>>,
        function: NativeFn,
        data: u32,
    ) -> Result<Value, RegistrationError> {
        let name = name.into();
        self.register_fresh(name.clone(), object::NativeBody::Plain(function), data)?;
        Ok(self.native(&name).expect("a registered native has a value"))
    }

    pub fn register_stepped(&mut self, name: impl Into<Cow<'static, str>>, function: SteppedFn) -> Result<Value, RegistrationError> {
        let name = name.into();
        self.register_fresh(name.clone(), object::NativeBody::Stepped(function), 0)?;
        Ok(self.native(&name).expect("a registered native has a value"))
    }

    pub fn native(&mut self, name: &str) -> Option<Value> {
        let function = *self.native_index.get(name)?.first()?;
        let builtin = self.natives[function.0 as usize].builtin;
        Some(Value::Native(self.heap.alloc(Native {
            function,
            upvalues: Vec::new(),
            builtin,
        })))
    }

    fn register_fresh(&mut self, name: Cow<'static, str>, body: object::NativeBody, data: u32) -> Result<(), RegistrationError> {
        if self.native_index.contains_key(name.as_ref()) {
            return Err(RegistrationError::new(&name));
        }
        self.register_entry(NativeEntry { name, body, data, builtin: None });
        Ok(())
    }

    fn register_entry(&mut self, entry: NativeEntry) -> object::NativeId {
        let id = object::NativeId(u32::try_from(self.natives.len()).expect("native index overflow"));
        self.native_index.entry(entry.name.clone()).or_default().push(id);
        self.natives.push(entry);
        id
    }

    pub(crate) fn register_builtin(&mut self, name: &'static str, body: object::NativeBody) {
        self.native_id(name, body);
    }

    fn native_id(&mut self, name: &'static str, body: object::NativeBody) -> object::NativeId {
        let existing = self.native_index.get(name).and_then(|ids| {
            ids.iter()
                .copied()
                .find(|id| object::same_body(self.natives[id.0 as usize].body, body))
        });
        match existing {
            Some(id) => id,
            None => self.register_entry(NativeEntry {
                name: Cow::Borrowed(name),
                body,
                data: 0,
                builtin: None,
            }),
        }
    }

    pub(crate) fn new_native(&mut self, name: &'static str, function: NativeFn) -> Value {
        self.new_native_with(name, object::NativeBody::Plain(function), Vec::new())
    }

    pub(crate) fn new_stepped(&mut self, name: &'static str, function: SteppedFn) -> Value {
        self.new_native_with(name, object::NativeBody::Stepped(function), Vec::new())
    }

    pub(crate) fn new_native_with(&mut self, name: &'static str, body: object::NativeBody, upvalues: Vec<Value>) -> Value {
        let function = self.native_id(name, body);
        let builtin = self.natives[function.0 as usize].builtin;
        Value::Native(self.heap.alloc(Native { function, upvalues, builtin }))
    }

    pub(crate) fn native_body(&self, native: Ref<Native>) -> object::NativeBody {
        self.natives[self.heap.natives.get(native).function.0 as usize].body
    }

    pub(crate) fn set_native_upvalue(&mut self, index: usize, value: Value) {
        if let Some(call) = self.current_native {
            self.heap.set_native_value(call.native, index, value);
        }
    }

    pub(crate) fn native_upvalue_count(&self, native: Ref<Native>) -> usize {
        self.heap.natives.get(native).upvalues.len()
    }

    pub(crate) fn native_upvalues(&self) -> Values {
        self.current_native
            .map(|call| Values::from_slice(&self.heap.natives.get(call.native).upvalues))
            .unwrap_or_default()
    }

    pub fn set_global(&mut self, name: &str, value: Value) {
        let key = self.heap.string_value(name.as_bytes());
        self.heap.table_set(self.globals, key, value).expect("string keys are never nil or NaN");
    }

    pub fn global(&mut self, name: &str) -> Value {
        let key = self.heap.string_value(name.as_bytes());
        self.heap.tables.get(self.globals).get(self.heap.key_view(), key)
    }

    /// The registry is a collector root, so the value lives as long as the VM.
    pub fn set_registry_value(&mut self, key: &str, value: Value) {
        let key = self.heap.string_value(key.as_bytes());
        self.heap.table_set(self.registry, key, value).expect("string keys are never nil or NaN");
    }

    pub fn registry_value(&mut self, key: &str) -> Value {
        let key = self.heap.string_value(key.as_bytes());
        self.heap.tables.get(self.registry).get(self.heap.key_view(), key)
    }

    pub fn string_bytes(&self, value: Value) -> Option<&[u8]> {
        match value {
            Value::String(string) => Some(self.heap.bytes(string)),
            _ => None,
        }
    }

    /// A nil `key` starts the traversal.
    pub fn next_entry(&mut self, table: Value, key: Value) -> Result<Option<(Value, Value)>, LuaError> {
        let Value::Table(table) = table else {
            return Err(self.error_string("table expected"));
        };
        self.table_next(table, key)
    }

    pub fn running_native(&self) -> Option<&str> {
        let entry = self.running_entry()?;
        Some(entry.name.as_ref())
    }

    pub fn running_native_data(&self) -> Option<u32> {
        Some(self.running_entry()?.data)
    }

    fn running_entry(&self) -> Option<&NativeEntry> {
        let call = self.current_native?;
        let id = self.heap.natives.get(call.native).function;
        Some(&self.natives[id.0 as usize])
    }

    /// A snapshot does not hold the host state, so a restored VM has none.
    pub fn set_host_data(&mut self, data: Box<dyn Any + Send>) {
        self.host_data = Some(data);
    }

    pub fn host_data(&self) -> Option<&(dyn Any + Send)> {
        self.host_data.as_deref()
    }

    pub fn host_data_mut(&mut self) -> Option<&mut (dyn Any + Send)> {
        self.host_data.as_deref_mut()
    }

    pub fn take_host_data(&mut self) -> Option<Box<dyn Any + Send>> {
        self.host_data.take()
    }

    pub(crate) fn error_string(&mut self, message: impl AsRef<[u8]>) -> LuaError {
        let string = self.heap.string(message.as_ref());
        LuaError::from_value(Value::String(string))
    }

    pub(crate) fn write_output(&mut self, bytes: &[u8]) {
        self.host.stdio.write_stdout(bytes);
    }

    pub(crate) fn write_error(&mut self, bytes: &[u8]) {
        self.host.stdio.write_stderr(bytes);
    }
}
