
use super::{
    arg_error, argument,
    machine::{Input, Machine, Next, Op, run},
};
use crate::{
    LuaError, Values, values, Vm,
    object::{NativeFn, Step},
    value::Value,
};

#[cfg(windows)]
const DEFAULT_PATH: &str = "!\\lua\\?.lua;!\\lua\\?\\init.lua;!\\?.lua;!\\?\\init.lua;!\\..\\share\\lua\\5.5\\?.lua;!\\..\\share\\lua\\5.5\\?\\init.lua;.\\?.lua;.\\?\\init.lua";
#[cfg(windows)]
const DEFAULT_CPATH: &str = "!\\?.dll;!\\..\\lib\\lua\\5.5\\?.dll;!\\loadall.dll;.\\?.dll";
#[cfg(windows)]
const DIRECTORY_SEPARATOR: &str = "\\";
#[cfg(windows)]
const CONFIG: &str = "\\\n;\n?\n!\n-\n";

#[cfg(not(windows))]
const DEFAULT_PATH: &str = "/usr/local/share/lua/5.5/?.lua;/usr/local/share/lua/5.5/?/init.lua;/usr/local/lib/lua/5.5/?.lua;/usr/local/lib/lua/5.5/?/init.lua;./?.lua;./?/init.lua";
#[cfg(not(windows))]
const DEFAULT_CPATH: &str = "/usr/local/lib/lua/5.5/?.so;/usr/local/lib/lua/5.5/loadall.so;./?.so";
#[cfg(not(windows))]
const DIRECTORY_SEPARATOR: &str = "/";
#[cfg(not(windows))]
const CONFIG: &str = "/\n;\n?\n!\n-\n";

const NO_DYNAMIC_LIBRARIES: &str = "dynamic libraries not enabled; check your Lua installation";

pub(super) fn open(vm: &mut Vm) {
    let package = vm.new_table();
    let loaded = vm.new_table();
    let preload = vm.new_table();
    let searchers = vm.new_table();
    let searcher_functions: &[(&'static str, NativeFn)] = &[
        ("preload", searcher_preload),
        ("lua", searcher_lua),
        ("c", searcher_c),
        ("croot", searcher_croot),
    ];
    for (index, (name, function)) in searcher_functions.iter().enumerate() {
        let value = vm.new_native(name, *function);
        vm.table_set(searchers, Value::Integer(index as i64 + 1), value)
            .expect("integer keys are never nil or NaN");
    }
    let path_text = environment_path(vm, "LUA_PATH", DEFAULT_PATH);
    let cpath_text = environment_path(vm, "LUA_CPATH", DEFAULT_CPATH);
    let path = vm.string_value(path_text.as_bytes());
    let cpath = vm.string_value(cpath_text.as_bytes());
    let config = vm.string_value(CONFIG.as_bytes());
    vm.set_field(package, "loaded", Value::Table(loaded));
    vm.set_field(package, "preload", Value::Table(preload));
    vm.set_field(package, "searchers", Value::Table(searchers));
    vm.set_field(package, "path", path);
    vm.set_field(package, "cpath", cpath);
    vm.set_field(package, "config", config);
    let searchpath = vm.new_native("searchpath", searchpath);
    vm.set_field(package, "searchpath", searchpath);
    let loadlib = vm.new_native("loadlib", loadlib);
    vm.set_field(package, "loadlib", loadlib);
    let require = vm.new_stepped("require", require);
    vm.set_global("require", require);
    vm.set_global("package", Value::Table(package));
    vm.set_hidden("package", Value::Table(package));
    vm.set_hidden("loaded", Value::Table(loaded));
    let registry = vm.registry();
    vm.set_field(registry, "_LOADED", Value::Table(loaded));
    vm.set_field(registry, "_PRELOAD", Value::Table(preload));
    let globals = vm.global_table();
    for name in ["package", "coroutine", "table", "string", "math", "utf8", "debug", "os", "io"] {
        let value = vm.global(name);
        if !value.is_nil() {
            vm.set_field(loaded, name, value);
        }
    }
    vm.set_field(loaded, "_G", globals);
}

fn environment_path(vm: &mut Vm, variable: &str, default: &str) -> String {
    let versioned = format!("{variable}_5_5");
    let value = vm
        .host
        .system
        .getenv(versioned.as_bytes())
        .or_else(|| vm.host.system.getenv(variable.as_bytes()));
    let text = match value {
        Some(value) if vm.options().environment_variables => String::from_utf8_lossy(&value).into_owned(),
        _ => return expand_executable_directory(vm, default),
    };
    let expanded = match text.find(";;") {
        None => text,
        Some(at) => join_default(&text[..at], default, &text[at + 2..]),
    };
    expand_executable_directory(vm, &expanded)
}

fn join_default(prefix: &str, default: &str, suffix: &str) -> String {
    let mut joined = String::new();
    if !prefix.is_empty() {
        joined.push_str(prefix);
        joined.push(';');
    }
    joined.push_str(default);
    if !suffix.is_empty() {
        joined.push(';');
        joined.push_str(suffix);
    }
    joined
}

fn expand_executable_directory(vm: &mut Vm, path: &str) -> String {
    if !cfg!(windows) || !path.contains('!') {
        return path.to_string();
    }
    let directory = vm
        .host
        .system
        .executable_directory()
        .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
        .unwrap_or_default();
    path.replace('!', &directory)
}

fn package_table(vm: &mut Vm) -> Value {
    vm.hidden("package")
}

fn package_field(vm: &mut Vm, field: &str) -> Value {
    match package_table(vm) {
        Value::Table(package) => vm.raw_field(package, field),
        _ => Value::Nil,
    }
}

fn search_path(vm: &mut Vm, name: &[u8], path: &[u8], separator: &[u8], replacement: &[u8]) -> Result<Vec<u8>, Vec<u8>> {
    let name = replace_all(name, separator, replacement);
    let mut tried: Vec<Vec<u8>> = Vec::new();
    for template in path.split(|&byte| byte == b';') {
        let filename = replace_all(template, b"?", &name);
        if vm.host.files.open(&filename, b"r").is_ok() {
            return Ok(filename);
        }
        tried.push(filename);
    }
    let mut message = Vec::new();
    for (index, filename) in tried.iter().enumerate() {
        if index > 0 {
            message.extend_from_slice(b"\n\t");
        }
        message.extend_from_slice(b"no file '");
        message.extend_from_slice(filename);
        message.push(b'\'');
    }
    Err(message)
}

fn replace_all(text: &[u8], pattern: &[u8], replacement: &[u8]) -> Vec<u8> {
    if pattern.is_empty() {
        return text.to_vec();
    }
    let mut out = Vec::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.windows(pattern.len()).position(|window| window == pattern) {
        out.extend_from_slice(&rest[..at]);
        out.extend_from_slice(replacement);
        rest = &rest[at + pattern.len()..];
    }
    out.extend_from_slice(rest);
    out
}

fn find_file(vm: &mut Vm, name: &[u8], field: &str) -> Result<Result<Vec<u8>, Vec<u8>>, LuaError> {
    let path = match package_field(vm, field) {
        Value::String(string) => vm.heap_bytes(string).to_vec(),
        _ => return Err(vm.runtime_error_text(&format!("'package.{field}' must be a string"))),
    };
    Ok(search_path(vm, name, &path, b".", DIRECTORY_SEPARATOR.as_bytes()))
}

fn searcher_preload(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let name = name_argument(vm, args, "searcher_preload")?;
    let preload = match package_field(vm, "preload") {
        Value::Table(table) => table,
        _ => return Err(vm.runtime_error_text("'package.preload' must be a table")),
    };
    let key = vm.string_value(&name);
    let loader = vm.raw_get(Value::Table(preload), key);
    if loader.is_nil() {
        let message = format!("no field package.preload['{}']", String::from_utf8_lossy(&name));
        return Ok(values![vm.string_value(message.as_bytes())]);
    }
    Ok(values![loader, vm.string_value(b":preload:")])
}

fn searcher_lua(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let name = name_argument(vm, args, "searcher_lua")?;
    match find_file(vm, &name, "path")? {
        Ok(filename) => load_module(vm, &name, filename),
        Err(message) => Ok(values![vm.string_value(&message)]),
    }
}

fn load_module(vm: &mut Vm, name: &[u8], filename: Vec<u8>) -> Result<Values, LuaError> {
    match vm.load_file(Some(&filename)) {
        Ok(loader) => {
            let file = vm.string_value(&filename);
            Ok(values![loader, file])
        }
        Err(message) => Err(loader_error(vm, name, &filename, &message)),
    }
}

fn loader_error(vm: &mut Vm, name: &[u8], filename: &[u8], message: &[u8]) -> LuaError {
    let text = format!(
        "error loading module '{}' from file '{}':\n\t{}",
        String::from_utf8_lossy(name),
        String::from_utf8_lossy(filename),
        String::from_utf8_lossy(message)
    );
    vm.runtime_error_text(&text)
}

fn searcher_c(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let name = name_argument(vm, args, "searcher_c")?;
    match find_file(vm, &name, "cpath")? {
        Ok(filename) => Err(loader_error(vm, &name, &filename, NO_DYNAMIC_LIBRARIES.as_bytes())),
        Err(message) => Ok(values![vm.string_value(&message)]),
    }
}

fn searcher_croot(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let name = name_argument(vm, args, "searcher_croot")?;
    let Some(dot) = name.iter().position(|&byte| byte == b'.') else {
        return Ok(Values::new());
    };
    match find_file(vm, &name[..dot], "cpath")? {
        Ok(filename) => Err(loader_error(vm, &name, &filename, NO_DYNAMIC_LIBRARIES.as_bytes())),
        Err(message) => Ok(values![vm.string_value(&message)]),
    }
}

fn name_argument(vm: &mut Vm, args: &[Value], function: &str) -> Result<Vec<u8>, LuaError> {
    match argument(args, 0) {
        Value::String(string) => Ok(vm.heap_bytes(string).to_vec()),
        value @ (Value::Integer(_) | Value::Float(_)) => Ok(vm.to_display_bytes(value)),
        _ => {
            let got = args.first().map_or("no value", |value| value.type_name());
            Err(arg_error(vm, 1, function, &format!("string expected, got {got}")))
        }
    }
}

fn searchpath(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let path = match argument(args, 1) {
        Value::String(string) => vm.heap_bytes(string).to_vec(),
        _ => {
            let got = args.get(1).map_or("no value", |value| value.type_name());
            return Err(arg_error(vm, 2, "searchpath", &format!("string expected, got {got}")));
        }
    };
    let name = name_argument(vm, args, "searchpath")?;
    let separator = optional_string(vm, args, 2, b".")?;
    let replacement = optional_string(vm, args, 3, DIRECTORY_SEPARATOR.as_bytes())?;
    match search_path(vm, &name, &path, &separator, &replacement) {
        Ok(filename) => Ok(values![vm.string_value(&filename)]),
        Err(message) => Ok(values![Value::Nil, vm.string_value(&message)]),
    }
}

fn optional_string(vm: &mut Vm, args: &[Value], index: usize, default: &[u8]) -> Result<Vec<u8>, LuaError> {
    match argument(args, index) {
        Value::Nil => Ok(default.to_vec()),
        Value::String(string) => Ok(vm.heap_bytes(string).to_vec()),
        _ => {
            let got = args.get(index).map_or("no value", |value| value.type_name());
            Err(arg_error(vm, index + 1, "searchpath", &format!("string expected, got {got}")))
        }
    }
}

fn loadlib(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    name_argument(vm, args, "loadlib")?;
    let message = vm.string_value(NO_DYNAMIC_LIBRARIES.as_bytes());
    let kind = vm.string_value(b"open");
    Ok(values![Value::Nil, message, kind])
}

impl RequireState {
    pub(crate) fn values(&self, out: &mut Values) {
        out.extend([self.name, self.loaded, self.searchers, self.data]);
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RequireState {
    name: Value,
    #[serde(with = "crate::bytes")]
    name_bytes: Vec<u8>,
    loaded: Value,
    searchers: Value,
    index: i64,
    #[serde(with = "crate::bytes")]
    messages: Vec<u8>,
    data: Value,
    phase: RequirePhase,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
enum RequirePhase {
    Search,
    Found,
    Searched,
    Loaded,
    Stored,
    ReadBack,
    MarkTrue,
}

fn require(vm: &mut Vm, args: &[Value]) -> Result<Step, LuaError> {
    let name_bytes = name_argument(vm, args, "require")?;
    let name = argument(args, 0);
    let loaded = match vm.hidden("loaded") {
        Value::Table(table) => Value::Table(table),
        _ => Value::Nil,
    };
    let cached = vm.raw_get(loaded, name);
    if !cached.is_falsy() {
        return Ok(Step::Done(values![cached]));
    }
    let searchers = match package_field(vm, "searchers") {
        table @ Value::Table(_) => table,
        _ => return Err(vm.runtime_error_text("'package.searchers' must be a table")),
    };
    let state = RequireState {
        name: if matches!(name, Value::String(_)) { name } else { vm.string_value(&name_bytes) },
        name_bytes,
        loaded,
        searchers,
        index: 1,
        messages: Vec::new(),
        data: Value::Nil,
        phase: RequirePhase::Search,
    };
    run(vm, Machine::Require(state), Input::Start)
}

impl RequireState {
    pub fn advance(&mut self, vm: &mut Vm, input: Input) -> Result<Next, LuaError> {
        match self.phase {
            RequirePhase::Search => Ok(self.request_searcher()),
            RequirePhase::Found => self.call_searcher(vm, input.value()),
            RequirePhase::Searched => Ok(self.take_loader(vm, input.results())),
            RequirePhase::Loaded => Ok(self.store_module(input.value())),
            RequirePhase::Stored => {
                self.phase = RequirePhase::ReadBack;
                Ok(self.read_module())
            }
            RequirePhase::ReadBack => Ok(self.finish_or_mark(input.value())),
            RequirePhase::MarkTrue => Ok(Next::Done(values![Value::Boolean(true), self.data])),
        }
    }

    fn request_searcher(&mut self) -> Next {
        self.phase = RequirePhase::Found;
        Next::Perform(Op::Get {
            object: self.searchers,
            key: Value::Integer(self.index),
        })
    }

    fn call_searcher(&mut self, vm: &mut Vm, searcher: Value) -> Result<Next, LuaError> {
        if searcher.is_nil() {
            let text = String::from_utf8_lossy(&self.name_bytes).into_owned();
            let mut message = format!("module '{text}' not found:").into_bytes();
            message.extend_from_slice(&self.messages);
            return Err(vm.runtime_error(&message));
        }
        self.phase = RequirePhase::Searched;
        Ok(Next::Perform(Op::Call {
            function: searcher,
            arguments: values![self.name],
        }))
    }

    fn take_loader(&mut self, vm: &mut Vm, results: Values) -> Next {
        let first = results.first().copied().unwrap_or(Value::Nil);
        if first.is_function() {
            self.data = results.get(1).copied().unwrap_or(Value::Nil);
            self.phase = RequirePhase::Loaded;
            return Next::Perform(Op::Call {
                function: first,
                arguments: values![self.name, self.data],
            });
        }
        if let Value::String(_) | Value::Integer(_) | Value::Float(_) = first {
            self.messages.extend_from_slice(b"\n\t");
            self.messages.extend_from_slice(&vm.to_display_bytes(first));
        }
        self.index += 1;
        self.request_searcher()
    }

    fn store_module(&mut self, module: Value) -> Next {
        if module.is_nil() {
            self.phase = RequirePhase::ReadBack;
            return self.read_module();
        }
        self.phase = RequirePhase::Stored;
        Next::Perform(Op::Set {
            object: self.loaded,
            key: self.name,
            value: module,
        })
    }

    fn read_module(&self) -> Next {
        Next::Perform(Op::Get {
            object: self.loaded,
            key: self.name,
        })
    }

    fn finish_or_mark(&mut self, value: Value) -> Next {
        if !value.is_nil() {
            return Next::Done(values![value, self.data]);
        }
        self.phase = RequirePhase::MarkTrue;
        Next::Perform(Op::Set {
            object: self.loaded,
            key: self.name,
            value: Value::Boolean(true),
        })
    }
}
