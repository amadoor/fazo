use std::{backtrace::Backtrace, cell::RefCell, panic, sync::Arc};

use cranelift_codegen::{
    Context,
    control::ControlPlane,
    gimli::{self, LittleEndian, UnwindSection},
    ir::{AbiParam, Function, InstBuilder, types},
    isa::{self, TargetIsa, unwind::UnwindInfo},
    settings::{self, Configurable},
};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITMemoryKind, JITMemoryProvider, JITModule};
use cranelift_module::{FuncId, Linkage, Module};
use target_lexicon::Triple;

use super::{
    JitResult,
    memory::{CodeBudget, CodeMemory, GRANULE},
    unit::{Compiler, JitUnit},
    unwind,
};

const HELPER: &str = "fazo_jit_test_helper";

thread_local! {
    static HELPER_BACKTRACE: RefCell<Option<String>> = const { RefCell::new(None) };
}

extern "C" fn helper(value: i64) -> i64 {
    panic::catch_unwind(|| {
        if value < 0 {
            let trace = Backtrace::force_capture().to_string();
            HELPER_BACKTRACE.with(|slot| *slot.borrow_mut() = Some(trace));
        }
        value + 1
    })
    .unwrap_or_else(|_| std::process::abort())
}

fn compiler() -> Compiler {
    Compiler::new().expect("the host has a Cranelift backend")
}

fn helper_address() -> *const u8 {
    (helper as extern "C" fn(i64) -> i64) as *const u8
}

fn add_two(module: &mut JITModule, func: &mut Function, context: &mut FunctionBuilderContext) -> JitResult<FuncId> {
    let mut signature = module.make_signature();
    signature.params.push(AbiParam::new(types::I64));
    signature.params.push(AbiParam::new(types::I64));
    signature.returns.push(AbiParam::new(types::I64));
    func.signature = signature.clone();
    let id = module.declare_function("add_two", Linkage::Export, &signature)?;

    let config = module.target_config();
    let mut builder = FunctionBuilder::new(func, context);
    let entry = builder.create_block();
    builder.append_block_params_for_function_params(entry);
    builder.switch_to_block(entry);
    let left = builder.block_params(entry)[0];
    let right = builder.block_params(entry)[1];
    let sum = builder.ins().iadd(left, right);
    builder.ins().return_(&[sum]);
    builder.seal_block(entry);
    builder.finalize(config);
    Ok(id)
}

fn call_helper_and_add_ten(module: &mut JITModule, func: &mut Function, context: &mut FunctionBuilderContext) -> JitResult<FuncId> {
    let mut helper_signature = module.make_signature();
    helper_signature.params.push(AbiParam::new(types::I64));
    helper_signature.returns.push(AbiParam::new(types::I64));
    let helper_id = module.declare_function(HELPER, Linkage::Import, &helper_signature)?;

    let mut signature = module.make_signature();
    signature.params.push(AbiParam::new(types::I64));
    signature.returns.push(AbiParam::new(types::I64));
    func.signature = signature.clone();
    let id = module.declare_function("call_helper_and_add_ten", Linkage::Export, &signature)?;

    let helper_ref = module.declare_func_in_func(helper_id, func);
    let config = module.target_config();
    let mut builder = FunctionBuilder::new(func, context);
    let entry = builder.create_block();
    builder.append_block_params_for_function_params(entry);
    builder.switch_to_block(entry);
    let argument = builder.block_params(entry)[0];
    let call = builder.ins().call(helper_ref, &[argument]);
    let returned = builder.inst_results(call)[0];
    let ten = builder.ins().iconst(types::I64, 10);
    let result = builder.ins().iadd(returned, ten);
    builder.ins().return_(&[result]);
    builder.seal_block(entry);
    builder.finalize(config);
    Ok(id)
}

type Unary = extern "C" fn(i64) -> i64;
type Binary = extern "C" fn(i64, i64) -> i64;

fn as_binary(entry: *const u8) -> Binary {
    debug_assert!(!entry.is_null());
    // SAFETY: `entry` is the finalized `add_two`, with two i64 parameters, an i64 result and the host's C convention.
    unsafe { std::mem::transmute::<*const u8, Binary>(entry) }
}

fn as_unary(entry: *const u8) -> Unary {
    debug_assert!(!entry.is_null());
    // SAFETY: `entry` is the finalized `call_helper_and_add_ten`, with one i64 parameter, an i64 result and the host's C convention.
    unsafe { std::mem::transmute::<*const u8, Unary>(entry) }
}

#[test]
fn a_compiled_add_runs_and_its_memory_returns_to_the_budget_on_drop() {
    let budget = CodeBudget::new(GRANULE * 4);
    let unit = JitUnit::compile(&mut compiler(), budget.clone(), &[], add_two).expect("the add compiles");
    let add = as_binary(unit.entry());
    assert_eq!(add(2, 40), 42);
    assert_eq!(budget.used(), GRANULE);
    drop(unit);
    assert_eq!(budget.used(), 0);
}

#[test]
fn compiled_code_calls_a_rust_helper_and_uses_its_result() {
    let unit = JitUnit::compile(
        &mut compiler(),
        CodeBudget::new(GRANULE),
        &[(HELPER, helper_address())],
        call_helper_and_add_ten,
    )
    .expect("the helper call compiles");
    let call = as_unary(unit.entry());
    assert_eq!(call(4), 15);
}

#[test]
fn a_backtrace_taken_in_a_helper_reaches_the_frames_above_the_jit_code() {
    let unit = JitUnit::compile(
        &mut compiler(),
        CodeBudget::new(GRANULE),
        &[(HELPER, helper_address())],
        call_helper_and_add_ten,
    )
    .expect("the helper call compiles");
    let call = as_unary(unit.entry());
    assert_eq!(call(-1), 10);
    let trace = HELPER_BACKTRACE
        .with(|slot| slot.borrow_mut().take())
        .expect("the helper captured a backtrace");
    assert!(
        trace.contains("a_backtrace_taken_in_a_helper_reaches_the_frames_above_the_jit_code"),
        "the backtrace stops at the JIT frame:\n{trace}"
    );
}

#[test]
fn code_memory_refuses_what_the_budget_cannot_hold_and_releases_on_drop() {
    let budget = CodeBudget::new(GRANULE);
    let mut memory = CodeMemory::new(budget.clone());
    assert!(
        memory
            .allocate(GRANULE + 1, 16, JITMemoryKind::Executable)
            .is_err()
    );
    assert_eq!(budget.used(), 0);
    memory
        .allocate(100, 16, JITMemoryKind::Executable)
        .expect("one granule fits the budget");
    assert_eq!(budget.used(), GRANULE);
    assert!(memory.allocate(1, 16, JITMemoryKind::Executable).is_err());
    assert!(memory.allocate(16, 8, JITMemoryKind::Writable).is_err());
    drop(memory);
    assert_eq!(budget.used(), 0);
}

fn host_isa_with_frame_pointers() -> Arc<dyn TargetIsa> {
    let mut flags = settings::builder();
    flags
        .set("preserve_frame_pointers", "true")
        .expect("the flag exists");
    isa::lookup(Triple::host())
        .expect("the host has a Cranelift backend")
        .finish(settings::Flags::new(flags))
        .expect("the host ISA builds")
}

fn unwind_info_for_host(
    isa: &dyn TargetIsa,
    kind: cranelift_codegen::isa::unwind::UnwindInfoKind,
) -> (Option<UnwindInfo>, usize) {
    let mut func = Function::new();
    let mut signature = cranelift_codegen::ir::Signature::new(isa.default_call_conv());
    signature.params.push(AbiParam::new(types::I64));
    signature.returns.push(AbiParam::new(types::I64));
    func.signature = signature;
    let mut context = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut func, &mut context);
    let entry = builder.create_block();
    builder.append_block_params_for_function_params(entry);
    builder.switch_to_block(entry);
    let value = builder.block_params(entry)[0];
    builder.ins().return_(&[value]);
    builder.seal_block(entry);
    builder.finalize(isa.frontend_config());

    let mut ctx = Context::new();
    ctx.func = func;
    let compiled = ctx
        .compile(isa, &mut ControlPlane::default())
        .expect("the function compiles");
    let code_len = compiled.code_buffer().len();
    let info = compiled
        .create_unwind_info_of_kind(isa, kind)
        .expect("unwind information is produced");
    (info, code_len)
}

const BLOB: u64 = 0x7000_0000;

#[test]
fn the_system_v_table_resolves_its_fde_to_the_start_of_the_blob() {
    let isa = host_isa_with_frame_pointers();
    let (info, code_len) = unwind_info_for_host(
        &*isa,
        cranelift_codegen::isa::unwind::UnwindInfoKind::SystemV,
    );
    let trailer_offset = 64;
    let trailer = unwind::build(&*isa, info, code_len, trailer_offset)
        .expect("the eh_frame table builds");
    let Some(unwind::Table::SystemV { eh_frame, fdes }) = trailer.table else {
        panic!("a System V table is expected");
    };
    assert_eq!(eh_frame, trailer_offset);

    let table = &trailer.bytes;
    let eh_address = BLOB + trailer_offset as u64;
    let frames = gimli::EhFrame::new(table, LittleEndian);
    let bases = gimli::BaseAddresses::default().set_eh_frame(eh_address);
    let mut entries = frames.entries(&bases);
    let mut resolved = None;
    while let Some(entry) = entries.next().expect("the table parses") {
        let gimli::CieOrFde::Fde(partial) = entry else {
            continue;
        };
        let fde = partial
            .parse(|_, bases, offset| frames.cie_from_offset(bases, offset))
            .expect("the FDE parses");
        resolved = Some((fde.initial_address(), fde.len()));
    }
    assert_eq!(resolved, Some((BLOB, code_len as u64)));
    assert_eq!(fdes.len(), 1);
    assert!(fdes[0] > trailer_offset);
    assert_eq!(table[table.len() - 4..], [0, 0, 0, 0]);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn the_windows_x64_table_names_the_function_and_its_unwind_data_by_blob_offset() {
    let isa = host_isa_with_frame_pointers();
    let (info, code_len) = unwind_info_for_host(
        &*isa,
        cranelift_codegen::isa::unwind::UnwindInfoKind::Windows,
    );
    let trailer_offset = 96;
    let trailer = unwind::build(&*isa, info, code_len, trailer_offset)
        .expect("the function table builds");
    let Some(unwind::Table::WindowsX64 { table }) = trailer.table else {
        panic!("a Windows x64 table is expected");
    };
    let entry = table - trailer_offset;
    let word = |at: usize| {
        u32::from_le_bytes(trailer.bytes[entry + at..entry + at + 4].try_into().unwrap())
    };
    assert_eq!(word(0), 0);
    assert_eq!(word(4), u32::try_from(code_len).unwrap());
    assert_eq!(word(8), u32::try_from(trailer_offset).unwrap());
    assert_eq!(trailer.bytes.len(), entry + 12);
}

const PROGRAM: &[u8] = b"
local function arithmetic(n)
  local s, f = 0, 0.5
  for i = 1, n do
    s = s + i * 3 - (i // 2) + (i % 7)
    f = f * 1.0001 + i / 3 - 2.0
    s = s ~ (i << 1) | (i >> 2) & 0xff
  end
  return s, f
end
print(arithmetic(1000))
print(math.maxinteger + 1 == math.mininteger, -math.mininteger, -(0.0), 3 // 0.0, -3 % 0.0 ~= -3 % 0.0)
print(1 < 2.5, 2.5 <= 2, 3 == 3.0, 1 ~= 1.0, 'a' < 'b', not nil, not 0, nil and 1 or 2)
local t = {}
for i = 1, 100 do t[i] = i * i end
local sum = 0
for _, v in ipairs(t) do if v % 2 == 0 then sum = sum + v end end
print(sum, #t, t[50], t[101])
local acc = {}
local k = 0
while true do
  k = k + 1
  if k > 20 then break end
  if k == 7 then goto skip end
  acc[#acc + 1] = k
  ::skip::
end
print(table.concat(acc, ','))
for x = 1.0, 2.0, 0.25 do io.write(x, ' ') end
print()
for x = 10, 1, -3 do io.write(x, ' ') end
print()
local function counter()
  local n = 0
  return function() n = n + 1 return n end
end
local c = counter()
for _ = 1, 50 do c() end
print(c(), select('#', 1, nil, 3), (...))
local mt = setmetatable({}, { __add = function(a, b) return 'added' end, __index = function(_, key) return key .. '!' end })
print(mt + 1, mt.hello, mt[3])
print(pcall(function() local z = nil; return z + 1 end))
print(pcall(function() return 1 < 'x' end))
print(string.format('%a', 0.1), 7 // -2, 7 % -2, -7 // 2.0, 2^10, 10 / 4)
local co = coroutine.wrap(function(a)
  for i = 1, 3 do a = coroutine.yield(a + i) end
  return 'done'
end)
print(co(1), co(10), co(20), co(30))
";

const MODES: &[&[&str]] = &[
    &["force=baseline"],
    &["force=peak"],
    &["stress-threshold=100000"],
    &["stress-threshold=100000", "stress-exit=7"],
    &["force=baseline", "stress-ic"],
    &["force=peak", "stress-ic", "stress-exit=3"],
];

const INTERPRETER: &[&str] = &["tier=interpreter"];

fn options_for(mode: &[&str]) -> crate::Options {
    let mut options = crate::Options {
        io: true,
        ..crate::Options::default()
    };
    for option in mode {
        assert!(options.set_extension(option.as_bytes()), "fazo knows -X {option}");
    }
    options
}

fn run_with(mode: &[&str]) -> (Vec<u8>, bool) {
    let mut vm = crate::Vm::new(options_for(mode));
    if let Err(error) = vm.run_source(b"=jit", PROGRAM) {
        panic!("the program runs: {}", String::from_utf8_lossy(&vm.error_message(&error)));
    }
    let compiled = vm.heap.protos.iter().any(|(_, proto)| proto.jit.is_compiled());
    (vm.take_output(), compiled)
}

#[test]
fn forced_compilation_prints_what_the_interpreter_prints() {
    let (interpreted, none_compiled) = run_with(INTERPRETER);
    assert!(!none_compiled, "the interpreter-only run compiles nothing");
    for mode in MODES {
        let (forced, compiled) = run_with(mode);
        assert!(compiled, "the {mode:?} run compiles the program's functions");
        assert_eq!(String::from_utf8_lossy(&forced), String::from_utf8_lossy(&interpreted), "mode {mode:?}");
    }
}

const GROUPS: &[&[u8]] = &[
    b"local a, b = math.maxinteger, 3 print(a + b, a - b, a * b, a + 1, a - (-1), a * 2, b & 1, b | 8, b ~ 5, a >> 1, 1 << 63, -a, ~b, 7 & 6 | 1 ~ 2)",
    b"local x, y = 1.5, -0.0 print(x + y, x - y, x * y, x + 2, x * 2.0, -x, -y, x - 0.5, 2.0 ^ 3, 1 / 0, x // 1, x % 1)",
    b"local i, f = 3, 2.5 print(i + f, f * i, i - 1.0, i * 0.5, f + i, 1 == 1.0, 'a' == 'a', {} == {})",
    b"print('10' + 1, '3' * '4', 10 .. 20, pcall(function() return {} + 1 end))",
    b"local a, b = 1, 2.0 print(a < b, a <= b, a > b, a >= b, a == b, a ~= b, a == 2, a < 2, a <= 1, a > 1, a >= 1, 1.5 < 2, 2.0 == 2)\nlocal n = 0 / 0 print(n == n, n < 1, n >= 1, n ~= n, n <= n)",
    b"local t = 0 for i = 1, 10 do if i % 2 == 0 and i > 3 or i == 1 then t = t + i elseif not (i < 8) then t = t - 1 end end print(t)",
    b"local x = nil print(x or 'd', false and 1, 0 and 'z', nil or false, x and x.y, 1 and 2 or 3, nil and 1)",
    b"for i = 10, 1, -2 do io.write(i, ' ') end for x = 0.5, 2.0, 0.5 do io.write(x, ' ') end for i = math.maxinteger - 2, math.maxinteger do io.write(i, ' ') end print()",
    b"local a, b, c = nil, nil, nil print(a, b, c, true, false, nil)",
    b"print(pcall(function() local a; return a.x end))\nprint(pcall(function() return 1 // 0 end))\nprint(pcall(function() return 3.5 & 1 end))\nprint(3.0 & 1, 2^53 | 0)",
    b"local s = 0 local i = 0 repeat i = i + 1 s = s + i until i >= 100 print(s)\nlocal w = 0 while w < 50 do w = w + 3 end print(w)",
    b"local t = {1, 2, 3, x = 10}
for i = 1, 200 do t[i] = i * 2 t.k = i end
print(t[1], t[200], t.k, t.x, t.missing, t[300], #t)
local p = setmetatable({}, {__index = function(_, k) return 'idx:' .. tostring(k) end, __newindex = function(s, k, v) rawset(s, k, v .. '!') end})
for i = 1, 3 do p.a = i p[i] = i print(p.a, p[i], p.zz) end
local q = setmetatable({}, {__index = {inherited = 42}})
for i = 1, 2 do print(q.inherited, q[1], q.nope) end
for i = 1, 2 do print(pcall(function() local z = {} z[nil] = 1 end)) end
for i = 1, 2 do print(pcall(function() local z = {} z[0/0] = 1 end)) end
for i = 1, 2 do print(pcall(function() local n = nil return n.field end)) end",
    b"local function f(...) local n = select('#', ...) return n, ... end print(f(1, nil, 3))\nlocal t = setmetatable({}, {__index = function(_, k) return k * 2 end}) print(t[21], t.x == nil or 'x')",
];

/// Each group runs three times, so a forced Peak compile has the feedback of the first run.
#[test]
fn each_opcode_group_matches_the_interpreter_when_compiled() {
    for group in GROUPS {
        let group = [&b"for _ = 1, 3 do\n"[..], group, b"\nend"].concat();
        let interpreted = run_source_with(options_for(INTERPRETER), &group);
        for mode in MODES {
            let forced = run_source_with(options_for(mode), &group);
            assert_eq!(
                String::from_utf8_lossy(&forced.0),
                String::from_utf8_lossy(&interpreted.0),
                "program {} in mode {mode:?}",
                String::from_utf8_lossy(&group)
            );
            assert!(forced.1 || !mode[0].starts_with("force"), "the {mode:?} run compiles {}", String::from_utf8_lossy(&group));
        }
    }
}

fn run_source_with(options: crate::Options, source: &[u8]) -> (Vec<u8>, bool) {
    let mut vm = crate::Vm::new(options);
    let outcome = vm.run_source(b"=group", source);
    let text = match outcome {
        Ok(()) => Vec::new(),
        Err(error) => vm.error_message(&error),
    };
    let compiled = vm.heap.protos.iter().any(|(_, proto)| proto.jit.is_compiled());
    let mut output = vm.take_output();
    output.extend_from_slice(&text);
    (output, compiled)
}

fn poke(vm: &mut crate::Vm, _arguments: &[crate::Value]) -> Result<crate::Values, crate::LuaError> {
    vm.request_pause();
    Ok(crate::Values::new())
}

#[test]
fn a_pause_inside_a_compiled_loop_survives_a_snapshot() {
    for mode in [["force=baseline"], ["force=peak"]] {
        pause_and_snapshot(options_for(&mode));
    }
}

fn pause_and_snapshot(options: crate::Options) {
    let mut vm = crate::Vm::new(options.clone());
    let poke_function = vm.register_native("poke", poke).expect("the name is free");
    vm.set_global("poke", poke_function);
    let chunk = vm
        .load(b"=loop", b"local acc = 0 for i = 1, 200 do acc = acc + i if i == 50 then poke() end end print(acc)")
        .expect("the chunk compiles");
    let outcome = vm.start(chunk, &[]).expect("the loop starts");
    assert!(matches!(outcome, crate::Outcome::Paused), "the request pauses the loop");
    assert!(vm.heap.protos.iter().any(|(_, proto)| proto.jit.is_compiled()), "the loop ran compiled");
    let bytes = vm.snapshot().expect("the paused VM is at an instruction boundary");

    let mut host = crate::Host::captured();
    host.register_native("poke", poke).expect("the name is free");
    let mut restored = crate::Vm::restore(&bytes, options, host).expect("the host provides poke");
    let finished = restored.resume().expect("the restored loop resumes");
    assert!(matches!(finished, crate::Outcome::Finished(_)));
    assert_eq!(restored.take_output(), b"20100\n");
}

#[test]
fn a_restored_vm_runs_in_the_tier_the_restoring_host_chooses() {
    let source: &[u8] = b"local function sum(n) local acc = 0 for i = 1, n do acc = acc + i end return acc end poke() print(sum(200))";
    let mut vm = crate::Vm::new(options_for(&["force=peak"]));
    let poke_function = vm.register_native("poke", poke).expect("the name is free");
    vm.set_global("poke", poke_function);
    let chunk = vm.load(b"=sum", source).expect("the chunk compiles");
    let outcome = vm.start(chunk, &[]).expect("the chunk starts");
    assert!(matches!(outcome, crate::Outcome::Paused), "poke pauses the chunk");
    let bytes = vm.snapshot().expect("the paused VM is at an instruction boundary");

    for (options, compiles) in [(options_for(&["tier=interpreter"]), false), (options_for(&["force=peak"]), true)] {
        let mut host = crate::Host::captured();
        host.register_native("poke", poke).expect("the name is free");
        let mut restored = crate::Vm::restore(&bytes, options, host).expect("the host provides poke");
        let finished = restored.resume().expect("the restored chunk resumes");
        assert!(matches!(finished, crate::Outcome::Finished(_)));
        assert_eq!(restored.take_output(), b"20100\n");
        let compiled = restored.heap.protos.iter().any(|(_, proto)| proto.jit.is_compiled());
        assert_eq!(compiled, compiles, "the restoring host's tier decides whether sum compiles");
    }
}

fn compiled_ir_size(statements: usize) -> usize {
    let mut source = String::new();
    for _ in 0..statements {
        source.push_str("aaa = aaa + 1\n");
    }
    let mut vm = crate::Vm::new(crate::Options::default());
    let chunk = vm.load(b"=big", source.as_bytes()).expect("the chunk loads");
    let crate::Value::Closure(closure) = chunk else {
        panic!("a chunk is a Lua function");
    };
    let proto = vm.heap.closures.live(closure).proto;
    let budget = crate::jit::CodeBudget::new(64 * 1024 * 1024);
    let facts = crate::jit::CompileFacts {
        layout: crate::jit::TableLayout::probe(),
        frames: crate::jit::FrameLayout::probe(),
        calls: Vec::new(),
        call_depth_limit: 0,
        interned: Vec::new(),
        inline_caches: true,
    };
    crate::jit::compile(&mut compiler(), budget, vm.heap.protos.live(proto), &facts)
        .expect("the chunk compiles")
        .ir_insts()
}

#[test]
fn peak_compile_work_grows_linearly_with_the_proto() {
    let small = compiled_ir_size(500);
    let large = compiled_ir_size(5000);
    assert!(
        large <= small * 12,
        "10x the statements may not cost more than 12x the IR: {small} instructions for 500, {large} for 5000"
    );
}

thread_local! {
    static PROBED_FRAMES: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn probe_frame(_vm: &mut crate::Vm, _arguments: &[crate::Value]) -> Result<crate::Values, crate::LuaError> {
    let marker = 0u8;
    let address = std::hint::black_box(&marker) as *const u8 as usize;
    PROBED_FRAMES.with(|frames| frames.borrow_mut().push(address));
    Ok(crate::Values::new())
}

/// Measures the native stack one level of compiled Lua nesting costs, to size `JIT_CALL_STACK`.
#[test]
#[ignore = "measures the native stack per compiled call level; run with --release --ignored"]
fn native_stack_per_compiled_call_level() {
    let options = crate::Options {
        jit_policy: crate::JitPolicy::forced(),
        ..crate::Options::default()
    };
    let mut vm = crate::Vm::new(options);
    let probe = vm.register_native("probe", probe_frame).expect("the name is free");
    vm.set_global("probe", probe);
    PROBED_FRAMES.with(|frames| frames.borrow_mut().clear());
    vm.run_source(
        b"=depth",
        b"local function f(n) probe() if n == 0 then return 0 end local r = f(n - 1) return r end f(60)",
    )
    .expect("the recursion runs");
    let frames = PROBED_FRAMES.with(|frames| frames.borrow().clone());
    let deltas: Vec<usize> = frames.windows(2).map(|pair| pair[0].abs_diff(pair[1])).collect();
    let steady = deltas[10..40].to_vec();
    let per_level = *steady.iter().max().expect("the recursion probed every level");
    let min_level = *steady.iter().min().expect("the recursion probed every level");
    eprintln!("NATIVE-STACK per level max={per_level} min={min_level} over {} levels", steady.len());
}

#[test]
fn the_table_layout_probe_finds_every_field() {
    let layout = crate::jit::TableLayout::probe().expect("the host's table layout is found");
    let table = crate::table::Table::default();
    assert!(layout.size >= 64 && layout.slots >= 0 && layout.metatable_tag >= 0);
    assert_ne!(layout.shaped, layout.shape);
    drop(table);
}

/// Compiles after the sites ran long enough to fill their inline caches.
const WARM: crate::JitPolicy = crate::JitPolicy {
    baseline: crate::TierPolicy {
        budget_base: 3000,
        ..crate::TierPolicy::immediate()
    },
    peak: crate::TierPolicy {
        budget_base: 3000,
        ..crate::TierPolicy::immediate()
    },
    ..crate::JitPolicy::forced()
};

const FIELD_PROGRAM: &[u8] = b"
collectgarbage('incremental')
local Point = {}
Point.__index = Point
function Point.new(x, y) return setmetatable({x = x, y = y}, Point) end
function Point:len() return self.x * self.x + self.y * self.y end
local Other = {}
Other.__index = Other
function Other:len() return -1 end
local function walk(n, list)
  local total = 0
  for i = 1, n do
    local p = list[i % #list + 1]
    total = total + p.x + (p.y or 0) + (p.z or 0) + p:len()
    p.x = p.x + 1
    p.name = 'n' .. i
    if p.missing then total = total + 1000 end
  end
  return total
end
local function calls(n, points)
  local total = 0
  for i = 1, n do total = total + points[i % #points + 1]:len() end
  return total
end
local points = { Point.new(1, 2), Point.new(2, 3), setmetatable({y = 1, x = 2}, Point) }
print(calls(3000, points))
function Point:len() return self.x end
print(calls(100, points))
points[4] = setmetatable({x = 4, y = 4}, Other)
print(calls(100, points))
Point.len = nil
print(pcall(calls, 100, points))
local plain = {x = 1, y = 2}
local list = { Point.new(1, 2), Point.new(3, 4), plain }
plain.len = function() return 7 end
print(walk(2000, list))
list[4] = {y = 5, x = 6, len = function() return 9 end}
list[5] = setmetatable({x = 1, y = 1}, Other)
print(walk(2000, list))
function Point:len() return 100 end
print(walk(500, list))
setmetatable(list[1], Other)
list[2].y = nil
list[2].y = 8
print(walk(500, list))
local dict = Point.new(1, 2)
for i = 1, 200 do dict['k' .. i] = i end
for i = 1, 200 do dict['k' .. i] = nil end
list[3] = dict
print(walk(500, list))
rawset(Point, 'missing', nil)
setmetatable(list[1], {__index = function(_, key) if key == 'missing' then return true end end})
print(pcall(walk, 500, list))
for i = 1, 30 do collectgarbage('step', 1) print(walk(50, list)) end
print(list[1].name, list[2].name, list[4].name, list[5].name)
";

#[test]
fn inline_field_access_matches_the_interpreter() {
    let interpreted = run_source_with(options_for(INTERPRETER), FIELD_PROGRAM);
    for ceiling in ["tier=baseline", "tier=peak"] {
        let mut options = options_for(&[ceiling]);
        options.jit_policy = WARM;
        let warm = run_source_with(options, FIELD_PROGRAM);
        assert!(warm.1, "the warm run compiles the loops");
        assert_eq!(String::from_utf8_lossy(&warm.0), String::from_utf8_lossy(&interpreted.0), "{ceiling}");
    }
}
