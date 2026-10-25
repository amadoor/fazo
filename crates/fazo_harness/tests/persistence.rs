use std::{fs, path::Path};


use fazo::{Host, LuaError, Options, Outcome, Value, Values, Vm};
use fazo_harness::{CASES_DIR, generator};

const TICK: &str = "persistence.tick";
const IDLE: &str = "persistence.idle";
const CHUNK: &[u8] = b"@case";
const PAUSE_POINTS: u64 = 10;
const TARGETED_PREFIX: &str = "persist_";
const TARGETED_DISTANCES: u64 = 40;

fn tick(vm: &mut Vm, _arguments: &[Value]) -> Result<Values, LuaError> {
    vm.request_pause();
    Ok(Values::new())
}

fn idle(_vm: &mut Vm, _arguments: &[Value]) -> Result<Values, LuaError> {
    Ok(Values::new())
}

fn host() -> Host {
    let mut host = Host::captured();
    host.register_native(TICK, tick).expect("the tick native is registered once");
    host.register_native(IDLE, idle).expect("the idle native is registered once");
    host
}

fn global_name(native: &str) -> &'static str {
    if native == TICK { "tick" } else { "idle" }
}

#[derive(Debug, PartialEq, Eq)]
struct Run {
    output: Vec<u8>,
    error: Option<Vec<u8>>,
}

/// Runs `source` with a count hook calling `hook` every `every` instructions. The hook is the same
/// kind of native for the reference and the interrupted run, so the stack and collector see the
/// same work. A pause requested by the hook snapshots the VM and restores it on a new host, then
/// resumes the restored VM.
fn run_with_hook(source: &[u8], every: u64, hook: &str) -> Run {
    let mut program = format!("debug.sethook({}, '', {every}) ", global_name(hook)).into_bytes();
    program.extend_from_slice(source);
    let mut vm = Vm::with_host(Options::default(), host());
    for name in [TICK, IDLE] {
        let value = vm.native(name).expect("the natives are registered");
        vm.set_global(global_name(name), value);
    }
    let function = vm.load(CHUNK, &program).expect("the source compiles");
    let mut output = Vec::new();
    let mut outcome = vm.start(function, &[]);
    let error = loop {
        match outcome {
            Ok(Outcome::Finished(_)) => break None,
            Ok(Outcome::Paused) => {
                output.extend(vm.take_output());
                let bytes = vm.snapshot().expect("a paused VM snapshots");
                vm = Vm::restore(&bytes, Options::default(), host()).expect("the snapshot restores on a host with the same natives");
                outcome = vm.resume();
            }
            Err(error) => break Some(vm.error_message(&error)),
        }
    };
    output.extend(vm.take_output());
    Run { output, error }
}

/// The number of instructions `source` runs, counted by a Lua count hook that fires on every
/// instruction. The count only chooses where the sweep pauses.
fn instruction_count(source: &[u8]) -> u64 {
    let mut program = b"counted = 0 debug.sethook(function() counted = counted + 1 end, '', 1) ".to_vec();
    program.extend_from_slice(source);
    let mut vm = Vm::with_host(Options::default(), host());
    let function = vm.load(CHUNK, &program).expect("the source compiles");
    let _ = vm.start(function, &[]);
    match vm.global("counted") {
        Value::Integer(count) => u64::try_from(count).unwrap_or(0),
        _ => 0,
    }
}

fn check_resumes_identically(name: &str, source: &[u8], every: u64) -> Result<(), String> {
    let expected = run_with_hook(source, every, IDLE);
    let actual = run_with_hook(source, every, TICK);
    if actual.output != expected.output {
        return Err(format!(
            "{name}: output differs after pausing every {every} instructions: {:?}; error {:?}",
            String::from_utf8_lossy(&actual.output),
            actual.error.as_deref().map(String::from_utf8_lossy)
        ));
    }
    if actual.error != expected.error {
        return Err(format!(
            "{name}: error differs after pausing every {every} instructions: {:?}",
            actual.error.as_deref().map(String::from_utf8_lossy)
        ));
    }
    Ok(())
}

fn sweep_distances(name: &str, instructions: u64) -> Vec<u64> {
    let mut distances: Vec<u64> = (1..=PAUSE_POINTS)
        .map(|point| (instructions * point / PAUSE_POINTS).max(1))
        .collect();
    if name.starts_with(TARGETED_PREFIX) {
        distances.extend(1..=TARGETED_DISTANCES);
    }
    distances.sort_unstable();
    distances.dedup();
    distances
}

fn sweep_case(file: &str) {
    let source = fs::read(Path::new(CASES_DIR).join(file)).expect("the case is readable");
    let instructions = instruction_count(&source);
    let mut failures = Vec::new();
    for every in sweep_distances(file, instructions) {
        failures.extend(check_resumes_identically(file, &source, every).err());
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

macro_rules! case_sweeps {
    ($($test:ident => $file:literal,)*) => {
        $(
            #[test]
            fn $test() {
                sweep_case(concat!($file, ".lua"));
            }
        )*
    };
}

case_sweeps! {
case_arithmetic_metamethods => "arithmetic_metamethods",
case_base_load_warn => "base_load_warn",
case_bitwise_metamethods => "bitwise_metamethods",
case_call_metamethod => "call_metamethod",
case_close_variables => "close_variables",
case_close_variables_order => "close_variables_order",
case_close_yield_during_error => "close_yield_during_error",
case_closures_in_loops => "closures_in_loops",
case_closures_shared_upvalues => "closures_shared_upvalues",
case_concat_len_unm_metamethods => "concat_len_unm_metamethods",
case_const_variables => "const_variables",
case_coroutine_basics => "coroutine_basics",
case_coroutine_errors_close => "coroutine_errors_close",
case_coroutine_yield_across => "coroutine_yield_across",
case_coroutines_basic => "coroutines_basic",
case_coroutines_errors => "coroutines_errors",
case_debug_upvalues_locals => "debug_upvalues_locals",
case_eq_lt_le_metamethods => "eq_lt_le_metamethods",
case_err_arith_field => "err_arith_field",
case_err_arith_global => "err_arith_global",
case_err_arith_local => "err_arith_local",
case_err_arith_upvalue => "err_arith_upvalue",
case_err_assert_default => "err_assert_default",
case_err_assert_message => "err_assert_message",
case_err_bitwise_float_local => "err_bitwise_float_local",
case_err_bitwise_string_constant => "err_bitwise_string_constant",
case_err_call_field => "err_call_field",
case_err_call_global => "err_call_global",
case_err_call_method => "err_call_method",
case_err_call_upvalue => "err_call_upvalue",
case_err_compare_number_nil => "err_compare_number_nil",
case_err_compare_two_tables => "err_compare_two_tables",
case_err_concat_nil_middle => "err_concat_nil_middle",
case_err_concat_table_local => "err_concat_table_local",
case_err_error_level_two => "err_error_level_two",
case_err_error_level_zero => "err_error_level_zero",
case_err_error_table_value => "err_error_table_value",
case_err_for_initial_nil => "err_for_initial_nil",
case_err_for_limit_string => "err_for_limit_string",
case_err_for_step_zero => "err_for_step_zero",
case_err_index_chain_loop => "err_index_chain_loop",
case_err_index_global_nil => "err_index_global_nil",
case_err_index_nested_field => "err_index_nested_field",
case_err_index_nil_local => "err_index_nil_local",
case_err_index_number_local => "err_index_number_local",
case_err_index_upvalue_store => "err_index_upvalue_store",
case_err_intdiv_zero => "err_intdiv_zero",
case_err_len_nil_local => "err_len_nil_local",
case_err_mod_zero => "err_mod_zero",
case_err_nested_call_result => "err_nested_call_result",
case_err_newindex_nil_local => "err_newindex_nil_local",
case_err_pattern_malformed => "err_pattern_malformed",
case_err_select_zero => "err_select_zero",
case_err_stack_overflow => "err_stack_overflow",
case_err_string_arith => "err_string_arith",
case_err_string_dump => "err_string_dump",
case_err_table_arith_plain => "err_table_arith_plain",
case_err_table_index_nan_key => "err_table_index_nan_key",
case_err_table_index_nil_key => "err_table_index_nil_key",
case_err_tonumber_base_range => "err_tonumber_base_range",
case_err_unary_minus_table => "err_unary_minus_table",
case_float_modulo_signs => "float_modulo_signs",
case_float_tostring => "float_tostring",
case_goto_labels => "goto_labels",
case_index_chain => "index_chain",
case_int_float_types => "int_float_types",
case_integer_conversions => "integer_conversions",
case_integer_division_by_zero => "integer_division_by_zero",
case_integer_overflow => "integer_overflow",
case_ipairs_wraparound => "ipairs_wraparound",
case_length_of_unique_borders => "length_of_unique_borders",
case_load_chunk_names_are_bytes => "load_chunk_names_are_bytes",
case_math_functions => "math_functions",
case_math_random => "math_random",
case_metamethod_error_names => "metamethod_error_names",
case_metatable_changes => "metatable_changes",
case_mixed_comparison => "mixed_comparison",
case_mm_call_arith => "mm_call_arith",
case_mm_compare_eq => "mm_compare_eq",
case_mm_index_newindex => "mm_index_newindex",
case_mm_tostring_name => "mm_tostring_name",
case_modulo_floor_division => "modulo_floor_division",
case_multiple_assignment_order => "multiple_assignment_order",
case_multiple_returns_tailcalls => "multiple_returns_tailcalls",
case_newindex => "newindex",
case_numeric_for => "numeric_for",
case_pcall_error_levels => "pcall_error_levels",
case_pcall_error_values => "pcall_error_values",
case_pcall_non_callable => "pcall_non_callable",
case_persist_close => "persist_close",
case_persist_coroutine => "persist_coroutine",
case_persist_coroutine_running_error => "persist_coroutine_running_error",
case_persist_coroutine_running_pause => "persist_coroutine_running_pause",
case_persist_metamethod => "persist_metamethod",
case_persist_pcall => "persist_pcall",
case_persist_stepped_gsub => "persist_stepped_gsub",
case_persist_stepped_sort => "persist_stepped_sort",
case_persist_weak_finalizers => "persist_weak_finalizers",
case_string_arith_coercion => "string_arith_coercion",
case_string_basics => "string_basics",
case_string_format => "string_format",
case_string_format_a_modifiers => "string_format_a_modifiers",
case_string_format_basics => "string_format_basics",
case_string_number_coercion => "string_number_coercion",
case_string_pack => "string_pack",
case_string_patterns => "string_patterns",
case_string_patterns_basics => "string_patterns_basics",
case_table_insert_remove => "table_insert_remove",
case_table_metamethods => "table_metamethods",
case_table_move_sort => "table_move_sort",
case_utf8_library => "utf8_library",
case_varargs_edges => "varargs_edges",
case_varargs_select => "varargs_select",
case_xpcall_handlers => "xpcall_handlers",
}

#[test]
fn two_hundred_generated_programs_resume_identically_after_pauses() {
    let mut failures = Vec::new();
    for seed in 0..200u64 {
        let source = generator::program(seed).into_bytes();
        let every = 500 + (seed * 7919) % 3500;
        failures.extend(check_resumes_identically(&format!("generated seed {seed}"), &source, every).err());
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn restore_rejects_a_snapshot_that_uses_a_native_the_host_lacks() {
    let mut vm = Vm::with_host(Options::default(), host());
    let tick_value = vm.native(TICK).expect("the tick native is registered");
    vm.set_global("tick", tick_value);
    let bytes = vm.snapshot().expect("an idle VM snapshots");
    let Err(error) = Vm::restore(&bytes, Options::default(), Host::captured()) else {
        panic!("the restore must fail");
    };
    assert_eq!(error.to_string(), "the snapshot uses the native function `persistence.tick`, which the restoring host does not provide");
}

#[test]
fn snapshot_refuses_while_a_host_file_is_open() {
    let mut options = Options::default();
    options.io = true;
    let mut host = Host::captured();
    host.files = Box::new(fazo::MemoryFiles::default());
    let mut vm = Vm::with_host(options, host);
    vm.run_source(CHUNK, b"kept = io.open('data.txt', 'w')").expect("the file opens");
    let error = vm.snapshot().expect_err("the snapshot must refuse an open file");
    assert_eq!(error.to_string(), "cannot snapshot the VM while a file it opened is open");
}

#[test]
fn memory_limit_raises_not_enough_memory_and_message_handlers_do_not_see_it() {
    let mut options = Options::default();
    options.memory_limit = Some(2_000_000);
    let mut vm = Vm::new(options);
    let grow = b"local t = {} for i = 1, 10000000 do t[i] = {i} end";
    let error = vm.run_source(CHUNK, grow).expect_err("the run must exceed the limit");
    assert_eq!(vm.error_message(&error), b"not enough memory");

    let mut options = Options::default();
    options.memory_limit = Some(2_000_000);
    let mut vm = Vm::new(options);
    vm.run_source(
        CHUNK,
        b"print(xpcall(function() local t = {} for i = 1, 10000000 do t[i] = {i} end end, function(m) return 'handled: ' .. m end))",
    )
    .expect("xpcall returns the error as a value");
    assert_eq!(vm.take_output(), b"false\tnot enough memory\n");
}
