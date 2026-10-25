use crate::{Host, LuaError, Options, Value, Values, Vm, values};

fn probe(_vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    Ok(values![args[0], Value::Integer(2), Value::Nil])
}

fn report(vm: &mut Vm, _args: &[Value]) -> Result<Values, LuaError> {
    let data = vm.running_native_data().unwrap_or(0);
    Ok(values![Value::Integer(i64::from(data))])
}

type BufferState = (usize, Vec<(usize, bool)>, usize, usize);

fn buffer_state(vm: &Vm) -> BufferState {
    let buffers = vm.argument_pool.iter().map(|buffer| (buffer.capacity(), buffer.spilled())).collect();
    (vm.argument_pool.len(), buffers, vm.thread().frames.capacity(), vm.thread().stack.capacity())
}

#[test]
fn a_native_call_reuses_its_argument_buffer() {
    let mut vm = Vm::new(Options::default());
    let function = vm.register_native("probe", probe).expect("the name is free");
    vm.set_global("probe", function);
    let program = b"for i = 1, 1000 do local a, b, c = probe(i, i) assert(b == 2 and c == nil) end";
    vm.run_source(b"=probe", program).expect("the first run succeeds");
    let warm = buffer_state(&vm);
    vm.run_source(b"=probe", program).expect("the second run succeeds");
    assert_eq!(buffer_state(&vm), warm);
    assert!(warm.1.iter().all(|&(capacity, spilled)| capacity == 4 && !spilled));
}

#[test]
fn a_native_sees_the_data_it_was_registered_with() {
    let mut vm = Vm::new(Options::default());
    let function = vm.register_native_with_data("report", report, 7).expect("the name is free");
    vm.set_global("report", function);
    vm.run_source(b"=report", b"assert(report() == 7)").expect("the data is 7");
}

#[test]
fn a_restored_vm_takes_the_data_its_host_registers() {
    let mut host = Host::captured();
    host.register_native_with_data("report", report, 9).expect("the name is free");
    let mut vm = Vm::with_host(Options::default(), host);
    let function = vm.native("report").expect("the host provides report");
    vm.set_global("report", function);
    let bytes = vm.snapshot().expect("the VM is at an instruction boundary");

    let mut host = Host::captured();
    host.register_native_with_data("report", report, 11).expect("the name is free");
    let mut restored = Vm::restore(&bytes, Options::default(), host).expect("the host provides every native");
    restored.run_source(b"=report", b"assert(report() == 11)").expect("the restored data is 11");
}

fn assert_interrupted(source: &[u8]) {
    let mut vm = Vm::new(Options::default());
    vm.request_interrupt();
    let error = vm.run_source(b"=hostile", source).expect_err("the native stops at its interrupt check");
    assert_eq!(vm.error_message(&error), b"interrupted");
    vm.run_source(b"=after", b"assert(string.find('a', 'a'))").expect("the request was taken, so the VM runs on");
}

#[test]
fn pattern_find_stops_on_backtracking_blowup() {
    assert_interrupted(b"string.find(string.rep('a', 30), 'a*a*a*a*a*b')");
}

#[test]
fn pattern_match_stops_on_long_subject() {
    assert_interrupted(b"string.match(string.rep('a', 1 << 20), '.-b')");
}

#[test]
fn gmatch_stops_between_matches() {
    assert_interrupted(b"for word in string.gmatch(string.rep('a', 1 << 20), 'a') do end");
}

#[test]
fn gsub_stops_on_long_subject() {
    assert_interrupted(b"string.gsub(string.rep('a', 1 << 20), 'a', 'b')");
}

#[test]
fn string_rep_stops_on_many_copies() {
    assert_interrupted(b"string.rep('a', 1 << 24)");
}

#[test]
fn string_format_stops_on_long_literal() {
    assert_interrupted(b"string.format(string.rep('x', 1 << 22))");
}

#[test]
fn string_byte_stops_on_long_range() {
    assert_interrupted(b"string.byte(string.rep('a', 1 << 20), 1, -1)");
}

#[test]
fn table_sort_stops_on_long_array() {
    assert_interrupted(b"local t = {} for i = 1, 1 << 18 do t[i] = -i end table.sort(t)");
}

#[test]
fn table_concat_stops_on_long_array() {
    assert_interrupted(b"local t = {} for i = 1, 1 << 18 do t[i] = 'x' end table.concat(t)");
}

#[test]
fn utf8_len_stops_on_long_string() {
    assert_interrupted(b"utf8.len(string.rep('a', 1 << 20))");
}

#[test]
fn utf8_codepoint_stops_on_long_string() {
    assert_interrupted(b"utf8.codepoint(string.rep('a', 1 << 20), 1, -1)");
}

#[test]
fn utf8_offset_stops_on_long_string() {
    assert_interrupted(b"utf8.offset(string.rep('a', 1 << 20), 1 << 20)");
}

#[test]
fn utf8_codes_stops_at_each_step() {
    assert_interrupted(b"for p, c in utf8.codes(string.rep('a', 1 << 20)) do end");
}

#[test]
fn string_pack_stops_on_long_format() {
    assert_interrupted(b"string.pack(string.rep('x', 1 << 20))");
}

#[test]
fn load_stops_on_long_chunk() {
    assert_interrupted(b"load(string.rep('a = 1;', 1 << 18))");
}

#[test]
fn pcall_cannot_catch_an_interrupt() {
    let mut vm = Vm::new(Options::default());
    vm.request_interrupt();
    let error = vm
        .run_source(b"=hostile", b"pcall(string.rep, 'a', 1 << 24)")
        .expect_err("the interrupt passes through pcall");
    assert_eq!(vm.error_message(&error), b"interrupted");
}
