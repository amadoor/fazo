use crate::{
    Host, Options, Vm,
    gc,
    shape::{MAX_DYNAMIC_TRANSITIONS, MAX_TRANSITIONS},
};

const BUILD: &[u8] = b"t = {} t.zeta = 1 t.alpha = 2 t.mid = 3 t.zeta = nil t.zeta = 4 t.beta = 5";
const LIST: &[u8] = b"local keys = {} for k in pairs(t) do keys[#keys + 1] = k end print(table.concat(keys, ' '))";
const INSERTION_ORDER: &[u8] = b"zeta alpha mid beta
";

fn list_keys(vm: &mut Vm) -> Vec<u8> {
    vm.run_source(b"=pairs", LIST).expect("pairs runs");
    vm.take_output()
}

#[test]
fn pairs_visits_shaped_string_keys_in_insertion_order() {
    let mut vm = Vm::new(Options::default());
    vm.run_source(b"=build", BUILD).expect("the table builds");
    assert_eq!(list_keys(&mut vm), INSERTION_ORDER);
}

#[test]
fn a_restored_table_keeps_its_pairs_order() {
    let mut vm = Vm::new(Options::default());
    vm.run_source(b"=build", BUILD).expect("the table builds");
    let bytes = vm.snapshot().expect("the VM is at an instruction boundary");
    let mut restored = Vm::restore(&bytes, Options::default(), Host::captured()).expect("the snapshot restores");
    assert_eq!(list_keys(&mut restored), INSERTION_ORDER);
}

#[test]
fn a_dictionary_table_lists_the_same_keys_in_the_same_order_on_every_run() {
    let build = b"big = {} for i = 1, 100 do big['key' .. i] = i end";
    let list = b"local keys = {} for k in pairs(big) do keys[#keys + 1] = k end print(table.concat(keys, ' '))";
    let mut first = Vm::new(Options::default());
    first.run_source(b"=build", build).expect("the table builds");
    let mut second = Vm::new(Options::default());
    second.run_source(b"=build", build).expect("the table builds");
    first.run_source(b"=pairs", list).expect("pairs runs");
    second.run_source(b"=pairs", list).expect("pairs runs");
    let order = first.take_output();
    assert_eq!(order.split(|&byte| byte == b' ').count(), 100);
    assert_eq!(order, second.take_output());
}

#[test]
fn interning_a_dead_unswept_string_revives_it() {
    let mut vm = Vm::new(Options::default());
    let heap = &mut vm.heap;
    let string = heap.string(b"revived");
    let marks = heap.strings.marks(string).expect("the string is alive");
    heap.strings.set_marks(string, (marks & !gc::GC_BITS) | heap.gc.other_white());
    assert_eq!(heap.string(b"revived"), string);
    let marks = heap.strings.marks(string).expect("the string is still alive");
    assert_eq!(marks & heap.gc.other_white(), 0);
    assert_eq!(marks & gc::WHITE_BITS, heap.gc.current_white);
}

fn shape_count(vm: &Vm) -> usize {
    vm.heap.shapes.iter().count()
}

#[test]
fn a_table_creates_no_more_shapes_from_dynamic_keys_than_its_budget() {
    let mut vm = Vm::new(Options::default());
    vm.run_source(b"=setup", b"keep = {} collectgarbage() collectgarbage() collectgarbage('stop')").expect("setup runs");
    let before = shape_count(&vm);
    vm.run_source(b"=dynamic", b"for j = 1, 30 do keep['unshared' .. j] = j end").expect("the table fills");
    assert_eq!(shape_count(&vm) - before, MAX_DYNAMIC_TRANSITIONS as usize);
}

#[test]
fn a_transition_node_at_its_fan_out_limit_refuses_new_dynamic_children() {
    let mut vm = Vm::new(Options::default());
    let fill = format!("holders = {{}} for i = 1, {MAX_TRANSITIONS} do local t = {{}} t['first' .. i] = i holders[i] = t end");
    vm.run_source(b"=fill", fill.as_bytes()).expect("the root fills");
    let before = shape_count(&vm);
    vm.run_source(b"=refused", b"local t = {} local k = 'fresh' t[k] = 1 holders[k] = t").expect("the store runs");
    assert_eq!(shape_count(&vm), before);
}

#[test]
fn constant_keys_are_not_refused_by_the_fan_out_limit() {
    let mut vm = Vm::new(Options::default());
    let fill = format!("holders = {{}} for i = 1, {MAX_TRANSITIONS} do local t = {{}} t['first' .. i] = i holders[i] = t end");
    vm.run_source(b"=fill", fill.as_bytes()).expect("the root fills");
    let before = shape_count(&vm);
    vm.run_source(b"=constant", b"local t = {} t.fresh = 1 holders.kept = t").expect("the store runs");
    assert!(shape_count(&vm) > before);
}

#[test]
fn a_restored_wide_shape_finds_its_keys() {
    let build: String = (1..=12).map(|i| format!("t.k{i} = {i} ")).collect();
    let mut vm = Vm::new(Options::default());
    vm.run_source(b"=build", format!("t = {{}} {build}").as_bytes()).expect("the table builds");
    let bytes = vm.snapshot().expect("the VM is at an instruction boundary");
    let mut restored = Vm::restore(&bytes, Options::default(), Host::captured()).expect("the snapshot restores");
    restored
        .run_source(b"=lookup", b"print(t.k3, t['k12'], t['k13'])")
        .expect("the lookup runs");
    assert_eq!(restored.take_output(), b"3\t12\tnil\n");
}

#[test]
fn collecting_dynamic_tables_releases_their_shapes() {
    let mut vm = Vm::new(Options::default());
    vm.run_source(b"=setup", b"keep = {} collectgarbage() collectgarbage()").expect("setup runs");
    let before = shape_count(&vm);
    vm.run_source(
        b"=churn",
        b"for i = 1, 200 do local t = {} for j = 1, 3 do t['c' .. i .. '_' .. j] = j end end collectgarbage() collectgarbage()",
    )
    .expect("the churn runs");
    assert_eq!(shape_count(&vm), before);
}

#[test]
fn a_table_entering_a_shape_the_collector_found_dead_keeps_it() {
    let source = b"
        collectgarbage('incremental')
        local orders = { 'a b c', 'b c a', 'c a b' }
        local function build(kind, i)
            local first, second, third = orders[kind]:match('(%a) (%a) (%a)')
            local t = {}
            t[first] = i
            t[second] = i
            t[third] = i
            return t
        end
        local kept = {}
        local wrong = 0
        for i = 1, 20000 do
            kept[i % 30 + 1] = build((i // 37) % 3 + 1, i)
            collectgarbage('step', 10)
            if i % 10 == 0 then
                for _, t in ipairs(kept) do
                    local sum = 0
                    for _, value in pairs(t) do sum = sum + value end
                    if sum ~= t.a + t.b + t.c then wrong = wrong + 1 end
                end
            end
        end
        print(wrong)
    ";
    let mut vm = Vm::new(Options::default());
    vm.run_source(b"=revive", source).expect("the tables keep their shapes");
    assert_eq!(vm.take_output(), b"0\n");
}
