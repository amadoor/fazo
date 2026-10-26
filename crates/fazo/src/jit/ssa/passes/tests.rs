use super::super::{
    build, check,
    ir::{BinOp, BlockId, BuiltinFn, Cc, Const, Counted, Edge, FrameState, Func, Loop, NumOp, Op, Rep, StateId, Term, Type, UnOp, Val},
};
use super::{bounds, dce, escape, fold, forward, hoist, inline, repr, types};
use crate::{arena::Ref, proto::Proto};

fn state(func: &mut Func, pc: u32) -> StateId {
    func.add_state(FrameState { pc, frames: Vec::new(), slots: Vec::new(), site: pc, lazy: false, objects: Vec::new() })
}

fn constant(func: &mut Func, block: BlockId, constant: Const) -> Val {
    let rep = match constant {
        Const::Int(_) => Rep::Int,
        Const::Float(_) => Rep::Float,
        Const::Bool(_) => Rep::Bool,
        _ => Rep::Boxed,
    };
    func.append(block, Op::Const(constant), constant.ty(), rep)
}

fn straight_line() -> (Func, BlockId) {
    let mut func = Func::default();
    let block = func.new_block();
    func.entry = Some(block);
    (func, block)
}

fn ops(func: &Func, block: BlockId) -> Vec<Op> {
    func.block(block).insts.iter().map(|&value| func.op(value).clone()).collect()
}

fn returns(func: &mut Func, block: BlockId, value: Val) {
    let boxed = func.append(block, Op::Unary(UnOp::Box, value), func.ty(value), Rep::Boxed);
    func.append(block, Op::StoreReg { slot: 0, value: boxed }, Type::NONE, Rep::None);
    func.block_mut(block).term = Term::Return { pc: 9, first: 0, count: 1, close: false };
}

/// `for i = 1, 10 do r5 = t[i] end` with `t` read from register 0 before the loop.
struct CountedLoop {
    func: Func,
    header: BlockId,
    preheader: BlockId,
    guard: Val,
    index: Val,
}

fn counted_loop() -> CountedLoop {
    let mut func = Func::default();
    let entry = func.new_block();
    let preheader = func.new_block();
    let header = func.new_block();
    let latch = func.new_block();
    let exit = func.new_block();
    func.entry = Some(entry);
    let entry_state = state(&mut func, 3);
    let body_state = state(&mut func, 4);
    let table = func.append(entry, Op::LoadReg { slot: 0 }, Type::ANY, Rep::Boxed);
    func.block_mut(entry).term = Term::Jump(Edge::to(preheader));
    let start = constant(&mut func, preheader, Const::Int(1));
    let count = constant(&mut func, preheader, Const::Int(9));
    func.block_mut(preheader).term = Term::Jump(Edge { block: header, args: vec![start, count] });
    let index = func.add_param(header, Type::INT, Rep::Int);
    let remaining = func.add_param(header, Type::INT, Rep::Int);
    let guard = func.append(header, Op::GuardType { value: table, ty: Type::TABLE, state: body_state }, Type::TABLE, Rep::Payload);
    let element = func.append(header, Op::Index { object: guard, key: index, state: body_state }, Type::ANY, Rep::Boxed);
    func.append(header, Op::StoreReg { slot: 5, value: element }, Type::NONE, Rep::None);
    let zero = constant(&mut func, header, Const::Int(0));
    let done = func.append(header, Op::Binary(BinOp::ICmp(Cc::Eq), remaining, zero), Type::BOOL, Rep::Bool);
    func.block_mut(header).term = Term::Branch { cond: done, then: Edge::to(exit), other: Edge::to(latch) };
    let one = constant(&mut func, latch, Const::Int(1));
    let next_count = func.append(latch, Op::Binary(BinOp::ISub, remaining, one), Type::INT, Rep::Int);
    let next_index = func.append(latch, Op::Binary(BinOp::IAdd, index, one), Type::INT, Rep::Int);
    func.block_mut(latch).term = Term::Jump(Edge { block: header, args: vec![next_index, next_count] });
    func.block_mut(exit).term = Term::Return { pc: 9, first: 0, count: 0, close: false };
    func.loops.push(Loop {
        header,
        preheader,
        blocks: vec![header, latch],
        entry_state,
        counted: Some(Counted { index, count: remaining, step: 1 }),
    });
    CountedLoop { func, header, preheader, guard, index }
}

fn child_proto(vm: &crate::Vm, main: crate::Value, index: usize) -> Ref<Proto> {
    let crate::Value::Closure(closure) = main else {
        panic!("a chunk is a Lua function");
    };
    let main = vm.heap.closures.live(closure).proto;
    vm.heap.protos.live(main).protos[index]
}

/// The graph of the chunk's first function, built without feedback.
fn built(source: &str) -> Func {
    let mut vm = crate::Vm::new(crate::Options::default());
    let chunk = vm.load(b"=test", source.as_bytes()).expect("the chunk loads");
    let reference = child_proto(&vm, chunk, 0);
    let heap = &vm.heap;
    let interned = |_: crate::Value| false;
    let proto_of = |reference: Ref<Proto>| heap.protos.try_get(reference);
    let node_hint = |_: &Proto, _: usize, _: bool| None;
    let builtin_of = |_: Ref<crate::object::Native>| None;
    let facts = build::Facts {
        inline_caches: true,
        avoid: &[],
        interned: &interned,
        proto_of: &proto_of,
        policy: inline::Policy::default(),
        osr_at: None,
        node_hint: &node_hint,
        builtin_of: &builtin_of,
        running_floats: &[],
    };
    build::build(heap.protos.live(reference), &facts)
}

fn all_ops(func: &Func) -> Vec<Op> {
    func.reverse_postorder().into_iter().flat_map(|block| ops(func, block)).collect()
}

#[test]
fn an_empty_function_optimizes() {
    for pipeline in [super::PEAK, super::OPTIMIZING] {
        let mut func = Func::default();
        super::optimize(&mut func, pipeline, true).expect("an empty function stays valid");
    }
}

#[test]
fn the_builder_keeps_a_numeric_loop_in_block_parameters() {
    let mut func = built("local function sum(n) local s = 0 for i = 1, n do s = s + i end return s end");
    check::check(&func, false).expect("the built graph is valid");
    assert_eq!(func.loops.len(), 1, "one loop");
    assert!(func.loops[0].counted.is_some(), "the integer loop with a constant step is counted");
    assert!(!func.osr.is_empty(), "the loop header can be entered from the interpreter");
    super::optimize(&mut func, super::PEAK, true).expect("the passes keep the graph valid");
    let ops = all_ops(&func);
    assert!(ops.iter().any(|op| matches!(op, Op::Binary(BinOp::IAdd, ..))), "the sum is an integer add:\n{func}");
    assert!(!ops.iter().any(|op| matches!(op, Op::Arith { .. })), "no generic arithmetic is left:\n{func}");
}

#[test]
fn the_builder_leaves_a_call_without_feedback_as_a_call() {
    let func = built("local function f(g) return g(1) + 1 end");
    check::check(&func, false).expect("the built graph is valid");
    let ops = all_ops(&func);
    assert!(ops.iter().any(|op| matches!(op, Op::CallLua { .. })), "a call without callee feedback stays a call:\n{func}");
}

#[test]
fn type_propagation_specializes_integer_and_mixed_arithmetic() {
    let (mut func, block) = straight_line();
    let s = state(&mut func, 1);
    let two = constant(&mut func, block, Const::Int(2));
    let half = constant(&mut func, block, Const::Float(0.5));
    let sum = func.append(block, Op::Arith { op: NumOp::Add, left: two, right: two, state: s }, Type::NUM, Rep::Boxed);
    let mixed = func.append(block, Op::Arith { op: NumOp::Mul, left: sum, right: half, state: s }, Type::NUM, Rep::Boxed);
    returns(&mut func, block, mixed);
    types::run(&mut func);
    assert_eq!(func.op(sum), &Op::Binary(BinOp::IAdd, two, two));
    assert_eq!(func.ty(sum), Type::INT);
    assert!(matches!(func.op(mixed), Op::Binary(BinOp::FMul, _, right) if *right == half));
    assert_eq!(func.ty(mixed), Type::FLOAT);
    assert!(ops(&func, block).iter().any(|op| matches!(op, Op::Unary(UnOp::IToF, operand) if *operand == sum)), "the integer operand converts");
}

#[test]
fn type_propagation_guards_integer_division_by_zero_and_drops_proven_guards() {
    let (mut func, block) = straight_line();
    let s = state(&mut func, 1);
    let loaded = func.append(block, Op::LoadReg { slot: 1 }, Type::ANY, Rep::Boxed);
    let integer = func.append(block, Op::GuardType { value: loaded, ty: Type::INT, state: s }, Type::INT, Rep::Boxed);
    let again = func.append(block, Op::GuardType { value: integer, ty: Type::INT, state: s }, Type::INT, Rep::Boxed);
    let seven = constant(&mut func, block, Const::Int(7));
    let quotient = func.append(block, Op::Arith { op: NumOp::IDiv, left: seven, right: again, state: s }, Type::NUM, Rep::Boxed);
    returns(&mut func, block, quotient);
    types::run(&mut func);
    assert!(!func.block(block).insts.contains(&again), "a guard its input already passes is removed");
    assert_eq!(func.op(quotient), &Op::Binary(BinOp::IDiv, seven, integer));
    assert!(ops(&func, block).iter().any(|op| matches!(op, Op::GuardFalse { .. })), "a zero divisor deoptimizes");
}

#[test]
fn type_propagation_meets_loop_parameters_at_a_fixpoint() {
    let mut func = built("local function f(n) local s = 0 for i = 1, n do s = s + 0.5 end return s end");
    types::run(&mut func);
    fold::run(&mut func);
    types::run(&mut func);
    let header = func.loops[0].header;
    let params: Vec<Type> = func.block(header).params.iter().map(|&param| func.ty(param)).collect();
    assert!(params.contains(&Type::NUM), "a sum that starts as an integer and adds floats is a number: {params:?}\n{func}");
}

#[test]
fn folding_uses_lua_arithmetic() {
    let (mut func, block) = straight_line();
    let minus_seven = constant(&mut func, block, Const::Int(-7));
    let three = constant(&mut func, block, Const::Int(3));
    let modulo = func.append(block, Op::Binary(BinOp::IMod, minus_seven, three), Type::INT, Rep::Int);
    let quotient = func.append(block, Op::Binary(BinOp::IDiv, minus_seven, three), Type::INT, Rep::Int);
    let sum = func.append(block, Op::Binary(BinOp::IAdd, modulo, quotient), Type::INT, Rep::Int);
    returns(&mut func, block, sum);
    fold::run(&mut func);
    assert_eq!(func.op(modulo), &Op::Const(Const::Int(2)), "-7 % 3 is 2 in Lua");
    assert_eq!(func.op(quotient), &Op::Const(Const::Int(-3)), "-7 // 3 is -3 in Lua");
    assert_eq!(func.op(sum), &Op::Const(Const::Int(-1)));
}

#[test]
fn folding_keeps_integer_overflow_wrapping_and_float_division_exact() {
    let (mut func, block) = straight_line();
    let max = constant(&mut func, block, Const::Int(i64::MAX));
    let one = constant(&mut func, block, Const::Int(1));
    let wrapped = func.append(block, Op::Binary(BinOp::IAdd, max, one), Type::INT, Rep::Int);
    let x = constant(&mut func, block, Const::Float(1.0));
    let zero = constant(&mut func, block, Const::Float(-0.0));
    let infinite = func.append(block, Op::Binary(BinOp::FDiv, x, zero), Type::FLOAT, Rep::Float);
    let boxed = func.append(block, Op::Unary(UnOp::Box, infinite), Type::FLOAT, Rep::Boxed);
    func.append(block, Op::StoreReg { slot: 1, value: boxed }, Type::NONE, Rep::None);
    returns(&mut func, block, wrapped);
    fold::run(&mut func);
    assert_eq!(func.op(wrapped), &Op::Const(Const::Int(i64::MIN)));
    assert_eq!(func.op(infinite), &Op::Const(Const::Float(f64::NEG_INFINITY)));
}

#[test]
fn folding_removes_trivial_parameters_and_folds_branches() {
    let mut func = Func::default();
    let entry = func.new_block();
    let join = func.new_block();
    let left = func.new_block();
    let right = func.new_block();
    func.entry = Some(entry);
    let value = constant(&mut func, entry, Const::Int(4));
    let flag = constant(&mut func, entry, Const::Bool(true));
    func.block_mut(entry).term = Term::Branch { cond: flag, then: Edge::to(left), other: Edge::to(right) };
    func.block_mut(left).term = Term::Jump(Edge { block: join, args: vec![value] });
    func.block_mut(right).term = Term::Jump(Edge { block: join, args: vec![value] });
    let param = func.add_param(join, Type::INT, Rep::Int);
    returns(&mut func, join, param);
    fold::run(&mut func);
    assert!(func.block(join).params.is_empty(), "a parameter given one value everywhere is that value");
    assert!(!matches!(func.block(entry).term, Term::Branch { .. }), "a constant condition picks its branch");
    check::check(&func, false).expect("the graph stays valid");
}

#[test]
fn cse_merges_guards_but_not_across_a_store_that_moves_shapes() {
    let (mut func, block) = straight_line();
    let s = state(&mut func, 1);
    let loaded = func.append(block, Op::LoadReg { slot: 0 }, Type::ANY, Rep::Boxed);
    let table = func.append(block, Op::GuardType { value: loaded, ty: Type::TABLE, state: s }, Type::TABLE, Rep::Payload);
    let again = func.append(block, Op::GuardType { value: loaded, ty: Type::TABLE, state: s }, Type::TABLE, Rep::Payload);
    let first = func.append(block, Op::GuardShape { table, shape: 7, state: s }, Type::NONE, Rep::None);
    let second = func.append(block, Op::GuardShape { table: again, shape: 7, state: s }, Type::NONE, Rep::None);
    let key = constant(&mut func, block, Const::Int(1));
    func.append(block, Op::NewIndex { object: table, key, value: loaded, state: s }, Type::NONE, Rep::None);
    let third = func.append(block, Op::GuardShape { table, shape: 7, state: s }, Type::NONE, Rep::None);
    func.block_mut(block).term = Term::Return { pc: 2, first: 0, count: 0, close: false };
    fold::run(&mut func);
    let insts = &func.block(block).insts;
    assert!(!insts.contains(&again), "the same type guard is merged");
    assert!(insts.contains(&first) && !insts.contains(&second), "the same shape guard is merged");
    assert!(insts.contains(&third), "a store may change the shape, so the guard after it stays");
}

#[test]
fn representation_selection_unboxes_loop_parameters_and_boxes_at_stores() {
    let mut func = built("local function f(n) local s = 0 for i = 1, n do s = s + i end return s end");
    types::run(&mut func);
    fold::run(&mut func);
    types::run(&mut func);
    fold::run(&mut func);
    repr::run(&mut func);
    check::check(&func, true).expect("every operand has the representation its op reads");
    let header = func.loops[0].header;
    let reps: Vec<Rep> = func.block(header).params.iter().map(|&param| func.rep(param)).collect();
    assert!(reps.iter().all(|&rep| rep == Rep::Int), "every loop parameter is an unboxed integer: {reps:?}\n{func}");
    let ops = all_ops(&func);
    assert!(ops.iter().any(|op| matches!(op, Op::Unary(UnOp::Box, _))), "the returned sum is boxed for its register");
}

#[test]
fn hoisting_moves_invariant_guards_to_the_preheader_with_their_site() {
    let mut fixture = counted_loop();
    hoist::run(&mut fixture.func);
    check::check(&fixture.func, false).expect("the graph stays valid");
    assert!(fixture.func.block(fixture.preheader).insts.contains(&fixture.guard), "the type guard runs once per loop entry");
    let Some(state) = fixture.func.op(fixture.guard).state() else {
        panic!("a guard has a frame state");
    };
    let moved = fixture.func.state(state);
    assert_eq!((moved.pc, moved.site), (3, 4), "it deoptimizes to the loop entry and blames the access");
}

#[test]
fn hoisting_leaves_memory_guards_in_loops_that_call_out() {
    let mut fixture = counted_loop();
    let s = state(&mut fixture.func, 5);
    let shape = fixture.func.add_inst(Op::GuardShape { table: fixture.guard, shape: 3, state: s }, Type::NONE, Rep::None);
    let header = fixture.header;
    fixture.func.block_mut(header).insts.insert(1, shape);
    let call = fixture.func.add_inst(Op::CallLua { pc: 6, a: 0, b: 1, c: 1 }, Type::NONE, Rep::Word);
    fixture.func.block_mut(header).insts.insert(2, call);
    hoist::run(&mut fixture.func);
    assert!(fixture.func.block(fixture.preheader).insts.contains(&fixture.guard), "a type guard reads no memory");
    assert!(fixture.func.block(header).insts.contains(&shape), "a call may change the shape");
}

#[test]
fn bounds_checks_on_the_loop_index_become_one_range_guard() {
    let mut fixture = counted_loop();
    hoist::run(&mut fixture.func);
    bounds::run(&mut fixture.func);
    check::check(&fixture.func, false).expect("the graph stays valid");
    let preheader = ops(&fixture.func, fixture.preheader);
    assert!(preheader.iter().any(|op| matches!(op, Op::GuardRange { .. })), "the preheader checks the whole range");
    let body = ops(&fixture.func, fixture.header);
    assert!(
        body.iter().any(|op| matches!(op, Op::LoadArray { index, .. } if *index == fixture.index)),
        "the access reads the array part without a bounds check"
    );
}

#[test]
fn dead_code_elimination_removes_unused_values_and_self_feeding_parameters() {
    let mut fixture = counted_loop();
    let header = fixture.header;
    let unused = constant(&mut fixture.func, header, Const::Int(42));
    let mut func = fixture.func.clone();
    dce::run(&mut fixture.func);
    assert!(!fixture.func.block(header).insts.contains(&unused));
    assert_eq!(fixture.func.block(header).params.len(), 2, "both loop variables are read");
    let kept: Vec<Val> = func.block(header).insts.iter().copied().filter(|&value| !matches!(func.op(value), Op::StoreReg { .. } | Op::Index { .. })).collect();
    func.block_mut(header).insts = kept;
    dce::run(&mut func);
    assert_eq!(func.block(header).params.len(), 1, "the index only feeds itself once nothing reads it");
}

#[test]
fn the_inline_policy_takes_small_functions_without_loops() {
    let mut vm = crate::Vm::new(crate::Options::default());
    let chunk = vm
        .load(b"=test", b"local function leaf(o) return o.x + 1 end local function looped(f) for i = 1, 2 do f(i) end end".as_slice())
        .expect("the chunk loads");
    let leaf = child_proto(&vm, chunk, 0);
    let looped = child_proto(&vm, chunk, 1);
    let policy = inline::Policy::default();
    assert!(inline::candidate(vm.heap.protos.live(leaf), leaf, &policy).is_some(), "a small function inlines");
    assert!(inline::candidate(vm.heap.protos.live(looped), looped, &policy).is_none(), "a function with a loop does not");
}

#[test]
fn the_checker_reports_a_use_its_definition_does_not_dominate() {
    let mut func = Func::default();
    let entry = func.new_block();
    let left = func.new_block();
    let right = func.new_block();
    let join = func.new_block();
    func.entry = Some(entry);
    let flag = constant(&mut func, entry, Const::Bool(true));
    func.block_mut(entry).term = Term::Branch { cond: flag, then: Edge::to(left), other: Edge::to(right) };
    let only_left = constant(&mut func, left, Const::Int(1));
    func.block_mut(left).term = Term::Jump(Edge::to(join));
    func.block_mut(right).term = Term::Jump(Edge::to(join));
    returns(&mut func, join, only_left);
    assert!(check::check(&func, false).is_err());
}

fn fresh_table(func: &mut Func, block: BlockId, fields: &[Val], s: StateId) -> Val {
    let op = Op::NewTable { hash_log: 2, top: 4, shape: Some(9), fields: fields.into(), state: s, after: s };
    func.append(block, op, Type::TABLE, Rep::Payload)
}

fn stored(func: &Func, block: BlockId, slot: u32) -> Option<Val> {
    func.block(block).insts.iter().find_map(|&value| match func.op(value) {
        Op::StoreReg { slot: at, value } if *at == slot => Some(*value),
        _ => None,
    })
}

#[test]
fn load_forwarding_reads_constructed_and_stored_fields_and_drops_known_shape_guards() {
    let (mut func, block) = straight_line();
    let s = state(&mut func, 1);
    let x = constant(&mut func, block, Const::Int(3));
    let fresh = fresh_table(&mut func, block, &[x], s);
    let known = func.append(block, Op::GuardShape { table: fresh, shape: 9, state: s }, Type::NONE, Rep::None);
    let constructed = func.append(block, Op::LoadSlot { table: fresh, slot: 0 }, Type::ANY, Rep::Boxed);
    let loaded = func.append(block, Op::LoadReg { slot: 0 }, Type::ANY, Rep::Boxed);
    let other = func.append(block, Op::GuardType { value: loaded, ty: Type::TABLE, state: s }, Type::TABLE, Rep::Payload);
    func.append(block, Op::StoreSlot { table: other, slot: 1, value: x, state: s }, Type::NONE, Rep::None);
    let reloaded = func.append(block, Op::LoadSlot { table: other, slot: 1 }, Type::ANY, Rep::Boxed);
    let untouched = func.append(block, Op::LoadSlot { table: fresh, slot: 0 }, Type::ANY, Rep::Boxed);
    func.append(block, Op::StoreReg { slot: 1, value: constructed }, Type::NONE, Rep::None);
    func.append(block, Op::StoreReg { slot: 2, value: reloaded }, Type::NONE, Rep::None);
    func.append(block, Op::StoreReg { slot: 3, value: untouched }, Type::NONE, Rep::None);
    func.block_mut(block).term = Term::Return { pc: 2, first: 0, count: 0, close: false };
    forward::run(&mut func);
    assert!(!func.block(block).insts.contains(&known), "the fresh table's shape is known");
    assert_eq!(stored(&func, block, 1), Some(x), "a constructed field is read from the constructor");
    assert_eq!(stored(&func, block, 2), Some(x), "a stored field is read from the store");
    assert_eq!(stored(&func, block, 3), Some(x), "a store to another table leaves the fresh table's fields");
}

#[test]
fn load_forwarding_forgets_fields_across_calls() {
    let (mut func, block) = straight_line();
    let s = state(&mut func, 1);
    let loaded = func.append(block, Op::LoadReg { slot: 0 }, Type::ANY, Rep::Boxed);
    let table = func.append(block, Op::GuardType { value: loaded, ty: Type::TABLE, state: s }, Type::TABLE, Rep::Payload);
    let first = func.append(block, Op::LoadSlot { table, slot: 0 }, Type::ANY, Rep::Boxed);
    func.append(block, Op::CallLua { pc: 1, a: 2, b: 1, c: 1 }, Type::NONE, Rep::None);
    let second = func.append(block, Op::LoadSlot { table, slot: 0 }, Type::ANY, Rep::Boxed);
    func.append(block, Op::StoreReg { slot: 1, value: first }, Type::NONE, Rep::None);
    func.append(block, Op::StoreReg { slot: 2, value: second }, Type::NONE, Rep::None);
    func.block_mut(block).term = Term::Return { pc: 2, first: 0, count: 0, close: false };
    forward::run(&mut func);
    assert_eq!(stored(&func, block, 2), Some(second), "a call may store to the table");
}

#[test]
fn escape_analysis_sinks_a_table_only_frame_states_see() {
    let mut fixture = counted_loop();
    let func = &mut fixture.func;
    func.sink_allocations = true;
    let body = func.state(StateId(1)).clone();
    let x = func.append(fixture.header, Op::Unary(UnOp::Box, fixture.index), Type::INT, Rep::Boxed);
    let fresh = fresh_table(func, fixture.header, &[x], StateId(0));
    let meta = func.append(fixture.header, Op::LoadReg { slot: 1 }, Type::ANY, Rep::Boxed);
    let set = func.append(fixture.header, Op::Builtin { function: BuiltinFn::SetMetatable, args: Box::new([fresh, meta]), state: StateId(0) }, Type::TABLE, Rep::Boxed);
    let seen = func.add_state(FrameState { slots: vec![(6, set)], ..body });
    let guard = func.append(fixture.header, Op::GuardType { value: fixture.index, ty: Type::INT, state: seen }, Type::INT, Rep::Int);
    escape::run(func);
    let header = &func.block(fixture.header).insts;
    assert!(!header.contains(&fresh) && !header.contains(&set), "the allocation and setmetatable are gone");
    assert!(header.iter().any(|&value| matches!(func.op(value), Op::GuardAbsent { .. })), "the metatable is checked for __gc");
    let object = func.state(seen).object(fresh).expect("the state describes the table");
    assert_eq!(object.fields, vec![x]);
    assert!(object.meta.is_some(), "past setmetatable the table has its metatable");
    assert_eq!(func.state(seen).slots, vec![(6, fresh)]);
    assert!(header.contains(&guard));
    check::check(func, false).expect("the sunk graph stays valid");
}

#[test]
fn escape_analysis_keeps_a_table_stored_to_a_register() {
    let mut fixture = counted_loop();
    let func = &mut fixture.func;
    func.sink_allocations = true;
    let x = func.append(fixture.header, Op::Unary(UnOp::Box, fixture.index), Type::INT, Rep::Boxed);
    let fresh = fresh_table(func, fixture.header, &[x], StateId(0));
    func.append(fixture.header, Op::StoreReg { slot: 6, value: fresh }, Type::NONE, Rep::None);
    escape::run(func);
    assert!(func.block(fixture.header).insts.contains(&fresh), "a table the interpreter's registers hold is allocated");
}

#[test]
fn escape_analysis_turns_a_carried_table_into_field_parameters() {
    let mut fixture = counted_loop();
    let func = &mut fixture.func;
    func.sink_allocations = true;
    let entry_state = StateId(0);
    let latch = func.loops[0].blocks[1];
    let entering = func.append(fixture.preheader, Op::LoadReg { slot: 2 }, Type::ANY, Rep::Boxed);
    let Term::Jump(edge) = &mut func.block_mut(fixture.preheader).term else {
        unreachable!("the preheader jumps to the header");
    };
    edge.args.push(entering);
    let carried = func.add_param(fixture.header, Type::ANY, Rep::Boxed);
    let table = func.append(fixture.header, Op::GuardType { value: carried, ty: Type::TABLE, state: entry_state }, Type::TABLE, Rep::Payload);
    func.append(fixture.header, Op::GuardShape { table, shape: 9, state: entry_state }, Type::NONE, Rep::None);
    let field = func.append(fixture.header, Op::LoadSlot { table, slot: 0 }, Type::ANY, Rep::Boxed);
    func.append(fixture.header, Op::StoreReg { slot: 7, value: field }, Type::NONE, Rep::None);
    let x = func.append(latch, Op::Unary(UnOp::Box, fixture.index), Type::INT, Rep::Boxed);
    let fresh = fresh_table(func, latch, &[x], entry_state);
    let Term::Jump(edge) = &mut func.block_mut(latch).term else {
        unreachable!("the latch jumps to the header");
    };
    edge.args.push(fresh);
    escape::run(func);
    assert!(!func.block(latch).insts.contains(&fresh), "the carried allocation is sunk");
    assert!(!func.block(fixture.header).params.contains(&carried), "the table parameter is gone");
    let stored_field = stored(func, fixture.header, 7).expect("the field is still stored");
    assert!(func.block(fixture.header).params.contains(&stored_field), "the field load reads its parameter");
    let preheader_shape = func.block(fixture.preheader).insts.iter().any(|&value| matches!(func.op(value), Op::GuardShape { shape: 9, .. }));
    assert!(preheader_shape, "the entering table's shape is checked before the loop");
    check::check(func, false).expect("the carried graph stays valid");
}
