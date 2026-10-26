//! Escape analysis and allocation sinking, following V8's `EscapeAnalysis` (src/compiler/escape-analysis.cc) and LuaJIT's `lj_opt_sink.c`.

use super::remove;
use crate::jit::ssa::ir::{BlockId, BuiltinFn, Const, Func, Loop, Op, Rep, StateId, Term, Type, Val, Virtual};

#[derive(Clone, Copy)]
struct Use {
    block: BlockId,
    inst: Option<Val>,
}

struct Candidate {
    table: Val,
    aliases: Vec<Val>,
    setmetatable: Option<Val>,
    exits: Vec<BlockId>,
    carried: Option<Carried>,
}

struct Carried {
    header: BlockId,
    preheader: BlockId,
    latch: BlockId,
    index: usize,
    aliases: Vec<Val>,
    loads: Vec<(Val, u32)>,
    metas: Vec<Val>,
    guards: Vec<Val>,
    entry_state: StateId,
}

struct Layout {
    hash_log: u8,
    shape: u64,
    fields: Vec<Val>,
}

pub(crate) fn run(func: &mut Func) {
    if !func.sink_allocations {
        return;
    }
    let tables: Vec<Val> = func.blocks.iter().flat_map(|block| block.insts.iter().copied()).filter(|&value| matches!(func.op(value), Op::NewTable { .. })).collect();
    for table in tables {
        let users = users(func);
        let Some(candidate) = candidate(func, &users, table) else {
            continue;
        };
        let Some(positions) = state_positions(func, &candidate) else {
            continue;
        };
        sink(func, &candidate, &positions);
    }
}

fn users(func: &Func) -> Vec<Vec<Use>> {
    let mut users = vec![Vec::new(); func.insts.len()];
    for block in func.reverse_postorder() {
        let data = func.block(block);
        for &value in &data.insts {
            func.op(value).for_each_operand(|operand| users[operand.0 as usize].push(Use { block, inst: Some(value) }));
        }
        for operand in data.term.operands() {
            users[operand.0 as usize].push(Use { block, inst: None });
        }
    }
    users
}

fn layout(func: &Func, table: Val) -> Layout {
    let Op::NewTable { hash_log, shape, fields, .. } = func.op(table) else {
        unreachable!("a sinking candidate is a table allocation");
    };
    Layout { hash_log: *hash_log, shape: shape.unwrap_or(crate::shape::ROOT.packed()), fields: fields.to_vec() }
}

fn innermost(func: &Func, block: BlockId) -> Option<&Loop> {
    func.loops.iter().filter(|candidate| candidate.blocks.contains(&block)).min_by_key(|candidate| candidate.blocks.len())
}

fn candidate(func: &Func, users: &[Vec<Use>], table: Val) -> Option<Candidate> {
    let definitions = func.definitions();
    let home = innermost(func, definitions[table.0 as usize]?)?;
    let idom = func.dominators();
    let mut aliases = vec![table];
    let mut setmetatable = None;
    let mut exits = Vec::new();
    let mut back_edge = None;
    let mut index = 0;
    while let Some(&alias) = aliases.get(index) {
        index += 1;
        for &user in &users[alias.0 as usize] {
            if !home.blocks.contains(&user.block) {
                let exit = exit_block(func, &idom, home, user.block)?;
                if !exits.contains(&exit) {
                    exits.push(exit);
                }
                continue;
            }
            let Some(inst) = user.inst else {
                if back_edge.is_some() {
                    return None;
                }
                back_edge = Some((user.block, alias));
                continue;
            };
            match func.op(inst) {
                Op::GuardType { ty, .. } if !ty.intersect(Type::TABLE).is_none() => {}
                Op::Builtin { function: BuiltinFn::SetMetatable, args, .. } if args[0] == alias && args[1] != alias && setmetatable.is_none() => setmetatable = Some(inst),
                _ => return None,
            }
            aliases.push(inst);
        }
    }
    let meta = setmetatable.map(|set| setmetatable_args(func, set).1);
    if meta.is_some_and(|meta| aliases.contains(&meta)) {
        return None;
    }
    let has_meta = meta.is_some_and(|meta| func.ty(meta) != Type::NIL);
    let carried = match back_edge {
        Some((latch, alias)) => Some(carried(func, users, home, (latch, alias), (table, has_meta))?),
        None => None,
    };
    Some(Candidate { table, aliases, setmetatable, exits, carried })
}

fn exit_block(func: &Func, idom: &[Option<BlockId>], home: &Loop, block: BlockId) -> Option<BlockId> {
    let mut current = block;
    loop {
        let parent = idom.get(current.0 as usize).copied().flatten()?;
        if home.blocks.contains(&parent) {
            break;
        }
        current = parent;
    }
    let predecessors = func.predecessors();
    let edges = func.blocks.iter().flat_map(|data| data.term.edges()).filter(|edge| edge.block == current).count();
    (edges == 1 && predecessors[current.0 as usize].len() == 1).then_some(current)
}

fn carried(func: &Func, users: &[Vec<Use>], home: &Loop, (latch, alias): (BlockId, Val), (table, has_meta): (Val, bool)) -> Option<Carried> {
    let Term::Jump(edge) = &func.block(latch).term else {
        return None;
    };
    if edge.block != home.header || edge.args.iter().filter(|&&arg| arg == alias).count() != 1 {
        return None;
    }
    let index = edge.args.iter().position(|&arg| arg == alias)?;
    let entries = func.blocks.iter().flat_map(|data| data.term.edges()).filter(|edge| edge.block == home.header).count();
    if entries != 2 || !func.block(home.preheader).term.edges().iter().any(|edge| edge.block == home.header) {
        return None;
    }
    let Layout { shape, fields, .. } = layout(func, table);
    let param = func.block(home.header).params[index];
    let mut carried = Carried {
        header: home.header,
        preheader: home.preheader,
        latch,
        index,
        aliases: vec![param],
        loads: Vec::new(),
        metas: Vec::new(),
        guards: Vec::new(),
        entry_state: home.entry_state,
    };
    let mut position = 0;
    while let Some(&alias) = carried.aliases.get(position) {
        position += 1;
        for user in &users[alias.0 as usize] {
            let inst = user.inst?;
            if !home.blocks.contains(&user.block) {
                return None;
            }
            match *func.op(inst) {
                Op::GuardType { ty, .. } if !ty.intersect(Type::TABLE).is_none() => carried.aliases.push(inst),
                Op::GuardShape { shape: guarded, .. } if guarded == shape => carried.guards.push(inst),
                Op::GuardHasMeta { .. } if has_meta => carried.metas.push(inst),
                Op::LoadSlot { slot, .. } if (slot as usize) < fields.len() => carried.loads.push((inst, slot)),
                _ => return None,
            }
        }
    }
    Some(carried)
}

fn setmetatable_args(func: &Func, set: Val) -> (Val, Val, StateId) {
    let Op::Builtin { args, state, .. } = func.op(set) else {
        unreachable!("a setmetatable candidate is a builtin");
    };
    (args[0], args[1], *state)
}

fn state_positions(func: &Func, candidate: &Candidate) -> Option<Vec<(StateId, bool)>> {
    let idom = func.dominators();
    let set_at = candidate.setmetatable.and_then(|set| position(func, set));
    let mut decided: Vec<(StateId, bool)> = Vec::new();
    for block in func.reverse_postorder() {
        if candidate.exits.iter().any(|&exit| Func::dominates(&idom, exit, block)) {
            continue;
        }
        let insts = &func.block(block).insts;
        let terminal = match func.block(block).term {
            Term::Deopt { state } => Some((insts.len(), state)),
            _ => None,
        };
        let states = insts.iter().enumerate().filter(|&(_, &value)| value != candidate.table).flat_map(|(index, &value)| states_of(func.op(value)).into_iter().map(move |state| (index, state)));
        for (index, state) in states.chain(terminal) {
            if !names(func, state, &candidate.aliases) {
                continue;
            }
            let after = past(&idom, set_at, (block, index))?;
            match decided.iter().find(|&&(known, _)| known == state) {
                Some(&(_, known)) if known != after => return None,
                Some(_) => {}
                None => decided.push((state, after)),
            }
        }
    }
    for &exit in &candidate.exits {
        past(&idom, set_at, (exit, 0))?;
    }
    Some(decided)
}

/// A table allocation also resumes at the state after its constructor when a finalizer runs.
fn states_of(op: &Op) -> Vec<StateId> {
    let after = match op {
        Op::NewTable { after, .. } => Some(*after),
        _ => None,
    };
    op.state().into_iter().chain(after).collect()
}

fn position(func: &Func, value: Val) -> Option<(BlockId, usize)> {
    func.blocks.iter().enumerate().find_map(|(block, data)| data.insts.iter().position(|&inst| inst == value).map(|index| (BlockId(block as u32), index)))
}

fn names(func: &Func, state: StateId, aliases: &[Val]) -> bool {
    func.state(state).slots.iter().any(|(_, value)| aliases.contains(value))
}

fn past(idom: &[Option<BlockId>], set_at: Option<(BlockId, usize)>, (block, index): (BlockId, usize)) -> Option<bool> {
    let Some((set_block, set_index)) = set_at else {
        return Some(false);
    };
    if block == set_block {
        return Some(index > set_index);
    }
    if Func::dominates(idom, set_block, block) {
        return Some(true);
    }
    Func::dominates(idom, block, set_block).then_some(false)
}

fn sink(func: &mut Func, candidate: &Candidate, positions: &[(StateId, bool)]) {
    let layout = layout(func, candidate.table);
    let meta = candidate.setmetatable.and_then(|set| metatable_checks(func, set));
    let object = |with_meta: bool| Virtual {
        value: candidate.table,
        hash_log: layout.hash_log,
        shape: layout.shape,
        fields: layout.fields.clone(),
        meta: meta.filter(|_| with_meta),
        original: None,
    };
    let mut dead = vec![false; func.insts.len()];
    if let Some(carried) = &candidate.carried {
        carry(func, carried, (&layout, meta), &mut dead);
    }
    for &exit in &candidate.exits {
        materialize_at(func, exit, object(meta.is_some()), &candidate.aliases);
    }
    for &(state, past_set) in positions {
        virtualize(func, state, object(past_set), &candidate.aliases);
    }
    dead.resize(func.insts.len(), false);
    for &alias in &candidate.aliases {
        dead[alias.0 as usize] = true;
    }
    remove(func, &dead);
}

fn virtualize(func: &mut Func, state: StateId, object: Virtual, aliases: &[Val]) {
    let data = &mut func.states[state.0 as usize];
    for (_, value) in &mut data.slots {
        if aliases.contains(value) {
            *value = object.value;
        }
    }
    data.objects.retain(|known| known.value != object.value);
    data.objects.push(object);
}

/// Building the table later registers no finalizer, so `setmetatable`'s `__gc` check still holds.
fn metatable_checks(func: &mut Func, set: Val) -> Option<Val> {
    let (_, meta, state) = setmetatable_args(func, set);
    if func.ty(meta) == Type::NIL {
        return None;
    }
    let (block, index) = position(func, set).expect("the setmetatable is placed");
    let mut checks = Vec::new();
    let table = if func.ty(meta).within(Type::TABLE) {
        meta
    } else {
        let guard = func.add_inst(Op::GuardType { value: meta, ty: Type::TABLE, state }, Type::TABLE, Rep::Boxed);
        checks.push(guard);
        guard
    };
    let gc = crate::vm::Event::Gc.absence_bit().expect("__gc has an absence bit");
    checks.push(func.add_inst(Op::GuardAbsent { table, bit: gc, state }, Type::NONE, Rep::None));
    func.block_mut(block).insts.splice(index..index, checks);
    Some(table)
}

fn materialize_at(func: &mut Func, exit: BlockId, object: Virtual, aliases: &[Val]) {
    let idom = func.dominators();
    let built = func.add_inst(Op::Materialize { object: Box::new(object) }, Type::TABLE, Rep::Payload);
    func.block_mut(exit).insts.insert(0, built);
    let region: Vec<BlockId> = (0..func.blocks.len() as u32).map(BlockId).filter(|&block| Func::dominates(&idom, exit, block)).collect();
    let replace = |value: Val| if aliases.contains(&value) { built } else { value };
    for &block in &region {
        let insts = func.block(block).insts.clone();
        for value in insts {
            if value == built {
                continue;
            }
            func.inst_mut(value).op.map_operands(replace);
            if let Some(state) = func.op(value).state()
                && names(func, state, aliases)
            {
                let copy = rebased_state(func, state, aliases, built);
                func.inst_mut(value).op.set_state(copy);
            }
            if let Op::NewTable { after, .. } = *func.op(value)
                && names(func, after, aliases)
            {
                let copy = rebased_state(func, after, aliases, built);
                if let Op::NewTable { after, .. } = &mut func.inst_mut(value).op {
                    *after = copy;
                }
            }
        }
        func.block_mut(block).term.map_operands(replace);
        if let Term::Deopt { state } = func.block(block).term
            && names(func, state, aliases)
        {
            let copy = rebased_state(func, state, aliases, built);
            func.block_mut(block).term = Term::Deopt { state: copy };
        }
    }
}

fn rebased_state(func: &mut Func, state: StateId, aliases: &[Val], built: Val) -> StateId {
    let mut copy = func.state(state).clone();
    for (_, value) in &mut copy.slots {
        if aliases.contains(value) {
            *value = built;
        }
    }
    func.states.push(copy);
    StateId(func.states.len() as u32 - 1)
}

fn carry(func: &mut Func, carried: &Carried, (layout, meta): (&Layout, Option<Val>), dead: &mut Vec<bool>) {
    let entry = entry_reads(func, carried, layout, meta.is_some());
    let param = func.block(carried.header).params[carried.index];
    let fields: Vec<Val> = layout.fields.iter().map(|_| func.add_param(carried.header, Type::ANY, Rep::Boxed)).collect();
    let meta_param = meta.map(|_| func.add_param(carried.header, Type::TABLE, Rep::Payload));
    let flag = func.add_param(carried.header, Type::BOOL, Rep::Bool);
    func.block_mut(carried.header).params.remove(carried.index);

    let set = func.append(carried.latch, Op::Const(Const::Bool(true)), Type::BOOL, Rep::Bool);
    let latch_args: Vec<Val> = layout.fields.iter().copied().chain(meta).chain([set]).collect();
    replace_edge_args(func, carried.latch, carried.header, carried.index, &latch_args);
    let entry_args: Vec<Val> = entry.fields.iter().copied().chain(entry.meta).chain([entry.flag]).collect();
    replace_edge_args(func, carried.preheader, carried.header, carried.index, &entry_args);

    let object = Virtual {
        value: param,
        hash_log: layout.hash_log,
        shape: layout.shape,
        fields: fields.clone(),
        meta: meta_param,
        original: Some((flag, entry.table)),
    };
    let states: Vec<StateId> = (0..func.states.len() as u32).map(StateId).filter(|&state| names(func, state, &carried.aliases)).collect();
    for state in states {
        virtualize(func, state, object.clone(), &carried.aliases);
    }
    let mut map = func.identity_map();
    for &(load, slot) in &carried.loads {
        map[load.0 as usize] = fields[slot as usize];
    }
    for &read in &carried.metas {
        map[read.0 as usize] = meta_param.expect("a carried metatable read has a metatable");
    }
    dead.resize(func.insts.len(), false);
    for &value in carried.loads.iter().map(|(load, _)| load).chain(&carried.metas).chain(&carried.guards).chain(&carried.aliases) {
        dead[value.0 as usize] = true;
    }
    func.rewrite_uses(&map);
}

struct EntryReads {
    table: Val,
    fields: Vec<Val>,
    meta: Option<Val>,
    flag: Val,
}

fn entry_reads(func: &mut Func, carried: &Carried, layout: &Layout, has_meta: bool) -> EntryReads {
    let preheader = carried.preheader;
    let state = carried.entry_state;
    let original = edge_args(func, preheader, carried.header)[carried.index];
    let table = func.append(preheader, Op::GuardType { value: original, ty: Type::TABLE, state }, Type::TABLE, Rep::Boxed);
    func.append(preheader, Op::GuardShape { table, shape: layout.shape, state }, Type::NONE, Rep::None);
    let meta = has_meta.then(|| func.append(preheader, Op::GuardHasMeta { table, state }, Type::TABLE, Rep::Payload));
    let fields = (0..layout.fields.len() as u32).map(|slot| func.append(preheader, Op::LoadSlot { table, slot }, Type::ANY, Rep::Boxed)).collect();
    let flag = func.append(preheader, Op::Const(Const::Bool(false)), Type::BOOL, Rep::Bool);
    EntryReads { table, fields, meta, flag }
}

fn edge_args(func: &Func, from: BlockId, to: BlockId) -> Vec<Val> {
    func.block(from).term.edges().into_iter().find(|edge| edge.block == to).map(|edge| edge.args.clone()).unwrap_or_default()
}

fn replace_edge_args(func: &mut Func, from: BlockId, to: BlockId, index: usize, appended: &[Val]) {
    for edge in func.block_mut(from).term.edges_mut() {
        if edge.block != to {
            continue;
        }
        edge.args.remove(index);
        edge.args.extend_from_slice(appended);
    }
}
