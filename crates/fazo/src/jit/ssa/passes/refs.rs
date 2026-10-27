//! A read and a later write of one key in one table share a single lookup, as LuaJIT's `AREF` and `HREF` feed both the
//! load and the store (lj_ir.h, lj_opt_mem.c), while nothing between can insert into a table.

use crate::jit::ssa::ir::{BlockId, Func, Op, Rep, Type, Val};

const STRING_TAG: u8 = 4;

pub(crate) fn run(func: &mut Func) {
    if !has_candidates(func) {
        return;
    }
    let order = func.reverse_postorder();
    let available = available_reads(func, &order);
    let pairs = matching_writes(func, &order, &available);
    share(func, &pairs);
}

fn has_candidates(func: &Func) -> bool {
    let reads: Vec<(Val, Val)> = func.insts.iter().filter(|inst| shareable_read(func, &inst.op)).filter_map(|inst| match inst.op {
        Op::Index { object, key, .. } => Some((object, key)),
        _ => None,
    }).collect();
    !reads.is_empty() && func.insts.iter().any(|inst| matches!(inst.op, Op::NewIndex { object, key, node: None, .. } if reads.contains(&(object, key))))
}

/// Reads whose probe is still valid on entry to each block, met over every predecessor.
fn available_reads(func: &Func, order: &[BlockId]) -> Vec<Option<Vec<Val>>> {
    let predecessors = func.predecessors();
    let roots: Vec<BlockId> = func.entry.into_iter().chain(func.osr.iter().map(|&(_, block)| block)).collect();
    let mut entry: Vec<Option<Vec<Val>>> = vec![None; func.blocks.len()];
    let mut exit: Vec<Option<Vec<Val>>> = vec![None; func.blocks.len()];
    let mut changed = true;
    while changed {
        changed = false;
        for &block in order {
            let incoming = if roots.contains(&block) { Some(Vec::new()) } else { meet(&predecessors[block.0 as usize], &exit) };
            let Some(incoming) = incoming else {
                continue;
            };
            let outgoing = transfer(func, block, incoming.clone(), |_, _| {});
            changed |= entry[block.0 as usize].as_ref() != Some(&incoming) || exit[block.0 as usize].as_ref() != Some(&outgoing);
            entry[block.0 as usize] = Some(incoming);
            exit[block.0 as usize] = Some(outgoing);
        }
    }
    entry
}

fn meet(predecessors: &[BlockId], exit: &[Option<Vec<Val>>]) -> Option<Vec<Val>> {
    let mut known = predecessors.iter().filter_map(|block| exit[block.0 as usize].as_ref());
    let first = known.next()?.clone();
    Some(known.fold(first, |mut kept, other| {
        kept.retain(|read| other.contains(read));
        kept
    }))
}

/// Calls `found` with each write and the read it can share a probe with.
fn transfer(func: &Func, block: BlockId, mut available: Vec<Val>, mut found: impl FnMut(Val, Val)) -> Vec<Val> {
    for &value in &func.block(block).insts {
        let op = func.op(value);
        if let Op::NewIndex { object, key, node: None, .. } = *op
            && let Some(&read) = available.iter().find(|&&read| reads(func, read) == Some((object, key)))
        {
            found(read, value);
        }
        if shareable_read(func, op) {
            available.push(value);
            continue;
        }
        if may_insert(op) {
            available.clear();
        }
    }
    available
}

fn matching_writes(func: &Func, order: &[BlockId], available: &[Option<Vec<Val>>]) -> Vec<(Val, Val)> {
    let mut pairs = Vec::new();
    for &block in order {
        let Some(incoming) = available[block.0 as usize].clone() else {
            continue;
        };
        transfer(func, block, incoming, |read, write| pairs.push((read, write)));
    }
    pairs
}

fn reads(func: &Func, read: Val) -> Option<(Val, Val)> {
    match *func.op(read) {
        Op::Index { object, key, .. } => Some((object, key)),
        _ => None,
    }
}

fn shareable_read(func: &Func, op: &Op) -> bool {
    let Op::Index { object, key, node: None, .. } = *op else {
        return false;
    };
    let keyed = match func.rep(key) {
        Rep::Int => true,
        Rep::Boxed => !func.ty(key).intersect(Type::INT.union(Type::of_tag(STRING_TAG))).is_none(),
        _ => false,
    };
    func.ty(object) == Type::TABLE && keyed && !matches!(func.op(key), Op::Const(_))
}

/// Inserting a key may rehash a table and free its nodes; the rest run no Lua code and leave every node in place.
fn may_insert(op: &Op) -> bool {
    match op {
        Op::Builtin { function, .. } => function.writes(),
        _ => matches!(op, Op::NewIndex { .. } | Op::AddSlot { .. } | Op::Helper(_) | Op::CallLua { .. } | Op::NewTable { .. } | Op::Materialize { .. } | Op::Safepoint { .. }),
    }
}

fn share(func: &mut Func, pairs: &[(Val, Val)]) {
    let mut refs: Vec<(Val, Val)> = Vec::new();
    for &(read, write) in pairs {
        let node = match refs.iter().find(|&&(known, _)| known == read) {
            Some(&(_, node)) => node,
            None => {
                let node = insert_ref(func, read);
                refs.push((read, node));
                node
            }
        };
        if let Op::NewIndex { node: shared, .. } = &mut func.inst_mut(write).op {
            *shared = Some(node);
        }
    }
}

fn insert_ref(func: &mut Func, read: Val) -> Val {
    let (table, key) = reads(func, read).expect("a shared read is an index");
    let (block, position) = place_of(func, read);
    let node = super::insert(func, block, position, Op::NodeRef { table, key }, Type::NONE, Rep::Word);
    if let Op::Index { node: shared, .. } = &mut func.inst_mut(read).op {
        *shared = Some(node);
    }
    node
}

fn place_of(func: &Func, value: Val) -> (BlockId, usize) {
    func.blocks
        .iter()
        .enumerate()
        .find_map(|(index, block)| Some((BlockId(index as u32), block.insts.iter().position(|&inst| inst == value)?)))
        .expect("a matched read sits in a block")
}
