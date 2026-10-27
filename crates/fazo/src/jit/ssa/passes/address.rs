//! Tables move only when the arena grows, so an address lives until the next node that can allocate (V8's
//! `OpProperties::CanAllocate`, src/maglev/maglev-ir.h).

use crate::jit::ssa::ir::{BlockId, Func, Loop, Op, Rep, Term, Type, Val};

pub(crate) fn run(func: &mut Func) {
    if !func.insts.iter().any(|inst| table_operand(&inst.op).is_some()) {
        return;
    }
    let quiet = quiet_loops(func);
    let inside = membership(func, &quiet);
    let defined = if quiet.is_empty() { Vec::new() } else { func.definitions() };
    let mut hoisted: Vec<(BlockId, Val, Val)> = Vec::new();
    let mut local: Vec<(Val, Val)> = Vec::new();
    for index in 0..func.blocks.len() {
        let block = BlockId(index as u32);
        local.clear();
        let mut position = 0;
        while position < func.block(block).insts.len() {
            let value = func.block(block).insts[position];
            position += 1;
            if moves_tables(func.op(value)) {
                local.clear();
                continue;
            }
            let Some(table) = table_operand(func.op(value)) else {
                continue;
            };
            if matches!(func.op(table), Op::TableAddr { .. }) {
                continue;
            }
            let hoistable = |index: &usize| inside[*index][block.0 as usize] && defined[table.0 as usize].is_none_or(|at| !inside[*index][at.0 as usize]);
            let address = match (0..quiet.len()).filter(hoistable).max_by_key(|&index| quiet[index].blocks.len()) {
                Some(index) => preheader_address(func, &mut hoisted, quiet[index].preheader, table),
                None => match local.iter().find(|&&(known, _)| known == table) {
                    Some(&(_, address)) => address,
                    None if read_again(func, block, position, table) => {
                        let address = super::insert(func, block, position - 1, Op::TableAddr { table }, Type::NONE, Rep::Word);
                        position += 1;
                        local.push((table, address));
                        address
                    }
                    None => continue,
                },
            };
            set_table_operand(&mut func.inst_mut(value).op, address);
        }
    }
}

fn membership(func: &Func, loops: &[Loop]) -> Vec<Vec<bool>> {
    loops
        .iter()
        .map(|entry| {
            let mut inside = vec![false; func.blocks.len()];
            entry.blocks.iter().for_each(|block| inside[block.0 as usize] = true);
            inside
        })
        .collect()
}

/// Loops whose preheader dominates the header and that never allocate.
fn quiet_loops(func: &Func) -> Vec<Loop> {
    if func.loops.is_empty() {
        return Vec::new();
    }
    let idom = func.dominators();
    func.loops
        .iter()
        .filter(|entry| Func::dominates(&idom, entry.preheader, entry.header))
        .filter(|entry| entry.blocks.iter().all(|&block| quiet_block(func, block)))
        .cloned()
        .collect()
}

fn quiet_block(func: &Func, block: BlockId) -> bool {
    let data = func.block(block);
    !matches!(data.term, Term::TailCall { .. }) && !data.insts.iter().any(|&value| moves_tables(func.op(value)))
}

fn preheader_address(func: &mut Func, hoisted: &mut Vec<(BlockId, Val, Val)>, preheader: BlockId, table: Val) -> Val {
    if let Some(&(_, _, address)) = hoisted.iter().find(|&&(at, known, _)| at == preheader && known == table) {
        return address;
    }
    let address = func.append(preheader, Op::TableAddr { table }, Type::NONE, Rep::Word);
    hoisted.push((preheader, table, address));
    address
}

fn read_again(func: &Func, block: BlockId, from: usize, table: Val) -> bool {
    for &value in &func.block(block).insts[from..] {
        let op = func.op(value);
        if moves_tables(op) {
            return false;
        }
        if table_operand(op) == Some(table) {
            return true;
        }
    }
    false
}

/// Only these can allocate a collectable object; the inline slow paths of the rest reach no allocation.
fn moves_tables(op: &Op) -> bool {
    matches!(op, Op::NewTable { .. } | Op::Materialize { .. } | Op::Helper(_) | Op::CallLua { .. })
}

fn table_operand(op: &Op) -> Option<Val> {
    match *op {
        Op::LoadSlot { table, .. }
        | Op::LoadNode { table, .. }
        | Op::NodeRef { table, .. }
        | Op::ShapeIs { table, .. }
        | Op::ArrayBase { table }
        | Op::GuardShape { table, .. }
        | Op::GuardNode { table, .. }
        | Op::GuardHasMeta { table, .. }
        | Op::GuardNoMeta { table, .. }
        | Op::GuardAbsent { table, .. }
        | Op::GuardRange { table, .. }
        | Op::StoreSlot { table, .. }
        | Op::StoreNode { table, .. }
        | Op::AddSlot { table, .. } => Some(table),
        Op::Index { object, .. } | Op::NewIndex { object, .. } => Some(object),
        _ => None,
    }
}

fn set_table_operand(op: &mut Op, address: Val) {
    match op {
        Op::LoadSlot { table, .. }
        | Op::LoadNode { table, .. }
        | Op::NodeRef { table, .. }
        | Op::ShapeIs { table, .. }
        | Op::ArrayBase { table }
        | Op::GuardShape { table, .. }
        | Op::GuardNode { table, .. }
        | Op::GuardHasMeta { table, .. }
        | Op::GuardNoMeta { table, .. }
        | Op::GuardAbsent { table, .. }
        | Op::GuardRange { table, .. }
        | Op::StoreSlot { table, .. }
        | Op::StoreNode { table, .. }
        | Op::AddSlot { table, .. } => *table = address,
        Op::Index { object, .. } | Op::NewIndex { object, .. } => *object = address,
        _ => {}
    }
}
