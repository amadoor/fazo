//! Hoists rather than peels, where LuaJIT's `lj_opt_loop.c` (lines 22-75) peels for the same effect.
//! Table-memory reads move only when nothing in the loop can change that memory.

use crate::jit::ssa::ir::{BlockId, Func, Loop, Op, StateId, Term, Val};

#[derive(Default)]
struct Clobbers {
    calls: bool,
    shapes: bool,
    upvalues: bool,
    slots: Vec<u32>,
}

pub(crate) fn run(func: &mut Func) {
    for index in (0..func.loops.len()).rev() {
        let entry = func.loops[index].clone();
        hoist_loop(func, &entry);
    }
}

/// An inner loop's OSR entry can reach this header past the preheader, so nothing may move out.
pub(crate) fn preheader_dominates(func: &Func, entry: &Loop) -> bool {
    let idom = func.dominators();
    Func::dominates(&idom, entry.preheader, entry.header)
}

fn hoist_loop(func: &mut Func, entry: &Loop) {
    if !preheader_dominates(func, entry) {
        return;
    }
    let defined = func.definitions();
    let mut inside = vec![false; func.blocks.len()];
    for &block in &entry.blocks {
        inside[block.0 as usize] = true;
    }
    let clobbers = clobbers(func, entry);
    let every_iteration = blocks_on_every_iteration(func, entry, &inside);
    let order: Vec<BlockId> = func.reverse_postorder().into_iter().filter(|block| inside[block.0 as usize]).collect();
    let mut invariant: Vec<bool> = defined.iter().map(|block| block.is_some_and(|block| !inside[block.0 as usize])).collect();
    let mut changed = true;
    while changed {
        changed = false;
        for &block in &order {
            let mut position = 0;
            while position < func.block(block).insts.len() {
                let value = func.block(block).insts[position];
                if !movable(func, value, &invariant, &clobbers, every_iteration.contains(&block)) {
                    position += 1;
                    continue;
                }
                func.block_mut(block).insts.remove(position);
                func.block_mut(entry.preheader).insts.push(value);
                if let Some(state) = func.op(value).state() {
                    let moved = entry_state_for(func, entry, state);
                    func.inst_mut(value).op.set_state(moved);
                }
                invariant[value.0 as usize] = true;
                changed = true;
            }
        }
    }
}

/// Keeps the original site, so a recompile drops the speculation that failed.
pub(crate) fn entry_state_for(func: &mut Func, entry: &Loop, state: StateId) -> StateId {
    let mut moved = func.state(entry.entry_state).clone();
    moved.site = func.state(state).site;
    func.add_state(moved)
}

fn movable(func: &Func, value: Val, invariant: &[bool], clobbers: &Clobbers, every_iteration: bool) -> bool {
    let op = func.op(value);
    if !op.operands().iter().all(|operand| invariant[operand.0 as usize]) {
        return false;
    }
    if op.is_pure() {
        return true;
    }
    if !every_iteration {
        return false;
    }
    match op {
        Op::GuardType { .. } | Op::GuardNotNil { .. } | Op::GuardEq { .. } | Op::GuardFalse { .. } => true,
        Op::GuardShape { .. } | Op::ShapeIs { .. } => !clobbers.calls && !clobbers.shapes,
        Op::GuardHasMeta { .. } | Op::GuardNoMeta { .. } => !clobbers.calls && !clobbers.shapes,
        Op::LoadSlot { slot, .. } => !clobbers.calls && !clobbers.shapes && !clobbers.slots.contains(slot),
        Op::LoadProto { .. } => true,
        Op::LoadUpval { .. } => !clobbers.calls && !clobbers.upvalues,
        Op::GuardRange { .. } => !clobbers.calls && !clobbers.shapes,
        _ => false,
    }
}

fn clobbers(func: &Func, entry: &Loop) -> Clobbers {
    let mut found = Clobbers::default();
    for &block in &entry.blocks {
        for &value in &func.block(block).insts {
            match func.op(value) {
                Op::Helper(_) | Op::CallLua { .. } => found.calls = true,
                Op::NewIndex { .. } => found.shapes = true,
                Op::StoreSlot { slot, .. } => found.slots.push(*slot),
                Op::StoreUpval { .. } | Op::StoreReg { .. } => found.upvalues = true,
                _ => {}
            }
        }
        if matches!(func.block(block).term, Term::TailCall { .. }) {
            found.calls = true;
        }
    }
    found
}

fn blocks_on_every_iteration(func: &Func, entry: &Loop, inside: &[bool]) -> Vec<BlockId> {
    let idom = func.dominators();
    let latches: Vec<BlockId> = entry
        .blocks
        .iter()
        .copied()
        .filter(|&block| inside[block.0 as usize] && func.block(block).term.successors().contains(&entry.header))
        .collect();
    entry
        .blocks
        .iter()
        .copied()
        .filter(|&block| latches.iter().all(|&latch| Func::dominates(&idom, block, latch)))
        .collect()
}
