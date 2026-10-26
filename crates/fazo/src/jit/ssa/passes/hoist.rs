//! Hoists rather than peels, where LuaJIT's `lj_opt_loop.c` (lines 22-75) peels for the same effect.
//! Table-memory reads move only when nothing in the loop can change that memory.

use crate::jit::ssa::ir::{BlockId, Func, Loop, Op, StateId, Term, Val};

#[derive(Default)]
struct Clobbers {
    calls: bool,
    shapes: bool,
    upvalues: bool,
    stores: bool,
    /// A collector step may clear entries of weak tables, so loads do not move across an allocation.
    allocates: bool,
    slots: Vec<u32>,
    nodes: Vec<u32>,
}

/// Moving instructions changes neither the blocks nor their dominators, so one analysis serves every loop.
struct Shape {
    idom: Vec<Option<BlockId>>,
    order: Vec<BlockId>,
    defined: Vec<Option<BlockId>>,
}

pub(crate) fn run(func: &mut Func) {
    if func.loops.is_empty() {
        return;
    }
    let mut shape = Shape {
        idom: func.dominators(),
        order: func.reverse_postorder(),
        defined: func.definitions(),
    };
    for index in (0..func.loops.len()).rev() {
        let entry = func.loops[index].clone();
        hoist_loop(func, &entry, &mut shape);
    }
}

/// An inner loop's OSR entry can reach this header past the preheader, so nothing may move out.
pub(crate) fn preheader_dominates(func: &Func, entry: &Loop) -> bool {
    let idom = func.dominators();
    Func::dominates(&idom, entry.preheader, entry.header)
}

fn hoist_loop(func: &mut Func, entry: &Loop, shape: &mut Shape) {
    if !Func::dominates(&shape.idom, entry.preheader, entry.header) {
        return;
    }
    let mut inside = vec![false; func.blocks.len()];
    for &block in &entry.blocks {
        inside[block.0 as usize] = true;
    }
    let clobbers = clobbers(func, entry);
    let every_iteration = blocks_on_every_iteration(func, entry, &inside, &shape.idom);
    let order: Vec<BlockId> = shape.order.iter().copied().filter(|block| inside[block.0 as usize]).collect();
    let mut invariant: Vec<bool> = shape.defined.iter().map(|block| block.is_some_and(|block| !inside[block.0 as usize])).collect();
    let mut changed = true;
    while changed {
        changed = false;
        for &block in &order {
            let mut position = 0;
            while position < func.block(block).insts.len() {
                let value = func.block(block).insts[position];
                if !movable(func, value, &invariant, &clobbers, every_iteration[block.0 as usize]) {
                    position += 1;
                    continue;
                }
                func.block_mut(block).insts.remove(position);
                func.block_mut(entry.preheader).insts.push(value);
                shape.defined[value.0 as usize] = Some(entry.preheader);
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

/// Kept in a register across calls, where a save around each call would cost more.
fn movable(func: &Func, value: Val, invariant: &[bool], clobbers: &Clobbers, every_iteration: bool) -> bool {
    let op = func.op(value);
    let mut operands_invariant = true;
    op.for_each_operand(|operand| operands_invariant &= invariant[operand.0 as usize]);
    if !operands_invariant {
        return false;
    }
    if matches!(op, Op::Requests) {
        return !clobbers.calls;
    }
    if op.is_pure() {
        return true;
    }
    if !every_iteration {
        return false;
    }
    match op {
        Op::GuardType { .. } | Op::GuardNotNil { .. } | Op::GuardEq { .. } | Op::GuardFalse { .. } => true,
        Op::GuardShape { .. } | Op::GuardNode { .. } | Op::ShapeIs { .. } => !clobbers.calls && !clobbers.shapes,
        Op::LoadNode { node, .. } => !clobbers.calls && !clobbers.shapes && !clobbers.allocates && !clobbers.nodes.contains(node),
        Op::GuardHasMeta { .. } | Op::GuardNoMeta { .. } => !clobbers.calls && !clobbers.shapes,
        Op::GuardAbsent { .. } => !clobbers.calls && !clobbers.shapes && !clobbers.stores,
        Op::LoadSlot { slot, .. } => !clobbers.calls && !clobbers.shapes && !clobbers.allocates && !clobbers.slots.contains(slot),
        Op::LoadProto { .. } => true,
        Op::LoadUpval { .. } => !clobbers.calls && !clobbers.upvalues,
        Op::GuardRange { .. } => !clobbers.calls && !clobbers.shapes,
        Op::Builtin { function, .. } => !function.writes() && !clobbers.calls && !clobbers.shapes && !clobbers.stores && !clobbers.allocates,
        _ => false,
    }
}

fn clobbers(func: &Func, entry: &Loop) -> Clobbers {
    let mut found = Clobbers::default();
    for &block in &entry.blocks {
        for &value in &func.block(block).insts {
            match func.op(value) {
                Op::Helper(_) | Op::CallLua { .. } => found.calls = true,
                Op::NewTable { .. } | Op::Materialize { .. } => found.allocates = true,
                Op::NewIndex { .. } => {
                    found.shapes = true;
                    found.stores = true;
                }
                Op::Builtin { function, .. } if function.writes() => {
                    found.shapes = true;
                    found.stores = true;
                }
                Op::AddSlot { .. } => {
                    found.shapes = true;
                    found.stores = true;
                }
                Op::StoreSlot { slot, .. } => {
                    found.slots.push(*slot);
                    found.stores = true;
                }
                Op::StoreNode { node, .. } => {
                    found.nodes.push(*node);
                    found.stores = true;
                }
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

fn blocks_on_every_iteration(func: &Func, entry: &Loop, inside: &[bool], idom: &[Option<BlockId>]) -> Vec<bool> {
    let latches: Vec<BlockId> = entry
        .blocks
        .iter()
        .copied()
        .filter(|&block| inside[block.0 as usize] && func.block(block).term.successors().contains(&entry.header))
        .collect();
    let mut every = vec![false; func.blocks.len()];
    for &block in &entry.blocks {
        every[block.0 as usize] = latches.iter().all(|&latch| Func::dominates(idom, block, latch));
    }
    every
}
