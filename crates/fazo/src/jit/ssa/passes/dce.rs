//! Frame states of live ops are roots, so deoptimization still sees the values it reads.

use super::{fold::{drop_params, incoming_edges}, remove};
use crate::jit::ssa::ir::{Func, Term, Val};

pub(crate) fn run(func: &mut Func) {
    let live = mark(func);
    let dead: Vec<bool> = live.iter().map(|&alive| !alive).collect();
    remove(func, &dead);
    let order = func.reverse_postorder();
    let incoming = incoming_edges(func, &order);
    for &block in &order {
        let keep: Vec<bool> = func.block(block).params.iter().map(|param| live[param.0 as usize]).collect();
        if keep.iter().all(|&kept| kept) {
            continue;
        }
        drop_params(func, block, &keep, &incoming[block.0 as usize]);
    }
}

fn mark(func: &Func) -> Vec<bool> {
    let mut live = vec![false; func.insts.len()];
    let mut work: Vec<Val> = Vec::new();
    let order = func.reverse_postorder();
    for &block in &order {
        let data = func.block(block);
        for &value in &data.insts {
            if !func.op(value).is_removable() {
                work.push(value);
            }
        }
        match &data.term {
            Term::Branch { cond, .. } => work.push(*cond),
            Term::Switch { code, .. } => work.push(*code),
            Term::Deopt { state } => work.extend(func.state(*state).slots.iter().map(|&(_, value)| value)),
            _ => {}
        }
    }
    let incoming = incoming_edges(func, &order);
    let position = |block: usize, param: Val| func.blocks[block].params.iter().position(|&known| known == param);
    let defined = func.definitions();
    while let Some(value) = work.pop() {
        if live[value.0 as usize] {
            continue;
        }
        live[value.0 as usize] = true;
        let op = func.op(value);
        work.extend(op.operands());
        if let Some(state) = op.state() {
            work.extend(func.state(state).slots.iter().map(|&(_, slot_value)| slot_value));
        }
        let Some(block) = defined[value.0 as usize] else {
            continue;
        };
        let Some(index) = position(block.0 as usize, value) else {
            continue;
        };
        for &(from, edge) in &incoming[block.0 as usize] {
            work.push(func.block(from).term.edges()[edge].args[index]);
        }
    }
    live
}
