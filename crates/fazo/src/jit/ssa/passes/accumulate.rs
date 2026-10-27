//! A loop parameter that comes back as a float on every back edge, and is only ever turned into a float in the loop,
//! loops as a float with a first-iteration flag. Lua turns an integer into a float before any arithmetic with a float
//! (lvm.c `op_arithf_aux`, lvm.h `tonumberns`), so the float sees the same numbers; frame states take the entry value
//! while the flag holds, so an exit in the first iteration still sees an integer.

use super::remove;
use crate::jit::ssa::ir::{Const, Func, Loop, Mixed, Op, Rep, Term, Type, UnOp, Val};

pub(crate) fn run(func: &mut Func) {
    let mut idom = None;
    for index in 0..func.loops.len() {
        let entry = func.loops[index].clone();
        let enters = matches!(&func.block(entry.preheader).term, Term::Jump(edge) if edge.block == entry.header);
        let mixed = (0..func.block(entry.header).params.len()).any(|position| may_split(func, &entry, position));
        if !enters || !mixed {
            continue;
        }
        let idom = idom.get_or_insert_with(|| func.dominators());
        if !Func::dominates(idom, entry.preheader, entry.header) {
            continue;
        }
        let mut position = 0;
        while position < func.block(entry.header).params.len() {
            match candidate(func, &entry, position) {
                Some(found) => split(func, &entry, position, found),
                None => position += 1,
            }
        }
    }
}

struct Found {
    param: Val,
    entry: Val,
    conversions: Vec<Val>,
}

fn may_split(func: &Func, entry: &Loop, position: usize) -> bool {
    func.ty(func.block(entry.header).params[position]) == Type::NUM && back_edges_float(func, entry, position)
}

fn candidate(func: &Func, entry: &Loop, position: usize) -> Option<Found> {
    let param = func.block(entry.header).params[position];
    if !may_split(func, entry, position) {
        return None;
    }
    let Term::Jump(edge) = &func.block(entry.preheader).term else {
        return None;
    };
    let conversions = conversions(func, param)?;
    Some(Found { param, entry: edge.args[position], conversions })
}

fn back_edges_float(func: &Func, entry: &Loop, position: usize) -> bool {
    let mut latches = 0;
    for &block in &entry.blocks {
        for edge in func.block(block).term.edges() {
            if edge.block != entry.header {
                continue;
            }
            latches += 1;
            if !edge.args.get(position).is_some_and(|&arg| func.ty(arg).within(Type::FLOAT)) {
                return false;
            }
        }
    }
    latches > 0
}

/// Every use but a frame state's slot must turn the parameter into a float.
fn conversions(func: &Func, param: Val) -> Option<Vec<Val>> {
    let mut found = Vec::new();
    for block in func.reverse_postorder() {
        let data = func.block(block);
        for &value in &data.insts {
            let op = func.op(value);
            if matches!(op, Op::Unary(UnOp::NumToF, operand) if *operand == param) {
                found.push(value);
                continue;
            }
            if op.operands().contains(&param) {
                return None;
            }
        }
        if data.term.operands().contains(&param) {
            return None;
        }
    }
    let in_object = func.states.iter().flat_map(|state| &state.objects).any(|object| {
        let mut uses = false;
        object.for_each_operand(&mut |value| uses |= value == param);
        uses
    });
    (!in_object).then_some(found)
}

fn split(func: &mut Func, entry: &Loop, position: usize, found: Found) {
    let Found { param, entry: start, conversions } = found;
    let conversion = if func.ty(start).within(Type::INT) { UnOp::IToF } else { UnOp::NumToF };
    let float_start = func.append(entry.preheader, Op::Unary(conversion, start), Type::FLOAT, Rep::Float);
    let first = func.append(entry.preheader, Op::Const(Const::Bool(true)), Type::BOOL, Rep::Bool);
    let later = func.append(entry.preheader, Op::Const(Const::Bool(false)), Type::BOOL, Rep::Bool);
    let float = func.add_param(entry.header, Type::FLOAT, Rep::Float);
    let flag = func.add_param(entry.header, Type::BOOL, Rep::Bool);
    func.block_mut(entry.header).params.remove(position);
    for block in std::iter::once(entry.preheader).chain(entry.blocks.iter().copied()) {
        let from_entry = block == entry.preheader;
        for edge in func.block_mut(block).term.edges_mut() {
            if edge.block != entry.header {
                continue;
            }
            let arg = edge.args.remove(position);
            edge.args.push(if from_entry { float_start } else { arg });
            edge.args.push(if from_entry { first } else { later });
        }
    }
    let mut map = func.identity_map();
    let mut dead = vec![false; func.insts.len()];
    for conversion in conversions {
        map[conversion.0 as usize] = float;
        dead[conversion.0 as usize] = true;
    }
    func.rewrite_uses(&map);
    remove(func, &dead);
    for state in &mut func.states {
        if state.slots.iter().any(|&(_, value)| value == param) {
            state.mixed.push(Mixed { value: param, first: flag, float, entry: start });
        }
    }
}
