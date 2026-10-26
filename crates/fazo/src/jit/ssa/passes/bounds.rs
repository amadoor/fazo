//! One range guard in the preheader covers every index a counted loop visits, as LuaJIT's `abc_invar` drops invariant bounds checks (lj_opt_fold.c line 1939).

use crate::jit::ssa::ir::{BinOp, Const, Counted, Func, Loop, Op, Rep, Term, Type, UnOp, Val};

pub(crate) fn run(func: &mut Func) {
    for index in 0..func.loops.len() {
        let entry = func.loops[index].clone();
        if let Some(counted) = entry.counted {
            eliminate(func, &entry, counted);
        }
    }
}

struct Access {
    value: Val,
    table: Val,
}

fn eliminate(func: &mut Func, entry: &Loop, counted: Counted) {
    if func.avoid.contains(&range_site(func, entry)) {
        return;
    }
    if !super::hoist::preheader_dominates(func, entry) {
        return;
    }
    let Some((start, count)) = entry_values(func, entry, counted) else {
        return;
    };
    let Some(accesses) = accesses(func, entry, counted) else {
        return;
    };
    let allocates = allocates(func, entry);
    let mut ranged: Vec<(Val, Val)> = Vec::new();
    for access in accesses {
        let table = invariant_table(func, entry, access.table);
        let state = func.op(access.value).state().unwrap_or(entry.entry_state);
        let base = match ranged.iter().find(|&&(known, _)| known == table) {
            Some(&(_, base)) => base,
            None => {
                add_range_guard(func, entry, table, (start, count, counted.step));
                let base = func.append(entry.preheader, Op::ArrayBase { table }, Type::NONE, Rep::Word);
                ranged.push((table, base));
                base
            }
        };
        let base = if allocates { base_before(func, access.value, table) } else { base };
        func.inst_mut(access.value).op = Op::LoadArray { table, base, index: counted.index, state };
    }
}

/// A collection step may move a table's array part, so such loops reload the address.
fn allocates(func: &Func, entry: &Loop) -> bool {
    entry.blocks.iter().any(|&block| func.block(block).insts.iter().any(|&value| matches!(func.op(value), Op::NewTable { .. } | Op::Materialize { .. })))
}

fn base_before(func: &mut Func, access: Val, table: Val) -> Val {
    let block = func.definitions()[access.0 as usize].expect("an access lies in a block");
    let position = func.block(block).insts.iter().position(|&value| value == access).unwrap_or(0);
    super::insert(func, block, position, Op::ArrayBase { table }, Type::NONE, Rep::Word)
}

fn entry_values(func: &Func, entry: &Loop, counted: Counted) -> Option<(Val, Val)> {
    let Term::Jump(edge) = &func.block(entry.preheader).term else {
        return None;
    };
    let params = &func.block(entry.header).params;
    let at = |param: Val| params.iter().position(|&known| known == param).map(|index| edge.args[index]);
    let start = at(counted.index)?;
    let count = at(counted.count)?;
    (func.rep(start) == Rep::Int && func.rep(count) == Rep::Int && func.rep(counted.index) == Rep::Int).then_some((start, count))
}

/// `None` when the loop could resize an array part some other way.
fn accesses(func: &Func, entry: &Loop, counted: Counted) -> Option<Vec<Access>> {
    let defined = func.definitions();
    let outside = |value: Val| defined[value.0 as usize].is_some_and(|block| !entry.blocks.contains(&block));
    let mut found = Vec::new();
    for &block in &entry.blocks {
        for &value in &func.block(block).insts {
            match *func.op(value) {
                Op::Index { object, key, .. } if key == counted.index && outside(object) && func.ty(object) == Type::TABLE => {
                    found.push(Access { value, table: object });
                }
                Op::NewIndex { .. } | Op::Helper(_) | Op::CallLua { .. } => return None,
                _ => {}
            }
        }
        if matches!(func.block(block).term, Term::TailCall { .. }) {
            return None;
        }
    }
    Some(found)
}

fn invariant_table(func: &mut Func, entry: &Loop, table: Val) -> Val {
    if func.rep(table) == Rep::Payload {
        return table;
    }
    let value = func.add_inst(Op::Unary(UnOp::Unbox, table), Type::TABLE, Rep::Payload);
    func.block_mut(entry.preheader).insts.push(value);
    value
}

/// A failed range guard disables this pass for its loop, not the accesses' own speculation.
pub(crate) const RANGE_SITE: u32 = 1 << 31;

fn range_site(func: &Func, entry: &Loop) -> u32 {
    RANGE_SITE | func.state(entry.entry_state).pc
}

fn add_range_guard(func: &mut Func, entry: &Loop, table: Val, (start, count, step): (Val, Val, i64)) {
    let mut moved = func.state(entry.entry_state).clone();
    moved.site = range_site(func, entry);
    let state = func.add_state(moved);
    let preheader = entry.preheader;
    let step_value = func.append(preheader, Op::Const(Const::Int(step)), Type::INT, Rep::Int);
    let span = func.append(preheader, Op::Binary(BinOp::IMul, count, step_value), Type::INT, Rep::Int);
    let end = func.append(preheader, Op::Binary(BinOp::IAdd, start, span), Type::INT, Rep::Int);
    let (low, high) = if step > 0 { (start, end) } else { (end, start) };
    func.append(preheader, Op::GuardRange { table, low, high, state }, Type::NONE, Rep::None);
}
