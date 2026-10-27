//! Follows V8's `MaglevPhiRepresentationSelector` (src/maglev/maglev-phi-representation-selector.cc).

use std::collections::HashMap;

use super::insert;
use crate::jit::ssa::{
    check::{binary_operand_rep, unary_operand_rep},
    ir::{BlockId, Closure, Edge, Func, Op, PureFn, Rep, Term, Type, UnOp, Val},
};

pub(crate) fn run(func: &mut Func) {
    assign(func);
    convert(func);
}

fn assign(func: &mut Func) {
    for block in func.reverse_postorder() {
        let data = func.block(block).clone();
        for value in data.params.into_iter().chain(data.insts) {
            let inst = func.inst(value);
            let rep = match &inst.op {
                Op::Param | Op::GuardType { .. } | Op::GuardNotNil { .. } | Op::Select { .. } => unboxed(inst.ty),
                Op::Unary(UnOp::Unbox, _) => unboxed(inst.ty),
                Op::Arith { .. } | Op::LoadReg { .. } | Op::LoadSlot { .. } | Op::LoadNode { .. } | Op::LoadArray { .. } | Op::Index { .. } | Op::LoadUpval { .. } | Op::Builtin { .. } => Rep::Boxed,
                _ => inst.rep,
            };
            func.inst_mut(value).rep = rep;
        }
    }
}

fn unboxed(ty: Type) -> Rep {
    if ty.is_none() {
        return Rep::Boxed;
    }
    ty.unboxed()
}

/// Ordered as `Op::map_operands` visits operands; `None` takes any representation.
fn required(func: &Func, value: Val) -> Vec<Option<Rep>> {
    let inst = func.inst(value);
    let boxed = Some(Rep::Boxed);
    let payload = Some(Rep::Payload);
    let int = Some(Rep::Int);
    match &inst.op {
        Op::Binary(op, ..) => vec![Some(binary_operand_rep(*op)); 2],
        Op::Unary(op, _) => vec![unary_operand_rep(*op)],
        Op::Pure(function, args) => match function {
            PureFn::FMod | PureFn::FPow => vec![Some(Rep::Float); 2],
            PureFn::Shl | PureFn::Shr | PureFn::LoopCount => vec![int; args.len()],
            PureFn::LtIF | PureFn::LeIF => vec![int, Some(Rep::Float)],
            PureFn::LtFI | PureFn::LeFI | PureFn::EqFI => vec![Some(Rep::Float), int],
        },
        Op::Select { .. } => vec![Some(Rep::Bool), Some(inst.rep), Some(inst.rep)],
        Op::StoreReg { .. } => vec![boxed],
        Op::StoreUpval { closure, .. } => match closure {
            Closure::Root | Closure::Known(_) => vec![boxed],
            Closure::Value(_) => vec![payload, boxed],
        },
        Op::LoadUpval { closure: Closure::Value(_), .. } | Op::LoadProto { .. } => vec![payload],
        Op::LoadSlot { .. } | Op::LoadNode { .. } | Op::GuardNode { .. } | Op::ShapeIs { .. } | Op::GuardShape { .. } | Op::GuardHasMeta { .. } | Op::GuardNoMeta { .. } | Op::GuardAbsent { .. } => {
            vec![payload]
        }
        Op::ArrayBase { .. } | Op::TableAddr { .. } => vec![payload],
        Op::LoadArray { .. } => vec![payload, None, int],
        Op::GuardRange { .. } => vec![payload, int, int],
        Op::StoreSlot { .. } | Op::StoreNode { .. } => vec![payload, boxed],
        Op::AddSlot { .. } => vec![payload, boxed, boxed],
        Op::NewIndex { .. } => vec![None, None, boxed],
        Op::GuardEq { value: operand, .. } => match func.rep(*operand) {
            Rep::Word | Rep::Payload => vec![None],
            _ => vec![payload],
        },
        Op::GuardFalse { .. } => vec![Some(Rep::Bool)],
        Op::Arith { .. } | Op::Compare { .. } => vec![boxed; 2],
        Op::Builtin { args, .. } => vec![boxed; args.len()],
        Op::NewTable { fields, .. } => vec![boxed; fields.len()],
        Op::Len { .. } => vec![boxed],
        _ => Vec::new(),
    }
}

/// A conversion serves every block its block dominates, as V8 keeps a node's converted alternatives in its `NodeInfo`
/// (src/maglev/maglev-graph-builder.cc, `GetTaggedValue`).
fn convert(func: &mut Func) {
    let order = func.reverse_postorder();
    let idom = func.dominators_in(&order);
    let mut children: Vec<Vec<BlockId>> = vec![Vec::new(); func.blocks.len()];
    let mut roots = Vec::new();
    for &block in &order {
        match idom[block.0 as usize] {
            Some(parent) if parent.0 != u32::MAX && parent != block => children[parent.0 as usize].push(block),
            _ => roots.push(block),
        }
    }
    let mut cache = Cache::default();
    for root in roots {
        let mut stack: Vec<(BlockId, Option<usize>)> = vec![(root, None)];
        while let Some((block, undo)) = stack.pop() {
            if let Some(mark) = undo {
                cache.restore(mark);
                continue;
            }
            stack.push((block, Some(cache.added.len())));
            convert_block(func, block, &mut cache);
            for &child in children[block.0 as usize].iter().rev() {
                stack.push((child, None));
            }
        }
    }
}

#[derive(Default)]
struct Cache {
    known: HashMap<(Val, Rep), Val>,
    added: Vec<(Val, Rep)>,
}

impl Cache {
    fn get(&self, key: (Val, Rep)) -> Option<Val> {
        self.known.get(&key).copied()
    }

    fn insert(&mut self, key: (Val, Rep), value: Val) {
        self.known.insert(key, value);
        self.added.push(key);
    }

    fn restore(&mut self, mark: usize) {
        for key in self.added.drain(mark..) {
            self.known.remove(&key);
        }
    }
}

fn convert_block(func: &mut Func, block: BlockId, cache: &mut Cache) {
    let mut position = 0;
    while position < func.block(block).insts.len() {
        let value = func.block(block).insts[position];
        let wants = required(func, value);
        let mut op = func.op(value).clone();
        let mut inserted = 0;
        let mut index = 0;
        let mut conversions = Vec::new();
        op.map_operands(|operand| {
            let wanted = wants.get(index).copied().flatten();
            index += 1;
            match wanted {
                Some(rep) if func.rep(operand) != rep => {
                    conversions.push((operand, rep));
                    operand
                }
                _ => operand,
            }
        });
        let mut converted = Vec::new();
        for (operand, rep) in conversions {
            let replacement = conversion(func, cache, (block, position + inserted), operand, rep, &mut inserted);
            converted.push((operand, rep, replacement));
        }
        let mut index = 0;
        op.map_operands(|operand| {
            let wanted = wants.get(index).copied().flatten();
            index += 1;
            match wanted {
                Some(rep) => converted.iter().find(|&&(from, to, _)| from == operand && to == rep).map_or(operand, |&(_, _, to)| to),
                None => operand,
            }
        });
        func.inst_mut(value).op = op;
        position += inserted + 1;
    }
    convert_edges(func, block, cache);
}

fn conversion(func: &mut Func, cache: &mut Cache, (block, position): (BlockId, usize), operand: Val, rep: Rep, inserted: &mut usize) -> Val {
    if let Some(known) = cache.get((operand, rep)) {
        return known;
    }
    let ty = func.ty(operand);
    let op = if rep == Rep::Boxed { Op::Unary(UnOp::Box, operand) } else { Op::Unary(UnOp::Unbox, operand) };
    let value = insert(func, block, position, op, ty, rep);
    *inserted += 1;
    cache.insert((operand, rep), value);
    value
}

/// Conversions go before a branch's trailing comparison, so the backend can still fuse the two.
fn convert_edges(func: &mut Func, block: BlockId, cache: &mut Cache) {
    let mut term = func.block(block).term.clone();
    let compared = match term {
        Term::Branch { cond, .. } if func.block(block).insts.last() == Some(&cond) => 1,
        _ => 0,
    };
    let branches = term.edges().len() > 1;
    for edge in term.edges_mut() {
        let params = func.block(edge.block).params.clone();
        let converts = edge.args.iter().zip(&params).any(|(&arg, &param)| func.rep(arg) != func.rep(param));
        if converts && branches && leaves_loop(func, block, edge.block) {
            *edge = convert_on_exit(func, (block, edge.block), &edge.args);
            continue;
        }
        for (arg, param) in edge.args.iter_mut().zip(params) {
            let rep = func.rep(param);
            if func.rep(*arg) == rep {
                continue;
            }
            let end = func.block(block).insts.len() - compared;
            let mut inserted = 0;
            *arg = conversion(func, cache, (block, end), *arg, rep, &mut inserted);
        }
    }
    func.block_mut(block).term = term;
}

fn leaves_loop(func: &Func, from: BlockId, to: BlockId) -> bool {
    func.loops.iter().any(|entry| entry.blocks.contains(&from) && !entry.blocks.contains(&to))
}

/// A loop exit converts in a block of its own, so the iterations that stay in the loop don't run the conversions.
fn convert_on_exit(func: &mut Func, (from, to): (BlockId, BlockId), args: &[Val]) -> Edge {
    let exit = func.new_block();
    let params = func.block(to).params.clone();
    let mut converted = Vec::with_capacity(args.len());
    for (&arg, param) in args.iter().zip(params) {
        let rep = func.rep(param);
        if func.rep(arg) == rep {
            converted.push(arg);
            continue;
        }
        let op = if rep == Rep::Boxed { Op::Unary(UnOp::Box, arg) } else { Op::Unary(UnOp::Unbox, arg) };
        let ty = func.ty(arg);
        converted.push(func.append(exit, op, ty, rep));
    }
    func.block_mut(exit).term = Term::Jump(Edge { block: to, args: converted });
    for entry in &mut func.loops {
        if entry.blocks.contains(&from) && entry.blocks.contains(&to) {
            entry.blocks.push(exit);
        }
    }
    Edge { block: exit, args: Vec::new() }
}
