//! Folds with `vm::fast` so results match the interpreter. Follows LuaJIT's `lj_opt_fold.c` (lj_opt_cse, line 2611).
//! Table-memory guards merge only within one block, since stores between blocks can change what they read.

use std::collections::HashMap;

use super::remove;
use crate::{
    instruction::{ArithOp, UnaryOp},
    jit::ssa::ir::{BinOp, BlockId, Cc, Const, Func, Op, PureFn, Rep, Term, Type, UnOp, Val},
    value::Value,
    vm::fast,
};

pub(crate) fn run(func: &mut Func) {
    for _ in 0..8 {
        let mut changed = remove_trivial_params(func);
        changed |= fold_constants(func);
        changed |= fold_branches(func);
        changed |= cse(func);
        if !changed {
            break;
        }
    }
}

/// Braun et al., "Simple and Efficient Construction of Static Single Assignment Form", section 3.1.
pub(crate) fn remove_trivial_params(func: &mut Func) -> bool {
    let mut map = func.identity_map();
    let mut changed = false;
    let order = func.reverse_postorder();
    let incoming = incoming_edges(func, &order);
    for &block in &order {
        let params = func.block(block).params.clone();
        let mut keep = Vec::with_capacity(params.len());
        for (index, &param) in params.iter().enumerate() {
            let mut only = None;
            let mut trivial = true;
            for &(from, edge) in &incoming[block.0 as usize] {
                let arg = super::super::ir::resolve(&map, func.block(from).term.edges()[edge].args[index]);
                if arg == param || Some(arg) == only {
                    continue;
                }
                if only.is_some() {
                    trivial = false;
                    break;
                }
                only = Some(arg);
            }
            match (trivial, only) {
                (true, Some(value)) => {
                    map[param.0 as usize] = value;
                    changed = true;
                    keep.push(false);
                }
                _ => keep.push(true),
            }
        }
        if keep.iter().all(|&kept| kept) {
            continue;
        }
        drop_params(func, block, &keep, &incoming[block.0 as usize]);
    }
    func.rewrite_uses(&map);
    changed
}

pub(crate) fn incoming_edges(func: &Func, order: &[BlockId]) -> Vec<Vec<(BlockId, usize)>> {
    let mut incoming = vec![Vec::new(); func.blocks.len()];
    for &block in order {
        for (index, edge) in func.block(block).term.edges().into_iter().enumerate() {
            incoming[edge.block.0 as usize].push((block, index));
        }
    }
    incoming
}

pub(crate) fn drop_params(func: &mut Func, block: BlockId, keep: &[bool], incoming: &[(BlockId, usize)]) {
    let params = func.block(block).params.clone();
    func.block_mut(block).params = params.iter().zip(keep).filter(|&(_, &kept)| kept).map(|(&param, _)| param).collect();
    for &(from, edge) in incoming {
        let mut edges = func.block_mut(from).term.edges_mut();
        let args = &mut edges[edge].args;
        let kept: Vec<Val> = args.iter().zip(keep).filter(|&(_, &kept)| kept).map(|(&arg, _)| arg).collect();
        *args = kept;
    }
}

fn constant(func: &Func, value: Val) -> Option<Const> {
    match func.op(value) {
        Op::Const(constant) => Some(*constant),
        _ => None,
    }
}

fn lua(constant: Const) -> Option<Value> {
    match constant {
        Const::Nil => Some(Value::Nil),
        Const::Bool(flag) => Some(Value::Boolean(flag)),
        Const::Int(integer) => Some(Value::Integer(integer)),
        Const::Float(float) => Some(Value::Float(float)),
        Const::Gc { .. } => None,
    }
}

fn from_lua(value: Value) -> Option<Const> {
    match value {
        Value::Integer(integer) => Some(Const::Int(integer)),
        Value::Float(float) => Some(Const::Float(float)),
        Value::Boolean(flag) => Some(Const::Bool(flag)),
        _ => None,
    }
}

fn arith_of(op: BinOp) -> Option<ArithOp> {
    Some(match op {
        BinOp::IAdd | BinOp::FAdd => ArithOp::Add,
        BinOp::ISub | BinOp::FSub => ArithOp::Sub,
        BinOp::IMul | BinOp::FMul => ArithOp::Mul,
        BinOp::IMod => ArithOp::Mod,
        BinOp::IDiv | BinOp::FIDiv => ArithOp::IDiv,
        BinOp::IBand => ArithOp::BAnd,
        BinOp::IBor => ArithOp::BOr,
        BinOp::IBxor => ArithOp::BXor,
        BinOp::FDiv => ArithOp::Div,
        BinOp::ICmp(_) | BinOp::FCmp(_) | BinOp::BoolEq => return None,
    })
}

fn compare(cc: Cc, left: Value, right: Value) -> Option<bool> {
    match cc {
        Cc::Eq => fast::equal(left, right),
        Cc::Lt => fast::less_than(left, right),
        Cc::Le => fast::less_equal(left, right),
    }
}

pub(crate) fn fold(func: &Func, op: &Op) -> Option<Const> {
    match op {
        Op::Binary(BinOp::BoolEq, left, right) => Some(Const::Bool(constant(func, *left)?.same(constant(func, *right)?))),
        Op::Binary(BinOp::ICmp(cc) | BinOp::FCmp(cc), left, right) => {
            Some(Const::Bool(compare(*cc, lua(constant(func, *left)?)?, lua(constant(func, *right)?)?)?))
        }
        Op::Binary(binary, left, right) => {
            let value = fast::arith(arith_of(*binary)?, lua(constant(func, *left)?)?, lua(constant(func, *right)?)?)?;
            from_lua(value)
        }
        Op::Unary(UnOp::INeg | UnOp::FNeg, operand) => from_lua(fast::unary(UnaryOp::Minus, lua(constant(func, *operand)?)?)?),
        Op::Unary(UnOp::IBnot, operand) => from_lua(fast::unary(UnaryOp::BNot, lua(constant(func, *operand)?)?)?),
        Op::Unary(UnOp::IToF, operand) => match constant(func, *operand)? {
            Const::Int(integer) => Some(Const::Float(integer as f64)),
            _ => None,
        },
        Op::Unary(UnOp::Truthy, operand) => truthy(func, *operand).map(Const::Bool),
        Op::Unary(UnOp::Not, operand) => match constant(func, *operand)? {
            Const::Bool(flag) => Some(Const::Bool(!flag)),
            _ => None,
        },
        Op::EqConst(value, expected) => {
            let actual = constant(func, *value)?;
            if let (Some(left), Some(right)) = (lua(actual), lua(*expected))
                && let Some(equal) = fast::equal(left, right)
            {
                return Some(Const::Bool(equal));
            }
            Some(Const::Bool(actual.same(*expected)))
        }
        Op::Select { cond, yes, no } => match constant(func, *cond)? {
            Const::Bool(true) => constant(func, *yes),
            Const::Bool(false) => constant(func, *no),
            _ => None,
        },
        Op::Pure(function, args) => fold_pure(func, *function, args),
        _ => None,
    }
}

fn fold_pure(func: &Func, function: PureFn, args: &[Val]) -> Option<Const> {
    let op = match function {
        PureFn::FMod => ArithOp::Mod,
        PureFn::FPow => ArithOp::Pow,
        PureFn::Shl => ArithOp::Shl,
        PureFn::Shr => ArithOp::Shr,
        PureFn::LoopCount | PureFn::LtIF | PureFn::LeIF | PureFn::LtFI | PureFn::LeFI | PureFn::EqFI => {
            let left = lua(constant(func, *args.first()?)?)?;
            let right = lua(constant(func, *args.get(1)?)?)?;
            let cc = match function {
                PureFn::LtIF | PureFn::LtFI => Cc::Lt,
                PureFn::LeIF | PureFn::LeFI => Cc::Le,
                PureFn::EqFI => Cc::Eq,
                _ => return None,
            };
            return Some(Const::Bool(compare(cc, left, right)?));
        }
    };
    let left = lua(constant(func, args[0])?)?;
    let right = lua(constant(func, args[1])?)?;
    from_lua(fast::arith(op, left, right)?)
}

fn truthy(func: &Func, value: Val) -> Option<bool> {
    let ty = func.ty(value);
    if !ty.may_be_falsy() && !ty.is_none() {
        return Some(true);
    }
    if ty.within(Type::NIL) && !ty.is_none() {
        return Some(false);
    }
    match constant(func, value)? {
        Const::Bool(flag) => Some(flag),
        Const::Nil => Some(false),
        _ => Some(true),
    }
}

fn rep_of(constant: Const) -> Rep {
    match constant {
        Const::Int(_) => Rep::Int,
        Const::Float(_) => Rep::Float,
        Const::Bool(_) => Rep::Bool,
        Const::Nil | Const::Gc { tag: 10, .. } => Rep::Boxed,
        Const::Gc { .. } => Rep::Payload,
    }
}

fn fold_constants(func: &mut Func) -> bool {
    let mut changed = false;
    let mut map = func.identity_map();
    let mut dead = vec![false; func.insts.len()];
    for block in func.reverse_postorder() {
        for value in func.block(block).insts.clone() {
            let op = func.op(value).clone();
            if let Some(redundant) = redundant_guard(func, &op) {
                if let Some(input) = redundant {
                    map[value.0 as usize] = input;
                }
                dead[value.0 as usize] = true;
                changed = true;
                continue;
            }
            if matches!(op, Op::Const(_)) || !op.is_pure() {
                continue;
            }
            let Some(folded) = fold(func, &op) else {
                continue;
            };
            let rep = if func.rep(value) == Rep::Boxed { Rep::Boxed } else { rep_of(folded) };
            let inst = func.inst_mut(value);
            inst.op = Op::Const(folded);
            inst.ty = folded.ty();
            inst.rep = rep;
            changed = true;
        }
    }
    func.rewrite_uses(&map);
    remove(func, &dead);
    changed
}

fn redundant_guard(func: &Func, op: &Op) -> Option<Option<Val>> {
    match op {
        Op::GuardType { value, ty, .. } if constant(func, *value).is_some_and(|constant| constant.ty().within(*ty)) => Some(Some(*value)),
        Op::GuardNotNil { value, .. } if constant(func, *value).is_some_and(|constant| constant != Const::Nil) => Some(Some(*value)),
        Op::GuardFalse { cond, .. } if constant(func, *cond) == Some(Const::Bool(false)) => Some(None),
        _ => None,
    }
}

fn fold_branches(func: &mut Func) -> bool {
    let mut changed = false;
    for block in func.reverse_postorder() {
        let Term::Branch { cond, then, other } = func.block(block).term.clone() else {
            continue;
        };
        let taken = match constant(func, cond) {
            Some(Const::Bool(true)) => then,
            Some(Const::Bool(false)) => other,
            _ => continue,
        };
        func.block_mut(block).term = Term::Jump(taken);
        changed = true;
    }
    changed
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct Key(Vec<u64>);

/// Frame states are left out, since the earlier op's frame serves the later one too.
fn key(op: &Op, rep: Rep) -> Option<(Key, bool)> {
    let operands: Vec<u64> = op.operands().into_iter().map(|value| u64::from(value.0)).collect();
    let (kind, extra, memory): (u64, Vec<u64>, bool) = match op {
        Op::Const(constant) => (1, vec![u64::from(constant.tag()), constant.payload()], false),
        Op::Unary(unary, _) => (2, vec![*unary as u64], false),
        Op::Binary(binary, ..) => (3, vec![binary_code(*binary)], false),
        Op::Pure(function, _) => (4, vec![*function as u64], false),
        Op::EqConst(_, constant) => (5, vec![u64::from(constant.tag()), constant.payload()], false),
        Op::Select { .. } => (6, Vec::new(), false),
        Op::GuardType { ty, .. } => (7, vec![u64::from(type_bits(*ty))], false),
        Op::GuardNotNil { .. } => (8, Vec::new(), false),
        Op::GuardEq { expected, .. } => (9, vec![*expected], false),
        Op::GuardFalse { .. } => (11, Vec::new(), false),
        Op::GuardShape { shape, .. } => (12, vec![*shape], true),
        Op::GuardHasMeta { .. } => (13, Vec::new(), true),
        Op::GuardNoMeta { bit, .. } => (14, vec![u64::from(*bit)], true),
        Op::ShapeIs { shape, .. } => (15, vec![*shape], true),
        _ => return None,
    };
    let mut words = vec![kind, rep_code(rep)];
    words.extend(extra);
    words.extend(operands);
    Some((Key(words), memory))
}

fn type_bits(ty: Type) -> u16 {
    (0..11u8).filter(|&tag| !ty.intersect(Type::of_tag(tag)).is_none()).fold(0, |bits, tag| bits | 1 << tag)
}

fn binary_code(op: BinOp) -> u64 {
    match op {
        BinOp::IAdd => 0,
        BinOp::ISub => 1,
        BinOp::IMul => 2,
        BinOp::IMod => 3,
        BinOp::IDiv => 4,
        BinOp::IBand => 5,
        BinOp::IBor => 6,
        BinOp::IBxor => 7,
        BinOp::FAdd => 8,
        BinOp::FSub => 9,
        BinOp::FMul => 10,
        BinOp::FDiv => 11,
        BinOp::FIDiv => 12,
        BinOp::BoolEq => 13,
        BinOp::ICmp(cc) => 100 + cc as u64,
        BinOp::FCmp(cc) => 200 + cc as u64,
    }
}

fn rep_code(rep: Rep) -> u64 {
    match rep {
        Rep::None => 0,
        Rep::Boxed => 1,
        Rep::Int => 2,
        Rep::Float => 3,
        Rep::Bool => 4,
        Rep::Payload => 5,
        Rep::Word => 6,
    }
}

/// Changes table shapes, metatables or absence bits, so table-memory guards before it say nothing after it.
fn clobbers_tables(op: &Op) -> bool {
    matches!(op, Op::NewIndex { .. } | Op::Helper(_) | Op::CallLua { .. } | Op::StoreSlot { .. })
}

fn cse(func: &mut Func) -> bool {
    let idom = func.dominators();
    let order = func.reverse_postorder();
    let mut children: Vec<Vec<BlockId>> = vec![Vec::new(); func.blocks.len()];
    let mut roots = Vec::new();
    for &block in &order {
        match idom[block.0 as usize] {
            Some(parent) if parent.0 != u32::MAX && parent != block => children[parent.0 as usize].push(block),
            _ => roots.push(block),
        }
    }
    let mut map = func.identity_map();
    let mut dead = vec![false; func.insts.len()];
    let mut table: HashMap<Key, Val> = HashMap::new();
    let mut changed = false;
    for root in roots {
        let mut stack: Vec<(BlockId, Option<Vec<Key>>)> = vec![(root, None)];
        while let Some((block, undo)) = stack.pop() {
            if let Some(added) = undo {
                for key in added {
                    table.remove(&key);
                }
                continue;
            }
            let added = cse_block(func, block, (&mut table, &mut map, &mut dead), &mut changed);
            stack.push((block, Some(added)));
            for &child in children[block.0 as usize].iter().rev() {
                stack.push((child, None));
            }
        }
    }
    func.rewrite_uses(&map);
    remove(func, &dead);
    changed
}

fn cse_block(func: &Func, block: BlockId, (table, map, dead): (&mut HashMap<Key, Val>, &mut Vec<Val>, &mut Vec<bool>), changed: &mut bool) -> Vec<Key> {
    let mut added = Vec::new();
    let mut local: HashMap<Key, Val> = HashMap::new();
    for &value in &func.block(block).insts {
        let mut op = func.op(value).clone();
        op.map_operands(|operand| super::super::ir::resolve(map, operand));
        if clobbers_tables(&op) {
            local.clear();
        }
        let Some((key, memory)) = key(&op, func.rep(value)) else {
            continue;
        };
        let known = if memory { local.get(&key) } else { table.get(&key) };
        if let Some(&earlier) = known {
            map[value.0 as usize] = earlier;
            dead[value.0 as usize] = true;
            *changed = true;
            continue;
        }
        if memory {
            local.insert(key, value);
        } else {
            table.insert(key.clone(), value);
            added.push(key);
        }
    }
    added
}
