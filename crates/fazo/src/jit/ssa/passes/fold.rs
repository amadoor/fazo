//! Folds with `vm::fast` so results match the interpreter. Follows LuaJIT's `lj_opt_fold.c` (lj_opt_cse, line 2611).
//! Table-memory guards merge only within one block, since stores between blocks can change what they read.

use hashbrown::HashMap;

use super::remove;
use crate::{
    instruction::{ArithOp, UnaryOp},
    jit::ssa::ir::{BinOp, BlockId, BuiltinFn, Cc, Closure, Const, Func, Op, PureFn, Rep, Term, Type, UnOp, Val},
    value::Value,
    vm::fast,
};

pub(crate) fn run(func: &mut Func) {
    for _ in 0..8 {
        let mut changed = merge_blocks(func);
        changed |= remove_trivial_params(func);
        changed |= fold_constants(func);
        changed |= fold_branches(func);
        changed |= cse(func);
        if !changed {
            break;
        }
    }
}

/// Joins a block to its only predecessor so block-local memory CSE sees the whole straight line; loop headers and preheaders stay.
pub(crate) fn merge_blocks(func: &mut Func) -> bool {
    let order = func.reverse_postorder();
    let incoming = incoming_edges(func, &order);
    let pinned: Vec<BlockId> = func.loops.iter().flat_map(|entry| [entry.header, entry.preheader]).chain(func.entry).chain(func.osr.iter().map(|&(_, block)| block)).collect();
    let mut map = func.identity_map();
    let mut changed = false;
    for &block in &order {
        while let Term::Jump(edge) = func.block(block).term.clone() {
            let next = edge.block;
            if next == block || pinned.contains(&next) || incoming[next.0 as usize].len() != 1 || func.block(next).cold != func.block(block).cold {
                break;
            }
            let taken = std::mem::replace(func.block_mut(next), super::super::ir::Block { params: Vec::new(), insts: Vec::new(), term: Term::Open, cold: false });
            for (&param, &arg) in taken.params.iter().zip(&edge.args) {
                map[param.0 as usize] = arg;
            }
            let data = func.block_mut(block);
            data.insts.extend(taken.insts);
            data.term = taken.term;
            changed = true;
        }
    }
    if changed {
        func.rewrite_uses(&map);
    }
    changed
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
        Op::Builtin { function: BuiltinFn::Type, args, .. } => type_name(func, args[0]),
        _ => None,
    }
}

/// Follows `luaB_type`.
fn type_name(func: &Func, value: Val) -> Option<Const> {
    let names = func.type_names?;
    let ty = func.ty(value);
    if ty.is_none() {
        return None;
    }
    let kinds: [Type; 8] = [
        Type::NIL,
        Type::BOOL,
        Type::NUM,
        Type::of_tag(4),
        Type::TABLE,
        Type::CLOSURE.union(Type::of_tag(7)),
        Type::of_tag(8),
        Type::of_tag(9).union(Type::of_tag(10)),
    ];
    let slot = kinds.iter().position(|&kind| ty.within(kind))?;
    Some(Const::Gc { tag: 4, payload: names[slot] })
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
        for index in 0..func.block(block).insts.len() {
            let value = func.block(block).insts[index];
            let op = func.op(value);
            if let Some(redundant) = redundant_guard(func, op) {
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
            let Some(folded) = fold(func, op) else {
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

const KEY_OPERANDS: usize = 3;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Key([u64; 4 + KEY_OPERANDS]);

/// Frame states are left out, since the earlier op's frame serves the later one too.
fn key(op: &Op, rep: Rep, map: &[Val]) -> Option<(Key, bool)> {
    let (kind, extra, memory): (u64, [u64; 2], bool) = match op {
        Op::Const(constant) => (1, [u64::from(constant.tag()), constant.payload()], false),
        Op::Unary(unary, _) => (2, [*unary as u64, 0], false),
        Op::Binary(binary, ..) => (3, [binary_code(*binary), 0], false),
        Op::Pure(function, _) => (4, [*function as u64, 0], false),
        Op::EqConst(_, constant) => (5, [u64::from(constant.tag()), constant.payload()], false),
        Op::Select { .. } => (6, [0, 0], false),
        Op::GuardType { ty, .. } => (7, [u64::from(type_bits(*ty)), 0], false),
        Op::GuardNotNil { .. } => (8, [0, 0], false),
        Op::GuardEq { expected, .. } => (9, [*expected, 0], false),
        Op::GuardFalse { .. } => (11, [0, 0], false),
        Op::GuardShape { shape, .. } => (12, [*shape, 0], true),
        Op::GuardHasMeta { .. } => (13, [0, 0], true),
        Op::GuardNoMeta { bit, .. } => (14, [u64::from(*bit), 0], true),
        Op::ShapeIs { shape, .. } => (15, [*shape, 0], true),
        Op::GuardNode { node, nodes, key, .. } => (16, [u64::from(*node) << 32 | u64::from(*nodes), *key], true),
        Op::LoadUpval { closure, index } => (UPVALUE_KIND, [u64::from(*index), u64::from(matches!(closure, Closure::Root))], true),
        Op::LoadNode { node, .. } => (NODE_KIND, [u64::from(*node), 0], true),
        Op::LoadProto { .. } => (19, [0, 0], false),
        Op::Builtin { function, .. } if function.is_pure() => (20, [*function as u64, 0], false),
        _ => return None,
    };
    let mut words = [kind, rep_code(rep), extra[0], extra[1], u64::MAX, u64::MAX, u64::MAX];
    let mut count = 0;
    op.for_each_operand(|operand| {
        if let Some(word) = words.get_mut(4 + count) {
            *word = u64::from(super::super::ir::resolve(map, operand).0);
        }
        count += 1;
    });
    (count <= KEY_OPERANDS).then_some((Key(words), memory))
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
    match op {
        Op::Builtin { function, .. } => function.writes(),
        _ => matches!(op, Op::NewIndex { .. } | Op::Helper(_) | Op::CallLua { .. } | Op::StoreSlot { .. } | Op::AddSlot { .. } | Op::NewTable { .. }),
    }
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

const UPVALUE_KIND: u64 = 17;
const NODE_KIND: u64 = 18;

/// Follows LuaJIT's `fwd_ahload` (lj_opt_mem.c): a load of the node just stored reads the stored value. A store to the same node of another table may alias every load of that node.
fn forward_store(local: &mut HashMap<Key, Val>, map: &[Val], (table, node, stored): (Val, u32, Val)) {
    local.retain(|key, _| key.0[0] != NODE_KIND || key.0[2] != u64::from(node));
    let table = u64::from(super::super::ir::resolve(map, table).0);
    let key = Key([NODE_KIND, rep_code(Rep::Boxed), u64::from(node), 0, table, u64::MAX, u64::MAX]);
    local.insert(key, super::super::ir::resolve(map, stored));
}

/// A store to a captured register writes through its open upvalue.
fn clobbers_upvalues(op: &Op) -> bool {
    matches!(op, Op::StoreUpval { .. } | Op::StoreReg { .. })
}

fn cse_block(func: &Func, block: BlockId, (table, map, dead): (&mut HashMap<Key, Val>, &mut Vec<Val>, &mut Vec<bool>), changed: &mut bool) -> Vec<Key> {
    let mut added = Vec::new();
    let mut local: HashMap<Key, Val> = HashMap::new();
    for &value in &func.block(block).insts {
        let op = func.op(value);
        if clobbers_tables(op) {
            local.clear();
        } else if clobbers_upvalues(op) {
            local.retain(|key, _| key.0[0] != UPVALUE_KIND);
        }
        if let Op::StoreNode { table, node, value: stored, .. } = op {
            forward_store(&mut local, map, (*table, *node, *stored));
        }
        let Some((key, memory)) = key(op, func.rep(value), map) else {
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
            table.insert(key, value);
            added.push(key);
        }
    }
    added
}
