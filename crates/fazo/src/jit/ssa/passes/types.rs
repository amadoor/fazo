//! Starts from no type and meets at block parameters until a fixpoint, so loop parameters settle at their true types.
//! Follows V8 Maglev's known node types (src/maglev/maglev-known-node-aspects.cc).

use super::{insert, remove};
use crate::jit::ssa::ir::{BinOp, BlockId, Cc, Const, Func, KIND_FLOAT, KIND_INT, NumOp, Op, PureFn, Rep, StateId, Type, UnOp, Val};

pub(crate) fn run(func: &mut Func) {
    propagate(func);
    speculate(func);
    propagate(func);
    eliminate_guards(func);
    specialize(func);
    propagate(func);
}

pub(crate) fn propagate(func: &mut Func) {
    let order = func.reverse_postorder();
    for &block in &order {
        for param in func.block(block).params.clone() {
            func.inst_mut(param).ty = Type::NONE;
        }
    }
    let mut changed = true;
    while changed {
        changed = false;
        for &block in &order {
            for value in func.block(block).insts.clone() {
                let ty = transfer(func, value);
                if func.inst(value).ty != ty {
                    func.inst_mut(value).ty = ty;
                    changed = true;
                }
            }
            changed |= flow_edges(func, block);
        }
    }
}

fn flow_edges(func: &mut Func, block: BlockId) -> bool {
    let mut changed = false;
    let edges: Vec<(BlockId, Vec<Val>)> = func.block(block).term.edges().into_iter().map(|edge| (edge.block, edge.args.clone())).collect();
    for (target, args) in edges {
        let params = func.block(target).params.clone();
        for (param, arg) in params.into_iter().zip(args) {
            let merged = func.ty(param).union(func.ty(arg));
            if merged != func.ty(param) {
                func.inst_mut(param).ty = merged;
                changed = true;
            }
        }
    }
    changed
}

pub(crate) fn transfer(func: &Func, value: Val) -> Type {
    let inst = func.inst(value);
    let ty = |operand: &Val| func.ty(*operand);
    match &inst.op {
        Op::Const(constant) => constant.ty(),
        Op::GuardType { value, ty: wanted, .. } => ty(value).intersect(*wanted),
        Op::GuardNotNil { value, .. } => ty(value).without(Type::NIL),
        Op::Unary(UnOp::Box | UnOp::Unbox, operand) => ty(operand).intersect(inst.ty.union(ty(operand))),
        Op::Select { yes, no, .. } => ty(yes).union(ty(no)),
        Op::Arith { op, left, right, .. } => arith_type(*op, ty(left), ty(right)),
        _ => inst.ty,
    }
}

/// Non-number operands deoptimize, so only their numeric kinds count.
pub(crate) fn arith_type(op: NumOp, left: Type, right: Type) -> Type {
    let left = left.intersect(Type::NUM);
    let right = right.intersect(Type::NUM);
    if left.is_none() || right.is_none() {
        return Type::NONE;
    }
    match op {
        NumOp::Div | NumOp::Pow => Type::FLOAT,
        NumOp::BAnd | NumOp::BOr | NumOp::BXor | NumOp::Shl | NumOp::Shr | NumOp::BNot => Type::INT,
        NumOp::Neg => left,
        NumOp::Add | NumOp::Sub | NumOp::Mul | NumOp::Mod | NumOp::IDiv => {
            let mut result = Type::NONE;
            if !left.intersect(Type::INT).is_none() && !right.intersect(Type::INT).is_none() {
                result = result.union(Type::INT);
            }
            if !left.intersect(Type::FLOAT).is_none() || !right.intersect(Type::FLOAT).is_none() {
                result = result.union(Type::FLOAT);
            }
            result
        }
    }
}

/// Follows Maglev's feedback speculation (src/maglev/maglev-graph-builder.cc); after a failed guess the function guesses floats.
fn speculate(func: &mut Func) {
    let idom = func.dominators();
    let mut refined = Refined { idom, guards: Vec::new() };
    for block in func.reverse_postorder() {
        let mut position = 0;
        while position < func.block(block).insts.len() {
            let value = func.block(block).insts[position];
            let inserted = speculate_operands(func, &mut refined, (block, position), value);
            if inserted == 0 {
                func.inst_mut(value).ty = transfer(func, value);
            }
            position += inserted + 1;
        }
    }
}

/// Follows Maglev's known node aspects (src/maglev/maglev-known-node-aspects.cc).
struct Refined {
    idom: Vec<Option<BlockId>>,
    guards: Vec<(Val, Val, BlockId)>,
}

impl Refined {
    fn lookup(&self, value: Val, block: BlockId) -> Val {
        self.guards
            .iter()
            .rev()
            .find(|&&(input, _, at)| input == value && Func::dominates(&self.idom, at, block))
            .map_or(value, |&(_, guard, _)| guard)
    }
}

fn speculate_operands(func: &mut Func, refined: &mut Refined, (block, position): (BlockId, usize), value: Val) -> usize {
    let (left, right, state) = match func.op(value) {
        Op::Arith { left, right, state, .. } | Op::Compare { cc: Cc::Lt | Cc::Le | Cc::Eq, left, right, state } => (*left, *right, *state),
        _ => return 0,
    };
    let (left, right) = (refined.lookup(left, block), refined.lookup(right, block));
    set_operands(func, value, left, right);
    if matches!(func.op(value), Op::Compare { cc: Cc::Eq, .. }) && kind(func.ty(left)).is_none() && kind(func.ty(right)).is_none() {
        return 0;
    }
    let site = func.state(state).site;
    let observed = func.observed.iter().find(|&&(known, _)| known == site).map(|&(_, kinds)| kinds);
    let fallback = match observed {
        Some(KIND_INT) => Type::INT,
        Some(KIND_FLOAT) => Type::FLOAT,
        Some(_) => Type::NUM,
        _ if func.float_guess => Type::FLOAT,
        _ => Type::INT,
    };
    let guess = if fallback == Type::NUM { fallback } else { kind(func.ty(left)).or(kind(func.ty(right))).unwrap_or(fallback) };
    let mut inserted = 0;
    let mut guarded = |func: &mut Func, operand: Val| {
        let ty = func.ty(operand);
        if kind(ty).is_some() || ty.intersect(guess).is_none() || ty.within(Type::NUM) {
            return operand;
        }
        let guard = insert(func, block, position + inserted, Op::GuardType { value: operand, ty: guess, state }, guess, Rep::Boxed);
        refined.guards.push((operand, guard, block));
        inserted += 1;
        guard
    };
    let new_left = guarded(func, left);
    let new_right = if right == left { new_left } else { guarded(func, right) };
    set_operands(func, value, new_left, new_right);
    func.inst_mut(value).ty = transfer(func, value);
    inserted
}

fn set_operands(func: &mut Func, value: Val, new_left: Val, new_right: Val) {
    if let Op::Arith { left, right, .. } | Op::Compare { left, right, .. } = &mut func.inst_mut(value).op {
        *left = new_left;
        *right = new_right;
    }
}

fn eliminate_guards(func: &mut Func) {
    let mut map = func.identity_map();
    let mut dead = vec![false; func.insts.len()];
    for block in func.reverse_postorder() {
        for &value in &func.block(block).insts {
            let redundant = match func.op(value) {
                Op::GuardType { value: input, ty, .. } if func.ty(*input).within(*ty) => Some(*input),
                Op::GuardNotNil { value: input, .. } if func.ty(*input).intersect(Type::NIL).is_none() => Some(*input),
                _ => None,
            };
            if let Some(input) = redundant {
                map[value.0 as usize] = input;
                dead[value.0 as usize] = true;
            }
        }
    }
    func.rewrite_uses(&map);
    remove(func, &dead);
}

fn specialize(func: &mut Func) {
    for block in func.reverse_postorder() {
        let mut position = 0;
        while position < func.block(block).insts.len() {
            let value = func.block(block).insts[position];
            let inserted = match func.op(value).clone() {
                Op::Arith { op, left, right, state } => specialize_arith(func, (block, position), value, (op, left, right, state)),
                Op::Compare { cc, left, right, .. } => specialize_compare(func, (block, position), value, (cc, left, right)),
                _ => 0,
            };
            position += inserted + 1;
        }
    }
}

struct Site {
    block: BlockId,
    position: usize,
    inserted: usize,
}

impl Site {
    fn add(&mut self, func: &mut Func, op: Op, ty: Type, rep: Rep) -> Val {
        let value = insert(func, self.block, self.position + self.inserted, op, ty, rep);
        self.inserted += 1;
        value
    }

    fn float(&mut self, func: &mut Func, operand: Val) -> Val {
        if func.ty(operand) == Type::FLOAT {
            return operand;
        }
        if let Op::Const(Const::Int(integer)) = func.op(operand) {
            let float = *integer as f64;
            return self.add(func, Op::Const(Const::Float(float)), Type::FLOAT, Rep::Float);
        }
        self.add(func, Op::Unary(UnOp::IToF, operand), Type::FLOAT, Rep::Float)
    }
}

fn kind(ty: Type) -> Option<Type> {
    if ty == Type::INT || ty == Type::FLOAT {
        return Some(ty);
    }
    None
}

fn specialize_arith(func: &mut Func, (block, position): (BlockId, usize), value: Val, (op, left, right, state): (NumOp, Val, Val, StateId)) -> usize {
    let (Some(left_kind), Some(right_kind)) = (kind(func.ty(left)), kind(func.ty(right))) else {
        return 0;
    };
    let mut site = Site { block, position, inserted: 0 };
    let integers = left_kind == Type::INT && right_kind == Type::INT;
    let new = if integers { integer_op(func, &mut site, op, (left, right), state) } else { float_op(func, &mut site, op, (left, right)) };
    let Some((new, ty)) = new else {
        return site.inserted;
    };
    let rep = if ty == Type::INT { Rep::Int } else { Rep::Float };
    let inst = func.inst_mut(value);
    inst.op = new;
    inst.ty = ty;
    inst.rep = rep;
    site.inserted
}

fn integer_op(func: &mut Func, site: &mut Site, op: NumOp, (left, right): (Val, Val), state: StateId) -> Option<(Op, Type)> {
    let binary = |op: BinOp| Some((Op::Binary(op, left, right), Type::INT));
    match op {
        NumOp::Add => binary(BinOp::IAdd),
        NumOp::Sub => binary(BinOp::ISub),
        NumOp::Mul => binary(BinOp::IMul),
        NumOp::BAnd => binary(BinOp::IBand),
        NumOp::BOr => binary(BinOp::IBor),
        NumOp::BXor => binary(BinOp::IBxor),
        NumOp::Mod | NumOp::IDiv => {
            guard_nonzero(func, site, right, state);
            binary(if op == NumOp::Mod { BinOp::IMod } else { BinOp::IDiv })
        }
        NumOp::Shl => Some((Op::Pure(PureFn::Shl, Box::new([left, right])), Type::INT)),
        NumOp::Shr => Some((Op::Pure(PureFn::Shr, Box::new([left, right])), Type::INT)),
        NumOp::Neg => Some((Op::Unary(UnOp::INeg, left), Type::INT)),
        NumOp::BNot => Some((Op::Unary(UnOp::IBnot, left), Type::INT)),
        NumOp::Div | NumOp::Pow => float_op(func, site, op, (left, right)),
    }
}

/// Integer `%` and `//` by zero raise an error, so the guard hands it to the interpreter.
fn guard_nonzero(func: &mut Func, site: &mut Site, divisor: Val, state: StateId) {
    if matches!(func.op(divisor), Op::Const(Const::Int(divisor)) if *divisor != 0) {
        return;
    }
    let zero = site.add(func, Op::Const(Const::Int(0)), Type::INT, Rep::Int);
    let is_zero = site.add(func, Op::Binary(BinOp::ICmp(Cc::Eq), divisor, zero), Type::BOOL, Rep::Bool);
    site.add(func, Op::GuardFalse { cond: is_zero, state }, Type::NONE, Rep::None);
}

/// Bitwise operators on floats need an exact integer conversion, which stays in the generic op.
fn float_op(func: &mut Func, site: &mut Site, op: NumOp, (left, right): (Val, Val)) -> Option<(Op, Type)> {
    let unary = matches!(op, NumOp::Neg | NumOp::BNot);
    if matches!(op, NumOp::BAnd | NumOp::BOr | NumOp::BXor | NumOp::Shl | NumOp::Shr | NumOp::BNot) {
        return None;
    }
    let left = site.float(func, left);
    if unary {
        return Some((Op::Unary(UnOp::FNeg, left), Type::FLOAT));
    }
    let right = site.float(func, right);
    let binary = |op: BinOp| Some((Op::Binary(op, left, right), Type::FLOAT));
    match op {
        NumOp::Add => binary(BinOp::FAdd),
        NumOp::Sub => binary(BinOp::FSub),
        NumOp::Mul => binary(BinOp::FMul),
        NumOp::Div => binary(BinOp::FDiv),
        NumOp::IDiv => binary(BinOp::FIDiv),
        NumOp::Mod => Some((Op::Pure(PureFn::FMod, Box::new([left, right])), Type::FLOAT)),
        NumOp::Pow => Some((Op::Pure(PureFn::FPow, Box::new([left, right])), Type::FLOAT)),
        NumOp::BAnd | NumOp::BOr | NumOp::BXor | NumOp::Shl | NumOp::Shr | NumOp::Neg | NumOp::BNot => None,
    }
}

/// Mixed integer and float comparisons call the interpreter's exact comparisons rather than converting.
fn specialize_compare(func: &mut Func, (block, position): (BlockId, usize), value: Val, (cc, left, right): (Cc, Val, Val)) -> usize {
    let (Some(left_kind), Some(right_kind)) = (kind(func.ty(left)), kind(func.ty(right))) else {
        return 0;
    };
    let mut site = Site { block, position, inserted: 0 };
    let new = match (left_kind == Type::INT, right_kind == Type::INT) {
        (true, true) => Op::Binary(BinOp::ICmp(cc), left, right),
        (false, false) => Op::Binary(BinOp::FCmp(cc), left, right),
        (true, false) => match cc {
            Cc::Eq => Op::Pure(PureFn::EqFI, Box::new([right, left])),
            Cc::Lt => Op::Pure(PureFn::LtIF, Box::new([left, right])),
            Cc::Le => Op::Pure(PureFn::LeIF, Box::new([left, right])),
        },
        (false, true) => match (cc, func.op(right).clone()) {
            (_, Op::Const(Const::Int(integer))) if integer.unsigned_abs() < 1 << 53 => {
                let float = site.add(func, Op::Const(Const::Float(integer as f64)), Type::FLOAT, Rep::Float);
                Op::Binary(BinOp::FCmp(cc), left, float)
            }
            (Cc::Eq, _) => Op::Pure(PureFn::EqFI, Box::new([left, right])),
            (Cc::Lt, _) => Op::Pure(PureFn::LtFI, Box::new([left, right])),
            (Cc::Le, _) => Op::Pure(PureFn::LeFI, Box::new([left, right])),
        },
    };
    let inst = func.inst_mut(value);
    inst.op = new;
    inst.ty = Type::BOOL;
    inst.rep = Rep::Bool;
    site.inserted
}
