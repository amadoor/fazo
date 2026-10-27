//! Load forwarding over table slots, shapes and metatables, following V8's `LoadElimination` (src/compiler/load-elimination.cc).
//! A fresh table cannot alias any other table, so stores elsewhere leave its known facts intact.

use super::{fold::incoming_edges, remove};
use crate::jit::ssa::ir::{BlockId, BuiltinFn, Func, Op, Type, UnOp, Val};

#[derive(Clone, Default)]
struct Table {
    fresh: bool,
    shape: Option<u64>,
    meta: Option<Option<Val>>,
    slots: Vec<(u32, Val)>,
}

#[derive(Clone, Default)]
struct Facts {
    tables: Vec<(Val, Table)>,
}

impl Facts {
    fn get(&self, table: Val) -> Option<&Table> {
        self.tables.iter().find(|(known, _)| *known == table).map(|(_, facts)| facts)
    }

    fn entry(&mut self, table: Val) -> &mut Table {
        let index = match self.tables.iter().position(|(known, _)| *known == table) {
            Some(index) => index,
            None => {
                self.tables.push((table, Table::default()));
                self.tables.len() - 1
            }
        };
        &mut self.tables[index].1
    }

    fn distinct(&self, left: Val, right: Val) -> bool {
        left != right && (self.get(left).is_some_and(|facts| facts.fresh) || self.get(right).is_some_and(|facts| facts.fresh))
    }

    /// Tables not known to be different objects may alias `table`, so its store kills their facts.
    fn store_slot(&mut self, table: Val, slot: u32, value: Val) {
        let others: Vec<Val> = self.tables.iter().map(|(known, _)| *known).filter(|&known| !self.distinct(known, table)).collect();
        for other in others {
            self.entry(other).slots.retain(|&(known, _)| known != slot);
        }
        let facts = self.entry(table);
        facts.slots.retain(|&(known, _)| known != slot);
        facts.slots.push((slot, value));
    }

    fn reshape(&mut self, table: Val) {
        let others: Vec<Val> = self.tables.iter().map(|(known, _)| *known).filter(|&known| !self.distinct(known, table)).collect();
        for other in others {
            let facts = self.entry(other);
            facts.shape = None;
            facts.meta = None;
            facts.slots.clear();
        }
    }

    fn remeta(&mut self, table: Val) {
        let others: Vec<Val> = self.tables.iter().map(|(known, _)| *known).filter(|&known| known != table && !self.distinct(known, table)).collect();
        for other in others {
            self.entry(other).meta = None;
        }
    }

    /// A collector step may clear entries of weak tables; only fresh tables are known not to be weak.
    fn collect(&mut self) {
        for (_, facts) in &mut self.tables {
            if !facts.fresh {
                facts.slots.clear();
            }
        }
    }
}

pub(crate) fn run(func: &mut Func) {
    let order = func.reverse_postorder();
    let idom = func.dominators_in(&order);
    let incoming = incoming_edges(func, &order);
    let mut children: Vec<Vec<BlockId>> = vec![Vec::new(); func.blocks.len()];
    let mut roots = Vec::new();
    for &block in &order {
        match idom[block.0 as usize] {
            Some(parent) if parent.0 != u32::MAX && parent != block => children[parent.0 as usize].push(block),
            _ => roots.push(block),
        }
    }
    let disturbs: Vec<bool> = func.blocks.iter().map(|block| block.insts.iter().any(|&value| disturbs_tables(func.op(value)))).collect();
    let mut seen = vec![u32::MAX; func.blocks.len()];
    let mut map = func.identity_map();
    let mut dead = vec![false; func.insts.len()];
    let mut stack: Vec<(BlockId, Facts)> = roots.into_iter().map(|root| (root, Facts::default())).collect();
    while let Some((block, inherited)) = stack.pop() {
        let keeps = match incoming[block.0 as usize].len() {
            1 => true,
            _ if inherited.tables.is_empty() => false,
            _ => idom[block.0 as usize].is_some_and(|parent| quiet_between(parent, block, (&incoming, &disturbs), &mut seen)),
        };
        let mut facts = if keeps { inherited } else { Facts::default() };
        forward_block(func, block, &mut facts, (&mut map, &mut dead));
        for &child in &children[block.0 as usize] {
            stack.push((child, facts.clone()));
        }
    }
    func.rewrite_uses(&map);
    remove(func, &dead);
}

type Incoming = [Vec<(BlockId, usize)>];

/// Facts from the dominator hold at a merge when no block on a path between them can change table memory, as Maglev
/// keeps known node aspects across a loop header its `LoopEffects` do not clobber (maglev-graph-builder.cc line 14786).
fn quiet_between(dominator: BlockId, merge: BlockId, (incoming, disturbs): (&Incoming, &[bool]), seen: &mut [u32]) -> bool {
    if dominator.0 == u32::MAX || dominator == merge {
        return false;
    }
    let stamp = merge.0;
    let mut work: Vec<BlockId> = incoming[merge.0 as usize].iter().map(|&(pred, _)| pred).collect();
    while let Some(block) = work.pop() {
        if block == dominator || seen[block.0 as usize] == stamp {
            continue;
        }
        seen[block.0 as usize] = stamp;
        if disturbs[block.0 as usize] {
            return false;
        }
        work.extend(incoming[block.0 as usize].iter().map(|&(pred, _)| pred));
    }
    true
}

/// Reads and guards leave table memory as it was; anything else may run code or allocate.
fn disturbs_tables(op: &Op) -> bool {
    !op.is_pure()
        && !matches!(
            op,
            Op::GuardType { .. }
                | Op::GuardNotNil { .. }
                | Op::GuardEq { .. }
                | Op::GuardFalse { .. }
                | Op::GuardShape { .. }
                | Op::GuardNode { .. }
                | Op::GuardHasMeta { .. }
                | Op::GuardNoMeta { .. }
                | Op::GuardRange { .. }
                | Op::ShapeIs { .. }
                | Op::LoadSlot { .. }
                | Op::LoadNode { .. }
                | Op::LoadProto { .. }
                | Op::LoadUpval { .. }
                | Op::LoadReg { .. }
                | Op::ArrayBase { .. }
                | Op::TableAddr { .. }
                | Op::Requests
                | Op::Safepoint { .. }
                | Op::StoreReg { .. }
                | Op::Param
        )
}

fn base(func: &Func, map: &[Val], mut value: Val) -> Val {
    loop {
        value = crate::jit::ssa::ir::resolve(map, value);
        value = match func.op(value) {
            Op::GuardType { value: input, .. } | Op::Unary(UnOp::Box | UnOp::Unbox, input) => *input,
            Op::Builtin { function: BuiltinFn::SetMetatable, args, .. } => args[0],
            _ => return value,
        };
    }
}

fn forward_block(func: &mut Func, block: BlockId, facts: &mut Facts, (map, dead): (&mut [Val], &mut [bool])) {
    for value in func.block(block).insts.clone() {
        let op = func.op(value).clone();
        match op {
            Op::NewTable { shape, fields, .. } => {
                facts.collect();
                let slots = fields.iter().enumerate().map(|(slot, &field)| (slot as u32, field)).collect();
                let root = crate::shape::ROOT.packed();
                facts.tables.retain(|(known, _)| *known != value);
                facts.tables.push((value, Table { fresh: true, shape: Some(shape.unwrap_or(root)), meta: Some(None), slots }));
            }
            Op::GuardShape { table, shape, .. } => {
                let table = base(func, map, table);
                if facts.get(table).and_then(|known| known.shape) == Some(shape) {
                    dead[value.0 as usize] = true;
                    continue;
                }
                facts.entry(table).shape = Some(shape);
            }
            Op::LoadSlot { table, slot } => {
                let table = base(func, map, table);
                if let Some(&(_, known)) = facts.get(table).and_then(|known| known.slots.iter().find(|&&(at, _)| at == slot)) {
                    map[value.0 as usize] = known;
                    dead[value.0 as usize] = true;
                    continue;
                }
                facts.entry(table).slots.push((slot, value));
            }
            Op::StoreSlot { table, slot, value: stored, .. } => {
                let table = base(func, map, table);
                facts.store_slot(table, slot, stored);
            }
            Op::GuardHasMeta { table, state } => {
                let table = base(func, map, table);
                match facts.get(table).and_then(|known| known.meta) {
                    Some(Some(meta)) if func.ty(meta).within(Type::TABLE) => {
                        map[value.0 as usize] = meta;
                        dead[value.0 as usize] = true;
                    }
                    Some(Some(meta)) => func.inst_mut(value).op = Op::GuardType { value: meta, ty: Type::TABLE, state },
                    _ => facts.entry(table).meta = Some(Some(value)),
                }
            }
            Op::GuardNoMeta { table, .. } => {
                let table = base(func, map, table);
                if facts.get(table).and_then(|known| known.meta) == Some(None) {
                    dead[value.0 as usize] = true;
                }
            }
            // The builtin either succeeded, so the metatable is nil or a table, or left compiled code.
            Op::Builtin { function: BuiltinFn::SetMetatable, args, .. } => {
                let table = base(func, map, args[0]);
                let meta = args[1];
                facts.remeta(table);
                let ty = func.ty(meta);
                facts.entry(table).meta = if ty == Type::NIL {
                    Some(None)
                } else if ty.intersect(Type::TABLE).is_none() {
                    None
                } else {
                    Some(Some(meta))
                };
            }
            Op::NewIndex { object, .. } | Op::AddSlot { table: object, .. } => {
                let table = base(func, map, object);
                facts.reshape(table);
            }
            Op::Helper(_) | Op::CallLua { .. } => facts.tables.clear(),
            _ => {}
        }
    }
}
