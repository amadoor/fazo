//! Block order and liveness, after Maglev's allocator (src/maglev/maglev-regalloc.cc).

use super::super::ir::{BinOp, BlockId, Const, Func, Op, Rep, Term, Val};

pub(crate) struct Liveness {
    pub(crate) order: Vec<BlockId>,
    words: usize,
    live_in: Vec<u64>,
    /// A comparison only its block's branch reads is compared at the branch.
    pub(crate) fused: Vec<bool>,
}

/// A value the allocator places: constants are rebuilt where they are used, and effects have no value.
pub(crate) fn has_location(func: &Func, value: Val) -> bool {
    let inst = func.inst(value);
    inst.rep != Rep::None && !matches!(inst.op, Op::Const(_))
}

pub(crate) fn safepoint_polls(func: &Func, pc: u32) -> bool {
    func.osr.iter().any(|&(at, _)| at == pc)
}

/// A new table's state after the call holds the table itself, which the instruction defines rather than uses.
pub(crate) fn for_each_use(func: &Func, value: Val, mut f: impl FnMut(Val)) {
    let op = &func.inst(value).op;
    op.for_each_operand(&mut f);
    match op {
        Op::Safepoint { pc, state, .. } if safepoint_polls(func, *pc) => func.state(*state).for_each_operand(&mut f),
        Op::Safepoint { .. } => {}
        Op::NewTable { state, after, .. } => {
            func.state(*state).for_each_operand(&mut f);
            func.state(*after).for_each_operand(|used| {
                if used != value {
                    f(used);
                }
            });
        }
        _ => {
            if let Some(state) = op.state() {
                func.state(state).for_each_operand(&mut f);
            }
        }
    }
}

pub(crate) fn for_each_term_use(func: &Func, term: &Term, mut f: impl FnMut(Val)) {
    for value in term.operands() {
        f(value);
    }
    if let Term::Deopt { state } = term {
        func.state(*state).for_each_operand(&mut f);
    }
}

/// Reverse postorder, with cold blocks after the rest so the hot path falls through.
fn block_order(func: &Func) -> Vec<BlockId> {
    let order = func.reverse_postorder();
    let (mut hot, cold): (Vec<BlockId>, Vec<BlockId>) = order.into_iter().partition(|&block| !func.block(block).cold);
    hot.extend(cold);
    hot
}

fn use_counts(func: &Func, order: &[BlockId]) -> Vec<u32> {
    let mut counts = vec![0u32; func.insts.len()];
    for &block in order {
        let data = func.block(block);
        for &value in &data.insts {
            for_each_use(func, value, |used| counts[used.0 as usize] += 1);
        }
        for_each_term_use(func, &data.term, |used| counts[used.0 as usize] += 1);
    }
    counts
}

fn fusable(func: &Func, block: BlockId, counts: &[u32]) -> Option<Val> {
    let data = func.block(block);
    let Term::Branch { cond, .. } = data.term else {
        return None;
    };
    if data.insts.last() != Some(&cond) || counts[cond.0 as usize] != 1 {
        return None;
    }
    fuses(func, cond).then_some(cond)
}

/// The tests `compare_branch` can make at the branch itself.
fn fuses(func: &Func, cond: Val) -> bool {
    match func.op(cond) {
        Op::Binary(BinOp::ICmp(_) | BinOp::FCmp(_), _, _) | Op::ShapeIs { .. } => true,
        Op::EqConst(operand, constant) => equality(func, *operand, *constant).is_some(),
        _ => false,
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Equality {
    Word(i64),
    Boxed { tag: u8, payload: i64 },
}

/// An `EqConst` the branch can compare: a word against an integer or a typed payload, or a boxed value against a
/// collectable (a boxed integer also equals a float, which the branch does not test).
pub(crate) fn equality(func: &Func, operand: Val, constant: Const) -> Option<Equality> {
    if matches!(func.op(operand), Op::Const(_)) {
        return None;
    }
    match (func.rep(operand), constant) {
        (Rep::Int | Rep::Word, Const::Int(k)) => Some(Equality::Word(k)),
        (Rep::Payload, Const::Gc { tag, payload }) if func.ty(operand).tag() == Some(tag) => Some(Equality::Word(payload as i64)),
        (Rep::Boxed, Const::Gc { .. }) => Some(Equality::Boxed { tag: constant.tag(), payload: constant.payload() as i64 }),
        _ => None,
    }
}

impl Liveness {
    pub(crate) fn new(func: &Func) -> Self {
        let order = block_order(func);
        let counts = use_counts(func, &order);
        let mut fused = vec![false; func.insts.len()];
        for &block in &order {
            if let Some(cond) = fusable(func, block, &counts) {
                fused[cond.0 as usize] = true;
            }
        }
        let words = func.insts.len().div_ceil(64).max(1);
        let mut liveness = Self {
            order,
            words,
            live_in: vec![0; func.blocks.len() * words],
            fused,
        };
        liveness.solve(func);
        liveness
    }

    fn solve(&mut self, func: &Func) {
        let words = self.words;
        let blocks = func.blocks.len();
        let mut gen_sets = vec![0u64; blocks * words];
        let mut kill = vec![0u64; blocks * words];
        for &block in &self.order {
            let range = block.0 as usize * words..(block.0 as usize + 1) * words;
            let (generated, killed) = self.local_sets(func, block);
            gen_sets[range.clone()].copy_from_slice(&generated);
            kill[range].copy_from_slice(&killed);
        }
        let mut out = vec![0u64; words];
        let mut changed = true;
        while changed {
            changed = false;
            for &block in self.order.iter().rev() {
                out.iter_mut().for_each(|word| *word = 0);
                for successor in func.block(block).term.successors() {
                    let at = successor.0 as usize * words;
                    for (word, &bits) in out.iter_mut().zip(&self.live_in[at..at + words]) {
                        *word |= bits;
                    }
                }
                let at = block.0 as usize * words;
                for index in 0..words {
                    let value = gen_sets[at + index] | (out[index] & !kill[at + index]);
                    if value != self.live_in[at + index] {
                        self.live_in[at + index] = value;
                        changed = true;
                    }
                }
            }
        }
    }

    /// Upward-exposed uses and definitions of one block.
    fn local_sets(&self, func: &Func, block: BlockId) -> (Vec<u64>, Vec<u64>) {
        let mut live = vec![0u64; self.words];
        let mut killed = vec![0u64; self.words];
        let data = func.block(block);
        self.term_uses(func, block, |value| set(&mut live, value));
        for &value in data.insts.iter().rev() {
            clear(&mut live, value);
            set(&mut killed, value);
            self.inst_uses(func, value, |used| set(&mut live, used));
        }
        for &param in &data.params {
            clear(&mut live, param);
            set(&mut killed, param);
        }
        (live, killed)
    }

    /// A fused comparison's operands are uses of its branch (see `term_uses`).
    pub(crate) fn inst_uses(&self, func: &Func, value: Val, mut f: impl FnMut(Val)) {
        if self.fused[value.0 as usize] {
            return;
        }
        for_each_use(func, value, |used| {
            if has_location(func, used) {
                f(used);
            }
        });
    }

    pub(crate) fn term_uses(&self, func: &Func, block: BlockId, mut f: impl FnMut(Val)) {
        let data = func.block(block);
        if let Term::Branch { cond, .. } = data.term
            && self.fused[cond.0 as usize]
        {
            func.inst(cond).op.for_each_operand(|used| {
                if has_location(func, used) {
                    f(used);
                }
            });
        }
        for_each_term_use(func, &data.term, |used| {
            if has_location(func, used) && !self.fused[used.0 as usize] {
                f(used);
            }
        });
    }

    pub(crate) fn live_in(&self, block: BlockId) -> impl Iterator<Item = Val> + '_ {
        let at = block.0 as usize * self.words;
        bits(&self.live_in[at..at + self.words])
    }

    pub(crate) fn is_live_in(&self, block: BlockId, value: Val) -> bool {
        let at = block.0 as usize * self.words + value.0 as usize / 64;
        self.live_in[at] & 1 << (value.0 % 64) != 0
    }

    pub(crate) fn live_out(&self, func: &Func, block: BlockId) -> Vec<Val> {
        let mut out = vec![0u64; self.words];
        for successor in func.block(block).term.successors() {
            let at = successor.0 as usize * self.words;
            for (word, &bits) in out.iter_mut().zip(&self.live_in[at..at + self.words]) {
                *word |= bits;
            }
        }
        bits(&out).collect()
    }

    /// `last` holds `u32::MAX` for values that live out of the block.
    pub(crate) fn deaths(&self, func: &Func, block: BlockId, last: &mut [u32]) -> Vec<Vec<Val>> {
        let data = func.block(block);
        let positions = data.insts.len() + 1;
        let mut dying: Vec<Vec<Val>> = vec![Vec::new(); positions];
        let mut live = vec![0u64; self.words];
        for value in self.live_out(func, block) {
            set(&mut live, value);
            last[value.0 as usize] = u32::MAX;
        }
        let mut uses = Vec::new();
        self.term_uses(func, block, |value| uses.push(value));
        for value in uses.drain(..) {
            note(&mut dying, last, &mut live, positions - 1, value);
        }
        for (position, &value) in data.insts.iter().enumerate().rev() {
            if has_location(func, value) && !self.fused[value.0 as usize] && !get(&live, value) {
                dying[position].push(value);
                last[value.0 as usize] = position as u32;
            }
            clear(&mut live, value);
            self.inst_uses(func, value, |used| uses.push(used));
            for used in uses.drain(..) {
                note(&mut dying, last, &mut live, position, used);
            }
        }
        for &param in &data.params {
            if has_location(func, param) && !get(&live, param) {
                dying[0].push(param);
                last[param.0 as usize] = 0;
            }
        }
        dying
    }
}

fn note(dying: &mut [Vec<Val>], last: &mut [u32], live: &mut [u64], position: usize, value: Val) {
    if get(live, value) {
        return;
    }
    set(live, value);
    last[value.0 as usize] = position as u32;
    dying[position].push(value);
}

fn set(bits: &mut [u64], value: Val) {
    bits[value.0 as usize / 64] |= 1 << (value.0 % 64);
}

fn clear(bits: &mut [u64], value: Val) {
    bits[value.0 as usize / 64] &= !(1 << (value.0 % 64));
}

fn get(bits: &[u64], value: Val) -> bool {
    bits[value.0 as usize / 64] & 1 << (value.0 % 64) != 0
}

fn bits(words: &[u64]) -> impl Iterator<Item = Val> + '_ {
    words.iter().enumerate().flat_map(|(index, &word)| {
        let mut rest = word;
        std::iter::from_fn(move || {
            if rest == 0 {
                return None;
            }
            let bit = rest.trailing_zeros();
            rest &= rest - 1;
            Some(Val(index as u32 * 64 + bit))
        })
    })
}
