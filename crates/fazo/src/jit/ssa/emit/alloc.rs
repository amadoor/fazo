//! Register allocation in one forward pass, after Maglev's `StraightForwardRegisterAllocator` (src/maglev/maglev-regalloc.cc).
//! Values never change, so a spill slot is written at most once per path.

use super::regs::{Class, PReg, RegSet, Registers};
use crate::jit::ssa::ir::Val;

/// Half 0 is a value's only word, or a boxed value's payload; half 1 is a boxed value's tag.
pub(crate) type Part = u32;

pub(crate) fn part(value: Val, half: u32) -> Part {
    value.0 * 2 + half
}

pub(crate) fn value_of(part: Part) -> Val {
    Val(part / 2)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Loc {
    Reg(PReg),
    Slot(u32),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Entry {
    pub(crate) part: Part,
    pub(crate) reg: Option<PReg>,
    pub(crate) in_slot: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(crate) struct Have {
    pub(crate) reg: Option<PReg>,
    pub(crate) slot: Option<u32>,
    pub(crate) constant: Option<u64>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Want {
    pub(crate) class: Class,
    pub(crate) reg: Option<PReg>,
    pub(crate) slot: Option<u32>,
}

pub(crate) trait Moves {
    fn copy(&mut self, dst: PReg, src: PReg);
    fn spill(&mut self, slot: u32, src: PReg);
    fn reload(&mut self, dst: PReg, slot: u32);
    fn constant(&mut self, dst: PReg, bits: u64);
    fn constant_slot(&mut self, slot: u32, bits: u64);
    fn slot_copy(&mut self, class: Class, dst: u32, src: u32);
    fn save_scratch(&mut self, src: PReg);
    fn restore_scratch(&mut self, dst: PReg);
}

pub(crate) struct Alloc {
    regs: Registers,
    owner: [Option<Part>; 64],
    reg_of: Vec<Option<PReg>>,
    in_slot: Vec<bool>,
    slot_of: Vec<u32>,
    class: Vec<Class>,
    pub(crate) slots: u32,
    pinned: RegSet,
    temps: RegSet,
    pub(crate) used: RegSet,
    /// Each value's last position in the current block, `u32::MAX` past it.
    pub(crate) last: Vec<u32>,
    /// Set when a part is asked for that is nowhere, so the compile is declined rather than reading garbage.
    pub(crate) broken: bool,
}

const NO_SLOT: u32 = u32::MAX;
const LOOP_RESERVE: usize = 4;

impl Alloc {
    pub(crate) fn new(regs: Registers, classes: Vec<Class>) -> Self {
        let parts = classes.len();
        Self {
            regs,
            owner: [None; 64],
            reg_of: vec![None; parts],
            in_slot: vec![false; parts],
            slot_of: vec![NO_SLOT; parts],
            class: classes,
            slots: 0,
            pinned: RegSet::default(),
            temps: RegSet::default(),
            used: RegSet::default(),
            last: vec![u32::MAX; parts / 2],
            broken: false,
        }
    }

    pub(crate) fn class_of(&self, part: Part) -> Class {
        self.class[part as usize]
    }

    pub(crate) fn slot(&mut self, part: Part) -> u32 {
        if self.slot_of[part as usize] == NO_SLOT {
            self.slot_of[part as usize] = self.slots;
            self.slots += 1;
        }
        self.slot_of[part as usize]
    }

    pub(crate) fn reg_of(&self, part: Part) -> Option<PReg> {
        self.reg_of[part as usize]
    }

    /// `None` for a part that is nowhere, which the caller treats as a broken compile.
    pub(crate) fn location(&self, part: Part) -> Option<Loc> {
        match self.reg_of[part as usize] {
            Some(reg) => Some(Loc::Reg(reg)),
            None if self.in_slot[part as usize] && self.slot_of[part as usize] != NO_SLOT => Some(Loc::Slot(self.slot_of[part as usize])),
            None => {
                debug_assert!(false, "a live part is in a register or its slot");
                None
            }
        }
    }

    pub(crate) fn have(&self, part: Part) -> Have {
        Have {
            reg: self.reg_of[part as usize],
            slot: self.in_slot[part as usize].then_some(self.slot_of[part as usize]),
            constant: None,
        }
    }

    pub(crate) fn enter(&mut self, entries: &[Entry]) {
        for index in 0..self.owner.len() {
            if let Some(part) = self.owner[index].take() {
                self.reg_of[part as usize] = None;
            }
        }
        self.pinned = RegSet::default();
        self.temps = RegSet::default();
        for entry in entries {
            self.reg_of[entry.part as usize] = entry.reg;
            self.in_slot[entry.part as usize] = entry.in_slot;
            if let Some(reg) = entry.reg {
                self.owner[reg.index()] = Some(entry.part);
            }
        }
    }

    pub(crate) fn entry(&self, part: Part) -> Entry {
        Entry {
            part,
            reg: self.reg_of[part as usize],
            in_slot: self.in_slot[part as usize],
        }
    }

    /// Parameters take a free register, preferring their argument's; `spilled` parts stay in their slots across the loop, as Maglev's `HoistLoopSpills` keeps them, and at a loop header the rest take the registers left, as its `HoistLoopReloads` does (src/maglev/maglev-regalloc.cc).
    pub(crate) fn fix(&mut self, live: &[Part], params: &[(Part, Option<PReg>)], spilled: impl Fn(Part) -> bool, hot: impl Fn(Part) -> bool) -> Vec<Entry> {
        let mut entries = Vec::with_capacity(live.len() + params.len());
        for &part in live {
            if spilled(part) {
                self.slot(part);
                entries.push(Entry { part, reg: None, in_slot: true });
                continue;
            }
            entries.push(self.entry(part));
        }
        let mut taken = RegSet::of(&entries.iter().filter_map(|entry| entry.reg).collect::<Vec<_>>());
        for &(param, preferred) in params {
            if spilled(param) {
                self.slot(param);
                entries.push(Entry { part: param, reg: None, in_slot: true });
                continue;
            }
            let class = self.class_of(param);
            let reg = preferred.filter(|&reg| !taken.contains(reg)).or_else(|| self.regs.order(class).find(|&reg| !taken.contains(reg)));
            match reg {
                Some(reg) => {
                    taken.insert(reg);
                    entries.push(Entry { part: param, reg: Some(reg), in_slot: false });
                }
                None => {
                    self.slot(param);
                    entries.push(Entry { part: param, reg: None, in_slot: true });
                }
            }
        }
        self.hoist_reloads(&mut entries, &mut taken, &hot);
        entries
    }

    /// Only values read on every iteration, leaving `LOOP_RESERVE` registers of each class for the body's own values.
    fn hoist_reloads(&self, entries: &mut [Entry], taken: &mut RegSet, hot: &impl Fn(Part) -> bool) {
        for entry in entries.iter_mut().filter(|entry| entry.reg.is_none() && hot(entry.part)) {
            let class = self.class_of(entry.part);
            let free = self.regs.order(class).filter(|&reg| !taken.contains(reg)).count();
            if free <= LOOP_RESERVE {
                continue;
            }
            let Some(reg) = self.regs.order(class).find(|&reg| !taken.contains(reg)) else {
                continue;
            };
            taken.insert(reg);
            entry.reg = Some(reg);
        }
    }

    /// A block no edge has reached yet finds every value in its slot.
    pub(crate) fn in_slots(&mut self, parts: &[Part]) -> Vec<Entry> {
        parts
            .iter()
            .map(|&part| {
                self.slot(part);
                Entry { part, reg: None, in_slot: true }
            })
            .collect()
    }

    pub(crate) fn want(&mut self, entry: &Entry) -> Want {
        let slot = entry.in_slot.then(|| self.slot(entry.part));
        Want {
            class: self.class_of(entry.part),
            reg: entry.reg,
            slot,
        }
    }

    fn free(&self, class: Class) -> Option<PReg> {
        let busy = self.pinned.union(self.temps);
        self.regs.order(class).find(|&reg| self.owner[reg.index()].is_none() && !busy.contains(reg))
    }

    /// Prefers a value already in its slot, then the one used last.
    fn victim(&self, class: Class) -> Option<PReg> {
        let busy = self.pinned.union(self.temps);
        self.regs
            .order(class)
            .filter(|&reg| !busy.contains(reg))
            .filter_map(|reg| self.owner[reg.index()].map(|part| (reg, part)))
            .max_by_key(|&(_, part)| (self.in_slot[part as usize], self.last[(part / 2) as usize]))
            .map(|(reg, _)| reg)
    }

    pub(crate) fn evict(&mut self, reg: PReg, moves: &mut impl Moves) {
        let Some(part) = self.owner[reg.index()].take() else {
            return;
        };
        if !self.in_slot[part as usize] {
            let slot = self.slot(part);
            moves.spill(slot, reg);
            self.in_slot[part as usize] = true;
        }
        self.reg_of[part as usize] = None;
    }

    fn pick(&mut self, class: Class, moves: &mut impl Moves) -> PReg {
        if let Some(reg) = self.free(class) {
            return reg;
        }
        let reg = self.victim(class).expect("a node needs fewer registers than the class has");
        self.evict(reg, moves);
        reg
    }

    pub(crate) fn use_reg(&mut self, part: Part, moves: &mut impl Moves) -> PReg {
        if let Some(reg) = self.reg_of[part as usize] {
            self.pinned.insert(reg);
            return reg;
        }
        debug_assert!(self.in_slot[part as usize] && self.slot_of[part as usize] != NO_SLOT, "a used part is in a register or its slot");
        if !self.in_slot[part as usize] || self.slot_of[part as usize] == NO_SLOT {
            self.broken = true;
        }
        let reg = self.pick(self.class_of(part), moves);
        moves.reload(reg, self.slot_of[part as usize]);
        self.place(part, reg);
        reg
    }

    fn place(&mut self, part: Part, reg: PReg) {
        self.reg_of[part as usize] = Some(reg);
        self.owner[reg.index()] = Some(part);
        self.pinned.insert(reg);
        self.used.insert(reg);
    }

    pub(crate) fn define(&mut self, part: Part, moves: &mut impl Moves) -> PReg {
        let reg = self.pick(self.class_of(part), moves);
        self.in_slot[part as usize] = false;
        self.place(part, reg);
        reg
    }

    /// Maglev's fixed-register result hint: the result takes `reg` only when nothing holds it.
    pub(crate) fn define_in(&mut self, part: Part, reg: PReg) -> Option<PReg> {
        let busy = self.pinned.union(self.temps);
        if reg.class != self.class_of(part) || self.owner[reg.index()].is_some() || busy.contains(reg) || !self.regs.order(reg.class).any(|candidate| candidate == reg) {
            return None;
        }
        self.in_slot[part as usize] = false;
        self.place(part, reg);
        Some(reg)
    }

    /// Maglev's `DefineSameAsFirst` (src/maglev/x64/maglev-ir-x64.cc); a two-address op then needs no copy.
    pub(crate) fn define_over(&mut self, part: Part, input: Part) -> Option<PReg> {
        let reg = self.reg_of[input as usize].filter(|reg| reg.class == self.class_of(part))?;
        self.reg_of[input as usize] = None;
        self.in_slot[part as usize] = false;
        self.place(part, reg);
        Some(reg)
    }

    /// A temporary dies within its node, so a free caller-saved register costs no prologue save.
    pub(crate) fn temp(&mut self, class: Class, moves: &mut impl Moves) -> PReg {
        let busy = self.pinned.union(self.temps);
        let callee_saved = self.regs.callee_saved;
        let scratch = self.regs.order(class).find(|&reg| self.owner[reg.index()].is_none() && !busy.contains(reg) && !callee_saved.contains(reg));
        let reg = match scratch {
            Some(reg) => reg,
            None => self.pick(class, moves),
        };
        self.temps.insert(reg);
        self.used.insert(reg);
        reg
    }

    pub(crate) fn pin(&mut self, reg: PReg) {
        self.pinned.insert(reg);
    }

    /// A value rebuilt rather than reloaded: eviction drops it without a store.
    pub(crate) fn mark_rebuilt(&mut self, part: Part) {
        self.in_slot[part as usize] = true;
    }

    pub(crate) fn release(&mut self, value: Val) {
        for half in 0..2 {
            let part = part(value, half);
            if let Some(reg) = self.reg_of.get(part as usize).copied().flatten() {
                self.owner[reg.index()] = None;
                self.reg_of[part as usize] = None;
            }
        }
    }

    pub(crate) fn end_node(&mut self, dying: &[Val]) {
        self.pinned = RegSet::default();
        self.temps = RegSet::default();
        for &value in dying {
            self.release(value);
        }
    }

    pub(crate) fn held(&self) -> impl Iterator<Item = (PReg, Part)> + '_ {
        self.owner.iter().enumerate().filter_map(|(index, part)| part.map(|part| (PReg::from_index(index), part)))
    }

    /// A part already in its slot needs no store.
    pub(crate) fn across_call(&mut self, keep: impl Fn(Part) -> bool) -> Vec<(PReg, u32, bool)> {
        let callee_saved = self.regs.callee_saved;
        let held: Vec<(PReg, Part)> = self.held().filter(|&(reg, part)| !callee_saved.contains(reg) && keep(part)).collect();
        held.into_iter()
            .map(|(reg, part)| {
                let stored = self.in_slot[part as usize];
                (reg, self.slot(part), stored)
            })
            .collect()
    }
}

/// Slots are written first, while every register still holds its value; then register moves in parallel, then loads.
pub(crate) fn resolve(pairs: &[(Want, Have)], moves: &mut impl Moves) {
    for &(want, have) in pairs {
        let Some(slot) = want.slot else {
            continue;
        };
        if have.slot == Some(slot) {
            continue;
        }
        match (have.reg, have.constant, have.slot) {
            (Some(reg), _, _) => moves.spill(slot, reg),
            (None, Some(bits), _) => moves.constant_slot(slot, bits),
            (None, None, Some(source)) => moves.slot_copy(want.class, slot, source),
            (None, None, None) => unreachable!("an edge's value is somewhere"),
        }
    }
    for class in [Class::Gp, Class::Fp] {
        let pending: Vec<(PReg, Option<PReg>)> = pairs
            .iter()
            .filter(|(want, _)| want.class == class)
            .filter_map(|&(want, have)| match (want.reg, have.reg) {
                (Some(dst), Some(src)) if dst != src => Some((dst, Some(src))),
                _ => None,
            })
            .collect();
        parallel(pending, moves);
    }
    for &(want, have) in pairs {
        let Some(dst) = want.reg else {
            continue;
        };
        if have.reg.is_some() {
            continue;
        }
        match (have.constant, have.slot) {
            (Some(bits), _) => moves.constant(dst, bits),
            (None, Some(slot)) => moves.reload(dst, slot),
            (None, None) => unreachable!("an edge's value is somewhere"),
        }
    }
}

/// `None` as a source is the scratch register that breaks a cycle.
fn parallel(mut pending: Vec<(PReg, Option<PReg>)>, moves: &mut impl Moves) {
    while !pending.is_empty() {
        let ready = pending.iter().position(|&(dst, _)| !pending.iter().any(|&(other, src)| src == Some(dst) && other != dst));
        let Some(index) = ready else {
            let (dst, src) = pending.remove(0);
            moves.save_scratch(dst);
            if let Some(src) = src {
                moves.copy(dst, src);
            }
            for (_, other) in &mut pending {
                if *other == Some(dst) {
                    *other = None;
                }
            }
            continue;
        };
        let (dst, src) = pending.remove(index);
        match src {
            Some(src) => moves.copy(dst, src),
            None => moves.restore_scratch(dst),
        }
    }
}
